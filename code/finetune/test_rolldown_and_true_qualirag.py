"""Test the combination of:
1. True QualiRAG Scene-Matched Clean Model Baseline Subtraction:
   For any query image x, retrieve its Top-K semantically similar Clean Reference Images in `va`
   (Leave-One-Source-Out for `va`, full `va` for `te`) and compute what our 48-model fine-tuned IQA ensemble
   predicts on those Clean Reference Images of the same scene (`mu_q_clean(x)`).
   Subtract the scene false-positive bias: `q_cal(x) = q(x) - beta * (mu_q_clean(x) - mean(mu_q_clean))`!
2. Top-Down Ordinal Probability Mass Roll-Down (fixing the `decode_pc` bug where failed `p0` mass was thrown in the trash instead of rolling down to `P1`, and failed `p1` mass was thrown in the trash instead of rolling down to `P2`!)
3. Stage-2A Dedicated Image-Level All-Clean Gate (protecting `S_clean_doc2` = 20 pts and `S_clean_doc1` = 20 pts!).
"""
import os, sys, csv, time, warnings
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from run_scheme1_scheme2_breakthrough import FULL_IDX, BIN_IDX, score_both, load_global_assets
from fast_test_qualirag_preserved_p0 import CACHE_PATH, align_prob_mean, build_p0_preserved_v9_14ch
from ablate_clean_gate_and_qualirag_lowdim import build_compact_clean_gate_features, fit_compact_clean_gate

t0 = time.time()
cache = torch.load(CACHE_PATH, weights_only=False)
assets = load_global_assets()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compute_qualirag_scene_model_residual(assets, va_idx, te_idx, gva, p0_va, p1_va, p2_va, p0_te, p1_te, p2_te, dev):
    """True QualiRAG in Fine-Tuned IQA Model Space:
    1. Build semantic scene similarity S(x, r) using SigLIP2 + PE-Core + EUPE + InternViT + graphic_score + resolution.
    2. For each query x (in `va` via Leave-One-Source-Out, in `te` querying `va`):
       - Find Top-15 most similar Clean Reference Images `r` in `va` (`(gva == 0).all(1)`).
       - Compute the fine-tuned IQA ensemble's expected false-positive baseline on clean images of that scene:
         `mu_clean_q(x) = sum_r w_r q(r)`
       - Also find Top-20 most similar overall reference images `r` in `va` and compute the empirical human-vs-model
         calibration residual:
         `res_q(x) = sum_r w_r (y_true(r) - q(r))`
    """
    blocks = assets['blocks']
    src_va = assets['all_sources'][va_idx]
    same_src_va = torch.from_numpy(src_va[:, None] == src_va[None, :]).to(dev)

    S_vv = 0.0
    S_tv = 0.0
    for idx_b, w_b in [(5, 0.35), (4, 0.25), (2, 0.20), (0, 0.20)]:
        B_va = torch.from_numpy(blocks[idx_b][va_idx]).to(dev)
        B_te = torch.from_numpy(blocks[idx_b][te_idx]).to(dev)
        S_vv = S_vv + w_b * (B_va @ B_va.T)
        S_tv = S_tv + w_b * (B_te @ B_va.T)

    gs_va = torch.from_numpy(assets['X_gs'][va_idx]).to(dev)
    gs_te = torch.from_numpy(assets['X_gs'][te_idx]).to(dev)
    logwh_va = torch.from_numpy(assets['X_res'][va_idx, 0]).to(dev)
    logwh_te = torch.from_numpy(assets['X_res'][te_idx, 0]).to(dev)

    dom_vv = 0.15 * torch.abs(gs_va[:, None] - gs_va[None, :]) + 0.04 * torch.abs(logwh_va[:, None] - logwh_va[None, :])
    dom_tv = 0.15 * torch.abs(gs_te[:, None] - gs_va[None, :]) + 0.04 * torch.abs(logwh_te[:, None] - logwh_va[None, :])

    S_vv = (S_vv - dom_vv).masked_fill(same_src_va, -1e4)
    S_tv = S_tv - dom_tv

    # Clean mask in `va`
    clean_va_mask = torch.from_numpy((gva == 0).all(axis=1)).to(dev)
    S_vv_clean = S_vv.masked_fill(~clean_va_mask[None, :], -1e4)
    S_tv_clean = S_tv.masked_fill(~clean_va_mask[None, :], -1e4)

    Q_va = np.concatenate([p0_va, p1_va, p2_va], axis=1).astype(np.float32)  # (500, 14)
    Q_te = np.concatenate([p0_te, p1_te, p2_te], axis=1).astype(np.float32)  # (1000, 14)
    Q_va_t = torch.from_numpy(Q_va).to(dev)

    # 1. Expected false-positive baseline on scene-matched Clean images
    tv_vc, ti_vc = torch.topk(S_vv_clean, k=12, dim=1)
    w_vc = F.softmax(tv_vc / 0.05, dim=1)
    mu_clean_va = (w_vc.unsqueeze(-1) * Q_va_t[ti_vc]).sum(dim=1).cpu().numpy()

    tv_tc, ti_tc = torch.topk(S_tv_clean, k=12, dim=1)
    w_tc = F.softmax(tv_tc / 0.05, dim=1)
    mu_clean_te = (w_tc.unsqueeze(-1) * Q_va_t[ti_tc]).sum(dim=1).cpu().numpy()

    # Center `mu_clean` around its global mean so subtracting `beta * (mu_clean - mean)` preserves marginal means!
    bar_mu = mu_clean_va.mean(axis=0, keepdims=True)
    fp_excess_va = mu_clean_va - bar_mu
    fp_excess_te = mu_clean_te - bar_mu

    # 2. Scene-matched residual `(Y_true - Q_va)` across Top-25 neighbors
    Y_va = np.concatenate([
        (gva == 3).astype(np.float32),
        (gva[:, FULL_IDX] == 2).astype(np.float32),
        (gva[:, FULL_IDX] == 1).astype(np.float32),
    ], axis=1)
    # Match Y_va scale to Q_va scale per column so residual is zero-mean!
    Y_va_scaled = Y_va * (Q_va.mean(axis=0, keepdims=True) / (Y_va.mean(axis=0, keepdims=True) + 1e-6))
    R_va_t = torch.from_numpy(Y_va_scaled - Q_va).to(dev)

    tv_v, ti_v = torch.topk(S_vv, k=25, dim=1)
    w_v = F.softmax(tv_v / 0.05, dim=1)
    res_va = (w_v.unsqueeze(-1) * R_va_t[ti_v]).sum(dim=1).cpu().numpy()

    tv_t, ti_t = torch.topk(S_tv, k=25, dim=1)
    w_t = F.softmax(tv_t / 0.05, dim=1)
    res_te = (w_t.unsqueeze(-1) * R_va_t[ti_t]).sum(dim=1).cpu().numpy()

    return (fp_excess_va, res_va), (fp_excess_te, res_te)


def apply_qualirag_calibration(p0, p1, p2, fp_excess, res, beta_fp=0.35, beta_res=0.20):
    """Apply Scene-Matched Clean Baseline Subtraction (`- beta_fp * fp_excess`)
    and Scene-Matched Residual Correction (`+ beta_res * res`) with mean-odds preservation!
    """
    Q = np.concatenate([p0, p1, p2], axis=1)
    Q_cal = np.clip(Q - beta_fp * fp_excess + beta_res * res, 1e-5, 0.999)
    for col in range(14):
        Q_cal[:, col] = align_prob_mean(Q_cal[:, col], Q[:, col])
    return Q_cal[:, :6], Q_cal[:, 6:10], Q_cal[:, 10:14]


def decode_ordinal_rolldown_two_stage(p0, p1, p2, cln0, cln1, cln2, p_clean, prm):
    """Scheme 2 Full Decoder:
    1. Top-Down Ordinal Probability Mass Roll-Down:
       - If an image fails `c0 = (p0 > T0)`, its `p0` mass is NOT thrown in the trash!
         Instead, the effective P1 score is `s1 = p1 + w01 * p0`!
       - If an image also fails `c1`, its `p0 + p1` mass rolls down into P2:
         `s2 = p2 + w12 * (p1 + p0)`!
    2. Cross-Defect Co-Occurrence Boost (on confirmed non-clean images):
       - In ground-truth annotations, `模糊`, `过锐`, and `过度生成` co-occur with `纹理损失` 88%~95% of the time!
       - Conversely, `虚化突兀_P0` (sharp subject + bokeh background) suppresses false global `模糊_P2`!
    3. Stage-2A Dedicated Image-Level All-Clean Gate:
       - Clears non-P0 false triggers on images where `p_clean >= tau_clean` AND `cln_cum.max(1) < tau_cln_max`!
    """
    n = p0.shape[0]
    pred = np.zeros((n, 6), dtype=int)

    cln_cum = cln0.copy()
    cum = p0.copy()
    for k, ci in enumerate(FULL_IDX):
        cln_cum[:, ci] = cln0[:, ci] + cln1[:, k] + cln2[:, k]
        cum[:, ci] = p0[:, ci] + p1[:, k] + p2[:, k]

    sh_p0 = (cln_cum.max(axis=1) < 0.28)
    tau_clean = prm['tau_clean']
    tau_cln_max = prm['tau_cln_max']
    is_gated_clean = (p_clean >= tau_clean) & (cln_cum.max(axis=1) < tau_cln_max)

    w01 = prm.get('w01', [0.45, 0.45, 0.45, 0.45])
    w12 = prm.get('w12', [0.35, 0.35, 0.35, 0.35])
    T0 = prm.get('T0', list(HAND['T0']))
    T1 = prm.get('T1', list(HAND['T1']))
    T2 = prm.get('T2', list(HAND['T2']))
    R12 = prm.get('R12', list(HAND['R12']))

    # Cross-defect interaction adjustments on p1/p2
    gamma_tex = prm.get('gamma_tex', 0.08)   # boost 纹理损失 when blur/sharp/overgen is strong
    gamma_bok = prm.get('gamma_bok', 0.10)   # suppress 模糊_P2 when 虚化突兀_P0 is strong

    p2_adj = p2.copy()
    p1_adj = p1.copy()
    # 纹理损失 (k=1) co-occurs with max(模糊, 过锐, 过度生成)
    other_max = np.maximum(cum[:, 0], np.maximum(cum[:, 2], cum[:, 4]))
    p2_adj[:, 1] = p2_adj[:, 1] + gamma_tex * np.maximum(0.0, other_max - 0.35)
    p1_adj[:, 1] = p1_adj[:, 1] + 0.5 * gamma_tex * np.maximum(0.0, other_max - 0.45)
    # 模糊 (k=0) is suppressed when 虚化突兀_P0 (ci=5) is active
    p2_adj[:, 0] = np.maximum(0.0, p2_adj[:, 0] - gamma_bok * p0[:, 5])

    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            q0 = p0[:, ci]
            q1 = p1_adj[:, k]
            q2 = p2_adj[:, k]

            # Roll-down scores:
            s1 = q1 + w01[k] * q0
            s2 = q2 + w12[k] * (q1 + q0)

            c2 = s2 > T2[k]
            c1 = (s1 > T1[k]) & ((s1 * R12[k]) >= (q2 - 0.05))
            c0 = (q0 > T0[ci]) & (~sh_p0)

            pr = np.where(c2, 1, 0)
            pr = np.where(c1, 2, pr)
            pr = np.where(c0, 3, pr)
            pred[:, ci] = pr
        else:
            pred[:, ci] = np.where((p0[:, ci] > T0[ci]) & (~sh_p0), 3, 0)

    # Apply Stage-2A All-Clean Gate (protecting all P0 predictions)
    no_p0 = (pred != 3).all(axis=1)
    pred[is_gated_clean & no_p0] = 0

    # Apply Defective Image Rescue on strongly non-clean images
    tau_rescue = prm.get('tau_rescue', 0.18)
    need_rescue = (pred == 0).all(axis=1) & (~is_gated_clean) & (p_clean <= tau_rescue)
    if need_rescue.any():
        cum4 = cum[:, FULL_IDX]
        cum4_norm = cum4 / (cum4.mean(axis=0, keepdims=True) + 0.05)
        best_k = np.argmax(cum4_norm[need_rescue], axis=1)
        for ii, kk in zip(np.where(need_rescue)[0], best_k):
            pred[ii, FULL_IDX[kk]] = 1

    return pred


def optimize_rolldown_decoder_on_val(p0_va, p1_va, p2_va, cln0_va, cln1_va, cln2_va, p_clean_va, gva):
    """Optimize the Top-Down Ordinal Roll-Down + Stage-2A Clean Gate parameters strictly on `va` (500 images)
    with strong shrinkage toward HAND so it generalizes cleanly to `te` (1,000 images)!
    """
    import copy
    best_prm = {
        'tau_clean': 0.50, 'tau_cln_max': 0.45, 'tau_rescue': 0.18,
        'w01': [0.35, 0.35, 0.35, 0.35],
        'w12': [0.25, 0.25, 0.25, 0.25],
        'T0': list(HAND['T0']),
        'T1': [0.32, 0.32, 0.32, 0.32],
        'T2': [0.33, 0.32, 0.34, 0.36],
        'R12': list(HAND['R12']),
        'gamma_tex': 0.06,
        'gamma_bok': 0.08,
    }

    def eval_p(prm):
        pr = decode_ordinal_rolldown_two_stage(p0_va, p1_va, p2_va, cln0_va, cln1_va, cln2_va, p_clean_va, prm)
        s = score_both(gva, pr)
        return s['total_doc2'] + 0.45 * s['total']

    best_obj = eval_p(best_prm)

    # 1. Tune Clean Gate
    for tc in np.linspace(0.36, 0.64, 15):
        for tcm in [0.38, 0.44, 0.50, 0.56]:
            for trs in [0.0, 0.15, 0.22]:
                cand = {**best_prm, 'tau_clean': float(tc), 'tau_cln_max': float(tcm), 'tau_rescue': float(trs)}
                v = eval_p(cand)
                if v > best_obj + 1e-5:
                    best_obj, best_prm = v, cand

    # 2. Tune roll-down weights w01, w12 and (T1, T2) per FULL class
    for k in range(4):
        for w01_k in [0.0, 0.20, 0.35, 0.50, 0.65]:
            for t1_k in np.linspace(0.26, 0.42, 9):
                cand = copy.deepcopy(best_prm)
                cand['w01'][k] = float(w01_k)
                cand['T1'][k] = float(t1_k)
                v = eval_p(cand)
                if v > best_obj + 1e-5:
                    best_obj, best_prm = v, cand

        for w12_k in [0.0, 0.15, 0.30, 0.45]:
            for t2_k in np.linspace(0.26, 0.44, 10):
                cand = copy.deepcopy(best_prm)
                cand['w12'][k] = float(w12_k)
                cand['T2'][k] = float(t2_k)
                v = eval_p(cand)
                if v > best_obj + 1e-5:
                    best_obj, best_prm = v, cand

        for r12_k in [0.70, 0.80, 0.85, 0.95, 1.05]:
            cand = copy.deepcopy(best_prm)
            cand['R12'][k] = float(r12_k)
            v = eval_p(cand)
            if v > best_obj + 1e-5:
                best_obj, best_prm = v, cand

    # 3. Tune cross-defect interaction weights
    for gt in [0.0, 0.05, 0.10, 0.15]:
        for gb in [0.0, 0.06, 0.12, 0.18]:
            cand = {**best_prm, 'gamma_tex': float(gt), 'gamma_bok': float(gb)}
            v = eval_p(cand)
            if v > best_obj + 1e-5:
                best_obj, best_prm = v, cand

    return best_prm


rows = []
for sp in (16, 17):
    print(f"\n=====================================================================================")
    print(f"🔥 TESTING TRUE QUALIRAG + TOP-DOWN ORDINAL ROLL-DOWN ON SPLIT s{sp}")
    print(f"=====================================================================================")
    models, gva, gte = load_healthy_split_models(sp)
    c_sp = cache['splits'][sp]
    va_idx, te_idx, src_va = c_sp['va_idx'], c_sp['te_idx'], c_sp['src_va']
    q_feats, deqa_res = c_sp['q_feats'], c_sp['deqa_res']

    p0_v8_te, p1_v8_te, p2_v8_te, _, _, _ = build_decoupled_14ch_probs(models, 'te')
    r_v8 = score_both(gte, decode_pc(p0_v8_te, p1_v8_te, p2_v8_te, HAND))

    p0_v9_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va = build_p0_preserved_v9_14ch(models, 'va', None)
    p0_v9_te, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te = build_p0_preserved_v9_14ch(models, 'te', None)
    r_v9 = score_both(gte, decode_pc(p0_v9_te, p1_v9_te, p2_v9_te, HAND))

    # True QualiRAG Scene-Matched Clean Model Baseline Subtraction + Residual Correction
    (fp_va, res_va), (fp_te, res_te) = compute_qualirag_scene_model_residual(
        assets, va_idx, te_idx, gva, p0_v9_va, p1_v9_va, p2_v9_va, p0_v9_te, p1_v9_te, p2_v9_te, dev
    )

    # Tune (beta_fp, beta_res) strictly on `va`
    best_b = (0.0, 0.0)
    best_va_sc = -1e9
    for b_fp in [0.0, 0.15, 0.25, 0.35, 0.45]:
        for b_res in [0.0, 0.10, 0.20, 0.30]:
            q0_v, q1_v, q2_v = apply_qualirag_calibration(p0_v9_va, p1_v9_va, p2_v9_va, fp_va, res_va, b_fp, b_res)
            sc_v = score_both(gva, decode_pc(q0_v, q1_v, q2_v, HAND))
            if sc_v['total_doc2'] + 0.4 * sc_v['total'] > best_va_sc:
                best_va_sc = sc_v['total_doc2'] + 0.4 * sc_v['total']
                best_b = (b_fp, b_res)

    p0_q_va, p1_q_va, p2_q_va = apply_qualirag_calibration(p0_v9_va, p1_v9_va, p2_v9_va, fp_va, res_va, best_b[0], best_b[1])
    p0_q_te, p1_q_te, p2_q_te = apply_qualirag_calibration(p0_v9_te, p1_v9_te, p2_v9_te, fp_te, res_te, best_b[0], best_b[1])
    r_q_hand = score_both(gte, decode_pc(p0_q_te, p1_q_te, p2_q_te, HAND))
    print(f"   Selected QualiRAG (beta_fp={best_b[0]}, beta_res={best_b[1]}) on s{sp}_va")

    # Stage-2A All-Clean Gate
    cln_cum_va = cln0_va.copy()
    cln_cum_te = cln0_te.copy()
    for k, ci in enumerate(FULL_IDX):
        cln_cum_va[:, ci] += cln1_va[:, k] + cln2_va[:, k]
        cln_cum_te[:, ci] += cln1_te[:, k] + cln2_te[:, k]

    X_cln_va = build_compact_clean_gate_features(models, 'va', va_idx, p0_q_va, p1_q_va, p2_q_va, cln0_va, cln1_va, cln2_va, q_feats, deqa_res)
    X_cln_te = build_compact_clean_gate_features(models, 'te', te_idx, p0_q_te, p1_q_te, p2_q_te, cln0_te, cln1_te, cln2_te, q_feats, deqa_res)
    pcln_va, pcln_te = fit_compact_clean_gate(X_cln_va, cln_cum_va, gva, src_va, X_cln_te, cln_cum_te)

    # Optimize Roll-Down + Stage-2A Clean Gate on `va`, evaluate blindly on `te`
    prm_rd = optimize_rolldown_decoder_on_val(p0_q_va, p1_q_va, p2_q_va, cln0_va, cln1_va, cln2_va, pcln_va, gva)
    pred_rd_te = decode_ordinal_rolldown_two_stage(p0_q_te, p1_q_te, p2_q_te, cln0_te, cln1_te, cln2_te, pcln_te, prm_rd)
    r_rd = score_both(gte, pred_rd_te)

    for tag, r in [
        ("0. V8 Decoupled 14ch (HAND)", r_v8),
        ("1. V9 P0-Preserved 14ch (HAND)", r_v9),
        ("2. + Scheme 1 (QualiRAG Scene Baseline Sub, HAND)", r_q_hand),
        ("3. + Scheme 1 & 2 (QualiRAG + Roll-Down + Clean Gate)", r_rd),
    ]:
        print(f"  {tag:52s} | Doc2={r['total_doc2']:5.2f} (S_pos={r['S_pos']:.3f}, S_p0={r['S_p0']:.3f}, Clean2={r['S_clean_doc2']:.3f}) | Doc1={r['total']:5.2f} (Clean1={r['S_clean']:.3f})")
    rows.append((r_v8, r_v9, r_q_hand, r_rd))

print("\n=====================================================================================")
print("🏆 2-SPLIT MEAN (s16_test + s17_test = 2,000 Blind Evaluations) SUMMARY")
print("=====================================================================================")
for idx_m, tag in enumerate([
    "0. V8 Decoupled 14ch (HAND)",
    "1. V9 P0-Preserved 14ch (HAND)",
    "2. + Scheme 1 (QualiRAG Scene Baseline Sub, HAND)",
    "3. + Scheme 1 & 2 (QualiRAG + Roll-Down + Clean Gate)",
]):
    r16, r17 = rows[0][idx_m], rows[1][idx_m]
    print(f"  {tag:52s} | Mean Doc2 = {0.5*(r16['total_doc2']+r17['total_doc2']):5.2f} "
          f"(S_pos={0.5*(r16['S_pos']+r17['S_pos']):.3f}, S_p0={0.5*(r16['S_p0']+r17['S_p0']):.3f}, Clean2={0.5*(r16['S_clean_doc2']+r17['S_clean_doc2']):.3f}) | "
          f"Mean Doc1 = {0.5*(r16['total']+r17['total']):5.2f} (Clean1={0.5*(r16['S_clean']+r17['S_clean']):.3f})")
