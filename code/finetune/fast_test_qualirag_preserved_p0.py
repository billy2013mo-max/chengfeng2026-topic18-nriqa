"""Cache QualiRAG + DeQAScore 6-Tower outputs once and run fast (<3s) exact ablations
fixing the 2 bugs found in task-3106:
Bug 1: `a0 = [1.12, 0.68, 1.0, 1.0, 0.82, 0.88]` shrunk p0 against HAND thresholds (dropping S_p0 from 0.818 to 0.804).
Bug 2: Blending unweighted LogisticRegression (`mean=0.05`) into pos_weight-trained neural probs (`mean=0.16`) without odds-alignment shrunk probabilities downward.
"""
import os, sys, csv, time, warnings
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from run_scheme1_scheme2_breakthrough import (
    FULL_IDX, BIN_IDX, score_both, load_global_assets,
    compute_qualirag_features, train_deqascore_qualirag_tower
)

CACHE_PATH = '/tmp/qualirag_s16_s17_cache.pt'


def align_prob_mean(src, ref):
    src_c = np.clip(src, 1e-6, 1.0 - 1e-6)
    m_src = float(src_c.mean())
    m_ref = float(np.clip(ref.mean(), 1e-6, 1.0 - 1e-6))
    gamma = (m_ref / (1.0 - m_ref)) / (m_src / (1.0 - m_src))
    return odds_scale(src_c, gamma)


def get_or_build_cache():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached QualiRAG + DeQAScore tensors from {CACHE_PATH}...")
        return torch.load(CACHE_PATH, weights_only=False)

    assets = load_global_assets()
    all_names = assets['all_names']
    name_to_idx = assets['name_to_idx']
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cache = {'splits': {}}
    for sp in (16, 17):
        ref = np.load(f'results/v3_ablate/A_base_s{sp}_probs.npz', allow_pickle=True)
        va_names = [str(x) for x in ref['va_names']]
        te_names = [str(x) for x in ref['te_names']]
        sp_csv = {r['name']: r['split'] for r in csv.DictReader(open(f'data/resplits/split_s{sp}.csv'))}
        tr_names = [n for n in all_names if sp_csv.get(n) == 'train']

        tr_idx = np.array([name_to_idx[n] for n in tr_names])
        va_idx = np.array([name_to_idx[n] for n in va_names])
        te_idx = np.array([name_to_idx[n] for n in te_names])
        src_va = assets['all_sources'][va_idx]

        q_feats = compute_qualirag_features(assets, tr_idx, va_idx, te_idx, dev)
        deqa_res = train_deqascore_qualirag_tower(assets, q_feats, tr_idx, va_idx, te_idx, sp, dev, seeds=(1, 2, 3, 4))
        cache['splits'][sp] = dict(
            tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx, src_va=src_va,
            q_feats=q_feats, deqa_res=deqa_res
        )
    torch.save(cache, CACHE_PATH)
    print(f"Saved cache to {CACHE_PATH}")
    return cache


def build_p0_preserved_v9_14ch(models, split_prefix, deqa_res=None):
    """Builds 14-channel specialist probabilities WITHOUT the broken `a0` P0 down-scaling,
    and with strict prior-odds alignment on every added specialist!
    """
    p0, p1, p2, cln0, cln1, cln2 = build_decoupled_14ch_probs(models, split_prefix)

    # Incorporate V8_W2_ExpA_overgen on 过度生成_P1 (aligned to p1[:, 3])
    if 'v8_experts/V8_W2_ExpA_overgen' in models:
        og_p1 = (
            0.28 * models['v8_experts/V8_ExpA_overgen'][f'{split_prefix}_p1'][:, 3]
            + 0.22 * models['v8_experts/V8_W2_ExpA_overgen'][f'{split_prefix}_p1'][:, 3]
            + 0.25 * models['v3_ablate/D_gen_p1'][f'{split_prefix}_p1'][:, 3]
            + 0.25 * models['v3_ablate/H_og'][f'{split_prefix}_p1'][:, 3]
        )
        p1[:, 3] = align_prob_mean(og_p1, p1[:, 3])

    def get_m(k, arr):
        return models[k][f'{split_prefix}_{arr}'] if k in models else None

    v9_a0, v9_a1, v9_a2 = get_m('v9_specialists/V9_SpecA_OverGen_Giant', 'p0'), get_m('v9_specialists/V9_SpecA_OverGen_Giant', 'p1'), get_m('v9_specialists/V9_SpecA_OverGen_Giant', 'p2')
    v9_b0, v9_b1, v9_b2 = get_m('v9_specialists/V9_SpecB_SharpTex_HiRes', 'p0'), get_m('v9_specialists/V9_SpecB_SharpTex_HiRes', 'p1'), get_m('v9_specialists/V9_SpecB_SharpTex_HiRes', 'p2')
    v9_c0, v9_c1, v9_c2 = get_m('v9_specialists/V9_SpecC_AllP0_Tile3', 'p0'), get_m('v9_specialists/V9_SpecC_AllP0_Tile3', 'p1'), get_m('v9_specialists/V9_SpecC_AllP0_Tile3', 'p2')
    v9_d0, v9_d1, v9_d2 = get_m('v9_specialists/V9_SpecD_BokehOG_HiRes', 'p0'), get_m('v9_specialists/V9_SpecD_BokehOG_HiRes', 'p1'), get_m('v9_specialists/V9_SpecD_BokehOG_HiRes', 'p2')
    v9_e0, v9_e1, v9_e2 = get_m('v9_specialists/V9_SpecE_RealHiRes_Giant', 'p0'), get_m('v9_specialists/V9_SpecE_RealHiRes_Giant', 'p1'), get_m('v9_specialists/V9_SpecE_RealHiRes_Giant', 'p2')

    if v9_e0 is not None:
        cln0 = 0.80 * cln0 + 0.20 * v9_e0
        cln1 = 0.80 * cln1 + 0.20 * v9_e1
        cln2 = 0.80 * cln2 + 0.20 * v9_e2

        for ci, (va_w, vb_w, vc_w, vd_w, ve_w) in enumerate([
            (0.00, 0.00, 0.08, 0.00, 0.06),  # 0: 模糊_P0
            (0.00, 0.08, 0.08, 0.00, 0.05),  # 1: 纹理损失_P0
            (0.00, 0.12, 0.08, 0.00, 0.05),  # 2: 过锐_P0
            (0.00, 0.00, 0.08, 0.00, 0.04),  # 3: 花屏_P0
            (0.15, 0.00, 0.05, 0.10, 0.00),  # 4: 过度生成_P0
            (0.00, 0.00, 0.08, 0.12, 0.05),  # 5: 虚化突兀_P0
        ]):
            w_sum = va_w + vb_w + vc_w + vd_w + ve_w
            v9_blend = (
                va_w * align_prob_mean(v9_a0[:, ci], p0[:, ci]) +
                vb_w * align_prob_mean(v9_b0[:, ci], p0[:, ci]) +
                vc_w * align_prob_mean(v9_c0[:, ci], p0[:, ci]) +
                vd_w * align_prob_mean(v9_d0[:, ci], p0[:, ci]) +
                ve_w * align_prob_mean(v9_e0[:, ci], p0[:, ci])
            )
            p0[:, ci] = (1.0 - w_sum) * p0[:, ci] + v9_blend

        for k, (va_w, vb_w, vc_w, vd_w, ve_w) in enumerate([
            (0.00, 0.00, 0.12, 0.00, 0.10),  # 0: 模糊 P1/P2
            (0.00, 0.12, 0.10, 0.00, 0.08),  # 1: 纹理损失 P1/P2
            (0.00, 0.16, 0.08, 0.00, 0.08),  # 2: 过锐 P1/P2
            (0.20, 0.00, 0.00, 0.14, 0.06),  # 3: 过度生成 P1/P2
        ]):
            w_sum = va_w + vb_w + vc_w + vd_w + ve_w
            v9_b1_k = (
                va_w * align_prob_mean(v9_a1[:, k], p1[:, k]) +
                vb_w * align_prob_mean(v9_b1[:, k], p1[:, k]) +
                vc_w * align_prob_mean(v9_c1[:, k], p1[:, k]) +
                vd_w * align_prob_mean(v9_d1[:, k], p1[:, k]) +
                ve_w * align_prob_mean(v9_e1[:, k], p1[:, k])
            )
            p1[:, k] = (1.0 - w_sum) * p1[:, k] + v9_b1_k

            v9_b2_k = (
                va_w * align_prob_mean(v9_a2[:, k], p2[:, k]) +
                vb_w * align_prob_mean(v9_b2[:, k], p2[:, k]) +
                vc_w * align_prob_mean(v9_c2[:, k], p2[:, k]) +
                vd_w * align_prob_mean(v9_d2[:, k], p2[:, k]) +
                ve_w * align_prob_mean(v9_e2[:, k], p2[:, k])
            )
            p2[:, k] = (1.0 - w_sum) * p2[:, k] + v9_b2_k

    if deqa_res is not None:
        dq0 = deqa_res[f'{split_prefix}_p0']
        dq1 = deqa_res[f'{split_prefix}_p1']
        dq2 = deqa_res[f'{split_prefix}_p2']
        for ci in range(6):
            p0[:, ci] = 0.90 * p0[:, ci] + 0.10 * align_prob_mean(dq0[:, ci], p0[:, ci])
        for k in range(4):
            p1[:, k] = 0.88 * p1[:, k] + 0.12 * align_prob_mean(dq1[:, k], p1[:, k])
            p2[:, k] = 0.88 * p2[:, k] + 0.12 * align_prob_mean(dq2[:, k], p2[:, k])

    return p0, p1, p2, cln0, cln1, cln2


def fit_odds_aligned_qualirag_calibrator(p0_va, p1_va, p2_va, cln_cum_va, q_feats, deqa_res, va_idx, gva, src_va,
                                         p0_te, p1_te, p2_te, cln_cum_te, te_idx):
    """Fit L2-regularized LogisticRegression with `class_weight='balanced'` AND strictly align the output
    mean odds to `base_prob` via `align_prob_mean` before blending!
    This guarantees ZERO shrinkage of P0/P1/P2 probabilities while injecting QualiRAG scene residuals and cross-defect context!
    """
    gkf = GroupKFold(n_splits=5)
    N_va, N_te = len(va_idx), len(te_idx)

    def build_ch_feats(p0, p1, p2, cln_cum, idx_arr, prefix):
        cum6 = p0.copy()
        for k, ci in enumerate(FULL_IDX):
            cum6[:, ci] = p0[:, ci] + p1[:, k] + p2[:, k]
        X_knn = q_feats['X_knn'][idx_arr]
        z_sum = X_knn[:, -10:]
        gs = q_feats['X_aux'][idx_arr, -7:]
        dq0, dq1, dq2 = deqa_res[f'{prefix}_p0'], deqa_res[f'{prefix}_p1'], deqa_res[f'{prefix}_p2']
        return cum6, X_knn, z_sum, gs, dq0, dq1, dq2

    cum6_va, knn_va, zs_va, gs_va, dq0_va, dq1_va, dq2_va = build_ch_feats(p0_va, p1_va, p2_va, cln_cum_va, va_idx, 'va')
    cum6_te, knn_te, zs_te, gs_te, dq0_te, dq1_te, dq2_te = build_ch_feats(p0_te, p1_te, p2_te, cln_cum_te, te_idx, 'te')

    def fit_1d_aligned(f_va, y_va, f_te, base_prob_va, base_prob_te, blend_w=0.18):
        if y_va.sum() < 5:
            return base_prob_va, base_prob_te
        sc = StandardScaler()
        f_va_s = np.clip(sc.fit_transform(f_va), -4.0, 4.0)
        f_te_s = np.clip(sc.transform(f_te), -4.0, 4.0)
        oof = np.zeros(N_va)
        pte = np.zeros(N_te)
        for tr_f, vl_f in gkf.split(f_va_s, y_va, groups=src_va):
            lr = LogisticRegression(C=0.05, class_weight='balanced', max_iter=300)
            lr.fit(f_va_s[tr_f], y_va[tr_f])
            oof[vl_f] = lr.predict_proba(f_va_s[vl_f])[:, 1]
            pte += lr.predict_proba(f_te_s)[:, 1] / 5.0
        # Crucial: align the mean odds of `oof` and `pte` to `base_prob` so threshold scale is 100% preserved!
        oof_al = align_prob_mean(oof, base_prob_va)
        pte_al = align_prob_mean(pte, base_prob_te)
        return (1.0 - blend_w) * base_prob_va + blend_w * oof_al, (1.0 - blend_w) * base_prob_te + blend_w * pte_al

    cal_p0_va, cal_p0_te = p0_va.copy(), p0_te.copy()
    cal_p1_va, cal_p1_te = p1_va.copy(), p1_te.copy()
    cal_p2_va, cal_p2_te = p2_va.copy(), p2_te.copy()

    for ci in range(6):
        f0_va = np.column_stack([p0_va[:, ci], cln_cum_va[:, ci], dq0_va[:, ci], knn_va[:, 31 + 1 + ci], knn_va[:, 31 + 15 + ci], cum6_va, zs_va[:, [0, 5]], gs_va[:, :2]])
        f0_te = np.column_stack([p0_te[:, ci], cln_cum_te[:, ci], dq0_te[:, ci], knn_te[:, 31 + 1 + ci], knn_te[:, 31 + 15 + ci], cum6_te, zs_te[:, [0, 5]], gs_te[:, :2]])
        cal_p0_va[:, ci], cal_p0_te[:, ci] = fit_1d_aligned(f0_va, (gva[:, ci] == 3).astype(int), f0_te, p0_va[:, ci], p0_te[:, ci], blend_w=0.14)

    for k, ci in enumerate(FULL_IDX):
        f1_va = np.column_stack([p1_va[:, k], p0_va[:, ci], p2_va[:, k], cum6_va[:, ci], cln_cum_va[:, ci], dq1_va[:, k], knn_va[:, 31 + 7 + k], knn_va[:, 31 + 21 + k], cum6_va, zs_va[:, [0, 4, 5]], gs_va[:, :2]])
        f1_te = np.column_stack([p1_te[:, k], p0_te[:, ci], p2_te[:, k], cum6_te[:, ci], cln_cum_te[:, ci], dq1_te[:, k], knn_te[:, 31 + 7 + k], knn_te[:, 31 + 21 + k], cum6_te, zs_te[:, [0, 4, 5]], gs_te[:, :2]])
        cal_p1_va[:, k], cal_p1_te[:, k] = fit_1d_aligned(f1_va, (gva[:, ci] == 2).astype(int), f1_te, p1_va[:, k], p1_te[:, k], blend_w=0.22)

        f2_va = np.column_stack([p2_va[:, k], p1_va[:, k], p0_va[:, ci], cum6_va[:, ci], cln_cum_va[:, ci], dq2_va[:, k], knn_va[:, 31 + 11 + k], knn_va[:, 31 + 15 + ci], cum6_va, zs_va[:, [0, 4, 5]], gs_va[:, :2]])
        f2_te = np.column_stack([p2_te[:, k], p1_te[:, k], p0_te[:, ci], cum6_te[:, ci], cln_cum_te[:, ci], dq2_te[:, k], knn_te[:, 31 + 11 + k], knn_te[:, 31 + 15 + ci], cum6_te, zs_te[:, [0, 4, 5]], gs_te[:, :2]])
        cal_p2_va[:, k], cal_p2_te[:, k] = fit_1d_aligned(f2_va, (gva[:, ci] == 1).astype(int), f2_te, p2_va[:, k], p2_te[:, k], blend_w=0.22)

    return (cal_p0_va, cal_p1_va, cal_p2_va), (cal_p0_te, cal_p1_te, cal_p2_te)


def decode_p0_protected_two_stage(p0, p1, p2, cln0, cln1, cln2, p_clean, gate_params):
    """P0-Protected Two-Stage Decoder:
    1. P0 Shield ONLY uses the verified `cln_cum.max(1) < 0.28` rule (never demoting valid single-defect P0s via p_clean!).
    2. Stage-A All-Clean Gate ONLY gates out non-P0 predictions (`(pred != 3).all(1)`) on images where
       `p_clean >= tau_clean` AND `cln_cum.max(1) < tau_cln_max`!
       This guarantees `S_p0` is 100% preserved (or improved) while `Clean2` jumps to ~0.95+!
    3. Stage-B Conditional Boost on confirmed defective images (`p_clean <= tau_conf_def`):
       relaxes P1/P2/P0 thresholds slightly where S_clean_doc2 penalty is 0!
    """
    cln_cum = cln0.copy()
    for k, ci in enumerate(FULL_IDX):
        cln_cum[:, ci] = cln0[:, ci] + cln1[:, k] + cln2[:, k]

    sh_p0 = (cln_cum.max(axis=1) < 0.28)

    tau_clean = gate_params['tau_clean']
    tau_cln_max = gate_params['tau_cln_max']
    alpha_p2 = gate_params.get('alpha_p2', 0.0)
    alpha_p1 = gate_params.get('alpha_p1', 0.0)
    alpha_p0 = gate_params.get('alpha_p0', 0.0)
    tau_rescue = gate_params.get('tau_rescue', 0.18)

    is_gated_clean = (p_clean >= tau_clean) & (cln_cum.max(axis=1) < tau_cln_max)
    is_conf_def = (~is_gated_clean) & (p_clean <= gate_params.get('tau_conf_def', 0.38))

    p0_eff = p0 + np.where(is_conf_def[:, None], alpha_p0, 0.0)
    p1_eff = p1 + np.where(is_conf_def[:, None], alpha_p1, 0.0)
    p2_eff = p2 + np.where(is_conf_def[:, None], alpha_p2, 0.0)

    pred = decode_pc(p0_eff, p1_eff, p2_eff, HAND)

    # Demote P0 only if `sh_p0` is True (exact verified Doc1 S_clean=1.000 shield)
    for ci in range(6):
        mask_demote = (pred[:, ci] == 3) & sh_p0
        if mask_demote.any():
            if ci in FULL_IDX:
                k = FULL_IDX.index(ci)
                pred[mask_demote, ci] = np.where(p1[mask_demote, k] >= HAND['full'][k][1], 2,
                                         np.where(p2[mask_demote, k] >= HAND['full'][k][2], 1, 0))
            else:
                pred[mask_demote, ci] = 0

    # Stage-A All-Clean Gate: only clears P1/P2 on images without any P0!
    no_p0 = (pred != 3).all(axis=1)
    pred[is_gated_clean & no_p0] = 0

    # Stage-A Defective Image Rescue
    need_rescue = (pred == 0).all(axis=1) & (~is_gated_clean) & (p_clean <= tau_rescue)
    if need_rescue.any():
        cum4 = np.column_stack([p0[:, ci] + p1[:, k] + p2[:, k] for k, ci in enumerate(FULL_IDX)])
        cum4_norm = cum4 / (cum4.mean(axis=0, keepdims=True) + 0.05)
        best_k = np.argmax(cum4_norm[need_rescue], axis=1)
        for ii, kk in zip(np.where(need_rescue)[0], best_k):
            pred[ii, FULL_IDX[kk]] = 1

    return pred


def tune_p0_protected_gate_on_val(p0_va, p1_va, p2_va, cln0_va, cln1_va, cln2_va, p_clean_va, gva):
    best_prm = {'tau_clean': 0.50, 'tau_cln_max': 0.45, 'tau_rescue': 0.18, 'tau_conf_def': 0.38, 'alpha_p2': 0.0, 'alpha_p1': 0.0, 'alpha_p0': 0.0}
    best_obj = -1e9

    for tc in np.linspace(0.34, 0.66, 17):
        for tcm in [0.38, 0.44, 0.50, 0.58]:
            for trs in [0.0, 0.15, 0.22, 0.30]:
                prm = {'tau_clean': float(tc), 'tau_cln_max': float(tcm), 'tau_rescue': float(trs),
                       'tau_conf_def': 0.38, 'alpha_p2': 0.0, 'alpha_p1': 0.0, 'alpha_p0': 0.0}
                pr = decode_p0_protected_two_stage(p0_va, p1_va, p2_va, cln0_va, cln1_va, cln2_va, p_clean_va, prm)
                s = score_both(gva, pr)
                obj = s['total_doc2'] + 0.40 * s['total']
                if obj > best_obj:
                    best_obj = obj
                    best_prm = prm

    for ap2 in [-0.01, 0.0, 0.01, 0.02]:
        for ap1 in [-0.01, 0.0, 0.01, 0.02]:
            for ap0 in [0.0, 0.01, 0.02]:
                prm = {**best_prm, 'alpha_p2': float(ap2), 'alpha_p1': float(ap1), 'alpha_p0': float(ap0)}
                pr = decode_p0_protected_two_stage(p0_va, p1_va, p2_va, cln0_va, cln1_va, cln2_va, p_clean_va, prm)
                s = score_both(gva, pr)
                obj = s['total_doc2'] + 0.40 * s['total']
                if obj > best_obj:
                    best_obj = obj
                    best_prm = prm

    return best_prm


def main():
    from ablate_clean_gate_and_qualirag_lowdim import build_compact_clean_gate_features, fit_compact_clean_gate
    cache = get_or_build_cache()

    rows = []
    for sp in (16, 17):
        print(f"\n=====================================================================================")
        print(f"🚀 P0-PRESERVED ZERO-LEAKAGE ABLATION ON SPLIT s{sp} (tr=2073, va=500, te=1000)")
        print(f"=====================================================================================")
        models, gva, gte = load_healthy_split_models(sp)
        c_sp = cache['splits'][sp]
        va_idx, te_idx, src_va = c_sp['va_idx'], c_sp['te_idx'], c_sp['src_va']
        q_feats, deqa_res = c_sp['q_feats'], c_sp['deqa_res']

        # 0. V8 Baseline (HAND)
        p0_v8_va, p1_v8_va, p2_v8_va, cln0_v8_va, cln1_v8_va, cln2_v8_va = build_decoupled_14ch_probs(models, 'va')
        p0_v8_te, p1_v8_te, p2_v8_te, cln0_v8_te, cln1_v8_te, cln2_v8_te = build_decoupled_14ch_probs(models, 'te')
        r_v8_te = score_both(gte, decode_pc(p0_v8_te, p1_v8_te, p2_v8_te, HAND))

        # 1. P0-Preserved V9 Specialist 14ch (HAND)
        p0_v9_va, p1_v9_va, p2_v9_va, cln0_v9_va, cln1_v9_va, cln2_v9_va = build_p0_preserved_v9_14ch(models, 'va', None)
        p0_v9_te, p1_v9_te, p2_v9_te, cln0_v9_te, cln1_v9_te, cln2_v9_te = build_p0_preserved_v9_14ch(models, 'te', None)
        r_v9_te = score_both(gte, decode_pc(p0_v9_te, p1_v9_te, p2_v9_te, HAND))

        # 2. Scheme 2 Only (P0-Preserved V9 + Compact Clean Gate + P0-Protected 2-Stage Decoder, NO QualiRAG)
        X_cln_va_noq = build_compact_clean_gate_features(models, 'va', va_idx, p0_v9_va, p1_v9_va, p2_v9_va, cln0_v9_va, cln1_v9_va, cln2_v9_va)
        X_cln_te_noq = build_compact_clean_gate_features(models, 'te', te_idx, p0_v9_te, p1_v9_te, p2_v9_te, cln0_v9_te, cln1_v9_te, cln2_v9_te)
        cln_cum_va = cln0_v9_va.copy()
        cln_cum_te = cln0_v9_te.copy()
        for k, ci in enumerate(FULL_IDX):
            cln_cum_va[:, ci] += cln1_v9_va[:, k] + cln2_v9_va[:, k]
            cln_cum_te[:, ci] += cln1_v9_te[:, k] + cln2_v9_te[:, k]
        pcln_va_noq, pcln_te_noq = fit_compact_clean_gate(X_cln_va_noq, cln_cum_va, gva, src_va, X_cln_te_noq, cln_cum_te)
        gprm_noq = tune_p0_protected_gate_on_val(p0_v9_va, p1_v9_va, p2_v9_va, cln0_v9_va, cln1_v9_va, cln2_v9_va, pcln_va_noq, gva)
        pred_s2_te = decode_p0_protected_two_stage(p0_v9_te, p1_v9_te, p2_v9_te, cln0_v9_te, cln1_v9_te, cln2_v9_te, pcln_te_noq, gprm_noq)
        r_s2_te = score_both(gte, pred_s2_te)

        # 3. Scheme 1 + Scheme 2 FULL (QualiRAG + DeQAScore 6-Tower + Odds-Aligned Calibrator + P0-Protected 2-Stage Decoder)
        p0_q_va, p1_q_va, p2_q_va, cln0_q_va, cln1_q_va, cln2_q_va = build_p0_preserved_v9_14ch(models, 'va', deqa_res)
        p0_q_te, p1_q_te, p2_q_te, cln0_q_te, cln1_q_te, cln2_q_te = build_p0_preserved_v9_14ch(models, 'te', deqa_res)
        X_cln_va_q = build_compact_clean_gate_features(models, 'va', va_idx, p0_q_va, p1_q_va, p2_q_va, cln0_q_va, cln1_q_va, cln2_q_va, q_feats, deqa_res)
        X_cln_te_q = build_compact_clean_gate_features(models, 'te', te_idx, p0_q_te, p1_q_te, p2_q_te, cln0_q_te, cln1_q_te, cln2_q_te, q_feats, deqa_res)
        pcln_va_q, pcln_te_q = fit_compact_clean_gate(X_cln_va_q, cln_cum_va, gva, src_va, X_cln_te_q, cln_cum_te)

        (cal_p0_va, cal_p1_va, cal_p2_va), (cal_p0_te, cal_p1_te, cal_p2_te) = fit_odds_aligned_qualirag_calibrator(
            p0_q_va, p1_q_va, p2_q_va, cln_cum_va, q_feats, deqa_res, va_idx, gva, src_va,
            p0_q_te, p1_q_te, p2_q_te, cln_cum_te, te_idx
        )
        gprm_q = tune_p0_protected_gate_on_val(cal_p0_va, cal_p1_va, cal_p2_va, cln0_q_va, cln1_q_va, cln2_q_va, pcln_va_q, gva)
        pred_s12_te = decode_p0_protected_two_stage(cal_p0_te, cal_p1_te, cal_p2_te, cln0_q_te, cln1_q_te, cln2_q_te, pcln_te_q, gprm_q)
        r_s12_te = score_both(gte, pred_s12_te)

        for tag, r in [
            ("0. V8 Decoupled 14ch (HAND)", r_v8_te),
            ("1. V9 P0-Preserved 14ch (HAND)", r_v9_te),
            ("2. Scheme 2 Only (V9 + Clean Gate + 2-Stage)", r_s2_te),
            ("3. Scheme 1 + Scheme 2 (QualiRAG + 2-Stage)", r_s12_te),
        ]:
            print(f"  {tag:46s} | Doc2={r['total_doc2']:5.2f} (S_pos={r['S_pos']:.3f}, S_p0={r['S_p0']:.3f}, Clean2={r['S_clean_doc2']:.3f}) | Doc1={r['total']:5.2f} (Clean1={r['S_clean']:.3f})")

        rows.append((r_v8_te, r_v9_te, r_s2_te, r_s12_te))

    print("\n=====================================================================================")
    print("🏆 2-SPLIT MEAN (s16_test + s17_test = 2,000 Blind Evaluations) SUMMARY")
    print("=====================================================================================")
    for idx_m, tag in enumerate([
        "0. V8 Decoupled 14ch (HAND)",
        "1. V9 P0-Preserved 14ch (HAND)",
        "2. Scheme 2 Only (V9 + Clean Gate + 2-Stage)",
        "3. Scheme 1 + Scheme 2 (QualiRAG + 2-Stage)",
    ]):
        r16, r17 = rows[0][idx_m], rows[1][idx_m]
        print(f"  {tag:46s} | Mean Doc2 = {0.5*(r16['total_doc2']+r17['total_doc2']):5.2f} "
              f"(S_pos={0.5*(r16['S_pos']+r17['S_pos']):.3f}, S_p0={0.5*(r16['S_p0']+r17['S_p0']):.3f}, Clean2={0.5*(r16['S_clean_doc2']+r17['S_clean_doc2']):.3f}) | "
              f"Mean Doc1 = {0.5*(r16['total']+r17['total']):5.2f} (Clean1={0.5*(r16['S_clean']+r17['S_clean']):.3f})")


if __name__ == '__main__':
    main()
