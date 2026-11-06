"""Test Decoupled Stage-2A Clean Gate + Conditional Non-Clean Grade Calibration (with P0 Channel Audit).
1. Audits each of the 14 channels across V8, V9_SpecA..E, V8_W2_ExpA..D, and QualiRAG to keep only genuine channel improvements.
2. Tests Stage-2A Clean Gate + Stage-2B Conditional Threshold Relaxation on Confirmed Non-Clean Images
   (fixing the double-conservatism where `HAND` used high T2/T1 even after `is_gated_clean` already filtered out clean images!).
"""
import os, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from run_scheme1_scheme2_breakthrough import FULL_IDX, BIN_IDX, score_both
from fast_test_qualirag_preserved_p0 import CACHE_PATH, align_prob_mean
from ablate_clean_gate_and_qualirag_lowdim import build_compact_clean_gate_features, fit_compact_clean_gate

cache = torch.load(CACHE_PATH, weights_only=False)


def build_best_verified_14ch(models, split_prefix, deqa_res=None):
    """Keep V8's exact top P0 channels where V8 is strongest, and blend V9 High-Res Specialists
    (SpecA..E) + V8_W2_ExpA..D only on the channels where they strictly improve Val OOF F1!
    """
    p0, p1, p2, cln0, cln1, cln2 = build_decoupled_14ch_probs(models, split_prefix)

    def get_m(k, arr):
        return models[k][f'{split_prefix}_{arr}'] if k in models else None

    # V8_W2_ExpA..D
    w2_a0, w2_a1, w2_a2 = get_m('v8_experts/V8_W2_ExpA_overgen', 'p0'), get_m('v8_experts/V8_W2_ExpA_overgen', 'p1'), get_m('v8_experts/V8_W2_ExpA_overgen', 'p2')
    w2_b0, w2_b1, w2_b2 = get_m('v8_experts/V8_W2_ExpB_sharp', 'p0'), get_m('v8_experts/V8_W2_ExpB_sharp', 'p1'), get_m('v8_experts/V8_W2_ExpB_sharp', 'p2')
    w2_c0, w2_c1, w2_c2 = get_m('v8_experts/V8_W2_ExpC_blurtex', 'p0'), get_m('v8_experts/V8_W2_ExpC_blurtex', 'p1'), get_m('v8_experts/V8_W2_ExpC_blurtex', 'p2')
    w2_d0, w2_d1, w2_d2 = get_m('v8_experts/V8_W2_ExpD_bokeh', 'p0'), get_m('v8_experts/V8_W2_ExpD_bokeh', 'p1'), get_m('v8_experts/V8_W2_ExpD_bokeh', 'p2')

    # V9_SpecA..E
    v9_a0, v9_a1, v9_a2 = get_m('v9_specialists/V9_SpecA_OverGen_Giant', 'p0'), get_m('v9_specialists/V9_SpecA_OverGen_Giant', 'p1'), get_m('v9_specialists/V9_SpecA_OverGen_Giant', 'p2')
    v9_b0, v9_b1, v9_b2 = get_m('v9_specialists/V9_SpecB_SharpTex_HiRes', 'p0'), get_m('v9_specialists/V9_SpecB_SharpTex_HiRes', 'p1'), get_m('v9_specialists/V9_SpecB_SharpTex_HiRes', 'p2')
    v9_c0, v9_c1, v9_c2 = get_m('v9_specialists/V9_SpecC_AllP0_Tile3', 'p0'), get_m('v9_specialists/V9_SpecC_AllP0_Tile3', 'p1'), get_m('v9_specialists/V9_SpecC_AllP0_Tile3', 'p2')
    v9_d0, v9_d1, v9_d2 = get_m('v9_specialists/V9_SpecD_BokehOG_HiRes', 'p0'), get_m('v9_specialists/V9_SpecD_BokehOG_HiRes', 'p1'), get_m('v9_specialists/V9_SpecD_BokehOG_HiRes', 'p2')
    v9_e0, v9_e1, v9_e2 = get_m('v9_specialists/V9_SpecE_RealHiRes_Giant', 'p0'), get_m('v9_specialists/V9_SpecE_RealHiRes_Giant', 'p1'), get_m('v9_specialists/V9_SpecE_RealHiRes_Giant', 'p2')

    if v9_e0 is not None:
        cln0 = 0.80 * cln0 + 0.20 * v9_e0
        cln1 = 0.80 * cln1 + 0.20 * v9_e1
        cln2 = 0.80 * cln2 + 0.20 * v9_e2

        # P0: keep V8's super-strong P0 anchor base (0.818!), only lightly injecting specialist P0s where beneficial
        p0[:, 2] = 0.85 * p0[:, 2] + 0.08 * align_prob_mean(v9_b0[:, 2], p0[:, 2]) + 0.07 * align_prob_mean(w2_b0[:, 2], p0[:, 2])
        p0[:, 3] = 0.90 * p0[:, 3] + 0.10 * align_prob_mean(v9_c0[:, 3], p0[:, 3])
        p0[:, 4] = 0.78 * p0[:, 4] + 0.12 * align_prob_mean(v9_a0[:, 4], p0[:, 4]) + 0.10 * align_prob_mean(w2_a0[:, 4], p0[:, 4])
        p0[:, 5] = 0.84 * p0[:, 5] + 0.08 * align_prob_mean(v9_d0[:, 5], p0[:, 5]) + 0.08 * align_prob_mean(w2_d0[:, 5], p0[:, 5])

        # P1 & P2: inject V9 High-Res + V8_W2 specialists
        for k, (va_w, vb_w, vc_w, vd_w, ve_w) in enumerate([
            (0.00, 0.00, 0.12, 0.00, 0.10),  # 0: 模糊 P1/P2
            (0.00, 0.12, 0.10, 0.00, 0.08),  # 1: 纹理损失 P1/P2
            (0.00, 0.16, 0.08, 0.00, 0.08),  # 2: 过锐 P1/P2
            (0.20, 0.00, 0.00, 0.14, 0.06),  # 3: 过度生成 P1/P2
        ]):
            w_sum = va_w + vb_w + vc_w + vd_w + ve_w
            p1[:, k] = (1.0 - w_sum) * p1[:, k] + (
                va_w * align_prob_mean(v9_a1[:, k], p1[:, k]) +
                vb_w * align_prob_mean(v9_b1[:, k], p1[:, k]) +
                vc_w * align_prob_mean(v9_c1[:, k], p1[:, k]) +
                vd_w * align_prob_mean(v9_d1[:, k], p1[:, k]) +
                ve_w * align_prob_mean(v9_e1[:, k], p1[:, k])
            )
            p2[:, k] = (1.0 - w_sum) * p2[:, k] + (
                va_w * align_prob_mean(v9_a2[:, k], p2[:, k]) +
                vb_w * align_prob_mean(v9_b2[:, k], p2[:, k]) +
                vc_w * align_prob_mean(v9_c2[:, k], p2[:, k]) +
                vd_w * align_prob_mean(v9_d2[:, k], p2[:, k]) +
                ve_w * align_prob_mean(v9_e2[:, k], p2[:, k])
            )

    if deqa_res is not None:
        dq1, dq2 = deqa_res[f'{split_prefix}_p1'], deqa_res[f'{split_prefix}_p2']
        for k in range(4):
            p1[:, k] = 0.92 * p1[:, k] + 0.08 * align_prob_mean(dq1[:, k], p1[:, k])
            p2[:, k] = 0.92 * p2[:, k] + 0.08 * align_prob_mean(dq2[:, k], p2[:, k])

    return p0, p1, p2, cln0, cln1, cln2


def decode_two_stage_clean_plus_nonclean_boost(p0, p1, p2, cln0, cln1, cln2, p_clean, prm):
    """Two-Stage Decoupled Decoder:
    - Stage 1 (All-Clean Gate):
      If `p_clean >= tau_clean` AND `cln_cum.max(1) < tau_cln_max` AND `(pred != 3).all(1)`:
      image is classified as All-Clean (`0` across all 6 classes).
    - Stage 2 (Confirmed Non-Clean Conditional Decoding):
      If an image is NOT classified as All-Clean (`~is_gated_clean`), then under Official (2).md:
      predicting P2 / P1 on this image has ZERO penalty on `S_clean_doc2`!
      Furthermore, for images where `p_clean < tau_def` (definitely defective), we lower the effective
      P2 and P1 thresholds by `d_t2` and `d_t1` and include `w01 * p0` roll-down into `P1`!
    """
    n = p0.shape[0]
    pred = np.zeros((n, 6), dtype=int)

    cln_cum = cln0.copy()
    cum = p0.copy()
    for k, ci in enumerate(FULL_IDX):
        cln_cum[:, ci] = cln0[:, ci] + cln1[:, k] + cln2[:, k]
        cum[:, ci] = p0[:, ci] + p1[:, k] + p2[:, k]

    sh_p0 = (cln_cum.max(axis=1) < 0.28)
    is_gated_clean = (p_clean >= prm['tau_clean']) & (cln_cum.max(axis=1) < prm['tau_cln_max'])
    is_def = (~is_gated_clean) & (p_clean <= prm['tau_def'])

    T0 = prm['T0']
    T1 = prm['T1']
    T2 = prm['T2']
    R12 = prm['R12']
    w01 = prm['w01']
    d_t2 = prm['d_t2']
    d_t1 = prm['d_t1']

    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            q0 = p0[:, ci]
            q1 = p1[:, k] + w01[k] * np.where(q0 <= T0[ci], q0, 0.0)  # ONLY roll down p0 when p0 fails T0!
            q2 = p2[:, k]

            eff_t2 = np.where(is_def, T2[k] - d_t2, T2[k])
            eff_t1 = np.where(is_def, T1[k] - d_t1, T1[k])

            c2 = q2 > eff_t2
            c1 = (q1 > eff_t1) & ((q1 * R12[k]) >= (q2 - 0.05))
            c0 = (q0 > T0[ci]) & (~sh_p0)

            pr = np.where(c2, 1, 0)
            pr = np.where(c1, 2, pr)
            pr = np.where(c0, 3, pr)
            pred[:, ci] = pr
        else:
            pred[:, ci] = np.where((p0[:, ci] > T0[ci]) & (~sh_p0), 3, 0)

    # Apply Stage-1 All-Clean Gate (never overwriting a valid P0)
    no_p0 = (pred != 3).all(axis=1)
    pred[is_gated_clean & no_p0] = 0

    # Rescue strongly defective images that ended up all-0
    need_rescue = (pred == 0).all(axis=1) & (~is_gated_clean) & (p_clean <= prm['tau_rescue'])
    if need_rescue.any():
        cum4 = cum[:, FULL_IDX]
        cum4_norm = cum4 / (cum4.mean(axis=0, keepdims=True) + 0.05)
        best_k = np.argmax(cum4_norm[need_rescue], axis=1)
        for ii, kk in zip(np.where(need_rescue)[0], best_k):
            pred[ii, FULL_IDX[kk]] = 1

    return pred


def tune_conditional_decoder_on_val(p0_va, p1_va, p2_va, cln0_va, cln1_va, cln2_va, p_clean_va, gva):
    import copy
    best_prm = {
        'tau_clean': 0.50, 'tau_cln_max': 0.45, 'tau_def': 0.35, 'tau_rescue': 0.18,
        'T0': list(HAND['T0']), 'T1': list(HAND['T1']), 'T2': list(HAND['T2']), 'R12': list(HAND['R12']),
        'w01': [0.25, 0.25, 0.25, 0.25], 'd_t2': 0.02, 'd_t1': 0.01
    }

    def eval_p(prm):
        pr = decode_two_stage_clean_plus_nonclean_boost(p0_va, p1_va, p2_va, cln0_va, cln1_va, cln2_va, p_clean_va, prm)
        s = score_both(gva, pr)
        return s['total_doc2'] + 0.50 * s['total']

    best_obj = eval_p(best_prm)

    for tc in np.linspace(0.36, 0.64, 15):
        for tcm in [0.38, 0.44, 0.50, 0.56]:
            for trs in [0.0, 0.15, 0.22, 0.28]:
                cand = {**best_prm, 'tau_clean': float(tc), 'tau_cln_max': float(tcm), 'tau_rescue': float(trs)}
                v = eval_p(cand)
                if v > best_obj + 1e-5:
                    best_obj, best_prm = v, cand

    for tdef in [0.25, 0.35, 0.45]:
        for dt2 in [0.0, 0.015, 0.03, 0.045]:
            for dt1 in [0.0, 0.015, 0.03]:
                for w_val in [0.0, 0.15, 0.25, 0.35]:
                    cand = {**best_prm, 'tau_def': float(tdef), 'd_t2': float(dt2), 'd_t1': float(dt1), 'w01': [float(w_val)] * 4}
                    v = eval_p(cand)
                    if v > best_obj + 1e-5:
                        best_obj, best_prm = v, cand

    return best_prm


rows = []
for sp in (16, 17):
    models, gva, gte = load_healthy_split_models(sp)
    c_sp = cache['splits'][sp]
    va_idx, te_idx, src_va = c_sp['va_idx'], c_sp['te_idx'], c_sp['src_va']
    q_feats, deqa_res = c_sp['q_feats'], c_sp['deqa_res']

    p0_v8_te, p1_v8_te, p2_v8_te, _, _, _ = build_decoupled_14ch_probs(models, 'te')
    r_v8 = score_both(gte, decode_pc(p0_v8_te, p1_v8_te, p2_v8_te, HAND))

    p0_v9_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va = build_best_verified_14ch(models, 'va', deqa_res)
    p0_v9_te, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te = build_best_verified_14ch(models, 'te', deqa_res)
    r_v9 = score_both(gte, decode_pc(p0_v9_te, p1_v9_te, p2_v9_te, HAND))

    cln_cum_va = cln0_va.copy()
    cln_cum_te = cln0_te.copy()
    for k, ci in enumerate(FULL_IDX):
        cln_cum_va[:, ci] += cln1_va[:, k] + cln2_va[:, k]
        cln_cum_te[:, ci] += cln1_te[:, k] + cln2_te[:, k]

    X_cln_va = build_compact_clean_gate_features(models, 'va', va_idx, p0_v9_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va, q_feats, deqa_res)
    X_cln_te = build_compact_clean_gate_features(models, 'te', te_idx, p0_v9_te, p1_v9_te, p2_q_te := p2_v9_te, cln0_te, cln1_te, cln2_te, q_feats, deqa_res)
    pcln_va, pcln_te = fit_compact_clean_gate(X_cln_va, cln_cum_va, gva, src_va, X_cln_te, cln_cum_te)

    prm = tune_conditional_decoder_on_val(p0_v9_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va, pcln_va, gva)
    pred_te = decode_two_stage_clean_plus_nonclean_boost(p0_v9_te, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te, pcln_te, prm)
    r_s12 = score_both(gte, pred_te)

    print(f"\n=== Split s{sp} Blind Test (1,000 Images) ===")
    for tag, r in [
        ("0. V8 Decoupled 14ch (HAND)", r_v8),
        ("1. V9 + QualiRAG 14ch (HAND)", r_v9),
        ("2. V9 + QualiRAG + Two-Stage Clean Gate & Cond Boost", r_s12),
    ]:
        print(f"  {tag:52s} | Doc2={r['total_doc2']:5.2f} (S_pos={r['S_pos']:.3f}, S_p0={r['S_p0']:.3f}, Clean2={r['S_clean_doc2']:.3f}) | Doc1={r['total']:5.2f} (Clean1={r['S_clean']:.3f})")
    rows.append((r_v8, r_v9, r_s12))

print("\n=== 2-SPLIT MEAN (2,000 Blind Evaluations) ===")
for idx_m, tag in enumerate([
    "0. V8 Decoupled 14ch (HAND)",
    "1. V9 + QualiRAG 14ch (HAND)",
    "2. V9 + QualiRAG + Two-Stage Clean Gate & Cond Boost",
]):
    r16, r17 = rows[0][idx_m], rows[1][idx_m]
    print(f"  {tag:52s} | Mean Doc2 = {0.5*(r16['total_doc2']+r17['total_doc2']):5.2f} "
          f"(S_pos={0.5*(r16['S_pos']+r17['S_pos']):.3f}, S_p0={0.5*(r16['S_p0']+r17['S_p0']):.3f}, Clean2={0.5*(r16['S_clean_doc2']+r17['S_clean_doc2']):.3f}) | "
          f"Mean Doc1 = {0.5*(r16['total']+r17['total']):5.2f} (Clean1={0.5*(r16['S_clean']+r17['S_clean']):.3f})")
