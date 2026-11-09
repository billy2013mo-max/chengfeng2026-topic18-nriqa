#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/upgrade_6b_4spec_synergy_router.py
Zero-Leakage Channel-Aware Synergy Router + Split-Mass Cumulative Rescue + Anti-Interception Ordinal Decoder
for the Production-Deliverable [InternViT-6B + Qwen3-VL-8B + 4 Domain Specialists] Architecture.

Strict Compliance with AGENTS.md / GEMINI.md:
  - ZERO reading of any legacy test_150 JSON.
  - ZERO reading of 69-model historical stacking cache.
  - All channel weights and thresholds are calibrated strictly on `va` (500 images per split)
    and verified on the isolated blind test sets `s16` (1,000) and `s17` (1,000) before
    running identical inference on `test_150` (and final 654-image evaluation set).
"""
import os, sys, json
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.extend(['code', 'code/finetune', 'code/decision_eval'])
from data import CLASSES, FULL
from utils import competition_score
import comp_metrics as cm
from unified_dual_metric_gate import evaluate_both_metrics
from infer_4spec_6b_test150 import fuse_6b_and_4spec_with_veto, decode_roll_veto, PRM_CALIBRATED

V18_DIR = 'results/v18_paired_kd_s16_s17'
V17_DIR = 'results/v17_end2end_s16_s17'
V16_DIR = 'results/v16_specialists'
V12_FIXED_DIR = 'results/v12_phaseB_fixed'
V12_6B_DIR = 'results/v12_phaseB'
V11_6B_DIR = 'results/v11_pure_real'
V13_VLM_DIR = 'results/v13_vlm_real'

ORDER_CLASSES = ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']
GRADE_MAP = {0: '无问题', 1: 'P2', 2: 'P1', 3: 'P0'}


def align_odds_1d(src, ref):
    s = np.clip(np.asarray(src, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    r = np.clip(np.asarray(ref, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    ms, mr = float(s.mean()), float(r.mean())
    if ms <= 1e-6 or mr <= 1e-6:
        return s.astype(np.float32)
    gamma = np.clip((mr / (1.0 - mr)) / (ms / (1.0 - ms)), 0.40, 2.50)
    return ((gamma * s) / (gamma * s + (1.0 - ms))).astype(np.float32) if False else ((gamma * s) / (gamma * s + (1.0 - s))).astype(np.float32)


def fuse_6b_vlm_4spec_synergy(p0_6b, p1_6b, p2_6b, mA, mB, mC, mD, mVLM=None, m16_tuple=None, prefix='te'):
    """
    Channel-Aware Synergy Fusion + Balanced OverGen Synergy Gate:
      1. Routes P0 of Blur(0) & TexLoss(1) with pure-real specialist anchor (preventing 960p synth P0 threshold inflation).
      2. Gives InternViT-6B high authority on Oversharpening (c=2: w6=0.65, k=2: w6=0.85) without downward odds distortion.
      3. Uses a Balanced Union + Geometric Synergy Gate on OverGen_P2/P1 (k=3) so localized OverGen recall is preserved
         while still suppressing single-model low-confidence noise.
    """
    k_p0, k_p1, k_p2, k_g = f'{prefix}_p0', f'{prefix}_p1', f'{prefix}_p2', f'{prefix}_gate'

    p0_17 = 0.25 * (mA[k_p0] + mB[k_p0] + mC[k_p0] + mD[k_p0])
    p1_17 = 0.25 * (mA[k_p1] + mB[k_p1] + mC[k_p1] + mD[k_p1])
    p2_17 = 0.25 * (mA[k_p2] + mB[k_p2] + mC[k_p2] + mD[k_p2])

    if m16_tuple is not None:
        mA16, mB16, mC16, mD16 = m16_tuple
        p0_16 = 0.25 * (mA16[k_p0] + mB16[k_p0] + mC16[k_p0] + mD16[k_p0])
        p1_16 = 0.25 * (mA16[k_p1] + mB16[k_p1] + mC16[k_p1] + mD16[k_p1])
        p2_16 = 0.25 * (mA16[k_p2] + mB16[k_p2] + mC16[k_p2] + mD16[k_p2])
        p0_4s = p0_17.copy()
        for c, a16 in [(0, 0.65), (1, 0.65), (2, 0.40), (3, 0.35), (4, 0.25), (5, 0.50)]:
            p0_4s[:, c] = a16 * p0_16[:, c] + (1.0 - a16) * p0_17[:, c]
        p1_4s = 0.80 * p1_17 + 0.20 * p1_16
        p2_4s = 0.80 * p2_17 + 0.20 * p2_16
        a_p2_raw = 0.65 * mA[k_p2][:, 3] + 0.35 * mA16[k_p2][:, 3]
        a_p1_raw = 0.65 * mA[k_p1][:, 3] + 0.35 * mA16[k_p1][:, 3]
    else:
        p0_4s, p1_4s, p2_4s = p0_17, p1_17, p2_17
        a_p2_raw = mA[k_p2][:, 3]
        a_p1_raw = mA[k_p1][:, 3]

    g_4s = 0.25 * (
        mA[k_g].reshape(-1) + mB[k_g].reshape(-1) + mC[k_g].reshape(-1) + mD[k_g].reshape(-1)
    )

    # Pure Base Clean Shield
    sh_pure = (g_4s > 0.65) & (p2_4s.max(1) < 0.28) & (p2_6b.max(1) < 0.28) & (p0_4s.max(1) < 0.24) & (p0_6b.max(1) < 0.24)

    # 2. Channel-Aware 6B Authority Blending (giving 6B strong weight on Oversharpening c=2, k=2)
    W6_P0 = [0.28, 0.30, 0.65, 0.28, 0.18, 0.18]
    W6_P1 = [0.28, 0.28, 0.85, 0.24]
    W6_P2 = [0.26, 0.26, 0.35, 0.32]

    p0 = np.zeros_like(p0_4s)
    p1 = np.zeros_like(p1_4s)
    p2 = np.zeros_like(p2_4s)

    for c in range(6):
        w6 = W6_P0[c]
        p0[:, c] = (1.0 - w6) * p0_4s[:, c] + w6 * (p0_6b[:, c] if c == 2 else align_odds_1d(p0_6b[:, c], p0_4s[:, c]))
    for k in range(4):
        w6_1, w6_2 = W6_P1[k], W6_P2[k]
        p1[:, k] = (1.0 - w6_1) * p1_4s[:, k] + w6_1 * (p1_6b[:, k] if k == 2 else align_odds_1d(p1_6b[:, k], p1_4s[:, k]))
        p2[:, k] = (1.0 - w6_2) * p2_4s[:, k] + w6_2 * (p2_6b[:, k] if k == 2 else align_odds_1d(p2_6b[:, k], p2_4s[:, k]))

    # Optional Qwen3-VL-8B (V13) orthogonal semantic injection on OverGen & Blur/TexLoss
    if mVLM is not None and k_p0 in mVLM:
        for k, w_vlm in [(0, 0.08), (1, 0.10), (3, 0.14)]:
            p1[:, k] = (1.0 - w_vlm) * p1[:, k] + w_vlm * align_odds_1d(mVLM[k_p1][:, k], p1[:, k])
            p2[:, k] = (1.0 - w_vlm) * p2[:, k] + w_vlm * align_odds_1d(mVLM[k_p2][:, k], p2[:, k])

    # 3. 6B High-Confidence P0 Recall Recovery on 模糊 (0), 纹理损失 (1), 花屏 (3)
    for c in [0, 1, 3]:
        p0[:, c] = np.maximum(p0[:, c], 0.88 * align_odds_1d(p0_6b[:, c], p0[:, c]))

    # 4. Balanced Synergy + Recall-Preserving Gate on 过度生成 (OverGen: c=4, k=3)
    a_p2_og = align_odds_1d(a_p2_raw, p2_6b[:, 3])
    b6_p2_og = p2_6b[:, 3]
    geom_p2_og = np.sqrt(np.maximum(0.0, a_p2_og * b6_p2_og))
    union_p2_og = np.maximum(a_p2_og, b6_p2_og)
    mean_p2_og = 0.5 * (a_p2_og + b6_p2_og)
    # Preserves high-confidence SpecA / 6B detections while boosting co-activated samples
    p2[:, 3] = np.maximum(p2[:, 3], 0.50 * union_p2_og + 0.30 * geom_p2_og + 0.20 * mean_p2_og)

    a_p1_og = align_odds_1d(a_p1_raw, p1_6b[:, 3])
    b6_p1_og = p1_6b[:, 3]
    p1[:, 3] = np.maximum(p1[:, 3], 0.86 * (0.55 * np.maximum(a_p1_og, b6_p1_og) + 0.45 * 0.5 * (a_p1_og + b6_p1_og)))

    a_p0_og = align_odds_1d(mA[k_p0][:, 4], p0[:, 4])
    p0[:, 4] = np.maximum(p0[:, 4], 0.90 * a_p0_og)

    # 5. SpecB 1:1 Microscope False-Positive Suppression on 过锐_P2 (k=2)
    b_p2_sh = align_odds_1d(mB[k_p2][:, 2], p2[:, 2])
    fp_mask = (b_p2_sh < 0.22) & (p2[:, 2] < 0.36)
    p2[fp_mask, 2] = np.minimum(p2[fp_mask, 2], b_p2_sh[fp_mask])

    return p0, p1, p2, sh_pure


def decode_anti_interception(p0, p1, p2, sh_pure, P):
    """
    Split-Mass Cumulative Rescue + Anti-Interception Ordinal Decoder:
      1. Split-Mass Cumulative Rescue: When probability mass is split across (P0, P1, P2) such that
         total defect probability `cum = p0 + p1 + p2` exceeds `T2[k] + 0.12` (or `p1 + p2 > T2[k] + 0.09`),
         rescues the cell from false-negative `无问题 (0)` to `P2 (1)` or `P1 (2)`.
      2. Protects high-confidence P1 on 过锐 (k=2) and 过度生成 (k=3) when borderline P0 only barely crosses T0.
    """
    T0, T1, T2, R12, sh = P['T0'], P['T1'], P['T2'], P['R12'], P['shield']
    n = p0.shape[0]
    pred = np.zeros((n, 6), int)
    cum = np.zeros((n, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum[:, ci] = p0[:, ci] + p1[:, k] + p2[:, k]
            s1_roll = p1[:, k] + 0.34 * p0[:, ci]
            c2 = (
                (p2[:, k] > T2[k])
                | ((p1[:, k] + p2[:, k]) > (T2[k] + 0.09))
                | ((cum[:, ci] > (T2[k] + 0.13)) & ((p1[:, k] + p2[:, k]) > 0.72 * T2[k]))
            )
            c1 = (s1_roll > T1[k]) & (((s1_roll * R12[k]) >= (p2[:, k] - 0.08)) | (s1_roll >= T1[k] + 0.02))
            c0 = p0[:, ci] > T0[ci]
            if k in (2, 3):
                # On Oversharpening (k=2) and OverGen (k=3), prevent borderline P0 from swallowing dominant P1
                p1_dom = (p1[:, k] > p0[:, ci] + 0.05) & (p0[:, ci] < T0[ci] + 0.02)
                c0 = c0 & (~p1_dom)
            pr = np.where(c2, 1, 0)
            pr = np.where(c1, 2, pr)
            pr = np.where(c0, 3, pr)
            pred[:, ci] = pr
        else:
            cum[:, ci] = p0[:, ci]
            pred[:, ci] = np.where(p0[:, ci] > T0[ci], 3, 0)
    if sh > 0:
        pred[cum.max(1) < sh] = 0
    if sh_pure is not None:
        pred[sh_pure] = 0
    return pred


def blend_npz_dicts(m1, m2, w):
    out = {}
    for k in m1.files:
        if m1[k].dtype.kind in ('f', 'c'):
            out[k] = (1.0 - w) * m1[k] + w * m2[k]
        else:
            out[k] = m1[k]
    return out


def load_split_models(sp, spec_dir=V17_DIR, spec_prefix='V17_S2R', blend_v18=True):
    mA = np.load(f'{spec_dir}/{spec_prefix}_SpecA_OverGen_TexLoss_s{sp}_probs.npz')
    mB = np.load(f'{spec_dir}/{spec_prefix}_SpecB_Sharp_TexLoss_s{sp}_probs.npz')
    mC = np.load(f'{spec_dir}/{spec_prefix}_SpecC_Blur_Bokeh_Screen_s{sp}_probs.npz')
    mD = np.load(f'{spec_dir}/{spec_prefix}_SpecD_All6_Giant_s{sp}_probs.npz')

    if blend_v18 and os.path.exists(f'{V18_DIR}/V18_SpecA_OverGen_TexLoss_s{sp}_probs.npz'):
        mA18 = np.load(f'{V18_DIR}/V18_SpecA_OverGen_TexLoss_s{sp}_probs.npz')
        mB18 = np.load(f'{V18_DIR}/V18_SpecB_Sharp_TexLoss_s{sp}_probs.npz')
        mC18 = np.load(f'{V18_DIR}/V18_SpecC_Blur_Bokeh_Screen_s{sp}_probs.npz')
        mD18 = np.load(f'{V18_DIR}/V18_SpecD_All6_Giant_s{sp}_probs.npz')
        mA = blend_npz_dicts(mA, mA18, 0.45)
        mB = blend_npz_dicts(mB, mB18, 0.65)
        mC = blend_npz_dicts(mC, mC18, 0.45)
        mD = blend_npz_dicts(mD, mD18, 0.65)

    m6b_fix = np.load(f'{V12_FIXED_DIR}/v12_abmil_s{sp}_probs.npz')
    m6b_12  = np.load(f'{V12_6B_DIR}/v12_abmil_s{sp}_probs.npz')
    m6b_11  = np.load(f'{V11_6B_DIR}/iv6b_s{sp}_probs.npz')
    vlm_path = f'{V13_VLM_DIR}/vlm_corn_s{sp}_probs.npz'
    mVLM = np.load(vlm_path) if os.path.exists(vlm_path) else None
    m16_tuple = None
    if os.path.exists(f'{V16_DIR}/V16_SpecA_OverGen_CleanMask_s{sp}_probs.npz'):
        m16_tuple = (
            np.load(f'{V16_DIR}/V16_SpecA_OverGen_CleanMask_s{sp}_probs.npz'),
            np.load(f'{V16_DIR}/V16_SpecB_SharpTex_Microscope_s{sp}_probs.npz'),
            np.load(f'{V16_DIR}/V16_SpecC_BlurBokeh_Microscope_s{sp}_probs.npz'),
            np.load(f'{V16_DIR}/V16_SpecD_RealGiant_AllGrade_s{sp}_probs.npz'),
        )
    return mA, mB, mC, mD, m6b_fix, m6b_12, m6b_11, mVLM, m16_tuple


def get_6b_probs(m6b_fix, m6b_12, m6b_11, prefix='te'):
    p0_6b = 0.45 * m6b_fix[f'{prefix}_p0'] + 0.30 * m6b_12[f'{prefix}_p0'] + 0.25 * m6b_11[f'{prefix}_p0']
    p1_6b = 0.45 * m6b_fix[f'{prefix}_p1'] + 0.30 * m6b_12[f'{prefix}_p1'] + 0.25 * m6b_11[f'{prefix}_p1']
    p2_6b = 0.45 * m6b_fix[f'{prefix}_p2'] + 0.30 * m6b_12[f'{prefix}_p2'] + 0.25 * m6b_11[f'{prefix}_p2']
    return p0_6b, p1_6b, p2_6b


def calibrate_thresholds_on_va(spec_dir=V17_DIR, spec_prefix='V17_S2R', blend_v18=True):
    """
    Strictly calibrates coordinate thresholds within a ±0.03 trust region on validation sets
    (`s16_va` + `s17_va`, N=1,000) while guaranteeing S_clean = 20.00.
    """
    va_data = []
    for sp in (16, 17):
        mA, mB, mC, mD, m6b_fix, m6b_12, m6b_11, mVLM, m16_tuple = load_split_models(sp, spec_dir, spec_prefix, blend_v18=blend_v18)
        p0_6b, p1_6b, p2_6b = get_6b_probs(m6b_fix, m6b_12, m6b_11, prefix='va')
        p0, p1, p2, sh_pure = fuse_6b_vlm_4spec_synergy(p0_6b, p1_6b, p2_6b, mA, mB, mC, mD, mVLM, m16_tuple, prefix='va')
        va_data.append((p0, p1, p2, sh_pure, mA['gva']))

    best_P = {
        'T0': [0.27, 0.25, 0.26, 0.40, 0.22, 0.44],
        'T1': [0.31, 0.27, 0.29, 0.29],
        'T2': [0.28, 0.27, 0.31, 0.27],
        'R12': [0.85, 0.85, 0.85, 0.85],
        'shield': 0.28,
    }

    def eval_va(P_cand):
        tot1, tot2, cln1 = 0.0, 0.0, 0.0
        for p0, p1, p2, sh_pure, gva in va_data:
            pr = decode_anti_interception(p0, p1, p2, sh_pure, P_cand)
            res = evaluate_both_metrics(pr, gva)
            tot1 += 0.5 * res['doc1_total']
            tot2 += 0.5 * res['doc2_strict_total']
            cln1 += 0.5 * res['doc1_S_clean']
        if cln1 < 19.99:
            return -1e9
        return 0.70 * tot1 + 0.30 * tot2

    best_obj = eval_va(best_P)
    for k_p in ['T0', 'T1', 'T2']:
        for idx in range(len(best_P[k_p])):
            cur_v = best_P[k_p][idx]
            for delta in [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]:
                P_try = {k: (list(v) if isinstance(v, list) else v) for k, v in best_P.items()}
                P_try[k_p][idx] = round(cur_v + delta, 3)
                sc = eval_va(P_try)
                if sc > best_obj + 1e-4:
                    best_obj = sc
                    best_P = P_try
    return best_P


def get_per14_map(res_dict):
    out = {}
    for c, g, P, R, F, tp, fp, fn_c in res_dict['per_14']:
        out[f'{c}_{g}'] = {'P': P, 'R': R, 'f1': F, 'tp': tp, 'fp': fp, 'fn': fn_c}
    return out


def main():
    print("=" * 105)
    print("=== Step 1.2: Zero-Leakage Validation Calibration (`va`) & Blind Test Verification (`s16 + s17`) ===")
    print("=" * 105)

    P_va = calibrate_thresholds_on_va(V17_DIR, 'V17_S2R', blend_v18=True)
    print(f"Calibrated on `va` (N=1,000, S_clean=20.00): {json.dumps(P_va)}")

    old_preds, new_preds, gts = [], [], []
    for sp in (16, 17):
        mA, mB, mC, mD, m6b_fix, m6b_12, m6b_11, mVLM, m16_tuple = load_split_models(sp, V17_DIR, 'V17_S2R', blend_v18=True)
        p0_6b, p1_6b, p2_6b = get_6b_probs(m6b_fix, m6b_12, m6b_11, prefix='te')
        gt = mA['gte']
        gts.append(gt)

        # Old baseline (53.351)
        p0_old, p1_old, p2_old, sh_old = fuse_6b_and_4spec_with_veto(p0_6b, p1_6b, p2_6b, mA, mB, mC, mD, prefix='te')
        pr_old = decode_roll_veto(p0_old, p1_old, p2_old, sh_old, PRM_CALIBRATED)
        old_preds.append(pr_old)

        # Upgraded Synergy Router + Split-Mass Cumulative Rescue + Anti-Interception Decoder
        p0_new, p1_new, p2_new, sh_new = fuse_6b_vlm_4spec_synergy(
            p0_6b, p1_6b, p2_6b, mA, mB, mC, mD, mVLM, m16_tuple, prefix='te'
        )
        pr_new = decode_anti_interception(p0_new, p1_new, p2_new, sh_new, P_va)
        new_preds.append(pr_new)

        r_old = evaluate_both_metrics(pr_old, gt)
        r_new = evaluate_both_metrics(pr_new, gt)
        print(
            f"Split s{sp} (N=1,000) | Old Veto: Doc1={r_old['doc1_total']:6.3f} (Doc2_strict={r_old['doc2_strict_total']:6.3f}) -> "
            f"Synergy Router: Doc1={r_new['doc1_total']:6.3f} ({r_new['doc1_total']-r_old['doc1_total']:+.3f}) | "
            f"Doc2_strict={r_new['doc2_strict_total']:6.3f} ({r_new['doc2_strict_total']-r_old['doc2_strict_total']:+.3f}) | "
            f"S_pos={r_new['doc1_S_pos']:5.2f} S_p0={r_new['doc1_S_p0']:5.2f} S_clean={r_new['doc1_S_clean']:5.2f}"
        )

    old_mean_d1 = 0.5 * sum(evaluate_both_metrics(old_preds[i], gts[i])['doc1_total'] for i in range(2))
    new_mean_d1 = 0.5 * sum(evaluate_both_metrics(new_preds[i], gts[i])['doc1_total'] for i in range(2))
    old_mean_d2 = 0.5 * sum(evaluate_both_metrics(old_preds[i], gts[i])['doc2_strict_total'] for i in range(2))
    new_mean_d2 = 0.5 * sum(evaluate_both_metrics(new_preds[i], gts[i])['doc2_strict_total'] for i in range(2))
    new_cln = 0.5 * sum(evaluate_both_metrics(new_preds[i], gts[i])['doc1_S_clean'] for i in range(2))

    all_old = np.vstack(old_preds)
    all_new = np.vstack(new_preds)
    all_gt  = np.vstack(gts)
    m_old_all = get_per14_map(evaluate_both_metrics(all_old, all_gt))
    m_new_all = get_per14_map(evaluate_both_metrics(all_new, all_gt))

    sharp_p1_old = 100.0 * m_old_all['过锐_P1']['f1']
    sharp_p1_new = 100.0 * m_new_all['过锐_P1']['f1']
    og_p2_fp_old = m_old_all['过度生成_P2']['fp']
    og_p2_fp_new = m_new_all['过度生成_P2']['fp']
    og_p2_f1_old = 100.0 * m_old_all['过度生成_P2']['f1']
    og_p2_f1_new = 100.0 * m_new_all['过度生成_P2']['f1']

    print("-" * 105)
    print(
        f"Mean Blind Test (N=2,000) | Old Veto: {old_mean_d1:6.3f} -> Upgraded Synergy Router: {new_mean_d1:6.3f} "
        f"({new_mean_d1 - old_mean_d1:+.3f}) | Doc2 Strict: {old_mean_d2:6.3f} -> {new_mean_d2:6.3f} ({new_mean_d2 - old_mean_d2:+.3f})"
    )
    print(
        f"Gate Checks: 过锐_P1 F1: {sharp_p1_old:.2f}% -> {sharp_p1_new:.2f}% ({sharp_p1_new - sharp_p1_old:+.2f}%) | "
        f"过度生成_P2 FP: {og_p2_fp_old} -> {og_p2_fp_new} ({og_p2_fp_new - og_p2_fp_old:+d}, F1 {og_p2_f1_old:.2f}% -> {og_p2_f1_new:.2f}%) | "
        f"S_clean: {new_cln:.2f}/20.00"
    )


if __name__ == '__main__':
    main()
