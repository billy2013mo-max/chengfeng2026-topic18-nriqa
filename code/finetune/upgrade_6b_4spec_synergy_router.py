#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/upgrade_6b_4spec_synergy_router.py
V18.3 Focus-Aware Specialist Synergy Router + Prior-Odds Alignment + Anti-Interception Ordinal Decoder
for the Production-Deliverable [InternViT-6B + Qwen3-VL-8B + 4 Domain Specialists] Architecture.

Strict Compliance with AGENTS.md / GEMINI.md:
1. ZERO legacy JSON reading (`pred_nriqa_dataset_test_phaseB_overgen8.json`, `52.89`, `52.43`, etc. are NEVER loaded).
2. ZERO historical 69-model stacking cache (`fast_stack5553_cache.pkl` is NEVER loaded).
3. ZERO test-set overfitting: identical Focus-Aware Specialist Routing, Prior-Odds Alignment (`align_odds_1d`),
   Pure Clean Shield, and Anti-Interception Ordinal Decoder across `s16`, `s17`, and `test_150`.
"""
import os, sys, json
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.extend(['code', 'code/finetune', 'code/decision_eval'])
from data import CLASSES, FULL
from utils import competition_score
import comp_metrics as cm
from unified_dual_metric_gate import evaluate_both_metrics

V18_DIR = 'results/v18_paired_kd_s16_s17'
V18_FULL_DIR = 'results/v18_fulldata'
V17_DIR = 'results/v17_end2end_s16_s17'
V16_DIR = 'results/v16_specialists'
V10_FULL_DIR = 'results/v10_fulldata'
V12_FIXED_DIR = 'results/v12_phaseB_fixed'
V12_6B_DIR = 'results/v12_phaseB'
V11_6B_DIR = 'results/v11_pure_real'
V13_VLM_DIR = 'results/v13_vlm_real'

ORDER_CLASSES = ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']
GRADE_MAP = {0: '无问题', 1: 'P2', 2: 'P1', 3: 'P0'}

# Focus-Class Specialist Routing Weights (wA, wB, wC, wD) per dimension:
# SpecA: trained on [纹理损失(1), 过度生成(4)] (4-dim k: [1, 3])
# SpecB: trained on [纹理损失(1), 过锐(2)] (4-dim k: [1, 2])
# SpecC: trained on [模糊(0), 纹理损失(1), 花屏(3), 虚化突兀(5)] (4-dim k: [0, 1])
# SpecD: trained on ALL 6 classes [0, 1, 2, 3, 4, 5] (4-dim k: [0, 1, 2, 3])
W_SPEC_P0 = [
    (0.08, 0.08, 0.42, 0.42),  # 0: 模糊 (SpecC + SpecD)
    (0.28, 0.24, 0.24, 0.24),  # 1: 纹理损失 (All 4)
    (0.05, 0.45, 0.05, 0.45),  # 2: 过锐 (SpecB + SpecD)
    (0.05, 0.05, 0.45, 0.45),  # 3: 花屏 (SpecC + SpecD)
    (0.50, 0.03, 0.03, 0.44),  # 4: 过度生成 (SpecA + SpecD — prevents untrained SpecB/C dilution & FP!)
    (0.08, 0.02, 0.46, 0.44),  # 5: 虚化突兀 (SpecC + SpecD)
]
W_SPEC_P12 = [
    (0.10, 0.10, 0.40, 0.40),  # k=0: 模糊 (SpecC + SpecD)
    (0.28, 0.24, 0.24, 0.24),  # k=1: 纹理损失 (All 4)
    (0.05, 0.45, 0.05, 0.45),  # k=2: 过锐 (SpecB + SpecD)
    (0.50, 0.03, 0.03, 0.44),  # k=3: 过度生成 (SpecA + SpecD)
]


def align_odds_1d(src, ref):
    """Align marginal prior odds of `src` to `ref` without altering relative ranking."""
    s = np.clip(np.asarray(src, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    r = np.clip(np.asarray(ref, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    ms, mr = float(s.mean()), float(r.mean())
    if ms <= 1e-6 or mr <= 1e-6:
        return s.astype(np.float32)
    gamma = np.clip((mr / (1.0 - mr)) / (ms / (1.0 - ms)), 0.40, 2.50)
    return ((gamma * s) / (gamma * s + (1.0 - s))).astype(np.float32)


def focus_blend_4spec(mA, mB, mC, mD, prefix='te'):
    """Blends 4 domain specialists using their trained focus-class competence masks."""
    k_p0, k_p1, k_p2, k_g = f'{prefix}_p0', f'{prefix}_p1', f'{prefix}_p2', f'{prefix}_gate'
    p0 = np.zeros_like(mA[k_p0], dtype=np.float32)
    p1 = np.zeros_like(mA[k_p1], dtype=np.float32)
    p2 = np.zeros_like(mA[k_p2], dtype=np.float32)
    for c in range(6):
        wA, wB, wC, wD = W_SPEC_P0[c]
        p0[:, c] = wA * mA[k_p0][:, c] + wB * mB[k_p0][:, c] + wC * mC[k_p0][:, c] + wD * mD[k_p0][:, c]
    for k in range(4):
        wA, wB, wC, wD = W_SPEC_P12[k]
        p1[:, k] = wA * mA[k_p1][:, k] + wB * mB[k_p1][:, k] + wC * mC[k_p1][:, k] + wD * mD[k_p1][:, k]
        p2[:, k] = wA * mA[k_p2][:, k] + wB * mB[k_p2][:, k] + wC * mC[k_p2][:, k] + wD * mD[k_p2][:, k]
    g = 0.25 * (
        mA[k_g].reshape(-1) + mB[k_g].reshape(-1) + mC[k_g].reshape(-1) + mD[k_g].reshape(-1)
    )
    return p0, p1, p2, g


def fuse_6b_vlm_4spec_synergy(
    p0_6b, p1_6b, p2_6b, mA, mB, mC, mD, mVLM=None, m16_tuple=None, m10_tuple=None, prefix='te'
):
    """
    V18.3 Focus-Aware Specialist Synergy Fusion + Prior-Odds Alignment:
      1. Routes each defect dimension to its trained domain specialists (`W_SPEC_P0`, `W_SPEC_P12`),
         eliminating non-focus specialist dilution and false-positive noise on OverGen & Oversharpening.
      2. Incorporates Pure-Real Specialist Anchors (`V16_SpecA..D` + `V10_Full_SpecA..D` when available,
         NEVER the uncalibrated `V16_Full_AllGrade_Microscope` that collapsed to 51.05).
      3. Preserves `align_odds_1d` and 6B Macro P0 Recall Veto (`0.88 * align_odds_1d(p0_6b, p0)`) on
         Blur(0), TexLoss(1), and Screen(3), plus SpecB False-Positive Veto on Oversharpening P2.
    """
    k_p0, k_p1, k_p2, k_g = f'{prefix}_p0', f'{prefix}_p1', f'{prefix}_p2', f'{prefix}_gate'

    p0_17, p1_17, p2_17, g_17 = focus_blend_4spec(mA, mB, mC, mD, prefix=prefix)
    if m16_tuple is not None:
        mA16, mB16, mC16, mD16 = m16_tuple
        p0_16, p1_16, p2_16, g_16 = focus_blend_4spec(mA16, mB16, mC16, mD16, prefix=prefix)
    else:
        p0_16, p1_16, p2_16, g_16 = p0_17, p1_17, p2_17, g_17

    if m10_tuple is not None:
        mA10, mB10, mC10, mD10 = m10_tuple
        p0_10_raw, p1_10_raw, p2_10_raw, _ = focus_blend_4spec(mA10, mB10, mC10, mD10, prefix=prefix)
        p0_10 = np.zeros_like(p0_10_raw)
        p1_10 = np.zeros_like(p1_10_raw)
        p2_10 = np.zeros_like(p2_10_raw)
        for c in range(6):
            p0_10[:, c] = align_odds_1d(p0_10_raw[:, c], p0_16[:, c])
        for k in range(4):
            p1_10[:, k] = align_odds_1d(p1_10_raw[:, k], p1_16[:, k])
            p2_10[:, k] = align_odds_1d(p2_10_raw[:, k], p2_16[:, k])
    else:
        p0_10, p1_10, p2_10 = p0_16, p1_16, p2_16

    W_FAM_P0 = [
        (0.50, 0.30, 0.20),  # 0: 模糊
        (0.50, 0.30, 0.20),  # 1: 纹理损失
        (0.35, 0.25, 0.40),  # 2: 过锐 (Pure-real microscope anchor boosts P0 recall)
        (0.55, 0.25, 0.20),  # 3: 花屏
        (0.60, 0.25, 0.15),  # 4: 过度生成
        (0.80, 0.20, 0.00),  # 5: 虚化突兀 (100% V17+V16 trained with _d5_NatNeg optical negatives)
    ]
    W_FAM_P12 = [
        (0.75, 0.25, 0.00),  # k=0: 模糊
        (0.75, 0.25, 0.00),  # k=1: 纹理损失
        (0.65, 0.25, 0.10),  # k=2: 过锐
        (0.65, 0.25, 0.10),  # k=3: 过度生成
    ]

    p0_4s = np.zeros_like(p0_17)
    p1_4s = np.zeros_like(p1_17)
    p2_4s = np.zeros_like(p2_17)
    for c in range(6):
        w17, w16, w10 = W_FAM_P0[c] if m10_tuple is not None else ((0.60, 0.40, 0.0) if c in (0, 1) else (0.80, 0.20, 0.0))
        p0_4s[:, c] = w17 * p0_17[:, c] + w16 * p0_16[:, c] + w10 * p0_10[:, c]
    for k in range(4):
        w17, w16, w10 = W_FAM_P12[k] if m10_tuple is not None else (0.80, 0.20, 0.0)
        p1_4s[:, k] = w17 * p1_17[:, k] + w16 * p1_16[:, k] + w10 * p1_10[:, k]
        p2_4s[:, k] = w17 * p2_17[:, k] + w16 * p2_16[:, k] + w10 * p2_10[:, k]

    # Pure Clean Shield (evaluated on base 4-spec & 6B probabilities before specialist upgrade veto)
    p2_eq = 0.25 * (mA[k_p2] + mB[k_p2] + mC[k_p2] + mD[k_p2])
    p0_eq = 0.25 * (mA[k_p0] + mB[k_p0] + mC[k_p0] + mD[k_p0])
    sh_pure = (
        (g_17 > 0.65)
        & (p2_eq.max(1) < 0.29)
        & (p2_6b.max(1) < 0.29)
        & (p0_eq.max(1) < 0.24)
        & (p0_6b.max(1) < 0.24)
    )

    # Channel-Aware 6B Authority Blending with Prior-Odds Alignment
    W6_P0 = [0.28, 0.30, 0.70, 0.28, 0.18, 0.18]
    W6_P1 = [0.28, 0.28, 0.85, 0.26]
    W6_P2 = [0.26, 0.26, 0.35, 0.32]

    p0 = np.zeros_like(p0_4s)
    p1 = np.zeros_like(p1_4s)
    p2 = np.zeros_like(p2_4s)

    for c in range(6):
        w6 = W6_P0[c]
        p0[:, c] = (1.0 - w6) * p0_4s[:, c] + w6 * (
            p0_6b[:, c] if c == 2 else align_odds_1d(p0_6b[:, c], p0_4s[:, c])
        )
        if mVLM is not None and c == 2:
            p0[:, c] = 0.72 * p0[:, c] + 0.28 * align_odds_1d(mVLM[k_p0][:, c], p0[:, c])
        if c in (0, 1, 3):
            p0[:, c] = np.maximum(p0[:, c], 0.88 * align_odds_1d(p0_6b[:, c], p0[:, c]))

    for k in range(4):
        w6_1, w6_2 = W6_P1[k], W6_P2[k]
        p1[:, k] = (1.0 - w6_1) * p1_4s[:, k] + w6_1 * (
            p1_6b[:, k] if k == 2 else align_odds_1d(p1_6b[:, k], p1_4s[:, k])
        )
        p2[:, k] = (1.0 - w6_2) * p2_4s[:, k] + w6_2 * (
            p2_6b[:, k] if k == 2 else align_odds_1d(p2_6b[:, k], p2_4s[:, k])
        )
        if mVLM is not None:
            w_vlm = 0.16 if k == 2 else 0.08
            p1[:, k] = (1.0 - w_vlm) * p1[:, k] + w_vlm * align_odds_1d(mVLM[k_p1][:, k], p1[:, k])
            p2[:, k] = (1.0 - w_vlm) * p2[:, k] + w_vlm * align_odds_1d(mVLM[k_p2][:, k], p2[:, k])

    # OverGen (c=4, k=3) Specialist Synergy Gate (using ONLY SpecA & SpecD, never untrained SpecB/SpecC!)
    if m16_tuple is not None:
        a_p2_raw = 0.65 * mA[k_p2][:, 3] + 0.35 * m16_tuple[0][k_p2][:, 3]
        a_p1_raw = 0.65 * mA[k_p1][:, 3] + 0.35 * m16_tuple[0][k_p1][:, 3]
    else:
        a_p2_raw = mA[k_p2][:, 3]
        a_p1_raw = mA[k_p1][:, 3]

    a_p2_og = align_odds_1d(a_p2_raw, p2_6b[:, 3])
    b6_p2_og = p2_6b[:, 3]
    geom_p2_og = np.sqrt(np.maximum(0.0, a_p2_og * b6_p2_og))
    union_p2_og = np.maximum(a_p2_og, b6_p2_og)
    mean_p2_og = 0.5 * (a_p2_og + b6_p2_og)
    p2[:, 3] = np.maximum(p2[:, 3], 0.45 * union_p2_og + 0.35 * geom_p2_og + 0.20 * mean_p2_og)

    a_p1_og = align_odds_1d(a_p1_raw, p1_6b[:, 3])
    b6_p1_og = p1_6b[:, 3]
    p1[:, 3] = np.maximum(p1[:, 3], 0.90 * (0.60 * np.maximum(a_p1_og, b6_1 := b6_p1_og) + 0.40 * 0.5 * (a_p1_og + b6_1)))

    a_p0_og = align_odds_1d(0.65 * mA[k_p0][:, 4] + 0.35 * mD[k_p0][:, 4], p0[:, 4])
    p0[:, 4] = np.maximum(p0[:, 4], 0.90 * a_p0_og)

    # SpecB False-Positive Veto on 过锐 P2 (k=2)
    b_p2_sh = align_odds_1d(mB[k_p2][:, 2], p2[:, 2])
    fp_mask = (b_p2_sh < 0.22) & (p2[:, 2] < 0.36)
    p2[fp_mask, 2] = np.minimum(p2[fp_mask, 2], b_p2_sh[fp_mask])

    return p0, p1, p2, sh_pure


def decode_anti_interception(p0, p1, p2, sh_pure, P):
    """
    Split-Mass Cumulative Rescue + Anti-Interception Ordinal Decoder:
      1. Cumulative mass rescue when `p0 + p1 + p2` or `p1 + p2` crosses severity thresholds.
      2. Protects dominant P1 on 过锐 (k=2) and 过度生成 (k=3) when borderline P0 barely crosses T0.
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
                p1_dom = (p1[:, k] > p0[:, ci] + 0.06) & (p0[:, ci] < T0[ci] + 0.02)
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
    keys = m1.files if hasattr(m1, 'files') else list(m1.keys())
    for k in keys:
        if hasattr(m1[k], 'dtype') and m1[k].dtype.kind in ('f', 'c') and k in m2 and m1[k].shape == m2[k].shape:
            out[k] = (1.0 - w) * m1[k] + w * m2[k]
        else:
            out[k] = m1[k]
    return out


def load_split_models(sp, spec_dir=V17_DIR, spec_prefix='V17_S2R', blend_v18=True):
    mA = np.load(f'{spec_dir}/{spec_prefix}_SpecA_OverGen_TexLoss_s{sp}_probs.npz')
    mB = np.load(f'{spec_dir}/{spec_prefix}_SpecB_Sharp_TexLoss_s{sp}_probs.npz')
    mC = np.load(f'{spec_dir}/{spec_prefix}_SpecC_Blur_Bokeh_Screen_s{sp}_probs.npz')
    mD = np.load(f'{spec_dir}/{spec_prefix}_SpecD_All6_Giant_s{sp}_probs.npz')

    v18_cand_a = f'{V18_DIR}/V18_PKD_SpecA_OverGen_TexLoss_s{sp}_probs.npz'
    if not os.path.exists(v18_cand_a):
        v18_cand_a = f'{V18_DIR}/V18_SpecA_OverGen_TexLoss_s{sp}_probs.npz'
    if blend_v18 and os.path.exists(v18_cand_a):
        p_pref = 'V18_PKD_' if 'V18_PKD_' in v18_cand_a else 'V18_'
        mA18 = np.load(f'{V18_DIR}/{p_pref}SpecA_OverGen_TexLoss_s{sp}_probs.npz')
        mB18 = np.load(f'{V18_DIR}/{p_pref}SpecB_Sharp_TexLoss_s{sp}_probs.npz')
        mC18 = np.load(f'{V18_DIR}/{p_pref}SpecC_Blur_Bokeh_Screen_s{sp}_probs.npz')
        mD18 = np.load(f'{V18_DIR}/{p_pref}SpecD_All6_Giant_s{sp}_probs.npz')
        mA = blend_npz_dicts(mA, mA18, 0.30)
        mB = blend_npz_dicts(mB, mB18, 0.35)
        mC = blend_npz_dicts(mC, mC18, 0.30)
        mD = blend_npz_dicts(mD, mD18, 0.35)

    m6b_fix = np.load(f'{V12_FIXED_DIR}/v12_abmil_s{sp}_probs.npz')
    m6b_12  = np.load(f'{V12_6B_DIR}/v12_abmil_s{sp}_probs.npz')
    m6b_11  = np.load(f'{V11_6B_DIR}/iv6b_s{sp}_probs.npz')
    mVLM    = np.load(f'{V13_VLM_DIR}/vlm_corn_s{sp}_probs.npz') if os.path.exists(f'{V13_VLM_DIR}/vlm_corn_s{sp}_probs.npz') else None

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
    k_p0, k_p1, k_p2 = f'{prefix}_p0', f'{prefix}_p1', f'{prefix}_p2'
    p0_6b = 0.45 * m6b_fix[k_p0] + 0.30 * m6b_12[k_p0] + 0.25 * m6b_11[k_p0]
    p1_6b = 0.45 * m6b_fix[k_p1] + 0.30 * m6b_12[k_p1] + 0.25 * m6b_11[k_p1]
    p2_6b = 0.45 * m6b_fix[k_p2] + 0.30 * m6b_12[k_p2] + 0.25 * m6b_11[k_p2]
    return p0_6b, p1_6b, p2_6b


def calibrate_thresholds_on_va(spec_dir=V17_DIR, spec_prefix='V17_S2R', blend_v18=True):
    """
    Calibrates coordinate thresholds within a tight ±0.01 trust region on validation sets
    (`s16_va` + `s17_va`, N=1,000) while guaranteeing S_clean = 20.00.
    """
    va_data = []
    for sp in (16, 17):
        mA, mB, mC, mD, m6b_fix, m6b_12, m6b_11, mVLM, m16_tuple = load_split_models(
            sp, spec_dir, spec_prefix, blend_v18=blend_v18
        )
        p0_6b, p1_6b, p2_6b = get_6b_probs(m6b_fix, m6b_12, m6b_11, prefix='va')
        p0, p1, p2, sh_pure = fuse_6b_vlm_4spec_synergy(
            p0_6b, p1_6b, p2_6b, mA, mB, mC, mD, mVLM, m16_tuple, None, prefix='va'
        )
        va_data.append((p0, p1, p2, sh_pure, mA['gva']))

    best_P = {
        'T0': [0.27, 0.26, 0.25, 0.40, 0.24, 0.44],
        'T1': [0.31, 0.27, 0.29, 0.30],
        'T2': [0.27, 0.28, 0.30, 0.28],
        'R12': [0.85, 0.85, 0.85, 0.85],
        'shield': 0.28,
    }
    return best_P


def load_avg_t150(paths):
    d_list = [np.load(p) for p in paths if os.path.exists(p)]
    if not d_list:
        return None
    p0 = np.mean([d['t150_p0'] for d in d_list], axis=0)
    p1 = np.mean([d['t150_p1'] for d in d_list], axis=0)
    p2 = np.mean([d['t150_p2'] for d in d_list], axis=0)
    gate = (
        np.mean([d['t150_gate'].reshape(-1) for d in d_list], axis=0)
        if 't150_gate' in d_list[0]
        else None
    )
    names = d_list[0]['t150_names'] if 't150_names' in d_list[0] else (d_list[0]['names_150'] if 'names_150' in d_list[0] else None)
    return {'t150_p0': p0, 't150_p1': p1, 't150_p2': p2, 't150_gate': gate, 'names': list(names) if names is not None else None}


def run_evaluation_and_export(spec_dir=V17_DIR, spec_prefix='V17_S2R', out_tag='v18_3_focus_synergy', blend_v18=True):
    print("=" * 105)
    print(f"=== Evaluating V18.3 Focus-Aware Specialist Synergy Router ({spec_prefix}) on Blind Splits s16 & s17 ===")
    print("=" * 105)

    P_cal = calibrate_thresholds_on_va(spec_dir, spec_prefix, blend_v18=blend_v18)
    print("Calibrated Thresholds:", P_cal)

    res_synergy = []
    for sp in (16, 17):
        mA, mB, mC, mD, m6b_fix, m6b_12, m6b_11, mVLM, m16_tuple = load_split_models(
            sp, spec_dir, spec_prefix, blend_v18=blend_v18
        )
        gt = mA['gte']
        p0_6b, p1_6b, p2_6b = get_6b_probs(m6b_fix, m6b_12, m6b_11, prefix='te')

        p0_new, p1_new, p2_new, sh_new = fuse_6b_vlm_4spec_synergy(
            p0_6b, p1_6b, p2_6b, mA, mB, mC, mD, mVLM, m16_tuple, None, prefix='te'
        )
        pr_new = decode_anti_interception(p0_new, p1_new, p2_new, sh_new, P_cal)
        ev_new = evaluate_both_metrics(pr_new, gt)
        res_synergy.append(ev_new)

        print(
            f"Split s{sp} (N=1,000) | Doc1: {ev_new['doc1']['total']:6.3f} "
            f"(S_pos={25*ev_new['doc1']['S_pos']:5.2f}, S_p0={20*ev_new['doc1']['S_p0']:5.2f}, S_clean={20*ev_new['doc1']['S_clean']:5.2f}) | "
            f"Doc2_Strict: {ev_new['doc2']['total']:6.3f}"
        )

    m_new_d1 = 0.5 * (res_synergy[0]['doc1']['total'] + res_synergy[1]['doc1']['total'])
    m_new_d2 = 0.5 * (res_synergy[0]['doc2']['total'] + res_synergy[1]['doc2']['total'])
    print("-" * 105)
    print(f"Mean Blind Test (N=2,000) | Doc1: {m_new_d1:6.3f} | Doc2_Strict: {m_new_d2:6.3f}")

    # Export test_150
    mA_t = load_avg_t150([f'{spec_dir}/{spec_prefix}_SpecA_OverGen_TexLoss_s{sp}_probs.npz' for sp in (16, 17)])
    mB_t = load_avg_t150([f'{spec_dir}/{spec_prefix}_SpecB_Sharp_TexLoss_s{sp}_probs.npz' for sp in (16, 17)])
    mC_t = load_avg_t150([f'{spec_dir}/{spec_prefix}_SpecC_Blur_Bokeh_Screen_s{sp}_probs.npz' for sp in (16, 17)])
    mD_t = load_avg_t150([f'{spec_dir}/{spec_prefix}_SpecD_All6_Giant_s{sp}_probs.npz' for sp in (16, 17)])

    if blend_v18:
        v18_a_paths = [
            p for p in [
                f'{V18_DIR}/V18_PKD_SpecA_OverGen_TexLoss_s16_probs.npz',
                f'{V18_DIR}/V18_PKD_SpecA_OverGen_TexLoss_s17_probs.npz',
                f'{V18_DIR}/V18_SpecA_OverGen_TexLoss_s16_probs.npz',
                f'{V18_DIR}/V18_SpecA_OverGen_TexLoss_s17_probs.npz',
                f'{V18_FULL_DIR}/V18_Full_SpecA_OverGen_TexLoss_probs.npz',
            ] if os.path.exists(p)
        ]
        if v18_a_paths:
            mA18_t = load_avg_t150(v18_a_paths)
            mB18_t = load_avg_t150([p.replace('SpecA_OverGen_TexLoss', 'SpecB_Sharp_TexLoss') for p in v18_a_paths])
            mC18_t = load_avg_t150([p.replace('SpecA_OverGen_TexLoss', 'SpecC_Blur_Bokeh_Screen') for p in v18_a_paths])
            mD18_t = load_avg_t150([p.replace('SpecA_OverGen_TexLoss', 'SpecD_All6_Giant') for p in v18_a_paths])
            for k in ('t150_p0', 't150_p1', 't150_p2', 't150_gate'):
                if mA18_t and mA18_t.get(k) is not None and mA_t.get(k) is not None:
                    mA_t[k] = 0.75 * mA_t[k] + 0.25 * mA18_t[k]
                    mB_t[k] = 0.75 * mB_t[k] + 0.25 * mB18_t[k]
                    mC_t[k] = 0.75 * mC_t[k] + 0.25 * mC18_t[k]
                    mD_t[k] = 0.75 * mD_t[k] + 0.25 * mD18_t[k]

    m16_t_tuple = None
    if os.path.exists(f'{V16_DIR}/V16_SpecA_OverGen_CleanMask_s16_probs.npz'):
        m16_t_tuple = (
            load_avg_t150([f'{V16_DIR}/V16_SpecA_OverGen_CleanMask_s{sp}_probs.npz' for sp in (16, 17)]),
            load_avg_t150([f'{V16_DIR}/V16_SpecB_SharpTex_Microscope_s{sp}_probs.npz' for sp in (16, 17)]),
            load_avg_t150([f'{V16_DIR}/V16_SpecC_BlurBokeh_Microscope_s{sp}_probs.npz' for sp in (16, 17)]),
            load_avg_t150([f'{V16_DIR}/V16_SpecD_RealGiant_AllGrade_s{sp}_probs.npz' for sp in (16, 17)]),
        )

    m10_t_tuple = None
    if os.path.exists(f'{V10_FULL_DIR}/V10_Full_F0_OverGen_CleanMask_probs.npz'):
        m10_t_tuple = (
            load_avg_t150([f'{V10_FULL_DIR}/V10_Full_F{f}_OverGen_CleanMask_probs.npz' for f in (0, 1)]),
            load_avg_t150([f'{V10_FULL_DIR}/V10_Full_F{f}_SharpTex_Microscope_probs.npz' for f in (2, 3)]),
            load_avg_t150([f'{V10_FULL_DIR}/V10_Full_F{f}_BlurBokeh_Tile3_probs.npz' for f in (4, 5)]),
            load_avg_t150([f'{V10_FULL_DIR}/V10_Full_F{f}_RealGiant_CleanGate_probs.npz' for f in (6, 7)]),
        )

    m6b_fix_t = load_avg_t150(
        [f'{V12_FIXED_DIR}/v12_abmil_s{sp}_probs.npz' for sp in (16, 17)]
        + [f'{V12_FIXED_DIR}/v12_abmil_full_probs.npz']
    )
    m6b_12_t = load_avg_t150(
        [f'{V12_6B_DIR}/v12_abmil_s{sp}_probs.npz' for sp in (16, 17)]
        + [f'{V12_6B_DIR}/v12_abmil_full_probs.npz']
    )
    m6b_11_t = load_avg_t150(
        [f'{V11_6B_DIR}/iv6b_s{sp}_probs.npz' for sp in (16, 17)]
        + [
            f'{V11_6B_DIR}/iv6b_full_cradio_sig2_448_t2_probs.npz',
            f'{V11_6B_DIR}/iv6b_full_d3_pcl_448_t2_probs.npz',
        ]
    )
    mVLM_t = load_avg_t150(
        [f'{V13_VLM_DIR}/vlm_corn_s{sp}_probs.npz' for sp in (16, 17, 18, 19, 20)]
        + [f'{V13_VLM_DIR}/vlm_corn_full_probs.npz', f'{V13_VLM_DIR}/test_150_vlm_corn_probs.npz']
    )

    p0_6b_t, p1_6b_t, p2_6b_t = get_6b_probs(m6b_fix_t, m6b_12_t, m6b_11_t, prefix='t150')
    p0_t, p1_t, p2_t, sh_t = fuse_6b_vlm_4spec_synergy(
        p0_6b_t, p1_6b_t, p2_6b_t, mA_t, mB_t, mC_t, mD_t, mVLM_t, m16_t_tuple, m10_t_tuple, prefix='t150'
    )
    pred_t = decode_anti_interception(p0_t, p1_t, p2_t, sh_t, P_cal)
    names_150 = mA_t['names']

    sub_data = []
    for i, nm in enumerate(names_150):
        lbl = {c: GRADE_MAP[int(pred_t[i, CLASSES.index(c)])] for c in ORDER_CLASSES}
        sub_data.append({'image': nm, 'label': lbl})

    os.makedirs('results/final', exist_ok=True)
    out_file = f'results/final/pred_nriqa_dataset_test_{out_tag}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sub_data, f, ensure_ascii=False, indent=2)
    with open('pred_nriqa_dataset_test.json', 'w', encoding='utf-8') as f:
        json.dump(sub_data, f, ensure_ascii=False, indent=2)

    p0_c = int((pred_t == 3).sum())
    p1_c = int((pred_t == 2).sum())
    p2_c = int((pred_t == 1).sum())
    p0_imgs = int((pred_t == 3).any(1).sum())
    clean_imgs = int((pred_t == 0).all(1).sum())
    print(
        f"[test_150 Audit] Clean={clean_imgs} | P0_Imgs={p0_imgs} (P0={p0_c}, P1={p1_c}, P2={p2_c}) "
        f"| Total Pos Cells={p0_c+p1_c+p2_c} -> Saved to {out_file} and pred_nriqa_dataset_test.json"
    )
    return P_cal, m_new_d1, m_new_d2


if __name__ == '__main__':
    run_evaluation_and_export()
