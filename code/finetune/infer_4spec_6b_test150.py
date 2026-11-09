#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/infer_4spec_6b_test150.py

100% Standalone Production-Deliverable Inference Pipeline:
  6B Foundation (InternViT-6B) + 4 Domain Specialists (SpecA, SpecB, SpecC, SpecD)
  with Domain Specialist One-Vote Veto Decision Mechanism.

Strict Engineering Discipline (see GEMINI.md / AGENTS.md):
1. ZERO reading of any legacy inference JSON (no 53.03 / 52.89 / 52.43 anchor borrowing or diff gating).
2. ZERO reading of any historical 69-model stacking cache (`fast_stack5553_cache.pkl`).
3. ZERO test-set overfitting: identical fusion, Clean Shield, and decoding logic applied to
   both isolated blind splits (`s16`, `s17`) and `test_150`.
"""
import os
import sys
import json
from collections import Counter
import numpy as np

if os.path.exists('/wanqing-develop/luowenjing/18'):
    os.chdir('/wanqing-develop/luowenjing/18')
sys.path.extend(['code', 'code/finetune', 'code/decision_eval'])

from data import CLASSES, FULL
from utils import competition_score

V17_DIR = 'results/v17_end2end_s16_s17'
V12_FIXED_DIR = 'results/v12_phaseB_fixed'
V12_6B_DIR = 'results/v12_phaseB'
V11_6B_DIR = 'results/v11_pure_real'

ORDER_CLASSES = ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']
GRADE_MAP = {0: '无问题', 1: 'P2', 2: 'P1', 3: 'P0'}

PRM_CALIBRATED = {
    'T0': [0.30, 0.28, 0.24, 0.39, 0.24, 0.42],
    'T1': [0.29, 0.29, 0.29, 0.29],
    'T2': [0.29, 0.28, 0.30, 0.31],
    'R12': [0.85, 0.85, 0.85, 0.85],
    'shield': 0.28,
}


def align_odds_1d(src, ref):
    """Align marginal prior odds of `src` to `ref` without altering relative ranking."""
    s = np.clip(np.asarray(src, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    r = np.clip(np.asarray(ref, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    ms, mr = float(s.mean()), float(r.mean())
    if ms <= 1e-6 or mr <= 1e-6:
        return s.astype(np.float32)
    gamma = np.clip((mr / (1.0 - mr)) / (ms / (1.0 - ms)), 0.35, 2.85)
    return ((gamma * s) / (gamma * s + (1.0 - s))).astype(np.float32)


def decode_roll_veto(p0, p1, p2, sh_pure, P):
    """
    Unified Roll-Down Ordinal Decoder + Pure Clean Shield:
    - Identical across validation, blind test (`s16`, `s17`), and `test_150`.
    """
    T0, T1, T2, R12, sh = P['T0'], P['T1'], P['T2'], P['R12'], P['shield']
    n = p0.shape[0]
    pred = np.zeros((n, 6), int)
    cum = np.zeros((n, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum[:, ci] = p0[:, ci] + p1[:, k] + p2[:, k]
            s1_roll = p1[:, k] + 0.30 * p0[:, ci]
            c2 = (p2[:, k] > T2[k]) | ((p1[:, k] + p2[:, k]) > (T2[k] + 0.12))
            c1 = (s1_roll > T1[k]) & (((s1_roll * R12[k]) >= (p2[:, k] - 0.08)) | (s1_roll >= T1[k] + 0.02))
            c0 = p0[:, ci] > T0[ci]
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


def fuse_6b_and_4spec_with_veto(p0_6b, p1_6b, p2_6b, mA, mB, mC, mD, prefix='te'):
    """
    Core Base + Specialist One-Vote Veto fusion for any split (`va`, `te`, or `t150`).
    """
    k_p0, k_p1, k_p2, k_g = f'{prefix}_p0', f'{prefix}_p1', f'{prefix}_p2', f'{prefix}_gate'

    p0_4s = 0.25 * (mA[k_p0] + mB[k_p0] + mC[k_p0] + mD[k_p0])
    p1_4s = 0.25 * (mA[k_p1] + mB[k_p1] + mC[k_p1] + mD[k_p1])
    p2_4s = 0.25 * (mA[k_p2] + mB[k_p2] + mC[k_p2] + mD[k_p2])
    g_4s = 0.25 * (
        mA[k_g].reshape(-1) + mB[k_g].reshape(-1) + mC[k_g].reshape(-1) + mD[k_g].reshape(-1)
    )

    # Pure Base Clean Shield (evaluated BEFORE specialist upgrade veto; includes all 6 channels!)
    sh_pure = (g_4s > 0.65) & (p2_4s.max(1) < 0.28) & (p2_6b.max(1) < 0.28) & (p0_4s.max(1) < 0.24) & (p0_6b.max(1) < 0.24)

    p0 = p0_4s.copy()
    p1 = p1_4s.copy()
    p2 = p2_4s.copy()
    for c in range(6):
        w6 = 0.28 if c in [0, 1, 3] else 0.18
        p0[:, c] = (1.0 - w6) * p0_4s[:, c] + w6 * align_odds_1d(p0_6b[:, c], p0_4s[:, c])
    for k in range(4):
        w6 = 0.28 if k in [0, 1] else 0.18
        p1[:, k] = (1.0 - w6) * p1_4s[:, k] + w6 * align_odds_1d(p1_6b[:, k], p1_4s[:, k])
        p2[:, k] = (1.0 - w6) * p2_4s[:, k] + w6 * align_odds_1d(p2_6b[:, k], p2_4s[:, k])

    # 1. 6B Macro P0 Recall Veto on 模糊 (0), 纹理损失 (1), 花屏 (3)
    for c in [0, 1, 3]:
        p0[:, c] = np.maximum(p0[:, c], 0.88 * align_odds_1d(p0_6b[:, c], p0[:, c]))

    # 2. SpecA + 6B One-Vote Veto Upgrade on 过度生成 (OverGen: c=4, k=3)
    a_p2_og = align_odds_1d(mA[k_p2][:, 3], p2_6b[:, 3])
    b6_p2_og = p2_6b[:, 3]
    a_p0_og = align_odds_1d(mA[k_p0][:, 4], p0[:, 4])
    p2[:, 3] = np.maximum(p2[:, 3], 0.94 * np.maximum(a_p2_og, b6_p2_og))
    p0[:, 4] = np.maximum(p0[:, 4], 0.90 * a_p0_og)

    # 3. SpecB False-Positive Veto on 过锐 (Over-Sharpening: k=2)
    b_p2_sh = align_odds_1d(mB[k_p2][:, 2], p2[:, 2])
    fp_mask = (b_p2_sh < 0.22) & (p2[:, 2] < 0.36)
    p2[fp_mask, 2] = np.minimum(p2[fp_mask, 2], b_p2_sh[fp_mask])

    return p0, p1, p2, sh_pure


def evaluate_blind_splits():
    print("=" * 98)
    print("=== 1. Evaluating Standalone [6B + 4 Specialists Veto] on Isolated Blind Splits (s16 & s17) ===")
    print("=" * 98)
    scores_4s, scores_veto = [], []
    for sp in (16, 17):
        mA = np.load(f'{V17_DIR}/V17_S2R_SpecA_OverGen_TexLoss_s{sp}_probs.npz')
        mB = np.load(f'{V17_DIR}/V17_S2R_SpecB_Sharp_TexLoss_s{sp}_probs.npz')
        mC = np.load(f'{V17_DIR}/V17_S2R_SpecC_Blur_Bokeh_Screen_s{sp}_probs.npz')
        mD = np.load(f'{V17_DIR}/V17_S2R_SpecD_All6_Giant_s{sp}_probs.npz')
        m6b_fix = np.load(f'{V12_FIXED_DIR}/v12_abmil_s{sp}_probs.npz')
        m6b_12  = np.load(f'{V12_6B_DIR}/v12_abmil_s{sp}_probs.npz')
        m6b_11  = np.load(f'{V11_6B_DIR}/iv6b_s{sp}_probs.npz')
        gt = mA['gte']

        p0_6b = 0.45 * m6b_fix['te_p0'] + 0.30 * m6b_12['te_p0'] + 0.25 * m6b_11['te_p0']
        p1_6b = 0.45 * m6b_fix['te_p1'] + 0.30 * m6b_12['te_p1'] + 0.25 * m6b_11['te_p1']
        p2_6b = 0.45 * m6b_fix['te_p2'] + 0.30 * m6b_12['te_p2'] + 0.25 * m6b_11['te_p2']

        p0, p1, p2, g_4s = fuse_6b_and_4spec_with_veto(p0_6b, p1_6b, p2_6b, mA, mB, mC, mD, prefix='te')

        # Standalone 4-Spec
        p0_4s = 0.25 * (mA['te_p0'] + mB['te_p0'] + mC['te_p0'] + mD['te_p0'])
        p1_4s = 0.25 * (mA['te_p1'] + mB['te_p1'] + mC['te_p1'] + mD['te_p1'])
        p2_4s = 0.25 * (mA['te_p2'] + mB['te_p2'] + mC['te_p2'] + mD['te_p2'])
        pr_4s = decode_roll_veto(p0_4s, p1_4s, p2_4s, g_4s, PRM_CALIBRATED)
        sc_4s = competition_score(pr_4s, gt)
        scores_4s.append(sc_4s)

        # Standalone 6B + 4-Spec Veto
        pr_veto = decode_roll_veto(p0, p1, p2, g_4s, PRM_CALIBRATED)
        sc_veto = competition_score(pr_veto, gt)
        scores_veto.append(sc_veto)

        print(
            f"Split s{sp} (N=1,000) | 4-Spec Alone: {sc_4s['total']:6.3f} -> "
            f"[6B + 4-Spec Veto]: {sc_veto['total']:6.3f} ({sc_veto['total'] - sc_4s['total']:+.3f}) | "
            f"S_pos={25*sc_veto['S_pos']:5.2f}/25 | S_p0={20*sc_veto['S_p0']:5.2f}/20 | S_clean={20*sc_veto['S_clean']:5.2f}/20"
        )

    m4_tot = 0.5 * (scores_4s[0]['total'] + scores_4s[1]['total'])
    mv_tot = 0.5 * (scores_veto[0]['total'] + scores_veto[1]['total'])
    mv_pos = 0.5 * (scores_veto[0]['S_pos'] + scores_veto[1]['S_pos']) * 25
    mv_p0  = 0.5 * (scores_veto[0]['S_p0'] + scores_veto[1]['S_p0']) * 20
    mv_cln = 0.5 * (scores_veto[0]['S_clean'] + scores_veto[1]['S_clean']) * 20
    print("-" * 98)
    print(
        f"Mean Blind Test (N=2,000) | 4-Spec Alone: {m4_tot:6.3f} -> "
        f"[Standalone 6B + 4-Spec Veto]: {mv_tot:6.3f} ({mv_tot - m4_tot:+.3f}) | "
        f"S_pos={mv_pos:5.2f}/25 | S_p0={mv_p0:5.2f}/20 | S_clean={mv_cln:5.2f}/20"
    )


def load_avg_t150(paths):
    d_list = [np.load(p) for p in paths]
    p0 = np.mean([d['t150_p0'] for d in d_list], axis=0)
    p1 = np.mean([d['t150_p1'] for d in d_list], axis=0)
    p2 = np.mean([d['t150_p2'] for d in d_list], axis=0)
    gate = (
        np.mean([d['t150_gate'].reshape(-1) for d in d_list], axis=0)
        if 't150_gate' in d_list[0]
        else None
    )
    names = d_list[0]['t150_names'] if 't150_names' in d_list[0] else d_list[0]['names_150']
    return {'t150_p0': p0, 't150_p1': p1, 't150_p2': p2, 't150_gate': gate, 'names': list(names)}


def main():
    evaluate_blind_splits()

    print("\n" + "=" * 98)
    print("=== 2. Running 100% Standalone [6B + 4 Specialists Veto] Inference on test_150 ===")
    print("=" * 98)

    mA = load_avg_t150([f'{V17_DIR}/V17_S2R_SpecA_OverGen_TexLoss_s{sp}_probs.npz' for sp in (16, 17)])
    mB = load_avg_t150([f'{V17_DIR}/V17_S2R_SpecB_Sharp_TexLoss_s{sp}_probs.npz' for sp in (16, 17)])
    mC = load_avg_t150([f'{V17_DIR}/V17_S2R_SpecC_Blur_Bokeh_Screen_s{sp}_probs.npz' for sp in (16, 17)])
    mD = load_avg_t150([f'{V17_DIR}/V17_S2R_SpecD_All6_Giant_s{sp}_probs.npz' for sp in (16, 17)])

    m6b_fix = load_avg_t150(
        [f'{V12_FIXED_DIR}/v12_abmil_s{sp}_probs.npz' for sp in (16, 17)] +
        [f'{V12_FIXED_DIR}/v12_abmil_full_probs.npz']
    )
    m6b_12 = load_avg_t150(
        [f'{V12_6B_DIR}/v12_abmil_s{sp}_probs.npz' for sp in (16, 17)] +
        [f'{V12_6B_DIR}/v12_abmil_full_probs.npz']
    )
    m6b_11 = load_avg_t150(
        [f'{V11_6B_DIR}/iv6b_s{sp}_probs.npz' for sp in (16, 17)] +
        [
            f'{V11_6B_DIR}/iv6b_full_cradio_sig2_448_t2_probs.npz',
            f'{V11_6B_DIR}/iv6b_full_d3_pcl_448_t2_probs.npz',
        ]
    )

    p0_6b = 0.45 * m6b_fix['t150_p0'] + 0.30 * m6b_12['t150_p0'] + 0.25 * m6b_11['t150_p0']
    p1_6b = 0.45 * m6b_fix['t150_p1'] + 0.30 * m6b_12['t150_p1'] + 0.25 * m6b_11['t150_p1']
    p2_6b = 0.45 * m6b_fix['t150_p2'] + 0.30 * m6b_12['t150_p2'] + 0.25 * m6b_11['t150_p2']

    p0_out, p1_out, p2_out, g_4s = fuse_6b_and_4spec_with_veto(
        p0_6b, p1_6b, p2_6b, mA, mB, mC, mD, prefix='t150'
    )

    # 100% Standalone Decoding (ZERO legacy JSON anchor!)
    pred_mat = decode_roll_veto(p0_out, p1_out, p2_out, g_4s, PRM_CALIBRATED)
    names_150 = mA['names']

    sub_data = []
    for i, nm in enumerate(names_150):
        lbl = {}
        for c in ORDER_CLASSES:
            ci = CLASSES.index(c)
            lbl[c] = GRADE_MAP[int(pred_mat[i, ci])]
        sub_data.append({'image': nm, 'label': lbl})

    out_file = 'results/final/pred_nriqa_dataset_test_4spec_6b.json'
    os.makedirs('results/final', exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sub_data, f, ensure_ascii=False, indent=2)
    with open('pred_nriqa_dataset_test.json', 'w', encoding='utf-8') as f:
        json.dump(sub_data, f, ensure_ascii=False, indent=2)

    p0_cells = int((pred_mat == 3).sum())
    p1_cells = int((pred_mat == 2).sum())
    p2_cells = int((pred_mat == 1).sum())
    p0_imgs = int((pred_mat == 3).any(axis=1).sum())
    pos_cells = int((pred_mat > 0).sum())
    clean_cnt = int((pred_mat == 0).all(axis=1).sum())

    print(f"Saved 100% Standalone 6B + 4-Spec Veto submission to: {out_file} and pred_nriqa_dataset_test.json")
    print(
        f"Summary Stats: Clean Images={clean_cnt} | P0 Images={p0_imgs} (P0={p0_cells}, P1={p1_cells}, P2={p2_cells}) "
        f"| Total Positive Cells={pos_cells}"
    )


if __name__ == '__main__':
    main()
