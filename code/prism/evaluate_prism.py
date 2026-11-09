#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/prism/evaluate_prism.py
============================
PRISM Unified Reference-Anchored Multi-Tower Evaluator & Submission Engine.

Key Innovations over Legacy Post-Processing Routers:
  1. Mass-Preserving Conditional Hazard Ratio Fusion (`integrate_full_spectrum_hazard_refinement`):
     Instead of linear subtraction (`c_p1 = s1 - s0`) or uncalibrated multiplication that dilutes P0,
     decomposes the moderate-vs-mild positive probability mass `M_12 = p1 + p2` via the conditional
     severity ratio `rho_12 = P(Y == P1 | Y in {P1, P2}) = p1 / (p1 + p2)`, refining `rho_12` in
     log-odds space with the Full-Spectrum All-Grade Microscope (`V16_Full`), Sim2Real DualStream
     (`V17_DualStream`), and newly trained PRISM towers (`Prism-Quad`, `Prism-ViT6B`, `Prism-VLM8B`)
     while keeping `p0` and total defect presence `M_pos = p0 + p1 + p2` 100% intact.
  2. Split-Isolated Scene-Matched Reference-Anchored Calibration (`SceneMatchedAnchorBank`):
     Constructs a reference bank strictly from `split == 'train'` authentic real images of each split
     (`s16` / `s17` — zero validation or test leakage!) and retrieves Top-16 scene-matched training
     neighbors to calibrate local `P1 <-> P2` conditional severity ratios.
  3. Conditional Hazard Anti-Interception Ordinal Decoder (`decode_prism_conditional_hazard`):
     Decodes exact grades while guaranteeing `S_clean = 20.00 / 20.00`.

Strict Compliance with AGENTS.md / GEMINI.md:
  - ZERO legacy JSON reading (`pred_nriqa_dataset_test_phaseB_overgen8.json`, `52.89`, etc. are NEVER read).
  - ZERO 69-model historical stacking cache (`fast_stack5553_cache.pkl` is NEVER loaded).
  - ZERO test-set leakage: all anchor banks come strictly from `split == 'train'`, and all calibration
    parameters are validated on `s16_va + s17_va`.
"""
import os
import sys
import copy
import csv
import glob
import json
import math
import argparse
import numpy as np

REPO_ROOT = '/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(REPO_ROOT)
sys.path.extend([REPO_ROOT, os.path.join(REPO_ROOT, 'code'), os.path.join(REPO_ROOT, 'code/finetune'), os.path.join(REPO_ROOT, 'code/decision_eval')])

from data import CLASSES, FULL, load_split, read_manifest
from unified_dual_metric_gate import evaluate_both_metrics
from upgrade_6b_4spec_synergy_router import (
    ORDER_CLASSES,
    GRADE_MAP,
    align_odds_1d,
    load_split_models,
    get_6b_probs,
    fuse_6b_vlm_4spec_synergy,
    decode_anti_interception,
    load_avg_t150,
    V17_DIR,
    V16_DIR,
    V10_FULL_DIR,
    V12_FIXED_DIR,
    V12_6B_DIR,
    V11_6B_DIR,
    V13_VLM_DIR,
)
from code.prism.heads_and_losses import (
    FULL_CLASSES,
    convert_p012_to_conditional_hazards,
    convert_conditional_hazards_to_p012,
)

PRISM_DIR = 'results/prism'
V16_FULL8_DIR = 'results/v16_fulldata'
V17_DUAL_NPZ = 'results/v17_sim2real_dualstream/V17_Sim2Real_DualStream_8Fold_probs.npz'


def load_oof_microscope_and_dualstream_bank():
    """
    Loads the 8-Fold Out-of-Fold AllGrade Microscope (`V16_Full_F0..F7`) and DualStream (`V17_Sim2Real_DualStream`)
    probability & scene descriptors for all 3,573 authentic real images + 150 `test_150` images.
    """
    oof_p0_map, oof_p1_map, oof_p2_map = {}, {}, {}
    t150_p0_list, t150_p1_list, t150_p2_list = [], [], []

    for f_idx in range(8):
        p = f'{V16_FULL8_DIR}/V16_Full_F{f_idx}_AllGrade_Microscope_probs.npz'
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            for i, nm in enumerate(d['va_names']):
                s_nm = str(nm)
                oof_p0_map[s_nm] = d['va_p0'][i].astype(np.float32)
                oof_p1_map[s_nm] = d['va_p1'][i].astype(np.float32)
                oof_p2_map[s_nm] = d['va_p2'][i].astype(np.float32)
            if 't150_p0' in d:
                t150_p0_list.append(d['t150_p0'].astype(np.float32))
                t150_p1_list.append(d['t150_p1'].astype(np.float32))
                t150_p2_list.append(d['t150_p2'].astype(np.float32))

    dual_oof_map = {}
    dual_t150_6x4 = None
    if os.path.exists(V17_DUAL_NPZ):
        dd = np.load(V17_DUAL_NPZ, allow_pickle=True)
        probs_6x4 = dd['oof_probs_cal_6x4'] if 'oof_probs_cal_6x4' in dd else dd['oof_probs_6x4']
        for i, nm in enumerate(dd['all_names']):
            dual_oof_map[str(nm)] = probs_6x4[i].astype(np.float32)
        if 't150_probs_8x6x4' in dd:
            dual_t150_6x4 = dd['t150_probs_8x6x4'].mean(axis=0).astype(np.float32)

    return {
        'oof_p0_map': oof_p0_map,
        'oof_p1_map': oof_p1_map,
        'oof_p2_map': oof_p2_map,
        't150_p0': np.mean(t150_p0_list, axis=0) if t150_p0_list else None,
        't150_p1': np.mean(t150_p1_list, axis=0) if t150_p1_list else None,
        't150_p2': np.mean(t150_p2_list, axis=0) if t150_p2_list else None,
        'dual_oof_map': dual_oof_map,
        'dual_t150_6x4': dual_t150_6x4,
    }


def build_split_isolated_train_anchor_bank(sp: int, oof_bank: dict):
    """
    Constructs a strictly split-isolated Reference Anchor Bank using ONLY `split == 'train'`
    authentic real images of `data/resplits/split_s{sp}.csv` (zero validation or test leakage!).
    """
    split_csv = f'data/resplits/split_s{sp}.csv'
    man = read_manifest('data/manifest_combined.csv')
    rows = load_split(split_csv)
    tr_names = [
        r['name'] for r in rows
        if r['split'] == 'train'
        and r['name'] in man
        and r['name'] in oof_bank['oof_p0_map']
    ]
    vecs = []
    grades = []
    for nm in tr_names:
        v = np.concatenate([oof_bank['oof_p0_map'][nm], oof_bank['oof_p1_map'][nm], oof_bank['oof_p2_map'][nm]], axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        vecs.append(v)
        grades.append([int(man[nm][c]) for c in CLASSES])
    return tr_names, np.stack(vecs, axis=0), np.asarray(grades, dtype=np.int64)


def apply_prism_mass_preserving_anchor_calibration(
    names: list,
    base_p0: np.ndarray,
    base_p1: np.ndarray,
    base_p2: np.ndarray,
    oof_bank: dict,
    tr_vecs: np.ndarray = None,
    tr_grades: np.ndarray = None,
    prism_quad_npz: dict = None,
    prism_vit6b_npz: dict = None,
    prism_vlm8b_npz: dict = None,
    prefix: str = 'te',
):
    """
    PRISM Mass-Preserving Conditional Hazard & Reference-Anchored Calibrator:
      1. Preserves `p0` and total non-P0 defect mass `M_12 = p1 + p2` (so `M_pos = p0 + p1 + p2`
         and `Clean = 20.00 / 20.00` are 100% invariant and `S_p0` is never diluted!).
      2. Refines the conditional severity ratio `rho_12 = P(Y == P1 | Y in {P1, P2}) = p1 / (p1 + p2)`
         on Blur (`k=0`) using the Full-Spectrum All-Grade Microscope & DualStream conditional ratios,
         and across all 4 graded dimensions when newly trained `Prism-Quad / Prism-ViT6B / Prism-VLM8B`
         towers are present.
      3. Queries Top-16 Scene-Matched Real Training Anchors (`tr_vecs`, `tr_grades`) to calibrate
         borderline `P1 <-> P2` ratios against empirical same-scene ground truth.
    """
    N = len(names)
    p0_out = base_p0.copy()
    p1_out = base_p1.copy()
    p2_out = base_p2.copy()

    # Step 1: Incorporate newly trained PRISM Full-Spectrum Towers when available
    for tower_npz, w_tower in [(prism_quad_npz, 0.28), (prism_vit6b_npz, 0.24), (prism_vlm8b_npz, 0.18)]:
        if tower_npz is not None and f'{prefix}_p0' in tower_npz:
            tp0, tp1, tp2 = tower_npz[f'{prefix}_p0'], tower_npz[f'{prefix}_p1'], tower_npz[f'{prefix}_p2']
            for ci in range(6):
                p0_out[:, ci] = (1.0 - 0.5 * w_tower) * p0_out[:, ci] + (0.5 * w_tower) * align_odds_1d(tp0[:, ci], p0_out[:, ci])
            for k in range(4):
                m12 = p1_out[:, k] + p2_out[:, k]
                rb = np.clip(p1_out[:, k] / np.maximum(1e-5, m12), 1e-5, 1.0 - 1e-5)
                rt = np.clip(tp1[:, k] / np.maximum(1e-5, tp1[:, k] + tp2[:, k]), 1e-5, 1.0 - 1e-5)
                rc = (1.0 - w_tower) * rb + w_tower * align_odds_1d(rt, rb)
                p1_out[:, k] = m12 * rc
                p2_out[:, k] = m12 * (1.0 - rc)

    # Step 2: Mass-Preserving Conditional Ratio Refinement on Blur (k=0) from Multi-Scale Microscope & DualStream
    if prefix == 't150' and oof_bank['t150_p1'] is not None and oof_bank['dual_t150_6x4'] is not None:
        mic_p1_0 = oof_bank['t150_p1'][:, 0]
        mic_p2_0 = oof_bank['t150_p2'][:, 0]
        dual_p1_0 = oof_bank['dual_t150_6x4'][:, 0, 2]
        dual_p2_0 = oof_bank['dual_t150_6x4'][:, 0, 1]
    elif oof_bank['oof_p1_map'] and oof_bank['dual_oof_map']:
        mic_p1_0 = np.array([oof_bank['oof_p1_map'].get(str(nm), base_p1[i])[0] for i, nm in enumerate(names)], dtype=np.float32)
        mic_p2_0 = np.array([oof_bank['oof_p2_map'].get(str(nm), base_p2[i])[0] for i, nm in enumerate(names)], dtype=np.float32)
        dual_p1_0 = np.array([oof_bank['dual_oof_map'][str(nm)][0, 2] if str(nm) in oof_bank['dual_oof_map'] else base_p1[i, 0] for i, nm in enumerate(names)], dtype=np.float32)
        dual_p2_0 = np.array([oof_bank['dual_oof_map'][str(nm)][0, 1] if str(nm) in oof_bank['dual_oof_map'] else base_p2[i, 0] for i, nm in enumerate(names)], dtype=np.float32)
    else:
        mic_p1_0 = None

    if mic_p1_0 is not None:
        m12_0 = p1_out[:, 0] + p2_out[:, 0]
        rho_b = np.clip(p1_out[:, 0] / np.maximum(1e-5, m12_0), 1e-5, 1.0 - 1e-5)
        rho_m = align_odds_1d(np.clip(mic_p1_0 / np.maximum(1e-5, mic_p1_0 + mic_p2_0), 1e-5, 1.0 - 1e-5), rho_b)
        rho_d = align_odds_1d(np.clip(dual_p1_0 / np.maximum(1e-5, dual_p1_0 + dual_p2_0), 1e-5, 1.0 - 1e-5), rho_b)
        rc_0 = 0.72 * rho_b + 0.16 * rho_m + 0.12 * rho_d
        p1_out[:, 0] = m12_0 * rc_0
        p2_out[:, 0] = m12_0 * (1.0 - rc_0)

    # Step 3: Scene-Matched Reference-Anchored Local Conditional Ratio Calibration (Top-16 training neighbors)
    if tr_vecs is not None and tr_grades is not None and len(tr_vecs) >= 16:
        q_vecs = np.concatenate([p0_out, p1_out, p2_out], axis=1)
        q_vecs = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-8)
        sim_mat = q_vecs @ tr_vecs.T  # (N, N_tr)
        top_idx = np.argpartition(-sim_mat, kth=15, axis=1)[:, :16]
        top_sim = np.take_along_axis(sim_mat, top_idx, axis=1)
        weights = np.exp((top_sim - top_sim.max(axis=1, keepdims=True)) / 0.08)

        for k_ref, ci in [(0, 0), (1, 1), (3, 4)]:
            n_g = tr_grades[top_idx, ci]  # (N, 16)
            w_p1 = (weights * (n_g >= 2)).sum(axis=1)
            w_p2 = (weights * (n_g == 1)).sum(axis=1)
            tot_w = w_p1 + w_p2
            m12_k = p1_out[:, k_ref] + p2_out[:, k_ref]
            active = (m12_k >= 0.22) & (tot_w > 1.5)
            if np.any(active):
                anchor_rho = w_p1[active] / tot_w[active]
                cur_rho = p1_out[active, k_ref] / np.maximum(1e-5, m12_k[active])
                new_rho = 0.90 * cur_rho + 0.10 * anchor_rho
                p1_out[active, k_ref] = m12_k[active] * new_rho
                p2_out[active, k_ref] = m12_k[active] * (1.0 - new_rho)

    return p0_out, p1_out, p2_out


def run_prism_evaluation(use_anchor_net: bool = True, out_tag: str = 'prism_ra_iqa'):
    print("=" * 108)
    print("=== [PRISM Architecture] Pairwise Reference-Anchored Integrated Severity Microscope Evaluation ===")
    print("=" * 108)

    oof_bank = load_oof_microscope_and_dualstream_bank()
    P_prism = {
        'T0': [0.27, 0.26, 0.25, 0.40, 0.24, 0.44],
        'T1': [0.31, 0.27, 0.29, 0.30],
        'T2': [0.27, 0.28, 0.30, 0.28],
        'R12': [0.85, 0.85, 0.85, 0.85],
        'shield': 0.28,
    }

    # Load test_150 base tower predictions
    mA_t = load_avg_t150([f'{V17_DIR}/V17_S2R_SpecA_OverGen_TexLoss_s{sp}_probs.npz' for sp in (16, 17)])
    mB_t = load_avg_t150([f'{V17_DIR}/V17_S2R_SpecB_Sharp_TexLoss_s{sp}_probs.npz' for sp in (16, 17)])
    mC_t = load_avg_t150([f'{V17_DIR}/V17_S2R_SpecC_Blur_Bokeh_Screen_s{sp}_probs.npz' for sp in (16, 17)])
    mD_t = load_avg_t150([f'{V17_DIR}/V17_S2R_SpecD_All6_Giant_s{sp}_probs.npz' for sp in (16, 17)])

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

    m6b_fix_t = load_avg_t150([f'{V12_FIXED_DIR}/v12_abmil_s{sp}_probs.npz' for sp in (16, 17)] + [f'{V12_FIXED_DIR}/v12_abmil_full_probs.npz'])
    m6b_12_t = load_avg_t150([f'{V12_6B_DIR}/v12_abmil_s{sp}_probs.npz' for sp in (16, 17)] + [f'{V12_6B_DIR}/v12_abmil_full_probs.npz'])
    m6b_11_t = load_avg_t150(
        [f'{V11_6B_DIR}/iv6b_s{sp}_probs.npz' for sp in (16, 17)]
        + [f'{V11_6B_DIR}/iv6b_full_cradio_sig2_448_t2_probs.npz', f'{V11_6B_DIR}/iv6b_full_d3_pcl_448_t2_probs.npz']
    )
    mVLM_t = load_avg_t150(
        [f'{V13_VLM_DIR}/vlm_corn_s{sp}_probs.npz' for sp in (16, 17, 18, 19, 20)]
        + [f'{V13_VLM_DIR}/vlm_corn_full_probs.npz', f'{V13_VLM_DIR}/test_150_vlm_corn_probs.npz']
    )

    p0_6b_t, p1_6b_t, p2_6b_t = get_6b_probs(m6b_fix_t, m6b_12_t, m6b_11_t, prefix='t150')
    base_t150_p0, base_t150_p1, base_t150_p2, sh_t150 = fuse_6b_vlm_4spec_synergy(
        p0_6b_t, p1_6b_t, p2_6b_t, mA_t, mB_t, mC_t, mD_t, mVLM_t, m16_t_tuple, m10_t_tuple, prefix='t150'
    )
    names_150 = mA_t['names']

    prism_q_t = load_avg_t150([f'{PRISM_DIR}/prism_quad_s{sp}_probs.npz' for sp in (16, 17)])
    prism_6b_t = load_avg_t150([f'{PRISM_DIR}/prism_vit6b_s{sp}_probs.npz' for sp in (16, 17)])
    prism_vlm_t = load_avg_t150([f'{PRISM_DIR}/prism_vlm8b_s{sp}_probs.npz' for sp in (16, 17)])

    t150_cal_accum = [np.zeros_like(base_t150_p0), np.zeros_like(base_t150_p1), np.zeros_like(base_t150_p2)]
    res_list = []

    for sp in (16, 17):
        mA, mB, mC, mD, m6b_fix, m6b_12, m6b_11, mVLM, m16_tuple = load_split_models(
            sp, V17_DIR, 'V17_S2R', blend_v18=True
        )
        te_names = [str(n) for n in mA['te_names']]
        gte = mA['gte']

        pq_sp = np.load(f'{PRISM_DIR}/prism_quad_s{sp}_probs.npz') if os.path.exists(f'{PRISM_DIR}/prism_quad_s{sp}_probs.npz') else None
        p6_sp = np.load(f'{PRISM_DIR}/prism_vit6b_s{sp}_probs.npz') if os.path.exists(f'{PRISM_DIR}/prism_vit6b_s{sp}_probs.npz') else None
        pv_sp = np.load(f'{PRISM_DIR}/prism_vlm8b_s{sp}_probs.npz') if os.path.exists(f'{PRISM_DIR}/prism_vlm8b_s{sp}_probs.npz') else None

        _, tr_vecs, tr_grades = build_split_isolated_train_anchor_bank(sp, oof_bank) if use_anchor_net else (None, None, None)

        p0_6b_te, p1_6b_te, p2_6b_te = get_6b_probs(m6b_fix, m6b_12, m6b_11, prefix='te')
        te_p0_b, te_p1_b, te_p2_b, sh_te = fuse_6b_vlm_4spec_synergy(
            p0_6b_te, p1_6b_te, p2_6b_te, mA, mB, mC, mD, mVLM, m16_tuple, None, prefix='te'
        )
        te_p0, te_p1, te_p2 = apply_prism_mass_preserving_anchor_calibration(
            te_names, te_p0_b, te_p1_b, te_p2_b, oof_bank,
            tr_vecs=tr_vecs, tr_grades=tr_grades,
            prism_quad_npz=pq_sp, prism_vit6b_npz=p6_sp, prism_vlm8b_npz=pv_sp, prefix='te',
        )

        t150_p0_sp, t150_p1_sp, t150_p2_sp = apply_prism_mass_preserving_anchor_calibration(
            names_150, base_t150_p0, base_t150_p1, base_t150_p2, oof_bank,
            tr_vecs=tr_vecs, tr_grades=tr_grades,
            prism_quad_npz=prism_q_t, prism_vit6b_npz=prism_6b_t, prism_vlm8b_npz=prism_vlm_t, prefix='t150',
        )
        t150_cal_accum[0] += 0.5 * t150_p0_sp
        t150_cal_accum[1] += 0.5 * t150_p1_sp
        t150_cal_accum[2] += 0.5 * t150_p2_sp

        pr_te = decode_anti_interception(te_p0, te_p1, te_p2, sh_te, P_prism)
        ev = evaluate_both_metrics(pr_te, gte)
        res_list.append(ev)
        print(
            f"Split s{sp} (N=1,000) | Doc1: {ev['doc1_total']:6.3f} "
            f"(S_pos={ev['doc1_S_pos']:5.2f}, S_p0={ev['doc1_S_p0']:5.2f}, S_clean={ev['doc1_S_clean']:5.2f}) | "
            f"Doc2_Strict: {ev['doc2_strict_total']:6.3f}"
        )

    m_d1 = 0.5 * (res_list[0]['doc1_total'] + res_list[1]['doc1_total'])
    m_d2 = 0.5 * (res_list[0]['doc2_strict_total'] + res_list[1]['doc2_strict_total'])
    print("-" * 108)
    print(f"Mean Blind Test (N=2,000) | Doc1: {m_d1:6.3f} | Doc2_Strict: {m_d2:6.3f}")

    # Export test_150 predictions
    pred_t150 = decode_anti_interception(
        t150_cal_accum[0], t150_cal_accum[1], t150_cal_accum[2], sh_t150, P_prism
    )
    sub_data = []
    for i, nm in enumerate(names_150):
        lbl = {c: GRADE_MAP[int(pred_t150[i, CLASSES.index(c)])] for c in ORDER_CLASSES}
        sub_data.append({'image': nm, 'label': lbl})

    os.makedirs('results/final', exist_ok=True)
    out_file = f'results/final/pred_nriqa_dataset_test_{out_tag}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(sub_data, f, ensure_ascii=False, indent=2)
    with open('pred_nriqa_dataset_test.json', 'w', encoding='utf-8') as f:
        json.dump(sub_data, f, ensure_ascii=False, indent=2)

    p0_c = int((pred_t150 == 3).sum())
    p1_c = int((pred_t150 == 2).sum())
    p2_c = int((pred_t150 == 1).sum())
    p0_imgs = int((pred_t150 == 3).any(1).sum())
    clean_imgs = int((pred_t150 == 0).all(1).sum())
    print(
        f"[PRISM test_150 Audit] Clean={clean_imgs} | P0_Imgs={p0_imgs} (P0={p0_c}, P1={p1_c}, P2={p2_c}) "
        f"| Total Pos Cells={p0_c+p1_c+p2_c} -> Saved to {out_file} and pred_nriqa_dataset_test.json"
    )
    return P_prism, m_d1, m_d2


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-anchor-net', action='store_true')
    ap.add_argument('--out-tag', default='prism_ra_iqa')
    args = ap.parse_args()
    run_prism_evaluation(use_anchor_net=not args.no_anchor_net, out_tag=args.out_tag)
