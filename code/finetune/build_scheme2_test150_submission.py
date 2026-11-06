"""Generate the Clean, Zero-Anchor-Lock `test_150` Submission (`results/pred_nriqa_dataset_test_scheme2.json`
and `pred_nriqa_dataset_test.json`) using:
1. The 1:1 Aligned Checkpoint Pool across `s16`, `s17`, and `test_150`:
   - Clean Multi-Backbone Shield & Base (`V2` 10 + `V3` 16 + `V5` 24 High-Res 448 + `V9_SpecE` + `V9_Full_F6/F7`)
   - V8 Wave-2 Single-Feature Dedicated Experts (`V8_W2_ExpA..D`) + V4/V6/V7 Specialists
   - V9 High-Resolution Spatial-Anomaly-Pooled Specialists (`V9_SpecA..E` + `V9_Full_F0..F7`)
2. Scheme 2: Dedicated Image-Level `All-Clean` Gate Classifier $P(\\text{All-Clean} \\mid x)$ + V8-P0 / V9-Grade Decoupled Routing,
   completely removing the old `b5243` hard-lock while achieving +1.35 ~ +1.56 pts on zero-leakage `s16/s17` blind tests!
"""
import os, sys, json, time, warnings
from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
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
from run_scheme1_scheme2_breakthrough import FULL_IDX, BIN_IDX, score_both
from fast_test_qualirag_preserved_p0 import (
    align_prob_mean, build_p0_preserved_v9_14ch,
    decode_p0_protected_two_stage, tune_p0_protected_gate_on_val
)
from generate_v9_150_submission import GRADE, load_150_context

def load_v9_fulldata_150():
    """Load the 8 Full-Data V9 Specialists (`V9_Full_F0..F7`) on `test_150`!"""
    files = {
        'A0': 'results/v9_fulldata/V9_Full_F0_OverGen_Giant_probs.npz',
        'A1': 'results/v9_fulldata/V9_Full_F1_OverGen_Giant_probs.npz',
        'B2': 'results/v9_fulldata/V9_Full_F2_SharpTex_HiRes_probs.npz',
        'B3': 'results/v9_fulldata/V9_Full_F3_SharpTex_HiRes_probs.npz',
        'C4': 'results/v9_fulldata/V9_Full_F4_AllP0_Tile3_probs.npz',
        'C5': 'results/v9_fulldata/V9_Full_F5_AllP0_Tile3_probs.npz',
        'E6': 'results/v9_fulldata/V9_Full_F6_RealHiRes_Giant_probs.npz',
        'E7': 'results/v9_fulldata/V9_Full_F7_RealHiRes_Giant_probs.npz',
    }
    loaded = {}
    for k, fp in files.items():
        if os.path.exists(fp):
            d = np.load(fp, allow_pickle=True)
            loaded[k] = {
                'p0': d['t150_p0'],
                'p1': d['t150_p1'],
                'p2': d['t150_p2'],
                'gate': d['t150_gate'],
            }
    return loaded


def build_1to1_clean_gate_features(p0, p1, p2, cln0, cln1, cln2, gate_mean, gate_min):
    """Build 100% 1:1 aligned 20-dimensional Image-Level All-Clean Gate feature matrix
    that is identically defined on `s16_va`, `s16_te`, `s17_va`, `s17_te`, and `test_150`!
    """
    cum6 = p0.copy()
    cln_cum6 = cln0.copy()
    for k, ci in enumerate(FULL_IDX):
        cum6[:, ci] = p0[:, ci] + p1[:, k] + p2[:, k]
        cln_cum6[:, ci] = cln0[:, ci] + cln1[:, k] + cln2[:, k]

    cln_sorted = np.sort(cln_cum6, axis=1)
    cum_sorted = np.sort(cum6, axis=1)

    p_clean_cln = np.prod(np.clip(1.0 - cln_cum6, 1e-4, 1.0), axis=1)
    p_clean_ens = np.prod(np.clip(1.0 - cum6, 1e-4, 1.0), axis=1)

    feats = [
        p_clean_cln,
        p_clean_ens,
        cln_sorted[:, -1],
        cln_sorted[:, -2],
        cln_cum6.sum(axis=1),
        cum_sorted[:, -1],
        cum_sorted[:, -2],
        cum6.sum(axis=1),
        cum6.max(axis=1) - cln_cum6.max(axis=1),
        p0.max(axis=1),
        p1.max(axis=1),
        p2.max(axis=1),
        gate_mean,
        gate_min,
        cln_cum6[:, 0], cln_cum6[:, 1], cln_cum6[:, 2], cln_cum6[:, 3], cln_cum6[:, 4], cln_cum6[:, 5],
    ]
    return np.column_stack(feats).astype(np.float32), cln_cum6


def extract_gate_summary_from_models(models, prefix, N):
    # Use the exact same 9 specialist checkpoints that have `gate` on both s16/s17 and test_150!
    spec_keys = [
        'v8_experts/V8_W2_ExpA_overgen', 'v8_experts/V8_W2_ExpB_sharp',
        'v8_experts/V8_W2_ExpC_blurtex', 'v8_experts/V8_W2_ExpD_bokeh',
        'v9_specialists/V9_SpecA_OverGen_Giant', 'v9_specialists/V9_SpecB_SharpTex_HiRes',
        'v9_specialists/V9_SpecC_AllP0_Tile3', 'v9_specialists/V9_SpecD_BokehOG_HiRes',
        'v9_specialists/V9_SpecE_RealHiRes_Giant',
    ]
    g_list = []
    for k in spec_keys:
        if k in models and f'{prefix}_gate' in models[k]:
            g_list.append(models[k][f'{prefix}_gate'])
    if not g_list:
        return np.zeros(N), np.zeros(N)
    g_arr = np.stack(g_list, axis=1)
    return g_arr.mean(axis=1), np.percentile(g_arr, 15, axis=1)


def main():
    t0 = time.time()
    import csv
    man_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
    name_to_src = {r['name']: r['source'] for r in man_rows}

    # 1. Verify on s16 and s17 with the exact 1:1 aligned 20-dim Clean Gate features
    va_X_all, va_clncum_all, va_y_all, va_src_all = [], [], [], []
    va_data_by_sp = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        ref = np.load(f'results/v3_ablate/A_base_s{sp}_probs.npz', allow_pickle=True)
        va_names = [str(x) for x in ref['va_names']]
        src_va = np.array([name_to_src[n] for n in va_names])

        p0_v8_va, _, _, _, _, _ = build_decoupled_14ch_probs(models, 'va')
        p0_v8_te, _, _, _, _, _ = build_decoupled_14ch_probs(models, 'te')
        _, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va = build_p0_preserved_v9_14ch(models, 'va', None)
        _, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te = build_p0_preserved_v9_14ch(models, 'te', None)

        gm_va, gmin_va = extract_gate_summary_from_models(models, 'va', len(gva))
        gm_te, gmin_te = extract_gate_summary_from_models(models, 'te', len(gte))

        X_va, cln_cum_va = build_1to1_clean_gate_features(p0_v8_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va, gm_va, gmin_va)
        X_te, cln_cum_te = build_1to1_clean_gate_features(p0_v8_te, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te, gm_te, gmin_te)

        # Fit 5-fold OOF on s{sp}_va and evaluate on s{sp}_te
        from ablate_clean_gate_and_qualirag_lowdim import fit_compact_clean_gate
        pcln_va, pcln_te = fit_compact_clean_gate(X_va, cln_cum_va, gva, src_va, X_te, cln_cum_te)
        gprm = tune_p0_protected_gate_on_val(p0_v8_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va, pcln_va, gva)
        pred_te = decode_p0_protected_two_stage(p0_v8_te, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te, pcln_te, gprm)
        r_te = score_both(gte, pred_te)
        print(f"[Verified 1:1 Aligned Pipeline on s{sp}_test] Doc2={r_te['total_doc2']:.2f} (S_pos={r_te['S_pos']:.3f}, S_p0={r_te['S_p0']:.3f}, Clean2={r_te['S_clean_doc2']:.3f}) | Doc1={r_te['total']:.2f} (Clean1={r_te['S_clean']:.3f})")
        print(f"   Learned Gate Params on s{sp}_va: {gprm}")

        va_X_all.append(X_va)
        va_clncum_all.append(cln_cum_va)
        va_y_all.append((gva == 0).all(axis=1).astype(int))
        va_src_all.append(src_va)
        va_data_by_sp[sp] = (p0_v8_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va, pcln_va, gva, gprm)

    # 2. Now build the exact 1:1 aligned channels & Clean Gate on `test_150`!
    ctx_150 = load_150_context()
    ordered_names = ctx_150['ordered_names']
    m150 = ctx_150['models']
    v9_full_150 = load_v9_fulldata_150()

    # Base Clean Shield on test_150 (`cln0_150, cln1_150, cln2_150`):
    # Blend `ctx_150['cln_p0/1/2']` (V2+V3+V5 Clean) with V9_SpecE (s16/s17) + V9_Full_F6/F7 (RealHiRes Giant)!
    cln0_150 = ctx_150['cln_p0'].copy()
    cln1_150 = ctx_150['cln_p1'].copy()
    cln2_150 = ctx_150['cln_p2'].copy()

    v9_e_150 = m150['v9_specialists/V9_SpecE_RealHiRes_Giant']
    if 'E6' in v9_full_150 and 'E7' in v9_full_150:
        e_full_0 = 0.5 * (v9_full_150['E6']['p0'] + v9_full_150['E7']['p0'])
        e_full_1 = 0.5 * (v9_full_150['E6']['p1'] + v9_full_150['E7']['p1'])
        e_full_2 = 0.5 * (v9_full_150['E6']['p2'] + v9_full_150['E7']['p2'])
        e0_blend = 0.5 * v9_e_150['te_p0'] + 0.5 * e_full_0
        e1_blend = 0.5 * v9_e_150['te_p1'] + 0.5 * e_full_1
        e2_blend = 0.5 * v9_e_150['te_p2'] + 0.5 * e_full_2
    else:
        e0_blend, e1_blend, e2_blend = v9_e_150['te_p0'], v9_e_150['te_p1'], v9_e_150['te_p2']

    cln0_150 = 0.80 * cln0_150 + 0.20 * e0_blend
    cln1_150 = 0.80 * cln1_150 + 0.20 * e1_blend
    cln2_150 = 0.80 * cln2_150 + 0.20 * e2_blend

    # Build V8-P0 (`p0_150`) and V9-Grade (`p1_150, p2_150`) on test_150 matching `s16/s17` + V9_Full_F0..F7!
    p0_150 = ctx_150['te_p0'].copy()
    p1_150 = ctx_150['te_p1'].copy()
    p2_150 = ctx_150['te_p2'].copy()

    # Combine s16/s17 V9_SpecA..E with 8-Fold Full-Data V9_Full_F0..F7 (50/50 within each specialist family)
    v9_a1 = 0.5 * m150['v9_specialists/V9_SpecA_OverGen_Giant']['te_p1'] + 0.25 * v9_full_150['A0']['p1'] + 0.25 * v9_full_150['A1']['p1']
    v9_a2 = 0.5 * m150['v9_specialists/V9_SpecA_OverGen_Giant']['te_p2'] + 0.25 * v9_full_150['A0']['p2'] + 0.25 * v9_full_150['A1']['p2']
    v9_a0 = 0.5 * m150['v9_specialists/V9_SpecA_OverGen_Giant']['te_p0'] + 0.25 * v9_full_150['A0']['p0'] + 0.25 * v9_full_150['A1']['p0']

    v9_b1 = 0.5 * m150['v9_specialists/V9_SpecB_SharpTex_HiRes']['te_p1'] + 0.25 * v9_full_150['B2']['p1'] + 0.25 * v9_full_150['B3']['p1']
    v9_b2 = 0.5 * m150['v9_specialists/V9_SpecB_SharpTex_HiRes']['te_p2'] + 0.25 * v9_full_150['B2']['p2'] + 0.25 * v9_full_150['B3']['p2']
    v9_b0 = 0.5 * m150['v9_specialists/V9_SpecB_SharpTex_HiRes']['te_p0'] + 0.25 * v9_full_150['B2']['p0'] + 0.25 * v9_full_150['B3']['p0']

    v9_c1 = 0.5 * m150['v9_specialists/V9_SpecC_AllP0_Tile3']['te_p1'] + 0.25 * v9_full_150['C4']['p1'] + 0.25 * v9_full_150['C5']['p1']
    v9_c2 = 0.5 * m150['v9_specialists/V9_SpecC_AllP0_Tile3']['te_p2'] + 0.25 * v9_full_150['C4']['p2'] + 0.25 * v9_full_150['C5']['p2']
    v9_c0 = 0.5 * m150['v9_specialists/V9_SpecC_AllP0_Tile3']['te_p0'] + 0.25 * v9_full_150['C4']['p0'] + 0.25 * v9_full_150['C5']['p0']

    v9_d1 = m150['v9_specialists/V9_SpecD_BokehOG_HiRes']['te_p1']
    v9_d2 = m150['v9_specialists/V9_SpecD_BokehOG_HiRes']['te_p2']
    v9_d0 = m150['v9_specialists/V9_SpecD_BokehOG_HiRes']['te_p0']

    v9_e1, v9_e2, v9_e0 = e1_blend, e2_blend, e0_blend

    # Inject V9 + V9_Full into p0_150 (15% weight, mean-odds aligned to preserve P0 calibration)
    for ci, (va_w, vb_w, vc_w, vd_w, ve_w) in enumerate([
        (0.00, 0.00, 0.08, 0.00, 0.06),  # 0: 模糊_P0
        (0.00, 0.08, 0.08, 0.00, 0.05),  # 1: 纹理损失_P0
        (0.00, 0.12, 0.08, 0.00, 0.05),  # 2: 过锐_P0
        (0.00, 0.00, 0.08, 0.00, 0.04),  # 3: 花屏_P0
        (0.15, 0.00, 0.05, 0.10, 0.00),  # 4: 过度生成_P0
        (0.00, 0.00, 0.08, 0.12, 0.05),  # 5: 虚化突兀_P0
    ]):
        w_sum = va_w + vb_w + vc_w + vd_w + ve_w
        p0_150[:, ci] = (1.0 - w_sum) * p0_150[:, ci] + (
            va_w * align_prob_mean(v9_a0[:, ci], p0_150[:, ci]) +
            vb_w * align_prob_mean(v9_b0[:, ci], p0_150[:, ci]) +
            vc_w * align_prob_mean(v9_c0[:, ci], p0_150[:, ci]) +
            vd_w * align_prob_mean(v9_d0[:, ci], p0_150[:, ci]) +
            ve_w * align_prob_mean(v9_e0[:, ci], p0_150[:, ci])
        )

    # Inject V9 + V9_Full into p1_150 and p2_150 (exact match to build_p0_preserved_v9_14ch)
    for k, (va_w, vb_w, vc_w, vd_w, ve_w) in enumerate([
        (0.00, 0.00, 0.12, 0.00, 0.10),  # 0: 模糊 P1/P2
        (0.00, 0.12, 0.10, 0.00, 0.08),  # 1: 纹理损失 P1/P2
        (0.00, 0.16, 0.08, 0.00, 0.08),  # 2: 过锐 P1/P2
        (0.20, 0.00, 0.00, 0.14, 0.06),  # 3: 过度生成 P1/P2
    ]):
        w_sum = va_w + vb_w + vc_w + vd_w + ve_w
        p1_150[:, k] = (1.0 - w_sum) * p1_150[:, k] + (
            va_w * align_prob_mean(v9_a1[:, k], p1_150[:, k]) +
            vb_w * align_prob_mean(v9_b1[:, k], p1_150[:, k]) +
            vc_w * align_prob_mean(v9_c1[:, k], p1_150[:, k]) +
            vd_w * align_prob_mean(v9_d1[:, k], p1_150[:, k]) +
            ve_w * align_prob_mean(v9_e1[:, k], p1_150[:, k])
        )
        p2_150[:, k] = (1.0 - w_sum) * p2_150[:, k] + (
            va_w * align_prob_mean(v9_a2[:, k], p2_150[:, k]) +
            vb_w * align_prob_mean(v9_b2[:, k], p2_150[:, k]) +
            vc_w * align_prob_mean(v9_c2[:, k], p2_150[:, k]) +
            vd_w * align_prob_mean(v9_d2[:, k], p2_150[:, k]) +
            ve_w * align_prob_mean(v9_e2[:, k], p2_150[:, k])
        )

    # Extract gate summary on test_150 from the exact same 9 specialist families + 8 V9_Full models
    g150_list = []
    for folder, pref in [
        ('v8_experts', 'V8_W2_ExpA_overgen'), ('v8_experts', 'V8_W2_ExpB_sharp'),
        ('v8_experts', 'V8_W2_ExpC_blurtex'), ('v8_experts', 'V8_W2_ExpD_bokeh'),
        ('v9_specialists', 'V9_SpecA_OverGen_Giant'), ('v9_specialists', 'V9_SpecB_SharpTex_HiRes'),
        ('v9_specialists', 'V9_SpecC_AllP0_Tile3'), ('v9_specialists', 'V9_SpecD_BokehOG_HiRes'),
        ('v9_specialists', 'V9_SpecE_RealHiRes_Giant'),
    ]:
        for sp in (16, 17):
            fp = f'results/{folder}/{pref}_s{sp}_probs.npz'
            if os.path.exists(fp):
                d = np.load(fp)
                if 't150_gate' in d.files:
                    g150_list.append(d['t150_gate'])
    for k, v in v9_full_150.items():
        g150_list.append(v['gate'])
    g150_arr = np.stack(g150_list, axis=1)
    gm_150 = g150_arr.mean(axis=1)
    gmin_150 = np.percentile(g150_arr, 15, axis=1)

    X_150, cln_cum_150 = build_1to1_clean_gate_features(
        p0_150, p1_150, p2_150, cln0_150, cln1_150, cln2_150, gm_150, gmin_150
    )

    # Predict `p_clean_150` by averaging the Clean Gate classifiers trained on s16_va and s17_va
    pcln_150_list = []
    for sp in (16, 17):
        X_va = va_X_all[0 if sp == 16 else 1]
        y_va = va_y_all[0 if sp == 16 else 1]
        sc = StandardScaler()
        X_va_s = np.clip(sc.fit_transform(X_va), -5.0, 5.0)
        X_150_s = np.clip(sc.transform(X_150), -5.0, 5.0)
        lr = LogisticRegression(C=0.20, max_iter=500, random_state=42)
        lr.fit(X_va_s, y_va)
        p_lr_150 = lr.predict_proba(X_150_s)[:, 1]
        s_np_150 = np.prod(np.clip(1.0 - cln_cum_150, 1e-4, 1.0), axis=1)
        pcln_150_list.append(0.65 * p_lr_150 + 0.35 * s_np_150)

    pcln_150 = 0.5 * (pcln_150_list[0] + pcln_150_list[1])

    # Average the gate parameters learned on s16_va and s17_va
    g16 = va_data_by_sp[16][8]
    g17 = va_data_by_sp[17][8]
    gprm_150 = {
        'tau_clean': 0.5 * (g16['tau_clean'] + g17['tau_clean']),
        'tau_cln_max': 0.5 * (g16['tau_cln_max'] + g17['tau_cln_max']),
        'tau_rescue': 0.5 * (g16['tau_rescue'] + g17['tau_rescue']),
        'tau_conf_def': 0.38,
        'alpha_p2': 0.5 * (g16['alpha_p2'] + g17['alpha_p2']),
        'alpha_p1': 0.5 * (g16['alpha_p1'] + g17['alpha_p1']),
        'alpha_p0': 0.5 * (g16['alpha_p0'] + g17['alpha_p0']),
    }
    print(f"\n[test_150] Averaged Gate Parameters: {gprm_150}")

    pred_150 = decode_p0_protected_two_stage(
        p0_150, p1_150, p2_150, cln0_150, cln1_150, cln2_150, pcln_150, gprm_150
    )

    # Format JSON submission and compare against b5243 (without locking to b5243!)
    b5243 = ctx_150['b5243']
    out_json = []
    n_img_diff = 0
    n_cell_diff = 0
    n_all_clean_new = 0
    n_all_clean_old = 0

    for i, name in enumerate(ordered_names):
        lbl = {CLASSES[ci]: GRADE[int(pred_150[i, ci])] for ci in range(6)}
        old_lbl = b5243[name]
        if all(v == '无问题' for v in lbl.values()):
            n_all_clean_new += 1
        if all(v == '无问题' for v in old_lbl.values()):
            n_all_clean_old += 1
        if lbl != old_lbl:
            n_img_diff += 1
            for c in CLASSES:
                if lbl[c] != old_lbl[c]:
                    n_cell_diff += 1
        out_json.append({'image': name, 'label': lbl})

    os.makedirs('results/scheme2_sub', exist_ok=True)
    for out_p in [
        'results/scheme2_sub/pred_nriqa_dataset_test_scheme2.json',
        'pred_nriqa_dataset_test.json',
    ]:
        with open(out_p, 'w', encoding='utf-8') as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        print(f"Saved {out_p} ({len(out_json)} images)")

    print(f"\n=== Comparison vs 52.43 Baseline on test_150 (ZERO Anchor Locking!) ===")
    print(f"  All-Clean images: 52.43={n_all_clean_old}/150 -> Scheme2={n_all_clean_new}/150")
    print(f"  Images modified vs 52.43: {n_img_diff}/150 ({n_cell_diff} total class-grade cells changed)")
    print(f"\n  Per-Class Grade Counts on test_150 (52.43 -> Scheme 2):")
    for ci, c in enumerate(CLASSES):
        cnt_old = Counter(b5243[n][c] for n in ordered_names)
        cnt_new = Counter(out_json[i]['label'][c] for i in range(150))
        print(f"    {c:6s}: P0 {cnt_old['P0']:2d}->{cnt_new['P0']:2d} | P1 {cnt_old['P1']:2d}->{cnt_new['P1']:2d} | P2 {cnt_old['P2']:2d}->{cnt_new['P2']:2d} | 无问题 {cnt_old['无问题']:3d}->{cnt_new['无问题']:3d}")


if __name__ == '__main__':
    main()
