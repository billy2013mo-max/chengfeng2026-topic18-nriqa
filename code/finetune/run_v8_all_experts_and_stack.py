"""Master Orchestrator for V8 Single-Feature Dedicated Experts + Counterfactual Ladders + Meta-Stacking:
1. Builds ~3,180 Counterfactual Severity Ladder images for s16 & s17 (`build_ladder_synth_v8.py`).
2. Extracts 60-dim 1:1 Native Physical Caliper features for all real & synthetic images (`extract_native_phys_feats.py`).
3. Evaluates Stage 0 (Physical-Caliper + Cross-Channel Meta-Calibrator on existing s16/s17 experts).
4. Launches all 8 V8 Single-Feature Dedicated Experts in parallel across GPUs 0..7:
   - GPU 0 (s16) & GPU 1 (s17): Expert A (`过度生成` 100% Dedicated Specialist)
   - GPU 2 (s16) & GPU 3 (s17): Expert B (`过锐` 100% Dedicated Specialist)
   - GPU 4 (s16) & GPU 5 (s17): Expert C (`模糊,纹理损失` Grade-Boundary Specialist)
   - GPU 6 (s16) & GPU 7 (s17): Expert D (`虚化突兀` 100% Dedicated Specialist)
5. Evaluates Stage 1 (Full V8 Single-Feature Dedicated Experts + Physical Caliper + Meta-Calibrator)
   on the 1,000-image isolated test sets (s16 & s17).
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from extract_native_phys_feats import extract_for_names

OUT_ABL = 'results/v3_ablate'
OUT_V8 = 'results/v8_experts'
os.makedirs(OUT_V8, exist_ok=True)

V8_JOBS = [
    # (tag, gpu, sp, focus_classes, suite_name, backbones)
    ('V8_ExpA_overgen_s16', 0, 16, '过度生成',      'overgen', 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4'),
    ('V8_ExpA_overgen_s17', 1, 17, '过度生成',      'overgen', 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4'),
    ('V8_ExpB_sharp_s16',   2, 16, '过锐',          'sharp',   'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4'),
    ('V8_ExpB_sharp_s17',   3, 17, '过锐',          'sharp',   'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4'),
    ('V8_ExpC_blurtex_s16', 4, 16, '模糊,纹理损失', 'blurtex', 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4'),
    ('V8_ExpC_blurtex_s17', 5, 17, '模糊,纹理损失', 'blurtex', 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4'),
    ('V8_ExpD_bokeh_s16',   6, 16, '虚化突兀',      'bokeh',   'siglip2:lastk4,cradio:lastk4,pecorel:lastk4'),
    ('V8_ExpD_bokeh_s17',   7, 17, '虚化突兀',      'bokeh',   'siglip2:lastk4,cradio:lastk4,pecorel:lastk4'),
]


def load_split_models(sp, include_v8=True):
    """Load all zero-leakage model probability caches for split `sp` (both val 357 and test 1,000)."""
    models = {}
    for tag in ['A_base', 'B_negw', 'C_asym', 'D_combo', 'E_synth', 'F_v5', 'G_v6', 'H_og']:
        p = f'{OUT_ABL}/{tag}_s{sp}_probs.npz'
        if os.path.exists(p):
            models[tag] = np.load(p)
    for p in sorted(glob.glob(f'results/v6_val/*s{sp}*probs.npz')):
        bn = os.path.basename(p).replace('_probs.npz', '')
        try:
            d = np.load(p)
            if 'va_p0' in d and 'te_p0' in d:
                models[bn] = d
        except Exception:
            pass
    if include_v8:
        for tag, _, s_id, *_ in V8_JOBS:
            if s_id == sp:
                p = f'{OUT_V8}/{tag}_probs.npz'
                if os.path.exists(p):
                    models[tag] = np.load(p)
    return models


def build_meta_features_for_split(models, names_list, phys_map, split_prefix='va'):
    """Construct rich per-image feature matrix combining:
    - All expert model probabilities (p0, p1, p2, cumulative probabilities, expected severity, entropy)
    - Cross-channel co-occurrence context (max defect prob, sum of texture/blur probs, clean gate)
    - 60-dim 1:1 Native Physical Caliper features
    """
    n_imgs = len(names_list)
    feat_blocks = []
    for mname, d in sorted(models.items()):
        p0 = d[f'{split_prefix}_p0']  # (N, 6)
        p1 = d[f'{split_prefix}_p1']  # (N, 4)
        p2 = d[f'{split_prefix}_p2']  # (N, 4)
        c01 = p0[:, [0, 1, 2, 4]] + p1
        c012 = c01 + p2
        sev = 3.0 * p0[:, [0, 1, 2, 4]] + 2.0 * p1 + 1.0 * p2
        max_any = np.maximum(np.max(c012, axis=1, keepdims=True), np.max(p0[:, [3, 5]], axis=1, keepdims=True))
        feat_blocks.append(np.hstack([p0, p1, p2, c01, c012, sev, max_any]))

    phys_mat = np.stack([phys_map.get(nm, np.zeros(60, dtype=np.float32)) for nm in names_list], axis=0)
    feat_blocks.append(phys_mat)
    return np.hstack(feat_blocks)


def fit_and_eval_two_stage_meta_calibrator(include_v8=True, stage_label='Stage 1'):
    """Fit a zero-leakage Two-Stage Presence + Grade Calibrator strictly on `val` (357 images)
    and evaluate on the 1,000-image isolated `test` set for s16 and s17.
    """
    with open('data/manifest_combined.csv') as f:
        all_names = [r['name'] for r in csv.DictReader(f)]
    phys_map = extract_for_names(all_names)

    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P2', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    print(f'\n================ {stage_label}: Zero-Leakage Evaluation on 1,000-Image Isolated Test Sets (s16 & s17) ================', flush=True)
    hdr = f"{'variant':24s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} {'S_cln':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)

    rows_base, rows_meta = [], []
    for sp in (16, 17):
        models = load_split_models(sp, include_v8=include_v8)
        da = models['A_base']
        gva = da['gva']
        gte = da['gte']

        with open(f'data/resplits/split_s{sp}.csv') as f:
            sp_rows = list(csv.DictReader(f))
        va_names = [r['name'] for r in sp_rows if r['split'] == 'val']
        te_names = [r['name'] for r in sp_rows if r['split'] == 'test']

        # 1. Baseline Clean_Ens on s{sp}
        df = models['F_v5']
        base_te_p0 = 0.45 * da['te_p0'] + 0.55 * df['te_p0']
        base_te_p1 = 0.45 * da['te_p1'] + 0.55 * df['te_p1']
        base_te_p2 = 0.45 * da['te_p2'] + 0.55 * df['te_p2']
        pred_base = decode_pc(base_te_p0, base_te_p1, base_te_p2, HAND)
        s_b = cm.score(gte, pred_base, TAU)
        f1_b = {f'{x[0]}_{x[1]}': x[4] for x in s_b['per']}
        rows_base.append((s_b, f1_b))

        # 2. Build dedicated expert probability blend per class + Meta-Calibrated probabilities
        # Collect clean base models
        clean_keys = [k for k in models.keys() if k in ('A_base', 'B_negw', 'C_asym', 'D_combo', 'F_v5') or k.startswith('train_v6_')]
        va_cln_p0 = np.mean([models[k]['va_p0'] for k in clean_keys], axis=0)
        va_cln_p1 = np.mean([models[k]['va_p1'] for k in clean_keys], axis=0)
        va_cln_p2 = np.mean([models[k]['va_p2'] for k in clean_keys], axis=0)
        te_cln_p0 = np.mean([models[k]['te_p0'] for k in clean_keys], axis=0)
        te_cln_p1 = np.mean([models[k]['te_p1'] for k in clean_keys], axis=0)
        te_cln_p2 = np.mean([models[k]['te_p2'] for k in clean_keys], axis=0)

        # Route each class to its Dedicated Single-Feature Expert Ensemble!
        va_p0, va_p1, va_p2 = va_cln_p0.copy(), va_cln_p1.copy(), va_cln_p2.copy()
        te_p0, te_p1, te_p2 = te_cln_p0.copy(), te_cln_p1.copy(), te_cln_p2.copy()

        # Class 4 (过度生成, fi=3): combine V8_ExpA_overgen + G_v6 + H_og + Clean
        og_models = [k for k in models if 'overgen' in k or k in ('G_v6', 'H_og')]
        if og_models:
            # Give V8 Dedicated Single-Feature Expert highest weight!
            w_og = [2.5 if 'V8_ExpA' in k else 1.0 for k in og_models]
            va_og_p0 = np.average([models[k]['va_p0'][:, 4] for k in og_models], axis=0, weights=w_og)
            va_og_p1 = np.average([models[k]['va_p1'][:, 3] for k in og_models], axis=0, weights=w_og)
            va_og_p2 = np.average([models[k]['va_p2'][:, 3] for k in og_models], axis=0, weights=w_og)
            te_og_p0 = np.average([models[k]['te_p0'][:, 4] for k in og_models], axis=0, weights=w_og)
            te_og_p1 = np.average([models[k]['te_p1'][:, 3] for k in og_models], axis=0, weights=w_og)
            te_og_p2 = np.average([models[k]['te_p2'][:, 3] for k in og_models], axis=0, weights=w_og)
            alpha_og = 0.70 if any('V8_ExpA' in k for k in og_models) else 0.45
            va_p0[:, 4] = (1 - alpha_og) * va_cln_p0[:, 4] + alpha_og * va_og_p0
            va_p1[:, 3] = (1 - alpha_og) * va_cln_p1[:, 3] + alpha_og * va_og_p1
            va_p2[:, 3] = (1 - alpha_og) * va_cln_p2[:, 3] + alpha_og * va_og_p2
            te_p0[:, 4] = (1 - alpha_og) * te_cln_p0[:, 4] + alpha_og * te_og_p0
            te_p1[:, 3] = (1 - alpha_og) * te_cln_p1[:, 3] + alpha_og * te_og_p1
            te_p2[:, 3] = (1 - alpha_og) * te_cln_p2[:, 3] + alpha_og * te_og_p2

        # Class 2 (过锐, fi=2): combine V8_ExpB_sharp + E_synth + G_v6
        sh_models = [k for k in models if 'sharp' in k or k in ('E_synth', 'G_v6')]
        if sh_models:
            w_sh = [2.5 if 'V8_ExpB' in k else 1.0 for k in sh_models]
            va_sh_p0 = np.average([models[k]['va_p0'][:, 2] for k in sh_models], axis=0, weights=w_sh)
            va_sh_p1 = np.average([models[k]['va_p1'][:, 2] for k in sh_models], axis=0, weights=w_sh)
            va_sh_p2 = np.average([models[k]['va_p2'][:, 2] for k in sh_models], axis=0, weights=w_sh)
            te_sh_p0 = np.average([models[k]['te_p0'][:, 2] for k in sh_models], axis=0, weights=w_sh)
            te_sh_p1 = np.average([models[k]['te_p1'][:, 2] for k in sh_models], axis=0, weights=w_sh)
            te_sh_p2 = np.average([models[k]['te_p2'][:, 2] for k in sh_models], axis=0, weights=w_sh)
            alpha_sh = 0.65 if any('V8_ExpB' in k for k in sh_models) else 0.40
            va_p0[:, 2] = (1 - alpha_sh) * va_cln_p0[:, 2] + alpha_sh * va_sh_p0
            va_p1[:, 2] = (1 - alpha_sh) * va_cln_p1[:, 2] + alpha_sh * va_sh_p1
            va_p2[:, 2] = (1 - alpha_sh) * va_cln_p2[:, 2] + alpha_sh * va_sh_p2
            te_p0[:, 2] = (1 - alpha_sh) * te_cln_p0[:, 2] + alpha_sh * te_sh_p0
            te_p1[:, 2] = (1 - alpha_sh) * te_cln_p1[:, 2] + alpha_sh * te_sh_p1
            te_p2[:, 2] = (1 - alpha_sh) * te_cln_p2[:, 2] + alpha_sh * te_sh_p2

        # Class 0 & 1 (模糊 & 纹理损失): combine V8_ExpC_blurtex + G_v6 + Clean
        bt_models = [k for k in models if 'blurtex' in k or k == 'G_v6']
        if bt_models:
            w_bt = [2.5 if 'V8_ExpC' in k else 1.0 for k in bt_models]
            alpha_bt = 0.55 if any('V8_ExpC' in k for k in bt_models) else 0.25
            for ci in [0, 1]:
                va_bt_p0 = np.average([models[k]['va_p0'][:, ci] for k in bt_models], axis=0, weights=w_bt)
                va_bt_p1 = np.average([models[k]['va_p1'][:, ci] for k in bt_models], axis=0, weights=w_bt)
                va_bt_p2 = np.average([models[k]['va_p2'][:, ci] for k in bt_models], axis=0, weights=w_bt)
                te_bt_p0 = np.average([models[k]['te_p0'][:, ci] for k in bt_models], axis=0, weights=w_bt)
                te_bt_p1 = np.average([models[k]['te_p1'][:, ci] for k in bt_models], axis=0, weights=w_bt)
                te_bt_p2 = np.average([models[k]['te_p2'][:, ci] for k in bt_models], axis=0, weights=w_bt)
                va_p0[:, ci] = (1 - alpha_bt) * va_cln_p0[:, ci] + alpha_bt * va_bt_p0
                va_p1[:, ci] = (1 - alpha_bt) * va_cln_p1[:, ci] + alpha_bt * va_bt_p1
                va_p2[:, ci] = (1 - alpha_bt) * va_cln_p2[:, ci] + alpha_bt * va_bt_p2
                te_p0[:, ci] = (1 - alpha_bt) * te_cln_p0[:, ci] + alpha_bt * te_bt_p0
                te_p1[:, ci] = (1 - alpha_bt) * te_cln_p1[:, ci] + alpha_bt * te_bt_p1
                te_p2[:, ci] = (1 - alpha_bt) * te_cln_p2[:, ci] + alpha_bt * te_bt_p2

        # Class 5 (虚化突兀): combine V8_ExpD_bokeh + G_v6 + Clean
        bk_models = [k for k in models if 'bokeh' in k or k == 'G_v6']
        if bk_models:
            w_bk = [2.5 if 'V8_ExpD' in k else 1.0 for k in bk_models]
            alpha_bk = 0.65 if any('V8_ExpD' in k for k in bk_models) else 0.35
            va_bk_p0 = np.average([models[k]['va_p0'][:, 5] for k in bk_models], axis=0, weights=w_bk)
            te_bk_p0 = np.average([models[k]['te_p0'][:, 5] for k in bk_models], axis=0, weights=w_bk)
            va_p0[:, 5] = (1 - alpha_bk) * va_cln_p0[:, 5] + alpha_bk * va_bk_p0
            te_p0[:, 5] = (1 - alpha_bk) * te_cln_p0[:, 5] + alpha_bk * te_bk_p0

        # 3. Cross-Channel + 60-Dim Physical Caliper Regularized Stacking (fit strictly on val using 5-fold OOF)
        X_va_raw = build_meta_features_for_split(models, va_names, phys_map, 'va')
        X_te_raw = build_meta_features_for_split(models, te_names, phys_map, 'te')
        scaler = StandardScaler()
        X_va = np.clip(scaler.fit_transform(X_va_raw), -5.0, 5.0)
        X_te = np.clip(scaler.transform(X_te_raw), -5.0, 5.0)

        # Out-of-fold predictions on val (to prevent threshold overfitting) + full-val fit for test
        n_va = len(va_names)
        folds = np.arange(n_va) % 5
        meta_va_p0, meta_va_p1, meta_va_p2 = np.zeros_like(va_p0), np.zeros_like(va_p1), np.zeros_like(va_p2)
        meta_te_p0, meta_te_p1, meta_te_p2 = np.zeros_like(te_p0), np.zeros_like(te_p1), np.zeros_like(te_p2)

        for ci, c in enumerate(CLASSES):
            if c in FULL:
                k = FULL.index(c)
                y_va = gva[:, ci]
                # Fit heavily L2-regularized multinomial logistic regression (C=0.015) on X_va
                oof_probs = np.zeros((n_va, 4), dtype=np.float64)
                for fold in range(5):
                    tr_m = folds != fold
                    va_m = folds == fold
                    clf = LogisticRegression(C=0.015, max_iter=300, class_weight={0: 1.0, 1: 1.15, 2: 1.25, 3: 1.15})
                    clf.fit(X_va[tr_m], y_va[tr_m])
                    oof_probs[va_m] = clf.predict_proba(X_va[va_m])
                clf_full = LogisticRegression(C=0.015, max_iter=300, class_weight={0: 1.0, 1: 1.15, 2: 1.25, 3: 1.15})
                clf_full.fit(X_va, y_va)
                te_probs = clf_full.predict_proba(X_te)

                meta_va_p2[:, k] = oof_probs[:, 1]
                meta_va_p1[:, k] = oof_probs[:, 2]
                meta_va_p0[:, ci] = oof_probs[:, 3]
                meta_te_p2[:, k] = te_probs[:, 1]
                meta_te_p1[:, k] = te_probs[:, 2]
                meta_te_p0[:, ci] = te_probs[:, 3]
            else:
                y_va_bin = (gva[:, ci] == 3).astype(int)
                oof_p0 = np.zeros(n_va, dtype=np.float64)
                for fold in range(5):
                    tr_m = folds != fold
                    va_m = folds == fold
                    clf = LogisticRegression(C=0.015, max_iter=300)
                    clf.fit(X_va[tr_m], y_va_bin[tr_m])
                    oof_p0[va_m] = clf.predict_proba(X_va[va_m])[:, 1]
                clf_full = LogisticRegression(C=0.015, max_iter=300)
                clf_full.fit(X_va, y_va_bin)
                meta_va_p0[:, ci] = oof_p0
                meta_te_p0[:, ci] = clf_full.predict_proba(X_te)[:, 1]

        # Blend 75% Dedicated Neural Experts + 25% Cross-Channel Physical Meta-Stacking
        f_va_p0 = 0.78 * va_p0 + 0.22 * meta_va_p0
        f_va_p1 = 0.78 * va_p1 + 0.22 * meta_va_p1
        f_va_p2 = 0.78 * va_p2 + 0.22 * meta_va_p2

        f_te_p0 = 0.78 * te_p0 + 0.22 * meta_te_p0
        f_te_p1 = 0.78 * te_p1 + 0.22 * meta_te_p1
        f_te_p2 = 0.78 * te_p2 + 0.22 * meta_te_p2

        # Coordinate-Ascent Per-Channel Decoder Calibration strictly on OOF Val (357 images),
        # starting from HAND and regularizing toward HAND so it generalizes cleanly to the 1,000 test images!
        best_th = copy_hand(HAND)
        best_val_sc = cm.score(gva, decode_with_shield(f_va_p0, f_va_p1, f_va_p2, va_cln_p0, va_cln_p1, va_cln_p2, best_th), TAU)['total']

        for _round in range(2):
            for ci, c in enumerate(CLASSES):
                if c in FULL:
                    k = FULL.index(c)
                    for key, idx_k, grid in [
                        ('T0', ci, np.linspace(0.24, 0.46, 12)),
                        ('T1', k, np.linspace(0.20, 0.38, 10)),
                        ('T2', k, np.linspace(0.20, 0.38, 10)),
                        ('R12', k, np.linspace(0.70, 1.05, 8)),
                    ]:
                        cur_best_v = best_th[key][idx_k]
                        for cand in grid:
                            best_th[key][idx_k] = float(cand)
                            pr_va = decode_with_shield(f_va_p0, f_va_p1, f_va_p2, va_cln_p0, va_cln_p1, va_cln_p2, best_th)
                            sc = cm.score(gva, pr_va, TAU)
                            if sc['S_clean'] >= 1.0 and sc['total'] > best_val_sc + 0.02:
                                best_val_sc = sc['total']
                                cur_best_v = float(cand)
                        best_th[key][idx_k] = 0.65 * cur_best_v + 0.35 * HAND[key][idx_k]
                else:
                    cur_best_v = best_th['T0'][ci]
                    for cand in np.linspace(0.24, 0.48, 13):
                        best_th['T0'][ci] = float(cand)
                        pr_va = decode_with_shield(f_va_p0, f_va_p1, f_va_p2, va_cln_p0, va_cln_p1, va_cln_p2, best_th)
                        sc = cm.score(gva, pr_va, TAU)
                        if sc['S_clean'] >= 1.0 and sc['total'] > best_val_sc + 0.02:
                            best_val_sc = sc['total']
                            cur_best_v = float(cand)
                    best_th['T0'][ci] = 0.65 * cur_best_v + 0.35 * HAND['T0'][ci]

        pred_te = decode_with_shield(f_te_p0, f_te_p1, f_te_p2, te_cln_p0, te_cln_p1, te_cln_p2, best_th)
        s_m = cm.score(gte, pred_te, TAU)
        f1_m = {f'{x[0]}_{x[1]}': x[4] for x in s_m['per']}
        rows_meta.append((s_m, f1_m))

    for label, rlist in [('Clean_384_Base', rows_base), (f'{stage_label}_4Exp_Stack', rows_meta)]:
        for idx, sp in enumerate((16, 17)):
            s, f1 = rlist[idx]
            print(f"{label+'_s'+str(sp):24s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want), flush=True)
        m = lambda k: np.mean([r[0][k] for r in rlist])
        mf = [np.mean([r[1][w] for r in rlist]) for w in want]
        print(f"{label+' MEAN':24s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf), flush=True)
        print(flush=True)


def copy_hand(h):
    return {k: list(v) if isinstance(v, (list, tuple)) else float(v) for k, v in h.items()}


def decode_with_shield(p0, p1, p2, cln_p0, cln_p1, cln_p2, th):
    pred = decode_pc(p0, p1, p2, th)
    # Strictly preserve Clean Shield from Pure Clean Base so S_clean = 1.000 is guaranteed!
    pred_cln = decode_pc(cln_p0, cln_p1, cln_p2, HAND)
    clean_mask = (pred_cln == 0).all(axis=1)
    pred[clean_mask] = 0
    return pred


def main():
    t0 = time.time()
    print('=== Step 1: Building V8 Counterfactual Severity Ladders (~3,180 images across s16 & s17) ===', flush=True)
    if not (os.path.exists('data/v8_ladders/manifest_s16_v8.csv') and os.path.exists('data/v8_ladders/manifest_s17_v8.csv')):
        subprocess.run(['python3', '-u', 'code/finetune/build_ladder_synth_v8.py'], check=True)
    else:
        print('[cached] s16 & s17 V8 ladders already built (+3,180 synthetic ladder rows).', flush=True)

    print('\n=== Step 2: Extracting 60-Dim 1:1 Native Physical Caliper Features ===', flush=True)
    for sp in (16, 17):
        with open(f'data/v8_ladders/manifest_s{sp}_v8.csv') as f:
            names = [r['name'] for r in csv.DictReader(f)]
        extract_for_names(names)

    # Evaluate Stage 0 (before V8 Single-Feature Dedicated neural training finishes)
    fit_and_eval_two_stage_meta_calibrator(include_v8=False, stage_label='Stage0_PreV8')

    print('\n=== Step 3: Launching 8 V8 Single-Feature Dedicated Experts in Parallel (GPUs 0..7) ===', flush=True)
    procs = []
    for tag, gpu, sp, focus_cls, suite, bbs in V8_JOBS:
        out_json = f'{OUT_V8}/{tag}.json'
        probs_npz = f'{OUT_V8}/{tag}_probs.npz'
        if os.path.exists(out_json) and os.path.exists(probs_npz):
            print(f'[skip] {tag} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_single_expert_v8.py',
            '--backbones', bbs,
            '--focus-classes', focus_cls,
            '--seed', '1',
            '--epochs', '4',
            '--batch', '4',
            '--grad-accum', '2',
            '--workers', '6',
            '--size', '384',
            '--tiles', '2',
            '--clean-repeat', '3',
            '--p0-clean-weight', '4.2',
            '--channel-neg-weight', '1.20',
            '--alpha-ex', '0.58',
            '--split', f'data/v8_ladders/split_s{sp}_{suite}.csv',
            '--manifest', f'data/v8_ladders/manifest_s{sp}_v8.csv',
            '--out', out_json,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        lf = open(f'{OUT_V8}/{tag}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((tag, gpu, p, lf))
        print(f'[launched] {tag:22s} on GPU {gpu} (focus={focus_cls}, suite={suite})', flush=True)

    for tag, gpu, p, lf in procs:
        rc = p.wait()
        lf.close()
        print(f'[{time.time()-t0:5.0f}s] Finished {tag:22s} on GPU {gpu} rc={rc}', flush=True)

    # Evaluate Stage 1 (with all 8 V8 Single-Feature Dedicated Experts!)
    fit_and_eval_two_stage_meta_calibrator(include_v8=True, stage_label='Stage1_FullV8')


if __name__ == '__main__':
    main()
