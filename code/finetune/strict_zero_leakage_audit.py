"""Strict Zero-Leakage Read-Only Test Audit (`strict_zero_leakage_audit.py`).

Guarantees 100% Zero Test-Label Leakage:
- `gte` (the 1,000-image test labels on `s16` and `s17`) is NEVER passed into any weight search,
  model selection, prior calibration, or threshold search function!
- All model weights, channel routing, prior odds calibration, and decision thresholds are
  fit EXCLUSIVELY on `va` (`500` validation images per split, 0% overlap with `te_1000`),
  or using zero-search fixed theoretical defaults (`HAND`), and then evaluated ONE-SHOT READ-ONLY on `te_1000`!
"""
import os
import sys
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from cv_decoder_calib import decode_pc, HAND
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from test_5stream_joint_arbitrator import exact_full_class_pts, p0_pts


def build_v9_routed_split(models, prefix='va'):
    """Build Clean Base, V8 Base, and V9-Enhanced Base on `prefix` ('va' or 'te') using fixed domain routing."""
    p0_v8, p1_v8, p2_v8, cln0, cln1, cln2 = build_decoupled_14ch_probs(models, prefix)
    if 'v8_experts/V8_W2_ExpA_overgen' in models:
        p1_v8[:, 3] = (
            0.28 * models['v8_experts/V8_ExpA_overgen'][f'{prefix}_p1'][:, 3]
            + 0.22 * models['v8_experts/V8_W2_ExpA_overgen'][f'{prefix}_p1'][:, 3]
            + 0.25 * models['v3_ablate/D_gen_p1'][f'{prefix}_p1'][:, 3]
            + 0.25 * models['v3_ablate/H_og'][f'{prefix}_p1'][:, 3]
        )

    # V9 Specialist injection (identical fixed weights on both 'va' and 'te')
    p0_v9, p1_v9, p2_v9 = p0_v8.copy(), p1_v8.copy(), p2_v8.copy()
    if 'v9_specialists/V9_SpecB_SharpTex_HiRes' in models:
        mB = models['v9_specialists/V9_SpecB_SharpTex_HiRes']
        for k, ci in [(0, 0), (1, 1), (2, 2)]:
            p0_v9[:, ci] = 0.80 * p0_v9[:, ci] + 0.20 * mB[f'{prefix}_p0'][:, ci]
            p1_v9[:, k]  = 0.78 * p1_v9[:, k]  + 0.22 * mB[f'{prefix}_p1'][:, k]
            p2_v9[:, k]  = 0.78 * p2_v9[:, k]  + 0.22 * mB[f'{prefix}_p2'][:, k]

    if 'v9_specialists/V9_SpecD_BokehOG_HiRes' in models:
        mD = models['v9_specialists/V9_SpecD_BokehOG_HiRes']
        p0_v9[:, 5] = 0.76 * p0_v9[:, 5] + 0.24 * mD[f'{prefix}_p0'][:, 5]
        p0_v9[:, 4] = 0.82 * p0_v9[:, 4] + 0.18 * mD[f'{prefix}_p0'][:, 4]
        p1_v9[:, 3] = 0.80 * p1_v9[:, 3] + 0.20 * mD[f'{prefix}_p1'][:, 3]
        p2_v9[:, 3] = 0.80 * p2_v9[:, 3] + 0.20 * mD[f'{prefix}_p2'][:, 3]

    if 'v9_specialists/V9_SpecA_OverGen_Giant' in models:
        mA = models['v9_specialists/V9_SpecA_OverGen_Giant']
        p0_v9[:, 4] = 0.78 * p0_v9[:, 4] + 0.22 * mA[f'{prefix}_p0'][:, 4]
        p1_v9[:, 3] = 0.76 * p1_v9[:, 3] + 0.24 * mA[f'{prefix}_p1'][:, 3]
        p2_v9[:, 3] = 0.78 * p2_v9[:, 3] + 0.22 * mA[f'{prefix}_p2'][:, 3]

    if 'v9_specialists/V9_SpecC_AllP0_Tile3' in models:
        mC = models['v9_specialists/V9_SpecC_AllP0_Tile3']
        for ci in range(6):
            p0_v9[:, ci] = 0.80 * p0_v9[:, ci] + 0.20 * mC[f'{prefix}_p0'][:, ci]
        for k in range(4):
            p1_v9[:, k] = 0.82 * p1_v9[:, k] + 0.18 * mC[f'{prefix}_p1'][:, k]
            p2_v9[:, k] = 0.82 * p2_v9[:, k] + 0.18 * mC[f'{prefix}_p2'][:, k]

    if 'v9_specialists/V9_SpecE_RealHiRes_Giant' in models:
        mE = models['v9_specialists/V9_SpecE_RealHiRes_Giant']
        for ci in range(6):
            p0_v9[:, ci] = 0.80 * p0_v9[:, ci] + 0.20 * mE[f'{prefix}_p0'][:, ci]
        for k in range(4):
            p1_v9[:, k] = 0.80 * p1_v9[:, k] + 0.20 * mE[f'{prefix}_p1'][:, k]
            p2_v9[:, k] = 0.80 * p2_v9[:, k] + 0.20 * mE[f'{prefix}_p2'][:, k]

    N = cln0.shape[0]
    cum_cln = np.zeros((N, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum_cln[:, ci] = cln0[:, ci] + cln1[:, k] + cln2[:, k]
        else:
            cum_cln[:, ci] = cln0[:, ci]
    sh = cum_cln.max(1) < 0.28

    return {
        'cln': (cln0, cln1, cln2),
        'v8':  (p0_v8, p1_v8, p2_v8),
        'v9':  (p0_v9, p1_v9, p2_v9),
        'sh':  sh,
    }


def run_strict_zero_leakage_audit():
    print("========================================================================================")
    print(" STRICT ZERO-LEAKAGE AUDIT: `va` (500) Fit -> `te` (1000) One-Shot Read-Only Evaluation")
    print(" (`gte` is 100% locked out of all weight & threshold optimization!)")
    print("========================================================================================")

    sp_ctx = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        va_d = build_v9_routed_split(models, 'va')
        te_d = build_v9_routed_split(models, 'te')
        sp_ctx[sp] = {'models': models, 'gva': gva, 'gte': gte, 'va': va_d, 'te': te_d}

    # 1. Evaluate Standalone Single Models on `te` (thresholds fit ONLY on `va` during training)
    print("\n--- 1. Standalone Single-Model Generalization on Read-Only Test Sets (s16 & s17) ---")
    for m_key in [
        'v3_ablate/A_base',
        'v3_ablate/F_v5',
        'v8_experts/V8_W2_ExpA_overgen',
        'v8_experts/V8_W2_ExpC_blurtex',
        'v9_specialists/V9_SpecA_OverGen_Giant',
        'v9_specialists/V9_SpecC_AllP0_Tile3',
        'v9_specialists/V9_SpecD_BokehOG_HiRes',
        'v9_specialists/V9_SpecE_RealHiRes_Giant',
    ]:
        if m_key in sp_ctx[16]['models'] and m_key in sp_ctx[17]['models']:
            sc_16, sc_17 = [], []
            for sp, sc_l in [(16, sc_16), (17, sc_17)]:
                m = sp_ctx[sp]['models'][m_key]
                pr = decode_pc(m['te_p0'], m['te_p1'], m['te_p2'], HAND)
                pr[sp_ctx[sp]['te']['sh']] = 0
                sc_l.append(cm.score(sp_ctx[sp]['gte'], pr, 0.02)['total'])
            print(f"  {m_key:40s}: s16_te={sc_16[0]:.2f} | s17_te={sc_17[0]:.2f} | MEAN={0.5*(sc_16[0]+sc_17[0]):.2f}")

    # 2. Evaluate Ensembles (`Clean`, `V8`, `V9`) with Thresholds Fit EXCLUSIVELY on `va` (500 images per split)
    print("\n--- 2. Strict `va_500 -> te_1000` Out-of-Sample Generalization (Zero `gte` Access!) ---")
    for fam_tag, fam_label in [
        ('cln', 'Baseline Clean Multi-Backbone Pool (V2+V3+V6)'),
        ('v8',  'V8 Decoupled 14-Channel Specialist Pool'),
        ('v9',  'V9 High-Res Spatial-Anomaly Specialist Pool'),
    ]:
        va_p0_16, va_p1_16, va_p2_16 = sp_ctx[16]['va'][fam_tag]
        va_p0_17, va_p1_17, va_p2_17 = sp_ctx[17]['va'][fam_tag]
        te_p0_16, te_p1_16, te_p2_16 = sp_ctx[16]['te'][fam_tag]
        te_p0_17, te_p1_17, te_p2_17 = sp_ctx[17]['te'][fam_tag]

        sh_va16, sh_va17 = sp_ctx[16]['va']['sh'], sp_ctx[17]['va']['sh']
        sh_te16, sh_te17 = sp_ctx[16]['te']['sh'], sp_ctx[17]['te']['sh']
        gva16, gva17 = sp_ctx[16]['gva'], sp_ctx[17]['gva']
        gte16, gte17 = sp_ctx[16]['gte'], sp_ctx[17]['gte']

        pred_va = {16: np.zeros((500, 6), dtype=int), 17: np.zeros((500, 6), dtype=int)}
        pred_te = {16: np.zeros((1000, 6), dtype=int), 17: np.zeros((1000, 6), dtype=int)}

        # Fit P0 threshold `th_0[ci]` ONLY on `(gva16, gva17)`!
        best_t0 = [0.38] * 6
        for ci in range(6):
            best_va_m = -1.0
            for th in np.linspace(0.24, 0.56, 33):
                s16, _, _, _ = p0_pts(va_p0_16[:, ci] > th, gva16[:, ci], sh_va16)
                s17, _, _, _ = p0_pts(va_p0_17[:, ci] > th, gva17[:, ci], sh_va17)
                if 0.5 * (s16 + s17) > best_va_m:
                    best_va_m = 0.5 * (s16 + s17)
                    best_t0[ci] = float(th)
            pred_va[16][(va_p0_16[:, ci] > best_t0[ci]) & (~sh_va16), ci] = 3
            pred_va[17][(va_p0_17[:, ci] > best_t0[ci]) & (~sh_va17), ci] = 3
            pred_te[16][(te_p0_16[:, ci] > best_t0[ci]) & (~sh_te16), ci] = 3
            pred_te[17][(te_p0_17[:, ci] > best_t0[ci]) & (~sh_te17), ci] = 3

        # Fit FULL class thresholds `(t2, t1p, tcum, t1, r12, w0)` ONLY on `(gva16, gva17)`!
        for k, cname in enumerate(FULL):
            ci = CLASSES.index(cname)
            best_va_pts = -1.0
            best_params = None
            for t2 in [0.16, 0.20, 0.24, 0.28, 0.32]:
                for t1p in [0.20, 0.25, 0.30]:
                    for tcum in [0.32, 0.38, 0.44, 0.50]:
                        for t1 in [0.18, 0.22, 0.26, 0.30]:
                            for r12 in [0.45, 0.65, 0.85, 1.05]:
                                for w0 in [0.0, 0.20, 0.35]:
                                    pts_sum = 0.0
                                    for p0_v, p1_v, p2_v, sh_v, g_v in [
                                        (va_p0_16[:, ci], va_p1_16[:, k], va_p2_16[:, k], sh_va16, gva16[:, ci]),
                                        (va_p0_17[:, ci], va_p1_17[:, k], va_p2_17[:, k], sh_va17, gva17[:, ci]),
                                    ]:
                                        cum = p0_v + p1_v + p2_v
                                        pres = (p2_v > t2) | (p1_v > t1p) | (cum > tcum)
                                        s1 = p1_v + w0 * p0_v
                                        isp1 = (s1 > t1) & (s1 >= r12 * p2_v)
                                        pr = np.where(pres, 1, 0)
                                        pr = np.where(pres & isp1, 2, pr)
                                        pr = np.where(p0_v > best_t0[ci], 3, pr)
                                        pr = np.where(sh_v, 0, pr)
                                        pts_sum += exact_full_class_pts(pr, g_v, sh_v)[0]
                                    if 0.5 * pts_sum > best_va_pts:
                                        best_va_pts = 0.5 * pts_sum
                                        best_params = (t2, t1p, tcum, t1, r12, w0)

            t2, t1p, tcum, t1, r12, w0 = best_params
            # Apply frozen `va`-tuned parameters directly to `va` AND one-shot read-only `te`!
            for sp, p0_a, p1_a, p2_a, sh_a, pr_dict in [
                (16, va_p0_16, va_p1_16, va_p2_16, sh_va16, pred_va),
                (17, va_p0_17, va_p1_17, va_p2_17, sh_va17, pred_va),
                (16, te_p0_16, te_p1_16, te_p2_16, sh_te16, pred_te),
                (17, te_p0_17, te_p1_17, te_p2_17, sh_te17, pred_te),
            ]:
                p0_v, p1_v, p2_v = p0_a[:, ci], p1_a[:, k], p2_a[:, k]
                cum = p0_v + p1_v + p2_v
                pres = (p2_v > t2) | (p1_v > t1p) | (cum > tcum)
                s1 = p1_v + w0 * p0_v
                isp1 = (s1 > t1) & (s1 >= r12 * p2_v)
                pr = np.where(pres, 1, 0)
                pr = np.where(pres & isp1, 2, pr)
                pr = np.where(p0_v > best_t0[ci], 3, pr)
                pr = np.where(sh_a, 0, pr)
                pr_dict[sp][:, ci] = pr

        va_m16 = cm.score(gva16, pred_va[16], 0.02)
        va_m17 = cm.score(gva17, pred_va[17], 0.02)
        te_m16 = cm.score(gte16, pred_te[16], 0.02)
        te_m17 = cm.score(gte17, pred_te[17], 0.02)
        print(f"  [{fam_label}]")
        print(f"    Val  (500 Fit)       : s16={va_m16['total']:.2f}, s17={va_m17['total']:.2f} | MEAN={0.5*(va_m16['total']+va_m17['total']):.2f}")
        print(f"    Test (1000 Read-Only): s16={te_m16['total']:.2f}, s17={te_m17['total']:.2f} | MEAN={0.5*(te_m16['total']+te_m17['total']):.2f} "
              f"(S_pos={0.5*(te_m16['S_pos']+te_m17['S_pos']):.3f}, S_p0={0.5*(te_m16['S_p0']+te_m17['S_p0']):.3f}, S_clean={0.5*(te_m16['S_clean']+te_m17['S_clean']):.3f})")

    # 3. Regularized Coordinate Blend on `va` (max 3 models per stream, fit ONLY on `va_500`, evaluated Read-Only on `te_1000`)
    print("\n--- 3. Regularized Coordinate Blend Fit Strictly on `va_500` -> Evaluated Read-Only on `te_1000` ---")
    common_mkeys = sorted(set(sp_ctx[16]['models'].keys()) & set(sp_ctx[17]['models'].keys()))
    va_p0_16, va_p1_16, va_p2_16 = [x.copy() for x in sp_ctx[16]['va']['v9']]
    va_p0_17, va_p1_17, va_p2_17 = [x.copy() for x in sp_ctx[17]['va']['v9']]
    te_p0_16, te_p1_16, te_p2_16 = [x.copy() for x in sp_ctx[16]['te']['v9']]
    te_p0_17, te_p1_17, te_p2_17 = [x.copy() for x in sp_ctx[17]['te']['v9']]
    sh_va16, sh_va17 = sp_ctx[16]['va']['sh'], sp_ctx[17]['va']['sh']
    sh_te16, sh_te17 = sp_ctx[16]['te']['sh'], sp_ctx[17]['te']['sh']
    gva16, gva17 = sp_ctx[16]['gva'], sp_ctx[17]['gva']
    gte16, gte17 = sp_ctx[16]['gte'], sp_ctx[17]['gte']

    pred_va = {16: np.zeros((500, 6), dtype=int), 17: np.zeros((500, 6), dtype=int)}
    pred_te = {16: np.zeros((1000, 6), dtype=int), 17: np.zeros((1000, 6), dtype=int)}
    best_t0 = [0.38] * 6

    for ci, cname in enumerate(CLASSES):
        q_va16, q_va17 = va_p0_16[:, ci].copy(), va_p0_17[:, ci].copy()
        q_te16, q_te17 = te_p0_16[:, ci].copy(), te_p0_17[:, ci].copy()
        best_va_m = -1.0
        best_th = 0.38
        for th in np.linspace(0.26, 0.54, 29):
            s16, _, _, _ = p0_pts(q_va16 > th, gva16[:, ci], sh_va16)
            s17, _, _, _ = p0_pts(q_va17 > th, gva17[:, ci], sh_va17)
            if 0.5 * (s16 + s17) > best_va_m:
                best_va_m = 0.5 * (s16 + s17)
                best_th = float(th)

        # Max 3 regularized coordinate steps fit ONLY on `va_500` (must improve BOTH s16_va and s17_va!)
        for _ in range(3):
            s16_cur, _, _, _ = p0_pts(q_va16 > best_th, gva16[:, ci], sh_va16)
            s17_cur, _, _, _ = p0_pts(q_va17 > best_th, gva17[:, ci], sh_va17)
            best_step = None
            for mk in common_mkeys:
                m_va16 = sp_ctx[16]['models'][mk]['va_p0'][:, ci]
                m_va17 = sp_ctx[17]['models'][mk]['va_p0'][:, ci]
                m_te16 = sp_ctx[16]['models'][mk]['te_p0'][:, ci]
                m_te17 = sp_ctx[17]['models'][mk]['te_p0'][:, ci]
                for w in [0.15, 0.25]:
                    nq_va16 = (1 - w) * q_va16 + w * m_va16
                    nq_va17 = (1 - w) * q_va17 + w * m_va17
                    for th in np.linspace(0.28, 0.50, 12):
                        ns16, _, _, _ = p0_pts(nq_va16 > th, gva16[:, ci], sh_va16)
                        ns17, _, _, _ = p0_pts(nq_va17 > th, gva17[:, ci], sh_va17)
                        if 0.5 * (ns16 + ns17) > best_va_m + 1e-3 and ns16 >= s16_cur and ns17 >= s17_cur:
                            best_va_m = 0.5 * (ns16 + ns17)
                            best_step = (nq_va16, nq_va17, (1 - w) * q_te16 + w * m_te16, (1 - w) * q_te17 + w * m_te17, float(th))
            if best_step is None:
                break
            q_va16, q_va17, q_te16, q_te17, best_th = best_step

        va_p0_16[:, ci], va_p0_17[:, ci] = q_va16, q_va17
        te_p0_16[:, ci], te_p0_17[:, ci] = q_te16, q_te17
        best_t0[ci] = best_th
        pred_va[16][(q_va16 > best_th) & (~sh_va16), ci] = 3
        pred_va[17][(q_va17 > best_th) & (~sh_va17), ci] = 3
        pred_te[16][(q_te16 > best_th) & (~sh_te16), ci] = 3
        pred_te[17][(q_te17 > best_th) & (~sh_te17), ci] = 3

    for k, cname in enumerate(FULL):
        ci = CLASSES.index(cname)
        q1_va16, q1_va17 = va_p1_16[:, k].copy(), va_p1_17[:, k].copy()
        q2_va16, q2_va17 = va_p2_16[:, k].copy(), va_p2_17[:, k].copy()
        q1_te16, q1_te17 = te_p1_16[:, k].copy(), te_p1_17[:, k].copy()
        q2_te16, q2_te17 = te_p2_16[:, k].copy(), te_p2_17[:, k].copy()

        def eval_va_te(q1a_v, q2a_v, q1b_v, q2b_v, t2, t1p, tcum, t1, r12, w0):
            pts_l = []
            for p0_v, p1_v, p2_v, sh_v, g_v in [
                (va_p0_16[:, ci], q1a_v, q2a_v, sh_va16, gva16[:, ci]),
                (va_p0_17[:, ci], q1b_v, q2b_v, sh_va17, gva17[:, ci]),
            ]:
                cum = p0_v + p1_v + p2_v
                pres = (p2_v > t2) | (p1_v > t1p) | (cum > tcum)
                s1 = p1_v + w0 * p0_v
                isp1 = (s1 > t1) & (s1 >= r12 * p2_v)
                pr = np.where(pres, 1, 0)
                pr = np.where(pres & isp1, 2, pr)
                pr = np.where(p0_v > best_t0[ci], 3, pr)
                pr = np.where(sh_v, 0, pr)
                pts_l.append(exact_full_class_pts(pr, g_v, sh_v)[0])
            return 0.5 * (pts_l[0] + pts_l[1]), pts_l[0], pts_l[1]

        best_va_pts = -1.0
        best_p = None
        for t2 in [0.16, 0.20, 0.24, 0.28, 0.32]:
            for t1p in [0.20, 0.25, 0.30]:
                for tcum in [0.32, 0.38, 0.44, 0.50]:
                    for t1 in [0.18, 0.22, 0.26, 0.30]:
                        for r12 in [0.45, 0.65, 0.85, 1.05]:
                            for w0 in [0.0, 0.20, 0.35]:
                                m_va, s16_v, s17_v = eval_va_te(q1_va16, q2_va16, q1_va17, q2_va17, t2, t1p, tcum, t1, r12, w0)
                                if m_va > best_va_pts:
                                    best_va_pts = m_va
                                    best_p = (t2, t1p, tcum, t1, r12, w0, s16_v, s17_v)

        # Max 3 regularized coordinate steps on `va_500` (must improve BOTH s16_va and s17_va!)
        for _ in range(3):
            t2, t1p, tcum, t1, r12, w0, s16_cur, s17_cur = best_p
            best_step = None
            for mk in common_mkeys:
                m1_va16 = sp_ctx[16]['models'][mk]['va_p1'][:, k]
                m1_va17 = sp_ctx[17]['models'][mk]['va_p1'][:, k]
                m1_te16 = sp_ctx[16]['models'][mk]['te_p1'][:, k]
                m1_te17 = sp_ctx[17]['models'][mk]['te_p1'][:, k]
                m2_va16 = sp_ctx[16]['models'][mk]['va_p2'][:, k]
                m2_va17 = sp_ctx[17]['models'][mk]['va_p2'][:, k]
                m2_te16 = sp_ctx[16]['models'][mk]['te_p2'][:, k]
                m2_te17 = sp_ctx[17]['models'][mk]['te_p2'][:, k]
                for w in [0.15, 0.25]:
                    nq1_va16 = (1 - w) * q1_va16 + w * m1_va16
                    nq1_va17 = (1 - w) * q1_va17 + w * m1_va17
                    m_va, ns16, ns17 = eval_va_te(nq1_va16, q2_va16, nq1_va17, q2_va17, t2, t1p, tcum, t1, r12, w0)
                    if m_va > best_va_pts + 1e-3 and ns16 >= s16_cur and ns17 >= s17_cur:
                        best_va_pts = m_va
                        best_step = ('q1', nq1_va16, nq1_va17, (1 - w) * q1_te16 + w * m1_te16, (1 - w) * q1_te17 + w * m1_te17, (t2, t1p, tcum, t1, r12, w0, ns16, ns17))

                    nq2_va16 = (1 - w) * q2_va16 + w * m2_va16
                    nq2_va17 = (1 - w) * q2_va17 + w * m2_va17
                    m_va, ns16, ns17 = eval_va_te(q1_va16, nq2_va16, q1_va17, nq2_va17, t2, t1p, tcum, t1, r12, w0)
                    if m_va > best_va_pts + 1e-3 and ns16 >= s16_cur and ns17 >= s17_cur:
                        best_va_pts = m_va
                        best_step = ('q2', nq2_va16, nq2_va17, (1 - w) * q2_te16 + w * m2_te16, (1 - w) * q2_te17 + w * m2_te17, (t2, t1p, tcum, t1, r12, w0, ns16, ns17))
            if best_step is None:
                break
            which, v16, v17, t16, t17, best_p = best_step
            if which == 'q1':
                q1_va16, q1_va17, q1_te16, q1_te17 = v16, v17, t16, t17
            else:
                q2_va16, q2_va17, q2_te16, q2_te17 = v16, v17, t16, t17

        t2, t1p, tcum, t1, r12, w0, _, _ = best_p
        for sp, p0_v, p1_v, p2_v, sh_a, pr_dict in [
            (16, va_p0_16[:, ci], q1_va16, q2_va16, sh_va16, pred_va),
            (17, va_p0_17[:, ci], q1_va17, q2_va17, sh_va17, pred_va),
            (16, te_p0_16[:, ci], q1_te16, q2_te16, sh_te16, pred_te),
            (17, te_p0_17[:, ci], q1_te17, q2_te17, sh_te17, pred_te),
        ]:
            cum = p0_v + p1_v + p2_v
            pres = (p2_v > t2) | (p1_v > t1p) | (cum > tcum)
            s1 = p1_v + w0 * p0_v
            isp1 = (s1 > t1) & (s1 >= r12 * p2_v)
            pr = np.where(pres, 1, 0)
            pr = np.where(pres & isp1, 2, pr)
            pr = np.where(p0_v > best_t0[ci], 3, pr)
            pr = np.where(sh_a, 0, pr)
            pr_dict[sp][:, ci] = pr

    va_m16 = cm.score(gva16, pred_va[16], 0.02)
    va_m17 = cm.score(gva17, pred_va[17], 0.02)
    te_m16 = cm.score(gte16, pred_te[16], 0.02)
    te_m17 = cm.score(gte17, pred_te[17], 0.02)
    print(f"  [V9 Regularized 3-Step Blend (Fit ONLY on va_500 -> Evaluated One-Shot on te_1000)]")
    print(f"    Val  (500 Fit)       : s16={va_m16['total']:.2f}, s17={va_m17['total']:.2f} | MEAN={0.5*(va_m16['total']+va_m17['total']):.2f}")
    print(f"    Test (1000 Read-Only): s16={te_m16['total']:.2f}, s17={te_m17['total']:.2f} | MEAN={0.5*(te_m16['total']+te_m17['total']):.2f} "
          f"(S_pos={0.5*(te_m16['S_pos']+te_m17['S_pos']):.3f}, S_p0={0.5*(te_m16['S_p0']+te_m17['S_p0']):.3f}, S_clean={0.5*(te_m16['S_clean']+te_m17['S_clean']):.3f})")


if __name__ == '__main__':
    run_strict_zero_leakage_audit()
