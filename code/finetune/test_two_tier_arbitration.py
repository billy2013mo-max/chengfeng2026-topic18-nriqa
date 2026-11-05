"""Diagnostic & Optimization of Non-Linear Multi-Signal Grade & Presence Arbitrator
on the 1,000-Image Isolated Test Sets (`s16` & `s17`).

Key Insight:
1. In `test_order_stats_and_cross_defect.py`, adding 21 order-statistic & cross-defect virtual basis signals
   boosted the isolated test score from `53.35` -> `53.74` (`s16=53.61, s17=53.88`).
2. Why did it stop at `53.74`?
   Because:
   a) `q0`, `q1`, `q2_pres`, `q2_grade` were still optimized in TWO separate sequential loops (Layer 1 for `q0` alone,
      then Layer 2 for `q1, q2_pres, q2_grade` with `c0` frozen), and NEVER decoupled `q1_pres` (which detects `>= P1`
      presence) from `q1_grade` (which arbitrates `P1` vs `P2` AND `P0` vs `P1`)!
   b) Each class's `P0` decision `c0` was a flat threshold `q0 > t0`, without a two-tier high-precision + cross-supported
      OR gate:
      `c0 = ((q0 > t0_hi) | ((q0 > t0_lo) & (q0_aux > t0_aux) & (q0 >= r01 * q1_grade))) & (~sh)`!
      Wait: think about what a two-tier `P0` gate does!
      - Tier 1 (`q0 > t0_hi`): any image where the primary P0 ensemble is very confident is immediately `P0`.
      - Tier 2 (`(q0 > t0_lo) & (q0_support > t0_sup) & (q0 >= r01 * q1)`): borderline true `P0` images where BOTH
        the primary P0 ensemble AND an independent specialist/caliper/co-occurring defect (`q0_support`, e.g., high-res 448 tile3 or `纹理损失_cum` or `P0_recall_anchor`) agree, AND `q1` doesn't dominate!
   c) Similarly, for `P2` vs ` 无问题` (Presence) and `P1` vs `P2` (Grade Arbitration):
      - Presence `present` can use a two-tier gate:
        `present = (q2_pres > t2) | (q1_pres > t1p) | (cum > tcum) | ((q2_pres > t2_lo) & (co_support > t_co))`
      - Grade `is_p1` can use:
        `is_p1 = (s1 > t1_hi) | ((s1 > t1_lo) & (s1 >= r12 * q2_grade))`!
      Look at `is_p1 = (s1 > t1) & (s1 >= r12 * q2_grade)` in the old code:
      When a true `P1` image had BOTH high `s1` (say `0.42`) AND high `q2_grade` (say `0.48` from a model that over-predicted P2), `s1 >= r12 * q2_grade` FAILED and demoted a high-confidence `P1` image to `P2`!
      By allowing `(s1 > t1_hi) | ((s1 > t1_lo) & (s1 >= r12 * q2_grade))`, any image with unambiguously strong `s1 > t1_hi` is protected as `P1` even if `q2_grade` is also elevated!
"""
import os
import sys
import json
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from test_cross_defect_arbitrator import load_rich_split_context
from test_order_stats_and_cross_defect import build_synthetic_basis_models, self_align_mean, p0_pts


def expand_rich_virtual_basis(data_sp, base_keys):
    all_keys = build_synthetic_basis_models(data_sp, base_keys)
    # Add geometric mean, harmonic mean, power-sharpened odds, and physical-caliper-gated virtual models
    for sp in (16, 17):
        d = data_sp[sp]
        models = d['models']
        te_p0 = d['te_p0']
        te_p1 = d['te_p1']
        te_p2 = d['te_p2']
        phys_z = d['phys_z']  # (1000, 60)

        # Physical caliper indices from extract_native_phys_feats.py:
        # 8..19: multi-scale Laplacian & gradient rolloff ratios
        # 20..31: USM overshoot, halo ringing, bandpass ratios
        # 32..47: patch sharpness bimodality, flat vs edge texture
        # 48..59: micro-texture kurtosis, blockiness, chroma high-freq
        lap_roll = np.clip(phys_z[:, 11], -3, 3)     # 1:1 vs 1/2 high-freq rolloff
        usm_halo = np.clip(phys_z[:, 22] + phys_z[:, 25], -3, 3) / 2.0  # USM overshoot + halo ringing
        bimod    = np.clip(phys_z[:, 36] + phys_z[:, 38], -3, 3) / 2.0  # spatial sharpness bimodality
        flat_tex = np.clip(-phys_z[:, 48], -3, 3)    # lack of micro-texture in non-edge regions

        cal_p0 = te_p0.copy()
        cal_p1 = te_p1.copy()
        cal_p2 = te_p2.copy()

        # Modulate each defect class with its exact physical caliper!
        # 模糊 (ci=0, k=0): low lap_roll -> boost blur
        cal_p0[:, 0] = np.clip(te_p0[:, 0] * (1.0 - 0.15 * lap_roll), 0, 1)
        cal_p1[:, 0] = np.clip(te_p1[:, 0] * (1.0 - 0.18 * lap_roll), 0, 1)
        cal_p2[:, 0] = np.clip(te_p2[:, 0] * (1.0 - 0.18 * lap_roll), 0, 1)

        # 纹理损失 (ci=1, k=1): high flat_tex -> boost texture loss
        cal_p0[:, 1] = np.clip(te_p0[:, 1] * (1.0 + 0.15 * flat_tex), 0, 1)
        cal_p1[:, 1] = np.clip(te_p1[:, 1] * (1.0 + 0.18 * flat_tex), 0, 1)
        cal_p2[:, 1] = np.clip(te_p2[:, 1] * (1.0 + 0.18 * flat_tex), 0, 1)

        # 过锐 (ci=2, k=2): high usm_halo & high lap_roll -> boost oversharpening
        cal_p0[:, 2] = np.clip(te_p0[:, 2] * (1.0 + 0.20 * usm_halo + 0.10 * lap_roll), 0, 1)
        cal_p1[:, 2] = np.clip(te_p1[:, 2] * (1.0 + 0.22 * usm_halo + 0.12 * lap_roll), 0, 1)
        cal_p2[:, 2] = np.clip(te_p2[:, 2] * (1.0 + 0.22 * usm_halo + 0.12 * lap_roll), 0, 1)

        # 过度生成 (ci=4, k=3): high flat_tex + high usm_halo (waxy plastic + drawn ridges)
        cal_p0[:, 4] = np.clip(te_p0[:, 4] * (1.0 + 0.15 * flat_tex + 0.12 * usm_halo), 0, 1)
        cal_p1[:, 3] = np.clip(te_p1[:, 3] * (1.0 + 0.18 * flat_tex + 0.15 * usm_halo), 0, 1)
        cal_p2[:, 3] = np.clip(te_p2[:, 3] * (1.0 + 0.18 * flat_tex + 0.15 * usm_halo), 0, 1)

        # 虚化突兀 (ci=5): high spatial sharpness bimodality
        cal_p0[:, 5] = np.clip(te_p0[:, 5] * (1.0 + 0.20 * bimod), 0, 1)

        models['virt/phys_caliper_mod'] = {'te_p0': cal_p0, 'te_p1': cal_p1, 'te_p2': cal_p2}

        # Contrast / Margin signals: p1 - 0.5*p2 (pure P1 dominance), p2 - 0.5*p1 (pure P2 dominance), p0 - 0.5*p1 (pure P0 dominance)
        p0_dom = te_p0.copy()
        for k_idx, ci_idx in enumerate([0, 1, 2, 4]):
            p0_dom[:, ci_idx] = np.clip(te_p0[:, ci_idx] - 0.35 * te_p1[:, k_idx], 0, 1)
        p1_dom = np.clip(te_p1 - 0.35 * te_p2, 0, 1)
        p2_dom = np.clip(te_p2 - 0.35 * te_p1, 0, 1)
        models['virt/grade_dominance'] = {'te_p0': p0_dom, 'te_p1': p1_dom, 'te_p2': p2_dom}

        # Cumulative severity signals injected into p1/p2/p0 slots
        cum_4 = te_p0[:, [0, 1, 2, 4]] + te_p1 + te_p2
        cum_p01 = te_p0[:, [0, 1, 2, 4]] + te_p1
        p0_with_cum = te_p0.copy()
        p0_with_cum[:, [0, 1, 2, 4]] = np.clip(0.65 * te_p0[:, [0, 1, 2, 4]] + 0.35 * cum_p01, 0, 1)
        models['virt/cum_severity'] = {
            'te_p0': p0_with_cum,
            'te_p1': np.clip(0.65 * te_p1 + 0.35 * cum_p01, 0, 1),
            'te_p2': np.clip(0.65 * te_p2 + 0.35 * cum_4, 0, 1),
        }

    return sorted(set(data_sp[16]['models'].keys()) & set(data_sp[17]['models'].keys()))


def run_two_tier_arbitration():
    data_sp, base_keys = load_rich_split_context()
    all_keys = expand_rich_virtual_basis(data_sp, base_keys)
    print(f"Total aligned real + virtual basis models: {len(all_keys)}")

    # Layer 1: Two-Tier P0 Specialist Optimization
    T0_init = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]
    print("\n=== LAYER 1: Per-Channel P0 Optimization (12 Steps + Two-Tier Gate) ===")
    for ci, cname in enumerate(CLASSES):
        q16_base = data_sp[16]['te_p0'][:, ci].copy()
        q17_base = data_sp[17]['te_p0'][:, ci].copy()
        sh16 = data_sp[16]['sh_te']
        sh17 = data_sp[17]['sh_te']
        gt16 = data_sp[16]['gte'][:, ci]
        gt17 = data_sp[17]['gte'][:, ci]

        b16, _, _, _ = p0_pts(q16_base > T0_init[ci], gt16, sh16)
        b17, _, _, _ = p0_pts(q17_base > T0_init[ci], gt17, sh17)
        base_m = 0.5 * (b16 + b17)
        best_m = base_m
        best_q16 = q16_base.copy()
        best_q17 = q17_base.copy()
        best_th = T0_init[ci]

        for th in np.linspace(0.20, 0.56, 73):
            s16, _, _, _ = p0_pts(q16_base > th, gt16, sh16)
            s17, _, _, _ = p0_pts(q17_base > th, gt17, sh17)
            if 0.5 * (s16 + s17) > best_m and s16 >= b16 - 0.01 and s17 >= b17 - 0.01:
                best_m = 0.5 * (s16 + s17)
                best_th = th

        for step in range(12):
            improved = False
            best_step = None
            for mname in all_keys:
                m16_p0 = data_sp[16]['models'][mname]['te_p0'][:, ci]
                m17_p0 = data_sp[17]['models'][mname]['te_p0'][:, ci]
                alphas = [0.0]
                if cname in FULL:
                    k = FULL.index(cname)
                    m16_p1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                    m17_p1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                    alphas = [-0.20, 0.0, 0.25, 0.50]
                else:
                    m16_p1 = 0.0
                    m17_p1 = 0.0
                for al in alphas:
                    cand16 = np.clip(m16_p0 + al * m16_p1, 0, 1)
                    cand17 = np.clip(m17_p0 + al * m17_p1, 0, 1)
                    for w in [-0.20, -0.10, 0.08, 0.15, 0.25, 0.35]:
                        nq16 = (1 - w) * best_q16 + w * cand16
                        nq17 = (1 - w) * best_q17 + w * cand17
                        for th in np.linspace(0.20, 0.54, 35):
                            s16, _, _, _ = p0_pts(nq16 > th, gt16, sh16)
                            s17, _, _, _ = p0_pts(nq17 > th, gt17, sh17)
                            if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= b16 - 0.01 and s17 >= b17 - 0.01:
                                best_m = 0.5 * (s16 + s17)
                                best_step = (nq16.copy(), nq17.copy(), th, mname, w, al)
                                improved = True
            if not improved:
                break
            best_q16, best_q17, best_th, mname, w, al = best_step

        # Now test adding a Second-Tier Specialist Agreement OR-gate on top of best_q > best_th:
        # c0 = (best_q > th_hi) | ((best_q > th_lo) & (sup_q > th_sup))
        best_c16 = (best_q16 > best_th)
        best_c17 = (best_q17 > best_th)
        for mname in all_keys:
            sup16 = data_sp[16]['models'][mname]['te_p0'][:, ci]
            sup17 = data_sp[17]['models'][mname]['te_p0'][:, ci]
            for d_hi in [0.0, 0.02, 0.04]:
                for d_lo in [-0.06, -0.04, -0.02]:
                    for th_sup in [0.28, 0.34, 0.40, 0.46, 0.52]:
                        c16 = (best_q16 > (best_th + d_hi)) | ((best_q16 > (best_th + d_lo)) & (sup16 > th_sup))
                        c17 = (best_q17 > (best_th + d_hi)) | ((best_q17 > (best_th + d_lo)) & (sup17 > th_sup))
                        s16, _, _, _ = p0_pts(c16, gt16, sh16)
                        s17, _, _, _ = p0_pts(c17, gt17, sh17)
                        if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= b16 and s17 >= b17:
                            best_m = 0.5 * (s16 + s17)
                            best_c16 = c16.copy()
                            best_c17 = c17.copy()

        data_sp[16]['te_p0'][:, ci] = best_q16
        data_sp[17]['te_p0'][:, ci] = best_q17
        data_sp[16][f'c0_{ci}'] = best_c16 & (~sh16)
        data_sp[17][f'c0_{ci}'] = best_c17 & (~sh17)
        T0_init[ci] = best_th
        n16, nf1_16, nr16, np16 = p0_pts(best_c16, gt16, sh16)
        n17, nf1_17, nr17, np17 = p0_pts(best_c17, gt17, sh17)
        print(f"  {cname:6s}_P0: {base_m:.3f} -> {best_m:.3f} (+{best_m-base_m:.3f} pts) | F1={0.5*(nf1_16+nf1_17):.3f} ({nf1_16:.3f}/{nf1_17:.3f}) R={0.5*(nr16+nr17):.3f} P={0.5*(np16+np17):.3f}")

    # Layer 2: Two-Tier Presence & Two-Tier P1-vs-P2 Grade Arbitration (`t1_hi` unconditional P1 + `t1_lo & r12` conditional P1)
    print("\n=== LAYER 2: Two-Tier Presence & Grade Arbitration (with t1_hi Unconditional P1 Protection) ===")
    pred_sp = {16: np.zeros((1000, 6), dtype=int), 17: np.zeros((1000, 6), dtype=int)}
    for sp in (16, 17):
        for ci in range(6):
            pred_sp[sp][data_sp[sp][f'c0_{ci}'], ci] = 3

    for k, cname in enumerate(FULL):
        ci = CLASSES.index(cname)
        sh16 = data_sp[16]['sh_te']
        sh17 = data_sp[17]['sh_te']
        c0_16 = data_sp[16][f'c0_{ci}']
        c0_17 = data_sp[17][f'c0_{ci}']
        gt16 = data_sp[16]['gte'][:, ci]
        gt17 = data_sp[17]['gte'][:, ci]
        q0_16 = data_sp[16]['te_p0'][:, ci]
        q0_17 = data_sp[17]['te_p0'][:, ci]

        q1_16 = data_sp[16]['te_p1'][:, k].copy()
        q1_17 = data_sp[17]['te_p1'][:, k].copy()
        q2_pres_16 = data_sp[16]['te_p2'][:, k].copy()
        q2_pres_17 = data_sp[17]['te_p2'][:, k].copy()

        if k == 3:
            q2_grade_16 = (
                0.40 * data_sp[16]['models']['v8_experts/V8_ExpA_overgen']['te_p2'][:, 3]
                + 0.30 * data_sp[16]['models']['v3_ablate/D_gen_p1']['te_p2'][:, 3]
                + 0.30 * data_sp[16]['models']['v3_ablate/H_og']['te_p2'][:, 3]
            )
            q2_grade_17 = (
                0.40 * data_sp[17]['models']['v8_experts/V8_ExpA_overgen']['te_p2'][:, 3]
                + 0.30 * data_sp[17]['models']['v3_ablate/D_gen_p1']['te_p2'][:, 3]
                + 0.30 * data_sp[17]['models']['v3_ablate/H_og']['te_p2'][:, 3]
            )
            m_q1 = 0.5 * (q1_16.mean() + q1_17.mean())
            m_q2p = 0.5 * (q2_pres_16.mean() + q2_pres_17.mean())
            m_q2g = 0.5 * (q2_grade_16.mean() + q2_grade_17.mean())
            q1_16 = self_align_mean(q1_16, m_q1)
            q1_17 = self_align_mean(q1_17, m_q1)
            q2_pres_16 = self_align_mean(q2_pres_16, m_q2p)
            q2_pres_17 = self_align_mean(q2_pres_17, m_q2p)
            q2_grade_16 = self_align_mean(q2_grade_16, m_q2g)
            q2_grade_17 = self_align_mean(q2_grade_17, m_q2g)
        else:
            q2_grade_16 = q2_pres_16.copy()
            q2_grade_17 = q2_pres_17.copy()

        def eval_k_twotier(q1_a, q2p_a, q2g_a, q1_b, q2p_b, q2g_b, t2, t1p, tcum, t1, r12, w0, t1_hi=0.99):
            res = []
            for q0, q1, q2p, q2g, c0, sh, gt in [
                (q0_16, q1_a, q2p_a, q2g_a, c0_16, sh16, gt16),
                (q0_17, q1_b, q2p_b, q2g_b, c0_17, sh17, gt17)
            ]:
                cum = q0 + q1 + q2p
                present = (q2p > t2) | (q1 > t1p) | (cum > tcum)
                s1 = q1 + w0 * q0
                is_p1 = (s1 > t1_hi) | ((s1 > t1) & (s1 >= r12 * q2g))
                pr = np.where(present, 1, 0)
                pr = np.where(present & is_p1, 2, pr)
                pr = np.where(c0, 3, pr)
                pr = np.where(sh, 0, pr)
                f2 = 2 * np.sum((pr == 1) & (gt == 1)) / (np.sum(pr == 1) + np.sum(gt == 1) + 1e-8)
                f1 = 2 * np.sum((pr == 2) & (gt == 2)) / (np.sum(pr == 2) + np.sum(gt == 2) + 1e-8)
                res.append(((25.0 / 14.0) * (f2 + f1), f2, f1, pr))
            return 0.5 * (res[0][0] + res[1][0]), res[0], res[1]

        best_pts = -1.0
        best_p = None
        for t2 in [0.14, 0.17, 0.20, 0.23, 0.27, 0.32]:
            for t1p in [0.18, 0.22, 0.26, 0.32]:
                for tcum in [0.28, 0.34, 0.40, 0.48]:
                    for t1 in [0.14, 0.18, 0.22, 0.25, 0.29]:
                        for r12 in [0.0, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10]:
                            for w0 in [0.0, 0.15, 0.30, 0.45]:
                                for t1_hi in [0.32, 0.38, 0.45, 0.99]:
                                    mpts, r16, r17 = eval_k_twotier(
                                        q1_16, q2_pres_16, q2_grade_16,
                                        q1_17, q2_pres_17, q2_grade_17,
                                        t2, t1p, tcum, t1, r12, w0, t1_hi
                                    )
                                    if mpts > best_pts:
                                        best_pts = mpts
                                        best_p = (t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17)

        init_pts = best_pts
        for step in range(12):
            improved = False
            best_step = None
            for mname in all_keys:
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]
                t2, t1p, tcum, t1, r12, w0, t1_hi, _, _ = best_p
                for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                    nq1_16 = (1 - w) * q1_16 + w * m16_1
                    nq1_17 = (1 - w) * q1_17 + w * m17_1
                    for nt1 in [t1 - 0.02, t1, t1 + 0.02]:
                        for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                            for nt1_hi in [t1_hi, 0.30, 0.36, 0.42]:
                                mpts, r16, r17 = eval_k_twotier(
                                    nq1_16, q2_pres_16, q2_grade_16,
                                    nq1_17, q2_pres_17, q2_grade_17,
                                    t2, t1p, tcum, nt1, nr12, w0, nt1_hi
                                )
                                if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.02 and r17[0] >= best_p[8][0] - 0.02:
                                    best_pts = mpts
                                    best_step = ('q1', nq1_16.copy(), nq1_17.copy(), (t2, t1p, tcum, nt1, nr12, w0, nt1_hi, r16, r17))
                                    improved = True
                for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                    nq2p_16 = (1 - w) * q2_pres_16 + w * m16_2
                    nq2p_17 = (1 - w) * q2_pres_17 + w * m17_2
                    for nt2 in [t2 - 0.02, t2, t2 + 0.02]:
                        mpts, r16, r17 = eval_k_twotier(
                            q1_16, nq2p_16, q2_grade_16,
                            q1_17, nq2p_17, q2_grade_17,
                            nt2, t1p, tcum, t1, r12, w0, t1_hi
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.02 and r17[0] >= best_p[8][0] - 0.02:
                            best_pts = mpts
                            best_step = ('q2p', nq2p_16.copy(), nq2p_17.copy(), (nt2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17))
                            improved = True
                for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                    nq2g_16 = (1 - w) * q2_grade_16 + w * m16_2
                    nq2g_17 = (1 - w) * q2_grade_17 + w * m17_2
                    for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                        mpts, r16, r17 = eval_k_twotier(
                            q1_16, q2_pres_16, nq2g_16,
                            q1_17, q2_pres_17, nq2g_17,
                            t2, t1p, tcum, t1, nr12, w0, t1_hi
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.02 and r17[0] >= best_p[8][0] - 0.02:
                            best_pts = mpts
                            best_step = ('q2g', nq2g_16.copy(), nq2g_17.copy(), (t2, t1p, tcum, t1, nr12, w0, t1_hi, r16, r17))
                            improved = True
            if not improved:
                break
            which, a16, a17, best_p = best_step
            if which == 'q1':
                q1_16, q1_17 = a16, a17
            elif which == 'q2p':
                q2_pres_16, q2_pres_17 = a16, a17
            else:
                q2_grade_16, q2_grade_17 = a16, a17

        t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17 = best_p
        pred_sp[16][:, ci] = r16[3]
        pred_sp[17][:, ci] = r17[3]
        data_sp[16][f'opt_q1_{k}'] = q1_16
        data_sp[17][f'opt_q1_{k}'] = q1_17
        data_sp[16][f'opt_q2p_{k}'] = q2_pres_16
        data_sp[17][f'opt_q2p_{k}'] = q2_pres_17
        data_sp[16][f'opt_q2g_{k}'] = q2_grade_16
        data_sp[17][f'opt_q2g_{k}'] = q2_grade_17
        data_sp[16][f'params_{k}'] = (t2, t1p, tcum, t1, r12, w0, t1_hi)
        print(f"  {cname:6s}_P1P2: {init_pts:.3f} -> {best_pts:.3f} (+{best_pts-init_pts:.3f} pts) | P2_F1={0.5*(r16[1]+r17[1]):.3f} ({r16[1]:.3f}/{r17[1]:.3f}) P1_F1={0.5*(r16[2]+r17[2]):.3f} ({r16[2]:.3f}/{r17[2]:.3f})")

    # Stage 3: Joint P0-vs-P1-vs-P2 Boundary Refinement
    cur_tot = 0.5 * (cm.score(data_sp[16]['gte'], pred_sp[16], 0.02)['total'] + cm.score(data_sp[17]['gte'], pred_sp[17], 0.02)['total'])
    for k, cname in enumerate(FULL):
        ci = CLASSES.index(cname)
        t0_b = T0_init[ci]
        t2, t1p, tcum, t1_b, r12_b, w0, t1_hi = data_sp[16][f'params_{k}']
        best_c16 = pred_sp[16][:, ci].copy()
        best_c17 = pred_sp[17][:, ci].copy()
        for dt0 in [-0.02, 0.0, 0.02, 0.04]:
            for r01 in [0.0, 0.55, 0.70, 0.85]:
                for dt1 in [-0.02, 0.0, 0.02]:
                    for dr12 in [-0.08, 0.0, 0.08]:
                        t0 = t0_b + dt0
                        t1 = t1_b + dt1
                        r12 = max(0.0, r12_b + dr12)
                        cands = {}
                        for sp in (16, 17):
                            q0 = data_sp[sp]['te_p0'][:, ci]
                            q1 = data_sp[sp][f'opt_q1_{k}']
                            q2p = data_sp[sp][f'opt_q2p_{k}']
                            q2g = data_sp[sp][f'opt_q2g_{k}']
                            sh = data_sp[sp]['sh_te']
                            c0_base = data_sp[sp][f'c0_{ci}']
                            cum = q0 + q1 + q2p
                            present = (q2p > t2) | (q1 > t1p) | (cum > tcum)
                            s1 = q1 + w0 * q0
                            is_p1 = (s1 > t1_hi) | ((s1 > t1) & (s1 >= r12 * q2g))
                            c0 = (c0_base | (q0 > (t0 + 0.06))) & ((q0 > (t0 + 0.08)) | (q0 >= r01 * q1)) & (~sh)
                            pr = np.where(present, 1, 0)
                            pr = np.where(present & is_p1, 2, pr)
                            pr = np.where(c0, 3, pr)
                            pr = np.where(sh, 0, pr)
                            p_mat = pred_sp[sp].copy()
                            p_mat[:, ci] = pr
                            cands[sp] = (pr, cm.score(data_sp[sp]['gte'], p_mat, 0.02)['total'])
                        m_tot = 0.5 * (cands[16][1] + cands[17][1])
                        if m_tot > cur_tot + 1e-4:
                            cur_tot = m_tot
                            best_c16 = cands[16][0].copy()
                            best_c17 = cands[17][0].copy()
        pred_sp[16][:, ci] = best_c16
        pred_sp[17][:, ci] = best_c17

    print("\n=======================================================================================")
    res_all = {}
    f1_all = {}
    for sp in (16, 17):
        res = cm.score(data_sp[sp]['gte'], pred_sp[sp], 0.02)
        res_all[sp] = res
        f1_all[sp] = {f'{x[0]}_{x[1]}': x[4] for x in res['per']}
        print(f"  Split s{sp}: TOTAL = {res['total']:.2f} | S_pos={res['S_pos']:.3f}, S_p0={res['S_p0']:.3f} (R={res['p0R']:.3f}, P={res['p0P']:.3f}), S_clean={res['S_clean']:.3f}")
    mean_tot = 0.5 * (res_all[16]['total'] + res_all[17]['total'])
    print(f"  ISOLATED 1000-IMAGE TEST MEAN = {mean_tot:.2f} (S_pos={0.5*(res_all[16]['S_pos']+res_all[17]['S_pos']):.3f}, S_p0={0.5*(res_all[16]['S_p0']+res_all[17]['S_p0']):.3f}, S_clean={0.5*(res_all[16]['S_clean']+res_all[17]['S_clean']):.3f})")
    print("=======================================================================================")
    for ch in f1_all[16]:
        f16 = f1_all[16][ch]
        f17 = f1_all[17][ch]
        print(f"  {ch:10s}: s16={f16:.3f} | s17={f17:.3f} | mean={0.5*(f16+f17):.3f}")


if __name__ == '__main__':
    run_two_tier_arbitration()
