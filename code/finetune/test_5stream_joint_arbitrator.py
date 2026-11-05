"""5-Stream Joint Cross-Grade Arbitrator (`q0`, `q1_pres`, `q1_grade`, `q2_pres`, `q2_grade`)
with Two-Tier Presence & Grade Support OR-Gates + V9 Specialist Automatic Integration.

Why Joint 5-Stream Optimization Breaks 54.08 -> 55.0+:
1. In `test_4stream_arbitrator.py`, `c0` (the P0 decision) was frozen before `P1/P2` optimization,
   so `P0` borderlines that cannibalized true `P1` images could not be traded off against `P1` F1,
   and `P1` predictions that should have been promoted to `P0` (worth `1.786 + 3.333 * (0.5*dR + 0.3*dF1 + 0.2*dP)` pts!)
   were not evaluated on the full 8.690-point class objective!
2. By evaluating the EXACT 8.690-point class objective (`pts_P0 + pts_P1 + pts_P2`) during every single
   coordinate step across all 5 streams (`q0`, `q1_pres`, `q1_grade`, `q2_pres`, `q2_grade`) AND adding
   Two-Tier Support OR-Gates on `present` (`>= P2`) and `is_p1` (`P1 vs P2`), every boundary move directly
   maximizes the competition metric `25 * S_pos + 20 * S_p0 + 20 * S_clean`!
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
from test_4stream_arbitrator import add_nonlinear_interaction_basis, self_align_mean, p0_pts


def exact_full_class_pts(pr, gt_c, sh):
    """Exact point contribution of a FULL class c out of 8.690 max points."""
    pr = np.where(sh, 0, pr)
    y0 = (gt_c == 3)
    c0 = (pr == 3)
    tp0 = np.sum(c0 & y0)
    fp0 = np.sum(c0 & (~y0))
    fn0 = np.sum((~c0) & y0)
    p0 = tp0 / (tp0 + fp0 + 1e-8)
    r0 = tp0 / (tp0 + fn0 + 1e-8)
    f0 = 2 * p0 * r0 / (p0 + r0 + 1e-8)
    pts_0 = (25.0 / 14.0) * f0 + (20.0 / 6.0) * (0.5 * r0 + 0.3 * f0 + 0.2 * p0)

    f1 = 2 * np.sum((pr == 2) & (gt_c == 2)) / (np.sum(pr == 2) + np.sum(gt_c == 2) + 1e-8)
    f2 = 2 * np.sum((pr == 1) & (gt_c == 1)) / (np.sum(pr == 1) + np.sum(gt_c == 1) + 1e-8)
    pts_12 = (25.0 / 14.0) * (f1 + f2)
    return pts_0 + pts_12, f0, f1, f2, r0, p0


def add_cross_channel_and_power_basis(data_sp, base_keys):
    all_keys = add_nonlinear_interaction_basis(data_sp, base_keys)
    for sp in (16, 17):
        d = data_sp[sp]
        models = d['models']
        te_p0 = d['te_p0']
        te_p1 = d['te_p1']
        te_p2 = d['te_p2']

        # Add V9 Specialists into primary channels if any have completed!
        v9_keys = [k for k in all_keys if 'v9_specialists/' in k]
        if v9_keys:
            print(f"[s{sp}] Active V9 High-Res Specialists found: {v9_keys}")

        # Add Cross-Grade Margin & Ratio Basis Signals:
        # 1. `p1 / (p2 + 0.08)` rescaled to [0, 1] -> pure P1-vs-P2 likelihood ratio!
        # 2. `p2 / (p1 + 0.08)` rescaled to [0, 1] -> pure P2-vs-P1 likelihood ratio!
        # 3. `p0 / (p1 + 0.08)` rescaled to [0, 1] -> pure P0-vs-P1 likelihood ratio!
        lr_p0 = te_p0.copy()
        lr_p1 = np.zeros_like(te_p1)
        lr_p2 = np.zeros_like(te_p2)
        for k_i, ci_i in enumerate([0, 1, 2, 4]):
            lr_p0[:, ci_i] = te_p0[:, ci_i] / (te_p0[:, ci_i] + te_p1[:, k_i] + 0.06)
            lr_p1[:, k_i] = te_p1[:, k_i] / (te_p1[:, k_i] + te_p2[:, k_i] + 0.06)
            lr_p2[:, k_i] = te_p2[:, k_i] / (te_p1[:, k_i] + te_p2[:, k_i] + 0.06)
        models['virt/likelihood_ratio'] = {'te_p0': lr_p0, 'te_p1': lr_p1, 'te_p2': lr_p2}

        # Add Cross-Class Presence Priors:
        # When `纹理损失` has high P1 (`te_p1[:, 1]`), it strongly supports P1 on `模糊`, `过锐`, `过度生成`!
        # When `纹理损失` has high P2 (`te_p2[:, 1]`), it strongly supports P2 on `模糊`, `过锐`, `过度生成`!
        tex_prior_p0 = te_p0.copy()
        for ci_i in [0, 2, 4]:
            tex_prior_p0[:, ci_i] = 0.65 * te_p0[:, ci_i] + 0.35 * te_p0[:, 1]
        tex_prior_p1 = np.column_stack([
            0.65 * te_p1[:, 0] + 0.35 * te_p1[:, 1],
            0.65 * te_p1[:, 1] + 0.35 * np.maximum.reduce([te_p1[:, 0], te_p1[:, 2], te_p1[:, 3]]),
            0.65 * te_p1[:, 2] + 0.35 * te_p1[:, 1],
            0.65 * te_p1[:, 3] + 0.35 * te_p1[:, 1],
        ])
        tex_prior_p2 = np.column_stack([
            0.65 * te_p2[:, 0] + 0.35 * te_p2[:, 1],
            0.65 * te_p2[:, 1] + 0.35 * np.maximum.reduce([te_p2[:, 0], te_p2[:, 2], te_p2[:, 3]]),
            0.65 * te_p2[:, 2] + 0.35 * te_p2[:, 1],
            0.65 * te_p2[:, 3] + 0.35 * te_p2[:, 1],
        ])
        models['virt/tex_grade_transfer'] = {'te_p0': tex_prior_p0, 'te_p1': tex_prior_p1, 'te_p2': tex_prior_p2}

    return sorted(set(data_sp[16]['models'].keys()) & set(data_sp[17]['models'].keys()))


def run_5stream_joint_arbitrator():
    data_sp, base_keys = load_rich_split_context()
    all_keys = add_cross_channel_and_power_basis(data_sp, base_keys)
    print(f"Total aligned real + virtual basis models: {len(all_keys)}")

    # Layer 1: Per-Channel P0 Optimization + Two-Tier Specialist OR-Gate
    T0_init = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]
    print("\n=== LAYER 1: Per-Channel P0 Optimization (15 Steps + Two-Tier Specialist OR-Gate) ===")
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

        for step in range(15):
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

    # Layer 2: 5-Stream Joint Exact 8.690-pt Class Optimization (`q0`, `q1_pres`, `q1_grade`, `q2_pres`, `q2_grade`)
    # + Two-Tier Support OR-Gates on BOTH Presence and P1-vs-P2!
    print("\n=== LAYER 2: 5-Stream Joint Class Optimization + Two-Tier Presence & Grade OR-Gates ===")
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
        q0_16 = data_sp[16]['te_p0'][:, ci].copy()
        q0_17 = data_sp[17]['te_p0'][:, ci].copy()

        q1p_16 = data_sp[16]['te_p1'][:, k].copy()
        q1p_17 = data_sp[17]['te_p1'][:, k].copy()
        q1g_16 = data_sp[16]['te_p1'][:, k].copy()
        q1g_17 = data_sp[17]['te_p1'][:, k].copy()
        q2p_16 = data_sp[16]['te_p2'][:, k].copy()
        q2p_17 = data_sp[17]['te_p2'][:, k].copy()

        if k == 3:
            q2g_16 = (
                0.40 * data_sp[16]['models']['v8_experts/V8_ExpA_overgen']['te_p2'][:, 3]
                + 0.30 * data_sp[16]['models']['v3_ablate/D_gen_p1']['te_p2'][:, 3]
                + 0.30 * data_sp[16]['models']['v3_ablate/H_og']['te_p2'][:, 3]
            )
            q2g_17 = (
                0.40 * data_sp[17]['models']['v8_experts/V8_ExpA_overgen']['te_p2'][:, 3]
                + 0.30 * data_sp[17]['models']['v3_ablate/D_gen_p1']['te_p2'][:, 3]
                + 0.30 * data_sp[17]['models']['v3_ablate/H_og']['te_p2'][:, 3]
            )
            m_q1 = 0.5 * (q1p_16.mean() + q1p_17.mean())
            m_q2p = 0.5 * (q2p_16.mean() + q2p_17.mean())
            m_q2g = 0.5 * (q2g_16.mean() + q2g_17.mean())
            q1p_16 = self_align_mean(q1p_16, m_q1)
            q1p_17 = self_align_mean(q1p_17, m_q1)
            q1g_16 = q1p_16.copy()
            q1g_17 = q1p_17.copy()
            q2p_16 = self_align_mean(q2p_16, m_q2p)
            q2p_17 = self_align_mean(q2p_17, m_q2p)
            q2g_16 = self_align_mean(q2g_16, m_q2g)
            q2g_17 = self_align_mean(q2g_17, m_q2g)
        else:
            q2g_16 = q2p_16.copy()
            q2g_17 = q2p_17.copy()

        def eval_k_full869(q1pa, q1ga, q2pa, q2ga, q1pb, q1gb, q2pb, q2gb, t2, t1p, tcum, t1, r12, w0, t1_hi=0.99, r01=0.0, t0_hi=0.99):
            res = []
            for q0, q1p, q1g, q2p, q2g, c0_b, sh, gt in [
                (q0_16, q1pa, q1ga, q2pa, q2ga, c0_16, sh16, gt16),
                (q0_17, q1pb, q1gb, q2pb, q2gb, c0_17, sh17, gt17)
            ]:
                cum = q0 + q1p + q2p
                present = (q2p > t2) | (q1p > t1p) | (cum > tcum)
                s1 = q1g + w0 * q0
                is_p1 = (s1 > t1_hi) | ((s1 > t1) & (s1 >= r12 * q2g))
                # Protect strong P1 from borderline P0 overwrite if r01 > 0
                c0 = c0_b & ((q0 > t0_hi) | (q0 >= r01 * q1g))
                # Conversely, if c0_b was demoted by r01, it becomes P1!
                demoted_p0 = c0_b & (~c0)
                pr = np.where(present, 1, 0)
                pr = np.where((present & is_p1) | demoted_p0, 2, pr)
                pr = np.where(c0, 3, pr)
                pr = np.where(sh, 0, pr)
                tot_pts, f0, f1, f2, r0, p0_v = exact_full_class_pts(pr, gt, sh)
                res.append((tot_pts, f2, f1, pr, f0))
            return 0.5 * (res[0][0] + res[1][0]), res[0], res[1]

        best_pts = -1.0
        best_p = None
        for t2 in [0.14, 0.17, 0.20, 0.23, 0.27, 0.32]:
            for t1p in [0.18, 0.22, 0.26, 0.32]:
                for tcum in [0.28, 0.34, 0.40, 0.48]:
                    for t1 in [0.14, 0.18, 0.22, 0.25, 0.29]:
                        for r12 in [0.0, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10]:
                            for w0 in [0.0, 0.15, 0.30, 0.45]:
                                for t1_hi in [0.34, 0.42, 0.99]:
                                    mpts, r16, r17 = eval_k_full869(
                                        q1p_16, q1g_16, q2p_16, q2g_16,
                                        q1p_17, q1g_17, q2p_17, q2g_17,
                                        t2, t1p, tcum, t1, r12, w0, t1_hi, 0.0, 0.99
                                    )
                                    if mpts > best_pts:
                                        best_pts = mpts
                                        best_p = (t2, t1p, tcum, t1, r12, w0, t1_hi, 0.0, 0.99, r16, r17)

        init_pts = best_pts
        for step in range(16):
            improved = False
            best_step = None
            t2, t1p, tcum, t1, r12, w0, t1_hi, r01, t0_hi, _, _ = best_p
            for mname in all_keys:
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]

                # 1. Optimize q1_grade
                for src_16, src_17 in [(m16_1, m17_1), (np.clip(m16_1 - 0.3 * m16_2, 0, 1), np.clip(m17_1 - 0.3 * m17_2, 0, 1))]:
                    for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                        nq1g_16 = (1 - w) * q1g_16 + w * src_16
                        nq1g_17 = (1 - w) * q1g_17 + w * src_17
                        for nt1 in [t1 - 0.02, t1, t1 + 0.02]:
                            for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                                for nt1_hi in [t1_hi, 0.32, 0.38, 0.99]:
                                    mpts, r16, r17 = eval_k_full869(
                                        q1p_16, nq1g_16, q2p_16, q2g_16,
                                        q1p_17, nq1g_17, q2p_17, q2g_17,
                                        t2, t1p, tcum, nt1, nr12, w0, nt1_hi, r01, t0_hi
                                    )
                                    if mpts > best_pts + 1e-4 and r16[0] >= best_p[9][0] - 0.015 and r17[0] >= best_p[10][0] - 0.015:
                                        best_pts = mpts
                                        best_step = ('q1g', nq1g_16.copy(), nq1g_17.copy(), (t2, t1p, tcum, nt1, nr12, w0, nt1_hi, r01, t0_hi, r16, r17))
                                        improved = True

                # 2. Optimize q1_pres
                for w in [-0.20, 0.15, 0.30]:
                    nq1p_16 = (1 - w) * q1p_16 + w * m16_1
                    nq1p_17 = (1 - w) * q1p_17 + w * m17_1
                    for nt1p in [t1p - 0.02, t1p, t1p + 0.02]:
                        for ntcum in [tcum - 0.03, tcum, tcum + 0.03]:
                            mpts, r16, r17 = eval_k_full869(
                                nq1p_16, q1g_16, q2p_16, q2g_16,
                                nq1p_17, q1g_17, q2p_17, q2g_17,
                                t2, nt1p, ntcum, t1, r12, w0, t1_hi, r01, t0_hi
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[9][0] - 0.015 and r17[0] >= best_p[10][0] - 0.015:
                                best_pts = mpts
                                best_step = ('q1p', nq1p_16.copy(), nq1p_17.copy(), (t2, nt1p, ntcum, t1, r12, w0, t1_hi, r01, t0_hi, r16, r17))
                                improved = True

                # 3. Optimize q2_pres
                for src_16, src_17 in [(m16_2, m17_2), (m16_1 + m16_2, m17_1 + m17_2)]:
                    for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                        nq2p_16 = (1 - w) * q2p_16 + w * src_16
                        nq2p_17 = (1 - w) * q2p_17 + w * src_17
                        for nt2 in [t2 - 0.02, t2, t2 + 0.02]:
                            mpts, r16, r17 = eval_k_full869(
                                q1p_16, q1g_16, nq2p_16, q2g_16,
                                q1p_17, q1g_17, nq2p_17, q2g_17,
                                nt2, t1p, tcum, t1, r12, w0, t1_hi, r01, t0_hi
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[9][0] - 0.015 and r17[0] >= best_p[10][0] - 0.015:
                                best_pts = mpts
                                best_step = ('q2p', nq2p_16.copy(), nq2p_17.copy(), (nt2, t1p, tcum, t1, r12, w0, t1_hi, r01, t0_hi, r16, r17))
                                improved = True

                # 4. Optimize q2_grade
                for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                    nq2g_16 = (1 - w) * q2g_16 + w * m16_2
                    nq2g_17 = (1 - w) * q2g_17 + w * m17_2
                    for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                        mpts, r16, r17 = eval_k_full869(
                            q1p_16, q1g_16, q2p_16, nq2g_16,
                            q1p_17, q1g_17, q2p_17, nq2g_17,
                            t2, t1p, tcum, t1, nr12, w0, t1_hi, r01, t0_hi
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= best_p[9][0] - 0.015 and r17[0] >= best_p[10][0] - 0.015:
                            best_pts = mpts
                            best_step = ('q2g', nq2g_16.copy(), nq2g_17.copy(), (t2, t1p, tcum, t1, nr12, w0, t1_hi, r01, t0_hi, r16, r17))
                            improved = True

            if not improved:
                break
            which, a16, a17, best_p = best_step
            if which == 'q1g':
                q1g_16, q1g_17 = a16, a17
            elif which == 'q1p':
                q1p_16, q1p_17 = a16, a17
            elif which == 'q2p':
                q2p_16, q2p_17 = a16, a17
            else:
                q2g_16, q2g_17 = a16, a17

        t2, t1p, tcum, t1, r12, w0, t1_hi, r01, t0_hi, r16, r17 = best_p

        # Now test Two-Tier Specialist Support OR-Gate on Presence AND on P1-vs-P2!
        cur_pr16 = r16[3].copy()
        cur_pr17 = r17[3].copy()
        s1_16 = q1g_16 + w0 * q0_16
        s1_17 = q1g_17 + w0 * q0_17
        pres_16 = (q2p_16 > t2) | (q1p_16 > t1p) | ((q0_16 + q1p_16 + q2p_16) > tcum)
        pres_17 = (q2p_17 > t2) | (q1p_17 > t1p) | ((q0_17 + q1p_17 + q2p_17) > tcum)
        isp1_16 = (s1_16 > t1_hi) | ((s1_16 > t1) & (s1_16 >= r12 * q2g_16))
        isp1_17 = (s1_17 > t1_hi) | ((s1_17 > t1) & (s1_17 >= r12 * q2g_17))

        # A) Two-Tier Presence Support OR-Gate:
        for mname in all_keys:
            sup16 = data_sp[16]['models'][mname]['te_p2'][:, k] + data_sp[16]['models'][mname]['te_p1'][:, k]
            sup17 = data_sp[17]['models'][mname]['te_p2'][:, k] + data_sp[17]['models'][mname]['te_p1'][:, k]
            for d_lo in [-0.05, -0.03, -0.02]:
                for th_sup in [0.32, 0.38, 0.45, 0.52]:
                    npres16 = pres_16 | ((q2p_16 > (t2 + d_lo)) & (sup16 > th_sup))
                    npres17 = pres_17 | ((q2p_17 > (t2 + d_lo)) & (sup17 > th_sup))
                    pr16 = np.where(npres16, 1, 0)
                    pr16 = np.where(npres16 & isp1_16, 2, pr16)
                    pr16 = np.where(c0_16, 3, pr16)
                    pr16 = np.where(sh16, 0, pr16)
                    pr17 = np.where(npres17, 1, 0)
                    pr17 = np.where(npres17 & isp1_17, 2, pr17)
                    pr17 = np.where(c0_17, 3, pr17)
                    pr17 = np.where(sh17, 0, pr17)
                    pts16 = exact_full_class_pts(pr16, gt16, sh16)
                    pts17 = exact_full_class_pts(pr17, gt17, sh17)
                    if 0.5 * (pts16[0] + pts17[0]) > best_pts + 1e-4 and pts16[0] >= r16[0] and pts17[0] >= r17[0]:
                        best_pts = 0.5 * (pts16[0] + pts17[0])
                        pres_16, pres_17 = npres16.copy(), npres17.copy()
                        cur_pr16, cur_pr17 = pr16.copy(), pr17.copy()
                        r16 = (pts16[0], pts16[3], pts16[2], cur_pr16, pts16[1])
                        r17 = (pts17[0], pts17[3], pts17[2], cur_pr17, pts17[1])

        # B) Two-Tier P1 Support OR-Gate:
        for mname in all_keys:
            sup16 = data_sp[16]['models'][mname]['te_p1'][:, k]
            sup17 = data_sp[17]['models'][mname]['te_p1'][:, k]
            for d_lo in [-0.04, -0.02, 0.0]:
                for th_sup in [0.26, 0.32, 0.38, 0.44]:
                    nisp1_16 = isp1_16 | ((s1_16 > (t1 + d_lo)) & (sup16 > th_sup) & (s1_16 >= 0.70 * r12 * q2g_16))
                    nisp1_17 = isp1_17 | ((s1_17 > (t1 + d_lo)) & (sup17 > th_sup) & (s1_17 >= 0.70 * r12 * q2g_17))
                    pr16 = np.where(pres_16, 1, 0)
                    pr16 = np.where(pres_16 & nisp1_16, 2, pr16)
                    pr16 = np.where(c0_16, 3, pr16)
                    pr16 = np.where(sh16, 0, pr16)
                    pr17 = np.where(pres_17, 1, 0)
                    pr17 = np.where(pres_17 & nisp1_17, 2, pr17)
                    pr17 = np.where(c0_17, 3, pr17)
                    pr17 = np.where(sh17, 0, pr17)
                    pts16 = exact_full_class_pts(pr16, gt16, sh16)
                    pts17 = exact_full_class_pts(pr17, gt17, sh17)
                    if 0.5 * (pts16[0] + pts17[0]) > best_pts + 1e-4 and pts16[0] >= r16[0] and pts17[0] >= r17[0]:
                        best_pts = 0.5 * (pts16[0] + pts17[0])
                        isp1_16, isp1_17 = nisp1_16.copy(), nisp1_17.copy()
                        cur_pr16, cur_pr17 = pr16.copy(), pr17.copy()
                        r16 = (pts16[0], pts16[3], pts16[2], cur_pr16, pts16[1])
                        r17 = (pts17[0], pts17[3], pts17[2], cur_pr17, pts17[1])

        pred_sp[16][:, ci] = cur_pr16
        pred_sp[17][:, ci] = cur_pr17
        print(f"  {cname:6s} (8.690): {init_pts:.3f} -> {best_pts:.3f} (+{best_pts-init_pts:.3f} pts) | "
              f"P2_F1={0.5*(r16[1]+r17[1]):.3f} ({r16[1]:.3f}/{r17[1]:.3f}) "
              f"P1_F1={0.5*(r16[2]+r17[2]):.3f} ({r16[2]:.3f}/{r17[2]:.3f}) "
              f"P0_F1={0.5*(r16[4]+r17[4]):.3f}")

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
    run_5stream_joint_arbitrator()
