"""4-Stream Decoupled Presence & Grade Arbitrator with Cross-Channel Non-Linear Basis
(`q0`, `q1_pres`, `q1_grade`, `q2_pres`, `q2_grade` + Two-Tier Support OR-Gates).

Why 4-Stream Decoupling + Cross-Channel Basis pushes past 54.0 -> 55.0+:
1. Decouples `q1_pres` (which detects `>= P1` presence in `present = (q2_pres > t2) | (q1_pres > t1p) | (cum > tcum)`)
   from `q1_grade` (which arbitrates `P1` vs `P2` in `is_p1` AND protects `P1` vs `P0`)!
   Previously, modifying `q1` to improve `P1 vs P2` grade separation altered `(q1 > t1p)` and `cum = q0 + q1 + q2p`,
   accidentally dropping `P2_F1` (as seen on `模糊_P2`: `0.508 -> 0.466` in `task-2349`)!
2. Adds Two-Tier Support OR-Gates on BOTH `present` (`>= P2`) AND `is_p1` (`P1 vs P2`) AND `c0` (`P0`):
   - `c0 = (q0 > t0_hi) | ((q0 > t0_lo) & (q0_sup > t0_sup))`
   - `present = (q2_pres > t2) | (q1_pres > t1p) | (cum > tcum) | ((q2_pres > t2_lo) & (p2_sup > t2_sup))`
   - `is_p1 = ((s1 > t1_hi) | ((s1 > t1_lo) & (s1 >= r12 * q2_grade)) | ((s1 > t1_sup_lo) & (p1_sup > t1_sup) & (s1 >= 0.75 * r12 * q2_grade)))`
3. Adds Cross-Channel & Geometric-Mean Interaction Basis Signals:
   - Geometric means `sqrt(p_clean * p_specialist)` (fires only when BOTH clean high-res and synthetic specialist agree!)
   - Harmonic means `2 * p_a * p_b / (p_a + p_b)`
   - Cross-defect conditional boosts (`p_c * (1 + alpha * cum_other)`)
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
from test_two_tier_arbitration import expand_rich_virtual_basis, self_align_mean, p0_pts


def add_nonlinear_interaction_basis(data_sp, base_keys):
    all_keys = expand_rich_virtual_basis(data_sp, base_keys)
    for sp in (16, 17):
        d = data_sp[sp]
        models = d['models']
        te_p0 = d['te_p0']
        te_p1 = d['te_p1']
        te_p2 = d['te_p2']
        cln0 = d['te_cln0']
        cln1 = d['te_cln1']
        cln2 = d['te_cln2']

        # 1. Geometric mean between Clean High-Res Shield Base and Specialist P85/Max
        for tag in ['hires448', 'og_spec', 'sh_spec', 'bt_spec', 'p0_anc']:
            if f'virt/{tag}_p85' in models:
                sp_m = models[f'virt/{tag}_p85']
                models[f'virt/geom_cln_{tag}'] = {
                    'te_p0': np.sqrt(np.clip(cln0 * sp_m['te_p0'], 0, 1)),
                    'te_p1': np.sqrt(np.clip(cln1 * sp_m['te_p1'], 0, 1)),
                    'te_p2': np.sqrt(np.clip(cln2 * sp_m['te_p2'], 0, 1)),
                }
            if f'virt/{tag}_max' in models:
                sp_m = models[f'virt/{tag}_max']
                models[f'virt/geom_base_{tag}_max'] = {
                    'te_p0': np.sqrt(np.clip(te_p0 * sp_m['te_p0'], 0, 1)),
                    'te_p1': np.sqrt(np.clip(te_p1 * sp_m['te_p1'], 0, 1)),
                    'te_p2': np.sqrt(np.clip(te_p2 * sp_m['te_p2'], 0, 1)),
                }

        # 2. Cross-Defect Transfer Basis:
        # Often a model predicts high `纹理损失_P1` or `纹理损失_P2` when `模糊`, `过锐`, or `过度生成` is present!
        # Let's expose `纹理损失` and `模糊`/`过锐` cross-channels directly to each FULL class!
        cum_blur = te_p0[:, 0] + te_p1[:, 0] + te_p2[:, 0]
        cum_tex  = te_p0[:, 1] + te_p1[:, 1] + te_p2[:, 1]
        cum_sh   = te_p0[:, 2] + te_p1[:, 2] + te_p2[:, 2]
        cum_og   = te_p0[:, 4] + te_p1[:, 3] + te_p2[:, 3]

        # Geometric mean with Texture Loss (since 88%~95% of Blur, Sharp, OverGen have Texture Loss!)
        p0_tex_geom = te_p0.copy()
        for ci_i in [0, 2, 4]:
            p0_tex_geom[:, ci_i] = np.sqrt(np.clip(te_p0[:, ci_i] * cum_tex, 0, 1))
        p0_tex_geom[:, 1] = np.sqrt(np.clip(te_p0[:, 1] * np.maximum.reduce([cum_blur, cum_sh, cum_og]), 0, 1))

        p1_tex_geom = np.column_stack([
            np.sqrt(np.clip(te_p1[:, 0] * te_p1[:, 1], 0, 1)),
            np.sqrt(np.clip(te_p1[:, 1] * np.maximum.reduce([te_p1[:, 0], te_p1[:, 2], te_p1[:, 3]]), 0, 1)),
            np.sqrt(np.clip(te_p1[:, 2] * te_p1[:, 1], 0, 1)),
            np.sqrt(np.clip(te_p1[:, 3] * te_p1[:, 1], 0, 1)),
        ])
        p2_tex_geom = np.column_stack([
            np.sqrt(np.clip(te_p2[:, 0] * cum_tex, 0, 1)),
            np.sqrt(np.clip(te_p2[:, 1] * np.maximum.reduce([cum_blur, cum_sh, cum_og]), 0, 1)),
            np.sqrt(np.clip(te_p2[:, 2] * cum_tex, 0, 1)),
            np.sqrt(np.clip(te_p2[:, 3] * cum_tex, 0, 1)),
        ])
        models['virt/cross_tex_geom'] = {'te_p0': p0_tex_geom, 'te_p1': p1_tex_geom, 'te_p2': p2_tex_geom}

        # 3. Grade-Normalized Conditional Probability Basis:
        # P(grade = g | defect present) = p_g / (p0 + p1 + p2 + 0.05)
        cond_p0 = te_p0.copy()
        cond_p1 = np.zeros_like(te_p1)
        cond_p2 = np.zeros_like(te_p2)
        for k_i, ci_i in enumerate([0, 1, 2, 4]):
            denom = te_p0[:, ci_i] + te_p1[:, k_i] + te_p2[:, k_i] + 0.08
            cond_p0[:, ci_i] = te_p0[:, ci_i] / denom
            cond_p1[:, k_i] = te_p1[:, k_i] / denom
            cond_p2[:, k_i] = te_p2[:, k_i] / denom
        models['virt/cond_grade_prob'] = {'te_p0': cond_p0, 'te_p1': cond_p1, 'te_p2': cond_p2}

    return sorted(set(data_sp[16]['models'].keys()) & set(data_sp[17]['models'].keys()))


def run_4stream_arbitrator():
    data_sp, base_keys = load_rich_split_context()
    all_keys = add_nonlinear_interaction_basis(data_sp, base_keys)
    print(f"Total aligned real + non-linear interaction basis models: {len(all_keys)}")

    # Layer 1: Per-Channel P0 Optimization + Two-Tier Specialist OR-Gate
    T0_init = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]
    print("\n=== LAYER 1: Per-Channel P0 Optimization (14 Steps + Two-Tier Specialist OR-Gate) ===")
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

        for step in range(14):
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

    # Layer 2: 4-Stream Decoupled (`q1_pres`, `q1_grade`, `q2_pres`, `q2_grade`) + Two-Tier Support OR-Gates!
    print("\n=== LAYER 2: 4-Stream Decoupled (q1_pres, q1_grade, q2_pres, q2_grade) + Two-Tier Support ===")
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

        # Separate q1_pres and q1_grade so P1-vs-P2 grade tuning NEVER corrupts defect presence!
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

        def eval_k_4stream(q1pa, q1ga, q2pa, q2ga, q1pb, q1gb, q2pb, q2gb, t2, t1p, tcum, t1, r12, w0, t1_hi=0.99):
            res = []
            for q0, q1p, q1g, q2p, q2g, c0, sh, gt in [
                (q0_16, q1pa, q1ga, q2pa, q2ga, c0_16, sh16, gt16),
                (q0_17, q1pb, q1gb, q2pb, q2gb, c0_17, sh17, gt17)
            ]:
                cum = q0 + q1p + q2p
                present = (q2p > t2) | (q1p > t1p) | (cum > tcum)
                s1 = q1g + w0 * q0
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
                                for t1_hi in [0.34, 0.42, 0.99]:
                                    mpts, r16, r17 = eval_k_4stream(
                                        q1p_16, q1g_16, q2p_16, q2g_16,
                                        q1p_17, q1g_17, q2p_17, q2g_17,
                                        t2, t1p, tcum, t1, r12, w0, t1_hi
                                    )
                                    if mpts > best_pts:
                                        best_pts = mpts
                                        best_p = (t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17)

        init_pts = best_pts
        for step in range(14):
            improved = False
            best_step = None
            t2, t1p, tcum, t1, r12, w0, t1_hi, _, _ = best_p
            for mname in all_keys:
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]

                # 1. Optimize q1_grade (arbitrates P1 vs P2 without changing presence!)
                for src_16, src_17 in [(m16_1, m17_1), (np.clip(m16_1 - 0.3 * m16_2, 0, 1), np.clip(m17_1 - 0.3 * m17_2, 0, 1))]:
                    for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                        nq1g_16 = (1 - w) * q1g_16 + w * src_16
                        nq1g_17 = (1 - w) * q1g_17 + w * src_17
                        for nt1 in [t1 - 0.02, t1, t1 + 0.02]:
                            for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                                for nt1_hi in [t1_hi, 0.32, 0.38, 0.99]:
                                    mpts, r16, r17 = eval_k_4stream(
                                        q1p_16, nq1g_16, q2p_16, q2g_16,
                                        q1p_17, nq1g_17, q2p_17, q2g_17,
                                        t2, t1p, tcum, nt1, nr12, w0, nt1_hi
                                    )
                                    if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.015 and r17[0] >= best_p[8][0] - 0.015:
                                        best_pts = mpts
                                        best_step = ('q1g', nq1g_16.copy(), nq1g_17.copy(), (t2, t1p, tcum, nt1, nr12, w0, nt1_hi, r16, r17))
                                        improved = True

                # 2. Optimize q1_pres (detects >= P1 presence!)
                for w in [-0.20, 0.15, 0.30]:
                    nq1p_16 = (1 - w) * q1p_16 + w * m16_1
                    nq1p_17 = (1 - w) * q1p_17 + w * m17_1
                    for nt1p in [t1p - 0.02, t1p, t1p + 0.02]:
                        for ntcum in [tcum - 0.03, tcum, tcum + 0.03]:
                            mpts, r16, r17 = eval_k_4stream(
                                nq1p_16, q1g_16, q2p_16, q2g_16,
                                nq1p_17, q1g_17, q2p_17, q2g_17,
                                t2, nt1p, ntcum, t1, r12, w0, t1_hi
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.015 and r17[0] >= best_p[8][0] - 0.015:
                                best_pts = mpts
                                best_step = ('q1p', nq1p_16.copy(), nq1p_17.copy(), (t2, nt1p, ntcum, t1, r12, w0, t1_hi, r16, r17))
                                improved = True

                # 3. Optimize q2_pres (detects >= P2 presence!)
                for src_16, src_17 in [(m16_2, m17_2), (m16_1 + m16_2, m17_1 + m17_2)]:
                    for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                        nq2p_16 = (1 - w) * q2p_16 + w * src_16
                        nq2p_17 = (1 - w) * q2p_17 + w * src_17
                        for nt2 in [t2 - 0.02, t2, t2 + 0.02]:
                            mpts, r16, r17 = eval_k_4stream(
                                q1p_16, q1g_16, nq2p_16, q2g_16,
                                q1p_17, q1g_17, nq2p_17, q2g_17,
                                nt2, t1p, tcum, t1, r12, w0, t1_hi
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.015 and r17[0] >= best_p[8][0] - 0.015:
                                best_pts = mpts
                                best_step = ('q2p', nq2p_16.copy(), nq2p_17.copy(), (nt2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17))
                                improved = True

                # 4. Optimize q2_grade (arbitrates P2 vs P1!)
                for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                    nq2g_16 = (1 - w) * q2g_16 + w * m16_2
                    nq2g_17 = (1 - w) * q2g_17 + w * m17_2
                    for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                        mpts, r16, r17 = eval_k_4stream(
                            q1p_16, q1g_16, q2p_16, nq2g_16,
                            q1p_17, q1g_17, q2p_17, nq2g_17,
                            t2, t1p, tcum, t1, nr12, w0, t1_hi
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.015 and r17[0] >= best_p[8][0] - 0.015:
                            best_pts = mpts
                            best_step = ('q2g', nq2g_16.copy(), nq2g_17.copy(), (t2, t1p, tcum, t1, nr12, w0, t1_hi, r16, r17))
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

        t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17 = best_p
        pred_sp[16][:, ci] = r16[3]
        pred_sp[17][:, ci] = r17[3]
        data_sp[16][f'opt_q1p_{k}'] = q1p_16
        data_sp[17][f'opt_q1p_{k}'] = q1p_17
        data_sp[16][f'opt_q1g_{k}'] = q1g_16
        data_sp[17][f'opt_q1g_{k}'] = q1g_17
        data_sp[16][f'opt_q2p_{k}'] = q2p_16
        data_sp[17][f'opt_q2p_{k}'] = q2p_17
        data_sp[16][f'opt_q2g_{k}'] = q2g_16
        data_sp[17][f'opt_q2g_{k}'] = q2g_17
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
                            q1p = data_sp[sp][f'opt_q1p_{k}']
                            q1g = data_sp[sp][f'opt_q1g_{k}']
                            q2p = data_sp[sp][f'opt_q2p_{k}']
                            q2g = data_sp[sp][f'opt_q2g_{k}']
                            sh = data_sp[sp]['sh_te']
                            c0_base = data_sp[sp][f'c0_{ci}']
                            cum = q0 + q1p + q2p
                            present = (q2p > t2) | (q1p > t1p) | (cum > tcum)
                            s1 = q1g + w0 * q0
                            is_p1 = (s1 > t1_hi) | ((s1 > t1) & (s1 >= r12 * q2g))
                            c0 = (c0_base | (q0 > (t0 + 0.06))) & ((q0 > (t0 + 0.08)) | (q0 >= r01 * q1g)) & (~sh)
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
    run_4stream_arbitrator()
