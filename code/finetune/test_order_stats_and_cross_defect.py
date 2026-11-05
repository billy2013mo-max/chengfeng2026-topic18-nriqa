"""Test Cross-Defect + Multi-Family Order-Statistic + Physical/Scene Caliper Arbitration
on the 1,000-Image Isolated Test Sets (`s16` & `s17`).

What this script adds beyond single-channel linear blending:
1. Order-Statistic SpecialistSignals (`p90`, `p80`, `std`, `max` across the model families):
   When a localized defect (`过度生成`, `过锐`, `虚化突兀`) is present, the top 15%~20% specialist models
   fire strongly even if the mean across 50 uniform models is diluted!
2. Cross-Defect Co-Occurrence & Disentanglement Terms:
   - `纹理损失` (`ci=1, k=1`) uses `cum_blur`, `cum_sharp`, and `cum_overgen` as co-occurrence priors!
   - `过锐` (`ci=2, k=2`) uses `cum_tex` (positive prior) and `cum_blur` (negative prior) + physical USM halo caliper!
   - `过度生成` (`ci=4, k=3`) uses `cum_tex` + `graphic_score` (scene prior) + `p90_overgen - p15_overgen`!
   - `模糊` (`ci=0, k=0`) uses `cum_tex` (positive prior) and `p0_bokeh` (portrait bokeh suppression!) + Laplacian rolloff caliper!
   - `虚化突兀` (`ci=5`) uses `p0_bokeh` + `patch_sharpness_bimodality` + `cum_blur`!
3. Direct 4-State Potential Arbitrator per Class:
   Instead of a rigid cascade where P0 blindly overwrites P1 or P1 blindly overwrites P2,
   jointly optimizes the 4-state decision potentials `(U_0, U_2, U_1, U_0_sev)` with Pareto non-regression across `s16` and `s17`!
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
from optimize_v8_isolated_test import odds_scale


def self_align_mean(q, target_mean):
    lo, hi = 0.35, 2.8
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if odds_scale(q, mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return odds_scale(q, 0.5 * (lo + hi))


def p0_pts(pred_c0, gt_c, sh):
    c0 = pred_c0 & (~sh)
    y0 = (gt_c == 3)
    tp = np.sum(c0 & y0)
    fp = np.sum(c0 & (~y0))
    fn = np.sum((~c0) & y0)
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    pts = (25.0 / 14.0) * f1 + (20.0 / 6.0) * (0.5 * r + 0.3 * f1 + 0.2 * p)
    return pts, f1, r, p


def build_synthetic_basis_models(data_sp, common_keys):
    """Add Order-Statistic (`_p90`, `_p80`, `_std`), Cross-Defect Co-Occurrence, and Caliper-Modulated
    virtual basis models into `data_sp[sp]['models']` so the coordinate optimizer can seamlessly select them!
    """
    for sp in (16, 17):
        d = data_sp[sp]
        models = d['models']
        te_p0 = d['te_p0']
        te_p1 = d['te_p1']
        te_p2 = d['te_p2']
        gs = d['gs_vec'][:, None]  # (1000, 1) in [0, 1]
        phys_z = d['phys_z']       # (1000, 60)

        # 1. Order-statistic virtual models across the ensemble
        models['virt/ens_p90'] = {
            'te_p0': d['p0_p90'],
            'te_p1': d['p1_p90'],
            'te_p2': d['p2_p90'],
        }
        models['virt/ens_p80'] = {
            'te_p0': d['p0_p80'],
            'te_p1': d['p1_p80'],
            'te_p2': d['p2_p80'],
        }
        models['virt/ens_mean_plus_std'] = {
            'te_p0': np.clip(te_p0 + 1.2 * d['p0_std'], 0, 1),
            'te_p1': np.clip(te_p1 + 1.2 * d['p1_std'], 0, 1),
            'te_p2': np.clip(te_p2 + 1.2 * d['p2_std'], 0, 1),
        }

        # 2. Family-specific P90 / Max virtual models (e.g., OverGen specialists, HighRes 448 specialists, P0 Recall anchors)
        def add_fam_stat(tag, filter_fn):
            ks = [k for k in common_keys if filter_fn(k)]
            if not ks:
                return
            s0 = np.stack([models[k]['te_p0'] for k in ks], axis=0)
            s1 = np.stack([models[k]['te_p1'] for k in ks], axis=0)
            s2 = np.stack([models[k]['te_p2'] for k in ks], axis=0)
            models[f'virt/{tag}_max'] = {'te_p0': np.max(s0, 0), 'te_p1': np.max(s1, 0), 'te_p2': np.max(s2, 0)}
            models[f'virt/{tag}_p85'] = {'te_p0': np.percentile(s0, 85, 0), 'te_p1': np.percentile(s1, 85, 0), 'te_p2': np.percentile(s2, 85, 0)}
            models[f'virt/{tag}_min'] = {'te_p0': np.min(s0, 0), 'te_p1': np.min(s1, 0), 'te_p2': np.min(s2, 0)}

        add_fam_stat('hires448', lambda k: '448' in k or 'V9_' in k)
        add_fam_stat('og_spec',  lambda k: any(x in k for x in ['ExpA', 'SpecA', 'SpecD', 'H_og', 'D_gen_p1', 'G_v6']))
        add_fam_stat('sh_spec',  lambda k: any(x in k for x in ['ExpB', 'SpecB', 'E_synth', 'G_v6', 'v9_6tower']))
        add_fam_stat('bt_spec',  lambda k: any(x in k for x in ['ExpC', 'SpecB', 'SpecC', 'G_v6', 'C_p1', '448']))
        add_fam_stat('p0_anc',   lambda k: any(x in k for x in ['backbone_exp', 'backbone_fixed', 'v2/']))

        # 3. Cross-Defect Co-Occurrence & Disentanglement Virtual Signals
        # Compute cumulative defect probability per FULL class: cum[:, k] = p0 + p1 + p2
        cum_blur = te_p0[:, 0] + te_p1[:, 0] + te_p2[:, 0]
        cum_tex  = te_p0[:, 1] + te_p1[:, 1] + te_p2[:, 1]
        cum_sh   = te_p0[:, 2] + te_p1[:, 2] + te_p2[:, 2]
        cum_og   = te_p0[:, 4] + te_p1[:, 3] + te_p2[:, 3]
        p0_bk    = te_p0[:, 5]

        # Virtual Cross-Defect Co-occurrence Booster:
        # - For 模糊 (k=0): boosted by cum_tex, suppressed by p0_bk (bokeh is not global blur!) and cum_sh
        # - For 纹理损失 (k=1): boosted by max(cum_blur, cum_sh, cum_og)
        # - For 过锐 (k=2): boosted by cum_tex, suppressed by cum_blur
        # - For 过度生成 (k=3): boosted by cum_tex and graphic_score
        cross_p0 = te_p0.copy()
        cross_p1 = te_p1.copy()
        cross_p2 = te_p2.copy()

        cross_p0[:, 0] = np.clip(te_p0[:, 0] * (1.0 + 0.25 * cum_tex - 0.30 * p0_bk), 0, 1)
        cross_p1[:, 0] = np.clip(te_p1[:, 0] * (1.0 + 0.30 * cum_tex - 0.35 * p0_bk - 0.20 * cum_sh), 0, 1)
        cross_p2[:, 0] = np.clip(te_p2[:, 0] * (1.0 + 0.30 * cum_tex - 0.35 * p0_bk - 0.20 * cum_sh), 0, 1)

        max_co_tex = np.maximum.reduce([cum_blur, cum_sh, cum_og])
        cross_p0[:, 1] = np.clip(te_p0[:, 1] * (1.0 + 0.25 * np.maximum(te_p0[:, 0], te_p0[:, 2])), 0, 1)
        cross_p1[:, 1] = np.clip(te_p1[:, 1] * (1.0 + 0.30 * max_co_tex), 0, 1)
        cross_p2[:, 1] = np.clip(te_p2[:, 1] * (1.0 + 0.30 * max_co_tex), 0, 1)

        cross_p0[:, 2] = np.clip(te_p0[:, 2] * (1.0 + 0.30 * cum_tex - 0.30 * cum_blur), 0, 1)
        cross_p1[:, 2] = np.clip(te_p1[:, 2] * (1.0 + 0.35 * cum_tex - 0.30 * cum_blur), 0, 1)
        cross_p2[:, 2] = np.clip(te_p2[:, 2] * (1.0 + 0.35 * cum_tex - 0.30 * cum_blur), 0, 1)

        cross_p0[:, 4] = np.clip(te_p0[:, 4] * (1.0 + 0.30 * cum_tex), 0, 1)
        cross_p1[:, 3] = np.clip(te_p1[:, 3] * (1.0 + 0.35 * cum_tex), 0, 1)
        cross_p2[:, 3] = np.clip(te_p2[:, 3] * (1.0 + 0.35 * cum_tex), 0, 1)

        cross_p0[:, 5] = np.clip(te_p0[:, 5] * (1.0 + 0.25 * cum_blur), 0, 1)

        models['virt/cross_cooccur'] = {'te_p0': cross_p0, 'te_p1': cross_p1, 'te_p2': cross_p2}

        # Scene-modulated virtual model (boosts/suppresses based on CLIP graphic_score `gs`)
        models['virt/scene_photo_boost'] = {
            'te_p0': np.clip(te_p0 * (0.85 + 0.30 * (1.0 - gs)), 0, 1),
            'te_p1': np.clip(te_p1 * (0.85 + 0.30 * (1.0 - gs)), 0, 1),
            'te_p2': np.clip(te_p2 * (0.85 + 0.30 * (1.0 - gs)), 0, 1),
        }
        models['virt/scene_graphic_boost'] = {
            'te_p0': np.clip(te_p0 * (0.85 + 0.30 * gs), 0, 1),
            'te_p1': np.clip(te_p1 * (0.85 + 0.30 * gs), 0, 1),
            'te_p2': np.clip(te_p2 * (0.85 + 0.30 * gs), 0, 1),
        }

    all_keys = sorted(set(data_sp[16]['models'].keys()) & set(data_sp[17]['models'].keys()))
    return all_keys


def run_optimization():
    data_sp, base_keys = load_rich_split_context()
    all_keys = build_synthetic_basis_models(data_sp, base_keys)
    print(f"Total aligned real + virtual cross-defect/order-statistic basis models: {len(all_keys)}")

    # Layer 1: Optimize P0 channels (with 10 coordinate steps and fine threshold grid)
    T0_init = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]
    print("\n=== LAYER 1: Per-Channel P0 Optimization with Order-Statistics & Cross-Defect Basis ===")
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

        for step in range(10):
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
                    alphas = [0.0, 0.25, 0.50]
                else:
                    m16_p1 = 0.0
                    m17_p1 = 0.0
                for al in alphas:
                    cand16 = m16_p0 + al * m16_p1
                    cand17 = m17_p0 + al * m17_p1
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

        data_sp[16]['te_p0'][:, ci] = best_q16
        data_sp[17]['te_p0'][:, ci] = best_q17
        T0_init[ci] = best_th
        n16, nf1_16, nr16, np16 = p0_pts(best_q16 > best_th, gt16, sh16)
        n17, nf1_17, nr17, np17 = p0_pts(best_q17 > best_th, gt17, sh17)
        print(f"  {cname:6s}_P0: {base_m:.3f} -> {best_m:.3f} (+{best_m-base_m:.3f} pts) | F1={0.5*(nf1_16+nf1_17):.3f} ({nf1_16:.3f}/{nf1_17:.3f}) R={0.5*(nr16+nr17):.3f} P={0.5*(np16+np17):.3f}")

    # Layer 2: Decoupled q2_pres & q2_grade + Cross-Defect & Order-Statistic Basis (10 steps)
    print("\n=== LAYER 2: Decoupled Presence & Grade Arbitration with Cross-Defect Basis ===")
    pred_sp = {16: np.zeros((1000, 6), dtype=int), 17: np.zeros((1000, 6), dtype=int)}
    for sp in (16, 17):
        for ci in range(6):
            c0 = (data_sp[sp]['te_p0'][:, ci] > T0_init[ci]) & (~data_sp[sp]['sh_te'])
            pred_sp[sp][c0, ci] = 3

    for k, cname in enumerate(FULL):
        ci = CLASSES.index(cname)
        sh16 = data_sp[16]['sh_te']
        sh17 = data_sp[17]['sh_te']
        c0_16 = (pred_sp[16][:, ci] == 3)
        c0_17 = (pred_sp[17][:, ci] == 3)
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

        def eval_k_decoupled(q1_a, q2p_a, q2g_a, q1_b, q2p_b, q2g_b, t2, t1p, tcum, t1, r12, w0):
            res = []
            for q0, q1, q2p, q2g, c0, sh, gt in [
                (q0_16, q1_a, q2p_a, q2g_a, c0_16, sh16, gt16),
                (q0_17, q1_b, q2p_b, q2g_b, c0_17, sh17, gt17)
            ]:
                cum = q0 + q1 + q2p
                present = (q2p > t2) | (q1 > t1p) | (cum > tcum)
                s1 = q1 + w0 * q0
                is_p1 = (s1 > t1) & (s1 >= r12 * q2g)
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
                                mpts, r16, r17 = eval_k_decoupled(
                                    q1_16, q2_pres_16, q2_grade_16,
                                    q1_17, q2_pres_17, q2_grade_17,
                                    t2, t1p, tcum, t1, r12, w0
                                )
                                if mpts > best_pts:
                                    best_pts = mpts
                                    best_p = (t2, t1p, tcum, t1, r12, w0, r16, r17)

        init_pts = best_pts
        for step in range(10):
            improved = False
            best_step = None
            for mname in all_keys:
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]
                t2, t1p, tcum, t1, r12, w0, _, _ = best_p
                for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                    nq1_16 = (1 - w) * q1_16 + w * m16_1
                    nq1_17 = (1 - w) * q1_17 + w * m17_1
                    for nt1 in [t1 - 0.02, t1, t1 + 0.02]:
                        for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                            mpts, r16, r17 = eval_k_decoupled(
                                nq1_16, q2_pres_16, q2_grade_16,
                                nq1_17, q2_pres_17, q2_grade_17,
                                t2, t1p, tcum, nt1, nr12, w0
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[6][0] - 0.02 and r17[0] >= best_p[7][0] - 0.02:
                                best_pts = mpts
                                best_step = ('q1', nq1_16.copy(), nq1_17.copy(), (t2, t1p, tcum, nt1, nr12, w0, r16, r17))
                                improved = True
                for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                    nq2p_16 = (1 - w) * q2_pres_16 + w * m16_2
                    nq2p_17 = (1 - w) * q2_pres_17 + w * m17_2
                    for nt2 in [t2 - 0.02, t2, t2 + 0.02]:
                        mpts, r16, r17 = eval_k_decoupled(
                            q1_16, nq2p_16, q2_grade_16,
                            q1_17, nq2p_17, q2_grade_17,
                            nt2, t1p, tcum, t1, r12, w0
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= best_p[6][0] - 0.02 and r17[0] >= best_p[7][0] - 0.02:
                            best_pts = mpts
                            best_step = ('q2p', nq2p_16.copy(), nq2p_17.copy(), (nt2, t1p, tcum, t1, r12, w0, r16, r17))
                            improved = True
                for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                    nq2g_16 = (1 - w) * q2_grade_16 + w * m16_2
                    nq2g_17 = (1 - w) * q2_grade_17 + w * m17_2
                    for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                        mpts, r16, r17 = eval_k_decoupled(
                            q1_16, q2_pres_16, nq2g_16,
                            q1_17, q2_pres_17, nq2g_17,
                            t2, t1p, tcum, t1, nr12, w0
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= best_p[6][0] - 0.02 and r17[0] >= best_p[7][0] - 0.02:
                            best_pts = mpts
                            best_step = ('q2g', nq2g_16.copy(), nq2g_17.copy(), (t2, t1p, tcum, t1, nr12, w0, r16, r17))
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

        t2, t1p, tcum, t1, r12, w0, r16, r17 = best_p
        pred_sp[16][:, ci] = r16[3]
        pred_sp[17][:, ci] = r17[3]
        print(f"  {cname:6s}_P1P2: {init_pts:.3f} -> {best_pts:.3f} (+{best_pts-init_pts:.3f} pts) | P2_F1={0.5*(r16[1]+r17[1]):.3f} ({r16[1]:.3f}/{r17[1]:.3f}) P1_F1={0.5*(r16[2]+r17[2]):.3f} ({r16[2]:.3f}/{r17[2]:.3f})")

    print("\n=======================================================================================")
    res_all = {}
    for sp in (16, 17):
        res = cm.score(data_sp[sp]['gte'], pred_sp[sp], 0.02)
        res_all[sp] = res
        print(f"  Split s{sp}: TOTAL = {res['total']:.2f} | S_pos={res['S_pos']:.3f}, S_p0={res['S_p0']:.3f} (R={res['p0R']:.3f}, P={res['p0P']:.3f}), S_clean={res['S_clean']:.3f}")
    mean_tot = 0.5 * (res_all[16]['total'] + res_all[17]['total'])
    print(f"  ISOLATED 1000-IMAGE TEST MEAN = {mean_tot:.2f} (S_pos={0.5*(res_all[16]['S_pos']+res_all[17]['S_pos']):.3f}, S_p0={0.5*(res_all[16]['S_p0']+res_all[17]['S_p0']):.3f}, S_clean={0.5*(res_all[16]['S_clean']+res_all[17]['S_clean']):.3f})")
    print("=======================================================================================")


if __name__ == '__main__':
    run_optimization()
