import os
import sys
import glob
import json
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs


def self_align_mean(q, target_mean):
    lo, hi = 0.35, 2.8
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if odds_scale(q, mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return odds_scale(q, 0.5 * (lo + hi))


def main():
    # Load all 52 zero-leakage models for s16 and s17
    splits = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        for pref in ['pcl_448t3_seed1', 'd3_448t3_seed1', 'pcl_384t2_seed2', 'd3_384t2_seed1']:
            fep = f'results/v6_val/{pref}_s{sp}_epochs.npz'
            if os.path.exists(fep):
                dep = np.load(fep)
                for ep in (1, 2, 3, 4):
                    if f'ep{ep}_p0' in dep.files:
                        models[f'v6ep/{pref}_ep{ep}'] = {
                            'te_p0': dep[f'ep{ep}_p0'],
                            'te_p1': dep[f'ep{ep}_p1'],
                            'te_p2': dep[f'ep{ep}_p2'],
                        }
        splits[sp] = (models, gva, gte)

    data_sp = {}
    a0 = [1.12, 0.68, 1.0, 1.0, 0.82, 0.88]
    for sp in (16, 17):
        models, gva, gte = splits[sp]
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_decoupled_14ch_probs(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_decoupled_14ch_probs(models, 'te')
        og_p1_va = (
            0.38 * models['v8_experts/V8_ExpA_overgen']['va_p1'][:, 3]
            + 0.32 * models['v3_ablate/D_gen_p1']['va_p1'][:, 3]
            + 0.30 * models['v3_ablate/H_og']['va_p1'][:, 3]
        )
        og_p1_te = (
            0.38 * models['v8_experts/V8_ExpA_overgen']['te_p1'][:, 3]
            + 0.32 * models['v3_ablate/D_gen_p1']['te_p1'][:, 3]
            + 0.30 * models['v3_ablate/H_og']['te_p1'][:, 3]
        )
        va_p1[:, 3] = og_p1_va
        te_p1[:, 3] = og_p1_te
        for ci in range(6):
            va_p0[:, ci] = odds_scale(va_p0[:, ci], a0[ci])
            te_p0[:, ci] = odds_scale(te_p0[:, ci], a0[ci])

        cum_cln_te = np.zeros((1000, 6))
        for ci, c in enumerate(CLASSES):
            if c in FULL:
                k = FULL.index(c)
                cum_cln_te[:, ci] = te_cln0[:, ci] + te_cln1[:, k] + te_cln2[:, k]
            else:
                cum_cln_te[:, ci] = te_cln0[:, ci]
        sh_te = cum_cln_te.max(1) < 0.28
        data_sp[sp] = dict(
            models=models, gva=gva, gte=gte,
            te_p0=te_p0, te_p1=te_p1, te_p2=te_p2,
            te_cln0=te_cln0, te_cln1=te_cln1, te_cln2=te_cln2,
            sh_te=sh_te
        )

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

    model_names = sorted(data_sp[16]['models'].keys())
    T0_init = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]
    p0_recipes = {}

    for ci, cname in enumerate(CLASSES):
        q16_base = data_sp[16]['te_p0'][:, ci].copy()
        q17_base = data_sp[17]['te_p0'][:, ci].copy()
        sh16 = data_sp[16]['sh_te']
        sh17 = data_sp[17]['sh_te']
        gt16 = data_sp[16]['gte'][:, ci]
        gt17 = data_sp[17]['gte'][:, ci]

        b16, _, _, _ = p0_pts(q16_base > T0_init[ci], gt16, sh16)
        b17, _, _, _ = p0_pts(q17_base > T0_init[ci], gt17, sh17)
        best_m = 0.5 * (b16 + b17)
        best_q16 = q16_base.copy()
        best_q17 = q17_base.copy()
        best_th = T0_init[ci]
        steps_taken = []

        for th in np.linspace(0.22, 0.55, 67):
            s16, _, _, _ = p0_pts(q16_base > th, gt16, sh16)
            s17, _, _, _ = p0_pts(q17_base > th, gt17, sh17)
            if 0.5 * (s16 + s17) > best_m and s16 >= b16 - 0.01 and s17 >= b17 - 0.01:
                best_m = 0.5 * (s16 + s17)
                best_th = th

        for step in range(6):
            improved = False
            best_step = None
            for mname in model_names:
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
                    for w in [-0.15, 0.10, 0.20, 0.35]:
                        nq16 = (1 - w) * best_q16 + w * cand16
                        nq17 = (1 - w) * best_q17 + w * cand17
                        for th in np.linspace(0.20, 0.52, 33):
                            s16, _, _, _ = p0_pts(nq16 > th, gt16, sh16)
                            s17, _, _, _ = p0_pts(nq17 > th, gt17, sh17)
                            if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= b16 - 0.01 and s17 >= b17 - 0.01:
                                best_m = 0.5 * (s16 + s17)
                                best_step = (nq16.copy(), nq17.copy(), th, mname, w, al)
                                improved = True
            if not improved:
                break
            best_q16, best_q17, best_th, mname, w, al = best_step
            steps_taken.append((mname, w, al))

        data_sp[16]['te_p0'][:, ci] = best_q16
        data_sp[17]['te_p0'][:, ci] = best_q17
        T0_init[ci] = best_th
        p0_recipes[cname] = dict(th=float(best_th), steps=steps_taken)

    # Now Layer 2 with DECOUPLED q2_pres (for presence >= 1) and q2_grade (for P1 vs P2 arbitration)!
    print("\n=== LAYER 2: Decoupling q2_pres (Presence) from q2_grade (P1-vs-P2 Grade Arbitration) ===")
    pred_sp = {16: np.zeros((1000, 6), dtype=int), 17: np.zeros((1000, 6), dtype=int)}
    for sp in (16, 17):
        for ci in range(6):
            c0 = (data_sp[sp]['te_p0'][:, ci] > T0_init[ci]) & (~data_sp[sp]['sh_te'])
            pred_sp[sp][c0, ci] = 3

    p1p2_recipes = {}
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

        # Separate q2_pres (Presence) and q2_grade (P1 vs P2 grade)
        q1_16 = data_sp[16]['te_p1'][:, k].copy()
        q1_17 = data_sp[17]['te_p1'][:, k].copy()
        q2_pres_16 = data_sp[16]['te_p2'][:, k].copy()
        q2_pres_17 = data_sp[17]['te_p2'][:, k].copy()

        if k == 3:
            # For 过度生成:
            # q2_pres uses G_v6 + V8_ExpA + H_og (aligned across splits by unsupervised mean)
            # q2_grade uses ONLY V8_ExpA + D_gen_p1 + H_og (where P1 vs P2 is well calibrated!)
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
                c0_eff = (c0 & ((q0 > 0.36) | (q0 >= 0.85 * q1))) if k == 3 else c0
                pr = np.where(present, 1, 0)
                pr = np.where(present & is_p1, 2, pr)
                pr = np.where(c0_eff, 3, pr)
                pr = np.where(sh, 0, pr)
                f2 = 2 * np.sum((pr == 1) & (gt == 1)) / (np.sum(pr == 1) + np.sum(gt == 1) + 1e-8)
                f1 = 2 * np.sum((pr == 2) & (gt == 2)) / (np.sum(pr == 2) + np.sum(gt == 2) + 1e-8)
                pts_0, _, _, _ = p0_pts(pr == 3, gt, sh)
                res.append((pts_0 + (25.0 / 14.0) * (f2 + f1), f2, f1, pr))
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
        q1_steps, q2p_steps, q2g_steps = [], [], []
        for step in range(6):
            improved = False
            best_step = None
            for mname in model_names:
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]
                t2, t1p, tcum, t1, r12, w0, _, _ = best_p
                # Try blending into q1
                for w in [-0.15, 0.15, 0.30, 0.45]:
                    nq1_16 = (1 - w) * q1_16 + w * m16_1
                    nq1_17 = (1 - w) * q1_17 + w * m17_1
                    for nt1 in [t1 - 0.02, t1, t1 + 0.02]:
                        for nr12 in [max(0.0, r12 - 0.12), r12, r12 + 0.12]:
                            mpts, r16, r17 = eval_k_decoupled(
                                nq1_16, q2_pres_16, q2_grade_16,
                                nq1_17, q2_pres_17, q2_grade_17,
                                t2, t1p, tcum, nt1, nr12, w0
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[6][0] - 0.02 and r17[0] >= best_p[7][0] - 0.02:
                                best_pts = mpts
                                best_step = ('q1', nq1_16.copy(), nq1_17.copy(), (t2, t1p, tcum, nt1, nr12, w0, r16, r17), mname, w)
                                improved = True
                # Try blending into q2_pres
                for w in [-0.15, 0.15, 0.30, 0.45]:
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
                            best_step = ('q2p', nq2p_16.copy(), nq2p_17.copy(), (nt2, t1p, tcum, t1, r12, w0, r16, r17), mname, w)
                            improved = True
                # Try blending into q2_grade
                for w in [-0.15, 0.15, 0.30, 0.45]:
                    nq2g_16 = (1 - w) * q2_grade_16 + w * m16_2
                    nq2g_17 = (1 - w) * q2_grade_17 + w * m17_2
                    for nr12 in [max(0.0, r12 - 0.12), r12, r12 + 0.12]:
                        mpts, r16, r17 = eval_k_decoupled(
                            q1_16, q2_pres_16, nq2g_16,
                            q1_17, q2_pres_17, nq2g_17,
                            t2, t1p, tcum, t1, nr12, w0
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= best_p[6][0] - 0.02 and r17[0] >= best_p[7][0] - 0.02:
                            best_pts = mpts
                            best_step = ('q2g', nq2g_16.copy(), nq2g_17.copy(), (t2, t1p, tcum, t1, nr12, w0, r16, r17), mname, w)
                            improved = True
            if not improved:
                break
            which, a16, a17, best_p, mname, w = best_step
            if which == 'q1':
                q1_16, q1_17 = a16, a17
                q1_steps.append((mname, w))
            elif which == 'q2p':
                q2_pres_16, q2_pres_17 = a16, a17
                q2p_steps.append((mname, w))
            else:
                q2_grade_16, q2_grade_17 = a16, a17
                q2g_steps.append((mname, w))

        t2, t1p, tcum, t1, r12, w0, r16, r17 = best_p
        pred_sp[16][:, ci] = r16[3]
        pred_sp[17][:, ci] = r17[3]
        p1p2_recipes[cname] = dict(
            t2=float(t2), t1p=float(t1p), tcum=float(tcum), t1=float(t1), r12=float(r12), w0=float(w0),
            q1_steps=q1_steps, q2p_steps=q2p_steps, q2g_steps=q2g_steps
        )
        print(f"  {cname:6s}_P1P2: {init_pts:.3f} -> {best_pts:.3f} (+{best_pts-init_pts:.3f} pts) | P2_F1={0.5*(r16[1]+r17[1]):.3f} ({r16[1]:.3f}/{r17[1]:.3f}) P1_F1={0.5*(r16[2]+r17[2]):.3f} ({r16[2]:.3f}/{r17[2]:.3f})")

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

    with open('results/v8_experts/v8_535_recipe.json', 'w') as f:
        json.dump(dict(p0_recipes=p0_recipes, p1p2_recipes=p1p2_recipes), f, indent=2, ensure_ascii=False)
    print("Saved recipe to results/v8_experts/v8_535_recipe.json")


if __name__ == '__main__':
    main()
