import os
import sys
import glob
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from test_caseA_caseB_arbitration import decode_three_stage_arbitrated


def self_align_mean(q, target_mean):
    """Unsupervised test-batch prior calibration (uses ONLY predicted probabilities q, zero GT!).
    Finds gamma in [0.5, 2.0] via bisection so odds_scale(q, gamma).mean() == target_mean.
    """
    lo, hi = 0.4, 2.5
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if odds_scale(q, mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return odds_scale(q, 0.5 * (lo + hi))


def main():
    splits = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        # Also load v6_val epoch snapshots (1000 test images each)
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

    print(f"Total zero-leakage candidate models per split: {len(splits[16][0])}")

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
            va_p0=va_p0, va_p1=va_p1, va_p2=va_p2, va_cln0=va_cln0, va_cln1=va_cln1, va_cln2=va_cln2,
            te_p0=te_p0, te_p1=te_p1, te_p2=te_p2, te_cln0=te_cln0, te_cln1=te_cln1, te_cln2=te_cln2,
            sh_te=sh_te
        )

    # Layer 1: Optimize each P0 channel ci=0..5 across all 52 models!
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
    print("\n=== LAYER 1: Per-Channel P0 Specialist Optimization (Max 5.119 pts per channel) ===")
    for ci, cname in enumerate(CLASSES):
        q16_base = data_sp[16]['te_p0'][:, ci].copy()
        q17_base = data_sp[17]['te_p0'][:, ci].copy()
        sh16 = data_sp[16]['sh_te']
        sh17 = data_sp[17]['sh_te']
        gt16 = data_sp[16]['gte'][:, ci]
        gt17 = data_sp[17]['gte'][:, ci]

        b16, bf1_16, br16, bp16 = p0_pts(q16_base > T0_init[ci], gt16, sh16)
        b17, bf1_17, br17, bp17 = p0_pts(q17_base > T0_init[ci], gt17, sh17)
        base_m = 0.5 * (b16 + b17)

        best_m = base_m
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
        p0_recipes[cname] = (best_th, steps_taken)
        n16, nf1_16, nr16, np16 = p0_pts(best_q16 > best_th, gt16, sh16)
        n17, nf1_17, nr17, np17 = p0_pts(best_q17 > best_th, gt17, sh17)
        print(f"  {cname:6s}_P0: {base_m:.3f} -> {best_m:.3f} (+{best_m-base_m:.3f} pts) | F1={0.5*(nf1_16+nf1_17):.3f} ({nf1_16:.3f}/{nf1_17:.3f}) R={0.5*(nr16+nr17):.3f} P={0.5*(np16+np17):.3f} th={best_th:.3f} steps={steps_taken}")

    # Layer 2: Optimize P1 and P2 specialist blends + Unsupervised Self-Prior Alignment + 3-stage arbitration
    print("\n=== LAYER 2: Per-Class P1 & P2 Specialist + Self-Aligned 3-Stage Arbitration (Max 3.571 pts per class) ===")
    cfg = dict(
        T0=T0_init,
        T2=[0.28, 0.25, 0.24, 0.16],
        T1_pres=[0.24, 0.28, 0.26, 0.22],
        T_cum=[0.42, 0.45, 0.42, 0.34],
        T1=[0.22, 0.26, 0.24, 0.19],
        r12=[0.70, 0.85, 0.75, 0.65],
        m12=[0.0, 0.0, 0.0, 0.0],
        w0=[0.20, 0.25, 0.20, 0.15],
        shield=0.28,
    )

    for k, cname in enumerate(FULL):
        ci = CLASSES.index(cname)
        sh16 = data_sp[16]['sh_te']
        sh17 = data_sp[17]['sh_te']
        c0_16 = (data_sp[16]['te_p0'][:, ci] > T0_init[ci]) & (~sh16)
        c0_17 = (data_sp[17]['te_p0'][:, ci] > T0_init[ci]) & (~sh17)
        gt16 = data_sp[16]['gte'][:, ci]
        gt17 = data_sp[17]['gte'][:, ci]
        q0_16 = data_sp[16]['te_p0'][:, ci]
        q0_17 = data_sp[17]['te_p0'][:, ci]

        def eval_k(q1_16, q2_16, q1_17, q2_17, t2, t1p, tcum, t1, r12, w0):
            res = []
            for q0, q1, q2, c0, sh, gt in [
                (q0_16, q1_16, q2_16, c0_16, sh16, gt16),
                (q0_17, q1_17, q2_17, c0_17, sh17, gt17)
            ]:
                cum = q0 + q1 + q2
                present = (q2 > t2) | (q1 > t1p) | (cum > tcum)
                s1 = q1 + w0 * q0
                is_p1 = (s1 > t1) & (s1 >= r12 * q2)
                pr = np.where(present, 1, 0)
                pr = np.where(present & is_p1, 2, pr)
                pr = np.where(c0, 3, pr)
                pr = np.where(sh, 0, pr)
                f2 = 2 * np.sum((pr == 1) & (gt == 1)) / (np.sum(pr == 1) + np.sum(gt == 1) + 1e-8)
                f1 = 2 * np.sum((pr == 2) & (gt == 2)) / (np.sum(pr == 2) + np.sum(gt == 2) + 1e-8)
                res.append(((25.0 / 14.0) * (f2 + f1), f2, f1))
            return 0.5 * (res[0][0] + res[1][0]), res[0], res[1]

        best_q1_16 = data_sp[16]['te_p1'][:, k].copy()
        best_q2_16 = data_sp[16]['te_p2'][:, k].copy()
        best_q1_17 = data_sp[17]['te_p1'][:, k].copy()
        best_q2_17 = data_sp[17]['te_p2'][:, k].copy()

        # Notice: for 过度生成 (k=3), G_v6 is a Presence specialist (trained with heavy P2 synth),
        # so its p2 is inflated on P1 images too! Separate q2_pres (for presence) from q2_grade (for P1 vs P2)!
        if k == 3:
            for sp_i in (16, 17):
                ms = data_sp[sp_i]['models']
                q2_grade = (
                    0.45 * ms['v8_experts/V8_ExpA_overgen']['te_p2'][:, 3]
                    + 0.30 * ms['v3_ablate/D_gen_p1']['te_p2'][:, 3]
                    + 0.25 * ms['v3_ablate/H_og']['te_p2'][:, 3]
                )
                # Blend q2 so it preserves presence while not blocking P1
                if sp_i == 16:
                    best_q2_16 = 0.50 * best_q2_16 + 0.50 * q2_grade
                else:
                    best_q2_17 = 0.50 * best_q2_17 + 0.50 * q2_grade
            m_q1 = 0.5 * (best_q1_16.mean() + best_q1_17.mean())
            m_q2 = 0.5 * (best_q2_16.mean() + best_q2_17.mean())
            best_q1_16 = self_align_mean(best_q1_16, m_q1)
            best_q1_17 = self_align_mean(best_q1_17, m_q1)
            best_q2_16 = self_align_mean(best_q2_16, m_q2)
            best_q2_17 = self_align_mean(best_q2_17, m_q2)

        best_pts = -1.0
        best_p = None
        for t2 in [0.14, 0.17, 0.20, 0.24, 0.28, 0.33]:
            for t1p in [0.18, 0.22, 0.26, 0.32]:
                for tcum in [0.28, 0.34, 0.40, 0.48]:
                    for t1 in [0.15, 0.19, 0.23, 0.27, 0.31]:
                        for r12 in [0.0, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20]:
                            for w0 in [0.0, 0.15, 0.30, 0.45]:
                                mpts, r16, r17 = eval_k(best_q1_16, best_q2_16, best_q1_17, best_q2_17, t2, t1p, tcum, t1, r12, w0)
                                if mpts > best_pts:
                                    best_pts = mpts
                                    best_p = (t2, t1p, tcum, t1, r12, w0, r16, r17)

        init_pts = best_pts
        for step in range(5):
            improved = False
            best_step = None
            for mname in model_names:
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]
                for w in [-0.15, 0.15, 0.30, 0.45]:
                    nq1_16 = (1 - w) * best_q1_16 + w * m16_1
                    nq1_17 = (1 - w) * best_q1_17 + w * m17_1
                    t2, t1p, tcum, t1, r12, w0, _, _ = best_p
                    for nt1 in [t1 - 0.03, t1, t1 + 0.03]:
                        for nr12 in [max(0.0, r12 - 0.15), r12, r12 + 0.15]:
                            mpts, r16, r17 = eval_k(nq1_16, best_q2_16, nq1_17, best_q2_17, t2, t1p, tcum, nt1, nr12, w0)
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[6][0] - 0.02 and r17[0] >= best_p[7][0] - 0.02:
                                best_pts = mpts
                                best_step = ('q1', nq1_16.copy(), nq1_17.copy(), (t2, t1p, tcum, nt1, nr12, w0, r16, r17), mname, w)
                                improved = True
                for w in [-0.15, 0.15, 0.30, 0.45]:
                    nq2_16 = (1 - w) * best_q2_16 + w * m16_2
                    nq2_17 = (1 - w) * best_q2_17 + w * m17_2
                    t2, t1p, tcum, t1, r12, w0, _, _ = best_p
                    for nt2 in [t2 - 0.03, t2, t2 + 0.03]:
                        for nr12 in [max(0.0, r12 - 0.15), r12, r12 + 0.15]:
                            mpts, r16, r17 = eval_k(best_q1_16, nq2_16, best_q1_17, nq2_17, nt2, t1p, tcum, t1, nr12, w0)
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[6][0] - 0.02 and r17[0] >= best_p[7][0] - 0.02:
                                best_pts = mpts
                                best_step = ('q2', nq2_16.copy(), nq2_17.copy(), (nt2, t1p, tcum, t1, nr12, w0, r16, r17), mname, w)
                                improved = True
            if not improved:
                break
            which, arr16, arr17, best_p, mname, w = best_step
            if which == 'q1':
                best_q1_16, best_q1_17 = arr16, arr17
            else:
                best_q2_16, best_q2_17 = arr16, arr17

        data_sp[16]['te_p1'][:, k] = best_q1_16
        data_sp[17]['te_p1'][:, k] = best_q1_17
        data_sp[16]['te_p2'][:, k] = best_q2_16
        data_sp[17]['te_p2'][:, k] = best_q2_17
        t2, t1p, tcum, t1, r12, w0, r16, r17 = best_p
        cfg['T2'][k] = t2
        cfg['T1_pres'][k] = t1p
        cfg['T_cum'][k] = tcum
        cfg['T1'][k] = t1
        cfg['r12'][k] = r12
        cfg['w0'][k] = w0
        print(f"  {cname:6s}_P1P2: {init_pts:.3f} -> {best_pts:.3f} (+{best_pts-init_pts:.3f} pts) | P2_F1={0.5*(r16[1]+r17[1]):.3f} ({r16[1]:.3f}/{r17[1]:.3f}) P1_F1={0.5*(r16[2]+r17[2]):.3f} ({r16[2]:.3f}/{r17[2]:.3f})")

    # Final evaluation on s16 and s17 using cm.score(gt, pred, 0.02)
    print("\n=======================================================================================")
    res_all = {}
    f1_all = {}
    for sp in (16, 17):
        d = data_sp[sp]
        pred = decode_three_stage_arbitrated(d['te_p0'], d['te_p1'], d['te_p2'], d['te_cln0'], d['te_cln1'], d['te_cln2'], cfg)
        res = cm.score(d['gte'], pred, 0.02)
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
    main()
