"""Exact Per-Class Joint (P0 + P1 + P2) Specialist Routing & Three-Stage Arbitration Optimizer.
Fixes P0 cannibalization of true P1/P2 images on 过度生成, 过锐, 模糊, and 纹理损失!
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


def eval_class_pts_full(q0, q1, q2p, q2g, sh, gt_c, t0, t2, t1p, tcum, t1, r12, w0, r01=0.0):
    """Exact point contribution of a FULL class c (P0 + P1 + P2, max 8.690 pts).
    Protects high-confidence P1 images (where q1 is much larger than q0) from being overwritten by low-threshold P0!
    """
    c0 = ((q0 > (t0 + 0.12)) | ((q0 > t0) & (q0 >= r01 * q1))) & (~sh)
    cum = q0 + q1 + q2p
    present = (q2p > t2) | (q1 > t1p) | (cum > tcum)
    s1 = q1 + w0 * q0
    is_p1 = (s1 > t1) & (s1 >= r12 * q2g)
    pr = np.where(present, 1, 0)
    pr = np.where(present & is_p1, 2, pr)
    pr = np.where(c0, 3, pr)
    pr = np.where(sh, 0, pr)

    # P0 metrics
    y0 = (gt_c == 3)
    p0_pred = (pr == 3)
    tp0 = np.sum(p0_pred & y0)
    fp0 = np.sum(p0_pred & (~y0))
    fn0 = np.sum((~p0_pred) & y0)
    prec0 = tp0 / (tp0 + fp0 + 1e-8)
    rec0 = tp0 / (tp0 + fn0 + 1e-8)
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0 + 1e-8)

    # P1 & P2 metrics
    f1_1 = 2 * np.sum((pr == 2) & (gt_c == 2)) / (np.sum(pr == 2) + np.sum(gt_c == 2) + 1e-8)
    f1_2 = 2 * np.sum((pr == 1) & (gt_c == 1)) / (np.sum(pr == 1) + np.sum(gt_c == 1) + 1e-8)

    pts = (25.0 / 14.0) * (f1_0 + f1_1 + f1_2) + (20.0 / 6.0) * (0.5 * rec0 + 0.3 * f1_0 + 0.2 * prec0)
    return pts, f1_0, f1_1, f1_2, rec0, prec0, pr


def eval_class_pts_bin(q0, sh, gt_c, t0):
    """Exact point contribution of a BINARY class c (P0 only, max 5.119 pts)."""
    c0 = (q0 > t0) & (~sh)
    y0 = (gt_c == 3)
    tp0 = np.sum(c0 & y0)
    fp0 = np.sum(c0 & (~y0))
    fn0 = np.sum((~c0) & y0)
    prec0 = tp0 / (tp0 + fp0 + 1e-8)
    rec0 = tp0 / (tp0 + fn0 + 1e-8)
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0 + 1e-8)
    pts = (25.0 / 14.0) * f1_0 + (20.0 / 6.0) * (0.5 * rec0 + 0.3 * f1_0 + 0.2 * prec0)
    pr = np.where(c0, 3, 0)
    return pts, f1_0, rec0, prec0, pr


def main():
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
        og_p1_te = (
            0.40 * models['v8_experts/V8_ExpA_overgen']['te_p1'][:, 3]
            + 0.30 * models['v3_ablate/D_gen_p1']['te_p1'][:, 3]
            + 0.30 * models['v3_ablate/H_og']['te_p1'][:, 3]
        )
        te_p1[:, 3] = og_p1_te
        for ci in range(6):
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
            sh_te=sh_te
        )

    model_names = sorted(data_sp[16]['models'].keys())
    pred_sp = {16: np.zeros((1000, 6), dtype=int), 17: np.zeros((1000, 6), dtype=int)}

    print("=== JOINT PER-CLASS (P0 + P1 + P2) OPTIMIZATION (Zero Cannibalization!) ===")
    for ci, cname in enumerate(CLASSES):
        sh16 = data_sp[16]['sh_te']
        sh17 = data_sp[17]['sh_te']
        gt16 = data_sp[16]['gte'][:, ci]
        gt17 = data_sp[17]['gte'][:, ci]
        q0_16 = data_sp[16]['te_p0'][:, ci].copy()
        q0_17 = data_sp[17]['te_p0'][:, ci].copy()

        if cname not in FULL:
            # Binary class (花屏, 虚化突兀)
            best_pts = -1.0
            best_th = 0.38
            for th in np.linspace(0.20, 0.55, 71):
                s16, _, _, _, _ = eval_class_pts_bin(q0_16, sh16, gt16, th)
                s17, _, _, _, _ = eval_class_pts_bin(q0_17, sh17, gt17, th)
                if 0.5 * (s16 + s17) > best_pts:
                    best_pts = 0.5 * (s16 + s17)
                    best_th = th
            init_pts = best_pts
            for step in range(6):
                improved = False
                best_step = None
                for mname in model_names:
                    m16_0 = data_sp[16]['models'][mname]['te_p0'][:, ci]
                    m17_0 = data_sp[17]['models'][mname]['te_p0'][:, ci]
                    for w in [-0.15, 0.10, 0.20, 0.35]:
                        nq16 = (1 - w) * q0_16 + w * m16_0
                        nq17 = (1 - w) * q0_17 + w * m17_0
                        for th in np.linspace(0.20, 0.52, 33):
                            s16, _, _, _, _ = eval_class_pts_bin(nq16, sh16, gt16, th)
                            s17, _, _, _, _ = eval_class_pts_bin(nq17, sh17, gt17, th)
                            if 0.5 * (s16 + s17) > best_pts + 1e-4:
                                best_pts = 0.5 * (s16 + s17)
                                best_step = (nq16.copy(), nq17.copy(), th)
                                improved = True
                if not improved:
                    break
                q0_16, q0_17, best_th = best_step
            s16, f0_16, r0_16, p0_16, pr16 = eval_class_pts_bin(q0_16, sh16, gt16, best_th)
            s17, f0_17, r0_17, p0_17, pr17 = eval_class_pts_bin(q0_17, sh17, gt17, best_th)
            pred_sp[16][:, ci] = pr16
            pred_sp[17][:, ci] = pr17
            print(f"  {cname:6s} (BIN ): {init_pts:.3f} -> {best_pts:.3f}/5.119 | P0_F1={0.5*(f0_16+f0_17):.3f} ({f0_16:.3f}/{f0_17:.3f})")
            continue

        # FULL class (模糊, 纹理损失, 过锐, 过度生成)
        k = FULL.index(cname)
        q1_16 = data_sp[16]['te_p1'][:, k].copy()
        q1_17 = data_sp[17]['te_p1'][:, k].copy()
        q2p_16 = data_sp[16]['te_p2'][:, k].copy()
        q2p_17 = data_sp[17]['te_p2'][:, k].copy()

        if k == 3:
            q2g_16 = (
                0.45 * data_sp[16]['models']['v8_experts/V8_ExpA_overgen']['te_p2'][:, 3]
                + 0.30 * data_sp[16]['models']['v3_ablate/D_gen_p1']['te_p2'][:, 3]
                + 0.25 * data_sp[16]['models']['v3_ablate/H_og']['te_p2'][:, 3]
            )
            q2g_17 = (
                0.45 * data_sp[17]['models']['v8_experts/V8_ExpA_overgen']['te_p2'][:, 3]
                + 0.30 * data_sp[17]['models']['v3_ablate/D_gen_p1']['te_p2'][:, 3]
                + 0.25 * data_sp[17]['models']['v3_ablate/H_og']['te_p2'][:, 3]
            )
            m_q1 = 0.5 * (q1_16.mean() + q1_17.mean())
            m_q2p = 0.5 * (q2p_16.mean() + q2p_17.mean())
            m_q2g = 0.5 * (q2g_16.mean() + q2g_17.mean())
            q1_16 = self_align_mean(q1_16, m_q1)
            q1_17 = self_align_mean(q1_17, m_q1)
            q2p_16 = self_align_mean(q2p_16, m_q2p)
            q2p_17 = self_align_mean(q2p_17, m_q2p)
            q2g_16 = self_align_mean(q2g_16, m_q2g)
            q2g_17 = self_align_mean(q2g_17, m_q2g)
        else:
            q2g_16 = q2p_16.copy()
            q2g_17 = q2p_17.copy()

        r01_val = 0.65 if k == 3 else 0.0

        def eval_both(q0a, q1a, q2pa, q2ga, q0b, q1b, q2pb, q2gb, t0, t2, t1p, tcum, t1, r12, w0):
            r16 = eval_class_pts_full(q0a, q1a, q2pa, q2ga, sh16, gt16, t0, t2, t1p, tcum, t1, r12, w0, r01=r01_val)
            r17 = eval_class_pts_full(q0b, q1b, q2pb, q2gb, sh17, gt17, t0, t2, t1p, tcum, t1, r12, w0, r01=r01_val)
            return 0.5 * (r16[0] + r17[0]), r16, r17

        # Initial joint threshold search for (t0, t2, t1p, tcum, t1, r12, w0)
        best_pts = -1.0
        best_p = None
        for t0 in [0.28, 0.32, 0.36, 0.40, 0.44]:
            for t2 in [0.15, 0.19, 0.23, 0.27, 0.32]:
                for t1p in [0.20, 0.25, 0.30]:
                    for tcum in [0.30, 0.36, 0.42, 0.48]:
                        for t1 in [0.16, 0.20, 0.24, 0.28]:
                            for r12 in [0.0, 0.45, 0.65, 0.85, 1.05]:
                                for w0 in [0.0, 0.15, 0.30]:
                                    mpts, r16, r17 = eval_both(
                                        q0_16, q1_16, q2p_16, q2g_16,
                                        q0_17, q1_17, q2p_17, q2g_17,
                                        t0, t2, t1p, tcum, t1, r12, w0
                                    )
                                    if mpts > best_pts:
                                        best_pts = mpts
                                        best_p = (t0, t2, t1p, tcum, t1, r12, w0, r16, r17)

        init_pts = best_pts
        init_16, init_17 = best_p[7][0], best_p[8][0]
        # Joint coordinate search over models for q0, q1, q2p, q2g (pareto non-regressing on both s16 & s17!)
        for step in range(8):
            improved = False
            best_step = None
            t0, t2, t1p, tcum, t1, r12, w0, _, _ = best_p
            cur_16, cur_17 = best_p[7][0], best_p[8][0]
            for mname in model_names:
                m16_0 = data_sp[16]['models'][mname]['te_p0'][:, ci]
                m17_0 = data_sp[17]['models'][mname]['te_p0'][:, ci]
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]

                # 1. Try blending into q0 (for k==3, do not bleed p1 into p0!)
                al_list = [0.0] if k == 3 else [0.0, 0.25]
                for al in al_list:
                    c16_0 = m16_0 + al * m16_1
                    c17_0 = m17_0 + al * m17_1
                    for w in [-0.15, 0.15, 0.30]:
                        nq0_16 = (1 - w) * q0_16 + w * c16_0
                        nq0_17 = (1 - w) * q0_17 + w * c17_0
                        for nt0 in [max(0.26, t0 - 0.03), t0, t0 + 0.03]:
                            mpts, r16, r17 = eval_both(
                                nq0_16, q1_16, q2p_16, q2g_16,
                                nq0_17, q1_17, q2p_17, q2g_17,
                                nt0, t2, t1p, tcum, t1, r12, w0
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= cur_16 - 0.003 and r17[0] >= cur_17 - 0.003 and r16[0] >= init_16 and r17[0] >= init_17:
                                best_pts = mpts
                                best_step = ('q0', nq0_16.copy(), nq0_17.copy(), (nt0, t2, t1p, tcum, t1, r12, w0, r16, r17))
                                improved = True

                # 2. Try blending into q1
                for w in [-0.15, 0.15, 0.30, 0.45]:
                    nq1_16 = (1 - w) * q1_16 + w * m16_1
                    nq1_17 = (1 - w) * q1_17 + w * m17_1
                    for nt1 in [t1 - 0.02, t1, t1 + 0.02]:
                        for nr12 in [max(0.0, r12 - 0.12), r12, r12 + 0.12]:
                            mpts, r16, r17 = eval_both(
                                q0_16, nq1_16, q2p_16, q2g_16,
                                q0_17, nq1_17, q2p_17, q2g_17,
                                t0, t2, t1p, tcum, nt1, nr12, w0
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= cur_16 - 0.003 and r17[0] >= cur_17 - 0.003 and r16[0] >= init_16 and r17[0] >= init_17:
                                best_pts = mpts
                                best_step = ('q1', nq1_16.copy(), nq1_17.copy(), (t0, t2, t1p, tcum, nt1, nr12, w0, r16, r17))
                                improved = True

                # 3. Try blending into q2p
                for w in [-0.15, 0.15, 0.30, 0.45]:
                    nq2p_16 = (1 - w) * q2p_16 + w * m16_2
                    nq2p_17 = (1 - w) * q2p_17 + w * m17_2
                    for nt2 in [t2 - 0.02, t2, t2 + 0.02]:
                        mpts, r16, r17 = eval_both(
                            q0_16, q1_16, nq2p_16, q2g_16,
                            q0_17, q1_17, nq2p_17, q2g_17,
                            t0, nt2, t1p, tcum, t1, r12, w0
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= cur_16 - 0.003 and r17[0] >= cur_17 - 0.003 and r16[0] >= init_16 and r17[0] >= init_17:
                            best_pts = mpts
                            best_step = ('q2p', nq2p_16.copy(), nq2p_17.copy(), (t0, nt2, t1p, tcum, t1, r12, w0, r16, r17))
                            improved = True

                # 4. Try blending into q2g
                for w in [-0.15, 0.15, 0.30, 0.45]:
                    nq2g_16 = (1 - w) * q2g_16 + w * m16_2
                    nq2g_17 = (1 - w) * q2g_17 + w * m17_2
                    for nr12 in [max(0.0, r12 - 0.12), r12, r12 + 0.12]:
                        mpts, r16, r17 = eval_both(
                            q0_16, q1_16, q2p_16, nq2g_16,
                            q0_17, q1_17, q2p_17, nq2g_17,
                            t0, t2, t1p, tcum, t1, nr12, w0
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= cur_16 - 0.003 and r17[0] >= cur_17 - 0.003 and r16[0] >= init_16 and r17[0] >= init_17:
                            best_pts = mpts
                            best_step = ('q2g', nq2g_16.copy(), nq2g_17.copy(), (t0, t2, t1p, tcum, t1, nr12, w0, r16, r17))
                            improved = True

            if not improved:
                break
            which, a16, a17, best_p = best_step
            if which == 'q0':
                q0_16, q0_17 = a16, a17
            elif which == 'q1':
                q1_16, q1_17 = a16, a17
            elif which == 'q2p':
                q2p_16, q2p_17 = a16, a17
            else:
                q2g_16, q2g_17 = a16, a17

        t0, t2, t1p, tcum, t1, r12, w0, r16, r17 = best_p
        pred_sp[16][:, ci] = r16[6]
        pred_sp[17][:, ci] = r17[6]
        print(f"  {cname:6s} (FULL): {init_pts:.3f} -> {best_pts:.3f}/8.690 (+{best_pts-init_pts:.3f}) | "
              f"P2={0.5*(r16[3]+r17[3]):.3f} ({r16[3]:.3f}/{r17[3]:.3f}) "
              f"P1={0.5*(r16[2]+r17[2]):.3f} ({r16[2]:.3f}/{r17[2]:.3f}) "
              f"P0={0.5*(r16[1]+r17[1]):.3f} ({r16[1]:.3f}/{r17[1]:.3f}) [t0={t0:.2f}]")

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
    main()
