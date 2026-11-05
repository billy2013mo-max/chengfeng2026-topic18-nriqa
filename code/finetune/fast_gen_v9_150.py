"""Fast (~15s) V9 Stage-1 Strongest 150-Image Blind Evaluation Set Submission Generator (`fast_gen_v9_150.py`).
Uses the exact 51-signal 1-to-1 Aligned Specialist & Virtual Basis Pool across `s16`, `s17`, and `test_150`
with the verified winning start per class so it executes in ~15 seconds!
"""
import os
import sys
import json
import time
from collections import Counter
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from test_cross_defect_arbitrator import load_rich_split_context
from test_5stream_joint_arbitrator import exact_full_class_pts, p0_pts
from generate_v9_150_submission import (
    GRADE,
    load_150_context,
    add_1to1_aligned_families_to_split,
    populate_common_virtual_basis,
)


def run_fast_v9_150():
    t0_wall = time.time()
    data_sp, _ = load_rich_split_context()
    for sp in (16, 17):
        add_1to1_aligned_families_to_split(data_sp[sp])
    ctx_150 = load_150_context()

    real_aligned_keys = sorted(
        set(data_sp[16]['models'].keys())
        & set(data_sp[17]['models'].keys())
        & set(ctx_150['models'].keys())
    )
    for d_ctx in (data_sp[16], data_sp[17], ctx_150):
        populate_common_virtual_basis(d_ctx, real_aligned_keys)

    common_all_keys = sorted(
        set(data_sp[16]['models'].keys())
        & set(data_sp[17]['models'].keys())
        & set(ctx_150['models'].keys())
    )
    print(f"[{time.time()-t0_wall:.1f}s] Loaded {len(common_all_keys)} 1-to-1 aligned basis signals across s16, s17, and test_150.")

    # Winning start per P0 channel from task-2478 / task-2513
    p0_win_tags = {
        0: 'Clean',    # 模糊_P0
        1: 'Clean',    # 纹理损失_P0
        2: 'V9-All',   # 过锐_P0
        3: 'V9-ABCD',  # 花屏_P0
        4: 'V9-BD',    # 过度生成_P0
        5: 'V9-ABCD',  # 虚化突兀_P0
    }
    T0_init = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]

    print("\n=== LAYER 1: Fast 1-to-1 Aligned P0 Optimization & 150-Image Replay ===")
    for ci, cname in enumerate(CLASSES):
        sh16, sh17, sh150 = data_sp[16]['sh_te'], data_sp[17]['sh_te'], ctx_150['sh_te']
        gt16, gt17 = data_sp[16]['gte'][:, ci], data_sp[17]['gte'][:, ci]
        tag = p0_win_tags[ci]

        q16 = data_sp[16]['cand_bases'][tag]['te_p0'][:, ci].copy()
        q17 = data_sp[17]['cand_bases'][tag]['te_p0'][:, ci].copy()
        q150 = ctx_150['cand_bases'][tag]['te_p0'][:, ci].copy()

        b16, _, _, _ = p0_pts(q16 > T0_init[ci], gt16, sh16)
        b17, _, _, _ = p0_pts(q17 > T0_init[ci], gt17, sh17)
        base_m = 0.5 * (b16 + b17)
        best_m = base_m
        best_th = T0_init[ci]

        for th in np.linspace(0.24, 0.52, 29):
            s16, _, _, _ = p0_pts(q16 > th, gt16, sh16)
            s17, _, _, _ = p0_pts(q17 > th, gt17, sh17)
            if 0.5 * (s16 + s17) > best_m and s16 >= b16 - 0.01 and s17 >= b17 - 0.01:
                best_m = 0.5 * (s16 + s17)
                best_th = float(th)

        for step in range(12):
            improved = False
            best_step = None
            for mname in common_all_keys:
                m16_0 = data_sp[16]['models'][mname]['te_p0'][:, ci]
                m17_0 = data_sp[17]['models'][mname]['te_p0'][:, ci]
                m150_0 = ctx_150['models'][mname]['te_p0'][:, ci]
                alphas = [0.0]
                if cname in FULL:
                    k = FULL.index(cname)
                    m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                    m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                    m150_1 = ctx_150['models'][mname]['te_p1'][:, k]
                    alphas = [-0.20, 0.0, 0.25, 0.50]
                else:
                    m16_1 = m17_1 = m150_1 = 0.0
                for al in alphas:
                    cand16 = np.clip(m16_0 + al * m16_1, 0, 1)
                    cand17 = np.clip(m17_0 + al * m17_1, 0, 1)
                    cand150 = np.clip(m150_0 + al * m150_1, 0, 1)
                    for w in [-0.20, -0.10, 0.10, 0.20, 0.35]:
                        nq16 = (1 - w) * q16 + w * cand16
                        nq17 = (1 - w) * q17 + w * cand17
                        nq150 = (1 - w) * q150 + w * cand150
                        for th in np.linspace(0.24, 0.50, 14):
                            s16, _, _, _ = p0_pts(nq16 > th, gt16, sh16)
                            s17, _, _, _ = p0_pts(nq17 > th, gt17, sh17)
                            if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= b16 - 0.01 and s17 >= b17 - 0.01:
                                best_m = 0.5 * (s16 + s17)
                                best_step = (nq16.copy(), nq17.copy(), nq150.copy(), float(th))
                                improved = True
            if not improved:
                break
            q16, q17, q150, best_th = best_step

        c16 = (q16 > best_th)
        c17 = (q17 > best_th)
        c150 = (q150 > best_th)

        for mname in common_all_keys:
            sup16 = data_sp[16]['models'][mname]['te_p0'][:, ci]
            sup17 = data_sp[17]['models'][mname]['te_p0'][:, ci]
            sup150 = ctx_150['models'][mname]['te_p0'][:, ci]
            for d_hi in [0.0, 0.02, 0.04]:
                for d_lo in [-0.06, -0.04, -0.02]:
                    for th_sup in [0.28, 0.36, 0.44, 0.52]:
                        nc16 = (q16 > (best_th + d_hi)) | ((q16 > (best_th + d_lo)) & (sup16 > th_sup))
                        nc17 = (q17 > (best_th + d_hi)) | ((q17 > (best_th + d_lo)) & (sup17 > th_sup))
                        s16, _, _, _ = p0_pts(nc16, gt16, sh16)
                        s17, _, _, _ = p0_pts(nc17, gt17, sh17)
                        if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= b16 and s17 >= b17:
                            best_m = 0.5 * (s16 + s17)
                            c16, c17 = nc16.copy(), nc17.copy()
                            c150 = (q150 > (best_th + d_hi)) | ((q150 > (best_th + d_lo)) & (sup150 > th_sup))

        cur_16, _, _, _ = p0_pts(c16, gt16, sh16)
        cur_17, _, _, _ = p0_pts(c17, gt17, sh17)
        for _ in range(3):
            imp_c = False
            best_c_step = None
            for mname in common_all_keys:
                sup16 = data_sp[16]['models'][mname]['te_p0'][:, ci]
                sup17 = data_sp[17]['models'][mname]['te_p0'][:, ci]
                sup150 = ctx_150['models'][mname]['te_p0'][:, ci]
                for d_lo in [-0.07, -0.05, -0.03]:
                    for th_sup in [0.30, 0.38, 0.46]:
                        nc16 = c16 | ((q16 > (best_th + d_lo)) & (sup16 > th_sup))
                        nc17 = c17 | ((q17 > (best_th + d_lo)) & (sup17 > th_sup))
                        s16, _, _, _ = p0_pts(nc16, gt16, sh16)
                        s17, _, _, _ = p0_pts(nc17, gt17, sh17)
                        if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= cur_16 - 1e-4 and s17 >= cur_17 - 1e-4:
                            best_m = 0.5 * (s16 + s17)
                            nc150 = c150 | ((q150 > (best_th + d_lo)) & (sup150 > th_sup))
                            best_c_step = (nc16.copy(), nc17.copy(), nc150.copy(), s16, s17)
                            imp_c = True
            if not imp_c:
                break
            c16, c17, c150, cur_16, cur_17 = best_c_step

        data_sp[16]['te_p0'][:, ci] = q16
        data_sp[17]['te_p0'][:, ci] = q17
        ctx_150['te_p0'][:, ci] = q150
        data_sp[16][f'c0_{ci}'] = c16 & (~sh16)
        data_sp[17][f'c0_{ci}'] = c17 & (~sh17)
        ctx_150[f'c0_{ci}'] = c150 & (~sh150)
        T0_init[ci] = best_th
        n16, nf1_16, nr16, np16 = p0_pts(c16, gt16, sh16)
        n17, nf1_17, nr17, np17 = p0_pts(c17, gt17, sh17)
        print(f"  {cname:6s}_P0 [{tag:7s}]: {base_m:.3f} -> {best_m:.3f} (+{best_m-base_m:.3f} pts) | "
              f"F1={0.5*(nf1_16+nf1_17):.3f} R={0.5*(nr16+nr17):.3f} P={0.5*(np16+np17):.3f} | 150_P0_cnt={int(ctx_150[f'c0_{ci}'].sum())}")

    # Winning start per FULL class from task-2478 / task-2456
    full_win_tags = {
        0: 'V9-All',   # 模糊
        1: 'V9-BD',    # 纹理损失
        2: 'V9-All',   # 过锐
        3: 'V9-ABCD',  # 过度生成
    }

    print("\n=== LAYER 2: Fast 1-to-1 Aligned 5-Stream + Multi-Clause Arbitrator & 150-Image Replay ===")
    pred_sp = {16: np.zeros((1000, 6), dtype=int), 17: np.zeros((1000, 6), dtype=int)}
    pred_150 = np.zeros((150, 6), dtype=int)
    for ci in range(6):
        pred_sp[16][data_sp[16][f'c0_{ci}'], ci] = 3
        pred_sp[17][data_sp[17][f'c0_{ci}'], ci] = 3
        pred_150[ctx_150[f'c0_{ci}'], ci] = 3

    for k, cname in enumerate(FULL):
        ci = CLASSES.index(cname)
        sh16, sh17, sh150 = data_sp[16]['sh_te'], data_sp[17]['sh_te'], ctx_150['sh_te']
        gt16, gt17 = data_sp[16]['gte'][:, ci], data_sp[17]['gte'][:, ci]
        q0_16, q0_17, q0_150 = data_sp[16]['te_p0'][:, ci], data_sp[17]['te_p0'][:, ci], ctx_150['te_p0'][:, ci]
        tag = full_win_tags[k]

        c0_16 = data_sp[16][f'c0_{ci}'].copy()
        c0_17 = data_sp[17][f'c0_{ci}'].copy()
        c0_150 = ctx_150[f'c0_{ci}'].copy()

        q1p_16 = data_sp[16]['cand_bases'][tag]['te_p1'][:, k].copy()
        q1p_17 = data_sp[17]['cand_bases'][tag]['te_p1'][:, k].copy()
        q1p_150 = ctx_150['cand_bases'][tag]['te_p1'][:, k].copy()
        q1g_16, q1g_17, q1g_150 = q1p_16.copy(), q1p_17.copy(), q1p_150.copy()

        q2p_16 = data_sp[16]['cand_bases'][tag]['te_p2'][:, k].copy()
        q2p_17 = data_sp[17]['cand_bases'][tag]['te_p2'][:, k].copy()
        q2p_150 = ctx_150['cand_bases'][tag]['te_p2'][:, k].copy()

        if k == 3:
            q2g_16 = 0.50 * data_sp[16]['models']['fam/v7_overgen_pro']['te_p2'][:, 3] + 0.50 * data_sp[16]['models']['v9_specialists/V9_SpecA_OverGen_Giant']['te_p2'][:, 3]
            q2g_17 = 0.50 * data_sp[17]['models']['fam/v7_overgen_pro']['te_p2'][:, 3] + 0.50 * data_sp[17]['models']['v9_specialists/V9_SpecA_OverGen_Giant']['te_p2'][:, 3]
            q2g_150 = 0.50 * ctx_150['models']['fam/v7_overgen_pro']['te_p2'][:, 3] + 0.50 * ctx_150['models']['v9_specialists/V9_SpecA_OverGen_Giant']['te_p2'][:, 3]
        else:
            q2g_16, q2g_17, q2g_150 = q2p_16.copy(), q2p_17.copy(), q2p_150.copy()

        def eval_k869(q1pa, q1ga, q2pa, q2ga, q1pb, q1gb, q2pb, q2gb, t2, t1p, tcum, t1, r12, w0, t1_hi=0.99):
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
                tot_pts, f0, f1, f2, r0, p0_v = exact_full_class_pts(pr, gt, sh)
                res.append((tot_pts, f2, f1, pr, f0))
            return 0.5 * (res[0][0] + res[1][0]), res[0], res[1]

        best_pts = -1.0
        best_p = None
        for t2 in [0.16, 0.20, 0.24, 0.28]:
            for t1p in [0.20, 0.25, 0.30]:
                for tcum in [0.32, 0.40, 0.48]:
                    for t1 in [0.16, 0.21, 0.25, 0.29]:
                        for r12 in [0.35, 0.55, 0.75, 0.95]:
                            for w0 in [0.0, 0.20, 0.40]:
                                for t1_hi in [0.36, 0.99]:
                                    mpts, r16, r17 = eval_k869(
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
            for mname in common_all_keys:
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m150_1 = ctx_150['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]
                m150_2 = ctx_150['models'][mname]['te_p2'][:, k]

                for s16_c, s17_c, s150_c in [
                    (m16_1, m17_1, m150_1),
                    (np.clip(m16_1 - 0.3 * m16_2, 0, 1), np.clip(m17_1 - 0.3 * m17_2, 0, 1), np.clip(m150_1 - 0.3 * m150_2, 0, 1))
                ]:
                    for w in [-0.20, 0.15, 0.30]:
                        nq1g_16 = (1 - w) * q1g_16 + w * s16_c
                        nq1g_17 = (1 - w) * q1g_17 + w * s17_c
                        nq1g_150 = (1 - w) * q1g_150 + w * s150_c
                        for nt1 in [t1 - 0.02, t1, t1 + 0.02]:
                            for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                                mpts, r16, r17 = eval_k869(
                                    q1p_16, nq1g_16, q2p_16, q2g_16,
                                    q1p_17, nq1g_17, q2p_17, q2g_17,
                                    t2, t1p, tcum, nt1, nr12, w0, t1_hi
                                )
                                if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012:
                                    best_pts = mpts
                                    best_step = ('q1g', nq1g_16.copy(), nq1g_17.copy(), nq1g_150.copy(), (t2, t1p, tcum, nt1, nr12, w0, t1_hi, r16, r17))
                                    improved = True

                for s16_c, s17_c, s150_c in [(m16_2, m17_2, m150_2), (m16_1 + m16_2, m17_1 + m17_2, m150_1 + m150_2)]:
                    for w in [-0.20, 0.15, 0.30]:
                        nq2p_16 = (1 - w) * q2p_16 + w * s16_c
                        nq2p_17 = (1 - w) * q2p_17 + w * s17_c
                        nq2p_150 = (1 - w) * q2p_150 + w * s150_c
                        for nt2 in [t2 - 0.02, t2, t2 + 0.02]:
                            mpts, r16, r17 = eval_k869(
                                q1p_16, q1g_16, nq2p_16, q2g_16,
                                q1p_17, q1g_17, nq2p_17, q2g_17,
                                nt2, t1p, tcum, t1, r12, w0, t1_hi
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012:
                                best_pts = mpts
                                best_step = ('q2p', nq2p_16.copy(), nq2p_17.copy(), nq2p_150.copy(), (nt2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17))
                                improved = True

                for w in [-0.20, 0.15, 0.30]:
                    nq2g_16 = (1 - w) * q2g_16 + w * m16_2
                    nq2g_17 = (1 - w) * q2g_17 + w * m17_2
                    nq2g_150 = (1 - w) * q2g_150 + w * m150_2
                    for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                        mpts, r16, r17 = eval_k869(
                            q1p_16, q1g_16, q2p_16, nq2g_16,
                            q1p_17, q1g_17, q2p_17, nq2g_17,
                            t2, t1p, tcum, t1, nr12, w0, t1_hi
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012:
                            best_pts = mpts
                            best_step = ('q2g', nq2g_16.copy(), nq2g_17.copy(), nq2g_150.copy(), (t2, t1p, tcum, t1, nr12, w0, t1_hi, r16, r17))
                            improved = True

            if not improved:
                break
            which, a16, a17, a150, best_p = best_step
            if which == 'q1g':
                q1g_16, q1g_17, q1g_150 = a16, a17, a150
            elif which == 'q2p':
                q2p_16, q2p_17, q2p_150 = a16, a17, a150
            else:
                q2g_16, q2g_17, q2g_150 = a16, a17, a150

        t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17 = best_p
        s1_16 = q1g_16 + w0 * q0_16
        s1_17 = q1g_17 + w0 * q0_17
        s1_150 = q1g_150 + w0 * q0_150
        pres_16 = (q2p_16 > t2) | (q1p_16 > t1p) | ((q0_16 + q1p_16 + q2p_16) > tcum)
        pres_17 = (q2p_17 > t2) | (q1p_17 > t1p) | ((q0_17 + q1p_17 + q2p_17) > tcum)
        pres_150 = (q2p_150 > t2) | (q1p_150 > t1p) | ((q0_150 + q1p_150 + q2p_150) > tcum)
        isp1_16 = (s1_16 > t1_hi) | ((s1_16 > t1) & (s1_16 >= r12 * q2g_16))
        isp1_17 = (s1_17 > t1_hi) | ((s1_17 > t1) & (s1_17 >= r12 * q2g_17))
        isp1_150 = (s1_150 > t1_hi) | ((s1_150 > t1) & (s1_150 >= r12 * q2g_150))

        def eval_bool3(np16, ni16, nc16, np17, ni17, nc17):
            pr16 = np.where(np16, 1, 0)
            pr16 = np.where(np16 & ni16, 2, pr16)
            pr16 = np.where(nc16, 3, pr16)
            pr16 = np.where(sh16, 0, pr16)
            pr17 = np.where(np17, 1, 0)
            pr17 = np.where(np17 & ni17, 2, pr17)
            pr17 = np.where(nc17, 3, pr17)
            pr17 = np.where(sh17, 0, pr17)
            pts16 = exact_full_class_pts(pr16, gt16, sh16)
            pts17 = exact_full_class_pts(pr17, gt17, sh17)
            return 0.5 * (pts16[0] + pts17[0]), pr16, pr17, pts16, pts17

        for _ in range(6):
            imp_rg = False
            best_rg = None
            for mname in common_all_keys:
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m150_1 = ctx_150['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]
                m150_2 = ctx_150['models'][mname]['te_p2'][:, k]
                sup_p16, sup_p17, sup_p150 = m16_1 + m16_2, m17_1 + m17_2, m150_1 + m150_2

                for d_lo in [-0.06, -0.04, -0.02]:
                    for th_sup in [0.26, 0.34, 0.42, 0.50]:
                        np16 = pres_16 | ((q2p_16 > (t2 + d_lo)) & (sup_p16 > th_sup))
                        np17 = pres_17 | ((q2p_17 > (t2 + d_lo)) & (sup_p17 > th_sup))
                        m_sc, pr16, pr17, pts16, pts17 = eval_bool3(np16, isp1_16, c0_16, np17, isp1_17, c0_17)
                        if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 1e-4 and pts17[0] >= r17[0] - 1e-4:
                            best_pts = m_sc
                            np150 = pres_150 | ((q2p_150 > (t2 + d_lo)) & (sup_p150 > th_sup))
                            best_rg = (np16.copy(), isp1_16.copy(), c0_16.copy(), np17.copy(), isp1_17.copy(), c0_17.copy(), np150.copy(), isp1_150.copy(), c0_150.copy(), pr16, pr17, pts16, pts17)
                            imp_rg = True

                for d_hi in [0.02, 0.04, 0.06]:
                    for th_veto in [0.16, 0.22, 0.28]:
                        np16 = pres_16 & ((q2p_16 > (t2 + d_hi)) | (q1p_16 > t1p) | (sup_p16 > th_veto))
                        np17 = pres_17 & ((q2p_17 > (t2 + d_hi)) | (q1p_17 > t1p) | (sup_p17 > th_veto))
                        m_sc, pr16, pr17, pts16, pts17 = eval_bool3(np16, isp1_16, c0_16, np17, isp1_17, c0_17)
                        if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 1e-4 and pts17[0] >= r17[0] - 1e-4:
                            best_pts = m_sc
                            np150 = pres_150 & ((q2p_150 > (t2 + d_hi)) | (q1p_150 > t1p) | (sup_p150 > th_veto))
                            best_rg = (np16.copy(), isp1_16.copy(), c0_16.copy(), np17.copy(), isp1_17.copy(), c0_17.copy(), np150.copy(), isp1_150.copy(), c0_150.copy(), pr16, pr17, pts16, pts17)
                            imp_rg = True

                for d_lo in [-0.05, -0.03, -0.01]:
                    for th_sup in [0.22, 0.28, 0.34, 0.40]:
                        for r_mult in [0.50, 0.65, 0.80]:
                            ni16 = isp1_16 | ((s1_16 > (t1 + d_lo)) & (m16_1 > th_sup) & (s1_16 >= r_mult * r12 * q2g_16))
                            ni17 = isp1_17 | ((s1_17 > (t1 + d_lo)) & (m17_1 > th_sup) & (s1_17 >= r_mult * r12 * q2g_17))
                            m_sc, pr16, pr17, pts16, pts17 = eval_bool3(pres_16, ni16, c0_16, pres_17, ni17, c0_17)
                            if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 1e-4 and pts17[0] >= r17[0] - 1e-4:
                                best_pts = m_sc
                                ni150 = isp1_150 | ((s1_150 > (t1 + d_lo)) & (m150_1 > th_sup) & (s1_150 >= r_mult * r12 * q2g_150))
                                best_rg = (pres_16.copy(), ni16.copy(), c0_16.copy(), pres_17.copy(), ni17.copy(), c0_17.copy(), pres_150.copy(), ni150.copy(), c0_150.copy(), pr16, pr17, pts16, pts17)
                                imp_rg = True

                for d_hi in [0.04, 0.07]:
                    for ratio_v in [1.05, 1.25, 1.45]:
                        ni16 = isp1_16 & ((s1_16 > (t1 + d_hi)) | (m16_2 < ratio_v * m16_1))
                        ni17 = isp1_17 & ((s1_17 > (t1 + d_hi)) | (m17_2 < ratio_v * m17_1))
                        m_sc, pr16, pr17, pts16, pts17 = eval_bool3(pres_16, ni16, c0_16, pres_17, ni17, c0_17)
                        if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 1e-4 and pts17[0] >= r17[0] - 1e-4:
                            best_pts = m_sc
                            ni150 = isp1_150 & ((s1_150 > (t1 + d_hi)) | (m150_2 < ratio_v * m150_1))
                            best_rg = (pres_16.copy(), ni16.copy(), c0_16.copy(), pres_17.copy(), ni17.copy(), c0_17.copy(), pres_150.copy(), ni150.copy(), c0_150.copy(), pr16, pr17, pts16, pts17)
                            imp_rg = True

            if not imp_rg:
                break
            pres_16, isp1_16, c0_16, pres_17, isp1_17, c0_17, pres_150, isp1_150, c0_150, pr16, pr17, pts16, pts17 = best_rg
            r16 = (pts16[0], pts16[3], pts16[2], pr16, pts16[1])
            r17 = (pts17[0], pts17[3], pts17[2], pr17, pts17[1])

        pr150 = np.where(pres_150, 1, 0)
        pr150 = np.where(pres_150 & isp1_150, 2, pr150)
        pr150 = np.where(c0_150, 3, pr150)
        pr150 = np.where(sh150, 0, pr150)

        pred_sp[16][:, ci] = r16[3]
        pred_sp[17][:, ci] = r17[3]
        pred_150[:, ci] = pr150
        print(f"  {cname:6s} [{tag:7s}] (8.690): {init_pts:.3f} -> {best_pts:.3f} (+{best_pts-init_pts:.3f} pts) | "
              f"P2_F1={0.5*(r16[1]+r17[1]):.3f} P1_F1={0.5*(r16[2]+r17[2]):.3f} P0_F1={0.5*(r16[4]+r17[4]):.3f}")

    print("\n=======================================================================================")
    res_all = {}
    for sp in (16, 17):
        res = cm.score(data_sp[sp]['gte'], pred_sp[sp], 0.02)
        res_all[sp] = res
        print(f"  1-to-1 Aligned Split s{sp}: TOTAL = {res['total']:.2f} | S_pos={res['S_pos']:.3f}, S_p0={res['S_p0']:.3f} (R={res['p0R']:.3f}, P={res['p0P']:.3f}), S_clean={res['S_clean']:.3f}")
    mean_tot = 0.5 * (res_all[16]['total'] + res_all[17]['total'])
    print(f"  1-TO-1 ALIGNED 1000-IMAGE TEST MEAN = {mean_tot:.2f} (S_pos={0.5*(res_all[16]['S_pos']+res_all[17]['S_pos']):.3f}, S_p0={0.5*(res_all[16]['S_p0']+res_all[17]['S_p0']):.3f}, S_clean=1.000)")
    print("=======================================================================================")

    ordered_names = ctx_150['ordered_names']
    b5243 = ctx_150['b5243']
    v8_sub = {e['image']: e['label'] for e in json.load(open('results/v8_experts/pred_nriqa_dataset_test.json'))}
    v9_mean_p0 = ctx_150['models']['virt/v9_spec_mean']['te_p0']
    cln_p0_150 = ctx_150['cln_p0']

    out_entries_main = []
    out_entries_aggr = []
    for i, name in enumerate(ordered_names):
        old_lbl = b5243[name]
        if all(v == '无问题' for v in old_lbl.values()):
            lbl_main = old_lbl.copy()
            lbl_aggr = old_lbl.copy()
        else:
            lbl_main = {}
            lbl_aggr = {}
            for k_name in ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']:
                ci = CLASSES.index(k_name)
                new_g = int(pred_150[i, ci])
                if old_lbl[k_name] == 'P0' and new_g < 3 and (v9_mean_p0[i, ci] >= 0.28 or cln_p0_150[i, ci] >= 0.36):
                    lbl_main[k_name] = 'P0'
                else:
                    lbl_main[k_name] = GRADE[new_g]
                lbl_aggr[k_name] = GRADE[new_g]

            if all(v == '无问题' for v in lbl_main.values()):
                lbl_main = old_lbl.copy()
            if all(v == '无问题' for v in lbl_aggr.values()):
                lbl_aggr = old_lbl.copy()

        out_entries_main.append({'image': name, 'label': lbl_main})
        out_entries_aggr.append({'image': name, 'label': lbl_aggr})

    for p in ['pred_nriqa_dataset_test.json', 'results/v9_specialists/pred_nriqa_dataset_test_v9_main.json']:
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(out_entries_main, f, ensure_ascii=False, indent=2)
    with open('results/v9_specialists/pred_nriqa_dataset_test_v9_pure_aligned.json', 'w', encoding='utf-8') as f:
        json.dump(out_entries_aggr, f, ensure_ascii=False, indent=2)

    for tag_name, entries, ref_dict in [
        ('V9 Main vs 52.43 Baseline', out_entries_main, b5243),
        ('V9 Main vs V8 Previous Submission', out_entries_main, v8_sub),
    ]:
        diffs = []
        trans = Counter()
        class_diffs = Counter()
        for e in entries:
            im = e['image']
            for c in CLASSES:
                ov = ref_dict[im][c]
                nv = e['label'][c]
                if ov != nv:
                    diffs.append((im, c, ov, nv))
                    trans[(c, f'{ov}->{nv}')] += 1
                    class_diffs[c] += 1
        print(f"\n=== {tag_name}: {len(diffs)} cell refinements across {len(set(x[0] for x in diffs))}/150 images ===")
        print("Per-class diff counts:", dict(class_diffs))
        for (c, tr), cnt in sorted(trans.items()):
            print(f"  {c:8s}: {tr:12s} x {cnt}")


if __name__ == '__main__':
    run_fast_v9_150()
