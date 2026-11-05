"""Unified V9 5-Stream + Multi-Start + Alternating 2-Pass 12-Round Multi-Clause Bidirectional Arbitrator
with Top-Real Pairwise Synergy Basis (`min/max/mean`) & Full Recipe Recording (`optimize_v9_to_55plus.py`).
"""
import os
import sys
import pickle
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from test_cross_defect_arbitrator import load_rich_split_context
from test_5stream_joint_arbitrator import (
    add_cross_channel_and_power_basis,
    exact_full_class_pts,
    self_align_mean,
    p0_pts,
)


def build_v9_candidate_bases(data_sp):
    """Create 4 candidate starting bases for each split:
    - 'Clean': original calibrated base (`te_p0, te_p1, te_p2`)
    - 'V9-BD': injected with `V9_SpecB` + `V9_SpecD` (optimal for global 模糊 & 纹理损失 P1/P2)
    - 'V9-ABCD': injected with `V9_SpecA` + `V9_SpecB` + `V9_SpecC` + `V9_SpecD`
    - 'V9-All': injected with `V9_SpecA` + `V9_SpecB` + `V9_SpecC` + `V9_SpecD` + `V9_SpecE`
    """
    cands = {'Clean': {}, 'V9-BD': {}, 'V9-ABCD': {}, 'V9-All': {}}
    for sp in (16, 17):
        models = data_sp[sp]['models']
        c_p0 = data_sp[sp]['te_p0'].copy()
        c_p1 = data_sp[sp]['te_p1'].copy()
        c_p2 = data_sp[sp]['te_p2'].copy()
        cands['Clean'][sp] = {'te_p0': c_p0.copy(), 'te_p1': c_p1.copy(), 'te_p2': c_p2.copy()}

        bd_p0, bd_p1, bd_p2 = c_p0.copy(), c_p1.copy(), c_p2.copy()
        if 'v9_specialists/V9_SpecB_SharpTex_HiRes' in models:
            mB = models['v9_specialists/V9_SpecB_SharpTex_HiRes']
            for k, ci in [(0, 0), (1, 1), (2, 2)]:
                bd_p0[:, ci] = 0.82 * bd_p0[:, ci] + 0.18 * mB['te_p0'][:, ci]
                bd_p1[:, k]  = 0.80 * bd_p1[:, k]  + 0.20 * mB['te_p1'][:, k]
                bd_p2[:, k]  = 0.80 * bd_p2[:, k]  + 0.20 * mB['te_p2'][:, k]
        if 'v9_specialists/V9_SpecD_BokehOG_HiRes' in models:
            mD = models['v9_specialists/V9_SpecD_BokehOG_HiRes']
            bd_p0[:, 5] = 0.78 * bd_p0[:, 5] + 0.22 * mD['te_p0'][:, 5]
            bd_p0[:, 4] = 0.82 * bd_p0[:, 4] + 0.18 * mD['te_p0'][:, 4]
            bd_p1[:, 3] = 0.80 * bd_p1[:, 3] + 0.20 * mD['te_p1'][:, 3]
            bd_p2[:, 3] = 0.80 * bd_p2[:, 3] + 0.20 * mD['te_p2'][:, 3]
        cands['V9-BD'][sp] = {'te_p0': bd_p0.copy(), 'te_p1': bd_p1.copy(), 'te_p2': bd_p2.copy()}

        abcd_p0, abcd_p1, abcd_p2 = bd_p0.copy(), bd_p1.copy(), bd_p2.copy()
        if 'v9_specialists/V9_SpecA_OverGen_Giant' in models:
            mA = models['v9_specialists/V9_SpecA_OverGen_Giant']
            abcd_p0[:, 4] = 0.78 * abcd_p0[:, 4] + 0.22 * mA['te_p0'][:, 4]
            abcd_p1[:, 3] = 0.76 * abcd_p1[:, 3] + 0.24 * mA['te_p1'][:, 3]
            abcd_p2[:, 3] = 0.78 * abcd_p2[:, 3] + 0.22 * mA['te_p2'][:, 3]
        if 'v9_specialists/V9_SpecC_AllP0_Tile3' in models:
            mC = models['v9_specialists/V9_SpecC_AllP0_Tile3']
            for ci in range(6):
                abcd_p0[:, ci] = 0.82 * abcd_p0[:, ci] + 0.18 * mC['te_p0'][:, ci]
            for k in range(4):
                abcd_p1[:, k] = 0.82 * abcd_p1[:, k] + 0.18 * mC['te_p1'][:, k]
                abcd_p2[:, k] = 0.82 * abcd_p2[:, k] + 0.18 * mC['te_p2'][:, k]
        cands['V9-ABCD'][sp] = {'te_p0': abcd_p0.copy(), 'te_p1': abcd_p1.copy(), 'te_p2': abcd_p2.copy()}

        all_p0, all_p1, all_p2 = abcd_p0.copy(), abcd_p1.copy(), abcd_p2.copy()
        if 'v9_specialists/V9_SpecE_RealHiRes_Giant' in models:
            mE = models['v9_specialists/V9_SpecE_RealHiRes_Giant']
            for ci in range(6):
                all_p0[:, ci] = 0.82 * all_p0[:, ci] + 0.18 * mE['te_p0'][:, ci]
            for k in range(4):
                all_p1[:, k] = 0.82 * all_p1[:, k] + 0.18 * mE['te_p1'][:, k]
                all_p2[:, k] = 0.82 * all_p2[:, k] + 0.18 * mE['te_p2'][:, k]
        cands['V9-All'][sp] = {'te_p0': all_p0.copy(), 'te_p1': all_p1.copy(), 'te_p2': all_p2.copy()}

        models['virt/clean_calibrated_base'] = cands['Clean'][sp]
        models['virt/v9_bd_injected_base']   = cands['V9-BD'][sp]
        models['virt/v9_abcd_injected_base'] = cands['V9-ABCD'][sp]
        models['virt/v9_injected_base']      = cands['V9-All'][sp]
    return cands


def add_v9_synergy_and_consensus_basis(data_sp, base_keys):
    """Add V9 high-res family statistics, geometric means, cross-grade margins,
    Top-3 channel consensus (`top3_channel_mean`, `top2_channel_min`, `top2_channel_max`),
    AND Top-5 Real-Model Pairwise Conjunction/Disjunction (`realpair_min/max/mean`).
    """
    real_cand_keys = [k for k in base_keys if not k.startswith('virt/')]
    all_keys = add_cross_channel_and_power_basis(data_sp, base_keys)
    v9_real_keys = [k for k in all_keys if k.startswith('v9_specialists/')]

    for sp in (16, 17):
        d = data_sp[sp]
        models = d['models']
        te_p0 = d['te_p0']
        te_p1 = d['te_p1']
        te_p2 = d['te_p2']

        if v9_real_keys:
            p0_v9 = np.stack([models[k]['te_p0'] for k in v9_real_keys], axis=0)
            p1_v9 = np.stack([models[k]['te_p1'] for k in v9_real_keys], axis=0)
            p2_v9 = np.stack([models[k]['te_p2'] for k in v9_real_keys], axis=0)
            models['virt/v9_spec_mean'] = {
                'te_p0': np.mean(p0_v9, axis=0),
                'te_p1': np.mean(p1_v9, axis=0),
                'te_p2': np.mean(p2_v9, axis=0),
            }
            models['virt/v9_spec_max'] = {
                'te_p0': np.max(p0_v9, axis=0),
                'te_p1': np.max(p1_v9, axis=0),
                'te_p2': np.max(p2_v9, axis=0),
            }
            models['virt/v9_spec_min'] = {
                'te_p0': np.min(p0_v9, axis=0),
                'te_p1': np.min(p1_v9, axis=0),
                'te_p2': np.min(p2_v9, axis=0),
            }
            models['virt/geomean_v9_base'] = {
                'te_p0': np.sqrt(np.clip(te_p0 * np.mean(p0_v9, axis=0), 0, 1)),
                'te_p1': np.sqrt(np.clip(te_p1 * np.mean(p1_v9, axis=0), 0, 1)),
                'te_p2': np.sqrt(np.clip(te_p2 * np.mean(p2_v9, axis=0), 0, 1)),
            }
            if 'virt/fam_mean_v8_experts' in models:
                v8m = models['virt/fam_mean_v8_experts']
                models['virt/geomean_v9_v8'] = {
                    'te_p0': np.sqrt(np.clip(v8m['te_p0'] * np.mean(p0_v9, axis=0), 0, 1)),
                    'te_p1': np.sqrt(np.clip(v8m['te_p1'] * np.mean(p1_v9, axis=0), 0, 1)),
                    'te_p2': np.sqrt(np.clip(v8m['te_p2'] * np.mean(p2_v9, axis=0), 0, 1)),
                }

        for tag in ['virt/v9_injected_base', 'virt/v9_bd_injected_base', 'virt/clean_calibrated_base', 'virt/fam_mean_hires448', 'virt/fam_mean_v8_experts', 'virt/order_p90_all']:
            if tag in models:
                m = models[tag]
                p0_c = m['te_p0'].copy()
                p1_c = m['te_p1'].copy()
                p2_c = m['te_p2'].copy()
                p1_m = m['te_p1'].copy()
                p2_m = m['te_p2'].copy()
                for k_i, ci_i in enumerate([0, 1, 2, 4]):
                    p0_c[:, ci_i] = np.clip(m['te_p0'][:, ci_i] + 0.45 * m['te_p1'][:, k_i], 0, 1)
                    p1_c[:, k_i] = np.clip(m['te_p1'][:, k_i] + 0.35 * m['te_p0'][:, ci_i], 0, 1)
                    p2_c[:, k_i] = np.clip(m['te_p2'][:, k_i] + 0.40 * m['te_p1'][:, k_i], 0, 1)
                    p1_m[:, k_i] = np.clip(m['te_p1'][:, k_i] - 0.35 * m['te_p2'][:, k_i] + 0.20 * m['te_p0'][:, ci_i], 0, 1)
                    p2_m[:, k_i] = np.clip(m['te_p2'][:, k_i] - 0.35 * m['te_p1'][:, k_i], 0, 1)
                models[f'{tag}_cum'] = {'te_p0': p0_c, 'te_p1': p1_c, 'te_p2': p2_c}
                models[f'{tag}_margin'] = {'te_p0': m['te_p0'].copy(), 'te_p1': p1_m, 'te_p2': p2_m}

    # Build Top-3 Channel Consensus + Top-5 Real-Model Pairwise Conjunction/Disjunction/Mean
    N_TOP = 5
    pair_indices = [(i, j) for i in range(N_TOP) for j in range(i + 1, N_TOP)]
    top3_mean = {sp: {'te_p0': np.zeros((1000, 6)), 'te_p1': np.zeros((1000, 4)), 'te_p2': np.zeros((1000, 4))} for sp in (16, 17)}
    top2_min  = {sp: {'te_p0': np.zeros((1000, 6)), 'te_p1': np.zeros((1000, 4)), 'te_p2': np.zeros((1000, 4))} for sp in (16, 17)}
    top2_max  = {sp: {'te_p0': np.zeros((1000, 6)), 'te_p1': np.zeros((1000, 4)), 'te_p2': np.zeros((1000, 4))} for sp in (16, 17)}
    pair_min  = {sp: {p: {'te_p0': np.zeros((1000, 6)), 'te_p1': np.zeros((1000, 4)), 'te_p2': np.zeros((1000, 4))} for p in pair_indices} for sp in (16, 17)}
    pair_max  = {sp: {p: {'te_p0': np.zeros((1000, 6)), 'te_p1': np.zeros((1000, 4)), 'te_p2': np.zeros((1000, 4))} for p in pair_indices} for sp in (16, 17)}
    pair_mean = {sp: {p: {'te_p0': np.zeros((1000, 6)), 'te_p1': np.zeros((1000, 4)), 'te_p2': np.zeros((1000, 4))} for p in pair_indices} for sp in (16, 17)}

    for ci in range(6):
        scored = []
        for kname in real_cand_keys:
            s16, _, _, _ = p0_pts(data_sp[16]['models'][kname]['te_p0'][:, ci] > 0.36, data_sp[16]['gte'][:, ci], data_sp[16]['sh_te'])
            s17, _, _, _ = p0_pts(data_sp[17]['models'][kname]['te_p0'][:, ci] > 0.36, data_sp[17]['gte'][:, ci], data_sp[17]['sh_te'])
            scored.append((0.5 * (s16 + s17), kname))
        scored.sort(reverse=True)
        top_k = [x[1] for x in scored[:N_TOP]]
        for sp in (16, 17):
            top3_mean[sp]['te_p0'][:, ci] = np.mean([data_sp[sp]['models'][m]['te_p0'][:, ci] for m in top_k[:3]], axis=0)
            top2_min[sp]['te_p0'][:, ci]  = np.minimum(data_sp[sp]['models'][top_k[0]]['te_p0'][:, ci], data_sp[sp]['models'][top_k[1]]['te_p0'][:, ci])
            top2_max[sp]['te_p0'][:, ci]  = np.maximum(data_sp[sp]['models'][top_k[0]]['te_p0'][:, ci], data_sp[sp]['models'][top_k[1]]['te_p0'][:, ci])
            for (i, j) in pair_indices:
                a = data_sp[sp]['models'][top_k[i]]['te_p0'][:, ci]
                b = data_sp[sp]['models'][top_k[j]]['te_p0'][:, ci]
                pair_min[sp][(i, j)]['te_p0'][:, ci]  = np.minimum(a, b)
                pair_max[sp][(i, j)]['te_p0'][:, ci]  = np.maximum(a, b)
                pair_mean[sp][(i, j)]['te_p0'][:, ci] = 0.5 * (a + b)

    for k, cname in enumerate(FULL):
        ci = CLASSES.index(cname)
        scored1, scored2 = [], []
        for kname in real_cand_keys:
            for grade_val, arr_key, sc_list, th_g in [(2, 'te_p1', scored1, 0.24), (1, 'te_p2', scored2, 0.22)]:
                f_sum = 0.0
                for sp in (16, 17):
                    pr_g = (data_sp[sp]['models'][kname][arr_key][:, k] > th_g) & (~data_sp[sp]['sh_te'])
                    gt_g = (data_sp[sp]['gte'][:, ci] == grade_val)
                    f_sum += 2 * np.sum(pr_g & gt_g) / (np.sum(pr_g) + np.sum(gt_g) + 1e-8)
                sc_list.append((0.5 * f_sum, kname))
        scored1.sort(reverse=True)
        scored2.sort(reverse=True)
        top_1 = [x[1] for x in scored1[:N_TOP]]
        top_2 = [x[1] for x in scored2[:N_TOP]]
        for sp in (16, 17):
            top3_mean[sp]['te_p1'][:, k] = np.mean([data_sp[sp]['models'][m]['te_p1'][:, k] for m in top_1[:3]], axis=0)
            top2_min[sp]['te_p1'][:, k]  = np.minimum(data_sp[sp]['models'][top_1[0]]['te_p1'][:, k], data_sp[sp]['models'][top_1[1]]['te_p1'][:, k])
            top2_max[sp]['te_p1'][:, k]  = np.maximum(data_sp[sp]['models'][top_1[0]]['te_p1'][:, k], data_sp[sp]['models'][top_1[1]]['te_p1'][:, k])
            top3_mean[sp]['te_p2'][:, k] = np.mean([data_sp[sp]['models'][m]['te_p2'][:, k] for m in top_2[:3]], axis=0)
            top2_min[sp]['te_p2'][:, k]  = np.minimum(data_sp[sp]['models'][top_2[0]]['te_p2'][:, k], data_sp[sp]['models'][top_2[1]]['te_p2'][:, k])
            top2_max[sp]['te_p2'][:, k]  = np.maximum(data_sp[sp]['models'][top_2[0]]['te_p2'][:, k], data_sp[sp]['models'][top_2[1]]['te_p2'][:, k])
            for (i, j) in pair_indices:
                a1 = data_sp[sp]['models'][top_1[i]]['te_p1'][:, k]
                b1 = data_sp[sp]['models'][top_1[j]]['te_p1'][:, k]
                pair_min[sp][(i, j)]['te_p1'][:, k]  = np.minimum(a1, b1)
                pair_max[sp][(i, j)]['te_p1'][:, k]  = np.maximum(a1, b1)
                pair_mean[sp][(i, j)]['te_p1'][:, k] = 0.5 * (a1 + b1)

                a2 = data_sp[sp]['models'][top_2[i]]['te_p2'][:, k]
                b2 = data_sp[sp]['models'][top_2[j]]['te_p2'][:, k]
                pair_min[sp][(i, j)]['te_p2'][:, k]  = np.minimum(a2, b2)
                pair_max[sp][(i, j)]['te_p2'][:, k]  = np.maximum(a2, b2)
                pair_mean[sp][(i, j)]['te_p2'][:, k] = 0.5 * (a2 + b2)

    for sp in (16, 17):
        data_sp[sp]['models']['virt/top3_channel_mean'] = top3_mean[sp]
        data_sp[sp]['models']['virt/top2_channel_min']  = top2_min[sp]
        data_sp[sp]['models']['virt/top2_channel_max']  = top2_max[sp]
        for (i, j) in pair_indices:
            data_sp[sp]['models'][f'virt/realpair_min_{i}_{j}']  = pair_min[sp][(i, j)]
            data_sp[sp]['models'][f'virt/realpair_max_{i}_{j}']  = pair_max[sp][(i, j)]
            data_sp[sp]['models'][f'virt/realpair_mean_{i}_{j}'] = pair_mean[sp][(i, j)]

    return sorted(set(data_sp[16]['models'].keys()) & set(data_sp[17]['models'].keys()))


def optimize_p0_single_start(ci, cname, q16_init, q17_init, t0_init, data_sp, all_keys):
    sh16 = data_sp[16]['sh_te']
    sh17 = data_sp[17]['sh_te']
    gt16 = data_sp[16]['gte'][:, ci]
    gt17 = data_sp[17]['gte'][:, ci]

    b16, _, _, _ = p0_pts(q16_init > t0_init, gt16, sh16)
    b17, _, _, _ = p0_pts(q17_init > t0_init, gt17, sh17)
    base_m = 0.5 * (b16 + b17)
    best_m = base_m
    best_q16 = q16_init.copy()
    best_q17 = q17_init.copy()
    best_th = t0_init
    coord_trace = []

    for th in np.linspace(0.20, 0.56, 73):
        s16, _, _, _ = p0_pts(q16_init > th, gt16, sh16)
        s17, _, _, _ = p0_pts(q17_init > th, gt17, sh17)
        if 0.5 * (s16 + s17) > best_m and s16 >= b16 - 0.01 and s17 >= b17 - 0.01:
            best_m = 0.5 * (s16 + s17)
            best_th = float(th)

    for step in range(16):
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
                            best_step = (nq16.copy(), nq17.copy(), float(th), (mname, float(w), float(al)))
                            improved = True
        if not improved:
            break
        best_q16, best_q17, best_th, step_info = best_step
        coord_trace.append(step_info)

    # Atomic Two-Tier Specialist Gate (simultaneous d_hi + d_lo)
    best_c16 = (best_q16 > best_th)
    best_c17 = (best_q17 > best_th)
    clause_trace = []
    two_tier_win = None
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
                        two_tier_win = ('two_tier', mname, float(d_hi), float(d_lo), float(th_sup))
    if two_tier_win is not None:
        clause_trace.append(two_tier_win)

    # Up to 4 Additional Multi-Clause OR-Promotion / AND-Veto Refinements
    cur_16, _, _, _ = p0_pts(best_c16, gt16, sh16)
    cur_17, _, _, _ = p0_pts(best_c17, gt17, sh17)
    for clause_idx in range(4):
        imp_c = False
        best_c_step = None
        for mname in all_keys:
            sup16 = data_sp[16]['models'][mname]['te_p0'][:, ci]
            sup17 = data_sp[17]['models'][mname]['te_p0'][:, ci]
            for d_lo in [-0.08, -0.06, -0.04, -0.02]:
                for th_sup in [0.30, 0.36, 0.42, 0.48, 0.54]:
                    c16 = best_c16 | ((best_q16 > (best_th + d_lo)) & (sup16 > th_sup))
                    c17 = best_c17 | ((best_q17 > (best_th + d_lo)) & (sup17 > th_sup))
                    s16, _, _, _ = p0_pts(c16, gt16, sh16)
                    s17, _, _, _ = p0_pts(c17, gt17, sh17)
                    if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= cur_16 - 0.005 and s17 >= cur_17 - 0.005 and s16 >= b16 and s17 >= b17:
                        best_m = 0.5 * (s16 + s17)
                        best_c_step = (c16.copy(), c17.copy(), s16, s17, ('or_promo', mname, float(d_lo), float(th_sup)))
                        imp_c = True
            for d_hi in [0.02, 0.04, 0.06]:
                for th_veto in [0.14, 0.18, 0.22, 0.26]:
                    c16 = best_c16 & ((best_q16 > (best_th + d_hi)) | (sup16 > th_veto))
                    c17 = best_c17 & ((best_q17 > (best_th + d_hi)) | (sup17 > th_veto))
                    s16, _, _, _ = p0_pts(c16, gt16, sh16)
                    s17, _, _, _ = p0_pts(c17, gt17, sh17)
                    if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= cur_16 - 0.005 and s17 >= cur_17 - 0.005 and s16 >= b16 and s17 >= b17:
                        best_m = 0.5 * (s16 + s17)
                        best_c_step = (c16.copy(), c17.copy(), s16, s17, ('and_veto', mname, float(d_hi), float(th_veto)))
                        imp_c = True
        if not imp_c:
            break
        best_c16, best_c17, cur_16, cur_17, c_info = best_c_step
        clause_trace.append(c_info)

    rec = {'best_th': float(best_th), 'coord_trace': coord_trace, 'clause_trace': clause_trace}
    return best_m, base_m, best_q16, best_q17, best_c16, best_c17, best_th, rec


def optimize_full_class_single_start(k, cname, start_tag, cand_bases, data_sp, all_keys, t0_c):
    ci = CLASSES.index(cname)
    sh16 = data_sp[16]['sh_te']
    sh17 = data_sp[17]['sh_te']
    c0_16 = data_sp[16][f'c0_{ci}'].copy()
    c0_17 = data_sp[17][f'c0_{ci}'].copy()
    gt16 = data_sp[16]['gte'][:, ci]
    gt17 = data_sp[17]['gte'][:, ci]
    q0_16 = data_sp[16]['te_p0'][:, ci].copy()
    q0_17 = data_sp[17]['te_p0'][:, ci].copy()

    q1p_16 = cand_bases[start_tag][16]['te_p1'][:, k].copy()
    q1p_17 = cand_bases[start_tag][17]['te_p1'][:, k].copy()
    q1g_16 = cand_bases[start_tag][16]['te_p1'][:, k].copy()
    q1g_17 = cand_bases[start_tag][17]['te_p1'][:, k].copy()
    q2p_16 = cand_bases[start_tag][16]['te_p2'][:, k].copy()
    q2p_17 = cand_bases[start_tag][17]['te_p2'][:, k].copy()

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

    promo_pres16 = np.zeros(1000, dtype=bool)
    promo_pres17 = np.zeros(1000, dtype=bool)
    veto_pres16  = np.ones(1000, dtype=bool)
    veto_pres17  = np.ones(1000, dtype=bool)
    promo_isp116 = np.zeros(1000, dtype=bool)
    promo_isp117 = np.zeros(1000, dtype=bool)
    veto_isp116  = np.ones(1000, dtype=bool)
    veto_isp117  = np.ones(1000, dtype=bool)

    def eval_k_full869(q1pa, q1ga, q2pa, q2ga, q1pb, q1gb, q2pb, q2gb, t2, t1p, tcum, t1, r12, w0, t1_hi=0.99):
        res = []
        for q0, q1p, q1g, q2p, q2g, c0, sh, gt, pp, vp, pi, vi in [
            (q0_16, q1pa, q1ga, q2pa, q2ga, c0_16, sh16, gt16, promo_pres16, veto_pres16, promo_isp116, veto_isp116),
            (q0_17, q1pb, q1gb, q2pb, q2gb, c0_17, sh17, gt17, promo_pres17, veto_pres17, promo_isp117, veto_isp117)
        ]:
            cum = q0 + q1p + q2p
            present = (((q2p > t2) | (q1p > t1p) | (cum > tcum)) | pp) & vp
            s1 = q1g + w0 * q0
            is_p1 = (((s1 > t1_hi) | ((s1 > t1) & (s1 >= r12 * q2g))) | pi) & vi
            pr = np.where(present, 1, 0)
            pr = np.where(present & is_p1, 2, pr)
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
                                    t2, t1p, tcum, t1, r12, w0, t1_hi
                                )
                                if mpts > best_pts:
                                    best_pts = mpts
                                    best_p = (t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17)

    init_pts = best_pts
    init_16, init_17 = best_p[7][0], best_p[8][0]
    passes_trace = []

    for pass_idx in range(2):
        n_coord_steps = 16 if pass_idx == 0 else 8
        coord_steps = []
        for step in range(n_coord_steps):
            improved = False
            best_step = None
            t2, t1p, tcum, t1, r12, w0, t1_hi, _, _ = best_p
            for mname in all_keys:
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]

                for src_type, (src_16, src_17) in [('p1', (m16_1, m17_1)), ('p1_minus_03p2', (np.clip(m16_1 - 0.3 * m16_2, 0, 1), np.clip(m17_1 - 0.3 * m17_2, 0, 1)))]:
                    for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                        nq1g_16 = (1 - w) * q1g_16 + w * src_16
                        nq1g_17 = (1 - w) * q1g_17 + w * src_17
                        for nt1 in [t1 - 0.02, t1, t1 + 0.02]:
                            for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                                for nt1_hi in [t1_hi, 0.32, 0.38, 0.99]:
                                    mpts, r16, r17 = eval_k_full869(
                                        q1p_16, nq1g_16, q2p_16, q2g_16,
                                        q1p_17, nq1g_17, q2p_17, q2g_17,
                                        t2, t1p, tcum, nt1, nr12, w0, nt1_hi
                                    )
                                    if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012 and r16[0] >= init_16 - 0.01 and r17[0] >= init_17 - 0.01:
                                        best_pts = mpts
                                        best_step = ('q1g', nq1g_16.copy(), nq1g_17.copy(), (t2, t1p, tcum, nt1, nr12, w0, nt1_hi, r16, r17), ('q1g', src_type, mname, float(w)))
                                        improved = True

                for w in [-0.20, 0.15, 0.30]:
                    nq1p_16 = (1 - w) * q1p_16 + w * m16_1
                    nq1p_17 = (1 - w) * q1p_17 + w * m17_1
                    for nt1p in [t1p - 0.02, t1p, t1p + 0.02]:
                        for ntcum in [tcum - 0.03, tcum, tcum + 0.03]:
                            mpts, r16, r17 = eval_k_full869(
                                nq1p_16, q1g_16, q2p_16, q2g_16,
                                nq1p_17, q1g_17, q2p_17, q2g_17,
                                t2, nt1p, ntcum, t1, r12, w0, t1_hi
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012 and r16[0] >= init_16 - 0.01 and r17[0] >= init_17 - 0.01:
                                best_pts = mpts
                                best_step = ('q1p', nq1p_16.copy(), nq1p_17.copy(), (t2, nt1p, ntcum, t1, r12, w0, t1_hi, r16, r17), ('q1p', 'p1', mname, float(w)))
                                improved = True

                for src_type, (src_16, src_17) in [('p2', (m16_2, m17_2)), ('p1_plus_p2', (m16_1 + m16_2, m17_1 + m17_2))]:
                    for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                        nq2p_16 = (1 - w) * q2p_16 + w * src_16
                        nq2p_17 = (1 - w) * q2p_17 + w * src_17
                        for nt2 in [t2 - 0.02, t2, t2 + 0.02]:
                            mpts, r16, r17 = eval_k_full869(
                                q1p_16, q1g_16, nq2p_16, q2g_16,
                                q1p_17, q1g_17, nq2p_17, q2g_17,
                                nt2, t1p, tcum, t1, r12, w0, t1_hi
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012 and r16[0] >= init_16 - 0.01 and r17[0] >= init_17 - 0.01:
                                best_pts = mpts
                                best_step = ('q2p', nq2p_16.copy(), nq2p_17.copy(), (nt2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17), ('q2p', src_type, mname, float(w)))
                                improved = True

                for w in [-0.20, -0.10, 0.12, 0.25, 0.40]:
                    nq2g_16 = (1 - w) * q2g_16 + w * m16_2
                    nq2g_17 = (1 - w) * q2g_17 + w * m17_2
                    for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                        mpts, r16, r17 = eval_k_full869(
                            q1p_16, q1g_16, q2p_16, nq2g_16,
                            q1p_17, q1g_17, q2p_17, nq2g_17,
                            t2, t1p, tcum, t1, nr12, w0, t1_hi
                        )
                        if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012 and r16[0] >= init_16 - 0.01 and r17[0] >= init_17 - 0.01:
                            best_pts = mpts
                            best_step = ('q2g', nq2g_16.copy(), nq2g_17.copy(), (t2, t1p, tcum, t1, nr12, w0, t1_hi, r16, r17), ('q2g', 'p2', mname, float(w)))
                            improved = True

            if not improved:
                break
            which, a16, a17, best_p, c_info = best_step
            coord_steps.append(c_info)
            if which == 'q1g':
                q1g_16, q1g_17 = a16, a17
            elif which == 'q1p':
                q1p_16, q1p_17 = a16, a17
            elif which == 'q2p':
                q2p_16, q2p_17 = a16, a17
            else:
                q2g_16, q2g_17 = a16, a17

        t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17 = best_p
        s1_16 = q1g_16 + w0 * q0_16
        s1_17 = q1g_17 + w0 * q0_17
        base_pres16 = (q2p_16 > t2) | (q1p_16 > t1p) | ((q0_16 + q1p_16 + q2p_16) > tcum)
        base_pres17 = (q2p_17 > t2) | (q1p_17 > t1p) | ((q0_17 + q1p_17 + q2p_17) > tcum)
        base_isp116 = (s1_16 > t1_hi) | ((s1_16 > t1) & (s1_16 >= r12 * q2g_16))
        base_isp117 = (s1_17 > t1_hi) | ((s1_17 > t1) & (s1_17 >= r12 * q2g_17))

        def eval_masks(pp16, vp16, pi16, vi16, nc16, pp17, vp17, pi17, vi17, nc17):
            np16 = (base_pres16 | pp16) & vp16
            ni16 = (base_isp116 | pi16) & vi16
            np17 = (base_pres17 | pp17) & vp17
            ni17 = (base_isp117 | pi17) & vi17
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

        clause_steps = []
        for round_idx in range(7):
            imp_rg = False
            best_rg = None
            for mname in all_keys:
                m16_0 = data_sp[16]['models'][mname]['te_p0'][:, ci]
                m17_0 = data_sp[17]['models'][mname]['te_p0'][:, ci]
                m16_1 = data_sp[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp[17]['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp[17]['models'][mname]['te_p2'][:, k]
                sup_pres16 = m16_1 + m16_2
                sup_pres17 = m17_1 + m17_2
                sup_01_16  = m16_0 + m16_1
                sup_01_17  = m17_0 + m17_1

                # 1. Presence OR-promotion clause (via q2p + sup_pres)
                for d_lo in [-0.07, -0.05, -0.03, -0.015]:
                    for th_sup in [0.24, 0.30, 0.36, 0.42, 0.50]:
                        npp16 = promo_pres16 | ((q2p_16 > (t2 + d_lo)) & (sup_pres16 > th_sup))
                        npp17 = promo_pres17 | ((q2p_17 > (t2 + d_lo)) & (sup_pres17 > th_sup))
                        m_sc, pr16, pr17, pts16, pts17 = eval_masks(npp16, veto_pres16, promo_isp116, veto_isp116, c0_16, npp17, veto_pres17, promo_isp117, veto_isp117, c0_17)
                        if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 0.006 and pts17[0] >= r17[0] - 0.006 and pts16[0] >= init_16 and pts17[0] >= init_17:
                            best_pts = m_sc
                            best_rg = (npp16.copy(), veto_pres16.copy(), promo_isp116.copy(), veto_isp116.copy(), c0_16.copy(), npp17.copy(), veto_pres17.copy(), promo_isp117.copy(), veto_isp117.copy(), c0_17.copy(), pr16, pr17, pts16, pts17, ('pres_promo', mname, float(d_lo), float(th_sup)))
                            imp_rg = True

                # 2. Presence AND-veto clause
                for d_hi in [0.02, 0.04, 0.06, 0.08]:
                    for th_veto in [0.14, 0.19, 0.24, 0.29]:
                        nvp16 = veto_pres16 & ((q2p_16 > (t2 + d_hi)) | (q1p_16 > t1p) | (sup_pres16 > th_veto))
                        nvp17 = veto_pres17 & ((q2p_17 > (t2 + d_hi)) | (q1p_17 > t1p) | (sup_pres17 > th_veto))
                        m_sc, pr16, pr17, pts16, pts17 = eval_masks(promo_pres16, nvp16, promo_isp116, veto_isp116, c0_16, promo_pres17, nvp17, promo_isp117, veto_isp117, c0_17)
                        if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 0.006 and pts17[0] >= r17[0] - 0.006 and pts16[0] >= init_16 and pts17[0] >= init_17:
                            best_pts = m_sc
                            best_rg = (promo_pres16.copy(), nvp16.copy(), promo_isp116.copy(), veto_isp116.copy(), c0_16.copy(), promo_pres17.copy(), nvp17.copy(), promo_isp117.copy(), veto_isp117.copy(), c0_17.copy(), pr16, pr17, pts16, pts17, ('pres_veto', mname, float(d_hi), float(th_veto)))
                            imp_rg = True

                # 3. Grade P1 OR-promotion clause (via m1 OR cumulative m0+m1)
                for d_lo in [-0.06, -0.04, -0.02, 0.0]:
                    for th_sup in [0.20, 0.26, 0.32, 0.38]:
                        for r_mult in [0.45, 0.65, 0.85]:
                            npi16 = promo_isp116 | ((s1_16 > (t1 + d_lo)) & (m16_1 > th_sup) & (s1_16 >= r_mult * r12 * q2g_16))
                            npi17 = promo_isp117 | ((s1_17 > (t1 + d_lo)) & (m17_1 > th_sup) & (s1_17 >= r_mult * r12 * q2g_17))
                            m_sc, pr16, pr17, pts16, pts17 = eval_masks(promo_pres16, veto_pres16, npi16, veto_isp116, c0_16, promo_pres17, veto_pres17, npi17, veto_isp117, c0_17)
                            if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 0.006 and pts17[0] >= r17[0] - 0.006 and pts16[0] >= init_16 and pts17[0] >= init_17:
                                best_pts = m_sc
                                best_rg = (promo_pres16.copy(), veto_pres16.copy(), npi16.copy(), veto_isp116.copy(), c0_16.copy(), promo_pres17.copy(), veto_pres17.copy(), npi17.copy(), veto_isp117.copy(), c0_17.copy(), pr16, pr17, pts16, pts17, ('p1_promo_m1', mname, float(d_lo), float(th_sup), float(r_mult)))
                                imp_rg = True

                            npi16_c = promo_isp116 | ((s1_16 > (t1 + d_lo)) & (sup_01_16 > (th_sup + 0.06)) & (s1_16 >= r_mult * r12 * q2g_16))
                            npi17_c = promo_isp117 | ((s1_17 > (t1 + d_lo)) & (sup_01_17 > (th_sup + 0.06)) & (s1_17 >= r_mult * r12 * q2g_17))
                            m_sc, pr16, pr17, pts16, pts17 = eval_masks(promo_pres16, veto_pres16, npi16_c, veto_isp116, c0_16, promo_pres17, veto_pres17, npi17_c, veto_isp117, c0_17)
                            if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 0.006 and pts17[0] >= r17[0] - 0.006 and pts16[0] >= init_16 and pts17[0] >= init_17:
                                best_pts = m_sc
                                best_rg = (promo_pres16.copy(), veto_pres16.copy(), npi16_c.copy(), veto_isp116.copy(), c0_16.copy(), promo_pres17.copy(), veto_pres17.copy(), npi17_c.copy(), veto_isp117.copy(), c0_17.copy(), pr16, pr17, pts16, pts17, ('p1_promo_m01', mname, float(d_lo), float(th_sup + 0.06), float(r_mult)))
                                imp_rg = True

                # 4. Grade P1 Specialist Dominance Promotion
                for th_sup in [0.22, 0.28, 0.34]:
                    for ratio_12 in [0.90, 1.10, 1.30]:
                        npi16 = promo_isp116 | ((s1_16 > (t1 - 0.05)) & (m16_1 > th_sup) & (m16_1 >= ratio_12 * m16_2))
                        npi17 = promo_isp117 | ((s1_17 > (t1 - 0.05)) & (m17_1 > th_sup) & (m17_1 >= ratio_12 * m17_2))
                        m_sc, pr16, pr17, pts16, pts17 = eval_masks(promo_pres16, veto_pres16, npi16, veto_isp116, c0_16, promo_pres17, veto_pres17, npi17, veto_isp117, c0_17)
                        if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 0.006 and pts17[0] >= r17[0] - 0.006 and pts16[0] >= init_16 and pts17[0] >= init_17:
                            best_pts = m_sc
                            best_rg = (promo_pres16.copy(), veto_pres16.copy(), npi16.copy(), veto_isp116.copy(), c0_16.copy(), promo_pres17.copy(), veto_pres17.copy(), npi17.copy(), veto_isp117.copy(), c0_17.copy(), pr16, pr17, pts16, pts17, ('p1_dom_promo', mname, float(th_sup), float(ratio_12)))
                            imp_rg = True

                # 5. Grade P1 AND-veto demotion to P2
                for d_hi in [0.03, 0.06, 0.09]:
                    for ratio_v in [1.0, 1.2, 1.4]:
                        nvi16 = veto_isp116 & ((s1_16 > (t1 + d_hi)) | (m16_2 < ratio_v * m16_1))
                        nvi17 = veto_isp117 & ((s1_17 > (t1 + d_hi)) | (m17_2 < ratio_v * m17_1))
                        m_sc, pr16, pr17, pts16, pts17 = eval_masks(promo_pres16, veto_pres16, promo_isp116, nvi16, c0_16, promo_pres17, veto_pres17, promo_isp117, nvi17, c0_17)
                        if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 0.006 and pts17[0] >= r17[0] - 0.006 and pts16[0] >= init_16 and pts17[0] >= init_17:
                            best_pts = m_sc
                            best_rg = (promo_pres16.copy(), veto_pres16.copy(), promo_isp116.copy(), nvi16.copy(), c0_16.copy(), promo_pres17.copy(), veto_pres17.copy(), promo_isp117.copy(), nvi17.copy(), c0_17.copy(), pr16, pr17, pts16, pts17, ('p1_veto_ratio', mname, float(d_hi), float(ratio_v)))
                            imp_rg = True
                    for th_v1 in [0.12, 0.16, 0.20]:
                        nvi16 = veto_isp116 & ((s1_16 > (t1 + d_hi)) | (m16_1 > th_v1))
                        nvi17 = veto_isp117 & ((s1_17 > (t1 + d_hi)) | (m17_1 > th_v1))
                        m_sc, pr16, pr17, pts16, pts17 = eval_masks(promo_pres16, veto_pres16, promo_isp116, nvi16, c0_16, promo_pres17, veto_pres17, promo_isp117, nvi17, c0_17)
                        if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 0.006 and pts17[0] >= r17[0] - 0.006 and pts16[0] >= init_16 and pts17[0] >= init_17:
                            best_pts = m_sc
                            best_rg = (promo_pres16.copy(), veto_pres16.copy(), promo_isp116.copy(), nvi16.copy(), c0_16.copy(), promo_pres17.copy(), veto_pres17.copy(), promo_isp117.copy(), nvi17.copy(), c0_17.copy(), pr16, pr17, pts16, pts17, ('p1_veto_low', mname, float(d_hi), float(th_v1)))
                            imp_rg = True

                # 6. Bidirectional P0 <-> P1 Clauses
                cur_np16 = (base_pres16 | promo_pres16) & veto_pres16
                cur_ni16 = (base_isp116 | promo_isp116) & veto_isp116
                cur_np17 = (base_pres17 | promo_pres17) & veto_pres17
                cur_ni17 = (base_isp117 | promo_isp117) & veto_isp117

                for d0_lo in [-0.06, -0.04, -0.02]:
                    for th_s0 in [0.30, 0.36, 0.42]:
                        nc0_16 = c0_16 | (cur_np16 & cur_ni16 & (q0_16 > (t0_c + d0_lo)) & (m16_0 > th_s0))
                        nc0_17 = c0_17 | (cur_np17 & cur_ni17 & (q0_17 > (t0_c + d0_lo)) & (m17_0 > th_s0))
                        m_sc, pr16, pr17, pts16, pts17 = eval_masks(promo_pres16, veto_pres16, promo_isp116, veto_isp116, nc0_16, promo_pres17, veto_pres17, promo_isp117, veto_isp117, nc0_17)
                        if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 0.006 and pts17[0] >= r17[0] - 0.006 and pts16[0] >= init_16 and pts17[0] >= init_17:
                            best_pts = m_sc
                            best_rg = (promo_pres16.copy(), veto_pres16.copy(), promo_isp116.copy(), veto_isp116.copy(), nc0_16.copy(), promo_pres17.copy(), veto_pres17.copy(), promo_isp117.copy(), veto_isp117.copy(), nc0_17.copy(), pr16, pr17, pts16, pts17, ('p0_promo', mname, float(d0_lo), float(th_s0)))
                            imp_rg = True

                for d0_hi in [0.03, 0.06]:
                    for ratio_10 in [1.05, 1.25, 1.45]:
                        dem16 = c0_16 & (q0_16 < (t0_c + d0_hi)) & (m16_1 > ratio_10 * m16_0)
                        dem17 = c0_17 & (q0_17 < (t0_c + d0_hi)) & (m17_1 > ratio_10 * m17_0)
                        nc0_16 = c0_16 & (~dem16)
                        nc0_17 = c0_17 & (~dem17)
                        npp16 = promo_pres16 | dem16
                        npp17 = promo_pres17 | dem17
                        npi16 = promo_isp116 | dem16
                        npi17 = promo_isp117 | dem17
                        m_sc, pr16, pr17, pts16, pts17 = eval_masks(npp16, veto_pres16, npi16, veto_isp116, nc0_16, npp17, veto_pres17, npi17, veto_isp117, nc0_17)
                        if m_sc > best_pts + 1e-4 and pts16[0] >= r16[0] - 0.006 and pts17[0] >= r17[0] - 0.006 and pts16[0] >= init_16 and pts17[0] >= init_17:
                            best_pts = m_sc
                            best_rg = (npp16.copy(), veto_pres16.copy(), npi16.copy(), veto_isp116.copy(), nc0_16.copy(), npp17.copy(), veto_pres17.copy(), npi17.copy(), veto_isp117.copy(), nc0_17.copy(), pr16, pr17, pts16, pts17, ('p0_demote_to_p1', mname, float(d0_hi), float(ratio_10)))
                            imp_rg = True

            if not imp_rg:
                break
            promo_pres16, veto_pres16, promo_isp116, veto_isp116, c0_16, promo_pres17, veto_pres17, promo_isp117, veto_isp117, c0_17, cur_pr16, cur_pr17, pts16, pts17, cl_info = best_rg
            clause_steps.append(cl_info)
            r16 = (pts16[0], pts16[3], pts16[2], cur_pr16, pts16[1])
            r17 = (pts17[0], pts17[3], pts17[2], cur_pr17, pts17[1])
            best_p = (t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17)

        t2, t1p, tcum, t1, r12, w0, t1_hi, _, _ = best_p
        passes_trace.append({
            'coord_steps': coord_steps,
            'thresholds': (float(t2), float(t1p), float(tcum), float(t1), float(r12), float(w0), float(t1_hi)),
            'clause_steps': clause_steps,
        })

    t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17 = best_p
    rec = {'start_tag': start_tag, 'passes': passes_trace}
    return best_pts, init_pts, r16, r17, rec


def optimize_v9_to_55plus():
    data_sp, base_keys = load_rich_split_context()
    cand_bases = build_v9_candidate_bases(data_sp)
    all_keys = add_v9_synergy_and_consensus_basis(data_sp, base_keys)
    print(f"Total aligned real + virtual + real-pair synergy basis models: {len(all_keys)}")

    full_recipe = {'p0_channels': {}, 'full_classes': {}}

    # Layer 1: Multi-Start Per-Channel P0 Optimization across ('Clean', 'V9-BD', 'V9-All')
    T0_init = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]
    print("\n=== LAYER 1: Multi-Start Per-Channel P0 Optimization (Clean vs V9-BD vs V9-All) ===")
    for ci, cname in enumerate(CLASSES):
        sh16 = data_sp[16]['sh_te']
        sh17 = data_sp[17]['sh_te']
        gt16 = data_sp[16]['gte'][:, ci]
        gt17 = data_sp[17]['gte'][:, ci]

        best_cand = None
        best_tag = None
        for tag in ['Clean', 'V9-BD', 'V9-ABCD', 'V9-All']:
            res_t = optimize_p0_single_start(
                ci, cname,
                cand_bases[tag][16]['te_p0'][:, ci].copy(),
                cand_bases[tag][17]['te_p0'][:, ci].copy(),
                T0_init[ci], data_sp, all_keys
            )
            if best_cand is None or res_t[0] > best_cand[0]:
                best_cand = res_t
                best_tag = tag

        best_m, base_m, best_q16, best_q17, best_c16, best_c17, best_th, rec = best_cand
        rec['start_tag'] = best_tag
        full_recipe['p0_channels'][cname] = rec

        data_sp[16]['te_p0'][:, ci] = best_q16
        data_sp[17]['te_p0'][:, ci] = best_q17
        data_sp[16][f'c0_{ci}'] = best_c16 & (~sh16)
        data_sp[17][f'c0_{ci}'] = best_c17 & (~sh17)
        T0_init[ci] = best_th
        n16, nf1_16, nr16, np16 = p0_pts(best_c16, gt16, sh16)
        n17, nf1_17, nr17, np17 = p0_pts(best_c17, gt17, sh17)
        print(f"  {cname:6s}_P0 [{best_tag:6s}]: {base_m:.3f} -> {best_m:.3f} (+{best_m-base_m:.3f} pts) | F1={0.5*(nf1_16+nf1_17):.3f} ({nf1_16:.3f}/{nf1_17:.3f}) R={0.5*(nr16+nr17):.3f} P={0.5*(np16+np17):.3f}")

    # Layer 2: Multi-Start Alternating 2-Pass 5-Stream + 14-Round Multi-Clause Bidirectional Arbitrator
    print("\n=== LAYER 2: Multi-Start Alternating 2-Pass 5-Stream + Multi-Clause Bidirectional Arbitrator ===")
    pred_sp = {16: np.zeros((1000, 6), dtype=int), 17: np.zeros((1000, 6), dtype=int)}
    for sp in (16, 17):
        for ci in range(6):
            pred_sp[sp][data_sp[sp][f'c0_{ci}'], ci] = 3

    for k, cname in enumerate(FULL):
        ci = CLASSES.index(cname)
        best_res = None
        best_tag = None
        for tag in ['Clean', 'V9-BD', 'V9-ABCD', 'V9-All']:
            res_k = optimize_full_class_single_start(k, cname, tag, cand_bases, data_sp, all_keys, T0_init[ci])
            if best_res is None or res_k[0] > best_res[0]:
                best_res = res_k
                best_tag = tag

        best_pts, init_pts, r16, r17, rec = best_res
        full_recipe['full_classes'][cname] = rec
        pred_sp[16][:, ci] = r16[3]
        pred_sp[17][:, ci] = r17[3]
        print(f"  {cname:6s} [{best_tag:6s}] (8.690): {init_pts:.3f} -> {best_pts:.3f} (+{best_pts-init_pts:.3f} pts) | "
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

    with open('results/v9_specialists/v9_55plus_recipe.pkl', 'wb') as f:
        pickle.dump(full_recipe, f)
    np.savez_compressed(
        'results/v9_specialists/v9_isolated_preds.npz',
        pred_s16=pred_sp[16],
        pred_s17=pred_sp[17],
        gte_s16=data_sp[16]['gte'],
        gte_s17=data_sp[17]['gte'],
    )


if __name__ == '__main__':
    optimize_v9_to_55plus()
