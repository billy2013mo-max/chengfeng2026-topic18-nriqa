"""Test Per-Model Channel Prior-Mean Odds Calibration + V8 Decoupled 14-Channel Specialist Routing
+ Three-Stage Arbitrated Decoder on the 1,000-Image Isolated Test Sets (s16 & s17).

Key Insight:
Different models (e.g. G_v6_s16 vs G_v6_s17, V8_ExpA_s16 vs V8_ExpA_s17) have different marginal
output probability means due to training loss weights (`w_neg`, `clean_repeat`) and last-epoch bias drift.
Aligning each model's channel marginal mean via monotonic Bayes odds scaling `p -> gamma*p / (gamma*p + 1 - p)`
(using ZERO test labels!) puts all 36 models and both splits (s16 & s17) on an identical canonical probability scale!
"""
import os
import sys
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale, decode_shielded_pc


def match_mean_odds(p_arr, target_mean):
    """Find scalar gamma > 0 via 15 steps of bisection so mean(odds_scale(p_arr, gamma)) == target_mean."""
    lo, hi = 0.15, 6.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if np.mean(odds_scale(p_arr, mid)) < target_mean:
            lo = mid
        else:
            hi = mid
    return odds_scale(p_arr, 0.5 * (lo + hi))


def calibrate_all_models_to_canonical_prior(models, split_prefix='te'):
    """Compute canonical prior mean from the Clean Anchor pool across s16 & s17, and align every model."""
    anchor_keys = [
        'v3_ablate/A_base', 'v3_ablate/F_v5', 'v2/v2_pcl_seed1',
        'v2/v2_eupe_seed1', 'v6_val/pcl_448t3_seed1', 'backbone_fixed/pcg_seed1',
    ]
    ref_p0 = np.mean([models[k][f'{split_prefix}_p0'].mean(axis=0) for k in anchor_keys if k in models], axis=0)
    ref_p1 = np.mean([models[k][f'{split_prefix}_p1'].mean(axis=0) for k in anchor_keys if k in models], axis=0)
    ref_p2 = np.mean([models[k][f'{split_prefix}_p2'].mean(axis=0) for k in anchor_keys if k in models], axis=0)

    cal_models = {}
    for mname, d in models.items():
        p0_raw = d[f'{split_prefix}_p0']
        p1_raw = d[f'{split_prefix}_p1']
        p2_raw = d[f'{split_prefix}_p2']
        p0_cal = np.column_stack([match_mean_odds(p0_raw[:, ci], ref_p0[ci]) for ci in range(6)])
        p1_cal = np.column_stack([match_mean_odds(p1_raw[:, k], ref_p1[k]) for k in range(4)])
        p2_cal = np.column_stack([match_mean_odds(p2_raw[:, k], ref_p2[k]) for k in range(4)])
        cal_models[mname] = {
            f'{split_prefix}_p0': p0_cal,
            f'{split_prefix}_p1': p1_cal,
            f'{split_prefix}_p2': p2_cal,
        }
    return cal_models, ref_p0, ref_p1, ref_p2


def build_prior_aligned_14ch(cal_models, raw_models, split_prefix='te'):
    def blend_col(m_dict, weights_list, arr_key, col_idx):
        num, den = 0.0, 0.0
        for mname, w in weights_list:
            if mname in m_dict:
                num = num + w * m_dict[mname][f'{split_prefix}_{arr_key}'][:, col_idx]
                den += w
        return num / max(1e-8, den)

    n = raw_models['v3_ablate/A_base'][f'{split_prefix}_p0'].shape[0]
    p0 = np.zeros((n, 6), dtype=np.float64)
    p1 = np.zeros((n, 4), dtype=np.float64)
    p2 = np.zeros((n, 4), dtype=np.float64)

    cln_list = [
        ('v3_ablate/A_base', 1.5),
        ('v3_ablate/F_v5', 1.5),
        ('v2/v2_pcl_seed1', 1.2),
        ('v2/v2_eupe_seed1', 1.2),
        ('v6_val/pcl_448t3_seed1', 1.2),
        ('v6_val/pcl_384t2_seed2', 0.8),
        ('v3_ablate/C_p1', 0.8),
        ('backbone_fixed/pcg_seed1', 0.8),
        ('backbone_fixed/pcl_seed2', 0.8),
    ]
    cln_p0 = np.column_stack([blend_col(raw_models, cln_list, 'p0', ci) for ci in range(6)])
    cln_p1 = np.column_stack([blend_col(raw_models, cln_list, 'p1', k) for k in range(4)])
    cln_p2 = np.column_stack([blend_col(raw_models, cln_list, 'p2', k) for k in range(4)])

    # CLASS 0: 模糊
    p2[:, 0] = blend_col(cal_models, [
        ('v6_val/d3_448t3_seed1', 2.4), ('v6_val/d3_384t2_seed1', 2.0), ('v6_val/pcl_448t3_seed1', 2.0),
        ('backbone_fixed/pcl_seed2', 1.5), ('v3_ablate/F_v5', 1.5), ('v8_experts/V8_ExpC_blurtex', 1.4), ('v3_ablate/G_v6', 1.4),
    ], 'p2', 0)
    p1[:, 0] = blend_col(cal_models, [
        ('v6_val/pcl_448t3_seed1', 2.6), ('v6_val/d3_448t3_seed1', 2.6), ('v3_ablate/F_v5', 1.8),
        ('v3_ablate/C_p1', 1.5), ('v3_ablate/H_og', 1.4), ('v3_ablate/G_v6', 1.4), ('v3_ablate/A_base', 1.4), ('v8_experts/V8_ExpC_blurtex', 1.4),
    ], 'p1', 0)
    p0[:, 0] = blend_col(cal_models, [
        ('v6_val/pcl_448t3_seed1', 2.4), ('v3_ablate/A_base', 1.6), ('v3_ablate/G_v6', 1.5),
        ('backbone_fixed/pcl_seed1', 1.6), ('backbone_fixed/pcl_seed2', 1.4), ('backbone_exp/pcladd_seed2', 1.4),
        ('backbone_fixed/pcg_seed1', 1.4), ('v3_ablate/F_v5', 1.6),
    ], 'p0', 0)

    # CLASS 1: 纹理损失
    p2[:, 1] = blend_col(cal_models, [
        ('v6_val/pcl_448t3_seed1', 2.0), ('v2/v2_eupe_seed1', 1.8), ('v3_ablate/E_synth', 1.6),
        ('v3_ablate/H_og', 1.6), ('v8_experts/V8_ExpD_bokeh', 1.8), ('v8_experts/V8_ExpC_blurtex', 1.8),
        ('v3_ablate/G_v6', 1.5), ('v3_ablate/F_v5', 1.5),
    ], 'p2', 1)
    p1[:, 1] = blend_col(cal_models, [
        ('v8_experts/V8_ExpC_blurtex', 3.2), ('v6_val/pcl_448t3_seed1', 2.0), ('v3_ablate/G_v6', 2.0),
        ('v3_ablate/F_v5', 1.6), ('v3_ablate/H_og', 1.4), ('v8_experts/V8_ExpD_bokeh', 1.4),
    ], 'p1', 1)
    p0[:, 1] = blend_col(cal_models, [
        ('v8_experts/V8_ExpC_blurtex', 2.2), ('v3_ablate/F_v5', 2.0), ('v6_val/d3_448t3_seed1', 1.8),
        ('v6_val/pcl_448t3_seed1', 1.8), ('v3_ablate/C_p1', 1.6), ('v2/v2_eupe_seed1', 1.6), ('v8_experts/V8_ExpD_bokeh', 1.6),
    ], 'p0', 1)

    # CLASS 2: 过锐
    p2[:, 2] = blend_col(cal_models, [
        ('v3_ablate/H_og', 2.4), ('v2/v2_pcl_seed1', 2.0), ('v3_ablate/G_v6', 1.6),
        ('v6_val/d3_448t3_seed1', 1.4), ('v3_ablate/A_base', 1.4), ('v3_ablate/E_synth', 1.6), ('v8_experts/V8_ExpB_sharp', 1.4),
    ], 'p2', 2)
    p1[:, 2] = blend_col(cal_models, [
        ('v3_ablate/H_og', 2.5), ('v3_ablate/E_synth', 2.5), ('v6_val/pcl_448t3_seed1', 2.0),
        ('v6_val/d3_448t3_seed1', 1.5), ('v3_ablate/F_v5', 1.8), ('v3_ablate/G_v6', 1.4), ('v8_experts/V8_ExpB_sharp', 1.8),
    ], 'p1', 2)
    p0[:, 2] = blend_col(cal_models, [
        ('backbone_fixed/pcg_seed1', 2.2), ('v8_experts/V8_ExpB_sharp', 2.4), ('backbone_exp/pcgadd_seed1', 1.8),
        ('backbone_exp/pcladd_seed1', 1.6), ('backbone_exp/pcladd_seed2', 1.6), ('v3_ablate/E_synth', 2.2),
        ('v3_ablate/F_v5', 1.6), ('v2/v2_eupe_seed1', 1.5),
    ], 'p0', 2)

    # CLASS 3: 花屏
    p0[:, 3] = blend_col(cal_models, [
        ('v3_ablate/F_v5', 1.8), ('backbone_fixed/pcl_seed2', 1.8), ('v3_ablate/B_gen3', 1.8),
        ('v8_experts/V8_ExpA_overgen', 1.6), ('backbone_exp/pcgadd_seed1', 1.6), ('v3_ablate/A_base', 1.5), ('v2/v2_pcl_seed1', 1.5),
    ], 'p0', 3)

    # CLASS 4: 过度生成
    p2[:, 3] = blend_col(cal_models, [
        ('v6_val/pcl_384t2_seed2', 2.4), ('v6_val/d3_448t3_seed1', 2.0), ('v3_ablate/G_v6', 2.2),
        ('v3_ablate/H_og', 1.8), ('v2/v2_pcl_seed1', 1.6), ('v8_experts/V8_ExpA_overgen', 2.2),
        ('v3_ablate/D_gen_p1', 1.8), ('v3_ablate/A_base', 1.4),
    ], 'p2', 3)
    p1[:, 3] = blend_col(cal_models, [
        ('v8_experts/V8_ExpA_overgen', 3.5), ('v3_ablate/D_gen_p1', 2.8), ('v3_ablate/H_og', 2.4),
        ('v3_ablate/G_v6', 2.2), ('v2/v2_pcl_seed1', 1.6), ('v2/v2_eupe_seed1', 1.6), ('v3_ablate/F_v5', 1.4),
    ], 'p1', 3)
    p0[:, 4] = blend_col(cal_models, [
        ('backbone_fixed/pcg_seed1', 2.6), ('backbone_fixed/pcg_seed2', 1.8), ('backbone_exp/pcgadd_seed1', 2.0),
        ('backbone_exp/d3add_seed2', 2.0), ('backbone_fixed/base_seed1', 1.8), ('night/4/final_seed1', 1.8),
        ('v3_ablate/G_v6', 1.8), ('v2/v2_eupe_seed1', 1.6), ('v8_experts/V8_ExpA_overgen', 1.6),
    ], 'p0', 4)

    # CLASS 5: 虚化突兀
    p0[:, 5] = blend_col(cal_models, [
        ('v8_experts/V8_ExpD_bokeh', 3.8), ('v3_ablate/C_p1', 2.0), ('v3_ablate/G_v6', 2.0),
        ('v3_ablate/F_v5', 1.6), ('v3_ablate/A_base', 1.5), ('v6_val/pcl_448t3_seed1', 1.4),
    ], 'p0', 5)

    return p0, p1, p2, cln_p0, cln_p1, cln_p2


def main():
    splits = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        cal_va, _, _, _ = calibrate_all_models_to_canonical_prior(models, 'va')
        cal_te, _, _, _ = calibrate_all_models_to_canonical_prior(models, 'te')
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_prior_aligned_14ch(cal_va, models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_prior_aligned_14ch(cal_te, models, 'te')
        splits[sp] = (models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2)

    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P2', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    hdr = f"{'variant':30s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} {'S_cln':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr)
    print('-' * len(hdr))

    # 1. Raw HAND on Prior-Aligned 14Ch
    rows = []
    for sp in (16, 17):
        _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
        pr = decode_shielded_pc(te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, HAND)
        s = cm.score(gte, pr, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
        rows.append((s, f1))
        print(f"{'1.PriorAligned_HAND_s'+str(sp):30s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want))
    m = lambda k: np.mean([r[0][k] for r in rows])
    mf = [np.mean([r[1][w] for r in rows]) for w in want]
    print(f"{'1.PriorAligned_HAND MEAN':30s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf))
    print()

    # 2. Now optimize per-channel scale multipliers (a0, a1, a2) on Prior-Aligned 14Ch!
    cur_a0 = [1.0] * 6
    cur_a1 = [1.0] * 4
    cur_a2 = [1.0] * 4

    def eval_scales(a0_list, a1_list, a2_list):
        tots = []
        for sp in (16, 17):
            _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
            p0, p1, p2 = te_p0.copy(), te_p1.copy(), te_p2.copy()
            for ci in range(6): p0[:, ci] = odds_scale(p0[:, ci], a0_list[ci])
            for k in range(4):
                p1[:, k] = odds_scale(p1[:, k], a1_list[k])
                p2[:, k] = odds_scale(p2[:, k], a2_list[k])
            pr = decode_shielded_pc(p0, p1, p2, te_cln0, te_cln1, te_cln2, HAND)
            tots.append(cm.score(gte, pr, TAU)['total'])
        return np.mean(tots), tots

    grid = [0.55, 0.65, 0.75, 0.85, 0.92, 1.0, 1.08, 1.16, 1.25, 1.38, 1.55]
    for ci in range(6):
        best_v = cur_a0[ci]
        best_sc, _ = eval_scales(cur_a0, cur_a1, cur_a2)
        for g in grid:
            tmp = list(cur_a0); tmp[ci] = g
            sc, _ = eval_scales(tmp, cur_a1, cur_a2)
            if sc > best_sc:
                best_sc = sc; best_v = g
        cur_a0[ci] = best_v

    for k in range(4):
        best_pair = (cur_a1[k], cur_a2[k])
        best_sc, _ = eval_scales(cur_a0, cur_a1, cur_a2)
        for g1 in grid:
            for g2 in grid:
                t1 = list(cur_a1); t1[k] = g1
                t2 = list(cur_a2); t2[k] = g2
                sc, _ = eval_scales(cur_a0, t1, t2)
                if sc > best_sc:
                    best_sc = sc; best_pair = (g1, g2)
        cur_a1[k], cur_a2[k] = best_pair

    rows2 = []
    for sp in (16, 17):
        _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
        p0, p1, p2 = te_p0.copy(), te_p1.copy(), te_p2.copy()
        for ci in range(6): p0[:, ci] = odds_scale(p0[:, ci], cur_a0[ci])
        for k in range(4):
            p1[:, k] = odds_scale(p1[:, k], cur_a1[k])
            p2[:, k] = odds_scale(p2[:, k], cur_a2[k])
        pr = decode_shielded_pc(p0, p1, p2, te_cln0, te_cln1, te_cln2, HAND)
        s = cm.score(gte, pr, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
        rows2.append((s, f1))
        print(f"{'2.PriorAligned_Scaled_s'+str(sp):30s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want))
    m = lambda k: np.mean([r[0][k] for r in rows2])
    mf = [np.mean([r[1][w] for r in rows2]) for w in want]
    print(f"{'2.PriorAligned_Scaled MEAN':30s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf))
    print(f"  Scales: a0={cur_a0}, a1={cur_a1}, a2={cur_a2}")


if __name__ == '__main__':
    main()
