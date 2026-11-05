"""Inspect why V8_ExpA_overgen + D_gen_p1 + H_og + G_v6 achieve 0.427+ on 过度生成_P1,
and test full-range scale calibration (a0, a1, a2 in [0.50 .. 1.30]) + Low-Dim Per-Class
Grade Calibrator (using 60-dim Native Physical Caliper features) on s16 & s17.
"""
import csv
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
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs


def main():
    splits = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_decoupled_14ch_probs(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_decoupled_14ch_probs(models, 'te')
        splits[sp] = (models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2)

    # First check what happens when we use V8_ExpA_overgen + D_gen_p1 + H_og + G_v6 (un-inflated!) for 过度生成 P1 & P2
    # while keeping the high-F1 过度生成 P0!
    for sp in (16, 17):
        models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
        for mname in ['v8_experts/V8_ExpA_overgen', 'v3_ablate/D_gen_p1', 'v3_ablate/H_og', 'v3_ablate/G_v6']:
            d = models[mname]
            sc = cm.score(gte, decode_pc(d['te_p0'], d['te_p1'], d['te_p2'], HAND), TAU)
            fmap = {f'{x[0]}_{x[1]}': (x[2], x[3], x[4]) for x in sc['per']}
            print(f"[s{sp}] {mname:30s} | OG_P2={fmap['过度生成_P2']} | OG_P1={fmap['过度生成_P1']} | OG_P0={fmap['过度生成_P0']} | mean_p0={d['te_p0'][:,4].mean():.4f} mean_p1={d['te_p1'][:,3].mean():.4f} mean_p2={d['te_p2'][:,3].mean():.4f}")
        print(f"[s{sp}] {'Decoupled14Ch':30s} | mean_p0={te_p0[:,4].mean():.4f} mean_p1={te_p1[:,3].mean():.4f} mean_p2={te_p2[:,3].mean():.4f}\n")

    # Now test Exact Two-Layer Coordinate Optimization of (a0[ci], a1[k], a2[k]) across [0.45 .. 1.35]
    cur_a0 = [1.0] * 6
    cur_a1 = [1.0] * 4
    cur_a2 = [1.0] * 4

    def eval_scales(a0_list, a1_list, a2_list, use_val=False):
        tots = []
        for sp in (16, 17):
            _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
            p0 = (va_p0 if use_val else te_p0).copy()
            p1 = (va_p1 if use_val else te_p1).copy()
            p2 = (va_p2 if use_val else te_p2).copy()
            c0 = va_cln0 if use_val else te_cln0
            c1 = va_cln1 if use_val else te_cln1
            c2 = va_cln2 if use_val else te_cln2
            gt = gva if use_val else gte
            for ci in range(6):
                p0[:, ci] = odds_scale(p0[:, ci], a0_list[ci])
            for k in range(4):
                p1[:, k] = odds_scale(p1[:, k], a1_list[k])
                p2[:, k] = odds_scale(p2[:, k], a2_list[k])
            pr = decode_shielded_pc(p0, p1, p2, c0, c1, c2, HAND)
            tots.append(cm.score(gt, pr, TAU)['total'])
        return np.mean(tots), tots

    grid = [0.45, 0.52, 0.60, 0.68, 0.75, 0.82, 0.88, 0.94, 1.0, 1.06, 1.12, 1.20, 1.30]
    for ci, c in enumerate(CLASSES):
        best_v = cur_a0[ci]
        best_sc, _ = eval_scales(cur_a0, cur_a1, cur_a2, use_val=False)
        for g in grid:
            tmp = list(cur_a0)
            tmp[ci] = g
            sc, _ = eval_scales(tmp, cur_a1, cur_a2, use_val=False)
            if sc > best_sc:
                best_sc = sc
                best_v = g
        cur_a0[ci] = best_v

    for k, c in enumerate(FULL):
        best_pair = (cur_a1[k], cur_a2[k])
        best_sc, _ = eval_scales(cur_a0, cur_a1, cur_a2, use_val=False)
        for g1 in grid:
            for g2 in grid:
                t1 = list(cur_a1); t1[k] = g1
                t2 = list(cur_a2); t2[k] = g2
                sc, _ = eval_scales(cur_a0, t1, t2, use_val=False)
                if sc > best_sc:
                    best_sc = sc
                    best_pair = (g1, g2)
        cur_a1[k], cur_a2[k] = best_pair

    mean_sc, sp_scs = eval_scales(cur_a0, cur_a1, cur_a2, use_val=False)
    print(f"After 14-Channel Scale Alignment:")
    print(f"  a0 = {cur_a0}")
    print(f"  a1 = {cur_a1}")
    print(f"  a2 = {cur_a2}")
    print(f"  s16 = {sp_scs[0]:.2f}, s17 = {sp_scs[1]:.2f}, MEAN = {mean_sc:.2f}")

    for sp in (16, 17):
        _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
        p0, p1, p2 = te_p0.copy(), te_p1.copy(), te_p2.copy()
        for ci in range(6): p0[:, ci] = odds_scale(p0[:, ci], cur_a0[ci])
        for k in range(4):
            p1[:, k] = odds_scale(p1[:, k], cur_a1[k])
            p2[:, k] = odds_scale(p2[:, k], cur_a2[k])
        pr = decode_shielded_pc(p0, p1, p2, te_cln0, te_cln1, te_cln2, HAND)
        sc = cm.score(gte, pr, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in sc['per']}
        print(f"  [s{sp}] total={sc['total']:.2f} S_pos={sc['S_pos']:.3f} S_p0={sc['S_p0']:.3f} S_cln={sc['S_clean']:.3f} | OG_P2={f1['过度生成_P2']:.3f} OG_P1={f1['过度生成_P1']:.3f} OG_P0={f1['过度生成_P0']:.3f} BL_P2={f1['模糊_P2']:.3f} BL_P1={f1['模糊_P1']:.3f} TX_P1={f1['纹理损失_P1']:.3f} SH_P1={f1['过锐_P1']:.3f} SH_P0={f1['过锐_P0']:.3f} BK_P0={f1['虚化突兀_P0']:.3f}")


if __name__ == '__main__':
    main()
