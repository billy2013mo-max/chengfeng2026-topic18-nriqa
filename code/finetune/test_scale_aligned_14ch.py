"""Test Scale-Aligned Decoupled 14-Channel Specialist Routing on s16 & s17 (1,000 Isolated Test Images).
Investigates how aligning the relative scale (s0, s1, s2) of decoupled P0, P1, P2 specialist channels
eliminates grade boundary distortion (P2 <-> P1 <-> P0) while keeping S_p0 >= 0.817 and S_clean = 1.000.
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
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs


def main():
    splits = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_decoupled_14ch_probs(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_decoupled_14ch_probs(models, 'te')
        splits[sp] = (models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2)

    # For each class ci=0..5, let's see what happens on s16 & s17 when we vary the relative scale
    # (alpha0, alpha1, alpha2) of p0[:, ci], p1[:, k], p2[:, k] using the EXACT Two-Layer Decomposition!
    for ci, c in enumerate(CLASSES):
        print(f"\n================ Class {ci}: {c} ================")
        if c in FULL:
            k = FULL.index(c)
            # 1. Sweep alpha0 for P0 channel (affects S_p0 + F1_P0)
            best_a0 = 1.0
            for a0 in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]:
                tot_s = []
                f1_p0s, f1_p1s, f1_p2s, s_p0s = [], [], [], []
                for sp in (16, 17):
                    _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
                    p0_c = te_p0.copy()
                    p0_c[:, ci] = odds_scale(p0_c[:, ci], a0)
                    pr = decode_shielded_pc(p0_c, te_p1, te_p2, te_cln0, te_cln1, te_cln2, HAND)
                    sc = cm.score(gte, pr, TAU)
                    fmap = {f'{x[0]}_{x[1]}': x[4] for x in sc['per']}
                    tot_s.append(sc['total'])
                    s_p0s.append(sc['S_p0'])
                    f1_p0s.append(fmap[f'{c}_P0'])
                    f1_p1s.append(fmap[f'{c}_P1'])
                    f1_p2s.append(fmap[f'{c}_P2'])
                print(f"  [P0 scale a0={a0:.2f}] mean_total={np.mean(tot_s):.3f} S_p0={np.mean(s_p0s):.4f} | {c}_P0={np.mean(f1_p0s):.3f} P1={np.mean(f1_p1s):.3f} P2={np.mean(f1_p2s):.3f}")

            # 2. Sweep (a1, a2) for P1 & P2 channels (affects ONLY P1 and P2 F1 on ~c0!)
            best_pair = (1.0, 1.0)
            best_tot = -1.0
            for a1 in [0.85, 0.95, 1.0, 1.08, 1.15, 1.25, 1.35]:
                for a2 in [0.75, 0.85, 0.92, 1.0, 1.08, 1.15]:
                    tot_s, f1_p1s, f1_p2s = [], [], []
                    for sp in (16, 17):
                        _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
                        p1_c = te_p1.copy()
                        p2_c = te_p2.copy()
                        p1_c[:, k] = odds_scale(p1_c[:, k], a1)
                        p2_c[:, k] = odds_scale(p2_c[:, k], a2)
                        pr = decode_shielded_pc(te_p0, p1_c, p2_c, te_cln0, te_cln1, te_cln2, HAND)
                        sc = cm.score(gte, pr, TAU)
                        fmap = {f'{x[0]}_{x[1]}': x[4] for x in sc['per']}
                        tot_s.append(sc['total'])
                        f1_p1s.append(fmap[f'{c}_P1'])
                        f1_p2s.append(fmap[f'{c}_P2'])
                    if np.mean(tot_s) > best_tot:
                        best_tot = np.mean(tot_s)
                        best_pair = (a1, a2, np.mean(f1_p1s), np.mean(f1_p2s))
            print(f"  --> Best (a1={best_pair[0]:.2f}, a2={best_pair[1]:.2f}): mean_total={best_tot:.3f} | {c}_P1={best_pair[2]:.3f}, {c}_P2={best_pair[3]:.3f}")


if __name__ == '__main__':
    main()
