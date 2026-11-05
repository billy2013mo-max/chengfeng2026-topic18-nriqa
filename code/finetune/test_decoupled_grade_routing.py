"""Decoupled 14-Channel (Class x Grade) Specialist Routing + Two-Stage Presence/Grade Calibration
on the 1,000-Image Isolated Test Sets (s16 & s17).

Key Insight:
Within the same defect class (e.g. `过度生成`, `过锐`, `模糊`, `纹理损失`), the best expert for
`P0` (severe defect recall/precision) is DIFFERENT from the best expert for `P1` and `P2`
(fine-grained borderline severity discrimination)!
- `P0` channels thrive on 2B-param `PE-Core-G` (`pcg_seed1`, `pcgadd_seed1`), `PE-Core-L` (`pcladd_seed1`, `v2_pcl`),
  `EUPE` (`v2_eupe`), `448x448` (`pcl_448t3`), and `V8_ExpD_bokeh` / `E_synth` / `G_v6`.
- `P1` and `P2` channels thrive on `V8_ExpA_overgen`, `V8_ExpB_sharp`, `V8_ExpC_blurtex`, `H_og`, `G_v6`,
  `D_gen_p1`, `C_p1`, `F_v5`, and `448x448` (`pcl_448t3`, `d3_448t3`).

Here we rank all 36 zero-leakage model families per individual (class, grade) channel on `Val` (s16+s17 val),
build a decoupled 14-channel specialist ensemble, and test Two-Stage Conditional Grade Calibration.
"""
import csv
import glob
import os
import sys
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale, decode_shielded_pc


def main():
    splits_data = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        splits_data[sp] = (models, gva, gte)

    common_keys = sorted(set(splits_data[16][0].keys()) & set(splits_data[17][0].keys()))
    print(f"Loaded {len(common_keys)} common zero-leakage model families across s16 & s17.")

    # 1. Rank all 36 models on each of the 14 individual (class, grade) channels by Val F1 and Val AP!
    label_names = [f"{x[0]}_{x[1]}" for x in cm.score(splits_data[16][2], np.zeros_like(splits_data[16][2]), TAU)['per']]
    print("\n=== Top-8 Specialists per Individual (Class, Grade) Channel (Ranked by Val F1) ===")
    top_per_label = {}
    for li, lname in enumerate(label_names):
        ranks = []
        for k in common_keys:
            va_f1s, te_f1s = [], []
            for sp in (16, 17):
                models, gva, gte = splits_data[sp]
                d = models[k]
                sc_va = cm.score(gva, decode_pc(d['va_p0'], d['va_p1'], d['va_p2'], HAND), TAU)
                sc_te = cm.score(gte, decode_pc(d['te_p0'], d['te_p1'], d['te_p2'], HAND), TAU)
                va_f1s.append(sc_va['per'][li][4])
                te_f1s.append(sc_te['per'][li][4])
            ranks.append((np.mean(va_f1s), np.mean(te_f1s), k))
        ranks.sort(reverse=True)
        top_per_label[lname] = ranks
        print(f"\n[{lname}]")
        for vf, tf, k in ranks[:7]:
            print(f"  {k:36s} | Val_F1={vf:.3f} | Test_F1={tf:.3f}")


if __name__ == '__main__':
    main()
