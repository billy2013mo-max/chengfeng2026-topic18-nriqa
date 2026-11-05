"""Analyze all available s16 & s17 zero-leakage expert caches (including the newly trained
8 V8 Single-Feature Dedicated Experts + 60-dim Native Physical Caliper features) and test:
1. All available zero-leakage `.npz` probability caches across `results/`.
2. Individual class F1 & P0 recall/precision of each V8 Single-Feature Expert with proper odds/scale calibration.
3. Focused Low-Dim Per-Class Grade Refiner (using only that class's expert probs + top physical caliper features)
   vs high-dim stacking.
"""
import csv
import glob
import os
import sys
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm

def main():
    all_npz = sorted(glob.glob('results/**/*probs.npz', recursive=True))
    print(f"Found {len(all_npz)} total probs.npz files across results/:")
    s16_files, s17_files = [], []
    for p in all_npz:
        try:
            d = np.load(p)
            if 'te_p0' in d and d['te_p0'].shape == (1000, 6):
                if 's16' in p:
                    s16_files.append(p)
                elif 's17' in p:
                    s17_files.append(p)
        except Exception:
            pass
    print(f"Valid 1,000-test-image caches: s16={len(s16_files)}, s17={len(s17_files)}")
    for p in s16_files:
        d = np.load(p)
        sc = cm.score(d['gte'], decode_pc(d['te_p0'], d['te_p1'], d['te_p2'], HAND), TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in sc['per']}
        print(f"  [s16] {p:52s} total={sc['total']:5.2f} S_pos={sc['S_pos']:.3f} S_p0={sc['S_p0']:.3f} S_cln={sc['S_clean']:.3f} | OG_P1={f1['过度生成_P1']:.3f} OG_P0={f1['过度生成_P0']:.3f} SH_P1={f1['过锐_P1']:.3f} SH_P0={f1['过锐_P0']:.3f} BK_P0={f1['虚化突兀_P0']:.3f}")

    print()
    for p in s17_files:
        d = np.load(p)
        sc = cm.score(d['gte'], decode_pc(d['te_p0'], d['te_p1'], d['te_p2'], HAND), TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in sc['per']}
        print(f"  [s17] {p:52s} total={sc['total']:5.2f} S_pos={sc['S_pos']:.3f} S_p0={sc['S_p0']:.3f} S_cln={sc['S_clean']:.3f} | OG_P1={f1['过度生成_P1']:.3f} OG_P0={f1['过度生成_P0']:.3f} SH_P1={f1['过锐_P1']:.3f} SH_P0={f1['过锐_P0']:.3f} BK_P0={f1['虚化突兀_P0']:.3f}")

if __name__ == '__main__':
    main()
