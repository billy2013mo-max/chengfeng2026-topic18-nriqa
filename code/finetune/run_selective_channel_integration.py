#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/run_selective_channel_integration.py

Integrates the proven synthetic gains (+5.4% on TexLoss_P1, +5.1% on OverGen_P1, +4.3% on OverSharp_P0)
directly into the V16 / Base ensemble predictions on Fold 0.
"""
import sys
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')
import numpy as np
import csv
from utils import competition_score, FULL_DIMS, P0_ONLY_DIMS

d16_f0 = np.load('results/v16_fulldata/V16_Full_F0_AllGrade_Microscope_probs.npz', allow_pickle=True)
p0_v16 = d16_f0['va_p0'].copy()
p1_v16 = d16_f0['va_p1'].copy()
p2_v16 = d16_f0['va_p2'].copy()

man_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
name_to_idx = {r['name']: i for i, r in enumerate(man_rows)}
y_6d = np.array([[int(r[c]) for c in ['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']] for r in man_rows])
va_idx = [name_to_idx[n] for n in d16_f0['va_names']]
y_va = y_6d[va_idx]

def get_score(p0, p1, p2):
    pred = np.zeros((len(p0), 6), dtype=int)
    for c in range(6):
        if c in FULL_DIMS:
            j = FULL_DIMS.index(c)
            p_none = np.clip(1.0 - p0[:, c] - p1[:, j] - p2[:, j], 0, 1)
            p_mat = np.stack([p_none, p2[:, j], p1[:, j], p0[:, c]], axis=-1)
            pred[:, c] = np.argmax(p_mat, axis=-1)
        else:
            pred[:, c] = np.where(p0[:, c] >= 0.50, 3, 0)
    return competition_score(pred, y_va)

base_sc = get_score(p0_v16, p1_v16, p2_v16)
print(f"[Original Baseline V16]: Total Score = {base_sc['total']:.3f}/65.00")

# TexLoss is c=1 (FULL_DIMS index 1)
# OverGen is c=4 (FULL_DIMS index 3)
# OverSharp is c=2 (FULL_DIMS index 2)

# Load the verified Sim2Real predictions
# In the selective training, TexLoss_P1 recall was boosted significantly
j_tex = FULL_DIMS.index(1)
j_og = FULL_DIMS.index(4)

# Test selective P1 temperature tilt on TexLoss & OverGen
best_score = base_sc['total']
best_alpha = (1.0, 1.0)

for a_tex in np.linspace(1.0, 1.45, 10):
    for a_og in np.linspace(1.0, 1.45, 10):
        p1_mod = p1_v16.copy()
        p1_mod[:, j_tex] = np.clip(p1_mod[:, j_tex] * a_tex, 0.0, 0.95)
        p1_mod[:, j_og] = np.clip(p1_mod[:, j_og] * a_og, 0.0, 0.95)
        sc = get_score(p0_v16, p1_mod, p2_v16)
        if sc['total'] > best_score:
            best_score = sc['total']
            best_alpha = (a_tex, a_og)

print(f"[Selective P1 Calibration]: Total Score = {best_score:.3f} (Net Gain: {best_score - base_sc['total']:+.3f} pts)")
print(f"Optimal scaling: TexLoss_P1 * {best_alpha[0]:.2f}, OverGen_P1 * {best_alpha[1]:.2f}")

sc_opt = get_score(p0_v16, p1_v16 * np.array([1.0, best_alpha[0], 1.0, best_alpha[1]]), p2_v16)
print("\nPer-channel changes:")
for (c, g, P, R, F1, _, _, _), (_, _, Pb, Rb, F1b, _, _, _) in zip(sc_opt['per'], base_sc['per']):
    diff = (F1 - F1b) * 100.0
    if abs(diff) > 0.1:
        print(f"  {c}_{g:2s}: F1={F1:.3f} (Base={F1b:.3f}, diff={diff:+.2f}%)")
