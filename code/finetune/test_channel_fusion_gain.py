import sys
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')
import numpy as np
import csv
from utils import competition_score, FULL_DIMS, P0_ONLY_DIMS

d16_f0 = np.load('results/v16_fulldata/V16_Full_F0_AllGrade_Microscope_probs.npz', allow_pickle=True)
p0_v16 = d16_f0['va_p0']
p1_v16 = d16_f0['va_p1']
p2_v16 = d16_f0['va_p2']

man_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
name_to_idx = {r['name']: i for i, r in enumerate(man_rows)}
y_6d = np.array([[int(r[c]) for c in ['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']] for r in man_rows])
va_idx = [name_to_idx[n] for n in d16_f0['va_names']]
y_va = y_6d[va_idx]

# Baseline prediction
pred_base = np.zeros((len(p0_v16), 6), dtype=int)
for c in range(6):
    if c in FULL_DIMS:
        j = FULL_DIMS.index(c)
        p_none = np.clip(1.0 - p0_v16[:, c] - p1_v16[:, j] - p2_v16[:, j], 0, 1)
        p_mat = np.stack([p_none, p2_v16[:, j], p1_v16[:, j], p0_v16[:, c]], axis=-1)
        pred_base[:, c] = np.argmax(p_mat, axis=-1)
    else:
        pred_base[:, c] = np.where(p0_v16[:, c] >= 0.50, 3, 0)
sc_base = competition_score(pred_base, y_va)

# Simulate replacing only the improved channel (TexLoss + OverGen) with improved predictions
pred_improved = pred_base.copy()
# From the verification, TexLoss_P1 F1 went from 0.551 -> 0.605 (+5.4%)
# Let us inspect how competition score behaves when substituting improved channels
print(f"Base Score: {sc_base['total']:.3f} | S_pos: {sc_base['S_pos']:.3f} | S_p0: {sc_base['S_p0']:.3f}")
