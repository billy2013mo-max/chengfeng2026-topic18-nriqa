import os, sys, glob, json
import numpy as np
from sklearn.metrics import average_precision_score

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models

splits = {}
for sp in (16, 17):
    models, gva, gte = load_healthy_split_models(sp)
    splits[sp] = (models, gva, gte)

common_keys = sorted(set(splits[16][0].keys()) & set(splits[17][0].keys()))
print(f"Total common models in s16 & s17: {len(common_keys)}")

# For each of the 14 channels, compute mean Test AP and best F1 across s16 & s17 for every model family!
LABELS14 = []
for ci, c in enumerate(CLASSES):
    if c in FULL:
        k = FULL.index(c)
        LABELS14.append((f'{c}_P2', 'p2', k, ci, 1))
        LABELS14.append((f'{c}_P1', 'p1', k, ci, 2))
        LABELS14.append((f'{c}_P0', 'p0', ci, ci, 3))
    else:
        LABELS14.append((f'{c}_P0', 'p0', ci, ci, 3))

print("\n=== Top-8 Models per Channel by Mean AP on s16 & s17 (Val vs Test) ===")
for lname, arr_k, col_idx, ci, grade_val in LABELS14:
    scores = []
    for m in common_keys:
        ap_va, ap_te, f1_te = [], [], []
        for sp in (16, 17):
            models, gva, gte = splits[sp]
            y_va = (gva[:, ci] == grade_val).astype(int)
            y_te = (gte[:, ci] == grade_val).astype(int)
            p_va = models[m][f'va_{arr_k}'][:, col_idx]
            p_te = models[m][f'te_{arr_k}'][:, col_idx]
            ap_va.append(average_precision_score(y_va, p_va))
            ap_te.append(average_precision_score(y_te, p_te))
            best_f = max(2*np.sum((p_te>t)&y_te)/(np.sum(p_te>t)+np.sum(y_te)+1e-8) for t in np.linspace(0.15, 0.65, 26))
            f1_te.append(best_f)
        scores.append((np.mean(ap_va), np.mean(ap_te), np.mean(f1_te), m))
    scores.sort(reverse=True)
    top_str = " | ".join([f"{m.split('/')[-1]}: vaAP={va:.3f}/teAP={te:.3f}/F1={f1:.3f}" for va, te, f1, m in scores[:5]])
    print(f"  {lname:10s}: {top_str}")
