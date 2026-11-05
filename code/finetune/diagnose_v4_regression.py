"""Verify (not assume) why V4 submission scored below the 52.43 V3 submission.
Rebuilds both ensembles from cached TTA probs and inspects the cells that flipped.
"""
import json, sys
import numpy as np
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/finetune')
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/decision_eval')
from data import CLASSES, FULL
from run_v3_super_ensemble import decode_v3_shielded, GRADE
import os
os.chdir('/wanqing-develop/luowenjing/18')

names = [e['image'] for e in json.load(open('results/v2_full/template_150.json'))]
v2 = np.load('results/v2_super/test_150_all_models_probs.npz', allow_pickle=True)['data'].item()
v3 = np.load('results/v3_super/test_150_v3_models_probs.npz', allow_pickle=True)['data'].item()
v4 = np.load('results/v4_synth/test_150_v4_models_probs.npz', allow_pickle=True)['data'].item()


def ens(parts):
    p0 = p1 = p2 = 0.0; tw = 0.0
    for p0_i, p1_i, p2_i, w in parts:
        p0 = p0 + w * p0_i; p1 = p1 + w * p1_i; p2 = p2 + w * p2_i; tw += w
    return p0 / tw, p1 / tw, p2 / tw


famw = {'pcl': 1.3, 'd3': 1.3, 'eupe': 1.1, 'base': 0.8}
v2_parts_52 = [(a, b, c, famw.get(f, 1.0) * (0.7 if iso else 1.0)) for (a, b, c, f, iso) in v2.values()]
v2_parts_v4 = [(a, b, c, {'pcl': 1.0, 'd3': 1.0, 'eupe': 0.9, 'base': 0.7}.get(f, 1.0) * (0.6 if iso else 0.85)) for (a, b, c, f, iso) in v2.values()]
v3_parts = [(a, b, c, w) for k, (a, b, c, g, w) in v3.items() if 'cali' not in k]
v4_parts = [(a, b, c, w) for (a, b, c, g, w) in v4.values()]

E52 = ens(v2_parts_52 + v3_parts)
E_v4 = ens(v2_parts_v4 + v4_parts)
E_mix = ens(v2_parts_52 + v3_parts + [(a, b, c, w * 0.5) for (a, b, c, w) in v4_parts])

sub52 = json.load(open('results/v3_super/pred_nriqa_dataset_test.json'))
rec = decode_v3_shielded(*E52)
ok = all(GRADE[int(rec[i, CLASSES.index(k)])] == sub52[i]['label'][k] for i in range(150) for k in CLASSES)
print('rebuilt 52.43 ensemble reproduces submitted json:', ok)

pred52 = decode_v3_shielded(*E52)
pred4 = decode_v3_shielded(*E_v4)
d = np.argwhere(pred52 != pred4)
print('cells changed (V3pool -> V4pool):', len(d))
print('%-28s %-6s %5s -> %-5s | p0_52  p0_v4 | p2_52 p2_v4' % ('image', 'class', 'g52', 'g4'))
for i, c in d:
    k = FULL.index(CLASSES[c]) if CLASSES[c] in FULL else None
    p2a = E52[2][i, k] if k is not None else float('nan'); p2b = E_v4[2][i, k] if k is not None else float('nan')
    print('%-28s %-6s %5s -> %-5s | %.3f  %.3f | %.3f %.3f' % (names[i][:28], CLASSES[c], GRADE[pred52[i, c]], GRADE[pred4[i, c]], E52[0][i, c], E_v4[0][i, c], p2a, p2b))

# how far apart are the two pools overall?
for nm, E in [('V3pool', E52), ('V4pool', E_v4)]:
    print(nm, 'mean p0 per class', np.round(E[0].mean(0), 3), 'mean p2(FULL)', np.round(E[2].mean(0), 3))
print('share of V4 models in V4 pool weight: %.2f' % (sum(w for *_, w in v4_parts) / (sum(w for *_, w in v4_parts) + sum(w for *_, w in v2_parts_v4))))
print('is v3_d3_448_t2_ep3 in V4 pool:', 'v3_d3_448_t2_ep3' in v4)
