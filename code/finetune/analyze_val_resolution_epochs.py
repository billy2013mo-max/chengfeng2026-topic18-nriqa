"""Analyze v6_val: resolution/tiles effect, epoch effect, seed-noise floor. Fixed decoder, unseen rows only."""
import os, sys
import numpy as np
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune'); sys.path.insert(0, 'code/decision_eval')
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm

R = 'results/v6_val'
def tot(g, p0, p1, p2):
    s = cm.score(g, decode_pc(p0, p1, p2, HAND), TAU); return s
def get(name):
    f = f'{R}/{name}_epochs.npz'
    return np.load(f) if os.path.exists(f) else None

rows = {}
print('%-22s | %-14s | %s' % ('run', 'metric', 'ep2    ep3    ep4    avg(3,4)'))
for tag in ['pcl_384t2_seed2', 'pcl_448t3_seed1', 'd3_384t2_seed1', 'd3_448t3_seed1']:
    for sp in (16, 17):
        d = get(f'{tag}_s{sp}')
        if d is None or 'ep4_p0' not in d.files: continue
        g = d['gte']; out = {}
        for e in (2, 3, 4):
            s = tot(g, d[f'ep{e}_p0'], d[f'ep{e}_p1'], d[f'ep{e}_p2']); out[e] = s
        avg = [(d[f'ep3_p{i}'] + d[f'ep4_p{i}']) / 2 for i in range(3)]
        out['avg'] = tot(g, avg[0], avg[1], avg[2])
        rows[(tag, sp)] = out
        for m in ('total', 'S_pos', 'p0R', 'p0P'):
            print('%-22s | %-14s | %s' % (f'{tag}_s{sp}', m, '  '.join('%6.3f' % out[k][m] for k in (2, 3, 4, 'avg'))))

print('\n--- summary over folds (mean total) ---')
for tag in ['pcl_384t2_seed2', 'pcl_448t3_seed1', 'd3_384t2_seed1', 'd3_448t3_seed1']:
    v = [rows[(tag, sp)] for sp in (16, 17) if (tag, sp) in rows]
    if v: print('%-18s n=%d  ep3=%.2f ep4=%.2f avg34=%.2f' % (tag, len(v), np.mean([x[3]['total'] for x in v]), np.mean([x[4]['total'] for x in v]), np.mean([x['avg']['total'] for x in v])))
print('reference A_base (pcl 384t2 seed1, best epoch): s16=50.20 s17=50.20')
