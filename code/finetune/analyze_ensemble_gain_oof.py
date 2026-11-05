"""How much does ensembling help on STRICTLY UNSEEN data? (no training, no test-set info)
Uses saved unseen-fold probabilities (te_* rows) of all single models trained on s16 / s17 splits.
Fixed decoder (HAND) for every number - nothing is tuned here.
"""
import glob, itertools, random, sys, os
import numpy as np
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune'); sys.path.insert(0, 'code/decision_eval')
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm

def load(path):
    d = np.load(path)
    names = list(d['te_names'])
    return names, d['te_p0'], d['te_p1'], d['te_p2'], d['gte']

def total(g, p, decoder=HAND):
    return cm.score(g, decode_pc(p[0], p[1], p[2], decoder), TAU)

for sp in (16, 17):
    files = sorted(glob.glob(f'results/v3_ablate/*_s{sp}_probs.npz')) + \
            sorted(glob.glob(f'results/v2/v2_*_s{sp}_seed1_probs.npz'))
    M = {}
    ref = None
    for f in files:
        n, p0, p1, p2, g = load(f)
        order = np.argsort(n); n = [n[i] for i in order]
        if ref is None: ref = n
        if n != ref:
            print('skip (rows differ):', f); continue
        M[os.path.basename(f).replace('_probs.npz', '')] = ([p0[order], p1[order], p2[order]], g[order])
    names = list(M); G = M[names[0]][1]
    singles = {k: total(G, M[k][0]) for k in names}
    print(f'\n=== split s{sp}: {len(names)} aligned single models, {len(G)} unseen images ===')
    for k, s in singles.items():
        print('  %-26s total=%.2f S_pos=%.3f p0R=%.3f p0P=%.3f clean=%.2f' % (k, s['total'], s['S_pos'], s['p0R'], s['p0P'], s['S_clean']))
    mean_single = np.mean([s['total'] for s in singles.values()])
    avg = [np.mean([M[k][0][i] for k in names], 0) for i in range(3)]
    se = total(G, avg)
    print('  mean single = %.2f | ensemble of all %d = %.2f (S_pos=%.3f p0R=%.3f p0P=%.3f clean=%.2f)' %
          (mean_single, len(names), se['total'], se['S_pos'], se['p0R'], se['p0P'], se['S_clean']))
    rng = random.Random(0)
    for k in (1, 2, 3, 4, 6):
        if k > len(names): continue
        combos = list(itertools.combinations(names, k)); rng.shuffle(combos)
        vals = []
        for c in combos[:40]:
            p = [np.mean([M[x][0][i] for x in c], 0) for i in range(3)]
            vals.append(total(G, p)['total'])
        print('  k=%d models: mean %.2f  (min %.2f max %.2f, %d subsets)' % (k, np.mean(vals), np.min(vals), np.max(vals), len(vals)))
