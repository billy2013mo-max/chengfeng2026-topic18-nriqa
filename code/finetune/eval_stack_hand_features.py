"""Does a physics-based feature stack add real, held-out signal on top of the deep-model probabilities?
Leave-one-split-out over s16..s20 (group-disjoint by source). OOF model probs: v2_pcl_s16..s20.
Decoder is fixed (HAND). Reports held-out mean total and per-label deltas for:
  base      : model probs decoded directly
  S_probs   : stacker on model probs only (checks stacking itself is not the source of change)
  S_hand    : stacker on hand features only
  S_both    : stacker on model probs + hand features
"""
import os, sys
import numpy as np
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune'); sys.path.insert(0, 'code/decision_eval')
from sklearn.ensemble import HistGradientBoostingClassifier
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from data import CLASSES, FULL

H = np.load('results/hand_features.npz', allow_pickle=True)
hname = {n: i for i, n in enumerate(H['names'])}; HX = H['X']

def split(sp):
    d = np.load(f'results/v2/v2_pcl_s{sp}_seed1_probs.npz', allow_pickle=True)
    names = list(d['va_names']) + list(d['te_names'])
    P = [np.concatenate([d['va_' + k], d['te_' + k]]) for k in ('p0', 'p1', 'p2')]
    G = np.concatenate([d['gva'], d['gte']])
    keep = [i for i, n in enumerate(names) if n in hname]
    return [names[i] for i in keep], [p[keep] for p in P], G[keep]

import csv
SRC = {}
for _sp in range(16, 21):
    for _r in csv.DictReader(open(f'data/resplits/split_s{_sp}.csv')): SRC[_r['name']] = _r['source']
S = {sp: split(sp) for sp in range(16, 21)}
print({sp: len(v[0]) for sp, v in S.items()})

def design(sp, use_p, use_h, exclude_sources=None):
    names, P, G = S[sp]
    if exclude_sources is not None:
        k = [i for i, n in enumerate(names) if SRC[n] not in exclude_sources]
        names = [names[i] for i in k]; P = [p[k] for p in P]; G = G[k]
    cols = []
    if use_p: cols.append(np.concatenate(P, 1))
    if use_h: cols.append(HX[[hname[n] for n in names]])
    return np.concatenate(cols, 1), G

def stack_predict(Xtr, Gtr, Xte, seed=0):
    p0 = np.zeros((len(Xte), 6)); p1 = np.zeros((len(Xte), 4)); p2 = np.zeros((len(Xte), 4))
    for ci, c in enumerate(CLASSES):
        y = Gtr[:, ci].astype(int)
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120, l2_regularization=1.0,
                                             min_samples_leaf=30, random_state=seed)
        clf.fit(Xtr, y)
        pr = np.zeros((len(Xte), 4)); pr[:, clf.classes_] = clf.predict_proba(Xte)
        p0[:, ci] = pr[:, 3]
        if c in FULL:
            k = FULL.index(c); p2[:, k] = pr[:, 1]; p1[:, k] = pr[:, 2]
    return p0, p1, p2

res = {k: [] for k in ('base', 'S_probs', 'S_hand', 'S_both')}
lab = {k: [] for k in res}
for ho in range(16, 21):
    tr = [s for s in range(16, 21) if s != ho]
    _, Pte, Gte = S[ho]
    s = cm.score(Gte, decode_pc(*Pte, HAND), TAU); res['base'].append(s['total']); lab['base'].append({f'{x[0]}_{x[1]}': x[4] for x in s['per']})
    for key, (up, uh) in {'S_probs': (1, 0), 'S_hand': (0, 1), 'S_both': (1, 1)}.items():
        hs = {SRC[n] for n in S[ho][0]}   # every source that appears in the held-out rows is removed from stacker training
        Xtr = np.concatenate([design(t, up, uh, hs)[0] for t in tr]); Gtr = np.concatenate([design(t, up, uh, hs)[1] for t in tr])
        Xte, _ = design(ho, up, uh)
        p = stack_predict(Xtr, Gtr, Xte)
        s = cm.score(Gte, decode_pc(*p, HAND), TAU); res[key].append(s['total']); lab[key].append({f'{x[0]}_{x[1]}': x[4] for x in s['per']})
    print(f'held-out s{ho}: ' + '  '.join(f'{k}={v[-1]:.2f}' for k, v in res.items()), flush=True)
print('\nMEAN held-out total: ' + '  '.join(f'{k}={np.mean(v):.3f}' for k, v in res.items()))
b = np.array(res['base'])
for k in ('S_probs', 'S_hand', 'S_both'):
    d = np.array(res[k]) - b
    print(f'{k} - base: mean {d.mean():+.3f}  per-fold {np.round(d, 2)}  (folds improved: {(d > 0).sum()}/5)')
print('\nper-label F1 (mean over folds): base -> S_both')
for key in lab['base'][0]:
    a = np.mean([l[key] for l in lab['base']]); c = np.mean([l[key] for l in lab['S_both']]); print(f'  {key:12s} {a:.3f} -> {c:.3f}  ({c - a:+.3f})')
