"""Competition metrics (doc appendix 2), vectorised."""
import argparse, json, collections, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import CLASSES, FULL

LET = 'ABCD'; FULL_IDX = [i for i,c in enumerate(CLASSES) if c in FULL]
BIN_IDX = [i for i,c in enumerate(CLASSES) if c not in FULL]
LABELS = [(i, c, g) for i, c in enumerate(CLASSES) for g in (['P2','P1','P0'] if c in FULL else ['P0'])]

def load_jsonl(path):
    by = collections.defaultdict(dict)
    for l in open(path):
        r = json.loads(l); by[r['image']][r['defect']] = r
    imgs = sorted(i for i, d in by.items() if len(d) == 6)
    n = len(imgs)
    gold = np.zeros((n, 6), int); p0 = np.zeros((n, 6)); p1 = np.zeros((n, 4)); p2 = np.zeros((n, 4))
    G2I = {'无问题':0,'P2':1,'P1':2,'P0':3}
    for k, im in enumerate(imgs):
        for ci, c in enumerate(CLASSES):
            r = by[im][c]; o = r['options']
            gold[k, ci] = G2I[r['gold']]
            def gp(g):
                return float(r['probs'].get(LET[o.index(g)], 0.0)) if g in o else 0.0
            p0[k, ci] = gp('P0')
            if c in FULL:
                j = FULL_IDX.index(ci); p1[k, j] = gp('P1'); p2[k, j] = gp('P2')
    return n, gold, p0, p1, p2

def load_probe(path):
    d = np.load(path, allow_pickle=True); P, names, grades = d['P'], list(d['names']), d['grades']
    n = len(grades); p0 = np.zeros((n,6)); p1 = np.zeros((n,4)); p2 = np.zeros((n,4))
    for ci, c in enumerate(CLASSES):
        i0 = names.index(f'{c}_P0'); p0[:, ci] = P[:, i0]
        if c in FULL:
            i2, i1 = names.index(f'{c}_P2'), names.index(f'{c}_P1')
            j = FULL_IDX.index(ci)
            p2[:, j] = np.clip(P[:,i2]-P[:,i1], 0, None); p1[:, j] = np.clip(P[:,i1]-P[:,i0], 0, None)
    return n, grades.astype(int), p0, p1, p2

def decode(p0, p1, p2, t0, t1, t2):
    n = p0.shape[0]; pred = np.zeros((n,6), int)
    pred[:, FULL_IDX] = np.where(p2 > t2, 1, 0)
    pred[:, FULL_IDX] = np.where(p1 > t1, 2, pred[:, FULL_IDX])
    pred[:, FULL_IDX] = np.where(p0[:, FULL_IDX] > t0, 3, pred[:, FULL_IDX])
    pred[:, BIN_IDX] = np.where(p0[:, BIN_IDX] > t0, 3, 0)
    return pred

def argmax_decode(p0, p1, p2):
    n = p0.shape[0]; pred = np.zeros((n,6), int)
    for j, ci in enumerate(FULL_IDX):
        pn = np.clip(1 - p0[:,ci] - p1[:,j] - p2[:,j], 0, None)
        M = np.stack([pn, p2[:,j], p1[:,j], p0[:,ci]], 1)
        pred[:, ci] = M.argmax(1)
    for ci in BIN_IDX:
        pred[:, ci] = np.where(p0[:,ci] > 0.5, 3, 0)
    return pred

def score(gold, pred, tau):
    per = []
    for ci, c, g in LABELS:
        gi = {'P2':1,'P1':2,'P0':3}[g]
        yt = (gold[:,ci] == gi); yp = (pred[:,ci] == gi)
        tp = int((yt & yp).sum()); fp = int((~yt & yp).sum()); fn = int((yt & ~yp).sum())
        P = tp/max(1,tp+fp); R = tp/max(1,tp+fn); F = 2*tp/max(1,2*tp+fp+fn)
        per.append((c, g, P, R, F, tp, fp, fn))
    S_pos = float(np.mean([x[4] for x in per]))
    p0s = [x for x in per if x[1]=='P0']
    r = float(np.mean([x[3] for x in p0s])); p = float(np.mean([x[2] for x in p0s])); f = float(np.mean([x[4] for x in p0s]))
    S_p0 = 0.5*r + 0.3*f + 0.2*p
    clean = (gold == 0).all(1)
    flagged = ((pred == 3).any(1) & clean)
    fpr = float(flagged.sum()/max(1, clean.sum()))
    S_clean = max(0.0, 1 - fpr/tau)
    return dict(per=per, S_pos=S_pos, S_p0=S_p0, S_clean=S_clean, p0R=r, p0P=p, p0F=f,
                fpr=fpr, total=25*S_pos+20*S_p0+20*S_clean)

def optimise(gold, p0, p1, p2, tau):
    best = None
    for t0 in np.arange(0.05, 0.96, 0.05):
        for t1 in np.arange(0.10, 0.91, 0.10):
            for t2 in np.arange(0.10, 0.91, 0.10):
                s = score(gold, decode(p0,p1,p2,t0,t1,t2), tau)
                if best is None or s['total'] > best[0]['total']:
                    best = (s, (t0,t1,t2))
    return best

def row(name, s):
    return (f"| {name} | {s['S_pos']:.3f} | {s['p0R']:.3f} | {s['p0P']:.3f} | {s['p0F']:.3f} | "
            f"{s['fpr']:.3f} | {s['S_p0']:.3f} | {s['S_clean']:.3f} | **{s['total']:.1f}** |")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tau', type=float, default=0.02)
    ap.add_argument('--detail', action='store_true'); a = ap.parse_args()
    runs = [('qwen3.8-27b','results/qwen38-27b.jsonl'), ('jev-27b-vl(原生)','results/jev-native.jsonl'),
            ('clef-27b','results/clef.jsonl'), ('clef-flash','results/clef-flash.jsonl'),
            ('qwen3-vl-2b','results/qwen3vl-2b.jsonl'), ('jev-gemma4(600子集)','results/jev-gemma_sub600.jsonl')]
    data = []
    for n, f in runs:
        if os.path.exists(f) and sum(1 for _ in open(f)) >= 600:
            data.append((n, *load_jsonl(f)))
    data.append(('siglip2-so400m(监督)', *load_probe('feats/siglip2-so400m_oof.npz')))
    hdr = '| 模型 | S_pos(F1宏均) | P0召回 | P0精确 | P0-F1 | P0误报率 | S_p0 | S_clean | 客观总分/65 |'
    sep = '|---|---|---|---|---|---|---|---|---|'
    print(f'\n## A. 不调阈值（argmax）\n\n{hdr}\n{sep}')
    for n, N, g, q0, q1, q2 in data:
        print(row(n, score(g, argmax_decode(q0,q1,q2), a.tau)))
    print(f'\n## B. 阈值调优（最大化客观总分，tau={a.tau:.0%}）\n\n{hdr}\n{sep}')
    bests = {}
    for n, N, g, q0, q1, q2 in data:
        s, th = optimise(g, q0, q1, q2, a.tau); bests[n] = s
        print(row(n, s) + f'  <!-- t={th} -->')
    if a.detail:
        for n, s in bests.items():
            print(f'\n### {n}\n\n| 类别 | P | R | F1 | TP | FP | FN |\n|---|---|---|---|---|---|---|')
            for c, g, P, R, F, tp, fp, fn in s['per']:
                print(f'| {c}_{g} | {P:.3f} | {R:.3f} | {F:.3f} | {tp} | {fp} | {fn} |')
