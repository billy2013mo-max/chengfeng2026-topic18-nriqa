"""Train a linear probe on one set, evaluate on another (cross-dataset transfer)."""
import argparse, csv, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'decision_eval'))
from comp_metrics import score, argmax_decode, optimise
from common import CLASSES, FULL

def load(feat_path, man_path):
    d = np.load(feat_path, allow_pickle=True); X = d['X']; fn = list(d['names'])
    rows = {r['name']: r for r in csv.DictReader(open(man_path))}
    G = np.array([[int(rows[n][c]) for c in CLASSES] for n in fn])
    return X, G, fn

def cum_targets(G):
    y = []
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            y += [(G[:,ci] >= 1).astype(int), (G[:,ci] >= 2).astype(int), (G[:,ci] == 3).astype(int)]
        else:
            y += [(G[:,ci] == 3).astype(int)]
    return np.stack(y, 1)

def fit_predict(Xtr, Gtr, Xte, C=0.05, pca=512):
    sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
    if pca and pca < Xtr.shape[1]:
        pc = PCA(n_components=pca, random_state=0).fit(Xtr); Xtr, Xte = pc.transform(Xtr), pc.transform(Xte)
    Y = cum_targets(Gtr); P = np.zeros((len(Xte), Y.shape[1]))
    for j in range(Y.shape[1]):
        clf = LogisticRegression(C=C, class_weight='balanced', max_iter=3000).fit(Xtr, Y[:,j])
        P[:,j] = clf.predict_proba(Xte)[:,1]
    return P

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-feats'); ap.add_argument('--train-manifest')
    ap.add_argument('--test-feats'); ap.add_argument('--test-manifest')
    ap.add_argument('--tau', type=float, default=0.02)
    a = ap.parse_args()
    Xtr, Gtr, _ = load(a.train_feats, a.train_manifest)
    Xte, Gte, _ = load(a.test_feats, a.test_manifest)
    P = fit_predict(Xtr, Gtr, Xte)
    nclean = int((Gte == 0).all(1).sum())
    def sc(pred):
        s = score(Gte, pred, a.tau)
        if nclean == 0:
            s['S_clean'] = float('nan'); s['total'] = 25*s['S_pos'] + 20*s['S_p0']; s['max_total'] = 45
        else:
            s['max_total'] = 65
        return s
    a_ = sc(argmax_decode(*exact(P)))
    best = None
    for t0 in np.arange(0.05,0.96,0.05):
        for t1 in np.arange(0.1,0.91,0.1):
            for t2 in np.arange(0.1,0.91,0.1):
                ss = sc(__import__('comp_metrics').decode(*exact(P), t0, t1, t2))
                if best is None or ss['total'] > best[0]['total']: best = (ss, (t0,t1,t2))
    b_, th = best
    print(f'clean samples in test: {nclean}')
    print(f'train={len(Xtr)} test={len(Xte)}')
    print(f"argmax: S_pos={a_['S_pos']:.3f} P0R={a_['p0R']:.3f} P0P={a_['p0P']:.3f} P0F1={a_['p0F']:.3f} total={a_['total']:.1f}/{a_['max_total']}")
    print(f"tuned : S_pos={b_['S_pos']:.3f} P0R={b_['p0R']:.3f} P0P={b_['p0P']:.3f} P0F1={b_['p0F']:.3f} total={b_['total']:.1f}/{b_['max_total']} (tau={a.tau}) S_clean={b_['S_clean']}")
    print('per-P0:', {f'{c}': round(x[4],3) for x in b_['per'] if x[1]=='P0' for c in [x[0]]})

def exact(P):
    """cumulative P (n,14) -> (p0[n,6], p1[n,4], p2[n,4])"""
    n = P.shape[0]; p0 = np.zeros((n,6)); p1 = np.zeros((n,4)); p2 = np.zeros((n,4)); j = 0
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            c2, c1, c0 = P[:,j], P[:,j+1], P[:,j+2]; j += 3
            k = [i for i,x in enumerate(CLASSES) if x in FULL].index(ci)
            p2[:,k] = np.clip(c2-c1,0,None); p1[:,k] = np.clip(c1-c0,0,None); p0[:,ci] = c0
        else:
            p0[:,ci] = P[:,j]; j += 1
    return p0, p1, p2

if __name__ == '__main__':
    main()
