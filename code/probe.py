"""Grouped CV linear probe + competition-metric evaluation."""
import argparse, csv, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

CLASSES = ['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']
FULL = ['模糊','纹理损失','过锐','过度生成']
BIN  = ['花屏','虚化突兀']
TAU = 0.02

def targets(grades, cumulative=True):
    """grades: (n,6) int 0=none,1=P2,2=P1,3=P0.

    cumulative=True  -> P(grade>=g), used as training targets for monotone heads
    cumulative=False -> exact one-vs-rest label, used for *scoring* (competition metric)
    """
    y, names = [], []
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            for g, name in [(1,'P2'),(2,'P1'),(3,'P0')]:
                y.append((grades[:,ci] >= g).astype(int) if cumulative
                         else (grades[:,ci] == g).astype(int))
                names.append(f'{c}_{name}')
        else:
            y.append((grades[:,ci] == 3).astype(int)); names.append(f'{c}_P0')
    return np.stack(y, 1), names

def decode(probs, names, t):
    """probs: dict name->prob. returns (n,6) predicted grade."""
    n = len(next(iter(probs.values())))
    out = np.zeros((n, 6), dtype=int)
    t0, t1, t2 = t
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            p2 = probs[f'{c}_P2']; p1 = np.minimum(probs[f'{c}_P1'], p2)
            p0 = np.minimum(probs[f'{c}_P0'], p1)
            g = np.where(p2 > t2, 1, 0)
            g = np.where(p1 > t1, 2, g)
            g = np.where(p0 > t0, 3, g)
            out[:, ci] = g
        else:
            out[:, ci] = np.where(probs[f'{c}_P0'] > t0, 3, 0)
    return out

def score(y_true, y_pred, names):
    """returns dict of competition metrics."""
    f1s, p0_pos = [], []
    for i, nm in enumerate(names):
        yt, yp = y_true[:, i], y_pred[:, i]
        tp = int(((yt==1)&(yp==1)).sum()); fp = int(((yt==0)&(yp==1)).sum()); fn = int(((yt==1)&(yp==0)).sum())
        f1 = 2*tp/max(1, 2*tp+fp+fn); f1s.append(f1)
        if nm.endswith('_P0'):
            rec = tp/max(1, tp+fn); prec = tp/max(1, tp+fp)
            p0_pos.append((rec, prec, f1))
    S_pos = float(np.mean(f1s))
    r = np.mean([x[0] for x in p0_pos]); p = np.mean([x[1] for x in p0_pos]); f = np.mean([x[2] for x in p0_pos])
    S_p0 = 0.5*r + 0.3*f + 0.2*p
    clean = (y_true.sum(1) == 0)
    if clean.sum():
        pred_p0 = np.zeros(len(y_true), bool)
        for i, nm in enumerate(names):
            if nm.endswith('_P0'): pred_p0 |= (y_pred[:, i] == 1)
        fpr = float((pred_p0 & clean).sum() / clean.sum())
    else:
        fpr = 0.0
    S_clean = max(0.0, 1 - fpr/TAU)
    obj = 25*S_pos + 20*S_p0 + 20*S_clean
    return dict(S_pos=S_pos, p0_recall=r, p0_prec=p, p0_f1=f, p0_fpr=fpr,
                S_clean=S_clean, objective=obj, per_label_f1=dict(zip(names, f1s)))

def fit_predict(X, Y, groups, C=1.0):
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    P = np.zeros_like(Y, dtype=float)
    for tr, va in gkf.split(X, Y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        for j in range(Y.shape[1]):
            yj = Y[tr, j]
            if yj.sum() < 5:
                P[va, j] = yj.mean() if yj.sum() else 0.0; continue
            clf = LogisticRegression(C=C, class_weight='balanced', max_iter=3000)
            clf.fit(Xtr, yj)
            P[va, j] = clf.predict_proba(Xva)[:, 1]
    return P

def best_thresholds(grades, names, P):
    y_true_g, _ = targets(grades, cumulative=False)
    best = None
    for t0 in np.arange(0.2, 0.81, 0.05):
        for t1 in np.arange(0.2, 0.81, 0.1):
            for t2 in np.arange(0.2, 0.81, 0.1):
                g = decode({n: P[:, i] for i, n in enumerate(names)}, names, (t0, t1, t2))
                yp, _ = targets(g, cumulative=False)
                s = score(y_true_g, yp, names)
                if best is None or s['objective'] > best[0]['objective']:
                    best = (s, (t0, t1, t2))
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--C', type=float, default=0.05)
    args = ap.parse_args()
    d = np.load(args.feats, allow_pickle=True)
    X = d['X']; names_feat = list(d['names'])
    rows = {r['name']: r for r in csv.DictReader(open(args.manifest))}
    rows_list = [rows[n] for n in names_feat]
    grades = np.array([[int(r[c]) for c in CLASSES] for r in rows_list])
    groups = np.array([r['source'] for r in rows_list])
    Y, names = targets(grades)
    P = fit_predict(X, Y, groups, C=args.C)
    base = decode({n: P[:, i] for i, n in enumerate(names)}, names, (0.5, 0.5, 0.5))
    yt, _ = targets(grades, cumulative=False)
    s05 = score(yt, targets(base, cumulative=False)[0], names)
    sbest, tbest = best_thresholds(grades, names, P)
    print(f'== {os.path.basename(args.feats)}  X{X.shape} ==')
    for tag, s, t in [('thr=.5', s05, (0.5,0.5,0.5)), ('thr=opt', sbest, tbest)]:
        print(f"[{tag} {t}] S_pos={s['S_pos']:.3f} P0R={s['p0_recall']:.3f} P0P={s['p0_prec']:.3f} "
              f"P0F1={s['p0_f1']:.3f} FPR={s['p0_fpr']:.3f} obj={s['objective']:.1f}/65")
    print('per-label F1 (opt thr):', {k: round(v,2) for k,v in sbest['per_label_f1'].items()})

if __name__ == '__main__':
    main()
