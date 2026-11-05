"""Threshold-free grouped-CV probe evaluation (AP / recall @ fixed clean-FPR)."""
import argparse, csv, os, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA

CLASSES = ['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']
FULL = ['模糊','纹理损失','过锐','过度生成']

def cum_targets(grades):
    y, names = [], []
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            for g, nm in [(1,'P2'),(2,'P1'),(3,'P0')]:
                y.append((grades[:,ci] >= g).astype(int) if g < 3 else (grades[:,ci] == 3).astype(int))
                names.append(f'{c}_{nm}')
        else:
            y.append((grades[:,ci] == 3).astype(int)); names.append(f'{c}_P0')
    return np.stack(y,1), names

def exact_targets(grades):
    y, names = [], []
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            for g, nm in [(1,'P2'),(2,'P1'),(3,'P0')]:
                y.append((grades[:,ci] == g).astype(int)); names.append(f'{c}_{nm}')
        else:
            y.append((grades[:,ci] == 3).astype(int)); names.append(f'{c}_P0')
    return np.stack(y,1), names

def exact_scores(P, names):
    """convert cumulative probs -> exact-grade scores."""
    S = {}
    for c in FULL:
        c2 = P[:, names.index(f'{c}_P2')]; c1 = P[:, names.index(f'{c}_P1')]; c0 = P[:, names.index(f'{c}_P0')]
        S[f'{c}_P2'] = np.clip(c2 - c1, 0, None)
        S[f'{c}_P1'] = np.clip(c1 - c0, 0, None)
        S[f'{c}_P0'] = c0
    for c in ['花屏','虚化突兀']:
        S[f'{c}_P0'] = P[:, names.index(f'{c}_P0')]
    return S

def decode(S, names, t0=0.5, t1=0.5, t2=0.5):
    n = len(next(iter(S.values()))); out = np.zeros((n,6), int)
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            out[:,ci] = np.where(S[f'{c}_P2'] > t2, 1, 0)
            out[:,ci] = np.where(S[f'{c}_P1'] > t1, 2, out[:,ci])
            out[:,ci] = np.where(S[f'{c}_P0'] > t0, 3, out[:,ci])
        else:
            out[:,ci] = np.where(S[f'{c}_P0'] > t0, 3, 0)
    return out

def metrics(grades, S, names, t0, t1, t2):
    pred = decode(S, names, t0, t1, t2)
    Y, nm = exact_targets(grades)
    f1s = []
    for i,n in enumerate(nm):
        yt, yp = Y[:,i], (pred[:, CLASSES.index(n.rsplit('_',1)[0])] == {'P2':1,'P1':2,'P0':3}[n.rsplit('_',1)[1]])
        tp = ((yt==1)&yp).sum(); fp = ((yt==0)&yp).sum(); fn=((yt==1)&~yp).sum()
        f1s.append(2*tp/max(1,2*tp+fp+fn))
    S_pos = float(np.mean(f1s))
    p0 = [i for i,n in enumerate(nm) if n.endswith('P0')]
    rec = np.mean([((Y[:,i]==1)&(pred[:, CLASSES.index(nm[i].rsplit('_',1)[0])]==3)).sum()/max(1,(Y[:,i]==1).sum()) for i in p0])
    prec = np.mean([((Y[:,i]==1)&(pred[:, CLASSES.index(nm[i].rsplit('_',1)[0])]==3)).sum()/max(1,(((pred[:, CLASSES.index(nm[i].rsplit('_',1)[0])]==3))).sum()) for i in p0])
    f1p = np.mean([f1s[i] for i in p0])
    return S_pos, rec, prec, f1p, f1s

def fpr_threshold(grades, S, target):
    clean = (np.stack([exact_targets(grades)[0][:,i] for i in range(14)],1).sum(1)==0)
    m = np.max(np.stack([S[f'{c}_P0'] for c in CLASSES],1), 1)
    return float(np.quantile(m[clean], 1-target))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats'); ap.add_argument('--manifest'); ap.add_argument('--C', type=float, default=0.05)
    ap.add_argument('--pca', type=int, default=512)
    a = ap.parse_args()
    d = np.load(a.feats, allow_pickle=True); X = d['X']; fn = list(d['names'])
    rows = {r['name']: r for r in csv.DictReader(open(a.manifest))}
    rl = [rows[n] for n in fn]
    grades = np.array([[int(r[c]) for c in CLASSES] for r in rl])
    groups = np.array([r['source'] for r in rl])
    Y, names = cum_targets(grades)
    P = np.zeros_like(Y, float)
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in gkf.split(X, Y, groups):
        sc = StandardScaler().fit(X[tr]); Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        if a.pca and a.pca < Xtr.shape[1]:
            pc = PCA(n_components=a.pca, random_state=0).fit(Xtr)
            Xtr, Xva = pc.transform(Xtr), pc.transform(Xva)
        for j in range(Y.shape[1]):
            yj = Y[tr, j]
            if yj.sum() < 5:
                P[va,j] = yj.mean(); continue
            clf = LogisticRegression(C=a.C, class_weight='balanced', max_iter=3000)
            clf.fit(Xtr, yj); P[va,j] = clf.predict_proba(Xva)[:,1]
    S = exact_scores(P, names)
    Ye, nm = exact_targets(grades)
    ap14 = float(np.mean([average_precision_score(Ye[:,i], S[n]) for i,n in enumerate(nm)]))
    p0 = [i for i,n in enumerate(nm) if n.endswith('P0')]
    ap_p0 = float(np.mean([average_precision_score(Ye[:,i], S[nm[i]]) for i in p0]))
    per = {nm[i]: round(float(average_precision_score(Ye[:,i], S[nm[i]])),3) for i in p0}
    np.savez_compressed(a.feats.replace('.npz','_oof.npz'), P=P, names=np.array(names),
                        grades=grades, groups=groups)
    out = [f'== {os.path.basename(a.feats)} X{X.shape} ==',
           f'mAP14={ap14:.3f}  mAP_P0={ap_p0:.3f}',
           f'P0 AP per class: {per}']
    for tgt in (0.005, 0.01, 0.02):
        t0 = fpr_threshold(grades, S, tgt)
        best = None
        for t1 in np.arange(0.2,0.8,0.1):
            for t2 in np.arange(0.2,0.8,0.1):
                m = metrics(grades, S, names, t0, t1, t2)
                if best is None or m[0] > best[0][0]: best = (m, t1, t2)
        m, t1, t2 = best
        out.append(f'FPR<={tgt:.1%}: P0R={m[1]:.3f} P0P={m[2]:.3f} P0F1={m[3]:.3f} S_pos={m[0]:.3f}')
    print('\n'.join(out), flush=True)

if __name__ == '__main__':
    main()
