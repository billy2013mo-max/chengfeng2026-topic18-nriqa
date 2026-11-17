import numpy as np, itertools
from harness import *
def cum_scores(Qd):  # P(y>=1),P(y>=2),P(y>=3)
    return np.stack([Qd[:, 1:].sum(1), Qd[:, 2:].sum(1), Qd[:, 3]], 1)
def dec_cum(C, t):  # monotone ordinal decode
    return (C[:, 0] > t[0]).astype(int) + ((C[:, 0] > t[0]) & (C[:, 1] > t[1])) + ((C[:, 0] > t[0]) & (C[:, 1] > t[1]) & (C[:, 2] > t[2]))
TG = np.linspace(0.05, 0.95, 37)
def fit_cum(C, y, w=(1, 1, 1)):
    t = np.array([0.5, 0.5, 0.5]); best = -1
    for _ in range(4):
        for j in range(3):
            for v in TG:
                tt = t.copy(); tt[j] = v; s = (f1s(y, dec_cum(C, tt)) * w).sum()
                if s > best + 1e-12: best, t = s, tt
    return t
def nested_cum(Qd, y, w=(1, 1, 1)):
    C = cum_scores(Qd); pr = np.zeros(len(y), int)
    for k in range(8):
        tr, va = fold != k, fold == k
        pr[va] = dec_cum(C[va], fit_cum(C[tr], y[tr], w))
    return pr
SG = np.linspace(0.1, 2.9, 57)
def fit_es(s, y):
    t = np.array([0.5, 1.5, 2.5]); best = -1
    for _ in range(4):
        for j in range(3):
            for v in SG:
                tt = t.copy(); tt[j] = v; tt = np.sort(tt); sc = f1s(y, np.digitize(s, tt)).sum()
                if sc > best + 1e-12: best, t = sc, tt
    return t
def nested_es(Qd, y):
    s = (Qd * np.arange(4)).sum(1); pr = np.zeros(len(y), int)
    for k in range(8):
        tr, va = fold != k, fold == k
        pr[va] = np.digitize(s[va], fit_es(s[tr], y[tr]))
    return pr
if __name__ == "__main__":
    Q = D["Q"]
    report("v7 Q + cumulative-threshold ordinal decode", {d: nested_cum(Q[:, d], g[:, d]) for d in (2, 4)})
    report("v7 Q + expected-severity 3-thr decode (fine grid)", {d: nested_es(Q[:, d], g[:, d]) for d in (2, 4)})

def nested_cum_bag(Qd, y, B=20, seed=0, w=(1, 1, 1)):
    """thresholds = median over group-bootstrap fits on the training folds (more stable)"""
    C = cum_scores(Qd); pr = np.zeros(len(y), int); rng = np.random.default_rng(seed)
    for k in range(8):
        tr = np.where(fold != k)[0]; va = fold == k
        ug = np.unique(groups[tr]); gi = {u: tr[groups[tr] == u] for u in ug}; ts = []
        for b in range(B):
            sel = np.concatenate([gi[u] for u in rng.choice(ug, len(ug), replace=True)])
            ts.append(fit_cum(C[sel], y[sel], w))
        pr[va] = dec_cum(C[va], np.median(np.array(ts), 0))
    return pr
