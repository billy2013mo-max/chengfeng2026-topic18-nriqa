import numpy as np, itertools, sys, pickle
from harness import *
TG = np.linspace(0.05, 0.95, 37)
def dec2(C, t):  # among non-P0: 0/1/2 using cum P(y>=1), P(y>=2)
    a = C[:, 0] > t[0]; return a.astype(int) + (a & (C[:, 1] > t[1]))
def hybrid(Qd, base_pred, y):
    C = np.stack([Qd[:, 1:].sum(1), Qd[:, 2:].sum(1)], 1); pr = base_pred.copy()
    for k in range(8):
        tr, va = fold != k, fold == k
        trn = tr & (base_pred != 3); t = np.array([.5, .5]); best = -1
        for _ in range(4):
            for j in range(2):
                for v in TG:
                    tt = t.copy(); tt[j] = v; full = base_pred[tr].copy(); m = full != 3
                    full[m] = dec2(C[tr][m], tt); s = f1s(y[tr], full).sum()
                    if s > best + 1e-12: best, t = s, tt
        m = va & (base_pred != 3); pr[m] = dec2(C[m], t)
    return pr
if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    Q, p = D["Q"], D["pred"]
    report("v7 hybrid: decoder P0 + cum-threshold P1/P2", {d: hybrid(Q[:, d], p[:, d], g[:, d]) for d in (2, 4)})
