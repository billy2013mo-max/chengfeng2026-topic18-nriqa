import os; os.environ["OMP_NUM_THREADS"]="2"
import numpy as np, sys
from scipy.optimize import minimize
from harness import *
from ordinal_dec import nested_cum_bag, nested_cum
def rps(Qd, y):
    c = np.cumsum(Qd, 1)[:, :3]; t = (y[:, None] <= np.arange(3)[None]).astype(float); return ((c - t) ** 2).sum(1).mean()
def fit_w(Ps, y, l2=0.01):
    M = len(Ps); A = np.stack(Ps, 0)
    def f(z):
        w = np.exp(z) / np.exp(z).sum(); Qd = np.tensordot(w, A, 1); return rps(Qd, y) + l2 * ((w - 1 / M) ** 2).sum()
    z = minimize(f, np.zeros(M), method="L-BFGS-B").x; return np.exp(z) / np.exp(z).sum()
def nested_dimpool(Pdict, d):
    mem = list(Pdict); A = [Pdict[m][:, d] for m in mem]; out = np.zeros((len(g), 4)); W = []
    for k in range(8):
        tr, va = fold != k, fold == k
        w = fit_w([a[tr] for a in A], g[tr, d]); W.append(w); out[va] = np.tensordot(w, np.stack([a[va] for a in A], 0), 1)
    return out, mem, np.mean(W, 0)
if __name__ == "__main__":
    res = {}; 
    for d in (2, 4):
        Qd, mem, w = nested_dimpool(D["P"], d); print(d, {m: round(float(x), 3) for m, x in zip(mem, w)})
        res[d] = nested_cum_bag(Qd, g[:, d])
    report("v7 members, per-dim RPS-optimal convex pool + bagged cum decode", res)
