import os; os.environ["OMP_NUM_THREADS"]="2"
import numpy as np
from harness import *
from dimpool import nested_dimpool
from ordinal_dec import cum_scores, dec_cum, TG, fit_cum
from joblib import Parallel, delayed
Qd = nested_dimpool(D["P"], 4)[0]; C = cum_scores(Qd); y = g[:, 4]
def fit_min(Ct, yt):
    t = np.array([0.5, 0.5, 0.5]); best = -1
    for _ in range(4):
        for j in range(3):
            for v in TG:
                tt = t.copy(); tt[j] = v; f = f1s(yt, dec_cum(Ct, tt)); s = f.min() + 0.01 * f.sum()
                if s > best + 1e-12: best, t = s, tt
    return t
def one(k):
    tr = np.where(fold != k)[0]; rng = np.random.default_rng(k); ug = np.unique(groups[tr]); gi = {u: tr[groups[tr] == u] for u in ug}; ts = []
    for b in range(20):
        sel = np.concatenate([gi[u] for u in rng.choice(ug, len(ug))]); ts.append(fit_min(C[sel], y[sel]))
    return k, np.median(ts, 0)
pr = np.zeros(len(y), int)
for k, t in Parallel(n_jobs=8)(delayed(one)(k) for k in range(8)): pr[fold == k] = dec_cum(C[fold == k], t)
report("过度生成 best-effort: per-dim pool + bagged max-min-F1 decode", {4: pr})
