import os, sys, numpy as np
REPO = "/wanqing-develop/luowenjing/18"
for p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
import common55 as c
from final_engine import SETS, oof_pool, weights
from probe_eval import rps_dim
from decoder55 import argmax_decode, nested_oof_decode
mem = SETS["primary"]
names, gold, fold, p_ar, P = oof_pool(mem)
gold = np.asarray(gold); w = weights(mem); w = w / w.sum()
X = np.stack([c.renorm(np.asarray(P[m], dtype=np.float64)) for m in mem])          # (M, N, 6, 4)
eps = 1e-6
def geo(X, w):
    return c.renorm(np.exp((w[:, None, None, None] * np.log(np.clip(X, eps, 1))).sum(0)))
def powm(X, w, a):
    return c.renorm(((w[:, None, None, None] * np.clip(X, eps, 1) ** a).sum(0)) ** (1 / a))
def cumlogit(X, w):
    # average the log-odds of P(y >= g), g = 1..3, then rebuild a monotone table
    S = np.clip(np.cumsum(X[..., ::-1], -1)[..., ::-1][..., 1:], eps, 1 - eps)   # (M,N,6,3): P(>=1),P(>=2),P(>=3)
    L = (w[:, None, None, None] * np.log(S / (1 - S))).sum(0)
    s = 1 / (1 + np.exp(-L)); s = np.minimum.accumulate(s, -1)
    q = np.stack([1 - s[..., 0], s[..., 0] - s[..., 1], s[..., 1] - s[..., 2], s[..., 2]], -1)
    return c.renorm(np.clip(q, eps, 1))
forms = {"arithmetic (current)": p_ar, "geometric": geo(X, w), "power 0.5": powm(X, w, 0.5), "power 2": powm(X, w, 2.0),
         "cumulative log-odds": cumlogit(X, w), "half arithmetic + half geometric": c.renorm(0.5 * p_ar + 0.5 * geo(X, w))}
grp = c.load_split()[3]; ug, inv = np.unique(grp, return_inverse=True)
membs = [np.where(inv == i)[0] for i in range(len(ug))]
rs = np.random.RandomState(0); boots = [np.concatenate([membs[i] for i in rs.randint(0, len(ug), len(ug))]) for _ in range(400)]
def per_img(p): return np.mean([rps_dim(p[:, d], gold[:, d], d) for d in range(6)], 0)
r0 = per_img(p_ar); K = "doc2_strict_total"
for k, p in forms.items():
    r = per_img(p); d = np.array([r[b].mean() - r0[b].mean() for b in boots])
    og = rps_dim(p[:, 4], gold[:, 4], 4).mean()
    print(f"{k:34s} mean RPS {r.mean():.5f}  d {r.mean() - r0.mean():+.5f} CI90 [{np.quantile(d, .05):+.5f}, {np.quantile(d, .95):+.5f}]  过度生成 RPS {og:.5f}  argmax {c.score(argmax_decode(p), gold)[K]:.2f}  nested {c.score(nested_oof_decode(p, gold, fold, K), gold)[K]:.2f}", flush=True)
