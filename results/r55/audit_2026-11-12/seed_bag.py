"""Probe: does averaging several training seeds of one tier-1 member improve the pooled probabilities?
Uses the three control seeds of the pe tower that exist for standard folds 0-1 (r55_fixed_s2026/2027/2028); no training.
  single = pe member replaced by ONE control seed (mean of the three choices), bag = pe member replaced by the 3-seed mean."""
import os, sys, numpy as np
REPO = "/wanqing-develop/luowenjing/18"
for p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
import common55 as c
from final_engine import SETS, oof_pool, pool
from probe_eval import rps_dim
from synth_eval import load_folds, TOW
mem = SETS["primary"]
names, gold, fold, p_ar, P = oof_pool(mem)
gold = np.asarray(gold); grp = c.load_split()[3]
L = [load_folds(f"{TOW}/r55_fixed_s{s}/pe_core_large", 8)[0] for s in (2026, 2027, 2028)]
keep = np.array([i for i, n in enumerate(names) if all(n in m and m[n][0] in (0, 1) for m in L)])
S = [c.renorm(np.stack([m[names[i]][1] for i in keep]).astype(np.float64)) for m in L]
g = gold[keep]; Pk = {k: np.asarray(v)[keep] for k, v in P.items()}
ug, inv = np.unique(grp[keep], return_inverse=True)
membs = [np.where(inv == i)[0] for i in range(len(ug))]
rs = np.random.RandomState(0); boots = [np.concatenate([membs[i] for i in rs.randint(0, len(ug), len(ug))]) for _ in range(1000)]
def per_img(p): return np.mean([rps_dim(p[:, d], g[:, d], d) for d in range(6)], 0)
def ci(a, b):
    d = np.array([b[x].mean() - a[x].mean() for x in boots]); return b.mean() - a.mean(), np.quantile(d, .05), np.quantile(d, .95)
print(f"folds 0-1, {len(keep)} images")
# tower alone
t1 = np.mean([per_img(s) for s in S], 0); tb = per_img(c.renorm(np.mean(S, 0)))
d = ci(t1, tb); print(f"pe tower alone : one seed {t1.mean():.5f} (seeds {' '.join(f'{per_img(s).mean():.5f}' for s in S)}) -> 3-seed mean {tb.mean():.5f}  d {d[0]:+.5f} CI90 [{d[1]:+.5f}, {d[2]:+.5f}] ({100 * d[0] / t1.mean():+.1f} %)")
# in the pool
def pooled(q):
    Q = dict(Pk); Q["pe"] = q; return pool(Q, mem)
p1 = np.mean([per_img(pooled(s)) for s in S], 0); pb = per_img(pooled(c.renorm(np.mean(S, 0))))
d = ci(p1, pb); print(f"primary pool   : pe = one seed {p1.mean():.5f} (seeds {' '.join(f'{per_img(pooled(s)).mean():.5f}' for s in S)}) -> pe = 3-seed mean {pb.mean():.5f}  d {d[0]:+.5f} CI90 [{d[1]:+.5f}, {d[2]:+.5f}] ({100 * d[0] / p1.mean():+.2f} %)")
# what a pool of ONLY the three tier-1 members gains, to bound the effect of bagging every tier-1 member
t3 = ["cradio", "pe", "sig"]
def pool3(q):
    return c.renorm(np.mean([Pk["cradio"], q, Pk["sig"]], 0))
a1 = np.mean([per_img(pool3(s)) for s in S], 0); ab = per_img(pool3(c.renorm(np.mean(S, 0))))
d = ci(a1, ab); print(f"tier-1 family  : pe = one seed {a1.mean():.5f} -> pe = 3-seed mean {ab.mean():.5f}  d {d[0]:+.5f} CI90 [{d[1]:+.5f}, {d[2]:+.5f}] ({100 * d[0] / a1.mean():+.2f} %)")
for dname, k in zip(["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"], range(6)):
    x1 = np.mean([rps_dim(pooled(s)[:, k], g[:, k], k) for s in S], 0); xb = rps_dim(pooled(c.renorm(np.mean(S, 0)))[:, k], g[:, k], k)
    d = ci(x1, xb); print(f"    pool {dname:5s} {x1.mean():.5f} -> {xb.mean():.5f} d {d[0]:+.5f} CI90 [{d[1]:+.5f}, {d[2]:+.5f}]")
