# Two-fold screen of the high-pass residual tower (preset `hp`): does it add information to the ensemble?  (CPU)
# Rows: validation images of folds 0 and 1.  Controls: same-recipe RGB towers (`fixed`, seeds 2026 / 2027 / 2028).
import os, sys
import numpy as np
REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c
from probe_eval import DIMS, rps_dim, skills
z = np.load(sys.argv[1], allow_pickle=True)
names = [str(x) for x in z["names"]]; gold, grp, fold = z["gold"], z["grp"], z["fold"]
pos = {n: i for i, n in enumerate(names)}
def load(d):
    P = np.full((len(names), 6, 4), np.nan)
    for f in (0, 1):
        q = np.load(os.path.join(REPO, "results/next55/towers", d, "pe_core_large", f"fold_{f}_probs.npz"), allow_pickle=True)
        for n, v in zip(q["va_names"], q["va_probs_6x4"]):
            P[pos[str(n)]] = v
    return P
T = {"hp": load("r55_hp_s2026"), "c26": load("r55_fixed_s2026"), "c27": load("r55_fixed_s2027"), "c28": load("r55_fixed_s2028")}
m = np.all([np.isfinite(v).all(axis=(1, 2)) for v in T.values()], axis=0)
assert set(fold[m]) == {0, 1}
T = {k: c.renorm(np.where(m[:, None, None], v, 0.25)) for k, v in T.items()}
M = {k: c.renorm(z["m_" + k]) for k in ("radio", "pe", "sig", "v16", "p1", "p3")}
M.update(T)
def rpsd(P): return np.stack([rps_dim(P[:, d], gold[:, d], d) for d in range(6)], 1)
def pool(extra_t1=(), fam=None):
    t1 = ["radio", "pe", "sig"] + list(extra_t1)
    w = {k: 1.0 / len(t1) for k in t1}; w.update({"v16": 1.0, "p1": 0.5, "p3": 0.5})
    if fam: w[fam] = 1.0
    return c.renorm(sum(v * M[k] for k, v in w.items()) / sum(w.values()))
idx = np.where(m)[0]
ug, inv = np.unique(grp[idx], return_inverse=True); mem = [idx[inv == k] for k in range(len(ug))]
rs = np.random.RandomState(0); boots = [np.concatenate([mem[j] for j in rs.randint(0, len(ug), len(ug))]) for _ in range(1000)]
print(f"rows: {len(idx)} validation images of folds 0-1, {len(ug)} videos")
print("single towers: mean RPS | detection AUC | adjacent-grade AUC | per-defect RPS")
for k in T:
    r = rpsd(M[k])[idx]; det, gr = [], []
    for d in range(6):
        s = skills(M[k][idx, d], gold[idx, d], d); det.append(s[0])
        if d in (0, 1, 2, 4): gr.append(np.nanmean(s[1:]))
    print(f"  {k:4s} {r.mean():.5f} | {np.mean(det):.4f} | {np.mean(gr):.4f} | " + " ".join(f"{DIMS[d]} {r[:, d].mean():.4f}" for d in range(6)))
ref = rpsd(pool())
res = lambda P: (rpsd(P) - ref).mean(1)          # per-row mean-RPS difference to the primary pool
E = {k: (rpsd(M[k]) - ref)[idx].mean(1) for k in T}
print("correlation of per-row RPS excess over the pool: hp vs controls %.2f %.2f %.2f | controls among themselves %.2f %.2f %.2f" % (
    np.corrcoef(E["hp"], E["c26"])[0, 1], np.corrcoef(E["hp"], E["c27"])[0, 1], np.corrcoef(E["hp"], E["c28"])[0, 1],
    np.corrcoef(E["c26"], E["c27"])[0, 1], np.corrcoef(E["c26"], E["c28"])[0, 1], np.corrcoef(E["c27"], E["c28"])[0, 1]))
print(f"primary pool on these rows: mean RPS {ref[idx].mean():.5f}")
for lab, mk in (("as a 4th tier-1 member", lambda k: pool(extra_t1=[k])), ("as a family of its own (weight 1)", lambda k: pool(fam=k))):
    D = {k: res(mk(k)) for k in T}
    ctrl = np.mean([D[k] for k in ("c26", "c27", "c28")], 0)
    def ci(x):
        dd = np.array([x[b].mean() for b in boots]); return x[idx].mean(), np.quantile(dd, .05), np.quantile(dd, .95)
    print(f"--- added {lab}: change of pooled mean RPS (negative = better), CI90 by video bootstrap")
    for k in T:
        print("  %-4s %+.5f [%+.5f, %+.5f] (%+.1f %%)" % ((k,) + ci(D[k]) + (100 * D[k][idx].mean() / ref[idx].mean(),)))
    print("  hp minus the mean of the three controls: %+.5f [%+.5f, %+.5f]" % ci(D["hp"] - ctrl))
    P_hp, P_c = mk("hp"), mk("c27")
    print("  per defect, hp minus control mean: " + " ".join(f"{DIMS[d]} {(rpsd(P_hp)[idx, d].mean() - np.mean([rpsd(mk(k))[idx, d].mean() for k in ('c26', 'c27', 'c28')])):+.5f}" for d in range(6)))
