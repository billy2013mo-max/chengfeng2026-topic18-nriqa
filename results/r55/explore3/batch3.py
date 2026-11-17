# First-batch-only "twins" of all three tier-1 towers inside the primary ensemble (CPU).
#   twins     r55_origonly_s2026/{cradio_v4_h,pe_core_large,siglip2_so400m_naflex}   trained on first-batch rows only
#   control   qy_s2027/{same three backbones}                                        both batches, another seed
# The control answers "is it the first-batch-only training or just three more towers in the tier-1 family?".
# Everything is judged on first-batch out-of-fold rows (the practice / final set looks like the first batch) and,
# for the cost, on supplement rows.  Usage: python3 batch3.py <pool.npz>
import csv
import os
import sys

import numpy as np

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
from decoder55 import argmax_decode, nested_oof_decode  # noqa: E402
from probe_eval import DIMS, rps_dim  # noqa: E402

K = "doc2_strict_total"
BB = {"r": "cradio_v4_h", "p": "pe_core_large", "s": "siglip2_so400m_naflex"}
if len(sys.argv) > 2 and sys.argv[2] == "dry":          # code check before the twins exist: every backbone -> pe
    BB = {k: "pe_core_large" for k in BB}
z = np.load(sys.argv[1], allow_pickle=True)
names = [str(x) for x in z["names"]]
gold, grp, fold = z["gold"], z["grp"], z["fold"]
sp = {r["name"]: r["split"] for r in csv.DictReader(open(REPO + "/data/manifest_combined.csv", encoding="utf-8"))}
o = np.array([sp[n] == "orig" for n in names])


def load(d, bb):
    pos = {n: i for i, n in enumerate(names)}
    P = np.full((len(names), 6, 4), np.nan)
    for f in range(8):
        p = os.path.join(REPO, "results/next55/towers", d, bb, f"fold_{f}_probs.npz")
        if os.path.exists(p):
            q = np.load(p, allow_pickle=True)
            for n, v in zip(q["va_names"], q["va_probs_6x4"]):
                P[pos[str(n)]] = v
    assert np.isfinite(P).all(), f"{d}/{bb}: folds missing"
    return c.renorm(P)


M = {k: c.renorm(z["m_" + k]) for k in ("radio", "pe", "sig", "v16", "p1", "p3")}
for k, bb in BB.items():
    M["oo_" + k] = load("r55_origonly_s2026", bb)
    M["q7_" + k] = load("qy_s2027", bb)
M["fx"] = load("r55_fixed_s2026", BB["p"])
HAVE_FX = all(os.path.exists(os.path.join(REPO, "results/next55/towers/r55_fixed_s2026", bb, "fold_7_probs.npz")) for bb in BB.values())
if HAVE_FX:                                               # same-recipe both-batch towers of all three backbones
    for k, bb in BB.items():
        M["fx_" + k] = load("r55_fixed_s2026", bb)


def rpsd(P):
    return np.stack([rps_dim(P[:, d], gold[:, d], d) for d in range(6)], 1)


def pool(t1, extra=None):
    """Family-balanced mean: tier-1 family = mean of the towers in t1; v16 weight 1; phoenix p1, p3 half each."""
    w = {k: 1.0 / len(t1) for k in t1}
    w.update({"v16": 1.0, "p1": 0.5, "p3": 0.5})
    if extra:
        w.update(extra)
    return c.renorm(sum(v * M[k] for k, v in w.items()) / sum(w.values()))


ug, inv = np.unique(grp, return_inverse=True)
rs = np.random.RandomState(0)


def boots_for(mask, n=400):
    mem = [np.where((inv == i) & mask)[0] for i in range(len(ug))]
    mem = [m for m in mem if len(m)]
    return [np.concatenate([mem[j] for j in rs.randint(0, len(mem), len(mem))]) for _ in range(n)]


bo, bs = boots_for(o), boots_for(~o)
T1 = ["radio", "pe", "sig"]
TW = ["oo_r", "oo_p", "oo_s"]
Q7 = ["q7_r", "q7_p", "q7_s"]
V = {
    "primary": pool(T1),
    "tier-1 = twins only": pool(TW),
    "tier-1 = primary + 3 twins": pool(T1 + TW),
    "tier-1 = primary + 3 both-batch seed-2 (control)": pool(T1 + Q7),
    "tier-1 = primary + pe twin": pool(T1 + ["oo_p"]),
    "tier-1 = primary + pe both-batch (control)": pool(T1 + ["fx"]),
    "primary + twins as own family (weight 1)": pool(T1, {k: 1 / 3 for k in TW}),
    "primary + seed-2 as own family (control)": pool(T1, {k: 1 / 3 for k in Q7}),
}
FX = ["fx_r", "fx_p", "fx_s"]
if HAVE_FX:
    V["tier-1 = same-recipe both-batch x3"] = pool(FX)
    V["tier-1 = primary + 3 same-recipe both-batch (control)"] = pool(T1 + FX)
R = {k: rpsd(P) for k, P in V.items()}


def cmp(a, b, boots, mask):
    x = R[a].mean(1) - R[b].mean(1)
    d = np.array([x[i].mean() for i in boots])
    return x[mask].mean(), np.quantile(d, .05), np.quantile(d, .95)


print("single towers, first-batch rows, mean RPS:")
for k in ("radio", "pe", "sig"):
    t = {"radio": "r", "pe": "p", "sig": "s"}[k]
    print(f"  {BB[t]:24s} primary member {rpsd(M[k])[o].mean():.5f} | first-batch-only twin {rpsd(M['oo_' + t])[o].mean():.5f}"
          f" | both-batch seed-2 {rpsd(M['q7_' + t])[o].mean():.5f}   (supplement rows: {rpsd(M[k])[~o].mean():.5f} / {rpsd(M['oo_' + t])[~o].mean():.5f} / {rpsd(M['q7_' + t])[~o].mean():.5f})")
print("\nensembles (difference to primary, CI90 by video bootstrap):")
for k in V:
    d, lo, hi = cmp(k, "primary", bo, o)
    ds, los, his = cmp(k, "primary", bs, ~o)
    pd = " ".join(f"{DIMS[j]} {R[k][o, j].mean() - R['primary'][o, j].mean():+.5f}" for j in range(6))
    print(f"  {k:50s} first batch RPS {R[k][o].mean():.5f} d {d:+.5f} [{lo:+.5f},{hi:+.5f}] ({100 * d / R['primary'][o].mean():+.1f} %)"
          f" | argmax {c.score(argmax_decode(V[k][o]), gold[o])[K]:.2f} | supplement d {ds:+.5f} [{los:+.5f},{his:+.5f}]\n      per defect: {pd}", flush=True)
for a, b in (("tier-1 = primary + 3 twins", "tier-1 = primary + 3 both-batch seed-2 (control)"),
             ("tier-1 = primary + pe twin", "tier-1 = primary + pe both-batch (control)"),
             ("primary + twins as own family (weight 1)", "primary + seed-2 as own family (control)")):
    d, lo, hi = cmp(a, b, bo, o)
    print(f"  twins vs control | {a}  minus  {b}: first-batch RPS {d:+.5f} [{lo:+.5f},{hi:+.5f}]", flush=True)

if HAVE_FX:
    print("\nsame recipe (r55 `fixed`, seed 2026), first-batch rows, mean RPS: both batches -> first batch only")
    for k, bb in BB.items():
        x = rpsd(M["oo_" + k]).mean(1) - rpsd(M["fx_" + k]).mean(1)
        dd = np.array([x[i].mean() for i in bo])
        xs = rpsd(M["oo_" + k])[~o].mean() - rpsd(M["fx_" + k])[~o].mean()
        print(f"  {bb:24s} {rpsd(M['fx_' + k])[o].mean():.5f} -> {rpsd(M['oo_' + k])[o].mean():.5f}  d {x[o].mean():+.5f} [{np.quantile(dd, .05):+.5f},{np.quantile(dd, .95):+.5f}]"
              f" ({100 * x[o].mean() / rpsd(M['fx_' + k])[o].mean():+.1f} %) | supplement rows d {xs:+.5f}")
    for a_, b_ in (("tier-1 = twins only", "tier-1 = same-recipe both-batch x3"),
                   ("tier-1 = primary + 3 twins", "tier-1 = primary + 3 same-recipe both-batch (control)")):
        d, lo, hi = cmp(a_, b_, bo, o)
        print(f"  ensemble | {a_}  minus  {b_}: first-batch RPS {d:+.5f} [{lo:+.5f},{hi:+.5f}] ({100 * d / R[b_][o].mean():+.1f} %)", flush=True)

print("\ndecoded Doc2_Strict on first-batch rows (decoder nested over folds, fitted on first-batch rows only):")
io = np.where(o)[0]
loc = {g: i for i, g in enumerate(io)}
D = {}
for k in ["primary", "tier-1 = primary + 3 twins", "tier-1 = primary + 3 both-batch seed-2 (control)", "tier-1 = twins only",
          "primary + twins as own family (weight 1)"] + (["tier-1 = same-recipe both-batch x3", "tier-1 = primary + 3 same-recipe both-batch (control)"] if HAVE_FX else []):
    D[k] = nested_oof_decode(V[k][io], gold[io], fold[io], K)
    s = c.score(D[k], gold[io])
    print(f"  {k:50s} {s[K]:.2f}", flush=True)
bl = [np.array([loc[i] for i in b]) for b in bo[:300]]
g = gold[io]
for a, b in ((("tier-1 = primary + 3 twins", "primary"), ("tier-1 = primary + 3 twins", "tier-1 = primary + 3 both-batch seed-2 (control)"),
             ("tier-1 = twins only", "primary"), ("primary + twins as own family (weight 1)", "primary"))
             + ((("tier-1 = twins only", "tier-1 = same-recipe both-batch x3"),
                 ("tier-1 = primary + 3 twins", "tier-1 = primary + 3 same-recipe both-batch (control)")) if HAVE_FX else ())):
    dd = np.array([c.score(D[a][i], g[i])[K] - c.score(D[b][i], g[i])[K] for i in bl])
    print(f"  {a}  minus  {b}: {c.score(D[a], g)[K] - c.score(D[b], g)[K]:+.2f}  CI90 [{np.quantile(dd, .05):+.2f},{np.quantile(dd, .95):+.2f}]", flush=True)
