# What are the three scored submissions on the final set worth?  (CPU, out-of-fold only)
# The organisers allow three scored submissions on the 654-image final set.  If three candidates that are equally
# good out of fold are submitted and the best reading is kept, the gain on THAT set is max(scores) - score(default).
# Estimated here on 654-image draws (by video) from first-batch out-of-fold rows; every candidate is a complete
# pre-declared system (member set + decoder nested over folds), nothing is fitted on the draw.
import csv
import os
import sys

import numpy as np

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
import final_engine as fe  # noqa: E402
from decoder55 import argmax_decode, nested_oof_decode  # noqa: E402

K = "doc2_strict_total"
names, gold, fold, grp = c.load_split()
sp = {r["name"]: r["split"] for r in csv.DictReader(open(REPO + "/data/manifest_combined.csv", encoding="utf-8"))}
o = np.array([sp[n] == "orig" for n in names])


def get(set_name):
    nm, _, _, P, _ = fe.oof_pool(fe.SETS[set_name])
    assert list(nm) == list(names)
    return np.asarray(P)


C = {}
for s in ("primary", "primary_v2", "fallback"):
    P = get(s)
    assert P.shape[0] == len(names), (s, P.shape)
    C[s] = nested_oof_decode(P, gold, fold, K)
    if s == "primary":
        C["primary, argmax"] = argmax_decode(P)
        C["primary, Doc1 decoder"] = nested_oof_decode(P, gold, fold, "doc1_final")
    sc = c.score(C[s], gold)
    sf = c.score(C[s][o], gold[o])
    print(f"{s:12s} nested, all rows {sc[K]:.2f} | first-batch rows {sf[K]:.2f}", flush=True)
for s in ("primary, argmax", "primary, Doc1 decoder"):
    print(f"{s:22s} all rows {c.score(C[s], gold)[K]:.2f} | first-batch rows {c.score(C[s][o], gold[o])[K]:.2f}", flush=True)

io = np.where(o)[0]
ug, inv = np.unique(grp[io], return_inverse=True)
mem = [io[inv == k] for k in range(len(ug))]
rs = np.random.RandomState(0)
keys = list(C)
S = []
for _ in range(1500):
    idx = []
    while len(idx) < 654:
        idx.extend(mem[rs.randint(len(ug))])
    idx = np.array(idx[:654])
    S.append([c.score(C[k][idx], gold[idx])[K] for k in keys])
S = np.array(S)
print("\n654-image draws from first-batch rows (mean, sd):")
for j, k in enumerate(keys):
    d = S[:, j] - S[:, 0]
    print(f"  {k:24s} {S[:, j].mean():.2f} +- {S[:, j].std():.2f} | minus primary: mean {d.mean():+.2f}, sd {d.std():.2f}, better than primary in {100 * (d > 0).mean():.0f} % of draws")
print("\nbest of three scored submissions, gain over submitting `primary` alone:")
import itertools  # noqa: E402
for trio in itertools.combinations(range(1, len(keys)), 2):
    j = [0] + list(trio)
    g = S[:, j].max(1) - S[:, 0]
    print(f"  primary + {keys[trio[0]]} + {keys[trio[1]]}: mean {g.mean():+.2f} (median {np.median(g):+.2f}, 90th percentile {np.quantile(g, .9):+.2f}); expected final {S[:, j].max(1).mean():.2f}")
