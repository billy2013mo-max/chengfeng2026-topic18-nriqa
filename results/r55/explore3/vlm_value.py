# What does the VLM member (phoenix p1, Qwen3-VL-8B) contribute to the primary ensemble?  (CPU, out of fold)
import csv, os, sys
import numpy as np
REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c
from decoder55 import argmax_decode, nested_oof_decode
from probe_eval import DIMS, rps_dim, skills
K = "doc2_strict_total"
z = np.load(sys.argv[1], allow_pickle=True)
names = [str(x) for x in z["names"]]; gold, grp, fold = z["gold"], z["grp"], z["fold"]
sp = {r["name"]: r["split"] for r in csv.DictReader(open(REPO + "/data/manifest_combined.csv", encoding="utf-8"))}
o = np.array([sp[n] == "orig" for n in names])
M = {k: c.renorm(z["m_" + k]) for k in ("radio", "pe", "sig", "v16", "p1", "p3")}
def rpsd(P): return np.stack([rps_dim(P[:, d], gold[:, d], d) for d in range(6)], 1)
def pool(w): return c.renorm(sum(v * M[k] for k, v in w.items()) / sum(w.values()))
ug, inv = np.unique(grp, return_inverse=True); rs = np.random.RandomState(0)
def boots(mask, n=400):
    mem = [np.where((inv == i) & mask)[0] for i in range(len(ug))]; mem = [m for m in mem if len(m)]
    return [np.concatenate([mem[j] for j in rs.randint(0, len(mem), len(mem))]) for _ in range(n)]
allm = np.ones(len(names), bool); ba, bo = boots(allm), boots(o)
print("single members, out of fold:  mean RPS all / first batch | detection AUC | adjacent-grade AUC | argmax score all rows")
for k in M:
    r = rpsd(M[k]); det, gr = [], []
    for d in range(6):
        s = skills(M[k][:, d], gold[:, d], d); det.append(s[0])
        if d in (0, 1, 2, 4): gr.append(np.nanmean(s[1:]))
    print(f"  {k:6s} {r.mean():.5f} / {r[o].mean():.5f} | {np.mean(det):.4f} | {np.mean(gr):.4f} | {c.score(argmax_decode(M[k]), gold)[K]:.2f}   per defect RPS: " + " ".join(f"{DIMS[d]} {r[:, d].mean():.4f}" for d in range(6)))
prim = {"radio": 1/3, "pe": 1/3, "sig": 1/3, "v16": 1.0, "p1": 0.5, "p3": 0.5}
V = {"primary": pool(prim),
     "without the VLM (p3 keeps the family weight)": pool({**{k: v for k, v in prim.items() if k != "p1"}, "p3": 1.0}),
     "without the VLM (p3 at weight 0.5)": pool({k: v for k, v in prim.items() if k != "p1"}),
     "without p3 (VLM keeps the family weight)": pool({**{k: v for k, v in prim.items() if k != "p3"}, "p1": 1.0}),
     "without both phoenix members": pool({k: v for k, v in prim.items() if k not in ("p1", "p3")}),
     "without v16": pool({k: v for k, v in prim.items() if k != "v16"}),
     "without tier-1": pool({k: v for k, v in prim.items() if k in ("v16", "p1", "p3")})}
R = {k: rpsd(P) for k, P in V.items()}
print("\nleave-out, difference to primary (positive RPS = worse), CI90 by video bootstrap:")
D = {}
for k in V:
    x = R[k].mean(1) - R["primary"].mean(1)
    da = np.array([x[i].mean() for i in ba]); do = np.array([x[i].mean() for i in bo])
    D[k] = nested_oof_decode(V[k], gold, fold, K)
    print(f"  {k:46s} RPS all {x.mean():+.5f} [{np.quantile(da, .05):+.5f},{np.quantile(da, .95):+.5f}] ({100 * x.mean() / R['primary'].mean():+.1f} %)"
          f" | first batch {x[o].mean():+.5f} [{np.quantile(do, .05):+.5f},{np.quantile(do, .95):+.5f}] | decoded all {c.score(D[k], gold)[K]:.2f}, first batch {c.score(D[k][o], gold[o])[K]:.2f}", flush=True)
    if k != "primary":
        print("      per defect RPS: " + " ".join(f"{DIMS[d]} {R[k][:, d].mean() - R['primary'][:, d].mean():+.5f}" for d in range(6)))
for k in list(V)[1:]:
    dd = np.array([c.score(D[k][i], gold[i])[K] - c.score(D["primary"][i], gold[i])[K] for i in ba[:300]])
    do = np.array([c.score(D[k][i], gold[i])[K] - c.score(D["primary"][i], gold[i])[K] for i in bo[:300]])
    print(f"  decoded {k:46s} all rows {c.score(D[k], gold)[K] - c.score(D['primary'], gold)[K]:+.2f} [{np.quantile(dd, .05):+.2f},{np.quantile(dd, .95):+.2f}] | first batch {c.score(D[k][o], gold[o])[K] - c.score(D['primary'][o], gold[o])[K]:+.2f} [{np.quantile(do, .05):+.2f},{np.quantile(do, .95):+.2f}]")
