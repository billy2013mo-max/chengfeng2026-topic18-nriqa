# 8-fold evaluation of the tiles + whole-frame tower (`ntg`) against same-recipe RGB controls (`fixed`) inside the primary pool.
import csv, os, sys
import numpy as np
REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c
import decoder_eu as M
from probe_eval import DIMS, rps_dim, skills
K = M.K
z = np.load(sys.argv[1], allow_pickle=True)
names = [str(x) for x in z["names"]]; gold, grp, fold = z["gold"], z["grp"], z["fold"]
pos = {n: i for i, n in enumerate(names)}
sp = {r["name"]: r["split"] for r in csv.DictReader(open(REPO + "/data/manifest_combined.csv", encoding="utf-8"))}
o = np.array([sp[n] == "orig" for n in names])
def load(d):
    P = np.full((len(names), 6, 4), np.nan)
    for f in range(8):
        q = np.load(os.path.join(REPO, "results/next55/towers", d, "pe_core_large", f"fold_{f}_probs.npz"), allow_pickle=True)
        for n, v in zip(q["va_names"], q["va_probs_6x4"]):
            P[pos[str(n)]] = v
    assert np.isfinite(P).all(), d
    return c.renorm(P)
T = {"ntg": load("r55_ntg_s2026"), "c26": load("r55_fixed_s2026"), "c27": load("r55_fixed_s2027")}
if len([f for f in os.listdir(os.path.join(REPO, "results/next55/towers/r55_fixed_s2028/pe_core_large")) if f.endswith("probs.npz")]) == 8:
    T["c28"] = load("r55_fixed_s2028")
CT = [k for k in T if k != "ntg"]
Mm = {k: c.renorm(z["m_" + k]) for k in ("radio", "pe", "sig", "v16", "p1", "p3")}
Mm.update(T)
def rpsd(P): return np.stack([rps_dim(P[:, d], gold[:, d], d) for d in range(6)], 1)
def pool(w): return c.renorm(sum(v * Mm[k] for k, v in w.items()) / sum(w.values()))
BASE = {"v16": 1.0, "p1": 0.5, "p3": 0.5}
def variant(kind, k):
    if kind == "primary": return pool({**BASE, "radio": 1 / 3, "pe": 1 / 3, "sig": 1 / 3})
    if kind == "4th tier-1 member": return pool({**BASE, "radio": .25, "pe": .25, "sig": .25, k: .25})
    if kind == "own family (weight 1)": return pool({**BASE, "radio": 1 / 3, "pe": 1 / 3, "sig": 1 / 3, k: 1.0})
    if kind == "replacing pe": return pool({**BASE, "radio": 1 / 3, k: 1 / 3, "sig": 1 / 3})
ug, inv = np.unique(grp, return_inverse=True); rs = np.random.RandomState(0)
def boots(mask, n=400):
    mem = [np.where((inv == i) & mask)[0] for i in range(len(ug))]; mem = [m for m in mem if len(m)]
    return [np.concatenate([mem[j] for j in rs.randint(0, len(mem), len(mem))]) for _ in range(n)]
ball, bo = boots(np.ones(len(names), bool)), boots(o)
print("single towers, all 3,573 rows: mean RPS | detection AUC | adjacent-grade AUC | per defect RPS")
for k in T:
    r = rpsd(T[k]); det, gr = [], []
    for d in range(6):
        s = skills(T[k][:, d], gold[:, d], d); det.append(s[0])
        if d in (0, 1, 2, 4): gr.append(np.nanmean(s[1:]))
    print(f"  {k:4s} {r.mean():.5f} | {np.mean(det):.4f} | {np.mean(gr):.4f} | " + " ".join(f"{DIMS[d]} {r[:, d].mean():.4f}" for d in range(6)) + f" | first-batch rows {r[o].mean():.5f}")
ref = rpsd(variant("primary", None))
print(f"primary pool: mean RPS {ref.mean():.5f} (first-batch rows {ref[o].mean():.5f})")
dec = {}
def decoded(P):
    pr = np.zeros_like(gold)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        pr[te] = M.DecoderEU().fit(P[tr], gold[tr]).predict(P[te])
    return pr
base_pred = decoded(variant("primary", None))
print(f"primary + eu: all rows {c.score(base_pred, gold)[K]:.2f}, first-batch rows {c.score(base_pred[o], gold[o])[K]:.2f}")
for kind in ("4th tier-1 member", "own family (weight 1)", "replacing pe"):
    print(f"--- {kind}")
    res = {}
    for k in T:
        P = variant(kind, k); r = rpsd(P)
        res[k] = (r.mean(1) - ref.mean(1), decoded(P), r)
    ctrl = np.mean([res[k][0] for k in CT], 0)
    for k in T:
        x = res[k][0]; d = np.array([x[b].mean() for b in ball]); do = np.array([x[b].mean() for b in bo])
        s_all = c.score(res[k][1], gold)[K]; s_o = c.score(res[k][1][o], gold[o])[K]
        print(f"  {k:4s} pooled RPS vs primary {x.mean():+.5f} [{np.quantile(d, .05):+.5f}, {np.quantile(d, .95):+.5f}] ({100 * x.mean() / ref.mean():+.1f} %) | first-batch {x[o].mean():+.5f} | decoded (eu, nested) all {s_all:.2f}, first-batch {s_o:.2f}")
    x = res["ntg"][0] - ctrl; d = np.array([x[b].mean() for b in ball]); do = np.array([x[b].mean() for b in bo])
    print(f"  ntg minus control mean: RPS {x.mean():+.5f} [{np.quantile(d, .05):+.5f}, {np.quantile(d, .95):+.5f}] | first-batch rows {x[o].mean():+.5f} [{np.quantile(do, .05):+.5f}, {np.quantile(do, .95):+.5f}]")
    print("    per defect: " + " ".join(f"{DIMS[j]} {res['ntg'][2][:, j].mean() - np.mean([res[k][2][:, j].mean() for k in CT]):+.5f}" for j in range(6)))
    cp = [res[k][1] for k in CT]
    for tag, mask, bb in (("all rows", np.ones(len(names), bool), ball), ("first-batch rows", o, bo)):
        dd = np.array([c.score(res["ntg"][1][i], gold[i])[K] - np.mean([c.score(p[i], gold[i])[K] for p in cp]) for i in bb[:300]])
        pt = c.score(res["ntg"][1][mask], gold[mask])[K] - np.mean([c.score(p[mask], gold[mask])[K] for p in cp])
        print(f"    decoded ntg minus control mean, {tag}: {pt:+.2f} [{np.quantile(dd, .05):+.2f}, {np.quantile(dd, .95):+.2f}]")
    dd = np.array([c.score(res["ntg"][1][i], gold[i])[K] - c.score(base_pred[i], gold[i])[K] for i in ball[:300]])
    print(f"    decoded ntg variant minus primary + eu, all rows: {c.score(res['ntg'][1], gold)[K] - c.score(base_pred, gold)[K]:+.2f} [{np.quantile(dd, .05):+.2f}, {np.quantile(dd, .95):+.2f}]", flush=True)
