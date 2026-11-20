#!/usr/bin/env python3
"""Evaluate the cross-defect invariance towers (direction B, 2026-11-17) against same-recipe controls.

  python code/r55/eval_disentangle.py TAG=tower_dir:control_dir [TAG=...]

For every pair (treated tower, control tower with the same recipe and seed, loss term off):
  1. single tower, all 3,573 rows: 过锐 / 过度生成 RPS, P0-vs-P1 and P1-vs-P2 AUC, P0 AP, and the
     "leakage slope" (expected severity regressed on the other defects' human grades at a fixed own grade);
  2. v7 (11 members) + the tower as a 12th member, nested 8-fold RoutedGateEU decode (all decision parameters fitted
     on training folds only): total score, P0 precision / recall and per-grade F1 for 过锐 / 过度生成, paired
     video-cluster bootstrap (B=1000) of treated minus control.
Training labels / OOF only; no test-set data is read."""
import os
import sys

REPO = "/wanqing-develop/luowenjing/18"
for p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
os.chdir(REPO)
import numpy as np  # noqa: E402
import common55 as c  # noqa: E402
import deconfound_calibrator as dc  # noqa: E402
import routed_eu_decoder as R  # noqa: E402
from eval_specialist_gate import load_oof_dir  # noqa: E402
from final_engine import oof_pool  # noqa: E402

V7 = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe", "p12_pe", "p1_qy", "pe_bkseg1", "ntg_convnext"]
DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
GRADED = (0, 1, 2, 4)
TOWERS = "results/next55/towers"


def split_extra():
    import csv
    names, gold, fold, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open("data/manifest_combined.csv", encoding="utf-8"))}
    return np.array([sm[str(n)] == "orig" for n in names]), np.array(groups)


def auc(sc, y):
    o = np.argsort(sc, kind="stable")
    rk = np.empty(len(sc))
    rk[o] = np.arange(1, len(sc) + 1)
    _, inv, cnt = np.unique(sc, return_inverse=True, return_counts=True)
    rk = (np.bincount(inv, rk) / cnt)[inv]
    return (rk[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum()))


def ap(sc, y):
    o = np.argsort(-sc)
    gg = y[o]
    cum = np.cumsum(gg)
    return float((cum[gg == 1] / (np.where(gg == 1)[0] + 1)).mean())


def rps(Q, gold, d):
    cdf_p = np.cumsum(Q[:, d, :3], axis=1)
    cdf_t = (gold[:, d][:, None] <= np.arange(3)[None]).astype(float)
    return ((cdf_p - cdf_t) ** 2).sum(1) / 3.0


def leak_slope(Q, gold, d):
    es = (Q[:, d] * np.arange(4)).sum(-1)
    others = [j for j in GRADED if j != d]
    X = np.column_stack([np.eye(4)[gold[:, d]], gold[:, others]])
    b = np.linalg.lstsq(X, es, rcond=None)[0]
    return float(b[4:].sum())


def tower_stats(P, gold):
    out = {}
    for d in (2, 4):
        g = gold[:, d]
        m23, m12 = np.isin(g, (2, 3)), np.isin(g, (1, 2))
        out[d] = dict(rps=float(rps(P, gold, d).mean()),
                      auc_p0_p1=float(auc(P[m23, d, 3], (g[m23] == 3).astype(int))),
                      auc_p1_p2=float(auc(P[m12, d, 2] + P[m12, d, 3], (g[m12] == 2).astype(int))),
                      ap_p0=ap(P[:, d, 3], (g == 3).astype(int)), leak=leak_slope(P, gold, d))
    return out


def nested(P, gold, fold, groups, is_orig):
    members = list(P)
    cfg = dc._build_routed_gate_cfg(members)
    Q = np.zeros((len(gold), 6, 4))
    pred = np.zeros((len(gold), 6), dtype=np.int64)
    for k in range(8):
        va, tr = fold == k, fold != k
        dec = R.RoutedGateEUDecoder(**cfg).fit({m: v[tr] for m, v in P.items()}, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
        Q[va] = dec.pool_from_member_dict({m: v[va] for m, v in P.items()})
        pred[va] = dec.eu.predict(Q[va])
    return Q, pred


def cell_stats(pred, gold, idx=None):
    if idx is not None:
        pred, gold = pred[idx], gold[idx]
    s = c.score(pred, gold)
    row = [s["doc2_strict_total"]]
    for d in (2, 4):
        gt, pd = gold[:, d] == 3, pred[:, d] == 3
        tp = (gt & pd).sum()
        row += [tp / max(1, pd.sum()), tp / max(1, gt.sum())]
        for k in (3, 2, 1):
            a, b = gold[:, d] == k, pred[:, d] == k
            row.append(2 * (a & b).sum() / max(1, a.sum() + b.sum()))
    return np.array(row, dtype=float)


LAB = ["total"] + [f"{DIMS[d]} {x}" for d in (2, 4) for x in ("P0 prec", "P0 rec", "F1 P0", "F1 P1", "F1 P2")]


def main():
    pairs = [a.split("=", 1) for a in sys.argv[1:]]
    names, gold, fold, _, Pv7 = oof_pool(V7)
    is_orig, groups = split_extra()
    uniq = np.unique(groups)
    gi = {u: np.where(groups == u)[0] for u in uniq}
    rng = np.random.default_rng(20261117)
    boots = [np.concatenate([gi[u] for u in rng.choice(uniq, size=len(uniq), replace=True)]) for _ in range(1000)]

    def load(d):
        arr, valid = load_oof_dir(os.path.join(TOWERS, d, "pe_core_large"), names)
        assert valid.all(), f"{d}: {valid.sum()} / {len(names)} rows"
        return c.renorm(arr)

    cache = {}
    for tag, spec in pairs:
        tdir, cdir = spec.split(":")
        print(f"\n================ {tag}: {tdir}  vs control {cdir}", flush=True)
        Pt, Pc = load(tdir), load(cdir)
        st, sc = tower_stats(Pt, gold), tower_stats(Pc, gold)
        for d in (2, 4):
            print(f"  single tower {DIMS[d]}: " + " | ".join(f"{k} {sc[d][k]:.4f} -> {st[d][k]:.4f}" for k in st[d]), flush=True)
        res = {}
        for nm, Px in (("control", Pc), ("treated", Pt)):
            key = cdir if nm == "control" else tdir
            if key not in cache:
                R.MEMBER_FAMILY_MAP["xtra"] = "ntg"
                cache[key] = nested({**Pv7, "xtra": Px}, gold, fold, groups, is_orig)
            res[nm] = cache[key][1]
        for lab, msk in (("ALL", np.ones(len(gold), bool)), ("first batch", is_orig)):
            a = cell_stats(res["control"], gold, np.where(msk)[0])
            b = cell_stats(res["treated"], gold, np.where(msk)[0])
            bd = np.array([cell_stats(res["treated"], gold, i[msk[i]]) - cell_stats(res["control"], gold, i[msk[i]]) for i in boots])
            print(f"  v7 + tower, nested decode, {lab}:", flush=True)
            for j, l in enumerate(LAB):
                lo, hi = np.percentile(bd[:, j], [5, 95])
                print(f"    {l:<18} control {a[j]:.4f} treated {b[j]:.4f} delta {b[j] - a[j]:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)


if __name__ == "__main__":
    main()
