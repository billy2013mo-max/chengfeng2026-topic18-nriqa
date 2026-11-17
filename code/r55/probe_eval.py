#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/probe_eval.py — does a bank of extra per-image features add information to the ensemble? (CPU)

Cheap refutation test used by docs/REASSESSMENT_AND_PLAN_2026-11-12.md (E1, E2).  For every defect a small gradient
boosting model is fitted inside the standard video-grouped 8 folds on
    pool        the 24 pooled out-of-fold probabilities of the primary ensemble         (control)
    pool+bank   the same plus the feature bank                                          (test)
    bank        the feature bank alone                                                  (how much it knows by itself)
and the out-of-fold predictions are compared per defect: RPS (Brier for the P0-only defects), detection AUC,
adjacent-grade AUC.  The decision statistic is pool+bank minus pool, with a paired cluster bootstrap over videos.
The control absorbs whatever a second-stage model gains by itself, so a difference can only come from the bank.
Hyper-parameters are fixed here and never tuned on the result.

Feature files: npz with `names` (N,), `X` (N, F) and optionally `feat_names`.  Images missing from a bank get NaN
(the boosting model handles missing values natively).

Usage:
  python code/r55/probe_eval.py --name ae_recon --feats results/r55/probes/ae_recon_feats.npz [more.npz ...]
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
from gate_plus import auc  # noqa: E402

DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
FULL = (0, 1, 2, 4)


def rps_dim(p, y, d):
    """per-image RPS of a (N, 4) probability table for defect d (binary Brier for the P0-only defects)."""
    if d not in FULL:
        q = p[:, 3] / np.clip(p[:, 0] + p[:, 3], 1e-9, None)
        return (q - (y == 3)) ** 2
    cdf = np.cumsum(p, 1)[:, :3]
    cy = (y[:, None] <= np.arange(3)[None]).astype(float)
    return ((cdf - cy) ** 2).sum(1) / 3.0


def skills(p, y, d):
    det = auc(1 - p[:, 0], y > 0)
    if d not in FULL:
        return det, np.nan, np.nan
    g = []
    for gr in (2, 3):
        m = (y == gr - 1) | (y == gr)
        g.append(auc(p[m, gr:].sum(1), y[m] == gr))
    return det, g[0], g[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--feats", nargs="+", required=True)
    ap.add_argument("--set", default="primary")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--dims", type=int, nargs="*", default=list(range(6)))
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from final_engine import SETS, oof_pool

    names, gold, fold, pp, _ = oof_pool(SETS[a.set])
    grp = c.load_split()[3]
    bank, fn = [], []
    for f in a.feats:
        z = np.load(os.path.join(REPO, f) if not os.path.isabs(f) else f, allow_pickle=True)
        pos = {str(n): i for i, n in enumerate(z["names"])}
        X = np.full((len(names), z["X"].shape[1]), np.nan, dtype=np.float32)
        hit = 0
        for i, n in enumerate(names):
            j = pos.get(n)
            if j is not None:
                X[i] = z["X"][j]
                hit += 1
        bank.append(X)
        fn += [str(x) for x in z["feat_names"]] if "feat_names" in z.files else [f"f{k}" for k in range(X.shape[1])]
        print(f"[probe_eval] {f}: {X.shape[1]} features, {hit}/{len(names)} labelled images covered")
    B = np.concatenate(bank, 1)
    P24 = pp.reshape(len(names), 24).astype(np.float32)
    ug, inv = np.unique(grp, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ug))]
    rs = np.random.RandomState(0)
    boots = [np.concatenate([members[i] for i in rs.randint(0, len(ug), len(ug))]) for _ in range(a.bootstrap)]

    def fit_predict(X, y):
        cls = np.unique(y)
        out = np.zeros((len(y), 4))
        for f in range(8):
            tr, te = fold != f, fold == f
            m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=25,
                                               l2_regularization=1.0, early_stopping=False, random_state=0)
            m.fit(X[tr], y[tr])
            pr = m.predict_proba(X[te])
            for k, cl in enumerate(m.classes_):
                out[te, int(cl)] = pr[:, k]
        out = np.clip(out, 1e-6, 1.0)
        return out / out.sum(1, keepdims=True), cls

    report = {"name": a.name, "feats": a.feats, "n_features": int(B.shape[1]), "dims": {}}
    print(f"=== probe_eval '{a.name}' | {B.shape[1]} bank features | N={len(names)} ===")
    print(f"{'defect':8s} {'side':10s} {'RPS':>8s} {'AUC det':>8s} {'AUC P2|P1':>9s} {'AUC P1|P0':>9s}")
    tot = {"pool": 0.0, "pool+bank": 0.0}
    for d in a.dims:
        y = gold[:, d]
        raw = pp[:, d]
        res = {"raw pool": raw}
        res["pool"], _ = fit_predict(P24, y)
        res["pool+bank"], _ = fit_predict(np.concatenate([P24, B], 1), y)
        res["bank"], _ = fit_predict(B, y)
        rec = {}
        for k, p in res.items():
            r = rps_dim(p, y, d)
            s = skills(p, y, d)
            rec[k] = dict(rps=float(r.mean()), auc_det=float(s[0]), auc_p2p1=float(s[1]), auc_p1p0=float(s[2]))
            print(f"{DIMS[d]:8s} {k:10s} {r.mean():8.5f} {s[0]:8.4f} {s[1]:9.4f} {s[2]:9.4f}")
        r0, r1 = rps_dim(res["pool"], y, d), rps_dim(res["pool+bank"], y, d)
        dr = np.array([r1[b].mean() - r0[b].mean() for b in boots])
        line = f"   pool+bank - pool: dRPS {r1.mean() - r0.mean():+.5f} CI90 [{np.quantile(dr, .05):+.5f}, {np.quantile(dr, .95):+.5f}] P(no gain)={(dr >= 0).mean():.2f}"
        rec["d_rps"] = dict(delta=float(r1.mean() - r0.mean()), ci90=[float(np.quantile(dr, .05)), float(np.quantile(dr, .95))], p_no_gain=float((dr >= 0).mean()))
        if d in FULL:
            da = []
            for b in boots[:200]:
                s1, s0 = skills(res["pool+bank"][b], y[b], d), skills(res["pool"][b], y[b], d)
                da.append(np.nanmean(s1[1:]) - np.nanmean(s0[1:]))
            da = np.array(da)
            s1, s0 = skills(res["pool+bank"], y, d), skills(res["pool"], y, d)
            dg = float(np.nanmean(s1[1:]) - np.nanmean(s0[1:]))
            line += f" | d(adjacent-grade AUC) {dg:+.4f} CI90 [{np.quantile(da, .05):+.4f}, {np.quantile(da, .95):+.4f}] P(no gain)={(da <= 0).mean():.2f}"
            rec["d_auc_grade"] = dict(delta=dg, ci90=[float(np.quantile(da, .05)), float(np.quantile(da, .95))], p_no_gain=float((da <= 0).mean()))
        print(line, flush=True)
        tot["pool"] += r0.mean() / len(a.dims)
        tot["pool+bank"] += r1.mean() / len(a.dims)
        report["dims"][DIMS[d]] = rec
    print(f"mean RPS over the evaluated defects: pool {tot['pool']:.5f} | pool+bank {tot['pool+bank']:.5f} | delta {tot['pool+bank'] - tot['pool']:+.5f}")
    report["mean_rps"] = tot
    od = os.path.join(REPO, "results/r55/probes")
    os.makedirs(od, exist_ok=True)
    json.dump(report, open(os.path.join(od, f"{a.name}_eval.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
