#!/usr/bin/env python3
"""
code/r55/decoder_eu2.py — audit 2026-11-14, P2: three changes to the `eu` rule of decoder_eu.py, each switchable, judged
nested (decoder fitted on 7 folds, applied to the 8th) on several group-wise fold partitions.  CPU.

  lam   wider clean-penalty grid (the shipped grid ends at 0.25 and every fold picked 0.25)
  cal   per-dimension temperature on the pooled 6x4 probabilities, fitted by NLL on the fitting rows
  clean learned P(all clean): logistic regression on [logit prod P(no defect), member clean heads] fitted on the fitting
        rows, used in place of prod P(no defect) in the clean gate

Members' clean heads: Tier-1 `va_qclean` (results/qingyu/<bb>/fold_*_probs.npz), V16 `va_gate`, Phoenix `p_clean_joint`.
`--v16-dir` switches V16 to another 8-fold OOF (e.g. results/r55/v16_qy, the video-grouped retrain).

  python3 code/r55/decoder_eu2.py --set fallback [--v16-dir results/r55/v16_qy] [--partitions 4]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
import decoder_eu as E  # noqa: E402
from common55 import renorm, score  # noqa: E402

K = "doc2_strict_total"
LAMS_OLD = (0.25, 0.5, 1.0, 2.0, 4.0)
LAMS_NEW = (0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0)


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def load_pool(set_name, v16_dir):
    import final_engine as fe
    members = fe.SETS[set_name]
    names, gold, fold, grp = c.load_split()
    idx = {n: i for i, n in enumerate(names)}
    _, _, _, _, P = fe.oof_pool(members)
    feats = []
    if v16_dir:
        P["v16"] = np.full((len(names), 6, 4), np.nan)
        files = sorted(glob.glob(os.path.join(REPO, v16_dir, "V16_Full_F*_probs.npz")))
        assert len(files) == 8, files
        for f in files:
            z = np.load(f, allow_pickle=True)
            for n, v in zip(z["va_names"], c.p012_to_6x4_simplex(z["va_p0"], z["va_p1"], z["va_p2"])):
                P["v16"][idx[str(n)]] = v
        assert not np.isnan(P["v16"]).any()
    pooled = fe.pool(P, members)
    # clean heads
    for m in members:
        if m in fe.TIER1:
            d = os.path.join(REPO, "results/qingyu", fe.TIER1[m]) if fe.OOF_KEY[m].startswith("qy_") else None
            if d is None:
                continue
            v = np.full(len(names), np.nan)
            for f in glob.glob(os.path.join(d, "fold_*_probs.npz")):
                z = np.load(f, allow_pickle=True)
                for n, q in zip(z["va_names"], z["va_qclean"]):
                    v[idx[str(n)]] = q
            if not np.isnan(v).any():
                feats.append(logit(v))
        elif m == "v16":
            d = os.path.join(REPO, v16_dir or "results/v16_fulldata")
            v = np.full(len(names), np.nan)
            for f in glob.glob(os.path.join(d, "V16_Full_F*_probs.npz")):
                z = np.load(f, allow_pickle=True)
                for n, q in zip(z["va_names"], z["va_gate"]):
                    v[idx[str(n)]] = q
            feats.append(logit(v))
        else:
            z = np.load(os.path.join(REPO, f"results/phoenix/pillar{m[1]}_stage2-rl2r_oof.npz"), allow_pickle=True)
            mp = {str(n): q for n, q in zip(z["names"], z["p_clean_joint"])}
            feats.append(logit(np.array([mp[n] for n in names])))
    return names, gold, fold, grp, pooled, np.stack(feats, 1)


def fit_temp(Q, gold, grid=np.round(np.arange(0.5, 2.01, 0.05), 3)):
    T = np.ones(6)
    L = np.log(np.clip(Q, 1e-9, 1))
    for d in range(6):
        best = None
        for t in grid:
            e = np.exp(L[:, d] / t)
            q = e / e.sum(1, keepdims=True)
            nll = -np.log(np.clip(q[np.arange(len(q)), gold[:, d]], 1e-9, 1)).mean()
            if best is None or nll < best[0]:
                best = (nll, t)
        T[d] = best[1]
    return T


def apply_temp(Q, T):
    return renorm(np.exp(np.log(np.clip(Q, 1e-9, 1)) / T[None, :, None]))


class EU2:
    def __init__(self, lams, cal, clean):
        self.lams, self.cal, self.clean = lams, cal, clean

    def _pc(self, Q, F):
        base = logit(np.prod(Q[:, :, 0], 1))
        if not self.clean:
            return np.prod(Q[:, :, 0], 1)
        X = np.column_stack([base, F])
        return self.lr.predict_proba(X)[:, 1]

    def fit(self, Q, F, gold):
        Q = renorm(Q)
        if self.cal:
            self.T = fit_temp(Q, gold)
            Q = apply_temp(Q, self.T)
        if self.clean:
            from sklearn.linear_model import LogisticRegression
            X = np.column_stack([logit(np.prod(Q[:, :, 0], 1)), F])
            self.lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, (gold == 0).all(1))
        pc = self._pc(Q, F)
        best = None
        for lam in self.lams:
            st = self._fit_stats(Q, pc, gold, lam)
            o = float(score(self._pred(Q, pc, st, lam), gold)[K])
            if best is None or o > best[0]:
                best = (o, st, lam)
        _, self.st, self.lam = best
        return self

    @staticmethod
    def _pred(Q, pc, st, lam):
        pred = E._eu_predict(Q, st, None)
        D = np.maximum(1, st["npred"] + st["npos"])
        p = Q[:, E.LD, E.LG]
        gain = E.W_F * (2 * p - st["F"]) / D + E.W_R * p / np.maximum(1, st["npos"]) + E.W_P * (p - st["P"]) / np.maximum(1, st["npred"])
        best = np.zeros((len(Q), 6))
        for l, (d, g) in enumerate(E.LABELS):
            best[:, d] = np.maximum(best[:, d], gain[:, l])
        pred[best.sum(1) < lam * pc * 20.0 / max(1.0, st["ncl"])] = 0
        return pred

    def _fit_stats(self, Q, pc, gold, lam, iters=15):
        pred = Q.argmax(-1)
        for _ in range(iters):
            new = self._pred(Q, pc, E._stats(pred, gold), lam)
            if (new == pred).all():
                break
            pred = new
        return E._stats(pred, gold)

    def predict(self, Q, F):
        Q = renorm(Q)
        if self.cal:
            Q = apply_temp(Q, self.T)
        return self._pred(Q, self._pc(Q, F), self.st, self.lam)


VARIANTS = {"eu": dict(lams=LAMS_OLD, cal=False, clean=False),
            "eu+lam": dict(lams=LAMS_NEW, cal=False, clean=False),
            "eu+cal": dict(lams=LAMS_OLD, cal=True, clean=False),
            "eu+clean": dict(lams=LAMS_NEW, cal=False, clean=True),
            "eu+lam+cal": dict(lams=LAMS_NEW, cal=True, clean=False),
            "eu+all": dict(lams=LAMS_NEW, cal=True, clean=True)}


def partitions(fold, grp, n):
    out = [fold]
    ug, inv = np.unique(grp, return_inverse=True)
    for s in range(1, n):
        rs = np.random.RandomState(1000 + s)
        out.append(rs.permutation(np.arange(len(ug)) % 8)[inv])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="fallback")
    ap.add_argument("--v16-dir", default="")
    ap.add_argument("--partitions", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    names, gold, fold, grp, Pp, F = load_pool(a.set, a.v16_dir)
    batch = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    orig = np.array([batch[n] == "orig" for n in names])
    ug, inv = np.unique(grp, return_inverse=True)
    rep = dict(set=a.set, v16_dir=a.v16_dir or "results/v16_fulldata", partitions=[])
    for pi, fp in enumerate(partitions(fold, grp, a.partitions)):
        pred = {k: np.zeros_like(gold) for k in VARIANTS}
        lam = {k: [] for k in VARIANTS}
        for f in range(8):
            tr, te = fp != f, fp == f
            for k, kw in VARIANTS.items():
                d = EU2(**kw).fit(Pp[tr], F[tr], gold[tr])
                pred[k][te] = d.predict(Pp[te], F[te])
                lam[k].append(d.lam)
        rs = np.random.RandomState(pi)
        prow = dict(partition=pi, lam=lam, res={})
        for tag, mask in (("all", np.ones(len(names), bool)), ("first", orig)):
            mem = [m for m in (np.where((inv == i) & mask)[0] for i in range(len(ug))) if len(m)]
            draws = [np.concatenate([mem[j] for j in rs.randint(0, len(mem), len(mem))]) for _ in range(a.bootstrap)]
            base = np.array([score(pred["eu"][i], gold[i])[K] for i in draws])
            for k in VARIANTS:
                s = float(score(pred[k][mask], gold[mask])[K])
                r = dict(total=s)
                if k != "eu":
                    dd = np.array([score(pred[k][i], gold[i])[K] for i in draws]) - base
                    r.update(delta=s - float(score(pred["eu"][mask], gold[mask])[K]), lo=float(np.quantile(dd, .05)),
                             hi=float(np.quantile(dd, .95)))
                prow["res"][f"{tag}:{k}"] = r
                print(f"p{pi} {tag:5s} {k:11s} {s:.2f}" + (f"  {r['delta']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]" if k != "eu" else "")
                      + (f"  lam {lam[k]}" if tag == "all" else ""), flush=True)
        rep["partitions"].append(prow)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
