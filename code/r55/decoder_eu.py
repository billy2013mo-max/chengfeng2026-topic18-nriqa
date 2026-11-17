#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/decoder_eu.py — two decision rules that beat the searched Decoder55 out of fold (study of 2026-11-13,
results/r55/explore4/decoding, reports/Breaking the 52 point plateau.md).  CPU.

  eu    expected-score ("plug-in") rule.  For each of the 14 positive labels the first-order gain of predicting it for
        an image with pooled probability p is
            w_F (2p - F)/(n_pred + n_pos)  +  w_R p/n_pos  +  w_P (p - P)/n_pred
        (F, P, n_pred, n_pos = F1, precision and counts of that label on the fitting rows; the weights are those of
        Doc2_Strict: 25/14 per label for the macro-F1, and for P0 labels 20 x 0.3/6, 0.5/6, 0.2/6 for F1, recall,
        precision).  Per defect the grade with the largest positive gain is predicted.  A row is left fully clean when
        the sum of its gains is below  lambda * prod_d P(no defect) * 20 / n_clean  (the expected loss of clean score).
        One fitted number: lambda in {0.25, 0.5, 1, 2, 4}.  Everything else follows from the metric.
  ord   ordinal rule: grade = highest k with P(grade >= k) >= t[d, k], plus Decoder55's clean gate (thresholds searched).
  vote  per image and defect, the majority of {Decoder55 (four starts), ord, eu with lambda = 1}; three different
        answers -> eu.

All statistics and thresholds come from the pooled OUT-OF-FOLD probabilities of the labelled images.  The decision for
an image depends on that image only (the fitting-set counts are constants), so the rule works one image at a time.
Nothing here reads a test set.

Usage:
  python code/r55/decoder_eu.py --set fallback          # nested report (canonical folds) + fit on all OOF rows
      -> results/r55/final/<set>_decoder_eu.pkl, <set>_decoder_vote.pkl, <set>_decoder_eu.json
"""
import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
from common55 import FULL_DIMS, P0_ONLY_DIMS, renorm, score  # noqa: E402
from decoder55 import GATE_GRID, Decoder55  # noqa: E402

K = "doc2_strict_total"
LABELS = [(d, g) for d in FULL_DIMS for g in (1, 2, 3)] + [(d, 3) for d in P0_ONLY_DIMS]     # 14 positive labels; grade 3 = P0
LD = np.array([l[0] for l in LABELS])
LG = np.array([l[1] for l in LABELS])
W_F = np.where(LG == 3, 25 / 14 + 20 * 0.3 / 6, 25 / 14)
W_R = np.where(LG == 3, 20 * 0.5 / 6, 0.0)
W_P = np.where(LG == 3, 20 * 0.2 / 6, 0.0)
LAMS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _stats(pred, gold):
    pl = pred[:, LD] == LG[None]
    gl = gold[:, LD] == LG[None]
    tp = (pl & gl).sum(0).astype(float)
    npred = pl.sum(0).astype(float)
    npos = gl.sum(0).astype(float)
    return dict(F=2 * tp / np.maximum(1, npred + npos), P=tp / np.maximum(1, npred), npos=npos, npred=npred,
                ncl=float((gold == 0).all(1).sum()), N=int(len(gold)))


def _eu_predict(Q, st, lam):
    """Counts are those of the fitting rows (no rescaling to the decoded set: every term scales alike, so the decision
    is the same for one image as for a batch)."""
    D = np.maximum(1, st["npred"] + st["npos"])
    p = Q[:, LD, LG]
    gain = W_F * (2 * p - st["F"]) / D + W_R * p / np.maximum(1, st["npos"]) + W_P * (p - st["P"]) / np.maximum(1, st["npred"])
    pred = np.zeros((len(Q), 6), dtype=np.int64)
    best = np.zeros((len(Q), 6))
    for l, (d, g) in enumerate(LABELS):
        m = gain[:, l] > best[:, d]
        pred[m, d] = g
        best[m, d] = gain[m, l]
    if lam is not None:
        pred[best.sum(1) < lam * np.prod(Q[:, :, 0], 1) * 20.0 / max(1.0, st["ncl"])] = 0
    return pred


class DecoderEU:
    def __init__(self, lams=LAMS):
        self.lams = tuple(lams)
        self.lam, self.st = None, None

    def _fit_stats(self, Q, gold, lam, iters=15):
        pred = Decoder55().predict(Q)                               # start from argmax
        for _ in range(iters):                                      # F, P, n_pred depend on the rule itself: fixed point
            new = _eu_predict(Q, _stats(pred, gold), lam)
            if (new == pred).all():
                break
            pred = new
        return _stats(pred, gold)

    def fit(self, s, gold):
        Q = renorm(s)
        best = None
        for lam in self.lams:
            st = self._fit_stats(Q, gold, lam)
            o = float(score(_eu_predict(Q, st, lam), gold)[K])
            if best is None or o > best[0]:
                best = (o, st, lam)
        _, self.st, self.lam = best
        return self

    def predict(self, s):
        return _eu_predict(renorm(s), self.st, self.lam)

    def state(self):
        return dict(lam=self.lam, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.st.items()})


class OrdDec(Decoder55):
    TG = np.round(np.arange(0.05, 0.96, 0.05), 3)

    def __init__(self, objective=K, t0=0.5):
        super().__init__(objective)
        self.t = np.full((6, 4), t0)

    def predict(self, s):
        s = renorm(s)
        cum = np.cumsum(s[:, :, ::-1], -1)[:, :, ::-1]              # cum[..., k] = P(grade >= k)
        pred = np.zeros(s.shape[:2], dtype=np.int64)
        for d in range(6):
            for k in ((1, 2, 3) if d in FULL_DIMS else (3,)):       # ascending: a higher grade overrides
                pred[cum[:, d, k] >= self.t[d, k], d] = k
        if self.t_clean > 0:
            pred[np.prod(s[:, :, 0], 1) >= self.t_clean] = 0
        return pred

    def fit(self, s, gold, sweeps=3, verbose=False):
        s = renorm(s)
        best = self._obj(s, gold)
        for _ in range(sweeps):
            imp = False
            for d in range(6):
                for k in ((1, 2, 3) if d in FULL_DIMS else (3,)):
                    keep = self.t[d, k]
                    for v in self.TG:
                        self.t[d, k] = v
                        o = self._obj(s, gold)
                        if o > best + 1e-9:
                            best, keep, imp = o, v, True
                    self.t[d, k] = keep
            keep = self.t_clean
            for v in GATE_GRID:
                self.t_clean = v
                o = self._obj(s, gold)
                if o > best + 1e-9:
                    best, keep, imp = o, v, True
            self.t_clean = keep
            if not imp:
                break
        return self


def fit_d55(s, gold, starts=(0.0, 0.3, 0.6, 0.9)):
    best = None
    for b in starts:
        d = Decoder55(K)
        d.bias[:, 1:] += b
        d.fit(s, gold)
        o = d._obj(renorm(s), gold)
        if best is None or o > best[0]:
            best = (o, d)
    return best[1]


def fit_ord(s, gold, starts=(0.5, 0.4, 0.3, 0.2)):
    best = None
    for t0 in starts:
        d = OrdDec(t0=t0).fit(s, gold)
        o = d._obj(renorm(s), gold)
        if best is None or o > best[0]:
            best = (o, d)
    return best[1]


class VoteDecoder:
    """Majority of Decoder55 (four starts), the ordinal rule and the expected-score rule with lambda = 1."""

    def fit(self, s, gold):
        self.d55 = fit_d55(s, gold)
        self.ord = fit_ord(s, gold)
        self.eu = DecoderEU(lams=(1.0,)).fit(s, gold)
        return self

    def predict(self, s):
        a, o, e = self.d55.predict(s), self.ord.predict(s), self.eu.predict(s)
        return np.where(o == e, o, np.where(a == o, a, np.where(a == e, a, e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="fallback")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--lams", type=float, nargs="+", default=list(LAMS), help="grid for the clean-penalty scale of `eu`")
    ap.add_argument("--no-save", action="store_true", help="report only (do not write the decoder files)")
    a = ap.parse_args()
    import final_engine as fe
    names, gold, fold, P, _ = fe.oof_pool(fe.SETS[a.set])
    names = [str(n) for n in names]
    P = np.asarray(P, dtype=np.float64)
    batch = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    o = np.array([batch[n] == "orig" for n in names])
    _, _, _, grp = c.load_split()
    pred = {k: np.zeros_like(gold) for k in ("d55", "eu", "vote")}
    lam_fold = {}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        v = VoteDecoder().fit(P[tr], gold[tr])
        e = DecoderEU(lams=a.lams).fit(P[tr], gold[tr])
        lam_fold[int(f)] = e.lam
        pred["d55"][te], pred["eu"][te], pred["vote"][te] = v.d55.predict(P[te]), e.predict(P[te]), v.predict(P[te])
        print(f"  fold {f}: lambda {e.lam}", flush=True)
    ug, inv = np.unique(np.asarray(grp), return_inverse=True)
    rs = np.random.RandomState(0)
    rep = dict(set=a.set, lambda_per_fold=lam_fold, nested={})
    for tag, mask in (("all rows", np.ones(len(names), bool)), ("first-batch rows", o)):
        mem = [np.where((inv == i) & mask)[0] for i in range(len(ug))]
        mem = [m for m in mem if len(m)]
        draws = [np.concatenate([mem[j] for j in rs.randint(0, len(mem), len(mem))]) for _ in range(a.bootstrap)]
        base = np.array([score(pred["d55"][i], gold[i])[K] for i in draws])
        print(f"--- nested over the 8 folds, {tag} (n={int(mask.sum())}); difference to Decoder55 with CI90 by video bootstrap")
        for k in pred:
            s = score(pred[k][mask], gold[mask])
            line = f"  {k:5s} {s[K]:.2f} (positive {s['doc2_S_pos']:.2f}, P0 {s['doc2_S_p0']:.2f}, clean {s[K] - s['doc2_S_pos'] - s['doc2_S_p0']:.2f})"
            r = dict(total=float(s[K]))
            if k != "d55":
                dd = np.array([score(pred[k][i], gold[i])[K] for i in draws]) - base
                r.update(delta=float(s[K] - score(pred["d55"][mask], gold[mask])[K]), ci90=[float(np.quantile(dd, .05)), float(np.quantile(dd, .95))],
                         p_gt0=float((dd > 0).mean()))
                line += f" | {r['delta']:+.2f} [{r['ci90'][0]:+.2f}, {r['ci90'][1]:+.2f}], P(>0) {r['p_gt0']:.2f}"
            rep["nested"][f"{tag}:{k}"] = r
            print(line, flush=True)
    import decoder_eu as M                                          # pickle the classes under the module name, not __main__
    eu = M.DecoderEU(lams=a.lams).fit(P, gold)
    vote = M.VoteDecoder().fit(P, gold)
    rep["full_fit"] = dict(eu=eu.state(), eu_insample=float(score(eu.predict(P), gold)[K]), vote_insample=float(score(vote.predict(P), gold)[K]),
                           d55_insample=float(score(vote.d55.predict(P), gold)[K]))
    out = os.path.join(REPO, "results/r55/final")
    os.makedirs(out, exist_ok=True)
    if a.no_save:
        print(f"[decoder_eu] report only: lambda on all rows {eu.lam}, in-sample eu {rep['full_fit']['eu_insample']:.2f}")
        return
    pickle.dump(dict(decoder=eu, members=fe.SETS[a.set]), open(os.path.join(out, f"{a.set}_decoder_eu.pkl"), "wb"))
    pickle.dump(dict(decoder=vote, members=fe.SETS[a.set]), open(os.path.join(out, f"{a.set}_decoder_vote.pkl"), "wb"))
    json.dump(rep, open(os.path.join(out, f"{a.set}_decoder_eu.json"), "w"), indent=2)
    print(f"[decoder_eu] fitted on all OOF rows of '{a.set}': lambda {eu.lam} | in-sample (optimistic) eu {rep['full_fit']['eu_insample']:.2f}, "
          f"vote {rep['full_fit']['vote_insample']:.2f}, Decoder55 {rep['full_fit']['d55_insample']:.2f} -> {out}/{a.set}_decoder_{{eu,vote}}.pkl")


if __name__ == "__main__":
    main()
