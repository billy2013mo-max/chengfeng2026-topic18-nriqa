#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/decoder_select.py — which decoding rule gives the highest expected objective score? (CPU)

All candidates turn the SAME pooled out-of-fold probabilities into labels, so they can be compared far more sharply
than two trainings can: every rule is fitted on seven folds and applied to the eighth (nested), and differences are
paired bootstraps over videos.  Objective = Doc2_Strict total (25 positive + 20 P0 + 20 clean).

  argmax          no fitting
  shift           argmax after adding one number b to every defect grade; b chosen on the training folds
  decoder         Decoder55 as shipped: coordinate ascent on 14 grade biases, 6 P0 floors, 1 clean gate, from zero
  multistart      Decoder55 from four starting points (defect shift 0, 0.3, 0.6, 0.9); best training objective wins
  bagged          Decoder55 fitted on 12 bootstrap resamples of the training folds; biases / floors / gate averaged
                  (the 21 thresholds are fitted on a step function; averaging removes the part that is noise)

Usage:  python code/r55/decoder_select.py --set primary
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
from decoder55 import Decoder55  # noqa: E402

K = "doc2_strict_total"


def shifted(p, b):
    q = p.copy()
    q[:, :, 1:] *= np.exp(b)
    return c.renorm(q)


def fit_multistart(p, g):
    best = None
    for b in (0.0, 0.3, 0.6, 0.9):
        d = Decoder55(K)
        d.bias[:, 1:] += b
        d.fit(p, g)
        o = d._obj(c.renorm(p), g)
        if best is None or o > best[0]:
            best = (o, d)
    return best[1]


def fit_bagged(p, g, grp, n=12, seed=0):
    rs = np.random.RandomState(seed)
    ug, inv = np.unique(grp, return_inverse=True)
    mem = [np.where(inv == i)[0] for i in range(len(ug))]
    ds = []
    for _ in range(n):
        idx = np.concatenate([mem[i] for i in rs.randint(0, len(ug), len(ug))])
        ds.append(Decoder55(K).fit(p[idx], g[idx]))
    d = Decoder55(K)
    d.bias = np.mean([x.bias for x in ds], 0)
    d.floor = np.median([x.floor for x in ds], 0)
    d.t_clean = float(np.median([x.t_clean for x in ds]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="primary")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    from final_engine import SETS, oof_pool
    names, gold, fold, p, _ = oof_pool(SETS[a.set])
    grp = c.load_split()[3]
    pred = {k: np.zeros_like(gold) for k in ("argmax", "shift", "decoder", "multistart", "bagged")}
    bs = []
    for f in range(8):
        tr, te = fold != f, fold == f
        pred["argmax"][te] = Decoder55().predict(p[te])
        b = max((0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8), key=lambda v: c.score(Decoder55().predict(shifted(p[tr], v)), gold[tr])[K])
        bs.append(b)
        pred["shift"][te] = Decoder55().predict(shifted(p[te], b))
        pred["decoder"][te] = Decoder55(K).fit(p[tr], gold[tr]).predict(p[te])
        pred["multistart"][te] = fit_multistart(p[tr], gold[tr]).predict(p[te])
        pred["bagged"][te] = fit_bagged(p[tr], gold[tr], grp[tr], seed=f).predict(p[te])
        print(f"[decoder_select] fold {f} done (shift b = {b})", flush=True)
    ug, inv = np.unique(grp, return_inverse=True)
    mem = [np.where(inv == i)[0] for i in range(len(ug))]
    rs = np.random.RandomState(0)
    boots = [np.concatenate([mem[i] for i in rs.randint(0, len(ug), len(ug))]) for _ in range(a.bootstrap)]
    S = {k: np.array([c.score(v[b], gold[b])[K] for b in boots]) for k, v in pred.items()}
    rep = {}
    print(f"=== decoder_select | set {a.set} | nested | paired bootstrap over videos vs 'decoder' ===")
    for k, v in pred.items():
        e = c.score(v, gold)
        d = S[k] - S["decoder"]
        print(f"  {k:11s} total {e[K]:6.2f} (pos {e['doc2_S_pos']:5.2f} p0 {e['doc2_S_p0']:5.2f} clean {e['doc2_Clean2']:5.2f}) | "
              f"vs decoder {d.mean():+.2f} CI90 [{np.quantile(d, .05):+.2f}, {np.quantile(d, .95):+.2f}] P(not better)={(d <= 0).mean():.2f}")
        rep[k] = dict(total=float(e[K]), pos=float(e["doc2_S_pos"]), p0=float(e["doc2_S_p0"]), clean=float(e["doc2_Clean2"]),
                      d_vs_decoder=float(d.mean()), ci90=[float(np.quantile(d, .05)), float(np.quantile(d, .95))])
    rep["shift_b_per_fold"] = bs
    json.dump(rep, open(os.path.join(REPO, f"results/r55/gates/decoder_select_{a.set}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
