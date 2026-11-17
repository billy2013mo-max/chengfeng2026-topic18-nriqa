#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/gate_aware_decoder.py — choose the operating point for the competition's pass/fail gates, not only for the total.

The statement sets four conditions that must ALL hold (any miss = not passed):
    objective total > 40,   positive-defect score >= 15 / 25,   P0 score >= 13 / 20,   clean score >= 12 / 20.
The ensemble sits at positive 15.5 (reference upload of s1: 15.08), P0 16.3, clean 19.6.  The positive gate has
0.5 points of margin; the clean gate has 7.6.  Decoder55 maximises the total, which is indifferent between
"call fewer defects, keep clean at 19.6" and "call more defects, clean at 19.0" (offline the two trade one for
one), so it leaves the binding gate with almost no margin: on 654-image sets drawn from the labelled data the
positive score falls below 15 in about 6 % of draws, before any distribution shift.

This script fits the same decoder with the clean term capped — utility = 25*S_pos + 20*S_p0 + 20*min(S_clean, cap)
— so nothing is gained by pushing clean above the cap and the freed budget goes to the positive labels.  Everything
is nested (thresholds of a fold are fitted on the other seven) and uses out-of-fold probabilities only; no test image
and no online score is involved.  The choice of cap is a decision about risk under the published rules.

Usage:
  python code/r55/gate_aware_decoder.py --set primary --caps 1.0 0.95 0.925 0.90        # evaluate (nested)
  python code/r55/gate_aware_decoder.py --set primary --save-cap 0.925                    # fit on all OOF and save
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
from decoder55 import Decoder55  # noqa: E402

K = "doc2_strict_total"
GATES = dict(pos=15.0, p0=13.0, clean=12.0, total=40.0)


class GateAwareDecoder(Decoder55):
    """Decoder55 whose objective does not reward the clean score above `cap` (as a fraction of its 20 points)."""
    def __init__(self, cap: float = 1.0, init_shift: float = 0.0):
        super().__init__("doc2_strict_total")
        self.cap = float(cap)
        # Coordinate ascent on one bias at a time cannot leave the "total-optimal" point (each single move loses more
        # clean than it gains); starting every defect grade `init_shift` higher puts the search on the defect-prone side.
        self.bias[:, 1:] += float(init_shift)

    def _obj(self, s, gold) -> float:
        ev = c.score(self.predict(s), gold)
        return float(ev["doc2_S_pos"] + ev["doc2_S_p0"] + min(ev["doc2_Clean2"], 20.0 * self.cap))


def nested(p, gold, fold, cap, init_shift=0.0):
    pred = np.zeros_like(gold)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        pred[te] = GateAwareDecoder(cap, init_shift).fit(p[tr], gold[tr]).predict(p[te])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="primary")
    ap.add_argument("--caps", type=float, nargs="*", default=[1.0, 0.95, 0.925, 0.90])
    ap.add_argument("--n-final", type=int, default=654)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--save-cap", type=float, default=None)
    ap.add_argument("--init-shift", type=float, nargs="*", default=[0.0])
    a = ap.parse_args()
    from final_engine import OUT, SETS, oof_pool, weights
    names, gold, fold, p, _ = oof_pool(SETS[a.set])
    if a.save_cap is not None:
        dec = GateAwareDecoder(a.save_cap, a.init_shift[0]).fit(p, gold)
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, f"{a.set}_decoder_gate{int(round(a.save_cap * 1000))}_s{int(round(a.init_shift[0] * 10))}.pkl")
        pickle.dump(dict(decoder=dec, members=SETS[a.set], weights=[float(x) for x in weights(SETS[a.set])], cap=a.save_cap), open(path, "wb"))
        ev = c.score(dec.predict(p), gold)
        print(f"[gate_aware] saved {path} | fitted on all OOF (optimistic): total {ev[K]:.2f} pos {ev['doc2_S_pos']:.2f} p0 {ev['doc2_S_p0']:.2f} clean {ev['doc2_Clean2']:.2f}")
        return
    grp = c.load_split()[3]
    ug, inv = np.unique(grp, return_inverse=True)
    mem = [np.where(inv == i)[0] for i in range(len(ug))]
    rs = np.random.RandomState(0)
    subs = []
    for _ in range(a.draws):
        idx = []
        for g in rs.permutation(len(ug)):
            idx += list(mem[g])
            if len(idx) >= a.n_final:
                break
        subs.append(np.array(idx[:a.n_final]))
    rep = {}
    print(f"=== gate-aware decoding | set {a.set} | nested OOF | {a.draws} draws of {a.n_final} images ===")
    print(f"{'clean cap':>9s} {'init':>5s} {'labels/img':>10s} | {'total':>6s} {'pos':>6s} {'p0':>6s} {'clean':>6s} | pos on draws (mean ± sd)  P(pos<15)  P(any gate missed)  total on draws")
    for cap, ish in [(cp, sh) for sh in a.init_shift for cp in a.caps]:
        pr = nested(p, gold, fold, cap, ish)
        e = c.score(pr, gold)
        S = np.array([[x["doc2_S_pos"], x["doc2_S_p0"], x["doc2_Clean2"], x[K]] for x in (c.score(pr[i], gold[i]) for i in subs)])
        miss = (S[:, 0] < GATES["pos"]) | (S[:, 1] < GATES["p0"]) | (S[:, 2] < GATES["clean"]) | (S[:, 3] <= GATES["total"])
        print(f"{cap * 20:9.1f} {ish:5.1f} {(pr > 0).sum(1).mean():10.2f} | {e[K]:6.2f} {e['doc2_S_pos']:6.2f} {e['doc2_S_p0']:6.2f} {e['doc2_Clean2']:6.2f} | "
              f"{S[:, 0].mean():6.2f} ± {S[:, 0].std():.2f}          {np.mean(S[:, 0] < 15):6.3f}      {miss.mean():6.3f}            {S[:, 3].mean():6.2f}", flush=True)
        rep[f"cap{cap}_init{ish}"] = dict(total=float(e[K]), pos=float(e["doc2_S_pos"]), p0=float(e["doc2_S_p0"]), clean=float(e["doc2_Clean2"]),
                             pos_draw_mean=float(S[:, 0].mean()), pos_draw_sd=float(S[:, 0].std()), p_pos_below_15=float(np.mean(S[:, 0] < 15)),
                             p_any_gate_missed=float(miss.mean()), labels_per_image=float((pr > 0).sum(1).mean()))
    od = os.path.join(REPO, "results/r55/gates")
    os.makedirs(od, exist_ok=True)
    json.dump(rep, open(os.path.join(od, f"gate_aware_{a.set}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
