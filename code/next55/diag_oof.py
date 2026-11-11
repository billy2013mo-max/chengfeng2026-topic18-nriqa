#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/diag_oof.py  — Step 0 / step 2 diagnostic (CPU only, ~3-10 min)

For every tower in the OOF bank and for simple ensembles, reports under the SAME protocol:
  * argmax decode (no tuning)                 -> raw separability
  * nested leave-one-fold-out Decoder55        -> what a deployable decoder actually gets
  * per-fold mean ± std (fold ≈ 447 imgs ≈ the 654-image final set scale)
  * per-channel F1 / P / R table + "which channel costs the most points"

Usage:
  python code/next55/diag_oof.py [--only qy_pe_core_large v16_full8 ...] [--objective blend] [--no-combos]
Writes results/next55/diag_oof.json and prints a table.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common55 import OUT_DIR, build_bank, load_split, logit_pool, score, score_line, stack_oof  # noqa: E402
from decoder55 import argmax_decode, nested_oof_decode  # noqa: E402


def per_fold(pred, gold, fold, key):
    v = [score(pred[fold == f], gold[fold == f])[key] for f in np.unique(fold)]
    return float(np.mean(v)), float(np.std(v))


def channel_table(ev):
    rows = []
    for c in ev["per_14"]:
        rows.append((c["channel"], c["F1"], c["P"], c["R"], c["tp"], c["fp"], c["fn"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these tower keys (default: all discovered)")
    ap.add_argument("--objective", default="doc1_final", choices=["doc1_final", "blend", "doc1_total", "doc2_strict_total"])
    ap.add_argument("--no-combos", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    names, gold, fold, _ = load_split()
    bank = build_bank(args.only)
    keys = sorted(bank)
    S = stack_oof(bank, keys, names)
    print(f"[diag] N={len(names)} clean={(gold == 0).all(1).sum()} towers={keys}")

    report = {"towers": {}, "ensembles": {}}
    for i, k in enumerate(keys):
        ev_a = score(argmax_decode(S[i]), gold)
        pred_n = nested_oof_decode(S[i], gold, fold, args.objective)
        ev_n = score(pred_n, gold)
        m1, s1 = per_fold(pred_n, gold, fold, "doc1_total")
        m2, s2 = per_fold(pred_n, gold, fold, "doc2_strict_total")
        print(f"\n== {k}\n   argmax : {score_line(ev_a)}\n   nested : {score_line(ev_n)}\n"
              f"   per-fold Doc1 {m1:.2f}±{s1:.2f}  Doc2S {m2:.2f}±{s2:.2f}")
        report["towers"][k] = {"argmax": {x: ev_a[x] for x in ("doc1_total", "doc2_strict_total")},
                               "nested": {x: ev_n[x] for x in ("doc1_final", "doc1_total", "doc2_strict_total", "blend")},
                               "fold_doc1": [m1, s1], "fold_doc2": [m2, s2]}

    combos = [tuple(keys)]
    if not args.no_combos and len(keys) <= 7:
        for r in range(2, len(keys)):
            combos += list(itertools.combinations(keys, r))
    best = None
    for cb in combos:
        idx = [keys.index(c) for c in cb]
        ens = logit_pool([S[i] for i in idx])
        ev_a = score(argmax_decode(ens), gold)
        report["ensembles"]["+".join(cb)] = {"argmax_blend": ev_a["blend"]}
        if best is None or ev_a["blend"] > best[1]:
            best = (cb, ev_a["blend"])
    print(f"\n[diag] best argmax ensemble subset: {best[0]}  blend={best[1]:.3f}")

    for tag, cb in (("ALL", tuple(keys)), ("BEST_SUBSET", best[0])):
        ens = logit_pool([S[keys.index(c)] for c in cb])
        pred_n = nested_oof_decode(ens, gold, fold, args.objective)
        ev = score(pred_n, gold)
        m1, s1 = per_fold(pred_n, gold, fold, "doc1_total")
        m2, s2 = per_fold(pred_n, gold, fold, "doc2_strict_total")
        print(f"\n== ENSEMBLE[{tag}] {cb}\n   nested : {score_line(ev)}\n"
              f"   per-fold Doc1 {m1:.2f}±{s1:.2f}  Doc2S {m2:.2f}±{s2:.2f}")
        print("   channel            F1     P      R     tp   fp   fn   (lost pts vs F1=1 @Doc2 S_pos)")
        for ch, f1, p, r, tp, fp, fn in channel_table(ev):
            print(f"   {ch:14s} {f1:6.3f} {p:6.3f} {r:6.3f} {tp:5d}{fp:5d}{fn:5d}   {25.0 * (1 - f1) / 14:5.2f}")
        report["ensembles"][f"NESTED_{tag}"] = {"members": list(cb),
                                                "doc1": ev["doc1_total"], "doc2_strict": ev["doc2_strict_total"],
                                                "fold_doc1": [m1, s1], "fold_doc2": [m2, s2],
                                                "per_14": ev["per_14"]}

    out = os.path.join(OUT_DIR, "diag_oof.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n[diag] wrote {out}")


if __name__ == "__main__":
    main()
