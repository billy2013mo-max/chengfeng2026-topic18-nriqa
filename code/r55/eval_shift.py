#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_shift.py — how much worse is a model on a processing pipeline / source it has never seen? (CPU)

`--shift`  tower trained on data/r55_tagsplit (every fold holds out a whole source family, build_tag_split.py)
`--ref`    the same recipe trained on the standard video-grouped 8 folds

Both give one out-of-fold prediction per labelled image.  For the SAME images the script compares
  ref    the model saw other videos of that family during training   (what the usual OOF number measures)
  shift  the model saw no image of that family                       (what a new pipeline looks like)
per family and overall: RPS, NLL, detection AUC, adjacent-grade AUC, argmax Doc2S, share of clean images flagged,
share of defective images called clean.

How to read the families:
  0-2  processing-pipeline tags: the clean "unseen pipeline" estimate.
  3,4  sources that carry whole label classes (family 4 holds 209 of the 211 花屏 positives): holding them out removes
       a defect class from training, so their gap is label shift, not pipeline shift.
  5/6  one family split by video: in the shift split the model sees half of this family instead of 7/8 of it, so
       their gap is the within-family data-size effect (small, not zero).  A second `--shift` tower (another recipe on the same folds) can be compared against the
first with a paired cluster bootstrap: does the recipe shrink the gap?

Reporting only: nothing here selects a model, a threshold or a checkpoint.

Usage:
  python code/r55/eval_shift.py --ref results/next55/towers/r55_fixed_s2026/pe_core_large \
      --shift results/next55/towers/r55_shift_s2026/pe_core_large [results/next55/towers/r55_shiftrc_s2026/pe_core_large]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
from decoder55 import argmax_decode  # noqa: E402
from gate_plus import aucs, per_image  # noqa: E402

FAMILY = ["tag v8L3HiV6", "tag v8SlHiV4", "other pipeline tags", "defect-word names", "other untagged",
          "timestamp/stock A (control)", "timestamp/stock B (control)"]
K = "doc2_strict_total"


def load(d, n_folds):
    m = {}
    for f in range(n_folds):
        p = os.path.join(REPO, d, f"fold_{f}_probs.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            m.update({str(n): q for n, q in zip(z["va_names"], z["va_probs_6x4"])})
    return m


def stats(p, gold):
    rps, nll = per_image(p, gold)
    det, grd = aucs(p, gold)
    pred = argmax_decode(p)
    clean = (gold == 0).all(1)
    return dict(rps=float(rps.mean()), nll=float(nll.mean()), auc_det=float(np.nanmean(det)) if np.isfinite(det).any() else float("nan"),
                auc_grade=float(np.nanmean(grd)) if np.isfinite(grd).any() else float("nan"),
                argmax=float(c.score(pred, gold)[K]),
                clean_flagged=float((pred[clean] > 0).any(1).mean()) if clean.any() else float("nan"),
                defect_missed=float((pred[~clean] == 0).all(1).mean()) if (~clean).any() else float("nan"),
                n=int(len(gold)), n_clean=int(clean.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--shift", nargs="+", required=True)
    ap.add_argument("--folds-csv", default="data/r55_tagsplit/folds.csv")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--name", default="shift")
    a = ap.parse_args()

    names, gold, _, grp = c.load_split()
    fam = {r["name"]: int(r["fold"]) for r in csv.DictReader(open(os.path.join(REPO, a.folds_csv), encoding="utf-8"))}
    fam = np.array([fam[n] for n in names])
    ref = load(a.ref, 8)
    shifts = [load(d, 7) for d in a.shift]
    have = np.array([n in ref and all(n in s for s in shifts) for n in names])
    done = sorted(set(fam[have]))
    print(f"=== eval_shift '{a.name}' | families with predictions: {done} | N={int(have.sum())} of {len(names)} ===")
    idx = np.where(have)[0]
    nm = [names[i] for i in idx]
    g, fm, gr = gold[idx], fam[idx], grp[idx]
    P = {"ref": c.renorm(np.stack([ref[n] for n in nm]))}
    for d, s in zip(a.shift, shifts):
        P[os.path.basename(os.path.dirname(d.rstrip("/"))).replace("r55_", "")] = c.renorm(np.stack([s[n] for n in nm]))
    report = {"ref": a.ref, "shift": a.shift, "families": {}, "overall": {}}

    hdr = f"{'family':30s} {'side':16s} {'n':>5s} {'clean':>5s} {'RPS':>8s} {'NLL':>7s} {'AUCdet':>7s} {'AUCgrd':>7s} {'argmax':>7s} {'cleanFP':>8s} {'missed':>7s}"
    print(hdr)
    for f in done + ["tags", "all"]:
        m = np.ones(len(nm), bool) if f == "all" else (fm <= 2 if f == "tags" else fm == f)
        lab = "ALL" if f == "all" else ("0-2 tagged pipelines pooled" if f == "tags" else f"{f} {FAMILY[f]}")
        for side, p in P.items():
            s = stats(p[m], g[m])
            print(f"{lab:30s} {side:16s} {s['n']:5d} {s['n_clean']:5d} {s['rps']:8.5f} {s['nll']:7.4f} {s['auc_det']:7.4f} {s['auc_grade']:7.4f} "
                  f"{s['argmax']:7.2f} {s['clean_flagged']:8.3f} {s['defect_missed']:7.3f}")
            report["overall" if f == "all" else "families"].setdefault(lab, {})[side] = s

    # paired cluster bootstrap over videos: gap of each shift tower to ref, and recipe vs recipe
    ug, inv = np.unique(gr, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ug))]
    rs = np.random.RandomState(0)
    boots = [np.concatenate([members[i] for i in rs.randint(0, len(ug), len(ug))]) for _ in range(a.bootstrap)]
    R = {k: per_image(p, g)[0] for k, p in P.items()}
    keys = list(P)
    pairs = [("ref", k) for k in keys[1:]] + ([(keys[1], keys[2])] if len(keys) > 2 else [])
    for mlab, m in [("all families", np.ones(len(nm), bool)), ("tagged pipelines 0-2", fm <= 2), ("unseen families 0-4", fm <= 4), ("control families 5-6", fm >= 5)]:
        if not m.any():
            continue
        for x, y in pairs:
            d = np.array([R[y][b][m[b]].mean() - R[x][b][m[b]].mean() for b in boots])
            print(f"  dRPS {y} - {x} on {mlab:22s}: {R[y][m].mean() - R[x][m].mean():+.5f}  CI90 [{np.quantile(d, .05):+.5f}, {np.quantile(d, .95):+.5f}]  P(<=0)={(d <= 0).mean():.2f}")
            report.setdefault("paired", []).append(dict(a=x, b=y, subset=mlab, d_rps=float(R[y][m].mean() - R[x][m].mean()),
                                                        ci90=[float(np.quantile(d, .05)), float(np.quantile(d, .95))]))
    od = os.path.join(REPO, "results/r55/gates")
    os.makedirs(od, exist_ok=True)
    json.dump(report, open(os.path.join(od, f"{a.name}_shift.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
