#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/gate_plus.py — low-variance companion to gate_r55.py (CPU).

gate_r55.py compares the thresholded competition score, which moves by +-0.4 between seeds of one recipe.
This script compares the PROBABILITIES of two sides on the same out-of-fold images with proper scoring rules and
rank statistics, which need no decoder and are far less noisy, plus the subgroup the business case cares about:

  RPS        ranked probability score over the 4 ordered grades (4-grade dims) / Brier (P0-only dims); lower = better
  NLL        negative log-likelihood of the human grade; lower = better
  AUC det    clean vs any positive, per dim (score = 1 - P(clean))
  AUC grade  adjacent grades only: P2 vs P1 and P1 vs P0 (score = P(grade >= upper)); this is the grading skill
  FP proc    false-positive rate on clean images that carry a processing-pipeline tag in the file name
             (the tag is used here ONLY to form an evaluation subgroup, never as model input)

Differences use a paired cluster bootstrap over videos (group_id), so sibling frames are resampled together.
It never decides alone: it is evidence next to gate_r55.py.  Works on any common subset of folds.

Usage:
  python code/r55/gate_plus.py --name rcrop --base <dir> [<dir> ...] --exp <dir> [<dir> ...]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "code/next55"))
import common55 as c  # noqa: E402
from decoder55 import nested_oof_decode  # noqa: E402
from gate_r55 import folds_of, load_dir  # noqa: E402

FULL, P0ONLY = (0, 1, 2, 4), (3, 5)
DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
TAG_RE = re.compile(r"_(v\d+[A-Za-z0-9]*?)(?:\.mp4|_|\.)")


def auc(score, pos):
    """Mann-Whitney AUC; nan if one class is empty."""
    pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(score)                               # average ranks for ties
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def per_image(p, gold):
    """Per-image RPS and NLL, averaged over the six dims.  p: (N,6,4) renormalised, gold: (N,6)."""
    n = len(gold)
    rps, nll = np.zeros(n), np.zeros(n)
    for d in FULL:
        cdf_p = np.cumsum(p[:, d], axis=1)[:, :3]
        cdf_y = (gold[:, d][:, None] <= np.arange(3)[None, :]).astype(float)
        rps += ((cdf_p - cdf_y) ** 2).sum(1) / 3.0
        nll += -np.log(np.clip(p[np.arange(n), d, gold[:, d]], 1e-6, 1))
    for d in P0ONLY:
        q = p[:, d, 3] / np.clip(p[:, d, 0] + p[:, d, 3], 1e-9, None)
        y = (gold[:, d] == 3).astype(float)
        rps += (q - y) ** 2
        nll += -np.log(np.clip(np.where(y == 1, q, 1 - q), 1e-6, 1))
    return rps / 6.0, nll / 6.0


def aucs(p, gold):
    det, grd = [], []
    for d in range(6):
        det.append(auc(1 - p[:, d, 0], gold[:, d] > 0))
    for d in FULL:
        for g in (2, 3):                               # (P2 vs P1), (P1 vs P0)
            m = (gold[:, d] == g - 1) | (gold[:, d] == g)
            grd.append(auc(p[m, d, g:].sum(1), gold[m, d] == g))
    return np.array(det), np.array(grd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--no-decode", action="store_true", help="skip the (slow) nested-decoder competition score")
    a = ap.parse_args()

    folds = sorted(set.intersection(*[folds_of(d) for d in a.base + a.exp]))
    names, gold, fold, grp = c.load_split()
    keep = np.isin(fold, folds)
    names = [n for n, k in zip(names, keep) if k]
    gold, fold, grp = gold[keep], fold[keep], grp[keep]

    def side(dirs):
        return c.renorm(c.mean_pool([np.stack([load_dir(d, folds)[n] for n in names]) for d in dirs]))

    pb, pe = side(a.base), side(a.exp)
    print(f"=== gate_plus '{a.name}' | folds {folds} | N={len(names)} | seeds base={len(a.base)} exp={len(a.exp)} ===")
    out = {"name": a.name, "folds": folds, "base": a.base, "exp": a.exp}

    rb, nb = per_image(pb, gold)
    re_, ne = per_image(pe, gold)
    db, gb = aucs(pb, gold)
    de, ge = aucs(pe, gold)
    clean = (gold == 0).all(1)
    proc = np.array([bool(TAG_RE.search(n)) for n in names])
    sub = clean & proc

    def fp(p):                                         # argmax decision, any dim positive
        return (p.argmax(-1) > 0).any(1)

    # paired cluster bootstrap over videos
    ug, inv = np.unique(grp, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ug))]
    rs = np.random.RandomState(0)
    boot = {k: [] for k in ("rps", "nll", "auc_grade", "auc_det")}
    for _ in range(a.bootstrap):
        idx = np.concatenate([members[i] for i in rs.randint(0, len(ug), len(ug))])
        boot["rps"].append(re_[idx].mean() - rb[idx].mean())
        boot["nll"].append(ne[idx].mean() - nb[idx].mean())
        d1, g1 = aucs(pe[idx], gold[idx])
        d0, g0 = aucs(pb[idx], gold[idx])
        boot["auc_grade"].append(np.nanmean(g1) - np.nanmean(g0))
        boot["auc_det"].append(np.nanmean(d1) - np.nanmean(d0))

    def row(lab, vb, ve, key, better_low):
        d = np.array(boot[key])
        lo, hi = np.quantile(d, [0.05, 0.95])
        p_bad = float((d >= 0).mean()) if better_low else float((d <= 0).mean())
        print(f"  {lab:<22s} base {vb:.4f}  exp {ve:.4f}  delta {ve - vb:+.4f}  CI90 [{lo:+.4f}, {hi:+.4f}]  P(no gain)={p_bad:.2f}")
        out[key] = dict(base=float(vb), exp=float(ve), delta=float(ve - vb), ci90=[float(lo), float(hi)], p_no_gain=p_bad)

    row("RPS (lower better)", rb.mean(), re_.mean(), "rps", True)
    row("NLL (lower better)", nb.mean(), ne.mean(), "nll", True)
    row("AUC adjacent grades", np.nanmean(gb), np.nanmean(ge), "auc_grade", False)
    row("AUC detection", np.nanmean(db), np.nanmean(de), "auc_det", False)
    labs = [f"{DIMS[d]} {t}" for d in FULL for t in ("P2|P1", "P1|P0")]
    print("  grade AUC  base: " + "  ".join(f"{l} {v:.3f}" for l, v in zip(labs, gb)))
    print("  grade AUC  exp : " + "  ".join(f"{l} {v:.3f}" for l, v in zip(labs, ge)))
    if sub.sum():
        print(f"  clean+processed images (n={int(sub.sum())}): flagged base {fp(pb)[sub].mean():.3f}  exp {fp(pe)[sub].mean():.3f}"
              f"   | other clean (n={int((clean & ~proc).sum())}): base {fp(pb)[clean & ~proc].mean():.3f}  exp {fp(pe)[clean & ~proc].mean():.3f}")
        out["fp_clean_processed"] = dict(n=int(sub.sum()), base=float(fp(pb)[sub].mean()), exp=float(fp(pe)[sub].mean()))
    if not a.no_decode:
        sb = c.score(nested_oof_decode(pb, gold, fold, "doc2_strict_total"), gold)
        se = c.score(nested_oof_decode(pe, gold, fold, "doc2_strict_total"), gold)
        for k, lab in (("doc2_strict_total", "Doc2S"), ("doc2_S_pos", "S_pos"), ("doc2_S_p0", "S_p0"), ("doc2_Clean2", "S_clean")):
            print(f"  {lab:8s} base {sb[k]:6.2f}   exp {se[k]:6.2f}   delta {se[k] - sb[k]:+.2f}")
        out["doc2s"] = dict(base=sb["doc2_strict_total"], exp=se["doc2_strict_total"])
    od = os.path.join(REPO, "results/r55/gates")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"{a.name}_plus.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
