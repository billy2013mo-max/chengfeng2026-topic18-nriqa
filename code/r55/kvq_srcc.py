#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/kvq_srcc.py — external, label-leak-free check: do the predicted defect grades track human quality scores
on a public set the models never saw?  (CPU)

KVQ (Kuaishou short-video quality, CVPR 2024) gives one mean opinion score (1-5) per video.  Nothing in this repo
was trained on those scores.  For every model the script reduces the (6, 4) grade probabilities of a frame to one
severity number and reports its Spearman correlation with the video's MOS:

  severity   max over the six dimensions of the expected grade (0 = clean ... 3 = P0)      [primary statistic]
  p_defect   1 - product over dimensions of P(clean)                                        [secondary]

The statistic was fixed before any engine result was seen; it is not tuned.  MOS is a different label from the
six defect grades (a video can be dull but defect-free), so the absolute value is not a score for the competition
metric; DIFFERENCES between models on the same frames are what matters (paired bootstrap over videos).

Inputs: one or more `name=path.npz` with arrays `names` and either `pooled`/`p_<member>` (final_engine.py output)
or `probs_6x4` (infer_tier1_ckpt.py output).  `--teacher` adds the 6-tower fold-teacher mean of aux_teacher_kvq.npz.

Usage:
  python code/r55/kvq_srcc.py --engine results/r55/kvq_val/primary_val800_probs.npz
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))


def severity(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum(-1, keepdims=True)
    return (p * np.arange(4)).sum(-1).max(1), 1.0 - np.prod(p[:, :, 0], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="", help="final_engine.py *_probs.npz (per-member + pooled probabilities)")
    ap.add_argument("--extra", nargs="*", default=[], help="name=path.npz with names + probs_6x4")
    ap.add_argument("--teacher", action="store_true", help="add the fold-teacher ensemble (results/r55/aux_teacher_kvq.npz)")
    ap.add_argument("--frames-csv", default="data/r55_aux/kvq_frames.csv")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--name", default="kvq_val")
    a = ap.parse_args()

    meta = {r["name"]: (float(r["mos"]), r["video"]) for r in csv.DictReader(open(os.path.join(REPO, a.frames_csv), encoding="utf-8"))}
    models = {}
    names = None
    if a.engine:
        z = np.load(os.path.join(REPO, a.engine), allow_pickle=True)
        names = [str(n) for n in z["names"]]
        for k in z.files:
            if k.startswith("p_"):
                models[k[2:]] = z[k]
        if "pooled" in z.files:
            models["engine pooled"] = z["pooled"]
        mem = [k[2:] for k in z.files if k.startswith("p_")]
        if "p1" in mem and len(mem) > 2:                      # the fallback set = the same members without p1
            from final_engine import pool
            models["pooled without p1"] = pool({m: z[f"p_{m}"] for m in mem if m != "p1"}, [m for m in mem if m != "p1"])
    for item in a.extra:
        k, path = item.split("=", 1)
        z = np.load(os.path.join(REPO, path), allow_pickle=True)
        pos = {str(n): i for i, n in enumerate(z["names"])}
        names = names or [str(n) for n in z["names"]]
        models[k] = z["probs_6x4"][[pos[n] for n in names]]
    if a.teacher:
        z = np.load(os.path.join(REPO, "results/r55/aux_teacher_kvq.npz"), allow_pickle=True)
        pos = {str(n): i for i, n in enumerate(z["names"])}
        names = names or [str(n) for n in z["names"]]
        models["fold teachers (6 Tier-1 towers x 8)"] = z["probs_folds"].astype(np.float64).mean(0)[[pos[n] for n in names]]
    assert models and names

    mos = np.array([meta[n][0] for n in names])
    vid = np.array([meta[n][1] for n in names])
    uv, inv = np.unique(vid, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uv))]
    rs = np.random.RandomState(0)
    boots = [np.concatenate([members[i] for i in rs.randint(0, len(uv), len(uv))]) for _ in range(a.bootstrap)]
    print(f"=== kvq_srcc '{a.name}' | {len(names)} frames of {len(uv)} videos | MOS {mos.min():.2f}-{mos.max():.2f} ===")
    print(f"{'model':38s} {'SRCC severity':>14s} {'CI90':>18s} {'SRCC p_defect':>14s}")
    rep, S = {}, {}
    for k, p in models.items():
        s1, s2 = severity(p)
        S[k] = s1
        r1 = -spearmanr(mos, s1).correlation
        r2 = -spearmanr(mos, s2).correlation
        b = np.array([-spearmanr(mos[i], s1[i]).correlation for i in boots[:300]])
        print(f"{k:38s} {r1:14.4f} [{np.quantile(b, .05):.4f}, {np.quantile(b, .95):.4f}] {r2:14.4f}")
        rep[k] = dict(srcc_severity=float(r1), ci90=[float(np.quantile(b, .05)), float(np.quantile(b, .95))], srcc_p_defect=float(r2))
    keys = list(models)
    pairs = [(x, y) for x, y in (("pooled without p1", "engine pooled"), ("v16", "engine pooled"), ("p3", "engine pooled")) if x in S and y in S]
    for x, y in pairs:
        d = np.array([(-spearmanr(mos[i], S[y][i]).correlation) - (-spearmanr(mos[i], S[x][i]).correlation) for i in boots])
        print(f"  SRCC {y} - {x}: {d.mean():+.4f}  CI90 [{np.quantile(d, .05):+.4f}, {np.quantile(d, .95):+.4f}]  P(<=0)={(d <= 0).mean():.2f}")
        rep[f"{y} - {x}"] = dict(delta=float(d.mean()), ci90=[float(np.quantile(d, .05)), float(np.quantile(d, .95))])
    od = os.path.join(REPO, "results/r55/kvq_val")
    os.makedirs(od, exist_ok=True)
    json.dump(rep, open(os.path.join(od, f"{a.name}_srcc.json"), "w"), indent=2)
    _ = keys


if __name__ == "__main__":
    main()
