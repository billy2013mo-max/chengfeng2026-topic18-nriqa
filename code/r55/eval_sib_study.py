#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_sib_study.py — does a tower predict a frame better when OTHER frames of the same video were in its
training data?  (CPU)

Context: 95 of the 150 practice images come from a video that also has labelled training frames, while the standard
folds never show the model the video it is validated on.  `build_sib_split.py` makes folds where one frame of each
multi-frame video is validated and its other frames are trained on (variants A and B hold out different frames).

Towers compared on exactly the same validation frames (pe_core_large, same trainer and seed, `--select fixed`):
    grouped   r55_fixed_s<seed>     standard folds: the video is unseen
    exposed   r55_sibxA/B_s<seed>   the frame's siblings (with their labels) are in the training data
Reported for (a) the held-out frames of multi-frame videos and (b) frames of single-frame videos (control: nothing
changes for them except ~440 more training rows): RPS per defect, paired cluster bootstrap over videos.

Usage: python code/r55/eval_sib_study.py [--seed 2026]
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
from decoder55 import argmax_decode  # noqa: E402
from probe_eval import DIMS, rps_dim  # noqa: E402

TOW = os.path.join(REPO, "results/next55/towers")


def vid(n):
    m = re.match(r"^(?:\d{3}_)?([0-9a-f]{8,}|\d{9,})", n)
    return m.group(1) if m else n


def load(tag, seed, names, bb="pe_core_large"):
    pos = {n: i for i, n in enumerate(names)}
    P = np.full((len(names), 6, 4), np.nan)
    for f in range(8):
        p = os.path.join(TOW, f"r55_{tag}_s{seed}", bb, f"fold_{f}_probs.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            for n, q in zip(z["va_names"], z["va_probs_6x4"]):
                P[pos[str(n)]] = q
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    names, gold, fold, _ = c.load_split()
    V = np.array([vid(n) for n in names])
    cnt = {}
    for v in V:
        cnt[v] = cnt.get(v, 0) + 1
    multi = np.array([cnt[v] > 1 for v in V])
    G = load("fixed", a.seed, names)
    rs = np.random.RandomState(0)
    rep = {}
    rows_m, rows_s = [], []           # (video, per-defect RPS grouped, per-defect RPS exposed, frame index, variant)
    for var in ("A", "B"):
        E = load(f"sibx{var}", a.seed, names)
        ok = np.isfinite(E).all(axis=(1, 2)) & np.isfinite(G).all(axis=(1, 2))
        if not ok.any():
            print(f"variant {var}: no finished folds")
            continue
        Gn, En = c.renorm(np.where(ok[:, None, None], G, 0.25)), c.renorm(np.where(ok[:, None, None], E, 0.25))
        rg = np.stack([rps_dim(Gn[:, d], gold[:, d], d) for d in range(6)], 1)
        re_ = np.stack([rps_dim(En[:, d], gold[:, d], d) for d in range(6)], 1)
        for i in np.where(ok)[0]:
            (rows_m if multi[i] else rows_s).append((V[i], rg[i], re_[i], i, var))
        m = ok & multi
        print(f"variant {var}: folds done {len(set(fold[ok]))} | held-out frames of multi-frame videos {int(m.sum())} | single-frame {int((ok & ~multi).sum())}"
              f" | argmax Doc2S on the multi-frame hold-outs: grouped {c.score(argmax_decode(Gn[m]), gold[m])['doc2_strict_total']:.2f}"
              f" exposed {c.score(argmax_decode(En[m]), gold[m])['doc2_strict_total']:.2f}")
    for lab, rows in (("frames whose siblings were trained on (multi-frame videos)", rows_m), ("frames of single-frame videos (control)", rows_s)):
        if not rows:
            continue
        vids = np.array([r[0] for r in rows])
        rg = np.stack([r[1] for r in rows])
        re_ = np.stack([r[2] for r in rows])
        ug, inv = np.unique(vids, return_inverse=True)
        mem = [np.where(inv == k)[0] for k in range(len(ug))]
        boots = [np.concatenate([mem[j] for j in rs.randint(0, len(ug), len(ug))]) for _ in range(a.bootstrap)]
        print(f"\n--- {lab}: {len(rows)} frame evaluations, {len(ug)} videos ---")
        print(f"{'':10s} " + " ".join(f"{x:>9s}" for x in DIMS) + f" {'mean':>9s}")
        print(f"{'grouped':10s} " + " ".join(f"{rg[:, d].mean():9.5f}" for d in range(6)) + f" {rg.mean():9.5f}")
        print(f"{'exposed':10s} " + " ".join(f"{re_[:, d].mean():9.5f}" for d in range(6)) + f" {re_.mean():9.5f}")
        out = []
        for d in list(range(6)) + [-1]:
            x = (re_[:, d] - rg[:, d]) if d >= 0 else (re_.mean(1) - rg.mean(1))
            dd = np.array([x[b].mean() for b in boots])
            nm = DIMS[d] if d >= 0 else "mean"
            out.append(f"{nm} {x.mean():+.5f} [{np.quantile(dd, .05):+.5f},{np.quantile(dd, .95):+.5f}]")
            rep[f"{lab[:6]}:{nm}"] = dict(delta=float(x.mean()), ci90=[float(np.quantile(dd, .05)), float(np.quantile(dd, .95))],
                                         rel=float(x.mean() / max(1e-9, (rg[:, d].mean() if d >= 0 else rg.mean()))))
        print("  exposed - grouped (CI90): " + " | ".join(out))
        print(f"  relative change of mean RPS: {100 * (re_.mean() - rg.mean()) / rg.mean():+.1f} %")
    od = os.path.join(REPO, "results/r55/gates")
    os.makedirs(od, exist_ok=True)
    json.dump(rep, open(os.path.join(od, f"sib_study_s{a.seed}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
