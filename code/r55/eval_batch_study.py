#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_batch_study.py — release-batch study (CPU).

Question: the labelled data came in two releases, the first batch ("orig", 1,902 images, all 513 clean images) and the
supplement ("supp", 1,671 images, no clean image, defect prevalence 1.2-2x higher).  Do they follow one labelling
convention, i.e. does each batch help the other?

Towers (pe_core_large, same trainer / seed / folds, `--select fixed`):
    all        r55_fixed_s2026      trained on both batches (the standard recipe)
    origonly   r55_origonly_s2026   trained on first-batch rows only
    supponly   r55_supponly_s2026   trained on supplement rows only
Each is scored on the validation images of each batch separately: RPS per defect (lower is better), detection AUC,
adjacent-grade AUC, with a paired cluster bootstrap over videos for the differences.

Reading: if `all` is clearly better than `origonly` on first-batch images, the supplement carries usable information
for them (one convention).  If it is equal or worse for a defect, the supplement's labels of that defect do not transfer.

Usage: python code/r55/eval_batch_study.py [--seed 2026] [--backbone cradio_v4_h --tags fixed origonly]
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
from probe_eval import DIMS, rps_dim, skills  # noqa: E402

TOW = os.path.join(REPO, "results/next55/towers")


def load(tag, seed, names, bb="pe_core_large"):
    pos = {n: i for i, n in enumerate(names)}
    P = np.full((len(names), 6, 4), np.nan)
    for f in range(8):
        p = os.path.join(TOW, f"r55_{tag}_s{seed}", bb, f"fold_{f}_probs.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        for n, q in zip(z["va_names"], z["va_probs_6x4"]):
            P[pos[str(n)]] = q
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--tags", nargs="+", default=["fixed", "origonly", "supponly"])
    ap.add_argument("--backbone", default="pe_core_large")
    a = ap.parse_args()
    names, gold, fold, grp = c.load_split()
    batch = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    B = np.array([batch[n] for n in names])
    T = {t: load(t, a.seed, names, a.backbone) for t in a.tags}
    have = np.all([np.isfinite(T[t]).all(axis=(1, 2)) for t in a.tags], axis=0)
    print(f"=== release-batch study | {a.backbone} | seed {a.seed} | images with all towers: {int(have.sum())}/{len(names)} | folds done: "
          + " ".join(f"{t}:{len(set(fold[np.isfinite(T[t]).all(axis=(1, 2))]))}" for t in a.tags) + " ===")
    for t in a.tags:
        T[t] = c.renorm(np.where(np.isfinite(T[t]), T[t], 0.25))
    ug, inv = np.unique(grp, return_inverse=True)
    rs = np.random.RandomState(0)
    report = {}
    for sub in ("orig", "supp"):
        m = have & (B == sub)
        idx = np.where(m)[0]
        mem = {}
        for i in idx:
            mem.setdefault(inv[i], []).append(i)
        mem = [np.array(v) for v in mem.values()]
        boots = [np.concatenate([mem[j] for j in rs.randint(0, len(mem), len(mem))]) for _ in range(a.bootstrap)]
        print(f"\n--- validation images of batch '{sub}' (n={len(idx)}, {len(mem)} videos) ---")
        R = {t: np.stack([rps_dim(T[t][:, d], gold[:, d], d) for d in range(6)], 1) for t in a.tags}      # (N, 6)
        print(f"{'tower':10s} " + " ".join(f"{x:>9s}" for x in DIMS) + f" {'mean RPS':>9s} {'argmax':>7s}")
        for t in a.tags:
            sc = c.score(argmax_decode(T[t][idx]), gold[idx])["doc2_strict_total"]
            print(f"{t:10s} " + " ".join(f"{R[t][idx, d].mean():9.5f}" for d in range(6)) + f" {R[t][idx].mean():9.5f} {sc:7.2f}")
        for t in a.tags:
            ad, ag = [], []
            for d in range(6):
                s = skills(T[t][idx, d], gold[idx, d], d)
                ad.append(s[0])
                ag.append(np.nanmean(s[1:]) if d in (0, 1, 2, 4) else np.nan)
            print(f"  {t:10s} detection AUC " + " ".join(f"{x:.3f}" for x in ad) + " | adjacent-grade AUC " + " ".join(f"{x:.3f}" for x in ag))
        ref = a.tags[0]
        for t in a.tags[1:]:
            out = []
            for d in list(range(6)) + [-1]:
                r1 = R[t][:, d] if d >= 0 else R[t].mean(1)
                r0 = R[ref][:, d] if d >= 0 else R[ref].mean(1)
                dd = np.array([r1[b].mean() - r0[b].mean() for b in boots])
                lab = DIMS[d] if d >= 0 else "mean"
                out.append(f"{lab} {r1[idx].mean() - r0[idx].mean():+.5f} [{np.quantile(dd, .05):+.5f},{np.quantile(dd, .95):+.5f}]")
                report[f"{sub}:{t}-{ref}:{lab}"] = dict(delta=float(r1[idx].mean() - r0[idx].mean()), ci90=[float(np.quantile(dd, .05)), float(np.quantile(dd, .95))])
            print(f"  RPS {t} - {ref} (CI90): " + " | ".join(out))
    od = os.path.join(REPO, "results/r55/gates")
    os.makedirs(od, exist_ok=True)
    json.dump(report, open(os.path.join(od, f"batch_study_s{a.seed}" + ("" if a.backbone == "pe_core_large" else "_" + a.backbone) + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
