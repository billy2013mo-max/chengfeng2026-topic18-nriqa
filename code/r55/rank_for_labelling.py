#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/rank_for_labelling.py — E5: which unlabelled frames are worth an annotator's time? (CPU)

docs/REASSESSMENT_AND_PLAN_2026-11-12.md §4 E.  The learning curve says the common defects (模糊, 纹理损失, 过锐) are close to
saturated in data, while 过度生成 and 虚化突兀 gain about +0.07 label F1 per doubling and the processed-but-clean and
unseen-pipeline cases have no fix other than labels.  So labels should be spent by purpose, not by volume.

Input: a `*_probs.npz` written by final_engine.py for the candidate frames (per-member and pooled probabilities;
run `final_engine.py --template <json of candidates> --img-dir <dir> --out <json>` first).
Output: a CSV with one row per selected frame, the bucket it was selected for, and the numbers behind it.

Buckets (a frame is assigned to the first one it qualifies for; quota per bucket = --per-bucket):
  overgen_grade   过度生成 is likely present but its grade is undecided
                  (P(positive) >= 0.35 and no single positive grade holds more than 60 % of the positive mass)
  overgen_detect  过度生成 presence itself is undecided (0.25 <= P(positive) <= 0.75) or the families disagree on it
  bokeh           虚化突兀 plausible (P >= 0.25)
  huaping         花屏 undecided (0.15 <= P <= 0.85)  — corruption from a source the model has not seen
  clean_border    the frame is on the clean / not-clean border (0.25 <= P(all clean) <= 0.75)
  disagreement    the three model families disagree most (mean total-variation distance), anything else
Inside a bucket frames are ordered by family disagreement, so the most informative come first.

Usage:
  python code/r55/rank_for_labelling.py --probs results/r55/kvq_val/primary_val800_probs.npz --out results/r55/labelling/kvq_val800_shortlist.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
FAMILY = {"cradio": 0, "pe": 0, "sig": 0, "v16": 1, "p1": 2, "p3": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bucket", type=int, default=100)
    a = ap.parse_args()
    z = np.load(os.path.join(REPO, a.probs) if not os.path.isabs(a.probs) else a.probs, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    pool = z["pooled"].astype(np.float64)
    pool = pool / pool.sum(-1, keepdims=True)
    mem = [k[2:] for k in z.files if k.startswith("p_") and k[2:] in FAMILY]
    fam = []
    for f in range(3):
        ms = [z[f"p_{m}"].astype(np.float64) for m in mem if FAMILY[m] == f]
        if ms:
            p = np.mean(ms, 0)
            fam.append(p / p.sum(-1, keepdims=True))
    fam = np.stack(fam)                                             # (families, N, 6, 4)
    tv = np.zeros((len(names), 6))
    npair = 0
    for i in range(len(fam)):
        for j in range(i + 1, len(fam)):
            tv += 0.5 * np.abs(fam[i] - fam[j]).sum(-1)
            npair += 1
    tv /= max(1, npair)

    og_pos = 1 - pool[:, 4, 0]
    og_top = pool[:, 4, 1:].max(1) / np.clip(og_pos, 1e-6, None)
    p_clean = np.prod(pool[:, :, 0], axis=1)
    rules = [("overgen_grade", (og_pos >= 0.35) & (og_top <= 0.60), tv[:, 4]),
             ("overgen_detect", ((og_pos >= 0.25) & (og_pos <= 0.75)) | (tv[:, 4] >= np.quantile(tv[:, 4], 0.97)), tv[:, 4]),
             ("bokeh", pool[:, 5, 3] >= 0.25, pool[:, 5, 3]),
             ("huaping", (pool[:, 3, 3] >= 0.15) & (pool[:, 3, 3] <= 0.85), tv[:, 3]),
             ("clean_border", (p_clean >= 0.25) & (p_clean <= 0.75), tv.mean(1)),
             ("disagreement", np.ones(len(names), bool), tv.mean(1))]
    taken = np.zeros(len(names), bool)
    rows = []
    for bucket, mask, key in rules:
        idx = np.where(mask & ~taken)[0]
        idx = idx[np.argsort(-key[idx])][:a.per_bucket]
        taken[idx] = True
        for i in idx:
            rows.append(dict(name=names[i], bucket=bucket, family_disagreement=f"{tv[i].mean():.3f}", p_overgen=f"{og_pos[i]:.2f}",
                             p_bokeh=f"{pool[i, 5, 3]:.2f}", p_huaping=f"{pool[i, 3, 3]:.2f}", p_all_clean=f"{p_clean[i]:.2f}",
                             predicted="/".join(str(int(x)) for x in pool[i].argmax(-1))))
        print(f"[rank_for_labelling] {bucket:15s} {int((mask).sum()):5d} candidates, {len(idx):4d} selected")
    out = os.path.join(REPO, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[rank_for_labelling] wrote {out}: {len(rows)} of {len(names)} frames")


if __name__ == "__main__":
    main()
