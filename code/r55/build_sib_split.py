#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_sib_split.py — "sibling-exposed" folds (CPU, seconds).

Why: 95 of the 150 practice images come from a video that also has labelled frames in the training data (same
leading video id in the file name; checked 2026-11-12), while the standard folds (data/qingyu_splits) keep all frames
of a video together, so out-of-fold numbers describe videos the model has never seen.  These folds reproduce the
practice condition: for a video with several labelled frames only ONE frame stays in the validation fold, the other
frames of that video go to the training side of the same fold.  Single-frame videos are untouched.

Variant A / B hold out a different frame of each multi-frame video (deterministic: md5 of the video id).
The file names are used only to group frames by video, exactly as the standard grouped folds do.

Usage: python code/r55/build_sib_split.py     ->  data/r55_sibsplit_{A,B}/fold_{0..7}.csv
"""
import collections
import csv
import hashlib
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def vid(n):
    m = re.match(r"^(?:\d{3}_)?([0-9a-f]{8,}|\d{9,})", n)
    return m.group(1) if m else n


def main():
    for var, pick in (("A", 0), ("B", 1)):
        od = os.path.join(REPO, f"data/r55_sibsplit_{var}")
        os.makedirs(od, exist_ok=True)
        tot = held = 0
        for k in range(8):
            rows = list(csv.DictReader(open(os.path.join(REPO, f"data/qingyu_splits/fold_{k}.csv"), encoding="utf-8")))
            byv = collections.defaultdict(list)
            for r in rows:
                if r["split"] == "val":
                    byv[vid(r["name"])].append(r)
            for v, rs in sorted(byv.items()):
                if len(rs) < 2:
                    continue
                rs = sorted(rs, key=lambda r: hashlib.md5((v + "|" + r["name"]).encode()).hexdigest())
                keep = rs[pick % len(rs)]
                held += 1
                for r in rs:
                    if r is not keep:
                        r["split"] = "train"
                        tot += 1
            with open(os.path.join(od, f"fold_{k}.csv"), "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"[sib_split] variant {var}: {held} multi-frame videos keep one validation frame, {tot} sibling frames moved to training")


if __name__ == "__main__":
    main()
