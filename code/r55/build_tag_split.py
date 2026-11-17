#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_tag_split.py — shift-test folds: every fold holds out a whole SOURCE FAMILY (CPU, seconds).

The standard 8 folds are grouped by video, so every processing pipeline appears on both sides.  Here a fold is a
family of file-name patterns, i.e. a processing pipeline or a data source the model has never seen in training:

  0  pipeline tag v8L3HiV6          3  names containing the defect word (supplement batch)
  1  pipeline tag v8SlHiV4          4  other untagged frames
  2  every other pipeline tag       5/6  untagged timestamp frames + stock photos, split by video

The file name is used only to FORM EVALUATION FOLDS; it is never a model input.  The result answers one question:
how much lower is the score on an unseen pipeline than the video-grouped OOF suggests?  It is for reporting, never
for model selection.  Output: data/r55_tagsplit/fold_{0..6}.csv (same columns as data/qingyu_splits) + folds.csv.
"""
import csv
import hashlib
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "data/qingyu_splits")
OUT = os.path.join(REPO, "data/r55_tagsplit")
TAG_RE = re.compile(r"_(v\d+[A-Za-z0-9]*?)(?:\.mp4|_|\.)")


def family(name: str, group: str) -> int:
    m = TAG_RE.search(name)
    if m:
        return {"v8L3HiV6": 0, "v8SlHiV4": 1}.get(m.group(1), 2)
    if re.search(r"[一-鿿]", name):
        return 3
    if re.search(r"\d+\.\d+s\.png$", name) or re.match(r"^[a-z]+_\d+\.jpe?g$", name):
        return 5 + int(hashlib.md5(group.encode()).hexdigest(), 16) % 2
    return 4


def main():
    rows = list(csv.DictReader(open(os.path.join(SRC, "fold_0.csv"), encoding="utf-8")))
    fields = list(rows[0].keys())
    os.makedirs(OUT, exist_ok=True)
    fam = {r["name"]: family(r["name"], r["group_id"]) for r in rows}
    # a video must not straddle folds: give every group the family of its first row
    gfam = {}
    for r in rows:
        gfam.setdefault(r["group_id"], fam[r["name"]])
    with open(os.path.join(OUT, "folds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "fold"])
        for r in rows:
            w.writerow([r["name"], gfam[r["group_id"]]])
    for k in range(7):
        n_val = 0
        with open(os.path.join(OUT, f"fold_{k}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                is_val = gfam[r["group_id"]] == k
                n_val += is_val
                w.writerow(dict(r, split="val" if is_val else "train"))
        clean = sum(1 for r in rows if gfam[r["group_id"]] == k and str(r.get("is_clean", "0")) == "1")
        print(f"fold {k}: val={n_val} train={len(rows) - n_val} clean_in_val={clean}")


if __name__ == "__main__":
    main()
