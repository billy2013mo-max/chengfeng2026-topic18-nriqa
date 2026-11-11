#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/make_fullfit_splits.py — CPU, seconds.

Deliverable models must stay <= 30B params and run on one 4090, so an 8-fold x multi-tower
ensemble can NOT be the shipped model (V16 alone is ~4.5B per fold).  The 8-fold runs are for
validation + decoder fitting only; the shipped model is ONE refit per tower(/seed) on 100% data.

Writes
  data/next55_fullfit/qingyu/fold_0.csv               all 3573 rows split=train, + fold-0 val rows
                                                      duplicated as split=val (logging / SWA check only)
  data/next55_fullfit/v16/split_full_v16_cleanog.csv  all real + fold-0 synthetic rows as train,
                                                      fold-0 val duplicated as val/test (logging only)
  data/next55_fullfit/v16/manifest_full_v16_cleanog.csv (= manifest_f0_v16_cleanog.csv)
"""
import csv
import os
import shutil

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    out_q = os.path.join(REPO, "data/next55_fullfit/qingyu")
    out_v = os.path.join(REPO, "data/next55_fullfit/v16")
    os.makedirs(out_q, exist_ok=True)
    os.makedirs(out_v, exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(REPO, "data/qingyu_splits/fold_0.csv"), encoding="utf-8")))
    fields = list(rows[0].keys())
    with open(os.path.join(out_q, "fold_0.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, split="train"))
        for r in rows:
            if r["split"] == "val":
                w.writerow(r)
    n_tr = len(rows)

    sp = list(csv.DictReader(open(os.path.join(REPO, "data/v9_fulldata/split_f0_v16_cleanog.csv"), encoding="utf-8")))
    val_names = [r["name"] for r in sp if r["split"] == "val"]
    train_names = [r["name"] for r in sp if r["split"] == "train"]
    seen = set(train_names)
    with open(os.path.join(out_v, "split_full_v16_cleanog.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "split"])
        w.writeheader()
        for n in train_names + [n for n in val_names if n not in seen]:
            w.writerow({"name": n, "split": "train"})
        for n in val_names:
            w.writerow({"name": n, "split": "val"})
            w.writerow({"name": n, "split": "test"})
    shutil.copy(os.path.join(REPO, "data/v9_fulldata/manifest_f0_v16_cleanog.csv"), os.path.join(out_v, "manifest_full_v16_cleanog.csv"))
    n_real = sum(1 for n in train_names + val_names if not n.startswith("v9_"))
    print(f"[fullfit-splits] qingyu: {n_tr} train rows | v16: {len(train_names) + len(val_names)} train rows ({n_real} real)")


if __name__ == "__main__":
    main()
