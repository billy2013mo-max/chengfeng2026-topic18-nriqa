#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_rank_tuples.py — same-scene degradation ladders at NATIVE resolution (CPU only).

For a labelled seed image x and one dimension d we write three versions x1, x2, x3 with strictly
increasing strength of ONE global operator.  The only supervision they carry is the ordering
    severity_d(x) <= severity_d(x1) <= severity_d(x2) <= severity_d(x3)
which holds whatever grade x has.  They are NEVER given a P0/P1/P2 label by recipe (that is what made the
V17 synthetic run lose S_p0), and the seed does not have to be clean.

Differences from the V17 pipeline (see docs/reports/NR_IQA_Synth_Data_and_54Plus_Breakthrough_Report.md):
  * no downsampling to 960 px — operators run at native size, strengths scale with min(h, w) / 1080
  * only GLOBAL operators, so every physics-guided 448 crop shows the degradation
  * the tuple stays together in one batch (train_testbed.py) and only a ranking loss is applied

  dim 0 模糊      gauss | defocus | resample
  dim 1 纹理损失  bilateral | nlm
  dim 2 过锐      usm | usm2 (fine + halo radius)

Usage:  python code/r55/build_rank_tuples.py [--per-dim 500] [--workers 48] [--max-side 2000]
"""
import argparse
import os
import random
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import AUX_IMG, REPO, append_manifest, load_split_rows, read_manifest  # noqa: E402

DIMS = ["模糊", "纹理损失", "过锐"]          # column names in the split csv
DIM_INDEX = {"模糊": 0, "纹理损失": 1, "过锐": 2}  # index in qingyu DIMS
DIM_TAG = {"模糊": "blur", "纹理损失": "tex", "过锐": "sharp"}
OPS = {"模糊": ["gauss", "defocus", "resample"], "纹理损失": ["bilateral", "nlm"], "过锐": ["usm", "usm2"]}
IMG_DIRS = ["data/images", "data/images_supp"]


def find_img(name):
    for d in IMG_DIRS:
        p = os.path.join(REPO, d, name)
        if os.path.exists(p):
            return p
    return None


def _disk(r):
    k = int(np.ceil(r)) * 2 + 1
    y, x = np.ogrid[-(k // 2):k // 2 + 1, -(k // 2):k // 2 + 1]
    m = (x * x + y * y <= r * r).astype(np.float32)
    return m / m.sum()


def _usm(img, sigma, amount):
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigma)
    return np.clip(f + amount * (f - blur), 0, 255).astype(np.uint8)


def apply_op(img, op, level, sc, jit):
    """level in {1,2,3}; sc = min(h,w)/1080; jit in [0.85,1.15] is shared by the tuple so order is kept."""
    i = level - 1
    if op == "gauss":
        return cv2.GaussianBlur(img, (0, 0), [0.8, 1.6, 3.0][i] * sc * jit)
    if op == "defocus":
        return cv2.filter2D(img, -1, _disk(max(1.0, [1.5, 3.0, 5.0][i] * sc * jit)))
    if op == "resample":
        f = [0.6, 0.4, 0.25][i] / jit
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if op == "bilateral":
        d = int(round(7 * sc)) | 1
        out = img
        for _ in range([1, 2, 4][i]):
            out = cv2.bilateralFilter(out, d, [20, 35, 55][i] * jit, 6 * sc)
        return out
    if op == "nlm":
        hh = [4, 8, 14][i] * jit
        return cv2.fastNlMeansDenoisingColored(img, None, hh, hh, 5, 15)
    if op == "usm":
        return _usm(img, 1.2 * sc, [0.8, 1.8, 3.5][i] * jit)
    if op == "usm2":
        return _usm(_usm(img, 0.8 * sc, [0.5, 1.0, 1.8][i] * jit), 3.0 * sc, [0.3, 0.7, 1.4][i] * jit)
    raise ValueError(op)


def work(job):
    name, fold, dim, op, jit, tid, max_side = job
    p = find_img(name)
    if p is None:
        return []
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or max(img.shape[:2]) > max_side or min(img.shape[:2]) < 448:
        return []
    sc = min(img.shape[:2]) / 1080.0
    stem = os.path.splitext(name)[0]
    rows = []
    for lv in (1, 2, 3):
        out_name = f"r55rk_{DIM_TAG[dim]}_{op}_L{lv}__{stem}.png"
        out_p = os.path.join(AUX_IMG, out_name)
        if not os.path.exists(out_p):
            ok, buf = cv2.imencode(".png", apply_op(img, op, lv, sc, jit), [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if not ok:
                return []
            buf.tofile(out_p + ".tmp")
            os.replace(out_p + ".tmp", out_p)
        rows.append(dict(name=out_name, kind="rank", seed=name, seed_fold=fold, tuple_id=tid,
                         dim=DIM_INDEX[dim], level=lv, op=op))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dim", type=int, default=500, help="seeds per dimension (x3 images each)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--max-side", type=int, default=2000, help="skip larger seeds (disk; 4K PNGs are ~20 MB)")
    ap.add_argument("--seed", type=int, default=55)
    a = ap.parse_args()
    os.makedirs(AUX_IMG, exist_ok=True)
    cv2.setNumThreads(1)
    rng = random.Random(a.seed)
    split = load_split_rows()
    done = {r["tuple_id"] for r in read_manifest()}
    jobs = []
    for dim in DIMS:
        # seeds that are not already P0 in this dimension leave room above them; balanced over folds
        cand = [r for r in split if int(r[dim]) < 3]
        rng.shuffle(cand)
        by_fold = {}
        for r in cand:
            by_fold.setdefault(r["fold"], []).append(r)
        picked, k = [], 0
        while len(picked) < a.per_dim and any(by_fold.values()):
            f = sorted(by_fold)[k % len(by_fold)]
            if by_fold[f]:
                picked.append(by_fold[f].pop())
            k += 1
        for r in picked:
            op = rng.choice(OPS[dim])
            tid = f"rk_{DIM_TAG[dim]}_{os.path.splitext(r['name'])[0]}"
            if tid not in done:
                jobs.append((r["name"], int(r["fold"]), dim, op, rng.uniform(0.85, 1.15), tid, a.max_side))
    print(f"[rank-tuples] {len(jobs)} tuples to build with {a.workers} workers", flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for i, out in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            rows.extend(out)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    n = append_manifest(rows)
    print(f"[rank-tuples] wrote {n} new images ({n // 3} tuples) -> {AUX_IMG}")


if __name__ == "__main__":
    main()
