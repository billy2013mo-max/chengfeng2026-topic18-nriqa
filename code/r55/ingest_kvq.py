#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/ingest_kvq.py — frames of the public KVQ short-video set as an IN-DOMAIN unlabelled pool (CPU).

KVQ (Lu et al., CVPR 2024, https://lixinustc.github.io/projects/KVQ/): 4,200 Kuaishou short videos — 600 user
uploads and 3,600 versions produced by the platform's pre-processing, transcoding and enhancement workflows —
each with a mean opinion score (1-5) from professional raters.  The training split (2,926 videos) is used here.

Why this pool and not KonIQ/Inter4K: the labelled data is short-video frames, mostly outputs of processing
pipelines; KVQ is the same platform and the same kind of processing, so its frames are in-domain.

What is written (kept apart from data/r55_aux/manifest.csv so the two pools never mix):
  data/r55_aux/images/r55kvq_<video>_<k>.png     native-resolution frames, k = 0,1 at 1/3 and 2/3 of the clip
  data/r55_aux/manifest_kvq.csv                  rows kind=kvq, seed_fold=-1, op=mos<score>
  data/r55_aux/kvq_frames.csv                    name, video, mos, width, height

Frames get NO grade here.  MOS is kept only as side information (to pick candidates for "processed and still clean"
negatives, which additionally need the fold-teacher certificate).  Near-duplicates of any labelled or practice image
are dropped with the same 64-bit difference hash as ingest_external_pool.py; processed versions of one clip are kept.

Usage:  python code/r55/ingest_kvq.py [--zip data/r55_aux/raw/kvq/train_video.zip] [--frames 2] [--workers 32]
"""
import argparse
import csv
import os
import sys
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import AUX_DIR, AUX_IMG, REPO, append_manifest  # noqa: E402
from ingest_external_pool import PROTECTED, HashIndex, dhash, imread, IMG_EXT  # noqa: E402

MAN_KVQ = os.path.join(AUX_DIR, "manifest_kvq.csv")
FRAMES_CSV = os.path.join(AUX_DIR, "kvq_frames.csv")


def grab(job):
    path, n_frames = job
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for k in range(n_frames):
        idx = int(n * (k + 1) / (n_frames + 1)) if n > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, im = cap.read()
        if ok and im is not None:
            out.append((k, im))
    cap.release()
    return path, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join(AUX_DIR, "raw/kvq/train_video.zip"))
    ap.add_argument("--labels", default=os.path.join(AUX_DIR, "raw/kvq/train_data.csv"))
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--min-side", type=int, default=448)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(AUX_IMG, exist_ok=True)
    vdir = os.path.join(os.path.dirname(a.zip), "videos")
    if not os.path.isdir(os.path.join(vdir, "train")):
        with zipfile.ZipFile(a.zip) as z:
            z.extractall(vdir)
    mos = {r["filename"]: float(r["score"]) for r in csv.DictReader(open(a.labels, encoding="utf-8"))}

    idx = HashIndex()
    for d in PROTECTED:
        full = os.path.join(REPO, d)
        for fn in sorted(os.listdir(full)) if os.path.isdir(full) else []:
            if fn.lower().endswith(IMG_EXT):
                im = imread(os.path.join(full, fn))
                if im is not None:
                    idx.add(dhash(im))
    print(f"[kvq] hash index: {idx.h.size} protected images", flush=True)

    jobs = [(os.path.join(vdir, f), a.frames) for f in sorted(mos) if os.path.exists(os.path.join(vdir, f))]
    rows, side, small, dup = [], [], 0, 0
    with Pool(a.workers) as pool:
        for path, frames in pool.imap(grab, jobs, chunksize=4):
            vid = os.path.splitext(os.path.basename(path))[0]
            key = "train/" + os.path.basename(path)
            for k, im in frames:
                if min(im.shape[:2]) < a.min_side:
                    small += 1
                    continue
                # Only labelled / practice images are excluded.  KVQ holds each source clip in several processed
                # versions; those are the point of this pool (same content, different processing), so frames are
                # NOT de-duplicated against each other.
                if idx.near(dhash(im)):
                    dup += 1
                    continue
                name = f"r55kvq_{vid}_{k}.png"
                cv2.imencode(".png", im, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(os.path.join(AUX_IMG, name))
                rows.append(dict(name=name, kind="kvq", seed="", seed_fold=-1, tuple_id="", dim=-1, level=0, op=f"mos{mos[key]:.3f}"))
                side.append(dict(name=name, video=vid, mos=f"{mos[key]:.4f}", width=im.shape[1], height=im.shape[0]))
    n = append_manifest(rows, MAN_KVQ)
    with open(FRAMES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "video", "mos", "width", "height"])
        w.writeheader()
        w.writerows(side)
    wh = np.array([[int(s["width"]), int(s["height"])] for s in side])
    print(f"[kvq] videos {len(jobs)} | kept {n} frames, dropped {small} too small, {dup} near-duplicates of labelled/practice images")
    if len(wh):
        print(f"[kvq] short side quantiles {np.percentile(wh.min(1), [0, 25, 50, 75, 100]).astype(int)} | "
              f"portrait {float((wh[:, 1] > wh[:, 0]).mean()):.2f}")


if __name__ == "__main__":
    main()
