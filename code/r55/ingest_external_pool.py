#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/ingest_external_pool.py — turn a folder of external images / videos into unlabelled pool rows (CPU).

  * images are copied as PNG; videos are sampled every --every-sec seconds (at most --max-per-video frames)
  * frames smaller than --min-side on the short side are dropped
  * near-duplicates are removed with a 64-bit difference hash, against the pool itself AND against every
    labelled image and the 150 practice images (so no evaluation image can enter the pool)
  * rows are written with kind=ext, seed_fold=-1.  They get NO label here: the fold-k teachers label them
    later (prep_aux.py) and certify_clean.py decides which ones may be used as clean negatives.

Usage:
  python code/r55/ingest_external_pool.py --src /path/to/frames_or_videos --tag lsvq [--max 4000]
  python code/r55/ingest_external_pool.py --hf-repo Shilin-LU/Inter4k --tag inter4k --max 1500   # downloads first

Candidate public sources found on the Hub on 2026-11-10 (contents and licences NOT verified — check before use):
  teowu/LSVQ-videos (UGC video, has MOS) · shuoxing/KoNViD_1k_videos (UGC video, has MOS) ·
  Shilin-LU/Inter4k (4K 60 fps clips) · danjacobellis/LSDIR (high-quality photos) · V-NovaLtd/UHD-IQA-JPG
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import AUX_DIR, AUX_IMG, REPO, append_manifest, read_manifest  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VID_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi")
PROTECTED = ["data/images", "data/images_supp", "data/test_150"]


def dhash(img_bgr) -> int:
    g = cv2.cvtColor(cv2.resize(img_bgr, (9, 8), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return int(np.packbits((g[:, 1:] > g[:, :-1]).flatten()).view(">u8")[0])


def imread(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


class HashIndex:
    def __init__(self):
        self.h = np.zeros(0, dtype=np.uint64)

    def add(self, v):
        self.h = np.append(self.h, np.uint64(v))

    def near(self, v, thr=4) -> bool:
        if self.h.size == 0:
            return False
        x = np.bitwise_xor(self.h, np.uint64(v))
        return bool((np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1) <= thr).any())


def frames_of(path, every_sec, max_per_video):
    if path.lower().endswith(IMG_EXT):
        im = imread(path)
        if im is not None:
            yield im
        return
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * every_sec)))
    k = 0
    for idx in range(step // 2, max(n, 1), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, im = cap.read()
        if not ok:
            break
        yield im
        k += 1
        if k >= max_per_video:
            break
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="")
    ap.add_argument("--hf-repo", default="", help="dataset repo to snapshot_download into data/r55_aux/raw/<tag>")
    ap.add_argument("--hf-allow", nargs="*", default=None, help="allow_patterns for the download, e.g. '*.mp4'")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max", type=int, default=4000)
    ap.add_argument("--min-side", type=int, default=720)
    ap.add_argument("--max-side", type=int, default=2160, help="larger frames are downscaled to this long side")
    ap.add_argument("--every-sec", type=float, default=3.0)
    ap.add_argument("--max-per-video", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(AUX_IMG, exist_ok=True)

    src = a.src
    if a.hf_repo:
        from huggingface_hub import snapshot_download
        src = snapshot_download(a.hf_repo, repo_type="dataset", local_dir=os.path.join(AUX_DIR, "raw", a.tag),
                                allow_patterns=a.hf_allow)
    assert src and os.path.isdir(src), f"no such source dir: {src}"

    idx = HashIndex()
    for d in PROTECTED:
        full = os.path.join(REPO, d)
        for fn in sorted(os.listdir(full)) if os.path.isdir(full) else []:
            if fn.lower().endswith(IMG_EXT):
                im = imread(os.path.join(full, fn))
                if im is not None:
                    idx.add(dhash(im))
    n_prot = idx.h.size
    for r in read_manifest():
        if r["kind"] == "ext":
            im = imread(os.path.join(AUX_IMG, r["name"]))
            if im is not None:
                idx.add(dhash(im))
    print(f"[ingest] hash index: {n_prot} protected + {idx.h.size - n_prot} pool images", flush=True)

    files = []
    for root, _, fns in os.walk(src):
        files += [os.path.join(root, f) for f in fns if f.lower().endswith(IMG_EXT + VID_EXT)]
    files.sort()
    rows, kept, small, dup = [], 0, 0, 0
    start = sum(1 for r in read_manifest() if r["name"].startswith(f"r55ext_{a.tag}_"))
    for fp in files:
        if kept >= a.max:
            break
        for im in frames_of(fp, a.every_sec, a.max_per_video):
            if min(im.shape[:2]) < a.min_side:
                small += 1
                continue
            if max(im.shape[:2]) > a.max_side:
                s = a.max_side / max(im.shape[:2])
                im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
            hv = dhash(im)
            if idx.near(hv):
                dup += 1
                continue
            idx.add(hv)
            name = f"r55ext_{a.tag}_{start + kept:06d}.png"
            cv2.imencode(".png", im, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(os.path.join(AUX_IMG, name))
            rows.append(dict(name=name, kind="ext", seed="", seed_fold=-1, tuple_id="", dim=-1, level=0, op=a.tag))
            kept += 1
            if kept >= a.max:
                break
    n = append_manifest(rows)
    print(f"[ingest] {a.tag}: kept {n}, dropped {small} too small, {dup} near-duplicates")


if __name__ == "__main__":
    main()
