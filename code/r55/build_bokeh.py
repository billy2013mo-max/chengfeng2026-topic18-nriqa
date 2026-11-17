#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_bokeh.py — simulated 虚化突兀 (abrupt background blur) with matched negatives. 1 GPU for depth.

Why: 虚化突兀 has only 225 labelled positives, 142 of them from one pipeline tag; it is a detection-only defect
(P0 / none), and its label F1 still rises by +0.076 when the labels are doubled.  The defect is defined as
"虚实过渡生硬、主体与背景边缘割裂": a sharp subject cut out against a blurred background.  That can be produced by
construction, exactly as 花屏 was (build_glitch.py), and with the same safeguard that made 花屏 work: matched
negatives, so the only thing that separates positive from negative is the abrupt transition itself.

For a KVQ frame with a clear foreground (Depth-Anything-V2-Small, local weights; foreground = nearer side of an
Otsu split of the depth map, 12-65 % of the frame, mostly one connected region):
  r55bokeh_<v>_<k>.png      background blurred (sigma 5-14 px), subject kept, composited through a HARD mask that is
                            additionally grown or shrunk by a few pixels            -> 虚化突兀 = P0
  r55bokehsoft_<v>_<k>.png  the same background blur composited through a wide feathered mask (smooth transition)
                                                                                    -> not 虚化突兀
  r55kvq_<v>_<k>.png        the untouched frame (already on disk)                    -> not 虚化突兀
Other defects keep the fold teacher's soft labels of the untouched frame.  No label of ours and no test image is used.

Usage:  python code/r55/build_bokeh.py --gpu 0 --max 1200
"""
import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import AUX_DIR, AUX_IMG, append_manifest  # noqa: E402

MAN = os.path.join(AUX_DIR, "manifest_bokeh.csv")
SIDE = os.path.join(AUX_DIR, "bokeh_frames.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dev = torch.device(f"cuda:{a.gpu}")
    mid = "depth-anything/Depth-Anything-V2-Small-hf"
    proc = AutoImageProcessor.from_pretrained(mid, local_files_only=True)
    net = AutoModelForDepthEstimation.from_pretrained(mid, local_files_only=True).to(dev).eval()
    rng = random.Random(a.seed)
    names = sorted(r["name"] for r in csv.DictReader(open(os.path.join(AUX_DIR, "kvq_frames.csv"), encoding="utf-8")))
    rng.shuffle(names)
    rows, side, tried = [], [], 0
    for n in names:
        if len(side) >= a.max:
            break
        tried += 1
        im = cv2.imdecode(np.fromfile(os.path.join(AUX_IMG, n), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        with torch.no_grad():
            inp = proc(images=cv2.cvtColor(im, cv2.COLOR_BGR2RGB), return_tensors="pt").to(dev)
            d = net(**inp).predicted_depth[0].float()
        d = torch.nn.functional.interpolate(d[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        d = (d - d.min()) / max(1e-6, d.max() - d.min())                     # larger = nearer
        d8 = (d * 255).astype(np.uint8)
        thr, fg = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        share = float((fg > 0).mean())
        if not (0.12 <= share <= 0.65):
            continue
        # the split must be a real foreground/background separation, not a depth ramp
        mu_f, mu_b = d[fg > 0].mean(), d[fg == 0].mean()
        if mu_f - mu_b < 0.30:
            continue
        ncc, lab, st, _ = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), connectivity=8)
        if ncc < 2:
            continue
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[big, cv2.CC_STAT_AREA] < 0.7 * (fg > 0).sum():
            continue
        m = (lab == big).astype(np.uint8)
        # background must have something to blur: skip flat backgrounds (screen recordings, plain walls)
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
        if lap[m == 0].mean() < 2.0:
            continue
        sigma = rng.uniform(5.0, 14.0)
        blur = cv2.GaussianBlur(im, (0, 0), sigma)
        k = rng.randint(2, 9)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mode = rng.choice(["grow", "shrink", "exact"])
        mh = cv2.dilate(m, ker) if mode == "grow" else (cv2.erode(m, ker) if mode == "shrink" else m)
        mh = cv2.GaussianBlur(mh.astype(np.float32), (0, 0), rng.uniform(0.6, 1.5))[..., None]          # hard edge (about 1 px)
        hard = (im.astype(np.float32) * mh + blur.astype(np.float32) * (1 - mh)).clip(0, 255).astype(np.uint8)
        ms = cv2.GaussianBlur(m.astype(np.float32), (0, 0), rng.uniform(18.0, 40.0))[..., None]         # wide, smooth transition
        ms = np.clip(ms * 1.6, 0, 1)
        soft = (im.astype(np.float32) * ms + blur.astype(np.float32) * (1 - ms)).clip(0, 255).astype(np.uint8)
        stem = n[len("r55kvq_"):-4]
        for tag, arr in (("r55bokeh_", hard), ("r55bokehsoft_", soft)):
            cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(os.path.join(AUX_IMG, f"{tag}{stem}.png"))
        rows += [dict(name=f"r55bokeh_{stem}.png", kind="bokeh", seed=n, seed_fold=-1, tuple_id="", dim=5, level=3, op=f"hard:{mode}{k}:s{sigma:.1f}"),
                 dict(name=f"r55bokehsoft_{stem}.png", kind="bokeh", seed=n, seed_fold=-1, tuple_id="", dim=5, level=0, op="soft"),
                 dict(name=f"r55bokehraw_{stem}.png", kind="bokeh", seed=n, seed_fold=-1, tuple_id="", dim=5, level=0, op="untouched")]
        # the untouched frame is stored once more under its own name so that every aux row has its own image file
        cv2.imencode(".png", im, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(os.path.join(AUX_IMG, f"r55bokehraw_{stem}.png"))
        side.append(dict(name=f"r55bokeh_{stem}.png", clean=n, fg_share=f"{share:.2f}", sigma=f"{sigma:.1f}", mask=f"{mode}{k}"))
        if len(side) % 100 == 0:
            print(f"[bokeh] {len(side)} built from {tried} frames tried", flush=True)
    n_add = append_manifest(rows, MAN)
    with open(SIDE, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["name", "clean", "fg_share", "sigma", "mask"])
        wr.writeheader()
        wr.writerows(side)
    print(f"[bokeh] {len(side)} positives, {2 * len(side)} matched negatives ({n_add} manifest rows) from {tried} frames tried")


if __name__ == "__main__":
    main()
