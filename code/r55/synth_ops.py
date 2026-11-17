#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/synth_ops.py — operator library for SynthLab (CPU, deterministic given the parameters).

Every operator takes and returns a BGR uint8 image of the same size.  Positives and their matched negatives must go
through the same code path and the same file format (PNG, same compression) so that nothing but the intended change
separates them — the lesson of the earlier synthetic rounds (docs/SYNTHETIC_DATA_BREAKTHROUGH_PLAN.md §1: JPEG
quality per grade, bilateral-filter fingerprints, geometric paste shapes) and of the first 花屏 run (re-encoded look).

Families
  blur        gauss(sigma) | motion(length, angle) | downup(factor)            -> 模糊
  texture     bilateral(sigma_color) | nlm(h) | x264(crf) | median(k)          -> 纹理损失
  sharpen     usm(amount, radius) | lap(amount)                                -> 过锐
  benign      x264(crf<=20) | resize(0.9) | gamma | jpeg(q>=92)                -> nothing (must NOT trigger a defect)
"""
import os
import subprocess
import tempfile

import cv2
import numpy as np


def gauss(im, sigma):
    return cv2.GaussianBlur(im, (0, 0), float(sigma))


def motion(im, length, angle=0.0):
    k = int(max(3, round(length)) | 1)
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(angle), 1.0)
    ker = cv2.warpAffine(ker, rot, (k, k))
    ker /= max(1e-6, ker.sum())
    return cv2.filter2D(im, -1, ker)


def downup(im, factor):
    h, w = im.shape[:2]
    small = cv2.resize(im, (max(8, int(w * factor)), max(8, int(h * factor))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def bilateral(im, sigma_color, sigma_space=7):
    return cv2.bilateralFilter(im, 9, float(sigma_color), float(sigma_space))


def nlm(im, h):
    return cv2.fastNlMeansDenoisingColored(im, None, float(h), float(h), 5, 15)


def median(im, k):
    return cv2.medianBlur(im, int(k) | 1)


def usm(im, amount, radius=1.5):
    blur = cv2.GaussianBlur(im, (0, 0), float(radius)).astype(np.float32)
    return np.clip(im.astype(np.float32) + float(amount) * (im.astype(np.float32) - blur), 0, 255).astype(np.uint8)


def lap(im, amount):
    l = cv2.Laplacian(im.astype(np.float32), cv2.CV_32F, ksize=3)
    return np.clip(im.astype(np.float32) - float(amount) * l, 0, 255).astype(np.uint8)


def gamma(im, g):
    return np.clip(255.0 * (im.astype(np.float32) / 255.0) ** float(g), 0, 255).astype(np.uint8)


def jpeg(im, q):
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def resize(im, factor):
    return downup(im, factor)


def x264(im, crf, tmp_root="/dev/shm"):
    """One frame through libx264 (yuv420p) and back: real codec loss, as in a transcoding step."""
    h, w = im.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    d = tempfile.mkdtemp(prefix="sop_", dir=tmp_root)
    try:
        src, mp4, out = os.path.join(d, "a.png"), os.path.join(d, "a.mp4"), os.path.join(d, "b.png")
        cv2.imwrite(src, im[:h2, :w2])
        subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-i", src, "-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)),
                        "-pix_fmt", "yuv420p", "-frames:v", "1", mp4], capture_output=True)
        subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-i", mp4, "-frames:v", "1", out], capture_output=True)
        o = cv2.imread(out, cv2.IMREAD_COLOR)
        if o is None:
            return im.copy()
        res = im.copy()
        res[:h2, :w2] = o
        return res
    finally:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        os.rmdir(d)


OPS = {"gauss": gauss, "motion": motion, "downup": downup, "bilateral": bilateral, "nlm": nlm, "median": median,
       "usm": usm, "lap": lap, "gamma": gamma, "jpeg": jpeg, "resize": resize, "x264": x264}

# The behavioural suite: (condition name, target defect index or -1 for benign, operator, kwargs).  Strengths go from
# barely visible to obvious; they are stimuli, not labels.
SUITE = [
    ("blur_gauss_1", 0, "gauss", dict(sigma=0.7)), ("blur_gauss_2", 0, "gauss", dict(sigma=1.2)), ("blur_gauss_3", 0, "gauss", dict(sigma=2.0)), ("blur_gauss_4", 0, "gauss", dict(sigma=3.5)),
    ("blur_downup_1", 0, "downup", dict(factor=0.75)), ("blur_downup_2", 0, "downup", dict(factor=0.5)), ("blur_downup_3", 0, "downup", dict(factor=0.33)),
    ("blur_motion_1", 0, "motion", dict(length=5)), ("blur_motion_2", 0, "motion", dict(length=11)), ("blur_motion_3", 0, "motion", dict(length=19)),
    ("tex_bilateral_1", 1, "bilateral", dict(sigma_color=25)), ("tex_bilateral_2", 1, "bilateral", dict(sigma_color=55)), ("tex_bilateral_3", 1, "bilateral", dict(sigma_color=100)),
    ("tex_nlm_1", 1, "nlm", dict(h=4)), ("tex_nlm_2", 1, "nlm", dict(h=9)), ("tex_nlm_3", 1, "nlm", dict(h=16)),
    ("tex_x264_1", 1, "x264", dict(crf=30)), ("tex_x264_2", 1, "x264", dict(crf=37)), ("tex_x264_3", 1, "x264", dict(crf=44)),
    ("sharp_usm_1", 2, "usm", dict(amount=0.6, radius=1.2)), ("sharp_usm_2", 2, "usm", dict(amount=1.3, radius=1.5)), ("sharp_usm_3", 2, "usm", dict(amount=2.2, radius=2.0)), ("sharp_usm_4", 2, "usm", dict(amount=3.5, radius=2.5)),
    ("sharp_lap_1", 2, "lap", dict(amount=0.4)), ("sharp_lap_2", 2, "lap", dict(amount=0.9)), ("sharp_lap_3", 2, "lap", dict(amount=1.6)),
    ("benign_x264", -1, "x264", dict(crf=18)), ("benign_jpeg", -1, "jpeg", dict(q=93)), ("benign_resize", -1, "resize", dict(factor=0.9)),
    ("benign_gamma_dn", -1, "gamma", dict(g=1.08)), ("benign_gamma_up", -1, "gamma", dict(g=0.93)),
]
