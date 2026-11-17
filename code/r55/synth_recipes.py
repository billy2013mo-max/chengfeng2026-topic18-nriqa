#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/synth_recipes.py — SynthLab recipe registry.

A recipe turns ONE labelled seed frame into a few synthetic frames and says, for each, what is known about its labels
BY CONSTRUCTION — never by a model's guess:

    (suffix, image, spec)      spec = dict(target=<defect index or -1>, level=<0 | 3 | 9>, mask=<6 floats>)

  level 0   the target defect is absent in this frame (matched negative; for target -1: nothing was changed that matters)
  level 3   the target defect is present at P0 (only for the two P0-only defects)
  level 9   the target defect is present, grade unknown (the trainer then teaches presence only)
  mask      1 = this defect keeps the SEED's human label on this frame, 0 = unknown after the operation, not supervised
            (the target defect is always supervised, by `level`)

Rules every recipe must follow (they are what separated the 花屏 recipe that worked from the ones that did not):
  1. Matched negatives: whatever is done to the positive besides the defect itself is also done to its negative.
  2. Same file path and format for both (PNG via the builder); no per-class compression, size or crop.
  3. Operators whose effect on another defect is not known mask that defect instead of asserting a label.
  4. Parameters are drawn per frame from a range, never one fixed value per class.

`SEEDS[name]` selects the labelled frames a recipe may start from (a function of the six human labels).
"""
import numpy as np

import synth_ops as so

ONES = [1.0] * 6


def _mask(*unknown):
    m = list(ONES)
    for d in unknown:
        m[d] = 0.0
    return m


def benign(im, rng):
    """Label-preserving processing: a mild transcode / re-quantisation / tone change does not create or remove a defect.
    Applied to clean AND defective frames with their human labels unchanged, it tells the model that 'has been
    through a processing step' is not evidence of a defect (clean frames with a pipeline tag are flagged 19-32 %
    of the time; unseen pipelines cost ~10 % RPS)."""
    k = rng.choice(["x264", "jpeg", "gamma", "x264+gamma"])
    out = im
    if "x264" in k:
        out = so.x264(out, crf=int(rng.randint(15, 21)))
    if k == "jpeg":
        out = so.jpeg(out, q=int(rng.randint(90, 97)))
    if "gamma" in k:
        out = so.gamma(out, g=float(rng.uniform(0.93, 1.08)))
    return [("b", out, dict(target=-1, level=0, mask=ONES))]


def sharpen_detect(im, rng):
    """过锐 present by construction (clearly visible unsharp masking with halo), grade unknown; matched negative is the
    same frame through the same benign transcode.  纹理损失 / 模糊 are masked on the positive (sharpening changes both)."""
    crf = int(rng.randint(15, 21))
    pos = so.x264(so.usm(im, amount=float(rng.uniform(1.6, 3.2)), radius=float(rng.uniform(1.2, 2.6))), crf=crf)
    neg = so.x264(im, crf=crf)
    return [("p", pos, dict(target=2, level=9, mask=_mask(0, 1))), ("n", neg, dict(target=2, level=0, mask=ONES))]


def blur_detect(im, rng):
    """模糊 present by construction (defocus or resolution loss well above visibility), grade unknown."""
    crf = int(rng.randint(15, 21))
    if rng.rand() < 0.5:
        deg = so.gauss(im, sigma=float(rng.uniform(1.6, 3.2)))
    else:
        deg = so.downup(im, factor=float(rng.uniform(0.3, 0.5)))
    return [("p", so.x264(deg, crf=crf), dict(target=0, level=9, mask=_mask(1, 2))), ("n", so.x264(im, crf=crf), dict(target=0, level=0, mask=ONES))]


def texture_detect(im, rng):
    """纹理损失 present by construction (strong denoising / smoothing that wipes fine texture), grade unknown."""
    crf = int(rng.randint(15, 21))
    if rng.rand() < 0.5:
        deg = so.nlm(im, h=float(rng.uniform(9, 16)))
    else:
        deg = so.bilateral(so.bilateral(im, sigma_color=float(rng.uniform(55, 100))), sigma_color=float(rng.uniform(40, 80)))
    return [("p", so.x264(deg, crf=crf), dict(target=1, level=9, mask=_mask(0, 2))), ("n", so.x264(im, crf=crf), dict(target=1, level=0, mask=ONES))]


RECIPES = {
    "benign1": dict(fn=benign, seeds="all", n_seeds=1600, doc=benign.__doc__),
    "sharp1": dict(fn=sharpen_detect, seeds="clean2", n_seeds=700, doc=sharpen_detect.__doc__),
    "blur1": dict(fn=blur_detect, seeds="clean0", n_seeds=700, doc=blur_detect.__doc__),
    "tex1": dict(fn=texture_detect, seeds="clean1", n_seeds=600, doc=texture_detect.__doc__),
}

SEEDS = {
    "all": lambda g: True,
    "clean": lambda g: all(x == 0 for x in g),
    "clean0": lambda g: g[0] == 0 and g[3] == 0,          # no 模糊 (and no 花屏) on the seed
    "clean1": lambda g: g[1] == 0 and g[3] == 0,
    "clean2": lambda g: g[2] == 0 and g[3] == 0,
}


def rng_for(recipe, name):
    import hashlib
    return np.random.RandomState(int(hashlib.md5(f"{recipe}:{name}".encode()).hexdigest()[:8], 16))
