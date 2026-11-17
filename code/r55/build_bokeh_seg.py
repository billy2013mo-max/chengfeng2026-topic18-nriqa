#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_bokeh_seg.py — SynthLab recipe `bkseg1`: 虚化突兀 built on LABELLED frames with a real subject mask.
Run with `python3 -I` (weights in data/r55_aux/raw/seg_weights are loaded as plain tensors).

Why (docs/REASSESSMENT_AND_PLAN_2026-11-12.md §10): 虚化突兀 is the most valuable single label (P0-only, counted in
both the positive and the P0 score), it is data-limited (+0.076 F1 per doubling of the labels), 79 of its 225
positives come from one batch, and outside that batch the ensemble finds 69 %.  The labelled positives are frames
with a SHARP SUBJECT CUT OUT AGAINST A BLURRED BACKGROUND (drama stills, privacy blur, panning shots).  The first
simulation (build_bokeh.py) failed on fidelity, not on principle: a depth split of KVQ frames rarely isolates a
subject and the blur destroyed the captions.  Here:
  * seeds are labelled frames without 虚化突兀 / 花屏 in which Mask R-CNN (COCO, torchvision) finds a person or animal
    covering 6-60 % of the frame;
  * positive  `r55syn_bkseg1_p__<stem>.png`: the background (subject removed before blurring, so nothing bleeds) is
    blurred with a Gaussian or lens kernel and composited through a HARD mask that is exact, grown or shrunk by a
    few pixels; in 40 % of the frames the subject is also sharpened (what an enhancer does to the in-focus region).
    Burnt-in captions keep their pixels.                                      -> 虚化突兀 = P0
  * negative  `r55syn_bkseg1_n__<stem>.png`: the SAME blur on the whole frame, captions kept: same content, same
    kernel, same information loss, no sharp/blurred boundary.                 -> 虚化突兀 absent
模糊 / 纹理损失 are not supervised on either frame (blurring changes both), 过锐 is not supervised when the subject was
sharpened; 花屏 and 过度生成 keep the seed's human labels.  No model prediction enters the labels.

Usage:
  for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r python3 -I code/r55/build_bokeh_seg.py --rank $r --world 4 & done; wait
  python3 -I code/r55/build_bokeh_seg.py --merge 4
  python code/r55/prep_aux.py --gpus 8 --steps template meta --manifest data/r55_aux/manifest_bkseg1.csv --suffix _bkseg1 --tag r55bkseg1
"""
import argparse
import csv
import hashlib
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RECIPE = "bkseg1"
AUX = os.path.join(REPO, "data/r55_aux")
IMG = os.path.join(AUX, "images")
WEI = os.path.join(AUX, "raw/seg_weights/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth")
DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
FIELDS = ["name", "kind", "seed", "seed_fold", "tuple_id", "dim", "level", "op"]
SUBJ = {1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}                      # COCO: person + animals


def find(name):
    for d in ("data/images", "data/images_supp"):
        p = os.path.join(REPO, d, name)
        if os.path.exists(p):
            return p


def labelled():
    lab, vfold = {}, {}
    for f in range(8):
        for r in csv.DictReader(open(os.path.join(REPO, f"data/qingyu_splits/fold_{f}.csv"), encoding="utf-8")):
            if r["split"] == "val":
                vfold[r["name"]] = f
                lab[r["name"]] = [int(r[d]) for d in DIMS]
    sfold = {r["name"]: int(r["fold"]) for r in csv.DictReader(open(os.path.join(REPO, "data/r55_tagsplit/folds.csv"), encoding="utf-8"))}
    return lab, vfold, sfold


def ui_mask(im):
    """Burnt-in captions: near-saturated fill bordered within 3 px by a dark outline (the heuristic of
    qingyu/dual_engine_preprocess.py), kept where strokes are dense; dilated so whole glyphs are covered."""
    import cv2
    import numpy as np
    b, g, r = [im[:, :, k].astype(np.float32) / 255.0 for k in range(3)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    bright = (y > 0.88) | ((r > 0.84) & (g > 0.72) & (b < 0.35))
    dark = y < 0.16
    k7 = np.ones((7, 7), np.uint8)
    stroke = (bright & (cv2.dilate(dark.astype(np.uint8), k7) > 0)) | (dark & (cv2.dilate(bright.astype(np.uint8), k7) > 0))
    dens = cv2.blur(stroke.astype(np.float32), (15, 15))
    h = im.shape[0]
    yy = (np.arange(h, dtype=np.float32) / max(1, h - 1))[:, None]
    band = (yy < 0.18) | (yy > 0.72)
    m = (stroke & ((band & (dens > 0.06)) | (dens > 0.25))).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    # a caption is a block of glyphs: drop small isolated blobs (specular highlights, twigs against the sky), which
    # would otherwise survive the blur as sharp specks
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = 1500.0 * (max(im.shape[:2]) / 1280.0) ** 2
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab_]


def lens_kernel(radius):
    import numpy as np
    r = max(1, int(round(radius)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (xx * xx + yy * yy <= r * r).astype(np.float32)
    return k / k.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--merge", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=900)
    a = ap.parse_args()
    import numpy as np
    lab, vfold, sfold = labelled()
    od = os.path.join(REPO, f"results/r55/synthlab/{RECIPE}")
    os.makedirs(od, exist_ok=True)
    if a.merge:
        made = []
        for r in range(a.merge):
            made += json.load(open(os.path.join(od, f"shard{r}.json"), encoding="utf-8"))
        made.sort(key=lambda t: t["name"])
        names, T, M, man, man_s = [], [], [], [], []
        for t in made:
            g = lab[t["seed"]]
            tt = np.full((6, 4), 1e-4, dtype=np.float32)
            for d in range(6):
                tt[d, g[d]] = 1.0
            tt[5] = 1e-4
            tt[5, 3 if t["level"] == 3 else 0] = 1.0
            names.append(t["name"])
            T.append(tt / tt.sum(-1, keepdims=True))
            M.append(np.array([0, 0, 0 if t["usm"] else 1, 1, 1, 1], dtype=np.float32))
            base = dict(name=t["name"], kind="synth", seed=t["seed"], tuple_id="", dim=5, level=t["level"], op=t["op"])
            man.append(dict(base, seed_fold=vfold[t["seed"]]))
            man_s.append(dict(base, seed_fold=sfold[t["seed"]]))
        for path, rr in ((os.path.join(AUX, f"manifest_{RECIPE}.csv"), man), (os.path.join(AUX, f"manifest_{RECIPE}_shift.csv"), man_s)):
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rr)
        for suf in ("", "_shift"):
            np.savez_compressed(os.path.join(REPO, f"results/r55/aux_teacher_{RECIPE}{suf}.npz"), names=np.array(names),
                                probs_folds=np.stack(T)[None].astype(np.float32), mask=np.stack(M), det=np.zeros(len(names), dtype=np.int64),
                                towers=np.array(["labels by construction (seed's human labels + recipe)"]))
        npos = sum(1 for t in made if t["level"] == 3)
        json.dump(dict(recipe=RECIPE, doc=" ".join(__doc__.split()[:150]), seeds=len({t["seed"] for t in made}), frames=len(made), positives=npos,
                       negatives=len(made) - npos), open(os.path.join(od, "build.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[bkseg] merged {len(made)} frames ({npos} cut-out positives, {len(made) - npos} uniformly blurred negatives) from {len({t['seed'] for t in made})} seeds")
        return
    import cv2
    import torch
    import torchvision
    dev = torch.device("cuda")
    net = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    net.load_state_dict(torch.load(WEI, map_location="cpu", weights_only=True))
    net.eval().to(dev)
    cand = sorted(n for n, g in lab.items() if g[5] == 0 and g[3] == 0 and n in sfold)
    rs = np.random.RandomState(0)
    order = [cand[i] for i in rs.permutation(len(cand))][a.rank::a.world]
    quota = (a.n_seeds + a.world - 1) // a.world
    made, tried, why = [], 0, {"no_subject": 0, "share": 0, "flat_background": 0}
    for n in order:
        if len(made) >= 2 * quota:
            break
        tried += 1
        rng = np.random.RandomState(int(hashlib.md5(f"{RECIPE}:{n}".encode()).hexdigest()[:8], 16))
        im = cv2.imdecode(np.fromfile(find(n), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        s = min(1.0, 1024.0 / max(h, w))
        small = cv2.resize(im, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else im
        x = torch.from_numpy(small[:, :, ::-1].copy()).permute(2, 0, 1).float().to(dev) / 255.0
        with torch.no_grad():
            o = net([x])[0]
        keep = [i for i in range(len(o["scores"])) if float(o["scores"][i]) >= 0.6 and int(o["labels"][i]) in SUBJ]
        ms = [(o["masks"][i, 0] > 0.5).cpu().numpy() for i in keep]
        ms = sorted([m for m in ms if m.mean() >= 0.015], key=lambda m: -m.mean())
        if not ms:
            why["no_subject"] += 1
            continue
        ms = [m for m in ms if m.mean() >= 0.35 * ms[0].mean()][:3]
        m = np.any(np.stack(ms), 0).astype(np.uint8)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        share = float(m.mean())
        if not (0.06 <= share <= 0.60):
            why["share"] += 1
            continue
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if np.abs(cv2.Laplacian(gray, cv2.CV_32F))[m == 0].mean() < 1.5:       # nothing to blur behind the subject
            why["flat_background"] += 1
            continue
        scale = max(h, w) / 1280.0
        imf = im.astype(np.float32)
        bgw = (1 - m).astype(np.float32)[..., None]
        if rng.rand() < 0.5:
            sig = float(rng.uniform(4.0, 12.0)) * scale
            blur_fn = lambda t: cv2.GaussianBlur(t, (0, 0), sig)                # noqa: E731
            kname = f"gauss{sig:.1f}"
        else:
            rad = float(rng.uniform(6.0, 16.0)) * scale
            ker = lens_kernel(rad)
            blur_fn = lambda t: cv2.filter2D(t, -1, ker, borderType=cv2.BORDER_REFLECT)   # noqa: E731
            kname = f"lens{rad:.1f}"
        num, den = blur_fn(imf * bgw), blur_fn(bgw[..., 0])
        bg = np.where(den[..., None] > 0.02, num / np.clip(den[..., None], 0.02, None), blur_fn(imf))   # background-only blur
        k = int(rng.randint(2, 9))
        ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        mode = ["exact", "grow", "shrink"][rng.randint(3)]
        mh = cv2.dilate(m, ell) if mode == "grow" else (cv2.erode(m, ell) if mode == "shrink" else m)
        mh = cv2.GaussianBlur(mh.astype(np.float32), (0, 0), float(rng.uniform(0.6, 1.5)))[..., None]   # edge about 1 px wide
        subj = imf
        usm = bool(rng.rand() < 0.4)
        if usm:
            amt, us = float(rng.uniform(0.4, 1.0)), float(rng.uniform(1.0, 1.6))
            subj = imf + amt * (imf - cv2.GaussianBlur(imf, (0, 0), us))
        ui = ui_mask(im).astype(np.float32)[..., None]
        pos = subj * mh + bg * (1 - mh)
        pos = (imf * ui + pos * (1 - ui)).clip(0, 255).astype(np.uint8)
        neg = blur_fn(imf)
        neg = (imf * ui + neg * (1 - ui)).clip(0, 255).astype(np.uint8)
        stem = os.path.splitext(n)[0]
        op = f"{kname}:{mode}{k}:usm{int(usm)}:share{share:.2f}"
        for suffix, arr, lv in (("p", pos, 3), ("n", neg, 0)):
            fn = f"r55syn_{RECIPE}_{suffix}__{stem}.png"
            cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(os.path.join(IMG, fn))
            made.append(dict(name=fn, seed=n, level=lv, op=op, usm=usm and suffix == "p"))
        if (len(made) // 2) % 50 == 0:
            print(f"[bkseg] rank {a.rank}: {len(made) // 2} seeds accepted of {tried} tried", flush=True)
    json.dump(made, open(os.path.join(od, f"shard{a.rank}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[bkseg] rank {a.rank}: {len(made)} frames from {len(made) // 2} seeds ({tried} tried; rejected {why})")


if __name__ == "__main__":
    main()
