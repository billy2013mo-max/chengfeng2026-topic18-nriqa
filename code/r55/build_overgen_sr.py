#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_overgen_sr.py — SynthLab recipe `ogsr1`: 过度生成 produced by real super-resolution / restoration models.
Run with `python3 -I` (the loader library lives in third_party/pydeps_sr; weights in data/r55_aux/raw/sr_weights).

Why: 过度生成 is the largest loss (label F1 0.32 / 0.27 / 0.63) and the earlier FLUX re-render ladder was a different
phenomenon (models that detect the real defect at AUC 0.96 saw it in 16-25 % of those frames).  The labelled defect is
what generative enhancement of a weak source looks like, so it is produced here the same way: a labelled frame without
过度生成 is degraded (down-scaled 2-4x, optionally re-compressed) and then restored by a public GAN super-resolution
model (Real-ESRGAN x4plus / x2plus, BSRGAN, realesr-general-x4v3) back to its original size.  The restored frame
contains texture that the model invented: 过度生成 present by construction, grade unknown (level 9).

Matched negative: the SAME degraded frame brought back with bicubic interpolation — identical content, size, file
format and information loss, but no invented texture: 过度生成 absent (level 0).
模糊 / 纹理损失 / 过锐 are masked on both (restoration changes all three); 花屏 and 虚化突兀 keep the seed's labels.
Labels use only the seed's human labels and the construction; no model prediction enters the build.

Weights are loaded as plain tensors (`torch.load(weights_only=True)`), architectures come from the `spandrel` package.

Usage:
  for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r python3 -I code/r55/build_overgen_sr.py --rank $r --world 4 & done; wait
  python3 -I code/r55/build_overgen_sr.py --merge 4          # manifests + labels; then prep_aux.py for the image metadata
"""
import argparse
import csv
import hashlib
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "third_party/pydeps_sr"))
RECIPE = "ogsr1"
AUX = os.path.join(REPO, "data/r55_aux")
IMG = os.path.join(AUX, "images")
WEI = os.path.join(AUX, "raw/sr_weights")
MODELS = ["RealESRGAN_x4plus.pth", "RealESRGAN_x2plus.pth", "BSRGAN.pth", "realesr-general-x4v3.pth"]
DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
FIELDS = ["name", "kind", "seed", "seed_fold", "tuple_id", "dim", "level", "op"]


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
        names, T, M, Dt, man, man_s = [], [], [], [], [], []
        for t in made:
            g = lab[t["seed"]]
            tt = np.full((6, 4), 1e-4, dtype=np.float32)
            for d in range(6):
                tt[d, g[d]] = 1.0
            tt[4] = 1e-4
            tt[4, 0] = 1.0                                             # negatives: absent; positives: replaced by the presence target in the loss
            names.append(t["name"])
            T.append(tt / tt.sum(-1, keepdims=True))
            M.append(np.array([0, 0, 0, 1, 1, 1], dtype=np.float32))
            Dt.append(1 if t["level"] == 9 else 0)
            base = dict(name=t["name"], kind="synth", seed=t["seed"], tuple_id="", dim=4, level=t["level"], op=t["op"])
            man.append(dict(base, seed_fold=vfold[t["seed"]]))
            man_s.append(dict(base, seed_fold=sfold[t["seed"]]))
        for path, rr in ((os.path.join(AUX, f"manifest_{RECIPE}.csv"), man), (os.path.join(AUX, f"manifest_{RECIPE}_shift.csv"), man_s)):
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rr)
        for suf in ("", "_shift"):
            np.savez_compressed(os.path.join(REPO, f"results/r55/aux_teacher_{RECIPE}{suf}.npz"), names=np.array(names),
                                probs_folds=np.stack(T)[None].astype(np.float32), mask=np.stack(M), det=np.array(Dt, dtype=np.int64),
                                towers=np.array(["labels by construction (seed's human labels + recipe)"]))
        npos = sum(1 for t in made if t["level"] == 9)
        json.dump(dict(recipe=RECIPE, doc=" ".join(__doc__.split()[:120]), seeds=len({t["seed"] for t in made}), frames=len(made), present_unknown_grade=npos,
                       negatives=len(made) - npos), open(os.path.join(od, "build.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[ogsr] merged {len(made)} frames ({npos} restored by a generative model, {len(made) - npos} matched bicubic negatives)")
        return
    import cv2
    import torch
    from spandrel import ModelLoader
    dev = torch.device("cuda")
    nets = {}
    for f in MODELS:
        sd = torch.load(os.path.join(WEI, f), map_location="cpu", weights_only=True)
        for k in ("params_ema", "params", "state_dict"):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
                break
        m = ModelLoader(device=dev).load_from_state_dict(sd)
        m.eval()
        nets[f] = m
    cand = sorted(n for n, g in lab.items() if g[4] == 0 and g[3] == 0 and n in sfold)
    rs = np.random.RandomState(0)
    pick = [cand[i] for i in sorted(rs.choice(len(cand), min(a.n_seeds, len(cand)), replace=False))][a.rank::a.world]
    made = []
    for i, n in enumerate(pick):
        rng = np.random.RandomState(int(hashlib.md5(f"{RECIPE}:{n}".encode()).hexdigest()[:8], 16))
        im = cv2.imdecode(np.fromfile(find(n), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        f = MODELS[rng.randint(len(MODELS))]
        net = nets[f]
        down = float(rng.choice([2.0, 2.5, 3.0, 4.0]))
        small = cv2.resize(im, (max(32, int(round(w / down))), max(32, int(round(h / down)))), interpolation=cv2.INTER_AREA)
        q = None
        if rng.rand() < 0.5:                                           # weak source: re-compressed before restoration
            q = int(rng.randint(60, 92))
            small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
        x = torch.from_numpy(small[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(dev) / 255.0
        with torch.no_grad():
            y = net(x).clamp(0, 1)
        up = (y[0].permute(1, 2, 0).float().cpu().numpy()[:, :, ::-1] * 255.0).round().astype(np.uint8)
        pos = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA if up.shape[0] >= h else cv2.INTER_CUBIC)
        neg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        stem = os.path.splitext(n)[0]
        op = f"{f.split('.')[0]}:down{down}:jpeg{q}"
        for suffix, arr, lv in (("p", pos, 9), ("n", neg, 0)):
            fn = f"r55syn_{RECIPE}_{suffix}__{stem}.png"
            cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(os.path.join(IMG, fn))
            made.append(dict(name=fn, seed=n, level=lv, op=op))
        if (i + 1) % 50 == 0:
            print(f"[ogsr] rank {a.rank}: {i + 1}/{len(pick)}", flush=True)
    json.dump(made, open(os.path.join(od, f"shard{a.rank}.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[ogsr] rank {a.rank}: {len(made)} frames")


if __name__ == "__main__":
    main()
