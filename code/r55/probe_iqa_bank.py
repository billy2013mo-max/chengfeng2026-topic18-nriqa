#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/probe_iqa_bank.py — E1: scores of public pretrained no-reference IQA models as a feature bank.
Run with `python3 -I` (isolated mode); the library lives in third_party/pydeps_iqa, weights in their own folder.

docs/REASSESSMENT_AND_PLAN_2026-11-12.md §4 A.  These models were trained on human opinion scores we do not have
(KonIQ, SPAQ, FLIVE, PIPAL, KADID, AVA, ...).  If their scores add nothing to the ensemble's probabilities, external
human supervision of this kind is not a lever and E4 (KVQ MOS multi-task) is dropped with it.

Safety: every `torch.load` in this process is forced to `weights_only=True`, so a downloaded checkpoint can only
yield tensors; a metric whose checkpoint needs more than that is skipped and reported.  Downloads go to
data/r55_aux/raw/pyiqa_weights (nothing else lives there).  Only the labelled images are read; nothing is uploaded.

Per image and metric: score of the whole frame (longer side resized to 768) and of five native-pixel 448 crops
(centre + four quadrant centres) -> features [global, crop mean, crop min, crop max].

Usage:
  python3 -I code/r55/probe_iqa_bank.py --warmup --gpu 0                      # download + load test, one process
  for r in 0..7: python3 -I code/r55/probe_iqa_bank.py --rank r --world 8 --gpu g &
  python3 -I code/r55/probe_iqa_bank.py --merge 8
"""
import argparse
import csv
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "third_party/pydeps_iqa"))
OUT = os.path.join(REPO, "results/r55/probes")
WEIGHTS = os.path.join(REPO, "data/r55_aux/raw/pyiqa_weights")
# topiq_*, the CLIP-based metrics and fgresq do not load in this environment (registry / missing `clip` / transformers
# version); they are left out rather than patched.
METRICS = ["musiq", "musiq-spaq", "musiq-paq2piq", "maniqa", "maniqa-pipal", "maniqa-kadid",
           "arniqa", "arniqa-kadid", "arniqa-spaq", "arniqa-flive", "hyperiqa", "dbcnn", "tres",
           "nima-koniq", "nima", "paq2piq", "afine_nr", "niqe", "brisque", "piqe"]


def setup(gpu):
    import numpy as np
    import torch
    os.makedirs(WEIGHTS, exist_ok=True)
    torch.hub.set_dir(os.path.join(WEIGHTS, "hub"))
    _load = torch.load

    def safe_load(*a, **k):                                       # tensors only: a checkpoint cannot run code
        k["weights_only"] = True
        return _load(*a, **k)
    torch.load = safe_load
    import pyiqa
    dev = torch.device(f"cuda:{gpu}")
    ok, bad = {}, {}
    x = torch.rand(2, 3, 448, 448, device=dev)
    for m in METRICS:
        try:
            torch.manual_seed(0)
            f = pyiqa.create_metric(m, device=dev)
            with torch.no_grad():
                v = f(x.clone())                                  # some metrics normalise their input in place
            assert np.isfinite(np.asarray(v.detach().float().cpu())).all()
            ok[m] = f
        except Exception as e:                                    # not loadable under tensors-only, or no weights reachable
            bad[m] = f"{type(e).__name__}: {str(e)[:140]}"
    return ok, bad, dev


def find(name):
    for d in ("data/images", "data/images_supp"):
        p = os.path.join(REPO, d, name)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--merge", type=int, default=0)
    a = ap.parse_args()
    import numpy as np
    os.makedirs(OUT, exist_ok=True)
    if a.merge:
        zs = [np.load(os.path.join(OUT, f"iqa_bank_shard{r}.npz"), allow_pickle=True) for r in range(a.merge)]
        fn = list(zs[0]["feat_names"])
        assert all(list(z["feat_names"]) == fn for z in zs), "shards loaded different metric sets"
        np.savez_compressed(os.path.join(OUT, "iqa_bank_feats.npz"), names=np.concatenate([z["names"] for z in zs]),
                            X=np.concatenate([z["X"] for z in zs]), feat_names=np.array(fn))
        print(f"[iqa_bank] merged {sum(len(z['names']) for z in zs)} images, {len(fn)} features")
        return
    import torch
    from PIL import Image
    ok, bad, dev = setup(a.gpu)
    print(f"[iqa_bank] {len(ok)} metrics loaded: {' '.join(ok)}", flush=True)
    for m, e in bad.items():
        print(f"[iqa_bank] skipped {m}: {e}", flush=True)
    if a.warmup:
        return
    names = [r["name"] for r in csv.DictReader(open(os.path.join(REPO, "data/qingyu_splits/fold_0.csv"), encoding="utf-8"))][a.rank::a.world]
    keys = [f"{m}:{s}" for m in ok for s in ("global", "crop_mean", "crop_min", "crop_max")]
    rows, keep = [], []
    for i, n in enumerate(names):
        p = find(n)
        if p is None:
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if min(w, h) < 448:
            s = 448 / min(w, h)
            im = im.resize((max(448, round(w * s)), max(448, round(h * s))), Image.BICUBIC)
            w, h = im.size
        s = 768 / max(w, h)
        s = min(1.0, s)
        gw, gh = max(224, int(round(w * s / 32)) * 32), max(224, int(round(h * s / 32)) * 32)      # multiples of 32 for every metric
        g = im.resize((gw, gh), Image.BICUBIC)
        crops = []
        for cx, cy in ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
            x0, y0 = int(min(max(0, cx * w - 224), w - 448)), int(min(max(0, cy * h - 224), h - 448))
            crops.append(np.asarray(im.crop((x0, y0, x0 + 448, y0 + 448)), dtype=np.float32) / 255.0)
        xc = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(dev)
        xg = torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
        f = []
        for m, fn in ok.items():
            try:
                torch.manual_seed(0)
                with torch.no_grad():
                    vg = float(fn(xg.clone()).float().mean())
                    vc = fn(xc.clone()).float().flatten().cpu().numpy()
                f += [vg, float(vc.mean()), float(vc.min()), float(vc.max())]
            except Exception:
                f += [np.nan] * 4
        rows.append(f)
        keep.append(n)
        if (i + 1) % 50 == 0:
            print(f"[iqa_bank] rank {a.rank}: {i + 1}/{len(names)}", flush=True)
    np.savez_compressed(os.path.join(OUT, f"iqa_bank_shard{a.rank}.npz"), names=np.array(keep), X=np.array(rows, dtype=np.float32),
                        feat_names=np.array(keys))
    print(f"[iqa_bank] rank {a.rank}: wrote {len(keep)} images, {len(keys)} features")


if __name__ == "__main__":
    main()
