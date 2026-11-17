#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/probe_iqa_bank2.py — second bank of public pretrained quality models: the ones E1 (probe_iqa_bank.py) could
not load, first of all FGResQ, which the competition statement cites (fine-grained quality of RESTORED images, trained
on human pairwise preferences between restoration outputs — the closest public supervision to 过度生成 / 纹理损失).
Run with `python3 -I`; library in third_party/pydeps_iqa, weights in data/r55_aux/raw/pyiqa_weights.

What differs from probe_iqa_bank.py
  * FGResQ: transformers >= 5 flattened CLIPVisionModel (no `.vision_model`); a read-only alias and a key rename
    (`vision_model.` dropped) make the published checkpoint load.  The load is verified: no missing / unexpected keys.
  * TOPIQ: its module imports facexlib only for the face variant; a stub module lets the non-face variants register.
  * A-FINE: takes one image at a time (H, W multiples of 32).
  * FGResQ also contributes its 2,304-d quality feature (global view), reduced to 16 principal components fitted
    without labels, to test the representation and not only the scalar.
  * The CLIP-text metrics (CLIP-IQA+, LIQE, QualiCLIP) need the external `clip` package, which is not installed here
    and was not installed (installing code from GitHub was not permitted); they stay untested.

Safety as before: every torch.load is forced to weights_only=True; only labelled images are read; nothing is uploaded.

Usage:
  python3 -I code/r55/probe_iqa_bank2.py --warmup --gpu 0
  for r in 0..7: python3 -I code/r55/probe_iqa_bank2.py --rank r --world 8 --gpu g &
  python3 -I code/r55/probe_iqa_bank2.py --merge 8
"""
import argparse
import csv
import os
import sys
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "third_party/pydeps_iqa"))
OUT = os.path.join(REPO, "results/r55/probes")
WEIGHTS = os.path.join(REPO, "data/r55_aux/raw/pyiqa_weights")
METRICS = ["fgresq", "afine_nr", "topiq_nr", "topiq_nr-flive", "topiq_nr-spaq", "unique", "wadiqam_nr", "cnniqa"]
ONE_AT_A_TIME = {"afine_nr"}


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

    # TOPIQ imports facexlib for its face variant only
    if "facexlib" not in sys.modules:
        for name in ("facexlib", "facexlib.utils", "facexlib.utils.face_restoration_helper"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["facexlib.utils.face_restoration_helper"].FaceRestoreHelper = None

    # FGResQ on transformers >= 5
    from transformers import CLIPVisionModel
    if not hasattr(CLIPVisionModel, "vision_model"):
        CLIPVisionModel.vision_model = property(lambda self: self)
    import pyiqa
    import pyiqa.archs.fgresq_arch as fa
    _lc = fa.load_checkpoint

    def load_checkpoint(path):
        sd = _lc(path)
        return {k.replace("vision_model.", ""): v for k, v in sd.items()}
    fa.load_checkpoint = load_checkpoint

    dev = torch.device(f"cuda:{gpu}")
    ok, bad = {}, {}
    x = torch.rand(2, 3, 448, 448, device=dev)
    import warnings
    for m in METRICS:
        try:
            torch.manual_seed(0)
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                f = pyiqa.create_metric(m, device=dev)
            # unexpected keys are the unused CLIP text branch of the published checkpoints; MISSING keys would
            # mean part of the model stayed at its initial values
            mism = [str(w.message)[:200] for w in wl if "missing=" in str(w.message) and "missing=0 " not in str(w.message)]
            for w in wl:
                if "missing=0 " in str(w.message):
                    print(f"[iqa_bank2] {m}: {str(w.message)[:90]}", flush=True)
            if mism:
                raise RuntimeError("checkpoint mismatch: " + " | ".join(mism))
            with torch.no_grad():
                v = f(x[:1].clone()) if m in ONE_AT_A_TIME else f(x.clone())
            assert np.isfinite(np.asarray(v.detach().float().cpu())).all()
            ok[m] = f
        except Exception as e:
            bad[m] = f"{type(e).__name__}: {str(e)[:300]}"
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
        zs = [np.load(os.path.join(OUT, f"iqa_bank2_shard{r}.npz"), allow_pickle=True) for r in range(a.merge)]
        fn = list(zs[0]["feat_names"])
        assert all(list(z["feat_names"]) == fn for z in zs), "shards loaded different metric sets"
        names = np.concatenate([z["names"] for z in zs])
        X = np.concatenate([z["X"] for z in zs])
        np.savez_compressed(os.path.join(OUT, "iqa_bank2_feats.npz"), names=names, X=X, feat_names=np.array(fn))
        print(f"[iqa_bank2] merged {len(names)} images, {len(fn)} scalar features")
        if all("E" in z.files for z in zs):
            E = np.concatenate([z["E"] for z in zs]).astype(np.float64)
            E = E - E.mean(0)
            _, s, vt = np.linalg.svd(E, full_matrices=False)          # unsupervised: no label is used
            Z = (E @ vt[:16].T).astype(np.float32)
            np.savez_compressed(os.path.join(OUT, "fgresq_pca16_feats.npz"), names=names, X=Z,
                                feat_names=np.array([f"fgresq_pc{k}" for k in range(16)]))
            print(f"[iqa_bank2] FGResQ embedding: {E.shape[1]} dims -> 16 PCs, variance kept {float((s[:16] ** 2).sum() / (s ** 2).sum()):.3f}")
        return
    import torch
    from PIL import Image
    ok, bad, dev = setup(a.gpu)
    print(f"[iqa_bank2] {len(ok)} metrics loaded: {' '.join(ok)}", flush=True)
    for m, e in bad.items():
        print(f"[iqa_bank2] skipped {m}: {e}", flush=True)
    if a.warmup:
        return
    names = [r["name"] for r in csv.DictReader(open(os.path.join(REPO, "data/qingyu_splits/fold_0.csv"), encoding="utf-8"))][a.rank::a.world]
    keys = [f"{m}:{s}" for m in ok for s in ("global", "crop_mean", "crop_min", "crop_max")]
    fg = ok.get("fgresq")
    rows, keep, embs = [], [], []
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
        s = min(1.0, 768 / max(w, h))
        gw, gh = max(224, int(round(w * s / 32)) * 32), max(224, int(round(h * s / 32)) * 32)
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
                    if m in ONE_AT_A_TIME:
                        vc = np.array([float(fn(xc[k:k + 1].clone()).float().mean()) for k in range(xc.shape[0])])
                    else:
                        vc = fn(xc.clone()).float().flatten().cpu().numpy()
                f += [vg, float(vc.mean()), float(vc.min()), float(vc.max())]
            except Exception:
                f += [np.nan] * 4
        if fg is not None:
            with torch.no_grad():
                e = fg.net.get_quality_features(fg.net.preprocess(xg.clone())).float().cpu().numpy()[0]
            embs.append(e)
        rows.append(f)
        keep.append(n)
        if (i + 1) % 50 == 0:
            print(f"[iqa_bank2] rank {a.rank}: {i + 1}/{len(names)}", flush=True)
    extra = dict(E=np.array(embs, dtype=np.float32)) if embs else {}
    np.savez_compressed(os.path.join(OUT, f"iqa_bank2_shard{a.rank}.npz"), names=np.array(keep), X=np.array(rows, dtype=np.float32),
                        feat_names=np.array(keys), **extra)
    print(f"[iqa_bank2] rank {a.rank}: wrote {len(keep)} images, {len(keys)} features")


if __name__ == "__main__":
    main()
