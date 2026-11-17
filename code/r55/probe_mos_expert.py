#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/probe_mos_expert.py — E1 (local variant): do independent HUMAN quality scores add information to the ensemble?

docs/REASSESSMENT_AND_PLAN_2026-11-12.md §4 A/D.  The planned bank of public pretrained IQA metrics could not be
run here (downloading and executing third-party weights was not permitted in this session), so the same question
is asked with local resources only: KVQ gives one human mean opinion score per video (2,926 videos, 5,657 frames in
data/r55_aux).  A ridge regressor from FROZEN pretrained features (PE-Core-L and DINOv3-L, local weights, never
fine-tuned on our labels) to KVQ MOS is a "MOS expert" whose knowledge comes only from KVQ's raters.  Its predictions
on the labelled images form the feature bank that probe_eval.py tests against the pool.

  --extract   frozen features for the KVQ frames and the labelled images (one GPU per shard)
  --fit       merge shards; 5-fold CV on KVQ grouped by SOURCE clip reports how well the expert predicts MOS;
              final fit on all KVQ frames -> results/r55/probes/mos_expert_feats.npz (labelled images only)

Views per image: the whole frame resized to 448x448 and five native-pixel 448 crops (centre + four quadrant centres).
Per view: class token and mean patch token.  Image feature = [mean over crops, global view].

Usage:
  for r in 0 1 2 3 4 5 6 7; do python code/r55/probe_mos_expert.py --extract --rank $r --world 8 --gpu $r & done; wait
  python code/r55/probe_mos_expert.py --fit --world 8
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results/r55/probes")
BACKBONES = {"pe": ("vit_pe_core_large_patch14_336", "models/pe-core-l14/model.safetensors"),
             "dino": ("vit_large_patch16_dinov3", "models/dinov3-vitl16/model.safetensors")}


def image_list():
    lab = [r["name"] for r in csv.DictReader(open(os.path.join(REPO, "data/qingyu_splits/fold_0.csv"), encoding="utf-8"))]
    kvq = [r["name"] for r in csv.DictReader(open(os.path.join(REPO, "data/r55_aux/kvq_frames.csv"), encoding="utf-8"))]
    out = []
    for n in lab:
        for d in ("data/images", "data/images_supp"):
            p = os.path.join(REPO, d, n)
            if os.path.exists(p):
                out.append((n, p, 0))
                break
    out += [(n, os.path.join(REPO, "data/r55_aux/images", n), 1) for n in kvq]
    return out


def views(path, size=448):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if min(w, h) < size:                                        # rare: upscale just enough to cut native-size crops
        s = size / min(w, h)
        im = im.resize((max(size, int(round(w * s))), max(size, int(round(h * s)))), Image.BICUBIC)
        w, h = im.size
    out = [im.resize((size, size), Image.BILINEAR)]
    for cx, cy in ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
        x0 = int(min(max(0, cx * w - size / 2), w - size))
        y0 = int(min(max(0, cy * h - size / 2), h - size))
        out.append(im.crop((x0, y0, x0 + size, y0 + size)))
    return np.stack([np.asarray(v, dtype=np.uint8) for v in out])            # (6, S, S, 3)


def extract(a):
    import timm
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, Dataset
    dev = torch.device(f"cuda:{a.gpu}")
    items = image_list()[a.rank::a.world]

    class DS(Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            return i, torch.from_numpy(views(items[i][1]))

    models = {}
    for k, (arch, w) in BACKBONES.items():
        m = timm.create_model(arch, pretrained=False, num_classes=0, dynamic_img_size=True)
        m.load_state_dict(load_file(os.path.join(REPO, w)), strict=False)
        dc = timm.data.resolve_data_config({}, model=m)
        models[k] = (m.to(dev).eval(), torch.tensor(dc["mean"], device=dev).view(1, 3, 1, 1), torch.tensor(dc["std"], device=dev).view(1, 3, 1, 1),
                     int(getattr(m, "num_prefix_tokens", 1)))
    feats = {k: np.zeros((len(items), 4 * models[k][0].embed_dim), dtype=np.float32) for k in models}
    dl = DataLoader(DS(), batch_size=8, num_workers=8, shuffle=False)
    done = 0
    with torch.no_grad():
        for idx, v in dl:
            b = v.shape[0]
            x = v.to(dev).permute(0, 1, 4, 2, 3).reshape(b * 6, 3, v.shape[2], v.shape[3]).float() / 255.0
            for k, (m, mu, sd, npre) in models.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):                   # fp16 overflows in DINOv3; bf16 does not
                    t = m.forward_features((x - mu) / sd).float()                    # (b*6, T, D)
                t = torch.nan_to_num(t)
                f = torch.cat([t[:, 0], t[:, npre:].mean(1)], 1).view(b, 6, -1)      # class token + mean patch token
                feats[k][idx.numpy()] = torch.cat([f[:, 1:].mean(1), f[:, 0]], 1).cpu().numpy().astype(np.float32)
            done += b
            if done % 400 < 8:
                print(f"[mos_expert] rank {a.rank}: {done}/{len(items)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"mos_expert_raw_shard{a.rank}.npz"), names=np.array([x[0] for x in items]),
                        is_kvq=np.array([x[2] for x in items]), **{f"F_{k}": v for k, v in feats.items()})
    print(f"[mos_expert] rank {a.rank}: wrote {len(items)} images")


def fit(a):
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    zs = [np.load(os.path.join(OUT, f"mos_expert_raw_shard{r}.npz"), allow_pickle=True) for r in range(a.world)]
    names = np.concatenate([z["names"] for z in zs])
    kv = np.concatenate([z["is_kvq"] for z in zs]).astype(bool)
    F = {k: np.concatenate([z[f"F_{k}"] for z in zs]).astype(np.float32) for k in BACKBONES}
    meta = {r["name"]: (float(r["mos"]), r["video"]) for r in csv.DictReader(open(os.path.join(REPO, "data/r55_aux/kvq_frames.csv"), encoding="utf-8"))}
    y = np.array([meta[n][0] for n in names[kv]])
    vid = np.array([meta[n][1] for n in names[kv]])
    uv = np.unique(vid)
    # KVQ holds ~7 processed versions of every source clip under different video ids.  Folds by video would let the
    # regressor recognise the content (frame-level SRCC 0.96 that way); group by SOURCE instead: frames whose global
    # DINOv3 features are near-identical, or that belong to the same video, are one source.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    g = F["dino"][kv][:, F["dino"].shape[1] // 2:]
    g = g - g.mean(0)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
    S = (g @ g.T) > 0.85
    S |= vid[:, None] == vid[None, :]
    n_src, src = connected_components(csr_matrix(S), directed=False)
    rs = np.random.RandomState(0)
    sf = rs.randint(0, 5, n_src)
    fold = sf[src]
    print(f"[mos_expert] {n_src} source clips found among {len(uv)} videos (median {np.median(np.bincount(src)):.0f} frames per source)")
    d = F["pe"].shape[1] // 2
    sets = {"pe": F["pe"], "dino": F["dino"], "both": np.concatenate([F["pe"], F["dino"]], 1),
            "pe_crops": F["pe"][:, :d], "pe_global": F["pe"][:, d:], "dino_crops": F["dino"][:, :d], "dino_global": F["dino"][:, d:]}
    X_out, fn = [], []
    print(f"[mos_expert] KVQ frames {int(kv.sum())} of {len(uv)} videos | labelled images {int((~kv).sum())}")
    for k, X in sets.items():
        sc = StandardScaler().fit(X[kv])
        Xk, Xl = sc.transform(X[kv]), sc.transform(X[~kv])
        best = None
        for alpha in (1e2, 1e3, 1e4, 1e5, 1e6):                               # chosen on KVQ only (source-grouped CV), never on our labels
            oof = np.zeros(len(y))
            for f in range(5):
                tr, te = fold != f, fold == f
                oof[te] = Ridge(alpha=alpha).fit(Xk[tr], y[tr]).predict(Xk[te])
            s = spearmanr(y, oof).correlation
            if best is None or s > best[0]:
                best = (s, alpha)
        print(f"  {k:12s} dim {X.shape[1]:5d}: KVQ frame-level SRCC {best[0]:.4f} (5 folds grouped by source clip, alpha {best[1]:g})")
        X_out.append(Ridge(alpha=best[1]).fit(Xk, y).predict(Xl))
        fn.append(f"mos_{k}")
    X_out = np.stack(X_out, 1).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, "mos_expert_feats.npz"), names=names[~kv], X=X_out, feat_names=np.array(fn))
    print(f"[mos_expert] wrote results/r55/probes/mos_expert_feats.npz: {X_out.shape}; predicted MOS on labelled images "
          f"mean {X_out[:, 2].mean():.2f} sd {X_out[:, 2].std():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    if args.extract:
        extract(args)
    if args.fit:
        fit(args)
