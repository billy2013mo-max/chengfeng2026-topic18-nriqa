#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/probe_ae_recon.py — E2: reconstruction error of a generative autoencoder as evidence for 过度生成.
Run with venv_flux/bin/python (needs diffusers); 1 GPU per shard; reads only the labelled images.

Hypothesis (docs/REASSESSMENT_AND_PLAN_2026-11-12.md §4 B): texture produced by a generative enhancer lies closer
to a generative model's manifold than camera texture does, so a latent autoencoder (FLUX AE, local weights)
reconstructs it better.  Every discriminative family grades 过度生成 equally badly (adjacent-grade AUC 0.70-0.75), so
this is a different kind of evidence, not another backbone.

Per image (native pixels, longer side capped at 1920, sides cropped to a multiple of 16):
  x -> AE -> x1 -> AE -> x2
  relative error of x1 against x in four Laplacian bands (fine -> coarse), globally and over a 6x6 grid of tiles
  (mean, sd, 10/50/90 % quantiles, min, max), the same split by tile texture energy (flat / mid / busy tiles),
  idempotence error x2 vs x1, error-gradient correlation, latent statistics.
Output: results/r55/probes/ae_recon_feats[_shardK].npz (names, X, feat_names); `--merge N` joins the shards.

Usage:
  for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r venv_flux/bin/python code/r55/probe_ae_recon.py --rank $r --world 4 & done; wait
  venv_flux/bin/python code/r55/probe_ae_recon.py --merge 4
"""
import argparse
import csv
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "results/r55/probes")
DIRS = ["data/images", "data/images_supp"]


def find(name):
    for d in DIRS:
        p = os.path.join(REPO, d, name)
        if os.path.exists(p):
            return p
    return None


def lap_bands(x, n=4):
    """x: (1,3,H,W) in [0,1] -> list of n band images (fine to coarse) at their own resolution."""
    import torch.nn.functional as F
    out, cur = [], x
    for _ in range(n):
        dn = F.avg_pool2d(cur, 2)
        up = F.interpolate(dn, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        out.append(cur - up)
        cur = dn
    return out


def tile_energy(t, g=6):
    """mean of t**2 over a g x g grid -> (g*g,)"""
    import torch.nn.functional as F
    e = (t ** 2).mean(1, keepdim=True)
    return F.adaptive_avg_pool2d(e, g).flatten().float().cpu().numpy()


def stats(v, prefix):
    q = np.quantile(v, [0.1, 0.5, 0.9])
    return {f"{prefix}_mean": v.mean(), f"{prefix}_sd": v.std(), f"{prefix}_q10": q[0], f"{prefix}_q50": q[1],
            f"{prefix}_q90": q[2], f"{prefix}_min": v.min(), f"{prefix}_max": v.max()}


def features(vae, pil, dev):
    import torch
    w, h = pil.size
    s = min(1.0, 1920.0 / max(w, h))
    if s < 1.0:
        pil = pil.resize((int(w * s), int(h * s)))
        w, h = pil.size
    w, h = w - w % 16, h - h % 16
    x = torch.from_numpy(np.asarray(pil.crop((0, 0, w, h)), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        def rec(t):
            lat = vae.encode(t * 2 - 1).latent_dist.mean
            return (vae.decode(lat).sample.clamp(-1, 1) + 1) / 2, lat
        x1, lat = rec(x)
        x2, _ = rec(x1)
    f = {}
    bx, b1, b2 = lap_bands(x), lap_bands(x1), lap_bands(x2)
    busy = tile_energy(bx[0])                                  # fine-band energy of the image per tile
    order = np.argsort(busy)
    third = len(order) // 3
    for k in range(4):
        ex = tile_energy(bx[k]) + 1e-8
        ee = tile_energy(b1[k] - bx[k])
        rel = np.log(ee / ex + 1e-8)                           # log relative error per tile
        f.update(stats(rel, f"b{k}_rel"))
        f[f"b{k}_rel_global"] = float(np.log(ee.mean() / ex.mean() + 1e-8))
        f[f"b{k}_energy"] = float(np.log(ex.mean()))
        f[f"b{k}_rel_flat"] = float(rel[order[:third]].mean())
        f[f"b{k}_rel_busy"] = float(rel[order[-third:]].mean())
        f[f"b{k}_gain"] = float(np.log((tile_energy(b1[k]).mean() + 1e-8) / ex.mean()))     # does the AE add or remove energy
        f[f"b{k}_idem"] = float(np.log(tile_energy(b2[k] - b1[k]).mean() / (tile_energy(b1[k]).mean() + 1e-8) + 1e-8))
    e = (x1 - x).abs().mean(1, keepdim=True)
    gx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(1, keepdim=True)[..., :-1, :]
    gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(1, keepdim=True)[..., :, :-1]
    g = (gx + gy).flatten().float()
    ee = e[..., :-1, :-1].flatten().float()
    f["err_l1"] = float(e.mean())
    f["err_grad_corr"] = float(torch.corrcoef(torch.stack([ee, g]))[0, 1])
    f["err_over_grad"] = float(ee.mean() / (g.mean() + 1e-6))
    lt = lat.float()
    f["lat_abs"] = float(lt.abs().mean())
    f["lat_sd_ch"] = float(lt.std(dim=(2, 3)).mean())
    ks = ((lt - lt.mean(dim=(2, 3), keepdim=True)) ** 4).mean(dim=(2, 3)) / (lt.var(dim=(2, 3)) ** 2 + 1e-8)
    f["lat_kurt"] = float(ks.mean())
    f["lat_tv"] = float((lt[..., 1:, :] - lt[..., :-1, :]).abs().mean() + (lt[..., :, 1:] - lt[..., :, :-1]).abs().mean())
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--merge", type=int, default=0)
    ap.add_argument("--vae", default="models/flux3-klein/vae")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.merge:
        zs = [np.load(os.path.join(OUT, f"ae_recon_feats_shard{r}.npz"), allow_pickle=True) for r in range(a.merge)]
        np.savez_compressed(os.path.join(OUT, "ae_recon_feats.npz"), names=np.concatenate([z["names"] for z in zs]),
                            X=np.concatenate([z["X"] for z in zs]), feat_names=zs[0]["feat_names"])
        print("[probe_ae_recon] merged", sum(len(z["names"]) for z in zs), "images,", zs[0]["X"].shape[1], "features")
        return
    import torch
    from diffusers import AutoencoderKLFlux2
    from PIL import Image
    dev = torch.device("cuda")
    vae = AutoencoderKLFlux2.from_pretrained(os.path.join(REPO, a.vae), torch_dtype=torch.float32).to(dev).eval()
    names = [r["name"] for r in csv.DictReader(open(os.path.join(REPO, "data/qingyu_splits/fold_0.csv"), encoding="utf-8"))]
    names = names[a.rank::a.world]
    rows, keep, keys = [], [], None
    for i, n in enumerate(names):
        p = find(n)
        if p is None:
            continue
        try:
            f = features(vae, Image.open(p).convert("RGB"), dev)
        except Exception as e:                                   # one unreadable / odd-sized image must not stop the shard
            print(f"[probe_ae_recon] skip {n}: {type(e).__name__} {str(e)[:120]}", flush=True)
            continue
        keys = keys or sorted(f)
        rows.append([f[k] for k in keys])
        keep.append(n)
        if (i + 1) % 100 == 0:
            print(f"[probe_ae_recon] rank {a.rank}: {i + 1}/{len(names)}", flush=True)
    np.savez_compressed(os.path.join(OUT, f"ae_recon_feats_shard{a.rank}.npz"), names=np.array(keep),
                        X=np.array(rows, dtype=np.float32), feat_names=np.array(keys))
    print(f"[probe_ae_recon] rank {a.rank}: wrote {len(keep)} images, {len(keys)} features")


if __name__ == "__main__":
    main()
