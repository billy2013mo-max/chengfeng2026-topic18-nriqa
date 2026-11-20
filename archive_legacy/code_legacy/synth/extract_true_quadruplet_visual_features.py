#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/synth/extract_true_quadruplet_visual_features.py

Extracts TRUE neural visual features (using SigLIP2-SO400M / DINOv3) for each state
(I0: Clean, I1: P2, I2: P1, I3: P0) of the verified quadruplets in /dev/shm/v17_synth_quads/
specifically focusing on the proven positive channels:
  - Dim 1: 纹理损失 (Texture Loss)
  - Dim 4: 过度生成 (AI OverGen from FLUX.3-Klein)

Eliminates the dummy feature bug and saves genuine multi-state representations.
"""
import os
import sys
import glob
import time
import csv
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')

from utils import DIMS, FULL_DIMS
from train_v9_specialist import build_v9_backbone

OUT_PATH = "results/v17_sim2real_dualstream/verified_quadruplets_features_true.npz"


def main():
    print("=" * 80)
    print("=== Extracting True Visual Features for Verified Quadruplets (Dims 1 & 4) ===")
    print("=" * 80)
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build SigLIP2 backbone
    m, fwd, mean, std, _, out_dim = build_v9_backbone('siglip2')
    m = m.to(device).eval().to(torch.bfloat16)
    d_emb = 1152
    mean_t = torch.tensor(mean, device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)

    # Load manifests and folds
    man_rows = list(csv.DictReader(open("data/manifest_combined.csv")))
    all_names = [r["name"] for r in man_rows]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    stem_to_name = {os.path.splitext(n)[0]: n for n in all_names}

    fold_ids = np.zeros(len(all_names), dtype=np.int64)
    for f in range(8):
        with open(f"data/v9_fulldata/split_f{f}_real.csv") as sf:
            for r in csv.DictReader(sf):
                if r["split"] == "val":
                    fold_ids[name_to_idx[r["name"]]] = f

    # Load native clean cache
    idx_d = np.load("/dev/shm/race18_native/index.npz", allow_pickle=True)
    shm_names = list(idx_d["names"])
    shm_offsets = idx_d["offsets"]
    shm_h, shm_w = idx_d["h"], idx_d["w"]
    shm_pos = {str(n): i for i, n in enumerate(shm_names)}
    shm_data = np.memmap("/dev/shm/race18_native/cache.bin", dtype=np.uint8, mode="r")

    def load_clean(nm):
        if nm in shm_pos:
            i = shm_pos[nm]
            off, h, w = int(shm_offsets[i]), int(shm_h[i]), int(shm_w[i])
            return np.frombuffer(shm_data, dtype=np.uint8, count=h * w * 3, offset=off).reshape((h, w, 3)).copy()
        for d in ["data/images", "data/images_supp"]:
            p = os.path.join(d, nm)
            if os.path.exists(p):
                return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        return None

    # Find all verified quadruplets for dim 1 and dim 4
    quad_items = []
    for dim_c in [1, 4]:
        p0_files = sorted(glob.glob(f"/dev/shm/v17_synth_quads/*_d{dim_c}_P0.png"))
        for p0_path in p0_files:
            bn = os.path.basename(p0_path)
            stem = bn[:bn.index(f"_d{dim_c}_P0.png")]
            p1_path = p0_path.replace("_P0.png", "_P1.png")
            p2_path = p0_path.replace("_P0.png", "_P2.png")
            if os.path.exists(p1_path) and os.path.exists(p2_path) and (stem in stem_to_name):
                clean_name = stem_to_name[stem]
                s_idx = name_to_idx[clean_name]
                quad_items.append({
                    "dim_c": dim_c,
                    "stem": stem,
                    "clean_name": clean_name,
                    "seed_idx": s_idx,
                    "fold_id": int(fold_ids[s_idx]),
                    "paths": [clean_name, p2_path, p1_path, p0_path],
                })

    print(f"Total quadruplets found: {len(quad_items)} (Dim 1: {sum(1 for q in quad_items if q['dim_c']==1)}, Dim 4: {sum(1 for q in quad_items if q['dim_c']==4)})")

    # Batch feature extraction for all 4 states
    def prep_tensor(img_arr):
        pil_im = Image.fromarray(img_arr).resize((384, 384), Image.BICUBIC) if img_arr.shape[:2] != (384, 384) else Image.fromarray(img_arr)
        t = torch.from_numpy(np.asarray(pil_im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return t

    feats_4states = []
    bs = 32

    # Group all images to extract: 4 images per quadruplet
    all_imgs_to_extract = []
    for q in quad_items:
        clean_img = load_clean(q["clean_name"])
        p2_img = np.asarray(Image.open(q["paths"][1]).convert("RGB"))
        p1_img = np.asarray(Image.open(q["paths"][2]).convert("RGB"))
        p0_img = np.asarray(Image.open(q["paths"][3]).convert("RGB"))
        all_imgs_to_extract.extend([clean_img, p2_img, p1_img, p0_img])

    print(f"Total images to extract features for: {len(all_imgs_to_extract)} images")
    extracted_feats = []

    for b0 in range(0, len(all_imgs_to_extract), bs):
        b_imgs = all_imgs_to_extract[b0:b0 + bs]
        b_t = torch.stack([prep_tensor(im) for im in b_imgs]).to(device).to(torch.bfloat16)
        b_norm = (b_t - mean_t) / std_t
        with torch.inference_mode():
            f = fwd(b_norm)
            f_norm = F.normalize(f.float(), dim=-1).cpu().numpy().astype(np.float32)
            extracted_feats.append(f_norm)

    all_feats = np.concatenate(extracted_feats, axis=0)  # (N_quads * 4, out_dim)
    quad_feats = all_feats.reshape(len(quad_items), 4, -1)  # (N_quads, 4, out_dim)

    # Verification: check delta monotonicity
    # Delta_p = ||feat_p - feat_0||
    f0 = quad_feats[:, 0:1, :]
    deltas = np.linalg.norm(quad_feats - f0, axis=-1)  # (N_quads, 4)
    print(f"\nMean neural distance from Clean anchor across all {len(quad_items)} quads:")
    print(f"  Clean: {np.mean(deltas[:, 0]):.4f}")
    print(f"  P2:    {np.mean(deltas[:, 1]):.4f}")
    print(f"  P1:    {np.mean(deltas[:, 2]):.4f}")
    print(f"  P0:    {np.mean(deltas[:, 3]):.4f}")

    # Monotonicity check: P2 < P1 < P0
    monotonic = (deltas[:, 1] < deltas[:, 2]) & (deltas[:, 2] < deltas[:, 3])
    print(f"Strict Monotonicity Rate (Clean < P2 < P1 < P0): {np.sum(monotonic)}/{len(quad_items)} ({100.0 * np.sum(monotonic)/len(quad_items):.1f}%)")

    # Save to file
    out_dict = {
        "seed_idx": np.array([q["seed_idx"] for q in quad_items], dtype=np.int64),
        "fold_id": np.array([q["fold_id"] for q in quad_items], dtype=np.int64),
        "dim_c": np.array([q["dim_c"] for q in quad_items], dtype=np.int64),
        "feats_4xdim": quad_feats,  # (N_quads, 4, 3456)
        "deltas_4": deltas,         # (N_quads, 4)
    }
    np.savez_compressed(OUT_PATH, **out_dict)
    print(f"\nSuccessfully saved TRUE quadruplet features to {OUT_PATH} in {time.time()-t0:.1f}s.")


if __name__ == '__main__':
    main()
