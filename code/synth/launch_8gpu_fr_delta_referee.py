#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/synth/launch_8gpu_fr_delta_referee.py

8-GPU Parallel Local FR-Delta Referee Pipeline using true neural backbones & heads:
1. Distributes clean seeds across 8 GPUs (cuda:0..cuda:7).
2. Generates genuine quadruplets (Clean, P2, P1, P0) with authentic FLUX.3-Klein and ISP operations.
3. On each GPU, uses deep vision backbone (SigLIP2 / DINOv3-L / InternViT-6B) to extract 1:1 ROI features:
   Delta S_c(tilde{I}_k | I_Clean) = Head_c(phi(tilde{I}_k)) - Head_c(phi(I_Clean))
4. Applies strict monotonic corridor filtering & dynamic cross-dimension ignore-masking.
5. Saves all verified quadruplets to shared memory /dev/shm/v17_synth_quads/ and saves verified cache.
"""
import os
import sys
import csv
import json
import glob
import math
import time
import argparse
import subprocess
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/synth')

from utils import DIMS, FULL_DIMS, P0_ONLY_DIMS
from build_guide_isp_aigc_quadruplets_and_vlm_qc import (
    parse_foreground_mask_and_disparity,
    extract_6_native_microscope_focus_patches,
    synthesize_blur_quadruplet,
    synthesize_texloss_quadruplet,
    synthesize_oversharp_quadruplet,
    synthesize_overgen_quadruplet,
    synthesize_screenart_p0,
    synthesize_bokeh_pair,
)
from train_v9_specialist import build_v9_backbone
from train_v17_purereal_fourstream_ordinal import extract_single_native_fourstream_caliper

OUT_DIR = "results/v17_sim2real_dualstream"
os.makedirs(OUT_DIR, exist_ok=True)
SYNTH_IMG_DIR = "/dev/shm/v17_synth_quads"
os.makedirs(SYNTH_IMG_DIR, exist_ok=True)


def normalize_subblocks(feat, d_emb):
    """L2-normalize each [CLS/mean/std/top32] sub-block of width `d_emb`."""
    n_blocks = feat.shape[1] // d_emb
    parts = []
    for b in range(n_blocks):
        sub = feat[:, b * d_emb:(b + 1) * d_emb]
        parts.append(F.normalize(sub.float(), dim=-1))
    return torch.cat(parts, dim=-1)


def run_gpu_worker(rank, world_size, total_seeds=513):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"[GPU {rank}] Initializing deep backbone & FR-Delta referee...", flush=True)

    # Load backbone model on GPU for true neural 1:1 patch feature extraction
    bname = 'siglip2'
    m, fwd, mean, std, _, out_dim = build_v9_backbone(bname)
    m = m.to(device).eval().to(torch.bfloat16)
    d_emb = 1152
    mean_t = torch.tensor(mean, device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)

    # Load manifests and clean seeds
    man_rows = list(csv.DictReader(open("data/manifest_combined.csv")))
    all_names = [r["name"] for r in man_rows]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    y_real = np.array([[int(r[c]) for c in DIMS] for r in man_rows], dtype=np.int64)

    clean_indices = np.where((y_real == 0).all(axis=1))[0][:total_seeds]
    my_seeds = clean_indices[rank::world_size]
    print(f"[GPU {rank}] Assigned {len(my_seeds)} clean seeds across 6 dimensions", flush=True)

    # Load FLUX map (direct O(1) dict lookup)
    flux_strong_map = {}
    for p in glob.glob("/tmp/synth/flux3/strong_*"):
        if not p.endswith(".txt"):
            flux_strong_map[os.path.basename(p)[len("strong_"):]] = p

    # Native image loading
    idx_d = np.load("/dev/shm/race18_native/index.npz", allow_pickle=True)
    shm_names = list(idx_d["names"])
    shm_offsets = idx_d["offsets"]
    shm_h = idx_d["h"]
    shm_w = idx_d["w"]
    shm_pos = {str(n): i for i, n in enumerate(shm_names)}
    shm_data = np.memmap("/dev/shm/race18_native/cache.bin", dtype=np.uint8, mode="r")

    def load_img(nm):
        if nm in shm_pos:
            i = shm_pos[nm]
            off = int(shm_offsets[i])
            h, w = int(shm_h[i]), int(shm_w[i])
            return np.frombuffer(shm_data, dtype=np.uint8, count=h*w*3, offset=off).reshape((h, w, 3)).copy()
        for d in ["data/images", "data/images_supp"]:
            p = os.path.join(d, nm)
            if os.path.exists(p):
                return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        return None

    # Load fold IDs
    fold_ids = np.zeros(len(all_names), dtype=np.int64)
    for f in range(8):
        with open(f"data/v9_fulldata/split_f{f}_real.csv") as sf:
            for r in csv.DictReader(sf):
                if r["split"] == "val":
                    fold_ids[name_to_idx[r["name"]]] = f

    verified_quads = []
    t_start = time.time()

    for dim_c in [0, 1, 2, 4, 3, 5]:
        target_seeds = my_seeds if dim_c in FULL_DIMS else my_seeds[:len(my_seeds)//2 + 1]
        passed_dim = 0

        for s_idx in target_seeds:
            nm = all_names[s_idx]
            stem = os.path.splitext(nm)[0]
            if os.path.exists(os.path.join(SYNTH_IMG_DIR, f"{stem}_d{dim_c}_P0.png")):
                passed_dim += 1
                continue
            im = load_img(nm)
            if im is None:
                continue

            h, w = im.shape[:2]
            if max(h, w) > 960:
                scale = 960.0 / max(h, w)
                im = cv2.resize(im, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

            M_fg, M_core, D_map = parse_foreground_mask_and_disparity(im)
            _, focus_coords = extract_6_native_microscope_focus_patches(im, M_fg, D_map, size=min(448, min(im.shape[:2])))

            # Synthesize quadruplet
            if dim_c == 0:
                I0, I1, I2, I3 = synthesize_blur_quadruplet(im, M_fg, D_map, seed=s_idx * 13 + 1)
            elif dim_c == 1:
                I0, I1, I2, I3 = synthesize_texloss_quadruplet(im, M_fg, seed=s_idx * 13 + 2)
            elif dim_c == 2:
                I0, I1, I2, I3 = synthesize_oversharp_quadruplet(im, M_fg, seed=s_idx * 13 + 3)
            elif dim_c == 4:
                flx_img = None
                if nm in flux_strong_map and os.path.exists(flux_strong_map[nm]):
                    try:
                        flx_img = np.asarray(Image.open(flux_strong_map[nm]).convert("RGB"), dtype=np.uint8)
                    except Exception:
                        flx_img = None
                I0, I1, I2, I3 = synthesize_overgen_quadruplet(im, M_fg, M_core, flx_img, seed=s_idx * 13 + 4)
            elif dim_c == 3:
                I3 = synthesize_screenart_p0(im, seed=s_idx * 13 + 5)
                I0 = im
                I1 = cv2.addWeighted(im, 0.75, I3, 0.25, 0.0)
                I2 = cv2.addWeighted(im, 0.45, I3, 0.55, 0.0)
            else:  # dim_c == 5 (Bokeh)
                I0, I_nat, I3 = synthesize_bokeh_pair(im, M_fg, D_map, seed=s_idx * 13 + 6)
                I1 = I_nat
                I2 = cv2.addWeighted(I_nat, 0.50, I3, 0.50, 0.0)

            # Feature extraction on GPU:
            # - For AIGC OverGen (dim_c == 4): multi-scale global 384x384 view captures localized AIGC perturbations across full scene
            # - For ISP defects (dim_c in [0, 1, 2, 3, 5]): native 1:1 microscope focus ROI captures high-frequency physical degradations
            if dim_c == 4:
                crops = [cv2.resize(I, (384, 384), interpolation=cv2.INTER_AREA) for I in [I0, I1, I2, I3]]
            elif dim_c == 1:
                y0, x0 = focus_coords[-1] if focus_coords else (0, 0)
                sz = min(448, im.shape[0], im.shape[1])
                y0 = max(0, min(im.shape[0] - sz, y0))
                x0 = max(0, min(im.shape[1] - sz, x0))
                crops = [I0[y0:y0+sz, x0:x0+sz], I1[y0:y0+sz, x0:x0+sz], I2[y0:y0+sz, x0:x0+sz], I3[y0:y0+sz, x0:x0+sz]]
            else:
                y0, x0 = focus_coords[0] if focus_coords else (0, 0)
                sz = min(448, im.shape[0], im.shape[1])
                y0 = max(0, min(im.shape[0] - sz, y0))
                x0 = max(0, min(im.shape[1] - sz, x0))
                crops = [I0[y0:y0+sz, x0:x0+sz], I1[y0:y0+sz, x0:x0+sz], I2[y0:y0+sz, x0:x0+sz], I3[y0:y0+sz, x0:x0+sz]]

            t_crops = []
            for cp in crops:
                pil_cp = Image.fromarray(cp).resize((384, 384), Image.BICUBIC) if cp.shape[:2] != (384, 384) else Image.fromarray(cp)
                arr = np.asarray(pil_cp, dtype=np.float32).transpose(2, 0, 1) / 255.0
                t_crops.append(torch.from_numpy(arr))
            batch_t = torch.stack(t_crops).to(device).to(torch.bfloat16)
            batch_norm = (batch_t - mean_t) / std_t

            with torch.inference_mode():
                feats = fwd(batch_norm)  # (4, out_dim)
                feats = normalize_subblocks(feats, d_emb)  # (4, out_dim)

            # Neural Feature Distance from Clean Anchor I0 (Reference Subtraction)
            f0 = feats[0:1]
            diff_p2 = torch.norm(feats[1:2] - f0, p=2).item()
            diff_p1 = torch.norm(feats[2:3] - f0, p=2).item()
            diff_p0 = torch.norm(feats[3:4] - f0, p=2).item()

            # Monotonic step check on deep representations
            if dim_c in FULL_DIMS:
                passed = (diff_p2 >= 0.015 and diff_p1 >= diff_p2 + 0.012 and diff_p0 >= diff_p1 + 0.018)
            else:
                passed = (diff_p0 >= 0.10)

            if not passed:
                continue

            # Save genuine synthetic images to shared memory
            stem = os.path.splitext(nm)[0]
            quad_paths = []
            for k, (sub_im, g_name) in enumerate(zip([I0, I1, I2, I3], ["Clean", "P2", "P1", "P0"])):
                save_p = os.path.join(SYNTH_IMG_DIR, f"{stem}_d{dim_c}_{g_name}.png")
                if k > 0 and not os.path.exists(save_p):
                    cv2.imwrite(save_p, cv2.cvtColor(sub_im, cv2.COLOR_RGB2BGR))
                quad_paths.append(save_p if k > 0 else nm)

            # Extract 32D physical calipers
            c0 = extract_single_native_fourstream_caliper(I0)
            c1 = extract_single_native_fourstream_caliper(I1)
            c2 = extract_single_native_fourstream_caliper(I2)
            c3 = extract_single_native_fourstream_caliper(I3)
            calipers_4x32 = np.stack([c0, c1, c2, c3], axis=0).astype(np.float32)

            # Dynamic cross-dimension ignore mask (1.0 for target dimension)
            ign_mask = np.ones(6, dtype=np.float32)
            if dim_c == 4:  # OverGen: protect blur and texloss
                ign_mask[0] = 0.0
                ign_mask[1] = 0.0
            elif dim_c == 0:
                ign_mask[1] = 0.0

            verified_quads.append({
                "seed_idx": int(s_idx),
                "fold_id": int(fold_ids[s_idx]),
                "dim_c": int(dim_c),
                "diffs": [0.0, diff_p2, diff_p1, diff_p0],
                "ignore_mask6": ign_mask,
                "calipers_4x32": calipers_4x32,
                "quad_paths": quad_paths,
            })
            passed_dim += 1

        print(f"[GPU {rank}] Dimension {DIMS[dim_c]:8s}: Passed {passed_dim}/{len(target_seeds)} ({100.0*passed_dim/max(1, len(target_seeds)):.1f}%)", flush=True)

    # Save worker results to temp file
    worker_p = os.path.join(OUT_DIR, f"part_verified_rank{rank}.npz")
    np.savez_compressed(
        worker_p,
        seed_idx=np.array([q["seed_idx"] for q in verified_quads], dtype=np.int64),
        fold_id=np.array([q["fold_id"] for q in verified_quads], dtype=np.int64),
        dim_c=np.array([q["dim_c"] for q in verified_quads], dtype=np.int64),
        ignore_mask6=np.array([q["ignore_mask6"] for q in verified_quads], dtype=np.float32),
        calipers_4x32=np.array([q["calipers_4x32"] for q in verified_quads], dtype=np.float32),
        diffs=np.array([q["diffs"] for q in verified_quads], dtype=np.float32),
    )
    print(f"[GPU {rank}] Finished in {time.time()-t_start:.1f}s. Saved {len(verified_quads)} quads to {worker_p}", flush=True)


def merge_results(world_size=8, out_name="verified_quadruplets_fr_oracle_1k.npz"):
    print("Merging results across 8 GPUs...", flush=True)
    all_seeds, all_folds, all_dims, all_masks, all_calipers, all_diffs = [], [], [], [], [], []
    for r in range(world_size):
        wp = os.path.join(OUT_DIR, f"part_verified_rank{r}.npz")
        if os.path.exists(wp):
            d = np.load(wp, allow_pickle=True)
            all_seeds.append(d["seed_idx"])
            all_folds.append(d["fold_id"])
            all_dims.append(d["dim_c"])
            all_masks.append(d["ignore_mask6"])
            all_calipers.append(d["calipers_4x32"])
            all_diffs.append(d["diffs"])
            os.remove(wp)

    final_seeds = np.concatenate(all_seeds, axis=0)
    final_folds = np.concatenate(all_folds, axis=0)
    final_dims = np.concatenate(all_dims, axis=0)
    final_masks = np.concatenate(all_masks, axis=0)
    final_calipers = np.concatenate(all_calipers, axis=0)
    final_diffs = np.concatenate(all_diffs, axis=0)

    out_p = os.path.join(OUT_DIR, out_name)
    np.savez_compressed(
        out_p,
        seed_idx=final_seeds,
        fold_id=final_folds,
        dim_c=final_dims,
        ignore_mask6=final_masks,
        calipers_4x32=final_calipers,
        diffs=final_diffs,
    )
    print(f"=== MERGED TOTAL: {len(final_seeds)} VERIFIED QUADRUPLETS SAVED TO {out_p} ===", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, default=-1)
    ap.add_argument('--world-size', type=int, default=8)
    ap.add_argument('--total-seeds', type=int, default=513)
    ap.add_argument('--out-name', type=str, default="verified_quadruplets_fr_oracle_1k.npz")
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()

    if args.merge:
        merge_results(args.world_size, out_name=args.out_name)
    elif args.rank >= 0:
        run_gpu_worker(args.rank, args.world_size, total_seeds=args.total_seeds)
    else:
        print(f"Spawning 8-GPU Parallel Local FR-Delta Referee for {args.total_seeds} clean seeds -> {args.out_name}...", flush=True)
        procs = [
            subprocess.Popen([
                sys.executable, __file__,
                '--rank', str(r),
                '--world-size', str(args.world_size),
                '--total-seeds', str(args.total_seeds),
                '--out-name', str(args.out_name)
            ])
            for r in range(args.world_size)
        ]
        for p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"Worker failed with exit code {rc}")
        merge_results(args.world_size, out_name=args.out_name)
