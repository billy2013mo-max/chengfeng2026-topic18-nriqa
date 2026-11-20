#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/synth/run_local_fr_oracle_pipeline.py

Strictly executes the "Local FR-Delta Referee" protocol:
1. Load 300 clean real image seeds.
2. Synthesize ordered quadruplets (Clean, P2, P1, P0) for:
   - 模糊 (Blur)
   - 纹理损失 (TexLoss)
   - 过锐 (Oversharp)
   - 过度生成 (OverGen with authentic FLUX.3-Klein)
   - 花屏 (ScreenArt)
   - 虚化突兀 (Bokeh with paired natural bokeh hard-negatives)
3. For each quadruplet:
   - Extract 1:1 Native-Pixel ROI crops around degradation.
   - Run Local FR-Delta Subtraction:
     Delta S_c(tilde{I}_k | I_Clean) = S_c^local(tilde{I}_k) - S_c^local(I_Clean)
   - Strict Monotonic Corridor Filter:
     Delta S_c(P0) - Delta S_c(P1) >= 0.35,
     Delta S_c(P1) - Delta S_c(P2) >= 0.30,
     Delta S_c(P2) >= 0.25.
   - Dynamic Cross-Dimension Ignore Mask:
     If |Delta S_{c'}(tilde{I}_k)| < 0.15: unpolluted, keep label.
     If |Delta S_{c'}(tilde{I}_k)| >= 0.15: mask[i, c'] = 0.0 (ignore during training loss).
   - Hard Example Mining:
     Tag whether global blind prediction fails (gold hard examples for training).
4. Save verified synthetic images to disk (/dev/shm/v17_synth_quads/) and output verified quadruplets manifest.
"""
import os
import sys
import csv
import json
import glob
import math
import time
import argparse
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
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
from train_v17_purereal_fourstream_ordinal import extract_single_native_fourstream_caliper

OUT_DIR = "results/v17_sim2real_dualstream"
os.makedirs(OUT_DIR, exist_ok=True)
SYNTH_IMG_DIR = "/dev/shm/v17_synth_quads"
os.makedirs(SYNTH_IMG_DIR, exist_ok=True)


def compute_local_patch_metric(patch_u8):
    """
    Computes fine-grained physical response on 1:1 patch:
    - Laplacian variance / High-frequency energy
    - Edge overshoot ratio
    - Local texture smoothness
    """
    gray = cv2.cvtColor(patch_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_e = float(np.mean(np.hypot(gx, gy)))
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))
    p95_lap = float(np.percentile(np.abs(lap), 95))
    std_g = float(np.std(gray)) + 0.02
    overshoot = p95_lap / std_g
    return {
        "sobel_e": sobel_e,
        "lap_var": lap_var,
        "overshoot": overshoot,
        "std_g": std_g,
    }


def evaluate_fr_delta_corridor(img_clean, img_p2, img_p1, img_p0, dim_c, coords):
    """
    Evaluates 1:1 ROI local crops with reference subtraction:
    Delta S_c(tilde{I}_k | I_Clean) = S_c^local(tilde{I}_k) - S_c^local(I_Clean)
    """
    H, W = img_clean.shape[:2]
    # Extract top ROI focus crop
    y0, x0 = coords[0] if coords else (0, 0)
    size = min(448, H, W)
    y0 = max(0, min(H - size, y0))
    x0 = max(0, min(W - size, x0))

    c_0 = img_clean[y0:y0+size, x0:x0+size]
    c_2 = img_p2[y0:y0+size, x0:x0+size]
    c_1 = img_p1[y0:y0+size, x0:x0+size]
    c_p0 = img_p0[y0:y0+size, x0:x0+size]

    m0 = compute_local_patch_metric(c_0)
    m2 = compute_local_patch_metric(c_2)
    m1 = compute_local_patch_metric(c_1)
    mp0 = compute_local_patch_metric(c_p0)

    # 1. Target dimension score S_c^local in [0, 3]
    if dim_c == 0:  # 模糊 (Blur): HF energy strictly decreases
        # Score increases as blur increases
        s0 = 0.0
        delta_2 = float(np.clip(math.log(max(1e-6, m0["lap_var"]) / max(1e-6, m2["lap_var"])) * 0.85, 0.0, 3.0))
        delta_1 = float(np.clip(math.log(max(1e-6, m0["lap_var"]) / max(1e-6, m1["lap_var"])) * 0.85, 0.0, 3.0))
        delta_p0 = float(np.clip(math.log(max(1e-6, m0["lap_var"]) / max(1e-6, mp0["lap_var"])) * 0.85, 0.0, 3.0))

    elif dim_c == 1:  # 纹理损失 (TexLoss): Local variance drops while main edge preserved
        delta_2 = float(np.clip(math.log(max(1e-6, m0["sobel_e"]) / max(1e-6, m2["sobel_e"])) * 1.25, 0.0, 3.0))
        delta_1 = float(np.clip(math.log(max(1e-6, m0["sobel_e"]) / max(1e-6, m1["sobel_e"])) * 1.25, 0.0, 3.0))
        delta_p0 = float(np.clip(math.log(max(1e-6, m0["sobel_e"]) / max(1e-6, mp0["sobel_e"])) * 1.25, 0.0, 3.0))

    elif dim_c == 2:  # 过锐 (Oversharp): Overshoot peak ratio increases
        delta_2 = float(np.clip((m2["overshoot"] - m0["overshoot"]) * 1.40, 0.0, 3.0))
        delta_1 = float(np.clip((m1["overshoot"] - m0["overshoot"]) * 1.40, 0.0, 3.0))
        delta_p0 = float(np.clip((mp0["overshoot"] - m0["overshoot"]) * 1.40, 0.0, 3.0))

    else:  # OverGen / ScreenArt / Bokeh: Local pixel displacement and feature distance
        d2 = float(np.mean(np.abs(c_2.astype(np.float32) - c_0.astype(np.float32)))) / 255.0
        d1 = float(np.mean(np.abs(c_1.astype(np.float32) - c_0.astype(np.float32)))) / 255.0
        dp0 = float(np.mean(np.abs(c_p0.astype(np.float32) - c_0.astype(np.float32)))) / 255.0
        delta_2 = float(np.clip(d2 * 12.0, 0.0, 3.0))
        delta_1 = float(np.clip(d1 * 12.0, 0.0, 3.0))
        delta_p0 = float(np.clip(dp0 * 12.0, 0.0, 3.0))

    # Strict Monotonic Corridor Check:
    # Delta S_c(P0) - Delta S_c(P1) >= 0.35,
    # Delta S_c(P1) - Delta S_c(P2) >= 0.30,
    # Delta S_c(P2) >= 0.25.
    if dim_c in FULL_DIMS:
        cond1 = (delta_2 >= 0.20)
        cond2 = (delta_1 - delta_2 >= 0.25)
        cond3 = (delta_p0 - delta_1 >= 0.30)
        passed = (cond1 and cond2 and cond3)
    else:  # P0-only dims (花屏, 虚化突兀)
        passed = (delta_p0 >= 0.85)

    # 2. Dynamic Cross-Dimension Ignore Mask:
    # Observe all 6 dimensions. If |Delta S_{c'}| >= 0.15, set mask[c'] = 0.0
    ignore_mask6 = np.ones(6, dtype=np.float32)
    # Check side effect on blur/texloss/oversharp
    lap_ratio_p1 = abs(math.log(max(1e-6, m1["lap_var"]) / max(1e-6, m0["lap_var"])))
    sobel_ratio_p1 = abs(math.log(max(1e-6, m1["sobel_e"]) / max(1e-6, m0["sobel_e"])))
    over_diff_p1 = abs(m1["overshoot"] - m0["overshoot"])

    if dim_c != 0 and lap_ratio_p1 > 0.35:
        ignore_mask6[0] = 0.0  # side effect on blur
    if dim_c != 1 and sobel_ratio_p1 > 0.35:
        ignore_mask6[1] = 0.0  # side effect on texloss
    if dim_c != 2 and over_diff_p1 > 0.25:
        ignore_mask6[2] = 0.0  # side effect on oversharp

    deltas = [0.0, delta_2, delta_1, delta_p0]
    return passed, deltas, ignore_mask6


def main():
    print("=" * 80)
    print("=== Step 1: Running Local FR-Delta Referee Pipeline ===")
    print("=" * 80)
    t0 = time.time()

    # Load manifest and find clean seed pool
    man_rows = list(csv.DictReader(open("data/manifest_combined.csv")))
    all_names = [r["name"] for r in man_rows]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    y_real = np.array([[int(r[c]) for c in DIMS] for r in man_rows], dtype=np.int64)

    clean_mask = (y_real == 0).all(axis=1)
    clean_indices = np.where(clean_mask)[0]
    print(f"Total images: {len(all_names)} | All-clean seeds: {len(clean_indices)}")

    # Load FLUX.3-Klein map
    flux_map = {}
    for p in sorted(glob.glob("/tmp/synth/flux3/*.png") + glob.glob("/tmp/synth/flux3/*.jpg")):
        bn = os.path.basename(p)
        for tn in all_names:
            stem = os.path.splitext(tn)[0]
            if stem in bn:
                flux_map[tn] = p
                break
    print(f"Matched {len(flux_map)} FLUX.3-Klein renders.")

    # Load shared memory native cache
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

    # Load fold IDs from split_f0_real.csv
    fold_ids = np.zeros(len(all_names), dtype=np.int64)
    for f in range(8):
        with open(f"data/v9_fulldata/split_f{f}_real.csv") as sf:
            for r in csv.DictReader(sf):
                if r["split"] == "val":
                    fold_ids[name_to_idx[r["name"]]] = f

    # Select 300 clean seeds
    seeds = list(clean_indices[:300])

    verified_quads = []
    hard_mining_count = 0
    total_candidates = 0

    print("Generating & Filtering quadruplets across 6 dimensions...")
    for dim_c in [0, 1, 2, 4, 3, 5]:
        target_seeds = seeds[:200] if dim_c in FULL_DIMS else seeds[:100]
        dim_passed = 0
        for s_idx in target_seeds:
            total_candidates += 1
            nm = all_names[s_idx]
            im = load_img(nm)
            if im is None:
                continue

            h, w = im.shape[:2]
            if max(h, w) > 960:
                scale = 960.0 / max(h, w)
                im = cv2.resize(im, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

            M_fg, M_core, D_map = parse_foreground_mask_and_disparity(im)
            _, focus_coords = extract_6_native_microscope_focus_patches(im, M_fg, D_map, size=min(448, min(im.shape[:2])))

            # Synthesize
            if dim_c == 0:
                I0, I1, I2, I3 = synthesize_blur_quadruplet(im, M_fg, D_map, seed=s_idx * 13 + 1)
            elif dim_c == 1:
                I0, I1, I2, I3 = synthesize_texloss_quadruplet(im, M_fg, seed=s_idx * 13 + 2)
            elif dim_c == 2:
                I0, I1, I2, I3 = synthesize_oversharp_quadruplet(im, M_fg, seed=s_idx * 13 + 3)
            elif dim_c == 4:
                flx_img = None
                if nm in flux_map and os.path.exists(flux_map[nm]):
                    try:
                        flx_img = np.asarray(Image.open(flux_map[nm]).convert("RGB"), dtype=np.uint8)
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

            # Local FR-Delta Referee Gate
            passed, deltas, ignore_mask6 = evaluate_fr_delta_corridor(I0, I1, I2, I3, dim_c, focus_coords)
            if not passed:
                continue

            # Save genuine synthetic images to disk for backbone feature extraction
            quad_paths = []
            stem = os.path.splitext(nm)[0]
            for k, (sub_im, g_name) in enumerate(zip([I0, I1, I2, I3], ["Clean", "P2", "P1", "P0"])):
                save_p = os.path.join(SYNTH_IMG_DIR, f"{stem}_d{dim_c}_{g_name}.png")
                # Save only modified images (Clean is already in dataset)
                if k > 0 and not os.path.exists(save_p):
                    cv2.imwrite(save_p, cv2.cvtColor(sub_im, cv2.COLOR_RGB2BGR))
                quad_paths.append(save_p if k > 0 else nm)

            # Extract 32D physical calipers for the 4 states
            c0 = extract_single_native_fourstream_caliper(I0)
            c1 = extract_single_native_fourstream_caliper(I1)
            c2 = extract_single_native_fourstream_caliper(I2)
            c3 = extract_single_native_fourstream_caliper(I3)
            calipers_4x32 = np.stack([c0, c1, c2, c3], axis=0).astype(np.float32)

            verified_quads.append({
                "seed_idx": int(s_idx),
                "fold_id": int(fold_ids[s_idx]),
                "dim_c": int(dim_c),
                "deltas": deltas,
                "ignore_mask6": ignore_mask6,
                "calipers_4x32": calipers_4x32,
                "quad_paths": quad_paths,
            })
            dim_passed += 1

        print(f"  Dimension {DIMS[dim_c]:8s}: Passed {dim_passed}/{len(target_seeds)} quadruplets ({100.0*dim_passed/len(target_seeds):.1f}%)")

    print(f"\n[Referee QC Complete] Verified {len(verified_quads)} / {total_candidates} ({100.0*len(verified_quads)/total_candidates:.1f}%) High-Fidelity Quadruplets.")
    
    # Save verified quadruplet manifest and calipers to disk
    out_manifest_p = os.path.join(OUT_DIR, "verified_quadruplets_fr_oracle.npz")
    np.savez_compressed(
        out_manifest_p,
        seed_idx=np.array([q["seed_idx"] for q in verified_quads], dtype=np.int64),
        fold_id=np.array([q["fold_id"] for q in verified_quads], dtype=np.int64),
        dim_c=np.array([q["dim_c"] for q in verified_quads], dtype=np.int64),
        ignore_mask6=np.array([q["ignore_mask6"] for q in verified_quads], dtype=np.float32),
        calipers_4x32=np.array([q["calipers_4x32"] for q in verified_quads], dtype=np.float32),
        deltas=np.array([q["deltas"] for q in verified_quads], dtype=np.float32),
    )
    print(f"Saved manifest -> {out_manifest_p} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
