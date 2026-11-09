#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/prism/real_defect_synthesizer.py
=====================================
PRISM Core Mathematical Module 3:
Real-Anchored Hard-Case Multi-Band Poisson Synthesizer & Differential Quadruplet Builder.

Why Previous Synthetic Data Failed to Yield Net Gains:
  1. Synthetic Gaussian blur / USM sharpening / naive FLUX blends had a domain distribution gap
     relative to authentic real video/ISP/AIGC artifacts.
  2. Feeding synthetic images directly into single-image BCE/CE classification heads caused the
     network to learn a "real vs synthetic" shortcut and shifted real-image classification biases.

How `PRISM` Fixes Synthetic Data From First Principles:
  1. **Authentic Real-Defect Multi-Band Blending (`synthesize_real_defect_poisson_quadruplet`)**:
     Extracts high-energy localized defect patches directly from authentic real training images
     annotated with `过度生成 (P0/P1)`, `过锐 (P0/P1)`, `模糊 (P0/P1)`, and `纹理损失 (P0/P1)`
     (strictly within the active training split `split == 'train'` — zero validation/test leakage!),
     and blends them onto scene-matched clean training images using 4-level Laplacian Pyramid
     Multi-Band Blending at calibrated severity alphas:
       - State 0 (Clean): alpha = 0.00 (Original clean real anchor)
       - State 1 (P2):    alpha = 0.38 (Mild localized authentic defect transfer)
       - State 2 (P1):    alpha = 0.68 (Moderate localized authentic defect transfer)
       - State 3 (P0):    alpha = 1.00 (Full localized authentic defect transfer)
  2. **Strict Differential-Only Routing (`mask7[6] = 0.0`)**:
     Every synthesized quadruplet is tagged with `is_real = 0.0` and its shared `seed_stem`.
     In `ConditionalHazardOrdinalHead` + `PrismDifferentialCalibratorLoss`, samples with `is_real = 0.0`
     NEVER update single-image classification biases (`hazard_out.bias` / `simplex_out.bias`),
     and ONLY train the relative severity direction `u_c` and backbone representations under a
     Domain-Adversarial Gradient Reversal Layer (GRL).
"""
import os
import csv
import json
import math
import glob
import random
import argparse
import numpy as np
import cv2
from PIL import Image

from .heads_and_losses import CLASSES, FULL_CLASSES

PRISM_SYNTH_DIR = '/dev/shm/prism_real_quads'
PRISM_META_OUT = 'data/v17_splits/prism_synth_metadata.json'


def multi_band_laplacian_blend(fg_rgb: np.ndarray, bg_rgb: np.ndarray, mask_1ch: np.ndarray, levels: int = 4) -> np.ndarray:
    """4-level Laplacian pyramid multi-band blending for zero-seam real defect transfer."""
    G_fg = [fg_rgb.astype(np.float32)]
    G_bg = [bg_rgb.astype(np.float32)]
    G_m = [np.clip(mask_1ch.astype(np.float32), 0.0, 1.0)[:, :, None]]
    for _ in range(levels):
        if min(G_fg[-1].shape[:2]) < 16:
            break
        G_fg.append(cv2.pyrDown(G_fg[-1]))
        G_bg.append(cv2.pyrDown(G_bg[-1]))
        m_down = cv2.pyrDown(G_m[-1])
        G_m.append(m_down[:, :, None] if m_down.ndim == 2 else m_down)

    L_out = [G_m[-1] * G_fg[-1] + (1.0 - G_m[-1]) * G_bg[-1]]
    for i in range(len(G_fg) - 2, -1, -1):
        size = (G_fg[i].shape[1], G_fg[i].shape[0])
        L_fg = G_fg[i] - cv2.pyrUp(G_fg[i + 1], dstsize=size)
        L_bg = G_bg[i] - cv2.pyrUp(G_bg[i + 1], dstsize=size)
        L_blend = G_m[i] * L_fg + (1.0 - G_m[i]) * L_bg
        L_out.append(cv2.pyrUp(L_out[-1], dstsize=size) + L_blend)
    return np.clip(np.round(L_out[-1]), 0, 255).astype(np.uint8)


def color_transfer_lab_local(src_rgb: np.ndarray, dst_rgb: np.ndarray) -> np.ndarray:
    """Matches low-frequency luminance & chrominance of `src_rgb` to `dst_rgb` before defect transfer
    so only the structural/high-frequency defect is transferred without artificial color seams."""
    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    for ch in range(3):
        mu_s, sd_s = float(src_lab[:, :, ch].mean()), float(src_lab[:, :, ch].std()) + 1e-4
        mu_d, sd_d = float(dst_lab[:, :, ch].mean()), float(dst_lab[:, :, ch].std()) + 1e-4
        ratio = np.clip(sd_d / sd_s, 0.65, 1.45)
        src_lab[:, :, ch] = np.clip((src_lab[:, :, ch] - mu_s) * ratio + mu_d, 0.0, 255.0)
    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def locate_peak_defect_region(im_rgb: np.ndarray, patch_size: int = 448) -> tuple:
    """Finds the highest gradient/Laplacian energy region in a defective donor image."""
    h, w = im_rgb.shape[:2]
    S = min(patch_size, h, w)
    gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    box = cv2.boxFilter(lap, -1, (S // 2, S // 2))
    cy, cx = np.unravel_index(np.argmax(box), box.shape)
    cy = int(np.clip(cy, S // 2, max(S // 2, h - S // 2)))
    cx = int(np.clip(cx, S // 2, max(S // 2, w - S // 2)))
    return cy, cx


def synthesize_real_defect_poisson_quadruplet(
    clean_rgb: np.ndarray,
    donor_rgb: np.ndarray,
    seed: int = 42,
):
    """
    Transfers an authentic defect region from `donor_rgb` (a real P0 training image) onto
    `clean_rgb` (a scene-matched clean real training image) at three continuous severity levels:
      - P2: alpha = 0.38, smaller soft mask (12%~16% area)
      - P1: alpha = 0.68, medium soft mask (22%~28% area)
      - P0: alpha = 1.00, large soft mask (38%~50% area)
    Returns (clean_rgb, img_p2, img_p1, img_p0, focus_yx).
    """
    rng = np.random.default_rng(seed)
    h, w = clean_rgb.shape[:2]
    donor_res = cv2.resize(donor_rgb, (w, h), interpolation=cv2.INTER_CUBIC) if donor_rgb.shape[:2] != (h, w) else donor_rgb
    donor_matched = color_transfer_lab_local(donor_res, clean_rgb)

    cy, cx = locate_peak_defect_region(donor_res, patch_size=min(448, h, w))
    outputs = []
    for alpha, area_frac in [(0.38, 0.14), (0.68, 0.25), (1.00, 0.44)]:
        ry = max(32, int(math.sqrt(area_frac * h * w / math.pi) * rng.uniform(0.90, 1.15)))
        rx = max(32, int(math.sqrt(area_frac * h * w / math.pi) * rng.uniform(0.90, 1.15)))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (rx, ry), float(rng.uniform(0, 180)), 0, 360, alpha, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(8.0, min(rx, ry) * 0.30))
        blended = multi_band_laplacian_blend(donor_matched, clean_rgb, mask, levels=4)
        outputs.append(blended)

    return clean_rgb, outputs[0], outputs[1], outputs[2], (int(cy), int(cx))


def build_split_isolated_real_defect_quadruplets(
    split_csv: str = 'data/resplits/split_s16.csv',
    manifest_csv: str = 'data/manifest_combined.csv',
    phys_cache: str = 'results/v8_experts/native_phys_feats.npz',
    max_quads_per_dim: int = 80,
    out_dir: str = PRISM_SYNTH_DIR,
    meta_out: str = PRISM_META_OUT,
):
    """
    Generates split-isolated Real-Defect Multi-Band Poisson Quadruplets strictly from `split == 'train'`
    images of `split_csv` (zero validation or test seed/donor leakage!).
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(meta_out), exist_ok=True)

    man = {r['name'].strip(): r for r in csv.DictReader(open(manifest_csv, encoding='utf-8'))}
    split_rows = list(csv.DictReader(open(split_csv, encoding='utf-8')))
    train_real_rows = [
        r for r in split_rows
        if r['split'].strip() == 'train'
        and r['name'].strip() in man
        and not r['name'].strip().startswith(('v9_', 'v8_', 'v7_', 'v6_', 'v4_'))
        and ('_d' not in r['name'].strip())
    ]

    for r in train_real_rows:
        nm = r['name'].strip()
        for c in CLASSES:
            r[c] = int(man[nm][c])

    phys_map = {}
    if os.path.exists(phys_cache):
        d = np.load(phys_cache, allow_pickle=True)
        if 'data' in d:
            phys_map = d['data'].item()

    clean_rows = [r for r in train_real_rows if all(r[c] == 0 for c in CLASSES)]
    if not clean_rows:
        return {}

    def _load_rgb(nm):
        for d_img in ['data/images', 'data/images_supp', 'data/train']:
            p = os.path.join(d_img, nm)
            if os.path.exists(p):
                return np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
        return None

    clean_vecs = np.stack([phys_map.get(r['name'], np.zeros(60, dtype=np.float32))[:16] for r in clean_rows], axis=0)
    c_mu = clean_vecs.mean(axis=0, keepdims=True)
    c_sd = clean_vecs.std(axis=0, keepdims=True) + 1e-5
    clean_norm = (clean_vecs - c_mu) / c_sd
    clean_norm /= (np.linalg.norm(clean_norm, axis=1, keepdims=True) + 1e-8)

    meta_records = {}
    if os.path.exists(meta_out):
        try:
            meta_records = json.load(open(meta_out, encoding='utf-8'))
        except Exception:
            meta_records = {}

    # Priority focus on OverGen (4), Oversharp (2), Blur (0), TexLoss (1)
    for dim_c in (4, 2, 0, 1):
        cname = CLASSES[dim_c]
        donors = [
            r for r in train_real_rows
            if r[cname] == 3 and sum(1 for c in CLASSES if r[c] > 0) == 1
        ]
        if len(donors) < 20:
            donors = [r for r in train_real_rows if r[cname] >= 2]
        donors = donors[:max_quads_per_dim]

        created = 0
        for d_idx, d_row in enumerate(donors):
            d_nm = d_row['name']
            d_vec = (phys_map.get(d_nm, np.zeros(60, dtype=np.float32))[:16] - c_mu[0]) / c_sd[0]
            d_vec /= (np.linalg.norm(d_vec) + 1e-8)
            sims = clean_norm @ d_vec
            c_idx = int(np.argmax(sims))
            c_nm = clean_rows[c_idx]['name']
            stem = os.path.splitext(c_nm)[0]

            clean_im = _load_rgb(c_nm)
            donor_im = _load_rgb(d_nm)
            if clean_im is None or donor_im is None:
                continue

            _, im_p2, im_p1, im_p0, (fy, fx) = synthesize_real_defect_poisson_quadruplet(
                clean_im, donor_im, seed=1000 * dim_c + d_idx
            )
            h, w = clean_im.shape[:2]
            for g_str, sub_im in [('P2', im_p2), ('P1', im_p1), ('P0', im_p0)]:
                fn = f'{stem}_prism_d{dim_c}_{g_str}.png'
                out_p = os.path.join(out_dir, fn)
                if not os.path.exists(out_p):
                    cv2.imwrite(out_p, cv2.cvtColor(sub_im, cv2.COLOR_RGB2BGR))
                meta_records[fn] = {
                    'seed_name': c_nm,
                    'donor_name': d_nm,
                    'dim_c': int(dim_c),
                    'grade': g_str,
                    'focus_yx': [int(fy), int(fx)],
                    'native_hw': [int(h), int(w)],
                    'is_real': 0,
                }
            created += 1
        print(f'[PRISM Real-Defect Synthesizer] Dim {cname:6s}: generated {created} authentic-donor Poisson quadruplets', flush=True)

    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(meta_records, f, indent=2, ensure_ascii=False)
    return meta_records


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='data/resplits/split_s16.csv')
    ap.add_argument('--manifest', default='data/manifest_combined.csv')
    ap.add_argument('--max-quads', type=int, default=80)
    args = ap.parse_args()
    build_split_isolated_real_defect_quadruplets(args.split, args.manifest, max_quads_per_dim=args.max_quads)
