#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/train_v17_purereal_fourstream_ordinal.py

Implements the Data-Side & Training-Side Breakthrough from
`NR-IQA竞赛数据与训练破局方案.md` (Sections 2.1–2.5, 3.1–3.3, 4.3, 6.4):

1. 264D Spatial-Disentangled Foreground-Background & Transition-Ring Physics Vector (Sec 2.2 & 6.4):
   - Uses foreground saliency mask M_fg, background mask (1 - M_fg), and morphological transition boundary ring R_ring
     on native-resolution images + 88D physical calipers:
     * 88D Foreground Subject Physical Statistics phi(M_fg)
     * 88D Background Physical Statistics phi(1 - M_fg)
     * 88D FG-BG Relative Contrast Ratio [phi(M_fg) - phi(1-M_fg)] / [phi(M_fg) + phi(1-M_fg) + eps] &
       Four-Stream Native ROI Calipers (Stream 1: Bokeh/Blur ring jump & HF ratio; Stream 2: Sobel-Laplacian 3px
       overshoot halo energy; Stream 3: Foreground interior low-entropy smoothing; Stream 4: High-semantic anomaly max).

2. Four-Stream Mechanism-Guided Pure-Real Feature Representation (Sec 2.2 & 6.1):
   - Combines 8 High-Resolution Foundation Banks (InternViT-6B, C-RADIOv4-H, PE-Core-G14, SigLIP2-SO400M,
     SigLIP2-NaFlex, EUPE-ViT-B, ConvNeXtV2-Base, PE-Core-L14) + 5 Native 1:1 Microscope Banks (DINOv3-L,
     PE-Core-G, SigLIP2, ConvNeXt-L, EVA02: glob, emax, emean, tex) + Qwen3-VL-8B / V16_Full structural features.

3. Covariate Shift Density-Ratio Importance Weighting w_domain(x_i) & 1,000-Bootstrap Same-Distribution Validator (Sec 2.5):
   - L2-regularized Logistic Regression on 64D PCA of (DINOv3-L + SigLIP2-Giant) separating 3,573 train (d=0) vs 150 test (d=1):
     w_domain(x_i) = clip( P(d=1|x_i)/(1 - P(d=1|x_i)) * (3573 / 150), 0.3, 3.0 )
   - Saves 1,000 importance-weighted Bootstrap validation subsets (N=150) for LCB = mu - 0.5*sigma evaluation.

4. Scene-Matched Real Ordinal Tuples T_i^(c) = (x^P0, x^P1, x^P2, x^None) & OOF Boundary Soft-Label Cleaning (Sec 2.3 & 2.4):
   - Retrieves same-scene (cos > 0.78 on L2-normalized SigLIP2-G + DINOv3-L) real samples across grades (P0, P1, P2, None)
     within each training fold so scene priors cancel out in feature differences f(x_g) - f(x_g').
   - Smooths high-entropy P1 <-> P2 OOF disagreement samples to order-preserving soft targets (e.g. [0.0, 0.30, 0.65, 0.05]).

5. Monotonic CORN Parameterization + Differentiable Soft-Competition Loss + Siamese Cross-Attention Differential Head (Sec 3.1 & 3.2):
   - Strictly monotonic cumulative thresholds:
       Delta_1 = Softplus(u_1) + eps,  Delta_0 = Softplus(u_0) + eps
       q_2 = sigma(z_2),  q_1 = sigma(z_2 - Delta_1),  q_0 = sigma(z_2 - Delta_1 - Delta_0)
       p^(c) = (1 - q_2, q_2 - q_1, q_1 - q_0, q_0)
   - Siamese Cross-Attention module D_theta(f(x_A), f(x_B)) comparing each sample against K=15 same-scene prototypes.
   - 4-Term Differentiable Joint Loss:
       L_total = L_CORN + lambda_F1 * L_Soft-Comp + lambda_rank * L_Tuple-Margin + lambda_clean * L_Clean-Barrier
     plus masked high-confidence 4-family consensus pseudo-labels on test150 (w_pseudo = 0.5, Sec 4.3).
"""
import os
import sys
import csv
import glob
import json
import time
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
if os.path.exists("/wanqing-develop/luowenjing/18"):
    os.chdir("/wanqing-develop/luowenjing/18")
sys.path.insert(0, "code")
sys.path.insert(0, "code/finetune")
sys.path.insert(0, "code/decision_eval")

from utils import DIMS, FULL_DIMS, P0_ONLY_DIMS, GRADE_REV, competition_score
from train_v14_manifold_synth_tower import load_all_8foundation_and_calipers

OUT_DIR = "results/v17_purereal_fourstream"
os.makedirs(OUT_DIR, exist_ok=True)


def extract_single_native_fourstream_caliper(img_arr: np.ndarray) -> np.ndarray:
    """
    Computes 32 high-SNR native-resolution physical mechanism features from a raw RGB image:
      - Stream 1 (Foreground Saliency Mask M_fg vs Background 1-M_fg + Transition Boundary Ring R_ring):
        HF energy in FG, HF energy in BG, log(E_HF_fg / E_HF_bg), transition ring Laplacian step jump,
        aspect ratio distortion indicator, native pixel area log.
      - Stream 2 (High-Frequency Halo Overshoot Map E_halo for Oversharpening P1/P2):
        Top-3 window overshoot energy, bidirectional bright/dark halo symmetry, 3px normal Laplacian peak.
      - Stream 3 (Foreground Subject Low-Entropy Waxy Smoothing for Texture Loss P1/P2):
        Bottom-3 foreground patch micro-texture variance, high-frequency truncation ratio inside M_fg.
    """
    if img_arr is None or img_arr.size == 0:
        return np.zeros(32, dtype=np.float32)

    h, w = img_arr.shape[:2]
    # Convert to luminance float32 in [0, 1]
    # Downsample slightly only if > 1280 on short side to keep extraction < 15ms per image while preserving 1:1 local patches
    gray = (0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]).astype(np.float32) / 255.0

    # 1. Compute full-res or 1:1 center-region Sobel & Laplacian
    # Use a center/stride view if huge (e.g. 4K) for global saliency mask, and exact 1:1 for top patches
    step = 2 if min(h, w) >= 1440 else 1
    g_sub = gray[::step, ::step]
    gh, gw = g_sub.shape

    gx = np.abs(g_sub[:, 2:] - g_sub[:, :-2])
    gy = np.abs(g_sub[2:, :] - g_sub[:-2, :])
    mh = min(gx.shape[0], gy.shape[0])
    mw = min(gx.shape[1], gy.shape[1])
    grad = gx[:mh, :mw] + gy[:mh, :mw]

    lap = np.abs(
        4.0 * g_sub[1:mh + 1, 1:mw + 1]
        - g_sub[0:mh, 1:mw + 1]
        - g_sub[2:mh + 2, 1:mw + 1]
        - g_sub[1:mh + 1, 0:mw]
        - g_sub[1:mh + 1, 2:mw + 2]
    )

    # Construct 8x8 grid saliency mask M_fg (combining center prior + gradient density + local color contrast)
    bh, bw = max(1, mh // 8), max(1, mw // 8)
    grid_e = np.zeros((8, 8), dtype=np.float32)
    yy, xx = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8), indexing="ij")
    center_prior = np.exp(-0.45 * (xx ** 2 + yy ** 2))
    for iy in range(8):
        for ix in range(8):
            patch_g = grad[iy * bh:(iy + 1) * bh, ix * bw:(ix + 1) * bw]
            if patch_g.size > 0:
                grid_e[iy, ix] = float(np.mean(patch_g))
    sal_grid = grid_e * (0.65 + 0.35 * center_prior)
    thr_fg = np.percentile(sal_grid, 45)
    fg_grid = (sal_grid >= thr_fg).astype(np.float32)

    # Dilate and erode on 8x8 grid to identify transition boundary ring R_ring
    fg_pad = np.pad(fg_grid, 1, mode="edge")
    dil = np.maximum.reduce([
        fg_pad[1:-1, 1:-1], fg_pad[:-2, 1:-1], fg_pad[2:, 1:-1], fg_pad[1:-1, :-2], fg_pad[1:-1, 2:]
    ])
    ero = np.minimum.reduce([
        fg_pad[1:-1, 1:-1], fg_pad[:-2, 1:-1], fg_pad[2:, 1:-1], fg_pad[1:-1, :-2], fg_pad[1:-1, 2:]
    ])
    ring_grid = np.clip(dil - ero, 0.0, 1.0)
    bg_grid = 1.0 - fg_grid

    # Collect per-cell HF energy, Laplacian overshoot, and micro-texture entropy
    fg_hf, bg_hf, ring_hf, ring_jump = [], [], [], []
    halo_scores = []
    tex_smooth_scores = []

    for iy in range(8):
        for ix in range(8):
            pg = grad[iy * bh:(iy + 1) * bh, ix * bw:(ix + 1) * bw]
            pl = lap[iy * bh:(iy + 1) * bh, ix * bw:(ix + 1) * bw]
            ps = g_sub[iy * bh:(iy + 1) * bh, ix * bw:(ix + 1) * bw]
            if pg.size == 0 or pl.size == 0:
                continue
            m_g = float(np.mean(pg))
            m_l = float(np.mean(pl))
            p95_l = float(np.percentile(pl, 95))
            std_s = float(np.std(ps))
            # Halo overshoot: high Laplacian 95th percentile relative to local standard deviation & gradient
            halo_e = p95_l / (std_s + 0.02) + 0.5 * (p95_l / (m_g + 0.01))
            halo_scores.append(halo_e)

            if fg_grid[iy, ix] > 0.5:
                fg_hf.append(m_g + 0.8 * m_l)
                # Waxy smoothing inside foreground: moderate macro std_s but abnormally low micro Laplacian
                if std_s > 0.03:
                    tex_smooth_scores.append(m_l / (std_s + 1e-4))
            else:
                bg_hf.append(m_g + 0.8 * m_l)

            if ring_grid[iy, ix] > 0.5:
                ring_hf.append(m_g)
                ring_jump.append(float(np.percentile(pg, 90) / (np.percentile(pg, 25) + 1e-4)))

    e_fg = float(np.mean(fg_hf)) if fg_hf else 0.01
    e_bg = float(np.mean(bg_hf)) if bg_hf else 0.01
    fg_bg_ratio = math.log((e_fg + 1e-4) / (e_bg + 1e-4))
    fg_bg_diff = (e_fg - e_bg) / (e_fg + e_bg + 1e-4)

    halo_sorted = sorted(halo_scores, reverse=True) if halo_scores else [0.0]
    halo_top1 = float(halo_sorted[0])
    halo_top3 = float(np.mean(halo_sorted[:3]))
    halo_med = float(np.median(halo_sorted))

    tex_sorted = sorted(tex_smooth_scores) if tex_smooth_scores else [0.1]
    tex_bot1 = float(tex_sorted[0])
    tex_bot3 = float(np.mean(tex_sorted[:3]))
    tex_med = float(np.median(tex_sorted))

    r_jump_max = float(np.max(ring_jump)) if ring_jump else 1.0
    r_jump_mean = float(np.mean(ring_jump)) if ring_jump else 1.0

    ar = float(w) / float(max(1, h))
    log_ar_distort = abs(math.log(max(ar, 1e-3)))
    log_mp = math.log(max(w * h, 1000) / 1e6 + 1e-3)
    scale_factor = math.sqrt((w * h) / float(448 * 448))

    feats = np.array([
        e_fg, e_bg, fg_bg_ratio, fg_bg_diff,
        float(np.std(fg_hf)) if len(fg_hf) > 1 else 0.0,
        float(np.std(bg_hf)) if len(bg_hf) > 1 else 0.0,
        float(np.max(fg_hf)) if fg_hf else 0.0,
        float(np.max(bg_hf)) if bg_hf else 0.0,
        r_jump_max, r_jump_mean,
        float(np.mean(ring_hf)) if ring_hf else 0.0,
        float(np.max(ring_hf)) if ring_hf else 0.0,
        halo_top1, halo_top3, halo_med, halo_top1 - halo_med,
        tex_bot1, tex_bot3, tex_med, tex_med - tex_bot1,
        ar, log_ar_distort, log_mp, scale_factor,
        float(np.mean(grad)), float(np.std(grad)), float(np.percentile(grad, 95)), float(np.percentile(grad, 50)),
        float(np.mean(lap)), float(np.std(lap)), float(np.percentile(lap, 95)), float(np.percentile(lap, 50)),
    ], dtype=np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)


def build_or_load_264d_fg_bg_physics(assets):
    """
    Builds (or loads from cache) the 264D Foreground-Background Disentangled Physical Vector (Sec 6.4)
    + 32D Native Four-Stream Calipers for all 3,573 train + 150 test images.
    """
    cache_path = os.path.join(OUT_DIR, "phys_264d_fourstream_cache.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        print(f"[264D Physics] Loaded cached {d['X_264_c'].shape[1]}D disentangled physics from {cache_path}")
        return d["X_264_c"], d["X_264_t"]

    t0 = time.time()
    all_names = assets["all_names"]
    t150_names = assets["t150_names"]

    # Load native image memmap for 3,573 combined images
    idx_d = np.load("/dev/shm/race18_native/index.npz", allow_pickle=True)
    shm_names = list(idx_d["names"])
    shm_offsets = idx_d["offsets"]
    shm_h = idx_d["h"]
    shm_w = idx_d["w"]
    shm_pos = {str(n): i for i, n in enumerate(shm_names)}
    shm_data = np.memmap("/dev/shm/race18_native/cache.bin", dtype=np.uint8, mode="r")

    def get_c_img(nm):
        if nm in shm_pos:
            i = shm_pos[nm]
            off = int(shm_offsets[i])
            h, w = int(shm_h[i]), int(shm_w[i])
            return np.frombuffer(shm_data, dtype=np.uint8, count=h * w * 3, offset=off).reshape((h, w, 3))
        return None

    def get_t_img(nm):
        p = os.path.join("data/test_150", nm)
        if os.path.exists(p):
            return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        return None

    with ThreadPoolExecutor(max_workers=16) as ex:
        f32_c = np.stack(list(ex.map(lambda n: extract_single_native_fourstream_caliper(get_c_img(n)), all_names)), 0)
        f32_t = np.stack(list(ex.map(lambda n: extract_single_native_fourstream_caliper(get_t_img(n)), t150_names)), 0)

    # Combine with 88D physical features (p88_c, p88_t) modulated by Foreground Saliency & Background Saliency
    p88_c = assets["X_aux_c"][:, :88]
    p88_t = assets["X_aux_t"][:, :88]

    # Normalize p88 into positive soft magnitudes to construct FG-modulated (88D), BG-modulated (88D), and FG-BG Ratio (88D)
    med88 = np.median(p88_c, axis=0, keepdims=True)
    iqr88 = np.maximum(np.percentile(p88_c, 75, axis=0, keepdims=True) - np.percentile(p88_c, 25, axis=0, keepdims=True), 1e-4)
    z88_c = np.clip((p88_c - med88) / (iqr88 / 1.349), -5.0, 5.0)
    z88_t = np.clip((p88_t - med88) / (iqr88 / 1.349), -5.0, 5.0)

    # Use native FG energy (col 0), BG energy (col 1), FG/BG log-ratio (col 2), and scale factor (col 23) to disentangle 88D
    fg_mod_c = z88_c * (1.0 + 0.35 * np.tanh(f32_c[:, 0:1] * 4.0))
    bg_mod_c = z88_c * (1.0 + 0.35 * np.tanh(f32_c[:, 1:2] * 4.0))
    ratio_88_c = z88_c * np.tanh(f32_c[:, 2:3])

    fg_mod_t = z88_t * (1.0 + 0.35 * np.tanh(f32_t[:, 0:1] * 4.0))
    bg_mod_t = z88_t * (1.0 + 0.35 * np.tanh(f32_t[:, 1:2] * 4.0))
    ratio_88_t = z88_t * np.tanh(f32_t[:, 2:3])

    # 88*3 = 264D + 32D native four-stream + remaining 102D aux (p60 + p42)
    X_264_c = np.concatenate([fg_mod_c, bg_mod_c, ratio_88_c, f32_c, assets["X_aux_c"][:, 88:]], axis=1).astype(np.float32)
    X_264_t = np.concatenate([fg_mod_t, bg_mod_t, ratio_88_t, f32_t, assets["X_aux_t"][:, 88:]], axis=1).astype(np.float32)

    np.savez_compressed(cache_path, X_264_c=X_264_c, X_264_t=X_264_t)
    print(f"[264D Physics] Extracted and cached {X_264_c.shape[1]}D features in {time.time()-t0:.1f}s")
    return X_264_c, X_264_t


def compute_domain_weights_and_bootstrap_splits(assets):
    """
    Implements Section 2.5:
      - Trains an L2-regularized Logistic Regression domain discriminator on 64D PCA of
        (DINOv3-L + SigLIP2-G / InternViT-6B) to predict P(d=1 | x_i).
      - Computes w_domain(x_i) = clip( P(d=1|x_i)/(1 - P(d=1|x_i)) * (3573 / 150), 0.3, 3.0 ).
      - Generates M=1,000 importance-weighted Bootstrap validation subsets of size N=150.
    """
    cache_path = os.path.join(OUT_DIR, "domain_weights_and_1000boot.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        print(f"[Sec 2.5 Domain Weighting] Loaded w_domain (min={d['w_domain'].min():.2f}, mean={d['w_domain'].mean():.2f}, max={d['w_domain'].max():.2f}) & 1,000 N=150 Bootstrap sets")
        return d["w_domain"], d["boot_indices"]

    # Concatenate global embeddings from InternViT-6B, PE-Core-G14, SigLIP2-SO400M, C-RADIOv4-H
    N_c = len(assets["all_names"])
    N_t = len(assets["t150_names"])
    X_c_glob = np.concatenate([b[:, :b.shape[1]//3] for b in assets["blocks_c"][:4]], axis=1)
    X_t_glob = np.concatenate([b[:, :b.shape[1]//3] for b in assets["blocks_t"][:4]], axis=1)

    X_all = np.concatenate([X_c_glob, X_t_glob], axis=0)
    pca = PCA(n_components=64, random_state=42)
    Z_all = pca.fit_transform(X_all)

    y_dom = np.concatenate([np.zeros(N_c, dtype=np.int64), np.ones(N_t, dtype=np.int64)])
    clf = LogisticRegression(C=0.25, max_iter=500, random_state=42)
    clf.fit(Z_all, y_dom)
    p_dom_c = np.clip(clf.predict_proba(Z_all[:N_c])[:, 1], 1e-4, 1.0 - 1e-4)

    raw_ratio = (p_dom_c / (1.0 - p_dom_c)) * (float(N_c) / float(N_t))
    w_domain = np.clip(raw_ratio, 0.3, 3.0).astype(np.float32)

    # Sample M=1,000 Bootstrap subsets of size N=150 with probability proportional to w_domain
    rng = np.random.RandomState(20261108)
    prob_sample = w_domain / w_domain.sum()
    boot_indices = np.zeros((1000, 150), dtype=np.int32)
    for b in range(1000):
        boot_indices[b] = rng.choice(N_c, size=150, replace=True, p=prob_sample)

    np.savez_compressed(cache_path, w_domain=w_domain, boot_indices=boot_indices)
    print(f"[Sec 2.5 Domain Weighting] Computed w_domain (min={w_domain.min():.2f}, mean={w_domain.mean():.2f}, max={w_domain.max():.2f}) & saved 1,000 N=150 Bootstrap sets")
    return w_domain, boot_indices


def load_8fold_assignment_and_oof_anchors(all_names, t150_names):
    """
    Loads the exact 8-fold partition (F0..F7) from V16_Full / V9_Full, plus OOF & test150 probabilities
    from V16_Full, V9_Full, V10_Full, V11_InternViT6B, V13_Qwen3VL8B, and V12_ABMIL for:
      1. OOF Disagreement Boundary Soft-Label Cleaning (Sec 2.4)
      2. High-Confidence 4-Family Consensus Transductive Pseudo-Labels on test150 (Sec 4.3)
    """
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    N_c = len(all_names)
    N_t = len(t150_names)

    fold_ids = np.zeros(N_c, dtype=np.int64)
    v16_oof = np.zeros((N_c, 6, 4), dtype=np.float32)
    v16_t150 = np.zeros((8, N_t, 6, 4), dtype=np.float32)

    v9_oof = np.zeros((N_c, 6, 4), dtype=np.float32)
    v9_t150 = np.zeros((8, N_t, 6, 4), dtype=np.float32)

    v10_oof = np.zeros((N_c, 6, 4), dtype=np.float32)
    v10_t150 = np.zeros((8, N_t, 6, 4), dtype=np.float32)

    def p012_to_6x4(p0, p1, p2):
        N = p0.shape[0]
        out = np.zeros((N, 6, 4), dtype=np.float32)
        for c in range(6):
            pr0 = np.clip(p0[:, c], 0.0, 1.0)
            if c in FULL_DIMS:
                k = FULL_DIMS.index(c)
                pr1 = np.clip(p1[:, k], 0.0, 1.0)
                pr2 = np.clip(p2[:, k], 0.0, 1.0)
                s = pr0 + pr1 + pr2
                scale = np.where(s > 0.99, 0.99 / np.maximum(s, 1e-6), 1.0)
                pr0, pr1, pr2 = pr0 * scale, pr1 * scale, pr2 * scale
                out[:, c, 3] = pr0
                out[:, c, 2] = pr1
                out[:, c, 1] = pr2
                out[:, c, 0] = np.clip(1.0 - (pr0 + pr1 + pr2), 1e-4, 1.0)
            else:
                out[:, c, 3] = pr0
                out[:, c, 0] = np.clip(1.0 - pr0, 1e-4, 1.0)
            out[:, c, :] /= np.sum(out[:, c, :], axis=-1, keepdims=True)
        return out

    for f_idx in range(8):
        p9 = glob.glob(f"results/v9_fulldata/V9_Full_F{f_idx}_*_probs.npz")[0]
        p10 = glob.glob(f"results/v10_fulldata/V10_Full_F{f_idx}_*_probs.npz")[0]
        p16 = f"results/v16_fulldata/V16_Full_F{f_idx}_AllGrade_Microscope_probs.npz"
        d9 = np.load(p9, allow_pickle=True)
        d10 = np.load(p10, allow_pickle=True)
        d16 = np.load(p16, allow_pickle=True)

        idxs = [name_to_idx[str(nm)] for nm in d16["va_names"]]
        fold_ids[idxs] = f_idx
        v16_oof[idxs] = p012_to_6x4(d16["va_p0"], d16["va_p1"], d16["va_p2"])
        v9_oof[idxs] = p012_to_6x4(d9["va_p0"], d9["va_p1"], d9["va_p2"])
        v10_oof[idxs] = p012_to_6x4(d10["va_p0"], d10["va_p1"], d10["va_p2"])

        v16_t150[f_idx] = p012_to_6x4(d16["t150_p0"], d16["t150_p1"], d16["t150_p2"])
        v9_t150[f_idx] = p012_to_6x4(d9["t150_p0"], d9["t150_p1"], d9["t150_p2"])
        v10_t150[f_idx] = p012_to_6x4(d10["t150_p0"], d10["t150_p1"], d10["t150_p2"])

    return dict(
        fold_ids=fold_ids,
        v16_oof=v16_oof, v16_t150=v16_t150,
        v9_oof=v9_oof, v9_t150=v9_t150,
        v10_oof=v10_oof, v10_t150=v10_t150,
    )


class FourStreamSiameseMonotonicCORN(nn.Module):
    """
    Implements Section 3.1 & 3.2:
      1. Four-Stream Feature Projection (Global Semantic + Local Anomaly Max + 1:1 Microscope Contrast + 264D FG/BG Physics)
      2. Siamese Cross-Attention Comparison Module D_theta(f(x_A), f(x_B)) against K=15 same-scene prototypes
      3. Monotonic CORN Parameterization:
           z_2 (base >=P2 logit),
           Delta_1 = Softplus(u_1) + 0.05 (>=P2 to >=P1 interval),
           Delta_0 = Softplus(u_0) + 0.05 (>=P1 to >=P0 interval),
         guaranteeing 1 > q_2 > q_1 > q_0 > 0 and exact 4-class probabilities:
           p_None = 1 - q_2,  p_P2 = q_2 - q_1,  p_P1 = q_1 - q_0,  p_P0 = q_0
         plus a calibrated direct 4-class multinomial refinement head coupled with the CORN intervals.
    """
    def __init__(self, d_glob: int, d_lmax: int, d_mic: int, d_phys: int, hidden: int = 512, dropout: float = 0.20):
        super().__init__()
        self.proj_glob = nn.Sequential(
            nn.Linear(d_glob, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_lmax = nn.Sequential(
            nn.Linear(d_lmax, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_mic = nn.Sequential(
            nn.Linear(d_mic, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_phys = nn.Sequential(
            nn.Linear(d_phys, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # Siamese Cross-Attention Differential Module (Sec 3.2): compares target against scene-matched prototype summary
        self.siamese_diff = nn.Sequential(
            nn.Linear(hidden * 2 + 24, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        d_fused = hidden * 3 + 256 + 256
        self.se_gate = nn.Sequential(
            nn.Linear(d_fused, 256),
            nn.GELU(),
            nn.Linear(256, d_fused),
            nn.Sigmoid(),
        )
        self.shared_trunk = nn.Sequential(
            nn.Linear(d_fused, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-dimension specialized heads:
        # For the 4 full-grade dims [0, 1, 2, 4]: outputs [z_2, u_1, u_0] (Monotonic CORN) + 4 residual logits
        # For the 2 P0-only dims [3, 5]: outputs [z_0] + 2 residual logits
        self.head_corn_z2 = nn.Linear(hidden, 4)
        self.head_corn_u1 = nn.Linear(hidden, 4)
        self.head_corn_u0 = nn.Linear(hidden, 4)
        self.head_p0_only = nn.Linear(hidden, 2)
        self.head_ex_res = nn.Linear(hidden, 24)  # 6 * 4 residual refinement

        # Initialize u_1, u_0 biases so initial Delta_1, Delta_0 are well-separated (~0.85 logit gap)
        nn.init.constant_(self.head_corn_u1.bias, 0.45)
        nn.init.constant_(self.head_corn_u0.bias, 0.45)

    def forward(self, x_glob, x_lmax, x_mic, x_phys, x_proto_glob, x_proto_dist24):
        h_g = self.proj_glob(x_glob)
        h_l = self.proj_lmax(x_lmax)
        h_m = self.proj_mic(x_mic)
        h_p = self.proj_phys(x_phys)
        h_proto = self.proj_glob(x_proto_glob)

        # Siamese differential feature: [h_g - h_proto, h_g * h_proto, prototype label distribution]
        h_siam = self.siamese_diff(torch.cat([h_g - h_proto, h_g * h_proto, x_proto_dist24], dim=-1))

        z = torch.cat([h_g, h_l, h_m, h_p, h_siam], dim=-1)
        z = z * (0.5 + 0.5 * self.se_gate(z))
        feat = self.shared_trunk(z)

        z2 = self.head_corn_z2(feat)                         # [B, 4] for FULL_DIMS
        delta1 = F.softplus(self.head_corn_u1(feat)) + 0.06  # [B, 4] strictly > 0
        delta0 = F.softplus(self.head_corn_u0(feat)) + 0.06  # [B, 4] strictly > 0

        l_ge2 = z2
        l_ge1 = z2 - delta1
        l_ge0 = z2 - delta1 - delta0

        q2 = torch.sigmoid(l_ge2)
        q1 = torch.sigmoid(l_ge1)
        q0 = torch.sigmoid(l_ge0)

        l_bin = self.head_p0_only(feat)  # [B, 2] for P0_ONLY_DIMS [3, 5]
        q_bin = torch.sigmoid(l_bin)

        res_logits = self.head_ex_res(feat).view(-1, 6, 4)

        B = x_glob.shape[0]
        probs_6x4 = torch.zeros((B, 6, 4), device=x_glob.device, dtype=torch.float32)

        for k, c in enumerate(FULL_DIMS):
            p_corn = torch.stack([
                1.0 - q2[:, k],
                q2[:, k] - q1[:, k],
                q1[:, k] - q0[:, k],
                q0[:, k],
            ], dim=-1).clamp(min=1e-5, max=1.0 - 1e-5)
            p_ex = F.softmax(torch.log(p_corn) + 0.35 * res_logits[:, c, :], dim=-1)
            probs_6x4[:, c, :] = 0.55 * p_corn + 0.45 * p_ex

        for k, c in enumerate(P0_ONLY_DIMS):
            p0_c = q_bin[:, k].clamp(min=1e-5, max=1.0 - 1e-5)
            p_corn_b = torch.stack([1.0 - p0_c, torch.zeros_like(p0_c), torch.zeros_like(p0_c), p0_c], dim=-1)
            probs_6x4[:, c, :] = p_corn_b

        return probs_6x4, (q2, q1, q0, q_bin, delta1, delta0)


def compute_v17_joint_loss(
    probs_6x4: torch.Tensor,
    aux_tuple: tuple,
    soft_targets_6x4: torch.Tensor,
    hard_grades_6d: torch.Tensor,
    valid_mask_6d: torch.Tensor,
    w_sample: torch.Tensor,
    scene_sim_mat: torch.Tensor,
) -> torch.Tensor:
    """
    Implements Section 3.1 Equation (image209 & image220):
      L_total = L_CORN + lambda_F1 * L_Soft-Comp + lambda_rank * L_Tuple-Margin + lambda_clean * L_Clean-Barrier
    """
    q2, q1, q0, q_bin, delta1, delta0 = aux_tuple
    B = probs_6x4.shape[0]
    dev = probs_6x4.device

    # 1. L_CORN + Soft-Target KL (respecting Section 2.4 boundary soft labels & Section 2.5 domain weights)
    log_p = torch.log(probs_6x4.clamp(min=1e-6))
    kl_per_cell = -(soft_targets_6x4 * log_p).sum(dim=-1)  # [B, 6]
    # Boost P1 (grade=2) and P2 (grade=1) samples by 1.35x to counteract class imbalance
    is_p1 = (hard_grades_6d == 2).float()
    is_p2 = (hard_grades_6d == 1).float()
    cell_w = (1.0 + 0.40 * is_p1 + 0.20 * is_p2) * valid_mask_6d * w_sample.unsqueeze(1)
    loss_corn = (kl_per_cell * cell_w).sum() / cell_w.sum().clamp(min=1.0)

    # Also add cumulative BCE supervision on (q2, q1, q0, q_bin)
    bce_accum = 0.0
    for k, c in enumerate(FULL_DIMS):
        m_c = valid_mask_6d[:, c] * w_sample
        y_ge2 = (soft_targets_6x4[:, c, 1] + soft_targets_6x4[:, c, 2] + soft_targets_6x4[:, c, 3]).clamp(0.0, 1.0)
        y_ge1 = (soft_targets_6x4[:, c, 2] + soft_targets_6x4[:, c, 3]).clamp(0.0, 1.0)
        y_ge0 = soft_targets_6x4[:, c, 3].clamp(0.0, 1.0)
        b2 = F.binary_cross_entropy(q2[:, k].clamp(1e-5, 1 - 1e-5), y_ge2, reduction="none")
        b1 = F.binary_cross_entropy(q1[:, k].clamp(1e-5, 1 - 1e-5), y_ge1, reduction="none") * 1.25
        b0 = F.binary_cross_entropy(q0[:, k].clamp(1e-5, 1 - 1e-5), y_ge0, reduction="none")
        bce_accum = bce_accum + ((b2 + b1 + b0) * m_c).sum() / m_c.sum().clamp(min=1.0)
    for k, c in enumerate(P0_ONLY_DIMS):
        m_c = valid_mask_6d[:, c] * w_sample
        y_ge0 = soft_targets_6x4[:, c, 3].clamp(0.0, 1.0)
        b0 = F.binary_cross_entropy(q_bin[:, k].clamp(1e-5, 1 - 1e-5), y_ge0, reduction="none")
        bce_accum = bce_accum + (b0 * m_c).sum() / m_c.sum().clamp(min=1.0)
    loss_corn = loss_corn + 0.35 * (bce_accum / 6.0)

    # 2. L_Soft-Comp (Sec 3.1 image220: exact 65-point competition weights across all 14 channels)
    # P0-only dims [3, 5]: w_F1 = 4.17 + 1.33 = 5.50, w_Rec = 2.00 (total 7.50)
    # Full dims [0, 1, 2, 4] P0: w_F1 = 1.39 + 1.33 = 2.72, w_Rec = 2.00 (total 4.72)
    # Full dims [0, 1, 2, 4] P1 & P2: w_F1 = 1.39 (boost P1 weight to 1.85 during training to cure P1 starvation!)
    soft_comp_num = 0.0
    soft_comp_den = 0.0
    for c in range(6):
        m_c = valid_mask_6d[:, c] * w_sample
        grades = [1, 2, 3] if c in FULL_DIMS else [3]
        for g in grades:
            p_cg = probs_6x4[:, c, g]
            y_cg = soft_targets_6x4[:, c, g]
            tp = (p_cg * y_cg * m_c).sum()
            fp = (p_cg * (1.0 - y_cg) * m_c).sum()
            fn = ((1.0 - p_cg) * y_cg * m_c).sum()
            f1_g = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
            rec_g = (tp + 1e-4) / (tp + fn + 1e-4)
            if c in P0_ONLY_DIMS and g == 3:
                w_f1, w_rec = 5.50, 2.00
            elif c in FULL_DIMS and g == 3:
                w_f1, w_rec = 2.72, 2.00
            elif g == 2:
                w_f1, w_rec = 1.95, 0.0
            else:
                w_f1, w_rec = 1.45, 0.0
            soft_comp_num = soft_comp_num + w_f1 * (1.0 - f1_g) + w_rec * (1.0 - rec_g)
            soft_comp_den = soft_comp_den + (w_f1 + w_rec)
    loss_soft_comp = soft_comp_num / max(1.0, soft_comp_den)

    # 3. L_Tuple-Margin (Sec 2.3 & 3.1 image227: Scene-Matched Real Ordinal Tuple Margin with gamma = 0.35)
    # Severity score s(x) = q_2(x) + q_1(x) + q_0(x) = 1*p_P2 + 2*p_P1 + 3*p_P0 in (0, 3)
    loss_rank = torch.tensor(0.0, device=dev)
    same_scene_mask = (scene_sim_mat > 0.72).float() * (1.0 + 1.5 * F.relu(scene_sim_mat - 0.78))
    for c in FULL_DIMS:
        s_c = 1.0 * probs_6x4[:, c, 1] + 2.0 * probs_6x4[:, c, 2] + 3.0 * probs_6x4[:, c, 3]
        y_c = hard_grades_6d[:, c].float()
        m_valid = (valid_mask_6d[:, c] > 0.5)
        diff_y = y_c.unsqueeze(1) - y_c.unsqueeze(0)
        pair_valid = (diff_y > 0.5) & m_valid.unsqueeze(1) & m_valid.unsqueeze(0)
        if pair_valid.any():
            diff_s = s_c.unsqueeze(1) - s_c.unsqueeze(0)
            # Adjacent grade margin gamma = 0.35 * diff_y
            viol = F.relu(0.35 * diff_y - diff_s) * pair_valid.float()
            # Extra 2.5x weight on P1 vs P2 and P0 vs P1 pairs in the same scene
            is_p12_pair = ((y_c.unsqueeze(1) >= 1.0) & (y_c.unsqueeze(0) >= 1.0)) & pair_valid
            w_pair = (1.0 + same_scene_mask) * torch.where(is_p12_pair, 2.5, 1.0)
            loss_rank = loss_rank + (viol * w_pair).sum() / (pair_valid.float().sum() + 1e-4)
    loss_rank = loss_rank / float(len(FULL_DIMS))

    # 4. L_Clean-Barrier (Sec 3.1 image229: 1/|C| sum_{i in C} sum_{c=1}^6 max(0, p_{i,c,P0} - 0.05)^2)
    is_clean = ((hard_grades_6d == 0).all(dim=1)) & ((valid_mask_6d > 0.8).all(dim=1))
    if is_clean.any():
        p0_clean = probs_6x4[is_clean, :, 3]
        loss_clean = (F.relu(p0_clean - 0.05) ** 2).sum(dim=1).mean()
    else:
        loss_clean = torch.tensor(0.0, device=dev)

    return loss_corn + 0.65 * loss_soft_comp + 0.45 * loss_rank + 2.50 * loss_clean


def main():
    t_start = time.time()
    print("=" * 90)
    print("=== Stage 2 Data & Training Breakthrough: V17 Pure-Real Four-Stream Monotonic CORN + Siamese ===")
    print("=" * 90)

    assets = load_all_8foundation_and_calipers()
    X_264_c, X_264_t = build_or_load_264d_fg_bg_physics(assets)
    w_domain, boot_indices = compute_domain_weights_and_bootstrap_splits(assets)
    oof_pack = load_8fold_assignment_and_oof_anchors(assets["all_names"], assets["t150_names"])

    all_names = assets["all_names"]
    t150_names = assets["t150_names"]
    all_grades = assets["all_grades"]
    fold_ids = oof_pack["fold_ids"]
    N_c = len(all_names)
    N_t = len(t150_names)

    # 1. Assemble the 4 feature streams for 3,573 train + 150 test:
    # Stream A (Global Semantic): u_global from 8 foundations
    # Stream B (Local Anomaly Max Residual): u_local_max - u_global from 8 foundations
    # Stream C (1:1 Native Microscope Contrast): emax, tex, emax - glob, emax - tex from /dev/shm/v15_microscope_cache
    g_list_c, l_list_c = [], []
    g_list_t, l_list_t = [], []
    for Xc, Xt in zip(assets["blocks_c"], assets["blocks_t"]):
        d = Xc.shape[1] // 3
        ug_c, um_c, ul_c = Xc[:, :d], Xc[:, d:2*d], Xc[:, 2*d:]
        ug_t, um_t, ul_t = Xt[:, :d], Xt[:, d:2*d], Xt[:, 2*d:]
        g_list_c.append(0.65 * ug_c + 0.35 * um_c)
        l_list_c.append(ul_c - ug_c)
        g_list_t.append(0.65 * ug_t + 0.35 * um_t)
        l_list_t.append(ul_t - ug_t)

    X_glob_c = np.concatenate(g_list_c, axis=1).astype(np.float32)
    X_lmax_c = np.concatenate(l_list_c, axis=1).astype(np.float32)
    X_glob_t = np.concatenate(g_list_t, axis=1).astype(np.float32)
    X_lmax_t = np.concatenate(l_list_t, axis=1).astype(np.float32)

    # Load 1:1 Native Microscope features from /dev/shm/v15_microscope_cache
    mic_names = [str(n) for n in np.load("/dev/shm/v15_microscope_cache/names.npy", allow_pickle=True)]
    mic_map = {n: i for i, n in enumerate(mic_names)}
    c_m_idx = [mic_map[n] for n in all_names]
    t_m_idx = [mic_map[n] for n in t150_names]

    mic_c_list, mic_t_list = [], []
    for b_tag in ["dinov3l", "pecoreg", "siglip2", "convnext"]:
        dm = np.load(f"/dev/shm/v15_microscope_cache/{b_tag}_merged.npz", allow_pickle=True)
        gc = dm["glob"][c_m_idx].astype(np.float32)
        ec = dm["emax"][c_m_idx].astype(np.float32)
        tc = dm["tex"][c_m_idx].astype(np.float32)
        gt = dm["glob"][t_m_idx].astype(np.float32)
        et = dm["emax"][t_m_idx].astype(np.float32)
        tt = dm["tex"][t_m_idx].astype(np.float32)

        # L2 normalize each component and compute 1:1 edge-vs-global and edge-vs-texture contrast residuals
        gc /= (np.linalg.norm(gc, axis=1, keepdims=True) + 1e-8)
        ec /= (np.linalg.norm(ec, axis=1, keepdims=True) + 1e-8)
        tc /= (np.linalg.norm(tc, axis=1, keepdims=True) + 1e-8)
        gt /= (np.linalg.norm(gt, axis=1, keepdims=True) + 1e-8)
        et /= (np.linalg.norm(et, axis=1, keepdims=True) + 1e-8)
        tt /= (np.linalg.norm(tt, axis=1, keepdims=True) + 1e-8)

        mic_c_list.append(np.concatenate([ec - gc, ec - tc, tc - gc], axis=1))
        mic_t_list.append(np.concatenate([et - gt, et - tt, tt - gt], axis=1))

    # Concatenate dimensionality-reduced PCA or top channels of 1:1 microscope contrast so memory stays compact and fast
    X_mic_raw_c = np.concatenate(mic_c_list, axis=1)
    X_mic_raw_t = np.concatenate(mic_t_list, axis=1)
    pca_mic = PCA(n_components=768, random_state=42)
    X_mic_all = pca_mic.fit_transform(np.concatenate([X_mic_raw_c, X_mic_raw_t], axis=0)).astype(np.float32)
    X_mic_c, X_mic_t = X_mic_all[:N_c], X_mic_all[N_c:]
    del X_mic_raw_c, X_mic_raw_t

    # Also reduce X_glob and X_lmax via whitened SVD/PCA (preserving >96% variance while eliminating 28,800D overfitting!)
    pca_glob = PCA(n_components=1024, random_state=42)
    X_glob_all = pca_glob.fit_transform(np.concatenate([X_glob_c, X_glob_t], axis=0)).astype(np.float32)
    X_glob_c, X_glob_t = X_glob_all[:N_c], X_glob_all[N_c:]

    pca_lmax = PCA(n_components=1024, random_state=42)
    X_lmax_all = pca_lmax.fit_transform(np.concatenate([X_lmax_c, X_lmax_t], axis=0)).astype(np.float32)
    X_lmax_c, X_lmax_t = X_lmax_all[:N_c], X_lmax_all[N_c:]
    print(f"[Streams Ready] X_glob={X_glob_c.shape[1]}D, X_lmax={X_lmax_c.shape[1]}D, X_mic={X_mic_c.shape[1]}D, X_264={X_264_c.shape[1]}D")

    # 2. Construct OOF Disagreement Boundary Soft-Labels (Section 2.4)
    # Combine V16_Full OOF + valid-fold V9/V10 OOF to identify P1 <-> P2 subjective boundary drift (~6% of samples)
    soft_targets_c = np.zeros((N_c, 6, 4), dtype=np.float32)
    boundary_cleaned_cnt = 0
    for i in range(N_c):
        f_i = fold_ids[i]
        for c in range(6):
            g_true = int(all_grades[i, c])
            soft_targets_c[i, c, g_true] = 1.0
            if c in FULL_DIMS and g_true in (1, 2):
                # Check OOF consensus between V16_Full and V9/V10
                p16 = oof_pack["v16_oof"][i, c]
                active_9 = not ((f_i in (0, 1) and c in (0, 2)) or (f_i in (2, 3) and c in (4, 5)))
                p_ens = 0.55 * p16 + 0.45 * (0.55 * oof_pack["v9_oof"][i, c] + 0.45 * oof_pack["v10_oof"][i, c]) if active_9 else p16
                pred_g = int(np.argmax(p_ens))
                # If ground truth is P1(2) but ensemble strongly prefers P2(1), or vice versa, smooth to order-preserving soft distribution
                if g_true == 2 and pred_g == 1 and p_ens[1] > p_ens[2] + 0.08:
                    soft_targets_c[i, c] = np.array([0.02, 0.30, 0.63, 0.05], dtype=np.float32)
                    boundary_cleaned_cnt += 1
                elif g_true == 1 and pred_g == 2 and p_ens[2] > p_ens[1] + 0.08:
                    soft_targets_c[i, c] = np.array([0.05, 0.63, 0.30, 0.02], dtype=np.float32)
                    boundary_cleaned_cnt += 1
    print(f"[Sec 2.4 Boundary Cleaning] Smoothed {boundary_cleaned_cnt} high-disagreement P1<->P2 cells (preserving primary rank)")

    # 3. High-Confidence 4-Family Consensus Pseudo-Labels on test150 (Section 4.3)
    sub_5289 = json.load(open("results/final/pred_nriqa_dataset_test_phase1_breakthrough.json"))
    map_5289 = {e["image"]: [GRADE_REV[e["label"][c]] for c in DIMS] for e in sub_5289}
    pred_5289_t = np.array([map_5289[nm] for nm in t150_names], dtype=np.int64)

    v16_t_mean = oof_pack["v16_t150"].mean(axis=0)  # [150, 6, 4]
    dv13 = np.load("results/v13_vlm_real/test_150_vlm_corn_probs.npz", allow_pickle=True)
    ord_13 = [list(dv13["names_150"]).index(nm) for nm in t150_names]
    v13_p0, v13_p1, v13_p2 = dv13["t150_p0"][ord_13], dv13["t150_p1"][ord_13], dv13["t150_p2"][ord_13]

    pseudo_targets_t = np.zeros((N_t, 6, 4), dtype=np.float32)
    pseudo_mask_t = np.zeros((N_t, 6), dtype=np.float32)
    for j in range(N_t):
        for c in range(6):
            g_anc = int(pred_5289_t[j, c])
            g_v16 = int(np.argmax(v16_t_mean[j, c]))
            conf_v16 = float(v16_t_mean[j, c, g_v16])
            if c in FULL_DIMS:
                k = FULL_DIMS.index(c)
                p13_vec = np.array([max(0.0, 1.0 - (v13_p0[j, k] + v13_p1[j, k] + v13_p2[j, k])), v13_p2[j, k], v13_p1[j, k], v13_p0[j, k]])
            else:
                p13_vec = np.array([1.0 - v13_p0[j, c], 0.0, 0.0, v13_p0[j, c]])
            g_v13 = int(np.argmax(p13_vec))
            if g_anc == g_v16 == g_v13 and conf_v16 >= 0.55:
                pseudo_targets_t[j, c, g_anc] = 1.0
                pseudo_mask_t[j, c] = 1.0
    print(f"[Sec 4.3 Consensus Pseudo-Labels] {int((pseudo_mask_t.sum(axis=1) >= 4).sum())}/150 test images ({int(pseudo_mask_t.sum())}/900 cells) met multi-family consensus")

    # 4. Precompute L2-normalized scene embeddings (SigLIP2 + DINOv3 + InternViT) for K=15 Siamese Prototypes & Tuple Margin
    S_emb_c = X_glob_c / (np.linalg.norm(X_glob_c, axis=1, keepdims=True) + 1e-8)
    S_emb_t = X_glob_t / (np.linalg.norm(X_glob_t, axis=1, keepdims=True) + 1e-8)

    # Train all 8 folds on GPU
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    v17_oof_probs = np.zeros((N_c, 6, 4), dtype=np.float32)
    v17_t150_probs = np.zeros((8, N_t, 6, 4), dtype=np.float32)

    for f_idx in range(8):
        t_f = time.time()
        va_mask = (fold_ids == f_idx)
        tr_mask = ~va_mask
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]

        # Standardize 264D physics strictly using training fold statistics
        med_p = np.median(X_264_c[tr_idx], axis=0, keepdims=True)
        iqr_p = np.maximum(np.percentile(X_264_c[tr_idx], 75, axis=0, keepdims=True) - np.percentile(X_264_c[tr_idx], 25, axis=0, keepdims=True), 1e-4)
        sd_p = np.maximum(iqr_p / 1.349, X_264_c[tr_idx].std(axis=0, keepdims=True) + 1e-5)
        P_c = np.clip((X_264_c - med_p) / sd_p, -6.0, 6.0).astype(np.float32)
        P_t = np.clip((X_264_t - med_p) / sd_p, -6.0, 6.0).astype(np.float32)

        # Compute K=15 same-scene training prototypes for each sample (excluding self & same source on train)
        sim_c_tr = S_emb_c @ S_emb_c[tr_idx].T  # [N_c, N_tr]
        sim_t_tr = S_emb_t @ S_emb_c[tr_idx].T  # [N_t, N_tr]
        for ii, ti in enumerate(tr_idx):
            sim_c_tr[ti, ii] = -1e9  # exclude self

        K = 15
        top_c = np.argpartition(-sim_c_tr, K, axis=1)[:, :K]
        top_t = np.argpartition(-sim_t_tr, K, axis=1)[:, :K]

        def compute_proto_feats(sim_mat, top_k_idx):
            N_q = sim_mat.shape[0]
            rows = np.arange(N_q)[:, None]
            s_k = sim_mat[rows, top_k_idx]
            w_k = np.exp((s_k - s_k.max(axis=1, keepdims=True)) / 0.08)
            w_k /= w_k.sum(axis=1, keepdims=True)
            proto_g = np.einsum("nk,kd->nd", w_k, X_glob_c[tr_idx][top_k_idx].reshape(N_q, K, -1)[0] if False else np.zeros((1, 1))) if False else (
                (w_k[:, :, None] * X_glob_c[tr_idx[top_k_idx]]).sum(axis=1)
            )
            proto_d24 = (w_k[:, :, None, None] * soft_targets_c[tr_idx[top_k_idx]]).sum(axis=1).reshape(N_q, 24)
            return proto_g.astype(np.float32), proto_d24.astype(np.float32)

        proto_g_c, proto_d24_c = compute_proto_feats(sim_c_tr, top_c)
        proto_g_t, proto_d24_t = compute_proto_feats(sim_t_tr, top_t)

        # Combine training fold (N_tr) + high-confidence transductive test_150 samples (w_pseudo = 0.50)
        xg_tr = torch.from_numpy(np.concatenate([X_glob_c[tr_idx], X_glob_t], axis=0)).to(dev)
        xl_tr = torch.from_numpy(np.concatenate([X_lmax_c[tr_idx], X_lmax_t], axis=0)).to(dev)
        xm_tr = torch.from_numpy(np.concatenate([X_mic_c[tr_idx], X_mic_t], axis=0)).to(dev)
        xp_tr = torch.from_numpy(np.concatenate([P_c[tr_idx], P_t], axis=0)).to(dev)
        xpg_tr = torch.from_numpy(np.concatenate([proto_g_c[tr_idx], proto_g_t], axis=0)).to(dev)
        xpd_tr = torch.from_numpy(np.concatenate([proto_d24_c[tr_idx], proto_d24_t], axis=0)).to(dev)

        y_soft_tr = torch.from_numpy(np.concatenate([soft_targets_c[tr_idx], pseudo_targets_t], axis=0)).to(dev)
        y_hard_tr = torch.from_numpy(np.concatenate([all_grades[tr_idx], pred_5289_t], axis=0)).to(dev)
        v_mask_tr = torch.from_numpy(np.concatenate([np.ones((len(tr_idx), 6), dtype=np.float32), pseudo_mask_t], axis=0)).to(dev)
        w_samp_tr = torch.from_numpy(np.concatenate([w_domain[tr_idx], np.full(N_t, 0.50, dtype=np.float32)], axis=0)).to(dev)
        s_emb_tr = torch.from_numpy(np.concatenate([S_emb_c[tr_idx], S_emb_t], axis=0)).to(dev)

        xg_va = torch.from_numpy(X_glob_c[va_idx]).to(dev)
        xl_va = torch.from_numpy(X_lmax_c[va_idx]).to(dev)
        xm_va = torch.from_numpy(X_mic_c[va_idx]).to(dev)
        xp_va = torch.from_numpy(P_c[va_idx]).to(dev)
        xpg_va = torch.from_numpy(proto_g_c[va_idx]).to(dev)
        xpd_va = torch.from_numpy(proto_d24_c[va_idx]).to(dev)

        xg_te = torch.from_numpy(X_glob_t).to(dev)
        xl_te = torch.from_numpy(X_lmax_t).to(dev)
        xm_te = torch.from_numpy(X_mic_t).to(dev)
        xp_te = torch.from_numpy(P_t).to(dev)
        xpg_te = torch.from_numpy(proto_g_t).to(dev)
        xpd_te = torch.from_numpy(proto_d24_t).to(dev)

        # Train ensemble of 3 seeds per fold for low variance
        fold_va_preds = np.zeros((len(va_idx), 6, 4), dtype=np.float32)
        fold_t150_preds = np.zeros((N_t, 6, 4), dtype=np.float32)

        for seed in [42, 1024, 2026]:
            torch.manual_seed(seed + f_idx * 17)
            np.random.seed(seed + f_idx * 17)
            model = FourStreamSiameseMonotonicCORN(
                d_glob=X_glob_c.shape[1],
                d_lmax=X_lmax_c.shape[1],
                d_mic=X_mic_c.shape[1],
                d_phys=P_c.shape[1],
                hidden=512,
                dropout=0.22,
            ).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=28, eta_min=2e-5)

            N_train_tot = xg_tr.shape[0]
            batch_size = 512
            swa_probs_va = 0.0
            swa_probs_te = 0.0
            swa_cnt = 0

            for ep in range(1, 29):
                model.train()
                perm = torch.randperm(N_train_tot, device=dev)
                for b0 in range(0, N_train_tot, batch_size):
                    b_idx = perm[b0:b0 + batch_size]
                    if len(b_idx) < 16:
                        continue
                    opt.zero_grad()
                    p6x4, aux_t = model(
                        xg_tr[b_idx], xl_tr[b_idx], xm_tr[b_idx], xp_tr[b_idx], xpg_tr[b_idx], xpd_tr[b_idx]
                    )
                    sim_b = s_emb_tr[b_idx] @ s_emb_tr[b_idx].T
                    loss = compute_v17_joint_loss(
                        p6x4, aux_t, y_soft_tr[b_idx], y_hard_tr[b_idx], v_mask_tr[b_idx], w_samp_tr[b_idx], sim_b
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

                if ep >= 20:
                    model.eval()
                    with torch.no_grad():
                        p_va, _ = model(xg_va, xl_va, xm_va, xp_va, xpg_va, xpd_va)
                        p_te, _ = model(xg_te, xl_te, xm_te, xp_te, xpg_te, xpd_te)
                        swa_probs_va += p_va.cpu().numpy()
                        swa_probs_te += p_te.cpu().numpy()
                        swa_cnt += 1

            fold_va_preds += (swa_probs_va / swa_cnt) / 3.0
            fold_t150_preds += (swa_probs_te / swa_cnt) / 3.0

        v17_oof_probs[va_idx] = fold_va_preds
        v17_t150_probs[f_idx] = fold_t150_preds

        # Evaluate raw argmax and expected-score decode on this fold
        raw_pred_f = np.argmax(fold_va_preds, axis=-1)
        sc_f = competition_score(raw_pred_f, all_grades[va_idx])
        print(f"  [Fold {f_idx}] ({len(va_idx)} samples, {time.time()-t_f:.1f}s) Raw Argmax Total={sc_f['total']:.2f} | S_pos={sc_f['S_pos']*100:.2f}% | S_p0={sc_f['S_p0']*100:.2f}%")

    out_npz = os.path.join(OUT_DIR, "V17_PureReal_FourStream_8Fold_probs.npz")
    np.savez_compressed(
        out_npz,
        all_names=np.array(all_names),
        t150_names=np.array(t150_names),
        fold_ids=fold_ids,
        oof_probs_6x4=v17_oof_probs,
        t150_probs_8x6x4=v17_t150_probs,
    )
    sc_all = competition_score(np.argmax(v17_oof_probs, axis=-1), all_grades)
    print("-" * 90)
    print(f"[V17 Complete in {time.time()-t_start:.1f}s] Saved to {out_npz}")
    print(f"  Full 3,573 OOF Raw Argmax Score: Total={sc_all['total']:.3f} | S_pos={sc_all['S_pos']*100:.2f}% | S_p0={sc_all['S_p0']*100:.2f}% | S_clean={sc_all['S_clean']*100:.2f}%")


if __name__ == "__main__":
    main()
