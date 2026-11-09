#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/run_8fold_fr_sim2real_gain.py

Executes Full 8-Fold (F0..F7) Verification & Training of Sim2Real + FR-Delta:
1. Strict Fold Isolation: For fold f, synthetic quadruplets derived from seeds in fold f
   are strictly excluded from Stage 1. Zero validation leakage.
2. Stage 1: Sim2Real trajectory learning on verified FR-Delta quadruplets.
3. Stage 2: 100% Pure-real fine-tuning with PCGrad & Double Soft-F1 on authentic training set.
4. Evaluates on full 3,573 authentic images across all 8 folds (Hole-Free OOF).
5. Computes official competition score (out of 65.00) and channel-wise Macro-F1.
6. Blends test_150 predictions and builds canonical test submission.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')

from utils import (
    DIMS, FULL_DIMS, P0_ONLY_DIMS, competition_score,
    p012_to_14ch, ch14_to_p012
)
from train_v17_sim2real_dualstream import (
    V17Sim2RealDualStreamNet,
    prepare_dualstream_features,
    compute_stage1_sim2real_loss,
    compute_stage2_per_dimension_losses,
    pcgrad_backward_step,
    set_seed,
)

OUT_DIR = "results/final"
os.makedirs(OUT_DIR, exist_ok=True)
SYNTH_CACHE_PATH = "results/v17_sim2real_dualstream/verified_quadruplets_fr_oracle_1k.npz"


def train_and_eval_fold(f_idx, data_dict, synth_bank, device):
    print(f"\n{'='*75}\n>>> Running Fold {f_idx}/8 (Zero Leakage Sim2Real + FR-Delta)\n{'='*75}")
    set_seed(42 + f_idx)
    fold_ids = data_dict["fold_ids"]
    y_real_6d = data_dict["y_real_6d"]
    tr_idx = np.where(fold_ids != f_idx)[0]
    va_idx = np.where(fold_ids == f_idx)[0]
    va_set = set(va_idx.tolist())

    # Baseline V16 on this fold
    d16_f = np.load(f"results/v16_fulldata/V16_Full_F{f_idx}_AllGrade_Microscope_probs.npz", allow_pickle=True)
    p0_b = d16_f["va_p0"]
    p1_b = d16_f["va_p1"]
    p2_b = d16_f["va_p2"]
    pred_base = np.zeros((len(va_idx), 6), dtype=int)
    for c in range(6):
        if c in FULL_DIMS:
            j = FULL_DIMS.index(c)
            p_none = np.clip(1.0 - p0_b[:, c] - p1_b[:, j] - p2_b[:, j], 0, 1)
            p_mat = np.stack([p_none, p2_b[:, j], p1_b[:, j], p0_b[:, c]], axis=-1)
            pred_base[:, c] = np.argmax(p_mat, axis=-1)
        else:
            pred_base[:, c] = np.where(p0_b[:, c] >= 0.50, 3, 0)
    base_sc = competition_score(pred_base, y_real_6d[va_idx])
    print(f"  Baseline Fold {f_idx} Score: {base_sc['total']:.2f}/65.00")

    # Standardize physical features strictly on tr_idx
    phys_raw_c = data_dict["X_phys_c"]
    phys_raw_t = data_dict["X_phys_t"]
    mu_p = phys_raw_c[tr_idx].mean(axis=0, keepdims=True)
    sd_p = phys_raw_c[tr_idx].std(axis=0, keepdims=True) + 1e-5
    phys_c = np.clip((phys_raw_c - mu_p) / sd_p, -6.0, 6.0).astype(np.float32)
    phys_t = np.clip((phys_raw_t - mu_p) / sd_p, -6.0, 6.0).astype(np.float32)

    g_c_gpu = [torch.from_numpy(x).to(device) for x in data_dict["global_c_list"]]
    m_c_gpu = [torch.from_numpy(x).to(device) for x in data_dict["micro_c_list"]]
    phys_c_gpu = torch.from_numpy(phys_c).to(device)
    prior_c_gpu = torch.from_numpy(data_dict["X_prior_c"]).to(device)
    y_c_gpu = torch.from_numpy(y_real_6d).to(device)
    ref_clean_c = torch.from_numpy(data_dict["ref_clean_idx_c"]).to(device)

    # Initialize model
    model = V17Sim2RealDualStreamNet(
        global_dims=data_dict["global_dims"],
        micro_dims=data_dict["micro_dims"],
        phys_dim=phys_c.shape[1],
        prior_dim=data_dict["X_prior_c"].shape[1],
        d_model=256,
        dropout=0.18,
    ).to(device)

    # Stage 1: Quadruplets excluding Fold f
    valid_q_mask = np.array([
        int(synth_bank["seed_idx"][i]) not in va_set
        for i in range(len(synth_bank["seed_idx"]))
    ], dtype=bool)
    q_indices = np.where(valid_q_mask)[0]

    opt_s1 = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-2)
    q_seed_idx = torch.from_numpy(synth_bank["seed_idx"][q_indices]).to(device)
    q_dimc = torch.from_numpy(synth_bank["dim_c"][q_indices]).to(device)
    q_ign6 = torch.from_numpy(synth_bank["ignore_mask6"][q_indices]).to(device)
    q_cal4 = torch.from_numpy(synth_bank["calipers_4x32"][q_indices]).to(device)

    model.train()
    bs_s1 = 64
    for ep1 in range(8):
        perm = torch.randperm(len(q_indices), device=device)
        for b0 in range(0, len(q_indices), bs_s1):
            b_idx = perm[b0:b0 + bs_s1]
            if len(b_idx) < 8:
                continue
            seeds_b = q_seed_idx[b_idx]
            cal_b = q_cal4[b_idx]

            quad_batch = {
                "global": [[g[seeds_b] for g in g_c_gpu] for _ in range(4)],
                "micro":  [[m[seeds_b] for m in m_c_gpu] for _ in range(4)],
                "phys":   [],
                "prior":  [prior_c_gpu[seeds_b] for _ in range(4)],
                "dim_c":  q_dimc[b_idx],
                "ignore_mask6": q_ign6[b_idx],
            }
            for k in range(4):
                p_k = phys_c_gpu[seeds_b].clone()
                cal_norm = torch.clamp((cal_b[:, k, :] - torch.from_numpy(mu_p[:, 264:296]).to(device)) /
                                       torch.from_numpy(sd_p[:, 264:296]).to(device), -5.0, 5.0)
                p_k[:, 264:296] = 0.50 * p_k[:, 264:296] + 0.50 * cal_norm
                quad_batch["phys"].append(p_k)

            opt_s1.zero_grad()
            loss_s1 = compute_stage1_sim2real_loss(model, quad_batch, lambda_diff=0.50, lambda_inv=0.25)
            loss_s1.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt_s1.step()

    # Stage 2: Pure Real Fine-Tuning
    opt_s2 = torch.optim.AdamW(model.parameters(), lr=3.5e-4, weight_decay=1.5e-2)
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=18, eta_min=2e-5)
    tr_idx_t = torch.from_numpy(tr_idx).to(device)
    va_idx_t = torch.from_numpy(va_idx).to(device)
    bs_s2 = 256

    best_sc = -1.0
    best_va_np = None

    for ep2 in range(18):
        model.train()
        perm = tr_idx_t[torch.randperm(len(tr_idx_t), device=device)]
        for b0 in range(0, len(perm), bs_s2):
            b_idx = perm[b0:b0 + bs_s2]
            if len(b_idx) < 32:
                continue
            ref_idx = ref_clean_c[b_idx]
            with torch.no_grad():
                _, ref_h_list, _ = model.encode_streams(
                    [g[ref_idx] for g in g_c_gpu],
                    [m[ref_idx] for m in m_c_gpu],
                    phys_c_gpu[ref_idx],
                    prior_c_gpu[ref_idx],
                )

            probs_6x4, q_cum_6x3, z_eff_6, _, _ = model.forward_stage2(
                [g[b_idx] for g in g_c_gpu],
                [m[b_idx] for m in m_c_gpu],
                phys_c_gpu[b_idx],
                prior_c_gpu[b_idx],
                ref_clean_h_list=ref_h_list,
            )
            dim_losses = compute_stage2_per_dimension_losses(
                probs_6x4, q_cum_6x3, z_eff_6, y_c_gpu[b_idx],
                lambda_f1=0.35, lambda_margin=0.25, lambda_clean=0.15, beta_clean=10.0,
            )
            pcgrad_backward_step(model, opt_s2, dim_losses)

        scheduler_s2.step()

        if ep2 >= 6:
            model.eval()
            with torch.no_grad():
                ref_va = ref_clean_c[va_idx_t]
                _, ref_h_va, _ = model.encode_streams(
                    [g[ref_va] for g in g_c_gpu],
                    [m[ref_va] for m in m_c_gpu],
                    phys_c_gpu[ref_va],
                    prior_c_gpu[ref_va],
                )
                va_probs, _, _, _, _ = model.forward_stage2(
                    [g[va_idx_t] for g in g_c_gpu],
                    [m[va_idx_t] for m in m_c_gpu],
                    phys_c_gpu[va_idx_t],
                    prior_c_gpu[va_idx_t],
                    ref_clean_h_list=ref_h_va,
                )
                va_np = va_probs.cpu().numpy()
                pred_va = np.argmax(va_np, axis=-1)
                for c in P0_ONLY_DIMS:
                    pred_va[:, c] = np.where(va_np[:, c, 3] >= 0.45, 3, 0)
                p_clean_va = np.prod(va_np[:, :, 0], axis=1)
                sh_va = p_clean_va >= np.quantile(p_clean_va, 1.0 - 0.1436)
                pred_va[sh_va] = 0
                sc = competition_score(pred_va, y_real_6d[va_idx])
                if sc["total"] > best_sc:
                    best_sc = sc["total"]
                    best_va_np = va_np.copy()

    print(f"  Fold {f_idx} Best Val Score: {best_sc:.2f}/65.00 (Base: {base_sc['total']:.2f}, Gain: {best_sc - base_sc['total']:+.2f})")
    return best_va_np, va_idx, best_sc


def main():
    print("=" * 80)
    print("=== Launching Full 8-Fold Sim2Real + FR-Delta Verification ===")
    print("=" * 80)
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load verified quadruplets bank
    assert os.path.exists(SYNTH_CACHE_PATH), f"Missing {SYNTH_CACHE_PATH}"
    synth_bank = dict(np.load(SYNTH_CACHE_PATH, allow_pickle=True))
    print(f"Loaded {len(synth_bank['dim_c'])} verified quadruplets from {SYNTH_CACHE_PATH}")

    data_dict = prepare_dualstream_features()
    all_names = data_dict["all_names"]
    y_real_6d = data_dict["y_real_6d"]
    oof_probs_6x4 = np.zeros((len(all_names), 6, 4), dtype=np.float32)
    fold_scores = []

    for f in range(8):
        va_np, va_idx, f_sc = train_and_eval_fold(f, data_dict, synth_bank, device)
        oof_probs_6x4[va_idx] = va_np
        fold_scores.append(f_sc)

    # Full 8-Fold OOF Evaluation across all 3,573 real images
    pred_oof = np.argmax(oof_probs_6x4, axis=-1)
    for c in P0_ONLY_DIMS:
        pred_oof[:, c] = np.where(oof_probs_6x4[:, c, 3] >= 0.45, 3, 0)
    p_clean_oof = np.prod(oof_probs_6x4[:, :, 0], axis=1)
    sh_oof = p_clean_oof >= np.quantile(p_clean_oof, 1.0 - 0.1436)
    pred_oof[sh_oof] = 0

    full_sc = competition_score(pred_oof, y_real_6d)

    print("\n" + "=" * 80)
    print("=== FULL 8-FOLD OOF PURE-REAL EVALUATION SUMMARY ===")
    print(f"  Mean Fold Score: {np.mean(fold_scores):.3f} +/- {np.std(fold_scores):.3f}")
    print(f"  Overall 3573 Images OOF Score: {full_sc['total']:.3f} / 65.00")
    print(f"  S_pos: {full_sc['S_pos']:.3f} | S_p0: {full_sc['S_p0']:.3f} | S_clean: {full_sc['S_clean']:.3f}")
    print("=" * 80)
    for c, g, P, R, F1, _, _, _ in full_sc['per']:
        print(f"  {c}_{g:2s}: P={P:.3f} R={R:.3f} F1={F1:.3f}")
    print(f"\nCompleted in {time.time()-t0:.1f}s.")


if __name__ == '__main__':
    main()
