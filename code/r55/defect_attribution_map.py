#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/defect_attribution_map.py
==================================
Stage 2 Out-of-the-Box Breakthrough & Dual-Use Semantic Attribution Engine:
  - Strictly within `< 1.0s` and `< 1.0 GB` extra VRAM on a single 24 GB RTX 4090.
  - Zero blur / zero inpainting fill (avoids removing 过锐/过度生成 evidence globally or injecting 模糊).
  - Subtracts the empirical operator footprint measured on Clean training images of the corresponding fold.
  - Operates on `UnimodalQuadraticResidualHead`'s unsaturated continuous latent severity `severity_s [6]`.

Modes:
  1) `--gate-g4`: Validates latency, VRAM, and Ground-Truth Pointing-Game Accuracy (`>= 0.80` gate) on
                  `r55glitch_<v>_<k>.png` vs `r55glitchref_<v>_<k>.png` twin frames, plus generates a
                  60-image (`20 P2 + 20 P1 + 20 P0`) out-of-fold `过度生成` visual localization & deletion audit.
  2) `--extract-oof --fold K`: Uses ONLY fold-K held-out `pe_core_large` checkpoints (`r55_fixed_s2026` +
                  `r55_glitch2_s2026`) to extract leak-free spatial attribution maps, Top-2 1:1 Peak Defect
                  Crop Boxes (`[2, 4]`), and 6 canonical spatial-topology features per defect for fold K's
                  validation rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    if os.path.join(REPO, _p) not in sys.path:
        sys.path.insert(0, os.path.join(REPO, _p))

from qingyu.backbones_qingyu import QingyuVisionQualityTower
from qingyu.config_and_paths import DIMS, REPO_ROOT, LosslessNativeImageLoader
from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM, DualEnginePhysicsExtractor, PrecomputedDualEngineStore


def _build_feathered_grid_masks(h: int, w: int, grid_n: int, feather_frac: float = 0.18) -> np.ndarray:
    """Returns `[grid_n*grid_n, h, w]` smooth cosine-feathered masks in `[0, 1]`."""
    masks = np.zeros((grid_n * grid_n, h, w), dtype=np.float32)
    ys = np.linspace(0.0, float(grid_n), h, endpoint=False) + 0.5 / grid_n
    xs = np.linspace(0.0, float(grid_n), w, endpoint=False) + 0.5 / grid_n
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    half = 0.5 + feather_frac
    inner = max(0.05, 0.5 - feather_frac)
    idx = 0
    for r in range(grid_n):
        cy = r + 0.5
        dy = np.abs(yy - cy)
        wy = np.clip((half - dy) / max(1e-5, half - inner), 0.0, 1.0)
        wy = 0.5 - 0.5 * np.cos(np.pi * wy)
        for c in range(grid_n):
            cx = c + 0.5
            dx = np.abs(xx - cx)
            wx = np.clip((half - dx) / max(1e-5, half - inner), 0.0, 1.0)
            wx = 0.5 - 0.5 * np.cos(np.pi * wx)
            masks[idx] = wy * wx
            idx += 1
    return masks


def load_pe_tower(ckpt_rel_path: str, device: torch.device, half: bool = True) -> QingyuVisionQualityTower:
    model = QingyuVisionQualityTower(backbone_key="pe_core_large", phys_dim=PHYS_FEAT_DIM)
    sd = torch.load(os.path.join(REPO, ckpt_rel_path), map_location="cpu")
    _, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"Unexpected keys in {ckpt_rel_path}: {unexpected[:5]}"
    if half:
        model.backbone.to(torch.bfloat16)
        for n, mod in model.backbone.named_modules():
            if any(t in n.split(".")[-1] for t in ("tap3", "dual_tap", "abmil")):
                mod.float()
    model.to(device).eval()
    return model


class PEAttributionOperator:
    """
    Fast single-tower attribution operator on `QingyuVisionQualityTower('pe_core_large')`.
    Supports:
      - `feathered_4x4`: 4x4 cosine-feathered cell-mean RGB replacement (`[17, 3, 448, 448]`).
      - `midblock_4x4`:  Runs blocks `0..14` ONCE (`[1, 3, 448, 448]`), replaces each `8x8` patch-token cell
                         with mean patch token at block 14 + masks cell in `abmil`, runs blocks `15..23` on `[17, L, D]`.
      - `midblock_8x8`:  Runs blocks `0..18` ONCE (`[1, 3, 448, 448]`), replaces each `4x4` patch-token cell
                         (`8x8 = 64` cells) at block 18 + masks cell in `abmil`, runs blocks `19..23` on `[65, L, D]`.
      - `readout_8x8`:   Runs full ViT `0..23` ONCE (`[1, 3, 448, 448]`), drops each `4x4` patch-token cell
                         from `abmil` (`[65]` vector readouts, `< 2 ms`).
    """

    def __init__(self, model: QingyuVisionQualityTower, device: torch.device):
        self.model = model
        self.dev = device
        self.masks_4x4_448 = torch.from_numpy(_build_feathered_grid_masks(448, 448, 4)).to(device)  # [16, 448, 448]
        self.footprints: Dict[str, np.ndarray] = {}

    def _pil_to_448_01(self, pil_img: Image.Image) -> torch.Tensor:
        im448 = pil_img.convert("RGB").resize((448, 448), Image.BILINEAR)
        arr = np.asarray(im448, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.dev)  # [1, 3, 448, 448]

    def _abmil_subset_pool(self, t_fused: torch.Tensor, keep_mask_bool: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        `t_fused`: `[M, 1025, 1024]` (`1 CLS + 1024` spatial tokens).
        `keep_mask_bool`: `[M, 1024]` bool mask of kept spatial patches (if provided, computes ABMIL on kept tokens).
        """
        bb = self.model.backbone
        with torch.autocast(device_type="cuda", enabled=False):
            tf = t_fused.float()
            cls_tok = tf[:, 0]
            patch_toks = tf[:, 1:]
            if keep_mask_bool is None:
                return torch.cat([cls_tok, bb.abmil(tf, has_cls=True)], dim=1)
            # Vectorized fixed-count gathering when every row keeps the exact same number of tokens!
            M, N, D = patch_toks.shape
            counts = keep_mask_bool.sum(dim=1)
            if int(counts.min().item()) == int(counts.max().item()):
                k_keep = int(counts[0].item())
                kept = patch_toks[keep_mask_bool].reshape(M, k_keep, D)
                pt = torch.cat([cls_tok.unsqueeze(1), kept], dim=1)
                return torch.cat([cls_tok, bb.abmil(pt, has_cls=True)], dim=1)
            outs = []
            for i in range(M):
                kept_i = patch_toks[i : i + 1, keep_mask_bool[i]]
                pt_i = torch.cat([cls_tok[i : i + 1].unsqueeze(1), kept_i], dim=1)
                outs.append(torch.cat([cls_tok[i : i + 1], bb.abmil(pt_i, has_cls=True)], dim=1))
            return torch.cat(outs, dim=0)

    def _head_severity(self, single_view_feat: torch.Tensor, phys_t: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs `view_agg` (single-view broadcast `[feat, feat, feat]`) + `head` -> `(severity_s [M, 6], probs [M, 6, 4])`."""
        with torch.autocast(device_type="cuda", enabled=False):
            vis_sum = self.model.view_agg(single_view_feat.unsqueeze(1).float())
            pf = phys_t.expand(vis_sum.shape[0], -1) if phys_t is not None else None
            out = self.model.head(vis_sum, phys_feat=pf)
            return out["severity_s"], out["probs_6x4"]

    @torch.no_grad()
    def raw_map(
        self,
        pil_img: Image.Image,
        mode: str = "midblock_8x8",
        phys_vec: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns `(delta_s [6, G, G], base_probs [6, 4])` BEFORE subtracting Clean footprint.
        Positive `delta_s[d, r, c]` means removing cell `(r, c)` LOWERED defect `d` severity!
        """
        x_01 = self._pil_to_448_01(pil_img)  # [1, 3, 448, 448]
        phys_t = (
            torch.from_numpy(phys_vec).float().unsqueeze(0).to(self.dev)
            if phys_vec is not None
            else torch.zeros((1, PHYS_FEAT_DIM), dtype=torch.float32, device=self.dev)
        )

        if mode == "feathered_4x4":
            G = 4
            # Compute per-cell weighted RGB mean [16, 3, 1, 1]
            m = self.masks_4x4_448.unsqueeze(1)  # [16, 1, 448, 448]
            cell_mean = (x_01 * m).sum(dim=(2, 3), keepdim=True) / m.sum(dim=(2, 3), keepdim=True).clamp(min=1e-5)
            x_occ = x_01 * (1.0 - m) + cell_mean * m  # [16, 3, 448, 448]
            batch_01 = torch.cat([x_01, x_occ], dim=0)  # [17, 3, 448, 448]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                v_feat = self.model._encode_single_view(self.model._norm_img(batch_01))
            sev, probs = self._head_severity(v_feat, phys_t)
            delta = (sev[0:1] - sev[1:]).transpose(0, 1).reshape(6, G, G)
            return delta.cpu().numpy(), probs[0].cpu().numpy()

        bb = self.model.backbone
        x_norm = self.model._norm_img(x_01)

        if mode == "readout_8x8":
            G = 8
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw_t = bb.forward_features(x_norm)
            t_fused = bb.tap3.fuse(raw_t.float())  # [1, 1025, 1024]
            base_feat = self._abmil_subset_pool(t_fused, None)  # [1, 2048]
            # Build 64 token-drop masks on 32x32 grid (each cell is 4x4=16 tokens)
            keep_64 = torch.ones((64, 32, 32), dtype=torch.bool, device=self.dev)
            idx = 0
            for r in range(8):
                for c in range(8):
                    keep_64[idx, r * 4 : (r + 1) * 4, c * 4 : (c + 1) * 4] = False
                    idx += 1
            occ_feat = self._abmil_subset_pool(t_fused.expand(64, -1, -1), keep_64.reshape(64, 1024))
            all_feat = torch.cat([base_feat, occ_feat], dim=0)
            sev, probs = self._head_severity(all_feat, phys_t)
            delta = (sev[0:1] - sev[1:]).transpose(0, 1).reshape(6, G, G)
            return delta.cpu().numpy(), probs[0].cpu().numpy()

        if mode == "sub8_blk15":
            # Full 8x8 sub-cell (4x4 tokens) map at split_blk=15, chunked by 16 so peak extra VRAM stays <=0.744 GB.
            # Used once during `fit_clean_footprint` to calibrate exact sub-cell Clean positional bias!
            split_blk = 15
            bb.tap3.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = bb.patch_embed(x_norm)
                h, rot_pos = bb._pos_embed(h)
                h = bb.norm_pre(h)
                rope_mixed = getattr(bb, "rope_mixed", False) and rot_pos is not None
                for bi in range(split_blk):
                    rp = rot_pos[bi] if rope_mixed else rot_pos
                    h = bb.blocks[bi](h, rope=rp)
                patch_h = h[:, 1:, :].reshape(1, 32, 32, -1)
                mean_tok = h[:, 1:, :].mean(dim=1, keepdim=True)

                h_base = h.clone()
                for bi in range(split_blk, len(bb.blocks)):
                    rp = rot_pos[bi] if rope_mixed else rot_pos
                    h_base = bb.blocks[bi](h_base, rope=rp)
                raw_b = bb.norm(h_base)
            tf_b = bb.tap3.fuse(raw_b.float())
            base_feat = self._abmil_subset_pool(tf_b, None)
            sev_b, probs_b = self._head_severity(base_feat, phys_t)

            all_specs = [(r8, c8) for r8 in range(8) for c8 in range(8)]
            sub_map = np.zeros((6, 8, 8), dtype=np.float32)
            for s0 in range(0, 64, 16):
                chunk_sp = all_specs[s0 : s0 + 16]
                K_sub = len(chunk_sp)
                bb.tap3.clear()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    h_sub = h.repeat(K_sub, 1, 1)
                    keep_sub = torch.ones((K_sub, 32, 32), dtype=torch.bool, device=self.dev)
                    for si, (r8, c8) in enumerate(chunk_sp):
                        keep_sub[si, r8 * 4 : (r8 + 1) * 4, c8 * 4 : (c8 + 1) * 4] = False
                        p_c = patch_h.clone()
                        p_c[:, r8 * 4 : (r8 + 1) * 4, c8 * 4 : (c8 + 1) * 4, :] = mean_tok.view(1, 1, 1, -1)
                        h_sub[si, 1:, :] = p_c.reshape(1, 1024, -1)
                    for bi in range(split_blk, len(bb.blocks)):
                        rp = rot_pos[bi] if rope_mixed else rot_pos
                        h_sub = bb.blocks[bi](h_sub, rope=rp)
                    raw_sub = bb.norm(h_sub)
                tf_sub = bb.tap3.fuse(raw_sub.float())
                occ_sub = self._abmil_subset_pool(tf_sub, keep_sub.reshape(K_sub, 1024))
                sev_sub, _ = self._head_severity(occ_sub, phys_t)
                d_sub = (sev_b - sev_sub).cpu().numpy()
                for si, (r8, c8) in enumerate(chunk_sp):
                    sub_map[:, r8, c8] = d_sub[si]
            return sub_map, probs_b[0].cpu().numpy()

        if mode in ("midblock_4x4", "midblock_8x8", "midblock_4x4_tok32", "midblock_hier8"):
            G = 4 if ("4x4" in mode or mode == "midblock_hier8") else 8
            split_blk = 15 if ("4x4" in mode or mode == "midblock_hier8") else 19  # tap3 hooks [15, 19, 22]!
            M = G * G
            cell_sz = 32 // G
            bb.tap3.clear()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = bb.patch_embed(x_norm)
                h, rot_pos = bb._pos_embed(h)
                h = bb.norm_pre(h)
                rope_mixed = getattr(bb, "rope_mixed", False) and rot_pos is not None
                for bi in range(split_blk):
                    rp = rot_pos[bi] if rope_mixed else rot_pos
                    h = bb.blocks[bi](h, rope=rp)

                tap0_saved = bb.tap3._caches.get(0, None)
                if tap0_saved is not None:
                    tap0_saved = tap0_saved.clone()

                patch_h = h[:, 1:, :].reshape(1, 32, 32, -1)
                mean_tok = h[:, 1:, :].mean(dim=1, keepdim=True)  # [1, 1, D]
                h_batch = h.repeat(1 + M, 1, 1)
                keep_m = torch.ones((M, 32, 32), dtype=torch.bool, device=self.dev)
                idx = 0
                for r in range(G):
                    for c in range(G):
                        keep_m[idx, r * cell_sz : (r + 1) * cell_sz, c * cell_sz : (c + 1) * cell_sz] = False
                        p_copy = patch_h.clone()
                        p_copy[:, r * cell_sz : (r + 1) * cell_sz, c * cell_sz : (c + 1) * cell_sz, :] = mean_tok.view(1, 1, 1, -1)
                        h_batch[1 + idx, 1:, :] = p_copy.reshape(1, 1024, -1)
                        idx += 1

                for bi in range(split_blk, len(bb.blocks)):
                    rp = rot_pos[bi] if rope_mixed else rot_pos
                    h_batch = bb.blocks[bi](h_batch, rope=rp)
                raw_out = bb.norm(h_batch)

            if split_blk > 15 and tap0_saved is not None:
                t0_b = tap0_saved.repeat(1 + M, 1, 1)
                t0_mean = tap0_saved[:, 1:, :].mean(dim=1, keepdim=True)
                t0_grid = tap0_saved[:, 1:, :].reshape(1, 32, 32, -1)
                idx = 0
                for r in range(G):
                    for c in range(G):
                        p_c = t0_grid.clone()
                        p_c[:, r * cell_sz : (r + 1) * cell_sz, c * cell_sz : (c + 1) * cell_sz, :] = t0_mean.view(1, 1, 1, -1)
                        t0_b[1 + idx, 1:, :] = p_c.reshape(1, 1024, -1)
                        idx += 1
                bb.tap3._caches[0] = t0_b

            t_fused = bb.tap3.fuse(raw_out.float())  # [1 + M, 1025, 1024]
            base_feat = self._abmil_subset_pool(t_fused[:1], None)
            occ_feat = self._abmil_subset_pool(t_fused[1:], keep_m.reshape(M, 1024))
            all_feat = torch.cat([base_feat, occ_feat], dim=0)
            sev, probs = self._head_severity(all_feat, phys_t)
            delta = (sev[0:1] - sev[1:]).transpose(0, 1).reshape(6, G, G)

            if mode == "midblock_hier8":
                # Coarse-to-fine 2-stage causal zoom:
                #   Stage 1: 4x4 coarse calibrated map (`CellHit = 90.0%`).
                #   Stage 2: Refine each defect's winning coarse cell into 2x2 sub-cells (4x4 tokens = 56x56 px)
                #            using Clean-calibrated `split_blk=15` sub-cell counterfactual drops (`<=16` seqs/batch -> 0.744 GB VRAM!).
                delta_np = delta.cpu().numpy()
                coarse_cal = delta_np - self.footprints.get("midblock_4x4", 0.0)
                fp_sub8 = self.footprints.get("sub8_blk15", np.zeros((6, 8, 8), dtype=np.float32))

                top_cells = set()
                for d_i in range(6):
                    top_cells.add(int(np.argmax(coarse_cal[d_i].ravel())))
                top_list = sorted(top_cells)  # <= 6 distinct coarse cells across 6 defects

                fine_8 = np.repeat(np.repeat(coarse_cal, 2, axis=1), 2, axis=2)  # [6, 8, 8]
                sub_specs = []
                for cid in top_list:
                    cr, cc = divmod(cid, 4)
                    for sr in range(2):
                        for sc in range(2):
                            sub_specs.append((cr * 2 + sr, cc * 2 + sc))

                sub_cal_8 = np.zeros((6, 8, 8), dtype=np.float32)
                for s0 in range(0, len(sub_specs), 16):
                    chunk_sp = sub_specs[s0 : s0 + 16]
                    K_sub = len(chunk_sp)
                    bb.tap3.clear()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        h_sub = h.repeat(K_sub, 1, 1)
                        keep_sub = torch.ones((K_sub, 32, 32), dtype=torch.bool, device=self.dev)
                        for si, (r8, c8) in enumerate(chunk_sp):
                            keep_sub[si, r8 * 4 : (r8 + 1) * 4, c8 * 4 : (c8 + 1) * 4] = False
                            p_c = patch_h.clone()
                            p_c[:, r8 * 4 : (r8 + 1) * 4, c8 * 4 : (c8 + 1) * 4, :] = mean_tok.view(1, 1, 1, -1)
                            h_sub[si, 1:, :] = p_c.reshape(1, 1024, -1)
                        for bi in range(split_blk, len(bb.blocks)):
                            rp = rot_pos[bi] if rope_mixed else rot_pos
                            h_sub = bb.blocks[bi](h_sub, rope=rp)
                        raw_sub = bb.norm(h_sub)
                    tf_sub = bb.tap3.fuse(raw_sub.float())
                    occ_sub = self._abmil_subset_pool(tf_sub, keep_sub.reshape(K_sub, 1024))
                    sev_sub, _ = self._head_severity(occ_sub, phys_t)
                    d_sub = (sev[0:1] - sev_sub).cpu().numpy()  # [K_sub, 6]
                    for si, (r8, c8) in enumerate(chunk_sp):
                        sub_cal_8[:, r8, c8] = d_sub[si] - fp_sub8[:, r8, c8]

                # Within each defect's winning coarse cell, sharpen sub-cell peak while strictly preserving global coarse cell rank!
                for d_i in range(6):
                    win_cid = int(np.argmax(coarse_cal[d_i].ravel()))
                    cr, cc = divmod(win_cid, 4)
                    q4 = sub_cal_8[d_i, cr * 2 : cr * 2 + 2, cc * 2 : cc * 2 + 2]
                    fine_8[d_i, cr * 2 : cr * 2 + 2, cc * 2 : cc * 2 + 2] = (
                        coarse_cal[d_i, cr, cc] + np.maximum(0.0, q4 - q4.mean()) + 1e-3 * (q4 - q4.min())
                    )
                return fine_8, probs[0].cpu().numpy()

            if mode == "midblock_4x4_tok32":
                # Zero-extra-pass 32x32 fine sub-cell localization inside each 8x8 token cell!
                with torch.autocast(device_type="cuda", enabled=False):
                    pt = t_fused[0:1, 1:, :].float()  # [1, 1024, 1024]
                    ab = bb.abmil
                    anom = ab.anom_scorer(pt).squeeze(-1) / math.sqrt(pt.shape[-1])
                    mil_a = ab.attn_w(torch.tanh(ab.attn_V(pt)) * torch.sigmoid(ab.attn_U(pt))).squeeze(-1)
                    tok_sal = (torch.sigmoid(anom) + torch.sigmoid(mil_a)).reshape(4, 8, 4, 8)
                    # Normalize sub-cell weights so mean within each 8x8 coarse cell == 1.0 (preserves exact cell causal total!)
                    tok_norm = tok_sal / tok_sal.mean(dim=(1, 3), keepdim=True).clamp(min=1e-5)
                    fine_32 = (delta[:, :, None, :, None] * tok_norm[None, ...]).reshape(6, 32, 32)
                return fine_32.cpu().numpy(), probs[0].cpu().numpy()

            return delta.cpu().numpy(), probs[0].cpu().numpy()

        raise ValueError(f"Unknown attribution mode: {mode}")

    def fit_clean_footprint(self, clean_pils: Sequence[Image.Image], mode: str, phys_vecs: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
        if mode == "midblock_hier8":
            if "midblock_4x4" not in self.footprints:
                self.fit_clean_footprint(clean_pils, mode="midblock_4x4", phys_vecs=phys_vecs)
            if "sub8_blk15" not in self.footprints:
                self.fit_clean_footprint(clean_pils, mode="sub8_blk15", phys_vecs=phys_vecs)
            fp0 = np.zeros((6, 8, 8), dtype=np.float32)
            self.footprints["midblock_hier8"] = fp0
            return fp0
        acc = []
        for i, im in enumerate(clean_pils):
            pv = phys_vecs[i] if phys_vecs is not None else None
            d_map, _ = self.raw_map(im, mode=mode, phys_vec=pv)
            acc.append(d_map)
        fp = np.mean(acc, axis=0).astype(np.float32)
        self.footprints[mode] = fp
        return fp

    def calibrated_map(self, pil_img: Image.Image, mode: str, phys_vec: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        d_map, probs = self.raw_map(pil_img, mode=mode, phys_vec=phys_vec)
        if mode in self.footprints:
            d_map = d_map - self.footprints[mode]
        return d_map, probs


def map_peak_xy_and_boxes(
    attr_2d: np.ndarray,
    orig_w: int,
    orig_h: int,
    crop_sz: int = 448,
) -> Tuple[Tuple[int, int], List[Tuple[int, int, int, int]]]:
    """
    Finds the sub-grid bilinear peak `(px, py)` in native `(orig_w, orig_h)` coordinates AND
    extracts Top-2 NMS-separated `1:1` `crop_sz x crop_sz` boxes around the highest attribution regions.
    """
    pos = np.maximum(attr_2d.astype(np.float32), 0.0)
    if float(pos.max()) <= 1e-7:
        pos = attr_2d.astype(np.float32) - float(attr_2d.min()) + 1e-6

    up_h, up_w = max(64, min(512, orig_h // 2)), max(64, min(512, orig_w // 2))
    up = cv2.resize(pos, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
    py_u, px_u = divmod(int(np.argmax(up)), up_w)
    px = int(np.clip(round((px_u + 0.5) * orig_w / up_w), 0, orig_w - 1))
    py = int(np.clip(round((py_u + 0.5) * orig_h / up_h), 0, orig_h - 1))

    from semantic_roi_meta import _pick_box_nms
    taken: List[Tuple[int, int, int, int]] = []
    b0 = _pick_box_nms(up, crop_sz=crop_sz, orig_h=orig_h, orig_w=orig_w, taken=taken, max_iou=0.20)
    b1 = _pick_box_nms(up, crop_sz=crop_sz, orig_h=orig_h, orig_w=orig_w, taken=taken, max_iou=0.20)
    return (px, py), [b0, b1]


def compute_spatial_evidence_features(
    cal_map_6xGxG: np.ndarray,
    roi_boxes_4x4: Optional[np.ndarray],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """
    Computes strictly `<= 6` compact, physically interpretable spatial-topology features per image
    focused on `过度生成` (dim 4), `过锐` (dim 2), `模糊` (dim 0), and `过度生成 <-> 虚化突兀` (dim 5) disambiguation:
      0: `og_peak_drop`:       Max positive attribution drop on `过度生成` (`cal_map[4].max()`)
      1: `og_area_frac`:       Fraction of grid cells with `cal_map[4] > 0.35 * max(cal_map[4], 0.05)` (localized vs global!)
      2: `og_gini_conc`:       Concentration ratio `max(pos_4) / (mean(pos_4) + 1e-4)` (distinguishes pinpoint hand/eye flaw vs diffuse noise)
      3: `og_anat_text_ov`:    Attribution mass fraction of `过度生成` falling inside `[subtitle_ocr, face_anat, hand_swirl]` semantic ROIs
      4: `og_bokeh_cliff_ov`:  Attribution mass fraction of `过度生成` falling inside `bokeh_halo` boundary ROI
      5: `og_vs_bokeh_cos`:    Spatial cosine similarity between positive `过度生成` map (`pos_4`) and `虚化突兀` map (`pos_5`)
    Also appends `[peak_drop, area_frac, gini_conc]` for dims `0 (模糊)`, `1 (纹理损失)`, `2 (过锐)` -> total 15 numbers
    (with the first 6 being the exact pre-registered 6 features for Step 0a!).
    """
    G = cal_map_6xGxG.shape[1]
    pos = np.maximum(cal_map_6xGxG, 0.0)

    def _stats(d: int) -> Tuple[float, float, float]:
        m = pos[d]
        pk = float(m.max())
        thr = max(0.05, 0.35 * pk)
        area = float((m >= thr).mean())
        conc = float(pk / (float(m.mean()) + 1e-4)) if pk > 1e-5 else 0.0
        return pk, area, conc

    og_pk, og_area, og_conc = _stats(4)

    # Overlap with semantic ROIs (0: subtitle, 1: face, 2: hand, 3: bokeh_halo)
    anat_ov, bokeh_ov = 0.0, 0.0
    m4 = pos[4]
    m4_sum = float(m4.sum()) + 1e-6
    if roi_boxes_4x4 is not None and len(roi_boxes_4x4) >= 4 and orig_w > 0 and orig_h > 0:
        yy, xx = np.meshgrid(
            (np.arange(G) + 0.5) * orig_h / G,
            (np.arange(G) + 0.5) * orig_w / G,
            indexing="ij",
        )
        anat_mask = np.zeros((G, G), dtype=bool)
        for bi in (0, 1, 2):
            x0, y0, x1, y1 = roi_boxes_4x4[bi]
            anat_mask |= (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        bx0, by0, bx1, by1 = roi_boxes_4x4[3]
        bokeh_mask = (xx >= bx0) & (xx <= bx1) & (yy >= by0) & (yy <= by1)
        anat_ov = float(m4[anat_mask].sum() / m4_sum)
        bokeh_ov = float(m4[bokeh_mask].sum() / m4_sum)

    m5 = pos[5]
    n4 = float(np.linalg.norm(m4))
    n5 = float(np.linalg.norm(m5))
    cos_45 = float(np.sum(m4 * m5) / (n4 * n5 + 1e-6)) if (n4 > 1e-5 and n5 > 1e-5) else 0.0

    feats = [og_pk, og_area, og_conc, anat_ov, bokeh_ov, cos_45]
    for d in (0, 1, 2):
        feats.extend(_stats(d))
    return np.array(feats, dtype=np.float32)


def run_gate_g4(gpu: int = 1, n_twins: int = 240) -> None:
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(dev)
    torch.cuda.reset_peak_memory_stats(dev)

    print("=== [Gate G4] Loading Fold-0 Glitch & Fixed PE-Core-Large Towers ===", flush=True)
    m_gl = load_pe_tower("results/next55/towers/r55_glitch2_s2026/pe_core_large/fold_0_best.pt", dev, half=True)
    m_fx = load_pe_tower("results/next55/towers/r55_fixed_s2026/pe_core_large/fold_0_best.pt", dev, half=True)
    op_gl = PEAttributionOperator(m_gl, dev)
    op_fx = PEAttributionOperator(m_fx, dev)

    # 1) Load glitch twins (`r55glitch_<v>_<k>.png` vs matched `r55glitchref_<v>_<k>.png`)
    aux_img_dir = os.path.join(REPO, "data/r55_aux/images")
    twin_rows = list(csv.DictReader(open(os.path.join(REPO, "data/r55_aux/glitch_frames.csv"), encoding="utf-8")))
    valid_twins = []
    for r in twin_rows:
        g_name = r["name"]
        ref_name = g_name.replace("r55glitch_", "r55glitchref_")
        gp = os.path.join(aux_img_dir, g_name)
        rp = os.path.join(aux_img_dir, ref_name)
        if os.path.exists(gp) and os.path.exists(rp):
            valid_twins.append((gp, rp, r["mode"], float(r["share"])))
        if len(valid_twins) >= n_twins + 60:
            break

    # Use first 60 uncorrupted `r55glitchref` frames purely to fit the Clean operator footprint; test on remaining `n_twins`!
    clean_cal_pils = [Image.open(rp).convert("RGB") for _, rp, _, _ in valid_twins[:60]]
    eval_twins = valid_twins[60 : 60 + n_twins]

    modes = ["midblock_4x4", "midblock_hier8"]
    report = {}
    op_gl.fit_clean_footprint(clean_cal_pils, mode="midblock_4x4")
    for mode in modes:
        if mode != "midblock_4x4":
            op_gl.fit_clean_footprint(clean_cal_pils, mode=mode)
        torch.cuda.synchronize(dev)
        base_mem = torch.cuda.memory_allocated(dev) / (2 ** 30)
        torch.cuda.reset_peak_memory_stats(dev)

        hits_block, hits_cell, del_drops = 0, 0, []
        t0 = time.perf_counter()
        for gp, rp, _, _ in eval_twins:
            im_g = Image.open(gp).convert("RGB")
            im_r = Image.open(rp).convert("RGB")
            w, h = im_g.size
            arr_g = np.asarray(im_g, dtype=np.int16)
            arr_r = np.asarray(im_r, dtype=np.int16)
            # True corruption mask: macroblocks (16x16 dilated) where max RGB diff > 18
            diff_px = np.max(np.abs(arr_g - arr_r), axis=-1) > 18
            # Standard 16x16 macroblock + 1-patch (1/32 image) ViT receptive-field tolerance
            tol = max(15, int(round(max(w, h) / 32.0)))
            if tol % 2 == 0:
                tol += 1
            gt_mask = cv2.dilate(diff_px.astype(np.uint8), np.ones((tol, tol), np.uint8)) > 0
            if not gt_mask.any():
                gt_mask = np.max(np.abs(arr_g - arr_r), axis=-1) > 5

            cal_m, _ = op_gl.calibrated_map(im_g, mode=mode)
            attr_gl = cal_m[3]  # dim 3 = 花屏
            (px, py), _ = map_peak_xy_and_boxes(attr_gl, w, h)
            hits_block += int(bool(gt_mask[py, px]))

            # Coarse 4x4 cell hit check
            attr_4 = attr_gl.reshape(4, attr_gl.shape[0] // 4, 4, attr_gl.shape[1] // 4).mean(axis=(1, 3))
            pr, pc = divmod(int(np.argmax(attr_4)), 4)
            y0, y1 = int(round(pr * h / 4)), int(round((pr + 1) * h / 4))
            x0, x1 = int(round(pc * w / 4)), int(round((pc + 1) * w / 4))
            cell_corrupt_frac = float(diff_px[y0:y1, x0:x1].mean())
            hits_cell += int(cell_corrupt_frac >= 0.05)
            del_drops.append(float(attr_4.max()))

        torch.cuda.synchronize(dev)
        sec_per_img = (time.perf_counter() - t0) / len(eval_twins)
        peak_extra_gb = max(0.0, torch.cuda.max_memory_allocated(dev) / (2 ** 30) - base_mem)
        pg_acc = hits_block / len(eval_twins)
        cell_acc = hits_cell / len(eval_twins)
        report[mode] = dict(
            n_eval=len(eval_twins),
            pointing_game_acc=round(pg_acc, 4),
            peak_cell_hit_acc=round(cell_acc, 4),
            mean_peak_sev_drop=round(float(np.mean(del_drops)), 4),
            h200_sec_per_img=round(sec_per_img, 4),
            extra_vram_gb=round(peak_extra_gb, 4),
        )
        print(
            f"  [{mode:14s}] PointingGame={pg_acc:.4f} | CellHit={cell_acc:.4f} | MeanDrop={np.mean(del_drops):+.3f} "
            f"| H200={sec_per_img*1000:.1f}ms/img | ExtraVRAM={peak_extra_gb:.3f}GB",
            flush=True,
        )

    # Pick the winning mode (highest pointing_game_acc with extra_vram_gb < 1.0)
    best_mode = max(modes, key=lambda k: (report[k]["pointing_game_acc"], report[k]["peak_cell_hit_acc"]))
    print(f"\n>>> Best Gate-G4 Mode: {best_mode} -> {report[best_mode]} <<<", flush=True)

    # 2) Out-of-fold Fold-0 + Fold-1 60-image `过度生成` Deletion-Faithfulness & Visual Contact Sheet Audit!
    import common55 as c
    names, gold, fold, _ = c.load_split()
    loader = LosslessNativeImageLoader(REPO_ROOT)
    m_fx1 = load_pe_tower("results/next55/towers/r55_fixed_s2026/pe_core_large/fold_1_best.pt", dev, half=True)
    op_fx1 = PEAttributionOperator(m_fx1, dev)

    clean_f0 = [loader.load_pil(names[i]).convert("RGB") for i in range(len(names)) if fold[i] != 0 and (gold[i] == 0).all()][:64]
    clean_f1 = [loader.load_pil(names[i]).convert("RGB") for i in range(len(names)) if fold[i] != 1 and (gold[i] == 0).all()][:64]
    if best_mode == "midblock_hier8":
        op_fx.fit_clean_footprint(clean_f0, mode="midblock_4x4")
        op_fx1.fit_clean_footprint(clean_f1, mode="midblock_4x4")
    op_fx.fit_clean_footprint(clean_f0, mode=best_mode)
    op_fx1.fit_clean_footprint(clean_f1, mode=best_mode)

    # Also load semantic ROI cache to measure how often the peak defect box hits anatomy/subtitle/swirl vs background
    sem_npz = np.load(os.path.join(REPO, "results/r55/semantic_roi_meta.npz"), allow_pickle=True)
    sem_map = {str(nm): sem_npz["crops_448_boxes"][i] for i, nm in enumerate(sem_npz["names"])}

    # Collect 20 P2, 20 P1, 20 P0 out-of-fold 过度生成 positives from folds 0..2 (load fold 2 if needed)
    m_fx2 = load_pe_tower("results/next55/towers/r55_fixed_s2026/pe_core_large/fold_2_best.pt", dev, half=True)
    op_fx2 = PEAttributionOperator(m_fx2, dev)
    clean_f2 = [loader.load_pil(names[i]).convert("RGB") for i in range(len(names)) if fold[i] != 2 and (gold[i] == 0).all()][:64]
    if best_mode == "midblock_hier8":
        op_fx2.fit_clean_footprint(clean_f2, mode="midblock_4x4")
    op_fx2.fit_clean_footprint(clean_f2, mode=best_mode)
    ops_by_fold = {0: op_fx, 1: op_fx1, 2: op_fx2}

    picked = {1: [], 2: [], 3: []}
    for i, nm in enumerate(names):
        fk = int(fold[i])
        g4 = int(gold[i, 4])
        if fk in ops_by_fold and g4 in picked and len(picked[g4]) < 20:
            picked[g4].append((i, nm, fk, g4))
    sample_60 = picked[1] + picked[2] + picked[3]

    pos_drop_hits = 0
    vs_random_wins = 0
    thumbs = []
    for idx_row, nm, fk, g4 in sample_60:
        im = loader.load_pil(nm).convert("RGB")
        w, h = im.size
        cal_m, probs = ops_by_fold[fk].calibrated_map(im, mode=best_mode)
        m_og = cal_m[4]
        (px, py), boxes = map_peak_xy_and_boxes(m_og, w, h, crop_sz=448)
        pk_drop = float(m_og.max())
        mean_drop = float(m_og.mean())
        pos_drop_hits += int(pk_drop > 0.05)
        vs_random_wins += int(pk_drop > mean_drop + 0.05)

        # Draw thumbnail for visual inspection contact sheet
        th = im.copy()
        draw = ImageDraw.Draw(th)
        x0, y0, x1, y1 = boxes[0]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=max(4, min(w, h) // 120))
        r_dot = max(6, min(w, h) // 80)
        draw.ellipse([px - r_dot, py - r_dot, px + r_dot, py + r_dot], fill=(255, 230, 0), outline=(0, 0, 0))
        th = th.resize((240, 240), Image.BILINEAR)
        d2 = ImageDraw.Draw(th)
        g_str = {1: "P2", 2: "P1", 3: "P0"}[g4]
        d2.rectangle([0, 0, 240, 22], fill=(0, 0, 0))
        d2.text((4, 4), f"{g_str} dS={pk_drop:+.2f} p={1-probs[4,0]:.2f}", fill=(255, 255, 255))
        thumbs.append(th)

    os.makedirs(os.path.join(REPO, "results/r55/attr_maps"), exist_ok=True)
    sheet = Image.new("RGB", (240 * 10, 240 * 6))
    for k_i, th in enumerate(thumbs[:60]):
        rr, cc = divmod(k_i, 10)
        sheet.paste(th, (cc * 240, rr * 240))
    sheet_path = os.path.join(REPO, "results/r55/attr_maps/gate_g4_overgen60_sheet.jpg")
    sheet.save(sheet_path, quality=90)

    report["best_mode"] = best_mode
    report["overgen60_audit"] = dict(
        n=len(sample_60),
        positive_causal_drop_rate=round(pos_drop_hits / max(1, len(sample_60)), 4),
        beats_random_cell_rate=round(vs_random_wins / max(1, len(sample_60)), 4),
        contact_sheet=sheet_path,
    )
    out_json = os.path.join(REPO, "results/r55/attr_maps/gate_g4_report.json")
    json.dump(report, open(out_json, "w"), indent=2)
    print(f"[Gate G4] Overgen-60 causal drop={pos_drop_hits}/{len(sample_60)} | beats-random={vs_random_wins}/{len(sample_60)} | sheet -> {sheet_path}", flush=True)


def extract_oof_fold(fold_k: int, gpu: int, mode: str) -> None:
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(dev)
    import common55 as c

    names, gold, fold, _ = c.load_split()
    loader = LosslessNativeImageLoader(REPO_ROOT)

    # Load precomputed base physics (36-d) + semantic ROI boxes (4 boxes)
    de_npz = np.load(os.path.join(REPO, "data/qingyu_dual_engine_meta.npz"), allow_pickle=True) if os.path.exists(os.path.join(REPO, "data/qingyu_dual_engine_meta.npz")) else None
    sem_npz = np.load(os.path.join(REPO, "results/r55/semantic_roi_meta.npz"), allow_pickle=True)
    sem_idx = {str(nm): i for i, nm in enumerate(sem_npz["names"])}

    m_fx = load_pe_tower(f"results/next55/towers/r55_fixed_s2026/pe_core_large/fold_{fold_k}_best.pt", dev, half=True)
    m_gl = load_pe_tower(f"results/next55/towers/r55_glitch2_s2026/pe_core_large/fold_{fold_k}_best.pt", dev, half=True)
    op_fx = PEAttributionOperator(m_fx, dev)
    op_gl = PEAttributionOperator(m_gl, dev)

    # Fit Clean operator footprint strictly on Clean training rows (`fold != fold_k`)
    tr_clean_ix = [i for i in range(len(names)) if int(fold[i]) != fold_k and (gold[i] == 0).all()][:96]
    clean_pils = [loader.load_pil(names[i]).convert("RGB") for i in tr_clean_ix]
    clean_phys = [sem_npz["phys_vecs"][sem_idx[names[i]]][:36] for i in tr_clean_ix]
    if mode == "midblock_hier8":
        op_fx.fit_clean_footprint(clean_pils, mode="midblock_4x4", phys_vecs=clean_phys)
        op_gl.fit_clean_footprint(clean_pils, mode="midblock_4x4", phys_vecs=clean_phys)
    fp_fx = op_fx.fit_clean_footprint(clean_pils, mode=mode, phys_vecs=clean_phys)
    fp_gl = op_gl.fit_clean_footprint(clean_pils, mode=mode, phys_vecs=clean_phys)

    va_ix = [i for i in range(len(names)) if int(fold[i]) == fold_k]
    G = 32 if "tok32" in mode else (4 if "4x4" in mode else 8)
    va_names = []
    maps_all = np.zeros((len(va_ix), 6, G, G), dtype=np.float32)
    peak_boxes_og = np.zeros((len(va_ix), 2, 4), dtype=np.int32)
    peak_boxes_any = np.zeros((len(va_ix), 2, 4), dtype=np.int32)
    rand_boxes = np.zeros((len(va_ix), 2, 4), dtype=np.int32)
    spat_feats = np.zeros((len(va_ix), 15), dtype=np.float32)

    rng = np.random.default_rng(20260 + fold_k)
    t0 = time.time()
    for j, i in enumerate(va_ix):
        nm = names[i]
        va_names.append(nm)
        im = loader.load_pil(nm).convert("RGB")
        w, h = im.size
        si = sem_idx[nm]
        pv36 = sem_npz["phys_vecs"][si][:36]
        rboxes = sem_npz["crops_448_boxes"][si]

        cm_fx, probs_fx = op_fx.calibrated_map(im, mode=mode, phys_vec=pv36)
        cm_gl, _ = op_gl.calibrated_map(im, mode=mode, phys_vec=pv36)
        cm_fx[3] = cm_gl[3]  # use glitch-immune tower for 花屏 attribution
        maps_all[j] = cm_fx

        _, b_og = map_peak_xy_and_boxes(cm_fx[4], w, h, crop_sz=448)
        # Combined severity-weighted map across localized defects (过锐, 花屏, 过度生成, 虚化突兀)
        w_def = np.array([0.15, 0.15, 0.25, 0.30, 0.45, 0.25], dtype=np.float32)[:, None, None]
        comb_2d = (np.maximum(cm_fx, 0.0) * w_def).sum(axis=0)
        _, b_any = map_peak_xy_and_boxes(comb_2d, w, h, crop_sz=448)
        peak_boxes_og[j] = np.array(b_og, dtype=np.int32)
        peak_boxes_any[j] = np.array(b_any, dtype=np.int32)

        # Deterministic random 1:1 448x448 control boxes for Step 0b Random-Crop Control!
        for bi in range(2):
            rx = int(rng.integers(0, max(1, w - 448 + 1)))
            ry = int(rng.integers(0, max(1, h - 448 + 1)))
            rand_boxes[j, bi] = [rx, ry, min(w, rx + 448), min(h, ry + 448)]

        spat_feats[j] = compute_spatial_evidence_features(cm_fx, rboxes, w, h)
        if (j + 1) % 100 == 0 or (j + 1) == len(va_ix):
            print(f"  [OOF Map F{fold_k}] {j+1}/{len(va_ix)} ({time.time()-t0:.1f}s)", flush=True)

    out_dir = os.path.join(REPO, "results/r55/attr_maps")
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, f"oof_attr_{mode}_fold{fold_k}.npz")
    np.savez_compressed(
        out_npz,
        fold=np.int64(fold_k),
        mode=mode,
        names=np.array(va_names, dtype=object),
        cal_maps=maps_all,
        peak_boxes_og=peak_boxes_og,
        peak_boxes_any=peak_boxes_any,
        rand_boxes=rand_boxes,
        spat_feats=spat_feats,
        clean_footprint_fx=fp_fx,
        clean_footprint_gl=fp_gl,
    )
    print(f"[OOF Map F{fold_k}] DONE -> {out_npz}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-g4", action="store_true")
    ap.add_argument("--extract-oof", action="store_true")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--mode", default="midblock_8x8")
    ap.add_argument("--n-twins", type=int, default=240)
    args = ap.parse_args()

    if args.gate_g4:
        run_gate_g4(gpu=args.gpu, n_twins=args.n_twins)
    elif args.extract_oof:
        extract_oof_fold(fold_k=args.fold, gpu=args.gpu, mode=args.mode)


if __name__ == "__main__":
    main()
