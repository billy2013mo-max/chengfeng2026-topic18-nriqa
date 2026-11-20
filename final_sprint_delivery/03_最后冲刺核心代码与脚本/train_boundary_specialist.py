#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/train_boundary_specialist.py
=====================================
Phase 2 (Stage 2B-V2): 100% Tag-Free, Zero-Leakage All-4-Dim P0/P1/P2 Boundary Native-Tile Specialist Trainer.

Synthesizes the three empirically proven mechanisms from Phase 1, Stage 2A, Stage 2B-v1, and V16:
  1. 1:1 Native Dynamic Tiles + Explicit Edge-vs-Texture Contrast Residuals (`r_edge_glob`, `r_edge_tex`):
     - Deterministic `<= 12` `448x448` native 1:1 tiles + `1` global `448x448` thumbnail + hflip-invariant
       spatial coordinates `(y_c, 2*|x_c-0.5|, h_f, w_f)` + hflip-invariant high-frequency edge energy `e_t`.
     - Explicit `r_edge_glob = edge_peak_tok - global_tok` and `r_edge_tex = edge_peak_tok - median_tex_tok`
       contrast tokens injected directly into `PerDefectRoutedTileHead` (exact V16 physical microscope cue!).
  2. Multi-Backbone Support with Lossless `--share-weights` Compatibility (`0.27-0.62 GB` incremental VRAM on 4090):
     - `pe_core_large` (0.3B, top-9 unfrozen, `0.27 GB` shared VRAM)
     - `cradio_v4_h` (0.67B, top-8 unfrozen, `~0.35 GB` shared VRAM)
     - `dual_pe_cradio` (joint `pe_core_large + cradio_v4_h` native-tile fusion, `~0.62 GB` shared VRAM!)
     - `dinov3_vitl16` (0.33B, top-8 unfrozen), `pe_core_g14` (1.8B, top-10 unfrozen)
     - Bottom blocks stay 100% frozen (bit-identical for `--share-weights`, zero autograd activation overhead).
  3. V2 Calibrated Boundary Loss (`All4BoundaryV2Loss` — Zero Overconfidence, Razor-Sharp `P0/P1/P2`):
     - Sharp class-weighted Multinomial CE (`cw4=[1.0, 1.20, 1.65, 1.50]`, `ls=0.02`, `1.35x` all-clean guard)
     - Cumulative 3-threshold Ordinal BCE (`>=P2, >=P1, ==P0` with `1.25x` on `>=P1` & `2.5x` clean P0 guard)
     - Boundary-weighted Squared-EMD / RPS (`b_w=[1.0, 1.85, 1.55]`, `dim_w=[1.25, 1.35, 1.30, 1.00, 1.35, 1.15]`)
     - Direct Bounded Hinge Separation Margin (`P2 vs P1`, `P1 vs P2/P0`, `P0 vs P1` — ZERO gradient once margin met!)
     - Cross-Video Conditional Multi-Boundary XBM (`M=512`, `cond_b1=True`, `group_i != group_j`)
     - Leak-Free Cosine-Annealed Nested KD (`--kd-npz results/r55/nested_kd_teacher.npz`, `lambda_kd: 0.25 -> 0.0`,
       hard-asserting `max_val_abs == 0.0` on val fold `k`) + optional Tag-Free real `P1/P0` oversampling.
  4. Strict Compliance:
     - 100% Tag-Free (`data/qingyu_splits/qingyu_8fold.csv`), fixed last-2 epoch SWA, 2x h-flip TTA,
       and hard `verify_n1_batch_invariance` assertion (`N=1 == N=Batch`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, roc_auc_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (
    os.path.join(REPO_ROOT, "code"),
    os.path.join(REPO_ROOT, "code", "next55"),
    os.path.join(REPO_ROOT, "code", "r55"),
    os.path.join(REPO_ROOT, "code", "finetune"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from qingyu.config_and_paths import (
    DIMS,
    FULL_DIMS,
    P0_ONLY_DIMS,
    resolve_model_path,
    seed_hf_dynamic_modules,
)
from qingyu.backbones_qingyu import IntermediateThreeLayerGatedTap, MaskedTokenGatedABMILPool
from qingyu.unimodal_xbm_losses import TipAdapterPrototypeBank, hash_group_id_int64
from qingyu.certainty_gated_stacker import compute_ece_6x4, simplex_6x4_to_p012
from abmil_layers import TokenGatedABMILPool
from train_iv6b_native_tiles import (
    TILE_SIZE,
    FastShmImageStore,
    load_canonical_8fold,
    load_leakfree_nested_kd,
    pad_to_min_tile_reflect,
    select_deterministic_tiles,
    tile_starts,
)

BALANCED_DIM_W = (1.25, 1.35, 1.30, 1.00, 1.35, 1.15)


# ---------------------------------------------------------------------------
# 1. Dataset & Collate (100% Tag-Free, Native 1:1 Tiles + Coords + Edge Energy)
# ---------------------------------------------------------------------------

def load_semantic_phys_map(npz_path: str) -> Tuple[Dict[str, np.ndarray], int]:
    full_p = npz_path if os.path.isabs(npz_path) else os.path.join(REPO_ROOT, npz_path)
    z = np.load(full_p, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    vecs = np.asarray(z["phys_vecs"], dtype=np.float32)
    vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)
    return {nm: vecs[i] for i, nm in enumerate(names)}, int(vecs.shape[1])


def extract_native_tiles_coords_energy(
    arr: np.ndarray,
    tile_cap: int = 12,
    rand_sample: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      global_u8:   (448, 448, 3) uint8
      tiles_u8:    (T, 448, 448, 3) uint8
      tile_coords: (T, 4) float32 -> (y_c, 2*|x_c - 0.5|, h_frac, w_frac) [100% hflip-invariant]
      tile_energy: (T,) float32 -> symmetric gradient + Laplacian high-freq energy [100% hflip-invariant]
    """
    H_orig, W_orig = arr.shape[:2]
    g_pil = Image.fromarray(arr, mode="RGB").resize((TILE_SIZE, TILE_SIZE), Image.BICUBIC)
    global_u8 = np.asarray(g_pil, dtype=np.uint8).copy()

    arr_pad = pad_to_min_tile_reflect(arr, TILE_SIZE)
    H_p, W_p = arr_pad.shape[:2]
    boxes = [(y, x) for y in tile_starts(H_p, TILE_SIZE) for x in tile_starts(W_p, TILE_SIZE)]

    if len(boxes) > tile_cap:
        if rand_sample:
            pick = sorted(random.sample(range(len(boxes)), tile_cap))
            boxes = [boxes[i] for i in pick]
        else:
            boxes = select_deterministic_tiles(arr_pad, boxes, tile_cap)

    tiles_list = [arr_pad[y : y + TILE_SIZE, x : x + TILE_SIZE] for y, x in boxes]
    tiles_u8 = np.stack(tiles_list, axis=0).copy()

    h_frac = min(1.0, float(TILE_SIZE) / max(1.0, float(H_orig)))
    w_frac = min(1.0, float(TILE_SIZE) / max(1.0, float(W_orig)))
    coords = np.zeros((len(boxes), 4), dtype=np.float32)
    energies = np.zeros((len(boxes),), dtype=np.float32)
    for i, ((y, x), tile) in enumerate(zip(boxes, tiles_list)):
        yc = (y + 0.5 * TILE_SIZE) / max(float(TILE_SIZE), float(H_p))
        xc = (x + 0.5 * TILE_SIZE) / max(float(TILE_SIZE), float(W_p))
        coords[i, 0] = float(np.clip(yc, 0.0, 1.0))
        coords[i, 1] = float(np.clip(abs(xc - 0.5) * 2.0, 0.0, 1.0))
        coords[i, 2] = h_frac
        coords[i, 3] = w_frac
        sub = tile[::4, ::4, 1].astype(np.float32)
        gx = float(np.abs(sub[:, 1:] - sub[:, :-1]).mean())
        gy = float(np.abs(sub[1:, :] - sub[:-1, :]).mean())
        lx = float(np.abs(2.0 * sub[:, 1:-1] - sub[:, :-2] - sub[:, 2:]).mean())
        ly = float(np.abs(2.0 * sub[1:-1, :] - sub[:-2, :] - sub[2:, :]).mean())
        energies[i] = round(gx + gy + 0.85 * (lx + ly), 5)
    return global_u8, tiles_u8, coords, energies


class BoundaryNativeTileDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        img_store: FastShmImageStore,
        phys_map: Dict[str, np.ndarray],
        tile_cap: int = 12,
        train: bool = False,
        hflip_tta: bool = False,
        train_rand_tile_prob: float = 0.40,
        teacher_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.rows = list(rows)
        self.img_store = img_store
        self.phys_map = phys_map
        self.tile_cap = int(tile_cap)
        self.train = bool(train)
        self.hflip_tta = bool(hflip_tta)
        self.train_rand_tile_prob = float(train_rand_tile_prob)
        self.teacher_map = teacher_map or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        r = self.rows[idx]
        nm = str(r["name"])
        arr = self.img_store.load_rgb_u8(nm)

        do_flip = self.hflip_tta or (self.train and random.random() < 0.5)
        if do_flip:
            arr = np.ascontiguousarray(arr[:, ::-1, :])

        rand_sample = self.train and (random.random() < self.train_rand_tile_prob)
        global_u8, tiles_u8, coords, energies = extract_native_tiles_coords_energy(
            arr, tile_cap=self.tile_cap, rand_sample=rand_sample
        )

        bright = 1.0 + random.uniform(-0.04, 0.04) if (self.train and random.random() < 0.30) else 1.0
        contr = 1.0 + random.uniform(-0.04, 0.04) if (self.train and random.random() < 0.30) else 1.0

        t_prob = self.teacher_map.get(nm, None)
        t_tensor = torch.from_numpy(t_prob.copy()) if t_prob is not None else torch.zeros((6, 4), dtype=torch.float32)

        return {
            "global_u8": torch.from_numpy(global_u8),
            "tiles_u8": torch.from_numpy(tiles_u8),
            "tile_coords": torch.from_numpy(coords),
            "tile_energy": torch.from_numpy(energies),
            "photo_bc": (bright, contr),
            "phys": torch.from_numpy(self.phys_map[nm].copy()),
            "grades": torch.tensor(r["grades"], dtype=torch.long),
            "ghash": hash_group_id_int64(str(r.get("group_id", nm))),
            "name": nm,
            "idx_in_ds": idx,
            "teacher_6x4": t_tensor,
            "has_teacher": 1 if t_prob is not None else 0,
        }


def collate_boundary_tiles(batch: List[Dict[str, object]]) -> Dict[str, object]:
    B = len(batch)
    Tm = max(int(b["tiles_u8"].shape[0]) for b in batch)
    tiles = torch.zeros((B, Tm, TILE_SIZE, TILE_SIZE, 3), dtype=torch.uint8)
    coords = torch.zeros((B, Tm, 4), dtype=torch.float32)
    energy = torch.zeros((B, Tm), dtype=torch.float32)
    mask = torch.zeros((B, Tm), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = int(b["tiles_u8"].shape[0])
        tiles[i, :n] = b["tiles_u8"]
        coords[i, :n] = b["tile_coords"]
        energy[i, :n] = b["tile_energy"]
        mask[i, :n] = True
    return {
        "global_u8": torch.stack([b["global_u8"] for b in batch], dim=0),
        "tiles_u8": tiles,
        "tile_coords": coords,
        "tile_energy": energy,
        "tile_mask": mask,
        "photo_bc": torch.tensor([b["photo_bc"] for b in batch], dtype=torch.float32),
        "phys": torch.stack([b["phys"] for b in batch], dim=0),
        "grades": torch.stack([b["grades"] for b in batch], dim=0),
        "ghash": torch.tensor([b["ghash"] for b in batch], dtype=torch.long),
        "names": [b["name"] for b in batch],
        "idx_in_ds": torch.tensor([b["idx_in_ds"] for b in batch], dtype=torch.long),
        "teacher_6x4": torch.stack([b["teacher_6x4"] for b in batch], dim=0),
        "has_teacher": torch.tensor([b["has_teacher"] for b in batch], dtype=torch.long),
    }


def apply_tagfree_p1p0_oversample(tr_rows: Sequence[Dict[str, object]], seed: int = 2026) -> List[Dict[str, object]]:
    """
    100% Tag-Free real P1/P0 boundary oversampling (zero P2 inflation, zero synthetic data):
      - Repeats 50% of real frames containing >=1 graded P1/P0 (`grade >= 2` on 模糊/纹理/过锐/过生),
        plus an equal share of real all-clean frames to preserve clean/positive balance.
    """
    rng = random.Random(seed)
    p1p0_rows, clean_rows = [], []
    for r in tr_rows:
        g = r["grades"]
        if any(int(g[d]) >= 2 for d in FULL_DIMS):
            p1p0_rows.append(r)
        elif sum(int(x) for x in g) == 0:
            clean_rows.append(r)
    extra_pos = rng.sample(p1p0_rows, k=len(p1p0_rows) // 2)
    extra_cln = rng.sample(clean_rows, k=min(len(clean_rows), len(extra_pos) // 2))
    out = list(tr_rows) + extra_pos + extra_cln
    return out


# ---------------------------------------------------------------------------
# 2. Per-Defect Routed Native-Tile Head (with V16 Contrast Residuals & Coverage)
# ---------------------------------------------------------------------------

class PerDefectRoutedTileHead(nn.Module):
    def __init__(
        self,
        single_view_dim: int,
        phys_dim: int = 48,
        hidden_dim: int = 512,
        proj_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.22,
        max_prototypes: int = 128,
    ):
        super().__init__()
        self.single_view_dim = int(single_view_dim)
        self.phys_dim = int(phys_dim)
        self.hidden_dim = int(hidden_dim)
        self.proj_dim = int(proj_dim)

        self.view_proj = nn.Sequential(
            nn.LayerNorm(single_view_dim),
            nn.Dropout(dropout),
            nn.Linear(single_view_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.global_role_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.tile_role_emb = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.contrast_role_emb = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)
        self.contrast_norm = nn.LayerNorm(hidden_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        if phys_dim > 0:
            self.register_buffer("phys_mu", torch.zeros(1, phys_dim, dtype=torch.float32))
            self.register_buffer("phys_sd", torch.ones(1, phys_dim, dtype=torch.float32))
            self.phys_token_mlp = nn.Sequential(
                nn.LayerNorm(phys_dim),
                nn.Linear(phys_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.phys_token_mlp = None

        # Per-tile defect intensity probe -> 6 order-statistic coverage features per defect
        self.tile_probe = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
        )
        self.cov_proj = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.cov_proj[-1].weight)
        nn.init.zeros_(self.cov_proj[-1].bias)
        self.cov_gate = nn.Parameter(torch.full((1, 6, 1), 0.25, dtype=torch.float32))
        self.norm_cov = nn.LayerNorm(hidden_dim)

        # 6 defect queries + 1 all-clean gate query
        self.queries = nn.Parameter(torch.randn(1, 7, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout * 0.5, batch_first=True
        )
        self.norm_c1 = nn.LayerNorm(hidden_dim)
        self.ffn_c1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_c2 = nn.LayerNorm(hidden_dim)

        # 2-layer Cross-Defect Self-Attention
        self.self_attn1 = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout * 0.5, batch_first=True
        )
        self.norm_s1 = nn.LayerNorm(hidden_dim)
        self.ffn_s1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_s1_ffn = nn.LayerNorm(hidden_dim)

        self.self_attn2 = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout * 0.5, batch_first=True
        )
        self.norm_s2 = nn.LayerNorm(hidden_dim)

        # Unconstrained per-defect 4-class linear heads + weak soft unimodal anchor
        self.grade_heads = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim, 4)) for _ in range(6)
        ])
        self.center_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(6)])
        self.log_kappa = nn.Parameter(torch.full((6,), -1.2, dtype=torch.float32))
        self.register_buffer("grade_grid", torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32))

        self.clean_head = nn.Linear(hidden_dim, 1)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.proto_bank = TipAdapterPrototypeBank(proj_dim=proj_dim, max_prototypes=max_prototypes)

    @torch.no_grad()
    def set_physics_scaler(self, mu_np: np.ndarray, sd_np: np.ndarray) -> None:
        if self.phys_token_mlp is not None:
            dev = self.phys_mu.device
            self.phys_mu.copy_(torch.from_numpy(np.asarray(mu_np, dtype=np.float32)).view(1, -1).to(dev))
            self.phys_sd.copy_(torch.from_numpy(np.asarray(sd_np, dtype=np.float32) + 1e-5).view(1, -1).to(dev))

    def _tile_coverage_stats(self, tile_toks: torch.Tensor, tile_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = tile_toks.shape
        a = self.tile_probe(tile_toks)  # [B, T, 6]
        mf = tile_mask.float().unsqueeze(-1)
        cnt = mf.sum(dim=1).clamp(min=1.0)

        a_mean = (a * mf).sum(dim=1) / cnt
        diff2 = ((a - a_mean.unsqueeze(1)) ** 2) * mf
        a_std = torch.sqrt(diff2.sum(dim=1) / cnt + 1e-5)

        neg_inf = -1e4
        a_masked = a.masked_fill(~tile_mask.unsqueeze(-1), neg_inf)
        a_max = a_masked.amax(dim=1)
        if T >= 2:
            top2_vals = torch.topk(a_masked, k=2, dim=1).values
            has2 = (cnt >= 1.5).float()
            a_top2 = has2 * (0.5 * (top2_vals[:, 0] + top2_vals[:, 1])) + (1.0 - has2) * a_max
        else:
            a_top2 = a_max

        a_frac = (torch.sigmoid(a) * mf).sum(dim=1) / cnt
        log_t = (torch.log1p(cnt) / math.log(13.0)).expand(B, 6)
        return torch.stack([a_mean, a_max, a_top2, a_std, a_frac, log_t], dim=-1)

    def _contrast_residual_tokens(
        self,
        g_raw: torch.Tensor,
        t_raw: torch.Tensor,
        tile_mask: torch.Tensor,
        tile_energy: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        V16-inspired 1:1 native edge-vs-global & edge-vs-interior-texture contrast tokens `[B, 2, H]`.
        Computed strictly over each image's valid tiles (`nv = tile_mask[b].sum()`), guaranteeing exact
        `N=1 == N=Batch` and horizontal-flip invariance.
        """
        B = g_raw.shape[0]
        c_list = []
        for b in range(B):
            nv = max(1, int(tile_mask[b].sum().item()))
            tb = t_raw[b, :nv]  # [nv, H]
            if tile_energy is not None and nv >= 2:
                order = torch.argsort(tile_energy[b, :nv])
                edge_tok = 0.5 * (tb[order[-1]] + tb[order[-2]])
                tex_tok = tb[order[nv // 2]]
            else:
                edge_tok = tb.amax(dim=0)
                tex_tok = tb.mean(dim=0)
            r_eg = edge_tok - g_raw[b]
            r_et = edge_tok - tex_tok
            c_list.append(torch.stack([r_eg, r_et], dim=0))
        c_toks = torch.stack(c_list, dim=0)  # [B, 2, H]
        return self.contrast_norm(c_toks) + self.contrast_role_emb

    def forward(
        self,
        global_feat: torch.Tensor,
        tile_feats: torch.Tensor,
        tile_mask: torch.Tensor,
        tile_coords: torch.Tensor,
        tile_energy: Optional[torch.Tensor] = None,
        phys_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = tile_feats.shape
        valid_b = tile_mask.bool()

        g_raw = self.view_proj(global_feat.float())  # [B, H]
        g_tok = g_raw.unsqueeze(1) + self.global_role_emb  # [B, 1, H]

        t_raw = g_tok.new_zeros((B, T, self.hidden_dim))
        t_tok = g_tok.new_zeros((B, T, self.hidden_dim))
        if valid_b.any():
            vt_raw = self.view_proj(tile_feats[valid_b].float())
            vt_pos = self.coord_mlp(tile_coords[valid_b].float())
            t_raw[valid_b] = vt_raw
            t_tok[valid_b] = vt_raw + self.tile_role_emb.squeeze(0).squeeze(0) + vt_pos

        c_toks = self._contrast_residual_tokens(g_raw, t_raw, valid_b, tile_energy)  # [B, 2, H]
        cov_stats = self._tile_coverage_stats(t_tok, valid_b)  # [B, 6, 6]

        tokens_list = [g_tok, t_tok, c_toks]
        kpm_list = [
            torch.zeros((B, 1), dtype=torch.bool, device=g_tok.device),
            ~valid_b,
            torch.zeros((B, 2), dtype=torch.bool, device=g_tok.device),
        ]
        if self.phys_token_mlp is not None and phys_feat is not None:
            pn = torch.clamp((phys_feat.float() - self.phys_mu) / self.phys_sd, -5.0, 5.0)
            ptok = self.phys_token_mlp(pn).unsqueeze(1)
            tokens_list.append(ptok)
            kpm_list.append(torch.zeros((B, 1), dtype=torch.bool, device=g_tok.device))

        tokens = torch.cat(tokens_list, dim=1)
        key_padding_mask = torch.cat(kpm_list, dim=1)

        q = self.queries.expand(B, -1, -1)
        qc, _ = self.cross_attn(q, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
        q = self.norm_c1(q + qc)
        q = self.norm_c2(q + self.ffn_c1(q))

        q_def = self.norm_cov(q[:, :6, :] + self.cov_gate * self.cov_proj(cov_stats))
        q = torch.cat([q_def, q[:, 6:, :]], dim=1)

        qs1, _ = self.self_attn1(q, q, q, need_weights=False)
        q = self.norm_s1(q + qs1)
        q = self.norm_s1_ffn(q + self.ffn_s1(q))

        qs2, _ = self.self_attn2(q, q, q, need_weights=False)
        q = self.norm_s2(q + qs2)

        mf = valid_b.float().unsqueeze(-1)
        t_mean = (t_tok * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
        z_cat = torch.cat([q[:, :6, :].mean(dim=1), g_tok.squeeze(1), t_mean], dim=-1)
        z_norm = F.normalize(self.proj_head(z_cat), dim=-1)
        proto_delta = self.proto_bank(z_norm)

        kappa = F.softplus(self.log_kappa) * 0.20
        grid = self.grade_grid.view(1, 4)

        logits_6x4 = g_tok.new_full((B, 6, 4), -1e4, dtype=torch.float32)
        probs_6x4 = torch.zeros_like(logits_6x4)

        for d in range(6):
            qd = q[:, d, :]
            raw_z = self.grade_heads[d](qd)
            c_est = 3.0 * torch.sigmoid(self.center_heads[d](qd))
            soft_prior = -kappa[d] * ((grid - c_est) ** 2)
            z = raw_z + soft_prior + proto_delta[:, d, :]
            if d in FULL_DIMS:
                z = z - z.mean(dim=-1, keepdim=True)
                logits_6x4[:, d, :] = z
                probs_6x4[:, d, :] = F.softmax(z, dim=-1)
            else:
                z03 = z[:, [0, 3]]
                z03 = z03 - z03.mean(dim=-1, keepdim=True)
                logits_6x4[:, d, 0] = z03[:, 0]
                logits_6x4[:, d, 3] = z03[:, 1]
                sm2 = F.softmax(z03, dim=-1)
                probs_6x4[:, d, 0] = sm2[:, 0]
                probs_6x4[:, d, 3] = sm2[:, 1]

        expected_s = (probs_6x4 * grid.view(1, 1, 4)).sum(dim=-1)
        clean_logit = self.clean_head(q[:, 6, :]).squeeze(-1)
        return {
            "logits_6x4": logits_6x4,
            "probs_6x4": probs_6x4,
            "severity_s": expected_s,
            "clean_logit": clean_logit,
            "z_norm": z_norm,
        }


# ---------------------------------------------------------------------------
# 3. Backbone Builders (`pe_core_large`, `cradio_v4_h`, `dual_pe_cradio`, `dinov3_vitl16`, `pe_core_g14`)
# ---------------------------------------------------------------------------

def _freeze_except_top_blocks(
    backbone: nn.Module,
    blocks: Sequence[nn.Module],
    unfreeze_last_k: int,
    extra_unfreeze_prefixes: Sequence[str],
) -> None:
    for p in backbone.parameters():
        p.requires_grad = False
    k_eff = min(len(blocks), max(1, int(unfreeze_last_k)))
    for blk in list(blocks)[-k_eff:]:
        for p in blk.parameters():
            p.requires_grad = True
    for n, p in backbone.named_parameters():
        if any(n.startswith(pref) for pref in extra_unfreeze_prefixes):
            p.requires_grad = True
    for p in backbone.parameters():
        if not p.requires_grad and p.is_floating_point():
            p.data = p.data.to(torch.bfloat16)


def _build_pe_core_large(unfreeze_last_k: int):
    import timm
    from safetensors.torch import load_file

    p_cand = resolve_model_path("pe_core_large")
    w_path = p_cand if os.path.isfile(p_cand) else os.path.join(p_cand, "model.safetensors")
    m = timm.create_model("vit_pe_core_large_patch14_336", pretrained=False, num_classes=0, dynamic_img_size=True)
    m.load_state_dict(load_file(w_path), strict=False)
    dc = timm.data.resolve_data_config({}, model=m)
    m.tap3 = IntermediateThreeLayerGatedTap(m.blocks, m.embed_dim, tap_indices=[15, 19, 22]).float()
    m.abmil = TokenGatedABMILPool(m.embed_dim).float()
    _freeze_except_top_blocks(
        m, m.blocks, unfreeze_last_k,
        extra_unfreeze_prefixes=("norm.", "fc_norm.", "attn_pool.", "tap3.", "abmil."),
    )
    mean = torch.tensor(dc["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(dc["std"], dtype=torch.float32).view(1, 3, 1, 1)

    def fwd_pe(x_01: torch.Tensor) -> torch.Tensor:
        x = (x_01 - mean.to(x_01.device)) / std.to(x_01.device)
        raw_t = m.forward_features(x)
        amp_dev = "cuda" if raw_t.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            t = m.tap3.fuse(raw_t.float())
            n_pref = getattr(m, "num_prefix_tokens", 1)
            if n_pref > 0:
                pt = torch.cat([t[:, :1], t[:, n_pref:]], dim=1)
                return torch.cat([t[:, 0], m.abmil(pt, has_cls=True)], dim=1)
            return torch.cat([t.mean(dim=1), m.abmil(t, has_cls=False)], dim=1)

    return m, fwd_pe, int(m.embed_dim * 2)


def _build_cradio(unfreeze_last_k: int):
    from transformers import AutoModel

    repo = resolve_model_path("cradio_v4_h")
    seed_hf_dynamic_modules(repo)
    m = AutoModel.from_pretrained(repo, trust_remote_code=True)
    blocks = m.radio_model.model.blocks
    m.tap3 = IntermediateThreeLayerGatedTap(
        blocks, 1280, tap_indices=[max(0, len(blocks) - 10), max(0, len(blocks) - 5), len(blocks) - 2]
    ).float()
    m.abmil = TokenGatedABMILPool(1280).float()
    _freeze_except_top_blocks(
        m, blocks, unfreeze_last_k,
        extra_unfreeze_prefixes=(
            "radio_model.model.norm.",
            "radio_model.model.fc_norm.",
            "tap3.",
            "abmil.",
        ),
    )

    def fwd_radio(x_01: torch.Tensor) -> torch.Tensor:
        summ, feat = m(x_01)
        amp_dev = "cuda" if feat.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            feat_f = m.tap3.fuse(feat.float())
            return torch.cat([summ.float(), m.abmil(feat_f, has_cls=False)], dim=1)

    return m, fwd_radio, int(2560 + 1280)


def _build_dinov3(unfreeze_last_k: int):
    import timm
    from safetensors.torch import load_file

    p_cand = resolve_model_path("dinov3_vitl16")
    w_path = p_cand if os.path.isfile(p_cand) else os.path.join(p_cand, "model.safetensors")
    m = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0, dynamic_img_size=True)
    m.load_state_dict(load_file(w_path), strict=True)
    dc = timm.data.resolve_data_config({}, model=m)
    m.tap3 = IntermediateThreeLayerGatedTap(m.blocks, m.embed_dim, tap_indices=[15, 19, 22]).float()
    m.abmil = TokenGatedABMILPool(m.embed_dim).float()
    _freeze_except_top_blocks(
        m, m.blocks, unfreeze_last_k,
        extra_unfreeze_prefixes=("norm.", "fc_norm.", "tap3.", "abmil."),
    )
    mean = torch.tensor(dc["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(dc["std"], dtype=torch.float32).view(1, 3, 1, 1)

    def fwd_dino(x_01: torch.Tensor) -> torch.Tensor:
        x = (x_01 - mean.to(x_01.device)) / std.to(x_01.device)
        raw_t = m.forward_features(x)
        amp_dev = "cuda" if raw_t.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            t = m.tap3.fuse(raw_t.float())
            n_pref = getattr(m, "num_prefix_tokens", 1)
            pt = torch.cat([t[:, :1], t[:, n_pref:]], dim=1)
            return torch.cat([t[:, 0], m.abmil(pt, has_cls=True)], dim=1)

    return m, fwd_dino, int(m.embed_dim * 2)


def _build_pe_core_g14(unfreeze_last_k: int):
    import timm
    from safetensors.torch import load_file

    shm_w = "/dev/shm/pe_core_g14_bf16.safetensors"
    nfs_w = os.path.join(REPO_ROOT, "models", "pe-core-g14", "model.safetensors")
    w_path = shm_w if os.path.exists(shm_w) else nfs_w
    m = timm.create_model("vit_pe_core_gigantic_patch14_448", pretrained=False, num_classes=0, dynamic_img_size=True)
    sd = load_file(w_path)
    miss, _ = m.load_state_dict(sd, strict=False)
    del sd
    assert len(miss) == 0, f"PE-Core-G14 missing keys: {miss[:5]}"
    hid = int(m.embed_dim)
    m.tap3 = IntermediateThreeLayerGatedTap(m.blocks, hid, tap_indices=[33, 41, 48]).float()
    m.masked_abmil = MaskedTokenGatedABMILPool(hid).float()
    _freeze_except_top_blocks(
        m, m.blocks, unfreeze_last_k,
        extra_unfreeze_prefixes=("norm.", "fc_norm.", "attn_pool.", "tap3.", "masked_abmil."),
    )
    mean = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32).view(1, 3, 1, 1)

    def fwd_g14(x_01: torch.Tensor) -> torch.Tensor:
        x = (x_01 - mean.to(x_01.device)) / std.to(x_01.device)
        raw_t = m.forward_features(x)
        map_feat = m.forward_head(raw_t, pre_logits=True).float()
        amp_dev = "cuda" if raw_t.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            t = m.tap3.fuse(raw_t.float())
            return torch.cat([map_feat, m.masked_abmil(t, mask=None)], dim=1)

    return m, fwd_g14, int(hid * 2)


class BoundaryTileTower(nn.Module):
    def __init__(
        self,
        backbone_key: str,
        phys_dim: int = 48,
        unfreeze_last_k: int = -1,
        view_chunk: int = 64,
        dropout: float = 0.22,
    ):
        super().__init__()
        self.backbone_key = str(backbone_key)
        self.view_chunk = int(view_chunk)

        if backbone_key == "pe_core_large":
            k_def = 9 if unfreeze_last_k <= 0 else unfreeze_last_k
            m, fn, dim = _build_pe_core_large(k_def)
            self.backbone = m
            self._encode_single_view = fn
            single_dim, hidden_dim = dim, 512
        elif backbone_key == "cradio_v4_h":
            k_def = 8 if unfreeze_last_k <= 0 else unfreeze_last_k
            m, fn, dim = _build_cradio(k_def)
            self.backbone = m
            self._encode_single_view = fn
            single_dim, hidden_dim = dim, 640
        elif backbone_key == "dual_pe_cradio":
            k_pe = 8 if unfreeze_last_k <= 0 else unfreeze_last_k
            k_cr = 6 if unfreeze_last_k <= 0 else max(4, unfreeze_last_k - 2)
            m_pe, fn_pe, d_pe = _build_pe_core_large(k_pe)
            m_cr, fn_cr, d_cr = _build_cradio(k_cr)
            self.backbone = nn.ModuleDict({"pe": m_pe, "cradio": m_cr})

            def fn_dual(x_01: torch.Tensor) -> torch.Tensor:
                return torch.cat([fn_pe(x_01), fn_cr(x_01)], dim=1)

            self._encode_single_view = fn_dual
            k_def = k_pe
            single_dim, hidden_dim = d_pe + d_cr, 640
        elif backbone_key == "dinov3_vitl16":
            k_def = 8 if unfreeze_last_k <= 0 else unfreeze_last_k
            m, fn, dim = _build_dinov3(k_def)
            self.backbone = m
            self._encode_single_view = fn
            single_dim, hidden_dim = dim, 512
        elif backbone_key == "pe_core_g14":
            k_def = 10 if unfreeze_last_k <= 0 else unfreeze_last_k
            m, fn, dim = _build_pe_core_g14(k_def)
            self.backbone = m
            self._encode_single_view = fn
            single_dim, hidden_dim = dim, 640
        else:
            raise ValueError(f"Unsupported backbone_key: {backbone_key}")

        self.unfreeze_last_k = int(k_def)
        self.single_view_dim = int(single_dim)
        self.head = PerDefectRoutedTileHead(
            single_view_dim=single_dim,
            phys_dim=phys_dim,
            hidden_dim=hidden_dim,
            proj_dim=512,
            num_heads=8,
            dropout=dropout,
            max_prototypes=128,
        ).float()

    def _prep_u8_01(self, x_u8: torch.Tensor, photo_bc: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x_u8.permute(0, 3, 1, 2).contiguous().float().mul_(1.0 / 255.0)
        if photo_bc is not None:
            b = photo_bc[:, 0].view(-1, 1, 1, 1)
            c = photo_bc[:, 1].view(-1, 1, 1, 1)
            x = (x * b - 0.5) * c + 0.5
            x = x.clamp_(0.0, 1.0)
        return x

    def _encode_chunked(self, x_01: torch.Tensor) -> torch.Tensor:
        N = x_01.shape[0]
        if N <= self.view_chunk:
            return self._encode_single_view(x_01)
        return torch.cat(
            [self._encode_single_view(x_01[s : s + self.view_chunk]) for s in range(0, N, self.view_chunk)],
            dim=0,
        )

    def forward(
        self,
        global_u8: torch.Tensor,
        tiles_u8: torch.Tensor,
        tile_mask: torch.Tensor,
        tile_coords: torch.Tensor,
        tile_energy: Optional[torch.Tensor] = None,
        phys_feat: Optional[torch.Tensor] = None,
        photo_bc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T = tiles_u8.shape[:2]
        valid = tile_mask.bool()

        g_01 = self._prep_u8_01(global_u8, photo_bc=photo_bc)
        bc_tiles = photo_bc.unsqueeze(1).expand(B, T, 2)[valid] if photo_bc is not None else None
        t_01 = self._prep_u8_01(tiles_u8[valid], photo_bc=bc_tiles)
        all_views = torch.cat([g_01, t_01], dim=0)

        emb_all = self._encode_chunked(all_views)
        g_feat = emb_all[:B]
        t_flat = emb_all[B:]
        t_feats = t_flat.new_zeros((B, T, t_flat.shape[-1]))
        t_feats[valid] = t_flat

        amp_dev = "cuda" if g_feat.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            return self.head(
                global_feat=g_feat.float(),
                tile_feats=t_feats.float(),
                tile_mask=valid,
                tile_coords=tile_coords.float(),
                tile_energy=tile_energy.float() if tile_energy is not None else None,
                phys_feat=phys_feat.float() if phys_feat is not None else None,
            )

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        keep_keys = set()
        for n, p in self.named_parameters():
            if p.requires_grad:
                keep_keys.add(n)
        for n, _ in self.named_buffers():
            if any(tok in n for tok in ("head.", "tap3.", "abmil.", "masked_abmil.")):
                keep_keys.add(n)
        sd = self.state_dict()
        return {k: sd[k].detach().cpu().clone() for k in sorted(keep_keys) if k in sd}


# ---------------------------------------------------------------------------
# 4. V2 Calibrated Boundary Loss (Hinge + CumBCE + Weighted-EMD + XBM + Annealed KD)
# ---------------------------------------------------------------------------

class All4BoundaryMarginXBMLoss(nn.Module):
    def __init__(self, capacity: int = 512, sigma0: float = 1.0, dim_weights: Sequence[float] = BALANCED_DIM_W):
        super().__init__()
        self.capacity = int(capacity)
        self.sigma0 = float(sigma0)
        self.dim_weights = tuple(float(w) for w in dim_weights)
        self.register_buffer("mem_b", torch.zeros(capacity, 6, 3, dtype=torch.float32))
        self.register_buffer("mem_y", torch.zeros(capacity, 6, dtype=torch.long))
        self.register_buffer("mem_g", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _boundary_scores(logits_6x4: torch.Tensor) -> torch.Tensor:
        B = logits_6x4.shape[0]
        out = logits_6x4.new_zeros((B, 6, 3))
        for d in range(6):
            z = logits_6x4[:, d, :]
            if d in FULL_DIMS:
                out[:, d, 0] = torch.logsumexp(z[:, 1:], dim=-1) - z[:, 0]
                out[:, d, 1] = torch.logsumexp(z[:, 2:], dim=-1) - z[:, 1]
                out[:, d, 2] = z[:, 3] - z[:, 2]
            else:
                b_bin = z[:, 3] - z[:, 0]
                out[:, d, 0] = b_bin
                out[:, d, 1] = b_bin
                out[:, d, 2] = b_bin
        return out

    @torch.no_grad()
    def _enqueue(self, b_det: torch.Tensor, y_det: torch.Tensor, g_det: torch.Tensor) -> None:
        n = b_det.shape[0]
        if n >= self.capacity:
            self.mem_b.copy_(b_det[-self.capacity:])
            self.mem_y.copy_(y_det[-self.capacity:])
            self.mem_g.copy_(g_det[-self.capacity:])
            self.ptr.fill_(0)
            self.filled.fill_(self.capacity)
            return
        p = int(self.ptr.item())
        end = p + n
        if end <= self.capacity:
            self.mem_b[p:end].copy_(b_det)
            self.mem_y[p:end].copy_(y_det)
            self.mem_g[p:end].copy_(g_det)
        else:
            first = self.capacity - p
            self.mem_b[p:].copy_(b_det[:first])
            self.mem_y[p:].copy_(y_det[:first])
            self.mem_g[p:].copy_(g_det[:first])
            rem = n - first
            self.mem_b[:rem].copy_(b_det[first:])
            self.mem_y[:rem].copy_(y_det[first:])
            self.mem_g[:rem].copy_(g_det[first:])
        self.ptr.fill_(end % self.capacity)
        self.filled.fill_(min(self.capacity, int(self.filled.item()) + n))

    def forward(self, logits_6x4: torch.Tensor, grades: torch.Tensor, group_hashes: torch.Tensor) -> torch.Tensor:
        b_cur = self._boundary_scores(logits_6x4.float())
        y_cur = grades.long()
        g_cur = group_hashes.long()

        n_mem = int(self.filled.item())
        if n_mem > 0:
            b_ref = torch.cat([b_cur.detach(), self.mem_b[:n_mem]], dim=0)
            y_ref = torch.cat([y_cur, self.mem_y[:n_mem]], dim=0)
            g_ref = torch.cat([g_cur, self.mem_g[:n_mem]], dim=0)
        else:
            b_ref = b_cur.detach()
            y_ref = y_cur
            g_ref = g_cur

        diff_group = g_cur.unsqueeze(1) != g_ref.unsqueeze(0)
        inv_s = 1.0 / (2.0 * self.sigma0)
        losses: List[torch.Tensor] = []

        for d in range(6):
            yi = y_cur[:, d].unsqueeze(1)
            yj = y_ref[:, d].unsqueeze(0)
            dw = self.dim_weights[d]
            if d in FULL_DIMS:
                c0_hi = (yi >= 1) & (yj == 0) & diff_group
                c0_lo = (yi == 0) & (yj >= 1) & diff_group
                c1_hi = (yi >= 2) & (yj == 1) & diff_group
                c1_lo = (yi == 1) & (yj >= 2) & diff_group
                c2_hi = (yi == 3) & (yj == 2) & diff_group
                c2_lo = (yi == 2) & (yj == 3) & diff_group
                cuts = [(0, c0_hi, c0_lo, 1.00), (1, c1_hi, c1_lo, 1.45), (2, c2_hi, c2_lo, 1.80)]
            else:
                cb_hi = (yi >= 3) & (yj == 0) & diff_group
                cb_lo = (yi == 0) & (yj >= 3) & diff_group
                cuts = [(0, cb_hi, cb_lo, 1.35)]

            for bi, m_hi, m_lo, bw in cuts:
                valid = m_hi | m_lo
                if not valid.any():
                    continue
                sign = torch.where(m_hi, 1.0, -1.0)
                si = b_cur[:, d, bi].unsqueeze(1)
                sj = b_ref[:, d, bi].unsqueeze(0)
                phi = 0.5 * (1.0 + torch.erf(sign * (si - sj) * inv_s))
                fl = 1.0 - torch.sqrt(phi.clamp(min=1e-6, max=1.0))
                losses.append(dw * bw * fl[valid].mean())

        if self.training:
            self._enqueue(b_cur.detach(), y_cur.detach(), g_cur.detach())

        if not losses:
            return logits_6x4.sum() * 0.0
        return torch.stack(losses).mean()


class All4BoundaryV2Loss(nn.Module):
    def __init__(
        self,
        lambda_cum: float = 0.50,
        lambda_rps: float = 0.65,
        lambda_hinge: float = 0.35,
        lambda_bmargin: float = 0.25,
        lambda_clean: float = 0.25,
        kd_temp: float = 1.25,
        dim_weights: Sequence[float] = BALANCED_DIM_W,
    ):
        super().__init__()
        self.lambda_cum = float(lambda_cum)
        self.lambda_rps = float(lambda_rps)
        self.lambda_hinge = float(lambda_hinge)
        self.lambda_bmargin = float(lambda_bmargin)
        self.lambda_clean = float(lambda_clean)
        self.kd_temp = float(kd_temp)
        self.register_buffer("dim_w", torch.tensor(dim_weights, dtype=torch.float32))
        self.register_buffer("cw4", torch.tensor([1.00, 1.20, 1.65, 1.50], dtype=torch.float32))
        self.register_buffer("cw2", torch.tensor([1.00, 1.45], dtype=torch.float32))
        self.register_buffer("bw_emd", torch.tensor([1.00, 1.85, 1.55], dtype=torch.float32))
        self.bmargin = All4BoundaryMarginXBMLoss(capacity=512, sigma0=1.0, dim_weights=dim_weights)

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        grades: torch.Tensor,
        group_hashes: torch.Tensor,
        teacher_probs: Optional[torch.Tensor] = None,
        has_teacher: Optional[torch.Tensor] = None,
        lambda_kd: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        logits = out["logits_6x4"].float()
        probs = out["probs_6x4"].float()
        clean_logit = out["clean_logit"].float()
        y = grades.long()
        B = y.shape[0]

        all_clean = (y.sum(dim=1) == 0).float()
        sample_w = 1.0 + 0.35 * all_clean

        # 1. Sharp Class-Weighted Multinomial CE (exact V16 cw4, zero Gaussian P1-into-P2 smearing!)
        ce_dims = []
        for d in range(6):
            if d in FULL_DIMS:
                ce = F.cross_entropy(logits[:, d, :], y[:, d], weight=self.cw4, reduction="none", label_smoothing=0.02)
            else:
                ce = F.cross_entropy(
                    logits[:, d, [0, 3]], (y[:, d] >= 3).long(), weight=self.cw2, reduction="none", label_smoothing=0.015
                )
            ce_dims.append(self.dim_w[d] * (ce * sample_w).mean())
        l_ce = torch.stack(ce_dims).mean()

        # 2. Cumulative 3-Threshold Ordinal BCE (`>=P2, >=P1, ==P0` with Clean-P0 guard)
        cum_dims = []
        for d in range(6):
            pd = probs[:, d, :]
            if d in FULL_DIMS:
                p_ge1 = (pd[:, 1] + pd[:, 2] + pd[:, 3]).clamp(1e-6, 1.0 - 1e-6)
                p_ge2 = (pd[:, 2] + pd[:, 3]).clamp(1e-6, 1.0 - 1e-6)
                p_ge3 = pd[:, 3].clamp(1e-6, 1.0 - 1e-6)
                b1 = F.binary_cross_entropy(p_ge1, (y[:, d] >= 1).float(), reduction="none")
                b2 = F.binary_cross_entropy(p_ge2, (y[:, d] >= 2).float(), reduction="none") * 1.25
                b3 = F.binary_cross_entropy(p_ge3, (y[:, d] >= 3).float(), reduction="none") * (1.15 + 1.50 * all_clean)
                cum_dims.append(self.dim_w[d] * ((b1 + b2 + b3) / 3.40).mean())
            else:
                p_ge3 = pd[:, 3].clamp(1e-6, 1.0 - 1e-6)
                b3 = F.binary_cross_entropy(p_ge3, (y[:, d] >= 3).float(), reduction="none") * (1.15 + 1.50 * all_clean)
                cum_dims.append(self.dim_w[d] * b3.mean())
        l_cum = torch.stack(cum_dims).mean()

        # 3. Boundary-Weighted Squared-EMD / RPS (`bw_emd = [1.0, 1.85, 1.55]`)
        oh = F.one_hot(y.clamp(0, 3), num_classes=4).float()
        cdf_p = torch.cumsum(probs, dim=-1)[..., :3]
        cdf_y = torch.cumsum(oh, dim=-1)[..., :3]
        rps_dim = (((cdf_p - cdf_y) ** 2) * self.bw_emd.view(1, 1, 3)).mean(dim=-1)
        l_rps = (rps_dim * self.dim_w.view(1, 6) * sample_w.view(B, 1)).mean()

        # 4. Direct Bounded Hinge Logit Separation Margin (`P2 vs P1`, `P1 vs P2/P0`, `P0 vs P1`)
        hinge_terms = []
        for d in FULL_DIMS:
            zd = logits[:, d, :]
            yd = y[:, d]
            m_pos = yd >= 1
            if not m_pos.any():
                continue
            is_p2 = (yd == 1).float()
            is_p1 = (yd == 2).float()
            is_p0 = (yd == 3).float()
            sep = (
                is_p2 * F.relu(0.30 - (zd[:, 1] - zd[:, 2]))
                + is_p1 * (F.relu(0.45 - (zd[:, 2] - zd[:, 1])) + 0.60 * F.relu(0.30 - (zd[:, 2] - zd[:, 3])))
                + is_p0 * F.relu(0.45 - (zd[:, 3] - zd[:, 2]))
            )
            hinge_terms.append(self.dim_w[d] * (sep.sum() / m_pos.float().sum().clamp(min=1.0)))
        l_hinge = torch.stack(hinge_terms).mean() if hinge_terms else logits.sum() * 0.0

        # 5. Cross-Video Conditional Multi-Boundary XBM Loss
        l_bmargin = self.bmargin(logits, y, group_hashes)

        # 6. Leak-Free Annealed Nested KD (`lambda_kd -> 0.0` in final epoch)
        l_kd = logits.sum() * 0.0
        if lambda_kd > 0.0 and teacher_probs is not None and has_teacher is not None and (has_teacher > 0).any():
            tm = has_teacher > 0
            T = self.kd_temp
            kd_dims = []
            for d in range(6):
                if d in FULL_DIMS:
                    tp = teacher_probs[tm, d, :].float().clamp(min=1e-6)
                    tp_t = F.softmax(torch.log(tp) / T, dim=-1)
                    sp_t = F.log_softmax(logits[tm, d, :] / T, dim=-1)
                    kd_dims.append(self.dim_w[d] * F.kl_div(sp_t, tp_t, reduction="batchmean") * (T * T))
                else:
                    tp2 = teacher_probs[tm, d, :][:, [0, 3]].float().clamp(min=1e-6)
                    tp_t = F.softmax(torch.log(tp2) / T, dim=-1)
                    sp_t = F.log_softmax(logits[tm, d, :][:, [0, 3]] / T, dim=-1)
                    kd_dims.append(self.dim_w[d] * F.kl_div(sp_t, tp_t, reduction="batchmean") * (T * T))
            l_kd = torch.stack(kd_dims).mean()

        # 7. Global All-Clean BCE
        l_clean = F.binary_cross_entropy_with_logits(clean_logit, all_clean)

        total = (
            l_ce
            + self.lambda_cum * l_cum
            + self.lambda_rps * l_rps
            + self.lambda_hinge * l_hinge
            + self.lambda_bmargin * l_bmargin
            + float(lambda_kd) * l_kd
            + self.lambda_clean * l_clean
        )
        return {
            "total": total,
            "ce": l_ce,
            "cum": l_cum,
            "rps": l_rps,
            "hinge": l_hinge,
            "bmargin": l_bmargin,
            "kd": l_kd,
        }


# ---------------------------------------------------------------------------
# 5. Prototype Bank, 2x TTA, N=1 Invariance Check & Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def populate_prototypes(
    model: BoundaryTileTower,
    tr_rows: Sequence[Dict[str, object]],
    img_store: FastShmImageStore,
    phys_map: Dict[str, np.ndarray],
    tile_cap: int,
    device: torch.device,
    batch_size: int = 8,
    workers: int = 4,
) -> int:
    model.eval()
    ds = BoundaryNativeTileDataset(tr_rows, img_store, phys_map, tile_cap=tile_cap, train=False, hflip_tta=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate_boundary_tiles)
    zs, gs, ps = [], [], []
    for b in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                b["global_u8"].to(device, non_blocking=True),
                b["tiles_u8"].to(device, non_blocking=True),
                b["tile_mask"].to(device, non_blocking=True),
                b["tile_coords"].to(device, non_blocking=True),
                tile_energy=b["tile_energy"].to(device, non_blocking=True),
                phys_feat=b["phys"].to(device, non_blocking=True),
            )
        zs.append(out["z_norm"].float().cpu().numpy())
        gs.append(b["grades"].long().cpu().numpy())
        ps.append(b["phys"].float().cpu().numpy())
    return int(
        model.head.proto_bank.fit_from_train_embeddings(
            np.concatenate(zs, axis=0), np.concatenate(gs, axis=0), phys_np=np.concatenate(ps, axis=0)
        )
    )


@torch.no_grad()
def predict_tta(
    model: BoundaryTileTower,
    rows: Sequence[Dict[str, object]],
    img_store: FastShmImageStore,
    phys_map: Dict[str, np.ndarray],
    tile_cap: int,
    device: torch.device,
    batch_size: int = 8,
    workers: int = 4,
    tta_2x: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    n = len(rows)
    probs_acc = np.zeros((n, 6, 4), dtype=np.float64)
    qcl_acc = np.zeros((n,), dtype=np.float64)
    g_out = np.zeros((n, 6), dtype=np.int64)
    names_out = [str(r["name"]) for r in rows]
    modes = (False, True) if tta_2x else (False,)

    for hf in modes:
        ds = BoundaryNativeTileDataset(rows, img_store, phys_map, tile_cap=tile_cap, train=False, hflip_tta=hf)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate_boundary_tiles)
        for b in dl:
            idxs = b["idx_in_ds"].numpy()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    b["global_u8"].to(device, non_blocking=True),
                    b["tiles_u8"].to(device, non_blocking=True),
                    b["tile_mask"].to(device, non_blocking=True),
                    b["tile_coords"].to(device, non_blocking=True),
                    tile_energy=b["tile_energy"].to(device, non_blocking=True),
                    phys_feat=b["phys"].to(device, non_blocking=True),
                )
            p = out["probs_6x4"].float().cpu().numpy()
            qc = torch.sigmoid(out["clean_logit"].float()).cpu().numpy()
            probs_acc[idxs] += p / float(len(modes))
            qcl_acc[idxs] += qc / float(len(modes))
            if not hf:
                g_out[idxs] = b["grades"].numpy()

    probs_acc = np.clip(probs_acc, 1e-7, 1.0)
    probs_acc /= probs_acc.sum(axis=-1, keepdims=True)
    return probs_acc.astype(np.float32), qcl_acc.astype(np.float32), g_out, names_out


@torch.no_grad()
def verify_n1_batch_invariance(
    model: BoundaryTileTower,
    rows: Sequence[Dict[str, object]],
    img_store: FastShmImageStore,
    phys_map: Dict[str, np.ndarray],
    tile_cap: int,
    device: torch.device,
    n_check: int = 8,
) -> float:
    model.eval()
    sub = list(rows[:n_check])
    ds = BoundaryNativeTileDataset(sub, img_store, phys_map, tile_cap=tile_cap, train=False, hflip_tta=False)
    items = [ds[i] for i in range(len(sub))]
    batch_all = collate_boundary_tiles(items)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p_bat = model(
            batch_all["global_u8"].to(device),
            batch_all["tiles_u8"].to(device),
            batch_all["tile_mask"].to(device),
            batch_all["tile_coords"].to(device),
            tile_energy=batch_all["tile_energy"].to(device),
            phys_feat=batch_all["phys"].to(device),
        )["probs_6x4"].float().cpu().numpy()

    max_diff = 0.0
    for i, it in enumerate(items):
        b1 = collate_boundary_tiles([it])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p1 = model(
                b1["global_u8"].to(device),
                b1["tiles_u8"].to(device),
                b1["tile_mask"].to(device),
                b1["tile_coords"].to(device),
                tile_energy=b1["tile_energy"].to(device),
                phys_feat=b1["phys"].to(device),
            )["probs_6x4"].float().cpu().numpy()[0]
        max_diff = max(max_diff, float(np.abs(p1 - p_bat[i]).max()))
    return max_diff


def compute_boundary_val_metrics(probs: np.ndarray, gold: np.ndarray) -> Dict[str, float]:
    oh = np.eye(4)[gold]
    rps_all = float(((np.cumsum(probs, -1)[..., :3] - np.cumsum(oh, -1)[..., :3]) ** 2).mean())
    out: Dict[str, float] = {"rps": rps_all, "ece": float(compute_ece_6x4(probs, gold))}
    short = {0: "blur", 1: "tex", 2: "os", 4: "og"}
    auc_b1_list, auc_b2_list, ap_all = [], [], []
    for d in FULL_DIMS:
        sn = short[d]
        out[f"{sn}_rps"] = float(((np.cumsum(probs[:, d], -1)[:, :3] - np.cumsum(oh[:, d], -1)[:, :3]) ** 2).mean())
        for g in (1, 2, 3):
            y_b = (gold[:, d] == g).astype(int)
            if 0 < y_b.sum() < len(y_b):
                ap = float(average_precision_score(y_b, probs[:, d, g]))
                ap_all.append(ap)
                if g == 3:
                    out[f"{sn}_ap0"] = ap
        m1 = gold[:, d] >= 1
        y1 = (gold[m1, d] >= 2).astype(int)
        if m1.sum() > 2 and 0 < y1.sum() < len(y1):
            sc1 = (probs[m1, d, 2] + probs[m1, d, 3]) / np.clip(1.0 - probs[m1, d, 0], 1e-6, 1.0)
            a1 = float(roc_auc_score(y1, sc1))
            out[f"{sn}_auc_ge1"] = a1
            auc_b1_list.append(a1)
        m2 = gold[:, d] >= 2
        y2 = (gold[m2, d] == 3).astype(int)
        if m2.sum() > 2 and 0 < y2.sum() < len(y2):
            sc2 = probs[m2, d, 3] / np.clip(probs[m2, d, 2] + probs[m2, d, 3], 1e-6, 1.0)
            a2 = float(roc_auc_score(y2, sc2))
            out[f"{sn}_auc_p0"] = a2
            auc_b2_list.append(a2)
    out["mAP12"] = float(np.mean(ap_all)) if ap_all else 0.0
    out["mean_auc_ge1"] = float(np.mean(auc_b1_list)) if auc_b1_list else 0.0
    out["mean_auc_p0"] = float(np.mean(auc_b2_list)) if auc_b2_list else 0.0
    return out


# ---------------------------------------------------------------------------
# 6. Training Main Loop
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_fold(args: argparse.Namespace) -> None:
    assert 0 <= args.gpu <= 6, f"GPU {args.gpu} forbidden (strictly GPUs 0..6 only; NEVER touch GPU 7!)"
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(args.seed + max(0, args.fold) * 101)

    os.makedirs(args.out_dir, exist_ok=True)
    tr_rows_base, va_rows, all_names = load_canonical_8fold(args.split_csv, args.fold)
    tr_rows = apply_tagfree_p1p0_oversample(tr_rows_base, seed=args.seed + max(0, args.fold)) if args.p1p0_oversample else tr_rows_base
    img_store = FastShmImageStore()
    phys_map, phys_dim = load_semantic_phys_map(args.semantic_npz)

    teacher_map: Dict[str, np.ndarray] = {}
    if args.lambda_kd > 0.0 and args.kd_npz:
        teacher_map = load_leakfree_nested_kd(args.kd_npz, args.fold, all_names, va_rows)

    t0 = time.perf_counter()
    model = BoundaryTileTower(
        backbone_key=args.backbone,
        phys_dim=phys_dim,
        unfreeze_last_k=args.unfreeze_last_k,
        view_chunk=args.view_chunk,
        dropout=args.dropout,
    ).to(device)

    # Fit physics scaler strictly on unique real training fold rows
    tr_phys = np.stack([phys_map[str(r["name"])] for r in tr_rows_base], axis=0)
    model.head.set_physics_scaler(tr_phys.mean(axis=0), tr_phys.std(axis=0))

    n_tot = sum(p.numel() for p in model.parameters())
    n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[BoundaryV2] bb={args.backbone} fold={args.fold} gpu={args.gpu} | "
        f"train={len(tr_rows)} (base={len(tr_rows_base)}, os={int(args.p1p0_oversample)}) val={len(va_rows)} | "
        f"kd={args.lambda_kd:.2f}->0 ({len(teacher_map)} rows) | "
        f"params={n_tot/1e6:.1f}M (trn={n_trn/1e6:.1f}M) | load={time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    chk_rows = va_rows if len(va_rows) >= 8 else tr_rows_base
    n1_pre = verify_n1_batch_invariance(model, chk_rows, img_store, phys_map, args.tile_cap, device, n_check=8)
    print(f"  [N=1 Check Pre] max_abs_diff={n1_pre:.2e}", flush=True)
    assert n1_pre < 2e-3, f"Pre-train N=1 invariance violated: {n1_pre:.2e}"

    bb_params, hd_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("head.") or any(tok in n for tok in (".tap3.", ".abmil.", ".masked_abmil.")):
            hd_params.append(p)
        else:
            bb_params.append(p)

    trainable_params = bb_params + hd_params
    try:
        optimizer = torch.optim.AdamW(
            [{"params": bb_params, "lr": args.lr_backbone}, {"params": hd_params, "lr": args.lr_head}],
            weight_decay=args.weight_decay,
            fused=True,
        )
    except Exception:
        optimizer = torch.optim.AdamW(
            [{"params": bb_params, "lr": args.lr_backbone}, {"params": hd_params, "lr": args.lr_head}],
            weight_decay=args.weight_decay,
        )

    ds_tr = BoundaryNativeTileDataset(
        tr_rows, img_store, phys_map, tile_cap=args.tile_cap, train=True,
        train_rand_tile_prob=args.train_rand_tile_prob, teacher_map=teacher_map,
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_boundary_tiles,
        persistent_workers=(args.workers > 0),
    )

    steps_per_epoch = math.ceil(len(dl_tr) / float(args.grad_accum))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_pct))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = All4BoundaryV2Loss(
        lambda_cum=args.lambda_cum,
        lambda_rps=args.lambda_rps,
        lambda_hinge=args.lambda_hinge,
        lambda_bmargin=args.lambda_bmargin,
        lambda_clean=args.lambda_clean,
    ).to(device)

    swa_sds: List[Dict[str, torch.Tensor]] = []
    torch.cuda.reset_peak_memory_stats(device)

    for ep in range(args.epochs):
        t_ep = time.perf_counter()
        model.train()
        criterion.train()
        optimizer.zero_grad(set_to_none=True)

        # Cosine-anneal leak-free KD down to 0.0 in final epoch!
        if args.lambda_kd > 0.0 and args.epochs > 1:
            cur_lam_kd = float(args.lambda_kd) * 0.5 * (1.0 + math.cos(math.pi * ep / float(args.epochs - 1)))
        else:
            cur_lam_kd = float(args.lambda_kd)

        sums = np.zeros(7, dtype=np.float64)
        n_b = 0

        for step_i, b in enumerate(dl_tr):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    b["global_u8"].to(device, non_blocking=True),
                    b["tiles_u8"].to(device, non_blocking=True),
                    b["tile_mask"].to(device, non_blocking=True),
                    b["tile_coords"].to(device, non_blocking=True),
                    tile_energy=b["tile_energy"].to(device, non_blocking=True),
                    phys_feat=b["phys"].to(device, non_blocking=True),
                    photo_bc=b["photo_bc"].to(device, non_blocking=True),
                )
            with torch.autocast("cuda", enabled=False):
                losses = criterion(
                    out,
                    b["grades"].to(device, non_blocking=True),
                    b["ghash"].to(device, non_blocking=True),
                    teacher_probs=b["teacher_6x4"].to(device, non_blocking=True) if cur_lam_kd > 0 else None,
                    has_teacher=b["has_teacher"].to(device, non_blocking=True) if cur_lam_kd > 0 else None,
                    lambda_kd=cur_lam_kd,
                )
                loss = losses["total"] / float(args.grad_accum)

            loss.backward()
            sums += np.array([
                float(losses["total"].item()),
                float(losses["ce"].item()),
                float(losses["cum"].item()),
                float(losses["rps"].item()),
                float(losses["hinge"].item()),
                float(losses["bmargin"].item()),
                float(losses["kd"].item()),
            ])
            n_b += 1

            if (step_i + 1) % args.grad_accum == 0 or (step_i + 1) == len(dl_tr):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if args.bench_steps > 0 and (step_i + 1) >= args.bench_steps:
                el = time.perf_counter() - t_ep
                pk = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(
                    f"  [BENCH {args.backbone}] {args.bench_steps} steps in {el:.1f}s "
                    f"({args.bench_steps * args.batch_size / el:.2f} imgs/s) | peak={pk:.1f} GiB",
                    flush=True,
                )
                return

        tr_s = time.perf_counter() - t_ep

        # Populate TipAdapterPrototypeBank in epochs E-3 and E-2 (strictly on unique real tr_rows_base!)
        n_proto = int(model.head.proto_bank.num_active.item())
        if max(1, args.epochs - 3) <= ep < args.epochs - 1:
            n_proto = populate_prototypes(
                model, tr_rows_base, img_store, phys_map, args.tile_cap, device,
                batch_size=args.eval_batch_size, workers=args.workers,
            )

        ml = sums / max(1, n_b)
        pk = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if len(va_rows) > 0:
            vp1, _, gv1, _ = predict_tta(
                model, va_rows, img_store, phys_map, args.tile_cap, device,
                batch_size=args.eval_batch_size, workers=args.workers, tta_2x=False,
            )
            vm = compute_boundary_val_metrics(vp1, gv1)
            print(
                f"  [{args.backbone} F{args.fold} Ep {ep+1}/{args.epochs}] "
                f"loss={ml[0]:.4f} (ce={ml[1]:.3f} cum={ml[2]:.3f} hng={ml[4]:.3f} bm={ml[5]:.3f} kd={ml[6]:.3f}@{cur_lam_kd:.2f}) | "
                f"1xVal RPS={vm['rps']:.5f} mAP12={vm['mAP12']:.4f} "
                f"AUC(ge1/p0)={vm['mean_auc_ge1']:.4f}/{vm['mean_auc_p0']:.4f} "
                f"ECE={vm['ece']:.4f} | prot={n_proto} pk={pk:.1f}G t={tr_s:.0f}s",
                flush=True,
            )
        else:
            print(
                f"  [{args.backbone} FullFit Ep {ep+1}/{args.epochs}] "
                f"loss={ml[0]:.4f} (ce={ml[1]:.3f} hng={ml[4]:.3f} bm={ml[5]:.3f} kd={ml[6]:.3f}@{cur_lam_kd:.2f}) | "
                f"prot={n_proto} pk={pk:.1f}G t={tr_s:.0f}s",
                flush=True,
            )

        if ep >= max(0, args.epochs - 2):
            swa_sds.append(model.trainable_state_dict())

    # Leak-free Fixed Last-2 Epoch SWA + Final Prototype Bank
    t_fin = time.perf_counter()
    avg_sd: Dict[str, torch.Tensor] = {}
    for k, v in swa_sds[-1].items():
        if v.is_floating_point():
            avg_sd[k] = (sum(sd[k].float() for sd in swa_sds) / float(len(swa_sds))).to(v.dtype)
        else:
            avg_sd[k] = v
    model.load_state_dict(avg_sd, strict=False)

    n_proto_fin = populate_prototypes(
        model, tr_rows_base, img_store, phys_map, args.tile_cap, device,
        batch_size=args.eval_batch_size, workers=args.workers,
    )
    n1_post = verify_n1_batch_invariance(model, chk_rows, img_store, phys_map, args.tile_cap, device, n_check=8)
    assert n1_post < 2e-3, f"Post-SWA N=1 invariance violated: {n1_post:.2e}"

    if len(va_rows) == 0:
        ckpt_path = os.path.join(args.out_dir, "fullfit_best.pt")
        torch.save(
            {
                "state_dict": model.trainable_state_dict(),
                "backbone": args.backbone,
                "unfreeze_last_k": model.unfreeze_last_k,
                "phys_dim": phys_dim,
                "tile_cap": args.tile_cap,
                "dropout": args.dropout,
                "fold": -1,
                "n_prototypes": n_proto_fin,
                "max_n1_batch_diff": n1_post,
            },
            ckpt_path,
        )
        print(f"=== [DONE FullFit {args.backbone}] prot={n_proto_fin} N1={n1_post:.2e} -> {ckpt_path} ===", flush=True)
        return

    va_probs, va_qclean, gva, va_names = predict_tta(
        model, va_rows, img_store, phys_map, args.tile_cap, device,
        batch_size=args.eval_batch_size, workers=args.workers, tta_2x=True,
    )
    va_p0, va_p1, va_p2 = simplex_6x4_to_p012(va_probs)
    vm_fin = compute_boundary_val_metrics(va_probs, gva)

    if not args.no_save_ckpt:
        ckpt_path = os.path.join(args.out_dir, f"fold_{args.fold}_best.pt")
        torch.save(
            {
                "state_dict": model.trainable_state_dict(),
                "backbone": args.backbone,
                "unfreeze_last_k": model.unfreeze_last_k,
                "phys_dim": phys_dim,
                "tile_cap": args.tile_cap,
                "dropout": args.dropout,
                "fold": args.fold,
                "n_prototypes": n_proto_fin,
                "val_metrics_2xtta": vm_fin,
            },
            ckpt_path,
        )

    out_npz = os.path.join(args.out_dir, f"fold_{args.fold}_probs.npz")
    np.savez_compressed(
        out_npz,
        fold=np.int64(args.fold),
        va_names=np.array(va_names, dtype=object),
        va_probs_6x4=va_probs.astype(np.float32),
        va_qclean=va_qclean.astype(np.float32),
        va_p0=va_p0.astype(np.float32),
        va_p1=va_p1.astype(np.float32),
        va_p2=va_p2.astype(np.float32),
        gva=gva.astype(np.int64),
    )
    with open(os.path.join(args.out_dir, f"fold_{args.fold}_summary.json"), "w", encoding="utf-8") as jf:
        json.dump(
            {
                "backbone": args.backbone,
                "fold": args.fold,
                "n_val": len(va_names),
                "n_prototypes": n_proto_fin,
                "max_n1_batch_diff": n1_post,
                "val_metrics_2xtta": vm_fin,
            },
            jf,
            indent=2,
        )
    print(
        f"=== [DONE {args.backbone} F{args.fold}] 2xTTA SWA -> RPS={vm_fin['rps']:.5f} mAP12={vm_fin['mAP12']:.4f} | "
        f"AUC(ge1/p0)={vm_fin['mean_auc_ge1']:.4f}/{vm_fin['mean_auc_p0']:.4f} | "
        f"AP0(B/T/OS/OG)={vm_fin.get('blur_ap0',0):.3f}/{vm_fin.get('tex_ap0',0):.3f}/"
        f"{vm_fin.get('os_ap0',0):.3f}/{vm_fin.get('og_ap0',0):.3f} | "
        f"ECE={vm_fin['ece']:.4f} N1={n1_post:.2e} fin={time.perf_counter()-t_fin:.1f}s -> {out_npz} ===",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 2B-V2: Tag-Free All-4-Dim Boundary Native-Tile Specialist Trainer")
    ap.add_argument("--backbone", required=True, choices=["pe_core_large", "cradio_v4_h", "dual_pe_cradio", "dinov3_vitl16", "pe_core_g14"])
    ap.add_argument("--fold", type=int, default=0, help="0..7 for 8-fold CV, -1 for 100% FullFit")
    ap.add_argument("--gpu", type=int, default=0, help="Physical GPU ID (strictly 0..6; NEVER 7)")
    ap.add_argument("--split-csv", default="data/qingyu_splits/qingyu_8fold.csv")
    ap.add_argument("--semantic-npz", default="results/r55/semantic_roi_meta.npz")
    ap.add_argument("--kd-npz", default="results/r55/nested_kd_teacher.npz")
    ap.add_argument("--lambda-kd", type=float, default=0.25)
    ap.add_argument("--p1p0-oversample", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--unfreeze-last-k", type=int, default=-1)
    ap.add_argument("--tile-cap", type=int, default=12)
    ap.add_argument("--train-rand-tile-prob", type=float, default=0.40)
    ap.add_argument("--epochs", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=-1)
    ap.add_argument("--grad-accum", type=int, default=-1)
    ap.add_argument("--eval-batch-size", type=int, default=12)
    ap.add_argument("--view-chunk", type=int, default=-1)
    ap.add_argument("--lr-backbone", type=float, default=-1.0)
    ap.add_argument("--lr-head", type=float, default=4.0e-4)
    ap.add_argument("--weight-decay", type=float, default=0.025)
    ap.add_argument("--dropout", type=float, default=0.22)
    ap.add_argument("--warmup-pct", type=float, default=0.10)
    ap.add_argument("--lambda-cum", type=float, default=0.50)
    ap.add_argument("--lambda-rps", type=float, default=0.65)
    ap.add_argument("--lambda-hinge", type=float, default=0.35)
    ap.add_argument("--lambda-bmargin", type=float, default=0.25)
    ap.add_argument("--lambda-clean", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-save-ckpt", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=0)
    args = ap.parse_args()

    if args.epochs <= 0:
        args.epochs = 5 if args.backbone == "pe_core_large" else 4
    if args.batch_size <= 0:
        args.batch_size = {"pe_core_large": 16, "cradio_v4_h": 16, "dual_pe_cradio": 12, "dinov3_vitl16": 16, "pe_core_g14": 8}[args.backbone]
    if args.grad_accum <= 0:
        args.grad_accum = 1 if args.batch_size >= 12 else max(1, 16 // args.batch_size)
    if args.view_chunk <= 0:
        args.view_chunk = {"pe_core_large": 128, "cradio_v4_h": 128, "dual_pe_cradio": 96, "dinov3_vitl16": 128, "pe_core_g14": 64}[args.backbone]
    if args.lr_backbone <= 0:
        args.lr_backbone = {"pe_core_large": 2.5e-5, "cradio_v4_h": 2.0e-5, "dual_pe_cradio": 2.0e-5, "dinov3_vitl16": 2.0e-5, "pe_core_g14": 1.6e-5}[args.backbone]
    return args


if __name__ == "__main__":
    train_fold(parse_args())
