#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/vlm_c233397_qy8.py
===========================
Exact Zero-Drift Vendored `c233397` (`Qwen3-VL-8B` / `pillar1`) 8-Fold Trainer, Schedule-Matched
Full-Fit Trainer, and 4090 NF4 Deployment Builder for `resident_v6` (Locked per Fable-5.2 Round 2):

1. Exact `c233397` Architecture & Recipe:
   - `CONCISE_INSPECTION_PROMPT_C233397` (exact commit `c233397` text).
   - `NativeVLMPillarModelC233397`: DoRA `r=24, alpha=48, dropout=0.05` on all `nn.Linear` layers
     (excluding `patch_embed`, `lm_head`, `embed_tokens`, `wte`), trainable multimodal `merger`,
     stripped `lm_head`, `(h[50%] + h[75%] + h[100%]) / 3.0` hypercolumn fusion, and `63` adaptive
     pooled valid tokens + `1` final instruction summary token (`[B, 64, D]`).
   - `PhoenixUnimodalDeQAHeadC233397`: `kappa = softplus(log_kappa) * 0.35`, `shift * 0.35`.
   - `DeQAOrdinalDistributionLossC233397` (`sigma_anchor=0.30, sigma_mid=0.46`, `beta_emd=1.50`,
     `beta_clean=0.65`, `p0_sample_boost=1.65`, `dim_w=[1.0, 1.0, 1.15, 1.35, 1.35, 1.40]`) +
     `0.35 * CrossBatchThurstoneRL2RLossC233397` from epoch >= 1.
   - NO `oversample_rare_real_p1_p0`, NO `grade_w` class-prior distortion, NO teacher distillation.
   - `lr_head=2.0e-4`, `lr_backbone=8.0e-5`, `epochs=4`, `batch_size=4`, `grad_accum=2`, SWA of epochs 3 & 4.

2. Honest 8-Fold `phoenix_qy8.csv` (`qingyu_8fold.csv`) Hygiene & Fast H200 Execution:
   - Verifies 1:1 row & fold alignment with `common55.load_split()` (zero sibling video leakage).
   - Computes `init_batch_shift` strictly on `tr_rows` inside each fold (zero validation label leakage).
   - Evaluates 2x HFlip at validation, saving `probs_orig_6x4`, `probs_hflip_6x4`, and `probs_6x4` separately.
   - Moves deterministic `processor(text, images)` tokenization/patchification into DataLoader worker
     collation and supports `--no-grad-ckpt` on 141 GB H200 GPUs (100% mathematically identical forward/backward,
     eliminating redundant activation recomputation & main-thread CPU stalls).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

from phoenix.config import (
    DIMS,
    FULL_DIMS,
    P0_ONLY_DIMS,
    PHOENIX_OOF_DIR,
    PHOENIX_SPLITS_DIR,
    PureRealImageResolver,
    evaluate_official_metrics,
    resolve_local_or_hf_model,
    set_deterministic_seed,
)
from phoenix.native_physics_and_views import resize_native_aspect_pil
from phoenix.train_pillar12_native_vlm import (
    crop_1to1_from_box,
    get_or_build_scrutiny_box_cache,
    load_split_rows,
)

RECIPE_TAG = "c233397_qy8"

CONCISE_INSPECTION_PROMPT_C233397 = (
    "请作为短视频无参考画质质检专家，综合图1（原生宽高比全局画面）与图2、图3（两处1:1原尺寸局部审视切片），"
    "直接形成6类画质缺陷（模糊、纹理损失、过锐、花屏、过度生成、虚化突兀）的客观严重度表征。"
)


def compute_fold_batch_shift(tr_rows: Sequence[Dict[str, object]]) -> np.ndarray:
    """
    Computes `log_odds_shift_supp_vs_orig` (`[6, 4]` float32) strictly on `tr_rows`
    using the exact Laplace-smoothed formula (`alpha_smooth = 2.0`) from `data_split_and_audit.py`.
    """
    orig_rows = [r for r in tr_rows if int(r["batch_id"]) == 0]
    supp_rows = [r for r in tr_rows if int(r["batch_id"]) == 1]
    n_orig = max(1, len(orig_rows))
    n_supp = max(1, len(supp_rows))
    alpha_smooth = 2.0
    out = np.zeros((6, 4), dtype=np.float32)
    for d_idx in range(6):
        valid_grades = [0, 1, 2, 3] if d_idx in FULL_DIMS else [0, 3]
        denom_k = float(len(valid_grades))
        for g in valid_grades:
            cnt_o = sum(1 for r in orig_rows if int(r["grades"][d_idx]) == g)  # type: ignore[index]
            cnt_s = sum(1 for r in supp_rows if int(r["grades"][d_idx]) == g)  # type: ignore[index]
            p_o = (cnt_o + alpha_smooth) / (n_orig + alpha_smooth * denom_k)
            p_s = (cnt_s + alpha_smooth) / (n_supp + alpha_smooth * denom_k)
            out[d_idx, g] = float(math.log(p_s / p_o))
    return out


class PhoenixUnimodalDeQAHeadC233397(nn.Module):
    """Exact `c233397` / `ef417c4` `PhoenixUnimodalDeQAHead` (`kappa = softplus * 0.35`)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.10,
        init_batch_shift: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.defect_queries = nn.Parameter(torch.randn(1, 7, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.inter_defect_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.grade_proj = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(6)])
        self.center_proj = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(6)])
        self.log_kappa = nn.Parameter(torch.zeros(6))
        self.clean_gate_proj = nn.Linear(hidden_dim, 1)
        self.anchor_emb_proj = nn.Sequential(
            nn.Linear(hidden_dim * 7, 256),
            nn.LayerNorm(256),
        )

        shift_tensor = torch.zeros(6, 4, dtype=torch.float32)
        if init_batch_shift is not None:
            arr = np.asarray(init_batch_shift, dtype=np.float32)
            if arr.shape == (6, 4):
                shift_tensor = torch.from_numpy(arr) * 0.35
        self.supp_batch_bias = nn.Parameter(shift_tensor)

        grade_grid = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        self.register_buffer("grade_grid", grade_grid, persistent=False)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        apply_batch_bias: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if spatial_tokens.ndim == 2:
            spatial_tokens = spatial_tokens.unsqueeze(1)
        amp_dev = "cuda" if spatial_tokens.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            tokens = self.in_proj(spatial_tokens.float())
            bsz = tokens.shape[0]

            q = self.defect_queries.float().expand(bsz, -1, -1)
            q_cross, attn_weights = self.cross_attn(
                q, tokens, tokens, need_weights=True, average_attn_weights=True
            )
            q = self.norm1(q + q_cross)
            q_self, _ = self.inter_defect_attn(q, q, q, need_weights=False)
            q = self.norm2(q + q_self)

            use_bias = self.training if apply_batch_bias is None else apply_batch_bias
            kappa = F.softplus(self.log_kappa.float()) * 0.35
            grid_1d = self.grade_grid.float()

            logits_list = []
            for d in range(6):
                qd = q[:, d, :]
                raw_z = self.grade_proj[d](qd)
                center = 3.0 * torch.sigmoid(self.center_proj[d](qd))
                unimodal_prior = -kappa[d] * (grid_1d.unsqueeze(0) - center) ** 2
                z = raw_z + unimodal_prior

                if use_bias and batch_ids is not None:
                    supp_w = (batch_ids == 1).float().unsqueeze(-1)
                    z = z + supp_w * self.supp_batch_bias[d].float().unsqueeze(0)

                if d in P0_ONLY_DIMS:
                    mask = torch.zeros_like(z)
                    mask[:, 1] = -1e4
                    mask[:, 2] = -1e4
                    z = z + mask
                logits_list.append(z)

            logits_6x4 = torch.stack(logits_list, dim=1)
            probs_6x4 = F.softmax(logits_6x4, dim=-1)

            clean_logit = self.clean_gate_proj(q[:, 6, :]).squeeze(-1)
            p_clean_head = torch.sigmoid(clean_logit)
            p_clean_prod = torch.prod(probs_6x4[:, :, 0], dim=1)
            p_clean_joint = 0.50 * p_clean_head + 0.50 * p_clean_prod

            grid = grid_1d.view(1, 1, 4)
            mu_6d = torch.sum(probs_6x4 * grid, dim=-1)
            var_6d = torch.sum(probs_6x4 * (grid - mu_6d.unsqueeze(-1)) ** 2, dim=-1)

            quality_emb = F.normalize(self.anchor_emb_proj(q.reshape(bsz, -1)), dim=-1)

            return {
                "logits_6x4": logits_6x4,
                "probs_6x4": probs_6x4,
                "clean_logit": clean_logit,
                "p_clean_joint": p_clean_joint,
                "mu_6d": mu_6d,
                "var_6d": var_6d,
                "quality_emb": quality_emb,
                "attn_map": attn_weights,
            }


def build_heteroscedastic_soft_targets_c233397(
    y_6d: torch.Tensor,
    noisy_flags: Optional[torch.Tensor] = None,
    sigma_anchor: float = 0.30,
    sigma_mid: float = 0.46,
) -> torch.Tensor:
    device = y_6d.device
    bsz = y_6d.shape[0]
    grid = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=torch.float32).view(1, 1, 4)
    y_f = y_6d.float().unsqueeze(-1)

    is_mid = ((y_6d == 1) | (y_6d == 2)).float().unsqueeze(-1)
    sigma = sigma_anchor + is_mid * (sigma_mid - sigma_anchor)
    if noisy_flags is not None:
        sigma = sigma + noisy_flags.float().view(bsz, 1, 1) * 0.16

    dist2 = (grid - y_f) ** 2
    raw = torch.exp(-dist2 / (2.0 * sigma ** 2))

    bin_mask = torch.ones((1, 6, 4), device=device, dtype=torch.float32)
    for d in P0_ONLY_DIMS:
        bin_mask[:, d, 1] = 0.0
        bin_mask[:, d, 2] = 0.0
    raw = raw * bin_mask
    return raw / torch.clamp(raw.sum(dim=-1, keepdim=True), min=1e-8)


class DeQAOrdinalDistributionLossC233397(nn.Module):
    """Exact undistorted `c233397` loss (no `grade_w` distortion, no `loss_p1_sep`)."""

    def __init__(
        self,
        beta_emd: float = 1.50,
        beta_clean: float = 0.65,
        p0_sample_boost: float = 1.65,
    ) -> None:
        super().__init__()
        self.beta_emd = beta_emd
        self.beta_clean = beta_clean
        self.p0_sample_boost = p0_sample_boost
        dim_w = torch.tensor([1.0, 1.0, 1.15, 1.35, 1.35, 1.40], dtype=torch.float32)
        self.register_buffer("dim_weights", dim_w, persistent=False)

    def forward(
        self,
        head_out: Dict[str, torch.Tensor],
        y_6d: torch.Tensor,
        noisy_flags: Optional[torch.Tensor] = None,
        supp_bias_param: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = head_out["logits_6x4"].float()
        probs = head_out["probs_6x4"].float()
        clean_logit = head_out["clean_logit"].float()

        q = build_heteroscedastic_soft_targets_c233397(y_6d=y_6d, noisy_flags=noisy_flags)
        log_p = F.log_softmax(logits, dim=-1)
        kl_per = torch.sum(q * (torch.log(torch.clamp(q, min=1e-7)) - log_p), dim=-1)

        cdf_p = torch.cumsum(probs, dim=-1)[:, :, :3]
        cdf_q = torch.cumsum(q, dim=-1)[:, :, :3]
        emd_per = torch.sum((cdf_p - cdf_q) ** 2, dim=-1)

        sample_w = 1.0 + (y_6d == 3).float() * (self.p0_sample_boost - 1.0)
        dim_w = self.dim_weights.float().view(1, 6)

        loss_kl = torch.mean(kl_per * sample_w * dim_w)
        loss_emd = torch.mean(emd_per * sample_w * dim_w)

        clean_gt = (y_6d == 0).all(dim=1).float()
        loss_clean = F.binary_cross_entropy_with_logits(clean_logit, clean_gt)

        total = loss_kl + self.beta_emd * loss_emd + self.beta_clean * loss_clean
        if supp_bias_param is not None:
            total = total + 0.01 * torch.mean(supp_bias_param.float() ** 2)

        return {
            "loss": total,
            "loss_kl": loss_kl.detach(),
            "loss_emd": loss_emd.detach(),
            "loss_clean": loss_clean.detach(),
        }


class CrossBatchThurstoneRL2RLossC233397(nn.Module):
    """Exact `c233397` CrossBatchThurstoneRL2RLoss."""

    def __init__(self, queue_size: int = 1024, emb_dim: int = 256) -> None:
        super().__init__()
        self.queue_size = queue_size
        self.register_buffer("q_mu", torch.zeros(queue_size, 6), persistent=False)
        self.register_buffer("q_var", torch.ones(queue_size, 6) * 0.25, persistent=False)
        self.register_buffer("q_y", torch.zeros(queue_size, 6, dtype=torch.long), persistent=False)
        self.register_buffer("q_emb", torch.zeros(queue_size, emb_dim), persistent=False)
        self.register_buffer("q_ptr", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("q_filled", torch.zeros(1, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def _enqueue(
        self,
        mu: torch.Tensor,
        var: torch.Tensor,
        y_6d: torch.Tensor,
        emb: torch.Tensor,
    ) -> None:
        b = mu.shape[0]
        if b == 0:
            return
        if b >= self.queue_size:
            self.q_mu.copy_(mu[-self.queue_size :].detach().float())
            self.q_var.copy_(var[-self.queue_size :].detach().float())
            self.q_y.copy_(y_6d[-self.queue_size :].detach())
            self.q_emb.copy_(emb[-self.queue_size :].detach().float())
            self.q_ptr[0] = 0
            self.q_filled[0] = self.queue_size
            return

        ptr = int(self.q_ptr.item())
        end = ptr + b
        if end <= self.queue_size:
            self.q_mu[ptr:end] = mu.detach().float()
            self.q_var[ptr:end] = var.detach().float()
            self.q_y[ptr:end] = y_6d.detach()
            self.q_emb[ptr:end] = emb.detach().float()
        else:
            first = self.queue_size - ptr
            self.q_mu[ptr:] = mu[:first].detach().float()
            self.q_var[ptr:] = var[:first].detach().float()
            self.q_y[ptr:] = y_6d[:first].detach()
            self.q_emb[ptr:] = emb[:first].detach().float()
            rem = b - first
            self.q_mu[:rem] = mu[first:].detach().float()
            self.q_var[:rem] = var[first:].detach().float()
            self.q_y[:rem] = y_6d[first:].detach()
            self.q_emb[:rem] = emb[first:].detach().float()

        self.q_ptr[0] = end % self.queue_size
        self.q_filled[0] = min(self.queue_size, int(self.q_filled.item()) + b)

    def forward(
        self,
        head_out: Dict[str, torch.Tensor],
        y_6d: torch.Tensor,
        grpo_group_samples: int = 6,
    ) -> torch.Tensor:
        mu = head_out["mu_6d"].float()
        var = head_out["var_6d"].float()
        probs = head_out["probs_6x4"].float()
        emb = head_out["quality_emb"].float()

        filled = int(self.q_filled.item())
        if filled > 0:
            ref_mu = torch.cat([mu.detach(), self.q_mu[:filled]], dim=0)
            ref_var = torch.cat([var.detach(), self.q_var[:filled]], dim=0)
            ref_y = torch.cat([y_6d, self.q_y[:filled]], dim=0)
            ref_emb = torch.cat([emb.detach(), self.q_emb[:filled]], dim=0)
        else:
            ref_mu = mu.detach()
            ref_var = var.detach()
            ref_y = y_6d
            ref_emb = emb.detach()

        sim_weight = torch.clamp(emb.detach() @ ref_emb.T, min=0.0)
        hard_scene_w = 1.0 + 1.5 * sim_weight

        sqrt2 = math.sqrt(2.0)
        pair_losses = []
        for d in range(6):
            yd_cur = y_6d[:, d].unsqueeze(1)
            yd_ref = ref_y[:, d].unsqueeze(0)
            diff_y = yd_cur - yd_ref
            valid = diff_y != 0
            if not torch.any(valid):
                continue

            sign = torch.sign(diff_y.float())
            adj_boost = torch.where(torch.abs(diff_y) == 1, 1.5, 1.0)
            denom = torch.sqrt(var[:, d].unsqueeze(1) + ref_var[:, d].unsqueeze(0) + 0.08)
            z_score = sign * (mu[:, d].unsqueeze(1) - ref_mu[:, d].unsqueeze(0)) / denom
            phi = 0.5 * (1.0 + torch.erf(z_score / sqrt2))
            fidelity_loss = (1.0 - phi) * adj_boost * hard_scene_w
            pair_losses.append(fidelity_loss[valid].mean())

        loss_thurstone = torch.stack(pair_losses).mean() if pair_losses else mu.sum() * 0.0

        if grpo_group_samples > 1:
            flat_p = torch.clamp(probs.reshape(-1, 4), min=1e-7)
            flat_p = flat_p / flat_p.sum(dim=-1, keepdim=True)
            actions = torch.multinomial(flat_p, num_samples=grpo_group_samples, replacement=True)
            gt_flat = y_6d.reshape(-1, 1).expand_as(actions)

            exact_hit = (actions == gt_flat).float()
            dist_pen = -0.35 * torch.abs(actions.float() - gt_flat.float())
            p0_miss = ((gt_flat == 3) & (actions != 3)).float() * (-1.25)
            clean_fp = ((gt_flat == 0) & (actions != 0)).float() * (-1.10)
            rewards = exact_hit + dist_pen + p0_miss + clean_fp

            r_mean = rewards.mean(dim=1, keepdim=True)
            r_std = rewards.std(dim=1, keepdim=True) + 1e-5
            adv = ((rewards - r_mean) / r_std).detach()

            log_probs_all = torch.log(flat_p)
            chosen_logp = torch.gather(log_probs_all, dim=1, index=actions)
            loss_grpo = -torch.mean(adv * chosen_logp) * 0.15
        else:
            loss_grpo = mu.sum() * 0.0

        self._enqueue(mu, var, y_6d, emb)
        return loss_thurstone + loss_grpo


def extract_trainable_state_dict_c233397(model: nn.Module) -> Dict[str, torch.Tensor]:
    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        kl = k.lower()
        if (
            (k in trainable_keys)
            or k.startswith("head.")
            or ("lora_" in kl)
            or ("dora_" in kl)
            or ("magnitude_vector" in kl)
        ):
            out[k] = v.detach().cpu().clone()
    return out


class BasinAlignedSWACollectorC233397:
    def __init__(self) -> None:
        self.count = 0
        self.total_weight = 0.0
        self.avg_state: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module, weight: float = 1.0) -> None:
        raw_sd = extract_trainable_state_dict_c233397(model)
        state = {k: (v.float() if v.is_floating_point() else v.clone()) for k, v in raw_sd.items()}
        if self.avg_state is None:
            self.avg_state = {k: (v * weight if v.is_floating_point() else v) for k, v in state.items()}
            self.total_weight = float(weight)
        else:
            for k, v in state.items():
                if v.is_floating_point():
                    self.avg_state[k] += v * weight
                else:
                    self.avg_state[k] = v
            self.total_weight += float(weight)
        self.count += 1

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.avg_state is None or self.count == 0:
            raise RuntimeError("BasinAlignedSWACollectorC233397 has no collected checkpoints.")
        out: Dict[str, torch.Tensor] = {}
        for k, v in self.avg_state.items():
            if v.is_floating_point():
                out[k] = v / max(1e-8, self.total_weight)
            else:
                out[k] = v.clone()
        return out


def format_3image_chat_prompt_c233397(processor: Any, prompt_text: str = CONCISE_INSPECTION_PROMPT_C233397) -> str:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        tpl_str = str(getattr(processor, "chat_template", "") or "")
        extra_kw: Dict[str, Any] = {}
        if "enable_thinking" in tpl_str:
            extra_kw["enable_thinking"] = False
        try:
            return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra_kw)
        except TypeError:
            return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<image>\n<image>\n<image>\n{prompt_text}"


class NativeVLMPillarModelC233397(nn.Module):
    """
    Exact vendored `c233397` `NativeVLMPillarModel` (`r=24`, 63-avg-pool + 1-summary readout)
    so training AND `final_engine.py` 4090 inference share 100% identical forward math.
    """

    def __init__(
        self,
        pillar_name: str = "pillar1",
        model_id_or_path: Optional[str] = None,
        init_batch_shift: Optional[np.ndarray] = None,
        apply_dora: bool = True,
        lora_rank: int = 24,
        enable_gc: bool = True,
        four_bit: bool = False,
        device_str: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.pillar_name = pillar_name
        self.lora_rank = int(lora_rank)
        self.four_bit = bool(four_bit)
        self.recipe_tag = RECIPE_TAG
        key = "pillar1_qwen3vl_8b" if pillar_name == "pillar1" else "pillar2_hetero_vlm_8b"
        resolved = model_id_or_path or resolve_local_or_hf_model(key)
        self.backbone, self.processor, in_dim = self._load_vlm_backbone(
            resolved_path=resolved,
            apply_dora=apply_dora,
            lora_rank=self.lora_rank,
            enable_gc=enable_gc,
            four_bit=self.four_bit,
            device_str=device_str,
        )
        self.prompt_text_formatted = format_3image_chat_prompt_c233397(
            self.processor, CONCISE_INSPECTION_PROMPT_C233397
        )
        self.head = PhoenixUnimodalDeQAHeadC233397(
            in_dim=in_dim,
            hidden_dim=512,
            num_heads=8,
            dropout=0.10,
            init_batch_shift=init_batch_shift,
        ).float()

    @staticmethod
    def _load_vlm_backbone(
        resolved_path: str,
        apply_dora: bool,
        lora_rank: int = 24,
        enable_gc: bool = True,
        four_bit: bool = False,
        device_str: Optional[str] = None,
    ):
        import transformers
        from transformers import AutoConfig, AutoProcessor

        if four_bit:
            pydeps = os.path.join(REPO, "third_party/pydeps")
            if os.path.isdir(pydeps) and pydeps not in sys.path:
                sys.path.insert(0, pydeps)
            from transformers import BitsAndBytesConfig

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        if hasattr(tok, "padding_side"):
            tok.padding_side = "right"
        cfg = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True)

        loader_classes = []
        for cls_name in (
            "AutoModelForImageTextToText",
            "Qwen3VLForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
            "AutoModel",
        ):
            cls_obj = getattr(transformers, cls_name, None)
            if cls_obj is not None and cls_obj not in loader_classes:
                loader_classes.append(cls_obj)

        full_model = None
        last_err: Optional[Exception] = None
        for attn_impl in ("flash_attention_2", "sdpa", None):
            for loader_cls in loader_classes:
                try:
                    kw: Dict[str, Any] = {
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                        "dtype": dtype,
                    }
                    if four_bit:
                        kw["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        kw["device_map"] = {"": (device_str or "cuda:0")}
                    if attn_impl is not None:
                        kw["attn_implementation"] = attn_impl
                    try:
                        full_model = loader_cls.from_pretrained(resolved_path, **kw)
                    except TypeError:
                        kw.pop("dtype", None)
                        kw["torch_dtype"] = dtype
                        full_model = loader_cls.from_pretrained(resolved_path, **kw)
                    break
                except Exception as exc:
                    last_err = exc
                    continue
            if full_model is not None:
                break

        if full_model is None:
            raise RuntimeError(f"Failed to load VLM from {resolved_path}: {last_err}")

        backbone = getattr(full_model, "model", full_model)
        if hasattr(full_model, "lm_head") and backbone is not full_model:
            try:
                del full_model.lm_head
            except Exception:
                pass

        if int(lora_rank) > 0:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if four_bit:
                backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=enable_gc)

            allowed_leafs = {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "qkv",
                "proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "linear_fc1",
                "linear_fc2",
                "fc1",
                "fc2",
                "in_proj_qkvz",
                "in_proj_ba",
                "out_proj",
            }
            exact_linear_paths: List[str] = []
            for full_name, mod in backbone.named_modules():
                if not (isinstance(mod, nn.Linear) or "Linear4bit" in type(mod).__name__):
                    continue
                nl = full_name.lower()
                if any(skip in nl for skip in ("patch_embed", "lm_head", "embed_tokens", "wte")):
                    continue
                leaf = full_name.split(".")[-1]
                if leaf in allowed_leafs:
                    exact_linear_paths.append(full_name)

            lora_cfg = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_rank) * 2,
                lora_dropout=0.05,
                target_modules=exact_linear_paths if exact_linear_paths else ["q_proj", "v_proj"],
                use_dora=bool(apply_dora),
                bias="none",
            )
            backbone = get_peft_model(backbone, lora_cfg)

            for n, p in backbone.named_parameters():
                nl = n.lower()
                if any(bk in nl for bk in ("merger", "multi_modal_projector", "aligner")) and p.is_floating_point():
                    p.requires_grad = True

            if enable_gc and hasattr(backbone, "gradient_checkpointing_enable"):
                try:
                    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    backbone.gradient_checkpointing_enable()
                if hasattr(backbone, "enable_input_require_grads"):
                    backbone.enable_input_require_grads()

        hidden_dim = None
        if hasattr(cfg, "text_config") and getattr(cfg.text_config, "hidden_size", None):
            hidden_dim = int(cfg.text_config.hidden_size)
        elif hasattr(cfg, "llm_config") and getattr(cfg.llm_config, "hidden_size", None):
            hidden_dim = int(cfg.llm_config.hidden_size)
        elif getattr(cfg, "hidden_size", None):
            hidden_dim = int(cfg.hidden_size)
        return backbone, processor, int(hidden_dim or 3584)

    def encode_multiview_tokens(
        self,
        batch_views_or_inputs: Any,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(batch_views_or_inputs, dict):
            model_inputs = batch_views_or_inputs
        else:
            batch_views = batch_views_or_inputs
            texts = [self.prompt_text_formatted for _ in batch_views]
            flat_images = [pil_v for views in batch_views for pil_v in views]
            model_inputs = self.processor(
                text=texts,
                images=flat_images,
                padding=True,
                return_tensors="pt",
            )

        moved_inputs: Dict[str, Any] = {}
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                moved_inputs[k] = v.to(device, non_blocking=True)
            else:
                moved_inputs[k] = v

        use_bf16 = device.type == "cuda"
        with torch.autocast(device_type="cuda" if use_bf16 else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = self.backbone(
                **moved_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        h_states = getattr(outputs, "hidden_states", None)
        if h_states is not None and len(h_states) >= 4:
            n_layers = len(h_states)
            idx_mid = max(0, n_layers // 2)
            idx_3q = max(0, (3 * n_layers) // 4)
            fused = (
                h_states[idx_mid].float() + h_states[idx_3q].float() + h_states[-1].float()
            ) / 3.0
        else:
            last_h = getattr(outputs, "last_hidden_state", h_states[-1])
            fused = last_h.float()

        attn_mask = moved_inputs.get("attention_mask", None)
        bsz = fused.shape[0]
        pooled_list = []
        for b in range(bsz):
            if attn_mask is not None:
                valid_toks = fused[b, attn_mask[b].bool()]
                if valid_toks.shape[0] == 0:
                    valid_toks = fused[b]
            else:
                valid_toks = fused[b]

            if valid_toks.shape[0] > 64:
                p63 = F.adaptive_avg_pool1d(
                    valid_toks.unsqueeze(0).transpose(1, 2), output_size=63
                ).transpose(1, 2).squeeze(0)
                seq_b = torch.cat([p63, valid_toks[-1:, :]], dim=0)
            else:
                pad_n = 64 - valid_toks.shape[0]
                if pad_n > 0:
                    seq_b = torch.cat([valid_toks, valid_toks[-1:, :].expand(pad_n, -1)], dim=0)
                else:
                    seq_b = valid_toks
            pooled_list.append(seq_b)

        return torch.stack(pooled_list, dim=0)

    def forward(
        self,
        batch_views: Any,
        batch_ids: Optional[torch.Tensor] = None,
        apply_batch_bias: Optional[bool] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        dev = device or next(self.head.parameters()).device
        tokens = self.encode_multiview_tokens(batch_views, dev)
        return self.head(tokens, batch_ids=batch_ids, apply_batch_bias=apply_batch_bias)


def build_c233397_pillar_for_4090(
    ckpt_path: str,
    pillar: str = "pillar1",
    device: str = "cuda",
    bnb_path: Optional[str] = None,
    four_bit: bool = True,
    verbose: bool = True,
) -> NativeVLMPillarModelC233397:
    """
    Exact 4090 deployment builder for `c233397` checkpoints:
      1. Builds `NativeVLMPillarModelC233397` (exact 64-token avg+summary readout, `r=24` DoRA).
      2. Asserts checkpoint `recipe == "c233397_qy8"` and zero missing/unexpected adapter/head keys.
      3. Merges DoRA adapters into bf16 base weights (`merge_and_unload`).
      4. Optionally quantizes LLM `nn.Linear` layers in-place to 4-bit NF4 (keeping vision + merger in bf16).
    """
    if bnb_path and bnb_path not in sys.path:
        sys.path.append(bnb_path)
    from phoenix_pillar_4090 import quantize_llm_inplace

    dev = torch.device(device)
    ck = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ck, dict) and "recipe" in ck:
        assert ck["recipe"] == RECIPE_TAG, f"Expected recipe={RECIPE_TAG}, got {ck['recipe']}"
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    lora_r = 24
    for k, v in sd.items():
        if "lora_a" in str(k).lower() and hasattr(v, "ndim") and v.ndim == 2:
            lora_r = int(v.shape[0])
            break

    m = NativeVLMPillarModelC233397(
        pillar_name=pillar,
        apply_dora=True,
        lora_rank=lora_r,
        enable_gc=False,
    ).to(dev)
    res = m.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if "lora" in k.lower() or k.startswith("head")] + list(res.unexpected_keys)
    assert not bad, f"c233397 checkpoint mismatch: {bad[:5]}"
    m.backbone = m.backbone.merge_and_unload()
    nq = quantize_llm_inplace(m.backbone, dev) if four_bit else 0
    torch.cuda.empty_cache()
    m.eval()
    if verbose:
        print(
            f"[build_c233397_pillar_for_4090] {pillar} ({RECIPE_TAG}): merged, {nq} linears 4-bit={four_bit}, "
            f"{torch.cuda.memory_allocated(dev) / 2 ** 30:.2f} GB on {dev}",
            flush=True,
        )
    return m


class NativeMultiViewVLMDatasetC233397(Dataset):
    """
    Yields `[global_view, crop0, crop1]` with pixel-exact horizontal-flip symmetry:
    flipping the unflipped views `[v.transpose(FLIP_LEFT_RIGHT) for v in views]` guarantees
    100% pixel identity between OOF validation (`force_hflip=True`) and `final_engine.py`!
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        resolver: PureRealImageResolver,
        box_cache: Dict[str, np.ndarray],
        is_train: bool = True,
        force_hflip: bool = False,
        max_global_pixels: int = 784 * 784,
    ) -> None:
        self.rows = list(rows)
        self.resolver = resolver
        self.box_cache = box_cache
        self.is_train = is_train
        self.force_hflip = force_hflip
        self.max_global_pixels = max_global_pixels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        from PIL import Image

        row = self.rows[idx]
        nm = str(row["name"])
        pil_img = self.resolver.load_rgb_pil(nm)
        boxes = self.box_cache[nm]

        crop0 = crop_1to1_from_box(pil_img, boxes[0], patch_size=384)
        crop1 = crop_1to1_from_box(pil_img, boxes[1], patch_size=384)
        global_view = resize_native_aspect_pil(
            pil_img,
            max_pixels=self.max_global_pixels,
            grid_factor=28,
        )
        views = [global_view, crop0, crop1]

        do_flip = self.force_hflip or (self.is_train and np.random.rand() < 0.5)
        if do_flip:
            views = [v.transpose(Image.FLIP_LEFT_RIGHT) for v in views]

        return {
            "name": nm,
            "views": views,
            "batch_id": int(row.get("batch_id", 0)),
            "noisy_flag": int(row.get("noisy_flag", 0)),
            "grades": np.asarray(row.get("grades", [0] * 6), dtype=np.int64),
        }


class FastProcessorCollate:
    """Runs deterministic VLM `processor(text, images)` inside CPU DataLoader workers."""

    def __init__(self, processor: Any, prompt_text_formatted: str) -> None:
        self.processor = processor
        self.prompt_text_formatted = prompt_text_formatted

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, object]:
        texts = [self.prompt_text_formatted for _ in batch]
        flat_images = [pil_v for b in batch for pil_v in b["views"]]  # type: ignore[index]
        model_inputs = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )
        return {
            "names": [str(b["name"]) for b in batch],
            "model_inputs": dict(model_inputs),
            "batch_ids": torch.tensor([int(b["batch_id"]) for b in batch], dtype=torch.long),
            "noisy_flags": torch.tensor([int(b["noisy_flag"]) for b in batch], dtype=torch.long),
            "grades": torch.from_numpy(np.stack([b["grades"] for b in batch], axis=0)).long(),
        }


@torch.no_grad()
def predict_single_orientation(
    model: NativeVLMPillarModelC233397,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    names_all: List[str] = []
    probs_list, logits_list, clean_l_list, p_clean_list = [], [], [], []
    mu_list, var_list, emb_list, gold_list = [], [], [], []
    for batch in loader:
        names_all.extend(batch["names"])
        inp = batch["model_inputs"] if "model_inputs" in batch else batch["views"]
        out = model(
            batch_views=inp,
            batch_ids=None,
            apply_batch_bias=False,
            device=device,
        )
        probs_list.append(out["probs_6x4"].cpu().numpy())
        logits_list.append(out["logits_6x4"].cpu().numpy())
        clean_l_list.append(out["clean_logit"].cpu().numpy())
        p_clean_list.append(out["p_clean_joint"].cpu().numpy())
        mu_list.append(out["mu_6d"].cpu().numpy())
        var_list.append(out["var_6d"].cpu().numpy())
        emb_list.append(out["quality_emb"].cpu().numpy())
        gold_list.append(batch["grades"].numpy())
    return {
        "names": np.array(names_all, dtype=object),
        "probs_6x4": np.concatenate(probs_list, axis=0).astype(np.float32),
        "logits_6x4": np.concatenate(logits_list, axis=0).astype(np.float32),
        "clean_logit": np.concatenate(clean_l_list, axis=0).astype(np.float32),
        "p_clean_joint": np.concatenate(p_clean_list, axis=0).astype(np.float32),
        "mu_6d": np.concatenate(mu_list, axis=0).astype(np.float32),
        "var_6d": np.concatenate(var_list, axis=0).astype(np.float32),
        "quality_emb": np.concatenate(emb_list, axis=0).astype(np.float32),
        "gold_6d": np.concatenate(gold_list, axis=0).astype(np.int64),
    }


def predict_2x_hflip_separate(
    model: NativeVLMPillarModelC233397,
    loader_orig: DataLoader,
    loader_hflip: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    p0 = predict_single_orientation(model, loader_orig, device)
    p1 = predict_single_orientation(model, loader_hflip, device)
    assert list(p0["names"]) == list(p1["names"])
    probs = 0.5 * (p0["probs_6x4"] + p1["probs_6x4"])
    logits = np.log(np.clip(probs, 1e-6, 1.0))
    for d in P0_ONLY_DIMS:
        logits[:, d, 1] = -1e4
        logits[:, d, 2] = -1e4
    emb = p0["quality_emb"] + p1["quality_emb"]
    emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
    grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 1, 4)
    mu = (probs * grid).sum(axis=-1)
    var = (probs * (grid - mu[:, :, None]) ** 2).sum(axis=-1)
    return {
        "names": p0["names"],
        "probs_orig_6x4": p0["probs_6x4"].astype(np.float32),
        "probs_hflip_6x4": p1["probs_6x4"].astype(np.float32),
        "probs_6x4": probs.astype(np.float32),
        "logits_6x4": logits.astype(np.float32),
        "clean_logit": (0.5 * (p0["clean_logit"] + p1["clean_logit"])).astype(np.float32),
        "p_clean_joint": (0.5 * (p0["p_clean_joint"] + p1["p_clean_joint"])).astype(np.float32),
        "mu_6d": mu.astype(np.float32),
        "var_6d": var.astype(np.float32),
        "quality_emb": emb.astype(np.float32),
        "gold_6d": p0["gold_6d"],
    }


def train_c233397_run(
    model: NativeVLMPillarModelC233397,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 4,
    lr_head: float = 2.0e-4,
    lr_backbone: float = 8.0e-5,
    grad_accum: int = 2,
    swa_start_epoch: Optional[int] = 2,
    max_steps_debug: int = 0,
) -> Optional[Dict[str, torch.Tensor]]:
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    bb_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]

    opt = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head, "weight_decay": 1e-2},
            {"params": bb_params, "lr": lr_backbone, "weight_decay": 1e-2},
        ]
    )
    grad_accum = max(1, int(grad_accum))
    opt_steps_per_ep = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_steps = max(1, epochs * opt_steps_per_ep)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_backbone * 0.1)

    deqa_loss_fn = DeQAOrdinalDistributionLossC233397().to(device)
    rl2r_loss_fn = CrossBatchThurstoneRL2RLossC233397(queue_size=1024, emb_dim=256).to(device)
    swa = BasinAlignedSWACollectorC233397() if swa_start_epoch is not None else None
    trainable_all = [p for p in model.parameters() if p.requires_grad]

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        n_steps_run = 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader):
            y_6d = batch["grades"].to(device, non_blocking=True)
            b_ids = batch["batch_ids"].to(device, non_blocking=True)
            noisy = batch["noisy_flags"].to(device, non_blocking=True)
            inp = batch["model_inputs"] if "model_inputs" in batch else batch["views"]

            out = model(
                batch_views=inp,
                batch_ids=b_ids,
                apply_batch_bias=True,
                device=device,
            )
            losses = deqa_loss_fn(
                head_out=out,
                y_6d=y_6d,
                noisy_flags=noisy,
                supp_bias_param=model.head.supp_batch_bias,
            )
            loss = losses["loss"]
            if ep >= 1:
                loss = loss + 0.35 * rl2r_loss_fn(out, y_6d)

            (loss / grad_accum).backward()
            if (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_all, max_norm=1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            ep_loss += float(loss.item())
            n_steps_run += 1
            if (step_idx + 1) % 100 == 0:
                print(
                    f"    [c233397 {model.pillar_name}] Ep {ep+1}/{epochs} step {step_idx+1}/{len(train_loader)} "
                    f"| avg_loss={ep_loss/n_steps_run:.4f} | peak_vram={torch.cuda.max_memory_allocated(device)/2**30:.1f}GB "
                    f"({time.time()-t0:.0f}s)",
                    flush=True,
                )
            if max_steps_debug > 0 and n_steps_run >= max_steps_debug:
                break

        if swa is not None and ep >= swa_start_epoch:
            swa.update(model, weight=1.0)
        print(
            f"  [c233397 {model.pillar_name}] Epoch {ep+1}/{epochs} done in {time.time()-t0:.1f}s "
            f"| loss={ep_loss / max(1, n_steps_run):.4f} | peak_vram={torch.cuda.max_memory_allocated(device)/2**30:.1f}GB",
            flush=True,
        )

    return swa.export_state_dict() if (swa is not None and swa.count > 0) else None


def verify_qy8_alignment(rows: Sequence[Dict[str, object]]) -> None:
    import common55 as c

    names_c, gold_c, fold_c, _ = c.load_split()
    assert len(rows) == len(names_c) == 3573, f"Expected 3573 rows, got {len(rows)}"
    mp = {str(r["name"]): r for r in rows}
    for i, nm in enumerate(names_c):
        r = mp[str(nm)]
        assert int(r["fold"]) == int(fold_c[i]), f"Fold mismatch on {nm}: {r['fold']} vs {fold_c[i]}"
        assert list(r["grades"]) == list(gold_c[i]), f"Grade mismatch on {nm}"


def aggregate_qy8_oof(
    rows: Sequence[Dict[str, object]],
    seed: int,
    out_dir: str,
    pillar: str = "pillar1",
) -> str:
    import common55 as c
    from eval_specialist_gate import per_dim_ap, rps_6x4

    names_c, gold_c, fold_c, _ = c.load_split()
    n = len(names_c)
    name_to_pos = {str(nm): i for i, nm in enumerate(names_c)}

    oof_orig = np.zeros((n, 6, 4), dtype=np.float32)
    oof_hflip = np.zeros((n, 6, 4), dtype=np.float32)
    oof_probs = np.zeros((n, 6, 4), dtype=np.float32)
    oof_logits = np.zeros((n, 6, 4), dtype=np.float32)
    oof_clean_l = np.zeros(n, dtype=np.float32)
    oof_p_clean = np.zeros(n, dtype=np.float32)
    oof_mu = np.zeros((n, 6), dtype=np.float32)
    oof_var = np.zeros((n, 6), dtype=np.float32)
    oof_emb = np.zeros((n, 256), dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)

    found_folds = []
    for f_idx in range(8):
        f_path = os.path.join(out_dir, f"p1_c233397_s{seed}_fold{f_idx}.npz")
        if not os.path.exists(f_path):
            continue
        found_folds.append(f_idx)
        part = np.load(f_path, allow_pickle=True)
        dest_idx = [name_to_pos[str(x)] for x in part["names"]]
        oof_orig[dest_idx] = part["probs_orig_6x4"]
        oof_hflip[dest_idx] = part["probs_hflip_6x4"]
        oof_probs[dest_idx] = part["probs_6x4"]
        oof_logits[dest_idx] = part["logits_6x4"]
        oof_clean_l[dest_idx] = part["clean_logit"]
        oof_p_clean[dest_idx] = part["p_clean_joint"]
        oof_mu[dest_idx] = part["mu_6d"]
        oof_var[dest_idx] = part["var_6d"]
        oof_emb[dest_idx] = part["quality_emb"]
        valid_mask[dest_idx] = True

    assert len(found_folds) > 0, f"No completed fold files found in {out_dir} for seed={seed}"
    out_npz = os.path.join(out_dir, f"p1_c233397_s{seed}_oof.npz")
    np.savez_compressed(
        out_npz,
        recipe=RECIPE_TAG,
        seed=np.int64(seed),
        completed_folds=np.array(found_folds, dtype=np.int64),
        valid_mask=valid_mask,
        names=np.array(names_c, dtype=object),
        folds=fold_c.astype(np.int64),
        gold_6d=gold_c.astype(np.int64),
        probs_orig_6x4=oof_orig,
        probs_hflip_6x4=oof_hflip,
        probs_6x4=oof_probs,
        logits_6x4=oof_logits,
        clean_logit=oof_clean_l,
        p_clean_joint=oof_p_clean,
        mu_6d=oof_mu,
        var_6d=oof_var,
        quality_emb=oof_emb,
    )
    pv = c.renorm(oof_probs[valid_mask].astype(np.float64))
    gv = gold_c[valid_mask]
    m = evaluate_official_metrics(np.argmax(pv, axis=-1), gv)
    _, map14 = per_dim_ap(pv, gv)
    print(
        f"=== [c233397 {pillar} s{seed} OOF ({len(found_folds)}/8 folds, N={valid_mask.sum()})] "
        f"RPS={rps_6x4(pv, gv):.5f} | mAP14={map14:.4f} | Argmax Doc2S={m['strict_total']:.2f} -> {out_npz} ===",
        flush=True,
    )
    return out_npz


def run_real_gpu_smoke(gpu: int, enable_gc: bool) -> None:
    set_deterministic_seed(2026)
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(dev)

    rows = load_split_rows(os.path.join(REPO, "data/phoenix_splits/phoenix_qy8.csv"))
    verify_qy8_alignment(rows)
    resolver = PureRealImageResolver()
    box_cache = get_or_build_scrutiny_box_cache(
        names=[str(r["name"]) for r in rows],
        resolver=resolver,
        device_str=str(dev),
        cache_npz_path=os.path.join(REPO, "results/phoenix/scrutiny_boxes_384_cache.npz"),
    )

    sub_rows = rows[:16]
    shift = compute_fold_batch_shift(rows[:500])
    model = NativeVLMPillarModelC233397(
        pillar_name="pillar1",
        init_batch_shift=shift,
        apply_dora=True,
        lora_rank=24,
        enable_gc=enable_gc,
    ).to(dev)
    collate = FastProcessorCollate(model.processor, model.prompt_text_formatted)
    ds_tr = NativeMultiViewVLMDatasetC233397(sub_rows, resolver, box_cache, is_train=True)
    dl_tr = DataLoader(ds_tr, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate)

    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    swa_sd = train_c233397_run(
        model, dl_tr, dev, epochs=2, lr_head=2.0e-4, lr_backbone=8.0e-5, grad_accum=2, swa_start_epoch=0, max_steps_debug=4
    )
    dt = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    assert swa_sd is not None
    model.load_state_dict(swa_sd, strict=False)

    ds_va0 = NativeMultiViewVLMDatasetC233397(sub_rows[:4], resolver, box_cache, is_train=False, force_hflip=False)
    ds_va1 = NativeMultiViewVLMDatasetC233397(sub_rows[:4], resolver, box_cache, is_train=False, force_hflip=True)
    dl_va0 = DataLoader(ds_va0, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)
    dl_va1 = DataLoader(ds_va1, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)
    va_out = predict_2x_hflip_separate(model, dl_va0, dl_va1, dev)

    tmp_ckpt = f"/tmp/smoke_c233397_{os.getpid()}.pt"
    torch.save({"pillar": "pillar1", "recipe": RECIPE_TAG, "state_dict": swa_sd}, tmp_ckpt)
    del model
    torch.cuda.empty_cache()

    # Parity test 1: BF16 merged deploy path (batch=1 raw PIL views, just like final_engine.py!)
    m_bf16 = build_c233397_pillar_for_4090(
        tmp_ckpt, pillar="pillar1", device=str(dev), bnb_path=os.path.join(REPO, "third_party/pydeps"), four_bit=False, verbose=True
    )
    p_bf16 = []
    with torch.no_grad():
        for i in range(4):
            v_orig = ds_va0[i]["views"]
            v_flip = ds_va1[i]["views"]
            o0 = m_bf16([v_orig], batch_ids=None, apply_batch_bias=False, device=dev)["probs_6x4"].float().cpu().numpy()[0]
            o1 = m_bf16([v_flip], batch_ids=None, apply_batch_bias=False, device=dev)["probs_6x4"].float().cpu().numpy()[0]
            p_bf16.append(0.5 * (o0 + o1))
    p_bf16_arr = np.stack(p_bf16)
    max_dp_bf16 = float(np.max(np.abs(p_bf16_arr - va_out["probs_6x4"])))
    del m_bf16
    torch.cuda.empty_cache()

    # Parity test 2: 4-bit NF4 merged deploy path
    m_q4 = build_c233397_pillar_for_4090(
        tmp_ckpt, pillar="pillar1", device=str(dev), bnb_path=os.path.join(REPO, "third_party/pydeps"), four_bit=True, verbose=True
    )
    p_q4 = []
    with torch.no_grad():
        for i in range(4):
            v_orig = ds_va0[i]["views"]
            v_flip = ds_va1[i]["views"]
            o0 = m_q4([v_orig], batch_ids=None, apply_batch_bias=False, device=dev)["probs_6x4"].float().cpu().numpy()[0]
            o1 = m_q4([v_flip], batch_ids=None, apply_batch_bias=False, device=dev)["probs_6x4"].float().cpu().numpy()[0]
            p_q4.append(0.5 * (o0 + o1))
    p_q4_arr = np.stack(p_q4)
    tv_q4 = float(0.5 * np.abs(p_q4_arr - va_out["probs_6x4"]).sum(axis=-1).mean())
    os.remove(tmp_ckpt)

    print(
        f"[SMOKE PASS] enable_gc={enable_gc} | 8 steps in {dt:.1f}s | train_peak={peak_gb:.2f}GB | "
        f"bf16_max|dp|={max_dp_bf16:.5f} (<=0.01) | q4_TV={tv_q4:.5f} (<=0.012)",
        flush=True,
    )
    assert max_dp_bf16 <= 0.01, f"BF16 parity failed: {max_dp_bf16}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pillar", choices=["pillar1", "pillar2"], default="pillar1")
    ap.add_argument("--model-path", default="", help="Optional override for local/HF VLM base model path")
    ap.add_argument("--lora-rank", type=int, default=24)
    ap.add_argument("--no-dora", action="store_true", help="Use standard LoRA instead of DoRA (saves ~35GB VRAM on >=38B models)")
    ap.add_argument("--four-bit", action="store_true", help="Load backbone in 4-bit NF4 QLoRA (via third_party/pydeps bitsandbytes)")
    ap.add_argument("--mode", choices=["qy8", "fullfit"], default="qy8")
    ap.add_argument("--split-csv", default=os.path.join(REPO, "data/phoenix_splits/phoenix_qy8.csv"))
    ap.add_argument("--box-cache", default=os.path.join(REPO, "results/phoenix/scrutiny_boxes_384_cache.npz"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results/r55/vlm_qy8"))
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO, "results/r55/vlm_qy8/ckpts"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr-head", type=float, default=2.0e-4)
    ap.add_argument("--lr-backbone", type=float, default=8.0e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-grad-ckpt", action="store_true", help="Disable gradient checkpointing on 141GB H200 for ~40% faster training")
    ap.add_argument("--aggregate-qy8", action="store_true")
    ap.add_argument("--smoke-gpu", type=int, default=-1)
    args = ap.parse_args()

    if args.smoke_gpu >= 0:
        run_real_gpu_smoke(args.smoke_gpu, enable_gc=not args.no_grad_ckpt)
        return

    rows = load_split_rows(args.split_csv)
    verify_qy8_alignment(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.aggregate_qy8:
        aggregate_qy8_oof(rows, args.seed, args.out_dir, pillar=args.pillar)
        return

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    resolver = PureRealImageResolver()
    box_cache = get_or_build_scrutiny_box_cache(
        names=[str(r["name"]) for r in rows],
        resolver=resolver,
        device_str=str(device),
        cache_npz_path=args.box_cache,
    )

    if args.mode == "qy8":
        f_idx = int(args.fold)
        assert 0 <= f_idx < 8
        fold_npz = os.path.join(args.out_dir, f"p1_c233397_s{args.seed}_fold{f_idx}.npz")
        fold_ckpt = os.path.join(args.ckpt_dir, f"p1_c233397_s{args.seed}_fold{f_idx}.pt")
        if os.path.exists(fold_npz) and os.path.exists(fold_ckpt):
            print(f"[Resume] Fold {f_idx} (seed={args.seed}) already done: {fold_npz}", flush=True)
            return

        set_deterministic_seed(args.seed + f_idx)
        tr_rows = [r for r in rows if int(r["fold"]) != f_idx]
        va_rows = [r for r in rows if int(r["fold"]) == f_idx]
        fold_shift = compute_fold_batch_shift(tr_rows)

        print(
            f"=== [c233397 {args.pillar} qy8] Fold {f_idx}/8 | seed={args.seed} | train={len(tr_rows)} val={len(va_rows)} "
            f"| gc={not args.no_grad_ckpt} | gpu={args.gpu} ===",
            flush=True,
        )
        model = NativeVLMPillarModelC233397(
            pillar_name=args.pillar,
            model_id_or_path=(args.model_path or None),
            init_batch_shift=fold_shift,
            apply_dora=not args.no_dora,
            lora_rank=args.lora_rank,
            enable_gc=not args.no_grad_ckpt,
            four_bit=args.four_bit,
            device_str=str(device),
        )
        if args.four_bit:
            model.head.to(device)
        else:
            model = model.to(device)
        collate = FastProcessorCollate(model.processor, model.prompt_text_formatted)

        ds_tr = NativeMultiViewVLMDatasetC233397(tr_rows, resolver, box_cache, is_train=True)
        ds_va0 = NativeMultiViewVLMDatasetC233397(va_rows, resolver, box_cache, is_train=False, force_hflip=False)
        ds_va1 = NativeMultiViewVLMDatasetC233397(va_rows, resolver, box_cache, is_train=False, force_hflip=True)

        dl_tr = DataLoader(
            ds_tr,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=(args.workers > 0),
            prefetch_factor=2 if args.workers > 0 else None,
            collate_fn=collate,
        )
        dl_va0 = DataLoader(
            ds_va0, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate
        )
        dl_va1 = DataLoader(
            ds_va1, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate
        )

        swa_sd = train_c233397_run(
            model=model,
            train_loader=dl_tr,
            device=device,
            epochs=args.epochs,
            lr_head=args.lr_head,
            lr_backbone=args.lr_backbone,
            grad_accum=args.grad_accum,
            swa_start_epoch=max(1, args.epochs - 2),
        )
        assert swa_sd is not None
        model.load_state_dict(swa_sd, strict=False)

        free_gb = shutil.disk_usage(args.ckpt_dir).free / (1024.0 ** 3)
        if free_gb >= 80.0 or f_idx == 0:
            torch.save(
                {"pillar": args.pillar, "recipe": RECIPE_TAG, "seed": args.seed, "fold": f_idx, "state_dict": swa_sd},
                fold_ckpt,
            )

        va_out = predict_2x_hflip_separate(model, dl_va0, dl_va1, device)
        np.savez_compressed(fold_npz, recipe=RECIPE_TAG, seed=np.int64(args.seed), fold=np.int64(f_idx), **va_out)

        from eval_specialist_gate import rps_6x4
        import common55 as c

        va_gold = np.array([r["grades"] for r in va_rows], dtype=np.int64)
        m = evaluate_official_metrics(np.argmax(va_out["probs_6x4"], axis=-1), va_gold)
        rps_v = rps_6x4(c.renorm(va_out["probs_6x4"].astype(np.float64)), va_gold)
        print(
            f"[c233397 {args.pillar} s{args.seed} Fold {f_idx}] RPS={rps_v:.5f} | Argmax Strict={m['strict_total']:.2f} "
            f"(Pos={m['strict_S_pos_pts']:.2f}, P0={m['strict_S_p0_pts']:.2f}, Clean={m['strict_S_clean_pts']:.2f}) -> {fold_npz}",
            flush=True,
        )

    elif args.mode == "fullfit":
        ckpt_path = os.path.join(args.ckpt_dir, f"p1_c233397_s{args.seed}_fullfit_swa.pt")
        if os.path.exists(ckpt_path):
            print(f"[Resume] Full-fit checkpoint already exists: {ckpt_path}", flush=True)
            return

        set_deterministic_seed(args.seed)
        full_shift = compute_fold_batch_shift(rows)
        print(
            f"=== [c233397 {args.pillar} FULL-FIT] seed={args.seed} | N={len(rows)} | epochs={args.epochs} (SWA ep 3-4) "
            f"| gc={not args.no_grad_ckpt} | gpu={args.gpu} ===",
            flush=True,
        )
        model = NativeVLMPillarModelC233397(
            pillar_name=args.pillar,
            model_id_or_path=(args.model_path or None),
            init_batch_shift=full_shift,
            apply_dora=not args.no_dora,
            lora_rank=args.lora_rank,
            enable_gc=not args.no_grad_ckpt,
            four_bit=args.four_bit,
            device_str=str(device),
        )
        if args.four_bit:
            model.head.to(device)
        else:
            model = model.to(device)
        collate = FastProcessorCollate(model.processor, model.prompt_text_formatted)
        ds_full = NativeMultiViewVLMDatasetC233397(rows, resolver, box_cache, is_train=True)
        dl_full = DataLoader(
            ds_full,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=(args.workers > 0),
            prefetch_factor=2 if args.workers > 0 else None,
            collate_fn=collate,
        )
        swa_sd = train_c233397_run(
            model=model,
            train_loader=dl_full,
            device=device,
            epochs=args.epochs,
            lr_head=args.lr_head,
            lr_backbone=args.lr_backbone,
            grad_accum=args.grad_accum,
            swa_start_epoch=max(1, args.epochs - 2),
        )
        assert swa_sd is not None
        torch.save(
            {"pillar": args.pillar, "recipe": RECIPE_TAG, "seed": args.seed, "fold": -1, "state_dict": swa_sd},
            ckpt_path,
        )
        print(f"=== [c233397 {args.pillar} FULL-FIT s{args.seed}] Saved -> {ckpt_path} ===", flush=True)


if __name__ == "__main__":
    main()
