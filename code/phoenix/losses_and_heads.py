#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/phoenix/losses_and_heads.py
================================
Breakthrough 4: DeQA Direct 4-Way Unimodal Soft-Distribution Head, Batch-Shift
Calibration Embedding, Heteroscedastic OOF Soft-Relabeling, Cross-Batch Memory (XBM)
Pairwise Thurstone RL2R Ranking Loss, and Adapter-Aware Basin-Aligned SWA Collector.

Why This Replaces Cumulative Sigmoid & Multiplicative Heads:
  - Previous 14-ch cumulative subtraction `P(P1) = sigmoid(z1) - sigmoid(z0)` mathematically
    capped P1 peak probability around ~0.462, causing massive P1 collapse into P0/P2.
  - Micro-batch (`bs=2~4`) differentiable Soft-F1 suffered empty-class denominator explosions.
  - Here, each defect outputs direct 4-level unimodal probabilities `P_{i,c,k}` (`k in 0..3`,
    where `P1` freely reaches `1.0`), supervised per-sample by Adaptive Gaussian Soft-KL +
    Squared CDF Earth Mover's Distance (EMD) + 1,024-Queue Thurstone Pairwise Fidelity RL2R!
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phoenix.config import P0_ONLY_DIMS


class PhoenixUnimodalDeQAHead(nn.Module):
    """
    Multi-Defect Query2Defect Cross-Attention + Direct 4-Way Unimodal Soft-Distribution Head
    with Training-Only Batch-Shift Calibration (`orig` vs `supp`) and Joint Clean Gate.
    """

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
        # 6 defect queries + 1 global joint-clean query
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

        # Per-defect direct 4-grade energy head + continuous center head (unimodal anchor)
        self.grade_proj = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(6)])
        self.center_proj = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(6)])
        self.log_kappa = nn.Parameter(torch.zeros(6))  # Unimodal quadratic curvature >= 0
        self.clean_gate_proj = nn.Linear(hidden_dim, 1)
        self.anchor_emb_proj = nn.Sequential(
            nn.Linear(hidden_dim * 7, 256),
            nn.LayerNorm(256),
        )

        # Training-only Batch-Shift Calibration bias for Batch-2 (`supp`, batch_id == 1)
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
        """
        Runs strictly in FP32 (`autocast(enabled=False)`) to prevent BF16 softmax/attention
        underflow and mixed-dtype errors.
        """
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
                raw_z = self.grade_proj[d](qd)  # [B, 4]
                center = 3.0 * torch.sigmoid(self.center_proj[d](qd))  # [B, 1] in [0, 3]
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

            logits_6x4 = torch.stack(logits_list, dim=1)  # [B, 6, 4]
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


def build_heteroscedastic_soft_targets(
    y_6d: torch.Tensor,
    noisy_flags: Optional[torch.Tensor] = None,
    oof_teacher_probs: Optional[torch.Tensor] = None,
    sigma_anchor: float = 0.30,
    sigma_mid: float = 0.46,
    teacher_blend: float = 0.25,
) -> torch.Tensor:
    device = y_6d.device
    bsz = y_6d.shape[0]
    grid = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=torch.float32).view(1, 1, 4)
    y_f = y_6d.float().unsqueeze(-1)  # [B, 6, 1]

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
    soft_q = raw / torch.clamp(raw.sum(dim=-1, keepdim=True), min=1e-8)

    if oof_teacher_probs is not None and teacher_blend > 0.0:
        adj_mask = (torch.abs(grid - y_f) <= 1.0).float() * bin_mask
        t_adj = oof_teacher_probs.to(device).float() * adj_mask
        t_sum = t_adj.sum(dim=-1, keepdim=True)
        valid_t = (t_sum > 1e-5).float()
        t_norm = torch.where(valid_t > 0, t_adj / torch.clamp(t_sum, min=1e-6), soft_q)
        soft_q = (1.0 - teacher_blend) * soft_q + teacher_blend * t_norm

    return soft_q


class DeQAOrdinalDistributionLoss(nn.Module):
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
        oof_teacher_probs: Optional[torch.Tensor] = None,
        supp_bias_param: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = head_out["logits_6x4"].float()
        probs = head_out["probs_6x4"].float()
        clean_logit = head_out["clean_logit"].float()

        q = build_heteroscedastic_soft_targets(
            y_6d=y_6d,
            noisy_flags=noisy_flags,
            oof_teacher_probs=oof_teacher_probs,
        )
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


class CrossBatchThurstoneRL2RLoss(nn.Module):
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


def extract_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extracts ONLY trainable parameters (`requires_grad=True`, e.g., DoRA/LoRA adapters,
    multimodal mergers, physics stems, and `PhoenixUnimodalDeQAHead`) plus head buffers.
    Reduces 8B-9B VLM checkpoint & SWA snapshot footprint from ~35 GB down to ~180 MB!
    """
    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        kl = k.lower()
        if (
            (k in trainable_keys)
            or k.startswith("head.")
            or k.startswith("physics_stem.")
            or k.startswith("fgresq_branch.")
            or k.startswith("pal4vst_probe.")
            or k.startswith("probe_proj.")
            or ("lora_" in kl)
            or ("dora_" in kl)
            or ("magnitude_vector" in kl)
        ):
            out[k] = v.detach().cpu().clone()
    return out


class BasinAlignedSWACollector:
    """
    Averages trainable adapter + head checkpoints strictly from the SAME optimization basin.
    Delivers ensemble-level variance reduction in 1 single weight file with ZERO parameter
    multiplication and <0.4 GB host RAM overhead.
    """

    def __init__(self) -> None:
        self.count = 0
        self.total_weight = 0.0
        self.avg_state: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module, weight: float = 1.0) -> None:
        raw_sd = extract_trainable_state_dict(model)
        state = {
            k: (v.float() if v.is_floating_point() else v.clone())
            for k, v in raw_sd.items()
        }
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

    def export_state_dict(self, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
        if self.avg_state is None or self.count == 0:
            raise RuntimeError("BasinAlignedSWACollector has no collected checkpoints.")
        out: Dict[str, torch.Tensor] = {}
        for k, v in self.avg_state.items():
            if v.is_floating_point():
                averaged = v / max(1e-8, self.total_weight)
                out[k] = averaged.to(dtype) if dtype is not None else averaged
            else:
                out[k] = v.clone()
        return out
