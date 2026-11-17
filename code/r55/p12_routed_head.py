#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/p12_routed_head.py
===========================
P1/P2 Architecture & Loss Upgrade:
  1. `PerDefectRoutedOrdinalHead`:
     - Keeps multi-view features `[B, V, D_vis]` uncollapsed (`[global, subtitle_ocr, face_anat, hand_swirl, bokeh_halo]`).
     - Learnable view-role embeddings + physics token -> 7 defect/clean queries route evidence via Multi-Head
       Cross-Attention + 2-layer Cross-Defect Self-Attention.
     - Unconstrained 4-class readout per graded defect (`模糊, 纹理损失, 过锐, 过度生成`) + weak monotonic prior residual
       (breaks the 1D quadratic bottleneck so localized P2 caption/hand flaws and global P0 collapse live in
       independent subspaces) + Pure-Real `TipAdapterPrototypeBank(128)`.
  2. `BoundaryMarginXBMLoss`:
     - Cross-batch memory queue (`M=512`, strictly `group_i != group_j` across videos) storing `[M, 6, 4]` logits.
     - Directly optimizes the 3 exact Bayes decision boundaries across videos:
         (a) Detection log-odds: `logsumexp(z_{1..3}) - z_0` (`Pos` vs `Clean`)
         (b) Moderate+ log-odds: `logsumexp(z_{2..3}) - logsumexp(z_{0..1})` (`P1/P0` vs `Clean/P2`)
         (c) Severe P0 log-odds: `z_3 - z_2` (`P0` vs `P1` — directly targets P1<->P0 cannibalization & 20-pt S_p0)
  3. `P12CompositeOrdinalLoss`:
     - Combines Soft-Ordinal KL + Squared-EMD (RPS) + `BoundaryMarginXBMLoss` + Clean Cliff Suppressor
       + optional Leak-Free Nested Cross-Family Teacher Soft-Target Distillation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qingyu.config_and_paths import FULL_DIMS, P0_ONLY_DIMS
from qingyu.unimodal_xbm_losses import TipAdapterPrototypeBank


class PerDefectRoutedOrdinalHead(nn.Module):
    def __init__(
        self,
        single_view_dim: int,
        phys_dim: int = 48,
        hidden_dim: int = 512,
        proj_dim: int = 512,
        num_heads: int = 8,
        max_views: int = 8,
        dropout: float = 0.20,
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
        self.view_role_emb = nn.Parameter(torch.randn(1, max_views, hidden_dim) * 0.02)

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

        # 6 defect queries + 1 all-clean gate query = 7 queries
        self.queries = nn.Parameter(torch.randn(1, 7, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout * 0.5, batch_first=True)
        self.norm_c1 = nn.LayerNorm(hidden_dim)
        self.ffn_c1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_c2 = nn.LayerNorm(hidden_dim)

        # Cross-defect self-attention (couples 模糊<->虚化突兀, 过锐<->过度生成字幕, 纹理损失<->模糊)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout * 0.5, batch_first=True)
        self.norm_s1 = nn.LayerNorm(hidden_dim)

        # Unconstrained per-defect 4-class linear heads + weak soft unimodal anchor residual
        self.grade_heads = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim, 4)) for _ in range(6)
        ])
        self.center_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(6)])
        self.log_kappa = nn.Parameter(torch.full((6,), -1.2, dtype=torch.float32))  # soft weak prior (~0.08)
        self.register_buffer("grade_grid", torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32))

        self.clean_head = nn.Linear(hidden_dim, 1)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        self.proto_bank = TipAdapterPrototypeBank(proj_dim=proj_dim, max_prototypes=max_prototypes)

    @torch.no_grad()
    def set_physics_scaler(self, mu_np: np.ndarray, sd_np: np.ndarray) -> None:
        if self.phys_token_mlp is not None:
            dev = self.phys_mu.device
            self.phys_mu.copy_(torch.from_numpy(np.asarray(mu_np, dtype=np.float32)).view(1, -1).to(dev))
            self.phys_sd.copy_(torch.from_numpy(np.asarray(sd_np, dtype=np.float32) + 1e-5).view(1, -1).to(dev))

    def forward(self, v_stack: torch.Tensor, phys_feat: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        `v_stack`: `[B, V, single_view_dim]` float32 tensor.
        """
        B, V, _ = v_stack.shape
        vt = self.view_proj(v_stack.float())
        if V <= self.view_role_emb.shape[1]:
            vt = vt + self.view_role_emb[:, :V, :]

        tokens = vt
        if self.phys_token_mlp is not None and phys_feat is not None:
            pn = torch.clamp((phys_feat.float() - self.phys_mu) / self.phys_sd, -5.0, 5.0)
            ptok = self.phys_token_mlp(pn).unsqueeze(1)  # [B, 1, H]
            tokens = torch.cat([vt, ptok], dim=1)        # [B, V+1, H]

        q = self.queries.expand(B, -1, -1)
        qc, _ = self.cross_attn(q, tokens, tokens, need_weights=False)
        q = self.norm_c1(q + qc)
        q = self.norm_c2(q + self.ffn_c1(q))

        qs, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.norm_s1(q + qs)

        # Global + defect pooled normalized embedding for TipAdapter prototypes
        z_cat = torch.cat([q[:, :6, :].mean(dim=1), vt.mean(dim=1)], dim=-1)
        z_norm = F.normalize(self.proj_head(z_cat), dim=-1)
        proto_delta = self.proto_bank(z_norm)  # [B, 6, 4]

        kappa = F.softplus(self.log_kappa) * 0.20
        grid = self.grade_grid.view(1, 4)

        logits_6x4 = v_stack.new_full((B, 6, 4), -1e4, dtype=torch.float32)
        probs_6x4 = torch.zeros_like(logits_6x4)

        for d in range(6):
            qd = q[:, d, :]
            raw_z = self.grade_heads[d](qd)  # Unconstrained 4-class logits!
            c_est = 3.0 * torch.sigmoid(self.center_heads[d](qd))  # [B, 1]
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

        expected_s = (probs_6x4 * grid.view(1, 1, 4)).sum(dim=-1)  # [B, 6]
        clean_logit = self.clean_head(q[:, 6, :]).squeeze(-1)
        return {
            "logits_6x4": logits_6x4,
            "probs_6x4": probs_6x4,
            "severity_s": expected_s,
            "delta_res": logits_6x4.new_zeros((B, 4, 4)),
            "clean_logit": clean_logit,
            "z_norm": z_norm,
        }


class BoundaryMarginXBMLoss(nn.Module):
    """
    Cross-Video Multi-Boundary Pairwise Ranking Loss with 512-entry Memory Queue.
    Directly optimizes the exact log-odds boundaries (`Pos vs Clean`, `P1+ vs P2`, `P0 vs P1`)
    across distinct videos (`group_i != group_j`).
    """
    def __init__(self, capacity: int = 512, sigma0: float = 1.0, cond_b1: bool = False):
        super().__init__()
        self.capacity = int(capacity)
        self.sigma0 = float(sigma0)
        self.cond_b1 = bool(cond_b1)
        self.register_buffer("mem_b", torch.zeros(capacity, 6, 3, dtype=torch.float32))
        self.register_buffer("mem_y", torch.zeros(capacity, 6, dtype=torch.long))
        self.register_buffer("mem_g", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _boundary_scores(logits_6x4: torch.Tensor, cond_b1: bool = False) -> torch.Tensor:
        """
        Extracts 3 monotone boundary log-odds `[B, 6, 3]` from `[B, 6, 4]` logits:
          b[..., 0]: Pos (>=1) vs Clean (0)
          b[..., 1]: Moderate+ (>=2) vs Mild/Clean (<=1), or conditional `>=2 vs 1` when `cond_b1=True`
          b[..., 2]: Severe P0 (3) vs P1 (2) [or P0 vs Clean for binary dims]
        """
        B = logits_6x4.shape[0]
        out = logits_6x4.new_zeros((B, 6, 3))
        for d in range(6):
            z = logits_6x4[:, d, :]
            if d in FULL_DIMS:
                b0 = torch.logsumexp(z[:, 1:], dim=-1) - z[:, 0]
                if cond_b1:
                    b1 = torch.logsumexp(z[:, 2:], dim=-1) - z[:, 1]
                else:
                    b1 = torch.logsumexp(z[:, 2:], dim=-1) - torch.logsumexp(z[:, :2], dim=-1)
                b2 = z[:, 3] - z[:, 2]
                out[:, d, 0] = b0
                out[:, d, 1] = b1
                out[:, d, 2] = b2
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
        b_cur = self._boundary_scores(logits_6x4.float(), cond_b1=self.cond_b1)  # [B, 6, 3]
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

        diff_group = g_cur.unsqueeze(1) != g_ref.unsqueeze(0)  # [B, M]
        inv_s = 1.0 / (2.0 * self.sigma0)
        losses: List[torch.Tensor] = []

        for d in range(6):
            yi = y_cur[:, d].unsqueeze(1)  # [B, 1]
            yj = y_ref[:, d].unsqueeze(0)  # [1, M]
            # Extra emphasis on OverGen (d=4) and OverSharp (d=2)
            dim_w = 1.45 if d == 4 else (1.25 if d == 2 else 1.0)

            if d in FULL_DIMS:
                # Cut 0: Pos (>=1) vs Clean (0)
                c0_hi = (yi >= 1) & (yj == 0) & diff_group
                c0_lo = (yi == 0) & (yj >= 1) & diff_group
                # Cut 1: >=2 vs 1 (P1/P0 vs P2)
                c1_hi = (yi >= 2) & (yj == 1) & diff_group
                c1_lo = (yi == 1) & (yj >= 2) & diff_group
                # Cut 2: 3 vs 2 (P0 vs P1 — high weight because P0 carries 20-pt S_p0 + 25-pt S_pos!)
                c2_hi = (yi == 3) & (yj == 2) & diff_group
                c2_lo = (yi == 2) & (yj == 3) & diff_group
                cuts = [(0, c0_hi, c0_lo, 1.0), (1, c1_hi, c1_lo, 1.25), (2, c2_hi, c2_lo, 1.65)]
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
                losses.append(dim_w * bw * fl[valid].mean())

        if self.training:
            self._enqueue(b_cur.detach(), y_cur.detach(), g_cur.detach())

        if not losses:
            return logits_6x4.sum() * 0.0
        return torch.stack(losses).mean()


class P12CompositeOrdinalLoss(nn.Module):
    def __init__(
        self,
        soft_sigma: float = 0.30,
        lambda_rps: float = 0.50,
        lambda_bmargin: float = 0.30,
        lambda_clean: float = 0.25,
        kd_temp: float = 1.25,
        cond_b1: bool = False,
    ):
        super().__init__()
        self.lambda_rps = float(lambda_rps)
        self.lambda_bmargin = float(lambda_bmargin)
        self.lambda_clean = float(lambda_clean)
        self.kd_temp = float(kd_temp)
        self.bmargin = BoundaryMarginXBMLoss(capacity=512, sigma0=1.1, cond_b1=cond_b1)

        q_table = torch.zeros(4, 4, dtype=torch.float32)
        ks = torch.arange(4, dtype=torch.float32)
        q_table[0, 0] = 1.0
        for yv in (1, 2, 3):
            w = torch.exp(-0.5 * ((ks - float(yv)) / soft_sigma) ** 2)
            if yv >= 2:
                w[0] = 0.0
            else:
                w[0] *= 0.12
            q_table[yv] = w / w.sum()
        self.register_buffer("q_soft_full", q_table)

        cls_w = torch.ones(6, 4, dtype=torch.float32)
        cls_w[:, 3] = 1.70
        cls_w[4, 1:] = 2.00  # Boost OverGen P2/P1/P0
        cls_w[2, 1:] = 1.55  # Boost OverSharp P2/P1/P0
        cls_w[3, 3] = 1.80
        cls_w[5, 3] = 1.95  # Boost Abrupt Bokeh P0
        self.register_buffer("cls_weights", cls_w)

    def forward(
        self,
        head_out: Dict[str, torch.Tensor],
        grades: torch.Tensor,
        group_hashes: torch.Tensor,
        teacher_probs_6x4: Optional[torch.Tensor] = None,
        has_teacher_mask: Optional[torch.Tensor] = None,
        lambda_kd: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        logits_6x4 = head_out["logits_6x4"].float()
        probs_6x4 = head_out["probs_6x4"].float()
        clean_logit = head_out["clean_logit"].float()
        y = grades.long()
        is_clean_img = (y == 0).all(dim=1)

        kl_losses = []
        rps_losses = []
        for d in FULL_DIMS:
            yd = y[:, d].clamp(0, 3)
            q_tgt = self.q_soft_full[yd]
            log_p = F.log_softmax(logits_6x4[:, d, :], dim=-1)
            kl_i = torch.sum(q_tgt * (torch.log(q_tgt.clamp(min=1e-8)) - log_p), dim=-1)
            w_i = torch.where(is_clean_img, self.cls_weights[d, yd] * 1.35, self.cls_weights[d, yd])
            kl_losses.append((kl_i * w_i).sum() / w_i.sum().clamp(min=1.0))

            cdf_pred = torch.cumsum(probs_6x4[:, d, :3], dim=-1)
            ks = torch.arange(3, device=y.device).unsqueeze(0)
            cdf_true = (yd.unsqueeze(1) <= ks).float()
            rps_losses.append(((cdf_pred - cdf_true) ** 2).sum(dim=-1).mean())

        for d in P0_ONLY_DIMS:
            yd_bin = (y[:, d] >= 3).long()
            log_p2 = F.log_softmax(logits_6x4[:, d, [0, 3]], dim=-1)
            q2 = torch.zeros_like(log_p2)
            q2[:, 1] = torch.where(yd_bin == 1, 0.985, 0.0)
            q2[:, 0] = 1.0 - q2[:, 1]
            kl_b = torch.sum(q2 * (torch.log(q2.clamp(min=1e-8)) - log_p2), dim=-1)
            w_b = torch.where(yd_bin == 1, self.cls_weights[d, 3], torch.where(is_clean_img, 1.50, 1.0))
            kl_losses.append((kl_b * w_b).sum() / w_b.sum().clamp(min=1.0))
            rps_losses.append(((probs_6x4[:, d, 3] - yd_bin.float()) ** 2).mean())

        loss_kl = torch.stack(kl_losses).mean()
        loss_rps = torch.stack(rps_losses).mean()
        loss_bm = self.bmargin(logits_6x4, y, group_hashes)
        loss_clean = F.binary_cross_entropy_with_logits(clean_logit, is_clean_img.float())
        if is_clean_img.any():
            loss_clean = loss_clean + 0.50 * probs_6x4[is_clean_img, :, 3].amax(dim=1).mean()

        total = loss_kl + self.lambda_rps * loss_rps + self.lambda_bmargin * loss_bm + self.lambda_clean * loss_clean

        loss_kd = total.new_tensor(0.0)
        if teacher_probs_6x4 is not None and lambda_kd > 1e-6:
            valid = has_teacher_mask.bool() if has_teacher_mask is not None else torch.ones(y.shape[0], dtype=torch.bool, device=y.device)
            if valid.any():
                tp = teacher_probs_6x4[valid].float().clamp(1e-6, 1.0 - 1e-6)
                lg = logits_6x4[valid]
                T = self.kd_temp
                kd_terms = []
                for d in FULL_DIMS:
                    t_q = F.softmax(torch.log(tp[:, d, :]) / T, dim=-1)
                    s_lp = F.log_softmax(lg[:, d, :] / T, dim=-1)
                    kd_terms.append((T * T) * F.kl_div(s_lp, t_q, reduction="batchmean"))
                for d in P0_ONLY_DIMS:
                    t_q2 = F.softmax(torch.log(tp[:, d, [0, 3]]) / T, dim=-1)
                    s_lp2 = F.log_softmax(lg[:, d, [0, 3]] / T, dim=-1)
                    kd_terms.append((T * T) * F.kl_div(s_lp2, t_q2, reduction="batchmean"))
                loss_kd = torch.stack(kd_terms).mean()
                total = total + float(lambda_kd) * loss_kd

        return {
            "total": total,
            "kl": loss_kl.detach(),
            "rps": loss_rps.detach(),
            "bmargin": loss_bm.detach(),
            "clean": loss_clean.detach(),
            "kd": loss_kd.detach(),
        }
