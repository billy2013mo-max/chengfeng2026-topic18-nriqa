#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/prism/heads_and_losses.py
==============================
PRISM Core Mathematical Module 1:
  1. `ConditionalHazardOrdinalHead`:
     Replaces the flawed `exact_probs_from_dual` (`c_p2 = s2 - s1`, `c_p1 = s1 - s0` subtraction of
     uncalibrated sigmoids) with an exact Conditional Hazard Chain:
       - h_det  = P(Y >= 1 | x)              (Presence of defect vs Clean 0)
       - h_sev1 = P(Y >= 2 | Y >= 1, x)      (P1/P0 vs P2, conditioned strictly on Y >= 1)
       - h_sev0 = P(Y == 3 | Y >= 2, x)      (P0 vs P1, conditioned strictly on Y >= 2)
     Exact grade probabilities follow the multiplicative probability law:
       - P(Clean) = 1 - h_det
       - P(P2)    = h_det * (1 - h_sev1)
       - P(P1)    = h_det * h_sev1 * (1 - h_sev0)
       - P(P0)    = h_det * h_sev1 * h_sev0
     This guarantees non-negative probabilities summing to 1.0 without ever squeezing P1!

  2. `PrismExactGradeLoss`:
     - Hard conditional supervision on each hazard branch's valid conditioning subset
       (ZERO `sigma=0.32` Gaussian soft-label smearing between P1 and P2!).
     - Direct 14-Channel Differentiable Exact-Grade Macro-F1 Surrogate Loss + Official S_p0 Loss
       (0.60 * R_p0 + 0.40 * F1_p0) + All-Clean P0 Shield Penalty.

  3. `PrismDifferentialCalibratorLoss` & `GradientReversalLayer`:
     - Ensures synthetic / Poisson-blended quadruplets NEVER update single-image classification biases
       (`head_hazard` bias parameters are detached for synthetic samples), and ONLY train the
       Pairwise Relative Severity Differential Operator Delta(f(x_a), f(x_b)) with a Domain-Adversarial
       Gradient Reversal Layer (GRL) so backbone features remain 100% invariant between real and synthetic domains.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
FULL_CLASSES = ['模糊', '纹理损失', '过锐', '过度生成']
P0_ONLY_CLASSES = ['花屏', '虚化突兀']
FULL_INDICES = [0, 1, 2, 4]
P0_ONLY_INDICES = [3, 5]


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    """Domain-Adversarial Gradient Reversal Layer (Ganin et al.) to strip synthetic-vs-real shortcuts."""
    def __init__(self, lambd: float = 0.35):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambd)


class ConditionalHazardOrdinalHead(nn.Module):
    """
    PRISM Sequential Conditional Hazard Head + Pairwise Differential Severity Comparator.
    Outputs:
      - `logits_hazard` (B, 14):
          For each of the 6 classes in `CLASSES`:
            * If class in `FULL_CLASSES` (3 channels):
                col+0: logit(h_det)  = logit P(Y >= 1 | x)
                col+1: logit(h_sev1) = logit P(Y >= 2 | Y >= 1, x)
                col+2: logit(h_sev0) = logit P(Y == 3 | Y >= 2, x)
            * If class in `P0_ONLY_CLASSES` (1 channel):
                col+0: logit(h_p0)   = logit P(Y == 3 | x)
      - `logits_simplex` (B, 20):
          Direct 4-way (or 2-way) residual logits fused in log-probability space with the hazard chain.
      - `gate_logit` (B,):
          All-Clean image authenticity gate logit.
      - `z_metric` (B, 6, D_proj):
          L2-normalized per-dimension severity metric embeddings for Reference-Anchored & Pairwise comparison.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, proj_dim: int = 128, dropout: float = 0.12):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim

        # Shared feature trunk with compact 256-dim Squeeze-and-Excitation channel calibration
        self.norm = nn.LayerNorm(in_dim)
        self.se_gate = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, in_dim),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(dropout)

        # Conditional Hazard Branch (14 conditional hazard logits)
        self.hazard_trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.hazard_out = nn.Linear(hidden_dim, 14)

        # Direct Simplex Branch (20 logits: 4x4 for full dims + 2x2 for P0-only dims)
        self.simplex_trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.simplex_out = nn.Linear(hidden_dim, 20)

        # All-Clean shield gate
        self.clean_gate = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Per-dimension Metric Embedding Projectors + Learned Monotonic Severity Direction u_c
        self.dim_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
            ) for _ in range(6)
        ])
        self.severity_dirs = nn.Parameter(torch.randn(6, proj_dim) * 0.05)

        # Domain discriminator with Gradient Reversal to enforce real/synthetic feature invariance
        self.grl = GradientReversalLayer(lambd=0.30)
        self.domain_disc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, is_real_mask: torch.Tensor = None):
        """
        If `is_real_mask` (B,) is provided during training, synthetic samples (is_real == 0)
        pass through `hazard_out` and `simplex_out` with detached bias/weights so they NEVER shift
        single-image classification thresholds, only training the backbone & differential metric space!
        """
        z_norm = self.norm(x)
        h = self.drop(z_norm * (0.5 + 0.5 * self.se_gate(z_norm)))

        h_haz = self.hazard_trunk(h)
        h_sim = self.simplex_trunk(h)

        if self.training and is_real_mask is not None and not bool(is_real_mask.all()):
            real_f = is_real_mask.float().unsqueeze(-1)  # (B, 1)
            # Authentic samples update full head parameters; synthetic samples detach single-image biases
            logits_haz_real = self.hazard_out(h_haz)
            logits_haz_syn = F.linear(h_haz, self.hazard_out.weight.detach(), self.hazard_out.bias.detach())
            logits_hazard = real_f * logits_haz_real + (1.0 - real_f) * logits_haz_syn

            logits_sim_real = self.simplex_out(h_sim)
            logits_sim_syn = F.linear(h_sim, self.simplex_out.weight.detach(), self.simplex_out.bias.detach())
            logits_simplex = real_f * logits_sim_real + (1.0 - real_f) * logits_sim_syn
        else:
            logits_hazard = self.hazard_out(h_haz)
            logits_simplex = self.simplex_out(h_sim)

        gate_logit = self.clean_gate(h).squeeze(-1)

        z_list = [F.normalize(proj(h), dim=-1) for proj in self.dim_projectors]
        z_metric = torch.stack(z_list, dim=1)  # (B, 6, proj_dim)

        domain_logit = self.domain_disc(self.grl(h_haz)).squeeze(-1)
        return logits_hazard, logits_simplex, gate_logit, z_metric, domain_logit

    def exact_probabilities(self, logits_hazard: torch.Tensor, logits_simplex: torch.Tensor, alpha_simplex: float = 0.35):
        """
        Computes exact grade probabilities (p0, p1, p2, P_exact14) using the Conditional Hazard Law
        fused in log-odds space with the direct simplex branch.
        Returns:
          - p0: (B, 6) exact P(Y == 3)
          - p1: (B, 4) exact P(Y == 2) for FULL_CLASSES
          - p2: (B, 4) exact P(Y == 1) for FULL_CLASSES
          - P_exact14: (B, 14) exact 14-channel probabilities matching competition evaluation columns
        """
        eps = 1e-6
        B = logits_hazard.shape[0]
        dev = logits_hazard.device

        p0_list, p1_list, p2_list, exact14_list = [], [], [], []
        col_h = 0
        col_s = 0

        for ci, cname in enumerate(CLASSES):
            if cname in FULL_CLASSES:
                h_det  = torch.sigmoid(logits_hazard[:, col_h])
                h_sev1 = torch.sigmoid(logits_hazard[:, col_h + 1])
                h_sev0 = torch.sigmoid(logits_hazard[:, col_h + 2])
                col_h += 3

                # Exact multiplicative conditional hazard probabilities (strictly sum to 1, zero P1 squeezing)
                p_clean_h = 1.0 - h_det
                p_p2_h    = h_det * (1.0 - h_sev1)
                p_p1_h    = h_det * h_sev1 * (1.0 - h_sev0)
                p_p0_h    = h_det * h_sev1 * h_sev0
                probs_h4  = torch.stack([p_clean_h, p_p2_h, p_p1_h, p_p0_h], dim=-1).clamp(eps, 1.0 - eps)

                probs_s4  = F.softmax(logits_simplex[:, col_s:col_s + 4], dim=-1).clamp(eps, 1.0 - eps)
                col_s += 4

                # Geometric log-linear pooling preserves relative ratios without linear subtraction collapse
                log_fused = (1.0 - alpha_simplex) * torch.log(probs_h4) + alpha_simplex * torch.log(probs_s4)
                probs4 = F.softmax(log_fused, dim=-1)

                c_p2 = probs4[:, 1]
                c_p1 = probs4[:, 2]
                c_p0 = probs4[:, 3]
                p0_list.append(c_p0)
                p1_list.append(c_p1)
                p2_list.append(c_p2)
                exact14_list.extend([c_p2, c_p1, c_p0])
            else:
                h_p0 = torch.sigmoid(logits_hazard[:, col_h]).clamp(eps, 1.0 - eps)
                col_h += 1
                probs_s2 = F.softmax(logits_simplex[:, col_s:col_s + 2], dim=-1)[:, 1].clamp(eps, 1.0 - eps)
                col_s += 2

                logit_h = torch.logit(h_p0)
                logit_s = torch.logit(probs_s2)
                c_p0 = torch.sigmoid((1.0 - alpha_simplex) * logit_h + alpha_simplex * logit_s)
                p0_list.append(c_p0)
                exact14_list.append(c_p0)

        p0 = torch.stack(p0_list, dim=-1)          # (B, 6)
        p1 = torch.stack(p1_list, dim=-1)          # (B, 4)
        p2 = torch.stack(p2_list, dim=-1)          # (B, 4)
        P_exact14 = torch.stack(exact14_list, dim=-1)  # (B, 14)
        return p0, p1, p2, P_exact14


class PrismExactGradeLoss(nn.Module):
    """
    PRISM Full-Spectrum Supervision Objective:
      1. Conditional Hazard Hard-Target BCE on valid conditioning subsets:
         - h_det  supervised on ALL valid samples (target: Y >= 1)
         - h_sev1 supervised strictly on defective samples Y >= 1 (target: Y >= 2)
         - h_sev0 supervised strictly on moderate/severe samples Y >= 2 (target: Y == 3)
      2. Direct P1-Boosted Exact Grade Cross-Entropy + Cumulative EMD (ZERO Gaussian label smearing!).
      3. Direct 14-Channel Differentiable Exact-Grade Macro-F1 Surrogate Loss + Official S_p0 Loss.
      4. All-Clean P0 Zero-Tolerance Shield Penalty.
    """
    def __init__(self, p0_clean_weight: float = 4.5, alpha_simplex: float = 0.35):
        super().__init__()
        self.p0_clean_weight = p0_clean_weight
        self.alpha_simplex = alpha_simplex

    def forward(
        self,
        head: ConditionalHazardOrdinalHead,
        logits_hazard: torch.Tensor,
        logits_simplex: torch.Tensor,
        gate_logit: torch.Tensor,
        g: torch.Tensor,
        mask7: torch.Tensor,
    ) -> torch.Tensor:
        dev = g.device
        mask6 = mask7[:, :6]          # (B, 6)
        is_real = (mask7[:, 6] > 0.5) # (B,)
        clean_real = (g == 0).all(dim=1) & is_real

        # 1. Conditional Hazard Loss on exact conditioning subsets (only on real samples or valid masked samples)
        loss_hazard = g.new_tensor(0.0, dtype=torch.float32)
        loss_simplex = g.new_tensor(0.0, dtype=torch.float32)
        loss_emd = g.new_tensor(0.0, dtype=torch.float32)
        loss_margin = g.new_tensor(0.0, dtype=torch.float32)

        cw4 = torch.tensor([1.0, 1.25, 1.75, 1.55], device=dev)
        b_w = torch.tensor([1.0, 1.90, 1.65], device=dev)

        col_h = 0
        col_s = 0
        for ci, cname in enumerate(CLASSES):
            m_ci = mask6[:, ci] * is_real.float()  # Single-image classification strictly uses authentic real labels!
            yc = g[:, ci]

            if cname in FULL_CLASSES:
                l_det  = logits_hazard[:, col_h]
                l_sev1 = logits_hazard[:, col_h + 1]
                l_sev0 = logits_hazard[:, col_h + 2]
                col_h += 3

                # (a) Presence detection P(Y >= 1) on all real samples
                t_det = (yc >= 1).float()
                w_det = torch.where(clean_real, torch.full_like(m_ci, 1.8), torch.ones_like(m_ci)) * m_ci
                if w_det.sum() > 0:
                    bce_det = F.binary_cross_entropy_with_logits(l_det, t_det, reduction='none')
                    loss_hazard = loss_hazard + (bce_det * w_det).sum() / w_det.sum().clamp(min=1.0)

                # (b) Conditional severity P(Y >= 2 | Y >= 1) strictly on samples with Y >= 1!
                m_ge1 = m_ci * (yc >= 1).float()
                if m_ge1.sum() > 0:
                    t_sev1 = (yc >= 2).float()
                    # Extra focus on P1/P2 boundary
                    w_s1 = torch.where(yc == 2, torch.full_like(m_ge1, 1.45), torch.ones_like(m_ge1)) * m_ge1
                    bce_s1 = F.binary_cross_entropy_with_logits(l_sev1, t_sev1, reduction='none')
                    loss_hazard = loss_hazard + 1.35 * (bce_s1 * w_s1).sum() / w_s1.sum().clamp(min=1.0)

                # (c) Conditional severity P(Y == 3 | Y >= 2) strictly on samples with Y >= 2!
                m_ge2 = m_ci * (yc >= 2).float()
                if m_ge2.sum() > 0:
                    t_sev0 = (yc == 3).float()
                    bce_s0 = F.binary_cross_entropy_with_logits(l_sev0, t_sev0, reduction='none')
                    loss_hazard = loss_hazard + 1.35 * (bce_s0 * m_ge2).sum() / m_ge2.sum().clamp(min=1.0)

                # (d) Direct 4-way Simplex CE + CDF EMD + Adjacent Logit Separation Margin
                lg4 = logits_simplex[:, col_s:col_s + 4]
                col_s += 4
                if m_ci.sum() > 0:
                    ce4 = F.cross_entropy(lg4, yc.long(), weight=cw4, reduction='none', label_smoothing=0.01)
                    loss_simplex = loss_simplex + (ce4 * m_ci).sum() / m_ci.sum().clamp(min=1.0)

                    probs4 = F.softmax(lg4, dim=-1)
                    cdf_p = torch.cumsum(probs4, dim=-1)[:, :3]
                    cdf_t = torch.cumsum(F.one_hot(yc.long(), num_classes=4).float(), dim=-1)[:, :3]
                    emd_c = (((cdf_p - cdf_t) ** 2) * b_w).mean(dim=-1)
                    loss_emd = loss_emd + (emd_c * m_ci).sum() / m_ci.sum().clamp(min=1.0)

                    # Explicit logit separation between P1 (2) vs P2 (1) and P0 (3) vs P1 (2)
                    is_p2 = (yc == 1).float() * m_ci
                    is_p1 = (yc == 2).float() * m_ci
                    is_p0 = (yc == 3).float() * m_ci
                    sep = (
                        is_p1 * (F.relu(0.45 - (lg4[:, 2] - lg4[:, 1])) + F.relu(0.35 - (lg4[:, 2] - lg4[:, 3])))
                        + is_p2 * F.relu(0.35 - (lg4[:, 1] - lg4[:, 2]))
                        + is_p0 * F.relu(0.45 - (lg4[:, 3] - lg4[:, 2]))
                    )
                    loss_margin = loss_margin + sep.sum() / (m_ci.sum() + 1e-5)
            else:
                l_p0 = logits_hazard[:, col_h]
                col_h += 1
                lg2 = logits_simplex[:, col_s:col_s + 2]
                col_s += 2
                t_p0 = (yc == 3).float()
                w_p0 = torch.where(clean_real, torch.full_like(m_ci, self.p0_clean_weight), torch.ones_like(m_ci)) * m_ci
                if w_p0.sum() > 0:
                    bce_p0 = F.binary_cross_entropy_with_logits(l_p0, t_p0, reduction='none')
                    loss_hazard = loss_hazard + (bce_p0 * w_p0).sum() / w_p0.sum().clamp(min=1.0)
                if m_ci.sum() > 0:
                    cw2 = torch.tensor([1.0, 1.55], device=dev)
                    ce2 = F.cross_entropy(lg2, t_p0.long(), weight=cw2, reduction='none', label_smoothing=0.01)
                    loss_simplex = loss_simplex + (ce2 * m_ci).sum() / m_ci.sum().clamp(min=1.0)

        # 2. Differentiable 14-Channel Exact-Grade Macro-F1 Surrogate + Official S_p0 Alignment
        p0, p1, p2, P_exact14 = head.exact_probabilities(logits_hazard, logits_simplex, alpha_simplex=self.alpha_simplex)
        y_exact14 = []
        m_exact14 = []
        for ci, cname in enumerate(CLASSES):
            m_real_c = (mask6[:, ci] > 0.1).float() * is_real.float()
            yc = g[:, ci]
            if cname in FULL_CLASSES:
                y_exact14.extend([(yc == 1).float(), (yc == 2).float(), (yc == 3).float()])
                m_exact14.extend([m_real_c, m_real_c, m_real_c])
            else:
                y_exact14.append((yc == 3).float())
                m_exact14.append(m_real_c)
        Y_exact14 = torch.stack(y_exact14, dim=-1)
        M_exact14 = torch.stack(m_exact14, dim=-1)

        tp = (P_exact14 * Y_exact14 * M_exact14).sum(dim=0)
        fp = (P_exact14 * (1.0 - Y_exact14) * M_exact14).sum(dim=0)
        fn = ((1.0 - P_exact14) * Y_exact14 * M_exact14).sum(dim=0)
        soft_f1_14 = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
        loss_s_pos = 1.0 - soft_f1_14.mean()

        Y_p0 = (g == 3).float()
        M_p0 = (mask6 > 0.1).float() * is_real.float().unsqueeze(-1)
        tp_p0 = (p0 * Y_p0 * M_p0).sum(dim=0)
        fp_p0 = (p0 * (1.0 - Y_p0) * M_p0).sum(dim=0)
        fn_p0 = ((1.0 - p0) * Y_p0 * M_p0).sum(dim=0)
        r_p0 = (tp_p0 + 1e-4) / (tp_p0 + fn_p0 + 1e-4)
        f1_p0 = (2.0 * tp_p0 + 1e-4) / (2.0 * tp_p0 + fp_p0 + fn_p0 + 1e-4)
        loss_s_p0 = 1.0 - (0.60 * r_p0 + 0.40 * f1_p0).mean()

        loss_clean_shield = p0[clean_real].amax(dim=-1).mean() if clean_real.any() else g.new_tensor(0.0, dtype=torch.float32)
        loss_gate = F.binary_cross_entropy_with_logits(gate_logit[is_real], clean_real[is_real].float()) if is_real.any() else g.new_tensor(0.0, dtype=torch.float32)

        return (
            loss_hazard / 6.0
            + 0.75 * (loss_simplex / 6.0)
            + 0.65 * (loss_emd / 4.0)
            + 0.50 * (loss_margin / 4.0)
            + 0.75 * loss_s_pos
            + 0.65 * loss_s_p0
            + 0.45 * loss_clean_shield
            + 0.30 * loss_gate
        )


class PrismDifferentialCalibratorLoss(nn.Module):
    """
    PRISM Pairwise Relative Severity Differential Loss for Synthetic & Real-Blended Quadruplets:
      - Enforces strict equidistant monotonic projection along the learned severity direction u_c:
          <z_metric(x_a) - z_metric(x_b), u_c> >= delta_margin * (y_a - y_b)
      - Strongly prioritizes same-seed / same-scene pairs (weight 3.5x) so synthetic quadruplets
        calibrate the exact relative step sizes (0 -> P2 -> P1 -> P0) without altering single-image biases.
      - Includes Domain-Adversarial Invariance Loss via GRL so backbone features cannot distinguish
        synthetic/Poisson-blended images from authentic real images.
    """
    def __init__(self, base_margin: float = 0.28, domain_weight: float = 0.20):
        super().__init__()
        self.base_margin = base_margin
        self.domain_weight = domain_weight

    def forward(
        self,
        head: ConditionalHazardOrdinalHead,
        z_metric: torch.Tensor,
        domain_logit: torch.Tensor,
        g: torch.Tensor,
        mask7: torch.Tensor,
        seed_stems: list,
    ) -> torch.Tensor:
        dev = g.device
        mask6 = mask7[:, :6]
        is_real = (mask7[:, 6] > 0.5).float()

        # 1. Domain-Adversarial Invariance Loss (GRL already reverses gradients into the trunk)
        if (is_real < 0.5).any() and (is_real > 0.5).any():
            loss_dom = F.binary_cross_entropy_with_logits(domain_logit, is_real)
        else:
            loss_dom = z_metric.new_tensor(0.0)

        # 2. Pairwise Relative Severity Differential Loss along u_c
        u_dirs = F.normalize(head.severity_dirs, dim=-1)  # (6, proj_dim)
        loss_diff = z_metric.new_tensor(0.0)
        n_active = 0

        for ci, cname in enumerate(CLASSES):
            valid = mask6[:, ci] > 0.1
            if valid.sum() < 2:
                continue
            v_idx = torch.where(valid)[0]
            yc = g[v_idx, ci].float()
            zc = z_metric[v_idx, ci, :]  # (M, proj_dim)
            scores = (zc * u_dirs[ci].unsqueeze(0)).sum(dim=-1)  # (M,)

            diff_y = yc.unsqueeze(1) - yc.unsqueeze(0)  # (M, M)
            pos_pairs = diff_y >= 1.0
            if not pos_pairs.any():
                continue

            v_stems = [seed_stems[int(i)] for i in v_idx.cpu().tolist()]
            same_seed = torch.tensor(
                [[1.0 if v_stems[a] == v_stems[b] else 0.0 for b in range(len(v_stems))] for a in range(len(v_stems))],
                device=dev, dtype=torch.float32,
            )
            both_defective = (yc.unsqueeze(1) >= 1.0) & (yc.unsqueeze(0) >= 1.0)
            pair_w = (1.0 + 2.5 * same_seed + 1.5 * both_defective.float())[pos_pairs]

            req_margin = self.base_margin * diff_y[pos_pairs]
            diff_s = (scores.unsqueeze(1) - scores.unsqueeze(0))[pos_pairs]
            viol = F.relu(req_margin - diff_s) ** 2
            loss_diff = loss_diff + (viol * pair_w).mean()
            n_active += 1

        return (loss_diff / max(1, n_active)) + self.domain_weight * loss_dom


def convert_p012_to_conditional_hazards(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, eps: float = 1e-5):
    """
    Converts (p0 [N,6], p1 [N,4], p2 [N,4]) into PRISM's exact orthogonal Conditional Hazard decomposition:
      - h_det  = P(Y >= 1)            [N, 6]
      - h_sev1 = P(Y >= 2 | Y >= 1)   [N, 4] (strictly conditioned on defect presence!)
      - h_sev0 = P(Y == 3 | Y >= 2)   [N, 4] (strictly conditioned on P1/P0!)
    """
    p0 = np.clip(np.asarray(p0, dtype=np.float64), eps, 1.0 - eps)
    p1 = np.clip(np.asarray(p1, dtype=np.float64), eps, 1.0 - eps)
    p2 = np.clip(np.asarray(p2, dtype=np.float64), eps, 1.0 - eps)

    N = p0.shape[0]
    h_det = np.zeros((N, 6), dtype=np.float64)
    h_sev1 = np.zeros((N, 4), dtype=np.float64)
    h_sev0 = np.zeros((N, 4), dtype=np.float64)

    for ci, cname in enumerate(CLASSES):
        if cname in FULL_CLASSES:
            k = FULL_CLASSES.index(cname)
            mass_pos = np.clip(p0[:, ci] + p1[:, k] + p2[:, k], eps, 1.0 - eps)
            mass_ge2 = np.clip(p0[:, ci] + p1[:, k], eps, mass_pos)
            h_det[:, ci] = mass_pos
            h_sev1[:, k] = np.clip(mass_ge2 / mass_pos, eps, 1.0 - eps)
            h_sev0[:, k] = np.clip(p0[:, ci] / mass_ge2, eps, 1.0 - eps)
        else:
            h_det[:, ci] = p0[:, ci]

    return h_det.astype(np.float32), h_sev1.astype(np.float32), h_sev0.astype(np.float32)


def convert_conditional_hazards_to_p012(h_det: np.ndarray, h_sev1: np.ndarray, h_sev0: np.ndarray):
    """
    Reconstructs exact (p0 [N,6], p1 [N,4], p2 [N,4]) from Conditional Hazards (h_det, h_sev1, h_sev0)
    via the exact multiplicative chain law:
      P(P2) = h_det * (1 - h_sev1)
      P(P1) = h_det * h_sev1 * (1 - h_sev0)
      P(P0) = h_det * h_sev1 * h_sev0
    """
    h_det = np.clip(np.asarray(h_det, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    h_sev1 = np.clip(np.asarray(h_sev1, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    h_sev0 = np.clip(np.asarray(h_sev0, dtype=np.float64), 1e-6, 1.0 - 1e-6)

    N = h_det.shape[0]
    p0 = np.zeros((N, 6), dtype=np.float32)
    p1 = np.zeros((N, 4), dtype=np.float32)
    p2 = np.zeros((N, 4), dtype=np.float32)

    for ci, cname in enumerate(CLASSES):
        if cname in FULL_CLASSES:
            k = FULL_CLASSES.index(cname)
            d = h_det[:, ci]
            s1 = h_sev1[:, k]
            s0 = h_sev0[:, k]
            p2[:, k] = (d * (1.0 - s1)).astype(np.float32)
            p1[:, k] = (d * s1 * (1.0 - s0)).astype(np.float32)
            p0[:, ci] = (d * s1 * s0).astype(np.float32)
        else:
            p0[:, ci] = h_det[:, ci].astype(np.float32)

    return p0, p1, p2
