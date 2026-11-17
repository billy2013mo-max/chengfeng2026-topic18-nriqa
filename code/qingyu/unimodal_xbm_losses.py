#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/unimodal_xbm_losses.py
==================================
Pillar 1 (Section 2.3) & Pillar 4 (Sections 5.1 – 5.2) Training Core:
  1. `UnimodalQuadraticResidualHead`:
     Replaces unconstrained 4-way classification & 3-way hazard multiplication (`P(>=1)*P(>=2|>=1)*P(3|>=2)`)
     with a continuous severity scalar `s_d(x)` anchored at `mu = (0.0, 1.5, 2.5, 3.5)` plus an L2-damped
     residual `delta_{d,c}(x)`:
         z_{d,c}(x) = -alpha_d * (s_d(x) - mu_{d,c})^2 + delta_{d,c}(x),   alpha_d = Softplus(a_d) + 0.10
     Structurally eliminates bimodal probability splitting (`P0 + Clean` swallowing `P1`) and 3-way product shrinkage!
  2. `TipAdapterPrototypeBank`:
     Pure-real training-fold visual prototype memory (`128` L2-normalized centroids = `14` positive `(defect, grade)`
     sub-clusters + `2` hard-clean clusters: optical shallow-DOF portraits & high-contrast UI/subtitle frames)
     providing non-parametric exponential-kernel residual log-odds calibration.
  3. `CrossBatchMemoryThurstoneLoss` (`M = 512`):
     Pure-real FIFO severity ring buffer across mini-batches (and across GPUs when DDP is active),
     computing Thurstone-Mosteller probit fidelity ranking loss `1 - sqrt(Phi((s_i - s_j)/(sqrt(2)*sigma_0)))`
     exclusively between distinct scene groups (`group_i != group_j`), with `2.0x` weight on adjacent boundaries
     `(3,2)`, `(2,1)`, `(1,0)`.
  4. `QingyuUnimodalCompositeLoss`:
     100% batch-size-independent composite objective combining Soft Ordinal KL (`sigma_d = 0.35`),
     1D Squared EMD / Ranked Probability Score (`RPS`), `XBM(512)` Thurstone ranking loss, `L2(delta)`
     unimodality regularizer, Global All-Clean calibration BCE, and optional fold-isolated OOF teacher KL distillation.
"""
import math
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from qingyu.config_and_paths import DIMS, FULL_DIMS, P0_ONLY_DIMS


def hash_group_id_int64(group_id: str) -> int:
    """Deterministic positive int64 hash of `group_id` string for fast tensor inequality checks."""
    h = hashlib.blake2b(str(group_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False) & 0x7FFFFFFFFFFFFFFF


class TipAdapterPrototypeBank(nn.Module):
    """
    Pure-Real Training-Fold Visual Prototype Bank (Section 2.3):
      - Stores `K` (`<= 128`) L2-normalized visual prototypes `proto_feats [K, D_proj]`
        and their smoothed empirical `[K, 6, 4]` log-prior templates `proto_log_probs [K, 6, 4]`.
      - At forward time, computes `affinity = exp(-beta * (1 - z_norm @ proto_feats.T))` and adds
        `alpha * (affinity / sum(affinity)) @ proto_log_probs` as a residual onto `[B, 6, 4]` logits.
      - Initialized as a no-op (`bank_active = 0`) during early warmup epochs and populated strictly
        from the current training fold's real embeddings (`fit_from_train_embeddings`).
    """
    def __init__(self, proj_dim: int = 512, max_prototypes: int = 128, init_alpha: float = 0.15, beta: float = 5.5):
        super().__init__()
        self.proj_dim = int(proj_dim)
        self.max_prototypes = int(max_prototypes)
        self.beta = float(beta)

        self.register_buffer("proto_feats", torch.zeros(max_prototypes, proj_dim, dtype=torch.float32))
        self.register_buffer("proto_log_probs", torch.zeros(max_prototypes, 6, 4, dtype=torch.float32))
        self.register_buffer("num_active", torch.tensor(0, dtype=torch.long))
        self.alpha_raw = nn.Parameter(torch.tensor(float(init_alpha), dtype=torch.float32))

    @torch.no_grad()
    def fit_from_train_embeddings(
        self,
        z_norm_np: np.ndarray,
        grades_np: np.ndarray,
        phys_np: Optional[np.ndarray] = None,
        per_pos_channel: int = 8,
        per_clean_cluster: int = 8,
    ) -> int:
        """
        Builds up to `14 * 8 + 2 * 8 = 128` pure-real prototypes strictly from the training fold:
          - 14 positive `(dim, grade)` sub-channels (`per_pos_channel` farthest-point / k-means centroids each)
          - 2 hard-clean clusters (`per_clean_cluster` centroids each):
              (a) Shallow-DOF optical bokeh clean images (high `fg_bg_sharp_ratio` `phys[:, 18]`)
              (b) High-UI/subtitle stroke clean images (high `ui_ratio` `phys[:, 8]`)
        """
        Z = np.asarray(z_norm_np, dtype=np.float32)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        Y = np.asarray(grades_np, dtype=np.int64)

        def _kmeans_light(sub_z: np.ndarray, sub_y: np.ndarray, k_req: int) -> List[Tuple[np.ndarray, np.ndarray]]:
            m = len(sub_z)
            if m == 0:
                return []
            k = min(k_req, m)
            if k == 1:
                c = sub_z.mean(axis=0)
                c /= np.linalg.norm(c) + 1e-8
                return [(c, sub_y)]
            # Deterministic Farthest-Point Initialization (k-means++) + 8 Lloyd steps
            centers = [sub_z[0]]
            for _ in range(1, k):
                sims = np.stack([sub_z @ c for c in centers], axis=1).max(axis=1)
                nxt = int(np.argmin(sims))
                centers.append(sub_z[nxt])
            C = np.stack(centers, axis=0)
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
            assign = np.zeros(m, dtype=np.int64)
            for _it in range(8):
                assign = np.argmax(sub_z @ C.T, axis=1)
                for ci in range(k):
                    mk = assign == ci
                    if np.any(mk):
                        nc = sub_z[mk].mean(axis=0)
                        C[ci] = nc / (np.linalg.norm(nc) + 1e-8)
            out_clusters = []
            for ci in range(k):
                mk = assign == ci
                if np.any(mk):
                    out_clusters.append((C[ci], sub_y[mk]))
            return out_clusters

        raw_protos: List[Tuple[np.ndarray, np.ndarray]] = []
        # 14 positive sub-channels
        for d in range(6):
            g_list = [1, 2, 3] if d in FULL_DIMS else [3]
            for gv in g_list:
                mask = Y[:, d] == gv
                if np.any(mask):
                    raw_protos.extend(_kmeans_light(Z[mask], Y[mask], per_pos_channel))

        # 2 hard-clean clusters (optical shallow-DOF bokeh clean + UI/subtitle clean)
        clean_mask = (Y == 0).all(axis=1)
        if np.any(clean_mask):
            z_cln = Z[clean_mask]
            y_cln = Y[clean_mask]
            if phys_np is not None and len(phys_np) == len(Y) and phys_np.shape[1] >= 19:
                p_cln = phys_np[clean_mask]
                # Top half by fg_bg_sharp_ratio (optical bokeh clean) vs top half by ui_ratio (subtitle clean)
                bokeh_ord = np.argsort(-p_cln[:, 18])[: max(4, len(z_cln) // 2)]
                ui_ord = np.argsort(-p_cln[:, 8])[: max(4, len(z_cln) // 2)]
                raw_protos.extend(_kmeans_light(z_cln[bokeh_ord], y_cln[bokeh_ord], per_clean_cluster))
                raw_protos.extend(_kmeans_light(z_cln[ui_ord], y_cln[ui_ord], per_clean_cluster))
            else:
                raw_protos.extend(_kmeans_light(z_cln, y_cln, per_clean_cluster * 2))

        raw_protos = raw_protos[: self.max_prototypes]
        K_act = len(raw_protos)
        if K_act == 0:
            self.num_active.fill_(0)
            return 0

        pf = np.zeros((self.max_prototypes, self.proj_dim), dtype=np.float32)
        plp = np.zeros((self.max_prototypes, 6, 4), dtype=np.float32)
        for idx_p, (cent, sub_y) in enumerate(raw_protos):
            pf[idx_p] = cent
            for d in range(6):
                act = [0, 1, 2, 3] if d in FULL_DIMS else [0, 3]
                cnt = np.zeros(4, dtype=np.float32)
                for v in sub_y[:, d]:
                    vv = 3 if (d in P0_ONLY_DIMS and v > 0) else int(v)
                    cnt[vv] += 1.0
                # Laplace smoothing across active grades only
                for a in act:
                    cnt[a] += 0.15
                cnt /= max(1e-6, cnt.sum())
                lp = np.zeros(4, dtype=np.float32)
                lp_act = np.log(np.clip(cnt[act], 1e-4, 1.0))
                lp_act -= lp_act.mean()
                for ia, a in enumerate(act):
                    lp[a] = lp_act[ia]
                plp[idx_p, d, :] = lp

        dev = self.proto_feats.device
        self.proto_feats.copy_(torch.from_numpy(pf).to(dev))
        self.proto_log_probs.copy_(torch.from_numpy(plp).to(dev))
        self.num_active.fill_(K_act)
        return K_act

    def forward(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Returns prototype log-odds residual `[B, 6, 4]` (zeros if bank not yet populated)."""
        k_act = int(self.num_active.item())
        if k_act <= 0:
            return z_norm.new_zeros((z_norm.shape[0], 6, 4))
        P = self.proto_feats[:k_act]          # [K, D]
        LP = self.proto_log_probs[:k_act]     # [K, 6, 4]
        cos_sim = z_norm.float() @ P.float().t()  # [B, K]
        aff = torch.exp(-self.beta * (1.0 - cos_sim).clamp(min=0.0))  # [B, K]
        aff_norm = aff / (aff.sum(dim=1, keepdim=True) + 1e-6)
        alpha = torch.clamp(self.alpha_raw, 0.0, 0.45)
        delta = torch.einsum("bk,kdc->bdc", aff_norm, LP.float())
        return alpha * delta


class UnimodalQuadraticResidualHead(nn.Module):
    """
    Pillar 4 (Section 5.1) Unimodal Quadratic-Residual Ordinal Head + Physics Gate + Prototype Bank:
      - Full 4-grade dimensions (`模糊=0, 纹理损失=1, 过锐=2, 过度生成=4`):
            z_{d,c}(x) = -alpha_d * (s_d(x) - mu_{d,c})^2 + delta_{d,c}(x)
            mu_d = (0.0, 1.5, 2.5, 3.5),  alpha_d = Softplus(alpha_raw_d) + 0.10
      - Binary P0-only dimensions (`花屏=3, 虚化突兀=5`):
            s_d(x) continuous severity scalar,  z_{d,3}(x) = alpha_bin_d * (s_d(x) - 2.0) + delta_bin_d(x),
            z_{d,0}(x) = -z_{d,3}(x),  z_{d,1} = z_{d,2} = -1e4 (strictly zero mass on P2/P1!)
      - Emits dictionary:
          * `logits_6x4`: `[B, 6, 4]`
          * `probs_6x4`:  `[B, 6, 4]` calibrated probability simplex
          * `severity_s`: `[B, 6]` continuous severity scalars for `XBM(512)` Thurstone ranking loss
          * `delta_res`:  `[B, 4, 4]` residual logits for L2 unimodality regularization
          * `clean_logit`: `[B]` auxiliary global all-clean logit
          * `z_norm`:     `[B, proj_dim]` L2-normalized embedding for `TipAdapterPrototypeBank`
    """
    def __init__(
        self,
        in_dim: int,
        phys_dim: int = 36,
        hidden_dim: int = 768,
        proj_dim: int = 512,
        dropout: float = 0.20,
        max_prototypes: int = 128,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.phys_dim = int(phys_dim)
        self.proj_dim = int(proj_dim)

        self.vis_trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Single-image standardized physics fusion branch (zero-init residual so step-0 relies on pretrained vision)
        if phys_dim > 0:
            self.register_buffer("phys_mu", torch.zeros(1, phys_dim, dtype=torch.float32))
            self.register_buffer("phys_sd", torch.ones(1, phys_dim, dtype=torch.float32))
            self.phys_mlp = nn.Sequential(
                nn.LayerNorm(phys_dim),
                nn.Linear(phys_dim, 128),
                nn.GELU(),
                nn.Linear(128, hidden_dim),
            )
            nn.init.zeros_(self.phys_mlp[-1].weight)
            nn.init.zeros_(self.phys_mlp[-1].bias)
        else:
            self.phys_mlp = None

        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Continuous latent severity scalar s_d(x) for all 6 dimensions
        self.severity_proj = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 6),
        )
        # Constrained residual delta_{d,c}(x) for the 4 full-grade dimensions (4 * 4 = 16)
        self.delta_full_proj = nn.Linear(hidden_dim, 16)
        nn.init.zeros_(self.delta_full_proj.weight)
        nn.init.zeros_(self.delta_full_proj.bias)

        # Binary P0-only bias/residual for dims [3, 5]
        self.bin_bias = nn.Parameter(torch.tensor([-2.2, -2.2], dtype=torch.float32))
        self.alpha_full_raw = nn.Parameter(torch.full((4,), 0.35, dtype=torch.float32))
        self.alpha_bin_raw = nn.Parameter(torch.full((2,), 0.65, dtype=torch.float32))

        self.register_buffer("mu_anchors", torch.tensor([0.0, 1.5, 2.5, 3.5], dtype=torch.float32).view(1, 1, 4))
        self.clean_head = nn.Linear(hidden_dim, 1)
        self.proto_bank = TipAdapterPrototypeBank(proj_dim=proj_dim, max_prototypes=max_prototypes)

    @torch.no_grad()
    def set_physics_scaler(self, mu_np: np.ndarray, sd_np: np.ndarray) -> None:
        if self.phys_mlp is not None:
            dev = self.phys_mu.device
            self.phys_mu.copy_(torch.from_numpy(np.asarray(mu_np, dtype=np.float32)).view(1, -1).to(dev))
            self.phys_sd.copy_(torch.from_numpy(np.asarray(sd_np, dtype=np.float32) + 1e-5).view(1, -1).to(dev))

    def forward(self, vis_feat: torch.Tensor, phys_feat: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h = self.vis_trunk(vis_feat.float())
        if self.phys_mlp is not None and phys_feat is not None:
            pn = torch.clamp((phys_feat.float() - self.phys_mu) / self.phys_sd, -5.0, 5.0)
            h = h + self.phys_mlp(pn)

        z_norm = F.normalize(self.proj_head(h), dim=-1)
        s_all = self.severity_proj(h)  # [B, 6]
        delta_raw = self.delta_full_proj(h).view(-1, 4, 4)
        # Zero-center delta across the 4 grades per dimension for unique identifiability with softmax
        delta = delta_raw - delta_raw.mean(dim=-1, keepdim=True)

        alpha_full = F.softplus(self.alpha_full_raw).view(1, 4, 1) + 0.10  # [1, 4, 1] > 0.10
        s_full = s_all[:, FULL_DIMS].unsqueeze(-1)                         # [B, 4, 1]
        z_full = -alpha_full * ((s_full - self.mu_anchors) ** 2) + delta   # [B, 4, 4]

        alpha_bin = F.softplus(self.alpha_bin_raw).view(1, 2) + 0.20
        s_bin = s_all[:, P0_ONLY_DIMS]                                     # [B, 2]
        logit_p0_bin = alpha_bin * s_bin + self.bin_bias.view(1, 2)        # [B, 2]

        B = vis_feat.shape[0]
        logits_6x4 = vis_feat.new_full((B, 6, 4), -1e4, dtype=torch.float32)
        for j, d in enumerate(FULL_DIMS):
            logits_6x4[:, d, :] = z_full[:, j, :]
        for j, d in enumerate(P0_ONLY_DIMS):
            logits_6x4[:, d, 0] = -0.5 * logit_p0_bin[:, j]
            logits_6x4[:, d, 3] = 0.5 * logit_p0_bin[:, j]

        # Add Tip-Adapter Pure-Real Prototype residual log-odds (masked strictly to active classes)
        proto_delta = self.proto_bank(z_norm)
        for d in FULL_DIMS:
            logits_6x4[:, d, :] = logits_6x4[:, d, :] + proto_delta[:, d, :]
        for d in P0_ONLY_DIMS:
            logits_6x4[:, d, 0] = logits_6x4[:, d, 0] + proto_delta[:, d, 0]
            logits_6x4[:, d, 3] = logits_6x4[:, d, 3] + proto_delta[:, d, 3]

        probs_6x4 = torch.zeros_like(logits_6x4)
        for d in FULL_DIMS:
            probs_6x4[:, d, :] = F.softmax(logits_6x4[:, d, :], dim=-1)
        for d in P0_ONLY_DIMS:
            p2_sm = F.softmax(logits_6x4[:, d, [0, 3]], dim=-1)
            probs_6x4[:, d, 0] = p2_sm[:, 0]
            probs_6x4[:, d, 3] = p2_sm[:, 1]

        clean_logit = self.clean_head(h).squeeze(-1)
        return {
            "logits_6x4": logits_6x4,
            "probs_6x4": probs_6x4,
            "severity_s": s_all,
            "delta_res": delta,
            "clean_logit": clean_logit,
            "z_norm": z_norm,
        }


class CrossBatchMemoryThurstoneLoss(nn.Module):
    """
    Pillar 1 (Section 2.3) Cross-Batch Memory (`M = 512`) Thurstone Fidelity Ranking Loss:
      L_XBM^(d) = (1 / |P_d|) * sum_{(i,j) in P_d} w(y_{i,d}, y_{j,d}) * [1 - sqrt(Phi((s_{i,d} - s_{j,d}) / (sqrt(2)*sigma_0)))]
    Strictly excludes same-scene pairs (`group_i != group_j`).
    """
    def __init__(self, capacity: int = 512, sigma0: float = 1.0, adj_weight: float = 2.0, same_group: bool = False):
        super().__init__()
        self.capacity = int(capacity)
        self.sigma0 = float(sigma0)
        # audit 2026-11-14 (P4b): weight of adjacent-grade pairs (2.0 = original) and whether pairs of frames from the
        # same video may be compared (False = original)
        self.adj_weight = float(adj_weight)
        self.same_group = bool(same_group)
        self.register_buffer("mem_s", torch.zeros(capacity, 6, dtype=torch.float32))
        self.register_buffer("mem_y", torch.zeros(capacity, 6, dtype=torch.long))
        self.register_buffer("mem_g", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, s_det: torch.Tensor, y_det: torch.Tensor, g_det: torch.Tensor) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            ws = dist.get_world_size()
            s_list = [torch.zeros_like(s_det) for _ in range(ws)]
            y_list = [torch.zeros_like(y_det) for _ in range(ws)]
            g_list = [torch.zeros_like(g_det) for _ in range(ws)]
            dist.all_gather(s_list, s_det.contiguous())
            dist.all_gather(y_list, y_det.contiguous())
            dist.all_gather(g_list, g_det.contiguous())
            s_det = torch.cat(s_list, dim=0)
            y_det = torch.cat(y_list, dim=0)
            g_det = torch.cat(g_list, dim=0)

        n = s_det.shape[0]
        if n >= self.capacity:
            self.mem_s.copy_(s_det[-self.capacity :])
            self.mem_y.copy_(y_det[-self.capacity :])
            self.mem_g.copy_(g_det[-self.capacity :])
            self.ptr.fill_(0)
            self.filled.fill_(self.capacity)
            return

        p = int(self.ptr.item())
        end = p + n
        if end <= self.capacity:
            self.mem_s[p:end].copy_(s_det)
            self.mem_y[p:end].copy_(y_det)
            self.mem_g[p:end].copy_(g_det)
        else:
            first = self.capacity - p
            self.mem_s[p:].copy_(s_det[:first])
            self.mem_y[p:].copy_(y_det[:first])
            self.mem_g[p:].copy_(g_det[:first])
            rem = n - first
            self.mem_s[:rem].copy_(s_det[first:])
            self.mem_y[:rem].copy_(y_det[first:])
            self.mem_g[:rem].copy_(g_det[first:])
        self.ptr.fill_(end % self.capacity)
        self.filled.fill_(min(self.capacity, int(self.filled.item()) + n))

    def forward(
        self,
        severity_s: torch.Tensor,
        grades: torch.Tensor,
        group_hashes: torch.Tensor,
        update_memory: bool = True,
    ) -> torch.Tensor:
        s_cur = severity_s.float()
        y_cur = grades.long()
        g_cur = group_hashes.long()

        n_mem = int(self.filled.item())
        if n_mem > 0:
            s_ref = torch.cat([s_cur.detach(), self.mem_s[:n_mem]], dim=0)
            y_ref = torch.cat([y_cur, self.mem_y[:n_mem]], dim=0)
            g_ref = torch.cat([g_cur, self.mem_g[:n_mem]], dim=0)
        else:
            s_ref = s_cur.detach()
            y_ref = y_cur
            g_ref = g_cur

        diff_group = g_cur.unsqueeze(1) != g_ref.unsqueeze(0)  # [B, M_tot]
        if self.same_group:
            diff_group = torch.ones_like(diff_group)
        inv_denom = 1.0 / (2.0 * self.sigma0)  # (s_i - s_j)/(sqrt(2)*sigma0) divided by sqrt(2) inside erf -> / (2*sigma0)

        dim_losses = []
        for d in range(6):
            yi = y_cur[:, d].unsqueeze(1)  # [B, 1]
            yj = y_ref[:, d].unsqueeze(0)  # [1, M_tot]
            si = s_cur[:, d].unsqueeze(1)
            sj = s_ref[:, d].unsqueeze(0)

            # Consider both (y_i > y_j) and (y_i < y_j) so gradient flows into s_cur whether i is higher or lower!
            higher = (yi > yj) & diff_group
            lower = (yi < yj) & diff_group
            valid = higher | lower
            if not valid.any():
                continue

            sign = torch.where(higher, 1.0, -1.0)
            gap = torch.abs(yi - yj)
            # Adjacent hard boundaries (gap == 1 on full dims, or (3,0) on binary dims) get weight 2.0
            if d in FULL_DIMS:
                w = torch.where(gap == 1, self.adj_weight, 1.0)
            else:
                w = torch.full_like(sign, 1.5)

            margin_z = sign * (si - sj) * inv_denom
            phi = 0.5 * (1.0 + torch.erf(margin_z))
            fidelity_loss = 1.0 - torch.sqrt(phi.clamp(min=1e-6, max=1.0))
            w_valid = w[valid]
            dim_losses.append((fidelity_loss[valid] * w_valid).sum() / w_valid.sum().clamp(min=1.0))

        if update_memory and self.training:
            self._enqueue(s_cur.detach(), y_cur.detach(), g_cur.detach())

        if not dim_losses:
            return s_cur.sum() * 0.0
        return torch.stack(dim_losses).mean()


class QingyuUnimodalCompositeLoss(nn.Module):
    """
    Pillar 4 (Section 5.2) Batch-Size-Independent Calibrated Ordinal Loss:
      L_total = SoftOrdinalKL(sigma=0.35) + lambda_rps * RPS_EMD2 + lambda_rank * XBM_Thurstone
                + lambda_delta * ||delta||_2^2 + lambda_clean * BCE_AllClean + lambda_kd * OOF_Teacher_KL
    """
    def __init__(
        self,
        soft_sigma: float = 0.35,
        lambda_rps: float = 0.45,
        lambda_rank: float = 0.25,
        lambda_delta: float = 1e-2,
        lambda_clean: float = 0.25,
        xbm_capacity: int = 512,
        kd_temp: float = 1.5,
        noise_T=None,
        noise_alpha: float = 0.0,
        rank_adj_weight: float = 2.0,
        rank_same_group: bool = False,
    ):
        super().__init__()
        self.soft_sigma = float(soft_sigma)
        self.lambda_rps = float(lambda_rps)
        self.lambda_rank = float(lambda_rank)
        self.lambda_delta = float(lambda_delta)
        self.lambda_clean = float(lambda_clean)
        self.kd_temp = float(kd_temp)
        self.xbm = CrossBatchMemoryThurstoneLoss(capacity=xbm_capacity, sigma0=1.0, adj_weight=rank_adj_weight,
                                                 same_group=rank_same_group)

        # Precompute [4, 4] narrow Gaussian soft target table q_y^{soft}(k) for y in {0, 1, 2, 3}
        q_table = torch.zeros(4, 4, dtype=torch.float32)
        ks = torch.arange(4, dtype=torch.float32)
        # Keep y == 0 (无问题) crisp (sigma_clean = 0.18 -> 99.9999% on 0) so Clean NEVER leaks into P2!
        q_table[0, 0] = 1.0
        for yv in (1, 2, 3):
            if soft_sigma > 0:
                w = torch.exp(-0.5 * ((ks - float(yv)) / soft_sigma) ** 2)
                # Prevent positive grade smoothing from leaking appreciable mass onto Clean (k=0) for P1/P0
                if yv >= 2:
                    w[0] = 0.0
                else:
                    w[0] *= 0.15
                q_table[yv] = w / w.sum()
            else:
                q_table[yv, yv] = 1.0
        self.register_buffer("q_soft_full", q_table)
        # audit 2026-11-14 (P4a): per-dimension soft targets mixed with the grade-transition matrix measured between
        # sibling frames of one video (row y = distribution of a sibling's label given label y; clean rows stay crisp)
        q6 = q_table.unsqueeze(0).repeat(6, 1, 1)
        if noise_T is not None and noise_alpha > 0:
            T = torch.as_tensor(np.asarray(noise_T), dtype=torch.float32)
            for d in FULL_DIMS:
                for yv in (1, 2, 3):
                    q6[d, yv] = (1.0 - noise_alpha) * q_table[yv] + noise_alpha * T[d, yv]
        self.register_buffer("q_soft_6", q6)

        # Moderate inverse-sqrt rarity weights per (dim, grade) so rare P0s/OverGen get crisp recall without calibration distortion
        cls_w = torch.ones(6, 4, dtype=torch.float32)
        cls_w[:, 3] = 1.65        # P0 priority
        cls_w[4, 1:] = 1.85       # 过度生成 P2/P1/P0
        cls_w[2, 2:] = 1.75       # 过锐 P1/P0
        cls_w[3, 3] = 1.90        # 花屏 P0
        cls_w[5, 3] = 1.90        # 虚化突兀 P0
        self.register_buffer("cls_weights", cls_w)

    def forward(
        self,
        head_out: Dict[str, torch.Tensor],
        grades: torch.Tensor,
        group_hashes: torch.Tensor,
        teacher_probs_6x4: Optional[torch.Tensor] = None,
        lambda_kd: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        logits_6x4 = head_out["logits_6x4"].float()
        probs_6x4 = head_out["probs_6x4"].float()
        s_all = head_out["severity_s"].float()
        delta_res = head_out["delta_res"].float()
        clean_logit = head_out["clean_logit"].float()
        y = grades.long()
        is_clean_img = (y == 0).all(dim=1)

        kl_losses = []
        rps_losses = []

        for d in FULL_DIMS:
            yd = y[:, d].clamp(0, 3)
            q_tgt = self.q_soft_6[d][yd]  # [B, 4] (== q_soft_full[yd] unless noise targets are on)
            log_p = F.log_softmax(logits_6x4[:, d, :], dim=-1)
            # KL(q_tgt || p) = sum_k q_tgt * (log q_tgt - log_p); drop constant entropy or include exact KL
            kl_per_i = torch.sum(q_tgt * (torch.log(q_tgt.clamp(min=1e-8)) - log_p), dim=-1)
            w_i = self.cls_weights[d, yd]
            # Extra protection on all-clean frames against P0 false positives
            w_i = torch.where(is_clean_img, w_i * 1.35, w_i)
            kl_losses.append((kl_per_i * w_i).sum() / w_i.sum().clamp(min=1.0))

            # 1D Squared EMD / Ranked Probability Score (RPS): sum_{k=0}^2 (CDF_pred(k) - I[y <= k])^2
            cdf_pred = torch.cumsum(probs_6x4[:, d, :3], dim=-1)  # [B, 3] for k=0,1,2
            ks = torch.arange(3, device=y.device).unsqueeze(0)    # [1, 3]
            cdf_true = (yd.unsqueeze(1) <= ks).float()
            rps_d = ((cdf_pred - cdf_true) ** 2).sum(dim=-1).mean()
            rps_losses.append(rps_d)

        for d in P0_ONLY_DIMS:
            yd_bin = (y[:, d] >= 3).long()  # 0 or 1
            logits_2 = logits_6x4[:, d, [0, 3]]
            log_p2 = F.log_softmax(logits_2, dim=-1)
            # Label smoothing 0.01 on positive only; strictly 0.0 on clean!
            q2 = torch.zeros_like(log_p2)
            q2[:, 1] = torch.where(yd_bin == 1, 0.985, 0.0)
            q2[:, 0] = 1.0 - q2[:, 1]
            kl_b = torch.sum(q2 * (torch.log(q2.clamp(min=1e-8)) - log_p2), dim=-1)
            w_b = torch.where(yd_bin == 1, self.cls_weights[d, 3], torch.where(is_clean_img, 1.50, 1.0))
            kl_losses.append((kl_b * w_b).sum() / w_b.sum().clamp(min=1.0))
            rps_losses.append(((probs_6x4[:, d, 3] - yd_bin.float()) ** 2).mean())

        dim_w = getattr(self, "dim_w", None)   # sprint: optional per-dim loss weights (canonical dim order), default off
        if dim_w is not None:
            order = list(FULL_DIMS) + list(P0_ONLY_DIMS)
            wv = torch.as_tensor([float(dim_w[d]) for d in order], device=logits_6x4.device)
            loss_kl = (torch.stack(kl_losses) * wv).sum() / wv.sum()
            loss_rps = (torch.stack(rps_losses) * wv).sum() / wv.sum()
        else:
            loss_kl = torch.stack(kl_losses).mean()
            loss_rps = torch.stack(rps_losses).mean()
        loss_rank = self.xbm(s_all, y, group_hashes, update_memory=self.training)
        loss_delta = (delta_res ** 2).mean()
        loss_clean = F.binary_cross_entropy_with_logits(clean_logit, is_clean_img.float())
        # Explicit Clean-P0 cliff suppressor: penalizes max P0 probability on true All-Clean images
        if is_clean_img.any():
            loss_clean = loss_clean + 0.50 * probs_6x4[is_clean_img, :, 3].amax(dim=1).mean()

        total = (
            loss_kl
            + self.lambda_rps * loss_rps
            + self.lambda_rank * loss_rank
            + self.lambda_delta * loss_delta
            + self.lambda_clean * loss_clean
        )

        loss_kd = total.new_tensor(0.0)
        if teacher_probs_6x4 is not None and lambda_kd > 1e-6:
            tp = teacher_probs_6x4.float().clamp(1e-6, 1.0 - 1e-6)
            kd_terms = []
            T = self.kd_temp
            for d in FULL_DIMS:
                t_log = torch.log(tp[:, d, :]) / T
                t_q = F.softmax(t_log, dim=-1)
                s_lp = F.log_softmax(logits_6x4[:, d, :] / T, dim=-1)
                kd_terms.append((T * T) * F.kl_div(s_lp, t_q, reduction="batchmean"))
            for d in P0_ONLY_DIMS:
                t_log2 = torch.log(tp[:, d, [0, 3]]) / T
                t_q2 = F.softmax(t_log2, dim=-1)
                s_lp2 = F.log_softmax(logits_6x4[:, d, [0, 3]] / T, dim=-1)
                kd_terms.append((T * T) * F.kl_div(s_lp2, t_q2, reduction="batchmean"))
            loss_kd = torch.stack(kd_terms).mean()
            total = total + float(lambda_kd) * loss_kd

        return {
            "total": total,
            "kl": loss_kl.detach(),
            "rps": loss_rps.detach(),
            "rank": loss_rank.detach(),
            "delta": loss_delta.detach(),
            "clean": loss_clean.detach(),
            "kd": loss_kd.detach(),
        }
