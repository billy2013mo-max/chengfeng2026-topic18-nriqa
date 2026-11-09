#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/prism/reference_anchor_net.py
==================================
PRISM Core Mathematical Module 2:
Scene-Matched Reference-Anchored Severity Comparator (RA-IQA).
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = '/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from code.prism.heads_and_losses import (
        CLASSES,
        FULL_CLASSES,
        convert_p012_to_conditional_hazards,
        convert_conditional_hazards_to_p012,
    )
except ImportError:
    from heads_and_losses import (
        CLASSES,
        FULL_CLASSES,
        convert_p012_to_conditional_hazards,
        convert_conditional_hazards_to_p012,
    )


class SceneMatchedAnchorBank:
    """
    Stores authentic real training anchors (`split == 'train'`) and retrieves Top-K scene-matched
    reference anchors across severity grades {0, P2, P1, P0} for any query image.
    """
    def __init__(
        self,
        train_names: list,
        train_scene_feats: np.ndarray,
        train_severity_feats: np.ndarray,
        train_grades: np.ndarray,
        top_k: int = 12,
        temperature: float = 0.12,
    ):
        self.train_names = [str(n) for n in train_names]
        self.name_to_train_idx = {n: i for i, n in enumerate(self.train_names)}
        self.top_k = top_k
        self.temperature = temperature

        self.scene_mu = train_scene_feats.mean(axis=0, keepdims=True)
        self.scene_sd = train_scene_feats.std(axis=0, keepdims=True) + 1e-5
        s_norm = np.clip((train_scene_feats - self.scene_mu) / self.scene_sd, -5.0, 5.0)
        self.train_scene_norm = s_norm / (np.linalg.norm(s_norm, axis=1, keepdims=True) + 1e-8)

        self.sev_mu = train_severity_feats.mean(axis=0, keepdims=True)
        self.sev_sd = train_severity_feats.std(axis=0, keepdims=True) + 1e-5
        self.train_sev_norm = np.clip((train_severity_feats - self.sev_mu) / self.sev_sd, -6.0, 6.0).astype(np.float32)
        self.train_grades = np.asarray(train_grades, dtype=np.int64)

        self.grade_pools = {}
        for ci in range(6):
            for g in (0, 1, 2, 3):
                idx = np.where(self.train_grades[:, ci] == g)[0]
                self.grade_pools[(ci, g)] = idx

    def normalize_query(self, query_scene_feats: np.ndarray, query_sev_feats: np.ndarray):
        s_norm = np.clip((query_scene_feats - self.scene_mu) / self.scene_sd, -5.0, 5.0)
        s_norm = s_norm / (np.linalg.norm(s_norm, axis=1, keepdims=True) + 1e-8)
        v_norm = np.clip((query_sev_feats - self.sev_mu) / self.sev_sd, -6.0, 6.0).astype(np.float32)
        return s_norm.astype(np.float32), v_norm

    def compute_anchor_relational_descriptors(
        self,
        query_names: list,
        query_scene_feats: np.ndarray,
        query_sev_feats: np.ndarray,
        query_p0: np.ndarray,
        query_p1: np.ndarray,
        query_p2: np.ndarray,
    ) -> np.ndarray:
        q_scene, q_sev = self.normalize_query(query_scene_feats, query_sev_feats)
        sim_mat = q_scene @ self.train_scene_norm.T

        for i, qnm in enumerate(query_names):
            s_qnm = str(qnm)
            if s_qnm in self.name_to_train_idx:
                sim_mat[i, self.name_to_train_idx[s_qnm]] = -1e9

        h_det, h_sev1, h_sev0 = convert_p012_to_conditional_hazards(query_p0, query_p1, query_p2)
        N_q = len(query_names)
        rel_feats = np.zeros((N_q, 4, 16), dtype=np.float32)

        K_loc = min(32, sim_mat.shape[1])
        top_loc_idx = np.argpartition(-sim_mat, kth=K_loc - 1, axis=1)[:, :K_loc]
        top_loc_sim = np.take_along_axis(sim_mat, top_loc_idx, axis=1)
        loc_w = np.exp((top_loc_sim - top_loc_sim.max(axis=1, keepdims=True)) / self.temperature)
        loc_w /= (loc_w.sum(axis=1, keepdims=True) + 1e-8)

        max_p0_all = query_p0.max(axis=1)
        max_ge1_all = h_det.max(axis=1)

        for k, cname in enumerate(FULL_CLASSES):
            ci = CLASSES.index(cname)
            l_det = np.log(np.clip(h_det[:, ci], 1e-5, 1.0 - 1e-5) / (1.0 - np.clip(h_det[:, ci], 1e-5, 1.0 - 1e-5)))
            l_s1  = np.log(np.clip(h_sev1[:, k], 1e-5, 1.0 - 1e-5) / (1.0 - np.clip(h_sev1[:, k], 1e-5, 1.0 - 1e-5)))
            l_s0  = np.log(np.clip(h_sev0[:, k], 1e-5, 1.0 - 1e-5) / (1.0 - np.clip(h_sev0[:, k], 1e-5, 1.0 - 1e-5)))

            rel_feats[:, k, 0] = l_det
            rel_feats[:, k, 1] = l_s1
            rel_feats[:, k, 2] = l_s0

            c_cols = slice(3 * k, 3 * k + 3)
            q_vec_k = q_sev[:, c_cols]

            anchor_centroids = []
            dists_g = []
            for g in (0, 1, 2, 3):
                pool = self.grade_pools[(ci, g)]
                if len(pool) == 0:
                    anchor_centroids.append(np.zeros_like(q_vec_k))
                    dists_g.append(np.ones(N_q, dtype=np.float32))
                    continue
                sim_pool = sim_mat[:, pool]
                kk = min(self.top_k, len(pool))
                idx_part = np.argpartition(-sim_pool, kth=kk - 1, axis=1)[:, :kk]
                s_top = np.take_along_axis(sim_pool, idx_part, axis=1)
                w_top = np.exp((s_top - s_top.max(axis=1, keepdims=True)) / self.temperature)
                w_top /= (w_top.sum(axis=1, keepdims=True) + 1e-8)

                tr_pool_vecs = self.train_sev_norm[pool[idx_part]][:, :, c_cols]
                cent_g = (w_top[:, :, None] * tr_pool_vecs).sum(axis=1)
                dist_g = np.linalg.norm(q_vec_k - cent_g, axis=1)
                anchor_centroids.append(cent_g)
                dists_g.append(dist_g)
                rel_feats[:, k, 3 + g] = dist_g

            span_12 = np.linalg.norm(anchor_centroids[2] - anchor_centroids[1], axis=1) + 0.25
            span_01 = np.linalg.norm(anchor_centroids[3] - anchor_centroids[2], axis=1) + 0.25
            span_02 = np.linalg.norm(anchor_centroids[1] - anchor_centroids[0], axis=1) + 0.25

            rel_feats[:, k, 7] = (dists_g[1] - dists_g[2]) / span_12
            rel_feats[:, k, 8] = (dists_g[2] - dists_g[3]) / span_01
            rel_feats[:, k, 9] = (dists_g[0] - dists_g[1]) / span_02

            loc_grades = self.train_grades[top_loc_idx, ci]
            for g in (0, 1, 2, 3):
                rel_feats[:, k, 10 + g] = (loc_w * (loc_grades == g)).sum(axis=1)

            rel_feats[:, k, 14] = max_p0_all
            rel_feats[:, k, 15] = max_ge1_all

        return rel_feats


class AnchorCrossAttentionComparator(nn.Module):
    def __init__(self, feat_dim: int = 16, hidden_dim: int = 32, max_delta: float = 0.45):
        super().__init__()
        self.max_delta = max_delta
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            ) for _ in range(4)
        ])
        self.anchor_pos_scale = nn.Parameter(torch.zeros(4, 2))
        self.hazard_bias = nn.Parameter(torch.zeros(4, 2))

        for net in self.nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def forward(self, rel_feats: torch.Tensor, h_det: torch.Tensor, h_sev1: torch.Tensor, h_sev0: torch.Tensor):
        eps = 1e-5
        h_s1_out = []
        h_s0_out = []

        for k in range(4):
            fk = rel_feats[:, k, :]
            mlp_delta = self.nets[k](fk)
            pos_12 = fk[:, 7]
            pos_01 = fk[:, 8]

            d_s1 = self.max_delta * torch.tanh(
                mlp_delta[:, 0] + self.anchor_pos_scale[k, 0] * pos_12 + self.hazard_bias[k, 0]
            )
            d_s0 = self.max_delta * torch.tanh(
                mlp_delta[:, 1] + self.anchor_pos_scale[k, 1] * pos_01 + self.hazard_bias[k, 1]
            )

            logit_s1 = torch.logit(h_sev1[:, k].clamp(eps, 1.0 - eps)) + d_s1
            logit_s0 = torch.logit(h_sev0[:, k].clamp(eps, 1.0 - eps)) + d_s0
            h_s1_out.append(torch.sigmoid(logit_s1))
            h_s0_out.append(torch.sigmoid(logit_s0))

        return h_det, torch.stack(h_s1_out, dim=-1), torch.stack(h_s0_out, dim=-1)


def build_severity_descriptor_from_p012(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    h_det, h_sev1, h_sev0 = convert_p012_to_conditional_hazards(p0, p1, p2)
    eps = 1e-5
    cols = []
    for k, cname in enumerate(FULL_CLASSES):
        ci = CLASSES.index(cname)
        for arr in (h_det[:, ci], h_sev1[:, k], h_sev0[:, k]):
            c = np.clip(arr, eps, 1.0 - eps)
            cols.append(np.log(c / (1.0 - c)))
    for cname in ('花屏', '虚化突兀'):
        ci = CLASSES.index(cname)
        c = np.clip(h_det[:, ci], eps, 1.0 - eps)
        cols.append(np.log(c / (1.0 - c)))
    return np.stack(cols, axis=-1).astype(np.float32)


def fit_and_apply_reference_anchor_calibrator(
    anchor_bank: SceneMatchedAnchorBank,
    va_names: list,
    va_scene_feats: np.ndarray,
    va_p0: np.ndarray,
    va_p1: np.ndarray,
    va_p2: np.ndarray,
    gva: np.ndarray,
    eval_tuples: list,
    steps: int = 160,
    lr: float = 0.015,
    weight_decay: float = 0.05,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    va_sev = build_severity_descriptor_from_p012(va_p0, va_p1, va_p2)
    rel_va_np = anchor_bank.compute_anchor_relational_descriptors(
        va_names, va_scene_feats, va_sev, va_p0, va_p1, va_p2
    )
    h_det_va_np, h_s1_va_np, h_s0_va_np = convert_p012_to_conditional_hazards(va_p0, va_p1, va_p2)

    rel_va = torch.from_numpy(rel_va_np).float()
    h_det_va = torch.from_numpy(h_det_va_np).float()
    h_s1_va = torch.from_numpy(h_s1_va_np).float()
    h_s0_va = torch.from_numpy(h_s0_va_np).float()
    y_va = torch.from_numpy(np.asarray(gva, dtype=np.int64))

    comparator = AnchorCrossAttentionComparator(feat_dim=16, hidden_dim=32, max_delta=0.35)
    opt = torch.optim.AdamW(comparator.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(steps):
        opt.zero_grad()
        _, s1_cal, s0_cal = comparator(rel_va, h_det_va, h_s1_va, h_s0_va)

        loss = torch.tensor(0.0)
        for k, cname in enumerate(FULL_CLASSES):
            ci = CLASSES.index(cname)
            yk = y_va[:, ci]
            d = h_det_va[:, ci]
            s1 = s1_cal[:, k]
            s0 = s0_cal[:, k]

            p_p2 = d * (1.0 - s1)
            p_p1 = d * s1 * (1.0 - s0)
            p_p0 = d * s1 * s0

            m_ge1 = (yk >= 1).float()
            if m_ge1.sum() > 0:
                t_s1 = (yk >= 2).float()
                loss = loss + (F.binary_cross_entropy(s1.clamp(1e-5, 1.0 - 1e-5), t_s1, reduction='none') * m_ge1).sum() / m_ge1.sum()

            m_ge2 = (yk >= 2).float()
            if m_ge2.sum() > 0:
                t_s0 = (yk == 3).float()
                loss = loss + (F.binary_cross_entropy(s0.clamp(1e-5, 1.0 - 1e-5), t_s0, reduction='none') * m_ge2).sum() / m_ge2.sum()

            for g_val, p_g, w_g in [(1, p_p2, 1.2), (2, p_p1, 1.6), (3, p_p0, 1.5)]:
                t_g = (yk == g_val).float()
                tp = (p_g * t_g).sum()
                fp = (p_g * (1.0 - t_g)).sum()
                fn = ((1.0 - p_g) * t_g).sum()
                f1_g = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
                loss = loss + w_g * (1.0 - f1_g)

        l2_reg = sum((p ** 2).mean() for p in comparator.parameters())
        (loss + 0.08 * l2_reg).backward()
        opt.step()

    comparator.eval()
    outputs = []
    with torch.no_grad():
        for q_names, q_scene, q_p0, q_p1, q_p2 in eval_tuples:
            q_sev = build_severity_descriptor_from_p012(q_p0, q_p1, q_p2)
            rel_q = torch.from_numpy(
                anchor_bank.compute_anchor_relational_descriptors(q_names, q_scene, q_sev, q_p0, q_p1, q_p2)
            ).float()
            hd_q, hs1_q, hs0_q = convert_p012_to_conditional_hazards(q_p0, q_p1, q_p2)
            _, s1_ref, s0_ref = comparator(
                rel_q,
                torch.from_numpy(hd_q).float(),
                torch.from_numpy(hs1_q).float(),
                torch.from_numpy(hs0_q).float(),
            )
            p0_out, p1_out, p2_out = convert_conditional_hazards_to_p012(
                hd_q, s1_ref.numpy(), s0_ref.numpy()
            )
            outputs.append((p0_out, p1_out, p2_out))

    return comparator, outputs
