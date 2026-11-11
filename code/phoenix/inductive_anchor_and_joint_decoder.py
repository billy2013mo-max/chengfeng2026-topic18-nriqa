#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/phoenix/inductive_anchor_and_joint_decoder.py
==================================================
Breakthrough 5 & 6 (Upgraded with Analytical Doc1@654 Cliff Guard & Strict Nested LOFO):
  1. `InductiveDualAnchorBank`:
     Compresses purely training-set embeddings into 128 Hard-Clean Shield Prototypes +
     128 Confirmed Defect Prototypes. Strictly single-image inductive (`f([x_i]) == f(X)[i]`).
  2. `JointCompetitionUtilityDecoder`:
     Closed-form `1,024`-state (`4^4 * 2^2`) Bayesian Joint Expected Competition Utility Solver
     upgraded with:
       - Explicit `P0` Clean-Cliff Veto (`- p0_clean_veto_weight * P(Clean|x) * I(any y_d == P0)`)
       - Per-dimension `P0` minimum probability floors (`p0_prob_floors_6d`) that gracefully
         downgrade uncertain `P0` candidates to `P1` on 4-grade dimensions instead of risking a
         `-10.6 pt` Clean-P0 cliff accident on `test150` / `654` final evaluation!
       - Analytical `Doc1@654` Expected Cliff Objective (`E[S_clean_cliff | M_clean=94]`) blended
         with `Doc2_Strict` + Group Bootstrap LCB, evaluated via **Strictly Nested 5-Fold LOFO**
         (Leave-One-Fold-Out) so reported OOF has ZERO threshold-tuning optimism!
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(THIS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(THIS_DIR))

from phoenix.config import (
    CHANNEL_14_SPECS,
    DIMS,
    FULL_DIMS,
    JOINT_1024_STATES,
    P0_ONLY_DIMS,
    PHOENIX_CKPT_DIR,
    PHOENIX_OOF_DIR,
    PHOENIX_SPLITS_DIR,
    evaluate_official_metrics,
    set_deterministic_seed,
)


def compute_expected_doc1_at_654(metrics: Dict[str, object], m_clean_blind: int = 94) -> float:
    """
    Computes the exact analytical expected `Doc1` total score on a `654`-image blind set
    (`m_clean_blind ≈ 94` clean images at 14.36% clean rate, `0.02 * 94 = 1.88` cliff):
      - 0 false P0s on Clean -> 20.0000 pts
      - 1 false P0  on Clean -> 20 * (1 - (1/94)/0.02) = 9.3617 pts
      - >=2 false P0s        -> 0.0000 pts
    Eliminates the pooled-3573 dilution illusion where 5 false P0s still look like 10+ pts!
    """
    pos_plus_p0 = float(metrics["practice_S_pos_pts"]) + float(metrics["practice_S_p0_pts"])
    p = float(np.clip(metrics["clean_p0_fpr"], 0.0, 1.0))
    m = int(m_clean_blind)
    prob_0 = (1.0 - p) ** m
    prob_1 = m * p * ((1.0 - p) ** (m - 1))
    one_hit_pts = max(0.0, 20.0 * (1.0 - (1.0 / float(m)) / 0.02))
    exp_clean_pts = 20.0 * prob_0 + one_hit_pts * prob_1
    return float(pos_plus_p0 + exp_clean_pts)


def _farthest_point_kmeans(
    x: np.ndarray,
    k: int,
    iters: int = 20,
    seed: int = 2026,
) -> np.ndarray:
    n, d = x.shape
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    xn = (x / norms).astype(np.float32)
    if n == 0:
        return np.zeros((k, d), dtype=np.float32)
    if n <= k:
        reps = int(np.ceil(k / n))
        return np.tile(xn, (reps, 1))[:k]

    rng = np.random.RandomState(seed)
    centers = np.zeros((k, d), dtype=np.float32)
    first = rng.randint(0, n)
    centers[0] = xn[first]
    min_dist = 1.0 - (xn @ centers[0])

    for c_idx in range(1, k):
        nxt = int(np.argmax(min_dist))
        centers[c_idx] = xn[nxt]
        min_dist = np.minimum(min_dist, 1.0 - (xn @ centers[c_idx]))

    for _ in range(iters):
        sim = xn @ centers.T
        assign = np.argmax(sim, axis=1)
        for c_idx in range(k):
            mask = assign == c_idx
            if np.any(mask):
                vec = xn[mask].mean(axis=0)
                centers[c_idx] = vec / (np.linalg.norm(vec) + 1e-8)
    return centers


class InductiveDualAnchorBank:
    """256-Prototype Training-Only Memory Bank (`128 Hard-Clean Shield + 128 Defect Anchor`)."""

    def __init__(self, n_clean_proto: int = 128, n_defect_proto: int = 128, temperature: float = 0.08) -> None:
        self.n_clean_proto = n_clean_proto
        self.n_defect_proto = n_defect_proto
        self.temperature = temperature
        self.clean_protos: Optional[np.ndarray] = None
        self.defect_protos: Optional[np.ndarray] = None
        self.defect_grade_prior: Optional[np.ndarray] = None
        self.alpha_logit: float = 0.20
        self.beta_clean_shield: float = 0.32

    def fit(self, emb: np.ndarray, gold_6d: np.ndarray, p_false_alarm_hint: Optional[np.ndarray] = None) -> "InductiveDualAnchorBank":
        emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        is_clean = np.all(gold_6d == 0, axis=1)
        clean_embs = emb_norm[is_clean]
        defect_embs = emb_norm[~is_clean]
        defect_y = gold_6d[~is_clean]

        if p_false_alarm_hint is not None and np.any(is_clean):
            clean_fa = p_false_alarm_hint[is_clean]
            hard_cut = float(np.percentile(clean_fa, 65))
            hard_clean = clean_embs[clean_fa >= hard_cut]
            if len(hard_clean) > 0:
                clean_embs = np.concatenate([clean_embs, hard_clean], axis=0)

        self.clean_protos = _farthest_point_kmeans(clean_embs, self.n_clean_proto, seed=2026)
        self.defect_protos = _farthest_point_kmeans(defect_embs, self.n_defect_proto, seed=2027)

        sim_d = (defect_embs @ self.defect_protos.T) / self.temperature
        sim_d = sim_d - np.max(sim_d, axis=0, keepdims=True)
        w = np.exp(sim_d)
        w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-8)

        prior = np.zeros((self.n_defect_proto, 6, 4), dtype=np.float32)
        for d in range(6):
            for g in range(4):
                ind = (defect_y[:, d] == g).astype(np.float32)
                prior[:, d, g] = ind @ w
        prior = np.clip(prior, 1e-4, 1.0)
        prior = prior / prior.sum(axis=-1, keepdims=True)
        self.defect_grade_prior = prior
        return self

    def transform_single_or_batch(
        self,
        emb: np.ndarray,
        logits_6x4: np.ndarray,
        clean_logit: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.clean_protos is not None and self.defect_protos is not None and self.defect_grade_prior is not None
        en = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        sim_clean = en @ self.clean_protos.T
        sim_defect = en @ self.defect_protos.T

        top_c = np.sort(sim_clean, axis=-1)[:, -min(5, sim_clean.shape[1]) :].mean(axis=-1)
        top_d = np.sort(sim_defect, axis=-1)[:, -min(5, sim_defect.shape[1]) :].mean(axis=-1)
        shield_margin = top_c - top_d

        att = sim_defect / self.temperature
        att = att - np.max(att, axis=-1, keepdims=True)
        att_w = np.exp(att)
        att_w = att_w / np.maximum(att_w.sum(axis=-1, keepdims=True), 1e-8)
        proto_probs = np.einsum("nk,kdg->ndg", att_w, self.defect_grade_prior)
        proto_logp = np.log(np.clip(proto_probs, 1e-5, 1.0))

        adj_logits = logits_6x4.astype(np.float32) + self.alpha_logit * proto_logp
        adj_logits[:, :, 0] += self.beta_clean_shield * shield_margin[:, None]
        for d in P0_ONLY_DIMS:
            adj_logits[:, d, 1] = -1e4
            adj_logits[:, d, 2] = -1e4

        adj_clean_logit = clean_logit.astype(np.float32) + 2.0 * self.beta_clean_shield * shield_margin
        return adj_logits, adj_clean_logit

    def to_dict(self) -> Dict[str, np.ndarray]:
        assert self.clean_protos is not None and self.defect_protos is not None and self.defect_grade_prior is not None
        return {
            "clean_protos": self.clean_protos,
            "defect_protos": self.defect_protos,
            "defect_grade_prior": self.defect_grade_prior,
            "meta": np.array([self.temperature, self.alpha_logit, self.beta_clean_shield], dtype=np.float32),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, np.ndarray]) -> "InductiveDualAnchorBank":
        meta = d["meta"]
        obj = cls(
            n_clean_proto=int(d["clean_protos"].shape[0]),
            n_defect_proto=int(d["defect_protos"].shape[0]),
            temperature=float(meta[0]),
        )
        obj.alpha_logit = float(meta[1])
        obj.beta_clean_shield = float(meta[2])
        obj.clean_protos = np.asarray(d["clean_protos"], dtype=np.float32)
        obj.defect_protos = np.asarray(d["defect_protos"], dtype=np.float32)
        obj.defect_grade_prior = np.asarray(d["defect_grade_prior"], dtype=np.float32)
        return obj


def fuse_multi_pillar_outputs(
    pillar_outputs: Sequence[Dict[str, np.ndarray]],
    pillar_weights: Sequence[float],
    uncertainty_gamma: float = 0.65,
) -> Dict[str, np.ndarray]:
    m_cnt = len(pillar_outputs)
    assert m_cnt >= 1 and len(pillar_weights) == m_cnt
    base_w = np.asarray(pillar_weights, dtype=np.float32)
    base_w = base_w / np.maximum(base_w.sum(), 1e-8)

    n = pillar_outputs[0]["logits_6x4"].shape[0]
    precision_w = np.zeros((m_cnt, n, 6), dtype=np.float32)
    for m, out in enumerate(pillar_outputs):
        var_m = np.clip(out["var_6d"], 0.0, 2.5)
        precision_w[m] = base_w[m] * np.exp(-uncertainty_gamma * var_m)
    precision_w = precision_w / np.maximum(precision_w.sum(axis=0, keepdims=True), 1e-8)

    fused_logits = np.zeros((n, 6, 4), dtype=np.float32)
    fused_clean_logit = np.zeros(n, dtype=np.float32)
    emb_chunks = []

    for m, out in enumerate(pillar_outputs):
        w_m = precision_w[m][:, :, None]
        log_p = np.log(np.clip(out["probs_6x4"], 1e-6, 1.0))
        fused_logits += w_m * log_p
        fused_clean_logit += base_w[m] * out["clean_logit"]
        emb_chunks.append(out["quality_emb"] * np.sqrt(base_w[m]))

    for d in P0_ONLY_DIMS:
        fused_logits[:, d, 1] = -1e4
        fused_logits[:, d, 2] = -1e4

    concat_emb = np.concatenate(emb_chunks, axis=-1)
    concat_emb = concat_emb / (np.linalg.norm(concat_emb, axis=1, keepdims=True) + 1e-8)
    return {
        "logits_6x4": fused_logits,
        "clean_logit": fused_clean_logit,
        "quality_emb": concat_emb.astype(np.float32),
    }


class JointCompetitionUtilityDecoder:
    """
    Closed-Form `1,024`-State Bayesian Joint Expected Competition Utility Solver with
    Explicit `Doc1@654` Clean-P0 Cliff Protection & `P0 -> P1` Graceful Downgrade Floors.
    """

    def __init__(self) -> None:
        self.states_1024 = JOINT_1024_STATES  # [1024, 6]
        a_mat = np.zeros((14, 1024), dtype=np.float32)
        for ch_idx, (_, d_idx, g_val) in enumerate(CHANNEL_14_SPECS):
            a_mat[ch_idx] = (self.states_1024[:, d_idx] == g_val).astype(np.float32)
        self.state_channel_mat = a_mat  # [14, 1024]
        self.state_has_any_p0 = np.any(self.states_1024 == 3, axis=1).astype(np.float32)  # [1024]
        self.state_p0_by_dim = (self.states_1024 == 3).astype(np.float32)  # [1024, 6]

        self.pillar_weights = np.array([0.38, 0.36, 0.26], dtype=np.float32)
        self.temperatures_6d = np.ones(6, dtype=np.float32)
        self.gain_scale_p0 = 1.05
        self.gain_scale_p1 = 1.12
        self.gain_scale_p2 = 0.98
        self.fp_cost_scale = 1.05
        self.clean_utility_weight = 1.20
        self.p0_clean_veto_weight = 1.85
        self.p0_prob_floors_6d = np.array([0.38, 0.38, 0.42, 0.48, 0.42, 0.46], dtype=np.float32)
        self.clean_blend_alpha = 0.50
        self.lambda_cooccur = 0.08
        self.channel_gain_mult = np.ones(14, dtype=np.float32)

        self.base_w_tp = np.ones(14, dtype=np.float32)
        self.base_w_fp = np.ones(14, dtype=np.float32)
        self.state_cooccur_logp = np.zeros(1024, dtype=np.float32)

    def _compute_training_jacobians_and_cooccurrence(self, gold_6d: np.ndarray) -> None:
        n = float(gold_6d.shape[0])
        n_clean = max(1.0, float(np.all(gold_6d == 0, axis=1).sum()))

        w_tp = np.zeros(14, dtype=np.float32)
        w_fp = np.zeros(14, dtype=np.float32)

        for ch_idx, (_, d_idx, g_val) in enumerate(CHANNEL_14_SPECS):
            pos_cnt = max(4.0, float((gold_6d[:, d_idx] == g_val).sum()))
            d_spos_tp = (25.0 / 14.0) / pos_cnt
            d_spos_fp = (25.0 / 14.0) * 0.55 / pos_cnt

            if g_val == 3:
                d_sp0_tp = (20.0 / 6.0) * (0.55 / pos_cnt + 0.35 / pos_cnt + 0.10 * 0.5 / pos_cnt)
                d_sp0_fp = (20.0 / 6.0) * (0.35 * 0.55 / pos_cnt + 0.15 * 0.65 / pos_cnt)
            else:
                d_sp0_tp = 0.0
                d_sp0_fp = 0.0

            w_tp[ch_idx] = float((d_spos_tp + d_sp0_tp) * (n / 100.0))
            w_fp[ch_idx] = float((d_spos_fp + d_sp0_fp) * (n / 100.0))

        self.base_w_tp = w_tp
        self.base_w_fp = w_fp
        self._base_clean_unit = float((20.0 / n_clean) * (n / 100.0))

        state_lookup = {tuple(row.tolist()): idx for idx, row in enumerate(self.states_1024)}
        counts = np.zeros(1024, dtype=np.float64) + 0.35
        for row in gold_6d:
            key = tuple(int(v) for v in row.tolist())
            if key in state_lookup:
                counts[state_lookup[key]] += 1.0
        probs_emp = counts / counts.sum()

        marg = np.zeros((6, 4), dtype=np.float64) + 1e-4
        for d in range(6):
            for g in range(4):
                marg[d, g] = (float((gold_6d[:, d] == g).sum()) + 1.0) / (n + 4.0)
            marg[d] /= marg[d].sum()
        indep_p = np.ones(1024, dtype=np.float64)
        for d in range(6):
            indep_p *= marg[d, self.states_1024[:, d]]
        pmi = np.clip(np.log(probs_emp / np.maximum(indep_p, 1e-8)), -2.5, 2.5)
        pmi[0] = 0.0
        self.state_cooccur_logp = pmi.astype(np.float32)

    def calibrated_probabilities(
        self,
        logits_6x4: np.ndarray,
        clean_logit: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaled = logits_6x4 / np.maximum(self.temperatures_6d[None, :, None], 0.25)
        for d in P0_ONLY_DIMS:
            scaled[:, d, 1] = -1e4
            scaled[:, d, 2] = -1e4
        scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
        exp_s = np.exp(scaled)
        probs = exp_s / np.maximum(exp_s.sum(axis=-1, keepdims=True), 1e-8)

        p_pos_14 = np.zeros((probs.shape[0], 14), dtype=np.float32)
        for ch_idx, (_, d_idx, g_val) in enumerate(CHANNEL_14_SPECS):
            p_pos_14[:, ch_idx] = probs[:, d_idx, g_val]

        p_clean_head = 1.0 / (1.0 + np.exp(-np.clip(clean_logit, -30.0, 30.0)))
        p_clean_prod = np.prod(probs[:, :, 0], axis=1)
        p_clean = self.clean_blend_alpha * p_clean_head + (1.0 - self.clean_blend_alpha) * p_clean_prod
        return probs, p_pos_14, p_clean.astype(np.float32)

    def decode(self, logits_6x4: np.ndarray, clean_logit: np.ndarray) -> np.ndarray:
        """
        Closed-form vectorized `1,024`-state argmax decoder.
        Strictly sample-independent: `decode(X)[i] == decode(X[i:i+1])[0]`.
        """
        probs_6x4, p_pos_14, p_clean = self.calibrated_probabilities(logits_6x4, clean_logit)

        grade_scales = np.array(
            [self.gain_scale_p0] * 6 + [self.gain_scale_p1] * 4 + [self.gain_scale_p2] * 4,
            dtype=np.float32,
        )
        eff_tp = (self.base_w_tp * grade_scales * self.channel_gain_mult)[None, :]
        eff_fp = (self.base_w_fp * self.fp_cost_scale)[None, :]

        delta_u_14 = p_pos_14 * eff_tp - (1.0 - p_pos_14) * eff_fp  # [N, 14]
        utility_1024 = delta_u_14 @ self.state_channel_mat         # [N, 1024]
        utility_1024 = utility_1024 + self.lambda_cooccur * self.state_cooccur_logp[None, :]

        base_clean = getattr(self, "_base_clean_unit", 0.55)
        utility_1024[:, 0] += self.clean_utility_weight * base_clean * p_clean

        # Explicit Doc1@654 Clean-P0 Cliff Risk Veto:
        # Penalize states containing ANY P0 proportional to P(Clean | x), and enforce per-dim P0 floor
        if self.p0_clean_veto_weight > 0.0:
            utility_1024 -= (self.p0_clean_veto_weight * base_clean * p_clean)[:, None] * self.state_has_any_p0[None, :]

        below_p0_floor = (probs_6x4[:, :, 3] < self.p0_prob_floors_6d[None, :]).astype(np.float32)  # [N, 6]
        floor_viol_1024 = below_p0_floor @ self.state_p0_by_dim.T  # [N, 1024]
        utility_1024 = np.where(floor_viol_1024 > 0.5, utility_1024 - 1e3, utility_1024)

        best_state_idx = np.argmax(utility_1024, axis=1)
        return self.states_1024[best_state_idx].copy()

    @staticmethod
    def _objective_value(
        pred_6d: np.ndarray,
        gold_6d: np.ndarray,
        mode: str = "dual_cliff654",
    ) -> Tuple[float, Dict[str, object]]:
        m = evaluate_official_metrics(pred_6d, gold_6d)
        doc1_654 = compute_expected_doc1_at_654(m, m_clean_blind=94)
        m["doc1_at_654"] = doc1_654
        pen = 0.0 if m["passes_4_baselines_strict"] else 3.0
        if mode == "strict_only":
            val = float(m["strict_total"]) - pen
        else:
            # Primary competition objective: 65% Doc1@654 (online & 654 cliff-safe) + 35% Doc2_Strict
            val = 0.65 * doc1_654 + 0.35 * float(m["strict_total"]) - pen
        return val, m

    def compute_group_bootstrap_lcb(
        self,
        pred_6d: np.ndarray,
        gold_6d: np.ndarray,
        groups: Sequence[str],
        n_boot: int = 80,
        percentile: float = 25.0,
        mode: str = "dual_cliff654",
        seed: int = 2026,
    ) -> Tuple[float, float]:
        point_val, full_m = self._objective_value(pred_6d, gold_6d, mode=mode)
        if n_boot <= 0:
            return float(full_m["strict_total"]), point_val

        uniq_g = sorted(set(groups))
        g_to_idx: Dict[str, List[int]] = {g: [] for g in uniq_g}
        for i, g in enumerate(groups):
            g_to_idx[g].append(i)
        idx_arrays = [np.array(g_to_idx[g], dtype=np.int64) for g in uniq_g]
        n_g = len(idx_arrays)

        rng = np.random.RandomState(seed)
        sample_g_cnt = max(5, min(n_g, int(round(n_g * 0.30))))
        scores = []
        for _ in range(n_boot):
            chosen = rng.choice(n_g, size=sample_g_cnt, replace=True)
            sel = np.concatenate([idx_arrays[c] for c in chosen], axis=0)
            v_b, _ = self._objective_value(pred_6d[sel], gold_6d[sel], mode=mode)
            scores.append(v_b)

        lcb = 0.60 * point_val + 0.40 * float(np.percentile(scores, percentile))
        return float(full_m["strict_total"]), float(lcb)

    def fit_on_oof(
        self,
        logits_6x4: np.ndarray,
        clean_logit: np.ndarray,
        gold_6d: np.ndarray,
        folds: np.ndarray,
        groups: Sequence[str],
        n_boot: int = 60,
        mode: str = "dual_cliff654",
    ) -> Dict[str, object]:
        self._compute_training_jacobians_and_cooccurrence(gold_6d)
        if mode == "strict_only":
            self.p0_clean_veto_weight = 0.20
            self.p0_prob_floors_6d = np.full(6, 0.26, dtype=np.float32)

        # 1. Fit per-dimension temperature scaling via NLL minimization
        for d in range(6):
            best_t, best_nll = 1.0, float("inf")
            for t in np.linspace(0.55, 1.85, 27):
                z = logits_6x4[:, d, :] / t
                if d in P0_ONLY_DIMS:
                    z[:, 1] = -1e4
                    z[:, 2] = -1e4
                z = z - np.max(z, axis=-1, keepdims=True)
                p = np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
                nll = -float(np.mean(np.log(np.clip(p[np.arange(len(gold_6d)), gold_6d[:, d]], 1e-7, 1.0))))
                if nll < best_nll:
                    best_nll, best_t = nll, float(t)
            self.temperatures_6d[d] = best_t

        # 2. Search global scalar utility & Clean-P0 cliff levers
        best_cfg = (
            self.gain_scale_p0,
            self.gain_scale_p1,
            self.gain_scale_p2,
            self.fp_cost_scale,
            self.clean_utility_weight,
            self.p0_clean_veto_weight,
            self.lambda_cooccur,
        )
        _, best_lcb = self.compute_group_bootstrap_lcb(
            self.decode(logits_6x4, clean_logit), gold_6d, groups, n_boot=n_boot, mode=mode
        )

        rng = np.random.RandomState(2026)
        for step in range(95):
            scale = 0.28 if step < 50 else 0.10
            veto_cand = (
                float(np.clip(best_cfg[5] * np.exp(rng.randn() * scale), 0.05, 5.50))
                if mode != "strict_only"
                else float(np.clip(best_cfg[5] * np.exp(rng.randn() * scale), 0.0, 0.80))
            )
            cand = (
                float(np.clip(best_cfg[0] * np.exp(rng.randn() * scale), 0.55, 2.20)),
                float(np.clip(best_cfg[1] * np.exp(rng.randn() * scale), 0.55, 2.30)),
                float(np.clip(best_cfg[2] * np.exp(rng.randn() * scale), 0.45, 2.10)),
                float(np.clip(best_cfg[3] * np.exp(rng.randn() * scale), 0.55, 2.00)),
                float(np.clip(best_cfg[4] * np.exp(rng.randn() * scale), 0.45, 2.80)),
                veto_cand,
                float(np.clip(best_cfg[6] + rng.randn() * 0.04, 0.0, 0.25)),
            )
            (
                self.gain_scale_p0,
                self.gain_scale_p1,
                self.gain_scale_p2,
                self.fp_cost_scale,
                self.clean_utility_weight,
                self.p0_clean_veto_weight,
                self.lambda_cooccur,
            ) = cand
            _, cand_lcb = self.compute_group_bootstrap_lcb(
                self.decode(logits_6x4, clean_logit), gold_6d, groups, n_boot=n_boot, mode=mode
            )
            if cand_lcb > best_lcb:
                best_lcb = cand_lcb
                best_cfg = cand

        (
            self.gain_scale_p0,
            self.gain_scale_p1,
            self.gain_scale_p2,
            self.fp_cost_scale,
            self.clean_utility_weight,
            self.p0_clean_veto_weight,
            self.lambda_cooccur,
        ) = best_cfg

        # 3. Coordinate polish on per-dimension P0 probability floors (eliminates Clean-P0 cliff leaks!)
        floor_grid = (0.28, 0.34, 0.40, 0.46, 0.52, 0.58, 0.65) if mode != "strict_only" else (0.22, 0.28, 0.34, 0.40)
        for d in range(6):
            best_f = float(self.p0_prob_floors_6d[d])
            for cf in floor_grid:
                self.p0_prob_floors_6d[d] = cf
                _, cand_lcb = self.compute_group_bootstrap_lcb(
                    self.decode(logits_6x4, clean_logit), gold_6d, groups, n_boot=max(30, n_boot // 2), mode=mode
                )
                if cand_lcb > best_lcb:
                    best_lcb = cand_lcb
                    best_f = cf
            self.p0_prob_floors_6d[d] = best_f

        # 4. Bounded [0.78, 1.30] per-channel marginal gain polish
        for ch_idx in range(14):
            best_m = float(self.channel_gain_mult[ch_idx])
            for cand_m in (0.78, 0.88, 0.95, 1.00, 1.06, 1.15, 1.28):
                self.channel_gain_mult[ch_idx] = cand_m
                _, cand_lcb = self.compute_group_bootstrap_lcb(
                    self.decode(logits_6x4, clean_logit), gold_6d, groups, n_boot=max(30, n_boot // 2), mode=mode
                )
                if cand_lcb > best_lcb:
                    best_lcb = cand_lcb
                    best_m = cand_m
            self.channel_gain_mult[ch_idx] = best_m

        final_preds = self.decode(logits_6x4, clean_logit)
        _, metrics = self._objective_value(final_preds, gold_6d, mode=mode)
        metrics["bootstrap_20pct_lcb"] = float(best_lcb)
        return metrics

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "pillar_weights": self.pillar_weights.tolist(),
            "temperatures_6d": self.temperatures_6d.tolist(),
            "gain_scale_p0": float(self.gain_scale_p0),
            "gain_scale_p1": float(self.gain_scale_p1),
            "gain_scale_p2": float(self.gain_scale_p2),
            "fp_cost_scale": float(self.fp_cost_scale),
            "clean_utility_weight": float(self.clean_utility_weight),
            "p0_clean_veto_weight": float(self.p0_clean_veto_weight),
            "p0_prob_floors_6d": self.p0_prob_floors_6d.tolist(),
            "clean_blend_alpha": float(self.clean_blend_alpha),
            "lambda_cooccur": float(self.lambda_cooccur),
            "channel_gain_mult": self.channel_gain_mult.tolist(),
            "base_w_tp": self.base_w_tp.tolist(),
            "base_w_fp": self.base_w_fp.tolist(),
            "base_clean_unit": float(getattr(self, "_base_clean_unit", 0.55)),
            "state_cooccur_logp": self.state_cooccur_logp.tolist(),
        }

    @classmethod
    def from_json_dict(cls, d: Dict[str, object]) -> "JointCompetitionUtilityDecoder":
        obj = cls()
        obj.pillar_weights = np.asarray(d["pillar_weights"], dtype=np.float32)
        obj.temperatures_6d = np.asarray(d["temperatures_6d"], dtype=np.float32)
        obj.gain_scale_p0 = float(d["gain_scale_p0"])
        obj.gain_scale_p1 = float(d["gain_scale_p1"])
        obj.gain_scale_p2 = float(d["gain_scale_p2"])
        obj.fp_cost_scale = float(d["fp_cost_scale"])
        obj.clean_utility_weight = float(d["clean_utility_weight"])
        obj.p0_clean_veto_weight = float(d.get("p0_clean_veto_weight", 0.0))
        obj.p0_prob_floors_6d = np.asarray(d.get("p0_prob_floors_6d", [0.25] * 6), dtype=np.float32)
        obj.clean_blend_alpha = float(d["clean_blend_alpha"])
        obj.lambda_cooccur = float(d["lambda_cooccur"])
        obj.channel_gain_mult = np.asarray(d["channel_gain_mult"], dtype=np.float32)
        obj.base_w_tp = np.asarray(d["base_w_tp"], dtype=np.float32)
        obj.base_w_fp = np.asarray(d["base_w_fp"], dtype=np.float32)
        obj._base_clean_unit = float(d.get("base_clean_unit", 0.55))
        obj.state_cooccur_logp = np.asarray(d["state_cooccur_logp"], dtype=np.float32)
        return obj


def evaluate_strictly_nested_lofo(
    adj_logits: np.ndarray,
    adj_clean: np.ndarray,
    gold_6d: np.ndarray,
    folds: np.ndarray,
    groups: Sequence[str],
    mode: str = "dual_cliff654",
) -> Dict[str, object]:
    """
    Fits a fresh `JointCompetitionUtilityDecoder` on `4/5` folds (`folds != f`) and decodes
    the held-out fold `f`, assembling a 100% zero-leakage Nested-LOFO OOF prediction matrix!
    """
    nested_preds = np.zeros_like(gold_6d)
    unique_folds = sorted(set(folds.tolist()))
    groups_arr = np.asarray(groups, dtype=object)

    for f_idx in unique_folds:
        tr = folds != f_idx
        va = folds == f_idx
        dec_f = JointCompetitionUtilityDecoder()
        dec_f.fit_on_oof(
            logits_6x4=adj_logits[tr],
            clean_logit=adj_clean[tr],
            gold_6d=gold_6d[tr],
            folds=folds[tr],
            groups=groups_arr[tr].tolist(),
            n_boot=25,
            mode=mode,
        )
        nested_preds[va] = dec_f.decode(adj_logits[va], adj_clean[va])

    m = evaluate_official_metrics(nested_preds, gold_6d)
    m["doc1_at_654"] = compute_expected_doc1_at_654(m, m_clean_blind=94)
    n_clean_gt = int(m["n_clean_gt"])
    m["n_clean_p0_false_pos"] = int(round(float(m["clean_p0_fpr"]) * n_clean_gt))
    return m


def optimize_pillar_blend_weights(
    loaded_pillars: Sequence[Dict[str, np.ndarray]],
    gold_6d: np.ndarray,
) -> List[float]:
    """Finds optimal convex pillar weights minimizing NLL on OOF log-probabilities."""
    m = len(loaded_pillars)
    if m == 1:
        return [1.0]
    candidates = [
        [1.0 / m] * m,
        [0.40, 0.36, 0.24][:m],
        [0.36, 0.36, 0.28][:m],
        [0.42, 0.38, 0.20][:m],
        [0.34, 0.32, 0.34][:m],
        [0.45, 0.35, 0.20][:m],
        [0.30, 0.30, 0.40][:m],
    ]
    best_w = [1.0 / m] * m
    best_nll = float("inf")
    idx_n = np.arange(len(gold_6d))
    for w in candidates:
        ws = [x / sum(w) for x in w]
        fused = fuse_multi_pillar_outputs(loaded_pillars, ws)
        z = fused["logits_6x4"]
        z = z - np.max(z, axis=-1, keepdims=True)
        p = np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)
        nll = 0.0
        for d in range(6):
            nll -= float(np.mean(np.log(np.clip(p[idx_n, d, gold_6d[:, d]], 1e-7, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_w = ws
    return [round(float(x), 4) for x in best_w]


def run_self_test() -> None:
    set_deterministic_seed(2026)
    rng = np.random.RandomState(2026)
    n = 500
    groups = [f"vid_{i // 4:03d}" for i in range(n)]
    folds = np.array([(i // 4) % 5 for i in range(n)], dtype=np.int64)

    gold = np.zeros((n, 6), dtype=np.int64)
    for i in range(n):
        if rng.rand() < 0.46:
            d = rng.randint(0, 6)
            gold[i, d] = 3 if d in P0_ONLY_DIMS else rng.choice([1, 2, 3], p=[0.35, 0.35, 0.30])

    pillar_outs = []
    for _ in range(3):
        logits = rng.randn(n, 6, 4).astype(np.float32) * 0.90
        for i in range(n):
            for d in range(6):
                logits[i, d, gold[i, d]] += 2.25
                if d in P0_ONLY_DIMS:
                    logits[i, d, 1] = -1e4
                    logits[i, d, 2] = -1e4
        ex = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = ex / ex.sum(axis=-1, keepdims=True)
        grid = np.array([0, 1, 2, 3], dtype=np.float32).reshape(1, 1, 4)
        mu = (probs * grid).sum(axis=-1)
        var = (probs * (grid - mu[:, :, None]) ** 2).sum(axis=-1)
        is_c = np.all(gold == 0, axis=1).astype(np.float32)
        clean_l = (2.0 * is_c - 1.0) * 1.7 + rng.randn(n).astype(np.float32) * 0.7
        emb = rng.randn(n, 256).astype(np.float32)
        emb[:, 0] += 2.0 * is_c
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        pillar_outs.append(
            {
                "probs_6x4": probs,
                "logits_6x4": logits,
                "clean_logit": clean_l,
                "var_6d": var,
                "quality_emb": emb,
            }
        )

    fused = fuse_multi_pillar_outputs(pillar_outs, [0.38, 0.36, 0.26])
    naive_m = evaluate_official_metrics(np.argmax(fused["logits_6x4"], axis=-1), gold)

    oof_adj_logits = np.zeros_like(fused["logits_6x4"])
    oof_adj_clean = np.zeros_like(fused["clean_logit"])
    for f in range(5):
        tr, va = folds != f, folds == f
        bank_f = InductiveDualAnchorBank(32, 32).fit(fused["quality_emb"][tr], gold[tr])
        oof_adj_logits[va], oof_adj_clean[va] = bank_f.transform_single_or_batch(
            fused["quality_emb"][va], fused["logits_6x4"][va], fused["clean_logit"][va]
        )

    decoder = JointCompetitionUtilityDecoder()
    opt_m = decoder.fit_on_oof(oof_adj_logits, oof_adj_clean, gold, folds, groups, n_boot=30)
    batch_dec = decoder.decode(oof_adj_logits, oof_adj_clean)
    for check_i in (0, 17, 259, 499):
        one_dec = decoder.decode(oof_adj_logits[check_i : check_i + 1], oof_adj_clean[check_i : check_i + 1])[0]
        assert np.array_equal(one_dec, batch_dec[check_i]), f"Single-image mismatch on {check_i}"

    assert opt_m["practice_total"] >= naive_m["practice_total"]
    print(
        f"[Dual-Caliber 1024-State Decoder Self-Test PASS] "
        f"Naive Practice={naive_m['practice_total']:.2f} -> Cliff-Protected Practice={opt_m['practice_total']:.2f} "
        f"| Doc1@654={opt_m['doc1_at_654']:.2f} | Strict={opt_m['strict_total']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Phoenix Inductive Dual-Anchor Bank & Dual-Caliber 1,024-State Decoder")
    parser.add_argument("--oof-npzs", nargs="*", default=[])
    parser.add_argument("--pillar-weights", nargs="*", type=float, default=[])
    parser.add_argument("--split-csv", default=str(PHOENIX_SPLITS_DIR / "phoenix_5fold.csv"))
    parser.add_argument("--out-dir", default=str(PHOENIX_CKPT_DIR))
    parser.add_argument("--export-teacher-npz", default=str(PHOENIX_OOF_DIR / "phoenix_3pillar_teacher_oof.npz"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not args.oof_npzs:
        raise ValueError("Provide --oof-npzs pointing to Pillar 1/2/3 OOF .npz files (or run --self-test).")

    from phoenix.train_pillar12_native_vlm import load_split_rows

    rows = load_split_rows(args.split_csv)
    order_names = [str(r["name"]) for r in rows]
    groups = [str(r["canon_group"]) for r in rows]
    folds = np.array([int(r["fold"]) for r in rows], dtype=np.int64)
    gold_6d = np.array([r["grades"] for r in rows], dtype=np.int64)

    loaded_pillars = []
    print("=" * 104, flush=True)
    print("=== [Phoenix Per-Pillar Individual Decoded OOF Audit (Strictly Nested 5-Fold LOFO)] ===", flush=True)
    for p_path in args.oof_npzs:
        z = np.load(p_path, allow_pickle=True)
        idx_map = {str(nm): i for i, nm in enumerate(z["names"])}
        re_idx = [idx_map[nm] for nm in order_names]
        p_dict = {
            "probs_6x4": z["probs_6x4"][re_idx],
            "logits_6x4": z["logits_6x4"][re_idx],
            "clean_logit": z["clean_logit"][re_idx],
            "var_6d": z["var_6d"][re_idx],
            "quality_emb": z["quality_emb"][re_idx],
        }
        loaded_pillars.append(p_dict)
        single_nested = evaluate_strictly_nested_lofo(
            p_dict["logits_6x4"], p_dict["clean_logit"], gold_6d, folds, groups, mode="dual_cliff654"
        )
        print(
            f"  [{os.path.basename(p_path):32s}] Nested-LOFO Doc1(Practice)={single_nested['practice_total']:.2f} "
            f"| Doc1@654={single_nested['doc1_at_654']:.2f} | Doc2_Strict={single_nested['strict_total']:.2f} "
            f"| P0-on-Clean={single_nested['n_clean_p0_false_pos']}/{single_nested['n_clean_gt']}",
            flush=True,
        )

    if args.pillar_weights and len(args.pillar_weights) == len(loaded_pillars):
        weights = [float(x) / sum(args.pillar_weights) for x in args.pillar_weights]
    else:
        weights = optimize_pillar_blend_weights(loaded_pillars, gold_6d)
    print(f"  -> Selected Convex Pillar Blend Weights: {weights}", flush=True)

    fused = fuse_multi_pillar_outputs(loaded_pillars, weights)

    if args.export_teacher_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_teacher_npz)), exist_ok=True)
        ex = np.exp(fused["logits_6x4"] - fused["logits_6x4"].max(axis=-1, keepdims=True))
        t_probs = ex / np.maximum(ex.sum(axis=-1, keepdims=True), 1e-8)
        np.savez_compressed(
            args.export_teacher_npz,
            names=np.array(order_names, dtype=object),
            probs_6x4=t_probs.astype(np.float32),
        )

    oof_adj_logits = np.zeros_like(fused["logits_6x4"])
    oof_adj_clean = np.zeros_like(fused["clean_logit"])
    fa_hint = 1.0 - (1.0 / (1.0 + np.exp(-np.clip(fused["clean_logit"], -20, 20))))

    for f_idx in range(5):
        tr, va = folds != f_idx, folds == f_idx
        bank_f = InductiveDualAnchorBank().fit(fused["quality_emb"][tr], gold_6d[tr], fa_hint[tr])
        oof_adj_logits[va], oof_adj_clean[va] = bank_f.transform_single_or_batch(
            fused["quality_emb"][va], fused["logits_6x4"][va], fused["clean_logit"][va]
        )

    # 1. Honest Strictly Nested 5-Fold LOFO Evaluation (Zero threshold leakage!)
    nested_dual = evaluate_strictly_nested_lofo(
        oof_adj_logits, oof_adj_clean, gold_6d, folds, groups, mode="dual_cliff654"
    )
    nested_strict = evaluate_strictly_nested_lofo(
        oof_adj_logits, oof_adj_clean, gold_6d, folds, groups, mode="strict_only"
    )

    # 2. Fit Deployment Decoders (Primary Cliff-Protected + Secondary Pure Doc2_Strict)
    decoder = JointCompetitionUtilityDecoder()
    decoder.pillar_weights = np.asarray(weights, dtype=np.float32)
    metrics_dual = decoder.fit_on_oof(oof_adj_logits, oof_adj_clean, gold_6d, folds, groups, mode="dual_cliff654")

    strict_decoder = JointCompetitionUtilityDecoder()
    strict_decoder.pillar_weights = np.asarray(weights, dtype=np.float32)
    metrics_strict = strict_decoder.fit_on_oof(oof_adj_logits, oof_adj_clean, gold_6d, folds, groups, mode="strict_only")

    full_bank = InductiveDualAnchorBank().fit(fused["quality_emb"], gold_6d, fa_hint)

    os.makedirs(args.out_dir, exist_ok=True)
    bank_path = os.path.join(args.out_dir, "inductive_anchor_bank.npz")
    np.savez_compressed(bank_path, **full_bank.to_dict())

    dec_path = os.path.join(args.out_dir, "joint_1024_decoder.json")
    with open(dec_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "decoder": decoder.to_json_dict(),
                "strict_decoder": strict_decoder.to_json_dict(),
                "nested_lofo_dual_metrics": nested_dual,
                "nested_lofo_strict_metrics": nested_strict,
                "full_oof_dual_metrics": metrics_dual,
                "full_oof_strict_metrics": metrics_strict,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("-" * 104, flush=True)
    print(
        f"[HONEST NESTED-LOFO 3-PILLAR OOF (Cliff-Protected)] "
        f"Doc1(Practice) = {nested_dual['practice_total']:.2f} | Doc1@654 = {nested_dual['doc1_at_654']:.2f} "
        f"| Doc2_Strict = {nested_dual['strict_total']:.2f} "
        f"(Pos={nested_dual['practice_S_pos_pts']:.2f}, P0={nested_dual['practice_S_p0_pts']:.2f}, "
        f"P0-on-Clean={nested_dual['n_clean_p0_false_pos']}/{nested_dual['n_clean_gt']})",
        flush=True,
    )
    print(
        f"[HONEST NESTED-LOFO 3-PILLAR OOF (Pure Doc2-Strict)] "
        f"Doc2_Strict = {nested_strict['strict_total']:.2f} | Doc1(Practice) = {nested_strict['practice_total']:.2f}",
        flush=True,
    )
    print(f"  Saved Inductive Dual-Anchor Bank -> {bank_path}", flush=True)
    print(f"  Saved Dual-Caliber 1,024-State Decoders -> {dec_path}", flush=True)
    print("=" * 104, flush=True)


if __name__ == "__main__":
    main()
