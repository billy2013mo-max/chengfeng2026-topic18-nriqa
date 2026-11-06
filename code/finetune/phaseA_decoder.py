#!/usr/bin/env python3
"""
code/finetune/phaseA_decoder.py
Phase A (zero-retraining) decoder upgrades from the 55+ roadmap PDF:

  1. probs_14ch_to_mutually_exclusive_6x4: 14 cumulative channels -> P(Y_{i,c}=g).
  2. sinkhorn_prior_shift_calibration: damped batch prior-shift calibration that
     preserves within-class ranking while pulling inflated P1/P2 counts back to the
     training base rates.
  3. decode_competition_ratio: hierarchical roll-down with an explicit P1-vs-P2
     relative competition constraint (fixes the unconditional P1 capture bug).
  4. eum_bca_joint_decode: a plug-in expected-utility block coordinate ascent on top
     of decode_competition_ratio (optional; enabled when `n_rounds > 0`).

All functions are numpy-only and operate on (N, 14) cumulative probability matrices
in the same channel layout as code/utils.py.
"""
import numpy as np

from utils import FULL_DIMS, P0_ONLY_DIMS, ch14_to_p012, p012_to_14ch, competition_score

# Expected counts per 150 images, estimated from s16~s20 GT (roadmap PDF page 5).
GT_EXPECTED_COUNTS_150 = {
    0: {3: 42.9, 2: 16.5, 1: 26.8},  # 模糊: P0, P1, P2
    1: {3: 30.5, 2: 43.4, 1: 37.1},  # 纹理损失
    2: {3: 4.6,  2: 13.1, 1: 17.5},  # 过锐
    3: {3: 6.1},                     # 花屏 (binary)
    4: {3: 7.9,  2: 11.6, 1: 11.0},  # 过度生成
    5: {3: 4.6},                     # 虚化突兀 (binary)
}
GRADE = {0: 0, 1: 1, 2: 2, 3: 3}  # grade id == utils grade id


def probs_14ch_to_mutually_exclusive_6x4(probs_14ch: np.ndarray) -> np.ndarray:
    """(N,14) cumulative -> (N,6,4) mutually exclusive P(Y_{i,c}=g), g in {0,1(P2),2(P1),3(P0)}."""
    p0, p1, p2 = ch14_to_p012(probs_14ch)  # p0: P(P0), p1: P(P1), p2: P(P2)
    N = probs_14ch.shape[0]
    P = np.zeros((N, 6, 4), dtype=np.float64)
    for j, d in enumerate(FULL_DIMS):
        P[:, d, 3] = np.clip(p0[:, d], 1e-6, 1.0 - 1e-6)
        P[:, d, 2] = np.clip(p1[:, j], 1e-6, 1.0 - 1e-6)
        P[:, d, 1] = np.clip(p2[:, j], 1e-6, 1.0 - 1e-6)
        p_pos = P[:, d, 1] + P[:, d, 2] + P[:, d, 3]
        over = p_pos > 0.999
        if np.any(over):
            P[over, d, 1:] /= (p_pos[over, None] / 0.999)
        P[:, d, 0] = np.clip(1.0 - P[:, d, 1:].sum(axis=1), 1e-6, 1.0)
    for d in P0_ONLY_DIMS:
        P[:, d, 3] = np.clip(p0[:, d], 1e-6, 1.0 - 1e-6)
        P[:, d, 0] = 1.0 - P[:, d, 3]
    return P


def mutually_exclusive_6x4_to_14ch(P: np.ndarray) -> np.ndarray:
    """Inverse of probs_14ch_to_mutually_exclusive_6x4."""
    N = P.shape[0]
    p0 = np.zeros((N, 6), dtype=np.float64)
    p1 = np.zeros((N, len(FULL_DIMS)), dtype=np.float64)
    p2 = np.zeros((N, len(FULL_DIMS)), dtype=np.float64)
    for j, d in enumerate(FULL_DIMS):
        p0[:, d] = P[:, d, 3]
        p1[:, j] = P[:, d, 2]
        p2[:, j] = P[:, d, 1]
    for d in P0_ONLY_DIMS:
        p0[:, d] = P[:, d, 3]
    return p012_to_14ch(p0, p1, p2)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sinkhorn_prior_shift_calibration(
    P_6x4: np.ndarray,
    clean_mask: np.ndarray = None,
    strength: float = 0.65,
    n_iters: int = 25,
    temperature: float = 1.0,
    target_counts: dict = None,
) -> np.ndarray:
    """
    Damped batch prior-shift calibration for the four FULL dimensions.

    For each dim, find a class-bias vector mu such that the expected marginal counts
    over the non-clean images match the training base rates, while preserving the
    within-class ranking (the bias is additive in log-probability). `strength`
    interpolates the bias between 0 (no change) and 1 (strict match).

    Clean rows are never modified (they are hard-anchored to all-zero).
    """
    P = P_6x4.copy()
    N = P.shape[0]
    clean_mask = np.zeros(N, dtype=bool) if clean_mask is None else np.asarray(clean_mask, bool)
    target_counts = GT_EXPECTED_COUNTS_150 if target_counts is None else target_counts
    scale = N / 150.0

    for d in FULL_DIMS:
        rows = ~clean_mask
        if rows.sum() == 0:
            continue
        logp = np.log(np.clip(P[rows, d, :], 1e-8, 1.0))
        # Class order: P columns are [0(clean), 1(P2), 2(P1), 3(P0)].
        # The positive counts in GT_EXPECTED_COUNTS_150 are over all 150 images;
        # clean images contribute 0 positives, so scale by N/150. The remaining
        # mass in the non-clean subset is the target clean count for this dim.
        n_rows = int(rows.sum())
        pos_target = np.array([
            target_counts[d].get(1, 0.0) * scale,   # P2
            target_counts[d].get(2, 0.0) * scale,   # P1
            target_counts[d].get(3, 0.0) * scale,   # P0
        ], dtype=np.float64)
        clean_target = max(1e-3, n_rows - pos_target.sum())
        target = np.array([clean_target, pos_target[0], pos_target[1], pos_target[2]])
        target = np.maximum(target, 1e-3)

        mu = np.zeros(4, dtype=np.float64)
        eta = 0.35
        for _ in range(n_iters):
            Q = _softmax(logp / temperature + mu[None, :])
            expected = Q.sum(axis=0)
            mu += eta * (np.log(target) - np.log(np.maximum(expected, 1e-8)))
        mu = strength * mu
        P[rows, d, :] = _softmax(logp / temperature + mu[None, :])

    return P


def _decode_core(p0, p1, p2, t0, t1, t2, tau_any, kappa, roll_p2):
    N = p0.shape[0]
    preds = np.zeros((N, 6), dtype=np.int64)
    for j, d in enumerate(FULL_DIMS):
        p_any = p0[:, d] + p1[:, j] + p2[:, j]
        s1_roll = p1[:, j] + 0.45 * p0[:, d]
        pass_any = (p_any >= t2[j]) if roll_p2 else (p_any >= tau_any[d])
        is_p0 = pass_any & (p0[:, d] >= t0[d])
        # Explicit P1-vs-P2 competition: P1 must not merely clear its own threshold,
        # it must also beat the P2 evidence by a relative margin.
        comp = s1_roll >= float(kappa[j]) * np.maximum(p2[:, j], 1e-6)
        is_p1 = pass_any & (~is_p0) & (s1_roll >= t1[j]) & comp
        if roll_p2:
            is_p2 = pass_any & (~is_p0) & (~is_p1)
        else:
            is_p2 = pass_any & (~is_p0) & (~is_p1) & (p2[:, j] >= t2[j])
        preds[is_p2, d] = 1
        preds[is_p1, d] = 2
        preds[is_p0, d] = 3
    for d in P0_ONLY_DIMS:
        c0 = (p0[:, d] >= t0[d]) if roll_p2 else ((p0[:, d] >= t0[d]) & (p0[:, d] >= tau_any[d]))
        preds[:, d] = np.where(c0, 3, 0)
    return preds


def decode_competition_ratio(
    probs_14ch: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    tau_any: np.ndarray,
    kappa_p1p2: np.ndarray = None,
    base_clean: np.ndarray = None,
    roll_p2: bool = False,
) -> np.ndarray:
    """Hierarchical roll-down + P1/P2 competition constraint + Clean anchor lock."""
    p0, p1, p2 = ch14_to_p012(probs_14ch)
    kappa = np.ones(len(FULL_DIMS), dtype=np.float64) if kappa_p1p2 is None else np.asarray(kappa_p1p2, float)
    preds = _decode_core(p0, p1, p2, t0, t1, t2, tau_any, kappa, roll_p2)
    if base_clean is not None:
        bc = np.asarray(base_clean, bool)
        preds[bc] = 0
        preds[bc[:, None] & (preds == 3)] = 0
    return preds


def _score_with_pseudo_gold(pred, pseudo_gold):
    return competition_score(pred, pseudo_gold)["total"]


def eum_bca_joint_decode(
    probs_14ch: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    tau_any: np.ndarray,
    kappa_p1p2: np.ndarray = None,
    base_clean: np.ndarray = None,
    n_rounds: int = 2,
    max_cells: int = None,
) -> np.ndarray:
    """
    Expected-utility block coordinate ascent (EUM-BCA).

    This is a tractable plug-in approximation: for each non-clean cell, hold the other
    predictions as pseudo-ground-truth, enumerate the 4 (or 2) actions, and pick the
    action maximizing E_g~P(i,c)[score(pred_with_action, pseudo_gold_with_g)].
    Only cells whose posterior is non-trivial (max grade prob >= 0.55) are visited,
    and `max_cells` caps the work for large blind splits.
    """
    P = probs_14ch_to_mutually_exclusive_6x4(probs_14ch)
    base = decode_competition_ratio(
        probs_14ch, t0, t1, t2, tau_any, kappa_p1p2, base_clean=base_clean, roll_p2=False
    )
    pred = base.copy()
    N = pred.shape[0]
    rows = np.ones(N, dtype=bool) if base_clean is None else (~np.asarray(base_clean, bool))
    cells = []
    for i in np.where(rows)[0]:
        for d in list(FULL_DIMS) + list(P0_ONLY_DIMS):
            if P[i, d].max() >= 0.55:
                cells.append((i, d))
    if max_cells is not None and len(cells) > max_cells:
        # Prioritise the most uncertain cells (highest entropy).
        ent = np.array([-(P[i, d] * np.log(P[i, d] + 1e-9)).sum() for i, d in cells])
        cells = [cells[k] for k in np.argsort(-ent)[:max_cells]]

    for _ in range(max(0, n_rounds)):
        changed = 0
        for i, d in cells:
            actions = (0, 3) if d in P0_ONLY_DIMS else (0, 1, 2, 3)
            pseudo = pred.copy()
            # Hold the other cells at their current predictions.
            best_a, best_u = int(pred[i, d]), -1e18
            for a in actions:
                exp_u = 0.0
                for g in range(4):
                    pg = float(P[i, d, g])
                    if pg < 1e-6:
                        continue
                    gold = pseudo.copy()
                    gold[i, d] = g
                    trial = pseudo.copy()
                    trial[i, d] = a
                    exp_u += pg * _score_with_pseudo_gold(trial, gold)
                if exp_u > best_u:
                    best_u, best_a = exp_u, a
            if best_a != pred[i, d]:
                pred[i, d] = best_a
                changed += 1
        if changed == 0:
            break
    if base_clean is not None:
        pred[np.asarray(base_clean, bool)] = 0
    return pred
