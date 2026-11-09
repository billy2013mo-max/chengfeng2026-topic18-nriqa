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


def fit_extremal_beta_calibration_14ch(
    oof_probs_list: list,
    y_oof_14ch: np.ndarray,
    fold_ids: np.ndarray = None,
) -> list:
    """
    Strictly monotonic 2-parameter Beta Calibration per model family and channel (PDF Sec 5.3):
        z = a * log(p + eps) - b * log(1 - p + eps) + d, with a > 0, b > 0.
    Preserves extremal right-tail P0 spikes without quantile clipping.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    M = len(oof_probs_list)
    N, C = y_oof_14ch.shape
    params_all = [[None for _ in range(C)] for _ in range(M)]
    for m in range(M):
        P_m = np.clip(oof_probs_list[m].astype(np.float64), 1e-6, 1.0 - 1e-6)
        log_p = np.log(P_m)
        log_1mp = np.log(1.0 - P_m)
        for c in range(C):
            y_c = y_oof_14ch[:, c].astype(np.float64)
            lp = log_p[:, c]
            l1m = log_1mp[:, c]

            def nll_loss(theta):
                a, b, d = theta
                z = a * lp - b * l1m + d
                p_hat = expit(z)
                w = np.where(y_c == 0, 1.25, 1.0)
                bce = -(y_c * np.log(p_hat + 1e-12) + (1.0 - y_c) * np.log(1.0 - p_hat + 1e-12))
                reg = 0.05 * ((a - 1.0) ** 2 + (b - 1.0) ** 2 + 0.25 * (d ** 2))
                return float(np.mean(w * bce) + reg)

            res = minimize(
                nll_loss,
                x0=np.array([1.0, 1.0, 0.0]),
                bounds=[(0.40, 2.50), (0.40, 2.50), (-2.5, 2.5)],
                method="L-BFGS-B",
            )
            params_all[m][c] = res.x
    return params_all


def apply_extremal_beta_calibration_14ch(
    probs_14ch: np.ndarray,
    family_params: list,
    anchor_probs_14ch: np.ndarray = None,
) -> np.ndarray:
    """Applies fitted Beta calibration parameters and optionally aligns first/second moments to anchor."""
    from scipy.special import expit, logit

    P = np.clip(probs_14ch.astype(np.float64), 1e-6, 1.0 - 1e-6)
    lp = np.log(P)
    l1m = np.log(1.0 - P)
    Z = np.zeros_like(P)
    for c in range(P.shape[1]):
        a, b, d = family_params[c]
        Z[:, c] = a * lp[:, c] - b * l1m[:, c] + d
    if anchor_probs_14ch is not None:
        l_base = logit(np.clip(anchor_probs_14ch.astype(np.float64), 1e-5, 1.0 - 1e-5))
        mu_b = l_base.mean(axis=0, keepdims=True)
        sd_b = l_base.std(axis=0, keepdims=True)
        mu_z = Z.mean(axis=0, keepdims=True)
        sd_z = Z.std(axis=0, keepdims=True) + 1e-6
        Z = mu_b + (sd_b / sd_z) * (Z - mu_z)
    return expit(Z)


def apply_true_oof_quali_rag_v2(
    oof_logits: np.ndarray,
    test_logits: np.ndarray,
    oof_scene_emb: np.ndarray,
    test_scene_emb: np.ndarray,
    y_oof_6d: np.ndarray,
    fold_ids: np.ndarray,
    k_ref: int = 12,
    k_all: int = 20,
    tau_rag: float = 0.08,
    beta_rag: float = 0.18,
    gamma_z: float = 0.04,
    eta_prior: float = 0.12,
):
    """
    QualiRAG 2.0 Four-Dimensional Manifold Relational Calibration (PDF Sec 5.2.3):
      f1 = L_ens(x, c)
      f2 = L_ens(x, c) - mu_clean_knn(x, c) + mu_clean_global(c)
      f3 = (L_ens(x, c) - mu_clean_knn(x, c)) / (sigma_clean_knn(x, c) + 0.15)
           (variance-adaptive Z-score significance: dampens subtraction when local clean
            variance is high, sharpens when local clean neighborhood is tight)
      f4 = sum_{k in KNN_all(x)} w_k * (Y_{k,c} - Y_bar_c)
           (same-scene Top-20 local defect prior density residual)
    """
    from utils import labels_to_14ch

    oof_emb = oof_scene_emb / (np.linalg.norm(oof_scene_emb, axis=1, keepdims=True) + 1e-8)
    test_emb = test_scene_emb / (np.linalg.norm(test_scene_emb, axis=1, keepdims=True) + 1e-8)

    is_clean = (y_oof_6d.sum(axis=1) == 0)
    clean_idx_all = np.where(is_clean)[0]
    if len(clean_idx_all) < k_ref:
        return oof_logits, test_logits

    y_oof_14 = labels_to_14ch(y_oof_6d).astype(np.float64)
    y_bar = y_oof_14.mean(axis=0, keepdims=True)
    global_clean_mean = oof_logits[clean_idx_all].mean(axis=0, keepdims=True)
    global_clean_std = oof_logits[clean_idx_all].std(axis=0, keepdims=True) + 0.15

    def _compute_manifold_shift(query_logits, query_emb, clean_pool_idx, all_pool_idx):
        # 1. Clean KNN mean & local variance (Top-k_ref)
        sim_c = query_emb @ oof_emb[clean_pool_idx].T
        k_c = min(k_ref, len(clean_pool_idx) - 1)
        top_c = np.argpartition(-sim_c, kth=k_c, axis=1)[:, :k_c]
        row_i = np.arange(sim_c.shape[0])[:, None]
        top_sim_c = sim_c[row_i, top_c]
        w_c = np.exp((top_sim_c - top_sim_c.max(axis=1, keepdims=True)) / tau_rag)
        w_c /= w_c.sum(axis=1, keepdims=True)

        ref_l = oof_logits[clean_pool_idx][top_c]  # [N, k_c, 14]
        mu_clean = np.einsum("nk,nkc->nc", w_c, ref_l)
        var_clean = np.einsum("nk,nkc->nc", w_c, (ref_l - mu_clean[:, None, :]) ** 2)
        sd_clean = np.sqrt(np.maximum(var_clean, 0.0)) + 0.15

        # Variance-adaptive confidence damping: if local clean variance is low relative to global,
        # trust the scene clean baseline more; if high, shrink subtraction.
        var_trust = np.clip(global_clean_std / sd_clean, 0.55, 1.35)
        f2_shift = -beta_rag * var_trust * (mu_clean - global_clean_mean)
        z_sig = np.clip((query_logits - mu_clean) / sd_clean - (query_logits - global_clean_mean) / global_clean_std, -2.5, 2.5)
        f3_shift = gamma_z * z_sig

        # 2. All-OOF KNN local defect prior density (Top-k_all)
        sim_a = query_emb @ oof_emb[all_pool_idx].T
        k_a = min(k_all, len(all_pool_idx) - 1)
        top_a = np.argpartition(-sim_a, kth=k_a, axis=1)[:, :k_a]
        top_sim_a = sim_a[row_i, top_a]
        w_a = np.exp((top_sim_a - top_sim_a.max(axis=1, keepdims=True)) / tau_rag)
        w_a /= w_a.sum(axis=1, keepdims=True)
        # Gate by max neighbor similarity so unrelated scenes are not shifted
        sim_gate = np.clip((top_sim_a.max(axis=1, keepdims=True) - 0.68) / 0.20, 0.0, 1.0)
        local_prior = np.einsum("nk,nkc->nc", w_a, y_oof_14[all_pool_idx][top_a])
        f4_shift = eta_prior * sim_gate * (local_prior - y_bar)

        return query_logits + f2_shift + f3_shift + f4_shift

    oof_rag = oof_logits.copy()
    for f in np.unique(fold_ids):
        val_mask = (fold_ids == f)
        ref_clean = np.where(is_clean & (fold_ids != f))[0]
        ref_all = np.where(fold_ids != f)[0]
        if len(ref_clean) < k_ref:
            ref_clean = clean_idx_all
        if len(ref_all) < k_all:
            ref_all = np.arange(len(oof_logits))
        oof_rag[val_mask] = _compute_manifold_shift(
            oof_logits[val_mask], oof_emb[val_mask], ref_clean, ref_all
        )

    all_idx = np.arange(len(oof_logits))
    test_rag = _compute_manifold_shift(test_logits, test_emb, clean_idx_all, all_idx)
    return oof_rag, test_rag


def eum_bca_joint_decode(
    probs_14ch: np.ndarray,
    t0: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    tau_any: np.ndarray,
    kappa_p1p2: np.ndarray = None,
    base_clean: np.ndarray = None,
    n_rounds: int = 3,
    max_cells: int = None,
) -> np.ndarray:
    """
    Fast O(1)-update Plug-in Expected-Utility Maximization Block Coordinate Ascent (EUM-BCA, PDF Sec 2.2):
    Directly maximizes the posterior expected competition score:
        E_{Y ~ P_cal}[25 * S_pos(Y_hat, Y) + 20 * S_p0(Y_hat, Y)]
    using sufficient statistics (expected TP and predicted positive count per (dim, grade))
    updated in O(1) time per coordinate move.
    """
    P_raw = probs_14ch_to_mutually_exclusive_6x4(probs_14ch)
    pred = decode_competition_ratio(
        probs_14ch, t0, t1, t2, tau_any, kappa_p1p2, base_clean=base_clean, roll_p2=False
    )
    N = pred.shape[0]
    bc = np.zeros(N, dtype=bool) if base_clean is None else np.asarray(base_clean, bool)
    non_clean_rows = np.where(~bc)[0]
    if len(non_clean_rows) == 0 or n_rounds <= 0:
        return pred

    # Calibrate cell posteriors around the decoded operating point so E[F1] is consistent
    # with the thresholded base-rate decision boundary.
    P = P_raw.copy()
    for j, d in enumerate(FULL_DIMS):
        # Softly align posterior class priors with the threshold ratios
        scale = np.array([
            1.0,
            0.28 / max(0.05, float(t2[j])),
            0.30 / max(0.05, float(t1[j])),
            0.38 / max(0.05, float(t0[d])),
        ], dtype=np.float64)
        P[:, d, :] *= scale[None, :]
        P[:, d, :] /= P[:, d, :].sum(axis=1, keepdims=True)
    for d in P0_ONLY_DIMS:
        scale0 = 0.38 / max(0.05, float(t0[d]))
        P[:, d, 3] *= scale0
        P[:, d, :] /= P[:, d, :].sum(axis=1, keepdims=True)

    # Sufficient statistics per (d, g):
    # G_exp[d, g] = sum_i P[i, d, g] (expected GT positives)
    # M_cnt[d, g] = sum_i I(pred[i, d] == g) (predicted positives)
    # T_exp[d, g] = sum_{i: pred[i, d] == g} P[i, d, g] (expected TPs)
    G_exp = P.sum(axis=0)  # [6, 4]
    M_cnt = np.zeros((6, 4), dtype=np.float64)
    T_exp = np.zeros((6, 4), dtype=np.float64)
    for d in range(6):
        for g in (1, 2, 3):
            mask = (pred[:, d] == g)
            M_cnt[d, g] = float(mask.sum())
            T_exp[d, g] = float(P[mask, d, g].sum())

    def _dim_utility(d: int, m_vec: np.ndarray, t_vec: np.ndarray) -> float:
        g_vec = G_exp[d]
        if d in FULL_DIMS:
            f1s = []
            for g in (1, 2, 3):
                denom = m_vec[g] + g_vec[g]
                f1 = (2.0 * t_vec[g] / denom) if denom > 1e-6 else 0.0
                f1s.append(f1)
            s_pos_d = (25.0 / 6.0) * (f1s[0] + f1s[1] + f1s[2]) / 3.0
            f1_p0 = f1s[2]
        else:
            denom = m_vec[3] + g_vec[3]
            f1_p0 = (2.0 * t_vec[3] / denom) if denom > 1e-6 else 0.0
            s_pos_d = (25.0 / 6.0) * f1_p0
        r_p0 = (t_vec[3] / g_vec[3]) if g_vec[3] > 1e-6 else 1.0
        s_p0_d = (20.0 / 6.0) * (0.6 * r_p0 + 0.4 * f1_p0)
        return s_pos_d + s_p0_d

    # Only allow coordinate transitions on boundary cells (entropy > 0.45) and within
    # high-confidence support of the threshold decoder so EUM-BCA refines ambiguous boundaries.
    p0_14, p1_14, p2_14 = ch14_to_p012(probs_14ch)
    for _ in range(n_rounds):
        changed = 0
        for d in FULL_DIMS:
            j = FULL_DIMS.index(d)
            cur_u = _dim_utility(d, M_cnt[d], T_exp[d])
            for i in non_clean_rows:
                a_old = int(pred[i, d])
                p_any = p0_14[i, d] + p1_14[i, j] + p2_14[i, j]
                # Only refine cells near the P1<->P2 or P0<->P1 boundary
                if a_old == 0 and p_any < tau_any[d] * 0.92:
                    continue
                best_a = a_old
                best_u = cur_u
                for a_new in (0, 1, 2, 3):
                    if a_new == a_old:
                        continue
                    # Enforce minimum evidence guardrails
                    if a_new == 3 and p0_14[i, d] < t0[d] * 0.90:
                        continue
                    if a_new == 2 and (p1_14[i, j] + 0.45 * p0_14[i, d]) < t1[j] * 0.88:
                        continue
                    if a_new == 1 and p_any < tau_any[d] * 0.92:
                        continue
                    if a_old == 3 and p0_14[i, d] >= t0[d] * 1.06:
                        continue
                    m_try = M_cnt[d].copy()
                    t_try = T_exp[d].copy()
                    if a_old > 0:
                        m_try[a_old] -= 1.0
                        t_try[a_old] -= P[i, d, a_old]
                    if a_new > 0:
                        m_try[a_new] += 1.0
                        t_try[a_new] += P[i, d, a_new]
                    u_try = _dim_utility(d, m_try, t_try)
                    if u_try > best_u + 1e-5:
                        best_u = u_try
                        best_a = a_new
                if best_a != a_old:
                    if a_old > 0:
                        M_cnt[d, a_old] -= 1.0
                        T_exp[d, a_old] -= P[i, d, a_old]
                    if best_a > 0:
                        M_cnt[d, best_a] += 1.0
                        T_exp[d, best_a] += P[i, d, best_a]
                    pred[i, d] = best_a
                    cur_u = best_u
                    changed += 1
        if changed == 0:
            break

    pred[bc] = 0
    return pred
