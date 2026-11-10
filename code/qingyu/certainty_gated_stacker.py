#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/certainty_gated_stacker.py
======================================
Pillar 3 (Section 4.5 of Blueprint):
Vector Temperature Scaling + Class-Aligned Structured Centered Log-Odds Certainty-Gated Meta-Stacker (`C = 0.1`).

Key Mathematical Properties:
  1. Per-Model Per-Dimension Vector Temperature Calibration (`T_{m,d}`):
     Calibrates each base model's raw probability simplex on OOF/val NLL prior to fusion
     so overconfident models cannot hijack log-odds space (`ECE < 0.02`).
  2. Centered Log-Odds Transformation:
       l_tilde^{(m)}_{d,k}(x) = log p^{(m)}_{d,k}(x) - (1 / K_d) * sum_{j in A_d} log p^{(m)}_{d,j}(x)
     Zero dependence on test-batch `.mean()` or `.std()`.
  3. Normalized Negative-Entropy Dynamic Certainty Gate:
       omega^{(m)}_d(x) = 1 + (1 / log K_d) * sum_{j in A_d} p^{(m)}_{d,j}(x) * log p^{(m)}_{d,j}(x)  in [0, 1]
     Amplifies decisive domain experts (`omega -> 1`) while muting uncertain guesses (`omega -> 0`).
  4. Non-Negative Class-Aligned Structured Log-Odds Combination + Defect-Specific Physics Routing:
       z_{d,k}(x) = b_{d,k}
                  + sum_{m=1}^M [alpha_{d,m} + beta_{d,m} * omega^{(m)}_d(x)] * l_tilde^{(m)}_{d,k}(x)
                  + gamma_d * e_max_{d,k}(x)
                  + (c_d(x)^T U_d)_k
     With non-negativity constraints `alpha_{d,m}, beta_{d,m}, gamma_d >= 0` (preventing collinear negative
     weight cancellation across correlated backbones!) and sample-size-invariant Ridge regularization on `U_d`
     using ONLY the physical operators physically matched to defect `d`.
  5. Dedicated Calibrated Global All-Clean Head `Q_clean(x_i)` & Strict Single-Image Invariance:
     `predict_proba([single_img])` == `predict_proba(batch)[i]` up to floating-point machine epsilon!
"""
import os
import sys
import math
import pickle
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.config_and_paths import DIMS, FULL_DIMS, P0_ONLY_DIMS

# Domain-matched physical feature slices from DualEnginePhysicsExtractor (36D):
#   0..9:   MEOR & UI stroke mask (过锐)
#   10..18: Foreground depth-decoupled defocus sharpness S_fg (模糊)
#   19..24: Cutout boundary discontinuity A_cutout & depth grad (虚化突兀)
#   25..31: Foreground micro-texture & Laplacian-to-gradient ratio (纹理损失)
#   32..35: 8x8 blockiness, chroma spike (花屏) & log aspect/area
DIM_PHYS_INDICES: Dict[int, Tuple[int, ...]] = {
    0: (10, 11, 12, 13, 14, 15, 16, 17, 18, 34, 35),
    1: (25, 26, 27, 28, 29, 30, 31, 12, 35),
    2: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    3: (32, 33, 31, 35),
    4: (28, 29, 30, 31, 23, 34, 35),
    5: (19, 20, 21, 22, 23, 24, 18),
}


def p012_to_6x4_simplex(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p0 = np.clip(np.asarray(p0, dtype=np.float64), eps, 1.0 - eps)
    p1 = np.clip(np.asarray(p1, dtype=np.float64), 0.0, 1.0 - eps)
    p2 = np.clip(np.asarray(p2, dtype=np.float64), 0.0, 1.0 - eps)
    N = p0.shape[0]
    out = np.zeros((N, 6, 4), dtype=np.float64)

    for j, d in enumerate(FULL_DIMS):
        q3 = p0[:, d]
        q2 = p1[:, j]
        q1 = p2[:, j]
        pos_sum = q1 + q2 + q3
        over = pos_sum > (1.0 - eps)
        if np.any(over):
            scale = (1.0 - eps) / np.maximum(pos_sum[over], 1e-8)
            q1 = q1.copy()
            q2 = q2.copy()
            q3 = q3.copy()
            q1[over] *= scale
            q2[over] *= scale
            q3[over] *= scale
        q0 = np.maximum(eps, 1.0 - (q1 + q2 + q3))
        vec = np.stack([q0, q1, q2, q3], axis=-1)
        vec /= vec.sum(axis=-1, keepdims=True)
        out[:, d, :] = np.clip(vec, eps, 1.0 - eps)
        out[:, d, :] /= out[:, d, :].sum(axis=-1, keepdims=True)

    for d in P0_ONLY_DIMS:
        q3 = np.clip(p0[:, d], eps, 1.0 - eps)
        out[:, d, 0] = 1.0 - q3
        out[:, d, 3] = q3

    return out.astype(np.float32)


def simplex_6x4_to_p012(probs_6x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs_6x4 = np.asarray(probs_6x4, dtype=np.float32)
    p0 = probs_6x4[:, :, 3].copy()
    p1 = np.stack([probs_6x4[:, d, 2] for d in FULL_DIMS], axis=-1)
    p2 = np.stack([probs_6x4[:, d, 1] for d in FULL_DIMS], axis=-1)
    return p0, p1, p2


def compute_ece_6x4(probs_6x4: np.ndarray, gold_6d: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs_6x4, dtype=np.float64)
    gold = np.asarray(gold_6d, dtype=np.int64)
    eces = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for d in range(6):
        conf = probs[:, d, :].max(axis=-1)
        pred = probs[:, d, :].argmax(axis=-1)
        acc = (pred == gold[:, d]).astype(np.float64)
        ece_d = 0.0
        n = float(len(conf))
        for b0, b1 in zip(bins[:-1], bins[1:]):
            m = (conf >= b0) & (conf < b1 if b1 < 1.0 else conf <= b1)
            if np.any(m):
                ece_d += (m.sum() / n) * abs(float(acc[m].mean() - conf[m].mean()))
        eces.append(ece_d)
    return float(np.mean(eces))


def _minimize_lbfgs_or_bb(
    loss_and_grad_fn,
    x0: np.ndarray,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    maxiter: int = 250,
) -> np.ndarray:
    """Uses `scipy.optimize.minimize(L-BFGS-B)` when available, with pure-NumPy projected Barzilai-Borwein fallback."""
    try:
        from scipy.optimize import minimize
        res = minimize(
            loss_and_grad_fn,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-10},
        )
        return np.asarray(res.x, dtype=np.float64)
    except ImportError:
        x = x0.astype(np.float64).copy()
        lo = np.array([-np.inf if (bounds is None or b[0] is None) else b[0] for b in (bounds or [(-np.inf, np.inf)] * len(x0))], dtype=np.float64)
        hi = np.array([np.inf if (bounds is None or b[1] is None) else b[1] for b in (bounds or [(-np.inf, np.inf)] * len(x0))], dtype=np.float64)
        x = np.clip(x, lo, hi)
        f, g = loss_and_grad_fn(x)
        step = 0.25
        prev_x, prev_g = None, None
        for _ in range(maxiter):
            if np.linalg.norm(g) < 1e-7:
                break
            if prev_x is not None:
                dx = x - prev_x
                dg = g - prev_g
                denom = float(np.dot(dx, dg))
                if denom > 1e-12:
                    step = float(np.clip(np.dot(dx, dx) / denom, 1e-4, 2.0))
            prev_x, prev_g = x.copy(), g.copy()
            cur_s = step
            for _ls in range(12):
                cand = np.clip(x - cur_s * g, lo, hi)
                f_new, g_new = loss_and_grad_fn(cand)
                if f_new <= f - 1e-4 * float(np.dot(g, x - cand)) + 1e-12:
                    x, f, g = cand, f_new, g_new
                    break
                cur_s *= 0.5
            else:
                break
        return x


class VectorTemperatureCalibrator:
    def __init__(self, reg_temp: float = 3.0, reg_bias: float = 8.0):
        self.reg_temp = reg_temp
        self.reg_bias = reg_bias
        self.inv_temp = np.ones(6, dtype=np.float64)
        self.bias = np.zeros((6, 4), dtype=np.float64)

    @staticmethod
    def _active_indices(d: int) -> List[int]:
        return [0, 1, 2, 3] if d in FULL_DIMS else [0, 3]

    def fit(self, probs_6x4: np.ndarray, gold_6d: np.ndarray) -> "VectorTemperatureCalibrator":
        P = np.clip(np.asarray(probs_6x4, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        Y = np.asarray(gold_6d, dtype=np.int64)
        N = float(len(Y))

        for d in range(6):
            act = self._active_indices(d)
            K = len(act)
            logP = np.log(P[:, d, act])
            logP -= logP.mean(axis=1, keepdims=True)
            y_raw = Y[:, d]
            y_loc = np.array([act.index(3 if (d in P0_ONLY_DIMS and v > 0) else int(v)) for v in y_raw], dtype=np.int64)

            def obj_and_grad(theta):
                inv_t = theta[0]
                b = theta[1:]
                b_centered = b - b.mean()
                logits = inv_t * logP + b_centered[None, :]
                logits -= logits.max(axis=1, keepdims=True)
                exp_l = np.exp(logits)
                sm = exp_l / exp_l.sum(axis=1, keepdims=True)
                nll = -np.log(np.clip(sm[np.arange(int(N)), y_loc], 1e-12, 1.0)).mean()
                reg = 0.5 * (self.reg_temp / N) * ((inv_t - 1.0) ** 2) + 0.5 * (self.reg_bias / N) * np.sum(b_centered ** 2)

                diff = sm.copy()
                diff[np.arange(int(N)), y_loc] -= 1.0
                diff /= N
                g_inv_t = float(np.sum(diff * logP) + (self.reg_temp / N) * (inv_t - 1.0))
                g_b = diff.sum(axis=0) + (self.reg_bias / N) * b_centered
                g_b -= g_b.mean()
                return float(nll + reg), np.concatenate([[g_inv_t], g_b])

            x0 = np.zeros(1 + K, dtype=np.float64)
            x0[0] = 1.0
            bounds: List[Tuple[Optional[float], Optional[float]]] = [(0.50, 1.85)] + [(-0.65, 0.65)] * K
            opt_x = _minimize_lbfgs_or_bb(obj_and_grad, x0, bounds=bounds, maxiter=180)
            self.inv_temp[d] = float(opt_x[0])
            b_opt = opt_x[1:] - opt_x[1:].mean()
            self.bias[d, :] = 0.0
            for idx_k, cls_k in enumerate(act):
                self.bias[d, cls_k] = float(b_opt[idx_k])
        return self

    def transform(self, probs_6x4: np.ndarray) -> np.ndarray:
        P = np.clip(np.asarray(probs_6x4, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        out = np.zeros_like(P)
        for d in range(6):
            act = self._active_indices(d)
            logP = np.log(P[:, d, act])
            logP -= logP.mean(axis=1, keepdims=True)
            logits = self.inv_temp[d] * logP + self.bias[d, act][None, :]
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            sm = exp_l / exp_l.sum(axis=1, keepdims=True)
            for idx_k, cls_k in enumerate(act):
                out[:, d, cls_k] = sm[:, idx_k]
        return out.astype(np.float32)


def _fit_structured_certainty_stacker_dim(
    lt_nmk: np.ndarray,
    gated_nmk: np.ndarray,
    max_env_nk: np.ndarray,
    ctx_nf: np.ndarray,
    y_loc: np.ndarray,
    C: float = 0.10,
) -> Dict[str, np.ndarray]:
    """
    Fits the non-negative class-aligned structured log-odds combiner for dimension `d`:
      logits[i, k] = b[k] + sum_m alpha[m] * lt[i,m,k] + sum_m beta[m] * gated[i,m,k]
                     + gamma * max_env[i, k] + (ctx[i] @ U)[k]
    Regularization is directly scaled on mean NLL (sample-size invariant, so N=400 and N=3,573
    both enjoy strong shrinkage without collinearity overfitting!).
    """
    N, M, K = lt_nmk.shape
    F_ctx = ctx_nf.shape[1]
    inv_m = 1.0 / float(max(1, M))

    # Sample-size-invariant Ridge penalties on mean NLL:
    reg_tower = 0.015 / max(0.02, float(C))   # 0.15 when C=0.10
    reg_env = 0.025 / max(0.02, float(C))     # 0.25 when C=0.10
    reg_ctx = 0.035 / max(0.02, float(C))     # 0.35 when C=0.10 (strongly damps noisy context/phys features!)
    reg_b = 0.005

    # Parameter layout: [alpha (M), beta (M), gamma (1), b (K), U_flat (F_ctx * K)]
    idx_beta = M
    idx_gamma = 2 * M
    idx_b = 2 * M + 1
    idx_u = 2 * M + 1 + K
    total_p = idx_u + F_ctx * K

    def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
        alpha = params[:idx_beta]
        beta = params[idx_beta:idx_gamma]
        gamma = params[idx_gamma]
        b = params[idx_b:idx_u]
        b_c = b - b.mean()
        U = params[idx_u:].reshape(F_ctx, K) if F_ctx > 0 else np.zeros((0, K), dtype=np.float64)
        if F_ctx > 0:
            U_c = U - U.mean(axis=1, keepdims=True)
        else:
            U_c = U

        # [N, K]
        logits = (
            np.einsum("nmk,m->nk", lt_nmk, alpha)
            + np.einsum("nmk,m->nk", gated_nmk, beta)
            + gamma * max_env_nk
            + b_c[None, :]
        )
        if F_ctx > 0:
            logits = logits + ctx_nf @ U_c

        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        p_true = np.clip(probs[np.arange(N), y_loc], 1e-12, 1.0)
        nll = -float(np.mean(np.log(p_true)))

        d_alpha = alpha - inv_m
        reg = (
            0.5 * reg_tower * float(np.sum(d_alpha ** 2) + np.sum(beta ** 2))
            + 0.5 * reg_env * float(gamma * gamma)
            + 0.5 * reg_b * float(np.sum(b_c ** 2))
            + (0.5 * reg_ctx * float(np.sum(U_c ** 2)) if F_ctx > 0 else 0.0)
        )

        diff = probs.copy()
        diff[np.arange(N), y_loc] -= 1.0
        diff /= float(N)

        g_alpha = np.einsum("nmk,nk->m", lt_nmk, diff) + reg_tower * d_alpha
        g_beta = np.einsum("nmk,nk->m", gated_nmk, diff) + reg_tower * beta
        g_gamma = np.array([float(np.sum(max_env_nk * diff)) + reg_env * gamma], dtype=np.float64)
        g_b = diff.sum(axis=0) + reg_b * b_c
        g_b -= g_b.mean()

        if F_ctx > 0:
            g_U = ctx_nf.T @ diff + reg_ctx * U_c
            g_U -= g_U.mean(axis=1, keepdims=True)
            grad = np.concatenate([g_alpha, g_beta, g_gamma, g_b, g_U.ravel()])
        else:
            grad = np.concatenate([g_alpha, g_beta, g_gamma, g_b])
        return float(nll + reg), grad

    x0 = np.zeros(total_p, dtype=np.float64)
    x0[:M] = inv_m  # Starts at calibrated equal-weight geometric pool!
    bounds: List[Tuple[Optional[float], Optional[float]]] = (
        [(0.0, 2.0)] * M          # alpha_m >= 0 (no negative cancellation!)
        + [(0.0, 2.0)] * M        # beta_m >= 0  (certainty gate boost >= 0!)
        + [(0.0, 0.55)]           # gamma >= 0   (max envelope weight)
        + [(-1.5, 1.5)] * K       # class bias
        + [(-0.8, 0.8)] * (F_ctx * K)
    )
    opt_p = _minimize_lbfgs_or_bb(loss_and_grad, x0, bounds=bounds, maxiter=240)
    U_opt = opt_p[idx_u:].reshape(F_ctx, K) if F_ctx > 0 else np.zeros((0, K), dtype=np.float64)
    if F_ctx > 0:
        U_opt = U_opt - U_opt.mean(axis=1, keepdims=True)
    b_opt = opt_p[idx_b:idx_u] - opt_p[idx_b:idx_u].mean()
    return {
        "alpha": opt_p[:idx_beta].copy(),
        "beta": opt_p[idx_beta:idx_gamma].copy(),
        "gamma": float(opt_p[idx_gamma]),
        "b": b_opt,
        "U": U_opt,
    }


def _fit_sample_invariant_binary_logreg(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.int64)
    N, F_dim = X64.shape
    reg_w = 0.02 / max(0.02, float(C))

    def loss_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
        W = params[: F_dim * 2].reshape(F_dim, 2)
        W_c = W - W.mean(axis=1, keepdims=True)
        b = params[F_dim * 2 :]
        b_c = b - b.mean()
        logits = X64 @ W_c + b_c[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        p_true = np.clip(probs[np.arange(N), y64], 1e-12, 1.0)
        nll = -float(np.mean(np.log(p_true)))
        reg = 0.5 * reg_w * float(np.sum(W_c ** 2))

        diff = probs.copy()
        diff[np.arange(N), y64] -= 1.0
        diff /= float(N)
        gW = X64.T @ diff + reg_w * W_c
        gW -= gW.mean(axis=1, keepdims=True)
        gb = diff.sum(axis=0)
        gb -= gb.mean()
        return float(nll + reg), np.concatenate([gW.ravel(), gb])

    init = np.zeros(F_dim * 2 + 2, dtype=np.float64)
    counts = np.bincount(y64, minlength=2).astype(np.float64) + 0.5
    lp = np.log(counts / counts.sum())
    init[F_dim * 2 :] = lp - lp.mean()
    opt = _minimize_lbfgs_or_bb(loss_and_grad, init, bounds=None, maxiter=200)
    W_opt = opt[: F_dim * 2].reshape(F_dim, 2)
    W_opt -= W_opt.mean(axis=1, keepdims=True)
    b_opt = opt[F_dim * 2 :] - opt[F_dim * 2 :].mean()
    return W_opt, b_opt


class CenteredLogOddsCertaintyGatedStacker:
    def __init__(self, C: float = 0.1, clean_C: float = 0.15):
        self.C = float(C)
        self.clean_C = float(clean_C)
        self.num_models: int = 0
        self.model_names: List[str] = []
        self.base_calibrators: List[VectorTemperatureCalibrator] = []
        self.phys_mean: Optional[np.ndarray] = None
        self.phys_std: Optional[np.ndarray] = None
        self.ctx_means: Dict[int, np.ndarray] = {}
        self.ctx_stds: Dict[int, np.ndarray] = {}
        self.dim_params: Dict[int, Dict[str, np.ndarray]] = {}
        self.clean_feat_mean: Optional[np.ndarray] = None
        self.clean_feat_std: Optional[np.ndarray] = None
        self.W_clean: Optional[np.ndarray] = None
        self.b_clean: Optional[np.ndarray] = None
        self.post_calibrator = VectorTemperatureCalibrator(reg_temp=3.0, reg_bias=8.0)

    @staticmethod
    def _active_indices(d: int) -> List[int]:
        return [0, 1, 2, 3] if d in FULL_DIMS else [0, 3]

    @staticmethod
    def _centered_log_odds_and_certainty(cal_models_6x4: Sequence[np.ndarray], d: int) -> Tuple[np.ndarray, np.ndarray]:
        act = CenteredLogOddsCertaintyGatedStacker._active_indices(d)
        K_d = len(act)
        log_k = math.log(float(K_d))
        l_list = []
        w_list = []
        for pm in cal_models_6x4:
            p_act = np.clip(np.asarray(pm[:, d, act], dtype=np.float64), 1e-6, 1.0 - 1e-6)
            p_act /= p_act.sum(axis=-1, keepdims=True)
            lp = np.log(p_act)
            lt = lp - lp.mean(axis=-1, keepdims=True)
            ent = -np.sum(p_act * lp, axis=-1, keepdims=True)
            omega = np.clip(1.0 - ent / log_k, 0.0, 1.0)
            l_list.append(lt)
            w_list.append(omega)
        return np.stack(l_list, axis=1), np.stack(w_list, axis=1)

    def _build_dim_components(
        self,
        cal_models_6x4: Sequence[np.ndarray],
        d: int,
        phys_norm: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lt, om = self._centered_log_odds_and_certainty(cal_models_6x4, d)  # [N, M, K_d], [N, M, 1]
        gated = om * lt
        max_env = lt.max(axis=1)
        max_env = max_env - max_env.mean(axis=-1, keepdims=True)

        # Compact 6D cross-defect co-occurrence context (mean log-odds of any defect across the 6 dims)
        ctx_cols = []
        for d_other in range(6):
            pc_m = np.stack([np.clip(pm[:, d_other, 0], 1e-5, 1.0 - 1e-5) for pm in cal_models_6x4], axis=1)
            ctx_cols.append(np.log((1.0 - pc_m) / pc_m).mean(axis=1))
        ctx_parts = [np.stack(ctx_cols, axis=1)]

        if phys_norm is not None and phys_norm.shape[1] > 0:
            valid_idx = [ix for ix in DIM_PHYS_INDICES.get(d, ()) if ix < phys_norm.shape[1]]
            if valid_idx:
                ctx_parts.append(phys_norm[:, valid_idx])

        ctx_raw = np.concatenate(ctx_parts, axis=1).astype(np.float64)
        return lt, gated, max_env, ctx_raw

    def _build_global_clean_features(
        self,
        cal_models_6x4: Sequence[np.ndarray],
        stacked_6x4: np.ndarray,
        phys_norm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        p_clean_d = np.clip(stacked_6x4[:, :, 0], 1e-6, 1.0 - 1e-6)
        log_joint_clean = np.sum(np.log(p_clean_d), axis=1, keepdims=True)
        min_clean_d = np.min(np.log(p_clean_d / (1.0 - p_clean_d)), axis=1, keepdims=True)
        max_p0_logit = np.max(
            np.log(np.clip(stacked_6x4[:, :, 3], 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(stacked_6x4[:, :, 3], 1e-6, 1.0 - 1e-6))),
            axis=1,
            keepdims=True,
        )
        per_dim_clean_logits = np.log(p_clean_d / (1.0 - p_clean_d))

        per_model_joint = []
        for pm in cal_models_6x4:
            pmc = np.clip(pm[:, :, 0], 1e-6, 1.0 - 1e-6)
            per_model_joint.append(np.sum(np.log(pmc), axis=1))
        pm_mat = np.stack(per_model_joint, axis=1)
        pm_mean = pm_mat.mean(axis=1, keepdims=True)
        pm_min = pm_mat.min(axis=1, keepdims=True)

        parts = [log_joint_clean, min_clean_d, max_p0_logit, per_dim_clean_logits, pm_mean, pm_min]
        if phys_norm is not None and phys_norm.shape[1] >= 4:
            # Summary physical cleanliness indicators only (MEOR_p99, S_fg_p90, A_cutout_p99, blockiness)
            key_ix = [ix for ix in (3, 13, 21, 32) if ix < phys_norm.shape[1]]
            if key_ix:
                parts.append(phys_norm[:, key_ix])
        return np.concatenate(parts, axis=1).astype(np.float64)

    def _standardize_phys(self, phys: Optional[np.ndarray], fit: bool = False) -> Optional[np.ndarray]:
        if phys is None:
            return None
        P = np.nan_to_num(np.asarray(phys, dtype=np.float64), nan=0.0, posinf=10.0, neginf=-10.0)
        if fit:
            self.phys_mean = P.mean(axis=0, keepdims=True)
            self.phys_std = P.std(axis=0, keepdims=True) + 1e-5
        if self.phys_mean is None or self.phys_std is None:
            return np.clip(P, -5.0, 5.0)
        return np.clip((P - self.phys_mean) / self.phys_std, -5.0, 5.0)

    @staticmethod
    def _eval_dim_logits(
        lt_nmk: np.ndarray,
        gated_nmk: np.ndarray,
        max_env_nk: np.ndarray,
        ctx_norm: np.ndarray,
        dp: Dict[str, np.ndarray],
    ) -> np.ndarray:
        logits = (
            np.einsum("nmk,m->nk", lt_nmk, dp["alpha"])
            + np.einsum("nmk,m->nk", gated_nmk, dp["beta"])
            + float(dp["gamma"]) * max_env_nk
            + dp["b"][None, :]
        )
        if ctx_norm.shape[1] > 0 and dp["U"].shape[0] == ctx_norm.shape[1]:
            logits = logits + ctx_norm @ dp["U"]
        return logits

    def fit(
        self,
        models_6x4: Sequence[np.ndarray],
        gold_6d: np.ndarray,
        phys_mat: Optional[np.ndarray] = None,
        model_names: Optional[Sequence[str]] = None,
    ) -> "CenteredLogOddsCertaintyGatedStacker":
        self.num_models = len(models_6x4)
        self.model_names = list(model_names) if model_names is not None else [f"m{i}" for i in range(self.num_models)]
        gold_6d = np.asarray(gold_6d, dtype=np.int64)

        self.base_calibrators = []
        cal_models = []
        for pm in models_6x4:
            vc = VectorTemperatureCalibrator().fit(pm, gold_6d)
            self.base_calibrators.append(vc)
            cal_models.append(vc.transform(pm))

        phys_norm = self._standardize_phys(phys_mat, fit=True)

        raw_stacked = np.zeros((len(gold_6d), 6, 4), dtype=np.float32)
        for d in range(6):
            act = self._active_indices(d)
            lt, gated, max_env, ctx_raw = self._build_dim_components(cal_models, d, phys_norm)
            mu = ctx_raw.mean(axis=0, keepdims=True)
            sd = ctx_raw.std(axis=0, keepdims=True) + 1e-5
            self.ctx_means[d] = mu
            self.ctx_stds[d] = sd
            ctx_n = np.clip((ctx_raw - mu) / sd, -5.0, 5.0)

            y_loc = np.array([act.index(3 if (d in P0_ONLY_DIMS and v > 0) else int(v)) for v in gold_6d[:, d]], dtype=np.int64)
            dp = _fit_structured_certainty_stacker_dim(lt, gated, max_env, ctx_n, y_loc, C=self.C)
            self.dim_params[d] = dp

            logits = self._eval_dim_logits(lt, gated, max_env, ctx_n, dp)
            logits -= logits.max(axis=1, keepdims=True)
            sm = np.exp(logits)
            sm /= sm.sum(axis=1, keepdims=True)
            for idx_k, cls_k in enumerate(act):
                raw_stacked[:, d, cls_k] = sm[:, idx_k].astype(np.float32)

        self.post_calibrator.fit(raw_stacked, gold_6d)
        cal_stacked = self.post_calibrator.transform(raw_stacked)

        y_clean = (gold_6d == 0).all(axis=1).astype(np.int64)
        Xc = self._build_global_clean_features(cal_models, cal_stacked, phys_norm)
        self.clean_feat_mean = Xc.mean(axis=0, keepdims=True)
        self.clean_feat_std = Xc.std(axis=0, keepdims=True) + 1e-5
        Xcn = np.clip((Xc - self.clean_feat_mean) / self.clean_feat_std, -5.0, 5.0)
        self.W_clean, self.b_clean = _fit_sample_invariant_binary_logreg(Xcn, y_clean, C=self.clean_C)
        return self

    def predict_proba(
        self,
        models_6x4: Sequence[np.ndarray],
        phys_mat: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert len(models_6x4) == self.num_models, f"Expected {self.num_models} models, got {len(models_6x4)}"
        cal_models = [self.base_calibrators[m].transform(models_6x4[m]) for m in range(self.num_models)]
        phys_norm = self._standardize_phys(phys_mat, fit=False)

        N = cal_models[0].shape[0]
        raw_stacked = np.zeros((N, 6, 4), dtype=np.float32)
        for d in range(6):
            act = self._active_indices(d)
            lt, gated, max_env, ctx_raw = self._build_dim_components(cal_models, d, phys_norm)
            ctx_n = np.clip((ctx_raw - self.ctx_means[d]) / self.ctx_stds[d], -5.0, 5.0)
            logits = self._eval_dim_logits(lt, gated, max_env, ctx_n, self.dim_params[d])
            logits -= logits.max(axis=1, keepdims=True)
            sm = np.exp(logits)
            sm /= sm.sum(axis=1, keepdims=True)
            for idx_k, cls_k in enumerate(act):
                raw_stacked[:, d, cls_k] = sm[:, idx_k].astype(np.float32)

        cal_stacked = self.post_calibrator.transform(raw_stacked)
        Xc = self._build_global_clean_features(cal_models, cal_stacked, phys_norm)
        Xcn = np.clip((Xc - self.clean_feat_mean) / self.clean_feat_std, -5.0, 5.0)
        c_logits = Xcn @ self.W_clean + self.b_clean[None, :]
        c_logits -= c_logits.max(axis=1, keepdims=True)
        c_sm = np.exp(c_logits)
        c_sm /= c_sm.sum(axis=1, keepdims=True)
        q_clean = c_sm[:, 1].astype(np.float32)
        return cal_stacked, q_clean

    def fit_oof_and_full(
        self,
        models_6x4: Sequence[np.ndarray],
        gold_6d: np.ndarray,
        fold_ids: np.ndarray,
        phys_mat: Optional[np.ndarray] = None,
        model_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = len(gold_6d)
        oof_probs = np.zeros((N, 6, 4), dtype=np.float32)
        oof_q_clean = np.zeros(N, dtype=np.float32)
        unique_folds = sorted(set(int(f) for f in fold_ids))

        for f in unique_folds:
            val_mask = fold_ids == f
            tr_mask = ~val_mask
            sub_stacker = CenteredLogOddsCertaintyGatedStacker(C=self.C, clean_C=self.clean_C)
            tr_models = [pm[tr_mask] for pm in models_6x4]
            va_models = [pm[val_mask] for pm in models_6x4]
            tr_phys = phys_mat[tr_mask] if phys_mat is not None else None
            va_phys = phys_mat[val_mask] if phys_mat is not None else None
            sub_stacker.fit(tr_models, gold_6d[tr_mask], phys_mat=tr_phys, model_names=model_names)
            vp, vq = sub_stacker.predict_proba(va_models, phys_mat=va_phys)
            oof_probs[val_mask] = vp
            oof_q_clean[val_mask] = vq

        self.fit(models_6x4, gold_6d, phys_mat=phys_mat, model_names=model_names)
        return oof_probs, oof_q_clean

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "CenteredLogOddsCertaintyGatedStacker":
        with open(path, "rb") as f:
            return pickle.load(f)
