#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/certainty_gated_stacker.py
======================================
Pillar 3 (Section 4.5 of Blueprint):
Vector Temperature Scaling + Centered Log-Odds Certainty-Gated Meta-Stacker (`C = 0.1`).

Key Mathematical Properties:
  1. Per-Model Per-Dimension Vector Temperature Calibration (`T_{m,d}`):
     Calibrates each base model's raw probability simplex on OOF/val NLL prior to fusion
     so overconfident models cannot hijack log-odds space (`ECE < 0.02`).
  2. Centered Log-Odds Transformation:
       l_tilde^{(m)}_{d,k}(x) = log p^{(m)}_{d,k}(x) - (1 / K_d) * sum_{j in A_d} log p^{(m)}_{d,j}(x)
     Replaces `align_odds_1d` completely — zero dependence on test-batch `.mean()`!
  3. Normalized Negative-Entropy Dynamic Certainty Gate:
       omega^{(m)}_d(x) = 1 + (1 / log K_d) * sum_{j in A_d} p^{(m)}_{d,j}(x) * log p^{(m)}_{d,j}(x)  in [0, 1]
     Amplifies decisive domain experts (`omega -> 1`) while muting uncertain guesses (`omega -> 0`).
  4. Cross-Model Max Log-Odds Envelope (`max_m l_tilde^{(m)}_{d,k}(x)`):
     Preserves sharp local detections (e.g. 1–2px MEOR halo or tiny 6-finger hand ROI)
     that linear probability averaging would dilute below threshold.
  5. Strong L2-Regularized Multinomial Logistic Regression (`C = 0.1`, i.e. L2 penalty `10.0`):
     Solved via deterministic L-BFGS-B (with built-in pure-NumPy Barzilai-Borwein fallback)
     with nested K-Fold OOF predictions, plus a dedicated calibrated Global All-Clean Probability Head `Q_clean(x_i)`.
  6. Strict Single-Image Invariance:
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
    bounds: Optional[List[Tuple[float, float]]] = None,
    maxiter: int = 250,
) -> np.ndarray:
    """Uses `scipy.optimize.minimize(L-BFGS-B)` when available, with pure-NumPy Barzilai-Borwein fallback."""
    try:
        from scipy.optimize import minimize
        res = minimize(
            loss_and_grad_fn, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-10}
        )
        return np.asarray(res.x, dtype=np.float64)
    except ImportError:
        x = x0.astype(np.float64).copy()
        if bounds is not None:
            lo = np.array([b[0] for b in bounds], dtype=np.float64)
            hi = np.array([b[1] for b in bounds], dtype=np.float64)
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
                cand = x - cur_s * g
                if bounds is not None:
                    cand = np.clip(cand, lo, hi)
                f_new, g_new = loss_and_grad_fn(cand)
                if f_new <= f - 1e-4 * float(np.dot(g, x - cand)) + 1e-12:
                    x, f, g = cand, f_new, g_new
                    break
                cur_s *= 0.5
            else:
                break
        return x


class VectorTemperatureCalibrator:
    def __init__(self, reg_temp: float = 2.0, reg_bias: float = 5.0):
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
            bounds = [(0.45, 2.10)] + [(-0.80, 0.80)] * K
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


def _fit_l2_multinomial_logreg(
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    C: float = 0.1,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X64 = np.asarray(X, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.int64)
    N, F_dim = X64.shape
    K = int(num_classes)
    sw = np.ones(N, dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    sw_sum = max(1e-8, float(sw.sum()))
    lam = 1.0 / max(1e-6, float(C))

    def loss_and_grad(params):
        W = params[: F_dim * K].reshape(F_dim, K)
        b = params[F_dim * K :]
        b_c = b - b.mean()
        logits = X64 @ W + b_c[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        p_true = np.clip(probs[np.arange(N), y64], 1e-12, 1.0)
        nll = -np.sum(sw * np.log(p_true)) / sw_sum
        reg = 0.5 * (lam / sw_sum) * np.sum(W * W)

        diff = probs.copy()
        diff[np.arange(N), y64] -= 1.0
        diff *= (sw / sw_sum)[:, None]

        gW = X64.T @ diff + (lam / sw_sum) * W
        gb = diff.sum(axis=0)
        gb -= gb.mean()
        return float(nll + reg), np.concatenate([gW.ravel(), gb])

    init = np.zeros(F_dim * K + K, dtype=np.float64)
    counts = np.bincount(y64, minlength=K).astype(np.float64) + 0.5
    log_prior = np.log(counts / counts.sum())
    log_prior -= log_prior.mean()
    init[F_dim * K :] = log_prior

    opt_x = _minimize_lbfgs_or_bb(loss_and_grad, init, bounds=None, maxiter=250)
    W_opt = opt_x[: F_dim * K].reshape(F_dim, K)
    b_opt = opt_x[F_dim * K :] - opt_x[F_dim * K :].mean()
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
        self.feat_means: Dict[int, np.ndarray] = {}
        self.feat_stds: Dict[int, np.ndarray] = {}
        self.W_dim: Dict[int, np.ndarray] = {}
        self.b_dim: Dict[int, np.ndarray] = {}
        self.clean_feat_mean: Optional[np.ndarray] = None
        self.clean_feat_std: Optional[np.ndarray] = None
        self.W_clean: Optional[np.ndarray] = None
        self.b_clean: Optional[np.ndarray] = None
        self.post_calibrator = VectorTemperatureCalibrator(reg_temp=1.5, reg_bias=4.0)

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

    def _build_dim_features(
        self,
        cal_models_6x4: Sequence[np.ndarray],
        d: int,
        phys_norm: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        lt, om = self._centered_log_odds_and_certainty(cal_models_6x4, d)
        N, M, K_d = lt.shape

        gated = (om * lt).reshape(N, M * K_d)
        raw_lt = lt.reshape(N, M * K_d)
        max_env = lt.max(axis=1)
        om_sum = np.maximum(1e-5, om.sum(axis=1))
        cert_mean = (om * lt).sum(axis=1) / om_sum

        ctx_list = []
        for d_other in range(6):
            p0_m = np.stack([np.clip(pm[:, d_other, 3], 1e-5, 1.0 - 1e-5) for pm in cal_models_6x4], axis=1)
            pc_m = np.stack([np.clip(pm[:, d_other, 0], 1e-5, 1.0 - 1e-5) for pm in cal_models_6x4], axis=1)
            ctx_list.append(np.log(p0_m / (1.0 - p0_m)).mean(axis=1))
            ctx_list.append(np.log((1.0 - pc_m) / pc_m).mean(axis=1))
        ctx_mat = np.stack(ctx_list, axis=1)

        parts = [gated, raw_lt, max_env, cert_mean, ctx_mat]
        if phys_norm is not None:
            parts.append(phys_norm)
        return np.concatenate(parts, axis=1).astype(np.float64)

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
        pm_min = pm_mat.min(axis=1, keepdims=True)

        parts = [log_joint_clean, min_clean_d, max_p0_logit, per_dim_clean_logits, pm_mat, pm_min]
        if phys_norm is not None:
            parts.append(phys_norm[:, : min(12, phys_norm.shape[1])])
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
            X_d = self._build_dim_features(cal_models, d, phys_norm)
            mu = X_d.mean(axis=0, keepdims=True)
            sd = X_d.std(axis=0, keepdims=True) + 1e-5
            self.feat_means[d] = mu
            self.feat_stds[d] = sd
            X_dn = np.clip((X_d - mu) / sd, -6.0, 6.0)

            y_loc = np.array([act.index(3 if (d in P0_ONLY_DIMS and v > 0) else int(v)) for v in gold_6d[:, d]], dtype=np.int64)
            W_d, b_d = _fit_l2_multinomial_logreg(X_dn, y_loc, num_classes=len(act), C=self.C)
            self.W_dim[d] = W_d
            self.b_dim[d] = b_d

            logits = X_dn @ W_d + b_d[None, :]
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
        Xcn = np.clip((Xc - self.clean_feat_mean) / self.clean_feat_std, -6.0, 6.0)
        self.W_clean, self.b_clean = _fit_l2_multinomial_logreg(Xcn, y_clean, num_classes=2, C=self.clean_C)
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
            X_d = self._build_dim_features(cal_models, d, phys_norm)
            X_dn = np.clip((X_d - self.feat_means[d]) / self.feat_stds[d], -6.0, 6.0)
            logits = X_dn @ self.W_dim[d] + self.b_dim[d][None, :]
            logits -= logits.max(axis=1, keepdims=True)
            sm = np.exp(logits)
            sm /= sm.sum(axis=1, keepdims=True)
            for idx_k, cls_k in enumerate(act):
                raw_stacked[:, d, cls_k] = sm[:, idx_k].astype(np.float32)

        cal_stacked = self.post_calibrator.transform(raw_stacked)
        Xc = self._build_global_clean_features(cal_models, cal_stacked, phys_norm)
        Xcn = np.clip((Xc - self.clean_feat_mean) / self.clean_feat_std, -6.0, 6.0)
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
