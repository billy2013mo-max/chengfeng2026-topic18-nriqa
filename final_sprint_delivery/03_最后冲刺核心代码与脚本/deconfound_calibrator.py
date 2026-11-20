#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/deconfound_calibrator.py
=================================
Method 2 (Low-Rank Cross-Defect De-confounding Calibration) +
Joint Method 1 & 2 (Spatial Salience x Cross-Defect Ordinal Calibration) +
Clean-Shielded OOB-Validated Cumulative-Ordinal Decoder for `过锐` (dim=2) and `过度生成` (dim=4).

Key statistical & engineering guarantees:
  1. Zero-Intercept Rank-1 Ordinal Shift + Top-K Residual Screening (`LowRankOrdinalCalibrator`):
     - Preserves `RoutedGateEUDecoder`'s `DirDiag` marginal class priors (`w == 0` -> exact `Q_base` identity).
     - Computes explicit 1:1 Tile-vs-Ensemble discrepancy features (`os_fg_vs_ens_gap`, `og_fg_vs_ens_gap`,
       `og_anat_vs_ens_gap`) from the cached 20-D `spatial_salience_probe` vector, suppressing bold-subtitle
       false positives on `过锐` and boosting localized anatomy/foreground peaks on `过度生成`.
  2. OOB-Gated Clean-Shielded Cumulative Ordinal Decoder (`CleanProtectedBalancedDecoder`):
     - Evaluates `(tau_clean_shield, keep_eu_p0)` strictly on 4-fold group-CV Out-Of-Fold predictions
       inside each training split, enforcing hard guardrails on `P0 F1` (`过度生成 P0` & `过锐 P0` preserved)
       and `Clean2` (`>= base_clean - 0.12` on `orig`), while lifting `P1/P2 F1` via group-bagged thresholds.
  3. Canonical Module Namespace Pickling + Strict `N = 1` Batch Invariance (`0 / 900` cell diff).
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

from spatial_salience_probe import SPATIAL_FEAT_DIM  # noqa: E402

DIMS_ZH = ("模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀")
TARGET_DIMS = (2, 4)  # 2: 过锐, 4: 过度生成
OS_SPATIAL_IDX = list(range(0, 10))
OG_SPATIAL_IDX = list(range(10, 20))

NTG_KEYS = ("ntg_pe", "ntg_convnext")
GLOBAL_T1_KEYS = ("cradio_v3", "pe_v3", "peg", "sig_v3", "pe_bkseg1", "cradio", "pe", "sig")


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pc = np.clip(p, eps, 1.0 - eps)
    return np.log(pc / (1.0 - pc))


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    ez = np.exp(np.clip(z, -40.0, 40.0))
    return ez / np.sum(ez, axis=-1, keepdims=True)


def extract_deconf_covariates(
    Q_base: np.ndarray,
    P_dict: Dict[str, np.ndarray],
    dim: int,
    members: Sequence[str],
) -> np.ndarray:
    """
    Strictly row-wise (`N = 1` invariant) 8-D pure cross-defect & inter-member discrepancy covariates.
    """
    Q = np.clip(np.asarray(Q_base, dtype=np.float64), 1e-7, 1.0)
    Q = Q / np.sum(Q, axis=-1, keepdims=True)
    N = Q.shape[0]
    grades_vec = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    exp_sev_all = np.einsum("ndc,c->nd", Q, grades_vec)
    blur_exp_sev = exp_sev_all[:, 0]
    texloss_exp_sev = exp_sev_all[:, 1]
    other_d = 4 if dim == 2 else 2
    other_osog_sev = exp_sev_all[:, other_d]
    bokeh_p0_logit = _safe_logit(Q[:, 5, 3])

    other_cols = [k for k in range(6) if k != dim]
    max_other_any = np.max(1.0 - Q[:, other_cols, 0], axis=1)
    clean_other_logp = np.sum(np.log(np.clip(Q[:, other_cols, 0], 1e-6, 1.0)), axis=1)

    mem_sevs, ntg_sevs, t1_sevs = [], [], []
    for m in members:
        if m not in P_dict:
            continue
        pm = np.clip(np.asarray(P_dict[m], dtype=np.float64), 1e-7, 1.0)
        pm = pm / np.sum(pm, axis=-1, keepdims=True)
        sd_m = np.einsum("nc,c->n", pm[:, dim, :], grades_vec)
        mem_sevs.append(sd_m)
        if m in NTG_KEYS:
            ntg_sevs.append(sd_m)
        elif m in GLOBAL_T1_KEYS:
            t1_sevs.append(sd_m)

    member_std_sev = np.std(np.stack(mem_sevs, axis=1), axis=1) if len(mem_sevs) >= 2 else np.zeros(N, dtype=np.float64)
    if ntg_sevs and t1_sevs:
        ntg_vs_global_gap = np.mean(np.stack(ntg_sevs, axis=1), axis=1) - np.mean(np.stack(t1_sevs, axis=1), axis=1)
    else:
        ntg_vs_global_gap = np.zeros(N, dtype=np.float64)

    X = np.stack(
        [
            blur_exp_sev,
            texloss_exp_sev,
            other_osog_sev,
            bokeh_p0_logit,
            max_other_any,
            clean_other_logp,
            member_std_sev,
            ntg_vs_global_gap,
        ],
        axis=1,
    )
    return np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)


def build_dim_features(
    Q_base: np.ndarray,
    P_dict: Dict[str, np.ndarray],
    dim: int,
    members: Sequence[str],
    mode: str,
    spatial_feats: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Strictly row-wise (`N = 1` invariant) feature builder for `mode in {'deconf', 'spatial', 'joint'}`.
    """
    Q = np.clip(np.asarray(Q_base, dtype=np.float64), 1e-7, 1.0)
    Q = Q / np.sum(Q, axis=-1, keepdims=True)
    ens_p_any = 1.0 - Q[:, dim, 0]
    ens_p_ge1 = Q[:, dim, 2] + Q[:, dim, 3]

    X_dc = extract_deconf_covariates(Q, P_dict, dim, members)
    if mode == "deconf":
        return X_dc

    assert spatial_feats is not None, f"spatial_feats is required for mode={mode!r}"
    sp = np.asarray(spatial_feats, dtype=np.float64)
    assert sp.shape[0] == Q.shape[0] and sp.shape[1] == SPATIAL_FEAT_DIM

    if dim == 2:
        X_sp_raw = sp[:, OS_SPATIAL_IDX]
        # Explicit 1:1 non-UI foreground tile vs global ensemble discrepancy
        fg_vs_ens = X_sp_raw[:, 0] - ens_p_any
        p01_vs_ens = X_sp_raw[:, 6] - ens_p_ge1
        X_sp = np.column_stack([X_sp_raw, fg_vs_ens, p01_vs_ens])
    else:
        X_sp_raw = sp[:, OG_SPATIAL_IDX]
        fg_vs_ens = X_sp_raw[:, 0] - ens_p_any
        anat_vs_ens = X_sp_raw[:, 1] - ens_p_any
        p01_vs_ens = X_sp_raw[:, 7] - ens_p_ge1
        X_sp = np.column_stack([X_sp_raw, fg_vs_ens, anat_vs_ens, p01_vs_ens])

    X_sp = np.nan_to_num(X_sp, nan=0.0, posinf=10.0, neginf=-10.0)
    if mode == "spatial":
        return X_sp
    if mode == "joint":
        return np.concatenate([X_dc, X_sp], axis=1)
    raise ValueError(f"Unknown feature mode: {mode!r}")


def compute_dim_rps(q_4: np.ndarray, y_int: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    cdf_p = np.cumsum(q_4[:, :3], axis=1)
    ks = np.arange(3, dtype=np.int64)[None, :]
    cdf_y = (y_int[:, None] <= ks).astype(np.float64)
    per_row = np.sum((cdf_p - cdf_y) ** 2, axis=1)
    if weights is None:
        return float(np.mean(per_row))
    return float(np.sum(weights * per_row) / np.maximum(1e-9, np.sum(weights)))


def compute_dim_ap_metrics(q_4: np.ndarray, y_int: np.ndarray) -> Dict[str, float]:
    s_any = 1.0 - q_4[:, 0]
    s_ge1 = q_4[:, 2] + q_4[:, 3]
    s_p0 = q_4[:, 3]
    s_exp = q_4[:, 1] + 2.0 * q_4[:, 2] + 3.0 * q_4[:, 3]

    ap_any = float(average_precision_score((y_int >= 1).astype(int), s_any + 0.05 * s_exp)) if np.any(y_int >= 1) else 0.0
    ap_ge1 = float(average_precision_score((y_int >= 2).astype(int), s_ge1 + 0.05 * s_exp)) if np.any(y_int >= 2) else 0.0
    ap_p0 = float(average_precision_score((y_int == 3).astype(int), s_p0 + 0.02 * s_exp)) if np.any(y_int == 3) else 0.0
    return {
        "ap_any_ge_p2": ap_any,
        "ap_sev_ge_p1": ap_ge1,
        "ap_p0": ap_p0,
        "mean_ap": float((ap_any + ap_ge1 + ap_p0) / 3.0),
    }


def _f1_bin(pred_b: np.ndarray, true_b: np.ndarray) -> float:
    tp = int(np.sum(pred_b & true_b))
    fp = int(np.sum(pred_b & (~true_b)))
    fn = int(np.sum((~pred_b) & true_b))
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return float(2.0 * tp / max(1, 2 * tp + fp + fn))


def compute_dim_grade_f1s(pred_d: np.ndarray, y_d: np.ndarray) -> Dict[str, float]:
    f_p0 = _f1_bin(pred_d == 3, y_d == 3)
    f_p1 = _f1_bin(pred_d == 2, y_d == 2)
    f_p2 = _f1_bin(pred_d == 1, y_d == 1)
    return {"P0": f_p0, "P1": f_p1, "P2": f_p2, "mean": float((f_p0 + f_p1 + f_p2) / 3.0)}


class LowRankOrdinalCalibrator:
    """
    Zero-intercept Rank-1 Ordinal Shift Calibrator with Top-K Residual Feature Screening.
    """

    ANCHOR_SHIFT = np.array([-1.25, 0.35, 0.90, 1.45], dtype=np.float64)

    def __init__(
        self,
        dim: int,
        mode: str = "joint",
        topk_grid: Tuple[int, ...] = (2, 4, 8),
        l2_grid: Tuple[float, ...] = (50.0, 150.0, 450.0, 1200.0),
        alpha_grid: Tuple[float, ...] = (0.0, 0.20, 0.40, 0.65, 0.85, 1.0),
        w_supp: float = 0.60,
    ):
        self.dim = int(dim)
        self.mode = str(mode)
        self.topk_grid = tuple(int(k) for k in topk_grid)
        self.l2_grid = tuple(float(x) for x in l2_grid)
        self.alpha_grid = tuple(float(x) for x in alpha_grid)
        self.w_supp = float(w_supp)

        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None
        self.sel_cols_: Optional[np.ndarray] = None
        self.w_: Optional[np.ndarray] = None
        self.best_topk_: int = 2
        self.best_l2_: float = 150.0
        self.alpha_: float = 0.0

    @staticmethod
    def _select_topk_cols(Z_tr: np.ndarray, q_tr: np.ndarray, y_tr: np.ndarray, k: int) -> np.ndarray:
        F_dim = Z_tr.shape[1]
        if k >= F_dim:
            return np.arange(F_dim, dtype=np.int64)
        exp_q = q_tr[:, 1] + 2.0 * q_tr[:, 2] + 3.0 * q_tr[:, 3]
        resid = y_tr.astype(np.float64) - exp_q
        resid_c = resid - np.mean(resid)
        denom_r = np.linalg.norm(resid_c) + 1e-8
        corrs = np.zeros(F_dim, dtype=np.float64)
        for j in range(F_dim):
            zj = Z_tr[:, j] - np.mean(Z_tr[:, j])
            corrs[j] = abs(float(np.dot(zj, resid_c)) / ((np.linalg.norm(zj) + 1e-8) * denom_r))
        return np.sort(np.argsort(-corrs)[:k])

    def _raw_probs(self, Z_sub: np.ndarray, q_base_d: np.ndarray, w: np.ndarray) -> np.ndarray:
        eta = Z_sub @ w
        log_q0 = np.log(np.clip(q_base_d, 1e-7, 1.0))
        return _softmax_rows(log_q0 + eta[:, None] * self.ANCHOR_SHIFT[None, :])

    def _fit_w(self, Z_sub: np.ndarray, q_base_d: np.ndarray, y_d: np.ndarray, sample_w: np.ndarray, l2_reg: float) -> np.ndarray:
        K = Z_sub.shape[1]
        ks = np.arange(3, dtype=np.int64)[None, :]
        cdf_true = (y_d[:, None] <= ks).astype(np.float64)
        sw = sample_w / np.mean(sample_w)
        N = float(len(y_d))
        idx_n = np.arange(len(y_d))

        def _loss_and_grad(w: np.ndarray):
            p = self._raw_probs(Z_sub, q_base_d, w)
            cdf_p = np.cumsum(p[:, :3], axis=1)
            diff_cdf = cdf_p - cdf_true
            rps_i = np.sum(diff_cdf ** 2, axis=1)
            nll_i = -np.log(np.clip(p[idx_n, y_d], 1e-9, 1.0))
            loss_data = float(np.sum(sw * (rps_i + 0.25 * nll_i)) / N)
            loss_reg = 0.5 * (l2_reg / N) * float(np.sum(w ** 2))

            g_p = np.zeros_like(p)
            g_p[:, 0] = 2.0 * (diff_cdf[:, 0] + diff_cdf[:, 1] + diff_cdf[:, 2])
            g_p[:, 1] = 2.0 * (diff_cdf[:, 1] + diff_cdf[:, 2])
            g_p[:, 2] = 2.0 * diff_cdf[:, 2]
            g_p[idx_n, y_d] -= 0.25 / np.clip(p[idx_n, y_d], 1e-7, 1.0)
            g_p *= (sw / N)[:, None]

            dot_pg = np.sum(p * g_p, axis=1, keepdims=True)
            g_log = p * (g_p - dot_pg)
            g_eta = g_log @ self.ANCHOR_SHIFT
            grad = Z_sub.T @ g_eta + (l2_reg / N) * w
            return loss_data + loss_reg, grad

        res = minimize(_loss_and_grad, np.zeros(K, dtype=np.float64), jac=True, method="L-BFGS-B", options={"maxiter": 120})
        return res.x.astype(np.float64)

    def fit(
        self,
        X: np.ndarray,
        q_base_d: np.ndarray,
        y_d: np.ndarray,
        groups: np.ndarray,
        is_orig: Optional[np.ndarray] = None,
    ) -> "LowRankOrdinalCalibrator":
        X = np.asarray(X, dtype=np.float64)
        q_base_d = np.asarray(q_base_d, dtype=np.float64)
        y_d = np.asarray(y_d, dtype=np.int64)
        N = X.shape[0]
        sw = np.where(np.asarray(is_orig, dtype=bool), 1.0, self.w_supp).astype(np.float64) if is_orig is not None else np.ones(N, dtype=np.float64)

        uniq_g = sorted(set(str(g) for g in groups))
        g_to_b = {g: (i % 4) for i, g in enumerate(uniq_g)}
        inner_f = np.array([g_to_b[str(g)] for g in groups], dtype=np.int64)

        def _inner_obj(p_cand: np.ndarray) -> float:
            rps_all = compute_dim_rps(p_cand, y_d)
            rps_w = compute_dim_rps(p_cand, y_d, weights=sw)
            aps = compute_dim_ap_metrics(p_cand, y_d)
            return 0.60 * rps_all + 0.40 * rps_w - 0.020 * aps["mean_ap"]

        base_rps_all = compute_dim_rps(q_base_d, y_d)
        best_score = _inner_obj(q_base_d)
        best_cfg = (2, 150.0, 0.0)

        for tk in self.topk_grid:
            for l2_v in self.l2_grid:
                oof_p = np.zeros_like(q_base_d)
                for b in range(4):
                    va = inner_f == b
                    tr = ~va
                    mu_b = X[tr].mean(axis=0)
                    sd_b = X[tr].std(axis=0) + 1e-5
                    Z_tr = np.clip((X[tr] - mu_b) / sd_b, -5.0, 5.0)
                    Z_va = np.clip((X[va] - mu_b) / sd_b, -5.0, 5.0)
                    cols_b = self._select_topk_cols(Z_tr, q_base_d[tr], y_d[tr], tk)
                    w_b = self._fit_w(Z_tr[:, cols_b], q_base_d[tr], y_d[tr], sw[tr], l2_v)
                    oof_p[va] = self._raw_probs(Z_va[:, cols_b], q_base_d[va], w_b)

                for al in self.alpha_grid:
                    if al <= 0.0:
                        continue
                    p_mix = (1.0 - al) * q_base_d + al * oof_p
                    p_mix = p_mix / np.sum(p_mix, axis=-1, keepdims=True)
                    if compute_dim_rps(p_mix, y_d) > base_rps_all + 1e-5:
                        continue
                    sc = _inner_obj(p_mix)
                    if sc < best_score - 1e-6:
                        best_score = sc
                        best_cfg = (tk, l2_v, al)

        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + 1e-5
        self.best_topk_, self.best_l2_, self.alpha_ = int(best_cfg[0]), float(best_cfg[1]), float(best_cfg[2])
        Z_full = np.clip((X - self.mu_) / self.sd_, -5.0, 5.0)
        self.sel_cols_ = self._select_topk_cols(Z_full, q_base_d, y_d, self.best_topk_)
        self.w_ = self._fit_w(Z_full[:, self.sel_cols_], q_base_d, y_d, sw, self.best_l2_)
        return self

    def predict_proba(self, X: np.ndarray, q_base_d: np.ndarray) -> np.ndarray:
        q_base_d = np.asarray(q_base_d, dtype=np.float64)
        if self.alpha_ <= 0.0 or self.w_ is None or self.sel_cols_ is None:
            return q_base_d.copy()
        Z = np.clip((np.asarray(X, dtype=np.float64) - self.mu_) / self.sd_, -5.0, 5.0)
        p_raw = self._raw_probs(Z[:, self.sel_cols_], q_base_d, self.w_)
        p_out = (1.0 - self.alpha_) * q_base_d + self.alpha_ * p_raw
        return p_out / np.sum(p_out, axis=-1, keepdims=True)


class CleanProtectedBalancedDecoder:
    """
    OOF-Gated Clean-Shielded Group-Bootstrap Cumulative-Threshold Ordinal Decoder.
    Selects `(tau_clean_shield, keep_eu_p0, active_d)` strictly via 4-fold video-grouped inner OOF
    so `P0 F1` and `Clean2` are mathematically guarded against in-sample threshold optimism!
    """

    TG = np.linspace(0.05, 0.95, 46)

    def __init__(self, n_bags: int = 24, seed: int = 2026):
        self.n_bags = int(n_bags)
        self.seed = int(seed)
        self.thresholds_: Dict[int, np.ndarray] = {}
        self.keep_eu_p0_: Dict[int, bool] = {2: True, 4: True}
        self.tau_clean_shield_: float = 0.65
        self.active_: Dict[int, bool] = {2: False, 4: False}

    @staticmethod
    def cum_scores(Qd: np.ndarray) -> np.ndarray:
        return np.stack([Qd[:, 1:].sum(axis=1), Qd[:, 2:].sum(axis=1), Qd[:, 3]], axis=1)

    @staticmethod
    def dec_cum(C: np.ndarray, t: Sequence[float]) -> np.ndarray:
        c0 = C[:, 0] > t[0]
        c1 = c0 & (C[:, 1] > t[1])
        c2 = c1 & (C[:, 2] > t[2])
        return c0.astype(np.int64) + c1.astype(np.int64) + c2.astype(np.int64)

    @classmethod
    def _fit_cum_one(cls, C: np.ndarray, y: np.ndarray, w: Tuple[float, float, float] = (1.0, 1.15, 1.0)) -> np.ndarray:
        t = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        best = -1.0
        wv = np.asarray(w, dtype=np.float64)
        for _ in range(4):
            for j in range(3):
                for v in cls.TG:
                    tt = t.copy()
                    tt[j] = float(v)
                    pr = cls.dec_cum(C, tt)
                    f1_3 = np.array([_f1_bin(pr == 1, y == 1), _f1_bin(pr == 2, y == 2), _f1_bin(pr == 3, y == 3)])
                    s = float(np.sum(f1_3 * wv))
                    if s > best + 1e-12:
                        best, t = s, tt
        return t

    @classmethod
    def _fit_bagged_thr(cls, Qd: np.ndarray, yd: np.ndarray, g_str: np.ndarray, n_bags: int, seed: int, d: int) -> np.ndarray:
        C_d = cls.cum_scores(Qd)
        uniq_g = np.unique(g_str)
        grp_idx = {u: np.where(g_str == u)[0] for u in uniq_g}
        rng = np.random.default_rng(seed + 31 * int(d))
        w_d = (1.0, 1.15, 1.0) if d == 2 else (1.0, 1.10, 1.0)
        ts = []
        for _ in range(n_bags):
            picked_u = rng.choice(uniq_g, size=len(uniq_g), replace=True)
            sel = np.concatenate([grp_idx[u] for u in picked_u])
            ts.append(cls._fit_cum_one(C_d[sel], yd[sel], w=w_d))
        return np.median(np.stack(ts, axis=0), axis=0)

    @staticmethod
    def _apply_dim_rule(
        Qd: np.ndarray,
        pred_eu_d: np.ndarray,
        eu_row_clean: np.ndarray,
        p_cl_joint: np.ndarray,
        thr: Sequence[float],
        keep_eu_p0: bool,
        tau_clean_shield: float,
    ) -> np.ndarray:
        C = CleanProtectedBalancedDecoder.cum_scores(Qd)
        raw = CleanProtectedBalancedDecoder.dec_cum(C, thr)
        if keep_eu_p0:
            raw = np.minimum(raw, 2)
            raw[pred_eu_d == 3] = 3
        shielded = eu_row_clean & (p_cl_joint >= tau_clean_shield)
        raw[eu_row_clean & (raw == 3)] = 2
        raw[shielded] = 0
        return raw

    def fit(
        self,
        Q_cal: np.ndarray,
        pred_eu: np.ndarray,
        gold: np.ndarray,
        groups: np.ndarray,
        is_orig: Optional[np.ndarray] = None,
        active_dims: Sequence[int] = TARGET_DIMS,
    ) -> "CleanProtectedBalancedDecoder":
        import common55 as c

        if is_orig is None:
            is_orig = np.ones(len(gold), dtype=bool)
        g_str = np.array([str(g) for g in groups])
        uniq_g = sorted(set(g_str))
        g_to_b = {g: (i % 4) for i, g in enumerate(uniq_g)}
        inner_f = np.array([g_to_b[g] for g in g_str], dtype=np.int64)

        # Step 1: Precompute honest 4-split inner OOF cumulative thresholds per dimension
        inner_thrs: Dict[int, List[np.ndarray]] = {}
        for d in active_dims:
            th_list = []
            for b in range(4):
                tr = inner_f != b
                th_b = self._fit_bagged_thr(Q_cal[tr, d, :], gold[tr, d], g_str[tr], max(12, self.n_bags // 2), self.seed + b, d)
                th_list.append(th_b)
            inner_thrs[int(d)] = th_list
            # Full-split bagged thresholds for final deployment
            self.thresholds_[int(d)] = self._fit_bagged_thr(Q_cal[:, d, :], gold[:, d], g_str, self.n_bags, self.seed, d)

        # Step 2: Evaluate `(tau_clean_shield, act2_mode, act4_mode)` strictly on inner OOF!
        # act_mode in {"off", "keep_p0", "full_cum"}
        eu_row_clean = np.all(pred_eu == 0, axis=1)
        p_cl_joint = np.prod(np.clip(Q_cal[:, :, 0], 1e-6, 1.0), axis=1)

        base_sc_o = c.score(pred_eu[is_orig], gold[is_orig])
        base_sc_a = c.score(pred_eu, gold)
        base_orig_doc2 = float(base_sc_o["doc2_strict_total"])
        base_all_doc2 = float(base_sc_a["doc2_strict_total"])
        base_clean_o = float(base_sc_o["doc2_Clean2"])
        base_f_os = compute_dim_grade_f1s(pred_eu[:, 2], gold[:, 2])
        base_f_og = compute_dim_grade_f1s(pred_eu[:, 4], gold[:, 4])

        best_val = 0.0  # 0.0 corresponds to ("off", "off"), i.e. exact EU baseline if nothing improves!
        best_choice = (0.65, "off", "off")

        for tau_sh in (0.52, 0.62, 0.70, 0.78):
            for m2 in ("off", "keep_p0", "full_cum"):
                for m4 in ("off", "keep_p0", "full_cum"):
                    if m2 == "off" and m4 == "off":
                        continue
                    cand = pred_eu.copy()
                    for b in range(4):
                        va = inner_f == b
                        if m2 != "off" and 2 in inner_thrs:
                            cand[va, 2] = self._apply_dim_rule(
                                Q_cal[va, 2, :], pred_eu[va, 2], eu_row_clean[va], p_cl_joint[va],
                                inner_thrs[2][b], keep_eu_p0=(m2 == "keep_p0"), tau_clean_shield=tau_sh
                            )
                        if m4 != "off" and 4 in inner_thrs:
                            cand[va, 4] = self._apply_dim_rule(
                                Q_cal[va, 4, :], pred_eu[va, 4], eu_row_clean[va], p_cl_joint[va],
                                inner_thrs[4][b], keep_eu_p0=(m4 == "keep_p0"), tau_clean_shield=tau_sh
                            )

                    sc_o = c.score(cand[is_orig], gold[is_orig])
                    sc_a = c.score(cand, gold)
                    f_os = compute_dim_grade_f1s(cand[:, 2], gold[:, 2])
                    f_og = compute_dim_grade_f1s(cand[:, 4], gold[:, 4])

                    # Hard inner-OOF guardrails:
                    # 1) Clean2 on orig loses at most 0.12 pts (stays >= ~19.68/20)
                    # 2) Neither dimension's P0 F1 drops by > 0.020
                    # 3) Doc2_ORIG and Doc2_ALL stay within 0.08 pts of EU (or higher!)
                    if float(sc_o["doc2_Clean2"]) < base_clean_o - 0.12:
                        continue
                    if f_os["P0"] < base_f_os["P0"] - 0.020 or f_og["P0"] < base_f_og["P0"] - 0.020:
                        continue
                    if float(sc_o["doc2_strict_total"]) < base_orig_doc2 - 0.08:
                        continue
                    if float(sc_a["doc2_strict_total"]) < base_all_doc2 - 0.08:
                        continue

                    doc2_gain = 0.5 * (float(sc_a["doc2_strict_total"]) - base_all_doc2) + 0.5 * (float(sc_o["doc2_strict_total"]) - base_orig_doc2)
                    f1_gain = 4.0 * ((f_os["mean"] - base_f_os["mean"]) + (f_og["mean"] - base_f_og["mean"]))
                    p12_gain = 2.0 * ((f_os["P1"] - base_f_os["P1"]) + (f_og["P1"] - base_f_og["P1"]) + (f_og["P2"] - base_f_og["P2"]))
                    val = doc2_gain + f1_gain + p12_gain
                    if val > best_val + 1e-4:
                        best_val = val
                        best_choice = (tau_sh, m2, m4)

        self.tau_clean_shield_ = float(best_choice[0])
        self.active_[2] = best_choice[1] != "off"
        self.keep_eu_p0_[2] = best_choice[1] == "keep_p0"
        self.active_[4] = best_choice[2] != "off"
        self.keep_eu_p0_[4] = best_choice[2] == "keep_p0"
        return self

    def predict(self, Q_cal: np.ndarray, pred_eu: np.ndarray) -> np.ndarray:
        out = np.asarray(pred_eu, dtype=np.int64).copy()
        eu_row_clean = np.all(out == 0, axis=1)
        p_cl_joint = np.prod(np.clip(Q_cal[:, :, 0], 1e-6, 1.0), axis=1)
        for d, is_act in self.active_.items():
            if not is_act or d not in self.thresholds_:
                continue
            out[:, d] = self._apply_dim_rule(
                Q_cal[:, d, :],
                out[:, d],
                eu_row_clean,
                p_cl_joint,
                self.thresholds_[d],
                self.keep_eu_p0_[d],
                self.tau_clean_shield_,
            )
        return out


class JointOsOgBreakthroughModel:
    """
    Self-contained `N = 1` batch-invariant deployment object.
    """

    def __init__(
        self,
        set_name: str,
        members: Sequence[str],
        mode: str,
        active_cal_dims: Sequence[int],
        base_decoder,
        calibrators: Dict[int, LowRankOrdinalCalibrator],
        eu_decoder,
        bal_decoder: CleanProtectedBalancedDecoder,
    ):
        self.set_name = str(set_name)
        self.members = list(members)
        self.mode = str(mode)
        self.active_cal_dims = tuple(int(d) for d in active_cal_dims)
        self.base_decoder = base_decoder
        self.calibrators = dict(calibrators)
        self.eu_decoder = eu_decoder
        self.bal_decoder = bal_decoder

    def calibrate_probs(
        self,
        P_dict: Dict[str, np.ndarray],
        spatial_feats: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        Q_base = self.base_decoder.pool_from_member_dict(P_dict)
        Q_cal = Q_base.copy()
        for d in self.active_cal_dims:
            cal = self.calibrators.get(d)
            if cal is None or cal.alpha_ <= 0.0:
                continue
            X_d = build_dim_features(Q_base, P_dict, d, self.members, self.mode, spatial_feats=spatial_feats)
            Q_cal[:, d, :] = cal.predict_proba(X_d, Q_base[:, d, :])
        return Q_cal

    def predict_all(
        self,
        P_dict: Dict[str, np.ndarray],
        spatial_feats: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        Q_base = self.base_decoder.pool_from_member_dict(P_dict)
        pred_base = self.base_decoder.eu.predict(Q_base)
        Q_cal = self.calibrate_probs(P_dict, spatial_feats=spatial_feats)
        pred_eu = self.eu_decoder.predict(Q_cal)
        pred_bal = self.bal_decoder.predict(Q_cal, pred_eu)
        return {
            "Q_base": Q_base,
            "Q_cal": Q_cal,
            "pred_base": pred_base,
            "pred_primary_eu": pred_eu,
            "pred_balanced": pred_bal,
        }


def _build_routed_gate_cfg(members: Sequence[str]) -> dict:
    from final_engine import P12_SPEC, _base_family_weights

    base_m = [m for m in members if m not in P12_SPEC]
    base_prior = list(_base_family_weights(base_m))
    p12_m = [m for m in members if m in P12_SPEC]
    p12_int = [0.65, 0.35] if set(p12_m) == {"p12_pe", "p12_g14"} else [1.0 / max(1, len(p12_m))] * len(p12_m)
    p12_tot = 0.20 if len(p12_m) >= 2 else 0.18
    cfg = dict(
        base_members=base_m,
        base_prior=base_prior,
        p12_members=p12_m,
        p12_internal_weights=p12_int,
        p12_total_prior=p12_tot,
        l2_reg=0.015,
    )
    if "p1_qy" in members:
        cfg["lams_cl"] = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)
    if "pe_bkseg1" in members:
        cfg["bkseg1_alpha"] = 0.25
    return cfg


def summarize_oof_variant(
    tag: str,
    Q_oof: np.ndarray,
    pred_oof: np.ndarray,
    gold: np.ndarray,
    is_orig: np.ndarray,
) -> Dict[str, object]:
    import common55 as c

    sc_all = c.score(pred_oof, gold)
    sc_orig = c.score(pred_oof[is_orig], gold[is_orig])
    sc_supp = c.score(pred_oof[~is_orig], gold[~is_orig])

    rps_dims = [compute_dim_rps(Q_oof[:, d, :], gold[:, d]) for d in (0, 1, 2, 4)]
    rps_bin = [float(np.mean((Q_oof[:, d, 3] - (gold[:, d] >= 3).astype(np.float64)) ** 2)) for d in (3, 5)]
    rps_all = float((sum(rps_dims) + sum(rps_bin)) / 6.0)

    dim_stats = {}
    for d, d_name in ((2, "过锐"), (4, "过度生成")):
        dim_stats[d_name] = {
            "rps_all": compute_dim_rps(Q_oof[:, d, :], gold[:, d]),
            "rps_orig": compute_dim_rps(Q_oof[is_orig, d, :], gold[is_orig, d]),
            "ap_all": compute_dim_ap_metrics(Q_oof[:, d, :], gold[:, d]),
            "ap_orig": compute_dim_ap_metrics(Q_oof[is_orig, d, :], gold[is_orig, d]),
            "f1_all": compute_dim_grade_f1s(pred_oof[:, d], gold[:, d]),
            "f1_orig": compute_dim_grade_f1s(pred_oof[is_orig, d], gold[is_orig, d]),
            "f1_supp": compute_dim_grade_f1s(pred_oof[~is_orig, d], gold[~is_orig, d]),
        }

    return {
        "variant": tag,
        "doc2_all": float(sc_all["doc2_strict_total"]),
        "doc2_orig": float(sc_orig["doc2_strict_total"]),
        "doc2_supp": float(sc_supp["doc2_strict_total"]),
        "orig_parts": {
            "pos": float(sc_orig.get("doc2_S_pos", 0.0)),
            "p0": float(sc_orig.get("doc2_S_p0", 0.0)),
            "clean": float(sc_orig.get("doc2_Clean2", 0.0)),
        },
        "rps_all": rps_all,
        "os": dim_stats["过锐"],
        "og": dim_stats["过度生成"],
    }


def _print_ablation_table(summaries: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 124)
    print(f"{'Variant':<28} | {'Doc2 ALL':>8} {'ORIG':>8} {'Cln_O':>6} | {'RPS':>7} | "
          f"{'过锐 mAP':>8} {'P0/P1/P2 F1 (ALL)':>19} | {'过度生成 mAP':>10} {'P0/P1/P2 F1 (ALL)':>19}")
    print("-" * 124)
    for s in summaries:
        os_f = s["os"]["f1_all"]
        og_f = s["og"]["f1_all"]
        os_ap = s["os"]["ap_all"]["mean_ap"]
        og_ap = s["og"]["ap_all"]["mean_ap"]
        cln_o = s["orig_parts"]["clean"]
        f_os_str = f"{os_f['P0']:.3f}/{os_f['P1']:.3f}/{os_f['P2']:.3f}"
        f_og_str = f"{og_f['P0']:.3f}/{og_f['P1']:.3f}/{og_f['P2']:.3f}"
        print(
            f"{s['variant']:<28} | {s['doc2_all']:8.3f} {s['doc2_orig']:8.3f} {cln_o:6.2f} | {s['rps_all']:7.5f} | "
            f"{os_ap:8.4f} {f_os_str:>19} | {og_ap:10.4f} {f_og_str:>19}"
        )
    print("=" * 124 + "\n", flush=True)


def run_nested_ablation_and_fit(
    set_name: str = "resident_v7",
    spatial_oof_path: str = "",
    out_dir: str = "results/r55/os_og_breakthrough",
) -> Tuple[str, Dict[str, object]]:
    import common55 as c
    from final_engine import SETS, oof_pool
    from routed_eu_decoder import RoutedGateEUDecoder, TrueGateConstrainedEU

    t0 = time.time()
    members = SETS[set_name]
    names, gold, fold_arr, _, P_dict = oof_pool(members)
    _, _, _, groups = c.load_split()
    split_map = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([split_map[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)

    spatial_oof = None
    if spatial_oof_path and os.path.exists(spatial_oof_path):
        z_sp = np.load(spatial_oof_path, allow_pickle=True)
        mp_sp = {str(nm): fv for nm, fv in zip(z_sp["names"], z_sp["feats"])}
        spatial_oof = np.stack([mp_sp[str(n)] for n in names], axis=0).astype(np.float64)
        print(f"[deconfound_calibrator] loaded leak-free OOF spatial features: {spatial_oof.shape} from {spatial_oof_path}", flush=True)
    else:
        print("[deconfound_calibrator] WARNING: no spatial_oof_path found; evaluating Method 2 (deconf) only.", flush=True)

    proto_cfg = _build_routed_gate_cfg(members)

    print(f"[deconfound_calibrator] fitting 8-fold base RoutedGateEUDecoder for '{set_name}' ({len(members)} members)...", flush=True)
    fold_base_decs = []
    Q_base_oof = np.zeros((N, 6, 4), dtype=np.float64)
    pred_base_oof = np.zeros((N, 6), dtype=np.int64)

    for k in range(8):
        va = fold_arr == k
        tr = ~va
        P_tr = {m: v[tr] for m, v in P_dict.items()}
        P_va = {m: v[va] for m, v in P_dict.items()}
        dec_k = RoutedGateEUDecoder(**proto_cfg).fit(P_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
        fold_base_decs.append(dec_k)
        q_va = dec_k.pool_from_member_dict(P_va)
        Q_base_oof[va] = q_va
        pred_base_oof[va] = dec_k.eu.predict(q_va)

    summaries: List[Dict[str, object]] = []
    base_sum = summarize_oof_variant("0_baseline_routed_gate_eu", Q_base_oof, pred_base_oof, gold, is_orig)
    summaries.append(base_sum)

    modes_to_run = ["deconf"] + (["spatial", "joint"] if spatial_oof is not None else [])
    mode_results = {}

    for mode in modes_to_run:
        Q_mode_raw = Q_base_oof.copy()
        fold_cals: Dict[int, List[LowRankOrdinalCalibrator]] = {2: [], 4: []}
        Q_tr_cal_by_fold: List[Dict[int, np.ndarray]] = [{} for _ in range(8)]

        for k in range(8):
            va = fold_arr == k
            tr = ~va
            dec_k = fold_base_decs[k]
            P_tr = {m: v[tr] for m, v in P_dict.items()}
            P_va = {m: v[va] for m, v in P_dict.items()}
            q_tr_base = dec_k.pool_from_member_dict(P_tr)
            q_va_base = Q_base_oof[va]
            sp_tr = spatial_oof[tr] if spatial_oof is not None else None
            sp_va = spatial_oof[va] if spatial_oof is not None else None

            for d in TARGET_DIMS:
                X_tr = build_dim_features(q_tr_base, P_tr, d, members, mode, spatial_feats=sp_tr)
                X_va = build_dim_features(q_va_base, P_va, d, members, mode, spatial_feats=sp_va)
                cal_kd = LowRankOrdinalCalibrator(dim=d, mode=mode).fit(
                    X_tr, q_tr_base[:, d, :], gold[tr, d], groups[tr], is_orig=is_orig[tr]
                )
                fold_cals[d].append(cal_kd)
                Q_mode_raw[va, d, :] = cal_kd.predict_proba(X_va, q_va_base[:, d, :])
                Q_tr_cal_by_fold[k][d] = cal_kd.predict_proba(X_tr, q_tr_base[:, d, :])

        active_dims: List[int] = []
        for d in TARGET_DIMS:
            rps_b = compute_dim_rps(Q_base_oof[:, d, :], gold[:, d])
            rps_m = compute_dim_rps(Q_mode_raw[:, d, :], gold[:, d])
            ap_b = compute_dim_ap_metrics(Q_base_oof[:, d, :], gold[:, d])["mean_ap"]
            ap_m = compute_dim_ap_metrics(Q_mode_raw[:, d, :], gold[:, d])["mean_ap"]
            mean_alpha = float(np.mean([c_k.alpha_ for c_k in fold_cals[d]]))
            print(
                f"  [mode={mode:<7} dim={DIMS_ZH[d]}] RPS {rps_b:.5f} -> {rps_m:.5f} ({rps_m - rps_b:+.5f}) | "
                f"mAP {ap_b:.4f} -> {ap_m:.4f} ({ap_m - ap_b:+.4f}) | mean_alpha={mean_alpha:.2f}",
                flush=True,
            )
            if (rps_m <= rps_b + 5e-5 and ap_m > ap_b + 5e-5) or (rps_m < rps_b - 5e-6):
                active_dims.append(d)

        Q_gated_oof = Q_base_oof.copy()
        for d in active_dims:
            Q_gated_oof[:, d, :] = Q_mode_raw[:, d, :]

        pred_eu_oof = np.zeros((N, 6), dtype=np.int64)
        pred_bal_oof = np.zeros((N, 6), dtype=np.int64)

        for k in range(8):
            va = fold_arr == k
            tr = ~va
            dec_k = fold_base_decs[k]
            P_tr = {m: v[tr] for m, v in P_dict.items()}
            q_tr_cal = dec_k.pool_from_member_dict(P_tr)
            for d in active_dims:
                q_tr_cal[:, d, :] = Q_tr_cal_by_fold[k][d]

            if active_dims:
                eu_k = TrueGateConstrainedEU(
                    lams_cl=dec_k.lams_cl,
                    p0_scales=dec_k.eu.p0_scales,
                    w_supp=dec_k.eu.w_supp,
                ).fit(q_tr_cal, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
            else:
                eu_k = dec_k.eu

            pe_tr = eu_k.predict(q_tr_cal)
            pe_va = eu_k.predict(Q_gated_oof[va])
            pred_eu_oof[va] = pe_va

            bal_k = CleanProtectedBalancedDecoder(n_bags=24, seed=2026 + k).fit(
                q_tr_cal, pe_tr, gold[tr], groups[tr], is_orig=is_orig[tr], active_dims=TARGET_DIMS
            )
            pred_bal_oof[va] = bal_k.predict(Q_gated_oof[va], pe_va)

        s_eu = summarize_oof_variant(f"{mode}_eu", Q_gated_oof, pred_eu_oof, gold, is_orig)
        s_bal = summarize_oof_variant(f"{mode}_balanced", Q_gated_oof, pred_bal_oof, gold, is_orig)
        summaries.extend([s_eu, s_bal])
        mode_results[mode] = dict(
            active_dims=active_dims,
            summary_eu=s_eu,
            summary_bal=s_bal,
            Q_oof=Q_gated_oof,
            pred_eu_oof=pred_eu_oof,
            pred_bal_oof=pred_bal_oof,
        )

    _print_ablation_table(summaries)

    best_mode = modes_to_run[-1]
    best_obj = -1e9
    base_doc2_mid = 0.5 * (base_sum["doc2_all"] + base_sum["doc2_orig"])
    base_map_mid = 0.5 * (base_sum["os"]["ap_all"]["mean_ap"] + base_sum["og"]["ap_all"]["mean_ap"])

    for mode, mr in mode_results.items():
        se = mr["summary_eu"]
        sb = mr["summary_bal"]
        doc2_mid = 0.5 * (se["doc2_all"] + se["doc2_orig"])
        bal_mid = 0.5 * (sb["doc2_all"] + sb["doc2_orig"])
        map_mid = 0.5 * (se["os"]["ap_all"]["mean_ap"] + se["og"]["ap_all"]["mean_ap"])
        score_m = (
            0.6 * (doc2_mid - base_doc2_mid)
            + 0.4 * (bal_mid - base_doc2_mid)
            + 15.0 * (map_mid - base_map_mid)
            - 20.0 * (se["rps_all"] - base_sum["rps_all"])
            + (0.005 if len(mr["active_dims"]) > 0 else 0.0)
        )
        if doc2_mid >= base_doc2_mid - 0.04 and score_m > best_obj:
            best_obj = score_m
            best_mode = mode

    chosen_active_dims = list(mode_results[best_mode]["active_dims"])
    se_best = mode_results[best_mode]["summary_eu"]
    sb_best = mode_results[best_mode]["summary_bal"]

    full_base_pkl = os.path.join(REPO, "results/r55/final", f"{set_name}_decoder_routed_gate_eu.pkl")
    if os.path.exists(full_base_pkl):
        base_dec_full = pickle.load(open(full_base_pkl, "rb"))["decoder"]
    else:
        base_dec_full = RoutedGateEUDecoder(**proto_cfg).fit(P_dict, gold, is_orig=is_orig, groups=groups)

    Q_full_base = base_dec_full.pool_from_member_dict(P_dict)
    full_cals: Dict[int, LowRankOrdinalCalibrator] = {}

    for d in chosen_active_dims:
        X_full = build_dim_features(Q_full_base, P_dict, d, members, best_mode, spatial_feats=spatial_oof)
        cal_d = LowRankOrdinalCalibrator(dim=d, mode=best_mode).fit(
            X_full, Q_full_base[:, d, :], gold[:, d], groups, is_orig=is_orig
        )
        full_cals[d] = cal_d

    Q_honest_cal = mode_results[best_mode]["Q_oof"]
    if chosen_active_dims and (0.5 * (se_best["doc2_all"] + se_best["doc2_orig"]) >= base_doc2_mid - 0.01):
        eu_full = TrueGateConstrainedEU(
            lams_cl=base_dec_full.lams_cl,
            p0_scales=base_dec_full.eu.p0_scales,
            w_supp=base_dec_full.eu.w_supp,
        ).fit(Q_honest_cal, gold, is_orig=is_orig, groups=groups)
    else:
        eu_full = base_dec_full.eu

    pred_eu_honest = eu_full.predict(Q_honest_cal)
    bal_full = CleanProtectedBalancedDecoder(n_bags=24, seed=2026).fit(
        Q_honest_cal, pred_eu_honest, gold, groups, is_orig=is_orig, active_dims=TARGET_DIMS
    )

    bal_dominates_doc2 = (
        sb_best["doc2_all"] > se_best["doc2_all"] + 0.01
        and sb_best["doc2_orig"] >= se_best["doc2_orig"] - 0.02
        and sb_best["orig_parts"]["clean"] >= 19.65
    )

    bundle = JointOsOgBreakthroughModel(
        set_name=set_name,
        members=members,
        mode=best_mode,
        active_cal_dims=chosen_active_dims,
        base_decoder=base_dec_full,
        calibrators=full_cals,
        eu_decoder=eu_full,
        bal_decoder=bal_full,
    )

    sub_P = {m: v[:16] for m, v in P_dict.items()}
    sub_sp = spatial_oof[:16] if spatial_oof is not None else None
    batch_out = bundle.predict_all(sub_P, spatial_feats=sub_sp)
    for i in range(16):
        p1 = {m: v[i : i + 1] for m, v in sub_P.items()}
        s1 = sub_sp[i : i + 1] if sub_sp is not None else None
        o1 = bundle.predict_all(p1, spatial_feats=s1)
        assert np.array_equal(o1["pred_primary_eu"][0], batch_out["pred_primary_eu"][i]), f"N=1 mismatch on primary_eu row {i}"
        assert np.array_equal(o1["pred_balanced"][0], batch_out["pred_balanced"][i]), f"N=1 mismatch on balanced row {i}"
        assert np.max(np.abs(o1["Q_cal"][0] - batch_out["Q_cal"][i])) < 1e-12, f"N=1 prob drift on row {i}"

    out_abs = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    os.makedirs(out_abs, exist_ok=True)
    pkl_path = os.path.join(out_abs, f"{set_name}_os_og_bundle.pkl")
    json_path = os.path.join(out_abs, f"{set_name}_os_og_report.json")

    report = dict(
        set=set_name,
        members=list(members),
        selected_mode=best_mode,
        active_calibrated_dims=[DIMS_ZH[d] for d in chosen_active_dims],
        balanced_dominates_doc2=bool(bal_dominates_doc2),
        recommended_submission=(
            "pred_nriqa_dataset_test_os_og_breakthrough_balanced.json"
            if bal_dominates_doc2
            else "pred_nriqa_dataset_test_os_og_breakthrough.json"
        ),
        balanced_active_dims={DIMS_ZH[d]: bool(bal_full.active_.get(d, False)) for d in TARGET_DIMS},
        balanced_thresholds={DIMS_ZH[d]: [float(x) for x in bal_full.thresholds_.get(d, ())] for d in TARGET_DIMS},
        balanced_keep_eu_p0={DIMS_ZH[d]: bool(bal_full.keep_eu_p0_.get(d, True)) for d in TARGET_DIMS},
        balanced_tau_clean_shield=float(bal_full.tau_clean_shield_),
        calibrator_alphas={DIMS_ZH[d]: float(full_cals[d].alpha_) for d in chosen_active_dims},
        calibrator_topk={DIMS_ZH[d]: int(full_cals[d].best_topk_) for d in chosen_active_dims},
        calibrator_l2={DIMS_ZH[d]: float(full_cals[d].best_l2_) for d in chosen_active_dims},
        n1_batch_invariance_verified=True,
        elapsed_sec=round(time.time() - t0, 2),
        ablation_summaries=summaries,
    )
    with open(pkl_path, "wb") as f:
        pickle.dump(dict(bundle=bundle, report=report), f)
    # Immediately test reading back the exact written file via standard pickle.load!
    with open(pkl_path, "rb") as f_chk:
        _reloaded = pickle.load(f_chk)
        assert isinstance(_reloaded["bundle"], JointOsOgBreakthroughModel)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(
        f"[deconfound_calibrator] saved & readback-verified bundle -> {pkl_path} (module={type(bundle).__module__})\n"
        f"  selected_mode={best_mode!r}, active_cal_dims={[DIMS_ZH[d] for d in chosen_active_dims]}, "
        f"balanced_active={report['balanced_active_dims']}, tau_clean_shield={bal_full.tau_clean_shield_:.2f}, "
        f"keep_eu_p0={report['balanced_keep_eu_p0']}, recommended={report['recommended_submission']} ({report['elapsed_sec']}s)",
        flush=True,
    )
    return pkl_path, report


def _run_self_test() -> None:
    rng = np.random.RandomState(7)
    N = 120
    members = ["ntg_pe", "ntg_convnext", "pe_v3", "v16"]
    P_dict = {m: rng.dirichlet([3.2, 0.4, 0.25, 0.15], size=(N, 6)) for m in members}
    Q_base = np.mean(list(P_dict.values()), axis=0)
    sp = rng.randn(N, SPATIAL_FEAT_DIM)
    y = rng.choice([0, 1, 2, 3], size=(N, 6), p=[0.70, 0.14, 0.09, 0.07])
    y[:, 3] = np.where(y[:, 3] > 0, 3, 0)
    y[:, 5] = np.where(y[:, 5] > 0, 3, 0)
    groups = np.array([f"vid_{i // 3}" for i in range(N)])
    is_orig = np.arange(N) % 2 == 0

    X_j = build_dim_features(Q_base, P_dict, 2, members, "joint", spatial_feats=sp)
    cal = LowRankOrdinalCalibrator(dim=2, mode="joint").fit(X_j, Q_base[:, 2, :], y[:, 2], groups, is_orig=is_orig)
    q_batch = cal.predict_proba(X_j, Q_base[:, 2, :])
    q_single = np.stack([cal.predict_proba(X_j[i : i + 1], Q_base[i : i + 1, 2, :])[0] for i in range(N)], axis=0)
    assert np.max(np.abs(q_batch - q_single)) < 1e-12
    assert type(cal).__module__ == "deconfound_calibrator", f"Expected module 'deconfound_calibrator', got {type(cal).__module__}"
    blob = pickle.dumps(cal)
    cal2 = pickle.loads(blob)
    assert np.max(np.abs(cal2.predict_proba(X_j, Q_base[:, 2, :]) - q_batch)) == 0.0
    print(f"[deconfound_calibrator][self-test] PASS: module={type(cal).__module__}, alpha={cal.alpha_}, max_n1_diff={np.max(np.abs(q_batch - q_single)):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="resident_v7")
    ap.add_argument("--spatial-oof", default="results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz")
    ap.add_argument("--out-dir", default="results/r55/os_og_breakthrough")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _run_self_test()

    sp_p = args.spatial_oof if os.path.isabs(args.spatial_oof) else os.path.join(REPO, args.spatial_oof)
    run_nested_ablation_and_fit(set_name=args.set, spatial_oof_path=sp_p, out_dir=args.out_dir)


if __name__ == "__main__":
    import deconfound_calibrator as _canonical_dc
    _canonical_dc.main()
