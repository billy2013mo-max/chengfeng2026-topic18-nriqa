#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/ablate_novlm_deep_calibration.py
=========================================
8-fold video-grouped Zero-Leakage Nested OOF laboratory testing 4 low-DOF,
mathematically grounded calibration upgrades on top of `resident_v8_boundary_novlm`
(`deconf_v6_boundary`, baseline ORIG=52.3274 / S1a_keepV5=52.354):

  1. Exact analytic-gradient log-linear blending (`L-BFGS-B` + simplex projection / SLSQP w/ jac)
     fixing SLSQP finite-diff stagnation on `(bndry_pe, bndry_dual)`.
  2. Low-DOF (3-param logit bias + 1-param temp) L2-regularized marginal recalibration
     fixing `bndry_pe/bndry_dual` boundary-weighted class-prior shift.
  3. 2-param / 3-param Clean-probability (`p_cl`) correlation calibrator replacing naive
     independence `prod_d Q_d0` inside `TrueGateConstrainedEU`.
  4. Fine inner-CV `(p0_base, p0_og, p0_os)` + keepV5 blur vs analytic pre-LRO OS/OG.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    _ap = os.path.join(REPO, _p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

import common55 as c  # noqa: E402
import decoder_eu as E  # noqa: E402
import deconfound_calibrator as dc  # noqa: E402
import routed_eu_decoder as red  # noqa: E402
import upgrade_deconf_v3_cleanlocked as v3m  # noqa: E402
import upgrade_deconf_v4_tile as v4m  # noqa: E402
import upgrade_deconf_v5_boundary as v5m  # noqa: E402
import upgrade_deconf_v6_boundary as v6m  # noqa: E402
import upgrade_os_og_bokeh_deconf as uob  # noqa: E402
from eval_specialist_gate import load_oof_dir  # noqa: E402
from final_engine import SETS, oof_pool  # noqa: E402

LAMS_CL_GRID = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)

_G: Dict = {}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_loglin_exact_dim(
    q_base_d: np.ndarray,
    extra_cals_d: Sequence[np.ndarray],
    y_d: np.ndarray,
    w_sample: np.ndarray,
    l2_w: float = 0.002,
    fit_bias: bool = False,
    l2_b: float = 0.01,
    max_each: float = 0.45,
    max_sum: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact analytic-gradient RPS minimizer for log-linear blend weights (+ optional 4-class logit bias)."""
    M = len(extra_cals_d)
    wn = (w_sample / np.mean(w_sample))[:, None]  # (N, 1)
    N = len(y_d)
    yd = np.cumsum(np.eye(4)[y_d], axis=-1)[:, :3]  # (N, 3)
    l0 = np.log(np.clip(q_base_d, 1e-7, 1.0))
    ls = np.stack([np.log(np.clip(cm, 1e-7, 1.0)) for cm in extra_cals_d], axis=0)  # (M, N, 4)
    dl = ls - l0[None, :, :]  # (M, N, 4)

    # Upper-triangular matrix U where cum_p[:, :3] = p @ U  (4 x 3)
    U = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.float64)

    def obj_and_grad(params: np.ndarray):
        w = params[:M]
        b = params[M:] if fit_bias else np.zeros(4, dtype=np.float64)
        z = l0 + np.tensordot(w, dl, axes=(0, 0)) + b[None, :]
        p = _softmax(z)
        pc = p @ U  # (N, 3)
        diff = pc - yd  # (N, 3)
        loss = float(np.mean(wn * (diff ** 2))) + l2_w * float(np.sum(w ** 2))
        if fit_bias:
            loss += l2_b * float(np.sum(b ** 2))

        # d(loss)/d(pc) = (2 / (N * 3)) * wn * diff
        g_pc = (2.0 / (N * 3.0)) * wn * diff  # (N, 3)
        g_p = g_pc @ U.T  # (N, 4)
        # Softmax Jacobian: g_z = p * (g_p - sum(p * g_p))
        dot = np.sum(p * g_p, axis=-1, keepdims=True)
        g_z = p * (g_p - dot)  # (N, 4)
        g_w = np.tensordot(dl, g_z, axes=([1, 2], [0, 1])) + 2.0 * l2_w * w
        if fit_bias:
            g_b = np.sum(g_z, axis=0) + 2.0 * l2_b * b
            return loss, np.concatenate([g_w, g_b])
        return loss, g_w

    p0 = np.concatenate([np.full(M, 0.08), np.zeros(4)]) if fit_bias else np.full(M, 0.08)
    bounds = [(0.0, max_each)] * M + ([(-1.0, 1.0)] * 4 if fit_bias else [])
    cons = {"type": "ineq", "fun": lambda x: max_sum - float(np.sum(x[:M]))}
    res = minimize(obj_and_grad, p0, method="SLSQP", jac=True, bounds=bounds, constraints=cons, options={"maxiter": 200, "ftol": 1e-11})
    w_opt = np.clip(res.x[:M], 0.0, max_each)
    b_opt = res.x[M:] - np.mean(res.x[M:]) if fit_bias else np.zeros(4, dtype=np.float64)
    return w_opt, b_opt


def apply_loglin_bias_dim(q_base_d: np.ndarray, extra_cals_d: Sequence[np.ndarray], w: np.ndarray, b: np.ndarray) -> np.ndarray:
    ws = float(np.sum(w))
    z = (1.0 - ws) * np.log(np.clip(q_base_d, 1e-7, 1.0)) + b[None, :]
    for wm, cm in zip(w, extra_cals_d):
        z = z + float(wm) * np.log(np.clip(cm, 1e-7, 1.0))
    return _softmax(z)


def fit_temp_bias_dim(q_d: np.ndarray, y_d: np.ndarray, w_sample: np.ndarray, l2_b: float = 0.02, l2_t: float = 0.02) -> Tuple[float, np.ndarray]:
    """Low-DOF (1 temp + 3 zero-sum logit biases) L2-shrunk RPS recalibrator."""
    wn = (w_sample / np.mean(w_sample))[:, None]
    N = len(y_d)
    yd = np.cumsum(np.eye(4)[y_d], axis=-1)[:, :3]
    lq = np.log(np.clip(q_d, 1e-7, 1.0))
    U = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.float64)

    def obj_and_grad(params: np.ndarray):
        inv_t = params[0]
        b = params[1:]
        z = inv_t * lq + b[None, :]
        p = _softmax(z)
        diff = p @ U - yd
        loss = float(np.mean(wn * (diff ** 2))) + l2_t * ((inv_t - 1.0) ** 2) + l2_b * float(np.sum(b ** 2))
        g_p = ((2.0 / (N * 3.0)) * wn * diff) @ U.T
        g_z = p * (g_p - np.sum(p * g_p, axis=-1, keepdims=True))
        g_invt = float(np.sum(g_z * lq)) + 2.0 * l2_t * (inv_t - 1.0)
        g_b = np.sum(g_z, axis=0) + 2.0 * l2_b * b
        return loss, np.concatenate([[g_invt], g_b])

    res = minimize(
        obj_and_grad,
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.5, 2.0)] + [(-0.8, 0.8)] * 4,
    )
    inv_t = float(res.x[0])
    b = res.x[1:] - np.mean(res.x[1:])
    return inv_t, b


def apply_temp_bias_dim(q_d: np.ndarray, inv_t: float, b: np.ndarray) -> np.ndarray:
    return _softmax(float(inv_t) * np.log(np.clip(q_d, 1e-7, 1.0)) + b[None, :])


def fit_clean_calibrator(Q_tr_orig: np.ndarray, y_clean_orig: np.ndarray) -> np.ndarray:
    """Fits 3-param logistic correlation calibrator for P(Clean | Q) strictly on orig training fold."""
    lq0 = np.log(np.clip(Q_tr_orig[:, :, 0], 1e-6, 1.0))
    f_sum = np.sum(lq0, axis=1)
    f_min = np.min(lq0, axis=1)
    X = np.column_stack([f_sum, f_min, np.ones_like(f_sum)])
    y = y_clean_orig.astype(np.float64)

    def obj(th):
        z = np.clip(X @ th, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        nll = -np.mean(y * np.log(np.clip(p, 1e-7, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-7, 1.0)))
        reg = 0.01 * ((th[0] - 1.0) ** 2 + th[1] ** 2 + 0.1 * th[2] ** 2)
        g = (X.T @ (p - y)) / len(y) + np.array([0.02 * (th[0] - 1.0), 0.02 * th[1], 0.002 * th[2]])
        return float(nll + reg), g

    res = minimize(obj, np.array([1.0, 0.0, 0.0]), method="L-BFGS-B", jac=True)
    return res.x


def predict_p_clean(Q: np.ndarray, th: np.ndarray) -> np.ndarray:
    lq0 = np.log(np.clip(Q[:, :, 0], 1e-6, 1.0))
    z = np.clip(th[0] * np.sum(lq0, axis=1) + th[1] * np.min(lq0, axis=1) + th[2], -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_eu_calclean(Q: np.ndarray, st: dict, lam_cl: float, p0_base: float, p0_og: float, cl_th: np.ndarray | None = None) -> np.ndarray:
    Q = c.renorm(Q)
    D = np.maximum(1.0, st["npred"] + st["npos"])
    p = Q[:, E.LD, E.LG]
    p0_sc_vec = np.where(E.LD == 4, float(p0_og), float(p0_base))
    w_f = np.where(E.LG == 3, 25.0 / 14.0 + p0_sc_vec * (20.0 * 0.3 / 6.0), 25.0 / 14.0)
    w_r = p0_sc_vec * E.W_R
    w_p = p0_sc_vec * E.W_P
    gain = w_f * (2.0 * p - st["F"]) / D + w_r * p / np.maximum(1.0, st["npos"]) + w_p * (p - st["P"]) / np.maximum(1.0, st["npred"])

    pred = np.zeros((len(Q), 6), dtype=np.int64)
    best_d = np.zeros((len(Q), 6), dtype=np.float64)
    best_nonp0_gain = np.full(len(Q), -1e9, dtype=np.float64)
    best_nonp0_d = np.zeros(len(Q), dtype=np.int64)
    best_nonp0_g = np.zeros(len(Q), dtype=np.int64)
    for l, (d, g) in enumerate(E.LABELS):
        gl = gain[:, l]
        m = gl > best_d[:, d]
        pred[m, d] = g
        best_d[m, d] = gl[m]
        if g in (1, 2):
            m_np0 = gl > best_nonp0_gain
            best_nonp0_gain[m_np0] = gl[m_np0]
            best_nonp0_d[m_np0] = d
            best_nonp0_g[m_np0] = g

    pos_gain_sum = best_d.sum(axis=1)
    has_pos = (pred > 0).any(axis=1)
    p_cl = predict_p_clean(Q, cl_th) if cl_th is not None else np.prod(Q[:, :, 0], axis=1)
    ndef = max(1.0, st["N"] - st["ncl"])
    ncl = max(1.0, st["ncl"])
    clean_diff = 10.0 * (1.0 - p_cl) / ndef - lam_cl * 10.0 * p_cl / ncl

    pred[has_pos & ((pos_gain_sum + clean_diff) < 0.0)] = 0
    promote = (~has_pos) & ((best_nonp0_gain + clean_diff) > 0.0)
    if promote.any():
        idx_p = np.where(promote)[0]
        pred[idx_p, best_nonp0_d[idx_p]] = best_nonp0_g[idx_p]
    return pred


def fit_st_calclean(Q: np.ndarray, gold: np.ndarray, rw: np.ndarray, lam_cl: float, p0_base: float, p0_og: float, cl_th: np.ndarray | None = None) -> dict:
    pr = E.Decoder55().predict(Q)
    st = {}
    for _ in range(15):
        pl = (pr[:, E.LD] == E.LG[None]).astype(np.float64)
        gl = (gold[:, E.LD] == E.LG[None]).astype(np.float64)
        tp = (rw[:, None] * pl * gl).sum(0)
        npred = (rw[:, None] * pl).sum(0)
        npos = (rw[:, None] * gl).sum(0)
        ncl = float((rw * (gold == 0).all(1).astype(np.float64)).sum())
        st = dict(
            F=2.0 * tp / np.maximum(1e-6, npred + npos),
            P=tp / np.maximum(1e-6, npred),
            npos=npos,
            npred=npred,
            ncl=max(1.0, ncl),
            N=float(rw.sum()),
        )
        nw = predict_eu_calclean(Q, st, lam_cl, p0_base, p0_og, cl_th=cl_th)
        if (nw == pr).all():
            break
        pr = nw
    return st


def select_lam_calclean_cv(Q_tr: np.ndarray, gold_tr: np.ndarray, groups_tr: np.ndarray, is_orig_tr: np.ndarray, cl_th: np.ndarray) -> Tuple[float, float, dict]:
    rw = np.where(is_orig_tr, 1.0, 0.25)
    # Match TrueGateConstrainedEU bootstrap gate on orig training fold
    base_d55 = E.Decoder55().predict(Q_tr[is_orig_tr])
    max_cp0 = max(1, int(((gold_tr[is_orig_tr] == 0).all(axis=1) & (base_d55 == 3).any(axis=1)).sum()))
    best = None
    for lam in (0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25):
        for p0b in (0.92, 1.00):
            st = fit_st_calclean(Q_tr, gold_tr, rw, lam, p0b, p0b, cl_th=cl_th)
            pr = predict_eu_calclean(Q_tr, st, lam, p0b, p0b, cl_th=cl_th)
            pr_o, g_o, grp_o = pr[is_orig_tr], gold_tr[is_orig_tr], groups_tr[is_orig_tr]
            pf, _, _ = red._fast_boot654_pos_risk(pr_o, g_o, grp_o, n_boot=4000)
            cp0 = int(((g_o == 0).all(axis=1) & (pr_o == 3).any(axis=1)).sum())
            sc_o = float(c.score(pr_o, g_o)["doc2_strict_total"])
            ok = (pf <= 1.0) and (cp0 <= max_cp0)
            key = (1 if ok else 0, sc_o)
            if best is None or key > best[0]:
                best = (key, lam, p0b, st)
    return float(best[1]), float(best[2]), best[3]


def _fold_worker(k: int) -> dict:
    fold_arr = _G["fold"]
    gold = _G["gold"]
    is_orig = _G["is_orig"]
    groups = _G["groups"]
    P_v7 = _G["P"]
    P_ex = _G["P_ex"]
    S_all = _G["S"]
    T_all = _G["T"]
    members_v7 = _G["members"]
    nobk_members = _G["nobk_members"]
    proto_cfg = _G["proto_cfg"]

    va = fold_arr == k
    tr = ~va
    P_tr = {m: v[tr] for m, v in P_v7.items()}
    P_va = {m: v[va] for m, v in P_v7.items()}
    P_nb_tr = {m: P_tr[m] for m in nobk_members}
    P_nb_va = {m: P_va[m] for m in nobk_members}
    rw = np.where(is_orig[tr], 1.0, 0.25)

    # ---- Base v2 -> v4 -> v5 ----
    dec_k = red.RoutedGateEUDecoder(**proto_cfg).fit(P_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    q_tr_base = dec_k.pool_from_member_dict(P_tr)
    q_va_base = dec_k.pool_from_member_dict(P_va)

    X_og2_tr = dc.extract_deconf_covariates(q_tr_base, P_tr, 4, members_v7)
    X_og2_va = dc.extract_deconf_covariates(q_va_base, P_va, 4, members_v7)
    cal_og2 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf").fit(X_og2_tr, q_tr_base[:, 4], gold[tr, 4], groups[tr], is_orig=is_orig[tr])
    X_os_tr = uob.extract_os_micro_covariates(q_tr_base, P_tr, members_v7)
    X_os_va = uob.extract_os_micro_covariates(q_va_base, P_va, members_v7)
    cal_os = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=(2, 4, 8, 11)).fit(X_os_tr, q_tr_base[:, 2], gold[tr, 2], groups[tr], is_orig=is_orig[tr])
    q_tr_os = cal_os.predict_proba(X_os_tr, q_tr_base[:, 2])
    q_va_os = cal_os.predict_proba(X_os_va, q_va_base[:, 2])

    q_tr_nobk_base = uob.pool_without_bkseg1(dec_k, P_tr)
    q_va_nobk_base = uob.pool_without_bkseg1(dec_k, P_va)
    cal_bk2 = uob.BokehSynergyCalibrator(w_supp=0.20).fit(q_tr_nobk_base[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5], gold[tr, 5], groups[tr], is_orig=is_orig[tr])

    qv2_tr = q_tr_base.copy()
    qv2_tr[:, 2], qv2_tr[:, 4], qv2_tr[:, 5] = q_tr_os, cal_og2.predict_proba(X_og2_tr, q_tr_base[:, 4]), cal_bk2.predict_proba(q_tr_nobk_base[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    qv2_tr = c.renorm(qv2_tr)
    qv2_va = q_va_base.copy()
    qv2_va[:, 2], qv2_va[:, 4], qv2_va[:, 5] = q_va_os, cal_og2.predict_proba(X_og2_va, q_va_base[:, 4]), cal_bk2.predict_proba(q_va_nobk_base[:, 5], c.renorm(P_va["pe_bkseg1"])[:, 5])
    qv2_va = c.renorm(qv2_va)

    eu_v2 = red.TrueGateConstrainedEU(lams_cl=dec_k.lams_cl, p0_scales=dec_k.eu.p0_scales, w_supp=dec_k.eu.w_supp).fit(
        qv2_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr]
    )
    pr_v2_va = eu_v2.predict(qv2_va)
    lam_cl, p0_base = float(eu_v2.lam_cl), float(eu_v2.p0_scale)

    # v4
    cal_bk3 = v3m.BokehSynergyV3(w_supp=0.20).fit(q_tr_nobk_base[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5], gold[tr, 5], groups[tr], is_orig=is_orig[tr])
    X24_tr = v4m.extract_og24_tile_covs(q_tr_base, P_tr, members_v7, S_all[tr], T_all[tr])
    X24_va = v4m.extract_og24_tile_covs(q_va_base, P_va, members_v7, S_all[va], T_all[va])
    cal_og24 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(X24_tr, q_tr_base[:, 4], gold[tr, 4], groups[tr], is_orig=is_orig[tr])
    qv4_tr = q_tr_base.copy()
    qv4_tr[:, 2], qv4_tr[:, 4], qv4_tr[:, 5] = q_tr_os, cal_og24.predict_proba(X24_tr, q_tr_base[:, 4]), cal_bk3.predict_proba(q_tr_nobk_base[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    qv4_tr = c.renorm(qv4_tr)
    qv4_va = q_va_base.copy()
    qv4_va[:, 2], qv4_va[:, 4], qv4_va[:, 5] = q_va_os, cal_og24.predict_proba(X24_va, q_va_base[:, 4]), cal_bk3.predict_proba(q_va_nobk_base[:, 5], c.renorm(P_va["pe_bkseg1"])[:, 5])
    qv4_va = c.renorm(qv4_va)
    p0_v4, st_v4 = v3m.select_p0_og_inner_cv(qv4_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=lam_cl, p0_base=p0_base)
    pr_v4_va = v3m.predict_og_p0_eu(qv4_va, st_v4, lam_cl=lam_cl, p0_base=p0_base, p0_og=p0_v4)
    pr_v4_va[:, [0, 1, 3]] = pr_v2_va[:, [0, 1, 3]]
    cl_v2 = (pr_v2_va == 0).all(axis=1)
    pr_v4_va[cl_v2] = 0
    nc4 = (~cl_v2) & (pr_v4_va == 0).all(axis=1)
    if np.any(nc4):
        pr_v4_va[nc4] = pr_v2_va[nc4]

    # v5
    nb_cfg = dc._build_routed_gate_cfg(nobk_members)
    nb_cfg["lams_cl"] = LAMS_CL_GRID
    dec_nb = red.RoutedGateEUDecoder(**nb_cfg).fit(P_nb_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    qnb_tr = dec_nb.pool_from_member_dict(P_nb_tr)
    qnb_va = dec_nb.pool_from_member_dict(P_nb_va)
    Xbl_tr = v5m.extract_blur33_tile_covs(qv4_tr, P_tr, nobk_members, T_all[tr])
    Xbl_va = v5m.extract_blur33_tile_covs(qv4_va, P_va, nobk_members, T_all[va])
    cal_bl = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 14, 24, Xbl_tr.shape[1])).fit(Xbl_tr, qv4_tr[:, 0], gold[tr, 0], groups[tr], is_orig=is_orig[tr])
    qv5_tr = qv4_tr.copy(); qv5_tr[:, 0], qv5_tr[:, 1] = cal_bl.predict_proba(Xbl_tr, qv4_tr[:, 0]), qnb_tr[:, 1]; qv5_tr = c.renorm(qv5_tr)
    qv5_va = qv4_va.copy(); qv5_va[:, 0], qv5_va[:, 1] = cal_bl.predict_proba(Xbl_va, qv4_va[:, 0]), qnb_va[:, 1]; qv5_va = c.renorm(qv5_va)
    p0_v5, st_v5 = v3m.select_p0_og_inner_cv(qv5_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=lam_cl, p0_base=p0_base)
    pr_v5_va = v3m.predict_og_p0_eu(qv5_va, st_v5, lam_cl=lam_cl, p0_base=p0_base, p0_og=p0_v5)
    pr_v5_va[:, [2, 3, 4, 5]] = pr_v4_va[:, [2, 3, 4, 5]]
    cl_v4 = (pr_v4_va == 0).all(axis=1)
    pr_v5_va[cl_v4] = 0
    nc5 = (~cl_v4) & (pr_v5_va == 0).all(axis=1)
    if np.any(nc5):
        pr_v5_va[nc5] = pr_v4_va[nc5]

    # Temperature calibrate extra models
    cal_tr, cal_va = {}, {}
    for nm, arr in P_ex.items():
        tk = v6m.fit_scalar_temp_weighted(arr[tr], gold[tr], rw)
        cal_tr[nm] = v6m.apply_scalar_temp(arr[tr], tk)
        cal_va[nm] = v6m.apply_scalar_temp(arr[va], tk)

    ex_keys = ("bndry_pe", "bndry_dual")

    # ---- Exact analytic-gradient OG pre-LRO (dim=4) ----
    w_og, b_og = fit_loglin_exact_dim(q_tr_base[:, 4], [cal_tr[nm][:, 4] for nm in ex_keys], gold[tr, 4], rw, l2_w=0.005, fit_bias=False)
    og0_tr = apply_loglin_bias_dim(q_tr_base[:, 4], [cal_tr[nm][:, 4] for nm in ex_keys], w_og, b_og)
    og0_va = apply_loglin_bias_dim(q_va_base[:, 4], [cal_va[nm][:, 4] for nm in ex_keys], w_og, b_og)
    qb_og_tr = q_tr_base.copy(); qb_og_tr[:, 4] = og0_tr
    qb_og_va = q_va_base.copy(); qb_og_va[:, 4] = og0_va
    c_og = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        v4m.extract_og24_tile_covs(qb_og_tr, P_tr, members_v7, S_all[tr], T_all[tr]), og0_tr, gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    og_tr = c_og.predict_proba(v4m.extract_og24_tile_covs(qb_og_tr, P_tr, members_v7, S_all[tr], T_all[tr]), og0_tr)
    og_va = c_og.predict_proba(v4m.extract_og24_tile_covs(qb_og_va, P_va, members_v7, S_all[va], T_all[va]), og0_va)

    # ---- Exact analytic-gradient OS pre-LRO (dim=2) w/ and w/o bias ----
    w_os, b_os = fit_loglin_exact_dim(q_tr_base[:, 2], [cal_tr[nm][:, 2] for nm in ex_keys], gold[tr, 2], rw, l2_w=0.005, fit_bias=False)
    os0_tr = apply_loglin_bias_dim(q_tr_base[:, 2], [cal_tr[nm][:, 2] for nm in ex_keys], w_os, b_os)
    os0_va = apply_loglin_bias_dim(q_va_base[:, 2], [cal_va[nm][:, 2] for nm in ex_keys], w_os, b_os)
    qb_os_tr = q_tr_base.copy(); qb_os_tr[:, 2] = os0_tr
    qb_os_va = q_va_base.copy(); qb_os_va[:, 2] = os0_va
    c_os = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=(2, 4, 8, 11)).fit(
        uob.extract_os_micro_covariates(qb_os_tr, P_tr, members_v7), os0_tr, gold[tr, 2], groups[tr], is_orig=is_orig[tr]
    )
    os_tr = c_os.predict_proba(uob.extract_os_micro_covariates(qb_os_tr, P_tr, members_v7), os0_tr)
    os_va = c_os.predict_proba(uob.extract_os_micro_covariates(qb_os_va, P_va, members_v7), os0_va)

    # Also test analytic-gradient OS w/ bias directly after LRO
    w_osb, b_osb = fit_loglin_exact_dim(qv5_tr[:, 2], [cal_tr[nm][:, 2] for nm in ex_keys], gold[tr, 2], rw, l2_w=0.005, fit_bias=True, l2_b=0.01)
    osb_tr = apply_loglin_bias_dim(qv5_tr[:, 2], [cal_tr[nm][:, 2] for nm in ex_keys], w_osb, b_osb)
    osb_va = apply_loglin_bias_dim(qv5_va[:, 2], [cal_va[nm][:, 2] for nm in ex_keys], w_osb, b_osb)

    # Also test analytic-gradient blur w/ bias after LRO
    w_blb, b_blb = fit_loglin_exact_dim(qv5_tr[:, 0], [cal_tr[nm][:, 0] for nm in ex_keys], gold[tr, 0], rw, l2_w=0.005, fit_bias=True, l2_b=0.01)
    blb_tr = apply_loglin_bias_dim(qv5_tr[:, 0], [cal_tr[nm][:, 0] for nm in ex_keys], w_blb, b_blb)
    blb_va = apply_loglin_bias_dim(qv5_va[:, 0], [cal_va[nm][:, 0] for nm in ex_keys], w_blb, b_blb)

    # Helper to apply clean lock against pr_v5_va
    def _dec_lock(Qtr, Qva, unlock):
        p0o, st = v3m.select_p0_og_inner_cv(Qtr, gold[tr], groups[tr], is_orig[tr], lam_cl=lam_cl, p0_base=p0_base)
        pr = v3m.predict_og_p0_eu(Qva, st, lam_cl=lam_cl, p0_base=p0_base, p0_og=p0o)
        ld = [d for d in range(6) if d not in unlock]
        if ld:
            pr[:, ld] = pr_v5_va[:, ld]
        cl = (pr_v5_va == 0).all(axis=1)
        pr[cl] = 0
        nc = (~cl) & (pr == 0).all(axis=1)
        if np.any(nc):
            pr[nc] = pr_v5_va[nc]
        return pr

    out = {"k": k}

    # D0: S1a_keepV5_blur + exact analytic OG pre-LRO
    qd0_tr = qv5_tr.copy(); qd0_tr[:, 4] = og_tr; qd0_tr = c.renorm(qd0_tr)
    qd0_va = qv5_va.copy(); qd0_va[:, 4] = og_va; qd0_va = c.renorm(qd0_va)
    out["D0_keepV5blur_exactOG"] = (qd0_va, _dec_lock(qd0_tr, qd0_va, unlock=(4,)))

    # D1: D0 + exact analytic OS pre-LRO (fix stuck-at-0.06 SLSQP!)
    qd1_tr = qd0_tr.copy(); qd1_tr[:, 2] = os_tr; qd1_tr = c.renorm(qd1_tr)
    qd1_va = qd0_va.copy(); qd1_va[:, 2] = os_va; qd1_va = c.renorm(qd1_va)
    out["D1_exact_OSpre_OGpre"] = (qd1_va, _dec_lock(qd1_tr, qd1_va, unlock=(2, 4)))

    # D2: D0 + exact analytic OS post-LRO w/ bias
    qd2_tr = qd0_tr.copy(); qd2_tr[:, 2] = osb_tr; qd2_tr = c.renorm(qd2_tr)
    qd2_va = qd0_va.copy(); qd2_va[:, 2] = osb_va; qd2_va = c.renorm(qd2_va)
    out["D2_exact_OSpostBias_OGpre"] = (qd2_va, _dec_lock(qd2_tr, qd2_va, unlock=(2, 4)))

    # D3: D2 + exact analytic Blur post-LRO w/ bias
    qd3_tr = qd2_tr.copy(); qd3_tr[:, 0] = blb_tr; qd3_tr = c.renorm(qd3_tr)
    qd3_va = qd2_va.copy(); qd3_va[:, 0] = blb_va; qd3_va = c.renorm(qd3_va)
    out["D3_exact_BL_OS_OG_bias"] = (qd3_va, _dec_lock(qd3_tr, qd3_va, unlock=(0, 2, 4)))

    # D4: D1 + Temp+Bias post-recalibration across all 6 dims (low-DOF 4 params/dim)
    qd4_tr = qd1_tr.copy()
    qd4_va = qd1_va.copy()
    for d in range(6):
        it_d, b_d = fit_temp_bias_dim(qd1_tr[:, d], gold[tr, d], rw, l2_b=0.02, l2_t=0.02)
        qd4_tr[:, d] = apply_temp_bias_dim(qd1_tr[:, d], it_d, b_d)
        qd4_va[:, d] = apply_temp_bias_dim(qd1_va[:, d], it_d, b_d)
    qd4_tr, qd4_va = c.renorm(qd4_tr), c.renorm(qd4_va)
    out["D4_D1_tempBias6d_lock"] = (qd4_va, _dec_lock(qd4_tr, qd4_va, unlock=(0, 1, 2, 3, 4, 5)))

    # D5: D0 + calibrated p_cl correlation model (3-param Platt on sum_log_q0 + min_log_q0)
    cl_th0 = fit_clean_calibrator(qd0_tr[is_orig[tr]], (gold[tr][is_orig[tr]] == 0).all(axis=1))
    lam_c0, p0b_c0, _ = select_lam_calclean_cv(qd0_tr, gold[tr], groups[tr], is_orig[tr], cl_th0)
    st_c0 = fit_st_calclean(qd0_tr, gold[tr], rw, lam_c0, p0b_c0, p0b_c0, cl_th=cl_th0)
    pr_c0 = predict_eu_calclean(qd0_va, st_c0, lam_c0, p0b_c0, p0b_c0, cl_th=cl_th0)
    out["D5_D0_calClean_EU"] = (qd0_va, pr_c0)

    # D6: D1 + calibrated p_cl correlation model (ALL dims including OS+OG + calibrated Clean!)
    cl_th1 = fit_clean_calibrator(qd1_tr[is_orig[tr]], (gold[tr][is_orig[tr]] == 0).all(axis=1))
    lam_c1, p0b_c1, _ = select_lam_calclean_cv(qd1_tr, gold[tr], groups[tr], is_orig[tr], cl_th1)
    st_c1 = fit_st_calclean(qd1_tr, gold[tr], rw, lam_c1, p0b_c1, p0b_c1, cl_th=cl_th1)
    pr_c1 = predict_eu_calclean(qd1_va, st_c1, lam_c1, p0b_c1, p0b_c1, cl_th=cl_th1)
    out["D6_D1_calClean_EU"] = (qd1_va, pr_c1)

    # D7: D4 (D1 + TempBias6d) + calibrated p_cl correlation model
    cl_th4 = fit_clean_calibrator(qd4_tr[is_orig[tr]], (gold[tr][is_orig[tr]] == 0).all(axis=1))
    lam_c4, p0b_c4, _ = select_lam_calclean_cv(qd4_tr, gold[tr], groups[tr], is_orig[tr], cl_th4)
    st_c4 = fit_st_calclean(qd4_tr, gold[tr], rw, lam_c4, p0b_c4, p0b_c4, cl_th=cl_th4)
    pr_c4 = predict_eu_calclean(qd4_va, st_c4, lam_c4, p0b_c4, p0b_c4, cl_th=cl_th4)
    out["D7_D4_calClean_EU"] = (qd4_va, pr_c4)

    out["weights"] = {
        "w_og": np.round(w_og, 4).tolist(),
        "w_os": np.round(w_os, 4).tolist(),
        "w_osb": np.round(w_osb, 4).tolist(),
        "w_blb": np.round(w_blb, 4).tolist(),
        "lam_c1": lam_c1,
    }
    return out


def main() -> None:
    t0 = time.time()
    m10 = list(SETS["resident_v7_novlm"])
    nb9 = [m for m in m10 if m != "pe_bkseg1"]

    names, gold, fold, _, P10 = oof_pool(m10)
    _, _, _, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)

    P_ex = {}
    for k_ex, d_rel in [("bndry_pe", "results/r55/boundary_v2/v2_pe_ntg"), ("bndry_dual", "results/r55/boundary_v2/v2_dual_pe_cradio")]:
        arr, valid = load_oof_dir(d_rel, names)
        assert valid.all()
        P_ex[k_ex] = c.renorm(arr)

    S_oof = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz"), names, "feats")
    T_oof = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_oof_3573.npz"), names, "stats")

    proto_cfg = dc._build_routed_gate_cfg(m10)
    proto_cfg["lams_cl"] = LAMS_CL_GRID
    _G.update(fold=fold, gold=gold, is_orig=is_orig, groups=groups, P=P10, P_ex=P_ex, S=S_oof, T=T_oof, members=m10, nobk_members=nb9, proto_cfg=proto_cfg)

    print(f"=== Deep Calibration 8-Fold Zero-Leakage Nested OOF Lab (N={N}) ===", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        f_res = sorted(pool.map(_fold_worker, list(range(8))), key=lambda d: d["k"])

    print("\n--- Reference ---")
    print("  [Ref] Full v8_boundary (+2 VLM)    | ORIG=52.665 | Doc1_o=52.440 | ALL=51.959 | RPS_o=0.11221")
    print("  [Ref] No-VLM deconf_v6 (54.52)     | ORIG=52.327 | Doc1_o=52.202 | ALL=51.864 | RPS_o=0.11344\n")

    keys = [k for k in f_res[0].keys() if k not in ("k", "weights")]
    results = {}
    for nm in keys:
        Q_oof = np.zeros((N, 6, 4))
        pr_oof = np.zeros((N, 6), dtype=np.int64)
        for r in f_res:
            va = fold == r["k"]
            Q_oof[va], pr_oof[va] = r[nm]
        rep = v5m._summarize_all4(nm, Q_oof, pr_oof, gold, is_orig)
        cp0 = int(((gold[is_orig] == 0).all(axis=1) & (pr_oof[is_orig] == 3).any(axis=1)).sum())
        pf, _, _ = red._fast_boot654_pos_risk(pr_oof[is_orig], gold[is_orig], groups[is_orig], n_boot=5000)
        print(f"      Clean-P0={cp0} | Boot654(Pos<15)={pf:.2f}%", flush=True)
        rep["clean_p0"] = cp0
        rep["boot654_pos_risk"] = round(pf, 3)
        results[nm] = rep

    print("\nMean exact analytic-gradient blend weights (bndry_pe, bndry_dual):")
    for wk in ("w_og", "w_os", "w_osb", "w_blb"):
        arr_w = np.mean([r["weights"][wk] for r in f_res], axis=0)
        print(f"  {wk:8s}: {np.round(arr_w, 4).tolist()}")
    print("  fold lam_c1:", [r["weights"]["lam_c1"] for r in f_res])

    out_p = os.path.join(REPO, "results/r55/final/novlm_deep_calibration_oof.json")
    json.dump(results, open(out_p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved -> {out_p} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
