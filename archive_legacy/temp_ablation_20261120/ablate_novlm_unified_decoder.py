#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/ablate_novlm_unified_decoder.py
========================================
Pure-CPU 8-fold video-grouped Zero-Leakage Nested OOF ablation (N = 3,573, ORIG = 1,902)
to discover the genuinely offline-optimal unified decoder for the 12 pure-vision members
(`resident_v8_boundary_novlm` = exact `resident_v8_boundary` minus `p1_qy` and `qwen_hr`).

Strictly zero test-set contact; evaluates all 4 metrics (`doc2_orig`, `doc1_orig`,
`doc2_all`, `rps_orig`) plus Clean-P0 count & per-dim F1/RPS across clean, low-DOF,
strongly-regularized architectural candidates.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

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
import deconfound_calibrator as dc  # noqa: E402
import routed_eu_decoder as red  # noqa: E402
import upgrade_deconf_v3_cleanlocked as v3m  # noqa: E402
import upgrade_deconf_v4_tile as v4m  # noqa: E402
import upgrade_deconf_v5_boundary as v5m  # noqa: E402
import upgrade_os_og_bokeh_deconf as uob  # noqa: E402
from eval_specialist_gate import load_oof_dir  # noqa: E402
from final_engine import SETS, oof_pool  # noqa: E402

LAMS_CL_GRID = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)

_G: Dict = {}


def weighted_rps_dim(q_d: np.ndarray, y_d: np.ndarray, w: np.ndarray) -> float:
    oh = np.cumsum(np.eye(4)[y_d], axis=-1)[:, :3]
    pc = np.cumsum(q_d, axis=-1)[:, :3]
    return float(np.average(((pc - oh) ** 2).mean(axis=-1), weights=w))


def fit_routed_pool_v2(
    P_list_tr: List[np.ndarray],
    Y_tr: np.ndarray,
    prior_w: np.ndarray,
    row_w: np.ndarray,
    per_dim_temp: bool = False,
    pool_mode: str = "linear",   # "linear" | "loglin" | "hybrid50"
    l2_reg: float = 0.015,
    t_max: float = 3.50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits temperature matrix `T` of shape `(M, 6)` and per-defect simplex weights `W[6, M]`
    minimizing weighted RPS + L2 shrinkage toward `prior_w`.
    """
    M = len(P_list_tr)
    rw = row_w / np.mean(row_w)
    t_grid = np.linspace(0.55, t_max, 30)
    oh_all = np.cumsum(np.eye(4)[Y_tr], axis=-1)[..., :3]  # (N, 6, 3)

    T_mat = np.ones((M, 6), dtype=np.float64)
    cal_tr = []
    for m in range(M):
        lp = np.log(np.clip(P_list_tr[m], 1e-7, 1.0))  # (N, 6, 4)
        # First fit scalar global T_m
        best_t, best_r = 1.0, 1e9
        for t in t_grid:
            z = lp / t
            z -= z.max(axis=-1, keepdims=True)
            q = np.exp(z)
            q /= q.sum(axis=-1, keepdims=True)
            pc = np.cumsum(q, axis=-1)[..., :3]
            r = float(np.mean(rw[:, None, None] * ((pc - oh_all) ** 2)))
            if r < best_r:
                best_r, best_t = r, float(t)
        if not per_dim_temp:
            T_mat[m, :] = best_t
        else:
            # Mildly regularized per-dim T around global best_t
            for d in range(6):
                lpd = lp[:, d, :]
                ohd = oh_all[:, d, :]
                bt_d, br_d = best_t, 1e9
                for t in t_grid:
                    z = lpd / t
                    z -= z.max(axis=-1, keepdims=True)
                    q = np.exp(z)
                    q /= q.sum(axis=-1, keepdims=True)
                    pc = np.cumsum(q, axis=-1)[:, :3]
                    r = float(np.mean(rw[:, None] * ((pc - ohd) ** 2))) + 0.0005 * ((t - best_t) ** 2)
                    if r < br_d:
                        br_d, bt_d = r, float(t)
                T_mat[m, d] = bt_d

        # Apply T_mat[m]
        z = lp / T_mat[m][None, :, None]
        z -= z.max(axis=-1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=-1, keepdims=True)
        cal_tr.append(q)

    stack_q = np.stack(cal_tr, axis=0)  # (M, N, 6, 4)
    stack_lq = np.log(np.clip(stack_q, 1e-7, 1.0))
    cum_M = np.cumsum(stack_q, axis=-1)[..., :3]  # (M, N, 6, 3)

    W = np.zeros((6, M), dtype=np.float64)
    bounds = [(0.0, 1.0)] * M
    cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    for d in range(6):
        Yd = oh_all[:, d, :]
        if pool_mode == "linear":
            Cd = cum_M[:, :, d, :]

            def obj(w):
                pred_c = np.tensordot(w, Cd, axes=(0, 0))
                return float(np.mean(rw[:, None] * ((pred_c - Yd) ** 2))) + l2_reg * float(np.sum((w - prior_w) ** 2))
        elif pool_mode == "loglin":
            Ld = stack_lq[:, :, d, :]

            def obj(w):
                z = np.tensordot(w, Ld, axes=(0, 0))
                z -= z.max(axis=-1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=-1, keepdims=True)
                pred_c = np.cumsum(p, axis=-1)[:, :3]
                return float(np.mean(rw[:, None] * ((pred_c - Yd) ** 2))) + l2_reg * float(np.sum((w - prior_w) ** 2))
        else:  # hybrid50
            Qd = stack_q[:, :, d, :]
            Ld = stack_lq[:, :, d, :]

            def obj(w):
                p_lin = np.tensordot(w, Qd, axes=(0, 0))
                z = np.tensordot(w, Ld, axes=(0, 0))
                z -= z.max(axis=-1, keepdims=True)
                p_log = np.exp(z)
                p_log /= p_log.sum(axis=-1, keepdims=True)
                p = 0.5 * (p_lin + p_log)
                pred_c = np.cumsum(p, axis=-1)[:, :3]
                return float(np.mean(rw[:, None] * ((pred_c - Yd) ** 2))) + l2_reg * float(np.sum((w - prior_w) ** 2))

        res = minimize(obj, prior_w.copy(), method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 80, "ftol": 1e-8})
        w_opt = np.clip(res.x, 0.0, 1.0)
        W[d] = w_opt / max(1e-9, w_opt.sum())
    return T_mat, W


def apply_routed_pool_v2(
    P_list: List[np.ndarray],
    T_mat: np.ndarray,
    W: np.ndarray,
    pool_mode: str = "linear",
    peg_arr: np.ndarray | None = None,
    bk_arr: np.ndarray | None = None,
    bk_alpha: float = 0.25,
) -> np.ndarray:
    cal = []
    for m, P in enumerate(P_list):
        z = np.log(np.clip(P, 1e-7, 1.0)) / T_mat[m][None, :, None]
        z -= z.max(axis=-1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=-1, keepdims=True)
        cal.append(q)
    stack = np.stack(cal, axis=0)  # (M, N, 6, 4)
    if pool_mode == "linear":
        Q = np.einsum("dm,mndg->ndg", W, stack)
    elif pool_mode == "loglin":
        lq = np.log(np.clip(stack, 1e-7, 1.0))
        z = np.einsum("dm,mndg->ndg", W, lq)
        z -= z.max(axis=-1, keepdims=True)
        Q = np.exp(z)
    else:
        q_lin = np.einsum("dm,mndg->ndg", W, stack)
        lq = np.log(np.clip(stack, 1e-7, 1.0))
        z = np.einsum("dm,mndg->ndg", W, lq)
        z -= z.max(axis=-1, keepdims=True)
        q_log = np.exp(z)
        q_log /= q_log.sum(axis=-1, keepdims=True)
        Q = 0.5 * (q_lin + q_log)
    Q = c.renorm(Q)
    if peg_arr is not None:
        pg = c.renorm(peg_arr)[:, 3, 3]
        m_res = pg >= 0.90
        if np.any(m_res):
            q3 = np.maximum(Q[m_res, 3, 3], pg[m_res])
            Q[m_res, 3, :] = np.stack([1.0 - q3, np.full_like(q3, 1e-6), np.full_like(q3, 1e-6), q3], axis=1)
            Q = c.renorm(Q)
    if bk_arr is not None and bk_alpha > 0.0:
        pb = c.renorm(bk_arr)[:, 5, :]
        Q[:, 5, :] = (1.0 - bk_alpha) * Q[:, 5, :] + bk_alpha * pb
        Q = c.renorm(Q)
    return Q


def extract_os_tile_bndry_covs(
    Q_base: np.ndarray,
    P_dict: Dict[str, np.ndarray],
    members: Sequence[str],
    T_peaks: np.ndarray,
) -> np.ndarray:
    """Extends `extract_os_micro_covariates` with `bndry_dual`/`bndry_pe` gap + `ntg_pe` OS tile-peaks."""
    X_base = uob.extract_os_micro_covariates(Q_base, P_dict, members)
    Q = c.renorm(Q_base)
    gv = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    os_sev = Q[:, 2] @ gv
    os_any = 1.0 - Q[:, 2, 0]
    os_ge1 = Q[:, 2, 2] + Q[:, 2, 3]
    bdual_gap = (c.renorm(P_dict["bndry_dual"])[:, 2] @ gv) - os_sev
    bpe_gap = (c.renorm(P_dict["bndry_pe"])[:, 2] @ gv) - os_sev
    td = T_peaks[:, 2, :]  # (N, 12) for dim=2 过锐
    X_extra = np.column_stack([
        bdual_gap,
        bpe_gap,
        td[:, 0] - os_any,
        td[:, 1],   # gap_p_any
        td[:, 2] - os_ge1,
        td[:, 3],   # gap_p_ge1
        td[:, 7],   # std_p_p0
        td[:, 9] - os_sev,
        td[:, 10],  # gap_sev
    ])
    return np.nan_to_num(np.hstack([X_base, X_extra]), nan=0.0, posinf=10.0, neginf=-10.0)


def extract_og_bndry_covs(
    Q_base: np.ndarray,
    P_dict: Dict[str, np.ndarray],
    members: Sequence[str],
    S_20d: np.ndarray,
    T_peaks: np.ndarray,
) -> np.ndarray:
    """Extends `extract_og24_tile_covs` with `bndry_dual`/`bndry_pe` explicit OG discrepancy."""
    X24 = v4m.extract_og24_tile_covs(Q_base, P_dict, members, S_20d, T_peaks)
    Q = c.renorm(Q_base)
    gv = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    og_sev = Q[:, 4] @ gv
    bdual_g = (c.renorm(P_dict["bndry_dual"])[:, 4] @ gv) - og_sev
    bpe_g = (c.renorm(P_dict["bndry_pe"])[:, 4] @ gv) - og_sev
    bdual_p0 = c.renorm(P_dict["bndry_dual"])[:, 4, 3] - Q[:, 4, 3]
    bpe_p0 = c.renorm(P_dict["bndry_pe"])[:, 4, 3] - Q[:, 4, 3]
    return np.nan_to_num(np.column_stack([X24, bdual_g, bpe_g, bdual_p0, bpe_p0]), nan=0.0, posinf=10.0, neginf=-10.0)


def extract_tex_tile_covs(
    Q_cur: np.ndarray,
    P_dict: Dict[str, np.ndarray],
    nobk_members: Sequence[str],
    T_peaks: np.ndarray,
) -> np.ndarray:
    """Tile-peak + cross-defect + member-gap covariates for `纹理损失` (`dim=1`)."""
    Q = c.renorm(Q_cur)
    gv = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    sevs = np.einsum("ndc,c->nd", Q, gv)
    other_cols = [0, 2, 3, 4, 5]
    other_sevs = sevs[:, other_cols]
    other_cl = np.sum(np.log(np.clip(Q[:, other_cols, 0], 1e-6, 1.0)), axis=1, keepdims=True)
    m_sevs = np.stack([c.renorm(P_dict[m])[:, 1, :] @ gv for m in nobk_members], axis=0)
    m_gaps = (m_sevs - sevs[:, 1][None, :]).T
    m_std = np.std(m_sevs, axis=0, keepdims=True).T
    td = T_peaks[:, 1, :]
    t_rel = np.column_stack([
        td[:, 0] - (1.0 - Q[:, 1, 0]),
        td[:, 2] - (Q[:, 1, 2] + Q[:, 1, 3]),
        td[:, 4] - Q[:, 1, 3],
        td[:, 9] - sevs[:, 1],
    ])
    return np.nan_to_num(np.hstack([other_sevs, other_cl, m_gaps, m_std, td, t_rel]), nan=0.0, posinf=10.0, neginf=-10.0)


def _run_unified_variant(
    tr: np.ndarray,
    va: np.ndarray,
    P12: Dict[str, np.ndarray],
    gold: np.ndarray,
    is_orig: np.ndarray,
    groups: np.ndarray,
    S_all: np.ndarray,
    T_all: np.ndarray,
    members12: List[str],
    nobk11: List[str],
    w_supp_pool: float,
    per_dim_temp: bool,
    pool_mode: str,
    use_deconf: bool,
    use_enhanced_covs: bool,
    use_tex_deconf: bool,
    use_clean_lock: bool,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    P_tr = {m: v[tr] for m, v in P12.items()}
    P_va = {m: v[va] for m, v in P12.items()}
    rw_pool = np.where(is_orig[tr], 1.0, float(w_supp_pool))

    # Build streams for 12-member unified router (matching RoutedGateEUDecoder prior convention)
    base_m = [m for m in members12 if m != "p12_pe"]
    prior_base = red.compute_family_base_prior(base_m)
    p12_tot = 0.18
    prior12 = np.append(prior_base * (1.0 - p12_tot), p12_tot)
    prior12 /= prior12.sum()
    stream_order = base_m + ["p12_pe"]

    T12, W12 = fit_routed_pool_v2(
        [c.renorm(P_tr[m]) for m in stream_order],
        gold[tr], prior12, rw_pool,
        per_dim_temp=per_dim_temp, pool_mode=pool_mode, l2_reg=0.015,
    )
    q_tr = apply_routed_pool_v2(
        [c.renorm(P_tr[m]) for m in stream_order], T12, W12, pool_mode=pool_mode,
        peg_arr=P_tr["peg"], bk_arr=P_tr["pe_bkseg1"], bk_alpha=0.25,
    )
    q_va = apply_routed_pool_v2(
        [c.renorm(P_va[m]) for m in stream_order], T12, W12, pool_mode=pool_mode,
        peg_arr=P_va["peg"], bk_arr=P_va["pe_bkseg1"], bk_alpha=0.25,
    )

    if not use_deconf:
        eu = red.TrueGateConstrainedEU(lams_cl=LAMS_CL_GRID).fit(q_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
        return q_va, eu.predict(q_va), dict(lam_cl=eu.lam_cl, p0_base=eu.p0_scale)

    # Also build nobk 11-member pool (excludes pe_bkseg1 for 纹理损失 + BokehSynergyV3)
    base_nb = [m for m in nobk11 if m != "p12_pe"]
    prior_nb_base = red.compute_family_base_prior(base_nb)
    prior11 = np.append(prior_nb_base * (1.0 - p12_tot), p12_tot)
    prior11 /= prior11.sum()
    nb_order = base_nb + ["p12_pe"]

    T11, W11 = fit_routed_pool_v2(
        [c.renorm(P_tr[m]) for m in nb_order],
        gold[tr], prior11, rw_pool,
        per_dim_temp=per_dim_temp, pool_mode=pool_mode, l2_reg=0.015,
    )
    qnb_tr = apply_routed_pool_v2(
        [c.renorm(P_tr[m]) for m in nb_order], T11, W11, pool_mode=pool_mode,
        peg_arr=P_tr["peg"], bk_arr=None, bk_alpha=0.0,
    )
    qnb_va = apply_routed_pool_v2(
        [c.renorm(P_va[m]) for m in nb_order], T11, W11, pool_mode=pool_mode,
        peg_arr=P_va["peg"], bk_arr=None, bk_alpha=0.0,
    )

    # 1) BokehSynergyV3 on dim=5 using 11-member nobk + pe_bkseg1
    cal_bk = v3m.BokehSynergyV3(w_supp=0.20).fit(
        qnb_tr[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5], gold[tr, 5], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_bk = cal_bk.predict_proba(qnb_tr[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    q_va_bk = cal_bk.predict_proba(qnb_va[:, 5], c.renorm(P_va["pe_bkseg1"])[:, 5])

    # 2) OS (dim=2)
    if use_enhanced_covs:
        Xos_tr = extract_os_tile_bndry_covs(q_tr, P_tr, members12, T_all[tr])
        Xos_va = extract_os_tile_bndry_covs(q_va, P_va, members12, T_all[va])
        os_grid = (2, 4, 8, 14, Xos_tr.shape[1])
    else:
        Xos_tr = uob.extract_os_micro_covariates(q_tr, P_tr, members12)
        Xos_va = uob.extract_os_micro_covariates(q_va, P_va, members12)
        os_grid = (2, 4, 8, Xos_tr.shape[1])
    cal_os = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=os_grid).fit(
        Xos_tr, q_tr[:, 2], gold[tr, 2], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_os = cal_os.predict_proba(Xos_tr, q_tr[:, 2])
    q_va_os = cal_os.predict_proba(Xos_va, q_va[:, 2])

    # 3) OG (dim=4)
    if use_enhanced_covs:
        Xog_tr = extract_og_bndry_covs(q_tr, P_tr, members12, S_all[tr], T_all[tr])
        Xog_va = extract_og_bndry_covs(q_va, P_va, members12, S_all[va], T_all[va])
        og_grid = (4, 8, 14, 24, Xog_tr.shape[1])
    else:
        Xog_tr = v4m.extract_og24_tile_covs(q_tr, P_tr, members12, S_all[tr], T_all[tr])
        Xog_va = v4m.extract_og24_tile_covs(q_va, P_va, members12, S_all[va], T_all[va])
        og_grid = (4, 8, 14, 24)
    cal_og = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=og_grid).fit(
        Xog_tr, q_tr[:, 4], gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_og = cal_og.predict_proba(Xog_tr, q_tr[:, 4])
    q_va_og = cal_og.predict_proba(Xog_va, q_va[:, 4])

    # 4) Blur (dim=0)
    Xbl_tr = v5m.extract_blur33_tile_covs(q_tr, P_tr, nobk11, T_all[tr])
    Xbl_va = v5m.extract_blur33_tile_covs(q_va, P_va, nobk11, T_all[va])
    cal_bl = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 14, 24, Xbl_tr.shape[1])).fit(
        Xbl_tr, q_tr[:, 0], gold[tr, 0], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_bl = cal_bl.predict_proba(Xbl_tr, q_tr[:, 0])
    q_va_bl = cal_bl.predict_proba(Xbl_va, q_va[:, 0])

    # 5) Texture (dim=1): nobk11 (+ optional Rank-1 tile-peak deconf)
    q_tr_tx = qnb_tr[:, 1]
    q_va_tx = qnb_va[:, 1]
    if use_tex_deconf:
        Xtx_tr = extract_tex_tile_covs(q_tr, P_tr, nobk11, T_all[tr])
        Xtx_va = extract_tex_tile_covs(q_va, P_va, nobk11, T_all[va])
        cal_tx = dc.LowRankOrdinalCalibrator(dim=1, mode="deconf", topk_grid=(4, 8, 14, 24, Xtx_tr.shape[1])).fit(
            Xtx_tr, qnb_tr[:, 1], gold[tr, 1], groups[tr], is_orig=is_orig[tr]
        )
        q_tr_tx = cal_tx.predict_proba(Xtx_tr, qnb_tr[:, 1])
        q_va_tx = cal_tx.predict_proba(Xtx_va, qnb_va[:, 1])

    q_tr_cal = q_tr.copy()
    q_tr_cal[:, 0] = q_tr_bl
    q_tr_cal[:, 1] = q_tr_tx
    q_tr_cal[:, 2] = q_tr_os
    q_tr_cal[:, 4] = q_tr_og
    q_tr_cal[:, 5] = q_tr_bk
    q_tr_cal = c.renorm(q_tr_cal)

    q_va_cal = q_va.copy()
    q_va_cal[:, 0] = q_va_bl
    q_va_cal[:, 1] = q_va_tx
    q_va_cal[:, 2] = q_va_os
    q_va_cal[:, 4] = q_va_og
    q_va_cal[:, 5] = q_va_bk
    q_va_cal = c.renorm(q_va_cal)

    # Single unified TrueGateConstrainedEU + inner-CV p0_og
    eu_u = red.TrueGateConstrainedEU(lams_cl=LAMS_CL_GRID).fit(q_tr_cal, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    lam_u, p0b_u = float(eu_u.lam_cl), float(eu_u.p0_scale)
    p0_og_u, st_u = v3m.select_p0_og_inner_cv(q_tr_cal, gold[tr], groups[tr], is_orig[tr], lam_cl=lam_u, p0_base=p0b_u)
    pr_va = v3m.predict_og_p0_eu(q_va_cal, st_u, lam_cl=lam_u, p0_base=p0b_u, p0_og=p0_og_u)

    if use_clean_lock:
        # Lock Clean mask strictly to base calibrated EU (before p0_og boost)
        pr_base_eu = eu_u.predict(q_va_cal)
        cl0 = (pr_base_eu == 0).all(axis=1)
        pr_va[cl0] = 0
        nc = (~cl0) & (pr_va == 0).all(axis=1)
        if np.any(nc):
            pr_va[nc] = pr_base_eu[nc]

    return q_va_cal, pr_va, dict(lam_cl=lam_u, p0_base=p0b_u, p0_og=float(p0_og_u))


VARIANTS = [
    # (name, w_supp_pool, per_dim_temp, pool_mode, use_deconf, use_enhanced_covs, use_tex_deconf, use_clean_lock)
    ("E0_raw12_lin_w1.0",       1.00, False, "linear",   False, False, False, False),
    ("E1_uni12_lin_w1.0",       1.00, False, "linear",   True,  False, False, True),
    ("E2_uni12_lin_w0.25",      0.25, False, "linear",   True,  False, False, True),
    ("E3_uni12_loglin_w0.25",   0.25, False, "loglin",   True,  False, False, True),
    ("E4_uni12_hyb50_w0.25",    0.25, False, "hybrid50", True,  False, False, True),
    ("E5_uni12_pdt_lin_w0.25",  0.25, True,  "linear",   True,  False, False, True),
    ("E6_uni12_pdt_hyb_w0.25",  0.25, True,  "hybrid50", True,  False, False, True),
    ("E7_uni12_enhcov_lin025",  0.25, False, "linear",   True,  True,  False, True),
    ("E8_uni12_enhcov_hyb025",  0.25, False, "hybrid50", True,  True,  False, True),
    ("E9_uni12_enhcov_tex_h025",0.25, False, "hybrid50", True,  True,  True,  True),
    ("E10_uni12_enhcov_log025", 0.25, False, "loglin",   True,  True,  False, True),
    ("E11_uni12_pdt_log_w0.25", 0.25, True,  "loglin",   True,  True,  False, True),
]


def _fold_worker(k: int) -> dict:
    fold = _G["fold"]
    gold = _G["gold"]
    is_orig = _G["is_orig"]
    groups = _G["groups"]
    P12 = _G["P12"]
    S = _G["S"]
    T = _G["T"]
    m12 = _G["m12"]
    nb11 = _G["nb11"]
    va = fold == k
    tr = ~va
    out = {"k": k}
    for cfg in VARIANTS:
        name = cfg[0]
        qv, pv, meta = _run_unified_variant(tr, va, P12, gold, is_orig, groups, S, T, m12, nb11, *cfg[1:])
        out[name] = (qv, pv, meta)
    return out


def main() -> None:
    t0 = time.time()
    m12 = list(SETS["resident_v8_boundary_novlm"])
    nb11 = [m for m in m12 if m != "pe_bkseg1"]
    m10 = list(SETS["resident_v7_novlm"])

    names, gold, fold, _, P10 = oof_pool(m10)
    _, _, _, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)

    P12 = dict(P10)
    for k_ex, d_rel in [("bndry_pe", "results/r55/boundary_v2/v2_pe_ntg"), ("bndry_dual", "results/r55/boundary_v2/v2_dual_pe_cradio")]:
        arr, valid = load_oof_dir(d_rel, names)
        assert valid.all()
        P12[k_ex] = c.renorm(arr)

    S_oof = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz"), names, "feats")
    T_oof = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_oof_3573.npz"), names, "stats")

    _G.update(fold=fold, gold=gold, is_orig=is_orig, groups=groups, P12=P12, S=S_oof, T=T_oof, m12=m12, nb11=nb11)

    print(f"=== Unified 12-Vision-Member Zero-Leakage 8-Fold Nested OOF Lab (N={N}, {len(VARIANTS)} variants) ===", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        f_res = sorted(pool.map(_fold_worker, list(range(8))), key=lambda d: d["k"])

    print("\n--- Reference Baselines (8-Fold Nested OOF) ---")
    print("  [Ref] Full v8_boundary (14m, +2 VLM)    | ORIG=52.665 | Doc1_o=52.440 | ALL=51.959 | RPS_o=0.11221")
    print("  [Ref] No-VLM deconf_v6_boundary (patch) | ORIG=52.327 | Doc1_o=52.202 | ALL=51.864 | RPS_o=0.11344\n")

    results = {}
    for cfg in VARIANTS:
        name = cfg[0]
        Q_oof = np.zeros((N, 6, 4))
        pr_oof = np.zeros((N, 6), dtype=np.int64)
        lams, p0ogs = [], []
        for r in f_res:
            va = fold == r["k"]
            Q_oof[va], pr_oof[va], meta = r[name]
            lams.append(meta["lam_cl"])
            p0ogs.append(meta.get("p0_og", 1.0))
        rep = v5m._summarize_all4(name, Q_oof, pr_oof, gold, is_orig)
        cp0 = int(((gold[is_orig] == 0).all(axis=1) & (pr_oof[is_orig] == 3).any(axis=1)).sum())
        pf, cf, _ = red._fast_boot654_pos_risk(pr_oof[is_orig], gold[is_orig], groups[is_orig], n_boot=5000)
        print(f"      Clean-P0={cp0} | Boot654(Pos<15)={pf:.2f}% | fold_lam={lams} | fold_p0og={p0ogs}", flush=True)
        rep["clean_p0"] = cp0
        rep["boot654_pos_risk"] = round(pf, 3)
        rep["fold_lams"] = lams
        rep["fold_p0og"] = p0ogs
        results[name] = rep

    out_p = os.path.join(REPO, "results/r55/final/novlm_unified_decoder_ablation_oof.json")
    json.dump(results, open(out_p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved -> {out_p} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
