#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/ablate_novlm_surgical_v6plus.py
========================================
Strict 8-fold video-grouped Zero-Leakage Nested OOF surgical ablation starting from
the proven `resident_v8_boundary_novlm + deconf_v6_boundary` baseline (ORIG=52.3274,
test_150=54.52). Tests adding `bndry_pe` & `bndry_dual` via temperature-calibrated
pre-LRO / post-LRO log-linear fusion on EVERY dimension while preserving the exact
well-calibrated 10-member base pool and bi-directional Clean lock.
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
import upgrade_deconf_v6_boundary as v6m  # noqa: E402
import upgrade_os_og_bokeh_deconf as uob  # noqa: E402
from eval_specialist_gate import load_oof_dir  # noqa: E402
from final_engine import SETS, oof_pool  # noqa: E402

LAMS_CL_GRID = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)

_G: Dict = {}


def fit_scalar_temp_dim(P_d: np.ndarray, y_d: np.ndarray, w: np.ndarray) -> float:
    oh = np.cumsum(np.eye(4)[y_d], axis=-1)[:, :3]
    lp = np.log(np.clip(np.asarray(P_d, dtype=np.float64), 1e-7, 1.0))
    best_t, best_r = 1.0, 1e9
    for t in np.linspace(0.60, 2.50, 25):
        z = lp / float(t)
        z -= z.max(axis=-1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=-1, keepdims=True)
        r = float(np.average(((np.cumsum(q, axis=-1)[:, :3] - oh) ** 2).mean(axis=-1), weights=w))
        if r < best_r:
            best_r, best_t = r, float(t)
    return best_t


def apply_temp_dim(P_d: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(np.asarray(P_d, dtype=np.float64), 1e-7, 1.0)) / float(T)
    z -= z.max(axis=-1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=-1, keepdims=True)
    return q


def eval_with_lock(
    Q_tr: np.ndarray,
    Q_va: np.ndarray,
    pr_lock_va: np.ndarray,
    gold_tr: np.ndarray,
    groups_tr: np.ndarray,
    is_orig_tr: np.ndarray,
    lam_cl: float,
    p0_base: float,
    unlock_dims: Sequence[int],
) -> Tuple[np.ndarray, float]:
    p0_og, st = v3m.select_p0_og_inner_cv(Q_tr, gold_tr, groups_tr, is_orig_tr, lam_cl=lam_cl, p0_base=p0_base)
    pr_va = v3m.predict_og_p0_eu(Q_va, st, lam_cl=lam_cl, p0_base=p0_base, p0_og=p0_og)
    lock_dims = [d for d in range(6) if d not in unlock_dims]
    if lock_dims:
        pr_va[:, lock_dims] = pr_lock_va[:, lock_dims]
    cl_lock = (pr_lock_va == 0).all(axis=1)
    pr_va[cl_lock] = 0
    nc = (~cl_lock) & (pr_va == 0).all(axis=1)
    if np.any(nc):
        pr_va[nc] = pr_lock_va[nc]
    return pr_va, float(p0_og)


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

    # ---- Exact v2 -> v4 -> v5 base on 10 vision members ----
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
    qv2_tr[:, 2] = q_tr_os
    qv2_tr[:, 4] = cal_og2.predict_proba(X_og2_tr, q_tr_base[:, 4])
    qv2_tr[:, 5] = cal_bk2.predict_proba(q_tr_nobk_base[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    qv2_tr = c.renorm(qv2_tr)

    qv2_va = q_va_base.copy()
    qv2_va[:, 2] = q_va_os
    qv2_va[:, 4] = cal_og2.predict_proba(X_og2_va, q_va_base[:, 4])
    qv2_va[:, 5] = cal_bk2.predict_proba(q_va_nobk_base[:, 5], c.renorm(P_va["pe_bkseg1"])[:, 5])
    qv2_va = c.renorm(qv2_va)

    eu_v2 = red.TrueGateConstrainedEU(lams_cl=dec_k.lams_cl, p0_scales=dec_k.eu.p0_scales, w_supp=dec_k.eu.w_supp).fit(
        qv2_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr]
    )
    pr_v2_va = eu_v2.predict(qv2_va)
    lam_cl, p0_base = float(eu_v2.lam_cl), float(eu_v2.p0_scale)

    # v4
    cal_bk3 = v3m.BokehSynergyV3(w_supp=0.20).fit(q_tr_nobk_base[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5], gold[tr, 5], groups[tr], is_orig=is_orig[tr])
    q_tr_bk3 = cal_bk3.predict_proba(q_tr_nobk_base[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    q_va_bk3 = cal_bk3.predict_proba(q_va_nobk_base[:, 5], c.renorm(P_va["pe_bkseg1"])[:, 5])

    X24_tr = v4m.extract_og24_tile_covs(q_tr_base, P_tr, members_v7, S_all[tr], T_all[tr])
    X24_va = v4m.extract_og24_tile_covs(q_va_base, P_va, members_v7, S_all[va], T_all[va])
    cal_og24 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(X24_tr, q_tr_base[:, 4], gold[tr, 4], groups[tr], is_orig=is_orig[tr])
    qv4_tr = q_tr_base.copy()
    qv4_tr[:, 2], qv4_tr[:, 4], qv4_tr[:, 5] = q_tr_os, cal_og24.predict_proba(X24_tr, q_tr_base[:, 4]), q_tr_bk3
    qv4_tr = c.renorm(qv4_tr)
    qv4_va = q_va_base.copy()
    qv4_va[:, 2], qv4_va[:, 4], qv4_va[:, 5] = q_va_os, cal_og24.predict_proba(X24_va, q_va_base[:, 4]), q_va_bk3
    qv4_va = c.renorm(qv4_va)
    pr_v4_va, _ = eval_with_lock(qv4_tr, qv4_va, pr_v2_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(2, 4, 5))

    # v5
    nb_cfg = dc._build_routed_gate_cfg(nobk_members)
    nb_cfg["lams_cl"] = LAMS_CL_GRID
    dec_nb = red.RoutedGateEUDecoder(**nb_cfg).fit(P_nb_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    qnb_tr = dec_nb.pool_from_member_dict(P_nb_tr)
    qnb_va = dec_nb.pool_from_member_dict(P_nb_va)

    Xbl_tr = v5m.extract_blur33_tile_covs(qv4_tr, P_tr, nobk_members, T_all[tr])
    Xbl_va = v5m.extract_blur33_tile_covs(qv4_va, P_va, nobk_members, T_all[va])
    cal_bl = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 14, 24, Xbl_tr.shape[1])).fit(Xbl_tr, qv4_tr[:, 0], gold[tr, 0], groups[tr], is_orig=is_orig[tr])
    qv5_tr = qv4_tr.copy()
    qv5_tr[:, 0], qv5_tr[:, 1] = cal_bl.predict_proba(Xbl_tr, qv4_tr[:, 0]), qnb_tr[:, 1]
    qv5_tr = c.renorm(qv5_tr)
    qv5_va = qv4_va.copy()
    qv5_va[:, 0], qv5_va[:, 1] = cal_bl.predict_proba(Xbl_va, qv4_va[:, 0]), qnb_va[:, 1]
    qv5_va = c.renorm(qv5_va)
    pr_v5_va, _ = eval_with_lock(qv5_tr, qv5_va, pr_v4_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(0, 1))

    # ---- Per-dim temperature-calibrated bndry_pe & bndry_dual ----
    cal_tr, cal_va = {}, {}
    cal_pdt_tr, cal_pdt_va = {}, {}
    for nm, arr in P_ex.items():
        tk = v6m.fit_scalar_temp_weighted(arr[tr], gold[tr], rw)
        cal_tr[nm] = v6m.apply_scalar_temp(arr[tr], tk)
        cal_va[nm] = v6m.apply_scalar_temp(arr[va], tk)
        # Per-dim temperature version
        pdt_tr = np.zeros_like(arr[tr])
        pdt_va = np.zeros_like(arr[va])
        for d in range(6):
            td = fit_scalar_temp_dim(arr[tr, d], gold[tr, d], rw)
            pdt_tr[:, d] = apply_temp_dim(arr[tr, d], td)
            pdt_va[:, d] = apply_temp_dim(arr[va, d], td)
        cal_pdt_tr[nm], cal_pdt_va[nm] = pdt_tr, pdt_va

    # ---- Baseline S0: Exact deconf_v6_boundary (novlm) ----
    w_bl_s0 = v6m.fit_loglin_blend_dim(qv5_tr[:, 0], [cal_tr["bndry_pe"][:, 0]], gold[tr, 0], rw)
    bl_tr_s0 = v6m.apply_loglin_blend_dim(qv5_tr[:, 0], [cal_tr["bndry_pe"][:, 0]], w_bl_s0)
    bl_va_s0 = v6m.apply_loglin_blend_dim(qv5_va[:, 0], [cal_va["bndry_pe"][:, 0]], w_bl_s0)

    ex_keys = ("bndry_pe", "bndry_dual")
    w_og_s0 = v6m.fit_loglin_blend_dim(q_tr_base[:, 4], [cal_tr[nm][:, 4] for nm in ex_keys], gold[tr, 4], rw)
    og0_tr = v6m.apply_loglin_blend_dim(q_tr_base[:, 4], [cal_tr[nm][:, 4] for nm in ex_keys], w_og_s0)
    og0_va = v6m.apply_loglin_blend_dim(q_va_base[:, 4], [cal_va[nm][:, 4] for nm in ex_keys], w_og_s0)
    qb_og_tr = q_tr_base.copy(); qb_og_tr[:, 4] = og0_tr
    qb_og_va = q_va_base.copy(); qb_og_va[:, 4] = og0_va
    c_og_s0 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        v4m.extract_og24_tile_covs(qb_og_tr, P_tr, members_v7, S_all[tr], T_all[tr]), og0_tr, gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    og_tr_s0 = c_og_s0.predict_proba(v4m.extract_og24_tile_covs(qb_og_tr, P_tr, members_v7, S_all[tr], T_all[tr]), og0_tr)
    og_va_s0 = c_og_s0.predict_proba(v4m.extract_og24_tile_covs(qb_og_va, P_va, members_v7, S_all[va], T_all[va]), og0_va)

    qs0_tr = qv5_tr.copy(); qs0_tr[:, 0], qs0_tr[:, 4] = bl_tr_s0, og_tr_s0; qs0_tr = c.renorm(qs0_tr)
    qs0_va = qv5_va.copy(); qs0_va[:, 0], qs0_va[:, 4] = bl_va_s0, og_va_s0; qs0_va = c.renorm(qs0_va)
    pr_s0, _ = eval_with_lock(qs0_tr, qs0_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(0, 4))

    out = {"k": k, "S0_v6_baseline": (qs0_va, pr_s0)}

    # ---- S1a: Keep v5 blur (drop post-LRO bndry_pe on blur) ----
    qs1a_tr = qs0_tr.copy(); qs1a_tr[:, 0] = qv5_tr[:, 0]
    qs1a_va = qs0_va.copy(); qs1a_va[:, 0] = qv5_va[:, 0]
    pr_s1a, _ = eval_with_lock(qs1a_tr, qs1a_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(4,))
    out["S1a_blur_keepV5"] = (qs1a_va, pr_s1a)

    # ---- S1b: Pre-LRO (bndry_pe, bndry_dual) on 模糊 -> cal_blur33 ----
    w_bl_pre = v6m.fit_loglin_blend_dim(qv4_tr[:, 0], [cal_pdt_tr[nm][:, 0] for nm in ex_keys], gold[tr, 0], rw)
    bl0_tr = v6m.apply_loglin_blend_dim(qv4_tr[:, 0], [cal_pdt_tr[nm][:, 0] for nm in ex_keys], w_bl_pre)
    bl0_va = v6m.apply_loglin_blend_dim(qv4_va[:, 0], [cal_pdt_va[nm][:, 0] for nm in ex_keys], w_bl_pre)
    qb_bl_tr = qv4_tr.copy(); qb_bl_tr[:, 0] = bl0_tr
    qb_bl_va = qv4_va.copy(); qb_bl_va[:, 0] = bl0_va
    Xbl_p_tr = v5m.extract_blur33_tile_covs(qb_bl_tr, P_tr, nobk_members, T_all[tr])
    Xbl_p_va = v5m.extract_blur33_tile_covs(qb_bl_va, P_va, nobk_members, T_all[va])
    c_bl_pre = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 14, 24, Xbl_p_tr.shape[1])).fit(
        Xbl_p_tr, bl0_tr, gold[tr, 0], groups[tr], is_orig=is_orig[tr]
    )
    bl_pre_tr = c_bl_pre.predict_proba(Xbl_p_tr, bl0_tr)
    bl_pre_va = c_bl_pre.predict_proba(Xbl_p_va, bl0_va)
    qs1b_tr = qs0_tr.copy(); qs1b_tr[:, 0] = bl_pre_tr
    qs1b_va = qs0_va.copy(); qs1b_va[:, 0] = bl_pre_va
    pr_s1b, _ = eval_with_lock(qs1b_tr, qs1b_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(0, 4))
    out["S1b_blur_preLRO_2b"] = (qs1b_va, pr_s1b)

    # ---- S2a: Pre-LRO (bndry_pe, bndry_dual) on 过锐 (dim=2) -> cal_os ----
    w_os = v6m.fit_loglin_blend_dim(q_tr_base[:, 2], [cal_pdt_tr[nm][:, 2] for nm in ex_keys], gold[tr, 2], rw)
    os0_tr = v6m.apply_loglin_blend_dim(q_tr_base[:, 2], [cal_pdt_tr[nm][:, 2] for nm in ex_keys], w_os)
    os0_va = v6m.apply_loglin_blend_dim(q_va_base[:, 2], [cal_pdt_va[nm][:, 2] for nm in ex_keys], w_os)
    qb_os_tr = q_tr_base.copy(); qb_os_tr[:, 2] = os0_tr
    qb_os_va = q_va_base.copy(); qb_os_va[:, 2] = os0_va
    Xos_p_tr = uob.extract_os_micro_covariates(qb_os_tr, P_tr, members_v7)
    Xos_p_va = uob.extract_os_micro_covariates(qb_os_va, P_va, members_v7)
    c_os_pre = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=(2, 4, 8, 11)).fit(
        Xos_p_tr, os0_tr, gold[tr, 2], groups[tr], is_orig=is_orig[tr]
    )
    os_pre_tr = c_os_pre.predict_proba(Xos_p_tr, os0_tr)
    os_pre_va = c_os_pre.predict_proba(Xos_p_va, os0_va)
    qs2a_tr = qs0_tr.copy(); qs2a_tr[:, 2] = os_pre_tr
    qs2a_va = qs0_va.copy(); qs2a_va[:, 2] = os_pre_va
    pr_s2a, _ = eval_with_lock(qs2a_tr, qs2a_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(0, 2, 4))
    out["S2a_os_preLRO_2b"] = (qs2a_va, pr_s2a)

    # ---- S2b: Post-LRO (bndry_pe, bndry_dual) on 过锐 (dim=2) ----
    w_os_post = v6m.fit_loglin_blend_dim(qv5_tr[:, 2], [cal_pdt_tr[nm][:, 2] for nm in ex_keys], gold[tr, 2], rw)
    os_post_tr = v6m.apply_loglin_blend_dim(qv5_tr[:, 2], [cal_pdt_tr[nm][:, 2] for nm in ex_keys], w_os_post)
    os_post_va = v6m.apply_loglin_blend_dim(qv5_va[:, 2], [cal_pdt_va[nm][:, 2] for nm in ex_keys], w_os_post)
    qs2b_tr = qs0_tr.copy(); qs2b_tr[:, 2] = os_post_tr
    qs2b_va = qs0_va.copy(); qs2b_va[:, 2] = os_post_va
    pr_s2b, _ = eval_with_lock(qs2b_tr, qs2b_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(0, 2, 4))
    out["S2b_os_postLRO_2b"] = (qs2b_va, pr_s2b)

    # ---- S3: Log-linear (bndry_pe, bndry_dual) on 纹理损失 (dim=1) ----
    w_tx = v6m.fit_loglin_blend_dim(qv5_tr[:, 1], [cal_pdt_tr[nm][:, 1] for nm in ex_keys], gold[tr, 1], rw)
    tx_tr = v6m.apply_loglin_blend_dim(qv5_tr[:, 1], [cal_pdt_tr[nm][:, 1] for nm in ex_keys], w_tx)
    tx_va = v6m.apply_loglin_blend_dim(qv5_va[:, 1], [cal_pdt_va[nm][:, 1] for nm in ex_keys], w_tx)
    qs3_tr = qs0_tr.copy(); qs3_tr[:, 1] = tx_tr
    qs3_va = qs0_va.copy(); qs3_va[:, 1] = tx_va
    pr_s3, _ = eval_with_lock(qs3_tr, qs3_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(0, 1, 4))
    out["S3_tex_loglin_2b"] = (qs3_va, pr_s3)

    # ---- S4: Log-linear (bndry_pe, bndry_dual) on 虚化突兀 (dim=5) ----
    w_bk = v6m.fit_loglin_blend_dim(qv5_tr[:, 5], [cal_pdt_tr[nm][:, 5] for nm in ex_keys], gold[tr, 5], rw)
    bk_tr = v6m.apply_loglin_blend_dim(qv5_tr[:, 5], [cal_pdt_tr[nm][:, 5] for nm in ex_keys], w_bk)
    bk_va = v6m.apply_loglin_blend_dim(qv5_va[:, 5], [cal_pdt_va[nm][:, 5] for nm in ex_keys], w_bk)
    qs4_tr = qs0_tr.copy(); qs4_tr[:, 5] = bk_tr
    qs4_va = qs0_va.copy(); qs4_va[:, 5] = bk_va
    pr_s4, _ = eval_with_lock(qs4_tr, qs4_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(0, 4, 5))
    out["S4_bk_loglin_2b"] = (qs4_va, pr_s4)

    # ---- S5: Per-dim temp on OG pre-LRO (dim=4) ----
    w_og_pdt = v6m.fit_loglin_blend_dim(q_tr_base[:, 4], [cal_pdt_tr[nm][:, 4] for nm in ex_keys], gold[tr, 4], rw)
    og0_p_tr = v6m.apply_loglin_blend_dim(q_tr_base[:, 4], [cal_pdt_tr[nm][:, 4] for nm in ex_keys], w_og_pdt)
    og0_p_va = v6m.apply_loglin_blend_dim(q_va_base[:, 4], [cal_pdt_va[nm][:, 4] for nm in ex_keys], w_og_pdt)
    qb_ogp_tr = q_tr_base.copy(); qb_ogp_tr[:, 4] = og0_p_tr
    qb_ogp_va = q_va_base.copy(); qb_ogp_va[:, 4] = og0_p_va
    c_og_p = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        v4m.extract_og24_tile_covs(qb_ogp_tr, P_tr, members_v7, S_all[tr], T_all[tr]), og0_p_tr, gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    og_p_tr = c_og_p.predict_proba(v4m.extract_og24_tile_covs(qb_ogp_tr, P_tr, members_v7, S_all[tr], T_all[tr]), og0_p_tr)
    og_p_va = c_og_p.predict_proba(v4m.extract_og24_tile_covs(qb_ogp_va, P_va, members_v7, S_all[va], T_all[va]), og0_p_va)
    qs5_tr = qs0_tr.copy(); qs5_tr[:, 4] = og_p_tr
    qs5_va = qs0_va.copy(); qs5_va[:, 4] = og_p_va
    pr_s5, _ = eval_with_lock(qs5_tr, qs5_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(0, 4))
    out["S5_og_pdt_preLRO"] = (qs5_va, pr_s5)

    # ---- S6: Full surgical combo (keepV5 blur + preLRO OS + loglin Tex + loglin BK + pdt OG) ----
    q_c1_tr = qv5_tr.copy()
    q_c1_tr[:, 1], q_c1_tr[:, 2], q_c1_tr[:, 4], q_c1_tr[:, 5] = tx_tr, os_pre_tr, og_p_tr, bk_tr
    q_c1_tr = c.renorm(q_c1_tr)
    q_c1_va = qv5_va.copy()
    q_c1_va[:, 1], q_c1_va[:, 2], q_c1_va[:, 4], q_c1_va[:, 5] = tx_va, os_pre_va, og_p_va, bk_va
    q_c1_va = c.renorm(q_c1_va)
    pr_c1, _ = eval_with_lock(q_c1_tr, q_c1_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(1, 2, 4, 5))
    out["S6_combo_all_cleanlock"] = (q_c1_va, pr_c1)

    # ---- S7: Full surgical combo + preLRO blur + unified EU (allow Clean to re-evaluate with 12m probs!) ----
    q_c2_tr = q_c1_tr.copy(); q_c2_tr[:, 0] = bl_pre_tr; q_c2_tr = c.renorm(q_c2_tr)
    q_c2_va = q_c1_va.copy(); q_c2_va[:, 0] = bl_pre_va; q_c2_va = c.renorm(q_c2_va)
    pr_c2_lock, _ = eval_with_lock(q_c2_tr, q_c2_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(0, 1, 2, 4, 5))
    out["S7_combo_with_blpre_lock"] = (q_c2_va, pr_c2_lock)

    # ---- S8: S6_combo + re-fit TrueGateConstrainedEU on full 12-member calibrated Q (unlocked Clean!) ----
    eu_c1 = red.TrueGateConstrainedEU(lams_cl=LAMS_CL_GRID).fit(q_c1_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    p0_c1, st_c1 = v3m.select_p0_og_inner_cv(q_c1_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=float(eu_c1.lam_cl), p0_base=float(eu_c1.p0_scale))
    pr_c1_u = v3m.predict_og_p0_eu(q_c1_va, st_c1, lam_cl=float(eu_c1.lam_cl), p0_base=float(eu_c1.p0_scale), p0_og=p0_c1)
    out["S8_combo_unlocked_EU"] = (q_c1_va, pr_c1_u)

    # ---- S9: Clean-locked surgical (keepV5 blur + preLRO OS + pdt OG + loglin BK; NO tex touch) ----
    q_c3_tr = qv5_tr.copy()
    q_c3_tr[:, 2], q_c3_tr[:, 4], q_c3_tr[:, 5] = os_pre_tr, og_p_tr, bk_tr
    q_c3_tr = c.renorm(q_c3_tr)
    q_c3_va = qv5_va.copy()
    q_c3_va[:, 2], q_c3_va[:, 4], q_c3_va[:, 5] = os_pre_va, og_p_va, bk_va
    q_c3_va = c.renorm(q_c3_va)
    pr_c3, _ = eval_with_lock(q_c3_tr, q_c3_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(2, 4, 5))
    out["S9_os_og_bk_cleanlock"] = (q_c3_va, pr_c3)

    # ---- S10: Clean-locked surgical (postLRO OS + pdt OG + loglin BK + keepV5 blur) ----
    q_c4_tr = qv5_tr.copy()
    q_c4_tr[:, 2], q_c4_tr[:, 4], q_c4_tr[:, 5] = os_post_tr, og_p_tr, bk_tr
    q_c4_tr = c.renorm(q_c4_tr)
    q_c4_va = qv5_va.copy()
    q_c4_va[:, 2], q_c4_va[:, 4], q_c4_va[:, 5] = os_post_va, og_p_va, bk_va
    q_c4_va = c.renorm(q_c4_va)
    pr_c4, _ = eval_with_lock(q_c4_tr, q_c4_va, pr_v5_va, gold[tr], groups[tr], is_orig[tr], lam_cl, p0_base, unlock_dims=(2, 4, 5))
    out["S10_ospost_og_bk_lock"] = (q_c4_va, pr_c4)

    out["weights"] = {
        "w_os_pre": np.round(w_os, 4).tolist(),
        "w_os_post": np.round(w_os_post, 4).tolist(),
        "w_tx": np.round(w_tx, 4).tolist(),
        "w_bk": np.round(w_bk, 4).tolist(),
        "w_bl_pre": np.round(w_bl_pre, 4).tolist(),
        "w_og_pdt": np.round(w_og_pdt, 4).tolist(),
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

    print(f"=== Surgical 12-Vision v6+ Zero-Leakage 8-Fold Nested OOF Lab (N={N}) ===", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        f_res = sorted(pool.map(_fold_worker, list(range(8))), key=lambda d: d["k"])

    print("\n--- Reference ---")
    print("  [Ref] Full v8_boundary (+2 VLM) | ORIG=52.665 | Doc1_o=52.440 | ALL=51.959 | RPS_o=0.11221\n")

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
        bk_f1_o = dc._f1_bin(pr_oof[is_orig, 5] == 3, gold[is_orig, 5] == 3)
        print(f"      BK_P0_F1_o={bk_f1_o:.4f} | Clean-P0={cp0} | Boot654(Pos<15)={pf:.2f}%", flush=True)
        rep["bk_p0_f1_orig"] = round(bk_f1_o, 4)
        rep["clean_p0"] = cp0
        rep["boot654_pos_risk"] = round(pf, 3)
        results[nm] = rep

    print("\nMean fold blend weights (bndry_pe, bndry_dual):")
    for wk in f_res[0]["weights"]:
        arr_w = np.mean([r["weights"][wk] for r in f_res], axis=0)
        print(f"  {wk:12s}: {np.round(arr_w, 4).tolist()}")

    out_p = os.path.join(REPO, "results/r55/final/novlm_surgical_v6plus_ablation_oof.json")
    json.dump(results, open(out_p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved -> {out_p} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
