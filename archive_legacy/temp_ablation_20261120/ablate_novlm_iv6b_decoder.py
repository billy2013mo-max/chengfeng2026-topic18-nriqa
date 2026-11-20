#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/ablate_novlm_iv6b_decoder.py
=====================================
8-fold video-grouped Zero-Leakage Nested OOF optimization (`N = 3,573`, `ORIG = 1,902`)
for `resident_v8_novlm_iv6b` (12 pure-vision members + `InternViT-6B` `iv6b_native_pilot`,
0 VLM!). Compares surgical placements of `iv6b` across dimensions, pre-LRO vs post-LRO,
P0-preserving Clean-lock vs unified EU, and verifies all 4 metrics (`doc2_orig`, `doc1_orig`,
`doc2_all`, `rps_orig`, `BK_P0_F1`, `Clean-P0`, `Boot654(Pos<15)`).
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

LAMS = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)

_G: Dict = {}


def _fit_scalar_temp_dim(P_d: np.ndarray, y_d: np.ndarray, w: np.ndarray) -> float:
    oh = np.cumsum(np.eye(4)[y_d], axis=-1)[:, :3]
    lp = np.log(np.clip(np.asarray(P_d, dtype=np.float64), 1e-7, 1.0))
    bt, br = 1.0, 1e9
    for t in np.linspace(0.60, 2.50, 25):
        z = lp / float(t)
        z -= z.max(axis=-1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=-1, keepdims=True)
        r = float(np.average(((np.cumsum(q, axis=-1)[:, :3] - oh) ** 2).mean(axis=-1), weights=w))
        if r < br:
            br, bt = r, float(t)
    return bt


def _apply_temp_dim(P_d: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(np.asarray(P_d, dtype=np.float64), 1e-7, 1.0)) / float(T)
    z -= z.max(axis=-1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=-1, keepdims=True)
    return q


def _blend_dim(q_tr_d, q_va_d, ctr, cva, keys, d, y_tr_d, rw, max_each=0.40, max_sum=0.45):
    w = v6m.fit_loglin_blend_dim(q_tr_d, [ctr[m][:, d] for m in keys], y_tr_d, rw, max_each=max_each, max_sum=max_sum)
    tr_o = v6m.apply_loglin_blend_dim(q_tr_d, [ctr[m][:, d] for m in keys], w)
    va_o = v6m.apply_loglin_blend_dim(q_va_d, [cva[m][:, d] for m in keys], w)
    return tr_o, va_o, w


def _eval_eu(Q_tr, Q_va, gold_tr, groups_tr, is_orig_tr, pr_lock_va=None, lam_cl=None, p0_base=None, lock_dims=()):
    if lam_cl is None or p0_base is None:
        eu = red.TrueGateConstrainedEU(lams_cl=LAMS).fit(Q_tr, gold_tr, is_orig=is_orig_tr, groups=groups_tr)
        lam_cl, p0_base = float(eu.lam_cl), float(eu.p0_scale)
    p0_og, st = v3m.select_p0_og_inner_cv(Q_tr, gold_tr, groups_tr, is_orig_tr, lam_cl=lam_cl, p0_base=p0_base)
    pr = v3m.predict_og_p0_eu(Q_va, st, lam_cl=lam_cl, p0_base=p0_base, p0_og=p0_og)
    if pr_lock_va is not None:
        if lock_dims:
            pr[:, list(lock_dims)] = pr_lock_va[:, list(lock_dims)]
        cl = (pr_lock_va == 0).all(axis=1)
        pr[cl] = 0
        nc = (~cl) & (pr == 0).all(axis=1)
        if np.any(nc):
            pr[nc] = pr_lock_va[nc]
    return pr, dict(lam_cl=lam_cl, p0_base=p0_base, p0_og=float(p0_og))


def _fold_worker(k: int) -> dict:
    fold_arr = _G["fold"]
    gold = _G["gold"]
    is_orig = _G["is_orig"]
    groups = _G["groups"]
    P10 = _G["P10"]
    P_ex = _G["P_ex"]
    S = _G["S"]
    T = _G["T"]
    m10 = _G["m10"]
    nb9 = _G["nb9"]
    proto_cfg = _G["proto_cfg"]

    va = fold_arr == k
    tr = ~va
    P_tr = {m: v[tr] for m, v in P10.items()}
    P_va = {m: v[va] for m, v in P10.items()}
    P_nb_tr = {m: P_tr[m] for m in nb9}
    P_nb_va = {m: P_va[m] for m in nb9}
    rw = np.where(is_orig[tr], 1.0, 0.25)

    # ---- Exact v2 -> v4 -> v5 base on 10 vision members ----
    dec_k = red.RoutedGateEUDecoder(**proto_cfg).fit(P_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    q_tr_b = dec_k.pool_from_member_dict(P_tr)
    q_va_b = dec_k.pool_from_member_dict(P_va)

    cal_og2 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf").fit(
        dc.extract_deconf_covariates(q_tr_b, P_tr, 4, m10), q_tr_b[:, 4], gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    cal_os = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=(2, 4, 8, 11)).fit(
        uob.extract_os_micro_covariates(q_tr_b, P_tr, m10), q_tr_b[:, 2], gold[tr, 2], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_os = cal_os.predict_proba(uob.extract_os_micro_covariates(q_tr_b, P_tr, m10), q_tr_b[:, 2])
    q_va_os = cal_os.predict_proba(uob.extract_os_micro_covariates(q_va_b, P_va, m10), q_va_b[:, 2])

    q_tr_nb0 = uob.pool_without_bkseg1(dec_k, P_tr)
    q_va_nb0 = uob.pool_without_bkseg1(dec_k, P_va)
    cal_bk2 = uob.BokehSynergyCalibrator(w_supp=0.20).fit(
        q_tr_nb0[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5], gold[tr, 5], groups[tr], is_orig=is_orig[tr]
    )

    qv2_tr = q_tr_b.copy()
    qv2_tr[:, 2] = q_tr_os
    qv2_tr[:, 4] = cal_og2.predict_proba(dc.extract_deconf_covariates(q_tr_b, P_tr, 4, m10), q_tr_b[:, 4])
    qv2_tr[:, 5] = cal_bk2.predict_proba(q_tr_nb0[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    qv2_tr = c.renorm(qv2_tr)
    qv2_va = q_va_b.copy()
    qv2_va[:, 2] = q_va_os
    qv2_va[:, 4] = cal_og2.predict_proba(dc.extract_deconf_covariates(q_va_b, P_va, 4, m10), q_va_b[:, 4])
    qv2_va[:, 5] = cal_bk2.predict_proba(q_va_nb0[:, 5], c.renorm(P_va["pe_bkseg1"])[:, 5])
    qv2_va = c.renorm(qv2_va)

    eu_v2 = red.TrueGateConstrainedEU(lams_cl=dec_k.lams_cl, p0_scales=dec_k.eu.p0_scales, w_supp=dec_k.eu.w_supp).fit(
        qv2_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr]
    )
    pr_v2_va = eu_v2.predict(qv2_va)
    lam_v2, p0b_v2 = float(eu_v2.lam_cl), float(eu_v2.p0_scale)

    # v4 + v5
    cal_bk3 = v3m.BokehSynergyV3(w_supp=0.20).fit(
        q_tr_nb0[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5], gold[tr, 5], groups[tr], is_orig=is_orig[tr]
    )
    cal_og24 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        v4m.extract_og24_tile_covs(q_tr_b, P_tr, m10, S[tr], T[tr]), q_tr_b[:, 4], gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    qv4_tr = q_tr_b.copy()
    qv4_tr[:, 2] = q_tr_os
    qv4_tr[:, 4] = cal_og24.predict_proba(v4m.extract_og24_tile_covs(q_tr_b, P_tr, m10, S[tr], T[tr]), q_tr_b[:, 4])
    qv4_tr[:, 5] = cal_bk3.predict_proba(q_tr_nb0[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    qv4_tr = c.renorm(qv4_tr)
    qv4_va = q_va_b.copy()
    qv4_va[:, 2] = q_va_os
    qv4_va[:, 4] = cal_og24.predict_proba(v4m.extract_og24_tile_covs(q_va_b, P_va, m10, S[va], T[va]), q_va_b[:, 4])
    qv4_va[:, 5] = cal_bk3.predict_proba(q_va_nb0[:, 5], c.renorm(P_va["pe_bkseg1"])[:, 5])
    qv4_va = c.renorm(qv4_va)

    nb_cfg = dc._build_routed_gate_cfg(nb9)
    nb_cfg["lams_cl"] = LAMS
    dec_nb = red.RoutedGateEUDecoder(**nb_cfg).fit(P_nb_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    qnb_tr, qnb_va = dec_nb.pool_from_member_dict(P_nb_tr), dec_nb.pool_from_member_dict(P_nb_va)
    Xbl_tr = v5m.extract_blur33_tile_covs(qv4_tr, P_tr, nb9, T[tr])
    Xbl_va = v5m.extract_blur33_tile_covs(qv4_va, P_va, nb9, T[va])
    cal_bl = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 14, 24, Xbl_tr.shape[1])).fit(
        Xbl_tr, qv4_tr[:, 0], gold[tr, 0], groups[tr], is_orig=is_orig[tr]
    )
    qv5_tr = qv4_tr.copy(); qv5_tr[:, 0], qv5_tr[:, 1] = cal_bl.predict_proba(Xbl_tr, qv4_tr[:, 0]), qnb_tr[:, 1]; qv5_tr = c.renorm(qv5_tr)
    qv5_va = qv4_va.copy(); qv5_va[:, 0], qv5_va[:, 1] = cal_bl.predict_proba(Xbl_va, qv4_va[:, 0]), qnb_va[:, 1]; qv5_va = c.renorm(qv5_va)

    # Temperature calibrate extra models (global scalar AND per-dim scalar)
    ctr, cva = {}, {}
    ctr_d, cva_d = {}, {}
    for nm, arr in P_ex.items():
        tk = v6m.fit_scalar_temp_weighted(arr[tr], gold[tr], rw)
        ctr[nm] = v6m.apply_scalar_temp(arr[tr], tk)
        cva[nm] = v6m.apply_scalar_temp(arr[va], tk)
        pd_tr = np.zeros_like(arr[tr]); pd_va = np.zeros_like(arr[va])
        for d in range(6):
            td = _fit_scalar_temp_dim(arr[tr, d], gold[tr, d], rw)
            pd_tr[:, d] = _apply_temp_dim(arr[tr, d], td)
            pd_va[:, d] = _apply_temp_dim(arr[va, d], td)
        ctr_d[nm], cva_d[nm] = pd_tr, pd_va

    # Pre-compute surgical dimension upgrades cleanly
    # 1) 纹理损失 (dim=1): iv6b only!
    tx_tr, tx_va, w_tx = _blend_dim(qv5_tr[:, 1], qv5_va[:, 1], ctr, cva, ("iv6b",), 1, gold[tr, 1], rw)
    tx_d_tr, tx_d_va, _ = _blend_dim(qv5_tr[:, 1], qv5_va[:, 1], ctr_d, cva_d, ("iv6b",), 1, gold[tr, 1], rw)

    # 2) 过锐 (dim=2): pre-LRO iv6b vs post-LRO iv6b vs pre-LRO (iv6b, bndry_dual)
    os_post_tr, os_post_va, w_os_post = _blend_dim(qv5_tr[:, 2], qv5_va[:, 2], ctr, cva, ("iv6b",), 2, gold[tr, 2], rw)
    os0_tr, os0_va, w_os_pre = _blend_dim(q_tr_b[:, 2], q_va_b[:, 2], ctr, cva, ("iv6b",), 2, gold[tr, 2], rw)
    qb_os_tr = q_tr_b.copy(); qb_os_tr[:, 2] = os0_tr
    qb_os_va = q_va_b.copy(); qb_os_va[:, 2] = os0_va
    c_os_pre = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=(2, 4, 8, 11)).fit(
        uob.extract_os_micro_covariates(qb_os_tr, P_tr, m10), os0_tr, gold[tr, 2], groups[tr], is_orig=is_orig[tr]
    )
    os_pre_tr = c_os_pre.predict_proba(uob.extract_os_micro_covariates(qb_os_tr, P_tr, m10), os0_tr)
    os_pre_va = c_os_pre.predict_proba(uob.extract_os_micro_covariates(qb_os_va, P_va, m10), os0_va)

    # 3) 模糊 (dim=0): iv6b only (post-LRO) vs keepV5
    bl_iv_tr, bl_iv_va, w_bl_iv = _blend_dim(qv5_tr[:, 0], qv5_va[:, 0], ctr, cva, ("iv6b",), 0, gold[tr, 0], rw)

    # 4) 虚化突兀 (dim=5): iv6b only (post-synergy) vs keepV5
    bk_iv_tr, bk_iv_va, w_bk_iv = _blend_dim(qv5_tr[:, 5], qv5_va[:, 5], ctr, cva, ("iv6b",), 5, gold[tr, 5], rw)

    # 5) 过度生成 (dim=4):
    #    (a) pre-LRO (bndry_pe, bndry_dual, iv6b) -> cal_og24
    og0_3_tr, og0_3_va, w_og3 = _blend_dim(q_tr_b[:, 4], q_va_b[:, 4], ctr, cva, ("bndry_pe", "bndry_dual", "iv6b"), 4, gold[tr, 4], rw, max_each=0.35, max_sum=0.45)
    qb_og3_tr = q_tr_b.copy(); qb_og3_tr[:, 4] = og0_3_tr
    qb_og3_va = q_va_b.copy(); qb_og3_va[:, 4] = og0_3_va
    c_og3 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        v4m.extract_og24_tile_covs(qb_og3_tr, P_tr, m10, S[tr], T[tr]), og0_3_tr, gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    og3_tr = c_og3.predict_proba(v4m.extract_og24_tile_covs(qb_og3_tr, P_tr, m10, S[tr], T[tr]), og0_3_tr)
    og3_va = c_og3.predict_proba(v4m.extract_og24_tile_covs(qb_og3_va, P_va, m10, S[va], T[va]), og0_3_va)

    #    (b) pre-LRO (bndry_pe, bndry_dual, iv6b) -> cal_og26 (adds iv6b sev & P0 gap!)
    gv = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    def _og26(qb, P_d, S_d, T_d, iv_d):
        x24 = v4m.extract_og24_tile_covs(qb, P_d, m10, S_d, T_d)
        iv_sev_gap = (iv_d[:, 4] @ gv) - (qb[:, 4] @ gv)
        iv_p0_gap = iv_d[:, 4, 3] - qb[:, 4, 3]
        return np.column_stack([x24, iv_sev_gap, iv_p0_gap])
    X26_tr = _og26(qb_og3_tr, P_tr, S[tr], T[tr], ctr["iv6b"])
    X26_va = _og26(qb_og3_va, P_va, S[va], T[va], cva["iv6b"])
    c_og26 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 26)).fit(
        X26_tr, og0_3_tr, gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    og26_tr = c_og26.predict_proba(X26_tr, og0_3_tr)
    og26_va = c_og26.predict_proba(X26_va, og0_3_va)

    #    (c) pre-LRO (bndry_pe, bndry_dual) only (no iv6b on OG, for ablation)
    og0_2_tr, og0_2_va, _ = _blend_dim(q_tr_b[:, 4], q_va_b[:, 4], ctr, cva, ("bndry_pe", "bndry_dual"), 4, gold[tr, 4], rw)
    qb_og2_tr = q_tr_b.copy(); qb_og2_tr[:, 4] = og0_2_tr
    qb_og2_va = q_va_b.copy(); qb_og2_va[:, 4] = og0_2_va
    c_og2 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        v4m.extract_og24_tile_covs(qb_og2_tr, P_tr, m10, S[tr], T[tr]), og0_2_tr, gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    og2_tr = c_og2.predict_proba(v4m.extract_og24_tile_covs(qb_og2_tr, P_tr, m10, S[tr], T[tr]), og0_2_tr)
    og2_va = c_og2.predict_proba(v4m.extract_og24_tile_covs(qb_og2_va, P_va, m10, S[va], T[va]), og0_2_va)

    def _assemble(bl_t, bl_v, tx_t, tx_v, os_t, os_v, og_t, og_v, bk_t, bk_v):
        Qtr = qv5_tr.copy(); Qva = qv5_va.copy()
        Qtr[:, 0], Qva[:, 0] = bl_t, bl_v
        Qtr[:, 1], Qva[:, 1] = tx_t, tx_v
        Qtr[:, 2], Qva[:, 2] = os_t, os_v
        Qtr[:, 4], Qva[:, 4] = og_t, og_v
        Qtr[:, 5], Qva[:, 5] = bk_t, bk_v
        return c.renorm(Qtr), c.renorm(Qva)

    out = {"k": k}
    configs = [
        # (name, (bl, tx, os, og, bk), use_v2_lock)
        ("V1_iv6b_tx_og3_keepBL_OS_BK_eu",    (qv5_tr[:,0], qv5_va[:,0], tx_tr, tx_va, qv5_tr[:,2], qv5_va[:,2], og3_tr, og3_va, qv5_tr[:,5], qv5_va[:,5]), False),
        ("V2_iv6b_tx_osPre_og3_keepBL_BK_eu", (qv5_tr[:,0], qv5_va[:,0], tx_tr, tx_va, os_pre_tr, os_pre_va, og3_tr, og3_va, qv5_tr[:,5], qv5_va[:,5]), False),
        ("V3_iv6b_tx_osPost_og3_keepBL_BK_eu",(qv5_tr[:,0], qv5_va[:,0], tx_tr, tx_va, os_post_tr, os_post_va, og3_tr, og3_va, qv5_tr[:,5], qv5_va[:,5]), False),
        ("V4_iv6b_bl_tx_osPre_og3_keepBK_eu", (bl_iv_tr, bl_iv_va, tx_tr, tx_va, os_pre_tr, os_pre_va, og3_tr, og3_va, qv5_tr[:,5], qv5_va[:,5]), False),
        ("V5_iv6b_bl_tx_osPre_og3_bk_eu",     (bl_iv_tr, bl_iv_va, tx_tr, tx_va, os_pre_tr, os_pre_va, og3_tr, og3_va, bk_iv_tr, bk_iv_va), False),
        ("V6_iv6b_tx_osPre_og26_keepBL_BK_eu",(qv5_tr[:,0], qv5_va[:,0], tx_tr, tx_va, os_pre_tr, os_pre_va, og26_tr, og26_va, qv5_tr[:,5], qv5_va[:,5]), False),
        ("V7_iv6b_bl_tx_osPre_og26_bk_eu",    (bl_iv_tr, bl_iv_va, tx_tr, tx_va, os_pre_tr, os_pre_va, og26_tr, og26_va, bk_iv_tr, bk_iv_va), False),
        ("V8_iv6b_tx_osPre_og3_keepBL_BK_cl", (qv5_tr[:,0], qv5_va[:,0], tx_tr, tx_va, os_pre_tr, os_pre_va, og3_tr, og3_va, qv5_tr[:,5], qv5_va[:,5]), True),
        ("V9_iv6b_tx_osPre_og2_keepBL_BK_eu", (qv5_tr[:,0], qv5_va[:,0], tx_tr, tx_va, os_pre_tr, os_pre_va, og2_tr, og2_va, qv5_tr[:,5], qv5_va[:,5]), False),
        ("V10_iv6b_txD_osPre_og3_bk_eu",      (qv5_tr[:,0], qv5_va[:,0], tx_d_tr, tx_d_va, os_pre_tr, os_pre_va, og3_tr, og3_va, bk_iv_tr, bk_iv_va), False),
    ]
    for nm, tup, use_cl in configs:
        Qtr, Qva = _assemble(*tup)
        if use_cl:
            pr, meta = _eval_eu(Qtr, Qva, gold[tr], groups[tr], is_orig[tr], pr_lock_va=pr_v2_va, lam_cl=lam_v2, p0_base=p0b_v2)
        else:
            pr, meta = _eval_eu(Qtr, Qva, gold[tr], groups[tr], is_orig[tr])
        out[nm] = (Qva, pr, meta)

    out["weights"] = {
        "w_tx": w_tx.tolist(),
        "w_os_pre": w_os_pre.tolist(),
        "w_os_post": w_os_post.tolist(),
        "w_bl_iv": w_bl_iv.tolist(),
        "w_bk_iv": w_bk_iv.tolist(),
        "w_og3": w_og3.tolist(),
    }
    return out


def main() -> None:
    t0 = time.time()
    m10 = list(SETS["resident_v7_novlm"])
    nb9 = [m for m in m10 if m != "pe_bkseg1"]

    names, gold, fold, _ = c.load_split()
    _, _, _, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)
    _, _, _, _, P10 = oof_pool(m10)

    P_ex = {}
    for k, d_rel in [
        ("bndry_pe", "results/r55/boundary_v2/v2_pe_ntg"),
        ("bndry_dual", "results/r55/boundary_v2/v2_dual_pe_cradio"),
        ("iv6b", "results/r55/iv6b_native_pilot"),
    ]:
        arr, valid = load_oof_dir(d_rel, names)
        assert valid.all()
        P_ex[k] = c.renorm(arr)

    S_oof = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz"), names, "feats")
    T_oof = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_oof_3573.npz"), names, "stats")

    proto_cfg = dc._build_routed_gate_cfg(m10)
    proto_cfg["lams_cl"] = LAMS
    _G.update(fold=fold, gold=gold, is_orig=is_orig, groups=groups, P10=P10, P_ex=P_ex, S=S_oof, T=T_oof, m10=m10, nb9=nb9, proto_cfg=proto_cfg)

    print(f"=== [Route A: novlm12 + iv6b] 8-Fold Zero-Leakage Nested OOF Lab (N={N}) ===", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        f_res = sorted(pool.map(_fold_worker, list(range(8))), key=lambda d: d["k"])

    print("\n  [Ref] Full v8_boundary (+2 VLM, 54.77) | ORIG=52.665 (Pos=16.217 P0=16.605 Cln=19.84) | Doc1_o=52.440 | ALL=51.959 | RPS_o=0.11221")
    print("  [Ref] No-VLM v8_boundary_novlm (54.52) | ORIG=52.327 (Pos=16.016 P0=16.507 Cln=19.81) | Doc1_o=52.202 | ALL=51.864 | RPS_o=0.11344\n")

    keys = [k for k in f_res[0] if k not in ("k", "weights")]
    results = {}
    for nm in keys:
        Q = np.zeros((N, 6, 4))
        pr = np.zeros((N, 6), dtype=np.int64)
        lams, p0bs, p0ogs = [], [], []
        for r in f_res:
            va = fold == r["k"]
            Q[va], pr[va], meta = r[nm]
            lams.append(meta["lam_cl"]); p0bs.append(meta["p0_base"]); p0ogs.append(meta["p0_og"])
        rep = v5m._summarize_all4(nm, Q, pr, gold, is_orig)
        bk_f1 = dc._f1_bin(pr[is_orig, 5] == 3, gold[is_orig, 5] == 3)
        gl_f1 = dc._f1_bin(pr[is_orig, 3] == 3, gold[is_orig, 3] == 3)
        cp0 = int(((gold[is_orig] == 0).all(axis=1) & (pr[is_orig] == 3).any(axis=1)).sum())
        pf, _, _ = red._fast_boot654_pos_risk(pr[is_orig], gold[is_orig], groups[is_orig], n_boot=5000)
        print(f"      BK_P0={bk_f1:.4f} GL_P0={gl_f1:.4f} Clean-P0={cp0} Boot654={pf:.2f}% | lam={lams} p0og={p0ogs}", flush=True)
        rep.update(bk_p0=round(bk_f1, 4), gl_p0=round(gl_f1, 4), clean_p0=cp0, boot654_pos_risk=round(pf, 3), fold_lams=lams, fold_p0og=p0ogs)
        results[nm] = rep

    print("\nMean fold blend weights:")
    for wk in f_res[0]["weights"]:
        print(f"  {wk:10s}: {np.round(np.mean([r['weights'][wk] for r in f_res], axis=0), 4).tolist()}")

    out_p = os.path.join(REPO, "results/r55/final/novlm_iv6b_decoder_ablation_oof.json")
    json.dump(results, open(out_p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n[DONE] Saved -> {out_p} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
