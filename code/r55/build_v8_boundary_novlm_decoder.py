#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_v8_boundary_novlm_decoder.py
===========================================
Builds a strictly controlled, 100% VLM-free `deconf_v6_boundary` decoder bundle
(`BoundaryDeconfV6Bundle`) for `resident_v8_boundary_novlm` (12 vision members = exact
`resident_v8_boundary` minus the two 8B VLM adapters `p1_qy` and `qwen_hr`).

All stages (`routed_gate_eu -> deconf_v2 -> deconf_v4_tile -> deconf_v5_boundary ->
deconf_v6_boundary`) are strictly validated under 8-fold video-grouped Zero-Leakage Nested OOF
(`N = 3,573`, `ORIG = 1,902`), using the exact same calibration pipeline and `lams_cl` search
grid as full `resident_v8_boundary`, with zero test-set tuning and zero legacy JSON anchoring.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from typing import Dict

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
from final_engine import OUT, SETS, load_decoder, oof_pool  # noqa: E402

NOVLM_EXTRA_OOF_DIRS = {
    "bndry_pe": "results/r55/boundary_v2/v2_pe_ntg",
    "bndry_dual": "results/r55/boundary_v2/v2_dual_pe_cradio",
}

LAMS_CL_GRID = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)

_G: Dict = {}


def _fold_worker_novlm(k: int) -> dict:
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

    # ---- Stage 0: Base RoutedGateEUDecoder (10 vision members) ----
    dec_k = red.RoutedGateEUDecoder(**proto_cfg).fit(P_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    q_tr_base = dec_k.pool_from_member_dict(P_tr)
    q_va_base = dec_k.pool_from_member_dict(P_va)
    pr_va_base = dec_k.eu.predict(q_va_base)

    # ---- Stage 1: deconf_v2 (cal_os, cal_og, cal_bk2, eu_v2) ----
    X_og_tr = dc.extract_deconf_covariates(q_tr_base, P_tr, 4, members_v7)
    X_og_va = dc.extract_deconf_covariates(q_va_base, P_va, 4, members_v7)
    cal_og2 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf").fit(
        X_og_tr, q_tr_base[:, 4, :], gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_og2 = cal_og2.predict_proba(X_og_tr, q_tr_base[:, 4, :])
    q_va_og2 = cal_og2.predict_proba(X_og_va, q_va_base[:, 4, :])

    X_os_tr = uob.extract_os_micro_covariates(q_tr_base, P_tr, members_v7)
    X_os_va = uob.extract_os_micro_covariates(q_va_base, P_va, members_v7)
    cal_os = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=(2, 4, 8, 11)).fit(
        X_os_tr, q_tr_base[:, 2, :], gold[tr, 2], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_os = cal_os.predict_proba(X_os_tr, q_tr_base[:, 2, :])
    q_va_os = cal_os.predict_proba(X_os_va, q_va_base[:, 2, :])

    q_tr_nobk_base = uob.pool_without_bkseg1(dec_k, P_tr)
    q_va_nobk_base = uob.pool_without_bkseg1(dec_k, P_va)
    cal_bk2 = uob.BokehSynergyCalibrator(w_supp=0.20).fit(
        q_tr_nobk_base[:, 5, :], c.renorm(P_tr["pe_bkseg1"])[:, 5, :], gold[tr, 5], groups[tr], is_orig[tr]
    )
    q_tr_bk2 = cal_bk2.predict_proba(q_tr_nobk_base[:, 5, :], c.renorm(P_tr["pe_bkseg1"])[:, 5, :])
    q_va_bk2 = cal_bk2.predict_proba(q_va_nobk_base[:, 5, :], c.renorm(P_va["pe_bkseg1"])[:, 5, :])

    qv2_tr = q_tr_base.copy()
    qv2_tr[:, 2] = q_tr_os
    qv2_tr[:, 4] = q_tr_og2
    qv2_tr[:, 5] = q_tr_bk2
    qv2_tr = c.renorm(qv2_tr)

    qv2_va = q_va_base.copy()
    qv2_va[:, 2] = q_va_os
    qv2_va[:, 4] = q_va_og2
    qv2_va[:, 5] = q_va_bk2
    qv2_va = c.renorm(qv2_va)

    eu_v2 = red.TrueGateConstrainedEU(
        lams_cl=dec_k.lams_cl, p0_scales=dec_k.eu.p0_scales, w_supp=dec_k.eu.w_supp
    ).fit(qv2_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    pr_v2_va = eu_v2.predict(qv2_va)
    lam_cl = float(eu_v2.lam_cl)
    p0_base = float(eu_v2.p0_scale)

    # ---- Stage 2: deconf_v4_tile (cal_og24 + BokehSynergyV3 + clean-locked EU) ----
    cal_bk3 = v3m.BokehSynergyV3(w_supp=0.20).fit(
        q_tr_nobk_base[:, 5, :], c.renorm(P_tr["pe_bkseg1"])[:, 5, :], gold[tr, 5], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_bk3 = cal_bk3.predict_proba(q_tr_nobk_base[:, 5, :], c.renorm(P_tr["pe_bkseg1"])[:, 5, :])
    q_va_bk3 = cal_bk3.predict_proba(q_va_nobk_base[:, 5, :], c.renorm(P_va["pe_bkseg1"])[:, 5, :])

    X24_tr = v4m.extract_og24_tile_covs(q_tr_base, P_tr, members_v7, S_all[tr], T_all[tr])
    X24_va = v4m.extract_og24_tile_covs(q_va_base, P_va, members_v7, S_all[va], T_all[va])
    cal_og24 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        X24_tr, q_tr_base[:, 4, :], gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_og4 = cal_og24.predict_proba(X24_tr, q_tr_base[:, 4, :])
    q_va_og4 = cal_og24.predict_proba(X24_va, q_va_base[:, 4, :])

    qv4_tr = q_tr_base.copy()
    qv4_tr[:, 2] = q_tr_os
    qv4_tr[:, 4] = q_tr_og4
    qv4_tr[:, 5] = q_tr_bk3
    qv4_tr = c.renorm(qv4_tr)

    qv4_va = q_va_base.copy()
    qv4_va[:, 2] = q_va_os
    qv4_va[:, 4] = q_va_og4
    qv4_va[:, 5] = q_va_bk3
    qv4_va = c.renorm(qv4_va)

    p0_v4, st_v4 = v3m.select_p0_og_inner_cv(qv4_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=lam_cl, p0_base=p0_base)
    pr_v4_va = v3m.predict_og_p0_eu(qv4_va, st_v4, lam_cl=lam_cl, p0_base=p0_base, p0_og=p0_v4)
    pr_v4_va[:, [0, 1, 3]] = pr_v2_va[:, [0, 1, 3]]
    cl_v2 = (pr_v2_va == 0).all(axis=1)
    pr_v4_va[cl_v2] = 0
    nc4 = (~cl_v2) & (pr_v4_va == 0).all(axis=1)
    if np.any(nc4):
        pr_v4_va[nc4] = pr_v2_va[nc4]

    # ---- Stage 3: deconf_v5_boundary (dec_nobk for 纹理损失 + cal_blur for 模糊) ----
    nb_cfg = dc._build_routed_gate_cfg(nobk_members)
    nb_cfg["lams_cl"] = LAMS_CL_GRID
    dec_nb = red.RoutedGateEUDecoder(**nb_cfg).fit(P_nb_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    qnb_tr = dec_nb.pool_from_member_dict(P_nb_tr)
    qnb_va = dec_nb.pool_from_member_dict(P_nb_va)

    Xbl_tr = v5m.extract_blur33_tile_covs(qv4_tr, P_tr, nobk_members, T_all[tr])
    Xbl_va = v5m.extract_blur33_tile_covs(qv4_va, P_va, nobk_members, T_all[va])
    bl_dim = Xbl_tr.shape[1]
    cal_bl = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 14, 24, bl_dim)).fit(
        Xbl_tr, qv4_tr[:, 0, :], gold[tr, 0], groups[tr], is_orig=is_orig[tr]
    )
    qv5_tr = qv4_tr.copy()
    qv5_tr[:, 0] = cal_bl.predict_proba(Xbl_tr, qv4_tr[:, 0, :])
    qv5_tr[:, 1] = qnb_tr[:, 1]
    qv5_tr = c.renorm(qv5_tr)

    qv5_va = qv4_va.copy()
    qv5_va[:, 0] = cal_bl.predict_proba(Xbl_va, qv4_va[:, 0, :])
    qv5_va[:, 1] = qnb_va[:, 1]
    qv5_va = c.renorm(qv5_va)

    p0_v5, st_v5 = v3m.select_p0_og_inner_cv(qv5_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=lam_cl, p0_base=p0_base)
    pr_v5_va = v3m.predict_og_p0_eu(qv5_va, st_v5, lam_cl=lam_cl, p0_base=p0_base, p0_og=p0_v5)
    pr_v5_va[:, [2, 3, 4, 5]] = pr_v4_va[:, [2, 3, 4, 5]]
    cl_v4 = (pr_v4_va == 0).all(axis=1)
    pr_v5_va[cl_v4] = 0
    nc5 = (~cl_v4) & (pr_v5_va == 0).all(axis=1)
    if np.any(nc5):
        pr_v5_va[nc5] = pr_v4_va[nc5]

    # ---- Stage 4: deconf_v6_boundary (+ bndry_pe on 模糊; pre-LRO (bndry_pe, bndry_dual) on 过度生成) ----
    temps_k = {}
    cal_tr, cal_va = {}, {}
    for nm, arr in P_ex.items():
        tk = v6m.fit_scalar_temp_weighted(arr[tr], gold[tr], rw)
        temps_k[nm] = tk
        cal_tr[nm] = v6m.apply_scalar_temp(arr[tr], tk)
        cal_va[nm] = v6m.apply_scalar_temp(arr[va], tk)

    w_bl = v6m.fit_loglin_blend_dim(qv5_tr[:, 0], [cal_tr["bndry_pe"][:, 0]], gold[tr, 0], rw)
    bl_tr = v6m.apply_loglin_blend_dim(qv5_tr[:, 0], [cal_tr["bndry_pe"][:, 0]], w_bl)
    bl_va = v6m.apply_loglin_blend_dim(qv5_va[:, 0], [cal_va["bndry_pe"][:, 0]], w_bl)

    og_keys = ("bndry_pe", "bndry_dual")
    w_og = v6m.fit_loglin_blend_dim(q_tr_base[:, 4], [cal_tr[nm][:, 4] for nm in og_keys], gold[tr, 4], rw)
    og0_tr = v6m.apply_loglin_blend_dim(q_tr_base[:, 4], [cal_tr[nm][:, 4] for nm in og_keys], w_og)
    og0_va = v6m.apply_loglin_blend_dim(q_va_base[:, 4], [cal_va[nm][:, 4] for nm in og_keys], w_og)

    qb_og_tr = q_tr_base.copy()
    qb_og_tr[:, 4] = og0_tr
    qb_og_va = q_va_base.copy()
    qb_og_va[:, 4] = og0_va
    c_og_v6 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        v4m.extract_og24_tile_covs(qb_og_tr, P_tr, members_v7, S_all[tr], T_all[tr]),
        og0_tr,
        gold[tr, 4],
        groups[tr],
        is_orig=is_orig[tr],
    )
    og_tr = c_og_v6.predict_proba(v4m.extract_og24_tile_covs(qb_og_tr, P_tr, members_v7, S_all[tr], T_all[tr]), og0_tr)
    og_va = c_og_v6.predict_proba(v4m.extract_og24_tile_covs(qb_og_va, P_va, members_v7, S_all[va], T_all[va]), og0_va)

    qv6_tr = qv5_tr.copy()
    qv6_tr[:, 0] = bl_tr
    qv6_tr[:, 4] = og_tr
    qv6_tr = c.renorm(qv6_tr)

    qv6_va = qv5_va.copy()
    qv6_va[:, 0] = bl_va
    qv6_va[:, 4] = og_va
    qv6_va = c.renorm(qv6_va)

    p0_v6, st_v6 = v3m.select_p0_og_inner_cv(qv6_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=lam_cl, p0_base=p0_base)
    pr_v6_va = v3m.predict_og_p0_eu(qv6_va, st_v6, lam_cl=lam_cl, p0_base=p0_base, p0_og=p0_v6)
    pr_v6_va[:, [1, 2, 3, 5]] = pr_v5_va[:, [1, 2, 3, 5]]
    cl_v5 = (pr_v5_va == 0).all(axis=1)
    pr_v6_va[cl_v5] = 0
    nc6 = (~cl_v5) & (pr_v6_va == 0).all(axis=1)
    if np.any(nc6):
        pr_v6_va[nc6] = pr_v5_va[nc6]

    return dict(
        k=k,
        q_va_base=q_va_base, pr_va_base=pr_va_base,
        qv2_va=qv2_va, pr_v2_va=pr_v2_va,
        qv4_va=qv4_va, pr_v4_va=pr_v4_va,
        qv5_va=qv5_va, pr_v5_va=pr_v5_va,
        qv6_va=qv6_va, pr_v6_va=pr_v6_va,
        lam_cl=lam_cl, p0_base=p0_base, p0_v6=float(p0_v6),
        w_bl=w_bl.tolist(), w_og=w_og.tolist(), temps=temps_k,
    )


def main() -> None:
    t0 = time.time()
    set_name = "resident_v8_boundary_novlm"
    members_v7 = list(SETS["resident_v7_novlm"])   # 10 base vision members (no p1_qy)
    members_v8 = list(SETS[set_name])              # 12 vision members (+ bndry_pe, bndry_dual; no p1_qy, no qwen_hr)
    assert "p1_qy" not in members_v8 and "qwen_hr" not in members_v8
    assert members_v8 == members_v7 + ["bndry_pe", "bndry_dual"]
    nobk_members = [m for m in members_v7 if m != "pe_bkseg1"]

    names, gold, fold, _, P_v7 = oof_pool(members_v7)
    _, _, _, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)

    P_ex: Dict[str, np.ndarray] = {}
    for k_ex, d_rel in NOVLM_EXTRA_OOF_DIRS.items():
        arr, valid = load_oof_dir(d_rel, names)
        assert valid.all(), f"Incomplete 8-fold OOF for {k_ex}: {valid.sum()}/{N}"
        P_ex[k_ex] = c.renorm(arr)

    S_oof = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz"), names, "feats")
    T_oof = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_oof_3573.npz"), names, "stats")

    proto_cfg = dc._build_routed_gate_cfg(members_v7)
    proto_cfg["lams_cl"] = LAMS_CL_GRID

    _G.update(
        fold=fold, gold=gold, is_orig=is_orig, groups=groups,
        P=P_v7, P_ex=P_ex, S=S_oof, T=T_oof,
        members=members_v7, nobk_members=nobk_members, proto_cfg=proto_cfg,
    )

    print(f"=== [{set_name}] Zero-Leakage 8-Fold Nested OOF Cascade (N={N}, 12 vision members, 0 VLM) ===", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        f_res = sorted(pool.map(_fold_worker_novlm, list(range(8))), key=lambda d: d["k"])

    stages = ["base", "v2", "v4", "v5", "v6"]
    Qs = {s: np.zeros((N, 6, 4)) for s in stages}
    Prs = {s: np.zeros((N, 6), dtype=np.int64) for s in stages}
    for r in f_res:
        va = fold == r["k"]
        Qs["base"][va], Prs["base"][va] = r["q_va_base"], r["pr_va_base"]
        Qs["v2"][va], Prs["v2"][va] = r["qv2_va"], r["pr_v2_va"]
        Qs["v4"][va], Prs["v4"][va] = r["qv4_va"], r["pr_v4_va"]
        Qs["v5"][va], Prs["v5"][va] = r["qv5_va"], r["pr_v5_va"]
        Qs["v6"][va], Prs["v6"][va] = r["qv6_va"], r["pr_v6_va"]

    reps = {}
    for tag, sk in [
        ("routed_gate_eu (10v novlm)", "base"),
        ("deconf_v2 (10v novlm)", "v2"),
        ("deconf_v4_tile (10v novlm)", "v4"),
        ("deconf_v5_boundary (10v novlm)", "v5"),
        ("deconf_v6_boundary (12v novlm)", "v6"),
    ]:
        reps[sk] = v5m._summarize_all4(tag, Qs[sk], Prs[sk], gold, is_orig)

    fold_lams = [r["lam_cl"] for r in f_res]
    fold_p0b = [r["p0_base"] for r in f_res]
    fold_p0og = [r["p0_v6"] for r in f_res]
    print(f"  fold lam_cl={fold_lams} | p0_base={fold_p0b} | p0_og={fold_p0og}", flush=True)

    # =========================================================================
    # Fit FINAL production bundle on all 3,573 honest OOF predictions (0 test peeking!)
    # =========================================================================
    rw_all = np.where(is_orig, 1.0, 0.25)

    base_dec_full = red.RoutedGateEUDecoder(**proto_cfg).fit(P_v7, gold, is_orig=is_orig, groups=groups)
    Q_raw_all = base_dec_full.pool_from_member_dict(P_v7)
    Q_nobk_base_all = uob.pool_without_bkseg1(base_dec_full, P_v7)

    # v2 components
    X_os_all = uob.extract_os_micro_covariates(Q_raw_all, P_v7, members_v7)
    cal_os_full = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=(2, 4, 8, 11)).fit(
        X_os_all, Q_raw_all[:, 2, :], gold[:, 2], groups, is_orig=is_orig
    )
    X_og_all = dc.extract_deconf_covariates(Q_raw_all, P_v7, 4, members_v7)
    cal_og2_full = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf").fit(
        X_og_all, Q_raw_all[:, 4, :], gold[:, 4], groups, is_orig=is_orig
    )
    cal_bk2_full = uob.BokehSynergyCalibrator(w_supp=0.20).fit(
        Q_nobk_base_all[:, 5, :], c.renorm(P_v7["pe_bkseg1"])[:, 5, :], gold[:, 5], groups, is_orig
    )
    Qv2_full = Q_raw_all.copy()
    Qv2_full[:, 2] = cal_os_full.predict_proba(X_os_all, Q_raw_all[:, 2, :])
    Qv2_full[:, 4] = cal_og2_full.predict_proba(X_og_all, Q_raw_all[:, 4, :])
    Qv2_full[:, 5] = cal_bk2_full.predict_proba(Q_nobk_base_all[:, 5, :], c.renorm(P_v7["pe_bkseg1"])[:, 5, :])
    Qv2_full = c.renorm(Qv2_full)

    eu_v2_full = red.TrueGateConstrainedEU(
        lams_cl=base_dec_full.lams_cl, p0_scales=base_dec_full.eu.p0_scales, w_supp=base_dec_full.eu.w_supp
    ).fit(Qv2_full, gold, is_orig=is_orig, groups=groups)
    lam_full = float(eu_v2_full.lam_cl)
    p0_base_full = float(eu_v2_full.p0_scale)

    v2_bundle = uob.UpgradedDeconfBundle(
        set_name="resident_v7_novlm",
        members=members_v7,
        base_decoder=base_dec_full,
        cal_os=cal_os_full,
        cal_og=cal_og2_full,
        cal_bokeh=cal_bk2_full,
        eu_decoder=eu_v2_full,
    )

    # v4 components
    cal_bk3_full = v3m.BokehSynergyV3(w_supp=0.20).fit(
        Q_nobk_base_all[:, 5, :], c.renorm(P_v7["pe_bkseg1"])[:, 5, :], gold[:, 5], groups, is_orig=is_orig
    )
    X24_all = v4m.extract_og24_tile_covs(Q_raw_all, P_v7, members_v7, S_oof, T_oof)
    cal_og24_full = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        X24_all, Q_raw_all[:, 4, :], gold[:, 4], groups, is_orig=is_orig
    )
    Qv4_full = Q_raw_all.copy()
    Qv4_full[:, 2] = Qv2_full[:, 2]
    Qv4_full[:, 4] = cal_og24_full.predict_proba(X24_all, Q_raw_all[:, 4, :])
    Qv4_full[:, 5] = cal_bk3_full.predict_proba(Q_nobk_base_all[:, 5, :], c.renorm(P_v7["pe_bkseg1"])[:, 5, :])
    Qv4_full = c.renorm(Qv4_full)
    p0_v4_full, st_v4_full = v3m.select_p0_og_inner_cv(
        Qv4_full, gold, groups, is_orig, lam_cl=lam_full, p0_base=p0_base_full
    )
    v4_bundle = v4m.TileSalienceDeconfV4Bundle(
        v2_bundle=v2_bundle,
        cal_og24=cal_og24_full,
        cal_bk3=cal_bk3_full,
        lam_cl=lam_full,
        p0_base=p0_base_full,
        p0_og=p0_v4_full,
        st_v4=st_v4_full,
        v3_fallback=None,
    )

    # v5 components
    nb_cfg_full = dc._build_routed_gate_cfg(nobk_members)
    nb_cfg_full["lams_cl"] = LAMS_CL_GRID
    P_nb_all = {m: P_v7[m] for m in nobk_members}
    dec_nb_full = red.RoutedGateEUDecoder(**nb_cfg_full).fit(P_nb_all, gold, is_orig=is_orig, groups=groups)
    Qnb_all = dec_nb_full.pool_from_member_dict(P_nb_all)

    Xbl_all = v5m.extract_blur33_tile_covs(Qv4_full, P_v7, nobk_members, T_oof)
    cal_bl_full = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 14, 24, Xbl_all.shape[1])).fit(
        Xbl_all, Qv4_full[:, 0, :], gold[:, 0], groups, is_orig=is_orig
    )
    Qv5_full = Qv4_full.copy()
    Qv5_full[:, 0] = cal_bl_full.predict_proba(Xbl_all, Qv4_full[:, 0, :])
    Qv5_full[:, 1] = Qnb_all[:, 1]
    Qv5_full = c.renorm(Qv5_full)
    p0_v5_full, st_v5_full = v3m.select_p0_og_inner_cv(
        Qv5_full, gold, groups, is_orig, lam_cl=lam_full, p0_base=p0_base_full
    )
    v5_bundle = v5m.BoundaryDeconfV5Bundle(
        v4_bundle=v4_bundle,
        dec_nobk=dec_nb_full,
        nobk_members=nobk_members,
        cal_blur33=cal_bl_full,
        lam_cl=lam_full,
        p0_base=p0_base_full,
        p0_og=p0_v5_full,
        st_v5=st_v5_full,
    )

    # v6 components (ONLY bndry_pe & bndry_dual — zero VLM!)
    extra_temps = {nm: v6m.fit_scalar_temp_weighted(arr, gold, rw_all) for nm, arr in P_ex.items()}
    cals_all = {nm: v6m.apply_scalar_temp(arr, extra_temps[nm]) for nm, arr in P_ex.items()}

    w_blur_keys = ("bndry_pe",)
    w_blur = v6m.fit_loglin_blend_dim(Qv5_full[:, 0], [cals_all["bndry_pe"][:, 0]], gold[:, 0], rw_all)

    w_og_keys = ("bndry_pe", "bndry_dual")
    w_og = v6m.fit_loglin_blend_dim(
        Q_raw_all[:, 4], [cals_all[nm][:, 4] for nm in w_og_keys], gold[:, 4], rw_all
    )
    og0_all = v6m.apply_loglin_blend_dim(Q_raw_all[:, 4], [cals_all[nm][:, 4] for nm in w_og_keys], w_og)
    Q_raw_og = Q_raw_all.copy()
    Q_raw_og[:, 4] = og0_all
    cal_og24_v6 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        v4m.extract_og24_tile_covs(Q_raw_og, P_v7, members_v7, S_oof, T_oof),
        og0_all,
        gold[:, 4],
        groups,
        is_orig=is_orig,
    )

    Qv6_full = Qv5_full.copy()
    Qv6_full[:, 0] = v6m.apply_loglin_blend_dim(Qv5_full[:, 0], [cals_all["bndry_pe"][:, 0]], w_blur)
    Qv6_full[:, 4] = cal_og24_v6.predict_proba(
        v4m.extract_og24_tile_covs(Q_raw_og, P_v7, members_v7, S_oof, T_oof), og0_all
    )
    Qv6_full = c.renorm(Qv6_full)
    p0_v6_full, st_v6_full = v3m.select_p0_og_inner_cv(
        Qv6_full, gold, groups, is_orig, lam_cl=lam_full, p0_base=p0_base_full
    )

    bundle = v6m.BoundaryDeconfV6Bundle(
        v5_bundle=v5_bundle,
        members_v8=members_v8,
        extra_temps=extra_temps,
        w_blur_keys=w_blur_keys,
        w_blur=w_blur,
        w_og_keys=w_og_keys,
        w_og=w_og,
        cal_og24_v6=cal_og24_v6,
        lam_cl=lam_full,
        p0_base=p0_base_full,
        p0_og=p0_v6_full,
        st_v6=st_v6_full,
    )
    bundle.set_name = set_name

    # Strict N=1 == N=Batch self-test!
    P_check = dict(P_v7)
    P_check.update(P_ex)
    P_check["_spatial_20d"] = S_oof
    P_check["_tile_peaks_6x12"] = T_oof
    P_sub = {k: v[:16] for k, v in P_check.items()}
    pr_bat, q_bat = bundle.predict(P_sub)
    max_q_diff = 0.0
    n_lbl_diff = 0
    for i in range(16):
        p1, q1 = bundle.predict({k: v[i : i + 1] for k, v in P_sub.items()})
        max_q_diff = max(max_q_diff, float(np.max(np.abs(q1[0] - q_bat[i]))))
        n_lbl_diff += int(np.sum(p1[0] != pr_bat[i]))
    assert max_q_diff < 1e-12 and n_lbl_diff == 0, f"Bundle N=1 drift: {max_q_diff}, {n_lbl_diff}"

    os.makedirs(OUT, exist_ok=True)
    out_pkl = os.path.join(OUT, f"{set_name}_decoder_deconf_v6_boundary.pkl")
    out_json = os.path.join(OUT, f"{set_name}_decoder_deconf_v6_boundary.json")
    with open(out_pkl, "wb") as f:
        pickle.dump(dict(decoder=bundle, members=members_v8, set=set_name, rule="deconf_v6_boundary"), f)

    # Verify final_engine.load_decoder round-trip
    dec_fn, _ = load_decoder(set_name, "deconf_v6_boundary")
    pr_rt, q_rt = dec_fn(P_sub)
    assert np.array_equal(pr_rt, pr_bat) and float(np.max(np.abs(q_rt - q_bat))) < 1e-12

    # Compare vs full v8_boundary Nested OOF if available
    full_v6_json = os.path.join(OUT, "resident_v8_boundary_decoder_deconf_v6_boundary.json")
    full_ref = json.load(open(full_v6_json))["nested_8fold_v6"] if os.path.exists(full_v6_json) else None

    summary = {
        "set": set_name,
        "rule": "deconf_v6_boundary",
        "members": members_v8,
        "lam_cl_full": lam_full,
        "p0_base_full": p0_base_full,
        "p0_og_full": float(p0_v6_full),
        "fold_lam_cl": fold_lams,
        "fold_p0_base": fold_p0b,
        "fold_p0_og": fold_p0og,
        "extra_temps": extra_temps,
        "w_blur_keys": list(w_blur_keys),
        "w_blur": [float(x) for x in w_blur],
        "w_og_keys": list(w_og_keys),
        "w_og": [float(x) for x in w_og],
        "og24_v6_cfg": {"alpha": float(cal_og24_v6.alpha_), "topk": int(cal_og24_v6.best_topk_), "l2": float(cal_og24_v6.best_l2_)},
        "n1_batch_invariance": {"max_q_diff": max_q_diff, "label_diffs": n_lbl_diff},
        "nested_8fold_cascade": reps,
        "full_v8_boundary_nested_8fold_ref": full_ref,
        "vlm_ablation_oof_delta": (
            {
                "doc2_orig": round(reps["v6"]["doc2_orig"] - full_ref["doc2_orig"], 4),
                "doc1_orig": round(reps["v6"]["doc1_orig"] - full_ref["doc1_orig"], 4),
                "doc2_all": round(reps["v6"]["doc2_all"] - full_ref["doc2_all"], 4),
                "rps_orig": round(reps["v6"]["rps_orig"] - full_ref["rps_orig"], 6),
            }
            if full_ref is not None
            else None
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"\n[SAVED] {out_pkl}\n"
        f"  lam_cl={lam_full:.2f} | p0_base={p0_base_full:.2f} | p0_og={p0_v6_full:.2f}\n"
        f"  extra_temps={ {k: round(v, 3) for k, v in extra_temps.items()} }\n"
        f"  w_blur={np.round(w_blur, 4).tolist()} | w_og={np.round(w_og, 4).tolist()}\n"
        f"  N=1 == N=Batch check: max_q_diff={max_q_diff:.2e}, label_diffs={n_lbl_diff}/96 | elapsed={time.time()-t0:.1f}s",
        flush=True,
    )
    if full_ref is not None:
        print(
            f"\n=== VLM Ablation (8-Fold Zero-Leakage Nested OOF) ===\n"
            f"  Full v8_boundary (14m, +2 VLM): doc2_orig={full_ref['doc2_orig']:.4f} | doc1_orig={full_ref['doc1_orig']:.4f} | doc2_all={full_ref['doc2_all']:.4f} | rps_orig={full_ref['rps_orig']:.6f}\n"
            f"  No-VLM v8_boundary_novlm (12m): doc2_orig={reps['v6']['doc2_orig']:.4f} | doc1_orig={reps['v6']['doc1_orig']:.4f} | doc2_all={reps['v6']['doc2_all']:.4f} | rps_orig={reps['v6']['rps_orig']:.6f}\n"
            f"  Delta (No-VLM - Full)         : doc2_orig={reps['v6']['doc2_orig']-full_ref['doc2_orig']:+.4f} | doc1_orig={reps['v6']['doc1_orig']-full_ref['doc1_orig']:+.4f} | doc2_all={reps['v6']['doc2_all']-full_ref['doc2_all']:+.4f} | rps_orig={reps['v6']['rps_orig']-full_ref['rps_orig']:+.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
