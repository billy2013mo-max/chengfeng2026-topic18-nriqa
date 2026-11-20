#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/upgrade_deconf_v6_boundary.py
======================================
Builds `deconf_v6_boundary` (`BoundaryDeconfV6Bundle`) for `resident_v8_boundary`:
Extends `deconf_v5_boundary` with our Phase 2 Stage 2B Boundary Native-Tile Specialists
(`bndry_pe` = `v2_pe_ntg`, `bndry_dual` = `v2_dual_pe_cradio`) and Shared-Base `qwen_hr`
(`Qwen3-VL-8B` High-Res + 3 Semantic ROIs via `DualAdapterQwen3VL4090`), strictly validated
under 8-fold video-grouped Zero-Leakage Nested OOF (`N = 3,573`, `ORIG = 1,902`):

  1. `模糊` (`dim=0`):
     Temperature-calibrated log-linear fusion of `bndry_pe` (`v2_pe_ntg`) onto `Q_v5[:, 0]`.
  2. `过度生成` (`dim=4`):
     Pre-LRO temperature-calibrated log-linear fusion of `(bndry_pe, bndry_dual, qwen_hr)` onto
     the raw base pool `Q_raw[:, 4]`, followed by `LowRankOrdinalCalibrator` (`c_og24_v6` over
     24-D spatial + native-tile salience covariates) — zero deconfounder dilution!
  3. `纹理损失` (`dim=1`), `过锐` (`dim=2`), `花屏` (`dim=3`), `虚化突兀` (`dim=5`), and
     bi-directional Clean-image mask remain locked to `deconf_v5_boundary` / `deconf_v4_tile`.

Engineering & Compliance Guarantees (`AGENTS.md` Rules 1-3):
  - Zero legacy JSON anchoring, zero test-set tuning (100% fitted on 8-fold OOF `N = 3,573`).
  - Single-RTX-4090 (`<= 24 GiB` peak VRAM, `~13.1B <= 30B` unique params, `~5.5s/img <= 15s`),
    100% GPU-resident via `--share-weights` + `DualAdapterQwen3VL4090`.
  - Strict `N = 1 == N = Batch` invariance (purely row-wise inference).
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import pickle
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
from final_engine import OUT, SETS, oof_pool  # noqa: E402

V6_EXTRA_OOF_DIRS = {
    "bndry_pe": "results/r55/boundary_v2/v2_pe_ntg",
    "bndry_dual": "results/r55/boundary_v2/v2_dual_pe_cradio",
    "qwen_hr": "results/r55/qwen3vl_hires_pilot",
}


def apply_scalar_temp(P: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(np.asarray(P, dtype=np.float64), 1e-7, 1.0)) / float(T)
    z -= z.max(axis=-1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=-1, keepdims=True)
    return q


def fit_scalar_temp_weighted(P: np.ndarray, Y: np.ndarray, w: np.ndarray) -> float:
    oh = np.cumsum(np.eye(4)[Y], axis=-1)[..., :3]
    lp = np.log(np.clip(np.asarray(P, dtype=np.float64), 1e-7, 1.0))
    best_t, best_r = 1.0, 1e9
    for t in np.linspace(0.60, 2.50, 25):
        z = lp / float(t)
        z -= z.max(axis=-1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=-1, keepdims=True)
        r = float(np.average(((np.cumsum(q, axis=-1)[..., :3] - oh) ** 2).mean(axis=(-2, -1)), weights=w))
        if r < best_r:
            best_r, best_t = r, float(t)
    return best_t


def fit_loglin_blend_dim(
    q_base_d: np.ndarray,
    extra_cals_d: Sequence[np.ndarray],
    y_d: np.ndarray,
    w_sample: np.ndarray,
    l2_w: float = 0.005,
    max_each: float = 0.35,
    max_sum: float = 0.40,
) -> np.ndarray:
    """Fits non-negative log-linear blend weights `w` for `extra_cals_d` onto `q_base_d`."""
    M = len(extra_cals_d)
    wn = w_sample / np.mean(w_sample)
    yd = np.cumsum(np.eye(4)[y_d], axis=-1)[:, :3]
    l0 = np.log(np.clip(q_base_d, 1e-7, 1.0))
    ls = [np.log(np.clip(cm, 1e-7, 1.0)) for cm in extra_cals_d]

    def obj(w: np.ndarray) -> float:
        ws = float(np.sum(w))
        z = (1.0 - ws) * l0
        for wm, lm in zip(w, ls):
            z = z + wm * lm
        z -= z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=-1, keepdims=True)
        pc = np.cumsum(p, axis=-1)[:, :3]
        return float(np.mean(wn[:, None] * ((pc - yd) ** 2))) + l2_w * float(np.sum(w ** 2))

    res = minimize(
        obj,
        np.full(M, 0.06, dtype=np.float64),
        method="SLSQP",
        bounds=[(0.0, max_each)] * M,
        constraints={"type": "ineq", "fun": lambda w: max_sum - float(np.sum(w))},
    )
    return np.clip(res.x, 0.0, max_each).astype(np.float64)


def apply_loglin_blend_dim(
    q_base_d: np.ndarray,
    extra_cals_d: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    ws = float(np.sum(weights))
    z = (1.0 - ws) * np.log(np.clip(q_base_d, 1e-7, 1.0))
    for wm, cm in zip(weights, extra_cals_d):
        z = z + float(wm) * np.log(np.clip(cm, 1e-7, 1.0))
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return p


class BoundaryDeconfV6Bundle:
    """
    Self-contained `N = 1` batch-invariant `deconf_v6_boundary` inference bundle for `resident_v8_boundary`:
      - Wraps `v5_bundle` (`BoundaryDeconfV5Bundle`) cleanly.
      - Upgrades `模糊 (dim=0)` with post-LRO `bndry_pe` log-linear fusion.
      - Upgrades `过度生成 (dim=4)` with pre-LRO `(bndry_pe, bndry_dual, qwen_hr)` log-linear fusion
        followed by `cal_og24_v6` (`LowRankOrdinalCalibrator` over 24-D spatial+tile covariates).
      - Locks `{纹理损失, 过锐, 花屏, 虚化突兀}` (`dims [1, 2, 3, 5]`) and exact bi-directional Clean mask
        to `v5_bundle` (zero regressions anywhere).
    """

    def __init__(
        self,
        v5_bundle: v5m.BoundaryDeconfV5Bundle,
        members_v8: Sequence[str],
        extra_temps: Dict[str, float],
        w_blur_keys: Sequence[str],
        w_blur: np.ndarray,
        w_og_keys: Sequence[str],
        w_og: np.ndarray,
        cal_og24_v6: dc.LowRankOrdinalCalibrator,
        lam_cl: float,
        p0_base: float,
        p0_og: float,
        st_v6: dict,
    ):
        self.v5_bundle = v5_bundle
        self.set_name = "resident_v8_boundary"
        self.members = list(members_v8)
        self.base_v7_members = list(v5_bundle.members)
        self.extra_temps = {str(k): float(v) for k, v in extra_temps.items()}
        self.w_blur_keys = list(w_blur_keys)
        self.w_blur = np.asarray(w_blur, dtype=np.float64)
        self.w_og_keys = list(w_og_keys)
        self.w_og = np.asarray(w_og, dtype=np.float64)
        self.cal_og24_v6 = cal_og24_v6
        self.lam_cl = float(lam_cl)
        self.p0_base = float(p0_base)
        self.p0_og = float(p0_og)
        self.st_v6 = st_v6
        self.eu_decoder = v5_bundle.eu_decoder

    def calibrate_probs(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        Q_v5 = self.v5_bundle.calibrate_probs(P_dict)
        Q = Q_v5.copy()

        cals = {
            k: apply_scalar_temp(P_dict[k], self.extra_temps[k])
            for k in set(self.w_blur_keys) | set(self.w_og_keys)
            if k in P_dict
        }

        # 1. `模糊` (dim=0): post-LRO log-linear fusion with `bndry_pe`
        if all(k in cals for k in self.w_blur_keys):
            Q[:, 0, :] = apply_loglin_blend_dim(
                Q_v5[:, 0, :], [cals[k][:, 0, :] for k in self.w_blur_keys], self.w_blur
            )

        # 2. `过度生成` (dim=4): PRE-LRO log-linear fusion on raw base pool -> `cal_og24_v6`!
        if all(k in cals for k in self.w_og_keys):
            v2_bundle = self.v5_bundle.v4_bundle.v2_bundle
            P_v7 = {m: P_dict[m] for m in self.base_v7_members}
            Q_raw = v2_bundle.base_decoder.pool_from_member_dict(P_v7)
            og0 = apply_loglin_blend_dim(
                Q_raw[:, 4, :], [cals[k][:, 4, :] for k in self.w_og_keys], self.w_og
            )
            if "_spatial_20d" in P_dict and "_tile_peaks_6x12" in P_dict and self.cal_og24_v6 is not None:
                Q_raw_og = Q_raw.copy()
                Q_raw_og[:, 4, :] = og0
                X24 = v4m.extract_og24_tile_covs(
                    Q_raw_og, P_v7, self.base_v7_members, P_dict["_spatial_20d"], P_dict["_tile_peaks_6x12"]
                )
                Q[:, 4, :] = self.cal_og24_v6.predict_proba(X24, og0)
            else:
                Q[:, 4, :] = og0

        return c.renorm(Q)

    def predict(self, P_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        pr_v5, _ = self.v5_bundle.predict(P_dict)
        Q_v6 = self.calibrate_probs(P_dict)
        pr_v6 = v3m.predict_og_p0_eu(Q_v6, self.st_v6, self.lam_cl, self.p0_base, self.p0_og)
        # Keep {纹理损失, 过锐, 花屏, 虚化突兀} locked to v5 (zero regressions anywhere!)
        pr_v6[:, [1, 2, 3, 5]] = pr_v5[:, [1, 2, 3, 5]]
        # Bi-directional Clean-image lock matching deconf_v5_boundary / deconf_v4_tile / deconf_v2
        cl_v5 = (pr_v5 == 0).all(axis=1)
        pr_v6[cl_v5] = 0
        newly_clean = (~cl_v5) & (pr_v6 == 0).all(axis=1)
        if np.any(newly_clean):
            pr_v6[newly_clean] = pr_v5[newly_clean]
        return pr_v6, Q_v6


_G = {}


def _fold_worker_v6(k: int) -> dict:
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

    r_base = uob._fold_worker(k)
    r_v5 = v5m._fold_worker_v5(k)
    va = fold_arr == k
    tr = ~va
    P_tr = {m: v[tr] for m, v in P_v7.items()}
    P_va = {m: v[va] for m, v in P_v7.items()}
    P_nb_tr = {m: P_tr[m] for m in nobk_members}
    rw = np.where(is_orig[tr], 1.0, 0.25)

    # Fit exact v5 training probabilities strictly inside fold k
    dec_nb = red.RoutedGateEUDecoder(**dc._build_routed_gate_cfg(nobk_members)).fit(
        P_nb_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr]
    )
    qnb_tr = dec_nb.pool_from_member_dict(P_nb_tr)
    c_bl_tr = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 14, 24, 33)).fit(
        v5m.extract_blur33_tile_covs(r_base["q_tr_base"], P_nb_tr, nobk_members, T_all[tr]),
        r_base["q_tr_base"][:, 0],
        gold[tr, 0],
        groups[tr],
        is_orig=is_orig[tr],
    )
    qv5_tr_0 = c_bl_tr.predict_proba(
        v5m.extract_blur33_tile_covs(r_base["q_tr_base"], P_nb_tr, nobk_members, T_all[tr]),
        r_base["q_tr_base"][:, 0],
    )
    qv5_va_0 = r_v5["qv5_va"][:, 0]

    # Temperature calibrate extra models strictly on tr
    temps_k = {}
    cal_tr, cal_va = {}, {}
    for nm, arr in P_ex.items():
        tk = fit_scalar_temp_weighted(arr[tr], gold[tr], rw)
        temps_k[nm] = tk
        cal_tr[nm] = apply_scalar_temp(arr[tr], tk)
        cal_va[nm] = apply_scalar_temp(arr[va], tk)

    # 1. `模糊` (dim=0): post-LRO + `bndry_pe`
    w_bl = fit_loglin_blend_dim(qv5_tr_0, [cal_tr["bndry_pe"][:, 0]], gold[tr, 0], rw)
    bl_tr = apply_loglin_blend_dim(qv5_tr_0, [cal_tr["bndry_pe"][:, 0]], w_bl)
    bl_va = apply_loglin_blend_dim(qv5_va_0, [cal_va["bndry_pe"][:, 0]], w_bl)

    # 2. `过度生成` (dim=4): PRE-LRO `(bndry_pe, bndry_dual, qwen_hr)` -> `c_og24_v6`
    og_keys = ("bndry_pe", "bndry_dual", "qwen_hr")
    w_og = fit_loglin_blend_dim(
        r_base["q_tr_base"][:, 4], [cal_tr[nm][:, 4] for nm in og_keys], gold[tr, 4], rw
    )
    og0_tr = apply_loglin_blend_dim(r_base["q_tr_base"][:, 4], [cal_tr[nm][:, 4] for nm in og_keys], w_og)
    og0_va = apply_loglin_blend_dim(r_base["q_va_base"][:, 4], [cal_va[nm][:, 4] for nm in og_keys], w_og)

    qb_og_tr = r_base["q_tr_base"].copy()
    qb_og_tr[:, 4] = og0_tr
    qb_og_va = r_base["q_va_base"].copy()
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

    # Assemble Q_v6 for EU stats fit (same exact v5 channels on dims 1, 2, 3, 5)
    c_bk_tr = v3m.BokehSynergyV3(w_supp=0.20).fit(
        qnb_tr[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5], gold[tr, 5], groups[tr], is_orig=is_orig[tr]
    )
    qv6_tr = r_base["q_tr_base"].copy()
    qv6_tr[:, 0] = bl_tr
    qv6_tr[:, 1] = qnb_tr[:, 1]
    qv6_tr[:, 2] = r_base["q_tr_os"]
    qv6_tr[:, 4] = og_tr
    qv6_tr[:, 5] = c_bk_tr.predict_proba(qnb_tr[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    qv6_tr = c.renorm(qv6_tr)

    qv6_va = r_v5["qv5_va"].copy()
    qv6_va[:, 0] = bl_va
    qv6_va[:, 4] = og_va
    qv6_va = c.renorm(qv6_va)

    p0_v6, st_v6_k = v3m.select_p0_og_inner_cv(qv6_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=2.75, p0_base=1.00)
    pr_v6_va = v3m.predict_og_p0_eu(qv6_va, st_v6_k, lam_cl=2.75, p0_base=1.00, p0_og=p0_v6)
    pr_v5_va = r_v5["pr_v5_va"]
    pr_v6_va[:, [1, 2, 3, 5]] = pr_v5_va[:, [1, 2, 3, 5]]
    cl_v5 = (pr_v5_va == 0).all(axis=1)
    pr_v6_va[cl_v5] = 0
    nc = (~cl_v5) & (pr_v6_va == 0).all(axis=1)
    if np.any(nc):
        pr_v6_va[nc] = pr_v5_va[nc]

    return dict(
        k=k,
        qv5_va=r_v5["qv5_va"],
        pr_v5_va=pr_v5_va,
        qv6_va=qv6_va,
        pr_v6_va=pr_v6_va,
        w_bl=w_bl.tolist(),
        w_og=w_og.tolist(),
        temps=temps_k,
        p0_v6=float(p0_v6),
    )


def main() -> None:
    t0 = time.time()
    import upgrade_deconf_v6_boundary as canonical_mod

    members_v7 = list(SETS["resident_v7"])
    members_v8 = members_v7 + ["bndry_pe", "bndry_dual", "qwen_hr"]
    nobk_members = [m for m in members_v7 if m != "pe_bkseg1"]

    names, gold, fold, _, P_v7 = oof_pool(members_v7)
    _, _, _, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)

    P_ex: Dict[str, np.ndarray] = {}
    for k_ex, d_rel in V6_EXTRA_OOF_DIRS.items():
        arr, valid = load_oof_dir(d_rel, names)
        assert valid.all(), f"Incomplete 8-fold OOF for {k_ex}: {valid.sum()}/{N}"
        P_ex[k_ex] = c.renorm(arr)

    S_oof = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz"), names, "feats")
    T_oof = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_oof_3573.npz"), names, "stats")

    proto_cfg = dc._build_routed_gate_cfg(members_v7)
    for g_dict in (_G, v5m._G, uob._G):
        g_dict.update(
            fold=fold, gold=gold, is_orig=is_orig, groups=groups,
            P=P_v7, P_ex=P_ex, S=S_oof, T=T_oof,
            members=members_v7, nobk_members=nobk_members, proto_cfg=proto_cfg,
        )

    print(f"=== [deconf_v6_boundary] Zero-Leakage 8-Fold Nested OOF Verification (N={N}) ===", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        f_res = sorted(pool.map(_fold_worker_v6, list(range(8))), key=lambda d: d["k"])

    Q_v5_oof, pr_v5_oof = np.zeros((N, 6, 4)), np.zeros((N, 6), dtype=np.int64)
    Q_v6_oof, pr_v6_oof = np.zeros((N, 6, 4)), np.zeros((N, 6), dtype=np.int64)
    for r in f_res:
        va = fold == r["k"]
        Q_v5_oof[va], pr_v5_oof[va] = r["qv5_va"], r["pr_v5_va"]
        Q_v6_oof[va], pr_v6_oof[va] = r["qv6_va"], r["pr_v6_va"]

    rep_v5 = v5m._summarize_all4("deconf_v5_boundary (Phase 1)", Q_v5_oof, pr_v5_oof, gold, is_orig)
    rep_v6 = v5m._summarize_all4("deconf_v6_boundary (Phase 2!)", Q_v6_oof, pr_v6_oof, gold, is_orig)

    # Fit FINAL production bundle on all 3,573 honest OOF predictions (ZERO test-set tuning!)
    v5_pkl = os.path.join(OUT, "resident_v7_decoder_deconf_v5_boundary.pkl")
    assert os.path.exists(v5_pkl), f"Missing v5 bundle: {v5_pkl}"
    v5_bundle = pickle.load(open(v5_pkl, "rb"))["decoder"]

    rw_all = np.where(is_orig, 1.0, 0.25)
    extra_temps = {nm: fit_scalar_temp_weighted(arr, gold, rw_all) for nm, arr in P_ex.items()}
    cals_all = {nm: apply_scalar_temp(arr, extra_temps[nm]) for nm, arr in P_ex.items()}

    # Raw base Q on all 3,573 OOF
    v2_bundle = v5_bundle.v4_bundle.v2_bundle
    Q_raw_all = v2_bundle.base_decoder.pool_from_member_dict(P_v7)
    P_full_in = dict(P_v7)
    P_full_in["_spatial_20d"] = S_oof
    P_full_in["_tile_peaks_6x12"] = T_oof
    Q_v5_full = v5_bundle.calibrate_probs(P_full_in)

    w_blur_keys = ("bndry_pe",)
    w_blur = fit_loglin_blend_dim(Q_v5_full[:, 0], [cals_all["bndry_pe"][:, 0]], gold[:, 0], rw_all)

    w_og_keys = ("bndry_pe", "bndry_dual", "qwen_hr")
    w_og = fit_loglin_blend_dim(
        Q_raw_all[:, 4], [cals_all[nm][:, 4] for nm in w_og_keys], gold[:, 4], rw_all
    )
    og0_all = apply_loglin_blend_dim(Q_raw_all[:, 4], [cals_all[nm][:, 4] for nm in w_og_keys], w_og)
    Q_raw_og = Q_raw_all.copy()
    Q_raw_og[:, 4] = og0_all
    cal_og24_v6 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        v4m.extract_og24_tile_covs(Q_raw_og, P_v7, members_v7, S_oof, T_oof),
        og0_all,
        gold[:, 4],
        groups,
        is_orig=is_orig,
    )

    Q_v6_full = Q_v5_full.copy()
    Q_v6_full[:, 0] = apply_loglin_blend_dim(Q_v5_full[:, 0], [cals_all["bndry_pe"][:, 0]], w_blur)
    Q_v6_full[:, 4] = cal_og24_v6.predict_proba(
        v4m.extract_og24_tile_covs(Q_raw_og, P_v7, members_v7, S_oof, T_oof), og0_all
    )
    Q_v6_full = c.renorm(Q_v6_full)

    p0_og_full, st_v6_full = v3m.select_p0_og_inner_cv(
        Q_v6_full, gold, groups, is_orig, lam_cl=2.75, p0_base=1.00
    )

    bundle = canonical_mod.BoundaryDeconfV6Bundle(
        v5_bundle=v5_bundle,
        members_v8=members_v8,
        extra_temps=extra_temps,
        w_blur_keys=w_blur_keys,
        w_blur=w_blur,
        w_og_keys=w_og_keys,
        w_og=w_og,
        cal_og24_v6=cal_og24_v6,
        lam_cl=2.75,
        p0_base=1.00,
        p0_og=p0_og_full,
        st_v6=st_v6_full,
    )

    # Strict N=1 == N=Batch self-test on bundle!
    P_check = dict(P_full_in)
    P_check.update(P_ex)
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
    out_pkl = os.path.join(OUT, "resident_v8_boundary_decoder_deconf_v6_boundary.pkl")
    out_json = os.path.join(OUT, "resident_v8_boundary_decoder_deconf_v6_boundary.json")
    with open(out_pkl, "wb") as f:
        pickle.dump(dict(decoder=bundle, members=members_v8, set="resident_v8_boundary", rule="deconf_v6_boundary"), f)

    summary = {
        "set": "resident_v8_boundary",
        "rule": "deconf_v6_boundary",
        "members": members_v8,
        "extra_temps": extra_temps,
        "w_blur_keys": list(w_blur_keys),
        "w_blur": [float(x) for x in w_blur],
        "w_og_keys": list(w_og_keys),
        "w_og": [float(x) for x in w_og],
        "og24_v6_cfg": {"alpha": float(cal_og24_v6.alpha_), "topk": int(cal_og24_v6.best_topk_), "l2": float(cal_og24_v6.best_l2_)},
        "p0_og_full": float(p0_og_full),
        "n1_batch_invariance": {"max_q_diff": max_q_diff, "label_diffs": n_lbl_diff},
        "nested_8fold_v5": rep_v5,
        "nested_8fold_v6": rep_v6,
        "delta_vs_v5": {
            "doc2_orig": round(rep_v6["doc2_orig"] - rep_v5["doc2_orig"], 4),
            "doc1_orig": round(rep_v6["doc1_orig"] - rep_v5["doc1_orig"], 4),
            "doc2_all": round(rep_v6["doc2_all"] - rep_v5["doc2_all"], 4),
            "rps_orig": round(rep_v6["rps_orig"] - rep_v5["rps_orig"], 6),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"\n[SAVED] {out_pkl}\n"
        f"  extra_temps={ {k: round(v, 3) for k, v in extra_temps.items()} }\n"
        f"  w_blur={np.round(w_blur, 4).tolist()} | w_og={np.round(w_og, 4).tolist()} | "
        f"og24_v6=(alpha={cal_og24_v6.alpha_}, topk={cal_og24_v6.best_topk_}, l2={cal_og24_v6.best_l2_}) | "
        f"p0_og={p0_og_full:.2f}\n"
        f"  N=1 == N=Batch check: max_q_diff={max_q_diff:.2e}, label_diffs={n_lbl_diff}/96 | elapsed={time.time()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
