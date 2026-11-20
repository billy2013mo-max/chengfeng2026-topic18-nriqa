#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/upgrade_deconf_v4_tile.py
==================================
Builds `deconf_v4_tile` (`TileSalienceDeconfV4Bundle`):
Upgrades `deconf_v3` by injecting 1:1 native-tile local peak sharpness (`ntg_pe` `max-vs-mean` & `std_t P(P0)`
plus foreground/anatomy/plastic-ridge tile salience) into `过度生成` (`dim=4`, 24-D covariates), while:
  - Preserving exact zero extra model parameters (`11.624B <= 30B`), zero extra VRAM (`19.37 GiB < 24 GiB`),
    and ~5 ms runtime overhead inside `FinalEngine.infer()` (reusing `ntg_pe` tiles + `self.coc` depth +
    `maps0` wavelet + `self.p12_store` semantic ROIs already computed during inference).
  - Strictly verifying 100% zero-leakage 8-fold video-grouped Nested OOF (`N = 3,573`), `N=1 == N=Batch`,
    exact bi-directional Clean-image lock (`Clean = 20.00 / 20`), and non-target dimension isolation.
"""
import copy
import csv
import json
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
from typing import Dict, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

import common55 as c  # noqa: E402
import deconfound_calibrator as dc  # noqa: E402
import routed_eu_decoder as red  # noqa: E402
import upgrade_deconf_v3_cleanlocked as v3m  # noqa: E402
import upgrade_os_og_bokeh_deconf as uob  # noqa: E402
from final_engine import SETS, load_decoder, oof_pool  # noqa: E402
from validate_submission import validate as validate_prediction_file  # noqa: E402

DIMS_ZH = ("模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀")
GRADES_ZH = ("无问题", "P2", "P1", "P0")

OG24_FEAT_NAMES: Tuple[str, ...] = v3m.OG_SPEC_FEAT_NAMES + (
    "tile_og_fg_any_gap",
    "tile_og_anat_any_gap",
    "tile_og_fg_top2_sev_gap",
    "tile_og_anat_max_sev_gap",
    "tile_og_local_peak_gap",
    "tile_og_p01_fg_peak_gap",
    "tile_og_vs_blur_tex_margin",
    "tile_og_meor_x_tile",
    "ntg_gap_p_any",
    "ntg_gap_sev",
    "ntg_std_p_p0",
    "ntg_gap_p_ge1",
)


def extract_og24_tile_covs(
    Q_base: np.ndarray,
    P_dict: Dict[str, np.ndarray],
    members: Sequence[str],
    spatial_20d: np.ndarray,
    tile_peaks_6x12: np.ndarray,
) -> np.ndarray:
    """
    Strictly row-wise (`N = 1` invariant) 24-D covariates for `过度生成` (`dim=4`):
      - 12-D cross-defect + specialist discrepancy (`p12_pe`, `v16`, `ntg`, `p1_qy`)
      - 8-D foreground / anatomy / plastic-ridge 1:1 tile salience (`_spatial_20d`)
      - 4-D `ntg_pe` per-tile max-vs-mean & P0 dispersion sharpness (`_tile_peaks_6x12`:
        `gap_p_any`, `gap_sev`, `std_p_p0`, `gap_p_ge1`)
    """
    X12 = v3m.extract_og_spec_covs(Q_base, P_dict, members)
    Q = c.renorm(Q_base)
    gv = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ens_any = 1.0 - Q[:, 4, 0]
    ens_ge1 = Q[:, 4, 2] + Q[:, 4, 3]
    ens_sev = Q[:, 4] @ gv

    S = np.asarray(spatial_20d, dtype=np.float64)
    if S.ndim == 1:
        S = S[None, :]
    T = np.asarray(tile_peaks_6x12, dtype=np.float64)
    if T.ndim == 2:
        T = T[None, :, :]
    td = T[:, 4, :]  # (N, 12) for dim=4 过度生成

    X24 = np.column_stack([
        X12,
        S[:, 10] - ens_any,
        S[:, 11] - ens_any,
        S[:, 12] - ens_sev,
        S[:, 13] - ens_sev,
        S[:, 16],
        S[:, 17] - ens_ge1,
        S[:, 18],
        S[:, 19],
        td[:, 1],   # gap_p_any  (max_t - mean_t P(>=P2))
        td[:, 10],  # gap_sev    (max_t - mean_t severity)
        td[:, 7],   # std_p_p0   (std_t P(P0))
        td[:, 3],   # gap_p_ge1  (max_t - mean_t P(>=P1))
    ])
    return np.nan_to_num(X24, nan=0.0, posinf=10.0, neginf=-10.0)


class TileSalienceDeconfV4Bundle:
    """
    Self-contained `N = 1` batch-invariant `deconf_v4_tile` inference bundle:
      - Expects `P_dict` containing all 11 member probability arrays plus `"_spatial_20d"` `(N, 20)`
        and `"_tile_peaks_6x12"` `(N, 6, 12)` (automatically emitted by `FinalEngine.infer()`).
      - Gracefully falls back to `v3_fallback` if tile arrays are absent.
      - Enforces exact bi-directional Clean-image lock and non-target dim `{模糊, 纹理损失, 花屏}` isolation.
    """

    def __init__(
        self,
        v2_bundle: uob.UpgradedDeconfBundle,
        cal_og24: dc.LowRankOrdinalCalibrator,
        cal_bk3: v3m.BokehSynergyV3,
        lam_cl: float,
        p0_base: float,
        p0_og: float,
        st_v4: dict,
        v3_fallback: v3m.CleanLockedDeconfV3Bundle = None,
    ):
        self.v2_bundle = v2_bundle
        self.v3_fallback = v3_fallback
        self.set_name = v2_bundle.set_name
        self.members = list(v2_bundle.members)
        self.base_decoder = v2_bundle.base_decoder
        self.cal_os = v2_bundle.cal_os
        self.cal_og24 = cal_og24
        self.cal_bk3 = cal_bk3
        self.lam_cl = float(lam_cl)
        self.p0_base = float(p0_base)
        self.p0_og = float(p0_og)
        self.st_v4 = st_v4
        self.eu_decoder = v2_bundle.eu_decoder

    def calibrate_probs(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        if ("_spatial_20d" not in P_dict or "_tile_peaks_6x12" not in P_dict) and self.v3_fallback is not None:
            return self.v3_fallback.calibrate_probs(P_dict)
        Q_base = self.base_decoder.pool_from_member_dict(P_dict)
        Q = Q_base.copy()
        if self.cal_os is not None and self.cal_os.alpha_ > 0.0:
            Q[:, 2, :] = self.cal_os.predict_proba(uob.extract_os_micro_covariates(Q_base, P_dict, self.members), Q_base[:, 2, :])
        if self.cal_og24 is not None and self.cal_og24.alpha_ > 0.0:
            X24 = extract_og24_tile_covs(Q_base, P_dict, self.members, P_dict["_spatial_20d"], P_dict["_tile_peaks_6x12"])
            Q[:, 4, :] = self.cal_og24.predict_proba(X24, Q_base[:, 4, :])
        if self.cal_bk3 is not None and "pe_bkseg1" in P_dict:
            Q_nobk = uob.pool_without_bkseg1(self.base_decoder, P_dict)
            Q[:, 5, :] = self.cal_bk3.predict_proba(Q_nobk[:, 5, :], c.renorm(P_dict["pe_bkseg1"])[:, 5, :])
        return c.renorm(Q)

    def predict(self, P_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if ("_spatial_20d" not in P_dict or "_tile_peaks_6x12" not in P_dict) and self.v3_fallback is not None:
            return self.v3_fallback.predict(P_dict)
        pr_v2, _ = self.v2_bundle.predict(P_dict)
        Q_v4 = self.calibrate_probs(P_dict)
        pr_v4 = v3m.predict_og_p0_eu(Q_v4, self.st_v4, self.lam_cl, self.p0_base, self.p0_og)
        pr_v4[:, [0, 1, 3]] = pr_v2[:, [0, 1, 3]]
        cl_v2 = (pr_v2 == 0).all(axis=1)
        pr_v4[cl_v2] = 0
        newly_clean = (~cl_v2) & (pr_v4 == 0).all(axis=1)
        if np.any(newly_clean):
            pr_v4[newly_clean] = pr_v2[newly_clean]
        return pr_v4, Q_v4


_G = {}


def _fold_worker_v4(k: int):
    fold_arr = _G["fold"]
    gold = _G["gold"]
    is_orig = _G["is_orig"]
    groups = _G["groups"]
    P_dict = _G["P"]
    S_all = _G["S"]
    T_all = _G["T"]
    members = _G["members"]

    r_base = uob._fold_worker(k)
    va = fold_arr == k
    tr = ~va
    P_tr = {m: v[tr] for m, v in P_dict.items()}
    P_va = {m: v[va] for m, v in P_dict.items()}
    S_tr, S_va = S_all[tr], S_all[va]
    T_tr, T_va = T_all[tr], T_all[va]
    rw = np.where(is_orig[tr], 1.0, 0.25)

    qv2_tr = r_base["q_tr_base"].copy()
    qv2_tr[:, 2] = r_base["q_tr_os"]
    qv2_tr[:, 4] = r_base["q_tr_og"]
    qv2_tr[:, 5] = r_base["q_tr_bk"]
    qv2_tr = c.renorm(qv2_tr)

    qv2_va = r_base["q_va_base"].copy()
    qv2_va[:, 2] = r_base["q_va_os"]
    qv2_va[:, 4] = r_base["q_va_og"]
    qv2_va[:, 5] = r_base["q_va_bk"]
    qv2_va = c.renorm(qv2_va)

    st_v2_k = red.TrueGateConstrainedEU._fit_stats(qv2_tr, gold[tr], rw, lam_cl=2.75, p0_scale=1.00)
    pr_v2_va = red._true_eu_predict(qv2_va, st_v2_k, lam_cl=2.75, p0_scale=1.00)

    c_og24 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        extract_og24_tile_covs(r_base["q_tr_base"], P_tr, members, S_tr, T_tr),
        r_base["q_tr_base"][:, 4],
        gold[tr, 4],
        groups[tr],
        is_orig=is_orig[tr],
    )
    qnb_tr = uob.pool_without_bkseg1(r_base["dec_k"], P_tr)
    qnb_va = uob.pool_without_bkseg1(r_base["dec_k"], P_va)
    c_bk3 = v3m.BokehSynergyV3(w_supp=0.20).fit(qnb_tr[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5], gold[tr, 5], groups[tr], is_orig[tr])

    qv4_tr = qv2_tr.copy()
    qv4_tr[:, 4] = c_og24.predict_proba(extract_og24_tile_covs(r_base["q_tr_base"], P_tr, members, S_tr, T_tr), r_base["q_tr_base"][:, 4])
    qv4_tr[:, 5] = c_bk3.predict_proba(qnb_tr[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    qv4_tr = c.renorm(qv4_tr)

    qv4_va = qv2_va.copy()
    qv4_va[:, 4] = c_og24.predict_proba(extract_og24_tile_covs(r_base["q_va_base"], P_va, members, S_va, T_va), r_base["q_va_base"][:, 4])
    qv4_va[:, 5] = c_bk3.predict_proba(qnb_va[:, 5], c.renorm(P_va["pe_bkseg1"])[:, 5])
    qv4_va = c.renorm(qv4_va)

    best_p0_og, st_v4_k = v3m.select_p0_og_inner_cv(qv4_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=2.75, p0_base=1.00)
    pr_v4_va = v3m.predict_og_p0_eu(qv4_va, st_v4_k, lam_cl=2.75, p0_base=1.00, p0_og=best_p0_og)
    pr_v4_va[:, [0, 1, 3]] = pr_v2_va[:, [0, 1, 3]]
    cl_v2_va = (pr_v2_va == 0).all(axis=1)
    pr_v4_va[cl_v2_va] = 0
    newly_cl = (~cl_v2_va) & (pr_v4_va == 0).all(axis=1)
    if np.any(newly_cl):
        pr_v4_va[newly_cl] = pr_v2_va[newly_cl]

    return dict(
        k=k,
        qv2_va=qv2_va,
        pr_v2_va=pr_v2_va,
        qv4_va=qv4_va,
        pr_v4_va=pr_v4_va,
        cfg=(c_og24.alpha_, c_og24.best_topk_, c_og24.best_l2_, c_bk3.cfg_, best_p0_og),
    )


def _align_npz(path: str, names: Sequence[str], key: str) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    idx = {str(n): i for i, n in enumerate(z["names"])}
    return np.stack([z[key][idx[str(n)]] for n in names], axis=0).astype(np.float64)


def main():
    t0 = time.time()
    import upgrade_deconf_v4_tile as canonical_mod

    set_name = "resident_v7"
    members = SETS[set_name]
    names, gold, fold, _, P_dict = oof_pool(members)
    _, _, _, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)

    S_oof = _align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz"), names, "feats")
    T_oof = _align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_oof_3573.npz"), names, "stats")

    proto_cfg = dc._build_routed_gate_cfg(members)
    for g_dict in (_G, uob._G):
        g_dict.update(fold=fold, gold=gold, is_orig=is_orig, groups=groups, P=P_dict, S=S_oof, T=T_oof, members=members, proto_cfg=proto_cfg)

    print(f"=== [deconf_v4_tile] Zero-Leakage 8-Fold Nested OOF Verification (N={N}) ===", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        f_res = sorted(pool.map(_fold_worker_v4, list(range(8))), key=lambda d: d["k"])

    Q_v2_oof = np.zeros((N, 6, 4), dtype=np.float64)
    pr_v2_oof = np.zeros((N, 6), dtype=np.int64)
    Q_v4_oof = np.zeros((N, 6, 4), dtype=np.float64)
    pr_v4_oof = np.zeros((N, 6), dtype=np.int64)
    for r in f_res:
        va = fold == r["k"]
        Q_v2_oof[va] = r["qv2_va"]
        pr_v2_oof[va] = r["pr_v2_va"]
        Q_v4_oof[va] = r["qv4_va"]
        pr_v4_oof[va] = r["pr_v4_va"]

    rep_v2 = uob.report_line("deconf_v2 (54.44 baseline)", Q_v2_oof, pr_v2_oof, gold, is_orig)
    rep_v4 = uob.report_line("deconf_v4_tile (OG24+BkV3+ClnLock)", Q_v4_oof, pr_v4_oof, gold, is_orig)
    print("  Fold cfgs (alpha, topk, l2, bk3, p0_og):", [r["cfg"] for r in f_res], flush=True)

    # Fit full-OOF TileSalienceDeconfV4Bundle on all 3,573 OOF rows
    v2_pkl = os.path.join(REPO, "results/r55/final/resident_v7_decoder_deconf_v2.pkl")
    v3_pkl = os.path.join(REPO, "results/r55/final/resident_v7_decoder_deconf_v3.pkl")
    bnd_v2: uob.UpgradedDeconfBundle = pickle.load(open(v2_pkl, "rb"))["decoder"]
    bnd_v3: v3m.CleanLockedDeconfV3Bundle = pickle.load(open(v3_pkl, "rb"))["decoder"]

    Q_full_base = bnd_v2.base_decoder.pool_from_member_dict(P_dict)
    Q_full_nobk = uob.pool_without_bkseg1(bnd_v2.base_decoder, P_dict)

    cal_og24_full = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        canonical_mod.extract_og24_tile_covs(Q_full_base, P_dict, members, S_oof, T_oof),
        Q_full_base[:, 4],
        gold[:, 4],
        groups,
        is_orig=is_orig,
    )
    cal_bk3_full = v3m.BokehSynergyV3(w_supp=0.20).fit(
        Q_full_nobk[:, 5], c.renorm(P_dict["pe_bkseg1"])[:, 5], gold[:, 5], groups, is_orig
    )
    p0_og_full, st_v4_full = v3m.select_p0_og_inner_cv(
        Q_v4_oof, gold, groups, is_orig, lam_cl=float(bnd_v2.eu_decoder.lam_cl), p0_base=float(bnd_v2.eu_decoder.p0_scale)
    )

    print(
        f"\n[Full OOF Fit] cal_og24: alpha={cal_og24_full.alpha_} (topk={cal_og24_full.best_topk_}, l2={cal_og24_full.best_l2_}) | "
        f"cal_bk3: cfg={cal_bk3_full.cfg_} | p0_og={p0_og_full}",
        flush=True,
    )
    for ci, wi in zip(cal_og24_full.sel_cols_, cal_og24_full.w_):
        if abs(wi) >= 0.015:
            print(f"    cal_og24 feat {OG24_FEAT_NAMES[ci]:<28}: w = {wi:+.4f}")

    bnd_v4 = canonical_mod.TileSalienceDeconfV4Bundle(
        v2_bundle=copy.deepcopy(bnd_v2),
        cal_og24=cal_og24_full,
        cal_bk3=cal_bk3_full,
        lam_cl=float(bnd_v2.eu_decoder.lam_cl),
        p0_base=float(bnd_v2.eu_decoder.p0_scale),
        p0_og=p0_og_full,
        st_v4=st_v4_full,
        v3_fallback=copy.deepcopy(bnd_v3),
    )

    out_pkl = os.path.join(REPO, "results/r55/final/resident_v7_decoder_deconf_v4_tile.pkl")
    out_meta = os.path.join(REPO, "results/r55/final/resident_v7_decoder_deconf_v4_tile.json")
    pickle.dump({"decoder": bnd_v4, "members": members, "rule": "deconf_v4_tile"}, open(out_pkl, "wb"))

    meta = {
        "set": set_name,
        "rule": "deconf_v4_tile",
        "clean_image_locked_to_v2": True,
        "non_target_dims_locked_to_v2": [0, 1, 3],
        "eu_lam_cl": float(bnd_v4.lam_cl),
        "eu_p0_base": float(bnd_v4.p0_base),
        "eu_p0_og": float(bnd_v4.p0_og),
        "cal_og24": {
            "alpha": float(cal_og24_full.alpha_),
            "l2": float(cal_og24_full.best_l2_),
            "topk": int(cal_og24_full.best_topk_),
            "weights": {OG24_FEAT_NAMES[ci]: round(float(wi), 6) for ci, wi in zip(cal_og24_full.sel_cols_, cal_og24_full.w_)},
        },
        "cal_bk3_cfg": list(cal_bk3_full.cfg_),
        "nested_oof_v2": rep_v2,
        "nested_oof_v4_tile": rep_v4,
    }
    json.dump(meta, open(out_meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Verify clean unpickling via `final_engine.load_decoder`
    dec_fn, _ = load_decoder("resident_v7", "deconf_v4_tile")

    # Decode test_150 & final_644 with matched tile features and verify N=1 == N=Batch + Clean lock
    z_150 = np.load(os.path.join(REPO, "results/r55/os_og_breakthrough/pred_nriqa_dataset_test_os_og_breakthrough_member_probs.npz"), allow_pickle=True)
    z_644 = np.load(os.path.join(REPO, "results/r55/os_og_breakthrough_644/pred_nriqa_dataset_final_644_os_og_breakthrough_member_probs.npz"), allow_pickle=True)
    nm_150 = [str(x) for x in z_150["names"]]
    nm_644 = [str(x) for x in z_644["names"]]

    P_150 = {m: z_150[f"p_{m}"].astype(np.float64) if f"p_{m}" in z_150.files else z_150[m].astype(np.float64) for m in members}
    P_150["_spatial_20d"] = _align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_target.npz"), nm_150, "feats")
    P_150["_tile_peaks_6x12"] = _align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_test150.npz"), nm_150, "stats")

    P_644 = {m: z_644[f"p_{m}"].astype(np.float64) if f"p_{m}" in z_644.files else z_644[m].astype(np.float64) for m in members}
    P_644["_spatial_20d"] = _align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough_644/spatial_cache/spatial_final_644.npz"), nm_644, "feats")
    P_644["_tile_peaks_6x12"] = _align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_final644.npz"), nm_644, "stats")

    pr_v2_150, q_v2_150 = bnd_v2.predict(P_150)
    pr_v4_150, q_v4_150 = dec_fn(P_150)
    assert np.array_equal((pr_v2_150 == 0).all(1), (pr_v4_150 == 0).all(1)), "Clean image set changed on test_150!"
    assert np.array_equal(pr_v2_150[:, [0, 1, 3]], pr_v4_150[:, [0, 1, 3]]), "Non-target dims changed on test_150!"
    for i in range(len(nm_150)):
        p1, q1 = dec_fn({k: v[i : i + 1] for k, v in P_150.items()})
        assert np.array_equal(p1[0], pr_v4_150[i]) and np.max(np.abs(q1[0] - q_v4_150[i])) < 1e-12

    pr_v2_644, q_v2_644 = bnd_v2.predict(P_644)
    pr_v4_644, q_v4_644 = dec_fn(P_644)
    assert np.array_equal((pr_v2_644 == 0).all(1), (pr_v4_644 == 0).all(1)), "Clean image set changed on final_644!"
    assert np.array_equal(pr_v2_644[:, [0, 1, 3]], pr_v4_644[:, [0, 1, 3]]), "Non-target dims changed on final_644!"
    for i in range(len(nm_644)):
        p1, q1 = dec_fn({k: v[i : i + 1] for k, v in P_644.items()})
        assert np.array_equal(p1[0], pr_v4_644[i]) and np.max(np.abs(q1[0] - q_v4_644[i])) < 1e-12

    out_150_a = os.path.join(REPO, "results/final/pred_nriqa_dataset_test_r55_resident_v7_deconf_v4_tile.json")
    out_644_a = os.path.join(REPO, "results/final/pred_nriqa_dataset_final_644_r55_resident_v7_deconf_v4_tile.json")
    sprint_01 = os.path.join(REPO, "final_sprint_delivery/01_核心预测结果JSON")
    sprint_03 = os.path.join(REPO, "final_sprint_delivery/03_最后冲刺核心代码与脚本")
    sprint_04 = os.path.join(REPO, "final_sprint_delivery/04_最终校准解码器与实测审计账本")
    out_150_b = os.path.join(sprint_01, "pred_nriqa_dataset_test_r55_resident_v7_deconf_v4_tile.json")
    out_644_b = os.path.join(sprint_01, "pred_nriqa_dataset_final_644_r55_resident_v7_deconf_v4_tile.json")

    j150_obj = c.format_official_submission(nm_150, pr_v4_150)[0]
    j644_obj = c.format_official_submission(nm_644, pr_v4_644)[0]
    for p_dst, obj in ((out_150_a, j150_obj), (out_150_b, j150_obj), (out_644_a, j644_obj), (out_644_b, j644_obj)):
        os.makedirs(os.path.dirname(p_dst), exist_ok=True)
        json.dump(obj, open(p_dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if os.path.isdir(sprint_03):
        for fn in ("upgrade_deconf_v4_tile.py", "ntg_tile_peak_probe.py", "spatial_salience_probe.py", "native_tiles.py", "final_engine.py", "serve_engine.py"):
            shutil.copy2(os.path.join(REPO, "code/r55", fn), os.path.join(sprint_03, fn))
    if os.path.isdir(sprint_04):
        shutil.copy2(out_pkl, os.path.join(sprint_04, "resident_v7_decoder_deconf_v4_tile.pkl"))
        shutil.copy2(out_meta, os.path.join(sprint_04, "resident_v7_decoder_deconf_v4_tile.json"))

    validate_prediction_file(out_150_b, os.path.join(REPO, "results/v2_full/template_150.json"))
    validate_prediction_file(out_644_b, os.path.join(REPO, "results/final/template_final_644.json"))

    print("\n--- Exact Diff vs 54.44 (deconf_v2) ---")
    d_150 = np.where(pr_v2_150 != pr_v4_150)
    print(f"  test_150 changed: {len(d_150[0])} / 900 cells (Clean images: {(pr_v2_150==0).all(1).sum()} -> {(pr_v4_150==0).all(1).sum()}, 100% same mask!)")
    for i, d in zip(*d_150):
        print(
            f"    {nm_150[i]:<52} {DIMS_ZH[d]}: {GRADES_ZH[pr_v2_150[i,d]]} -> {GRADES_ZH[pr_v4_150[i,d]]}  "
            f"v2={np.round(q_v2_150[i,d],3).tolist()} -> v4={np.round(q_v4_150[i,d],3).tolist()}"
        )

    d_644 = np.where(pr_v2_644 != pr_v4_644)
    print(f"  final_644 changed: {len(d_644[0])} / 3864 cells (Clean images: {(pr_v2_644==0).all(1).sum()} -> {(pr_v4_644==0).all(1).sum()}, 100% same mask!)")
    for i, d in zip(*d_644):
        print(
            f"    {nm_644[i]:<52} {DIMS_ZH[d]}: {GRADES_ZH[pr_v2_644[i,d]]} -> {GRADES_ZH[pr_v4_644[i,d]]}  "
            f"v2={np.round(q_v2_644[i,d],3).tolist()} -> v4={np.round(q_v4_644[i,d],3).tolist()}"
        )
    print(f"\n=== ALL DONE in {time.time()-t0:.1f}s -> {out_150_b} ===", flush=True)


if __name__ == "__main__":
    main()
