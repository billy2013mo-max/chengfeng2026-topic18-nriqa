#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_v8_novlm_iv6b_decoder.py
=======================================
Builds the production `deconf_v6_boundary` decoder (`Iv6bPureVisionDeconfBundle`,
winning architecture `W2_V6_plus_blIV_keepBK`) for `resident_v8_novlm_iv6b`
(13 pure-vision members = `resident_v8_boundary_novlm` + `InternViT-6B` `iv6b`, 0 VLM!).

Strictly validated under 8-fold video-grouped Zero-Leakage Nested OOF (`N = 3,573`,
`ORIG = 1,902`):
  - `ORIG = 52.606` (+0.279 vs 12m No-VLM `52.327`, within -0.059 of 14m +2×8B VLM `52.665`)
  - `Pos = 16.236` (exceeds +2×8B VLM `16.217`), `Clean = 19.844` (ties +2×8B VLM),
    `ALL = 51.952`, `RPS_o = 0.11242`, `Boot654(Pos<15) = 0.10%`.
  - Strict `N = 1 == N = Batch` invariance (`max_q_diff < 1e-12`, 0 label diffs).
  - Zero test-set tuning, zero legacy JSON anchoring.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import pickle
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
from final_engine import OUT, SETS, load_decoder, oof_pool  # noqa: E402

LAMS_CL_GRID = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)
EXTRA_OOF_DIRS = {
    "bndry_pe": "results/r55/boundary_v2/v2_pe_ntg",
    "bndry_dual": "results/r55/boundary_v2/v2_dual_pe_cradio",
    "iv6b": "results/r55/iv6b_native_pilot",
}


def extract_og26_iv6b_covs(
    Q_raw_og: np.ndarray,
    P_v7: Dict[str, np.ndarray],
    base_members: Sequence[str],
    spatial_20d: np.ndarray,
    tile_peaks_6x12: np.ndarray,
    iv6b_cal: np.ndarray,
) -> np.ndarray:
    """24-D spatial/tile OG covariates + 2-D `iv6b` severity & P0 discrepancy (strictly row-wise N=1 invariant)."""
    X24 = v4m.extract_og24_tile_covs(Q_raw_og, P_v7, base_members, spatial_20d, tile_peaks_6x12)
    gv = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    og0 = c.renorm(Q_raw_og)[:, 4, :]
    iv4 = c.renorm(iv6b_cal)[:, 4, :]
    iv_sev_gap = (iv4 @ gv) - (og0 @ gv)
    iv_p0_gap = iv4[:, 3] - og0[:, 3]
    return np.nan_to_num(np.column_stack([X24, iv_sev_gap, iv_p0_gap]), nan=0.0, posinf=10.0, neginf=-10.0)


class Iv6bPureVisionDeconfBundle:
    """
    Self-contained, strictly `N = 1` batch-invariant `deconf_v6_boundary` bundle for
    `resident_v8_novlm_iv6b` (13 pure-vision members, 0 VLM):
      - Base 10-member `RoutedGateEUDecoder` + `dec_nobk` (9-member `pe_bkseg1`-free pool)
      - `dim=0 (模糊)`: `cal_blur33` -> post-LRO log-linear `iv6b`
      - `dim=1 (纹理损失)`: `dec_nobk` -> log-linear `iv6b`
      - `dim=2 (过锐)`: pre-LRO log-linear `iv6b` -> `cal_os`
      - `dim=3 (花屏)`: base `peg`-rescued pool
      - `dim=4 (过度生成)`: pre-LRO log-linear `(bndry_pe, bndry_dual, iv6b)` -> `cal_og26`
      - `dim=5 (虚化突兀)`: `BokehSynergyV3` (`nobk` + `pe_bkseg1`, unperturbed)
      - Unified `TrueGateConstrainedEU` + inner-CV `p0_og`.
    """

    def __init__(
        self,
        set_name: str,
        members_all: Sequence[str],
        base_members: Sequence[str],
        nobk_members: Sequence[str],
        base_decoder: red.RoutedGateEUDecoder,
        dec_nobk: red.RoutedGateEUDecoder,
        cal_bk3: v3m.BokehSynergyV3,
        cal_blur33: dc.LowRankOrdinalCalibrator,
        cal_os: dc.LowRankOrdinalCalibrator,
        cal_og26: dc.LowRankOrdinalCalibrator,
        extra_temps: Dict[str, float],
        w_bl: np.ndarray,
        w_tx: np.ndarray,
        w_os: np.ndarray,
        w_og: np.ndarray,
        eu_decoder: red.TrueGateConstrainedEU,
        lam_cl: float,
        p0_base: float,
        p0_og: float,
        st_eu: dict,
    ):
        self.set_name = str(set_name)
        self.members = list(members_all)
        self.base_members = list(base_members)
        self.nobk_members = list(nobk_members)
        self.base_decoder = base_decoder
        self.dec_nobk = dec_nobk
        self.cal_bk3 = cal_bk3
        self.cal_blur33 = cal_blur33
        self.cal_os = cal_os
        self.cal_og26 = cal_og26
        self.extra_temps = {str(k): float(v) for k, v in extra_temps.items()}
        self.w_bl = np.asarray(w_bl, dtype=np.float64)
        self.w_tx = np.asarray(w_tx, dtype=np.float64)
        self.w_os = np.asarray(w_os, dtype=np.float64)
        self.w_og = np.asarray(w_og, dtype=np.float64)
        self.eu_decoder = eu_decoder
        self.lam_cl = float(lam_cl)
        self.p0_base = float(p0_base)
        self.p0_og = float(p0_og)
        self.st_eu = st_eu

    def calibrate_probs(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        P_base = {m: P_dict[m] for m in self.base_members}
        P_nobk = {m: P_dict[m] for m in self.nobk_members}

        Q_raw = self.base_decoder.pool_from_member_dict(P_base)
        Q_nobk_base = uob.pool_without_bkseg1(self.base_decoder, P_base)
        Q_nb = self.dec_nobk.pool_from_member_dict(P_nobk)

        # Temperature-calibrated extra pure-vision specialists
        cals = {
            k: v6m.apply_scalar_temp(P_dict[k], self.extra_temps[k])
            for k in ("bndry_pe", "bndry_dual", "iv6b")
            if k in P_dict
        }

        Q = Q_raw.copy()
        # dim=5 (虚化突兀): BokehSynergyV3
        if "pe_bkseg1" in P_dict:
            Q[:, 5, :] = self.cal_bk3.predict_proba(Q_nobk_base[:, 5, :], c.renorm(P_dict["pe_bkseg1"])[:, 5, :])

        # dim=0 (模糊): cal_blur33 -> post-LRO iv6b
        if "_tile_peaks_6x12" in P_dict and self.cal_blur33 is not None and self.cal_blur33.alpha_ > 0.0:
            Xbl = v5m.extract_blur33_tile_covs(Q, P_base, self.nobk_members, P_dict["_tile_peaks_6x12"])
            bl_v5 = self.cal_blur33.predict_proba(Xbl, Q[:, 0, :])
        else:
            bl_v5 = Q[:, 0, :]
        if "iv6b" in cals:
            Q[:, 0, :] = v6m.apply_loglin_blend_dim(bl_v5, [cals["iv6b"][:, 0, :]], self.w_bl)
        else:
            Q[:, 0, :] = bl_v5

        # dim=1 (纹理损失): nobk -> log-linear iv6b
        if "iv6b" in cals:
            Q[:, 1, :] = v6m.apply_loglin_blend_dim(Q_nb[:, 1, :], [cals["iv6b"][:, 1, :]], self.w_tx)
        else:
            Q[:, 1, :] = Q_nb[:, 1, :]

        # dim=2 (过锐): pre-LRO iv6b -> cal_os
        os0 = v6m.apply_loglin_blend_dim(Q_raw[:, 2, :], [cals["iv6b"][:, 2, :]], self.w_os) if "iv6b" in cals else Q_raw[:, 2, :]
        Q_raw_os = Q_raw.copy()
        Q_raw_os[:, 2, :] = os0
        if self.cal_os is not None and self.cal_os.alpha_ > 0.0:
            Xos = uob.extract_os_micro_covariates(Q_raw_os, P_base, self.base_members)
            Q[:, 2, :] = self.cal_os.predict_proba(Xos, os0)
        else:
            Q[:, 2, :] = os0

        # dim=4 (过度生成): pre-LRO (bndry_pe, bndry_dual, iv6b) -> cal_og26
        og_keys = ("bndry_pe", "bndry_dual", "iv6b")
        if all(k in cals for k in og_keys):
            og0 = v6m.apply_loglin_blend_dim(Q_raw[:, 4, :], [cals[k][:, 4, :] for k in og_keys], self.w_og)
            Q_raw_og = Q_raw.copy()
            Q_raw_og[:, 4, :] = og0
            if "_spatial_20d" in P_dict and "_tile_peaks_6x12" in P_dict and self.cal_og26 is not None and self.cal_og26.alpha_ > 0.0:
                X26 = extract_og26_iv6b_covs(
                    Q_raw_og, P_base, self.base_members, P_dict["_spatial_20d"], P_dict["_tile_peaks_6x12"], cals["iv6b"]
                )
                Q[:, 4, :] = self.cal_og26.predict_proba(X26, og0)
            else:
                Q[:, 4, :] = og0

        return c.renorm(Q)

    def predict(self, P_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        Q = self.calibrate_probs(P_dict)
        pr = v3m.predict_og_p0_eu(Q, self.st_eu, self.lam_cl, self.p0_base, self.p0_og)
        return pr, Q


_G: Dict = {}


def _fit_fold_or_full(
    P10_tr: Dict[str, np.ndarray],
    Pex_tr: Dict[str, np.ndarray],
    S_tr: np.ndarray,
    T_tr: np.ndarray,
    gold_tr: np.ndarray,
    is_orig_tr: np.ndarray,
    groups_tr: np.ndarray,
    m10: Sequence[str],
    nb9: Sequence[str],
    members_all: Sequence[str],
    set_name: str,
) -> Iv6bPureVisionDeconfBundle:
    rw = np.where(is_orig_tr, 1.0, 0.25)

    proto_cfg = dc._build_routed_gate_cfg(m10)
    proto_cfg["lams_cl"] = LAMS_CL_GRID
    base_dec = red.RoutedGateEUDecoder(**proto_cfg).fit(P10_tr, gold_tr, is_orig=is_orig_tr, groups=groups_tr)
    Q_raw = base_dec.pool_from_member_dict(P10_tr)
    Q_nobk_base = uob.pool_without_bkseg1(base_dec, P10_tr)

    cal_bk3 = v3m.BokehSynergyV3(w_supp=0.20).fit(
        Q_nobk_base[:, 5], c.renorm(P10_tr["pe_bkseg1"])[:, 5], gold_tr[:, 5], groups_tr, is_orig=is_orig_tr
    )
    nb_cfg = dc._build_routed_gate_cfg(nb9)
    nb_cfg["lams_cl"] = LAMS_CL_GRID
    P_nb_tr = {m: P10_tr[m] for m in nb9}
    dec_nb = red.RoutedGateEUDecoder(**nb_cfg).fit(P_nb_tr, gold_tr, is_orig=is_orig_tr, groups=groups_tr)
    Q_nb = dec_nb.pool_from_member_dict(P_nb_tr)

    Q_for_bl = Q_raw.copy()
    Q_for_bl[:, 5] = cal_bk3.predict_proba(Q_nobk_base[:, 5], c.renorm(P10_tr["pe_bkseg1"])[:, 5])
    Q_for_bl = c.renorm(Q_for_bl)
    Xbl = v5m.extract_blur33_tile_covs(Q_for_bl, P10_tr, nb9, T_tr)
    cal_blur33 = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 14, 24, Xbl.shape[1])).fit(
        Xbl, Q_for_bl[:, 0], gold_tr[:, 0], groups_tr, is_orig=is_orig_tr
    )
    bl_v5 = cal_blur33.predict_proba(Xbl, Q_for_bl[:, 0])

    # Temperature calibrate extra specialists strictly on training data
    extra_temps = {nm: v6m.fit_scalar_temp_weighted(arr, gold_tr, rw) for nm, arr in Pex_tr.items()}
    cals = {nm: v6m.apply_scalar_temp(arr, extra_temps[nm]) for nm, arr in Pex_tr.items()}

    # dim=0 (模糊): post-LRO iv6b
    w_bl = v6m.fit_loglin_blend_dim(bl_v5, [cals["iv6b"][:, 0]], gold_tr[:, 0], rw, max_each=0.40, max_sum=0.45)
    bl_f = v6m.apply_loglin_blend_dim(bl_v5, [cals["iv6b"][:, 0]], w_bl)

    # dim=1 (纹理损失): nobk + iv6b
    w_tx = v6m.fit_loglin_blend_dim(Q_nb[:, 1], [cals["iv6b"][:, 1]], gold_tr[:, 1], rw, max_each=0.40, max_sum=0.45)
    tx_f = v6m.apply_loglin_blend_dim(Q_nb[:, 1], [cals["iv6b"][:, 1]], w_tx)

    # dim=2 (过锐): pre-LRO iv6b -> cal_os
    w_os = v6m.fit_loglin_blend_dim(Q_raw[:, 2], [cals["iv6b"][:, 2]], gold_tr[:, 2], rw, max_each=0.40, max_sum=0.45)
    os0 = v6m.apply_loglin_blend_dim(Q_raw[:, 2], [cals["iv6b"][:, 2]], w_os)
    Q_raw_os = Q_raw.copy()
    Q_raw_os[:, 2] = os0
    Xos = uob.extract_os_micro_covariates(Q_raw_os, P10_tr, m10)
    cal_os = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=(2, 4, 8, 11)).fit(
        Xos, os0, gold_tr[:, 2], groups_tr, is_orig=is_orig_tr
    )
    os_f = cal_os.predict_proba(Xos, os0)

    # dim=4 (过度生成): pre-LRO (bndry_pe, bndry_dual, iv6b) -> cal_og26
    og_keys = ("bndry_pe", "bndry_dual", "iv6b")
    w_og = v6m.fit_loglin_blend_dim(
        Q_raw[:, 4], [cals[nm][:, 4] for nm in og_keys], gold_tr[:, 4], rw, max_each=0.35, max_sum=0.45
    )
    og0 = v6m.apply_loglin_blend_dim(Q_raw[:, 4], [cals[nm][:, 4] for nm in og_keys], w_og)
    Q_raw_og = Q_raw.copy()
    Q_raw_og[:, 4] = og0
    X26 = extract_og26_iv6b_covs(Q_raw_og, P10_tr, m10, S_tr, T_tr, cals["iv6b"])
    cal_og26 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 26)).fit(
        X26, og0, gold_tr[:, 4], groups_tr, is_orig=is_orig_tr
    )
    og_f = cal_og26.predict_proba(X26, og0)

    Q_cal = Q_for_bl.copy()
    Q_cal[:, 0] = bl_f
    Q_cal[:, 1] = tx_f
    Q_cal[:, 2] = os_f
    Q_cal[:, 4] = og_f
    Q_cal = c.renorm(Q_cal)

    eu_dec = red.TrueGateConstrainedEU(lams_cl=LAMS_CL_GRID).fit(Q_cal, gold_tr, is_orig=is_orig_tr, groups=groups_tr)
    lam_cl, p0_base = float(eu_dec.lam_cl), float(eu_dec.p0_scale)
    p0_og, st_eu = v3m.select_p0_og_inner_cv(Q_cal, gold_tr, groups_tr, is_orig_tr, lam_cl=lam_cl, p0_base=p0_base)

    import build_v8_novlm_iv6b_decoder as canonical_mod

    return canonical_mod.Iv6bPureVisionDeconfBundle(
        set_name=set_name,
        members_all=members_all,
        base_members=m10,
        nobk_members=nb9,
        base_decoder=base_dec,
        dec_nobk=dec_nb,
        cal_bk3=cal_bk3,
        cal_blur33=cal_blur33,
        cal_os=cal_os,
        cal_og26=cal_og26,
        extra_temps=extra_temps,
        w_bl=w_bl,
        w_tx=w_tx,
        w_os=w_os,
        w_og=w_og,
        eu_decoder=eu_dec,
        lam_cl=lam_cl,
        p0_base=p0_base,
        p0_og=p0_og,
        st_eu=st_eu,
    )


def _fold_worker(k: int) -> dict:
    fold = _G["fold"]
    va = fold == k
    tr = ~va
    P10_tr = {m: v[tr] for m, v in _G["P10"].items()}
    Pex_tr = {m: v[tr] for m, v in _G["Pex"].items()}
    bnd = _fit_fold_or_full(
        P10_tr, Pex_tr, _G["S"][tr], _G["T"][tr],
        _G["gold"][tr], _G["is_orig"][tr], _G["groups"][tr],
        _G["m10"], _G["nb9"], _G["members_all"], _G["set_name"],
    )
    P_va = {m: v[va] for m, v in _G["P10"].items()}
    for m, v in _G["Pex"].items():
        P_va[m] = v[va]
    P_va["_spatial_20d"] = _G["S"][va]
    P_va["_tile_peaks_6x12"] = _G["T"][va]
    pr_va, q_va = bnd.predict(P_va)
    return dict(k=k, q_va=q_va, pr_va=pr_va, lam_cl=bnd.lam_cl, p0_base=bnd.p0_base, p0_og=bnd.p0_og)


def main() -> None:
    t0 = time.time()
    set_name = "resident_v8_novlm_iv6b"
    m10 = list(SETS["resident_v7_novlm"])
    nb9 = [m for m in m10 if m != "pe_bkseg1"]
    members_all = list(SETS[set_name])
    assert "p1_qy" not in members_all and "qwen_hr" not in members_all and "iv6b" in members_all

    names, gold, fold, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)
    _, _, _, _, P10 = oof_pool(m10)

    Pex: Dict[str, np.ndarray] = {}
    for k_ex, d_rel in EXTRA_OOF_DIRS.items():
        arr, valid = load_oof_dir(d_rel, names)
        assert valid.all(), f"Incomplete 8-fold OOF for {k_ex}: {valid.sum()}/{N}"
        Pex[k_ex] = c.renorm(arr)

    S_oof = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz"), names, "feats")
    T_oof = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_oof_3573.npz"), names, "stats")

    _G.update(fold=fold, gold=gold, is_orig=is_orig, groups=groups, P10=P10, Pex=Pex, S=S_oof, T=T_oof, m10=m10, nb9=nb9, members_all=members_all, set_name=set_name)

    print(f"=== [{set_name}] Zero-Leakage 8-Fold Nested OOF Verification (N={N}, 13 pure-vision members, 0 VLM) ===", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        f_res = sorted(pool.map(_fold_worker, list(range(8))), key=lambda d: d["k"])

    Q_oof = np.zeros((N, 6, 4))
    pr_oof = np.zeros((N, 6), dtype=np.int64)
    for r in f_res:
        va = fold == r["k"]
        Q_oof[va], pr_oof[va] = r["q_va"], r["pr_va"]

    rep = v5m._summarize_all4("deconf_v6_boundary (13v novlm+iv6b)", Q_oof, pr_oof, gold, is_orig)
    bk_f1 = dc._f1_bin(pr_oof[is_orig, 5] == 3, gold[is_orig, 5] == 3)
    gl_f1 = dc._f1_bin(pr_oof[is_orig, 3] == 3, gold[is_orig, 3] == 3)
    cp0 = int(((gold[is_orig] == 0).all(axis=1) & (pr_oof[is_orig] == 3).any(axis=1)).sum())
    pf, _, _ = red._fast_boot654_pos_risk(pr_oof[is_orig], gold[is_orig], groups[is_orig], n_boot=5000)
    print(f"      BK_P0={bk_f1:.4f} GL_P0={gl_f1:.4f} Clean-P0={cp0} Boot654(Pos<15)={pf:.2f}%", flush=True)
    print(f"      fold lam_cl={[r['lam_cl'] for r in f_res]} | p0_base={[r['p0_base'] for r in f_res]} | p0_og={[r['p0_og'] for r in f_res]}", flush=True)

    # Fit FINAL production bundle on all 3,573 honest OOF predictions (ZERO test-set peeking!)
    bundle = _fit_fold_or_full(P10, Pex, S_oof, T_oof, gold, is_orig, groups, m10, nb9, members_all, set_name)

    # Strict N=1 == N=Batch self-test!
    P_check = dict(P10)
    P_check.update(Pex)
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
        pickle.dump(dict(decoder=bundle, members=members_all, set=set_name, rule="deconf_v6_boundary"), f)

    dec_fn, _ = load_decoder(set_name, "deconf_v6_boundary")
    pr_rt, q_rt = dec_fn(P_sub)
    assert np.array_equal(pr_rt, pr_bat) and float(np.max(np.abs(q_rt - q_bat))) < 1e-12

    summary = {
        "set": set_name,
        "rule": "deconf_v6_boundary",
        "members": members_all,
        "lam_cl_full": bundle.lam_cl,
        "p0_base_full": bundle.p0_base,
        "p0_og_full": bundle.p0_og,
        "extra_temps": bundle.extra_temps,
        "w_bl": bundle.w_bl.tolist(),
        "w_tx": bundle.w_tx.tolist(),
        "w_os": bundle.w_os.tolist(),
        "w_og": bundle.w_og.tolist(),
        "nested_8fold": rep,
        "bk_p0_orig": round(bk_f1, 4),
        "gl_p0_orig": round(gl_f1, 4),
        "clean_p0_orig": cp0,
        "boot654_pos_risk": round(pf, 3),
        "n1_batch_invariance": {"max_q_diff": max_q_diff, "label_diffs": n_lbl_diff},
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"\n[SAVED] {out_pkl}\n"
        f"  lam_cl={bundle.lam_cl:.2f} | p0_base={bundle.p0_base:.2f} | p0_og={bundle.p0_og:.2f}\n"
        f"  extra_temps={ {k: round(v, 3) for k, v in bundle.extra_temps.items()} }\n"
        f"  w_bl={np.round(bundle.w_bl, 4).tolist()} | w_tx={np.round(bundle.w_tx, 4).tolist()} | "
        f"w_os={np.round(bundle.w_os, 4).tolist()} | w_og={np.round(bundle.w_og, 4).tolist()}\n"
        f"  N=1 == N=Batch: max_q_diff={max_q_diff:.2e}, label_diffs={n_lbl_diff}/96 | elapsed={time.time()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
