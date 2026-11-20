#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/upgrade_deconf_v3_cleanlocked.py
=========================================
Builds `deconf_v3` (`CleanLockedDeconfV3Bundle`) on top of `deconf_v2` (`54.44`, where Clean Score = `20.00/20`),
verified under 100% zero-leakage 8-fold video-grouped Nested OOF (`N = 3,573` real images, zero test-set tuning,
zero legacy JSON anchoring, strict `N = 1 == N = Batch` invariance):

  1. `过度生成` (`dim=4`, `extract_og_spec_covs`): Upgrades `dim=4` `LowRankOrdinalCalibrator` from 8-D to 12-D by
     adding 1:1 semantic-ROI & local-microscope discrepancy covariates (`p12_pe` face/hand/anatomy ROI specialist,
     `v16` 1:1 crop microscope, `p1_qy` Qwen3-VL-8B semantic VLM) — symmetric counterpart to `过锐` (`dim=2`).
  2. `虚化突兀` (`dim=5`, `BokehSynergyV3`): Expands inner-CV synergistic `pe_bkseg1` blend (`alpha_hi` up to `0.32`
     when real-domain ensemble agrees `q_real >= tau`, keeping `alpha_lo in {0.04, 0.05}` Clean shield otherwise).
  3. `Clean-Locked Target-Dimension EU` (`OG` `p0_og_scale in (1.00, 1.08, 1.16)` selected via 4-fold video-grouped
     inner CV + exact `v2` Clean-image lock): De-dilutes `过度生成` `P0/P1` while mathematically guaranteeing that
     every Clean prediction of `deconf_v2` (`Clean Score = 20.00 / 20`) remains 100% invariant.
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
from typing import Dict, Optional, Sequence, Tuple

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
import decoder_eu as E  # noqa: E402
import deconfound_calibrator as dc  # noqa: E402
import routed_eu_decoder as red  # noqa: E402
import upgrade_os_og_bokeh_deconf as uob  # noqa: E402
from final_engine import SETS, load_decoder, oof_pool  # noqa: E402
from validate_submission import validate as validate_prediction_file  # noqa: E402

DIMS_ZH = ("模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀")
GRADES_ZH = ("无问题", "P2", "P1", "P0")
OG_SPEC_FEAT_NAMES = (
    "blur_exp_sev",
    "texloss_exp_sev",
    "os_exp_sev",
    "bokeh_p0_logit",
    "max_other_any",
    "clean_other_logp",
    "member_std_sev",
    "ntg_vs_global_gap",
    "loc4_vs_glo3_gap",
    "p12_ge1_gap",
    "v16_p0_gap",
    "vlm_sev_gap",
)


def extract_og_spec_covs(
    Q_base: np.ndarray,
    P_dict: Dict[str, np.ndarray],
    members: Sequence[str],
) -> np.ndarray:
    """
    Strictly row-wise (`N = 1` invariant) 12-D covariates for `过度生成` (`dim=4`):
    augments the 8-D cross-defect covariates with 1:1 local/semantic-ROI specialist vs global discrepancy
    (`p12_pe` anatomy/face/hand ROI, `v16` 1:1 microscope, `ntg_pe + ntg_convnext`, `p1_qy` Qwen3-VL-8B).
    """
    X8 = dc.extract_deconf_covariates(Q_base, P_dict, 4, members)
    Q = c.renorm(Q_base)
    N = Q.shape[0]
    gv = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    loc_keys = [m for m in ("ntg_pe", "ntg_convnext", "p12_pe", "v16") if m in members and m in P_dict]
    glo_keys = [m for m in ("cradio_v3", "pe_v3", "sig_v3") if m in members and m in P_dict]
    loc_m = np.mean([c.renorm(P_dict[m])[:, 4] @ gv for m in loc_keys], axis=0) if loc_keys else np.zeros(N)
    glo_m = np.mean([c.renorm(P_dict[m])[:, 4] @ gv for m in glo_keys], axis=0) if glo_keys else np.zeros(N)

    if "p12_pe" in members and "p12_pe" in P_dict:
        p_p12 = c.renorm(P_dict["p12_pe"])[:, 4]
        p12_ge1_gap = (p_p12[:, 2] + p_p12[:, 3]) - (Q[:, 4, 2] + Q[:, 4, 3])
    else:
        p12_ge1_gap = np.zeros(N)

    v16_p0_gap = (c.renorm(P_dict["v16"])[:, 4, 3] - Q[:, 4, 3]) if ("v16" in members and "v16" in P_dict) else np.zeros(N)
    vlm_sev_gap = ((c.renorm(P_dict["p1_qy"])[:, 4] @ gv) - (Q[:, 4] @ gv)) if ("p1_qy" in members and "p1_qy" in P_dict) else np.zeros(N)

    X12 = np.column_stack([X8, loc_m - glo_m, p12_ge1_gap, v16_p0_gap, vlm_sev_gap])
    return np.nan_to_num(X12, nan=0.0, posinf=10.0, neginf=-10.0)


class BokehSynergyV3(uob.BokehSynergyCalibrator):
    """Expands synergistic `alpha_hi` up to `0.32` when real-domain & `pe_bkseg1` agree, keeping Clean-safe `alpha_lo`."""

    GRID: Tuple[Tuple[float, float, float], ...] = uob.BokehSynergyCalibrator.GRID + (
        (0.28, 0.05, 0.35),
        (0.32, 0.05, 0.35),
        (0.28, 0.04, 0.32),
        (0.32, 0.04, 0.38),
    )


def predict_og_p0_eu(Q: np.ndarray, st: dict, lam_cl: float, p0_base: float, p0_og: float) -> np.ndarray:
    """Strictly row-wise (`N = 1` invariant) Bayes EU decoder with dedicated `p0_og` scale for `过度生成` (`dim=4`)."""
    Q = c.renorm(Q)
    D = np.maximum(1.0, st["npred"] + st["npos"])
    p = Q[:, E.LD, E.LG]
    ps_vec = np.where(E.LD == 4, p0_og, p0_base)
    w_f = np.where(E.LG == 3, 25.0 / 14.0 + ps_vec * (20.0 * 0.3 / 6.0), 25.0 / 14.0)
    w_r = ps_vec * E.W_R
    w_p = ps_vec * E.W_P
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
    p_cl = np.prod(Q[:, :, 0], axis=1)
    clean_diff = 10.0 * (1.0 - p_cl) / max(1.0, st["N"] - st["ncl"]) - lam_cl * 10.0 * p_cl / max(1.0, st["ncl"])

    pred[has_pos & ((pos_gain_sum + clean_diff) < 0.0)] = 0
    promote = (~has_pos) & ((best_nonp0_gain + clean_diff) > 0.0)
    if promote.any():
        idx_p = np.where(promote)[0]
        pred[idx_p, best_nonp0_d[idx_p]] = best_nonp0_g[idx_p]
    return pred


def fit_st_og(Q: np.ndarray, gold: np.ndarray, row_w: np.ndarray, lam_cl: float, p0_base: float, p0_og: float, iters: int = 25) -> dict:
    pred0 = E.Decoder55().predict(Q)
    st = red.TrueGateConstrainedEU._raw_stats(pred0, gold, row_w)
    for _ in range(iters):
        pr_now = predict_og_p0_eu(Q, st, lam_cl, p0_base, p0_og)
        st_ind = red.TrueGateConstrainedEU._raw_stats(pr_now, gold, row_w)
        if float(np.max(np.abs(st_ind["F"] - st["F"]))) < 1e-6:
            return st_ind
        st = dict(
            F=0.5 * st["F"] + 0.5 * st_ind["F"],
            P=0.5 * st["P"] + 0.5 * st_ind["P"],
            npos=st_ind["npos"],
            npred=0.5 * st["npred"] + 0.5 * st_ind["npred"],
            ncl=st_ind["ncl"],
            N=st_ind["N"],
        )
    return st


def select_p0_og_inner_cv(
    Q_tr: np.ndarray,
    gold_tr: np.ndarray,
    groups_tr: np.ndarray,
    is_orig_tr: np.ndarray,
    lam_cl: float = 2.75,
    p0_base: float = 1.00,
    candidates: Tuple[float, ...] = (1.00, 1.08, 1.16),
) -> Tuple[float, dict]:
    rw = np.where(is_orig_tr, 1.0, 0.25)
    ug = sorted(set(str(g) for g in groups_tr))
    g2b = {g: (i % 4) for i, g in enumerate(ug)}
    inf = np.array([g2b[str(g)] for g in groups_tr], dtype=np.int64)

    best_p0_og = float(p0_base)
    best_sc = -1e9
    for cand in candidates:
        pr_in = np.zeros((len(gold_tr), 6), dtype=np.int64)
        for b in range(4):
            iva = inf == b
            itr = ~iva
            st_in = fit_st_og(Q_tr[itr], gold_tr[itr], rw[itr], lam_cl, p0_base, cand)
            pr_in[iva] = predict_og_p0_eu(Q_tr[iva], st_in, lam_cl, p0_base, cand)
        sc_all = float(c.score(pr_in, gold_tr)["doc2_strict_total"])
        sc_orig = float(c.score(pr_in[is_orig_tr], gold_tr[is_orig_tr])["doc2_strict_total"])
        sc_mid = 0.5 * sc_all + 0.5 * sc_orig
        if sc_mid > best_sc + 0.005:
            best_sc = sc_mid
            best_p0_og = float(cand)

    st_full = fit_st_og(Q_tr, gold_tr, rw, lam_cl, p0_base, best_p0_og)
    return best_p0_og, st_full


class CleanLockedDeconfV3Bundle:
    """
    Self-contained `N = 1` batch-invariant `deconf_v3` inference bundle:
      - Calibrates `过锐` (`cal_os` 11-D), `过度生成` (`cal_og12` 12-D specialist discrepancy), and `虚化突兀` (`cal_bk3`).
      - Decodes via `predict_og_p0_eu` (`lam_cl`, `p0_base`, `p0_og`) and enforces an exact row-wise Clean-image lock
        against `v2_bundle` (`(pred_v2 == 0).all(axis=1) -> pred_v3 = 0`), mathematically protecting `Clean = 20.00/20`.
    """

    def __init__(
        self,
        v2_bundle: uob.UpgradedDeconfBundle,
        cal_og12: dc.LowRankOrdinalCalibrator,
        cal_bk3: BokehSynergyV3,
        lam_cl: float,
        p0_base: float,
        p0_og: float,
        st_v3: dict,
    ):
        self.v2_bundle = v2_bundle
        self.set_name = v2_bundle.set_name
        self.members = list(v2_bundle.members)
        self.base_decoder = v2_bundle.base_decoder
        self.cal_os = v2_bundle.cal_os
        self.cal_og12 = cal_og12
        self.cal_bk3 = cal_bk3
        self.lam_cl = float(lam_cl)
        self.p0_base = float(p0_base)
        self.p0_og = float(p0_og)
        self.st_v3 = st_v3
        self.eu_decoder = v2_bundle.eu_decoder

    def calibrate_probs(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        Q_base = self.base_decoder.pool_from_member_dict(P_dict)
        Q = Q_base.copy()
        if self.cal_os is not None and self.cal_os.alpha_ > 0.0:
            Q[:, 2, :] = self.cal_os.predict_proba(uob.extract_os_micro_covariates(Q_base, P_dict, self.members), Q_base[:, 2, :])
        if self.cal_og12 is not None and self.cal_og12.alpha_ > 0.0:
            Q[:, 4, :] = self.cal_og12.predict_proba(extract_og_spec_covs(Q_base, P_dict, self.members), Q_base[:, 4, :])
        if self.cal_bk3 is not None and "pe_bkseg1" in P_dict:
            Q_nobk = uob.pool_without_bkseg1(self.base_decoder, P_dict)
            Q[:, 5, :] = self.cal_bk3.predict_proba(Q_nobk[:, 5, :], c.renorm(P_dict["pe_bkseg1"])[:, 5, :])
        return c.renorm(Q)

    def predict(self, P_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        pr_v2, _ = self.v2_bundle.predict(P_dict)
        Q_v3 = self.calibrate_probs(P_dict)
        pr_v3 = predict_og_p0_eu(Q_v3, self.st_v3, self.lam_cl, self.p0_base, self.p0_og)
        # Strictly isolate upgrades to the 3 target dimensions {2:过锐, 4:过度生成, 5:虚化突兀}:
        # non-target dims {0:模糊, 1:纹理损失, 3:花屏} and exact Clean-image mask remain 100% invariant to `deconf_v2` (`54.44`)!
        pr_v3[:, [0, 1, 3]] = pr_v2[:, [0, 1, 3]]
        cl_v2 = (pr_v2 == 0).all(axis=1)
        pr_v3[cl_v2] = 0
        newly_clean = (~cl_v2) & (pr_v3 == 0).all(axis=1)
        if np.any(newly_clean):
            pr_v3[newly_clean] = pr_v2[newly_clean]
        return pr_v3, Q_v3


_G = {}


def _fold_worker_v3(k: int):
    fold_arr = _G["fold"]
    gold = _G["gold"]
    is_orig = _G["is_orig"]
    groups = _G["groups"]
    P_dict = _G["P"]
    members = _G["members"]

    r_base = uob._fold_worker(k)
    va = fold_arr == k
    tr = ~va
    P_tr = {m: v[tr] for m, v in P_dict.items()}
    P_va = {m: v[va] for m, v in P_dict.items()}
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

    c_og12 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(2, 4, 8, 12)).fit(
        extract_og_spec_covs(r_base["q_tr_base"], P_tr, members), r_base["q_tr_base"][:, 4], gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    qnb_tr = uob.pool_without_bkseg1(r_base["dec_k"], P_tr)
    qnb_va = uob.pool_without_bkseg1(r_base["dec_k"], P_va)
    c_bk3 = BokehSynergyV3(w_supp=0.20).fit(qnb_tr[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5], gold[tr, 5], groups[tr], is_orig[tr])

    qv3_tr = qv2_tr.copy()
    qv3_tr[:, 4] = c_og12.predict_proba(extract_og_spec_covs(r_base["q_tr_base"], P_tr, members), r_base["q_tr_base"][:, 4])
    qv3_tr[:, 5] = c_bk3.predict_proba(qnb_tr[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    qv3_tr = c.renorm(qv3_tr)

    qv3_va = qv2_va.copy()
    qv3_va[:, 4] = c_og12.predict_proba(extract_og_spec_covs(r_base["q_va_base"], P_va, members), r_base["q_va_base"][:, 4])
    qv3_va[:, 5] = c_bk3.predict_proba(qnb_va[:, 5], c.renorm(P_va["pe_bkseg1"])[:, 5])
    qv3_va = c.renorm(qv3_va)

    best_p0_og, st_v3_k = select_p0_og_inner_cv(qv3_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=2.75, p0_base=1.00)
    pr_v3_va = predict_og_p0_eu(qv3_va, st_v3_k, lam_cl=2.75, p0_base=1.00, p0_og=best_p0_og)
    pr_v3_va[:, [0, 1, 3]] = pr_v2_va[:, [0, 1, 3]]
    cl_v2_va = (pr_v2_va == 0).all(axis=1)
    pr_v3_va[cl_v2_va] = 0
    newly_cl = (~cl_v2_va) & (pr_v3_va == 0).all(axis=1)
    if np.any(newly_cl):
        pr_v3_va[newly_cl] = pr_v2_va[newly_cl]

    return dict(
        k=k,
        qv2_va=qv2_va,
        pr_v2_va=pr_v2_va,
        qv3_va=qv3_va,
        pr_v3_va=pr_v3_va,
        og12_alpha=c_og12.alpha_,
        bk3_cfg=c_bk3.cfg_,
        p0_og=best_p0_og,
    )


def main():
    t0 = time.time()
    # Import canonical module name so pickle never stores `__main__`!
    import upgrade_deconf_v3_cleanlocked as canonical_mod

    set_name = "resident_v7"
    members = SETS[set_name]
    names, gold, fold, _, P_dict = oof_pool(members)
    _, _, _, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)

    proto_cfg = dc._build_routed_gate_cfg(members)
    for g_dict in (_G, uob._G):
        g_dict.update(fold=fold, gold=gold, is_orig=is_orig, groups=groups, P=P_dict, members=members, proto_cfg=proto_cfg)

    print(f"=== [deconf_v3] Running Zero-Leakage 8-Fold Nested OOF Verification (N={N}) ===", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        f_res = sorted(pool.map(_fold_worker_v3, list(range(8))), key=lambda d: d["k"])

    Q_v2_oof = np.zeros((N, 6, 4), dtype=np.float64)
    pr_v2_oof = np.zeros((N, 6), dtype=np.int64)
    Q_v3_oof = np.zeros((N, 6, 4), dtype=np.float64)
    pr_v3_oof = np.zeros((N, 6), dtype=np.int64)
    for r in f_res:
        va = fold == r["k"]
        Q_v2_oof[va] = r["qv2_va"]
        pr_v2_oof[va] = r["pr_v2_va"]
        Q_v3_oof[va] = r["qv3_va"]
        pr_v3_oof[va] = r["pr_v3_va"]

    rep_v2 = uob.report_line("deconf_v2 (54.44 baseline)", Q_v2_oof, pr_v2_oof, gold, is_orig)
    rep_v3 = uob.report_line("deconf_v3 (OG12+BkV3+OGp0+ClnLock)", Q_v3_oof, pr_v3_oof, gold, is_orig)
    print("  Fold configs (og12_alpha, bk3_cfg, p0_og):", [(r["og12_alpha"], r["bk3_cfg"], r["p0_og"]) for r in f_res], flush=True)

    # Fit full-OOF CleanLockedDeconfV3Bundle on all 3,573 OOF rows
    v2_pkl = os.path.join(REPO, "results/r55/final/resident_v7_decoder_deconf_v2.pkl")
    bnd_v2: uob.UpgradedDeconfBundle = pickle.load(open(v2_pkl, "rb"))["decoder"]
    Q_full_base = bnd_v2.base_decoder.pool_from_member_dict(P_dict)
    Q_full_nobk = uob.pool_without_bkseg1(bnd_v2.base_decoder, P_dict)

    cal_og12_full = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(2, 4, 8, 12)).fit(
        canonical_mod.extract_og_spec_covs(Q_full_base, P_dict, members), Q_full_base[:, 4], gold[:, 4], groups, is_orig=is_orig
    )
    cal_bk3_full = canonical_mod.BokehSynergyV3(w_supp=0.20).fit(
        Q_full_nobk[:, 5], c.renorm(P_dict["pe_bkseg1"])[:, 5], gold[:, 5], groups, is_orig
    )
    p0_og_full, st_v3_full = canonical_mod.select_p0_og_inner_cv(
        Q_v3_oof, gold, groups, is_orig, lam_cl=float(bnd_v2.eu_decoder.lam_cl), p0_base=float(bnd_v2.eu_decoder.p0_scale)
    )

    print(
        f"\n[Full OOF Fit] cal_og12: alpha={cal_og12_full.alpha_} (topk={cal_og12_full.best_topk_}, l2={cal_og12_full.best_l2_}) | "
        f"cal_bk3: cfg={cal_bk3_full.cfg_} | p0_og={p0_og_full}",
        flush=True,
    )
    if cal_og12_full.alpha_ > 0:
        for ci, wi in zip(cal_og12_full.sel_cols_, cal_og12_full.w_):
            print(f"    cal_og12 feat {OG_SPEC_FEAT_NAMES[ci]:<20}: w = {wi:+.4f}")

    bnd_v3 = canonical_mod.CleanLockedDeconfV3Bundle(
        v2_bundle=copy.deepcopy(bnd_v2),
        cal_og12=cal_og12_full,
        cal_bk3=cal_bk3_full,
        lam_cl=float(bnd_v2.eu_decoder.lam_cl),
        p0_base=float(bnd_v2.eu_decoder.p0_scale),
        p0_og=p0_og_full,
        st_v3=st_v3_full,
    )

    out_pkl = os.path.join(REPO, "results/r55/final/resident_v7_decoder_deconf_v3.pkl")
    out_meta = os.path.join(REPO, "results/r55/final/resident_v7_decoder_deconf_v3.json")
    pickle.dump({"decoder": bnd_v3, "members": members, "rule": "deconf_v3"}, open(out_pkl, "wb"))

    meta = {
        "set": set_name,
        "rule": "deconf_v3",
        "clean_image_locked_to_v2": True,
        "eu_lam_cl": float(bnd_v3.lam_cl),
        "eu_p0_base": float(bnd_v3.p0_base),
        "eu_p0_og": float(bnd_v3.p0_og),
        "cal_og12": {
            "alpha": float(cal_og12_full.alpha_),
            "l2": float(cal_og12_full.best_l2_),
            "topk": int(cal_og12_full.best_topk_),
            "weights": {OG_SPEC_FEAT_NAMES[ci]: round(float(wi), 6) for ci, wi in zip(cal_og12_full.sel_cols_, cal_og12_full.w_)},
        },
        "cal_bk3_cfg": list(cal_bk3_full.cfg_),
        "nested_oof_v2": rep_v2,
        "nested_oof_v3": rep_v3,
    }
    json.dump(meta, open(out_meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Verify `final_engine.load_decoder("resident_v7", "deconf_v3")` unpickles cleanly!
    dec_fn, _ = load_decoder("resident_v7", "deconf_v3")

    # Decode test_150 & final_644 and verify N=1 == N=batch + exact Clean-image invariance
    z_150 = np.load(os.path.join(REPO, "results/r55/os_og_breakthrough/pred_nriqa_dataset_test_os_og_breakthrough_member_probs.npz"), allow_pickle=True)
    z_644 = np.load(os.path.join(REPO, "results/r55/os_og_breakthrough_644/pred_nriqa_dataset_final_644_os_og_breakthrough_member_probs.npz"), allow_pickle=True)
    P_150 = {m: z_150[f"p_{m}"].astype(np.float64) if f"p_{m}" in z_150.files else z_150[m].astype(np.float64) for m in members}
    P_644 = {m: z_644[f"p_{m}"].astype(np.float64) if f"p_{m}" in z_644.files else z_644[m].astype(np.float64) for m in members}
    nm_150 = [str(x) for x in z_150["names"]]
    nm_644 = [str(x) for x in z_644["names"]]

    pr_v2_150, q_v2_150 = bnd_v2.predict(P_150)
    pr_v3_150, q_v3_150 = dec_fn(P_150)
    assert np.array_equal((pr_v2_150 == 0).all(1), (pr_v3_150 == 0).all(1)), "Clean image set changed on test_150!"
    for i in range(len(nm_150)):
        p1, q1 = dec_fn({m: v[i : i + 1] for m, v in P_150.items()})
        assert np.array_equal(p1[0], pr_v3_150[i]) and np.max(np.abs(q1[0] - q_v3_150[i])) < 1e-12

    pr_v2_644, q_v2_644 = bnd_v2.predict(P_644)
    pr_v3_644, q_v3_644 = dec_fn(P_644)
    assert np.array_equal((pr_v2_644 == 0).all(1), (pr_v3_644 == 0).all(1)), "Clean image set changed on final_644!"
    for i in range(len(nm_644)):
        p1, q1 = dec_fn({m: v[i : i + 1] for m, v in P_644.items()})
        assert np.array_equal(p1[0], pr_v3_644[i]) and np.max(np.abs(q1[0] - q_v3_644[i])) < 1e-12

    out_150_a = os.path.join(REPO, "results/final/pred_nriqa_dataset_test_r55_resident_v7_deconf_v3.json")
    out_644_a = os.path.join(REPO, "results/final/pred_nriqa_dataset_final_644_r55_resident_v7_deconf_v3.json")
    sprint_01 = os.path.join(REPO, "final_sprint_delivery/01_核心预测结果JSON")
    sprint_03 = os.path.join(REPO, "final_sprint_delivery/03_最后冲刺核心代码与脚本")
    sprint_04 = os.path.join(REPO, "final_sprint_delivery/04_最终校准解码器与实测审计账本")
    out_150_b = os.path.join(sprint_01, "pred_nriqa_dataset_test_r55_resident_v7_deconf_v3.json")
    out_644_b = os.path.join(sprint_01, "pred_nriqa_dataset_final_644_r55_resident_v7_deconf_v3.json")

    j150_obj = c.format_official_submission(nm_150, pr_v3_150)[0]
    j644_obj = c.format_official_submission(nm_644, pr_v3_644)[0]
    for p_dst, obj in ((out_150_a, j150_obj), (out_150_b, j150_obj), (out_644_a, j644_obj), (out_644_b, j644_obj)):
        os.makedirs(os.path.dirname(p_dst), exist_ok=True)
        json.dump(obj, open(p_dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if os.path.isdir(sprint_03):
        shutil.copy2(__file__, os.path.join(sprint_03, "upgrade_deconf_v3_cleanlocked.py"))
        shutil.copy2(os.path.join(REPO, "code/r55/final_engine.py"), os.path.join(sprint_03, "final_engine.py"))
        shutil.copy2(os.path.join(REPO, "code/r55/serve_engine.py"), os.path.join(sprint_03, "serve_engine.py"))
    if os.path.isdir(sprint_04):
        shutil.copy2(out_pkl, os.path.join(sprint_04, "resident_v7_decoder_deconf_v3.pkl"))
        shutil.copy2(out_meta, os.path.join(sprint_04, "resident_v7_decoder_deconf_v3.json"))

    validate_prediction_file(out_150_b, os.path.join(REPO, "results/v2_full/template_150.json"))
    validate_prediction_file(out_644_b, os.path.join(REPO, "results/final/template_final_644.json"))

    print("\n--- Exact Diff vs 54.44 (deconf_v2) ---")
    d_150 = np.where(pr_v2_150 != pr_v3_150)
    print(f"  test_150 changed: {len(d_150[0])} / 900 cells (Clean images: {(pr_v2_150==0).all(1).sum()} -> {(pr_v3_150==0).all(1).sum()}, 100% same mask!)")
    for i, d in zip(*d_150):
        print(
            f"    {nm_150[i]:<52} {DIMS_ZH[d]}: {GRADES_ZH[pr_v2_150[i,d]]} -> {GRADES_ZH[pr_v3_150[i,d]]}  "
            f"v2={np.round(q_v2_150[i,d],3).tolist()} -> v3={np.round(q_v3_150[i,d],3).tolist()}"
        )

    d_644 = np.where(pr_v2_644 != pr_v3_644)
    print(f"  final_644 changed: {len(d_644[0])} / 3864 cells (Clean images: {(pr_v2_644==0).all(1).sum()} -> {(pr_v3_644==0).all(1).sum()}, 100% same mask!)")
    for i, d in zip(*d_644):
        print(
            f"    {nm_644[i]:<52} {DIMS_ZH[d]}: {GRADES_ZH[pr_v2_644[i,d]]} -> {GRADES_ZH[pr_v3_644[i,d]]}  "
            f"v2={np.round(q_v2_644[i,d],3).tolist()} -> v3={np.round(q_v3_644[i,d],3).tolist()}"
        )
    print(f"\n=== ALL DONE in {time.time()-t0:.1f}s -> {out_150_b} ===", flush=True)


if __name__ == "__main__":
    main()
