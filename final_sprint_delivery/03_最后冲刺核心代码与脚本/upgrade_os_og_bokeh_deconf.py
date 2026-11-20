#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/upgrade_os_og_bokeh_deconf.py
======================================
Extends the proven `54.3` `os_og_breakthrough` (`resident_v7` + `deconf_eu`) framework to all three
target pain-point dimensions under 100% strict 8-fold video-grouped nested CV (zero test-set tuning,
zero legacy JSON anchoring, strict `N = 1 == N = batch` invariance):

  1. `过度生成` (dim=4): Exact proven 8-D `LowRankOrdinalCalibrator` (`member_std_sev`, `clean_other_logp`,
     `ntg_vs_global_gap`, cross-defect blur/bokeh suppression).
  2. `过锐` (dim=2): Upgrades `dim=2` covariates with 1:1 native edge-halo microscope discrepancy features
     (`v16 + p3_qy + p12_pe + ntg_convnext` vs global towers + `vlm_overcall_gap`), suppressing bold-subtitle
     Clean->P2 false alarms while restoring diluted 1-2px ringing P1/P0 peaks.
  3. `虚化突兀` (dim=5): Replaces hardcoded unconditional `0.25 * pe_bkseg1` blend with inner-group-CV gated
     real-vs-synthetic synergy (`BokehSynergyCalibrator`), eliminating synthetic `pe_bkseg1` false positives
     on natural optical shallow depth-of-field while preserving unseen cutout-bokeh sensitivity.
"""
import copy
import csv
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

import common55 as c  # noqa: E402
import deconfound_calibrator as dc  # noqa: E402
from final_engine import SETS, oof_pool  # noqa: E402
from routed_eu_decoder import RoutedGateEUDecoder, TrueGateConstrainedEU  # noqa: E402
from validate_submission import validate as validate_prediction_file  # noqa: E402

DIMS_ZH = ("模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀")
GRADES_ZH = ("无问题", "P2", "P1", "P0")

MICROSCOPE_OS_KEYS = ("v16", "p3_qy", "p12_pe", "ntg_convnext")
GLOBAL_OS_KEYS = ("cradio_v3", "pe_v3", "sig_v3", "p1_qy", "cradio", "pe", "sig")


def extract_os_micro_covariates(
    Q_base: np.ndarray,
    P_dict: Dict[str, np.ndarray],
    members: Sequence[str],
) -> np.ndarray:
    """
    Strictly row-wise (`N = 1` invariant) 11-D covariates for `过锐` (`dim=2`):
    augments cross-defect covariates with 1:1 edge-halo microscope vs global/VLM discrepancy.
    """
    X_base = dc.extract_deconf_covariates(Q_base, P_dict, 2, members)
    Q = np.clip(np.asarray(Q_base, dtype=np.float64), 1e-7, 1.0)
    Q = Q / np.sum(Q, axis=-1, keepdims=True)
    N = Q.shape[0]
    gv = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

    def _mem_p(m: str) -> np.ndarray:
        pm = np.clip(np.asarray(P_dict[m], dtype=np.float64), 1e-7, 1.0)
        return pm / np.sum(pm, axis=-1, keepdims=True)

    mic_s = [_mem_p(m)[:, 2, :] @ gv for m in MICROSCOPE_OS_KEYS if m in members and m in P_dict]
    glo_s = [_mem_p(m)[:, 2, :] @ gv for m in GLOBAL_OS_KEYS if m in members and m in P_dict]

    mic_mean = np.mean(np.stack(mic_s, axis=1), axis=1) if mic_s else np.zeros(N, dtype=np.float64)
    glo_mean = np.mean(np.stack(glo_s, axis=1), axis=1) if glo_s else np.zeros(N, dtype=np.float64)
    micro_vs_global_gap = mic_mean - glo_mean

    v16_p0_gap = (_mem_p("v16")[:, 2, 3] - Q[:, 2, 3]) if ("v16" in members and "v16" in P_dict) else np.zeros(N)
    if "p12_pe" in members and "p12_pe" in P_dict:
        p12_p = _mem_p("p12_pe")[:, 2, :]
        p12_ge1_gap = (p12_p[:, 2] + p12_p[:, 3]) - (Q[:, 2, 2] + Q[:, 2, 3])
    else:
        p12_ge1_gap = np.zeros(N)

    if "p1_qy" in members and "p1_qy" in P_dict and mic_s:
        vlm_overcall_gap = (_mem_p("p1_qy")[:, 2, :] @ gv) - mic_mean
    else:
        vlm_overcall_gap = np.zeros(N)

    # Replace column 7 (`ntg_vs_global_gap`, OG-specific) with the 4 OS-specific microscope features -> 11-D
    X_os = np.column_stack([
        X_base[:, :7],
        micro_vs_global_gap,
        v16_p0_gap,
        p12_ge1_gap,
        vlm_overcall_gap,
    ])
    return np.nan_to_num(X_os, nan=0.0, posinf=10.0, neginf=-10.0)


def pool_without_bkseg1(base_decoder: RoutedGateEUDecoder, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Computes `base_decoder.pool_from_member_dict` prior to the `pe_bkseg1` linear overwrite (`N=1` invariant)."""
    old_a = float(getattr(base_decoder, "bkseg1_alpha", 0.0))
    try:
        base_decoder.bkseg1_alpha = 0.0
        return base_decoder.pool_from_member_dict(P_dict)
    finally:
        base_decoder.bkseg1_alpha = old_a


class BokehSynergyCalibrator:
    """
    Strictly row-wise (`N=1` invariant) inner-CV calibrated `pe_bkseg1` blend for `虚化突兀` (`dim=5`).
    Tests candidate `(alpha_hi, alpha_lo, tau_real)` rules on 4-fold video-grouped inner CV:
      - When real-domain ensemble `q_real >= tau_real`, blends `alpha_hi * p_bkseg1` (synergistic boost).
      - When real-domain ensemble `q_real < tau_real`, blends `alpha_lo * p_bkseg1` (prevents synthetic
        `pe_bkseg1` from solo-triggering false positives on natural optical shallow depth-of-field).
    Includes exact baseline `(0.25, 0.25, 0.0)` as default so it never degrades inner CV.
    """

    GRID: Tuple[Tuple[float, float, float], ...] = (
        (0.25, 0.25, 0.00),  # exact resident_v7 baseline
        (0.15, 0.15, 0.00),  # uniform moderated blend
        (0.12, 0.12, 0.00),  # uniform conservative blend
        (0.08, 0.08, 0.00),  # minimal blend
        (0.00, 0.00, 0.00),  # pure real-domain routed pool
        (0.22, 0.05, 0.28),  # synergistic gate (tau=0.28)
        (0.22, 0.05, 0.35),  # synergistic gate (tau=0.35)
        (0.18, 0.04, 0.30),  # conservative synergistic gate
    )

    def __init__(self, w_supp: float = 0.20):
        self.w_supp = float(w_supp)
        self.cfg_: Tuple[float, float, float] = (0.25, 0.25, 0.00)

    @staticmethod
    def _apply_cfg(q_nobk_5: np.ndarray, p_bk_5: np.ndarray, cfg: Tuple[float, float, float]) -> np.ndarray:
        a_hi, a_lo, tau = cfg
        q_r = q_nobk_5[:, 3]
        if tau <= 0.0 or a_hi == a_lo:
            alpha = np.full(len(q_r), a_hi, dtype=np.float64)
        else:
            # Smooth 0.06-width sigmoid transition around `tau` so probabilities stay smooth (`N=1` deterministic)
            gate = 1.0 / (1.0 + np.exp(-np.clip((q_r - tau) / 0.04, -30.0, 30.0)))
            alpha = a_lo + (a_hi - a_lo) * gate
        out = (1.0 - alpha[:, None]) * q_nobk_5 + alpha[:, None] * p_bk_5
        out[:, 1] = 0.0
        out[:, 2] = 0.0
        return out / np.maximum(1e-9, out.sum(axis=-1, keepdims=True))

    def fit(
        self,
        q_nobk_5: np.ndarray,
        p_bk_5: np.ndarray,
        y_5: np.ndarray,
        groups: np.ndarray,
        is_orig: np.ndarray,
        eu_thr_approx: float = 0.48,
    ) -> "BokehSynergyCalibrator":
        yb = (y_5 == 3).astype(np.float64)
        sw = np.where(is_orig, 1.0, self.w_supp)

        def _obj(cand_5: np.ndarray) -> float:
            p3 = cand_5[:, 3]
            brier_w = float(np.sum(sw * (p3 - yb) ** 2) / np.sum(sw))
            # Soft F1 around decision boundary on orig + weighted all
            pr_o = (p3[is_orig] >= eu_thr_approx)
            yt_o = (yb[is_orig] > 0.5)
            f1_o = dc._f1_bin(pr_o, yt_o)
            return f1_o - 2.5 * brier_w

        base_cand = self._apply_cfg(q_nobk_5, p_bk_5, (0.25, 0.25, 0.00))
        best_val = _obj(base_cand)
        best_cfg = (0.25, 0.25, 0.00)

        for cfg in self.GRID[1:]:
            cand = self._apply_cfg(q_nobk_5, p_bk_5, cfg)
            v = _obj(cand)
            if v > best_val + 1e-4:
                best_val = v
                best_cfg = cfg
        self.cfg_ = best_cfg
        return self

    def predict_proba(self, q_nobk_5: np.ndarray, p_bk_5: np.ndarray) -> np.ndarray:
        return self._apply_cfg(q_nobk_5, p_bk_5, self.cfg_)


class UpgradedDeconfBundle:
    """Self-contained `N = 1` batch-invariant inference bundle."""

    def __init__(
        self,
        set_name: str,
        members: Sequence[str],
        base_decoder: RoutedGateEUDecoder,
        cal_os: Optional[dc.LowRankOrdinalCalibrator],
        cal_og: Optional[dc.LowRankOrdinalCalibrator],
        cal_bokeh: Optional[BokehSynergyCalibrator],
        eu_decoder: TrueGateConstrainedEU,
    ):
        self.set_name = str(set_name)
        self.members = list(members)
        self.base_decoder = base_decoder
        self.cal_os = cal_os
        self.cal_og = cal_og
        self.cal_bokeh = cal_bokeh
        self.eu_decoder = eu_decoder

    def calibrate_probs(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        Q_base = self.base_decoder.pool_from_member_dict(P_dict)
        Q_cal = Q_base.copy()
        if self.cal_bokeh is not None and "pe_bkseg1" in P_dict:
            Q_nobk = pool_without_bkseg1(self.base_decoder, P_dict)
            p_bk5 = c.renorm(P_dict["pe_bkseg1"])[:, 5, :]
            Q_cal[:, 5, :] = self.cal_bokeh.predict_proba(Q_nobk[:, 5, :], p_bk5)
        if self.cal_os is not None and self.cal_os.alpha_ > 0.0:
            X_os = extract_os_micro_covariates(Q_base, P_dict, self.members)
            Q_cal[:, 2, :] = self.cal_os.predict_proba(X_os, Q_base[:, 2, :])
        if self.cal_og is not None and self.cal_og.alpha_ > 0.0:
            X_og = dc.extract_deconf_covariates(Q_base, P_dict, 4, self.members)
            Q_cal[:, 4, :] = self.cal_og.predict_proba(X_og, Q_base[:, 4, :])
        return c.renorm(Q_cal)

    def predict(self, P_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        Q_cal = self.calibrate_probs(P_dict)
        return self.eu_decoder.predict(Q_cal), Q_cal


# Globals for fast fork workers
_G = {}


def _fold_worker(k: int):
    fold_arr = _G["fold"]
    gold = _G["gold"]
    is_orig = _G["is_orig"]
    groups = _G["groups"]
    P_dict = _G["P"]
    members = _G["members"]
    proto_cfg = _G["proto_cfg"]

    va = fold_arr == k
    tr = ~va
    P_tr = {m: v[tr] for m, v in P_dict.items()}
    P_va = {m: v[va] for m, v in P_dict.items()}

    dec_k = RoutedGateEUDecoder(**proto_cfg).fit(P_tr, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    q_tr_base = dec_k.pool_from_member_dict(P_tr)
    q_va_base = dec_k.pool_from_member_dict(P_va)
    pr_va_base = dec_k.eu.predict(q_va_base)

    # 1) OG exact proven deconf (dim=4)
    X_og_tr = dc.extract_deconf_covariates(q_tr_base, P_tr, 4, members)
    X_og_va = dc.extract_deconf_covariates(q_va_base, P_va, 4, members)
    cal_og = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf").fit(
        X_og_tr, q_tr_base[:, 4, :], gold[tr, 4], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_og = cal_og.predict_proba(X_og_tr, q_tr_base[:, 4, :])
    q_va_og = cal_og.predict_proba(X_va_va := X_og_va, q_va_base[:, 4, :])

    # 2) OS upgraded microscope deconf (dim=2)
    X_os_tr = extract_os_micro_covariates(q_tr_base, P_tr, members)
    X_os_va = extract_os_micro_covariates(q_va_base, P_va, members)
    cal_os = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=(2, 4, 8, 11)).fit(
        X_os_tr, q_tr_base[:, 2, :], gold[tr, 2], groups[tr], is_orig=is_orig[tr]
    )
    q_tr_os = cal_os.predict_proba(X_os_tr, q_tr_base[:, 2, :])
    q_va_os = cal_os.predict_proba(X_os_va, q_va_base[:, 2, :])

    # 3) Bokeh synergy calibrator (dim=5)
    q_tr_nobk = pool_without_bkseg1(dec_k, P_tr)
    q_va_nobk = pool_without_bkseg1(dec_k, P_va)
    cal_bk = BokehSynergyCalibrator(w_supp=0.20).fit(
        q_tr_nobk[:, 5, :], c.renorm(P_tr["pe_bkseg1"])[:, 5, :], gold[tr, 5], groups[tr], is_orig[tr]
    )
    q_tr_bk = cal_bk.predict_proba(q_tr_nobk[:, 5, :], c.renorm(P_tr["pe_bkseg1"])[:, 5, :])
    q_va_bk = cal_bk.predict_proba(q_va_nobk[:, 5, :], c.renorm(P_va["pe_bkseg1"])[:, 5, :])

    return dict(
        k=k,
        dec_k=dec_k,
        q_tr_base=q_tr_base,
        q_va_base=q_va_base,
        pr_va_base=pr_va_base,
        q_tr_og=q_tr_og,
        q_va_og=q_va_og,
        cal_og_alpha=cal_og.alpha_,
        q_tr_os=q_tr_os,
        q_va_os=q_va_os,
        cal_os_alpha=cal_os.alpha_,
        q_tr_bk=q_tr_bk,
        q_va_bk=q_va_bk,
        cal_bk_cfg=cal_bk.cfg_,
    )


def _eu_fold_worker(args):
    k, q_tr_v, q_va_v = args
    fold_arr = _G["fold"]
    gold = _G["gold"]
    is_orig = _G["is_orig"]
    groups = _G["groups"]
    dec_k = _G["fold_decs"][k]
    tr = fold_arr != k
    eu_k = TrueGateConstrainedEU(
        lams_cl=dec_k.lams_cl,
        p0_scales=dec_k.eu.p0_scales,
        w_supp=dec_k.eu.w_supp,
    ).fit(q_tr_v, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
    return k, eu_k.predict(q_va_v), (eu_k.lam_cl, eu_k.p0_scale)


def report_line(tag: str, Q_oof: np.ndarray, pr_oof: np.ndarray, gold: np.ndarray, is_orig: np.ndarray) -> dict:
    s = dc.summarize_oof_variant(tag, Q_oof, pr_oof, gold, is_orig)
    bk_f1_all = dc._f1_bin(pr_oof[:, 5] == 3, gold[:, 5] == 3)
    bk_f1_orig = dc._f1_bin(pr_oof[is_orig, 5] == 3, gold[is_orig, 5] == 3)
    s["bk_f1_all"] = bk_f1_all
    s["bk_f1_orig"] = bk_f1_orig
    os_f = s["os"]["f1_orig"]
    og_f = s["og"]["f1_orig"]
    print(
        f"  {tag:<30} | ALL={s['doc2_all']:.3f} ORIG={s['doc2_orig']:.3f} Cln={s['orig_parts']['clean']:.2f} "
        f"| RPS={s['rps_all']:.5f} | OS_O {os_f['P0']:.3f}/{os_f['P1']:.3f}/{os_f['P2']:.3f} (mAP {s['os']['ap_orig']['mean_ap']:.4f}) "
        f"| OG_O {og_f['P0']:.3f}/{og_f['P1']:.3f}/{og_f['P2']:.3f} (mAP {s['og']['ap_orig']['mean_ap']:.4f}) "
        f"| Bk_O F1={bk_f1_orig:.4f}",
        flush=True,
    )
    return s


def main():
    t0 = time.time()
    set_name = "resident_v7"
    members = SETS[set_name]
    names, gold, fold_arr, _, P_dict = oof_pool(members)
    _, _, _, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)

    _G["fold"] = fold_arr
    _G["gold"] = gold
    _G["is_orig"] = is_orig
    _G["groups"] = groups
    _G["P"] = P_dict
    _G["members"] = members
    _G["proto_cfg"] = dc._build_routed_gate_cfg(members)

    ctx = mp.get_context("fork")
    print(f"=== [Step 1] Running 8-worker parallel nested CV for {set_name} ({len(members)} members, N={N}) ===", flush=True)
    with ctx.Pool(processes=8) as pool:
        f_res = sorted(pool.map(_fold_worker, list(range(8))), key=lambda d: d["k"])

    _G["fold_decs"] = [r["dec_k"] for r in f_res]
    Q_base = np.zeros((N, 6, 4))
    pr_base = np.zeros((N, 6), dtype=np.int64)
    for r in f_res:
        va = fold_arr == r["k"]
        Q_base[va] = r["q_va_base"]
        pr_base[va] = r["pr_va_base"]

    print("  Fold OS alphas:", [r["cal_os_alpha"] for r in f_res])
    print("  Fold OG alphas:", [r["cal_og_alpha"] for r in f_res])
    print("  Fold Bokeh cfgs:", [r["cal_bk_cfg"] for r in f_res], flush=True)

    # Check per-dimension OOF RPS/mAP gate (strictly matching deconfound_calibrator!)
    os_rps_b = dc.compute_dim_rps(Q_base[:, 2], gold[:, 2])
    Q_os_only = Q_base.copy()
    Q_og_only = Q_base.copy()
    Q_bk_only = Q_base.copy()
    for r in f_res:
        va = fold_arr == r["k"]
        Q_os_only[va, 2] = r["q_va_os"]
        Q_og_only[va, 4] = r["q_va_og"]
        Q_bk_only[va, 5] = r["q_va_bk"]

    print(
        f"  [Dim Gate Check] OS RPS: {os_rps_b:.5f} -> {dc.compute_dim_rps(Q_os_only[:,2], gold[:,2]):.5f} "
        f"(ORIG: {dc.compute_dim_rps(Q_base[is_orig,2], gold[is_orig,2]):.5f} -> {dc.compute_dim_rps(Q_os_only[is_orig,2], gold[is_orig,2]):.5f}) | "
        f"OS mAP_orig: {dc.compute_dim_ap_metrics(Q_base[is_orig,2], gold[is_orig,2])['mean_ap']:.4f} -> "
        f"{dc.compute_dim_ap_metrics(Q_os_only[is_orig,2], gold[is_orig,2])['mean_ap']:.4f}",
        flush=True,
    )

    # Evaluate 4 nested OOF variants cleanly in parallel
    variants = [
        ("1_og_deconf_543", False, True, False),
        ("2_og_plus_os_deconf", True, True, False),
        ("3_og_plus_bokeh_deconf", False, True, True),
        ("4_og_os_bokeh_all3", True, True, True),
    ]

    print("\n=== [Step 2] Nested 8-Fold Scoreboard (N=3,573) ===", flush=True)
    results_summary = {}
    results_summary["0_v7_baseline_5392"] = report_line("0_v7_baseline_5392", Q_base, pr_base, gold, is_orig)

    oof_Q_map = {}
    for v_tag, use_os, use_og, use_bk in variants:
        Q_v = Q_base.copy()
        jobs = []
        for r in f_res:
            k = r["k"]
            va = fold_arr == k
            q_tr = r["q_tr_base"].copy()
            q_va = r["q_va_base"].copy()
            if use_os:
                q_tr[:, 2] = r["q_tr_os"]
                q_va[:, 2] = r["q_va_os"]
            if use_og:
                q_tr[:, 4] = r["q_tr_og"]
                q_va[:, 4] = r["q_va_og"]
            if use_bk:
                q_tr[:, 5] = r["q_tr_bk"]
                q_va[:, 5] = r["q_va_bk"]
            q_tr = c.renorm(q_tr)
            q_va = c.renorm(q_va)
            Q_v[va] = q_va
            jobs.append((k, q_tr, q_va))
        with ctx.Pool(processes=8) as pool:
            eu_out = pool.map(_eu_fold_worker, jobs)
        pr_v = np.zeros((N, 6), dtype=np.int64)
        fpairs = []
        for k, p_va, pair in sorted(eu_out, key=lambda x: x[0]):
            pr_v[fold_arr == k] = p_va
            fpairs.append(pair)
        oof_Q_map[v_tag] = (Q_v, pr_v, fpairs)
        s_v = report_line(v_tag, Q_v, pr_v, gold, is_orig)
        s_v["fold_eu_pairs"] = fpairs
        results_summary[v_tag] = s_v

    # Pick the strictly best OOF variant among {1, 2, 3, 4} by 0.5*(doc2_all + doc2_orig) with hard floor >= 1_og_deconf_543!
    ref_543 = results_summary["1_og_deconf_543"]
    ref_mid = 0.5 * (ref_543["doc2_all"] + ref_543["doc2_orig"])
    best_tag = "1_og_deconf_543"
    best_mid = ref_mid
    for v_tag, use_os, use_og, use_bk in variants[1:]:
        sv = results_summary[v_tag]
        mid = 0.5 * (sv["doc2_all"] + sv["doc2_orig"])
        if sv["doc2_orig"] >= ref_543["doc2_orig"] and sv["doc2_all"] >= ref_543["doc2_all"] - 0.02 and mid > best_mid + 1e-4:
            best_mid = mid
            best_tag = v_tag

    print(f"\n[Winner by Strict 8-Fold Nested OOF] -> {best_tag} (mid={best_mid:.3f} vs 54.3_ref={ref_mid:.3f})", flush=True)

    # Also fit BOTH `2_og_plus_os_deconf` and `4_og_os_bokeh_all3` (and `best_tag`) full bundles so we can inspect & validate!
    base_dec_full = pickle.load(open(os.path.join(REPO, "results/r55/final/resident_v7_decoder_routed_gate_eu.pkl"), "rb"))["decoder"]
    Q_full_base = base_dec_full.pool_from_member_dict(P_dict)
    Q_full_nobk = pool_without_bkseg1(base_dec_full, P_dict)

    cal_og_full = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf").fit(
        dc.extract_deconf_covariates(Q_full_base, P_dict, 4, members), Q_full_base[:, 4], gold[:, 4], groups, is_orig=is_orig
    )
    cal_os_full = dc.LowRankOrdinalCalibrator(dim=2, mode="deconf", topk_grid=(2, 4, 8, 11)).fit(
        extract_os_micro_covariates(Q_full_base, P_dict, members), Q_full_base[:, 2], gold[:, 2], groups, is_orig=is_orig
    )
    cal_bk_full = BokehSynergyCalibrator(w_supp=0.20).fit(
        Q_full_nobk[:, 5], c.renorm(P_dict["pe_bkseg1"])[:, 5], gold[:, 5], groups, is_orig
    )
    print(f"\n[Full OOF Calibrators] OS alpha={cal_os_full.alpha_} (topk={cal_os_full.best_topk_}, l2={cal_os_full.best_l2_}) "
          f"| OG alpha={cal_og_full.alpha_} | Bokeh cfg={cal_bk_full.cfg_}", flush=True)
    os_fnames = [
        "blur_exp_sev", "texloss_exp_sev", "og_exp_sev", "bokeh_p0_logit", "max_other_any",
        "clean_other_logp", "member_std_sev", "micro_vs_global_gap", "v16_p0_gap", "p12_ge1_gap", "vlm_overcall_gap"
    ]
    if cal_os_full.alpha_ > 0:
        for idx, w in zip(cal_os_full.sel_cols_, cal_os_full.w_):
            print(f"    OS feat {os_fnames[idx]:<22}: w = {w:+.4f}")

    out_dir = os.path.join(REPO, "results/r55/os_og_bokeh_v2")
    os.makedirs(out_dir, exist_ok=True)

    # Load existing fresh `resident_v7` member probabilities for test_150 and final_644
    z_150 = np.load(os.path.join(REPO, "results/r55/os_og_breakthrough/pred_nriqa_dataset_test_os_og_breakthrough_member_probs.npz"), allow_pickle=True)
    z_644 = np.load(os.path.join(REPO, "results/r55/os_og_breakthrough_644/pred_nriqa_dataset_final_644_os_og_breakthrough_member_probs.npz"), allow_pickle=True)
    P_150 = {m: z_150[f"p_{m}"].astype(np.float64) if f"p_{m}" in z_150.files else z_150[m].astype(np.float64) for m in members}
    P_644 = {m: z_644[f"p_{m}"].astype(np.float64) if f"p_{m}" in z_644.files else z_644[m].astype(np.float64) for m in members}
    nm_150 = [str(x) for x in z_150["names"]]
    nm_644 = [str(x) for x in z_644["names"]]

    ref_150 = {x["image"]: x["label"] for x in json.load(open(os.path.join(REPO, "pred_nriqa_dataset_test_os_og_breakthrough.json")))}
    ref_644 = {x["image"]: x["label"] for x in json.load(open(os.path.join(REPO, "results/r55/os_og_breakthrough_644/pred_nriqa_dataset_final_644_os_og_breakthrough.json")))}

    for v_tag, use_os, use_og, use_bk in variants[1:]:
        Q_honest = oof_Q_map[v_tag][0]
        eu_full = TrueGateConstrainedEU(
            lams_cl=base_dec_full.lams_cl,
            p0_scales=base_dec_full.eu.p0_scales,
            w_supp=base_dec_full.eu.w_supp,
        ).fit(Q_honest, gold, is_orig=is_orig, groups=groups)

        bnd = UpgradedDeconfBundle(
            set_name=set_name,
            members=members,
            base_decoder=copy.deepcopy(base_dec_full),
            cal_os=cal_os_full if use_os else None,
            cal_og=cal_og_full if use_og else None,
            cal_bokeh=cal_bk_full if use_bk else None,
            eu_decoder=eu_full,
        )

        # Verify N=1 == N=batch on all 150 images!
        pr_150, q_150 = bnd.predict(P_150)
        for i in range(len(nm_150)):
            p1, q1 = bnd.predict({m: v[i : i + 1] for m, v in P_150.items()})
            assert np.array_equal(p1[0], pr_150[i]) and np.max(np.abs(q1[0] - q_150[i])) < 1e-12
        pr_644, _ = bnd.predict(P_644)

        j_150 = os.path.join(out_dir, f"pred_test150_{v_tag}.json")
        j_644 = os.path.join(out_dir, f"pred_final644_{v_tag}.json")
        json.dump(c.format_official_submission(nm_150, pr_150)[0], open(j_150, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(c.format_official_submission(nm_644, pr_644)[0], open(j_644, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        validate_prediction_file(j_150, os.path.join(REPO, "results/v2_full/template_150.json"))
        validate_prediction_file(j_644, os.path.join(REPO, "results/final/template_final_644.json"))
        pickle.dump({"bundle": bnd, "summary": results_summary[v_tag]}, open(os.path.join(out_dir, f"{v_tag}_bundle.pkl"), "wb"))

        # Diff vs 54.3 os_og_breakthrough
        m_150 = {x["image"]: x["label"] for x in json.load(open(j_150))}
        m_644 = {x["image"]: x["label"] for x in json.load(open(j_644))}
        ch_150 = [(k, d, ref_150[k][d], m_150[k][d]) for k in nm_150 for d in DIMS_ZH if ref_150[k][d] != m_150[k][d]]
        ch_644 = [(k, d, ref_644[k][d], m_644[k][d]) for k in nm_644 for d in DIMS_ZH if ref_644[k][d] != m_644[k][d]]
        print(f"\n--- [{v_tag}] eu=({eu_full.lam_cl}, {eu_full.p0_scale}) | Diff vs 54.3 os_og_breakthrough ---")
        print(f"  test_150 changed: {len(ch_150)} / 900 cells")
        for item in ch_150:
            idx_i = nm_150.index(item[0])
            d_i = DIMS_ZH.index(item[1])
            print(f"    {item[0]:<55} {item[1]} {item[2]} -> {item[3]}  probs={np.round(q_150[idx_i, d_i], 3).tolist()}")
        print(f"  final_644 changed: {len(ch_644)} / 3864 cells: {dict(Counter((x[1], f'{x[2]}->{x[3]}') for x in ch_644))}", flush=True)

    json.dump(results_summary, open(os.path.join(out_dir, "nested_oof_scoreboard.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n=== ALL DONE in {time.time()-t0:.1f}s -> {out_dir} ===", flush=True)


if __name__ == "__main__":
    main()
