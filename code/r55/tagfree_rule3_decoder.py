#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/tagfree_rule3_decoder.py
=================================
Step 1 of the 100% batch-tag-free clean deliverable plan (owner: opus).

1. `TagFreeGateConstrainedEU` & `TagFreeRoutedGateEU`:
   - 100% batch-tag-free (`TAG-FREE`): uniform weight 1.0 on every training row in DirDiag post-pool calibration,
     fixed-point EU statistics, `(lam_cl, p0_scale)` selection, `Perm654` / cluster-bootstrap `S_pos` safety gates,
     and `Clean-P0` guardrails.
   - Zero `is_orig` / `is_supp` / `w_supp` parameters or branches anywhere in the model or decoder classes.
2. `baseline`:
   - Runs strict 8-fold video-grouped nested CV (`data/qingyu_splits/qingyu_8fold.csv`) for a member set
     (`resident_v7_core`, `resident_v7`, or `rule3_prev`) and saves OOF predictions, 100%-data pickle, and JSON report
     under `results/r55/tagfree_baseline/`.
3. `gate`:
   - Deterministic CLI gate evaluator for Fold 0-1 pilots (or full 8-fold OOF candidates):
     * Solo `OS` & `OG` `AP_any`, `AP_p0`, within-positive `P2vP1` AUC, `P1vP0` AUC on the exact covered rows;
     * Cluster-bootstrap Gate #1 (`dAP_p0`, `dP1vP0`, `dP2vP1` with 500-draw video-cluster CI90) alongside 3 null-tower
       baselines on the identical rows;
     * Tag-Free **Rule-3 5-Member Stack** (`base_6b + ntg_convnext + p12_pe + spec_c + peg`) vs `RULE3_prev`
       (`iv6b_ep2 + ntg_convnext + p12_pe + p1_qy + peg`), `4specs`, and `resident_v7_core` on the identical rows,
       plus `N=1 == N=batch` determinism verification.
   - Supports evaluating a single candidate (`--cand --tag --slot`) or BOTH pilots jointly
     (`--cand ... --slot base_6b --cand2 ... --tag2 ... --slot2 spec_c`).

Usage:
  python code/r55/tagfree_rule3_decoder.py self-test
  python code/r55/tagfree_rule3_decoder.py baseline --set resident_v7_core
  python code/r55/tagfree_rule3_decoder.py gate --cand results/r55/iv6b_native_pilot --tag iv6b_native --slot base_6b
  python code/r55/tagfree_rule3_decoder.py gate --cand results/r55/qwen3vl_hires_pilot --tag qwen_hires --slot spec_c
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from scipy.optimize import minimize

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    _a = os.path.join(REPO, _p)
    if _a not in sys.path:
        sys.path.insert(0, _a)

import common55 as c  # noqa: E402
import decoder_eu as E  # noqa: E402
from eval_specialist_gate import apply_nested_routed_pool, fit_nested_routed_pool  # noqa: E402
from routed_eu_decoder import _fast_boot654_pos_risk, _true_eu_predict, compute_family_base_prior  # noqa: E402

OUT_DIR = os.path.join(REPO, "results/r55/tagfree_baseline")
IV6B_EP2_OOF = os.path.join(REPO, "results/r55/v6_debate/iv6b_ep2_8fold_oof.npz")
NULL_TOWERS = {
    "null_pe_glitch": "results/next55/towers/r55_glitch_s2026/pe_core_large",
    "null_pe_hp": "results/next55/towers/r55_hp_s2026/pe_core_large",
    "null_pe_fixed_s2028": "results/next55/towers/r55_fixed_s2028/pe_core_large",
}
RULE3_FIXED_SPECS = ["peg", "ntg_convnext", "p12_pe"]  # + base_6b + spec_c = 5 members


# ----------------------------------------------------------------------------------------------------------------------
# 100% Batch-Tag-Free Decoder Classes (Zero split/batch metadata anywhere)
# ----------------------------------------------------------------------------------------------------------------------
class TagFreeGateConstrainedEU:
    """
    100% uniform-weight fixed-point Expected Utility decoder (`beta == 1.0`).
    Fits fixed-point statistics and selects `(lam_cl, p0_scale)` across ALL training rows uniformly
    subject to 10,000-draw 654-image video-cluster bootstrap safety gates and Clean-P0 protection.
    """

    def __init__(
        self,
        lams_cl: Sequence[float] = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25),
        p0_scales: Sequence[float] = (0.85, 0.92, 1.00),
        max_perm_fail_pct: float = 1.00,
        max_clboot_fail_pct: float = 3.20,
    ):
        self.lams_cl = tuple(float(x) for x in lams_cl)
        self.p0_scales = tuple(float(x) for x in p0_scales)
        self.max_perm_fail_pct = float(max_perm_fail_pct)
        self.max_clboot_fail_pct = float(max_clboot_fail_pct)
        self.lam_cl: float = 2.25
        self.p0_scale: float = 0.92
        self.st: dict = {}

    @staticmethod
    def _uniform_stats(pred: np.ndarray, gold: np.ndarray) -> dict:
        pl = (pred[:, E.LD] == E.LG[None]).astype(np.float64)
        gl = (gold[:, E.LD] == E.LG[None]).astype(np.float64)
        tp = (pl * gl).sum(0)
        npred = pl.sum(0)
        npos = gl.sum(0)
        ncl = float((gold == 0).all(1).sum())
        return dict(
            F=2.0 * tp / np.maximum(1e-6, npred + npos),
            P=tp / np.maximum(1e-6, npred),
            npos=npos,
            npred=npred,
            ncl=max(1e-6, ncl),
            N=float(len(gold)),
        )

    @classmethod
    def _fit_stats(
        cls,
        Q: np.ndarray,
        gold: np.ndarray,
        lam_cl: float,
        p0_scale: float,
        iters: int = 30,
        alpha: float = 0.5,
    ) -> dict:
        pred0 = E.Decoder55().predict(Q)
        st = cls._uniform_stats(pred0, gold)
        best_res = float("inf")
        best_st = st
        for _ in range(iters):
            pr_now = _true_eu_predict(Q, st, lam_cl=lam_cl, p0_scale=p0_scale)
            st_ind = cls._uniform_stats(pr_now, gold)
            res = float(np.max(np.abs(st_ind["F"] - st["F"])))
            if res < best_res:
                best_res = res
                best_st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
            if res < 1e-6:
                return st_ind
            st = dict(
                F=(1.0 - alpha) * st["F"] + alpha * st_ind["F"],
                P=(1.0 - alpha) * st["P"] + alpha * st_ind["P"],
                npos=st_ind["npos"],
                npred=(1.0 - alpha) * st["npred"] + alpha * st_ind["npred"],
                ncl=st_ind["ncl"],
                N=st_ind["N"],
            )
        return best_st

    def fit(self, Q: np.ndarray, gold: np.ndarray, groups: Optional[np.ndarray] = None) -> "TagFreeGateConstrainedEU":
        Q = c.renorm(Q)
        if groups is None:
            groups = np.arange(len(gold))

        base_d55 = E.Decoder55().predict(Q)
        base_cp0 = int(((gold == 0).all(axis=1) & (base_d55 == 3).any(axis=1)).sum())
        max_allowed_cp0 = max(1, base_cp0)

        best = None
        for lam_cl in self.lams_cl:
            for p0_sc in self.p0_scales:
                st = self._fit_stats(Q, gold, lam_cl, p0_sc)
                pr = _true_eu_predict(Q, st, lam_cl=lam_cl, p0_scale=p0_sc)
                perm_fail, _, _ = _fast_boot654_pos_risk(
                    pr, gold, groups, n_boot=10000, seed=2026, without_replacement=True, margin_buffer=0.0
                )
                clb_fail, _, _ = _fast_boot654_pos_risk(
                    pr, gold, groups, n_boot=10000, seed=2026, without_replacement=False, margin_buffer=0.0
                )
                n_cp0 = int(((gold == 0).all(axis=1) & (pr == 3).any(axis=1)).sum())
                sc_all = float(c.score(pr, gold)["doc2_strict_total"])
                feasible = (
                    (perm_fail <= self.max_perm_fail_pct)
                    and (clb_fail <= self.max_clboot_fail_pct)
                    and (n_cp0 <= max_allowed_cp0)
                )
                penalty = -round(perm_fail + 0.5 * clb_fail + 5.0 * max(0, n_cp0 - max_allowed_cp0), 3) if not feasible else 0.0
                key = (1 if feasible else 0, penalty, sc_all)
                if best is None or key > best[0]:
                    best = (key, st, lam_cl, p0_sc)
        _, self.st, self.lam_cl, self.p0_scale = best
        return self

    def predict(self, Q: np.ndarray) -> np.ndarray:
        return _true_eu_predict(Q, self.st, lam_cl=self.lam_cl, p0_scale=self.p0_scale)


class TagFreeRoutedGateEU:
    """
    100% Batch-Tag-Free Routed Gate-Constrained EU Decoder:
      1) Per-stream scalar temperature + L2-shrunk per-defect simplex routing (`W[6, S]`) on all training rows.
      2) Uniform-weight Diagonal Dirichlet / Logit calibration (`DirDiag`, 20 params, `l2=0.02`) on all training rows.
      3) High-confidence `peg >= 0.90` unseen-source `花屏` rescue (if `peg` is in pool) + optional `pe_bkseg1` blend.
      4) `TagFreeGateConstrainedEU` fitted on all training rows uniformly.
      5) Strictly `N = 1` batch invariant at inference (`predict_from_member_dict([x]) == predict_from_member_dict(X)[0]`).
    """

    def __init__(
        self,
        base_members: Sequence[str],
        base_prior: Sequence[float],
        p12_members: Sequence[str],
        p12_internal_weights: Sequence[float],
        p12_total_prior: float = 0.18,
        l2_reg: float = 0.015,
        peg_rescue_thr: float = 0.90,
        use_dirdiag: bool = True,
        dirdiag_l2: float = 0.02,
        t_max: float = 3.80,
        lams_cl: Sequence[float] = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25),
        bkseg1_alpha: float = 0.0,
    ):
        self.base_members = list(base_members)
        bp = np.asarray(base_prior, dtype=np.float64)
        if len(bp) != len(self.base_members):
            bp = compute_family_base_prior(self.base_members)
        self.base_prior = bp / bp.sum()

        self.p12_members = list(p12_members)
        pw = np.asarray(p12_internal_weights, dtype=np.float64)
        self.p12_internal_weights = pw / pw.sum() if len(pw) > 0 else pw
        self.p12_total_prior = float(p12_total_prior) if len(self.p12_members) > 0 else 0.0
        self.l2_reg = float(l2_reg)
        self.peg_rescue_thr = float(peg_rescue_thr)
        self.use_dirdiag = bool(use_dirdiag)
        self.dirdiag_l2 = float(dirdiag_l2)
        self.t_max = float(t_max)
        self.lams_cl = tuple(float(x) for x in lams_cl)
        self.bkseg1_alpha = float(bkseg1_alpha)
        self.tag_free = True

        self.temps: np.ndarray = np.array([], dtype=np.float64)
        self.W: np.ndarray = np.array([], dtype=np.float64)
        self.dirdiag_params: Optional[np.ndarray] = None
        self.eu: TagFreeGateConstrainedEU = TagFreeGateConstrainedEU(lams_cl=self.lams_cl)

    @staticmethod
    def _fit_uniform_dirdiag(Q_tr: np.ndarray, Y_tr: np.ndarray, l2: float = 0.02) -> np.ndarray:
        params = np.zeros((6, 4), dtype=np.float64)
        Q_tr_clip = np.clip(Q_tr, 1e-6, 1.0 - 1e-6)
        idx_n = np.arange(len(Y_tr))
        for d in range(6):
            lp = np.log(Q_tr_clip[:, d, :])
            yd = Y_tr[:, d]
            if d in c.FULL_DIMS:
                def obj(th):
                    sc = np.exp(th[0])
                    b = np.array([0.0, th[1], th[2], th[3]])
                    z = lp * sc + b
                    z -= z.max(axis=1, keepdims=True)
                    sm = np.exp(z)
                    sm /= sm.sum(axis=1, keepdims=True)
                    nll = -np.mean(np.log(np.clip(sm[idx_n, yd], 1e-9, 1.0)))
                    return float(nll + l2 * np.sum(th ** 2))
                params[d] = minimize(obj, np.zeros(4), method="BFGS").x
            else:
                lp2 = np.log(Q_tr_clip[:, d, [0, 3]])
                yb = (yd == 3).astype(int)
                def obj2(th):
                    sc = np.exp(th[0])
                    b = np.array([0.0, th[1]])
                    z = lp2 * sc + b
                    z -= z.max(axis=1, keepdims=True)
                    sm = np.exp(z)
                    sm /= sm.sum(axis=1, keepdims=True)
                    nll = -np.mean(np.log(np.clip(sm[idx_n, yb], 1e-9, 1.0)))
                    return float(nll + l2 * np.sum(th ** 2))
                params[d, :2] = minimize(obj2, np.zeros(2), method="BFGS").x
        return params

    @staticmethod
    def _apply_dirdiag(Q_ev: np.ndarray, params: np.ndarray) -> np.ndarray:
        Qc = np.zeros_like(Q_ev)
        Qe = np.clip(Q_ev, 1e-6, 1.0 - 1e-6)
        for d in range(6):
            if d in c.FULL_DIMS:
                th = params[d]
                z = np.log(Qe[:, d, :]) * np.exp(th[0]) + np.array([0.0, th[1], th[2], th[3]])
                z -= z.max(axis=1, keepdims=True)
                sm = np.exp(z)
                Qc[:, d, :] = sm / sm.sum(axis=1, keepdims=True)
            else:
                th = params[d, :2]
                z = np.log(Qe[:, d, [0, 3]]) * np.exp(th[0]) + np.array([0.0, th[1]])
                z -= z.max(axis=1, keepdims=True)
                sm = np.exp(z)
                sm /= sm.sum(axis=1, keepdims=True)
                Qc[:, d, 0] = sm[:, 0]
                Qc[:, d, 3] = sm[:, 1]
        return c.renorm(Qc)

    def _build_streams(self, P_dict: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        streams = [c.renorm(P_dict[m]) for m in self.base_members]
        if len(self.p12_members) > 0:
            p12_comb = c.renorm(c.mean_pool([P_dict[m] for m in self.p12_members], list(self.p12_internal_weights)))
            streams.append(p12_comb)
            prior = np.append(self.base_prior * (1.0 - self.p12_total_prior), self.p12_total_prior)
        else:
            prior = self.base_prior.copy()
        prior /= prior.sum()
        return streams, prior

    def fit(self, P_dict: Dict[str, np.ndarray], gold: np.ndarray, groups: np.ndarray) -> "TagFreeRoutedGateEU":
        streams, prior = self._build_streams(P_dict)
        self.temps, self.W = fit_nested_routed_pool(streams, gold, prior, l2_reg=self.l2_reg, t_max=self.t_max)
        if self.use_dirdiag:
            q_raw = apply_nested_routed_pool(streams, self.temps, self.W)
            self.dirdiag_params = self._fit_uniform_dirdiag(q_raw, gold, l2=self.dirdiag_l2)
        else:
            self.dirdiag_params = None
        q = self.pool_from_member_dict(P_dict)
        self.eu = TagFreeGateConstrainedEU(lams_cl=self.lams_cl).fit(q, gold, groups=groups)
        return self

    def pool_from_member_dict(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        streams, _ = self._build_streams(P_dict)
        Q = apply_nested_routed_pool(streams, self.temps, self.W)
        if self.dirdiag_params is not None:
            Q = self._apply_dirdiag(Q, self.dirdiag_params)
        if "peg" in P_dict and self.peg_rescue_thr <= 1.0:
            pg = c.renorm(P_dict["peg"])[:, 3, 3]
            m_res = pg >= self.peg_rescue_thr
            if np.any(m_res):
                Q = Q.copy()
                q3 = np.maximum(Q[m_res, 3, 3], pg[m_res])
                Q[m_res, 3, :] = np.stack([1.0 - q3, np.full_like(q3, 1e-6), np.full_like(q3, 1e-6), q3], axis=1)
                Q[m_res] = c.renorm(Q[m_res])
        if "pe_bkseg1" in P_dict and self.bkseg1_alpha > 0.0:
            Q = Q.copy()
            pb = c.renorm(P_dict["pe_bkseg1"])[:, 5, :]
            Q[:, 5, :] = (1.0 - self.bkseg1_alpha) * Q[:, 5, :] + self.bkseg1_alpha * pb
            Q = c.renorm(Q)
        return Q

    def predict_from_member_dict(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return self.eu.predict(self.pool_from_member_dict(P_dict))


def build_cfg(members: Sequence[str]) -> dict:
    from final_engine import P12_SPEC

    base_m = [m for m in members if m not in P12_SPEC]
    p12_m = [m for m in members if m in P12_SPEC]
    cfg = dict(
        base_members=base_m,
        base_prior=list(compute_family_base_prior(base_m)),
        p12_members=p12_m,
        p12_internal_weights=([0.65, 0.35] if set(p12_m) == {"p12_pe", "p12_g14"} else [1.0 / max(1, len(p12_m))] * len(p12_m)),
        p12_total_prior=(0.20 if len(p12_m) >= 2 else 0.18),
        l2_reg=0.015,
    )
    if "pe_bkseg1" in members:
        cfg["bkseg1_alpha"] = 0.25
    return cfg


# ----------------------------------------------------------------------------------------------------------------------
# Data & Reporting Helpers
# ----------------------------------------------------------------------------------------------------------------------
def load_member_oof(members: Sequence[str]):
    from final_engine import oof_pool

    std_m = [m for m in members if m != "iv6b_ep2"]
    names, gold, fold, _, P = oof_pool(std_m)
    _, _, _, groups = c.load_split()
    if "iv6b_ep2" in members:
        iv, ok = load_candidate(IV6B_EP2_OOF, names)
        assert bool(ok.all()), f"Incomplete iv6b_ep2 OOF: {ok.sum()}/{len(names)}"
        P["iv6b_ep2"] = iv
    return list(names), gold, fold, groups, P


def load_candidate(path: str, names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Directory of fold_{k}_probs.npz (`va_names`, `va_probs_6x4`) or a single `.npz` (`names`, `probs_6x4`)."""
    idx = {str(n): i for i, n in enumerate(names)}
    arr = np.full((len(names), 6, 4), np.nan, dtype=np.float64)
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    files: List[str] = []
    if os.path.isdir(p):
        for k in range(8):
            for pat in (f"fold_{k}_probs.npz", f"oof_fold_{k}.npz"):
                fp = os.path.join(p, pat)
                if os.path.exists(fp):
                    files.append(fp)
                    break
    elif os.path.exists(p):
        files = [p]
    assert files, f"No candidate OOF files found at {p}"
    for f in files:
        z = np.load(f, allow_pickle=True)
        nk = "va_names" if "va_names" in z.files else "names"
        pk = "va_probs_6x4" if "va_probs_6x4" in z.files else "probs_6x4"
        for n, q in zip(z[nk], z[pk]):
            i = idx.get(str(n))
            if i is not None:
                arr[i] = q
    ok = ~np.isnan(arr[:, 0, 0])
    arr[~ok] = 0.25
    return c.renorm(arr), ok


def _eval_only_orig_mask(names: Sequence[str]) -> np.ndarray:
    """EVALUATION DISPLAY ONLY: never passed to any model or decoder `fit`."""
    mp = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    return np.array([mp[str(n)] == "orig" for n in names], dtype=bool)


def _f1(pb: np.ndarray, tb: np.ndarray) -> float:
    tp = int(np.sum(pb & tb))
    fp = int(np.sum(pb & ~tb))
    fn = int(np.sum(~pb & tb))
    return 1.0 if tp + fp + fn == 0 else 2.0 * tp / max(1, 2 * tp + fp + fn)


def grade_f1(pred_d: np.ndarray, y_d: np.ndarray) -> Dict[str, float]:
    p0, p1, p2 = _f1(pred_d == 3, y_d == 3), _f1(pred_d == 2, y_d == 2), _f1(pred_d == 1, y_d == 1)
    return {"P0": p0, "P1": p1, "P2": p2, "mean": float((p0 + p1 + p2) / 3.0)}


def summarize(pred: np.ndarray, gold: np.ndarray, orig_disp_mask: np.ndarray) -> dict:
    sa = c.score(pred, gold)
    so = c.score(pred[orig_disp_mask], gold[orig_disp_mask]) if orig_disp_mask.any() else sa
    return dict(
        doc2_all=float(sa["doc2_strict_total"]),
        doc2_orig=float(so["doc2_strict_total"]),
        S_pos_all=float(sa["doc2_S_pos"]),
        S_p0_all=float(sa["doc2_S_p0"]),
        Clean2_all=float(sa["doc2_Clean2"]),
        Clean2_orig=float(so["doc2_Clean2"]),
        n_clean_p0_flag=int(sa["n_clean_p0_flag"]),
        os_f1=grade_f1(pred[:, 2], gold[:, 2]),
        og_f1=grade_f1(pred[:, 4], gold[:, 4]),
    )


def _line(tag: str, s: dict) -> str:
    o, g = s["os_f1"], s["og_f1"]
    return (
        f"{tag:<50} doc2_all={s['doc2_all']:.3f} doc2_orig={s['doc2_orig']:.3f} Clean2={s['Clean2_all']:.2f} "
        f"| OS F1 {o['P0']:.3f}/{o['P1']:.3f}/{o['P2']:.3f} | OG F1 {g['P0']:.3f}/{g['P1']:.3f}/{g['P2']:.3f}"
    )


def severity_metrics(q6: np.ndarray, gold: np.ndarray, ok: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

    out = {}
    for d, nm in ((2, "OS"), (4, "OG")):
        y, q = gold[:, d], q6[:, d, :]
        sc = (q[:, 1] + 2 * q[:, 2] + 3 * q[:, 3]) / np.maximum(1e-6, 1 - q[:, 0])
        m21, m32 = ok & np.isin(y, [1, 2]), ok & np.isin(y, [2, 3])
        out[nm] = dict(
            AP_any=float(AP(y[ok] >= 1, 1 - q[ok, 0])),
            AP_ge1=float(AP(y[ok] >= 2, q[ok, 2] + q[ok, 3])),
            AP_p0=float(AP(y[ok] == 3, q[ok, 3])),
            P2vP1=float(AUC(y[m21] == 2, sc[m21])),
            P1vP0=float(AUC(y[m32] == 3, sc[m32])),
        )
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Nested OOF & N=1 Verification
# ----------------------------------------------------------------------------------------------------------------------
def _subfold_ids_within_covered(fold: np.ndarray, groups: np.ndarray, cov_folds: Sequence[int]) -> np.ndarray:
    """For 2-fold pilots (872 rows), splits each covered fold's video groups into 2 deterministic halves -> 4 balanced
    video-disjoint folds (~654 train rows / ~218 val rows per split), matching Perm654's 654-image calibration scale."""
    sub = np.full(len(fold), -1, dtype=np.int64)
    sid = 0
    for fk in cov_folds:
        m = fold == fk
        ug = sorted(set(str(g) for g in groups[m]))
        g2h = {g: (i % 2) for i, g in enumerate(ug)}
        for idx_i in np.where(m)[0]:
            sub[idx_i] = sid + g2h[str(groups[idx_i])]
        sid += 2
    return sub


def nested_oof(
    P: Dict[str, np.ndarray],
    members: Sequence[str],
    gold: np.ndarray,
    fold: np.ndarray,
    groups: np.ndarray,
    eval_folds: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = build_cfg(members)
    N = len(gold)
    Q = np.zeros((N, 6, 4), dtype=np.float64)
    pred = np.zeros((N, 6), dtype=np.int64)
    if eval_folds is None or len(eval_folds) == 8:
        pool_rows = np.ones(N, dtype=bool)
        for k in range(8):
            va, tr = fold == k, fold != k
            dec = TagFreeRoutedGateEU(**cfg).fit({m: P[m][tr] for m in members}, gold[tr], groups[tr])
            Q[va] = dec.pool_from_member_dict({m: P[m][va] for m in members})
            pred[va] = dec.eu.predict(Q[va])
        return Q, pred, pool_rows

    pool_rows = np.isin(fold, list(eval_folds))
    sub = _subfold_ids_within_covered(fold, groups, eval_folds)
    for sk in sorted(k for k in np.unique(sub) if k >= 0):
        va = sub == sk
        tr = pool_rows & (sub != sk)
        dec = TagFreeRoutedGateEU(**cfg).fit({m: P[m][tr] for m in members}, gold[tr], groups[tr])
        Q[va] = dec.pool_from_member_dict({m: P[m][va] for m in members})
        pred[va] = dec.eu.predict(Q[va])
    return Q, pred, pool_rows


def n1_check(dec: TagFreeRoutedGateEU, P: Dict[str, np.ndarray], members: Sequence[str], rows: np.ndarray) -> int:
    sel = np.where(rows)[0][:32]
    Pb = {m: P[m][sel] for m in members}
    pb = dec.predict_from_member_dict(Pb)
    qb = dec.pool_from_member_dict(Pb)
    diff = 0
    for j, i in enumerate(sel):
        p1_in = {m: P[m][i : i + 1] for m in members}
        p1 = dec.predict_from_member_dict(p1_in)
        q1 = dec.pool_from_member_dict(p1_in)
        diff += int(np.sum(p1[0] != pb[j]))
        assert float(np.max(np.abs(q1[0] - qb[j]))) < 1e-12, f"N=1 probability drift at row {i}"
    return diff


# ----------------------------------------------------------------------------------------------------------------------
# Gate #1: Severity Complementarity with Video-Cluster Bootstrap CI90
# ----------------------------------------------------------------------------------------------------------------------
def _cv_scores(X: np.ndarray, yb: np.ndarray, fm: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    s = np.zeros(len(yb), dtype=np.float64)
    for k in np.unique(fm):
        tr, va = fm != k, fm == k
        if len(np.unique(yb[tr])) < 2:
            s[va] = float(yb[tr].mean())
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[tr], yb[tr])
        s[va] = clf.predict_proba(X[va])[:, 1]
    return s


def gate1(
    Q_ref: np.ndarray,
    cand: np.ndarray,
    ok: np.ndarray,
    gold: np.ndarray,
    fold: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

    L = lambda a: np.log(np.clip(a, 1e-6, 1.0))  # noqa: E731
    rng = np.random.default_rng(seed)
    cov_folds = sorted(int(k) for k in np.unique(fold[ok]))
    cv_f = fold if len(cov_folds) >= 4 else _subfold_ids_within_covered(fold, groups, cov_folds)
    out = {}
    for d, nm in ((2, "OS"), (4, "OG")):
        y = gold[:, d]
        res = {}
        for lab, sel, fn, met in (
            ("dAP_p0", ok, lambda yy: yy == 3, AP),
            ("dP1vP0", ok & np.isin(y, [2, 3]), lambda yy: yy == 3, AUC),
            ("dP2vP1", ok & np.isin(y, [1, 2]), lambda yy: yy == 2, AUC),
        ):
            yb = fn(y[sel]).astype(int)
            fm = cv_f[sel]
            gm = groups[sel]
            s0 = _cv_scores(L(Q_ref[sel, d]), yb, fm)
            s1 = _cv_scores(np.column_stack([L(Q_ref[sel, d]), L(cand[sel, d])]), yb, fm)
            delta = float(met(yb, s1) - met(yb, s0))
            ug = np.unique(gm)
            gi = {g: np.where(gm == g)[0] for g in ug}
            bs = []
            for _ in range(n_boot):
                ii = np.concatenate([gi[g] for g in rng.choice(ug, len(ug))])
                if yb[ii].min() == yb[ii].max():
                    continue
                bs.append(float(met(yb[ii], s1[ii]) - met(yb[ii], s0[ii])))
            lo, hi = np.percentile(bs, [5, 95])
            res[lab] = dict(delta=delta, ci90=[float(lo), float(hi)])
        out[nm] = res
    return out


# ----------------------------------------------------------------------------------------------------------------------
# CLI Commands
# ----------------------------------------------------------------------------------------------------------------------
def _resolve_set_members(set_name: str) -> List[str]:
    from final_engine import SETS

    if set_name == "rule3_prev":
        return ["iv6b_ep2"] + RULE3_FIXED_SPECS + ["p1_qy"]
    return list(SETS[set_name])


def cmd_baseline(set_name: str) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    members = _resolve_set_members(set_name)
    t0 = time.time()
    names, gold, fold, groups, P = load_member_oof(members)
    orig_disp = _eval_only_orig_mask(names)
    Q, pred, _ = nested_oof(P, members, gold, fold, groups)
    s = summarize(pred, gold, orig_disp)
    print(_line(f"TAG-FREE nested 8-fold [{set_name}]", s), flush=True)

    dec_full = TagFreeRoutedGateEU(**build_cfg(members)).fit(P, gold, groups)
    n1 = n1_check(dec_full, P, members, np.ones(len(gold), dtype=bool))
    pkl_path = os.path.join(OUT_DIR, f"{set_name}_tagfree_decoder.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(dict(decoder=dec_full, members=list(members), tag_free=True), f)
    with open(pkl_path, "rb") as f_chk:
        reloaded = pickle.load(f_chk)["decoder"]
        assert isinstance(reloaded, TagFreeRoutedGateEU) and type(reloaded).__module__ == "tagfree_rule3_decoder"

    np.savez_compressed(
        os.path.join(OUT_DIR, f"{set_name}_tagfree_oof.npz"),
        names=np.array(names),
        Q_oof=Q,
        pred_oof=pred,
        fold=fold,
    )
    rep = dict(
        set=set_name,
        members=list(members),
        nested_8fold=s,
        solo_pool_metrics=severity_metrics(Q, gold, np.ones(len(gold), dtype=bool)),
        n1_cell_diffs=n1,
        fullfit_lam_cl=float(dec_full.eu.lam_cl),
        fullfit_p0_scale=float(dec_full.eu.p0_scale),
        elapsed_s=round(time.time() - t0, 1),
        compliance="100% tag-free (uniform weights=1.0 across all 3,573 rows; doc2_orig is evaluation display only)",
    )
    with open(os.path.join(OUT_DIR, f"{set_name}_tagfree_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"  saved {pkl_path} (N=1 cell diffs={n1}, {rep['elapsed_s']}s)", flush=True)
    return rep


def cmd_gate(
    cand_path: str,
    tag: str,
    slot: str,
    ref_set: str,
    cand2_path: str = "",
    tag2: str = "",
    slot2: str = "",
) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    ref_members = _resolve_set_members(ref_set)
    all_m = list(dict.fromkeys(list(ref_members) + ["iv6b_ep2", "p1_qy"] + RULE3_FIXED_SPECS))
    names, gold, fold, groups, P = load_member_oof(all_m)
    orig_disp = _eval_only_orig_mask(names)

    cand, ok = load_candidate(cand_path, names)
    if cand2_path:
        assert slot2 and slot2 != slot, "--slot2 must differ from --slot when --cand2 is set"
        cand2, ok2 = load_candidate(cand2_path, names)
        ok = ok & ok2
    else:
        cand2 = None

    cov_folds = sorted(int(k) for k in np.unique(fold[ok]))
    assert np.all(ok == np.isin(fold, cov_folds)), "Candidate OOF must cover complete canonical folds"
    label = f"{tag}+{tag2}" if cand2 is not None else tag
    print(f"[gate] candidate={label} covered_folds={cov_folds} n_rows={int(ok.sum())}", flush=True)

    ref_npz = os.path.join(OUT_DIR, f"{ref_set}_tagfree_oof.npz")
    if not os.path.exists(ref_npz):
        cmd_baseline(ref_set)
    Q_ref = np.load(ref_npz, allow_pickle=True)["Q_oof"]

    report: Dict[str, object] = dict(tag=label, covered_folds=cov_folds, n_rows=int(ok.sum()))
    report["solo"] = {tag: severity_metrics(cand, gold, ok)}
    if cand2 is not None:
        report["solo"][tag2] = severity_metrics(cand2, gold, ok)
    report["ref_pool_same_rows"] = severity_metrics(Q_ref, gold, ok)

    g1_map = {tag: gate1(Q_ref, cand, ok, gold, fold, groups)}
    if cand2 is not None:
        g1_map[tag2] = gate1(Q_ref, cand2, ok, gold, fold, groups)
    for ntag, npath in NULL_TOWERS.items():
        try:
            nq, nok = load_candidate(npath, names)
        except AssertionError:
            continue
        if np.all(nok[ok]):
            g1_map[ntag] = gate1(Q_ref, nq, ok, gold, fold, groups)
    report["gate1"] = g1_map

    for c_nm, sm in report["solo"].items():
        for nm in ("OS", "OG"):
            so, rp = sm[nm], report["ref_pool_same_rows"][nm]
            print(
                f"  solo [{c_nm}] {nm}: AP_any={so['AP_any']:.3f} AP_p0={so['AP_p0']:.3f} "
                f"P2vP1={so['P2vP1']:.3f} P1vP0={so['P1vP0']:.3f}  "
                f"(ref pool: AP_p0={rp['AP_p0']:.3f} P2vP1={rp['P2vP1']:.3f} P1vP0={rp['P1vP0']:.3f})",
                flush=True,
            )
    for t, g in g1_map.items():
        s = " | ".join(
            f"{nm} " + " ".join(f"{k}={v['delta']:+.3f}[{v['ci90'][0]:+.3f},{v['ci90'][1]:+.3f}]" for k, v in g[nm].items())
            for nm in ("OS", "OG")
        )
        print(f"  gate1 {t:<24} {s}", flush=True)

    P2 = dict(P)
    P2[f"cand_{slot}"] = cand
    if cand2 is not None:
        P2[f"cand_{slot2}"] = cand2

    base_6b_mem = "cand_base_6b" if "cand_base_6b" in P2 else "iv6b_ep2"
    spec_c_mem = "cand_spec_c" if "cand_spec_c" in P2 else "p1_qy"
    rule3_cand_members = [base_6b_mem] + RULE3_FIXED_SPECS + [spec_c_mem]

    stacks = {
        f"ref_{ref_set}": list(ref_members),
        "4specs(ntg_convnext,p12_pe,p1_qy,peg)": RULE3_FIXED_SPECS + ["p1_qy"],
        "RULE3_prev(iv6b_ep2+4specs)": ["iv6b_ep2"] + RULE3_FIXED_SPECS + ["p1_qy"],
        f"RULE3_cand({label})": rule3_cand_members,
    }
    eval_folds = None if len(cov_folds) == 8 else cov_folds
    report["stacks"] = {}
    for st_name, mem in stacks.items():
        _, pred, rows = nested_oof(P2, mem, gold, fold, groups, eval_folds=eval_folds)
        s = summarize(pred[rows], gold[rows], orig_disp[rows])
        report["stacks"][st_name] = s
        print(_line(st_name, s), flush=True)

    dec_r3 = TagFreeRoutedGateEU(**build_cfg(rule3_cand_members)).fit(
        {m: P2[m][ok] for m in rule3_cand_members}, gold[ok], groups[ok]
    )
    n1 = n1_check(dec_r3, {m: P2[m][ok] for m in rule3_cand_members}, rule3_cand_members, np.ones(int(ok.sum()), dtype=bool))
    report["n1_cell_diffs_rule3"] = n1
    report["rule3_members"] = rule3_cand_members
    print(f"  N=1 vs batch cell diffs (Rule-3 5-member decoder): {n1}", flush=True)

    out_json = os.path.join(OUT_DIR, f"gate_{label}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  wrote {out_json}", flush=True)
    return report


def cmd_self_test() -> None:
    import inspect

    rng = np.random.default_rng(7)
    N = 400
    members = ["pe_v3", "ntg_convnext", "p12_pe", "p1_qy", "peg"]
    gold = rng.choice(4, size=(N, 6), p=[0.70, 0.14, 0.09, 0.07])
    gold[:, 3] = np.where(gold[:, 3] > 0, 3, 0)
    gold[:, 5] = np.where(gold[:, 5] > 0, 3, 0)
    P = {m: c.renorm(0.62 * np.eye(4)[gold] + 0.38 * rng.dirichlet([1, 1, 1, 1], size=(N, 6))) for m in members}
    groups = np.array([f"g{i // 2}" for i in range(N)])

    dec = TagFreeRoutedGateEU(**build_cfg(members)).fit(P, gold, groups)
    assert dec.tag_free is True
    assert "is_orig" not in inspect.signature(TagFreeRoutedGateEU.fit).parameters
    assert "is_orig" not in inspect.signature(TagFreeGateConstrainedEU.fit).parameters
    assert type(dec).__module__ == "tagfree_rule3_decoder", f"Bad module: {type(dec).__module__}"
    blob = pickle.dumps(dec)
    dec2 = pickle.loads(blob)
    assert n1_check(dec2, P, members, np.ones(N, dtype=bool)) == 0
    print("[tagfree_rule3_decoder][self-test] PASS (100% tag-free, canonical pickle, N=1==N=batch)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--set", default="resident_v7_core", help="resident_v7_core | resident_v7 | rule3_prev")
    g = sub.add_parser("gate")
    g.add_argument("--cand", required=True)
    g.add_argument("--tag", required=True)
    g.add_argument("--slot", choices=["base_6b", "spec_c"], required=True)
    g.add_argument("--ref-set", default="resident_v7_core")
    g.add_argument("--cand2", default="")
    g.add_argument("--tag2", default="")
    g.add_argument("--slot2", choices=["", "base_6b", "spec_c"], default="")
    sub.add_parser("self-test")
    a = ap.parse_args()
    if a.cmd == "baseline":
        cmd_baseline(a.set)
    elif a.cmd == "gate":
        cmd_gate(a.cand, a.tag, a.slot, a.ref_set, a.cand2, a.tag2, a.slot2)
    else:
        cmd_self_test()


if __name__ == "__main__":
    import tagfree_rule3_decoder as _canonical_mod
    _canonical_mod.main()
