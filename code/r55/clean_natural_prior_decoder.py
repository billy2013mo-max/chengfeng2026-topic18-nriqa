#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/clean_natural_prior_decoder.py
=======================================
100% Package-Free (`ZERO` `orig`/`supp`/`split` concept anywhere!) Natural-Prior Calibrated +
Cross-Defect Deconfounded Expected-Utility Decoder (`NaturalPriorRoutedEU`).

Why this completely discards the "package/batch" concept while matching real evaluation priors:
  - In production/competition NR-IQA datasets, hard/defective frames are routinely oversampled (case-control
    sampling: our 3,573-frame pool has only 14.4% all-clean frames, whereas natural evaluation streams have ~25%
    all-clean frames).
  - Instead of referencing ANY dataset batch/package tag (`orig` vs `supp`), `compute_case_control_weights(gold, pi_clean)`
    assigns standard statistical case-control importance weights purely from the ground-truth label `y == 0` vs `y != 0`:
        w_i = (pi_clean / f_clean)        if all(y_i == 0)
        w_i = ((1 - pi_clean) / f_defect) otherwise
  - Optionally applies 100% package-free Zero-Intercept Rank-1 Cross-Defect Deconfounding (`LowRankDeconfounder`)
    on `过锐` (dim 2) and `过度生成` (dim 4), validated strictly via inner group-CV on weighted RPS/Doc2.
  - Strictly `N = 1 == N = Batch` invariant (`0` cell diff) at inference.
"""
from __future__ import annotations

import argparse
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
import routed_eu_decoder as red  # noqa: E402
import tagfree_rule3_decoder as tfd  # noqa: E402
from deconfound_calibrator import extract_deconf_covariates  # noqa: E402
from eval_specialist_gate import apply_nested_routed_pool, fit_nested_routed_pool  # noqa: E402
from routed_eu_decoder import _true_eu_predict, compute_family_base_prior  # noqa: E402

red.MEMBER_FAMILY_MAP["qwen3vl_hires"] = "ph_vlm"
red.MEMBER_FAMILY_MAP["iv6b_native"] = "iv6b"

ANCHOR_SHIFT = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)


def compute_case_control_weights(gold: np.ndarray, pi_clean: float = 0.25) -> np.ndarray:
    """
    Purely label-driven (`y == 0` vs `y != 0`) importance weights restoring natural Clean prevalence `pi_clean`.
    100% invariant to row order and 100% free of any `orig`/`supp` batch metadata.
    """
    is_cl = (gold == 0).all(axis=1)
    n_tot = float(len(gold))
    n_cl = float(is_cl.sum())
    n_def = n_tot - n_cl
    if n_cl <= 0 or n_def <= 0:
        return np.ones(len(gold), dtype=np.float64)
    f_cl = n_cl / n_tot
    f_def = n_def / n_tot
    w = np.where(is_cl, float(pi_clean) / f_cl, (1.0 - float(pi_clean)) / f_def)
    return w * (n_tot / np.sum(w))


def weighted_doc2_strict(pred: np.ndarray, gold: np.ndarray, row_w: np.ndarray) -> Dict[str, float]:
    """Importance-weighted `Doc2_Strict` (`25*S_pos + 20*S_p0 + 20*Clean2`) using only `(pred, gold, row_w)`."""
    w = row_w[:, None]
    pl = (pred[:, E.LD] == E.LG[None]).astype(np.float64)
    gl = (gold[:, E.LD] == E.LG[None]).astype(np.float64)
    tp = (w * pl * gl).sum(0)
    npred = (w * pl).sum(0)
    npos = (w * gl).sum(0)

    f1_14 = np.where(npred + npos > 1e-8, 2.0 * tp / np.maximum(1e-8, npred + npos), 1.0)
    s_pos = 25.0 * float(np.mean(f1_14))

    # P0 channels in E.LABELS: indices where E.LG == 3
    p0_idx = np.where(E.LG == 3)[0]
    tp0, np0_pred, np0_pos = tp[p0_idx], npred[p0_idx], npos[p0_idx]
    r_p0 = float(np.mean(tp0 / np.maximum(1e-8, np0_pos)))
    p_p0 = float(np.mean(np.where(np0_pred > 1e-8, tp0 / np.maximum(1e-8, np0_pred), 1.0)))
    f_p0 = float(np.mean(np.where(np0_pred + np0_pos > 1e-8, 2.0 * tp0 / np.maximum(1e-8, np0_pred + np0_pos), 1.0)))
    s_p0 = 20.0 * (0.5 * r_p0 + 0.3 * f_p0 + 0.2 * p_p0)

    is_cl = (gold == 0).all(axis=1)
    w_cl = row_w[is_cl]
    if w_cl.sum() > 1e-8:
        strict_zero_rate = float(np.sum(w_cl * (pred[is_cl] == 0).all(axis=1)) / np.sum(w_cl))
        cp0_rate = float(np.sum(w_cl * (pred[is_cl] == 3).any(axis=1)) / np.sum(w_cl))
        cliff = max(0.0, 1.0 - cp0_rate / 0.02)
        clean2 = 10.0 * strict_zero_rate + 10.0 * cliff
    else:
        clean2 = 20.0

    return {
        "total": s_pos + s_p0 + clean2,
        "S_pos": s_pos,
        "S_p0": s_p0,
        "Clean2": clean2,
    }


def weighted_cluster_pos_fail_pct(
    pred: np.ndarray,
    gold: np.ndarray,
    row_w: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 4000,
    seed: int = 2026,
) -> float:
    """Video-cluster bootstrap probability (%) that weighted `25 * S_pos < 15.0`."""
    rng = np.random.default_rng(seed)
    w = row_w[:, None]
    pl = (pred[:, E.LD] == E.LG[None]).astype(np.float64)
    gl = (gold[:, E.LD] == E.LG[None]).astype(np.float64)
    tp_row = w * pl * gl
    denom_row = w * (pl + gl)

    ug = np.unique(groups)
    n_g = len(ug)
    g_tp = np.zeros((n_g, len(E.LD)), dtype=np.float64)
    g_den = np.zeros((n_g, len(E.LD)), dtype=np.float64)
    g2i = {g: idx for idx, g in enumerate(ug)}
    for r_i, g in enumerate(groups):
        gi = g2i[g]
        g_tp[gi] += tp_row[r_i]
        g_den[gi] += denom_row[r_i]

    # Sample ~500 video clusters per draw (matching ~644-image evaluation set size)
    k_draw = min(n_g, 520)
    picks = rng.integers(0, n_g, size=(n_boot, k_draw))
    b_tp = g_tp[picks].sum(axis=1)
    b_den = g_den[picks].sum(axis=1)
    b_f1 = np.where(b_den > 1e-8, 2.0 * b_tp / np.maximum(1e-8, b_den), 1.0)
    b_spos = 25.0 * b_f1.mean(axis=1)
    return float(100.0 * np.mean(b_spos < 15.0))


class PackageFreeDeconfounder:
    """
    100% Package-Free Cross-Defect Rank-1 Ordinal Shift for `dim in (2, 4)` (`过锐`, `过度生成`).
    Suppresses multi-defect crosstalk (e.g., severe blur/texloss falsely inflating `过度生成`) using
    8-D row-wise cross-defect covariates (`N = 1` invariant). Activated ONLY if inner 4-fold group-CV
    strictly improves weighted RPS on that dimension.
    """

    def __init__(self, l2_grid: Sequence[float] = (50.0, 150.0, 400.0)):
        self.l2_grid = tuple(float(x) for x in l2_grid)
        self.active: Dict[int, bool] = {2: False, 4: False}
        self.mu: Dict[int, np.ndarray] = {}
        self.sd: Dict[int, np.ndarray] = {}
        self.w_coef: Dict[int, np.ndarray] = {}

    @staticmethod
    def _apply_shift(q_d: np.ndarray, shift: np.ndarray) -> np.ndarray:
        lp = np.log(np.clip(q_d, 1e-7, 1.0)) + shift[:, None] * ANCHOR_SHIFT[None, :]
        lp -= lp.max(axis=1, keepdims=True)
        ez = np.exp(np.clip(lp, -35.0, 35.0))
        return ez / ez.sum(axis=1, keepdims=True)

    @classmethod
    def _fit_w(cls, q_d: np.ndarray, X_std: np.ndarray, y_d: np.ndarray, row_w: np.ndarray, l2: float) -> np.ndarray:
        cdf_y = (np.arange(4)[None, :] >= y_d[:, None]).astype(np.float64)
        # Note: CDF_pred(k) = sum_{c <= k} p_c, CDF_gold(k) = I(y <= k)
        cdf_gold = (np.arange(3)[None, :] >= y_d[:, None]).astype(np.float64)
        w_norm = row_w / np.sum(row_w)

        def obj(w_vec: np.ndarray) -> float:
            sh = np.clip(X_std @ w_vec, -1.2, 1.2)
            p_adj = cls._apply_shift(q_d, sh)
            cdf_p = np.cumsum(p_adj, axis=1)[:, :3]
            rps = float(np.sum(w_norm[:, None] * (cdf_p - cdf_gold) ** 2))
            return rps + (float(l2) / max(1.0, len(y_d))) * float(np.sum(w_vec ** 2))

        return minimize(obj, np.zeros(X_std.shape[1]), method="BFGS").x

    def fit(
        self,
        Q_base: np.ndarray,
        P_dict: Dict[str, np.ndarray],
        members: Sequence[str],
        gold: np.ndarray,
        row_w: np.ndarray,
        groups: np.ndarray,
    ) -> "PackageFreeDeconfounder":
        ug = sorted(set(str(g) for g in groups))
        g2f = {g: (i % 4) for i, g in enumerate(ug)}
        sub_f = np.array([g2f[str(g)] for g in groups], dtype=np.int64)
        w_norm = row_w / np.sum(row_w)

        for d in (2, 4):
            X_raw = extract_deconf_covariates(Q_base, P_dict, d, members)
            y_d = gold[:, d]
            cdf_gold = (np.arange(3)[None, :] >= y_d[:, None]).astype(np.float64)
            base_rps = float(np.sum(w_norm[:, None] * (np.cumsum(Q_base[:, d, :], axis=1)[:, :3] - cdf_gold) ** 2))

            best_l2 = None
            best_cv_rps = base_rps
            for l2 in self.l2_grid:
                q_oof = np.zeros_like(Q_base[:, d, :])
                for kf in range(4):
                    tr, va = (sub_f != kf), (sub_f == kf)
                    mu = X_raw[tr].mean(axis=0)
                    sd = X_raw[tr].std(axis=0) + 1e-6
                    wv = self._fit_w(Q_base[tr, d, :], np.clip((X_raw[tr] - mu) / sd, -5.0, 5.0), y_d[tr], row_w[tr], l2)
                    sh_va = np.clip(np.clip((X_raw[va] - mu) / sd, -5.0, 5.0) @ wv, -1.2, 1.2)
                    q_oof[va] = self._apply_shift(Q_base[va, d, :], sh_va)
                cv_rps = float(np.sum(w_norm[:, None] * (np.cumsum(q_oof, axis=1)[:, :3] - cdf_gold) ** 2))
                if cv_rps < best_cv_rps - 1e-5:
                    best_cv_rps = cv_rps
                    best_l2 = l2

            if best_l2 is not None:
                mu = X_raw.mean(axis=0)
                sd = X_raw.std(axis=0) + 1e-6
                wv = self._fit_w(Q_base[:, d, :], np.clip((X_raw - mu) / sd, -5.0, 5.0), y_d, row_w, best_l2)
                self.active[d] = True
                self.mu[d] = mu
                self.sd[d] = sd
                self.w_coef[d] = wv
            else:
                self.active[d] = False
        return self

    def transform(self, Q_base: np.ndarray, P_dict: Dict[str, np.ndarray], members: Sequence[str]) -> np.ndarray:
        Q_out = Q_base.copy()
        for d in (2, 4):
            if not self.active.get(d, False):
                continue
            X_raw = extract_deconf_covariates(Q_base, P_dict, d, members)
            X_std = np.clip((X_raw - self.mu[d]) / self.sd[d], -5.0, 5.0)
            sh = np.clip(X_std @ self.w_coef[d], -1.2, 1.2)
            Q_out[:, d, :] = self._apply_shift(Q_base[:, d, :], sh)
        return c.renorm(Q_out)


class NaturalPriorGateEU:
    """
    100% Package-Free Expected-Utility Decoder calibrated to natural Clean prevalence `pi_clean = 0.25`.
    """

    def __init__(
        self,
        pi_clean: float = 0.25,
        lams_cl: Sequence[float] = (2.00, 2.25, 2.50, 2.75, 3.00, 3.25),
        p0_scales: Sequence[float] = (0.85, 0.92, 1.00),
        max_clboot_fail_pct: float = 2.50,
    ):
        self.pi_clean = float(pi_clean)
        self.lams_cl = tuple(float(x) for x in lams_cl)
        self.p0_scales = tuple(float(x) for x in p0_scales)
        self.max_clboot_fail_pct = float(max_clboot_fail_pct)
        self.lam_cl: float = 2.75
        self.p0_scale: float = 1.00
        self.st: dict = {}

    @staticmethod
    def _weighted_stats(pred: np.ndarray, gold: np.ndarray, row_w: np.ndarray) -> dict:
        w = row_w[:, None]
        pl = (pred[:, E.LD] == E.LG[None]).astype(np.float64)
        gl = (gold[:, E.LD] == E.LG[None]).astype(np.float64)
        tp = (w * pl * gl).sum(0)
        npred = (w * pl).sum(0)
        npos = (w * gl).sum(0)
        ncl = float((row_w * (gold == 0).all(1).astype(np.float64)).sum())
        return dict(
            F=2.0 * tp / np.maximum(1e-6, npred + npos),
            P=tp / np.maximum(1e-6, npred),
            npos=npos,
            npred=npred,
            ncl=max(1e-6, ncl),
            N=float(row_w.sum()),
        )

    @classmethod
    def _fit_stats(cls, Q: np.ndarray, gold: np.ndarray, row_w: np.ndarray, lam_cl: float, p0_scale: float) -> dict:
        pred0 = E.Decoder55().predict(Q)
        st = cls._weighted_stats(pred0, gold, row_w)
        best_res = float("inf")
        best_st = st
        for _ in range(30):
            pr_now = _true_eu_predict(Q, st, lam_cl=lam_cl, p0_scale=p0_scale)
            st_ind = cls._weighted_stats(pr_now, gold, row_w)
            res = float(np.max(np.abs(st_ind["F"] - st["F"])))
            if res < best_res:
                best_res = res
                best_st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
            if res < 1e-6:
                return st_ind
            st = dict(
                F=0.5 * st["F"] + 0.5 * st_ind["F"],
                P=0.5 * st["P"] + 0.5 * st_ind["P"],
                npos=st_ind["npos"],
                npred=0.5 * st["npred"] + 0.5 * st_ind["npred"],
                ncl=st_ind["ncl"],
                N=st_ind["N"],
            )
        return best_st

    def fit(self, Q: np.ndarray, gold: np.ndarray, groups: Optional[np.ndarray] = None) -> "NaturalPriorGateEU":
        Q = c.renorm(Q)
        if groups is None:
            groups = np.arange(len(gold))
        row_w = compute_case_control_weights(gold, pi_clean=self.pi_clean)

        base_d55 = E.Decoder55().predict(Q)
        base_cp0 = int(((gold == 0).all(axis=1) & (base_d55 == 3).any(axis=1)).sum())
        max_allowed_cp0 = max(1, base_cp0)

        best = None
        for lam_cl in self.lams_cl:
            for p0_sc in self.p0_scales:
                st = self._fit_stats(Q, gold, row_w, lam_cl, p0_sc)
                pr = _true_eu_predict(Q, st, lam_cl=lam_cl, p0_scale=p0_sc)
                clb_fail = weighted_cluster_pos_fail_pct(pr, gold, row_w, groups, n_boot=4000, seed=2026)
                n_cp0 = int(((gold == 0).all(axis=1) & (pr == 3).any(axis=1)).sum())
                sc_w = weighted_doc2_strict(pr, gold, row_w)["total"]
                feasible = (clb_fail <= self.max_clboot_fail_pct) and (n_cp0 <= max_allowed_cp0)
                pen = -round(clb_fail + 5.0 * max(0, n_cp0 - max_allowed_cp0), 3) if not feasible else 0.0
                key = (1 if feasible else 0, pen, sc_w)
                if best is None or key > best[0]:
                    best = (key, st, lam_cl, p0_sc)
        _, self.st, self.lam_cl, self.p0_scale = best
        return self

    def predict(self, Q: np.ndarray) -> np.ndarray:
        return _true_eu_predict(Q, self.st, lam_cl=self.lam_cl, p0_scale=self.p0_scale)


class NaturalPriorRoutedDecoder:
    """
    Complete 100% Package-Free (`ZERO` `orig`/`supp` concept!) Decoder:
      1) Simplex Routing `W[6, S]` + Temperatures `temps[S]`
      2) Natural-Prior Weighted `DirDiag` Calibration (`pi_clean = 0.25`)
      3) `peg >= 0.90` Glitch Rescue + optional `pe_bkseg1` Bokeh Blend
      4) Inner-CV Gated `PackageFreeDeconfounder` (`过锐` & `过度生成` cross-defect crosstalk suppression)
      5) `NaturalPriorGateEU` (`pi_clean = 0.25`)
    """

    def __init__(
        self,
        members: Sequence[str],
        pi_clean: float = 0.25,
        use_deconf: bool = True,
        l2_reg: float = 0.015,
        dirdiag_l2: float = 0.02,
        peg_rescue_thr: float = 0.90,
    ):
        from final_engine import P12_SPEC

        self.members = list(members)
        self.base_members = [m for m in self.members if m not in P12_SPEC]
        self.p12_members = [m for m in self.members if m in P12_SPEC]
        bp = compute_family_base_prior(self.base_members)
        self.base_prior = bp / bp.sum()
        self.p12_internal_weights = (
            np.array([0.65, 0.35])
            if set(self.p12_members) == {"p12_pe", "p12_g14"}
            else np.ones(len(self.p12_members)) / max(1, len(self.p12_members))
        )
        self.p12_total_prior = (0.20 if len(self.p12_members) >= 2 else 0.18) if self.p12_members else 0.0
        self.bkseg1_alpha = 0.25 if "pe_bkseg1" in self.members else 0.0
        self.pi_clean = float(pi_clean)
        self.use_deconf = bool(use_deconf)
        self.l2_reg = float(l2_reg)
        self.dirdiag_l2 = float(dirdiag_l2)
        self.peg_rescue_thr = float(peg_rescue_thr)

        self.temps: np.ndarray = np.array([], dtype=np.float64)
        self.W: np.ndarray = np.array([], dtype=np.float64)
        self.dirdiag_params: Optional[np.ndarray] = None
        self.deconf: Optional[PackageFreeDeconfounder] = None
        self.eu: NaturalPriorGateEU = NaturalPriorGateEU(pi_clean=self.pi_clean)

    def _build_streams(self, P_dict: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        streams = [c.renorm(P_dict[m]) for m in self.base_members]
        if self.p12_members:
            p12_c = c.renorm(c.mean_pool([P_dict[m] for m in self.p12_members], list(self.p12_internal_weights)))
            streams.append(p12_c)
            prior = np.append(self.base_prior * (1.0 - self.p12_total_prior), self.p12_total_prior)
        else:
            prior = self.base_prior.copy()
        return streams, prior / prior.sum()

    @staticmethod
    def _fit_weighted_dirdiag(Q_tr: np.ndarray, Y_tr: np.ndarray, row_w: np.ndarray, l2: float = 0.02) -> np.ndarray:
        params = np.zeros((6, 4), dtype=np.float64)
        Q_c = np.clip(Q_tr, 1e-6, 1.0 - 1e-6)
        idx_n = np.arange(len(Y_tr))
        w_norm = row_w / np.sum(row_w)
        for d in range(6):
            lp = np.log(Q_c[:, d, :])
            yd = Y_tr[:, d]
            if d in c.FULL_DIMS:
                def obj(th):
                    sc = np.exp(th[0])
                    z = lp * sc + np.array([0.0, th[1], th[2], th[3]])
                    z -= z.max(axis=1, keepdims=True)
                    sm = np.exp(z)
                    sm /= sm.sum(axis=1, keepdims=True)
                    nll = -float(np.sum(w_norm * np.log(np.clip(sm[idx_n, yd], 1e-9, 1.0))))
                    return nll + l2 * float(np.sum(th ** 2))
                params[d] = minimize(obj, np.zeros(4), method="BFGS").x
            else:
                lp2 = np.log(Q_c[:, d, [0, 3]])
                yb = (yd == 3).astype(int)
                def obj2(th):
                    sc = np.exp(th[0])
                    z = lp2 * sc + np.array([0.0, th[1]])
                    z -= z.max(axis=1, keepdims=True)
                    sm = np.exp(z)
                    sm /= sm.sum(axis=1, keepdims=True)
                    nll = -float(np.sum(w_norm * np.log(np.clip(sm[idx_n, yb], 1e-9, 1.0))))
                    return nll + l2 * float(np.sum(th ** 2))
                params[d, :2] = minimize(obj2, np.zeros(2), method="BFGS").x
        return params

    def pool_from_member_dict(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        streams, _ = self._build_streams(P_dict)
        Q = apply_nested_routed_pool(streams, self.temps, self.W)
        if self.dirdiag_params is not None:
            Q = tfd.TagFreeRoutedGateEU._apply_dirdiag(Q, self.dirdiag_params)
        if "peg" in P_dict and self.peg_rescue_thr < 1.0:
            p_peg_g = c.renorm(P_dict["peg"])[:, 3, 3]
            m_res = (p_peg_g >= self.peg_rescue_thr) & (p_peg_g > Q[:, 3, 3])
            if np.any(m_res):
                Q = Q.copy()
                Q[m_res, 3, 3] = p_peg_g[m_res]
                Q[m_res, 3, 0] = 1.0 - p_peg_g[m_res]
                Q[m_res, 3, 1:3] = 0.0
                Q[m_res] = c.renorm(Q[m_res])
        if self.bkseg1_alpha > 0.0 and "pe_bkseg1" in P_dict:
            pb = c.renorm(P_dict["pe_bkseg1"])[:, 5, :]
            Q = Q.copy()
            Q[:, 5, :] = (1.0 - self.bkseg1_alpha) * Q[:, 5, :] + self.bkseg1_alpha * pb
            Q = c.renorm(Q)
        if self.deconf is not None:
            Q = self.deconf.transform(Q, P_dict, self.members)
        return Q

    def fit(self, P_dict: Dict[str, np.ndarray], gold: np.ndarray, groups: np.ndarray) -> "NaturalPriorRoutedDecoder":
        row_w = compute_case_control_weights(gold, pi_clean=self.pi_clean)
        streams, prior = self._build_streams(P_dict)
        self.temps, self.W = fit_nested_routed_pool(streams, gold, prior, l2_reg=self.l2_reg, t_max=3.80)
        q_raw = apply_nested_routed_pool(streams, self.temps, self.W)
        self.dirdiag_params = self._fit_weighted_dirdiag(q_raw, gold, row_w, l2=self.dirdiag_l2)

        self.deconf = None
        q_pre = self.pool_from_member_dict(P_dict)
        if self.use_deconf:
            self.deconf = PackageFreeDeconfounder().fit(q_pre, P_dict, self.members, gold, row_w, groups)
            q_fin = self.pool_from_member_dict(P_dict)
        else:
            q_fin = q_pre

        self.eu = NaturalPriorGateEU(pi_clean=self.pi_clean).fit(q_fin, gold, groups=groups)
        return self

    def predict_from_member_dict(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return self.eu.predict(self.pool_from_member_dict(P_dict))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="Verify Step-1 NaturalPrior+Deconf on OOF + emit pilot natprior JSONs")
    ap.add_argument("--finalize-full8", action="store_true", help="Fit on full 8-fold 3,573 OOF & run FullFit test150/final644 inference")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()

    if args.verify:
        from run_clean_eval_infer import decode_and_verify_n1, write_and_validate_json

        base_m = ["cradio_v3", "peg", "v16", "ntg_pe", "p12_pe", "p1_qy", "pe_bkseg1", "ntg_convnext"]
        names, gold, fold, groups, P = tfd.load_member_oof(base_m)
        orig_m = tfd._eval_only_orig_mask(names)
        P["qwen3vl_hires"], ok_q = tfd.load_candidate(os.path.join(REPO, "results/r55/qwen3vl_hires_pilot"), names)
        P["iv6b_native"], ok_i = tfd.load_candidate(os.path.join(REPO, "results/r55/iv6b_native_pilot"), names)
        ok = ok_q & ok_i

        # 1) Verify on full 8-fold `resident_v7_core`: Uniform vs NaturalPrior+Deconf (100% package-free!)
        pred_np = np.zeros_like(gold)
        for k in range(8):
            tr, va = (fold != k), (fold == k)
            dec = NaturalPriorRoutedDecoder(base_m, pi_clean=0.25, use_deconf=True).fit(
                {m: P[m][tr] for m in base_m}, gold[tr], groups[tr]
            )
            pred_np[va] = dec.predict_from_member_dict({m: P[m][va] for m in base_m})
        dec_full_v7c = NaturalPriorRoutedDecoder(base_m, pi_clean=0.25, use_deconf=True).fit(
            {m: P[m] for m in base_m}, gold, groups
        )
        print(
            f"[8-Fold Full 3573] v7_core NaturalPrior(pi=0.25)+Deconf: "
            f"lam_cl={dec_full_v7c.eu.lam_cl} p0_scale={dec_full_v7c.eu.p0_scale} deconf_active={dec_full_v7c.deconf.active}"
        )
        print("  " + tfd._line("v7_core NaturalPrior+Deconf", tfd.summarize(pred_np, gold, orig_m)))

        # 2) Verify on Fold 0-1 (N=872) `dualcore_6m_plus_cradio` & `dualcore_5m_ideal`
        sub = tfd._subfold_ids_within_covered(fold, groups, [0, 1])
        out_eval_dir = os.path.join(REPO, "results/r55/clean_tagfree_eval")
        datasets = [
            ("test150", os.path.join(REPO, "results/v2_full/template_150.json")),
            ("final644", os.path.join(REPO, "results/final/template_final_644.json")),
        ]
        for st_tag, mems in [
            ("dualcore_6m_natprior", ["cradio_v3", "iv6b_native", "qwen3vl_hires", "p12_pe", "ntg_convnext", "peg"]),
            ("dualcore_5m_natprior", ["iv6b_native", "qwen3vl_hires", "p12_pe", "ntg_convnext", "peg"]),
        ]:
            pred_cv = np.zeros_like(gold)
            for sk in [0, 1, 2, 3]:
                va, tr = (sub == sk), (ok & (sub != sk))
                dm = NaturalPriorRoutedDecoder(mems, pi_clean=0.25, use_deconf=True).fit(
                    {m: P[m][tr] for m in mems}, gold[tr], groups[tr]
                )
                pred_cv[va] = dm.predict_from_member_dict({m: P[m][va] for m in mems})
            dm_full = NaturalPriorRoutedDecoder(mems, pi_clean=0.25, use_deconf=True).fit(
                {m: P[m][ok] for m in mems}, gold[ok], groups[ok]
            )
            perm = np.random.default_rng(42).permutation(int(ok.sum()))
            dm_shuf = NaturalPriorRoutedDecoder(mems, pi_clean=0.25, use_deconf=True).fit(
                {m: P[m][ok][perm] for m in mems}, gold[ok][perm], groups[ok][perm]
            )
            p_a = dm_full.predict_from_member_dict({m: P[m][ok][:32] for m in mems})
            p_b = dm_shuf.predict_from_member_dict({m: P[m][ok][:32] for m in mems})
            n1_diff = sum(
                int(np.sum(dm_full.predict_from_member_dict({m: P[m][ok][i : i + 1] for m in mems})[0] != p_a[i]))
                for i in range(32)
            )
            print(
                f"[Fold 0-1 N=872] {st_tag}: lam_cl={dm_full.eu.lam_cl} p0_scale={dm_full.eu.p0_scale} "
                f"deconf={dm_full.deconf.active} shuffle_diffs={int(np.sum(p_a != p_b))} n1_diffs={n1_diff}"
            )
            print("  " + tfd._line(st_tag, tfd.summarize(pred_cv[ok], gold[ok], orig_m[ok])))

            # Re-decode existing fresh Fold-0 pilot probabilities on test150 & final644 immediately
            for ds_tag, tmpl_p in datasets:
                zb = np.load(os.path.join(out_eval_dir, f"{ds_tag}_resident_v7_core_fresh_probs.npz"), allow_pickle=True)
                zq = np.load(os.path.join(out_eval_dir, f"{ds_tag}_qwen3vl_hires_fresh_probs.npz"), allow_pickle=True)
                zi = np.load(os.path.join(out_eval_dir, f"{ds_tag}_iv6b_native_fresh_probs.npz"), allow_pickle=True)
                ds_names = [str(x) for x in zb["names"]]
                P_ds = {m: zb[f"p_{m}"].astype(np.float64) for m in base_m if f"p_{m}" in zb.files}
                P_ds["qwen3vl_hires"] = c.renorm(zq["probs_6x4"].astype(np.float64))
                P_ds["iv6b_native"] = c.renorm(zi["probs_6x4"].astype(np.float64))
                pr_ds = decode_and_verify_n1(dm_full, P_ds, mems)
                out_j = os.path.join(out_eval_dir, f"pred_{ds_tag}_{st_tag}_pilot.json")
                write_and_validate_json(ds_names, pr_ds, tmpl_p, out_j)

    if args.finalize_full8:
        from run_clean_eval_infer import (
            decode_and_verify_n1,
            run_sharded_iv6b_native,
            run_sharded_qwen_hires,
            write_and_validate_json,
        )

        gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        base_m = ["cradio_v3", "peg", "v16", "ntg_pe", "p12_pe", "p1_qy", "pe_bkseg1", "ntg_convnext"]
        names, gold, fold, groups, P = tfd.load_member_oof(base_m)
        orig_m = tfd._eval_only_orig_mask(names)
        P["qwen3vl_hires"], ok_q = tfd.load_candidate(os.path.join(REPO, "results/r55/qwen3vl_hires_pilot"), names)
        P["iv6b_native"], ok_i = tfd.load_candidate(os.path.join(REPO, "results/r55/iv6b_native_pilot"), names)
        assert bool(ok_q.all()) and bool(ok_i.all()), f"Incomplete 8-fold OOF: qwen={ok_q.sum()}/3573, iv6b={ok_i.sum()}/3573"

        out_final_dir = os.path.join(REPO, "results/r55/dualcore_fullfit_final")
        os.makedirs(out_final_dir, exist_ok=True)
        old_eval_dir = os.path.join(REPO, "results/r55/clean_tagfree_eval")

        stacks = [
            ("clean6m_v6_upgraded_fullfit", ["cradio_v3", "peg", "iv6b_native", "ntg_pe", "p12_pe", "qwen3vl_hires"]),
            ("clean7m_plus_cnx_fullfit", ["cradio_v3", "peg", "iv6b_native", "ntg_pe", "ntg_convnext", "p12_pe", "qwen3vl_hires"]),
            ("dualcore_6m_nocradio_ntgpe_fullfit", ["peg", "iv6b_native", "ntg_pe", "ntg_convnext", "p12_pe", "qwen3vl_hires"]),
        ]

        full_report = {"nested_8fold_3573": {}, "datasets": {}}
        fitted_decs = {}
        for st_tag, mems in stacks:
            _, pred_oof, _ = tfd.nested_oof(P, mems, gold, fold, groups)
            s_oof = tfd.summarize(pred_oof, gold, orig_m)
            d_all = tfd.TagFreeRoutedGateEU(**tfd.build_cfg(mems)).fit(
                {m: P[m] for m in mems}, gold, groups
            )
            s_oof["lam_cl"] = d_all.eu.lam_cl
            s_oof["p0_scale"] = d_all.eu.p0_scale
            full_report["nested_8fold_3573"][st_tag] = s_oof
            print("[FULL 8-FOLD 3573 TAG-FREE] " + tfd._line(st_tag, s_oof), flush=True)

            dec_pkl = os.path.join(out_final_dir, f"{st_tag}_decoder.pkl")
            with open(dec_pkl, "wb") as f:
                pickle.dump({"decoder": d_all, "members": mems, "tag_free": True}, f)
            fitted_decs[st_tag] = (d_all, mems)

        qwen_ff_ckpt = os.path.join(REPO, "results/r55/qwen3vl_hires_pilot/ckpts/qwen3vl_hires_s2026_fullfit_swa.pt")
        iv6b_ff_ckpt = os.path.join(REPO, "results/r55/iv6b_native_pilot/fullfit_best.pt")
        assert os.path.exists(qwen_ff_ckpt) and os.path.exists(iv6b_ff_ckpt), "Missing 100% FullFit checkpoints!"

        datasets = [
            ("test150", os.path.join(REPO, "results/v2_full/template_150.json"), os.path.join(REPO, "data/test_150")),
            ("final644", os.path.join(REPO, "results/final/template_final_644.json"), "/opt/final_eval_644"),
        ]
        for ds_tag, tmpl_p, img_d in datasets:
            zb = np.load(os.path.join(old_eval_dir, f"{ds_tag}_resident_v7_core_fresh_probs.npz"), allow_pickle=True)
            ds_names = [str(x) for x in zb["names"]]
            P_ds = {m: zb[f"p_{m}"].astype(np.float64) for m in base_m}
            P_ds["qwen3vl_hires"] = run_sharded_qwen_hires(
                ckpt_path=qwen_ff_ckpt,
                gpus=gpus,
                template_path=tmpl_p,
                img_dir=img_d,
                merged_npz=os.path.join(out_final_dir, f"{ds_tag}_qwen3vl_hires_fullfit_probs.npz"),
            )
            P_ds["iv6b_native"] = run_sharded_iv6b_native(
                ckpt_path=iv6b_ff_ckpt,
                gpus=gpus,
                template_path=tmpl_p,
                img_dir=img_d,
                merged_npz=os.path.join(out_final_dir, f"{ds_tag}_iv6b_native_fullfit_probs.npz"),
            )
            full_report["datasets"][ds_tag] = {}
            for st_tag, (dec_obj, mems) in fitted_decs.items():
                pr_ds = decode_and_verify_n1(dec_obj, P_ds, mems)
                out_j = os.path.join(out_final_dir, f"pred_{ds_tag}_{st_tag}.json")
                full_report["datasets"][ds_tag][st_tag] = write_and_validate_json(ds_names, pr_ds, tmpl_p, out_j)

        with open(os.path.join(out_final_dir, "fullfit_final_report.json"), "w", encoding="utf-8") as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)
        print("=== [FINAL FULL8 + FULLFIT COMPLETE] All submissions saved to results/r55/dualcore_fullfit_final/ ===", flush=True)
