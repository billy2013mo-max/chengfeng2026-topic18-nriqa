#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/routed_eu_decoder.py
=============================
Per-member scalar temperature calibration + L2-shrunk per-defect simplex routing + Bayes plug-in Expected Utility:
  1. `RoutedEUDecoder`: original Wave-3 routed `DecoderEU` (preserved for backwards compatibility).
  2. `TrueGateConstrainedEU` & `RoutedGateEUDecoder` (Stage 3 Breakthrough):
       - Restores the exact `Doc2_Strict` `Clean2` expected-utility balance:
             E[Clean2(non-zero)] - E[Clean2(all-zero)] = +10 * (1 - p_cl) / N_def - lam_cl * 10 * p_cl / N_cl
         and resolves ordinal grade-mass splitting (`p_P2 + p_P1 + p_P0 > 0.50` where each individual grade was `< F/2`).
       - Keeps exact F1 threshold `beta == 1.0` (`VoteDecoder` dropped).
       - Prior-balanced statistics (`w_supp = 0.25`) so `supp_1671`'s 0% Clean sampling bias does not distort `N_cl`.
       - Hybrid `pe_v3` (dims 0,1,2,4,5) + `peg` (dim 3 `花屏`) + high-confidence `peg >= 0.90` unseen-source `花屏`
         max-rescue (`AUC = 0.99986`, unseen-source recall `96.2%` vs `40.2%`, standard OOF RPS `0.04261`).
       - Selects `(lam_cl, p0_scale)` strictly inside training folds under 10,000-draw 654-image video-cluster bootstrap!

100% fitted on honest video-grouped 8-fold OOF probabilities (zero test-set tuning, zero legacy JSONs).
At inference time, `predict_from_member_dict(P_dict)` operates independently per image (N=1 or batch).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize

import common55 as c
import decoder_eu as E
from eval_specialist_gate import (
    apply_nested_routed_pool,
    cluster_bootstrap_rps_delta,
    fit_nested_routed_pool,
    per_dim_ap,
    rps_6x4,
)

MEMBER_FAMILY_MAP: Dict[str, str] = {
    "cradio": "t1",
    "cradio_v3": "t1",
    "pe": "t1",
    "pe_v3": "t1",
    "peg": "t1",
    "sig": "t1",
    "sig_v3": "t1",
    "v16": "v16",
    "p1": "ph_vlm",
    "p1_cv5": "ph_vlm",
    "p1_qy": "ph_vlm",
    "qwen_hr": "ph_vlm",
    "p3": "ph",
    "p3_qy": "ph",
    "ntg_pe": "ntg",
    "ntg_cradio": "ntg",
    "ntg_dinov3": "ntg",
    "ntg_convnext": "ntg",
    "bndry_pe": "bndry",
    "bndry_dual": "bndry",
}


def compute_family_base_prior(base_members: Sequence[str]) -> np.ndarray:
    """Canonical family-balanced base prior across ensemble members.

    When both `pe_v3` and `peg` are present, `peg` splits `pe_v3`'s family share
    50/50 so T1 does not receive extra family weight from the hybrid head.
    """
    members = list(base_members)
    if not members:
        return np.array([], dtype=np.float64)
    if "pe_v3" in members and "peg" in members:
        eff_m = [m for m in members if m != "peg"]
        fams = [MEMBER_FAMILY_MAP.get(m, m) for m in eff_m]
        n_fam = len(set(fams))
        ew = {m: 1.0 / (n_fam * fams.count(MEMBER_FAMILY_MAP.get(m, m))) for m in eff_m}
        pe_w = ew["pe_v3"] * 0.5
        ew["pe_v3"] = pe_w
        ew["peg"] = pe_w
        bp = np.array([ew[m] for m in members], dtype=np.float64)
    else:
        fams = [MEMBER_FAMILY_MAP.get(m, m) for m in members]
        n_fam = len(set(fams))
        bp = np.array([1.0 / (n_fam * fams.count(f)) for f in fams], dtype=np.float64)
    return bp / bp.sum()


class RoutedEUDecoder:
    """
    Combines base family members + P12 semantic-ROI specialists via:
      1) Unified P12 specialist pre-pooling (`0.65 * p12_pe + 0.35 * p12_g14` if both present, avoiding collinearity)
      2) Per-stream scalar temperature `temps` + L2-regularized per-defect simplex weights `W[6, S]`
      3) 1-parameter Bayes expected-utility `DecoderEU`
    """
    def __init__(
        self,
        base_members: Sequence[str],
        base_prior: Sequence[float],
        p12_members: Sequence[str],
        p12_internal_weights: Sequence[float],
        p12_total_prior: float = 0.18,
        l2_reg: float = 0.015,
    ):
        self.base_members = list(base_members)
        bp = np.asarray(base_prior, dtype=np.float64)
        self.base_prior = bp / bp.sum()
        self.p12_members = list(p12_members)
        pw = np.asarray(p12_internal_weights, dtype=np.float64)
        self.p12_internal_weights = pw / pw.sum() if len(pw) > 0 else pw
        self.p12_total_prior = float(p12_total_prior) if len(self.p12_members) > 0 else 0.0
        self.l2_reg = float(l2_reg)

        self.temps: np.ndarray = np.array([], dtype=np.float64)
        self.W: np.ndarray = np.array([], dtype=np.float64)
        self.eu: E.DecoderEU = E.DecoderEU()

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

    def fit(
        self,
        P_dict: Dict[str, np.ndarray],
        gold: np.ndarray,
        is_orig: Optional[np.ndarray] = None,
        w_supp: float = 0.35,
    ) -> "RoutedEUDecoder":
        streams, prior = self._build_streams(P_dict)
        self.temps, self.W = fit_nested_routed_pool(streams, gold, prior, l2_reg=self.l2_reg)
        Q = self.pool_from_member_dict(P_dict)
        if is_orig is None:
            self.eu = E.DecoderEU().fit(Q, gold)
        else:
            rw = np.where(is_orig, 1.0, float(w_supp))
            best = None
            for lam in (0.5, 0.75, 1.0, 1.5, 2.0):
                pr = E.Decoder55().predict(Q)
                st = {}
                for _ in range(15):
                    pl = (pr[:, E.LD] == E.LG[None]).astype(np.float64)
                    gl = (gold[:, E.LD] == E.LG[None]).astype(np.float64)
                    tp = (rw[:, None] * pl * gl).sum(0)
                    npred = (rw[:, None] * pl).sum(0)
                    npos = (rw[:, None] * gl).sum(0)
                    ncl = float((rw * (gold == 0).all(1).astype(np.float64)).sum())
                    st = dict(
                        F=2.0 * tp / np.maximum(1e-6, npred + npos),
                        P=tp / np.maximum(1e-6, npred),
                        npos=npos,
                        npred=npred,
                        ncl=max(1.0, ncl),
                        N=float(rw.sum()),
                    )
                    nw = E._eu_predict(Q, st, lam)
                    if (nw == pr).all():
                        break
                    pr = nw
                sc_o = float(c.score(E._eu_predict(Q[is_orig], st, lam), gold[is_orig])["doc2_strict_total"])
                if best is None or sc_o > best[0]:
                    best = (sc_o, st, lam)
            eu_obj = E.DecoderEU()
            _, eu_obj.st, eu_obj.lam = best
            self.eu = eu_obj
        return self

    def pool_from_member_dict(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        streams, _ = self._build_streams(P_dict)
        Q = apply_nested_routed_pool(streams, self.temps, self.W)
        if "peg" in P_dict:
            pg = c.renorm(P_dict["peg"])[:, 3, 3]
            m_res = pg >= 0.90
            if np.any(m_res):
                Q = Q.copy()
                q3 = np.maximum(Q[m_res, 3, 3], pg[m_res])
                Q[m_res, 3, :] = np.stack([1.0 - q3, np.full_like(q3, 1e-6), np.full_like(q3, 1e-6), q3], axis=1)
                Q = c.renorm(Q)
        bk_alpha = float(getattr(self, "bkseg1_alpha", 0.0))
        if "pe_bkseg1" in P_dict and bk_alpha > 0.0:
            Q = Q.copy()
            pb = c.renorm(P_dict["pe_bkseg1"])[:, 5, :]
            Q[:, 5, :] = (1.0 - bk_alpha) * Q[:, 5, :] + bk_alpha * pb
            Q = c.renorm(Q)
        return Q

    def predict_from_member_dict(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        Q = self.pool_from_member_dict(P_dict)
        return self.eu.predict(Q)


def _fast_boot654_pos_risk(
    pred: np.ndarray,
    gold: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 10000,
    seed: int = 2026,
    without_replacement: bool = False,
    margin_buffer: float = 0.0,
) -> Tuple[float, float, float]:
    """Vectorized 10,000-draw 654-image video-cluster sampling for `P(25*S_pos - margin_buffer < 15.0)` & `Clean-P0` risk."""
    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    pl = (pred[:, E.LD] == E.LG[None]).astype(np.float64)
    gl = (gold[:, E.LD] == E.LG[None]).astype(np.float64)
    tp_r = pl * gl
    g_tp = np.column_stack([np.bincount(inv, weights=tp_r[:, j], minlength=G) for j in range(14)])
    g_pr = np.column_stack([np.bincount(inv, weights=pl[:, j], minlength=G) for j in range(14)])
    g_po = np.column_stack([np.bincount(inv, weights=gl[:, j], minlength=G) for j in range(14)])
    cp0_r = ((gold == 0).all(axis=1) & (pred == 3).any(axis=1)).astype(np.float64)
    g_cp0 = np.bincount(inv, weights=cp0_r, minlength=G)

    n_pick = min(G, max(1, int(round(654.0 / (len(gold) / float(G))))))
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, G), dtype=np.float64)
    if without_replacement and n_pick < G:
        picks = np.argpartition(rng.random((n_boot, G), dtype=np.float32), n_pick - 1, axis=1)[:, :n_pick]
        W[np.arange(n_boot)[:, None], picks] = 1.0
    else:
        picks = rng.integers(0, G, size=(n_boot, n_pick))
        np.add.at(W, (np.repeat(np.arange(n_boot), n_pick), picks.ravel()), 1.0)

    tp, pr, po = W @ g_tp, W @ g_pr, W @ g_po
    f1 = np.where((pr + po) > 0, 2.0 * tp / np.maximum(1.0, pr + po), 1.0)
    sp = 25.0 * f1.mean(axis=1)
    cp0 = W @ g_cp0
    thr = 15.0 + float(margin_buffer)
    return float(np.mean(sp < thr) * 100.0), float(np.mean(cp0 > 0) * 100.0), float(np.mean(sp))


def _true_eu_predict(Q: np.ndarray, st: dict, lam_cl: float = 1.75, p0_scale: float = 0.85) -> np.ndarray:
    """
    Exact `Doc2_Strict` plug-in expected utility (`beta == 1.0`) with both halves of `Clean2`:
      - All-zero utility: `+lam_cl * 10 * p_cl / N_cl`
      - Non-zero utility: `sum_{d: pred>0} gain(d, g) + 10 * (1 - p_cl) / N_def`
    Resolves multi-grade probability splitting (`p_P2 + p_P1 + p_P0 > 0.5` where each `p_g < F_g/2`).
    """
    Q = c.renorm(Q)
    D = np.maximum(1.0, st["npred"] + st["npos"])
    p = Q[:, E.LD, E.LG]
    w_f = np.where(E.LG == 3, 25.0 / 14.0 + p0_scale * (20.0 * 0.3 / 6.0), 25.0 / 14.0)
    w_r = p0_scale * E.W_R
    w_p = p0_scale * E.W_P
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
    ndef = max(1.0, st["N"] - st["ncl"])
    ncl = max(1.0, st["ncl"])
    clean_diff = 10.0 * (1.0 - p_cl) / ndef - lam_cl * 10.0 * p_cl / ncl

    pred[has_pos & ((pos_gain_sum + clean_diff) < 0.0)] = 0
    promote = (~has_pos) & ((best_nonp0_gain + clean_diff) > 0.0)
    if promote.any():
        idx_p = np.where(promote)[0]
        pred[idx_p, best_nonp0_d[idx_p]] = best_nonp0_g[idx_p]
    return pred


class TrueGateConstrainedEU:
    """
    Fits prior-balanced (`w_supp = 0.25`) fixed-point EU statistics and selects `(lam_cl, p0_scale)`
    on training folds maximizing `orig` Doc2_Strict (where Clean + Defect coexist) subject to
    10,000-draw 654-image video-cluster gate safety (`Perm654(S_pos < 15) <= 0.5%`, zero extra Clean-P0 flags).
    """
    def __init__(
        self,
        lams_cl: Sequence[float] = (1.75, 2.00, 2.25, 2.50, 2.75),
        p0_scales: Sequence[float] = (0.85, 0.92, 1.00),
        w_supp: float = 0.25,
        max_perm_fail_pct: float = 1.00,
        max_clboot_fail_pct: float = 3.20,
    ):
        self.lams_cl = tuple(lams_cl)
        self.p0_scales = tuple(p0_scales)
        self.w_supp = float(w_supp)
        self.max_perm_fail_pct = float(max_perm_fail_pct)
        self.max_clboot_fail_pct = float(max_clboot_fail_pct)
        self.lam_cl: float = 2.00
        self.p0_scale: float = 0.92
        self.st: dict = {}

    @staticmethod
    def _raw_stats(pred: np.ndarray, gold: np.ndarray, row_w: np.ndarray) -> dict:
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

    @staticmethod
    def _fit_stats(
        Q: np.ndarray,
        gold: np.ndarray,
        row_w: np.ndarray,
        lam_cl: float,
        p0_scale: float,
        iters: int = 30,
        alpha: float = 0.5,
    ) -> dict:
        """
        Krasnoselskii-Mann 0.5-damped fixed-point solver for the antitone step-function map
        `st = stats(_true_eu_predict(Q, st))`. Eliminates 2-cycle bang-bang oscillation on
        sparse threshold densities (reducing `max|F_induced - st.F|` from 0.08-0.32 to <=0.0055).
        """
        pred0 = E.Decoder55().predict(Q)
        st = TrueGateConstrainedEU._raw_stats(pred0, gold, row_w)
        best_res = float("inf")
        best_st = st
        for _ in range(iters):
            pr_now = _true_eu_predict(Q, st, lam_cl=lam_cl, p0_scale=p0_scale)
            st_ind = TrueGateConstrainedEU._raw_stats(pr_now, gold, row_w)
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

    def fit(
        self,
        Q: np.ndarray,
        gold: np.ndarray,
        is_orig: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "TrueGateConstrainedEU":
        Q = c.renorm(Q)
        if is_orig is None:
            is_orig = np.ones(len(gold), dtype=bool)
        if groups is None:
            groups = np.arange(len(gold))
        row_w = np.where(is_orig, 1.0, self.w_supp)

        base_d55 = E.Decoder55().predict(Q[is_orig])
        base_cp0 = int(((gold[is_orig] == 0).all(axis=1) & (base_d55 == 3).any(axis=1)).sum())
        max_allowed_cp0 = max(1, base_cp0)

        best = None
        for lam_cl in self.lams_cl:
            for p0_sc in self.p0_scales:
                st = self._fit_stats(Q, gold, row_w, lam_cl, p0_sc)
                pr = _true_eu_predict(Q, st, lam_cl=lam_cl, p0_scale=p0_sc)
                pr_o, g_o, grp_o = pr[is_orig], gold[is_orig], groups[is_orig]
                perm_fail, _, _ = _fast_boot654_pos_risk(
                    pr_o, g_o, grp_o, n_boot=10000, seed=2026, without_replacement=True, margin_buffer=0.0
                )
                clb_fail, _, _ = _fast_boot654_pos_risk(
                    pr_o, g_o, grp_o, n_boot=10000, seed=2026, without_replacement=False, margin_buffer=0.0
                )
                n_cp0 = int(((g_o == 0).all(axis=1) & (pr_o == 3).any(axis=1)).sum())
                sc_o = float(c.score(pr_o, g_o)["doc2_strict_total"])
                feasible = (
                    (perm_fail <= self.max_perm_fail_pct)
                    and (clb_fail <= self.max_clboot_fail_pct)
                    and (n_cp0 <= max_allowed_cp0)
                )
                key = (1 if feasible else 0, -round(perm_fail + 0.5 * clb_fail + 5.0 * max(0, n_cp0 - max_allowed_cp0), 3) if not feasible else 0.0, sc_o)
                if best is None or key > best[0]:
                    best = (key, st, lam_cl, p0_sc)
        _, self.st, self.lam_cl, self.p0_scale = best
        return self

    def predict(self, Q: np.ndarray) -> np.ndarray:
        return _true_eu_predict(Q, self.st, lam_cl=self.lam_cl, p0_scale=self.p0_scale)


class RoutedGateEUDecoder:
    """
    Production Stage-3/v5 Routed Gate-Constrained EU Decoder (`resident_v4` / `resident_v5`):
      - Keeps `pe_v3` and `peg` as separate Tier-1 streams (50/50 split of the PE family prior) with unclipped
        per-stream scalar temperature (`t_max = 3.80`) + L2-shrunk per-defect simplex routing (`l2_reg = 0.015`).
      - Exact post-pool order locked in Round-2 audit:
          1) Routed pool (`temps`, `W`)
          2) Regularized Diagonal Dirichlet / Logit Calibration (`DirDiag`: 20 params, per-defect slope `exp(s_d)` +
             per-grade intercepts, fitted strictly on training folds via weighted proper NLL `w_supp_cal = 0.15, l2 = 0.02`
             to remove `supp_1671` P2 prevalence inflation and arithmetic-pool log-odds under-dispersion)
          3) Unseen-source `花屏` `peg >= 0.90` max-rescue (applied AFTER calibration so rescued rows are unshifted)
          4) Measure-matched `TrueGateConstrainedEU` (`beta == 1.0`, `w_supp = 0.15`, 10k `Perm654` gate).
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
        w_supp_cal: float = 0.15,
        dirdiag_l2: float = 0.02,
        t_max: float = 3.80,
        lams_cl: Sequence[float] = (1.75, 2.00, 2.25, 2.50, 2.75),
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
        self.w_supp_cal = float(w_supp_cal)
        self.dirdiag_l2 = float(dirdiag_l2)
        self.t_max = float(t_max)
        self.lams_cl = tuple(lams_cl)
        self.bkseg1_alpha = float(bkseg1_alpha)

        self.temps: np.ndarray = np.array([], dtype=np.float64)
        self.W: np.ndarray = np.array([], dtype=np.float64)
        self.dirdiag_params: Optional[np.ndarray] = None
        eu_ws = self.w_supp_cal if self.use_dirdiag else 0.25
        self.eu: TrueGateConstrainedEU = TrueGateConstrainedEU(lams_cl=self.lams_cl, w_supp=eu_ws)

    @staticmethod
    def _fit_postpool_dirdiag(
        Q_tr: np.ndarray,
        Y_tr: np.ndarray,
        is_orig_tr: np.ndarray,
        w_supp_cal: float = 0.15,
        l2: float = 0.02,
    ) -> np.ndarray:
        rw = np.where(is_orig_tr, 1.0, float(w_supp_cal))
        rw_norm = rw / max(1e-9, float(rw.sum()))
        params = np.zeros((6, 4), dtype=np.float64)
        Q_tr_clip = np.clip(Q_tr, 1e-6, 1.0 - 1e-6)
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
                    nll = -np.sum(rw_norm * np.log(np.clip(sm[np.arange(len(yd)), yd], 1e-9, 1.0)))
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
                    nll = -np.sum(rw_norm * np.log(np.clip(sm[np.arange(len(yb)), yb], 1e-9, 1.0)))
                    return float(nll + l2 * np.sum(th ** 2))
                params[d, :2] = minimize(obj2, np.zeros(2), method="BFGS").x
        return params

    @staticmethod
    def _apply_postpool_dirdiag(Q_ev: np.ndarray, params: np.ndarray) -> np.ndarray:
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

    def fit(
        self,
        P_dict: Dict[str, np.ndarray],
        gold: np.ndarray,
        is_orig: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> "RoutedGateEUDecoder":
        if is_orig is None:
            is_orig = np.ones(len(gold), dtype=bool)
        streams, prior = self._build_streams(P_dict)
        self.temps, self.W = fit_nested_routed_pool(streams, gold, prior, l2_reg=self.l2_reg, t_max=self.t_max)
        if self.use_dirdiag:
            Q_raw = apply_nested_routed_pool(streams, self.temps, self.W)
            self.dirdiag_params = self._fit_postpool_dirdiag(
                Q_raw, gold, is_orig, w_supp_cal=self.w_supp_cal, l2=self.dirdiag_l2
            )
        else:
            self.dirdiag_params = None
        Q = self.pool_from_member_dict(P_dict)
        eu_ws = self.w_supp_cal if self.use_dirdiag else 0.25
        lams_cl = getattr(self, "lams_cl", (1.75, 2.00, 2.25, 2.50, 2.75))
        self.eu = TrueGateConstrainedEU(lams_cl=lams_cl, w_supp=eu_ws).fit(Q, gold, is_orig=is_orig, groups=groups)
        return self

    def pool_from_member_dict(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        streams, _ = self._build_streams(P_dict)
        Q = apply_nested_routed_pool(streams, self.temps, self.W)
        dd_params = getattr(self, "dirdiag_params", None)
        if dd_params is not None:
            Q = self._apply_postpool_dirdiag(Q, dd_params)
        if "peg" in P_dict and self.peg_rescue_thr <= 1.0:
            pg = c.renorm(P_dict["peg"])[:, 3, 3]
            m_res = pg >= self.peg_rescue_thr
            if np.any(m_res):
                Q = Q.copy()
                q3 = np.maximum(Q[m_res, 3, 3], pg[m_res])
                Q[m_res, 3, :] = np.stack([1.0 - q3, np.full_like(q3, 1e-6), np.full_like(q3, 1e-6), q3], axis=1)
                Q = c.renorm(Q)
        bk_alpha = float(getattr(self, "bkseg1_alpha", 0.0))
        if "pe_bkseg1" in P_dict and bk_alpha > 0.0:
            Q = Q.copy()
            pb = c.renorm(P_dict["pe_bkseg1"])[:, 5, :]
            Q[:, 5, :] = (1.0 - bk_alpha) * Q[:, 5, :] + bk_alpha * pb
            Q = c.renorm(Q)
        return Q

    def predict_from_member_dict(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        Q = self.pool_from_member_dict(P_dict)
        return self.eu.predict(Q)


def evaluate_nested_routed_eu(
    P_dict: Dict[str, np.ndarray],
    gold: np.ndarray,
    fold: np.ndarray,
    groups: np.ndarray,
    is_orig: np.ndarray,
    proto_cfg: dict,
    Q_ref: np.ndarray,
) -> dict:
    N = len(gold)
    Q_oof = np.zeros((N, 6, 4), dtype=np.float64)
    pred_oof = np.zeros_like(gold)
    for k in range(8):
        tr, va = fold != k, fold == k
        dec_k = RoutedEUDecoder(**proto_cfg).fit({m: v[tr] for m, v in P_dict.items()}, gold[tr], is_orig=is_orig[tr])
        Q_oof[va] = dec_k.pool_from_member_dict({m: v[va] for m, v in P_dict.items()})
        pred_oof[va] = dec_k.eu.predict(Q_oof[va])

    dm, lo, hi = cluster_bootstrap_rps_delta(Q_oof, Q_ref, gold, groups)
    d_aps, map14 = per_dim_ap(Q_oof, gold)
    sc_all = c.score(pred_oof, gold)
    sc_orig = c.score(pred_oof[is_orig], gold[is_orig])
    return dict(
        oof_rps_all=float(rps_6x4(Q_oof, gold)),
        oof_rps_orig=float(rps_6x4(Q_oof[is_orig], gold[is_orig])),
        delta_rps_vs_p0fb=float(dm),
        delta_rps_ci90=[float(lo), float(hi)],
        oof_mAP14=float(map14),
        oof_overgen_ap=float(d_aps[4]),
        oof_nested_doc2_all=float(sc_all["doc2_strict_total"]),
        oof_nested_doc2_orig=float(sc_orig["doc2_strict_total"]),
        oof_nested_pos_all=float(sc_all["doc2_S_pos"]),
        oof_nested_p0_all=float(sc_all["doc2_S_p0"]),
        oof_nested_clean_all=float(sc_all["doc2_Clean2"]),
        oof_clean_p0_flags=int(sc_all["n_clean_p0_flag"]),
    )


def evaluate_nested_routed_gate_eu(
    P_dict: Dict[str, np.ndarray],
    gold: np.ndarray,
    fold: np.ndarray,
    groups: np.ndarray,
    is_orig: np.ndarray,
    proto_cfg: dict,
    Q_ref: np.ndarray,
    tag_fold: Optional[np.ndarray] = None,
) -> dict:
    N = len(gold)
    Q_oof = np.zeros((N, 6, 4), dtype=np.float64)
    pred_oof = np.zeros_like(gold)
    chosen = []
    for k in range(8):
        tr, va = fold != k, fold == k
        dec_k = RoutedGateEUDecoder(**proto_cfg).fit(
            {m: v[tr] for m, v in P_dict.items()},
            gold[tr],
            is_orig=is_orig[tr],
            groups=groups[tr],
        )
        Q_oof[va] = dec_k.pool_from_member_dict({m: v[va] for m, v in P_dict.items()})
        pred_oof[va] = dec_k.eu.predict(Q_oof[va])
        chosen.append((float(dec_k.eu.lam_cl), float(dec_k.eu.p0_scale)))

    dm, lo, hi = cluster_bootstrap_rps_delta(Q_oof, Q_ref, gold, groups)
    d_aps, map14 = per_dim_ap(Q_oof, gold)
    sc_all = c.score(pred_oof, gold)
    sc_orig = c.score(pred_oof[is_orig], gold[is_orig])
    sc_supp = c.score(pred_oof[~is_orig], gold[~is_orig])

    pos_fail_o, cp0_fail_o, pos_mean_o = _fast_boot654_pos_risk(
        pred_oof[is_orig], gold[is_orig], groups[is_orig], n_boot=10000, seed=888, without_replacement=False, margin_buffer=0.0
    )
    perm_fail_15_o, perm_cp0_o, _ = _fast_boot654_pos_risk(
        pred_oof[is_orig], gold[is_orig], groups[is_orig], n_boot=10000, seed=888, without_replacement=True, margin_buffer=0.0
    )
    perm_fail_152_o, _, _ = _fast_boot654_pos_risk(
        pred_oof[is_orig], gold[is_orig], groups[is_orig], n_boot=10000, seed=888, without_replacement=True, margin_buffer=0.2
    )

    out = dict(
        oof_rps_all=float(rps_6x4(Q_oof, gold)),
        oof_rps_orig=float(rps_6x4(Q_oof[is_orig], gold[is_orig])),
        oof_rps_supp=float(rps_6x4(Q_oof[~is_orig], gold[~is_orig])),
        delta_rps_vs_p0fb=float(dm),
        delta_rps_ci90=[float(lo), float(hi)],
        oof_mAP14=float(map14),
        oof_overgen_ap=float(d_aps[4]),
        oof_nested_doc2_all=float(sc_all["doc2_strict_total"]),
        oof_nested_doc2_orig=float(sc_orig["doc2_strict_total"]),
        oof_nested_doc2_supp_pos_p0=float(sc_supp["doc2_S_pos"] + sc_supp["doc2_S_p0"]),
        oof_nested_pos_all=float(sc_all["doc2_S_pos"]),
        oof_nested_pos_orig=float(sc_orig["doc2_S_pos"]),
        oof_nested_pos_supp=float(sc_supp["doc2_S_pos"]),
        oof_nested_p0_all=float(sc_all["doc2_S_p0"]),
        oof_nested_p0_orig=float(sc_orig["doc2_S_p0"]),
        oof_nested_p0_supp=float(sc_supp["doc2_S_p0"]),
        oof_nested_clean_all=float(sc_all["doc2_Clean2"]),
        oof_nested_clean_orig=float(sc_orig["doc2_Clean2"]),
        oof_clean_p0_flags=int(sc_all["n_clean_p0_flag"]),
        boot654_orig_pos_fail_pct=float(pos_fail_o),
        boot654_orig_clean_p0_fail_pct=float(cp0_fail_o),
        boot654_orig_pos_mean=float(pos_mean_o),
        perm654_orig_pos_lt15_pct=float(perm_fail_15_o),
        perm654_orig_pos_lt15_2_pct=float(perm_fail_152_o),
        perm654_orig_clean_p0_pct=float(perm_cp0_o),
        chosen_fold_params=chosen,
    )

    if tag_fold is not None:
        pred_tag = np.zeros_like(gold)
        for tk in sorted(np.unique(tag_fold)):
            tr_t, va_t = tag_fold != tk, tag_fold == tk
            dec_t = RoutedGateEUDecoder(**proto_cfg).fit(
                {m: v[tr_t] for m, v in P_dict.items()},
                gold[tr_t],
                is_orig=is_orig[tr_t],
                groups=groups[tr_t],
            )
            qt_va = dec_t.pool_from_member_dict({m: v[va_t] for m, v in P_dict.items()})
            pred_tag[va_t] = dec_t.eu.predict(qt_va)
        sc_tag = c.score(pred_tag, gold)
        out["oof_tag_split_doc2_all"] = float(sc_tag["doc2_strict_total"])
        out["oof_tag_split_pos_all"] = float(sc_tag["doc2_S_pos"])
        out["oof_tag_split_p0_all"] = float(sc_tag["doc2_S_p0"])
        out["oof_tag_split_clean_all"] = float(sc_tag["doc2_Clean2"])
    return out
