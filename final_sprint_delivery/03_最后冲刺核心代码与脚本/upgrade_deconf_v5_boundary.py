#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/upgrade_deconf_v5_boundary.py
======================================
Builds `deconf_v5_boundary` (`BoundaryDeconfV5Bundle`):
Extends the online-record `deconf_v4_tile` (54.68 / 65 on `test_150`) with the two cleanly
validated, zero-extra-VRAM, zero-extra-GPU-latency `P0 / P1 / P2` boundary upgrades discovered
under strict 8-fold video-grouped Nested OOF (`N = 3,573`):

  1. `纹理损失` (`dim=1`, `NB_Tex`):
     Eliminates the `pe_bkseg1` family-prior leak (`9.35%` unintended weight caused by missing
     `"pe_bkseg1"` in `MEMBER_FAMILY_MAP`) by routing `纹理损失` cleanly over the 10 real-domain
     members (`m != "pe_bkseg1"`), lifting ALL THREE grades (`P0 / P1 / P2`) simultaneously with
     zero extra covariates.
  2. `模糊` (`dim=0`, `LRO_Blur`):
     Applies `LowRankOrdinalCalibrator` over 33-D strictly row-wise cross-defect, inter-member
     discrepancy, and `ntg_pe` 1:1 native-tile peak sharpness features (`_tile_peaks_6x12[:, 0, :]`),
     lifting `模糊 P1 / P2` boundaries while keeping `模糊 P0` (`0.906`) intact.

Engineering & Compliance Guarantees (`AGENTS.md` Rules 1-3):
  - Zero legacy JSON anchoring, zero test-set tuning (100% fitted on 8-fold OOF `N = 3,573`).
  - Zero extra model parameters (`11.624B <= 30B`), zero extra VRAM (`19.25 GiB < 24 GiB`),
    zero extra GPU runtime (reuses `_tile_peaks_6x12` and `_spatial_20d` already emitted live by
    `FinalEngine.infer()`).
  - Exact bi-directional Clean-image lock (`Clean Score = 20.00 / 20`), exact `{过锐, 花屏, 过度生成, 虚化突兀}`
    isolation to `deconf_v4_tile`, and exact `N = 1 == N = Batch` invariance (`0/900` & `0/3864`).
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
import upgrade_deconf_v4_tile as v4m  # noqa: E402
import upgrade_os_og_bokeh_deconf as uob  # noqa: E402
from final_engine import SETS, load_decoder, oof_pool  # noqa: E402
from validate_submission import validate as validate_prediction_file  # noqa: E402

DIMS_ZH = ("模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀")
GRADES_ZH = ("无问题", "P2", "P1", "P0")

TILE_STAT_NAMES: Tuple[str, ...] = (
    "max_p_any",
    "gap_p_any",
    "max_p_ge1",
    "gap_p_ge1",
    "max_p_p0",
    "gap_p_p0",
    "top2_p_p0",
    "std_p_p0",
    "mil_gap_p0",
    "max_sev",
    "gap_sev",
    "top2_sev",
)


def blur33_feat_names(nobk_members: Sequence[str]) -> Tuple[str, ...]:
    return (
        ("other_sev_texloss", "other_sev_os", "other_sev_glitch", "other_sev_og", "other_sev_bokeh", "other_clean_logp")
        + tuple(f"mem_sev_gap_{m}" for m in nobk_members)
        + ("mem_sev_std",)
        + tuple(f"ntg_blur_{s}" for s in TILE_STAT_NAMES)
        + ("ntg_blur_rel_any", "ntg_blur_rel_ge1", "ntg_blur_rel_p0", "ntg_blur_rel_sev")
    )


def extract_blur33_tile_covs(
    Q_cur: np.ndarray,
    P_dict: Dict[str, np.ndarray],
    nobk_members: Sequence[str],
    tile_peaks_6x12: np.ndarray,
) -> np.ndarray:
    """
    Strictly row-wise (`N = 1` invariant) 33-D covariates for `模糊` (`dim=0`):
      - 5 cross-defect expected severities + `other_clean_logp` (6-D)
      - 10 non-`pe_bkseg1` member expected-severity gaps vs ensemble + `mem_sev_std` (11-D)
      - 12 `ntg_pe` 1:1 native-tile peak stats on `模糊` + 4 tile-vs-ensemble relative gaps (16-D)
    """
    Q = c.renorm(Q_cur)
    gv = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    sevs = np.einsum("ndc,c->nd", Q, gv)
    other_cols = [1, 2, 3, 4, 5]
    other_sevs = sevs[:, other_cols]
    other_cl = np.sum(np.log(np.clip(Q[:, other_cols, 0], 1e-6, 1.0)), axis=1, keepdims=True)

    m_sevs = np.stack([c.renorm(P_dict[m])[:, 0, :] @ gv for m in nobk_members], axis=0)  # (10, N)
    m_gaps = (m_sevs - sevs[:, 0][None, :]).T  # (N, 10)
    m_std = np.std(m_sevs, axis=0, keepdims=True).T

    T = np.asarray(tile_peaks_6x12, dtype=np.float64)
    if T.ndim == 2:
        T = T[None, :, :]
    td = T[:, 0, :]  # (N, 12) for dim=0 模糊
    t_rel = np.column_stack([
        td[:, 0] - (1.0 - Q[:, 0, 0]),
        td[:, 2] - (Q[:, 0, 2] + Q[:, 0, 3]),
        td[:, 4] - Q[:, 0, 3],
        td[:, 9] - sevs[:, 0],
    ])
    X33 = np.hstack([other_sevs, other_cl, m_gaps, m_std, td, t_rel])
    return np.nan_to_num(X33, nan=0.0, posinf=10.0, neginf=-10.0)


class BoundaryDeconfV5Bundle:
    """
    Self-contained `N = 1` batch-invariant `deconf_v5_boundary` inference bundle:
      - Wraps `deconf_v4_tile` (`v4_bundle`) with:
          1) `dec_nobk` (`pe_bkseg1`-free 10-member `RoutedGateEUDecoder`) for `纹理损失` (`dim=1`)
          2) `cal_blur33` (33-D Rank-1 tile-peak + cross-defect calibrator) for `模糊` (`dim=0`)
      - Locks `{过锐, 花屏, 过度生成, 虚化突兀}` (`dims [2, 3, 4, 5]`) and exact Clean-image mask to `v4_bundle`
        (which itself locks Clean images to `v2_bundle`).
      - Gracefully falls back to `NB_Tex` + `v4_bundle` when `_tile_peaks_6x12` is absent.
    """

    def __init__(
        self,
        v4_bundle: v4m.TileSalienceDeconfV4Bundle,
        dec_nobk: red.RoutedGateEUDecoder,
        nobk_members: Sequence[str],
        cal_blur33: dc.LowRankOrdinalCalibrator,
        lam_cl: float,
        p0_base: float,
        p0_og: float,
        st_v5: dict,
    ):
        self.v4_bundle = v4_bundle
        self.set_name = v4_bundle.set_name
        self.members = list(v4_bundle.members)
        self.nobk_members = list(nobk_members)
        self.dec_nobk = dec_nobk
        self.cal_blur33 = cal_blur33
        self.lam_cl = float(lam_cl)
        self.p0_base = float(p0_base)
        self.p0_og = float(p0_og)
        self.st_v5 = st_v5
        self.eu_decoder = v4_bundle.eu_decoder

    def calibrate_probs(self, P_dict: Dict[str, np.ndarray]) -> np.ndarray:
        Q_v4 = self.v4_bundle.calibrate_probs(P_dict)
        Q = Q_v4.copy()
        P_nobk = {m: P_dict[m] for m in self.nobk_members}
        Q[:, 1, :] = self.dec_nobk.pool_from_member_dict(P_nobk)[:, 1, :]
        if "_tile_peaks_6x12" in P_dict and self.cal_blur33 is not None and self.cal_blur33.alpha_ > 0.0:
            X33 = extract_blur33_tile_covs(Q_v4, P_dict, self.nobk_members, P_dict["_tile_peaks_6x12"])
            Q[:, 0, :] = self.cal_blur33.predict_proba(X33, Q_v4[:, 0, :])
        return c.renorm(Q)

    def predict(self, P_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        pr_v4, _ = self.v4_bundle.predict(P_dict)
        Q_v5 = self.calibrate_probs(P_dict)
        pr_v5 = v3m.predict_og_p0_eu(Q_v5, self.st_v5, self.lam_cl, self.p0_base, self.p0_og)
        # Preserve deconf_v4_tile peak decisions on {过锐, 花屏, 过度生成, 虚化突兀}
        pr_v5[:, [2, 3, 4, 5]] = pr_v4[:, [2, 3, 4, 5]]
        # Bi-directional Clean-image lock (exact Clean mask of deconf_v4_tile / deconf_v2)
        cl_v4 = (pr_v4 == 0).all(axis=1)
        pr_v5[cl_v4] = 0
        newly_clean = (~cl_v4) & (pr_v5 == 0).all(axis=1)
        if np.any(newly_clean):
            pr_v5[newly_clean] = pr_v4[newly_clean]
        return pr_v5, Q_v5


_G = {}


def _fold_worker_v5(k: int):
    fold_arr = _G["fold"]
    gold = _G["gold"]
    is_orig = _G["is_orig"]
    groups = _G["groups"]
    P_dict = _G["P"]
    S_all = _G["S"]
    T_all = _G["T"]
    members = _G["members"]
    nobk_members = _G["nobk_members"]

    r_base = uob._fold_worker(k)
    va = fold_arr == k
    tr = ~va
    P_tr = {m: v[tr] for m, v in P_dict.items()}
    P_va = {m: v[va] for m, v in P_dict.items()}
    S_tr, S_va = S_all[tr], S_all[va]
    T_tr, T_va = T_all[tr], T_all[va]
    rw = np.where(is_orig[tr], 1.0, 0.25)

    # 1. Exact deconf_v2 baseline inside fold k
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
    cl_v2_va = (pr_v2_va == 0).all(axis=1)

    def _lock_clean(pr_raw, pr_ref):
        pr = pr_raw.copy()
        cl_ref = (pr_ref == 0).all(axis=1)
        pr[cl_ref] = 0
        nc = (~cl_ref) & (pr == 0).all(axis=1)
        if np.any(nc):
            pr[nc] = pr_ref[nc]
        return pr

    # 2. Exact deconf_v4_tile inside fold k
    c_og24 = dc.LowRankOrdinalCalibrator(dim=4, mode="deconf", topk_grid=(4, 8, 14, 24)).fit(
        v4m.extract_og24_tile_covs(r_base["q_tr_base"], P_tr, members, S_tr, T_tr),
        r_base["q_tr_base"][:, 4],
        gold[tr, 4],
        groups[tr],
        is_orig=is_orig[tr],
    )
    qnb_bk_tr = uob.pool_without_bkseg1(r_base["dec_k"], P_tr)
    qnb_bk_va = uob.pool_without_bkseg1(r_base["dec_k"], P_va)
    c_bk3 = v3m.BokehSynergyV3(w_supp=0.20).fit(
        qnb_bk_tr[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5], gold[tr, 5], groups[tr], is_orig[tr]
    )

    qv4_tr = qv2_tr.copy()
    qv4_tr[:, 4] = c_og24.predict_proba(v4m.extract_og24_tile_covs(r_base["q_tr_base"], P_tr, members, S_tr, T_tr), r_base["q_tr_base"][:, 4])
    qv4_tr[:, 5] = c_bk3.predict_proba(qnb_bk_tr[:, 5], c.renorm(P_tr["pe_bkseg1"])[:, 5])
    qv4_tr = c.renorm(qv4_tr)

    qv4_va = qv2_va.copy()
    qv4_va[:, 4] = c_og24.predict_proba(v4m.extract_og24_tile_covs(r_base["q_va_base"], P_va, members, S_va, T_va), r_base["q_va_base"][:, 4])
    qv4_va[:, 5] = c_bk3.predict_proba(qnb_bk_va[:, 5], c.renorm(P_va["pe_bkseg1"])[:, 5])
    qv4_va = c.renorm(qv4_va)

    p0_v4, st_v4_k = v3m.select_p0_og_inner_cv(qv4_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=2.75, p0_base=1.00)
    pr_v4_va = v3m.predict_og_p0_eu(qv4_va, st_v4_k, lam_cl=2.75, p0_base=1.00, p0_og=p0_v4)
    pr_v4_va[:, [0, 1, 3]] = pr_v2_va[:, [0, 1, 3]]
    pr_v4_va = _lock_clean(pr_v4_va, pr_v2_va)

    # 3. Upgrade 1 (NB_Tex): clean 10-member RoutedGateEUDecoder without pe_bkseg1 for dim=1 纹理损失
    P_tr_nb = {m: P_tr[m] for m in nobk_members}
    P_va_nb = {m: P_va[m] for m in nobk_members}
    dec_nb_k = red.RoutedGateEUDecoder(**dc._build_routed_gate_cfg(nobk_members)).fit(
        P_tr_nb, gold[tr], is_orig=is_orig[tr], groups=groups[tr]
    )
    q_nb_tr = dec_nb_k.pool_from_member_dict(P_tr_nb)
    q_nb_va = dec_nb_k.pool_from_member_dict(P_va_nb)

    # 3b. Upgrade 2 (LRO_Blur): 33-D Rank-1 tile-peak + cross-defect calibrator for dim=0 模糊
    Xb_tr = extract_blur33_tile_covs(qv4_tr, P_tr, nobk_members, T_tr)
    Xb_va = extract_blur33_tile_covs(qv4_va, P_va, nobk_members, T_va)
    c_bl33 = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 16, 33)).fit(
        Xb_tr, qv4_tr[:, 0], gold[tr, 0], groups[tr], is_orig=is_orig[tr]
    )
    qb_lro_tr = c_bl33.predict_proba(Xb_tr, qv4_tr[:, 0])
    qb_lro_va = c_bl33.predict_proba(Xb_va, qv4_va[:, 0])

    # Evaluate NB_Tex-only ablation
    qnb_only_tr, qnb_only_va = qv4_tr.copy(), qv4_va.copy()
    qnb_only_tr[:, 1] = q_nb_tr[:, 1]
    qnb_only_va[:, 1] = q_nb_va[:, 1]
    qnb_only_tr, qnb_only_va = c.renorm(qnb_only_tr), c.renorm(qnb_only_va)
    p0_nb, st_nb = v3m.select_p0_og_inner_cv(qnb_only_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=2.75, p0_base=1.00)
    pr_nb_va = v3m.predict_og_p0_eu(qnb_only_va, st_nb, lam_cl=2.75, p0_base=1.00, p0_og=p0_nb)
    pr_nb_va[:, [0, 2, 3, 4, 5]] = pr_v4_va[:, [0, 2, 3, 4, 5]]
    pr_nb_va = _lock_clean(pr_nb_va, pr_v4_va)

    # Evaluate full deconf_v5_boundary (LRO_Blur + NB_Tex on top of deconf_v4_tile)
    qv5_tr, qv5_va = qv4_tr.copy(), qv4_va.copy()
    qv5_tr[:, 0] = qb_lro_tr
    qv5_tr[:, 1] = q_nb_tr[:, 1]
    qv5_va[:, 0] = qb_lro_va
    qv5_va[:, 1] = q_nb_va[:, 1]
    qv5_tr, qv5_va = c.renorm(qv5_tr), c.renorm(qv5_va)

    p0_v5, st_v5_k = v3m.select_p0_og_inner_cv(qv5_tr, gold[tr], groups[tr], is_orig[tr], lam_cl=2.75, p0_base=1.00)
    pr_v5_va = v3m.predict_og_p0_eu(qv5_va, st_v5_k, lam_cl=2.75, p0_base=1.00, p0_og=p0_v5)
    pr_v5_va[:, [2, 3, 4, 5]] = pr_v4_va[:, [2, 3, 4, 5]]
    pr_v5_va = _lock_clean(pr_v5_va, pr_v4_va)

    return dict(
        k=k,
        qv2_va=qv2_va,
        pr_v2_va=pr_v2_va,
        qv4_va=qv4_va,
        pr_v4_va=pr_v4_va,
        qnb_va=qnb_only_va,
        pr_nb_va=pr_nb_va,
        qv5_va=qv5_va,
        pr_v5_va=pr_v5_va,
        blur_cfg=(c_bl33.alpha_, c_bl33.best_topk_, c_bl33.best_l2_, p0_v5),
    )


def _summarize_all4(tag: str, Q: np.ndarray, P: np.ndarray, gold: np.ndarray, is_orig: np.ndarray) -> dict:
    sc_all = c.score(P, gold)
    sc_o = c.score(P[is_orig], gold[is_orig])
    rps_o = float(np.mean([dc.compute_dim_rps(Q[is_orig, d], gold[is_orig, d]) for d in range(6)]))
    rps_a = float(np.mean([dc.compute_dim_rps(Q[:, d], gold[:, d]) for d in range(6)]))
    per_dim = {}
    parts = []
    for d, nm in ((0, "模糊"), (1, "纹理损失"), (2, "过锐"), (4, "过度生成")):
        fo = dc.compute_dim_grade_f1s(P[is_orig, d], gold[is_orig, d])
        fa = dc.compute_dim_grade_f1s(P[:, d], gold[:, d])
        ro = dc.compute_dim_rps(Q[is_orig, d], gold[is_orig, d])
        ra = dc.compute_dim_rps(Q[:, d], gold[:, d])
        per_dim[nm] = {"f1_orig": fo, "f1_all": fa, "rps_orig": ro, "rps_all": ra}
        parts.append(f"{nm[:2]}:{fo['P0']:.3f}/{fo['P1']:.3f}/{fo['P2']:.3f}({ro:.4f})")
    print(
        f"  {tag:<30} | ORIG={sc_o['doc2_strict_total']:.3f} "
        f"(Pos={sc_o['doc2_S_pos']:.3f} P0={sc_o['doc2_S_p0']:.3f} Cln={sc_o['doc2_Clean2']:.2f}) | "
        f"Doc1_o={sc_o['doc1_total']:.3f} | ALL={sc_all['doc2_strict_total']:.3f} | RPS_o={rps_o:.5f}"
    )
    print("      " + " | ".join(parts), flush=True)
    return {
        "variant": tag,
        "doc2_orig": float(sc_o["doc2_strict_total"]),
        "doc1_orig": float(sc_o["doc1_total"]),
        "doc2_all": float(sc_all["doc2_strict_total"]),
        "orig_parts": {
            "pos": float(sc_o["doc2_S_pos"]),
            "p0": float(sc_o["doc2_S_p0"]),
            "clean": float(sc_o["doc2_Clean2"]),
        },
        "rps_orig": rps_o,
        "rps_all": rps_a,
        "per_dim": per_dim,
    }


def main():
    t0 = time.time()
    import upgrade_deconf_v5_boundary as canonical_mod

    set_name = "resident_v7"
    members = SETS[set_name]
    nobk_members = [m for m in members if m != "pe_bkseg1"]
    feat_names = blur33_feat_names(nobk_members)
    names, gold, fold, _, P_dict = oof_pool(members)
    _, _, _, groups = c.load_split()
    sm = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    is_orig = np.array([sm[str(n)] == "orig" for n in names], dtype=bool)
    N = len(names)

    S_oof = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz"), names, "feats")
    T_oof = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_oof_3573.npz"), names, "stats")

    proto_cfg = dc._build_routed_gate_cfg(members)
    for g_dict in (_G, uob._G):
        g_dict.update(
            fold=fold,
            gold=gold,
            is_orig=is_orig,
            groups=groups,
            P=P_dict,
            S=S_oof,
            T=T_oof,
            members=members,
            nobk_members=nobk_members,
            proto_cfg=proto_cfg,
        )

    print(f"=== [deconf_v5_boundary] Zero-Leakage 8-Fold Nested OOF Verification (N={N}) ===", flush=True)
    with mp.get_context("fork").Pool(8) as pool:
        f_res = sorted(pool.map(_fold_worker_v5, list(range(8))), key=lambda d: d["k"])

    Q_v2_oof, pr_v2_oof = np.zeros((N, 6, 4)), np.zeros((N, 6), dtype=np.int64)
    Q_v4_oof, pr_v4_oof = np.zeros((N, 6, 4)), np.zeros((N, 6), dtype=np.int64)
    Q_nb_oof, pr_nb_oof = np.zeros((N, 6, 4)), np.zeros((N, 6), dtype=np.int64)
    Q_v5_oof, pr_v5_oof = np.zeros((N, 6, 4)), np.zeros((N, 6), dtype=np.int64)
    for r in f_res:
        va = fold == r["k"]
        Q_v2_oof[va], pr_v2_oof[va] = r["qv2_va"], r["pr_v2_va"]
        Q_v4_oof[va], pr_v4_oof[va] = r["qv4_va"], r["pr_v4_va"]
        Q_nb_oof[va], pr_nb_oof[va] = r["qnb_va"], r["pr_nb_va"]
        Q_v5_oof[va], pr_v5_oof[va] = r["qv5_va"], r["pr_v5_va"]

    rep_v2 = _summarize_all4("deconf_v2 (54.44)", Q_v2_oof, pr_v2_oof, gold, is_orig)
    rep_v4 = _summarize_all4("deconf_v4_tile (54.68)", Q_v4_oof, pr_v4_oof, gold, is_orig)
    rep_nb = _summarize_all4("v4_tile + NB_Tex only", Q_nb_oof, pr_nb_oof, gold, is_orig)
    rep_v5 = _summarize_all4("deconf_v5_boundary (NB+Blur)", Q_v5_oof, pr_v5_oof, gold, is_orig)
    print("  Fold blur cfgs (alpha, topk, l2, p0_og):", [r["blur_cfg"] for r in f_res], flush=True)

    # Fit full-OOF BoundaryDeconfV5Bundle on all 3,573 OOF rows
    v4_pkl = os.path.join(REPO, "results/r55/final/resident_v7_decoder_deconf_v4_tile.pkl")
    bnd_v4: v4m.TileSalienceDeconfV4Bundle = pickle.load(open(v4_pkl, "rb"))["decoder"]

    P_oof_with_tile = dict(P_dict)
    P_oof_with_tile["_spatial_20d"] = S_oof
    P_oof_with_tile["_tile_peaks_6x12"] = T_oof
    Q_full_v4 = bnd_v4.calibrate_probs(P_oof_with_tile)

    P_oof_nobk = {m: P_dict[m] for m in nobk_members}
    dec_nobk_full = red.RoutedGateEUDecoder(**dc._build_routed_gate_cfg(nobk_members)).fit(
        P_oof_nobk, gold, is_orig=is_orig, groups=groups
    )

    X_blur_full = canonical_mod.extract_blur33_tile_covs(Q_full_v4, P_dict, nobk_members, T_oof)
    cal_blur33_full = dc.LowRankOrdinalCalibrator(dim=0, mode="deconf", topk_grid=(4, 8, 16, 33)).fit(
        X_blur_full, Q_full_v4[:, 0], gold[:, 0], groups, is_orig=is_orig
    )

    p0_og_v5, st_v5_full = v3m.select_p0_og_inner_cv(
        Q_v5_oof, gold, groups, is_orig, lam_cl=float(bnd_v4.lam_cl), p0_base=float(bnd_v4.p0_base)
    )

    print(
        f"\n[Full OOF Fit] cal_blur33: alpha={cal_blur33_full.alpha_} (topk={cal_blur33_full.best_topk_}, "
        f"l2={cal_blur33_full.best_l2_}) | p0_og={p0_og_v5}",
        flush=True,
    )
    for ci, wi in zip(cal_blur33_full.sel_cols_, cal_blur33_full.w_):
        if abs(wi) >= 0.010:
            print(f"    cal_blur33 feat {feat_names[ci]:<28}: w = {wi:+.4f}")

    # Inspect router weights on dim=1 (纹理损失) before vs after dropping pe_bkseg1
    bms_11 = bnd_v4.base_decoder.base_members + ["p12_pool"]
    bms_10 = dec_nobk_full.base_members + ["p12_pool"]
    print("\n[Full OOF Fit] 纹理损失 (dim=1) Router Weights (10-member NB_Tex vs 11-member old):")
    print("    OLD (11m):", {m: round(float(w), 4) for m, w in zip(bms_11, bnd_v4.base_decoder.W[1])})
    print("    NEW (10m):", {m: round(float(w), 4) for m, w in zip(bms_10, dec_nobk_full.W[1])})

    bnd_v5 = canonical_mod.BoundaryDeconfV5Bundle(
        v4_bundle=copy.deepcopy(bnd_v4),
        dec_nobk=dec_nobk_full,
        nobk_members=nobk_members,
        cal_blur33=cal_blur33_full,
        lam_cl=float(bnd_v4.lam_cl),
        p0_base=float(bnd_v4.p0_base),
        p0_og=p0_og_v5,
        st_v5=st_v5_full,
    )

    out_pkl = os.path.join(REPO, "results/r55/final/resident_v7_decoder_deconf_v5_boundary.pkl")
    out_meta = os.path.join(REPO, "results/r55/final/resident_v7_decoder_deconf_v5_boundary.json")
    pickle.dump({"decoder": bnd_v5, "members": members, "rule": "deconf_v5_boundary"}, open(out_pkl, "wb"))

    meta = {
        "set": set_name,
        "rule": "deconf_v5_boundary",
        "clean_image_locked_to_v4_and_v2": True,
        "dims_locked_to_v4_tile": [2, 3, 4, 5],
        "upgraded_dims": [0, 1],
        "eu_lam_cl": float(bnd_v5.lam_cl),
        "eu_p0_base": float(bnd_v5.p0_base),
        "eu_p0_og": float(bnd_v5.p0_og),
        "nb_tex_weights_dim1": {m: round(float(w), 6) for m, w in zip(bms_10, dec_nobk_full.W[1])},
        "cal_blur33": {
            "alpha": float(cal_blur33_full.alpha_),
            "l2": float(cal_blur33_full.best_l2_),
            "topk": int(cal_blur33_full.best_topk_),
            "weights": {feat_names[ci]: round(float(wi), 6) for ci, wi in zip(cal_blur33_full.sel_cols_, cal_blur33_full.w_)},
        },
        "nested_oof_v2": rep_v2,
        "nested_oof_v4_tile": rep_v4,
        "nested_oof_nb_tex_only": rep_nb,
        "nested_oof_v5_boundary": rep_v5,
    }
    json.dump(meta, open(out_meta, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # Verify clean unpickling via `final_engine.load_decoder`
    dec_fn, _ = load_decoder("resident_v7", "deconf_v5_boundary")

    # Decode test_150 & final_644 with matched tile features and verify N=1 == N=Batch + locks
    z_150 = np.load(os.path.join(REPO, "results/r55/os_og_breakthrough/pred_nriqa_dataset_test_os_og_breakthrough_member_probs.npz"), allow_pickle=True)
    z_644 = np.load(os.path.join(REPO, "results/r55/os_og_breakthrough_644/pred_nriqa_dataset_final_644_os_og_breakthrough_member_probs.npz"), allow_pickle=True)
    nm_150 = [str(x) for x in z_150["names"]]
    nm_644 = [str(x) for x in z_644["names"]]

    P_150 = {m: z_150[f"p_{m}"].astype(np.float64) if f"p_{m}" in z_150.files else z_150[m].astype(np.float64) for m in members}
    P_150["_spatial_20d"] = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_target.npz"), nm_150, "feats")
    P_150["_tile_peaks_6x12"] = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_test150.npz"), nm_150, "stats")

    P_644 = {m: z_644[f"p_{m}"].astype(np.float64) if f"p_{m}" in z_644.files else z_644[m].astype(np.float64) for m in members}
    P_644["_spatial_20d"] = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough_644/spatial_cache/spatial_final_644.npz"), nm_644, "feats")
    P_644["_tile_peaks_6x12"] = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_final644.npz"), nm_644, "stats")

    pr_v4_150, q_v4_150 = bnd_v4.predict(P_150)
    pr_v5_150, q_v5_150 = dec_fn(P_150)
    assert np.array_equal((pr_v4_150 == 0).all(1), (pr_v5_150 == 0).all(1)), "Clean image set changed on test_150!"
    assert np.array_equal(pr_v4_150[:, [2, 3, 4, 5]], pr_v5_150[:, [2, 3, 4, 5]]), "Locked dims changed on test_150!"
    for i in range(len(nm_150)):
        p1, q1 = dec_fn({k: v[i : i + 1] for k, v in P_150.items()})
        assert np.array_equal(p1[0], pr_v5_150[i]) and np.max(np.abs(q1[0] - q_v5_150[i])) < 1e-12

    pr_v4_644, q_v4_644 = bnd_v4.predict(P_644)
    pr_v5_644, q_v5_644 = dec_fn(P_644)
    assert np.array_equal((pr_v4_644 == 0).all(1), (pr_v5_644 == 0).all(1)), "Clean image set changed on final_644!"
    assert np.array_equal(pr_v4_644[:, [2, 3, 4, 5]], pr_v5_644[:, [2, 3, 4, 5]]), "Locked dims changed on final_644!"
    for i in range(len(nm_644)):
        p1, q1 = dec_fn({k: v[i : i + 1] for k, v in P_644.items()})
        assert np.array_equal(p1[0], pr_v5_644[i]) and np.max(np.abs(q1[0] - q_v5_644[i])) < 1e-12

    # Fresh 11-model test_150 prob parity check (if verify_test150_attr_off_probs.npz exists)
    fresh_npz = os.path.join(REPO, "results/final/verify_test150_attr_off_probs.npz")
    if os.path.exists(fresh_npz):
        zf = np.load(fresh_npz, allow_pickle=True)
        nm_f = [str(x) for x in zf["names"]]
        idx_f = {n: i for i, n in enumerate(nm_f)}
        P_f = {m: np.stack([zf[m][idx_f[n]] for n in nm_150], axis=0).astype(np.float64) for m in members}
        P_f["_spatial_20d"] = P_150["_spatial_20d"]
        P_f["_tile_peaks_6x12"] = P_150["_tile_peaks_6x12"]
        pr_f, _ = dec_fn(P_f)
        n_diff_f = int((pr_f != pr_v5_150).sum())
        print(f"\n[Fresh-Prob Parity Check] vs verify_test150_attr_off_probs.npz: {n_diff_f} / 900 cell diff", flush=True)

    out_150_a = os.path.join(REPO, "results/final/pred_nriqa_dataset_test_r55_resident_v7_deconf_v5_boundary.json")
    out_644_a = os.path.join(REPO, "results/final/pred_nriqa_dataset_final_644_r55_resident_v7_deconf_v5_boundary.json")
    sprint_01 = os.path.join(REPO, "final_sprint_delivery/01_核心预测结果JSON")
    sprint_03 = os.path.join(REPO, "final_sprint_delivery/03_最后冲刺核心代码与脚本")
    sprint_04 = os.path.join(REPO, "final_sprint_delivery/04_最终校准解码器与实测审计账本")
    out_150_b = os.path.join(sprint_01, "pred_nriqa_dataset_test_r55_resident_v7_deconf_v5_boundary.json")
    out_644_b = os.path.join(sprint_01, "pred_nriqa_dataset_final_644_r55_resident_v7_deconf_v5_boundary.json")

    j150_obj = c.format_official_submission(nm_150, pr_v5_150)[0]
    j644_obj = c.format_official_submission(nm_644, pr_v5_644)[0]
    for p_dst, obj in ((out_150_a, j150_obj), (out_150_b, j150_obj), (out_644_a, j644_obj), (out_644_b, j644_obj)):
        os.makedirs(os.path.dirname(p_dst), exist_ok=True)
        json.dump(obj, open(p_dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if os.path.isdir(sprint_03):
        for fn in ("upgrade_deconf_v5_boundary.py", "upgrade_deconf_v4_tile.py", "final_engine.py", "serve_engine.py"):
            shutil.copy2(os.path.join(REPO, "code/r55", fn), os.path.join(sprint_03, fn))
    if os.path.isdir(sprint_04):
        shutil.copy2(out_pkl, os.path.join(sprint_04, "resident_v7_decoder_deconf_v5_boundary.pkl"))
        shutil.copy2(out_meta, os.path.join(sprint_04, "resident_v7_decoder_deconf_v5_boundary.json"))

    validate_prediction_file(out_150_b, os.path.join(REPO, "results/v2_full/template_150.json"))
    validate_prediction_file(out_644_b, os.path.join(REPO, "results/final/template_final_644.json"))

    print("\n--- Exact Diff vs 54.68 (deconf_v4_tile) ---")
    d_150 = np.where(pr_v4_150 != pr_v5_150)
    print(
        f"  test_150 changed: {len(d_150[0])} / 900 cells "
        f"(Clean images: {(pr_v4_150 == 0).all(1).sum()} -> {(pr_v5_150 == 0).all(1).sum()}, 100% same mask!)"
    )
    for i, d in zip(*d_150):
        print(
            f"    {nm_150[i]:<52} {DIMS_ZH[d]}: {GRADES_ZH[pr_v4_150[i, d]]} -> {GRADES_ZH[pr_v5_150[i, d]]}  "
            f"v4={np.round(q_v4_150[i, d], 3).tolist()} -> v5={np.round(q_v5_150[i, d], 3).tolist()}"
        )

    d_644 = np.where(pr_v4_644 != pr_v5_644)
    print(
        f"\n  final_644 changed: {len(d_644[0])} / 3864 cells "
        f"(Clean images: {(pr_v4_644 == 0).all(1).sum()} -> {(pr_v5_644 == 0).all(1).sum()}, 100% same mask!)"
    )
    from collections import Counter
    cnt_644 = Counter((DIMS_ZH[d], f"{GRADES_ZH[pr_v4_644[i, d]]}->{GRADES_ZH[pr_v5_644[i, d]]}") for i, d in zip(*d_644))
    for (dm, tr_s), cnt in cnt_644.most_common():
        print(f"    {dm:<5} {tr_s:<10}: {cnt:3d}")

    print(f"\n=== ALL DONE in {time.time() - t0:.1f}s -> {out_150_b} ===", flush=True)


if __name__ == "__main__":
    main()
