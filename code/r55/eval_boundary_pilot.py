#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_boundary_pilot.py
===============================
Honest Zero-Leakage Pilot (2-Fold) & Full 8-Fold Evaluator for Stage 2B BoundaryTileSpecialists.

Evaluates candidates against the exact `resident_v7 + deconf_v5_boundary` production baseline,
measuring:
  - Standalone boundary discrimination (`RPS_o`, `PosRPS_o`, `mAP12_o`, `AUC(>=P1|pos)`, `AUC(P0|>=P1)`)
    across ALL 4 graded dimensions (`模糊, 纹理损失, 过锐, 过度生成`) vs all 11 `resident_v7` members
    + `iv6b_native` + `qwen3vl_hires`.
  - Zero-leakage ensemble blending (linear & log-linear) onto `Q_v5` (using strictly out-of-fold
    temperature & `st_v5` EU parameters from `fold != k`), plus synergy with `iv6b_native` / `qwen3vl_hires`.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

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
import upgrade_deconf_v5_boundary as v5m  # noqa: E402
import upgrade_os_og_bokeh_deconf as uob  # noqa: E402
from eval_specialist_gate import load_oof_dir  # noqa: E402
from final_engine import SETS, oof_pool  # noqa: E402

GRADED_DIMS = (0, 1, 2, 4)
SHORT_ZH = {0: "模糊", 1: "纹理", 2: "过锐", 4: "过生"}


def load_or_build_v5_oof(cache_path: str = "/tmp/v5_nested_oof_cache.npz") -> Dict[str, np.ndarray]:
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    members = SETS["resident_v7"]
    nobk_members = [m for m in members if m != "pe_bkseg1"]
    names, gold, fold, _, P_dict = oof_pool(members)
    _, _, _, groups = c.load_split()
    split_map = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    is_orig = np.array([split_map[n] == "orig" for n in names], dtype=bool)
    S_all = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz"), names, "feats")
    T_all = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_oof_3573.npz"), names, "stats")
    proto_cfg = dc._build_routed_gate_cfg(members)

    for g_dict in (uob._G, v5m._G):
        g_dict.update(
            fold=fold, gold=gold, is_orig=is_orig, groups=groups, P=P_dict,
            S=S_all, T=T_all, members=members, nobk_members=nobk_members, proto_cfg=proto_cfg,
        )
    with mp.get_context("fork").Pool(8) as pool:
        res_list = pool.map(v5m._fold_worker_v5, list(range(8)))

    Q_v5 = np.zeros((len(names), 6, 4), dtype=np.float64)
    pr_v5 = np.zeros_like(gold)
    pr_v4 = np.zeros_like(gold)
    for r in res_list:
        va = fold == r["k"]
        Q_v5[va] = r["qv5_va"]
        pr_v5[va] = r["pr_v5_va"]
        pr_v4[va] = r["pr_v4_va"]

    np.savez_compressed(
        cache_path,
        Q_v5=Q_v5.astype(np.float64),
        pr_v5=pr_v5.astype(np.int64),
        pr_v4=pr_v4.astype(np.int64),
        is_orig=is_orig,
    )
    return {"Q_v5": Q_v5, "pr_v5": pr_v5, "pr_v4": pr_v4, "is_orig": is_orig}


def rps_6x4(P: np.ndarray, Y: np.ndarray) -> float:
    oh = np.eye(4)[Y]
    return float(((np.cumsum(P, -1)[..., :3] - np.cumsum(oh, -1)[..., :3]) ** 2).mean())


def pos_rps_graded(P: np.ndarray, Y: np.ndarray) -> float:
    oh = np.eye(4)[Y]
    err = ((np.cumsum(P, -1)[..., :3] - np.cumsum(oh, -1)[..., :3]) ** 2).mean(axis=-1)  # [N, 6]
    vals = [float(err[Y[:, d] >= 1, d].mean()) for d in GRADED_DIMS if (Y[:, d] >= 1).any()]
    return float(np.mean(vals))


def boundary_summary(P: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    out = {"rps": rps_6x4(P, Y), "pos_rps": pos_rps_graded(P, Y)}
    aps, a_ge1, a_p0 = [], [], []
    oh = np.eye(4)[Y]
    for d in GRADED_DIMS:
        out[f"rps_{d}"] = float(((np.cumsum(P[:, d], -1)[:, :3] - np.cumsum(oh[:, d], -1)[:, :3]) ** 2).mean())
        for g in (1, 2, 3):
            yb = (Y[:, d] == g).astype(int)
            if 0 < yb.sum() < len(yb):
                ap = float(average_precision_score(yb, P[:, d, g]))
                aps.append(ap)
                out[f"ap_{d}_{g}"] = ap
        m1 = Y[:, d] >= 1
        y1 = (Y[m1, d] >= 2).astype(int)
        if 0 < y1.sum() < len(y1):
            sc1 = (P[m1, d, 2] + P[m1, d, 3]) / np.clip(1.0 - P[m1, d, 0], 1e-6, 1.0)
            v1 = float(roc_auc_score(y1, sc1))
            out[f"auc1_{d}"] = v1
            a_ge1.append(v1)
        m2 = Y[:, d] >= 2
        y2 = (Y[m2, d] == 3).astype(int)
        if 0 < y2.sum() < len(y2):
            sc2 = P[m2, d, 3] / np.clip(P[m2, d, 2] + P[m2, d, 3], 1e-6, 1.0)
            v2 = float(roc_auc_score(y2, sc2))
            out[f"auc2_{d}"] = v2
            a_p0.append(v2)
    out["mAP12"] = float(np.mean(aps))
    out["auc_ge1"] = float(np.mean(a_ge1))
    out["auc_p0"] = float(np.mean(a_p0))
    return out


def apply_temp(P: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(P, 1e-7, 1.0)) / float(T)
    z -= z.max(axis=-1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=-1, keepdims=True)
    return q


def fit_scalar_temp(P: np.ndarray, Y: np.ndarray) -> float:
    best_t, best_r = 1.0, 1e9
    for t in np.linspace(0.55, 2.50, 40):
        r = rps_6x4(apply_temp(P, t), Y)
        if r < best_r:
            best_r, best_t = r, float(t)
    return best_t


def per_grade_f1(pred: np.ndarray, gold: np.ndarray) -> Dict[int, Tuple[float, float, float]]:
    res = {}
    for d in GRADED_DIMS:
        f1s = []
        for g in (3, 2, 1):
            tp = int(((pred[:, d] == g) & (gold[:, d] == g)).sum())
            fp = int(((pred[:, d] == g) & (gold[:, d] != g)).sum())
            fn = int(((pred[:, d] != g) & (gold[:, d] == g)).sum())
            f1s.append(2.0 * tp / max(1, 2 * tp + fp + fn))
        res[d] = tuple(f1s)  # (P0, P1, P2)
    return res


def load_iv6b_and_qwen_hr(names: Sequence[str]) -> Dict[str, np.ndarray]:
    out = {}
    idx = {n: i for i, n in enumerate(names)}
    # iv6b_native
    p_iv, v_iv = load_oof_dir("results/r55/iv6b_native_pilot", list(names))
    if v_iv.all():
        out["iv6b_native"] = c.renorm(p_iv)
    # qwen3vl_hires
    qh_path = os.path.join(REPO, "results/r55/dualcore_fullfit_final/qwen3vl_hires_8fold_oof.npz")
    if os.path.exists(qh_path):
        z = np.load(qh_path, allow_pickle=True)
        mp = {str(n): q for n, q in zip(z["names"], z["probs_6x4"].astype(np.float64))}
        if all(n in mp for n in names):
            out["qwen3vl_hires"] = c.renorm(np.stack([mp[n] for n in names]))
    return out


def fit_dirdiag_1d(Q: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fits per-dim 5-param DirDiag (log_temp + 4 class biases) minimizing RPS + L2 reg."""
    from scipy.optimize import minimize
    N = len(Y)
    oh = np.eye(4)[Y]
    params = np.zeros((6, 5), dtype=np.float64)
    for d in range(6):
        lp = np.log(np.clip(Q[:, d, :], 1e-7, 1.0))
        yd = np.cumsum(oh[:, d, :], axis=-1)[:, :3]
        if d in GRADED_DIMS:
            def obj(th):
                sc = np.exp(np.clip(th[0], -1.5, 1.5))
                z = sc * lp + th[1:5][None, :]
                z -= z.max(axis=-1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=-1, keepdims=True)
                cp = np.cumsum(p, axis=-1)[:, :3]
                return float(np.mean((cp - yd) ** 2)) + 0.002 * float(np.sum(th ** 2))
            res = minimize(obj, np.zeros(5), method="BFGS", options={"maxiter": 40})
            params[d] = res.x
        else:
            def obj2(th):
                sc = np.exp(np.clip(th[0], -1.5, 1.5))
                z0 = sc * lp[:, 0] + th[1]
                z3 = sc * lp[:, 3] + th[2]
                m = np.maximum(z0, z3)
                p3 = np.exp(z3 - m) / (np.exp(z0 - m) + np.exp(z3 - m))
                return float(np.mean((p3 - (Y[:, d] == 3)) ** 2)) + 0.002 * float(np.sum(th ** 2))
            res = minimize(obj2, np.zeros(3), method="BFGS", options={"maxiter": 40})
            params[d, 0], params[d, 1], params[d, 4] = res.x[0], res.x[1], res.x[2]
    return params


def apply_dirdiag_1d(Q: np.ndarray, params: np.ndarray) -> np.ndarray:
    out = np.zeros_like(Q)
    for d in range(6):
        lp = np.log(np.clip(Q[:, d, :], 1e-7, 1.0))
        sc = np.exp(np.clip(params[d, 0], -1.5, 1.5))
        if d in GRADED_DIMS:
            z = sc * lp + params[d, 1:5][None, :]
            z -= z.max(axis=-1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=-1, keepdims=True)
            out[:, d, :] = p
        else:
            z0 = sc * lp[:, 0] + params[d, 1]
            z3 = sc * lp[:, 3] + params[d, 4]
            m = np.maximum(z0, z3)
            e0, e3 = np.exp(z0 - m), np.exp(z3 - m)
            out[:, d, 0] = e0 / (e0 + e3)
            out[:, d, 3] = e3 / (e0 + e3)
    return c.renorm(out)


def eval_nested_blend_on_folds(
    Q_v5: np.ndarray,
    pr_v4: np.ndarray,
    extra_list: List[np.ndarray],
    extra_weights: List[float],
    gold: np.ndarray,
    fold: np.ndarray,
    is_orig: np.ndarray,
    eval_folds: Sequence[int],
    log_linear: bool = False,
    lock_clean: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strictly leak-free evaluation on `eval_folds`:
      - For each fold k in `eval_folds`, fits candidate scalar temp strictly on `eval_folds \\ {k}`.
      - Blends onto `Q_v5` on both `tr_cand` and `va`, fits DirDiag strictly on `tr_cand`,
        and fits `st_k` on `Q_tr_aligned` (`Q_mix` on `tr_cand` when all 8 folds present, else `Q_v5` with DirDiag-aligned `q_mix_va`).
    """
    Q_out = Q_v5.copy()
    pr_out = np.zeros_like(gold)
    eval_set = set(eval_folds)
    w_sum = float(sum(extra_weights))

    def _mix(Q_base, P_cals):
        if log_linear:
            lp = (1.0 - w_sum) * np.log(np.clip(Q_base, 1e-7, 1.0))
            for w_m, q_m in zip(extra_weights, P_cals):
                lp = lp + w_m * np.log(np.clip(q_m, 1e-7, 1.0))
            lp -= lp.max(axis=-1, keepdims=True)
            qm = np.exp(lp)
            qm /= qm.sum(axis=-1, keepdims=True)
            return qm
        qm = (1.0 - w_sum) * Q_base
        for w_m, q_m in zip(extra_weights, P_cals):
            qm = qm + w_m * q_m
        return c.renorm(qm)

    for k in eval_folds:
        va = fold == k
        tr_all = ~va
        tr_cand = np.isin(fold, list(eval_set - {k}))

        cals_va, cals_tr = [], []
        for P_ext in extra_list:
            t_k = fit_scalar_temp(P_ext[tr_cand], gold[tr_cand]) if tr_cand.any() else 1.0
            cals_va.append(apply_temp(P_ext[va], t_k))
            if tr_cand.any():
                cals_tr.append(apply_temp(P_ext[tr_cand], t_k))

        q_mix_va = _mix(Q_v5[va], cals_va)
        if tr_cand.any():
            q_mix_tr = _mix(Q_v5[tr_cand], cals_tr)
            if len(eval_folds) == 8:
                dd = fit_dirdiag_1d(q_mix_tr, gold[tr_cand])
                q_mix_tr = apply_dirdiag_1d(q_mix_tr, dd)
                q_mix_va = apply_dirdiag_1d(q_mix_va, dd)
                rw = np.where(is_orig[tr_cand], 1.0, 0.25)
                st_k = red.TrueGateConstrainedEU._fit_stats(q_mix_tr, gold[tr_cand], rw, lam_cl=2.75, p0_scale=1.00)
            else:
                # Align marginal prior to Q_v5 scale using strictly tr_cand (other pilot fold)
                Q_tr_hybrid = Q_v5[tr_all].copy()
                # Also fit EU stats on Q_v5[tr_all] after aligning q_mix via 1-param temp on tr_cand
                t_mix = fit_scalar_temp(q_mix_tr, gold[tr_cand])
                q_mix_va = apply_temp(q_mix_va, t_mix)
                rw = np.where(is_orig[tr_all], 1.0, 0.25)
                st_k = red.TrueGateConstrainedEU._fit_stats(Q_tr_hybrid, gold[tr_all], rw, lam_cl=2.75, p0_scale=1.00)
        else:
            rw = np.where(is_orig[tr_all], 1.0, 0.25)
            st_k = red.TrueGateConstrainedEU._fit_stats(Q_v5[tr_all], gold[tr_all], rw, lam_cl=2.75, p0_scale=1.00)

        Q_out[va] = q_mix_va
        pr_raw = v3m.predict_og_p0_eu(q_mix_va, st_k, lam_cl=2.75, p0_base=1.00, p0_og=0.90)
        if lock_clean:
            cl_ref = (pr_v4[va] == 0).all(axis=1)
            pr_raw[cl_ref] = 0
            nc = (~cl_ref) & (pr_raw == 0).all(axis=1)
            if np.any(nc):
                pr_raw[nc] = pr_v4[va][nc]
        pr_out[va] = pr_raw

    return Q_out, pr_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", default=[], help="name=dir_path")
    args = ap.parse_args()

    members = SETS["resident_v7"]
    names, gold, fold, _, P_v7 = oof_pool(members)
    v5 = load_or_build_v5_oof()
    Q_v5, pr_v5, pr_v4, is_orig = v5["Q_v5"], v5["pr_v5"], v5["pr_v4"], v5["is_orig"]

    extra_existing = load_iv6b_and_qwen_hr(names)
    cands: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for spec in args.cand:
        lbl, path = spec.split("=", 1)
        arr, valid = load_oof_dir(path, names)
        cands[lbl] = (c.renorm(np.where(np.isnan(arr), 0.25, arr)), valid)

    if not cands:
        print("No --cand passed; exiting.")
        return

    # Intersection of valid folds across all passed candidates
    common_valid = np.ones(len(names), dtype=bool)
    for _, (_, v) in cands.items():
        common_valid &= v
    eval_folds = sorted({int(f) for f in fold[common_valid]})
    m_eval = np.isin(fold, eval_folds)
    m_orig = m_eval & is_orig
    print(f"=== Evaluating on Folds {eval_folds}: total={m_eval.sum()} | ORIG={m_orig.sum()} ===\n")

    # Table A: Standalone Comparison on ORIG subset of eval_folds
    print("--- [TABLE A] Standalone Boundary Quality on Held-Out ORIG Images ---")
    hdr = (
        f"{'Model':18s} | {'RPS_o':7s} {'PosRPS':7s} {'mAP12':6s} | "
        f"{'AUC>=1':6s} {'AUC_P0':6s} | "
        f"{'Blur(r/a1/a2)':17s} {'Tex(r/a1/a2)':17s} {'OS(r/a1/a2)':17s} {'OG(r/a1/a2)':17s}"
    )
    print(hdr)
    print("-" * len(hdr))

    stand_pool: List[Tuple[str, np.ndarray]] = [("deconf_v5(ens)", Q_v5)]
    for lbl, (arr, _) in cands.items():
        stand_pool.append((f"NEW:{lbl}", arr))
    for k in ("iv6b_native", "qwen3vl_hires"):
        if k in extra_existing:
            stand_pool.append((k, extra_existing[k]))
    for m in ("p1_qy", "p12_pe", "cradio_v3", "ntg_pe", "p3_qy", "v16", "sig_v3"):
        if m in P_v7:
            stand_pool.append((m, P_v7[m]))

    for lbl, P in stand_pool:
        s = boundary_summary(P[m_orig], gold[m_orig])
        cells = []
        for d in GRADED_DIMS:
            cells.append(f"{s[f'rps_{d}']:.3f}/{s.get(f'auc1_{d}',0):.2f}/{s.get(f'auc2_{d}',0):.2f}")
        print(
            f"{lbl:18s} | {s['rps']:.5f} {s['pos_rps']:.5f} {s['mAP12']:.4f} | "
            f"{s['auc_ge1']:.4f} {s['auc_p0']:.4f} | "
            f"{cells[0]:17s} {cells[1]:17s} {cells[2]:17s} {cells[3]:17s}"
        )

    # Table B: Zero-Leakage Ensemble Blending onto Q_v5
    print("\n--- [TABLE B] Zero-Leakage Blend onto `deconf_v5_boundary` (Held-Out ORIG) ---")
    sc_base = c.score(pr_v5[m_orig], gold[m_orig])
    sb_base = boundary_summary(Q_v5[m_orig], gold[m_orig])
    f1_base = per_grade_f1(pr_v5[m_orig], gold[m_orig])

    def _fmt_row(tag: str, Q: np.ndarray, pr: np.ndarray) -> str:
        sc = c.score(pr[m_orig], gold[m_orig])
        sb = boundary_summary(Q[m_orig], gold[m_orig])
        f1 = per_grade_f1(pr[m_orig], gold[m_orig])
        d_tot = sc["doc2_strict_total"] - sc_base["doc2_strict_total"]
        return (
            f"{tag:34s} | ORIG={sc['doc2_strict_total']:.3f} ({d_tot:+.3f}) "
            f"[Pos={sc['doc2_S_pos']:.2f} P0={sc['doc2_S_p0']:.2f} Cln={sc['doc2_Clean2']:.2f}] | "
            f"RPS={sb['rps']:.5f} PosR={sb['pos_rps']:.5f} mAP={sb['mAP12']:.4f} "
            f"A1/A2={sb['auc_ge1']:.4f}/{sb['auc_p0']:.4f} | "
            f"P0_F1(B/T/OS/OG)={f1[0][0]:.3f}/{f1[1][0]:.3f}/{f1[2][0]:.3f}/{f1[4][0]:.3f} "
            f"P1_F1={f1[0][1]:.3f}/{f1[1][1]:.3f}/{f1[2][1]:.3f}/{f1[4][1]:.3f}"
        )

    print(_fmt_row("BASE: deconf_v5_boundary", Q_v5, pr_v5))

    # Individual new candidates (linear & log-linear)
    for lbl, (arr, _) in cands.items():
        for mode_name, use_ll in (("lin", False), ("log", True)):
            for w in (0.12, 0.18, 0.25):
                Qm, prm = eval_nested_blend_on_folds(
                    Q_v5, pr_v4, [arr], [w], gold, fold, is_orig, eval_folds, log_linear=use_ll
                )
                print(_fmt_row(f"+{lbl} ({mode_name} w={w:.2f})", Qm, prm))

    # Multi-candidate combos
    cand_names = list(cands.keys())
    if len(cand_names) >= 2:
        print()
        # All pairs & all-3 equal weight
        all_arrs = [cands[k][0] for k in cand_names]
        for mode_name, use_ll in (("lin", False), ("log", True)):
            for tot_w in (0.24, 0.33):
                ws = [tot_w / len(all_arrs)] * len(all_arrs)
                Qm, prm = eval_nested_blend_on_folds(
                    Q_v5, pr_v4, all_arrs, ws, gold, fold, is_orig, eval_folds, log_linear=use_ll
                )
                print(_fmt_row(f"+ALL({len(all_arrs)}) ({mode_name} tot={tot_w:.2f})", Qm, prm))

        # Synergy with iv6b_native & qwen3vl_hires!
        if "iv6b_native" in extra_existing and "qwen3vl_hires" in extra_existing:
            print()
            for tag, ext_arrs, ws, use_ll in [
                ("Stage2A: +iv6b+qwen_hr (log 0.12x2)", [extra_existing["iv6b_native"], extra_existing["qwen3vl_hires"]], [0.12, 0.12], True),
                ("ALL_NEW + iv6b (log 0.08x3+0.12)", all_arrs + [extra_existing["iv6b_native"]], [0.08] * len(all_arrs) + [0.12], True),
                ("ALL_NEW + qwen_hr (log 0.08x3+0.12)", all_arrs + [extra_existing["qwen3vl_hires"]], [0.08] * len(all_arrs) + [0.12], True),
                ("ALL_NEW + iv6b + qwen_hr (log)", all_arrs + [extra_existing["iv6b_native"], extra_existing["qwen3vl_hires"]], [0.08] * len(all_arrs) + [0.10, 0.10], True),
            ]:
                Qm, prm = eval_nested_blend_on_folds(
                    Q_v5, pr_v4, ext_arrs, ws, gold, fold, is_orig, eval_folds, log_linear=use_ll
                )
                print(_fmt_row(tag, Qm, prm))


if __name__ == "__main__":
    main()
