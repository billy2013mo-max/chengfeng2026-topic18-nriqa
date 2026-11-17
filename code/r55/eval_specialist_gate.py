#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_specialist_gate.py
================================
Honest 8-Fold Video-Grouped OOF Screening & Ensemble Gate for Wave 2/3 Specialists.

Evaluates any candidate 8-fold OOF directory (`fold_0_probs.npz` .. `fold_7_probs.npz`,
or subset of folds if screening on partial folds):
  1. Standalone metrics: `RPS_all`, `RPS_orig`, `ECE`, `mAP14`, and per-defect `mAP` across
     `[模糊, 纹理损失, 过锐, 花屏, 过度生成, 虚化突兀]`.
  2. Ensemble-level gate when added to the simultaneously-resident pure-vision `P0-fallback`
     baseline (`cradio_e4`, `pe_e8`/`peg`, `sig_e8`, `v16_qy`, `p3`):
     - Pooled `RPS_all` & `RPS_orig` delta + 1,000x video-group cluster bootstrap 90% CI!
     - Nested 8-fold `DecoderEU` and `DecoderEU2 (EU+cal)` `Doc2_Strict` (`all_3573` & `orig_1902`).
     - Fold-nested per-defect regularized specialist routing (`NestedRoutedPool`: fits per-defect
       temperature + L2-shrunk simplex weights strictly on `tr = (fold != k)`, zero test/val leakage!).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    if _p not in sys.path:
        sys.path.insert(0, os.path.join(REPO, _p))

import common55 as c
import decoder_eu as E
import decoder_eu2 as E2
import final_engine as fe
from qingyu.certainty_gated_stacker import compute_ece_6x4

DIMS = ("模糊", "纹理", "过锐", "花屏", "过度生成", "虚化突兀")


def rps_per_sample(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    oh = np.eye(4)[Y]
    return ((np.cumsum(P, -1)[..., :3] - np.cumsum(oh, -1)[..., :3]) ** 2).mean(axis=(-2, -1))


def rps_6x4(P: np.ndarray, Y: np.ndarray) -> float:
    return float(rps_per_sample(P, Y).mean())


def per_dim_ap(P: np.ndarray, Y: np.ndarray) -> Tuple[List[float], float]:
    dim_aps = []
    all14 = []
    for d in range(6):
        gs = (1, 2, 3) if d in (0, 1, 2, 4) else (3,)
        sub = []
        for g in gs:
            ap = float(average_precision_score((Y[:, d] == g).astype(int), P[:, d, g]))
            sub.append(ap)
            all14.append(ap)
        dim_aps.append(float(np.mean(sub)))
    return dim_aps, float(np.mean(all14))


def cluster_bootstrap_rps_delta(
    P_new: np.ndarray, P_base: np.ndarray, Y: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 2026
) -> Tuple[float, float, float]:
    diff = rps_per_sample(P_new, Y) - rps_per_sample(P_base, Y)
    uniq_g, inv = np.unique(groups, return_inverse=True)
    g_sum = np.bincount(inv, weights=diff, minlength=len(uniq_g))
    g_cnt = np.bincount(inv, minlength=len(uniq_g)).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq_g), size=(n_boot, len(uniq_g)))
    boot_means = (g_sum[idx].sum(axis=1)) / np.maximum(1.0, g_cnt[idx].sum(axis=1))
    lo, hi = np.percentile(boot_means, [5.0, 95.0])
    return float(diff.mean()), float(lo), float(hi)


def fit_nested_routed_pool(
    P_list_tr: List[np.ndarray], Y_tr: np.ndarray, prior_w: np.ndarray, l2_reg: float = 0.015, t_max: float = 3.80
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits per-member scalar temperature `T_m` and per-defect simplex weights `W[6, M]`
    minimizing RPS + L2 shrinkage toward `prior_w` strictly inside training fold `tr`.
    """
    M = len(P_list_tr)
    temps = np.ones(M, dtype=np.float64)
    cal_tr = []
    n_t = max(2, int(round((t_max - 0.50) / 0.05)) + 1)
    t_grid = np.linspace(0.50, t_max, n_t)
    for m in range(M):
        lp = np.log(np.clip(P_list_tr[m], 1e-7, 1.0))
        best_t, best_r = 1.0, 1e9
        for t in t_grid:
            z = lp / t
            z -= z.max(axis=-1, keepdims=True)
            q = np.exp(z)
            q /= q.sum(axis=-1, keepdims=True)
            r = rps_6x4(q, Y_tr)
            if r < best_r:
                best_r, best_t = r, t
        temps[m] = best_t
        z = lp / best_t
        z -= z.max(axis=-1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=-1, keepdims=True)
        cal_tr.append(q)

    # Stack [M, N, 6, 4] -> cumulative [M, N, 6, 3]
    cum_M = np.stack([np.cumsum(q, axis=-1)[..., :3] for q in cal_tr], axis=0)
    oh_cum = np.cumsum(np.eye(4)[Y_tr], axis=-1)[..., :3]  # [N, 6, 3]

    W = np.zeros((6, M), dtype=np.float64)
    bounds = [(0.0, 1.0)] * M
    cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    for d in range(6):
        Cd = cum_M[:, :, d, :]  # [M, N, 3]
        Yd = oh_cum[:, d, :]    # [N, 3]

        def obj(w):
            pred_c = np.tensordot(w, Cd, axes=(0, 0))  # [N, 3]
            rps_d = float(np.mean((pred_c - Yd) ** 2))
            reg = l2_reg * float(np.sum((w - prior_w) ** 2))
            return rps_d + reg

        res = minimize(obj, prior_w.copy(), method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 80, "ftol": 1e-8})
        w_opt = np.clip(res.x, 0.0, 1.0)
        W[d] = w_opt / max(1e-9, w_opt.sum())
    return temps, W


def apply_nested_routed_pool(P_list: List[np.ndarray], temps: np.ndarray, W: np.ndarray) -> np.ndarray:
    cal = []
    for m, P in enumerate(P_list):
        z = np.log(np.clip(P, 1e-7, 1.0)) / temps[m]
        z -= z.max(axis=-1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=-1, keepdims=True)
        cal.append(q)
    stack = np.stack(cal, axis=0)  # [M, N, 6, 4]
    out = np.einsum("dm,mndg->ndg", W, stack)
    return c.renorm(out)


def load_oof_dir(path: str, names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    idx = {n: i for i, n in enumerate(names)}
    arr = np.full((len(names), 6, 4), np.nan, dtype=np.float64)
    present_folds = []
    for f in range(8):
        fp = os.path.join(REPO, path, f"fold_{f}_probs.npz") if not os.path.isabs(path) else os.path.join(path, f"fold_{f}_probs.npz")
        if not os.path.exists(fp):
            continue
        z = np.load(fp, allow_pickle=True)
        if "va_names" not in z:
            continue
        present_folds.append(f)
        for n, p in zip(z["va_names"], z["va_probs_6x4"]):
            arr[idx[str(n)]] = p
    valid = ~np.isnan(arr[:, 0, 0])
    return arr, valid


def load_p0_fallback_base(names: List[str]) -> Dict[str, np.ndarray]:
    idx = {n: i for i, n in enumerate(names)}
    _, _, _, _, P = fe.oof_pool(["cradio", "pe", "sig", "v16", "p3"])
    P_v16 = np.zeros((len(names), 6, 4))
    for f in sorted(glob.glob(os.path.join(REPO, "results/r55/v16_qy/V16_Full_F*_probs.npz"))):
        z = np.load(f, allow_pickle=True)
        for n, v in zip(z["va_names"], c.p012_to_6x4_simplex(z["va_p0"], z["va_p1"], z["va_p2"])):
            P_v16[idx[str(n)]] = v
    P["v16"] = P_v16
    # Use P0 optimal fixed schedules: cradio=E4, pe=E8, sig=E8
    c_e4, v_c = load_oof_dir("results/next55/towers/r55_fixed_e4_s2026/cradio_v4_h", names)
    p_e8, v_p = load_oof_dir("results/next55/towers/r55_fixed_s2026/pe_core_large", names)
    s_e8, v_s = load_oof_dir("results/next55/towers/r55_fixed_s2026/siglip2_so400m_naflex", names)
    assert v_c.all() and v_p.all() and v_s.all()
    P["cradio"] = c_e4
    P["pe"] = p_e8
    P["sig"] = s_e8
    return P


def eval_eu_nested(Q: np.ndarray, gold: np.ndarray, fold: np.ndarray, orig: np.ndarray) -> Tuple[float, float, float, float]:
    pred_eu = np.zeros_like(gold)
    pred_cal = np.zeros_like(gold)
    F_dummy = np.zeros((len(gold), 1))
    for k in range(8):
        tr, va = fold != k, fold == k
        pred_eu[va] = E.DecoderEU().fit(Q[tr], gold[tr]).predict(Q[va])
        dec2 = E2.EU2(E2.LAMS_OLD, cal=True, clean=False).fit(Q[tr], F_dummy[tr], gold[tr])
        Qva = E2.apply_temp(c.renorm(Q[va]), dec2.T)
        pred_cal[va] = dec2._pred(Qva, np.prod(Qva[:, :, 0], 1), dec2.st, dec2.lam)
    return (
        c.score(pred_eu, gold)["doc2_strict_total"],
        c.score(pred_eu[orig], gold[orig])["doc2_strict_total"],
        c.score(pred_cal, gold)["doc2_strict_total"],
        c.score(pred_cal[orig], gold[orig])["doc2_strict_total"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", default=[], help="label=path_to_fold_npz_dir (repeatable)")
    args = ap.parse_args()

    names, gold, fold, groups = c.load_split()
    split_map = {row["name"]: row["split"] for row in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    orig = np.array([split_map[n] == "orig" for n in names])

    P_base = load_p0_fallback_base(names)
    base_members = ["cradio", "pe", "sig", "v16", "p3"]
    Q_base = fe.pool(P_base, base_members)
    eu_a, eu_o, cal_a, cal_o = eval_eu_nested(Q_base, gold, fold, orig)
    d_aps_b, map14_b = per_dim_ap(Q_base, gold)
    print("=== [Baseline: P0-Fallback (NO VLM, 13.6GB resident)] ===")
    print(f"  RPS_all={rps_6x4(Q_base, gold):.5f} RPS_orig={rps_6x4(Q_base[orig], gold[orig]):.5f} | mAP14={map14_b:.4f} | "
          f"EU={eu_a:.3f}/{eu_o:.3f} | EU+cal={cal_a:.3f}/{cal_o:.3f}")
    print("  Per-Dim AP: " + " ".join(f"{dn}={v:.4f}" for dn, v in zip(DIMS, d_aps_b)))

    # Also test NestedRoutedPool on P0-Fallback alone!
    P_list_base = [P_base[m] for m in base_members]
    prior_base = fe.weights(base_members)
    Q_base_routed = np.zeros_like(Q_base)
    for k in range(8):
        tr, va = fold != k, fold == k
        temps, W = fit_nested_routed_pool([p[tr] for p in P_list_base], gold[tr], prior_base)
        Q_base_routed[va] = apply_nested_routed_pool([p[va] for p in P_list_base], temps, W)
    reu_a, reu_o, rcal_a, rcal_o = eval_eu_nested(Q_base_routed, gold, fold, orig)
    dm, lo, hi = cluster_bootstrap_rps_delta(Q_base_routed, Q_base, gold, groups)
    print(f"  [+NestedRoutedPool on P0-Fallback] RPS_all={rps_6x4(Q_base_routed, gold):.5f} (delta={dm:+.5f} CI90=[{lo:+.5f},{hi:+.5f}]) "
          f"| EU={reu_a:.3f}/{reu_o:.3f} | EU+cal={rcal_a:.3f}/{rcal_o:.3f}")

    loaded_cands: Dict[str, np.ndarray] = {}
    for spec in args.cand:
        label, path = spec.split("=", 1)
        arr, valid = load_oof_dir(path, names)
        n_val = int(valid.sum())
        d_aps, m14 = per_dim_ap(arr[valid], gold[valid])
        ece = compute_ece_6x4(arr[valid], gold[valid])
        vo = valid & orig
        print(f"\n=== [Candidate Standalone: {label} (N={n_val}/{len(names)})] ===")
        print(f"  RPS_all={rps_6x4(arr[valid], gold[valid]):.5f} RPS_orig={rps_6x4(arr[vo], gold[vo]):.5f} | mAP14={m14:.4f} | ECE={ece:.4f}")
        print("  Per-Dim AP: " + " ".join(f"{dn}={v:.4f}" for dn, v in zip(DIMS, d_aps)))

        if valid.all():
            loaded_cands[label] = arr
            for w in (0.12, 0.18, 0.24):
                Qb = c.renorm((1.0 - w) * Q_base + w * arr)
                dm, lo, hi = cluster_bootstrap_rps_delta(Qb, Q_base, gold, groups)
                ea, eo, ca, co = eval_eu_nested(Qb, gold, fold, orig)
                _, m14_ens = per_dim_ap(Qb, gold)
                print(f"  [+Blend w={w:.2f}] RPS={rps_6x4(Qb, gold):.5f} ({dm:+.5f} CI90=[{lo:+.5f},{hi:+.5f}]) mAP14={m14_ens:.4f} "
                      f"| EU={ea:.3f}/{eo:.3f} | EU+cal={ca:.3f}/{co:.3f}")

            # NestedRoutedPool with candidate added as specialist!
            P_list_c = P_list_base + [arr]
            prior_c = np.append(prior_base * 0.82, 0.18)
            Qr = np.zeros_like(Q_base)
            W_mean = np.zeros((6, len(P_list_c)))
            for k in range(8):
                tr, va = fold != k, fold == k
                temps, W = fit_nested_routed_pool([p[tr] for p in P_list_c], gold[tr], prior_c)
                W_mean += W / 8.0
                Qr[va] = apply_nested_routed_pool([p[va] for p in P_list_c], temps, W)
            dm, lo, hi = cluster_bootstrap_rps_delta(Qr, Q_base, gold, groups)
            ea, eo, ca, co = eval_eu_nested(Qr, gold, fold, orig)
            print(f"  [+NestedRouted] RPS={rps_6x4(Qr, gold):.5f} ({dm:+.5f} CI90=[{lo:+.5f},{hi:+.5f}]) "
                  f"| EU={ea:.3f}/{eo:.3f} | EU+cal={ca:.3f}/{co:.3f} | cand_w_by_dim=" +
                  ",".join(f"{DIMS[d]}:{W_mean[d, -1]:.2f}" for d in range(6)))

    if len(loaded_cands) >= 2:
        print(f"\n=== [Combined All {len(loaded_cands)} Candidates + P0-Fallback via NestedRoutedPool] ===")
        P_list_all = P_list_base + list(loaded_cands.values())
        k_c = len(loaded_cands)
        prior_all = np.concatenate([prior_base * 0.72, np.full(k_c, 0.28 / k_c)])
        Qr_all = np.zeros_like(Q_base)
        for k in range(8):
            tr, va = fold != k, fold == k
            temps, W = fit_nested_routed_pool([p[tr] for p in P_list_all], gold[tr], prior_all)
            Qr_all[va] = apply_nested_routed_pool([p[va] for p in P_list_all], temps, W)
        dm, lo, hi = cluster_bootstrap_rps_delta(Qr_all, Q_base, gold, groups)
        ea, eo, ca, co = eval_eu_nested(Qr_all, gold, fold, orig)
        d_aps_a, m14_a = per_dim_ap(Qr_all, gold)
        print(f"  RPS_all={rps_6x4(Qr_all, gold):.5f} RPS_orig={rps_6x4(Qr_all[orig], gold[orig]):.5f} ({dm:+.5f} CI90=[{lo:+.5f},{hi:+.5f}]) | mAP14={m14_a:.4f}")
        print(f"  EU={ea:.3f}/{eo:.3f} | EU+cal={ca:.3f}/{co:.3f}")
        print("  Per-Dim AP: " + " ".join(f"{dn}={v:.4f}" for dn, v in zip(DIMS, d_aps_a)))


if __name__ == "__main__":
    main()
