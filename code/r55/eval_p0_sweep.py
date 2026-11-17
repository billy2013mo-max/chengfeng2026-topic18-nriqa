#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_p0_sweep.py
=========================
Evaluates all completed P0 fixed-epoch schedules E in {3, 4, 5, 8} (plus val-best reference)
for each Tier-1 backbone (`cradio_v4_h`, `pe_core_large`, `siglip2_so400m_naflex`),
both standalone (RPS, ECE) and inside the honest `fallback` (no VLM) and `primary` ensembles
under `DecoderEU` and `DecoderEU2 (eu+cal)`.
"""
from __future__ import annotations

import csv
import glob
import itertools
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    if _p not in sys.path:
        sys.path.insert(0, os.path.join(REPO, _p))

import common55 as c
import decoder_eu as E
import decoder_eu2 as E2
import final_engine as fe
from qingyu.certainty_gated_stacker import compute_ece_6x4


def rps_6x4(P: np.ndarray, Y: np.ndarray) -> float:
    oh = np.eye(4)[Y]
    return float(np.mean((np.cumsum(P, -1)[..., :3] - np.cumsum(oh, -1)[..., :3]) ** 2))


def main():
    names, gold, fold, _ = c.load_split()
    idx = {n: i for i, n in enumerate(names)}
    split_map = {row["name"]: row["split"] for row in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    orig_mask = np.array([split_map[n] == "orig" for n in names])

    def load_dir(d):
        arr = np.full((len(names), 6, 4), np.nan, dtype=np.float64)
        for f in range(8):
            fp = os.path.join(REPO, d, f"fold_{f}_probs.npz")
            if not os.path.exists(fp):
                return None
            z = np.load(fp, allow_pickle=True)
            for n, p in zip(z["va_names"], z["va_probs_6x4"]):
                arr[idx[str(n)]] = p
        return None if np.isnan(arr).any() else arr

    candidates = {
        "E3": "results/next55/towers/r55_fixed_e3_s2026",
        "E4": "results/next55/towers/r55_fixed_e4_s2026",
        "E5": "results/next55/towers/r55_fixed_e5_s2026",
        "E8": "results/next55/towers/r55_fixed_s2026",
        "val": "results/qingyu",
    }
    bbs = [("cradio", "cradio_v4_h"), ("pe", "pe_core_large"), ("sig", "siglip2_so400m_naflex")]

    loaded = {m: {} for m, _ in bbs}
    print("=== [P0 Step 1] Standalone Per-Backbone Fixed-Epoch Comparison ===")
    for m, bb in bbs:
        for tag, root in candidates.items():
            d = f"{root}/{bb}" if tag != "val" else f"{root}/{bb}"
            arr = load_dir(d)
            if arr is None:
                continue
            loaded[m][tag] = arr
            ece = compute_ece_6x4(arr, gold)
            print(f"  {bb:24s} | {tag:4s} | RPS_all={rps_6x4(arr, gold):.5f} RPS_orig={rps_6x4(arr[orig_mask], gold[orig_mask]):.5f} | ECE={ece:.4f}")

    # Load honest v16_qy + phoenix
    _, _, _, _, P_base = fe.oof_pool(["cradio", "pe", "sig", "v16", "p1", "p3"])
    P_v16 = np.zeros((len(names), 6, 4))
    for f in sorted(glob.glob(os.path.join(REPO, "results/r55/v16_qy/V16_Full_F*_probs.npz"))):
        z = np.load(f, allow_pickle=True)
        for n, v in zip(z["va_names"], c.p012_to_6x4_simplex(z["va_p0"], z["va_p1"], z["va_p2"])):
            P_v16[idx[str(n)]] = v
    P_base["v16"] = P_v16

    fixed_tags = [t for t in ("E3", "E4", "E5", "E8") if all(t in loaded[m] for m, _ in bbs)]
    print("\n=== [P0 Step 2] Ensemble Grid Search Over Fixed Epoch Combos (cradio, pe, sig) ===")
    rows_out = []
    F_dummy = np.zeros((len(names), 1))
    for tc, tp, ts in itertools.product(fixed_tags, repeat=3):
        P_cur = dict(P_base)
        P_cur["cradio"] = loaded["cradio"][tc]
        P_cur["pe"] = loaded["pe"][tp]
        P_cur["sig"] = loaded["sig"][ts]
        Q_fb = fe.pool(P_cur, fe.SETS["fallback"])
        rps_a = rps_6x4(Q_fb, gold)
        rps_o = rps_6x4(Q_fb[orig_mask], gold[orig_mask])
        rows_out.append((rps_a, rps_o, tc, tp, ts))

    rows_out.sort(key=lambda x: x[0])
    for rps_a, rps_o, tc, tp, ts in rows_out[:8]:
        P_cur = dict(P_base)
        P_cur["cradio"], P_cur["pe"], P_cur["sig"] = loaded["cradio"][tc], loaded["pe"][tp], loaded["sig"][ts]
        for sname in ("fallback", "primary"):
            Q = fe.pool(P_cur, fe.SETS[sname])
            pred_eu = np.zeros_like(gold)
            pred_cal = np.zeros_like(gold)
            for k in range(8):
                tr, va = fold != k, fold == k
                pred_eu[va] = E.DecoderEU().fit(Q[tr], gold[tr]).predict(Q[va])
                dec2 = E2.EU2(E2.LAMS_OLD, cal=True, clean=False).fit(Q[tr], F_dummy[tr], gold[tr])
                Qva = E2.apply_temp(c.renorm(Q[va]), dec2.T)
                pred_cal[va] = dec2._pred(Qva, np.prod(Qva[:, :, 0], 1), dec2.st, dec2.lam)
            sa = c.score(pred_eu, gold)["doc2_strict_total"]
            so = c.score(pred_eu[orig_mask], gold[orig_mask])["doc2_strict_total"]
            sca = c.score(pred_cal, gold)["doc2_strict_total"]
            sco = c.score(pred_cal[orig_mask], gold[orig_mask])["doc2_strict_total"]
            print(f"  ({tc},{tp},{ts}) [{sname:8s}] | RPS_all={rps_6x4(Q, gold):.5f} RPS_orig={rps_6x4(Q[orig_mask], gold[orig_mask]):.5f} "
                  f"| EU all={sa:.3f} orig={so:.3f} | EU+cal all={sca:.3f} orig={sco:.3f}")


if __name__ == "__main__":
    main()
