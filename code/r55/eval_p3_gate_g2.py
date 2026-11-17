#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_p3_gate_g2.py
===========================
Pre-registered Gate G2 Evaluator for Stage 1A Honest 8-Fold `Pillar3` (`p3_qy`):
  1. Verifies zero sibling-frame leakage on `data/phoenix_splits/phoenix_qy8.csv` (`0 / 3,573` vs `100 / 3,573` in legacy 5-fold).
  2. Compares Arm A (`pillar3_cv5_oof.npz`, 5-epoch SWA) vs Arm B (`pillar3_stage2-rl2r_oof.npz`, +5 cool-down SWA epochs)
     both standalone and inside the `resident_v4` (`cradio_v3, pe_v3, peg, sig_v3, v16, p3_qy, p12_pe`) ensemble.
  3. Adopts Arm A by default unless Arm B lowers pooled ensemble RPS by `> 2 * SE` (video-cluster bootstrap).
  4. Unconditionally installs the winning honest 8-fold OOF to `results/r55/p3_qy_honest/pillar3_best_oof.npz`.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    if os.path.join(REPO, _p) not in sys.path:
        sys.path.insert(0, os.path.join(REPO, _p))

import common55 as c
import final_engine as fe
from eval_specialist_gate import rps_6x4


def _load_aligned_p3(npz_path: str, names: list[str]) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    mp = {str(n): q for n, q in zip(z["names"], z["probs_6x4"].astype(np.float64))}
    assert len(mp) == len(names), f"Size mismatch in {npz_path}: {len(mp)} vs {len(names)}"
    return c.renorm(np.stack([mp[n] for n in names], axis=0))


def _cluster_boot_se(q_b: np.ndarray, q_a: np.ndarray, gold: np.ndarray, groups: np.ndarray, n_boot: int = 4000, seed: int = 2026):
    cdf_b = np.cumsum(c.renorm(q_b), axis=-1)[..., :3]
    cdf_a = np.cumsum(c.renorm(q_a), axis=-1)[..., :3]
    cdf_y = (np.arange(3)[None, None, :] >= gold[..., None]).astype(np.float64)
    row_diff = np.mean((cdf_b - cdf_y) ** 2 - (cdf_a - cdf_y) ** 2, axis=(1, 2))

    ug, inv = np.unique(groups, return_inverse=True)
    G = len(ug)
    g_sum = np.bincount(inv, weights=row_diff, minlength=G)
    g_cnt = np.bincount(inv, minlength=G).astype(np.float64)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, G, size=(n_boot, G))
    W = np.zeros((n_boot, G), dtype=np.float64)
    np.add.at(W, (np.repeat(np.arange(n_boot), G), picks.ravel()), 1.0)
    diffs = (W @ g_sum) / np.maximum(1.0, W @ g_cnt)
    return float(np.mean(row_diff)), float(np.std(diffs, ddof=1)), [float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))]


def main() -> None:
    names, gold, fold, groups = c.load_split()

    # Verify zero sibling-frame leakage between phoenix_qy8.csv and qingyu_8fold.csv
    qy8_rows = {r["name"]: int(r["fold"]) for r in csv.DictReader(open(os.path.join(REPO, "data/phoenix_splits/phoenix_qy8.csv")))}
    fold_qy8 = np.array([qy8_rows[n] for n in names], dtype=np.int64)
    assert np.array_equal(fold_qy8, fold), "phoenix_qy8.csv fold assignment does not match qingyu_8fold.csv!"
    leaked_groups = 0
    for g in np.unique(groups):
        if len(np.unique(fold_qy8[groups == g])) > 1:
            leaked_groups += 1
    assert leaked_groups == 0, f"Found {leaked_groups} video groups straddling folds!"

    arm_a_path = os.path.join(REPO, "results/r55/p3_qy_s2026/pillar3_cv5_oof.npz")
    arm_b_path = os.path.join(REPO, "results/r55/p3_qy_s2026/pillar3_stage2-rl2r_oof.npz")
    p3_a = _load_aligned_p3(arm_a_path, names)
    p3_b = _load_aligned_p3(arm_b_path, names)
    p3_legacy5 = _load_aligned_p3(os.path.join(REPO, "results/phoenix/pillar3_stage2-rl2r_oof.npz"), names)

    base_others = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p12_pe"]
    _, _, _, _, P_base = fe.oof_pool(base_others)
    members_v4 = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "p12_pe"]

    P_arm_a = dict(P_base, p3_qy=p3_a)
    P_arm_b = dict(P_base, p3_qy=p3_b)
    q_ens_a = fe.pool(P_arm_a, members_v4)
    q_ens_b = fe.pool(P_arm_b, members_v4)

    delta_ens, se_ens, ci90_ens = _cluster_boot_se(q_ens_b, q_ens_a, gold, groups)
    t_stat = delta_ens / max(1e-12, se_ens)
    adopt_arm_b = bool(delta_ens < -2.0 * se_ens)
    winner = "stage2-rl2r" if adopt_arm_b else "cv5"
    src_oof = arm_b_path if adopt_arm_b else arm_a_path

    dst_dir = os.path.join(REPO, "results/r55/p3_qy_honest")
    os.makedirs(dst_dir, exist_ok=True)
    dst_oof = os.path.join(dst_dir, "pillar3_best_oof.npz")
    shutil.copyfile(src_oof, dst_oof)

    rep = {
        "zero_video_group_leakage": True,
        "leaked_video_groups": 0,
        "standalone_rps": {
            "arm_a_cv5_5ep": round(float(rps_6x4(p3_a, gold)), 6),
            "arm_b_stage2_5ep": round(float(rps_6x4(p3_b, gold)), 6),
            "legacy_leaky_5fold_ref": round(float(rps_6x4(p3_legacy5, gold)), 6),
        },
        "standalone_argmax_doc2_strict": {
            "arm_a_cv5_5ep": round(float(c.score(p3_a.argmax(-1), gold)["doc2_strict_total"]), 3),
            "arm_b_stage2_5ep": round(float(c.score(p3_b.argmax(-1), gold)["doc2_strict_total"]), 3),
        },
        "resident_v4_pooled_rps": {
            "arm_a_cv5_5ep": round(float(rps_6x4(q_ens_a, gold)), 6),
            "arm_b_stage2_5ep": round(float(rps_6x4(q_ens_b, gold)), 6),
            "delta_b_minus_a": round(delta_ens, 6),
            "cluster_boot_se": round(se_ens, 6),
            "z_score_b_minus_a": round(t_stat, 3),
            "ci90_b_minus_a": [round(x, 6) for x in ci90_ens],
        },
        "adopted_schedule": winner,
        "installed_oof_path": dst_oof,
    }
    json.dump(rep, open(os.path.join(dst_dir, "gate_g2_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
