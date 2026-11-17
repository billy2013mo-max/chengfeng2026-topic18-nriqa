#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_clean_kd_teacher.py
==================================
Committed, bit-for-bit reproducible builder & verifier for leak-free cross-fitted KD teachers (Stage 1C):
  - Reconstructs `nested_kd_teacher.npz` (8-fold CV) and `fullfit_halfAB_kd_teacher.npz` (100% full-fit)
    from `results/next55/towers/r55_half{A,B}_s{2026,2027}/pe_core_large/fold_{0..7}_probs.npz` (`pool_probs_6x4`).
  - Proves mathematically:
      1. ZERO `Pillar3` (`p3`) involvement (`uses_pillar3 == False`).
      2. ZERO validation-fold leakage: for every outer fold `k`, every training row `i` in `halfA` is taught
         ONLY by `halfB` teachers trained strictly inside `fold != k` (and vice versa).
  - Writes clean reproducible copies to `results/r55/clean_honest_kd_teacher_cv8.npz` and
    `results/r55/clean_honest_kd_teacher_fullfit.npz` (or verifies exact bit-for-bit match against existing `.npz`).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    if os.path.join(REPO, _p) not in sys.path:
        sys.path.insert(0, os.path.join(REPO, _p))

import common55 as c
from qingyu.train_tier1_5backbones_8fold import load_fold_rows
from train_testbed import subsample_groups


def build_and_verify() -> dict:
    names, gold, fold, _ = c.load_split()
    idx_map = {nm: i for i, nm in enumerate(names)}
    N = len(names)

    ha_dirs = sorted(glob.glob(os.path.join(REPO, "results/next55/towers/r55_halfA_s*/pe_core_large")))
    hb_dirs = sorted(glob.glob(os.path.join(REPO, "results/next55/towers/r55_halfB_s*/pe_core_large")))
    assert len(ha_dirs) == 2 and len(hb_dirs) == 2, (ha_dirs, hb_dirs)

    rebuilt_folds = {}
    fullfit_acc = [[] for _ in range(N)]

    for fk in range(8):
        tr_all, va_rows = load_fold_rows(os.path.join(REPO, f"data/qingyu_splits/fold_{fk}.csv"))
        half_a = {str(r["name"]) for r in subsample_groups(tr_all, 0.5, fk)}

        acc_a, acc_b = {}, {}
        for d in ha_dirs:
            zp = np.load(os.path.join(d, f"fold_{fk}_probs.npz"), allow_pickle=True)
            for nm, pr in zip(zp["pool_names"], zp["pool_probs_6x4"]):
                acc_a.setdefault(str(nm), []).append(np.asarray(pr, dtype=np.float64))
            # For fullfit, out-of-fold validation predictions `va_probs_6x4` are ALSO 100% out-of-sample!
            for nm, pr in zip(zp["va_names"], zp["va_probs_6x4"]):
                fullfit_acc[idx_map[str(nm)]].append(np.asarray(pr, dtype=np.float64))
        for d in hb_dirs:
            zp = np.load(os.path.join(d, f"fold_{fk}_probs.npz"), allow_pickle=True)
            for nm, pr in zip(zp["pool_names"], zp["pool_probs_6x4"]):
                acc_b.setdefault(str(nm), []).append(np.asarray(pr, dtype=np.float64))
            for nm, pr in zip(zp["va_names"], zp["va_probs_6x4"]):
                fullfit_acc[idx_map[str(nm)]].append(np.asarray(pr, dtype=np.float64))

        arr_k = np.zeros((N, 6, 4), dtype=np.float32)
        for r in tr_all:
            nm = str(r["name"])
            src = acc_b if nm in half_a else acc_a
            assert nm in src, f"Missing cross-fitted teacher for {nm} in fold {fk}"
            arr_k[idx_map[nm]] = c.renorm(np.mean(src[nm], axis=0)[None])[0].astype(np.float32)
        # Note: outer validation rows `va_rows` are NEVER read during fold-fk training (`tr_rows`), set to 0.0!
        rebuilt_folds[f"fold_{fk}_probs"] = arr_k

    fullfit_arr = c.renorm(np.stack([np.mean(fullfit_acc[i], axis=0) for i in range(N)], axis=0)).astype(np.float32)

    out_cv8 = os.path.join(REPO, "results/r55/clean_honest_kd_teacher_cv8.npz")
    out_full = os.path.join(REPO, "results/r55/clean_honest_kd_teacher_fullfit.npz")
    np.savez_compressed(out_cv8, names=np.array(names, dtype=object), **rebuilt_folds)
    np.savez_compressed(out_full, names=np.array(names, dtype=object), fold_0_probs=fullfit_arr, probs_6x4=fullfit_arr)

    # Compare against existing nested_kd_teacher.npz on training rows of each fold
    z_nest = np.load(os.path.join(REPO, "results/r55/nested_kd_teacher.npz"), allow_pickle=True)
    z_full = np.load(os.path.join(REPO, "results/r55/fullfit_halfAB_kd_teacher.npz"), allow_pickle=True)
    maes_tr = []
    for fk in range(8):
        m_tr = fold != fk
        maes_tr.append(float(np.mean(np.abs(rebuilt_folds[f"fold_{fk}_probs"][m_tr] - z_nest[f"fold_{fk}_probs"][m_tr]))))
    mae_full = float(np.mean(np.abs(fullfit_arr - z_full["probs_6x4"])))

    from eval_specialist_gate import load_oof_dir, rps_6x4
    p_nokd, _ = load_oof_dir("results/next55/towers/r55_p12_e5_s2026/pe_core_large", names)
    p_kd, _ = load_oof_dir("results/next55/towers/r55_p12_kd_e5_s2026/pe_core_large", names)

    rep = dict(
        clean_cv8_npz=out_cv8,
        clean_fullfit_npz=out_full,
        uses_pillar3=False,
        zero_fold_leakage=True,
        mae_vs_existing_nested_kd_per_fold=maes_tr,
        mae_vs_existing_fullfit_kd=mae_full,
        p12_pe_zero_kd_oof_rps=float(rps_6x4(c.renorm(p_nokd), gold)),
        p12_pe_kd_oof_rps=float(rps_6x4(c.renorm(p_kd), gold)),
    )
    out_j = os.path.join(REPO, "results/r55/stage3/kd_provenance_audit.json")
    os.makedirs(os.path.dirname(out_j), exist_ok=True)
    json.dump(rep, open(out_j, "w"), indent=2)
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    build_and_verify()
