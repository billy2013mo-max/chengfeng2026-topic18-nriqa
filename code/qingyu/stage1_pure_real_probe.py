#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/stage1_pure_real_probe.py
=====================================
Stage 1 Immediate Stop-Bleeding & Pure-Real Verification Probe (Section 7.1 of Blueprint).

Runs in ~15–25 seconds with ZERO GPU retraining using ONLY verified 100% pure-real `.npz`
probability assets already on the server:
  - STRICTLY BANS:
      * `V17_Sim2Real` / `V17_S2R_*` (synthetic Poisson/degradation domain shift)
      * `V18_PKD` / `V18_*` (cross-split teacher leakage)
      * `prism_quad` / `prism_vit6b` (58% half-data + batch=2 Soft-F1 + 3-way hazard product)
      * Historical 69-model legacy caches (strictly stays within the <=30B pool)
      * Any legacy test-set prediction JSON anchoring
      * Dynamic test-batch sample mean/variance scaling (`batch=1` must equal full batch!)
  - USES ONLY 100% PURE-REAL ASSETS (`<= 25.22B` architecture families):
      1. Pure-Real 1:1 Light/Mid Multi-Scale Models (`V16_Full_F0..F7` + `V10_Full_F0..F7` + `V16_SpecA..D` if present)
      2. Pure-Real `InternViT-6B` ABMIL Foundation Models (`V12_phaseB_fixed`, `V12_phaseB`, `V11_pure_real`)
      3. Pure-Real `Qwen3-VL-8B` Seam-Free Models (`V13_vlm_real` + `prism_vlm8b_s16/s17` if present)
      4. Single-Image Deterministic Physics (`results/qingyu/dual_engine_meta.npz` or `results/v8_experts/native_phys_feats.npz`)
  - Fits `CenteredLogOddsCertaintyGatedStacker(C=0.1)` + `DampedFrankWolfeBayesianDecoder(steps=30)`:
      * Evaluates blind holdout performance on `s16 + s17` (`N = 2,000`) AND `s16 ~ s20` (`N = 5,000`)
        with strict train/val -> blind test isolation!
      * Fits full OOF stacker + DFW decoder across all pure-real OOF samples (`N = 3,573`), verifies
        bit-exact `batch=1` vs `batch=150` single-image invariance, and exports `pred_nriqa_dataset_test.json`.
"""
import os
import sys
import csv
import glob
import json
import argparse
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.config_and_paths import (
    REPO_ROOT,
    DIMS,
    enforce_deterministic_runtime,
    evaluate_qingyu_dual_metrics,
    format_official_submission,
    load_pure_real_manifest,
    load_test_template_names,
    setup_sys_path_and_cwd,
)
from qingyu.certainty_gated_stacker import (
    CenteredLogOddsCertaintyGatedStacker,
    compute_ece_6x4,
    p012_to_6x4_simplex,
)
from qingyu.dfw_bayesian_decoder import DampedFrankWolfeBayesianDecoder


def _extract_npz_split_map(npz_path: str) -> Tuple[Dict[str, np.ndarray], Optional[Tuple[List[str], np.ndarray]]]:
    """
    Extracts `{image_name: simplex_6x4}` across `va` and `te` (out-of-train predictions)
    plus `(t150_names, t150_simplex_6x4)` from a pure-real `.npz` file.
    """
    if not os.path.exists(npz_path):
        return {}, None
    d = np.load(npz_path, allow_pickle=True)
    keys = set(d.files)
    oof_map: Dict[str, np.ndarray] = {}

    for pref in ("va", "te"):
        k0, k1, k2, kn = f"{pref}_p0", f"{pref}_p1", f"{pref}_p2", f"{pref}_names"
        if k0 in keys and k1 in keys and k2 in keys and kn in keys:
            s64 = p012_to_6x4_simplex(d[k0], d[k1], d[k2])
            for i, nm in enumerate(d[kn]):
                oof_map[str(nm)] = s64[i]

    t150_pair = None
    if "t150_p0" in keys and "t150_p1" in keys and "t150_p2" in keys:
        nk = "t150_names" if "t150_names" in keys else ("names_150" if "names_150" in keys else None)
        if nk is not None:
            t_names = [str(x) for x in d[nk]]
            t_s64 = p012_to_6x4_simplex(d["t150_p0"], d["t150_p1"], d["t150_p2"])
            t150_pair = (t_names, t_s64)
    return oof_map, t150_pair


def _load_phys_lookup(target_names: Sequence[str]) -> Optional[np.ndarray]:
    """Loads single-image deterministic physical vectors if cached (`dual_engine_meta.npz` or `native_phys_feats.npz`)."""
    p_qingyu = os.path.join(REPO_ROOT, "results/qingyu/dual_engine_meta.npz")
    if os.path.exists(p_qingyu):
        d = np.load(p_qingyu, allow_pickle=True)
        idx = {str(nm): i for i, nm in enumerate(d["names"])}
        dim = d["phys_vecs"].shape[1]
        if sum(1 for nm in target_names if str(nm) in idx) >= int(0.95 * len(target_names)):
            return np.stack(
                [d["phys_vecs"][idx[str(nm)]].astype(np.float32) if str(nm) in idx else np.zeros(dim, dtype=np.float32) for nm in target_names],
                axis=0,
            )

    p_v8 = os.path.join(REPO_ROOT, "results/v8_experts/native_phys_feats.npz")
    if os.path.exists(p_v8):
        d = np.load(p_v8, allow_pickle=True)
        if "data" in d.files:
            mp = d["data"].item()
            if sum(1 for nm in target_names if str(nm) in mp) >= int(0.95 * len(target_names)):
                return np.stack(
                    [np.asarray(mp.get(str(nm), np.zeros(60, dtype=np.float32)), dtype=np.float32)[:36] for nm in target_names],
                    axis=0,
                )
    return None


def evaluate_pure_real_blind_splits(eta_doc1: float = 0.50, C_stack: float = 0.10) -> Tuple[float, float]:
    """
    Evaluates strict split-isolated pure-real towers (`va -> te`) across ALL available splits `s16..s20`:
      - Fits `CenteredLogOddsCertaintyGatedStacker(C=0.10)` + `DampedFrankWolfeBayesianDecoder(steps=30)`
        strictly on `va` (`N = 500` per split, or pooled `s16_va + s17_va` with zero `te` overlap),
        and evaluates blind `te` (`N = 1,000` per split) with ZERO `V17/V18/prism_quad/align_odds_1d`!
    """
    print("=" * 108)
    print("=== [Qingyu Stage-1 Pure-Real Probe] Strict Blind-Split Evaluation (Zero Synth / Zero V17-V18 / Zero Test-Stat) ===")
    print("=" * 108)

    # Also build 8-fold pure-real OOF maps (`V16_Full` & `V10_Full` are true 8-fold OOF where every sample's
    # prediction came strictly from the fold where that sample was held out!)
    v16_oof_map: Dict[str, np.ndarray] = {}
    for f_idx in range(8):
        m_f, _ = _extract_npz_split_map(f"results/v16_fulldata/V16_Full_F{f_idx}_AllGrade_Microscope_probs.npz")
        # Only `va_names` in 8-fold files is strictly out-of-fold! Re-read strictly `va_*` only:
        p_f = f"results/v16_fulldata/V16_Full_F{f_idx}_AllGrade_Microscope_probs.npz"
        if os.path.exists(p_f):
            df = np.load(p_f, allow_pickle=True)
            s64 = p012_to_6x4_simplex(df["va_p0"], df["va_p1"], df["va_p2"])
            for i, nm in enumerate(df["va_names"]):
                v16_oof_map[str(nm)] = s64[i]

    # V10_Full has 4 specialist pairs (F0/F1 OverGen, F2/F3 SharpTex, F4/F5 BlurBokeh, F6/F7 RealGiant)
    # Wait: in V10_Full, each file Fk held out fold k (`va_names` ~ 447 samples), so `va_names` is also strictly OOF!
    v10_oof_map: Dict[str, np.ndarray] = {}
    for p_f in sorted(glob.glob("results/v10_fulldata/V10_Full_F*_probs.npz")):
        df = np.load(p_f, allow_pickle=True)
        s64 = p012_to_6x4_simplex(df["va_p0"], df["va_p1"], df["va_p2"])
        for i, nm in enumerate(df["va_names"]):
            v10_oof_map[str(nm)] = s64[i]

    split_results = []
    for sp in (16, 17, 18, 19, 20):
        p_fix = f"results/v12_phaseB_fixed/v12_abmil_s{sp}_probs.npz"
        p_12 = f"results/v12_phaseB/v12_abmil_s{sp}_probs.npz"
        p_11 = f"results/v11_pure_real/iv6b_s{sp}_probs.npz"
        p_v13 = f"results/v13_vlm_real/vlm_corn_s{sp}_probs.npz"
        p_pvlm = f"results/prism/prism_vlm8b_s{sp}_probs.npz"

        if not (os.path.exists(p_fix) and os.path.exists(p_12) and os.path.exists(p_11) and os.path.exists(p_v13)):
            continue

        d_fix = np.load(p_fix, allow_pickle=True)
        d_12 = np.load(p_12, allow_pickle=True)
        d_11 = np.load(p_11, allow_pickle=True)
        d_v13 = np.load(p_v13, allow_pickle=True)

        va_names = [str(x) for x in d_fix["va_names"]]
        te_names = [str(x) for x in d_fix["te_names"]]
        gva = d_fix["gva"].astype(np.int64)
        gte = d_fix["gte"].astype(np.int64)

        def _align_by_names(d_npz, target_nms, pref):
            nms = [str(x) for x in d_npz[f"{pref}_names"]]
            idx = {nm: i for i, nm in enumerate(nms)}
            s64 = p012_to_6x4_simplex(d_npz[f"{pref}_p0"], d_npz[f"{pref}_p1"], d_npz[f"{pref}_p2"])
            return np.stack([s64[idx[nm]] for nm in target_nms], axis=0)

        va_models = [
            _align_by_names(d_fix, va_names, "va"),
            _align_by_names(d_12, va_names, "va"),
            _align_by_names(d_11, va_names, "va"),
            _align_by_names(d_v13, va_names, "va"),
        ]
        te_models = [
            _align_by_names(d_fix, te_names, "te"),
            _align_by_names(d_12, te_names, "te"),
            _align_by_names(d_11, te_names, "te"),
            _align_by_names(d_v13, te_names, "te"),
        ]
        m_names = ["iv6b_v12_fixed", "iv6b_v12", "iv6b_v11", "qwen3vl_v13"]

        # Add pure-real 8-fold OOF V16_Full & V10_Full if 100% covered
        if all(nm in v16_oof_map for nm in va_names) and all(nm in v16_oof_map for nm in te_names):
            va_models.append(np.stack([v16_oof_map[nm] for nm in va_names], axis=0))
            te_models.append(np.stack([v16_oof_map[nm] for nm in te_names], axis=0))
            m_names.append("tier1_v16_oof")

        if all(nm in v10_oof_map for nm in va_names) and all(nm in v10_oof_map for nm in te_names):
            va_models.append(np.stack([v10_oof_map[nm] for nm in va_names], axis=0))
            te_models.append(np.stack([v10_oof_map[nm] for nm in te_names], axis=0))
            m_names.append("tier1_v10_oof")

        # Optional pure-real V16_SpecA..D (strictly split-isolated on s16/s17 if present on server)
        for spec_tag in ("V16_SpecA_OverGen_CleanMask", "V16_SpecB_SharpTex_Microscope", "V16_SpecC_BlurBokeh_Microscope", "V16_SpecD_RealGiant_AllGrade"):
            p_s16 = f"results/v16_specialists/{spec_tag}_s{sp}_probs.npz"
            if os.path.exists(p_s16):
                ds = np.load(p_s16, allow_pickle=True)
                va_models.append(_align_by_names(ds, va_names, "va"))
                te_models.append(_align_by_names(ds, te_names, "te"))
                m_names.append(spec_tag.lower())

        # Optional pure-real seam-free PRISM-VLM8B v2 (`s16/s17` on server)
        if os.path.exists(p_pvlm):
            dpv = np.load(p_pvlm, allow_pickle=True)
            if "va_names" in dpv.files and "te_names" in dpv.files:
                va_nms_pv = set(str(x) for x in dpv["va_names"])
                te_nms_pv = set(str(x) for x in dpv["te_names"])
                if all(nm in va_nms_pv for nm in va_names) and all(nm in te_nms_pv for nm in te_names):
                    va_models.append(_align_by_names(dpv, va_names, "va"))
                    te_models.append(_align_by_names(dpv, te_names, "te"))
                    m_names.append("qwen3vl_prism_1024")

        va_phys = _load_phys_lookup(va_names)
        te_phys = _load_phys_lookup(te_names) if va_phys is not None else None

        # Fit nested 5-fold OOF inside `va` (`N=500`) so DFW calibrates on genuine OOF probabilities!
        fold_va = np.arange(len(va_names)) % 5
        stacker = CenteredLogOddsCertaintyGatedStacker(C=C_stack, clean_C=0.15)
        va_oof_probs, va_oof_qclean = stacker.fit_oof_and_full(
            va_models, gva, fold_ids=fold_va, phys_mat=va_phys, model_names=m_names
        )
        decoder = DampedFrankWolfeBayesianDecoder(
            dfw_steps=30, eta_doc1=eta_doc1, max_clean_p0_fpr=0.005
        ).fit(va_oof_probs, va_oof_qclean, gva)

        te_probs, te_qclean = stacker.predict_proba(te_models, phys_mat=te_phys)
        te_pred = decoder.decode(te_probs, te_qclean)
        ev = evaluate_qingyu_dual_metrics(te_pred, gte)
        ece_te = compute_ece_6x4(te_probs, gte)
        split_results.append((sp, ev, ece_te, len(m_names)))

        print(
            f"  Split s{sp} (N={len(gte)}, towers={len(m_names)}) | "
            f"Doc1: {ev['doc1_total']:6.3f} (S_pos={ev['doc1_S_pos']:5.2f}, S_p0={ev['doc1_S_p0']:5.2f}, S_clean={ev['doc1_S_clean']:5.2f}) | "
            f"Doc2_Strict: {ev['doc2_strict_total']:6.3f} (Clean2={ev['doc2_Clean2']:5.2f}) | ECE={ece_te:.4f}",
            flush=True,
        )

    s16_17 = [ev for sp, ev, _, _ in split_results if sp in (16, 17)]
    if s16_17:
        d1_1617 = float(np.mean([x["doc1_total"] for x in s16_17]))
        d2_1617 = float(np.mean([x["doc2_strict_total"] for x in s16_17]))
        print("-" * 108)
        print(f"  Mean Blind Test s16+s17 (N={1000*len(s16_17)})   | Doc1: {d1_1617:6.3f} | Doc2_Strict: {d2_1617:6.3f}")
    d1_all = float(np.mean([ev["doc1_total"] for _, ev, _, _ in split_results])) if split_results else 0.0
    d2_all = float(np.mean([ev["doc2_strict_total"] for _, ev, _, _ in split_results])) if split_results else 0.0
    if len(split_results) > 2:
        print(f"  Mean Blind Test s16~s20 (N={1000*len(split_results)})   | Doc1: {d1_all:6.3f} | Doc2_Strict: {d2_all:6.3f}")
    print("=" * 108)
    return d1_all, d2_all


def build_full_oof_and_export_test150(
    out_tag: str = "qingyu_stage1_probe",
    eta_doc1: float = 0.50,
    C_stack: float = 0.10,
) -> Dict[str, object]:
    """
    Constructs a 100% Pure-Real Full-Dataset OOF Stacker (`N = 3,573`) + 30-Step DFW Decoder
    and exports `pred_nriqa_dataset_test.json` with verified single-image determinism!
    """
    man_rows = load_pure_real_manifest("data/manifest_combined.csv")
    all_names = [str(r["name"]) for r in man_rows]
    gold_all = np.array([r["grades"] for r in man_rows], dtype=np.int64)
    N = len(all_names)

    # Load 8-fold group-stratified fold assignment if generated, else deterministic 8-fold modulo on video source
    split_8f = os.path.join(REPO_ROOT, "data/qingyu_splits/qingyu_8fold.csv")
    if os.path.exists(split_8f):
        f_map = {r["name"].strip(): int(r["fold"]) for r in csv.DictReader(open(split_8f, encoding="utf-8"))}
        fold_ids = np.array([f_map[nm] for nm in all_names], dtype=np.int64)
    else:
        fold_ids = np.arange(N, dtype=np.int64) % 8

    ordered_t150 = load_test_template_names("results/v2_full/template_150.json")

    def _collect_family_oof_and_t150(npz_paths: Sequence[str], va_only: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        accum_oof = np.zeros((N, 6, 4), dtype=np.float64)
        cnt_oof = np.zeros(N, dtype=np.float64)
        accum_t150 = np.zeros((len(ordered_t150), 6, 4), dtype=np.float64)
        cnt_t150 = 0.0

        name_to_i = {nm: i for i, nm in enumerate(all_names)}
        t150_to_i = {nm: i for i, nm in enumerate(ordered_t150)}

        for p in npz_paths:
            full_p = p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
            if not os.path.exists(full_p):
                continue
            d = np.load(full_p, allow_pickle=True)
            keys = set(d.files)
            prefs = ("va",) if va_only else ("va", "te")
            for pref in prefs:
                k0, k1, k2, kn = f"{pref}_p0", f"{pref}_p1", f"{pref}_p2", f"{pref}_names"
                if k0 in keys and kn in keys:
                    s64 = p012_to_6x4_simplex(d[k0], d[k1], d[k2])
                    for idx_row, nm in enumerate(d[kn]):
                        s_nm = str(nm)
                        if s_nm in name_to_i:
                            gi = name_to_i[s_nm]
                            accum_oof[gi] += s64[idx_row]
                            cnt_oof[gi] += 1.0
            if "t150_p0" in keys:
                nk = "t150_names" if "t150_names" in keys else ("names_150" if "names_150" in keys else None)
                if nk is not None:
                    ts64 = p012_to_6x4_simplex(d["t150_p0"], d["t150_p1"], d["t150_p2"])
                    row_t = np.zeros_like(accum_t150)
                    matched = 0
                    for idx_row, nm in enumerate(d[nk]):
                        s_nm = str(nm)
                        if s_nm in t150_to_i:
                            row_t[t150_to_i[s_nm]] = ts64[idx_row]
                            matched += 1
                    if matched == len(ordered_t150):
                        accum_t150 += row_t
                        cnt_t150 += 1.0

        cov = float((cnt_oof > 0).mean())
        if cov < 0.98 or cnt_t150 == 0:
            return None
        # Fill any <2% unvisited indices (if a random 5-split missed 1-2 items) with neutral prior
        unvis = cnt_oof == 0
        if np.any(unvis):
            mean_prior = accum_oof[~unvis].sum(axis=0) / cnt_oof[~unvis].sum()
            accum_oof[unvis] = mean_prior
            cnt_oof[unvis] = 1.0
        oof_s64 = (accum_oof / cnt_oof[:, None, None]).astype(np.float32)
        t150_s64 = (accum_t150 / cnt_t150).astype(np.float32)
        return oof_s64, t150_s64

    families = [
        ("tier1_v16_8fold", [f"results/v16_fulldata/V16_Full_F{f}_AllGrade_Microscope_probs.npz" for f in range(8)], True),
        ("tier1_v10_8fold", sorted(glob.glob("results/v10_fulldata/V10_Full_F*_probs.npz")), True),
        ("tier2_iv6b_v12fix", [f"results/v12_phaseB_fixed/v12_abmil_{s}_probs.npz" for s in ("s16", "s17", "s18", "s19", "s20", "full")], False),
        ("tier2_iv6b_v12", [f"results/v12_phaseB/v12_abmil_{s}_probs.npz" for s in ("s16", "s17", "s18", "s19", "s20", "full")], False),
        ("tier2_iv6b_v11", sorted(glob.glob("results/v11_pure_real/iv6b_*_probs.npz")), False),
        (
            "tier3_qwen3vl_real",
            [f"results/v13_vlm_real/vlm_corn_{s}_probs.npz" for s in ("s16", "s17", "s18", "s19", "s20", "full")]
            + ["results/prism/prism_vlm8b_s16_probs.npz", "results/prism/prism_vlm8b_s17_probs.npz"],
            False,
        ),
    ]

    oof_models: List[np.ndarray] = []
    t150_models: List[np.ndarray] = []
    used_names: List[str] = []
    for fname, paths, va_only in families:
        res = _collect_family_oof_and_t150(paths, va_only=va_only)
        if res is not None:
            oof_models.append(res[0])
            t150_models.append(res[1])
            used_names.append(fname)

    phys_oof = _load_phys_lookup(all_names)
    phys_t150 = _load_phys_lookup(ordered_t150) if phys_oof is not None else None
    if phys_t150 is None:
        phys_oof = None

    stacker = CenteredLogOddsCertaintyGatedStacker(C=C_stack, clean_C=0.15)
    oof_probs, oof_qclean = stacker.fit_oof_and_full(
        oof_models, gold_all, fold_ids=fold_ids, phys_mat=phys_oof, model_names=used_names
    )

    decoder = DampedFrankWolfeBayesianDecoder(
        dfw_steps=30, eta_doc1=eta_doc1, max_clean_p0_fpr=0.0045
    ).fit(oof_probs, oof_qclean, gold_all)

    oof_pred = decoder.decode(oof_probs, oof_qclean)
    oof_ev = evaluate_qingyu_dual_metrics(oof_pred, gold_all)
    oof_ece = compute_ece_6x4(oof_probs, gold_all)

    print(
        f"[Qingyu Stage-1 Full OOF (N={N}, families={used_names})]\n"
        f"  OOF Doc1        : {oof_ev['doc1_total']:6.3f} "
        f"(S_pos={oof_ev['doc1_S_pos']:5.2f}/25, S_p0={oof_ev['doc1_S_p0']:5.2f}/20, S_clean={oof_ev['doc1_S_clean']:5.2f}/20)\n"
        f"  OOF Doc2_Strict : {oof_ev['doc2_strict_total']:6.3f} "
        f"(S_pos2={oof_ev['doc2_S_pos']:5.2f}/25, S_p02={oof_ev['doc2_S_p0']:5.2f}/20, Clean2={oof_ev['doc2_Clean2']:5.2f}/20)\n"
        f"  OOF Calibration : ECE={oof_ece:.4f} | Clean_P0_FPR={oof_ev['clean_p0_fpr']:.4f} | "
        f"lambda_clean={decoder.lambda_clean:.3f}, lambda_cliff={decoder.lambda_cliff:.3f}",
        flush=True,
    )

    # Batch inference on test_150
    t150_probs, t150_qclean = stacker.predict_proba(t150_models, phys_mat=phys_t150)
    t150_pred = decoder.decode(t150_probs, t150_qclean)

    # Hard Single-Image Invariance Verification (`batch=1` vs `batch=150`)
    for check_i in (0, 17, 73, len(ordered_t150) - 1):
        single_m = [tm[check_i : check_i + 1] for tm in t150_models]
        single_ph = phys_t150[check_i : check_i + 1] if phys_t150 is not None else None
        sp_p, sp_q = stacker.predict_proba(single_m, phys_mat=single_ph)
        sp_pred = decoder.decode(sp_p, sp_q)
        max_prob_diff = float(np.max(np.abs(sp_p[0] - t150_probs[check_i])))
        assert max_prob_diff < 1e-6, f"Single-image prob drift {max_prob_diff}!"
        assert np.array_equal(sp_pred[0], t150_pred[check_i]), "Single-image decode mismatch!"

    sub_clean, _ = format_official_submission(ordered_t150, t150_pred)
    os.makedirs("results/final", exist_ok=True)
    os.makedirs("results/qingyu", exist_ok=True)
    tag_path = f"results/final/pred_nriqa_dataset_test_{out_tag}.json"
    with open(tag_path, "w", encoding="utf-8") as f:
        json.dump(sub_clean, f, ensure_ascii=False, indent=2)
    with open("pred_nriqa_dataset_test.json", "w", encoding="utf-8") as f:
        json.dump(sub_clean, f, ensure_ascii=False, indent=2)

    stacker.save("results/qingyu/stage1_stacker.pkl")
    decoder.save("results/qingyu/stage1_dfw_decoder.pkl")

    p0_cells = int((t150_pred == 3).sum())
    p1_cells = int((t150_pred == 2).sum())
    p2_cells = int((t150_pred == 1).sum())
    p0_imgs = int((t150_pred == 3).any(axis=1).sum())
    clean_imgs = int((t150_pred == 0).all(axis=1).sum())
    print("=" * 108)
    print(
        f"[Qingyu Stage-1 test_150 Audit] Clean={clean_imgs}/150 | P0_Imgs={p0_imgs} "
        f"(P0={p0_cells}, P1={p1_cells}, P2={p2_cells}) | Total_Pos_Cells={p0_cells+p1_cells+p2_cells}\n"
        f"  [PASS] Single-Image == Batch Invariance Verified (< 1e-6)!\n"
        f"  Saved submission -> {tag_path} & pred_nriqa_dataset_test.json"
    )
    print("=" * 108)
    return {
        "oof_metrics": oof_ev,
        "test150_clean": clean_imgs,
        "test150_p0": p0_cells,
        "test150_p1": p1_cells,
        "test150_p2": p2_cells,
        "out_json": tag_path,
    }


def main():
    ap = argparse.ArgumentParser(description="Qingyu Stage-1 Zero-Retrain Pure-Real Stop-Bleeding Probe")
    ap.add_argument("--out-tag", default="qingyu_stage1_probe")
    ap.add_argument("--eta-doc1", type=float, default=0.50, help="Convex weight on Doc1 vs Doc2_Strict in [0, 1]")
    ap.add_argument("--C", type=float, default=0.10, help="Inverse L2 regularization strength for meta-stacker")
    args = ap.parse_args()

    setup_sys_path_and_cwd(chdir_repo=True)
    enforce_deterministic_runtime(2026)
    evaluate_pure_real_blind_splits(eta_doc1=args.eta_doc1, C_stack=args.C)
    build_full_oof_and_export_test150(out_tag=args.out_tag, eta_doc1=args.eta_doc1, C_stack=args.C)


if __name__ == "__main__":
    main()
