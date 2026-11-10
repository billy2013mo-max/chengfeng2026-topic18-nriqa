#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/stage1_pure_real_probe.py
=====================================
Stage 1 Immediate Stop-Bleeding & Pure-Real Verification Probe (Section 7.1 of Blueprint).

Runs in ~15–25 seconds with ZERO GPU retraining using ONLY verified 100% pure-real,
full-6-dimension `.npz` probability assets already on the server:
  - STRICTLY EXCLUDES:
      * `V17_Sim2Real` / `V17_S2R_*` (synthetic Poisson/degradation domain shift)
      * `V18_PKD` / `V18_*` (cross-split teacher leakage)
      * `prism_quad` / `prism_vit6b` (58% half-data + batch=2 Soft-F1 + 3-way hazard product)
      * Stitched single-defect partial specialists (`V10_Full_F0..5` / `V16_SpecA..C` whose non-target dims are uncalibrated)
      * `*_full_probs.npz` in OOF fitting (strictly uses out-of-train `va` + `te` predictions ONLY!)
      * Historical 69-model legacy caches, legacy JSONs, and dynamic test-batch mean/std scaling.
  - USES ONLY 100% PURE-REAL FULL-6D ASSETS (`<= 25.28B` architecture families):
      1. `tier1_v16_8fold`: Pure-Real 8-fold `V16_Full_F0..F7_AllGrade_Microscope` (1:1 light/mid multi-scale)
      2. `tier2_iv6b_v12fix`: Pure-Real `InternViT-6B` ABMIL (`v12_phaseB_fixed` out-of-train `s16..s20`)
      3. `tier2_iv6b_v12`: Pure-Real `InternViT-6B` ABMIL (`v12_phaseB` out-of-train `s16..s20`)
      4. `tier2_iv6b_v11`: Pure-Real `InternViT-6B` (`v11_pure_real` out-of-train `s16..s20`)
      5. `tier3_qwen3vl_v13`: Pure-Real `Qwen3-VL-8B` (`v13_vlm_real` out-of-train `s16..s20`)
      6. `qwen3vl_prism_1024`: Pure-Real Seam-Free `Qwen3-VL-8B` (`prism_vlm8b_s16/s17` on blind `s16/s17`)
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
    Evaluates strict split-isolated pure-real full-6D towers (`va -> te`) across `s16..s20`.
    """
    print("=" * 108)
    print("=== [Qingyu Stage-1 Pure-Real Probe] Strict Blind-Split Evaluation (Homogeneous Full-6D Pure-Real Towers) ===")
    print("=" * 108)

    v16_oof_map: Dict[str, np.ndarray] = {}
    for f_idx in range(8):
        p_f = f"results/v16_fulldata/V16_Full_F{f_idx}_AllGrade_Microscope_probs.npz"
        if os.path.exists(p_f):
            df = np.load(p_f, allow_pickle=True)
            s64 = p012_to_6x4_simplex(df["va_p0"], df["va_p1"], df["va_p2"])
            for i, nm in enumerate(df["va_names"]):
                v16_oof_map[str(nm)] = s64[i]

    split_results = []
    for sp in (16, 17, 18, 19, 20):
        p_fix = f"results/v12_phaseB_fixed/v12_abmil_s{sp}_probs.npz"
        p_12 = f"results/v12_phaseB/v12_abmil_s{sp}_probs.npz"
        p_11 = f"results/v11_pure_real/iv6b_s{sp}_probs.npz"
        p_v13 = f"results/v13_vlm_real/vlm_corn_s{sp}_probs.npz"
        p_pvlm = f"results/prism/prism_vlm8b_s{sp}_probs.npz"
        p_specd = f"results/v16_specialists/V16_SpecD_RealGiant_AllGrade_s{sp}_probs.npz"

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

        if all(nm in v16_oof_map for nm in va_names) and all(nm in v16_oof_map for nm in te_names):
            va_models.append(np.stack([v16_oof_map[nm] for nm in va_names], axis=0))
            te_models.append(np.stack([v16_oof_map[nm] for nm in te_names], axis=0))
            m_names.append("tier1_v16_oof")

        # Include ONLY All-6-Dim pure-real towers (SpecD_RealGiant_AllGrade & seam-free PRISM-VLM8B_1024)
        if os.path.exists(p_specd):
            ds = np.load(p_specd, allow_pickle=True)
            va_models.append(_align_by_names(ds, va_names, "va"))
            te_models.append(_align_by_names(ds, te_names, "te"))
            m_names.append("tier1_v16_giant_all6")

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

        fold_va = np.arange(len(va_names)) % 5
        stacker = CenteredLogOddsCertaintyGatedStacker(C=C_stack, clean_C=0.15)
        va_oof_probs, va_oof_qclean = stacker.fit_oof_and_full(
            va_models, gva, fold_ids=fold_va, phys_mat=va_phys, model_names=m_names
        )
        decoder = DampedFrankWolfeBayesianDecoder(
            dfw_steps=30, eta_doc1=eta_doc1, max_clean_p0_fpr=0.002
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
    Constructs a 100% Pure-Real 3-Tier OOF Stacker across ALL covered pure-real samples (`~3,338 / 3,573`,
    strictly `va` + `te` out-of-train predictions ONLY, zero `full` in-sample leakage, zero prior imputation!)
    and exports `pred_nriqa_dataset_test.json` with verified single-image invariance!
    """
    man_rows = load_pure_real_manifest("data/manifest_combined.csv")
    all_names = [str(r["name"]) for r in man_rows]
    gold_all = np.array([r["grades"] for r in man_rows], dtype=np.int64)
    N = len(all_names)

    split_8f = os.path.join(REPO_ROOT, "data/qingyu_splits/qingyu_8fold.csv")
    if os.path.exists(split_8f):
        f_map = {r["name"].strip(): int(r["fold"]) for r in csv.DictReader(open(split_8f, encoding="utf-8"))}
        fold_ids_all = np.array([f_map[nm] for nm in all_names], dtype=np.int64)
    else:
        fold_ids_all = np.arange(N, dtype=np.int64) % 8

    ordered_t150 = load_test_template_names("results/v2_full/template_150.json")

    def _collect_family_oof_and_t150(
        npz_paths: Sequence[str],
        va_only: bool = False,
        extra_t150_paths: Sequence[str] = (),
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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

        for p in list(npz_paths) + list(extra_t150_paths):
            full_p = p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
            if not os.path.exists(full_p):
                continue
            d = np.load(full_p, allow_pickle=True)
            keys = set(d.files)
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

        valid_mask = cnt_oof > 0
        cov = float(valid_mask.mean())
        # 5 random splits s16..s20 cover 1 - (1 - 1500/3573)^5 = 93.4% of N=3573!
        if cov < 0.85 or cnt_t150 == 0:
            return None
        denom = np.where(valid_mask, cnt_oof, 1.0)
        oof_s64 = (accum_oof / denom[:, None, None]).astype(np.float32)
        t150_s64 = (accum_t150 / cnt_t150).astype(np.float32)
        return oof_s64, t150_s64, valid_mask

    # Strictly out-of-train files ONLY for OOF (`s16..s20`), never `*_full_probs.npz` in OOF!
    families = [
        (
            "tier1_v16_8fold",
            [f"results/v16_fulldata/V16_Full_F{f}_AllGrade_Microscope_probs.npz" for f in range(8)],
            True,
            (),
        ),
        (
            "tier2_iv6b_v12fix",
            [f"results/v12_phaseB_fixed/v12_abmil_{s}_probs.npz" for s in ("s16", "s17", "s18", "s19", "s20")],
            False,
            ("results/v12_phaseB_fixed/v12_abmil_full_probs.npz",),
        ),
        (
            "tier2_iv6b_v12",
            [f"results/v12_phaseB/v12_abmil_{s}_probs.npz" for s in ("s16", "s17", "s18", "s19", "s20")],
            False,
            ("results/v12_phaseB/v12_abmil_full_probs.npz",),
        ),
        (
            "tier2_iv6b_v11",
            [f"results/v11_pure_real/iv6b_{s}_probs.npz" for s in ("s16", "s17", "s18", "s19", "s20")],
            False,
            (),
        ),
        (
            "tier3_qwen3vl_real",
            [f"results/v13_vlm_real/vlm_corn_{s}_probs.npz" for s in ("s16", "s17", "s18", "s19", "s20")]
            + ["results/prism/prism_vlm8b_s16_probs.npz", "results/prism/prism_vlm8b_s17_probs.npz"],
            False,
            ("results/v13_vlm_real/vlm_corn_full_probs.npz",),
        ),
    ]

    oof_models_raw: List[np.ndarray] = []
    t150_models: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []
    used_names: List[str] = []
    for fname, paths, va_only, ext_t150 in families:
        res = _collect_family_oof_and_t150(paths, va_only=va_only, extra_t150_paths=ext_t150)
        if res is not None:
            oof_models_raw.append(res[0])
            t150_models.append(res[1])
            masks_list.append(res[2])
            used_names.append(fname)

    # Fit strictly on the exact pure-real subset covered out-of-fold by ALL loaded families (zero prior imputation!)
    common_mask = np.ones(N, dtype=bool)
    for mk in masks_list:
        common_mask &= mk

    oof_models = [m[common_mask] for m in oof_models_raw]
    gold_sub = gold_all[common_mask]
    folds_sub = fold_ids_all[common_mask]
    names_sub = [all_names[i] for i in range(N) if common_mask[i]]

    phys_oof = _load_phys_lookup(names_sub)
    phys_t150 = _load_phys_lookup(ordered_t150) if phys_oof is not None else None
    if phys_t150 is None:
        phys_oof = None

    stacker = CenteredLogOddsCertaintyGatedStacker(C=C_stack, clean_C=0.15)
    oof_probs, oof_qclean = stacker.fit_oof_and_full(
        oof_models, gold_sub, fold_ids=folds_sub, phys_mat=phys_oof, model_names=used_names
    )

    decoder = DampedFrankWolfeBayesianDecoder(
        dfw_steps=30, eta_doc1=eta_doc1, max_clean_p0_fpr=0.002
    ).fit(oof_probs, oof_qclean, gold_sub)

    oof_pred = decoder.decode(oof_probs, oof_qclean)
    oof_ev = evaluate_qingyu_dual_metrics(oof_pred, gold_sub)
    oof_ece = compute_ece_6x4(oof_probs, gold_sub)

    print(
        f"[Qingyu Stage-1 Full 3-Tier OOF (N_covered={len(gold_sub)}/{N}, families={used_names})]\n"
        f"  OOF Doc1        : {oof_ev['doc1_total']:6.3f} "
        f"(S_pos={oof_ev['doc1_S_pos']:5.2f}/25, S_p0={oof_ev['doc1_S_p0']:5.2f}/20, S_clean={oof_ev['doc1_S_clean']:5.2f}/20)\n"
        f"  OOF Doc2_Strict : {oof_ev['doc2_strict_total']:6.3f} "
        f"(S_pos2={oof_ev['doc2_S_pos']:5.2f}/25, S_p02={oof_ev['doc2_S_p0']:5.2f}/20, Clean2={oof_ev['doc2_Clean2']:5.2f}/20)\n"
        f"  OOF Calibration : ECE={oof_ece:.4f} | Clean_P0_FPR={oof_ev['clean_p0_fpr']:.4f} | "
        f"kappa_p0={decoder.kappa_p0:.3f}, lambda_clean={decoder.lambda_clean:.3f}, lambda_cliff={decoder.lambda_cliff:.3f}",
        flush=True,
    )

    t150_probs, t150_qclean = stacker.predict_proba(t150_models, phys_mat=phys_t150)
    t150_pred = decoder.decode(t150_probs, t150_qclean)

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
