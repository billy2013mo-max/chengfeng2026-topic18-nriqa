#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/common55.py
=======================
Shared, dependency-light helpers for the "next55" plan:

  * official metric (both online-practice `Doc1` and official-spec `Doc2_Strict`)
  * a unified OOF bank: every tower is reduced to a (N, 6, 4) simplex keyed by image name
    (grade order along the last axis = [无问题, P2, P1, P0]; gold encoding 0/1/2/3 identical)
  * the canonical 8-fold, group-constrained split (`data/qingyu_splits/qingyu_8fold.csv`)

Nothing in this file reads test-set labels or legacy submission JSONs (AGENTS.md rule 1/2).
"""
import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE = os.path.join(REPO, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from qingyu.config_and_paths import (  # noqa: E402
    DIMS,
    FULL_DIMS,
    P0_ONLY_DIMS,
    evaluate_qingyu_dual_metrics,
    format_official_submission,
)
from qingyu.certainty_gated_stacker import p012_to_6x4_simplex  # noqa: E402

SPLIT_CSV = os.path.join(REPO, "data/qingyu_splits/qingyu_8fold.csv")
TEMPLATE_150 = os.path.join(REPO, "results/v2_full/template_150.json")
OUT_DIR = os.path.join(REPO, "results/next55")
EPS = 1e-6


# ----------------------------------------------------------------------------------------------
# split / labels
# ----------------------------------------------------------------------------------------------
def load_split() -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Returns names, gold (N,6) int, fold (N,) int, group ids (N,) str."""
    rows = list(csv.DictReader(open(SPLIT_CSV, encoding="utf-8")))
    names = [r["name"] for r in rows]
    gold = np.array([[int(r[d]) for d in DIMS] for r in rows], dtype=np.int64)
    fold = np.array([int(r["fold"]) for r in rows], dtype=np.int64)
    grp = np.array([r["group_id"] for r in rows])
    return names, gold, fold, grp


def load_template_names(path: str = TEMPLATE_150) -> List[str]:
    obj = json.load(open(path, encoding="utf-8"))
    if isinstance(obj, dict):
        obj = obj.get("data", obj.get("items", list(obj.values())))
    return [str(x["image"]) for x in obj]


# ----------------------------------------------------------------------------------------------
# metric
# ----------------------------------------------------------------------------------------------
FINAL_N = 654          # size of the hidden final set (competition doc §7)
TAU = 0.02             # Doc1 clean cliff


def expected_clean_cliff(n_flag: int, n_clean: int, n_total: int, n_final: int = FINAL_N) -> float:
    """E[20 * S_clean_cliff] on a random final-size subset: flagged-clean count ~ Binomial(m, n_flag/n_clean),
    m = expected #clean images in the final set. On a 654 set (~94 clean) ONE P0-on-clean costs ~10.6 pts,
    while on the pooled 3573 OOF it costs only ~2 -> pooled Doc1 badly under-penalises the cliff."""
    from math import comb
    if n_clean == 0:
        return 20.0
    m = max(1, int(round(n_final * n_clean / max(1, n_total))))
    r = n_flag / n_clean
    return float(sum(comb(m, x) * r ** x * (1 - r) ** (m - x) * 20.0 * max(0.0, 1.0 - x / (m * TAU)) for x in range(0, m + 1)))


def score(pred: np.ndarray, gold: np.ndarray) -> Dict[str, float]:
    ev = evaluate_qingyu_dual_metrics(pred, gold)
    ev["blend"] = 0.5 * ev["doc1_total"] + 0.5 * ev["doc2_strict_total"]
    clean = (np.asarray(gold) == 0).all(1)
    n_flag = int(((np.asarray(pred) == 3).any(1) & clean).sum())
    ev["n_clean_p0_flag"] = n_flag
    ev["doc1_final_clean"] = expected_clean_cliff(n_flag, int(clean.sum()), len(gold))
    # Objective most likely used by the website (see handoff §2): Doc1 with the cliff evaluated at final-set scale
    ev["doc1_final"] = ev["doc1_S_pos"] + ev["doc1_S_p0"] + ev["doc1_final_clean"]
    return ev


def score_line(ev: Dict[str, float]) -> str:
    return (
        f"Doc1={ev['doc1_total']:6.3f} (pos {ev['doc1_S_pos']:5.2f} p0 {ev['doc1_S_p0']:5.2f} cl {ev['doc1_S_clean']:5.2f}) | "
        f"Doc2S={ev['doc2_strict_total']:6.3f} (pos {ev['doc2_S_pos']:5.2f} p0 {ev['doc2_S_p0']:5.2f} cl {ev['doc2_Clean2']:5.2f}) | "
        f"P0 R/P={ev['p0_recall']:.3f}/{ev['p0_precision']:.3f} | Doc1@654={ev['doc1_final']:6.3f} (P0-on-clean={ev['n_clean_p0_flag']})"
    )


# ----------------------------------------------------------------------------------------------
# simplex helpers
# ----------------------------------------------------------------------------------------------
def renorm(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=np.float64), EPS, None)
    s[:, P0_ONLY_DIMS, 1] = EPS
    s[:, P0_ONLY_DIMS, 2] = EPS
    return s / s.sum(-1, keepdims=True)


def logit_pool(stack: Sequence[np.ndarray], w: Optional[Sequence[float]] = None) -> np.ndarray:
    """Weighted geometric pooling of simplices (log-linear opinion pool)."""
    w = np.ones(len(stack)) / len(stack) if w is None else np.asarray(w, dtype=np.float64) / np.sum(w)
    lg = sum(wi * np.log(renorm(s)) for wi, s in zip(w, stack))
    lg -= lg.max(-1, keepdims=True)
    return renorm(np.exp(lg))


def mean_pool(stack: Sequence[np.ndarray], w: Optional[Sequence[float]] = None) -> np.ndarray:
    w = np.ones(len(stack)) / len(stack) if w is None else np.asarray(w, dtype=np.float64) / np.sum(w)
    return renorm(sum(wi * renorm(s) for wi, s in zip(w, stack)))


# ----------------------------------------------------------------------------------------------
# OOF bank
# ----------------------------------------------------------------------------------------------
# A "tower" = one model family trained on all 8 folds. Two on-disk formats are understood:
#   qingyu format : <dir>/fold_{k}_probs.npz with va_probs_6x4 / va_names / t150_probs_6x4 / t150_names
#                   (+ optional extra_probs_6x4 / extra_names), optional <dir>/ext_<tag>_probs.npz
#                   written by code/next55/infer_tier1_ckpt.py
#   v16 format    : <dir>/V16_Full_F{k}_*_probs.npz with va_p0/va_p1/va_p2 (+ t150_*, ext_*)
NEXT55_TOWERS = os.path.join(REPO, "results/next55/towers")
QINGYU_TIER1 = ["convnextv2_base", "pe_core_large", "dinov3_vitl16", "siglip2_so400m_naflex", "cradio_v4_h"]


def discover_towers() -> Dict[str, Tuple[str, str]]:
    """key -> (format, dir)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(REPO, "results/qingyu/*/"))):
        if os.path.exists(os.path.join(d, "fold_7_probs.npz")):
            out["qy_" + os.path.basename(d.rstrip("/"))] = ("qingyu", d)
    for d in sorted(glob.glob(os.path.join(NEXT55_TOWERS, "*/*/"))):
        if os.path.exists(os.path.join(d, "fold_7_probs.npz")):
            parts = d.rstrip("/").split("/")
            out[f"{parts[-2]}_{parts[-1]}"] = ("qingyu", d)
    if len(glob.glob(os.path.join(REPO, "results/v16_fulldata/V16_Full_F*_probs.npz"))) == 8:
        out["v16_full8"] = ("v16", os.path.join(REPO, "results/v16_fulldata"))
    for d in sorted(glob.glob(os.path.join(NEXT55_TOWERS, "v16_s*/"))):
        if len(glob.glob(os.path.join(d, "V16_Full_F*_probs.npz"))) == 8:
            out[os.path.basename(d.rstrip("/"))] = ("v16", d)
    return out


def _v16_files(d):
    return sorted(glob.glob(os.path.join(d, "V16_Full_F*_probs.npz")))


def load_oof(fmt: str, d: str) -> Dict[str, np.ndarray]:
    oof = {}
    if fmt == "qingyu":
        for f in sorted(glob.glob(os.path.join(d, "fold_*_probs.npz"))):
            z = np.load(f, allow_pickle=True)
            for nm, v in zip(z["va_names"], z["va_probs_6x4"]):
                oof[str(nm)] = v
    else:
        for f in _v16_files(d):
            z = np.load(f, allow_pickle=True)
            for nm, v in zip(z["va_names"], p012_to_6x4_simplex(z["va_p0"], z["va_p1"], z["va_p2"])):
                oof[str(nm)] = v
    return oof


FULLFIT_ROOT = os.path.join(REPO, "results/next55/fullfit")


def fullfit_dir(key: str) -> Tuple[str, str]:
    """(format, dir) of the 100%-data refit that ships in place of 8-fold tower `key`.
    qingyu towers: results/next55/fullfit/<prefix>/<backbone>/fold_0_*   (prefix 'qy' for results/qingyu/*)
    v16 towers   : results/next55/fullfit/<key>/V16_Full_F0_*"""
    fmt, d = discover_towers()[key]
    if fmt == "v16":
        return fmt, os.path.join(FULLFIT_ROOT, key)
    d = d.rstrip("/")
    prefix = "qy" if os.path.dirname(d) == os.path.join(REPO, "results/qingyu") else os.path.basename(os.path.dirname(d))
    return fmt, os.path.join(FULLFIT_ROOT, prefix, os.path.basename(d))


def load_target_folds(fmt: str, d: str, target: str, min_parts: int = 8) -> Optional[List[Dict[str, np.ndarray]]]:
    """8 per-fold-model test probability maps (one per fold model).
    target = 't150' or an ext tag: ext_<tag>_probs.npz (infer_tier1_ckpt) / ext_<tag>_F*.npz (infer_v16_ckpt) /
    in-training extra_* or ext_* arrays."""
    parts = []
    if fmt == "qingyu":
        side = os.path.join(d, f"ext_{target}_probs.npz")
        if target != "t150" and os.path.exists(side):
            z = np.load(side, allow_pickle=True)
            nm = list(map(str, z["names"]))
            return [dict(zip(nm, z["probs_6x4_folds"][k])) for k in range(z["probs_6x4_folds"].shape[0])]
        for f in sorted(glob.glob(os.path.join(d, "fold_*_probs.npz"))):
            z = np.load(f, allow_pickle=True)
            kp, kn = ("t150_probs_6x4", "t150_names") if target == "t150" else ("extra_probs_6x4", "extra_names")
            if kp not in z.files:
                return None
            parts.append((list(map(str, z[kn])), z[kp]))
    else:
        side = sorted(glob.glob(os.path.join(d, f"ext_{target}_F*.npz"))) if target != "t150" else []
        for f in (side if (side and len(side) == len(_v16_files(d))) else _v16_files(d)):
            z = np.load(f, allow_pickle=True)
            pre = "t150" if target == "t150" else "ext"
            if f"{pre}_p0" not in z.files:
                return None
            parts.append((list(map(str, z[f"{pre}_names"])), p012_to_6x4_simplex(z[f"{pre}_p0"], z[f"{pre}_p1"], z[f"{pre}_p2"])))
    if len(parts) < min_parts:
        return None
    return [dict(zip(nm, v)) for nm, v in parts]


def load_target(fmt: str, d: str, target: str) -> Optional[Dict[str, np.ndarray]]:
    """Fold-averaged (arithmetic mean, same as qingyu aggregate_8fold) test probabilities."""
    folds = load_target_folds(fmt, d, target)
    if folds is None:
        return None
    keys = list(folds[0])
    if any(set(f) != set(keys) for f in folds):
        return None
    return {n: np.mean([f[n] for f in folds], axis=0) for n in keys}


def build_bank(only: Optional[Sequence[str]] = None, min_cov: float = 0.99) -> Dict[str, Dict[str, np.ndarray]]:
    """key -> OOF map. Only towers whose OOF covers >= min_cov of the 8-fold set are kept."""
    names, _, _, _ = load_split()
    bank = {}
    for k, (fmt, d) in discover_towers().items():
        if only and k not in only:
            continue
        oof = load_oof(fmt, d)
        cov = sum(1 for n in names if n in oof) / len(names)
        if cov >= min_cov:
            bank[k] = oof
        else:
            print(f"[bank] skip {k}: OOF coverage {cov*100:.1f}%")
    return bank


def stack_maps(maps: Sequence[Dict[str, np.ndarray]], names: Sequence[str]) -> np.ndarray:
    """(M, N, 6, 4). Missing rows fall back to a 'no problem' prior (should be <1%)."""
    prior = np.full((6, 4), EPS)
    prior[:, 0] = 1.0
    out = np.stack([np.stack([mp.get(n, prior) for n in names]) for mp in maps])
    M, N = out.shape[:2]
    return renorm(out.reshape(-1, 6, 4)).reshape(M, N, 6, 4)


def stack_oof(bank, keys: Sequence[str], names: Sequence[str]) -> np.ndarray:
    return stack_maps([bank[k] for k in keys], names)
