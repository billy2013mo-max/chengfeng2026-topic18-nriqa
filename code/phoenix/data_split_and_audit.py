#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/phoenix/data_split_and_audit.py
====================================
Breakthrough 1: Zero-Leakage Canonical Video-ID + Visual Graph Connected-Component
5-Fold Stratification, Timestamp Duplicate Noise Detector, and Batch-Shift Auditor.

Why This Module Replaces Legacy Splits:
  1. Fixes the critical `manifest_combined.csv` prefix leakage bug where filenames like
     `0016__211661657305_v8L3HiV6_24.99s_frame_02_虚化突兀.png` or
     `002_212299294787_75.83s.png` were grouped under `'0016'` / `'002'` instead of the
     true Kuaishou video ID `'211661657305'` / `'212299294787'`, leaking identical videos
     across train and validation folds!
  2. Optionally merges any pair of images whose precomputed DINOv3/SigLIP2 visual cosine
     similarity exceeds `sim_threshold=0.90` via Union-Find connected components.
  3. Quantifies the systematic annotator log-odds shift between Batch-1 (`orig`, 1,902)
     and Batch-2 (`supp`, 1,671) to initialize `BatchShiftCalibration` priors, and flags
     same-video near-duplicate timestamp label conflicts for heteroscedastic soft-label
     variance boosting.
  4. Runs seamlessly on BOTH GPU servers (with NumPy) AND bare stdlib Python 3 environments.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(THIS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(THIS_DIR))

from phoenix.config import (
    CHANNEL_14_SPECS,
    DIMS,
    FULL_DIMS,
    PHOENIX_SPLITS_DIR,
    load_real_manifest,
    set_deterministic_seed,
)

# Regex patterns ordered from most specific to fallback
_RE_INDEX_PREFIXED_VID = re.compile(r"^\d{1,5}__?(\d{9,22})[_.]")
_RE_LONG_NUMERIC_VID = re.compile(r"^(\d{9,22})(?:[_.-]|\.mp4|$)")
_RE_UUID_PREFIX = re.compile(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)
_RE_HEX_HASH_VID = re.compile(r"^([0-9a-f]{14,20})[_.-]", re.I)
_RE_NATURAL_PACK = re.compile(
    r"^(blinds_bokeh|fj_grass|light_effect|forest|grass|greenc|greens|indoor|reception|glitch)[_.]",
    re.I,
)
_RE_TIMESTAMP_SEC = re.compile(r"(?:_|-)(\d+(?:[._]\d+)?)s(?:[_.]|$)")


def extract_canonical_video_group(filename: str, raw_source: str = "") -> str:
    """
    Extracts the true underlying video/scene source identifier from Kuaishou frame filenames,
    eliminating serial-index prefix bugs (`0016__211661657305_...`, `002_212299294787_...`).
    """
    base = os.path.basename(filename.strip())

    m = _RE_INDEX_PREFIXED_VID.match(base)
    if m:
        return f"vid_{m.group(1)}"

    m = _RE_LONG_NUMERIC_VID.match(base)
    if m:
        return f"vid_{m.group(1)}"

    m = _RE_UUID_PREFIX.match(base)
    if m:
        return f"uuid_{m.group(1).lower()}"

    m = _RE_HEX_HASH_VID.match(base)
    if m:
        return f"hex_{m.group(1).lower()}"

    m = _RE_NATURAL_PACK.match(base)
    if m:
        stem = os.path.splitext(base)[0]
        digits = re.findall(r"\d+", stem)
        bucket = int(digits[-1]) // 10 if digits else 0
        return f"pack_{m.group(1).lower()}_{bucket:03d}"

    any_vid = re.search(r"(?<!\d)(\d{10,20})(?!\d)", base)
    if any_vid:
        return f"vid_{any_vid.group(1)}"

    stem = os.path.splitext(base)[0]
    if raw_source and not raw_source.isdigit():
        return f"src_{raw_source}"
    return f"misc_{stem}"


def extract_timestamp_or_frame_hint(filename: str) -> Optional[float]:
    """Extracts approximate timestamp (seconds) if encoded in filename for duplicate auditing."""
    base = os.path.basename(filename)
    m = _RE_TIMESTAMP_SEC.search(base)
    if m:
        try:
            return float(m.group(1).replace("_", "."))
        except ValueError:
            return None
    return None


class UnionFind:
    """Disjoint-set structure for merging video-ID groups with high-similarity visual pairs."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def merge_visual_similarity_groups(
    names: Sequence[str],
    initial_groups: Sequence[str],
    emb_npz_path: Optional[str] = None,
    sim_threshold: float = 0.90,
) -> Tuple[List[str], int]:
    """
    Builds connected components combining canonical video IDs AND optional visual cosine
    similarity edges (`cos_sim >= sim_threshold`) from an offline feature cache if available.
    """
    n = len(names)
    uf = UnionFind(n)

    by_group: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(initial_groups):
        by_group[g].append(i)
    for idxs in by_group.values():
        first = idxs[0]
        for j in idxs[1:]:
            uf.union(first, j)

    visual_merges = 0
    if np is not None and emb_npz_path and os.path.exists(emb_npz_path):
        try:
            cache = np.load(emb_npz_path, allow_pickle=True)
            if "names" in cache and "feats" in cache:
                c_names = [str(x) for x in cache["names"]]
                c_feats = np.asarray(cache["feats"], dtype=np.float32)
                pos = {nm: idx for idx, nm in enumerate(c_names)}
                valid_idx = [i for i, nm in enumerate(names) if nm in pos]
                if len(valid_idx) >= 100:
                    mat = np.stack([c_feats[pos[names[i]]] for i in valid_idx], axis=0)
                    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
                    mat = mat / norms
                    sim = mat @ mat.T
                    ii, jj = np.where(np.triu(sim, k=1) >= sim_threshold)
                    for a, b in zip(ii.tolist(), jj.tolist()):
                        ia, ib = valid_idx[a], valid_idx[b]
                        if uf.find(ia) != uf.find(ib):
                            uf.union(ia, ib)
                            visual_merges += 1
        except Exception:
            visual_merges = 0

    root_to_gid: Dict[int, str] = {}
    final_groups: List[str] = []
    for i in range(n):
        r = uf.find(i)
        if r not in root_to_gid:
            root_to_gid[r] = initial_groups[r]
        final_groups.append(root_to_gid[r])
    return final_groups, visual_merges


def iterative_multilabel_group_kfold(
    y_rows: Sequence[Sequence[int]],
    batch_ids: Sequence[int],
    groups: Sequence[str],
    n_splits: int = 5,
    seed: int = 2026,
) -> List[int]:
    """
    Stratifies connected scene groups into `n_splits` folds balancing:
      - All 14 positive defect×grade channels
      - Clean (all-zero) images
      - Batch-1 (`orig`) vs Batch-2 (`supp`) indicators
      - Total image counts per fold
    Guarantees 100% zero group overlap across folds.
    """
    set_deterministic_seed(seed)
    n = len(y_rows)

    # 17 stratification targets per sample: 14 pos channels + Clean + orig + supp
    strat_rows: List[List[float]] = []
    col_totals = [0.0] * 17
    for i in range(n):
        row_vec = [0.0] * 17
        yi = y_rows[i]
        for ch_idx, (_, d_idx, g_val) in enumerate(CHANNEL_14_SPECS):
            if yi[d_idx] == g_val:
                row_vec[ch_idx] = 1.0
        if all(v == 0 for v in yi):
            row_vec[14] = 1.0
        if batch_ids[i] == 0:
            row_vec[15] = 1.0
        else:
            row_vec[16] = 1.0
        for c in range(17):
            col_totals[c] += row_vec[c]
        strat_rows.append(row_vec)

    col_totals = [max(v, 1.0) for v in col_totals]
    inv_freq = [1.0 / v for v in col_totals]

    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_indices[g].append(i)

    unique_groups = list(group_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    group_sums: Dict[str, List[float]] = {}
    for g_key, idxs in group_to_indices.items():
        s = [0.0] * 17
        for idx in idxs:
            sr = strat_rows[idx]
            for c in range(17):
                s[c] += sr[c]
        group_sums[g_key] = s

    def group_priority(g_key: str) -> Tuple[float, int]:
        g_vec = group_sums[g_key]
        rarity = max((g_vec[c] * inv_freq[c] for c in range(17)), default=0.0)
        return (rarity, len(group_to_indices[g_key]))

    unique_groups.sort(key=group_priority, reverse=True)

    desired_per_fold = [v / float(n_splits) for v in col_totals]
    sqrt_desired = [math.sqrt(v) for v in desired_per_fold]
    desired_size = n / float(n_splits)
    sqrt_size = math.sqrt(desired_size)

    fold_counts = [[0.0] * 17 for _ in range(n_splits)]
    fold_sizes = [0.0] * n_splits
    folds = [-1] * n

    for g_key in unique_groups:
        idxs = group_to_indices[g_key]
        g_vec = group_sums[g_key]
        g_sz = float(len(idxs))
        active_cols = [c for c in range(17) if g_vec[c] > 0.0]

        best_f = 0
        best_cost = float("inf")
        for f in range(n_splits):
            delta_label_cost = 0.0
            fc = fold_counts[f]
            for c in active_cols:
                cur_rel = (fc[c] - desired_per_fold[c]) / sqrt_desired[c]
                new_rel = (fc[c] + g_vec[c] - desired_per_fold[c]) / sqrt_desired[c]
                delta_label_cost += new_rel * new_rel - cur_rel * cur_rel

            cur_sz_rel = (fold_sizes[f] - desired_size) / sqrt_size
            new_sz_rel = (fold_sizes[f] + g_sz - desired_size) / sqrt_size
            size_cost = 0.35 * (new_sz_rel * new_sz_rel - cur_sz_rel * cur_sz_rel)

            total_cost = delta_label_cost + size_cost + 1e-6 * fold_sizes[f]
            if total_cost < best_cost:
                best_cost = total_cost
                best_f = f

        for idx in idxs:
            folds[idx] = best_f
        for c in range(17):
            fold_counts[best_f][c] += g_vec[c]
        fold_sizes[best_f] += g_sz

    assert all(f >= 0 for f in folds), "Unassigned sample in K-fold split!"
    for g_key, idxs in group_to_indices.items():
        assigned = {folds[idx] for idx in idxs}
        if len(assigned) != 1:
            raise AssertionError(f"Group leakage detected for {g_key}: folds={assigned}")
    return folds


def audit_batch_shift_and_timestamp_conflicts(
    records: Sequence[Dict[str, object]],
    groups: Sequence[str],
) -> Tuple[Dict[str, object], List[int]]:
    """
    1. Computes empirical grade priors for Batch-1 (`orig`) vs Batch-2 (`supp`) and their
       smoothed log-odds shift matrix `[6, 4]` for `BatchShiftCalibration`.
    2. Detects same-video timestamp collisions (`<= 0.25s` apart) with conflicting human labels
       so training automatically boosts soft-label sigma on noisy boundary frames.
    """
    n = len(records)
    y_rows: List[List[int]] = [list(r["grades"]) for r in records]  # type: ignore[arg-type]
    batch_ids: List[int] = [int(r["batch_id"]) for r in records]

    orig_indices = [i for i, b in enumerate(batch_ids) if b == 0]
    supp_indices = [i for i, b in enumerate(batch_ids) if b == 1]
    n_orig = max(1, len(orig_indices))
    n_supp = max(1, len(supp_indices))

    prior_orig = [[0.0] * 4 for _ in range(6)]
    prior_supp = [[0.0] * 4 for _ in range(6)]
    log_odds_shift_supp_vs_orig = [[0.0] * 4 for _ in range(6)]

    batch_table: Dict[str, Dict[str, object]] = {}
    alpha_smooth = 2.0
    for d_idx, d_name in enumerate(DIMS):
        valid_grades = [0, 1, 2, 3] if d_idx in FULL_DIMS else [0, 3]
        denom_k = float(len(valid_grades))
        for g in valid_grades:
            cnt_o = sum(1 for i in orig_indices if y_rows[i][d_idx] == g)
            cnt_s = sum(1 for i in supp_indices if y_rows[i][d_idx] == g)
            p_o = (cnt_o + alpha_smooth) / (n_orig + alpha_smooth * denom_k)
            p_s = (cnt_s + alpha_smooth) / (n_supp + alpha_smooth * denom_k)
            prior_orig[d_idx][g] = p_o
            prior_supp[d_idx][g] = p_s
            log_odds_shift_supp_vs_orig[d_idx][g] = math.log(p_s / p_o)
        batch_table[d_name] = {
            "orig_dist": [round(prior_orig[d_idx][g], 4) for g in range(4)],
            "supp_dist": [round(prior_supp[d_idx][g], 4) for g in range(4)],
            "log_ratio_supp_over_orig": [
                round(log_odds_shift_supp_vs_orig[d_idx][g], 4) for g in range(4)
            ],
        }

    # Detect same-video close-timestamp label conflicts (<= 0.25s apart with grade diff)
    by_vid_ts: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for i, (r, g_id) in enumerate(zip(records, groups)):
        if not g_id.startswith("vid_"):
            continue
        ts = extract_timestamp_or_frame_hint(str(r["name"]))
        if ts is not None:
            by_vid_ts[g_id].append((ts, i))

    noisy_sample_boost = [0] * n
    conflict_pairs = []
    for g_id, ts_list in by_vid_ts.items():
        if len(ts_list) < 2:
            continue
        ts_list.sort(key=lambda x: x[0])
        for a in range(len(ts_list)):
            for b in range(a + 1, len(ts_list)):
                dt = abs(ts_list[b][0] - ts_list[a][0])
                if dt > 0.25:
                    break
                ia, ib = ts_list[a][1], ts_list[b][1]
                if y_rows[ia] != y_rows[ib]:
                    noisy_sample_boost[ia] = 1
                    noisy_sample_boost[ib] = 1
                    conflict_pairs.append(
                        {
                            "group": g_id,
                            "dt_sec": round(dt, 3),
                            "img_a": str(records[ia]["name"]),
                            "grades_a": y_rows[ia],
                            "img_b": str(records[ib]["name"]),
                            "grades_b": y_rows[ib],
                        }
                    )

    canon_to_raw_srcs: Dict[str, set] = defaultdict(set)
    for r, g_id in zip(records, groups):
        canon_to_raw_srcs[g_id].add(str(r["raw_source"]))
    fixed_prefix_leaks = sum(len(s) - 1 for s in canon_to_raw_srcs.values() if len(s) > 1)

    clean_orig = sum(1 for i in orig_indices if all(v == 0 for v in y_rows[i])) / float(n_orig)
    clean_supp = sum(1 for i in supp_indices if all(v == 0 for v in y_rows[i])) / float(n_supp)
    clean_all = sum(1 for i in range(n) if all(v == 0 for v in y_rows[i])) / float(max(1, n))

    audit_summary = {
        "n_total_real": n,
        "n_orig_batch1": len(orig_indices),
        "n_supp_batch2": len(supp_indices),
        "n_unique_canonical_groups": len(set(groups)),
        "n_fixed_legacy_prefix_leak_clusters": fixed_prefix_leaks,
        "n_close_timestamp_label_conflicts": len(conflict_pairs),
        "conflict_examples": conflict_pairs[:15],
        "clean_rate_orig": round(clean_orig, 4),
        "clean_rate_supp": round(clean_supp, 4),
        "clean_rate_overall": round(clean_all, 4),
        "batch_distribution_shift": batch_table,
        "log_odds_shift_supp_vs_orig": log_odds_shift_supp_vs_orig,
    }
    return audit_summary, noisy_sample_boost


def build_and_save_phoenix_splits(
    manifest_path: str = "data/manifest_combined.csv",
    out_dir: str = str(PHOENIX_SPLITS_DIR),
    n_splits: int = 5,
    seed: int = 2026,
    emb_npz_path: Optional[str] = None,
) -> Dict[str, object]:
    records = load_real_manifest(manifest_path)
    names = [str(r["name"]) for r in records]
    canon_groups = [extract_canonical_video_group(str(r["name"]), str(r["raw_source"])) for r in records]
    merged_groups, n_vis_merges = merge_visual_similarity_groups(
        names=names,
        initial_groups=canon_groups,
        emb_npz_path=emb_npz_path,
        sim_threshold=0.90,
    )

    y_rows: List[List[int]] = [list(r["grades"]) for r in records]  # type: ignore[arg-type]
    batch_ids = [int(r["batch_id"]) for r in records]
    folds = iterative_multilabel_group_kfold(
        y_rows=y_rows,
        batch_ids=batch_ids,
        groups=merged_groups,
        n_splits=n_splits,
        seed=seed,
    )

    audit_summary, noisy_boost = audit_batch_shift_and_timestamp_conflicts(records, merged_groups)
    audit_summary["n_visual_similarity_merges"] = n_vis_merges
    audit_summary["n_splits"] = n_splits

    # Verify every fold has >0 positives in all 14 channels & non-empty Clean set
    fold_stats = []
    for f in range(n_splits):
        f_indices = [i for i, fv in enumerate(folds) if fv == f]
        ch_counts = {
            ch_name: sum(1 for i in f_indices if y_rows[i][d_idx] == g_val)
            for ch_name, d_idx, g_val in CHANNEL_14_SPECS
        }
        min_pos = min(ch_counts.values())
        n_clean_f = sum(1 for i in f_indices if all(v == 0 for v in y_rows[i]))
        assert min_pos > 0, f"Fold {f} has an empty positive channel: {ch_counts}"
        assert n_clean_f > 0, f"Fold {f} has zero Clean samples!"
        fold_stats.append(
            {
                "fold": f,
                "n_samples": len(f_indices),
                "n_groups": len({merged_groups[i] for i in f_indices}),
                "n_clean": n_clean_f,
                "n_orig": sum(1 for i in f_indices if batch_ids[i] == 0),
                "n_supp": sum(1 for i in f_indices if batch_ids[i] == 1),
                "min_14ch_positives": min_pos,
                "channel_counts": ch_counts,
            }
        )
    audit_summary["fold_stats"] = fold_stats

    os.makedirs(out_dir, exist_ok=True)
    split_csv = os.path.join(out_dir, f"phoenix_{n_splits}fold.csv")
    fieldnames = [
        "name",
        "canon_group",
        "fold",
        "batch_id",
        "split",
        "width",
        "height",
        "noisy_boundary_flag",
        *DIMS,
    ]
    with open(split_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(records):
            writer.writerow(
                {
                    "name": r["name"],
                    "canon_group": merged_groups[i],
                    "fold": int(folds[i]),
                    "batch_id": int(r["batch_id"]),
                    "split": r["split"],
                    "width": r["width"],
                    "height": r["height"],
                    "noisy_boundary_flag": int(noisy_boost[i] > 0),
                    **{c: int(r[c]) for c in DIMS},
                }
            )

    audit_json = os.path.join(out_dir, "batch_shift_and_audit.json")
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, ensure_ascii=False, indent=2)

    print("=" * 98)
    print(f"[Phoenix Split & Audit] Saved {n_splits}-Fold Zero-Leakage Split -> {split_csv}")
    print(
        f"  Total Real={audit_summary['n_total_real']} (orig={audit_summary['n_orig_batch1']}, "
        f"supp={audit_summary['n_supp_batch2']}) | Canonical Groups={audit_summary['n_unique_canonical_groups']} "
        f"| Fixed Prefix Leaks={audit_summary['n_fixed_legacy_prefix_leak_clusters']} "
        f"| Close-TS Conflicts={audit_summary['n_close_timestamp_label_conflicts']}"
    )
    print(
        f"  Clean Rate: orig={audit_summary['clean_rate_orig']*100:.2f}% vs "
        f"supp={audit_summary['clean_rate_supp']*100:.2f}% (overall={audit_summary['clean_rate_overall']*100:.2f}%)"
    )
    for fs in fold_stats:
        print(
            f"  Fold {fs['fold']}: N={fs['n_samples']:4d} | Groups={fs['n_groups']:4d} | "
            f"Clean={fs['n_clean']:3d} | orig/supp={fs['n_orig']:3d}/{fs['n_supp']:3d} | "
            f"min_14ch_pos={fs['min_14ch_positives']:2d}"
        )
    print("=" * 98)
    return audit_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix 5-Fold Zero-Leakage Video-Group Split & Batch Audit")
    parser.add_argument("--manifest", default="data/manifest_combined.csv")
    parser.add_argument("--out-dir", default=str(PHOENIX_SPLITS_DIR))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--emb-npz", default="", help="Optional DINOv3 feature .npz for visual graph union")
    args = parser.parse_args()
    build_and_save_phoenix_splits(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        n_splits=args.n_splits,
        seed=args.seed,
        emb_npz_path=args.emb_npz or None,
    )


if __name__ == "__main__":
    main()
