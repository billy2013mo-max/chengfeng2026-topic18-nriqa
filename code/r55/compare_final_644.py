#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/compare_final_644.py
=============================
Comprehensive, zero-label forensic diff & Clean-safety audit comparing:
  1) `ref` (53.99 Process-Review Baseline `resident_v6`):
     `results/final/pred_nriqa_dataset_final_644_r55_resident_v6_routed_gate_eu.json`
  2) `v7_base` (11-member `resident_v7` + `routed_gate_eu`, adding `pe_bkseg1` & `ntg_convnext`):
     `results/r55/os_og_breakthrough_644/pred_nriqa_dataset_final_644_os_og_breakthrough_v7_baseline_fresh.json`
  3) `new` (Current Best `resident_v7` + `deconf_eu` Cross-Defect De-confounding):
     `results/r55/os_og_breakthrough_644/pred_nriqa_dataset_final_644_os_og_breakthrough.json`

Read-only diagnostic tool: NEVER modifies any prediction JSON.
"""
import argparse
import collections
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
GRADES_4 = ["P0", "P1", "P2", "无问题"]
GRADE_TO_INT = {"无问题": 0, "P2": 1, "P1": 2, "P0": 3}


def _load_pred_map(path: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid prediction JSON: {path}")
    order = [str(r["image"]) for r in rows]
    mp = {str(r["image"]): dict(r["label"]) for r in rows}
    if len(order) != len(mp):
        raise ValueError(f"Duplicate image IDs in {path}")
    return order, mp


def _is_clean(lbl: Dict[str, str]) -> bool:
    return all(lbl[d] == "无问题" for d in DIMS)


def _has_p0(lbl: Dict[str, str]) -> bool:
    return any(lbl[d] == "P0" for d in DIMS)


def _pair_stats(
    names: List[str],
    mp_a: Dict[str, Dict[str, str]],
    mp_b: Dict[str, Dict[str, str]],
) -> Dict[str, object]:
    n_imgs = len(names)
    n_cells = n_imgs * len(DIMS)
    changed_cells = 0
    changed_imgs = 0
    per_dim_changed = {d: 0 for d in DIMS}
    per_dim_transitions = {d: collections.Counter() for d in DIMS}
    img_diffs = []

    for nm in names:
        la, lb = mp_a[nm], mp_b[nm]
        diffs_nm = {}
        for d in DIMS:
            va, vb = la[d], lb[d]
            if va != vb:
                changed_cells += 1
                per_dim_changed[d] += 1
                per_dim_transitions[d][f"{va}->{vb}"] += 1
                diffs_nm[d] = {"old": va, "new": vb}
        if diffs_nm:
            changed_imgs += 1
            img_diffs.append(
                {
                    "image": nm,
                    "num_changed_dims": len(diffs_nm),
                    "old_clean": _is_clean(la),
                    "new_clean": _is_clean(lb),
                    "changes": diffs_nm,
                    "old_label": la,
                    "new_label": lb,
                }
            )

    clean_a = {nm for nm in names if _is_clean(mp_a[nm])}
    clean_b = {nm for nm in names if _is_clean(mp_b[nm])}
    clean_to_nonclean = sorted(clean_a - clean_b)
    nonclean_to_clean = sorted(clean_b - clean_a)
    clean_to_p0 = [nm for nm in clean_to_nonclean if _has_p0(mp_b[nm])]
    p0_to_clean = [nm for nm in nonclean_to_clean if _has_p0(mp_a[nm])]

    dist_a = {d: collections.Counter(mp_a[nm][d] for nm in names) for d in DIMS}
    dist_b = {d: collections.Counter(mp_b[nm][d] for nm in names) for d in DIMS}

    return {
        "n_images": n_imgs,
        "n_cells": n_cells,
        "changed_cells": changed_cells,
        "changed_cells_pct": round(100.0 * changed_cells / max(1, n_cells), 2),
        "changed_images": changed_imgs,
        "changed_images_pct": round(100.0 * changed_imgs / max(1, n_imgs), 2),
        "clean_old": len(clean_a),
        "clean_new": len(clean_b),
        "clean_shared": len(clean_a & clean_b),
        "clean_to_nonclean": clean_to_nonclean,
        "nonclean_to_clean": nonclean_to_clean,
        "clean_to_any_p0": clean_to_p0,
        "any_p0_to_clean": p0_to_clean,
        "per_dim_changed": per_dim_changed,
        "per_dim_transitions": {
            d: dict(per_dim_transitions[d].most_common()) for d in DIMS
        },
        "dist_old": {d: {g: int(dist_a[d].get(g, 0)) for g in GRADES_4} for d in DIMS},
        "dist_new": {d: {g: int(dist_b[d].get(g, 0)) for g in GRADES_4} for d in DIMS},
        "image_diffs": img_diffs,
    }


def _fmt_delta(v: int) -> str:
    return f"+{v}" if v > 0 else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref",
        default="results/final/pred_nriqa_dataset_final_644_r55_resident_v6_routed_gate_eu.json",
        help="53.99 process-review baseline JSON (resident_v6)",
    )
    ap.add_argument(
        "--new",
        required=True,
        help="New primary 644 prediction JSON (resident_v7 + os_og_breakthrough)",
    )
    ap.add_argument(
        "--v7-base",
        default="",
        help="Optional fresh resident_v7 baseline JSON (without deconf_eu) for 2-stage attribution",
    )
    ap.add_argument(
        "--probs-npz",
        default="",
        help="Optional *_calibrated_probs.npz from multi_gpu_infer.py to display Q_base -> Q_cal probabilities",
    )
    ap.add_argument(
        "--out-report",
        default="",
        help="Optional path to save full JSON diff report",
    )
    args = ap.parse_args()

    ref_p = args.ref if os.path.isabs(args.ref) else os.path.join(REPO, args.ref)
    new_p = args.new if os.path.isabs(args.new) else os.path.join(REPO, args.new)
    v7b_p = (args.v7_base if os.path.isabs(args.v7_base) else os.path.join(REPO, args.v7_base)) if args.v7_base else ""
    npz_p = (args.probs_npz if os.path.isabs(args.probs_npz) else os.path.join(REPO, args.probs_npz)) if args.probs_npz else ""

    assert os.path.isfile(ref_p), f"Missing 53.99 reference JSON: {ref_p}"
    assert os.path.isfile(new_p), f"Missing new 644 JSON: {new_p}"

    names_ref, mp_ref = _load_pred_map(ref_p)
    names_new, mp_new = _load_pred_map(new_p)
    if set(names_ref) != set(names_new):
        missing = set(names_ref) - set(names_new)
        extra = set(names_new) - set(names_ref)
        raise ValueError(f"ID mismatch between ref and new: missing={len(missing)}, extra={len(extra)}")

    names = names_ref
    st_main = _pair_stats(names, mp_ref, mp_new)

    st_v6_to_v7b: Optional[Dict[str, object]] = None
    st_v7b_to_new: Optional[Dict[str, object]] = None
    if v7b_p and os.path.isfile(v7b_p):
        _, mp_v7b = _load_pred_map(v7b_p)
        st_v6_to_v7b = _pair_stats(names, mp_ref, mp_v7b)
        st_v7b_to_new = _pair_stats(names, mp_v7b, mp_new)

    q_base_map: Dict[str, Any] = {}
    q_cal_map: Dict[str, Any] = {}
    if npz_p and os.path.isfile(npz_p):
        import numpy as np
        z = np.load(npz_p, allow_pickle=True)
        for nm, qb, qc in zip(z["names"], z["Q_base"], z["Q_cal"]):
            q_base_map[str(nm)] = np.asarray(qb, dtype=np.float64)
            q_cal_map[str(nm)] = np.asarray(qc, dtype=np.float64)

    print("\n" + "=" * 116)
    print("=== [644 Final Eval Diff Audit] Current Model (v7 + deconf_eu) vs 53.99 Baseline (resident_v6) ===")
    print("=" * 116)
    print(f"  53.99 Baseline (ref) : {os.path.relpath(ref_p, REPO)}")
    if st_v6_to_v7b is not None:
        print(f"  v7 Pure Base         : {os.path.relpath(v7b_p, REPO)}")
    print(f"  Current Model (new)  : {os.path.relpath(new_p, REPO)}")
    print("-" * 116)
    print(
        f"  [1] Total Diff (53.99 v6 -> new)  : {st_main['changed_cells']} / {st_main['n_cells']} cells "
        f"({st_main['changed_cells_pct']:.2f}%), across {st_main['changed_images']} / {st_main['n_images']} images "
        f"({st_main['changed_images_pct']:.2f}%)"
    )
    if st_v6_to_v7b is not None and st_v7b_to_new is not None:
        print(
            f"      ├─ Step A (53.99 v6 -> v7_base: +pe_bkseg1 +ntg_convnext) : "
            f"{st_v6_to_v7b['changed_cells']} / {st_v6_to_v7b['n_cells']} cells "
            f"({st_v6_to_v7b['changed_images']} imgs)"
        )
        print(
            f"      └─ Step B (v7_base -> new: +过度生成 Cross-Defect Deconf)  : "
            f"{st_v7b_to_new['changed_cells']} / {st_v7b_to_new['n_cells']} cells "
            f"({st_v7b_to_new['changed_images']} imgs)"
        )

    print(
        f"  [2] Clean-Score Safety (20/20 Gate): "
        f"All-Clean count {st_main['clean_old']} (v6) -> {st_main['clean_new']} (new) "
        f"[{_fmt_delta(st_main['clean_new'] - st_main['clean_old'])}, shared={st_main['clean_shared']}]"
    )
    print(
        f"      ├─ v6 Clean -> new Non-Clean   : {len(st_main['clean_to_nonclean'])} imgs "
        f"| CRITICAL v6 Clean -> new Any-P0: {len(st_main['clean_to_any_p0'])} imgs"
    )
    print(
        f"      └─ v6 Non-Clean -> new Clean   : {len(st_main['nonclean_to_clean'])} imgs "
        f"| v6 Any-P0 -> new Clean: {len(st_main['any_p0_to_clean'])} imgs"
    )

    print("\n" + "-" * 116)
    print(
        f"{'维度':<8} | {'改动格数':>8} | {'P0 (v6->new)':>16} | {'P1 (v6->new)':>16} | "
        f"{'P2 (v6->new)':>16} | {'无问题 (v6->new)':>18} | 主要翻转流向 (Top 4)"
    )
    print("-" * 116)
    for d in DIMS:
        do = st_main["dist_old"][d]
        dn = st_main["dist_new"][d]
        chg = st_main["per_dim_changed"][d]

        def _cell_s(g: str) -> str:
            if d in ("花屏", "虚化突兀") and g in ("P1", "P2"):
                return "—"
            vo, vn = do[g], dn[g]
            return f"{vo:3d}->{vn:3d}({_fmt_delta(vn - vo)})"

        top_tr = ", ".join(
            f"{k}:{v}" for k, v in list(st_main["per_dim_transitions"][d].items())[:4]
        ) or "无变化"
        print(
            f"{d:<8} | {chg:5d}/644 | {_cell_s('P0'):>16} | {_cell_s('P1'):>16} | "
            f"{_cell_s('P2'):>16} | {_cell_s('无问题'):>18} | {top_tr}"
        )
    print("-" * 116)

    if st_v7b_to_new is not None:
        print(
            f"\n=== [Step B Detail] Exact Cells Flipped by `deconf_eu` (v7_base -> new): "
            f"{st_v7b_to_new['changed_cells']} cells across {st_v7b_to_new['changed_images']} images ==="
        )
        if not st_v7b_to_new["image_diffs"]:
            print("  (0 cells differed between v7_base and new on 644 set)")
        for item in st_v7b_to_new["image_diffs"]:
            nm = item["image"]
            v6_lbl = mp_ref[nm]
            for d, ch in item["changes"].items():
                d_idx = DIMS.index(d)
                prob_str = ""
                if nm in q_base_map and nm in q_cal_map:
                    qb = [round(float(x), 3) for x in q_base_map[nm][d_idx]]
                    qc = [round(float(x), 3) for x in q_cal_map[nm][d_idx]]
                    prob_str = f" | Q_base[无,P2,P1,P0]={qb} -> Q_cal={qc}"
                print(
                    f"  - {nm:<52} [{d}] 53.99_v6={v6_lbl[d]} | v7_base={ch['old']} -> new={ch['new']}{prob_str}"
                )

    bokeh_diffs = [
        it for it in st_main["image_diffs"] if "虚化突兀" in it["changes"]
    ]
    print(f"\n=== [Focus Audit: 虚化突兀] 53.99 v6 -> new changed {len(bokeh_diffs)} images ===")
    for it in bokeh_diffs[:20]:
        nm = it["image"]
        ch = it["changes"]["虚化突兀"]
        p_s = ""
        if nm in q_cal_map:
            p_s = f" | new P(虚化_P0)={float(q_cal_map[nm][5, 3]):.3f}"
        print(f"  - {nm:<52} : {ch['old']} -> {ch['new']}{p_s}")
    if len(bokeh_diffs) > 20:
        print(f"  ... ({len(bokeh_diffs) - 20} more in JSON report)")

    if st_main["clean_to_any_p0"]:
        print(f"\n[WARN] Images predicted All-Clean in 53.99 v6 that became P0 in new ({len(st_main['clean_to_any_p0'])}):")
        for nm in st_main["clean_to_any_p0"]:
            print(f"  - {nm}: new_label={mp_new[nm]}")

    if q_cal_map:
        import numpy as np
        q_stack = np.stack([q_cal_map[nm] for nm in names], axis=0)
        max_p = np.max(q_stack, axis=-1)
        print("\n=== [644 Calibrated Confidence Summary (new Q_cal)] ===")
        print(f"  Overall mean max-prob (6x644): {float(np.mean(max_p)):.4f} (median {float(np.median(max_p)):.4f})")
        for di, d in enumerate(DIMS):
            mp_d = max_p[:, di]
            print(
                f"  - {d:<6}: mean_max_p={float(np.mean(mp_d)):.4f} | "
                f">=0.80: {float(np.mean(mp_d >= 0.80) * 100.0):5.1f}% | "
                f">=0.90: {float(np.mean(mp_d >= 0.90) * 100.0):5.1f}%"
            )

    report_payload = {
        "ref_5399_path": ref_p,
        "v7_base_path": v7b_p,
        "new_primary_path": new_p,
        "diff_v6_5399_to_new": st_main,
        "diff_v6_5399_to_v7_base": st_v6_to_v7b,
        "diff_v7_base_to_new_deconf": st_v7b_to_new,
    }
    if args.out_report:
        rep_abs = args.out_report if os.path.isabs(args.out_report) else os.path.join(REPO, args.out_report)
        os.makedirs(os.path.dirname(rep_abs), exist_ok=True)
        with open(rep_abs, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)
        print(f"\n[compare_final_644] full diff report saved -> {rep_abs}")
    print("=" * 116 + "\n", flush=True)


if __name__ == "__main__":
    main()
