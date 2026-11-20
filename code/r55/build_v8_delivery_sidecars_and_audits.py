#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_v8_delivery_sidecars_and_audits.py
=================================================
Generates the full suite of zero-leakage OOF audit JSONs, video aggregator,
cost-Pareto table, and 100% label-locked subjective-score sidecars
(attribution JSON + contact sheet + 极目/TAT payload + video summary) for
`resident_v8_boundary` + `deconf_v6_boundary` (Online 54.77 / 65).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/finetune", "code/decision_eval", "code/r55"):
    _ap = os.path.join(REPO, _p)
    if os.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)

DIMS_ZH = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
GRADE_INT_TO_STR = {0: "无问题", 1: "P2", 2: "P1", 3: "P0"}
GRADE_STR_TO_INT = {v: k for k, v in GRADE_INT_TO_STR.items()}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_grid(path: str, names: List[str]) -> np.ndarray:
    rows = json.load(open(path, "r", encoding="utf-8"))
    mp_d = {str(r["image"]): r["label"] for r in rows}
    g = np.zeros((len(names), 6), dtype=np.int64)
    for i, nm in enumerate(names):
        lbl = mp_d[str(nm)]
        for d, dn in enumerate(DIMS_ZH):
            g[i, d] = GRADE_STR_TO_INT[str(lbl[dn])]
    return g


def _compute_map14(probs: np.ndarray, gold: np.ndarray) -> float:
    aps = []
    for d in range(6):
        gs = (1, 2, 3) if d in (0, 1, 2, 4) else (3,)
        for g in gs:
            yb = (gold[:, d] == g).astype(int)
            if 0 < yb.sum() < len(yb):
                aps.append(float(average_precision_score(yb, probs[:, d, g])))
    return float(np.mean(aps))


def _compute_rps_6x4(probs: np.ndarray, gold: np.ndarray) -> float:
    oh = np.eye(4)[gold]
    return float(((np.cumsum(probs, axis=-1)[..., :3] - np.cumsum(oh, axis=-1)[..., :3]) ** 2).mean())


def _per_dim_f1(pred: np.ndarray, gold: np.ndarray) -> Dict[str, dict]:
    out = {}
    for d, dn in enumerate(DIMS_ZH):
        gs = (3, 2, 1) if d in (0, 1, 2, 4) else (3,)
        dd = {}
        vals = []
        for g in gs:
            tp = int(((pred[:, d] == g) & (gold[:, d] == g)).sum())
            fp = int(((pred[:, d] == g) & (gold[:, d] != g)).sum())
            fn = int(((pred[:, d] != g) & (gold[:, d] == g)).sum())
            f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            dd[GRADE_INT_TO_STR[g]] = round(float(f1), 6)
            vals.append(f1)
        if len(vals) > 1:
            dd["mean"] = round(float(np.mean(vals)), 6)
        out[dn] = dd
    return out


def _fast_doc2_s_pos(pred: np.ndarray, gold: np.ndarray) -> float:
    f1s = []
    for d in range(6):
        gs = (1, 2, 3) if d in (0, 1, 2, 4) else (3,)
        for g in gs:
            tp = int(((pred[:, d] == g) & (gold[:, d] == g)).sum())
            fp = int(((pred[:, d] == g) & (gold[:, d] != g)).sum())
            fn = int(((pred[:, d] != g) & (gold[:, d] == g)).sum())
            denom = 2 * tp + fp + fn
            f1s.append((2.0 * tp / denom) if denom > 0 else 1.0)
    return 25.0 * float(np.mean(f1s))


def _pos_risk_audit(pred: np.ndarray, gold: np.ndarray, groups: np.ndarray, n_trials: int = 10000, target_n: int = 654) -> dict:
    import common55 as c

    ref_spos = float(c.score(pred, gold)["doc2_S_pos"])
    fast_spos = _fast_doc2_s_pos(pred, gold)
    assert abs(ref_spos - fast_spos) < 1e-9, f"S_pos mismatch: {ref_spos} vs {fast_spos}"

    rng = np.random.default_rng(2026)
    N = len(gold)
    # 1. Uniform subsampling without replacement (Perm654)
    pos_perm = np.zeros(n_trials, dtype=np.float64)
    for t in range(n_trials):
        ix = rng.choice(N, size=min(target_n, N), replace=False)
        pos_perm[t] = _fast_doc2_s_pos(pred[ix], gold[ix])

    # 2. Video-cluster bootstrap (ClBoot654)
    uniq_g = np.unique(groups)
    g_to_ix = {g: np.where(groups == g)[0] for g in uniq_g}
    n_g = len(uniq_g)
    avg_len = float(N) / max(1, n_g)
    n_draw_g = max(1, int(round(target_n / avg_len)))
    pos_clb = np.zeros(n_trials, dtype=np.float64)
    for t in range(n_trials):
        draw = rng.choice(n_g, size=n_draw_g, replace=True)
        ix = np.concatenate([g_to_ix[uniq_g[j]] for j in draw])
        if len(ix) > target_n:
            ix = ix[:target_n]
        pos_clb[t] = _fast_doc2_s_pos(pred[ix], gold[ix])

    return {
        "n_trials": int(n_trials),
        "target_n": int(target_n),
        "perm654_pos_lt15_pct": round(float((pos_perm < 15.0).mean() * 100.0), 2),
        "perm654_pos_mean": round(float(pos_perm.mean()), 4),
        "perm654_pos_p05": round(float(np.quantile(pos_perm, 0.05)), 4),
        "clboot654_pos_lt15_pct": round(float((pos_clb < 15.0).mean() * 100.0), 2),
        "clboot654_pos_mean": round(float(pos_clb.mean()), 4),
        "clboot654_pos_p05": round(float(np.quantile(pos_clb, 0.05)), 4),
    }


def _compute_nested_v6_oof():
    import common55 as c
    import deconfound_calibrator as dc
    import final_engine as fe
    import upgrade_deconf_v4_tile as v4m
    import upgrade_deconf_v5_boundary as v5m
    import upgrade_deconf_v6_boundary as v6m
    import upgrade_os_og_bokeh_deconf as uob
    from eval_specialist_gate import load_oof_dir

    t0 = time.time()
    members_v7 = fe.SETS["resident_v7"]
    nobk_members = [m for m in members_v7 if m != "pe_bkseg1"]
    names, gold, fold, _, P_v7 = fe.oof_pool(members_v7)
    _, _, _, groups = c.load_split()
    split_mp = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    is_orig = np.array([split_mp.get(n, "orig") == "orig" for n in names], dtype=bool)

    cache_p = "/tmp/v6_nested_oof_cache.npz"
    if os.path.exists(cache_p):
        zc = np.load(cache_p, allow_pickle=True)
        if list(zc["names"]) == names:
            print(f"[OOF] loaded cached 8-fold nested deconf_v6_boundary from {cache_p}", flush=True)
            return names, gold, fold, groups, is_orig, zc["Q_v6"].astype(np.float64), zc["pr_v6"].astype(np.int64)

    S_all = v4m._align_npz(os.path.join(REPO, "results/r55/os_og_breakthrough/spatial_cache/spatial_oof_3573.npz"), names, "feats")
    T_all = v4m._align_npz(os.path.join(REPO, "results/r55/ntg_tile_peaks/ntg_tile_peaks_oof_3573.npz"), names, "stats")

    P_ex = {}
    for k, rel in v6m.V6_EXTRA_OOF_DIRS.items():
        arr, val = load_oof_dir(rel, names)
        assert val.all(), f"Incomplete OOF for {k}"
        P_ex[k] = c.renorm(arr)

    proto_cfg = dc._build_routed_gate_cfg(members_v7)
    for g_dict in (v6m._G, v5m._G, uob._G):
        g_dict.update(
            fold=fold,
            gold=gold,
            is_orig=is_orig,
            groups=groups,
            P=P_v7,
            P_ex=P_ex,
            S=S_all,
            T=T_all,
            members=members_v7,
            nobk_members=nobk_members,
            proto_cfg=proto_cfg,
        )
    with mp.get_context("fork").Pool(8) as pool:
        outs = sorted(pool.map(v6m._fold_worker_v6, list(range(8))), key=lambda d: d["k"])

    N = len(names)
    Q_v6 = np.zeros((N, 6, 4), dtype=np.float64)
    pr_v6 = np.zeros((N, 6), dtype=np.int64)
    for o in outs:
        va = fold == o["k"]
        Q_v6[va] = o["qv6_va"]
        pr_v6[va] = o["pr_v6_va"]

    np.savez_compressed(cache_p, names=np.array(names), Q_v6=Q_v6, pr_v6=pr_v6)
    print(f"[OOF] 8-fold nested deconf_v6_boundary reconstructed in {time.time() - t0:.1f}s (cached -> {cache_p})", flush=True)
    return names, gold, fold, groups, is_orig, Q_v6, pr_v6


def _fit_and_audit_video_aggregator(Q_v6: np.ndarray, pr_v6: np.ndarray, gold: np.ndarray, groups: np.ndarray):
    import common55 as c
    import final_engine as fe
    from subjective_engineering import EffectiveFrameORVideoAggregator

    _, sf_dec = fe.load_decoder("resident_v8_boundary", "deconf_v6_boundary")
    agg = EffectiveFrameORVideoAggregator().fit(Q_v6, gold, groups, pr_v6)

    out_pkl = os.path.join(REPO, "results/r55/final/resident_v8_boundary_video_aggregator.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "set": "resident_v8_boundary",
                "decoder": "deconf_v6_boundary",
                "video_aggregator": agg,
                "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
        )

    # Audit on real multi-frame OOF videos (M >= 2)
    uniq_g = np.unique(groups)
    mf_groups = [g for g in uniq_g if int((groups == g).sum()) >= 2]
    vid_gold = np.zeros((len(mf_groups), 6), dtype=np.int64)
    vid_or = np.zeros((len(mf_groups), 6), dtype=np.int64)
    vid_ours = np.zeros((len(mf_groups), 6), dtype=np.int64)
    vid_cusum = np.zeros((len(mf_groups), 6), dtype=np.int64)

    for vi, g in enumerate(mf_groups):
        ix = np.where(groups == g)[0]
        vg = gold[ix].max(axis=0)
        vid_gold[vi] = vg
        vid_or[vi] = pr_v6[ix].max(axis=0)
        vid_ours[vi] = agg.aggregate_video(Q_v6[ix], sf_dec)["video_grade_6d"]
        q_mean = Q_v6[ix].mean(axis=0, keepdims=True)
        vid_cusum[vi] = sf_dec.predict(q_mean)[0]

    def _vid_metrics(vp):
        is_cln = (vid_gold == 0).all(axis=1)
        p0_rec = float(((vp == 3) & (vid_gold == 3)).sum() / max(1, (vid_gold == 3).sum()))
        cln_spec = float(((vp[is_cln] == 0).all(axis=1)).mean()) if is_cln.any() else 1.0
        d2 = float(c.score(vp, vid_gold)["doc2_strict_total"])
        return {
            "video_p0_recall": round(p0_rec, 4),
            "clean_video_specificity": round(cln_spec, 4),
            "clean_video_fp_rate": round(1.0 - cln_spec, 4),
            "video_doc2_strict": round(d2, 3),
        }

    # Multi-frame Clean video sampling FP sweep (M = 1, 4, 8, 16, 30)
    cln_idx = np.where((gold == 0).all(axis=1))[0]
    rng = np.random.default_rng(42)
    sweep = {}
    for M in (1, 4, 8, 16, 30):
        n_sim = 3000
        or_any, me_any, or_p0, me_p0 = 0, 0, 0, 0
        for _ in range(n_sim):
            ix = rng.choice(cln_idx, size=M, replace=True)
            fp_or = pr_v6[ix].max(axis=0)
            fp_me = agg.aggregate_video(Q_v6[ix], sf_dec)["video_grade_6d"]
            or_any += int(fp_or.max() > 0)
            me_any += int(fp_me.max() > 0)
            or_p0 += int(fp_or.max() == 3)
            me_p0 += int(fp_me.max() == 3)
        sweep[f"M={M}"] = {
            "naive_or_any_fp_pct": round(100.0 * or_any / n_sim, 2),
            "meff_or_any_fp_pct": round(100.0 * me_any / n_sim, 2),
            "naive_or_p0_fp_pct": round(100.0 * or_p0 / n_sim, 2),
            "meff_or_p0_fp_pct": round(100.0 * me_p0 / n_sim, 2),
        }

    audit = {
        "set": "resident_v8_boundary",
        "decoder": "deconf_v6_boundary",
        "n_real_multiframe_videos_oof": len(mf_groups),
        "n_clean_multiframe_videos": int((vid_gold == 0).all(axis=1).sum()),
        "n_p0_video_defect_instances": int((vid_gold == 3).sum()),
        "fitted_within_video_rho_p0": {DIMS_ZH[d]: round(float(agg.rho_p0[d]), 4) for d in range(6)},
        "fitted_base_tau_p0": {DIMS_ZH[d]: round(float(agg.base_tau_p0[d]), 4) for d in range(6)},
        "real_multiframe_oof_comparison": {
            "legacy_cusum_mean": _vid_metrics(vid_cusum),
            "naive_frame_or": _vid_metrics(vid_or),
            "effective_frame_or_ours": _vid_metrics(vid_ours),
        },
        "multi_frame_sampling_fp_sweep": sweep,
    }
    out_json = os.path.join(REPO, "results/r55/final/resident_v8_boundary_video_oof_audit.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"[VIDEO] saved {out_pkl} & {out_json}", flush=True)
    return agg, sf_dec


def _generate_sidecars_and_audit(gpu: int, agg, sf_dec):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import final_engine as fe
    from subjective_engineering import (
        export_tat_jimu_payload,
        infer_video_group_id,
        render_attribution_contact_sheet,
    )

    # Load ONLY the 3 lightweight PE attribution towers (`pe_v3`, `peg`, `pe_bkseg1`)
    pe_towers = ["pe_v3", "peg", "pe_bkseg1"]
    datasets = [
        (
            "test_150",
            "test150",
            os.path.join(REPO, "data/test_150"),
            os.path.join(REPO, "results/final/pred_nriqa_dataset_test_r55_resident_v8_boundary_deconf_v6_boundary.json"),
            os.path.join(REPO, "results/r55/final/test_150_resident_v8_boundary_probs.npz"),
        ),
        (
            "final_644",
            "final644",
            "/opt/final_eval_644",
            os.path.join(REPO, "results/final/pred_nriqa_dataset_final_644_r55_resident_v8_boundary_deconf_v6_boundary.json"),
            os.path.join(REPO, "results/r55/final/final_644_resident_v8_boundary_probs.npz"),
        ),
    ]

    eng = fe.FinalEngine(pe_towers, img_dir="/opt/final_eval_644", half=True, share_weights=True)
    attr_rt = fe.AttributionRuntime(eng, "resident_v8_boundary")

    audit_all = {"set": "resident_v8_boundary", "decoder": "deconf_v6_boundary"}

    for ds_key, prefix, img_dir, pred_json, probs_npz in datasets:
        z = np.load(probs_npz, allow_pickle=True)
        names = [str(x) for x in z["names"]]
        routed = z["routed"].astype(np.float64)
        pred_grid = _load_json_grid(pred_json, names)

        attr_records, tat_records = [], []
        t0 = time.time()
        attr_secs = []
        for i, nm in enumerate(names):
            pil = eng.loader.load_pil(nm).convert("RGB")
            att, ta = attr_rt.attribute(pil, pred_grid[i], routed[i])
            attr_secs.append(ta)
            lbl = {DIMS_ZH[d]: GRADE_INT_TO_STR[int(pred_grid[i, d])] for d in range(6)}
            attr_records.append(
                {
                    "image": nm,
                    "label": lbl,
                    "attribution": att,
                    "latency_s": {"attr_s": round(float(ta), 4)},
                }
            )
            tat_records.append(export_tat_jimu_payload(nm, pred_grid[i], routed[i], att, latency_ms=ta * 1000.0))

        # Video summary
        vid_groups: Dict[str, List[int]] = {}
        for i, nm in enumerate(names):
            vid_groups.setdefault(infer_video_group_id(nm), []).append(i)
        vid_records = []
        for gid, ix_list in vid_groups.items():
            res = agg.aggregate_video(routed[ix_list], sf_dec)
            vg = res["video_grade_6d"]
            vid_records.append(
                {
                    "video_id": gid,
                    "num_frames": len(ix_list),
                    "frames": [names[i] for i in ix_list],
                    "video_label": {DIMS_ZH[d]: GRADE_INT_TO_STR[int(vg[d])] for d in range(6)},
                    "trigger_frames": res["trigger_frames"],
                    "m_eff_p0": res["m_eff_p0"],
                }
            )

        out_dir = os.path.join(REPO, "results/r55/final")
        p_attr = os.path.join(out_dir, f"{prefix}_attribution_v8.json")
        p_sheet = os.path.join(out_dir, f"{prefix}_attribution_v8_sheet.jpg")
        p_tat = os.path.join(out_dir, f"{prefix}_tat_jimu_v8.json")
        p_vid = os.path.join(out_dir, f"{prefix}_video_summary_v8.json")
        json.dump(attr_records, open(p_attr, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        render_attribution_contact_sheet(attr_records, eng.loader, p_sheet)
        json.dump(tat_records, open(p_tat, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(vid_records, open(p_vid, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        # Strict attribution audit
        clean_ok, clean_tot = 0, 0
        pos_tot, pos_match = 0, 0
        max_len, over40, ellip, contra, unc_ok, zero_gv_bugs = 0, 0, 0, 0, 0, 0
        verdicts = Counter()
        supp_types = Counter()
        loc_by_dim = {dn: Counter() for dn in DIMS_ZH}
        box_contrast_hits = {dn: [0, 0] for dn in DIMS_ZH}  # [hits, total_boxes_with_rel]
        all_reasons = []
        examples = []
        for i, rec in enumerate(attr_records):
            att = rec["attribution"]
            is_cln = bool((pred_grid[i] == 0).all())
            if is_cln:
                clean_tot += 1
                if att.get("is_clean") and len(att.get("defects", [])) == 0:
                    clean_ok += 1
                continue
            exp_pos = {DIMS_ZH[d]: GRADE_INT_TO_STR[int(pred_grid[i, d])] for d in range(6) if pred_grid[i, d] > 0}
            got_pos = {it["defect"]: it["severity"] for it in att.get("defects", [])}
            pos_tot += len(exp_pos)
            for dn, gs in exp_pos.items():
                if got_pos.get(dn) == gs:
                    pos_match += 1
            for it in att.get("defects", []):
                r40 = str(it.get("reason_40ch", ""))
                all_reasons.append(r40)
                max_len = max(max_len, len(r40))
                if len(r40) > 40:
                    over40 += 1
                if "..." in r40 or "…" in r40:
                    ellip += 1
                if "无问题" in r40:
                    contra += 1
                if it.get("uncertainty_note"):
                    unc_ok += 1
                ev_d = it.get("evidence", {})
                verdicts[str(ev_d.get("verdict", "none"))] += 1
                loc_by_dim[it["defect"]]["局部框" if it.get("box_xyxy") is not None else "全图"] += 1
                meas = ev_d.get("measurements", [])
                for m_it in meas:
                    if m_it.get("supports"):
                        supp_types[str(m_it.get("support_type", "clean_pct"))] += 1
                        break
                if ev_d.get("scope") == "box" and meas:
                    any_zero_gv = any(float(m_it.get("global_value", 1.0)) <= 0.0 for m_it in meas if "global_value" in m_it)
                    if any_zero_gv:
                        zero_gv_bugs += 1
                    rel_list = [m_it for m_it in meas if "rel_ratio" in m_it]
                    if rel_list:
                        box_contrast_hits[it["defect"]][1] += 1
                        hit_dir = any(
                            (float(m_it["rel_ratio"]) > 1.0 if m_it.get("direction") == "high" else float(m_it["rel_ratio"]) < 1.0)
                            for m_it in rel_list
                        )
                        if hit_dir:
                            box_contrast_hits[it["defect"]][0] += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "image": rec["image"],
                            "defect": it["defect"],
                            "severity": it["severity"],
                            "location": it["location"],
                            "region_cue": it.get("region_cue"),
                            "box_xyxy": it["box_xyxy"],
                            "evidence_verdict": ev_d.get("verdict"),
                            "reason_40ch": r40,
                            "len_reason": len(r40),
                            "uncertainty_note": it.get("uncertainty_note"),
                        }
                    )

        assert clean_ok == clean_tot and pos_match == pos_tot and over40 == 0 and contra == 0
        tot_bh = sum(v[0] for v in box_contrast_hits.values())
        tot_bn = sum(v[1] for v in box_contrast_hits.values())
        audit_all[f"{ds_key}_attribution_audit"] = {
            "n_images": len(names),
            "pred_json_sha256": _sha256(pred_json)[:16],
            "clean_images_consistent": f"{clean_ok}/{clean_tot}",
            "total_positive_defect_cells": pos_tot,
            "label_attribution_exact_match": f"{pos_match}/{pos_tot} ({100.0 * pos_match / max(1, pos_tot):.1f}%)",
            "max_reason_40ch_len": max_len,
            "over_40ch_count": over40,
            "truncated_ellipsis_count": ellip,
            "self_contradiction_count": contra,
            "zero_global_value_bug_count": zero_gv_bugs,
            "unique_reason_40ch_ratio": f"{len(set(all_reasons))}/{len(all_reasons)} ({100.0 * len(set(all_reasons)) / max(1, len(all_reasons)):.1f}%)",
            "uncertainty_note_coverage": f"{unc_ok}/{pos_tot}",
            "attr_latency_mean_s": round(float(np.mean(attr_secs)), 3),
            "attr_latency_p95_s": round(float(np.quantile(attr_secs, 0.95)), 3),
            "n_video_clips": len(vid_records),
            "evidence_verdict_counts": dict(verdicts),
            "evidence_support_types": dict(supp_types),
            "localization_by_dim": {k: dict(v) for k, v in loc_by_dim.items() if v},
            "localization_physical_contrast_hit_rate": {
                "overall": f"{tot_bh}/{tot_bn} ({100.0 * tot_bh / max(1, tot_bn):.1f}%)",
                "by_dim": {
                    k: f"{v[0]}/{v[1]} ({100.0 * v[0] / max(1, v[1]):.1f}%)"
                    for k, v in box_contrast_hits.items()
                    if v[1] > 0
                },
            },
            "sample_examples": examples,
        }
        print(
            f"[SIDECAR {ds_key}] {len(names)} imgs in {time.time() - t0:.1f}s | "
            f"clean={clean_ok}/{clean_tot} pos_match={pos_match}/{pos_tot} max_len={max_len} "
            f"supports={verdicts['supports']}/{pos_tot} unique={len(set(all_reasons))}/{len(all_reasons)} "
            f"zero_gv={zero_gv_bugs} box_hit={tot_bh}/{tot_bn}",
            flush=True,
        )

    out_audit = os.path.join(REPO, "results/r55/final/resident_v8_boundary_attribution_audit.json")
    json.dump(audit_all, open(out_audit, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[AUDIT] saved {out_audit}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sidecars-only", action="store_true", help="Refresh attribution/TAT/video sidecars & Pareto table without recomputing OOF")
    args = ap.parse_args()
    assert args.gpu != 7, "GPU 7 is forbidden!"

    import common55 as c
    import final_engine as fe

    if args.sidecars_only:
        vap = os.path.join(REPO, "results/r55/final/resident_v8_boundary_video_aggregator.pkl")
        agg = pickle.load(open(vap, "rb"))["video_aggregator"]
        _, sf_dec = fe.load_decoder("resident_v8_boundary", "deconf_v6_boundary")
        _generate_sidecars_and_audit(args.gpu, agg, sf_dec)
        print("=== V8 SIDECARS REFRESH COMPLETED ===", flush=True)
        return

    names, gold, fold, groups, is_orig, Q_v6, pr_v6 = _compute_nested_v6_oof()

    ev_all = c.score(pr_v6, gold)
    ev_o = c.score(pr_v6[is_orig], gold[is_orig])
    d2_all, sp_all, p0_all, cl_all = ev_all["doc2_strict_total"], ev_all["doc2_S_pos"], ev_all["doc2_S_p0"], ev_all["doc2_Clean2"]
    d2_o, sp_o, p0_o, cl_o = ev_o["doc2_strict_total"], ev_o["doc2_S_pos"], ev_o["doc2_S_p0"], ev_o["doc2_Clean2"]
    d1_o = float(ev_o["doc1_total"])
    risk_orig = _pos_risk_audit(pr_v6[is_orig], gold[is_orig], groups[is_orig], n_trials=10000, target_n=654)

    v6_dec_meta = json.load(open(os.path.join(REPO, "results/r55/final/resident_v8_boundary_decoder_deconf_v6_boundary.json")))
    oof_summary = {
        "set": "resident_v8_boundary",
        "rule": "deconf_v6_boundary",
        "members": fe.SETS["resident_v8_boundary"],
        "online_test_150_verified": {
            "total": 54.77,
            "S_pos": 17.46,
            "S_p0": 17.31,
            "S_clean": 20.00,
        },
        "decoder_params": {
            "extra_temps": v6_dec_meta["extra_temps"],
            "w_blur_keys": v6_dec_meta["w_blur_keys"],
            "w_blur": v6_dec_meta["w_blur"],
            "w_og_keys": v6_dec_meta["w_og_keys"],
            "w_og": v6_dec_meta["w_og"],
            "og24_v6_cfg": v6_dec_meta["og24_v6_cfg"],
            "p0_og_full": v6_dec_meta["p0_og_full"],
        },
        "oof_nested_3573": {
            "doc2_all": round(float(d2_all), 4),
            "doc2_orig": round(float(d2_o), 4),
            "doc1_orig": round(float(d1_o), 4),
            "S_pos_orig": round(float(sp_o), 4),
            "S_p0_orig": round(float(p0_o), 4),
            "S_clean_orig": round(float(cl_o), 4),
            "S_pos_all": round(float(sp_all), 4),
            "S_p0_all": round(float(p0_all), 4),
            "S_clean_all": round(float(cl_all), 4),
            "rps_6x4_all": round(_compute_rps_6x4(Q_v6, gold), 5),
            "rps_6x4_orig": round(_compute_rps_6x4(Q_v6[is_orig], gold[is_orig]), 5),
            "mAP14_all": round(_compute_map14(Q_v6, gold), 4),
            "mAP14_orig": round(_compute_map14(Q_v6[is_orig], gold[is_orig]), 4),
            **risk_orig,
            "per_dim_f1_orig": _per_dim_f1(pr_v6[is_orig], gold[is_orig]),
            "per_dim_f1_all": _per_dim_f1(pr_v6, gold),
        },
    }
    p_sum = os.path.join(REPO, "results/r55/final/resident_v8_boundary_oof_deconf_v6_boundary_summary.json")
    json.dump(oof_summary, open(p_sum, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(
        f"[OOF SUMMARY] saved {p_sum} | ORIG={d2_o:.4f} (Pos={sp_o:.4f}, P0={p0_o:.4f}, Cln={cl_o:.4f}) "
        f"| ALL={d2_all:.4f} | Perm654<15={risk_orig['perm654_pos_lt15_pct']}% ClBoot654<15={risk_orig['clboot654_pos_lt15_pct']}%",
        flush=True,
    )

    # Cost-Pareto comparison table
    pareto_old = json.load(open(os.path.join(REPO, "results/r55/final/resident_v7_cost_pareto_oof.json")))
    pareto = dict(pareto_old)
    pareto["resident_v7 + deconf_v4_tile (11m Prev Best 54.68, 11.62B, 19.37 GiB)"] = {
        "doc2_all": 51.809,
        "doc2_orig": 52.464,
        "S_pos_orig": 16.078,
        "S_p0_orig": 16.542,
        "S_clean_orig": 19.844,
        "mAP14_orig": 0.6647,
        "rps_orig": 0.03760,
        "online_test_150": 54.68,
    }
    pareto["resident_v8_boundary + deconf_v6_boundary (14m Flagship 54.77, 13.26B, 21.02 GiB)"] = {
        "doc2_all": round(float(d2_all), 4),
        "doc2_orig": round(float(d2_o), 4),
        "S_pos_orig": round(float(sp_o), 4),
        "S_p0_orig": round(float(p0_o), 4),
        "S_clean_orig": round(float(cl_o), 4),
        "mAP14_orig": round(_compute_map14(Q_v6[is_orig], gold[is_orig]), 4),
        "rps_orig": round(_compute_rps_6x4(Q_v6[is_orig], gold[is_orig]), 5),
        "online_test_150": 54.77,
    }
    p_par = os.path.join(REPO, "results/r55/final/resident_v8_boundary_cost_pareto_oof.json")
    json.dump(pareto, open(p_par, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[PARETO] saved {p_par}", flush=True)

    agg, sf_dec = _fit_and_audit_video_aggregator(Q_v6, pr_v6, gold, groups)
    _generate_sidecars_and_audit(args.gpu, agg, sf_dec)
    print("=== ALL V8 DELIVERY SIDECARS & AUDITS COMPLETED ===", flush=True)


if __name__ == "__main__":
    main()
