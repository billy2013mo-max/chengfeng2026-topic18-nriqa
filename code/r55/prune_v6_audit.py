#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/prune_v6_audit.py
==========================
Committed pruning & convergence audit for `resident_v6` (9 members -> minimal converged set).
Evaluates:
  1. Leave-One-Out (LOO) ablation from `resident_v6` (9m) with paired 10,000-draw video-cluster
     CI90 for ΔRPS on BOTH ALL and ORIG, plus decoded ΔDoc2.
  2. Multi-partition (4x8-fold) + TagSplit + hardware matrix for candidate subsets (`9m -> 3m`),
     plus a zero-leakage CPU check of no-KD `p12_pe` (`r55_p12_e5_s2026`) inside `v6_core_nop3_6m`.
  3. Applies the convergence rule (`smallest set with point ΔRPS <= +0.00010 and CI90 upper <= +0.00030
     on BOTH ALL and ORIG`).
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Dict, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

import common55 as c
import final_engine as fe
from eval_specialist_gate import cluster_bootstrap_rps_delta, load_oof_dir, per_dim_ap, rps_6x4
from eval_v6_gates import LAMS_V6, build_proto_cfg, make_partitions
from routed_eu_decoder import RoutedGateEUDecoder, _fast_boot654_pos_risk


def cluster_bootstrap_doc2_delta(
    pred_new: np.ndarray,
    pred_ref: np.ndarray,
    gold: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 2026,
) -> Tuple[float, float, float]:
    diff_pt = float(c.score(pred_new, gold)["doc2_strict_total"] - c.score(pred_ref, gold)["doc2_strict_total"])
    ug, inv = np.unique(groups, return_inverse=True)
    by_g = [np.where(inv == g)[0] for g in range(len(ug))]
    rs = np.random.RandomState(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    G = len(ug)
    for b in range(n_boot):
        picks = rs.randint(0, G, size=G)
        idx = np.concatenate([by_g[g] for g in picks])
        deltas[b] = float(
            c.score(pred_new[idx], gold[idx])["doc2_strict_total"]
            - c.score(pred_ref[idx], gold[idx])["doc2_strict_total"]
        )
    return diff_pt, float(np.quantile(deltas, 0.05)), float(np.quantile(deltas, 0.95))


def eval_p0_only(
    P_sub: Dict[str, np.ndarray],
    members: Sequence[str],
    gold: np.ndarray,
    fold: np.ndarray,
    groups: np.ndarray,
    is_orig: np.ndarray,
):
    cfg = build_proto_cfg(members)
    Q_oof = np.zeros((len(gold), 6, 4), dtype=np.float64)
    pr_oof = np.zeros_like(gold)
    for k in range(8):
        tr, va = fold != k, fold == k
        dk = RoutedGateEUDecoder(**cfg, lams_cl=LAMS_V6).fit(
            {m: v[tr] for m, v in P_sub.items()}, gold[tr], is_orig[tr], groups[tr]
        )
        Q_oof[va] = dk.pool_from_member_dict({m: v[va] for m, v in P_sub.items()})
        pr_oof[va] = dk.eu.predict(Q_oof[va])
    return Q_oof, pr_oof


def main() -> None:
    t_start = time.time()
    names, gold, fold, groups = c.load_split()
    man = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    is_orig = np.array([man[n] == "orig" for n in names], dtype=bool)
    tm = {r["name"]: int(r["fold"]) for r in csv.DictReader(open(os.path.join(REPO, "data/r55_tagsplit/folds.csv")))}
    tag_fold = np.array([tm[n] for n in names], dtype=np.int64)

    v6_m = fe.SETS["resident_v6"]
    _, _, _, _, P_v6 = fe.oof_pool(v6_m)
    p12_nokd_arr, p12_nokd_valid = load_oof_dir(os.path.join(REPO, "results/next55/towers/r55_p12_e5_s2026/pe_core_large"), names)
    assert bool(p12_nokd_valid.all()), "Incomplete r55_p12_e5_s2026 OOF"
    P_v6_nokd = dict(P_v6)
    P_v6_nokd["p12_pe"] = c.renorm(p12_nokd_arr)

    bench = json.load(open(os.path.join(REPO, "results/r55/final/resident_v6_bench_half.json")))
    params_b = {
        "cradio_v3": 0.680, "pe_v3": 0.332, "peg": 0.332, "sig_v3": 0.449,
        "ntg_pe": 0.332, "p12_pe": 0.332, "p3_qy": 1.270, "v16": 3.305, "p1_qy": 4.789,
    }

    Q_full, pr_full = eval_p0_only(P_v6, v6_m, gold, fold, groups, is_orig)
    sa_full = c.score(pr_full, gold)
    so_full = c.score(pr_full[is_orig], gold[is_orig])

    # 1. Leave-One-Out with BOTH ALL & ORIG CIs
    loo_rows = []
    print("=== 1. Leave-One-Out Ablation from resident_v6 (9m) ===", flush=True)
    for m in v6_m:
        sub = [x for x in v6_m if x != m]
        Q_s, pr_s = eval_p0_only({k: P_v6[k] for k in sub}, sub, gold, fold, groups, is_orig)
        dm_a, lo_a, hi_a = cluster_bootstrap_rps_delta(Q_s, Q_full, gold, groups)
        dm_o, lo_o, hi_o = cluster_bootstrap_rps_delta(Q_s[is_orig], Q_full[is_orig], gold[is_orig], groups[is_orig])
        dd_a, dlo_a, dhi_a = cluster_bootstrap_doc2_delta(pr_s, pr_full, gold, groups, n_boot=1000)
        dd_o, dlo_o, dhi_o = cluster_bootstrap_doc2_delta(pr_s[is_orig], pr_full[is_orig], gold[is_orig], groups[is_orig], n_boot=1000)
        row = dict(
            dropped=m,
            dRPS_ALL=dm_a, ci90_ALL=[lo_a, hi_a],
            dRPS_ORIG=dm_o, ci90_ORIG=[lo_o, hi_o],
            dDoc2_ALL=dd_a, doc2_ci90_ALL=[dlo_a, dhi_a],
            dDoc2_ORIG=dd_o, doc2_ci90_ORIG=[dlo_o, dhi_o],
        )
        loo_rows.append(row)
        print(
            f"  drop_{m:10s} | dRPS_ALL={dm_a:+.5f} [{lo_a:+.5f},{hi_a:+.5f}] | "
            f"dRPS_ORIG={dm_o:+.5f} [{lo_o:+.5f},{hi_o:+.5f}] | "
            f"dDoc2_ALL={dd_a:+.3f} [{dlo_a:+.3f},{dhi_a:+.3f}] ORIG={dd_o:+.3f} [{dlo_o:+.3f},{dhi_o:+.3f}]",
            flush=True,
        )

    # 2. Candidate Subset Matrix (4-partition + TagSplit + both CIs + no-KD p12 check)
    CANDIDATES = {
        "v6_full_9m":             (v6_m, P_v6),
        "v6_nosig_8m":            (["cradio_v3", "pe_v3", "peg", "v16", "p3_qy", "ntg_pe", "p12_pe", "p1_qy"], P_v6),
        "v6_pegonly_7m":          (["cradio_v3", "peg", "v16", "p3_qy", "ntg_pe", "p12_pe", "p1_qy"], P_v6),
        "v6_core_nop12_6m":       (["cradio_v3", "peg", "v16", "p3_qy", "ntg_pe", "p1_qy"], P_v6),
        "v6_core_nop3_6m":        (["cradio_v3", "peg", "v16", "ntg_pe", "p12_pe", "p1_qy"], P_v6),
        "v6_core_nop3_6m_nokd12": (["cradio_v3", "peg", "v16", "ntg_pe", "p12_pe", "p1_qy"], P_v6_nokd),
        "v6_slim_5m":             (["peg", "v16", "ntg_pe", "p12_pe", "p1_qy"], P_v6),
        "v6_ultra_4m":            (["peg", "v16", "ntg_pe", "p1_qy"], P_v6),
        "v6_3fam_3m":             (["v16", "ntg_pe", "p1_qy"], P_v6),
    }

    parts = make_partitions(fold, groups, 4)
    subset_rows = []
    print("\n=== 2. Subset Convergence Matrix (with ALL & ORIG CIs) ===", flush=True)
    for label, (mems, P_src) in CANDIDATES.items():
        P_sub = {m: P_src[m] for m in mems}
        cfg = build_proto_cfg(mems)
        d_all, d_orig, cl_all = [], [], []
        q_p0, pr_p0 = None, None
        for pi, fp in enumerate(parts):
            Qo = np.zeros((len(gold), 6, 4))
            po = np.zeros_like(gold)
            for k in range(8):
                tr, va = fp != k, fp == k
                dk = RoutedGateEUDecoder(**cfg, lams_cl=LAMS_V6).fit(
                    {m: v[tr] for m, v in P_sub.items()}, gold[tr], is_orig[tr], groups[tr]
                )
                Qo[va] = dk.pool_from_member_dict({m: v[va] for m, v in P_sub.items()})
                po[va] = dk.eu.predict(Qo[va])
            sa, so = c.score(po, gold), c.score(po[is_orig], gold[is_orig])
            d_all.append(float(sa["doc2_strict_total"]))
            d_orig.append(float(so["doc2_strict_total"]))
            cl_all.append(float(sa["doc2_Clean2"]))
            if pi == 0:
                q_p0, pr_p0 = Qo, po

        pt = np.zeros_like(gold)
        for tk in sorted(np.unique(tag_fold)):
            tr, va = tag_fold != tk, tag_fold == tk
            dt = RoutedGateEUDecoder(**cfg, lams_cl=LAMS_V6).fit(
                {m: v[tr] for m, v in P_sub.items()}, gold[tr], is_orig[tr], groups[tr]
            )
            pt[va] = dt.predict_from_member_dict({m: v[va] for m, v in P_sub.items()})
        st_o = c.score(pt[is_orig], gold[is_orig])

        dm_a, lo_a, hi_a = cluster_bootstrap_rps_delta(q_p0, Q_full, gold, groups)
        dm_o, lo_o, hi_o = cluster_bootstrap_rps_delta(q_p0[is_orig], Q_full[is_orig], gold[is_orig], groups[is_orig])
        dd_a, dlo_a, dhi_a = cluster_bootstrap_doc2_delta(pr_p0, pr_full, gold, groups, n_boot=1000)
        dd_o, dlo_o, dhi_o = cluster_bootstrap_doc2_delta(pr_p0[is_orig], pr_full[is_orig], gold[is_orig], groups[is_orig], n_boot=1000)
        pf, _, _ = _fast_boot654_pos_risk(pr_p0[is_orig], gold[is_orig], groups[is_orig], n_boot=10000, seed=888, without_replacement=True)
        _, m14 = per_dim_ap(q_p0, gold)

        passes_rule = (
            (dm_a <= 0.00010) and (hi_a <= 0.00030)
            and (dm_o <= 0.00010) and (hi_o <= 0.00030)
        )
        srow = dict(
            label=label, members=list(mems), n_members=len(mems),
            params_b=round(sum(params_b[m] for m in mems), 3),
            bench_sum_s=round(sum(bench["per_member"][m] for m in mems), 3),
            rps_all=float(rps_6x4(q_p0, gold)), rps_orig=float(rps_6x4(q_p0[is_orig], gold[is_orig])),
            mAP14=float(m14),
            dRPS_ALL=dm_a, ci90_ALL=[lo_a, hi_a],
            dRPS_ORIG=dm_o, ci90_ORIG=[lo_o, hi_o],
            dDoc2_ALL=dd_a, doc2_ci90_ALL=[dlo_a, dhi_a],
            dDoc2_ORIG=dd_o, doc2_ci90_ORIG=[dlo_o, dhi_o],
            mean4_ALL=float(np.mean(d_all)), mean4_ORIG=float(np.mean(d_orig)), mean4_Clean=float(np.mean(cl_all)),
            p0_ALL=d_all[0], p0_ORIG=d_orig[0],
            tag_ORIG=float(st_o["doc2_strict_total"]),
            perm654_pos_lt15_pct=float(pf),
            passes_convergence_rule=bool(passes_rule),
        )
        subset_rows.append(srow)
        print(
            f"  {label:22s} | {len(mems)}m {srow['params_b']:5.2f}B {srow['bench_sum_s']:4.2f}s | "
            f"dRPS_ALL={dm_a:+.5f} [{lo_a:+.5f},{hi_a:+.5f}] ORIG={dm_o:+.5f} [{lo_o:+.5f},{hi_o:+.5f}] | "
            f"mAP={m14:.4f} | mean4 ORIG={srow['mean4_ORIG']:.3f} | P<15={pf:.2f}% | RULE={passes_rule}",
            flush=True,
        )

    out_p = os.path.join(REPO, "results/r55/vlm_qy8/prune_v6_audit.json")
    json.dump(dict(elapsed_s=round(time.time() - t_start, 1), loo=loo_rows, subsets=subset_rows), open(out_p, "w"), indent=2)
    print(f"Saved -> {out_p}", flush=True)


if __name__ == "__main__":
    main()
