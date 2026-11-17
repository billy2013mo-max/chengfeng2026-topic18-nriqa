#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_v6_gates.py
=========================
Pre-registered baseline lock & Gate G1/G2 evaluator for `resident_v6` (`v5_8m + p1_qy`)
per `claude-fable-5-2` Round-2 Convergence Spec (`results/r55/v6_debate/round2_reply.md`):

- Extended `lams_cl = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)` applied identically to BOTH
  `resident_v5` and `resident_v6` (with guard logging if >= 2 folds select 3.25).
- Gate G1 (Evidence): `p1_qy` at fixed 0.18 on top of `resident_v5`:
    `dRPS_ALL <= -0.00025` and `dRPS_ORIG <= -0.00035`, video-cluster CI90 upper bound < 0,
    and `seed 2027` control same negative sign.
- Gate G2 (No Harm): 4-partition mean `Doc2S >= v5 - 0.05` and `Clean2 >= v5 - 0.05`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

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
from eval_specialist_gate import cluster_bootstrap_rps_delta, per_dim_ap, rps_6x4
from routed_eu_decoder import RoutedGateEUDecoder, TrueGateConstrainedEU, _fast_boot654_pos_risk, compute_family_base_prior

LAMS_V6 = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)


def make_partitions(fold: np.ndarray, grp: np.ndarray, n: int = 4) -> List[np.ndarray]:
    out = [fold.copy()]
    ug, inv = np.unique(grp, return_inverse=True)
    for s in range(1, n):
        rs = np.random.RandomState(1000 + s)
        out.append(rs.permutation(np.arange(len(ug)) % 8)[inv])
    return out


def build_proto_cfg(members: Sequence[str], p1_family_prior_mode: str = "family") -> dict:
    """
    Builds `RoutedGateEUDecoder` config for `members`.
    For `resident_v5` (`8m`): `base_m = [cradio_v3, pe_v3, peg, sig_v3, v16, p3_qy, ntg_pe]`, `p12_m = [p12_pe]`.
    For `resident_v6` (`9m = v5 + p1_qy`): `p1_qy` is its own orthogonal VLM family (`phoenix_vlm`).
    """
    p12_m = [m for m in members if m in fe.P12_SPEC]
    base_m = [m for m in members if m not in fe.P12_SPEC]
    bp = compute_family_base_prior(base_m)

    if p1_family_prior_mode == "fixed018" and any(m in ("p1_qy", "p1_cv5") for m in base_m):
        # Give p1_qy 0.18 of TOTAL prior (= 0.18 / (1 - 0.18) inside base_m when p12_total_prior=0.18)
        vlm_k = "p1_qy" if "p1_qy" in base_m else "p1_cv5"
        non_vlm = [m for m in base_m if m != vlm_k]
        nv_sum = sum(bp[base_m.index(m)] for m in non_vlm)
        target_base_vlm = 0.18 / (1.0 - (0.18 if p12_m else 0.0))
        new_bp = []
        for m in base_m:
            if m == vlm_k:
                new_bp.append(target_base_vlm)
            else:
                new_bp.append(bp[base_m.index(m)] / nv_sum * (1.0 - target_base_vlm))
        bp = np.array(new_bp, dtype=np.float64)

    return dict(
        base_members=base_m,
        base_prior=list(bp / bp.sum()),
        p12_members=p12_m,
        p12_internal_weights=[1.0 / max(1, len(p12_m))] * len(p12_m),
        p12_total_prior=0.18 if p12_m else 0.0,
        l2_reg=0.015,
    )


def fit_routed_gate_eu_lams(
    P_tr: Dict[str, np.ndarray],
    gold_tr: np.ndarray,
    is_orig_tr: np.ndarray,
    groups_tr: np.ndarray,
    proto_cfg: dict,
    lams_cl: Sequence[float] = LAMS_V6,
) -> RoutedGateEUDecoder:
    dec = RoutedGateEUDecoder(**proto_cfg)
    streams, prior = dec._build_streams(P_tr)
    from eval_specialist_gate import apply_nested_routed_pool, fit_nested_routed_pool

    dec.temps, dec.W = fit_nested_routed_pool(streams, gold_tr, prior, l2_reg=dec.l2_reg, t_max=dec.t_max)
    if dec.use_dirdiag:
        Q_raw = apply_nested_routed_pool(streams, dec.temps, dec.W)
        dec.dirdiag_params = dec._fit_postpool_dirdiag(
            Q_raw, gold_tr, is_orig_tr, w_supp_cal=dec.w_supp_cal, l2=dec.dirdiag_l2
        )
    else:
        dec.dirdiag_params = None
    Q = dec.pool_from_member_dict(P_tr)
    eu_ws = dec.w_supp_cal if dec.use_dirdiag else 0.25
    dec.eu = TrueGateConstrainedEU(lams_cl=lams_cl, w_supp=eu_ws).fit(Q, gold_tr, is_orig=is_orig_tr, groups=groups_tr)
    return dec


def eval_4part_and_tag(
    P_dict: Dict[str, np.ndarray],
    gold: np.ndarray,
    fold: np.ndarray,
    groups: np.ndarray,
    is_orig: np.ndarray,
    tag_fold: np.ndarray,
    proto_cfg: dict,
    lams_cl: Sequence[float] = LAMS_V6,
) -> dict:
    parts = make_partitions(fold, groups, 4)
    part_rows = []
    q_p0 = None
    pred_p0 = None
    full_dec = fit_routed_gate_eu_lams(P_dict, gold, is_orig, groups, proto_cfg, lams_cl=lams_cl)

    for pi, fp in enumerate(parts):
        N = len(gold)
        Q_oof = np.zeros((N, 6, 4), dtype=np.float64)
        pred_oof = np.zeros_like(gold)
        chosen_lams = []
        for k in range(8):
            tr, va = fp != k, fp == k
            dk = fit_routed_gate_eu_lams(
                {m: v[tr] for m, v in P_dict.items()}, gold[tr], is_orig[tr], groups[tr], proto_cfg, lams_cl=lams_cl
            )
            Q_oof[va] = dk.pool_from_member_dict({m: v[va] for m, v in P_dict.items()})
            pred_oof[va] = dk.eu.predict(Q_oof[va])
            chosen_lams.append((float(dk.eu.lam_cl), float(dk.eu.p0_scale)))
        if pi == 0:
            q_p0 = Q_oof
            pred_p0 = pred_oof
        sc_a = c.score(pred_oof, gold)
        sc_o = c.score(pred_oof[is_orig], gold[is_orig])
        part_rows.append(
            dict(
                partition=pi,
                doc2_all=float(sc_a["doc2_strict_total"]),
                pos_all=float(sc_a["doc2_S_pos"]),
                p0_all=float(sc_a["doc2_S_p0"]),
                clean_all=float(sc_a["doc2_Clean2"]),
                doc2_orig=float(sc_o["doc2_strict_total"]),
                pos_orig=float(sc_o["doc2_S_pos"]),
                p0_orig=float(sc_o["doc2_S_p0"]),
                clean_orig=float(sc_o["doc2_Clean2"]),
                rps_all=float(rps_6x4(Q_oof, gold)),
                rps_orig=float(rps_6x4(Q_oof[is_orig], gold[is_orig])),
                chosen_lams=chosen_lams,
                n_edge_325=sum(1 for l_val, _ in chosen_lams if abs(l_val - 3.25) < 1e-6),
            )
        )

    # Tag-split evaluation
    pred_tag = np.zeros_like(gold)
    for tk in sorted(np.unique(tag_fold)):
        tr_t, va_t = tag_fold != tk, tag_fold == tk
        dt = fit_routed_gate_eu_lams(
            {m: v[tr_t] for m, v in P_dict.items()}, gold[tr_t], is_orig[tr_t], groups[tr_t], proto_cfg, lams_cl=lams_cl
        )
        qt = dt.pool_from_member_dict({m: v[va_t] for m, v in P_dict.items()})
        pred_tag[va_t] = dt.eu.predict(qt)
    sc_tag_a = c.score(pred_tag, gold)
    sc_tag_o = c.score(pred_tag[is_orig], gold[is_orig])

    assert q_p0 is not None and pred_p0 is not None
    d_aps, map14 = per_dim_ap(q_p0, gold)
    perm_fail, perm_cp0, _ = _fast_boot654_pos_risk(
        pred_p0[is_orig], gold[is_orig], groups[is_orig], n_boot=10000, seed=888, without_replacement=True
    )
    clb_fail, _, _ = _fast_boot654_pos_risk(
        pred_p0[is_orig], gold[is_orig], groups[is_orig], n_boot=10000, seed=888, without_replacement=False
    )

    return dict(
        lams_grid=list(lams_cl),
        p0_rps_all=float(rps_6x4(q_p0, gold)),
        p0_rps_orig=float(rps_6x4(q_p0[is_orig], gold[is_orig])),
        p0_mAP14=float(map14),
        p0_per_dim_ap=[float(x) for x in d_aps],
        mean4_doc2_all=float(np.mean([r["doc2_all"] for r in part_rows])),
        mean4_clean_all=float(np.mean([r["clean_all"] for r in part_rows])),
        mean4_doc2_orig=float(np.mean([r["doc2_orig"] for r in part_rows])),
        mean4_clean_orig=float(np.mean([r["clean_orig"] for r in part_rows])),
        p0_doc2_all=part_rows[0]["doc2_all"],
        p0_doc2_orig=part_rows[0]["doc2_orig"],
        p0_pos_orig=part_rows[0]["pos_orig"],
        p0_p0_orig=part_rows[0]["p0_orig"],
        p0_clean_orig=part_rows[0]["clean_orig"],
        p0_n_edge_325=part_rows[0]["n_edge_325"],
        tag_doc2_all=float(sc_tag_a["doc2_strict_total"]),
        tag_doc2_orig=float(sc_tag_o["doc2_strict_total"]),
        perm654_orig_pos_lt15_pct=float(perm_fail),
        clboot654_orig_pos_lt15_pct=float(clb_fail),
        perm654_orig_clean_p0_pct=float(perm_cp0),
        full_temps=[float(x) for x in full_dec.temps],
        full_W=full_dec.W.tolist(),
        full_lam_cl=float(full_dec.eu.lam_cl),
        full_p0_scale=float(full_dec.eu.p0_scale),
        partitions=part_rows,
        q_p0=q_p0,
    )


def run_pre_register() -> None:
    names, gold, fold, groups = c.load_split()
    man = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    is_orig = np.array([man[n] == "orig" for n in names], dtype=bool)
    tm = {r["name"]: int(r["fold"]) for r in csv.DictReader(open(os.path.join(REPO, "data/r55_tagsplit/folds.csv")))}
    tag_fold = np.array([tm[n] for n in names], dtype=np.int64)

    v5_m = fe.SETS["resident_v5"]
    _, _, _, q_v5_fixed, P_v5 = fe.oof_pool(v5_m)
    cfg_v5 = build_proto_cfg(v5_m)

    rep_v5_old = eval_4part_and_tag(P_v5, gold, fold, groups, is_orig, tag_fold, cfg_v5, lams_cl=(1.75, 2.00, 2.25, 2.50, 2.75))
    rep_v5_ext = eval_4part_and_tag(P_v5, gold, fold, groups, is_orig, tag_fold, cfg_v5, lams_cl=LAMS_V6)
    rep_v5_old.pop("q_p0", None)
    rep_v5_ext.pop("q_p0", None)

    # Also log the exact historical 5-fold Strict_3473 check requested in Round-2 audit
    ph5_rows = list(csv.DictReader(open(os.path.join(REPO, "data/phoenix_splits/phoenix_5fold.csv"))))
    grp_folds: Dict[str, set] = {}
    for r in ph5_rows:
        grp_folds.setdefault(r["canon_group"], set()).add(int(r["fold"]))
    leaked_grps = {g for g, fs in grp_folds.items() if len(fs) > 1}
    nm_to_grp = {r["name"]: r["canon_group"] for r in ph5_rows}
    strict_mask = np.array([nm_to_grp[n] not in leaked_grps for n in names], dtype=bool)

    z_p1_cv5 = np.load(os.path.join(REPO, "results/phoenix/pillar1_cv5_oof.npz"), allow_pickle=True)
    mp_p1 = {str(nm): q for nm, q in zip(z_p1_cv5["names"], z_p1_cv5["probs_6x4"].astype(np.float64))}
    P_9m = dict(P_v5)
    P_9m["p1_cv5"] = c.renorm(np.stack([mp_p1[n] for n in names]))
    cfg_9m = build_proto_cfg(v5_m + ["p1_cv5"])

    s_idx = np.where(strict_mask)[0]
    dec_9m_full = fit_routed_gate_eu_lams(
        {m: v[s_idx] for m, v in P_9m.items()},
        gold[s_idx],
        is_orig[s_idx],
        groups[s_idx],
        cfg_9m,
        lams_cl=(1.75, 2.00, 2.25, 2.50, 2.75),
    )

    out_path = os.path.join(REPO, "results/r55/v6_debate/pre_registered_v5_and_gates.json")
    payload = dict(
        locked_at_utc="2026-11-15T00:44:00Z",
        gates=dict(
            G1_fixed018_dRPS_ALL_max=-0.00025,
            G1_fixed018_dRPS_ORIG_max=-0.00035,
            G1_ci90_upper_max=0.0,
            G1_seed2027_same_sign=True,
            G2_mean4_doc2_min_delta=-0.05,
            G2_mean4_clean_min_delta=-0.05,
            G3_bf16_parity_max_dp=0.01,
            G3_nf4_tv_max=0.012,
            G3_p3_cast_tv_max=0.002,
            G3_peak_gib_max=22.0,
            guard_lams_325_max_folds=1,
        ),
        v5_fixed_pool_rps_all=float(rps_6x4(q_v5_fixed, gold)),
        v5_fixed_pool_rps_orig=float(rps_6x4(q_v5_fixed[is_orig], gold[is_orig])),
        resident_v5_old_grid=rep_v5_old,
        resident_v5_ext_grid_325=rep_v5_ext,
        strict3473_9m_audit=dict(
            n_strict=int(strict_mask.sum()),
            n_strict_orig=int((strict_mask & is_orig).sum()),
            stream_order=cfg_9m["base_members"] + ["p12_comb"],
            temps=[float(x) for x in dec_9m_full.temps],
            W_by_dim=dec_9m_full.W.tolist(),
        ),
    )
    json.dump(payload, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(
        f"[Pre-Registered Lock Saved -> {out_path}]\n"
        f"  resident_v5 (old grid 1.75..2.75): p0 Doc2S ALL={rep_v5_old['p0_doc2_all']:.3f} ORIG={rep_v5_old['p0_doc2_orig']:.3f} | "
        f"mean4 ALL={rep_v5_old['mean4_doc2_all']:.3f} Cln={rep_v5_old['mean4_clean_all']:.3f}\n"
        f"  resident_v5 (ext grid 1.75..3.25): p0 Doc2S ALL={rep_v5_ext['p0_doc2_all']:.3f} ORIG={rep_v5_ext['p0_doc2_orig']:.3f} | "
        f"mean4 ALL={rep_v5_ext['mean4_doc2_all']:.3f} Cln={rep_v5_ext['mean4_clean_all']:.3f} | edge325={rep_v5_ext['p0_n_edge_325']}",
        flush=True,
    )


def run_eval_v6(oof_s2026: str, oof_s2027: str = "") -> None:
    names, gold, fold, groups = c.load_split()
    man = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    is_orig = np.array([man[n] == "orig" for n in names], dtype=bool)
    tm = {r["name"]: int(r["fold"]) for r in csv.DictReader(open(os.path.join(REPO, "data/r55_tagsplit/folds.csv")))}
    tag_fold = np.array([tm[n] for n in names], dtype=np.int64)

    v5_m = fe.SETS["resident_v5"]
    _, _, _, q_v5_fixed, P_v5 = fe.oof_pool(v5_m)
    cfg_v5 = build_proto_cfg(v5_m)
    rep_v5 = eval_4part_and_tag(P_v5, gold, fold, groups, is_orig, tag_fold, cfg_v5, lams_cl=LAMS_V6)
    q_v5_routed = rep_v5.pop("q_p0")

    z26 = np.load(oof_s2026, allow_pickle=True)
    assert bool(np.all(z26["valid_mask"])), f"Incomplete s2026 OOF: {z26['valid_mask'].sum()}/3573"
    mp26 = {str(nm): q for nm, q in zip(z26["names"], z26["probs_6x4"].astype(np.float64))}
    p_s26 = c.renorm(np.stack([mp26[n] for n in names]))

    solo_aps, solo_map14 = per_dim_ap(p_s26, gold)
    solo_sc = c.score(np.argmax(p_s26, axis=-1), gold)

    # Gate G1: fixed 0.18 blend on top of fixed-pool v5 AND on top of routed-pool v5
    q_fix018 = c.renorm(0.82 * q_v5_fixed + 0.18 * p_s26)
    dm_all, lo_all, hi_all = cluster_bootstrap_rps_delta(q_fix018, q_v5_fixed, gold, groups)
    dm_orig, lo_orig, hi_orig = cluster_bootstrap_rps_delta(
        q_fix018[is_orig], q_v5_fixed[is_orig], gold[is_orig], groups[is_orig]
    )

    s27_info = None
    s27_same_sign = True
    if oof_s2027 and os.path.exists(oof_s2027):
        z27 = np.load(oof_s2027, allow_pickle=True)
        vm = np.asarray(z27["valid_mask"], dtype=bool)
        mp27 = {str(nm): q for nm, q in zip(z27["names"], z27["probs_6x4"].astype(np.float64))}
        p_s27 = c.renorm(np.stack([mp27[n] for n in names]))
        q27_sub = c.renorm(0.82 * q_v5_fixed[vm] + 0.18 * p_s27[vm])
        dm27_a, lo27_a, hi27_a = cluster_bootstrap_rps_delta(q27_sub, q_v5_fixed[vm], gold[vm], groups[vm])
        vm_o = vm & is_orig
        dm27_o, lo27_o, hi27_o = cluster_bootstrap_rps_delta(
            c.renorm(0.82 * q_v5_fixed[vm_o] + 0.18 * p_s27[vm_o]), q_v5_fixed[vm_o], gold[vm_o], groups[vm_o]
        )
        s27_same_sign = (dm27_a < 0.0) and (dm27_o < 0.0)
        s27_info = dict(
            n_valid=int(vm.sum()),
            completed_folds=[int(x) for x in z27["completed_folds"]],
            solo_rps_valid=float(rps_6x4(p_s27[vm], gold[vm])),
            dRPS_all=float(dm27_a),
            ci90_all=[float(lo27_a), float(hi27_a)],
            dRPS_orig=float(dm27_o),
            ci90_orig=[float(lo27_o), float(hi27_o)],
            same_negative_sign=bool(s27_same_sign),
        )

    # Full 9-member RoutedGateEUDecoder evaluation
    P_v6 = dict(P_v5)
    P_v6["p1_qy"] = p_s26
    cfg_v6 = build_proto_cfg(v5_m + ["p1_qy"])
    rep_v6 = eval_4part_and_tag(P_v6, gold, fold, groups, is_orig, tag_fold, cfg_v6, lams_cl=LAMS_V6)
    q_v6_routed = rep_v6.pop("q_p0")
    rdm_all, rlo_all, rhi_all = cluster_bootstrap_rps_delta(q_v6_routed, q_v5_routed, gold, groups)
    rdm_orig, rlo_orig, rhi_orig = cluster_bootstrap_rps_delta(
        q_v6_routed[is_orig], q_v5_routed[is_orig], gold[is_orig], groups[is_orig]
    )

    g1_pass = (
        (dm_all <= -0.00025)
        and (dm_orig <= -0.00035)
        and (hi_all < 0.0)
        and (hi_orig < 0.0)
        and bool(s27_same_sign)
    )
    g2_pass = (
        (rep_v6["mean4_doc2_all"] >= rep_v5["mean4_doc2_all"] - 0.05)
        and (rep_v6["mean4_clean_all"] >= rep_v5["mean4_clean_all"] - 0.05)
    )

    out_rep = dict(
        solo_p1_s2026=dict(
            rps_all=float(rps_6x4(p_s26, gold)),
            rps_orig=float(rps_6x4(p_s26[is_orig], gold[is_orig])),
            mAP14=float(solo_map14),
            per_dim_ap=[float(x) for x in solo_aps],
            argmax_doc2_all=float(solo_sc["doc2_strict_total"]),
        ),
        G1_fixed018=dict(
            pass_gate=bool(g1_pass),
            dRPS_all=float(dm_all),
            ci90_all=[float(lo_all), float(hi_all)],
            dRPS_orig=float(dm_orig),
            ci90_orig=[float(lo_orig), float(hi_orig)],
            routed_dRPS_all=float(rdm_all),
            routed_ci90_all=[float(rlo_all), float(rhi_all)],
            routed_dRPS_orig=float(rdm_orig),
            routed_ci90_orig=[float(rlo_orig), float(rhi_orig)],
            seed2027_control=s27_info,
        ),
        G2_no_harm=dict(
            pass_gate=bool(g2_pass),
            delta_mean4_doc2_all=float(rep_v6["mean4_doc2_all"] - rep_v5["mean4_doc2_all"]),
            delta_mean4_clean_all=float(rep_v6["mean4_clean_all"] - rep_v5["mean4_clean_all"]),
            delta_mean4_doc2_orig=float(rep_v6["mean4_doc2_orig"] - rep_v5["mean4_doc2_orig"]),
            delta_mean4_clean_orig=float(rep_v6["mean4_clean_orig"] - rep_v5["mean4_clean_orig"]),
            guard_edge325_folds=int(rep_v6["p0_n_edge_325"]),
        ),
        resident_v5=rep_v5,
        resident_v6=rep_v6,
    )
    out_json = os.path.join(os.path.dirname(os.path.abspath(oof_s2026)), "v6_gate_report.json")
    json.dump(out_rep, open(out_json, "w"), indent=2, ensure_ascii=False)

    print(
        f"=== [resident_v6 Honest 8-Fold Gate Evaluation] ===\n"
        f"  Solo p1_qy (s2026): RPS_ALL={out_rep['solo_p1_s2026']['rps_all']:.5f} | RPS_ORIG={out_rep['solo_p1_s2026']['rps_orig']:.5f} "
        f"| mAP14={solo_map14:.4f} | OG_AP={solo_aps[4]:.4f} | Argmax={solo_sc['doc2_strict_total']:.2f}\n"
        f"  G1 (fixed 0.18): PASS={g1_pass} | dRPS_ALL={dm_all:+.5f} [{lo_all:+.5f}, {hi_all:+.5f}] "
        f"| dRPS_ORIG={dm_orig:+.5f} [{lo_orig:+.5f}, {hi_orig:+.5f}] | s2027_same_sign={s27_same_sign}\n"
        f"  G2 (4-part mean): PASS={g2_pass} | Doc2_ALL: {rep_v5['mean4_doc2_all']:.3f} -> {rep_v6['mean4_doc2_all']:.3f} "
        f"({rep_v6['mean4_doc2_all']-rep_v5['mean4_doc2_all']:+.3f}) | Doc2_ORIG: {rep_v5['mean4_doc2_orig']:.3f} -> "
        f"{rep_v6['mean4_doc2_orig']:.3f} ({rep_v6['mean4_doc2_orig']-rep_v5['mean4_doc2_orig']:+.3f}) | "
        f"Clean_ALL: {rep_v5['mean4_clean_all']:.3f} -> {rep_v6['mean4_clean_all']:.3f} | edge325={rep_v6['p0_n_edge_325']}\n"
        f"  TagSplit Doc2S: ALL {rep_v5['tag_doc2_all']:.3f} -> {rep_v6['tag_doc2_all']:.3f} | "
        f"ORIG {rep_v5['tag_doc2_orig']:.3f} -> {rep_v6['tag_doc2_orig']:.3f}\n"
        f"  Saved -> {out_json}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-register", action="store_true")
    ap.add_argument("--eval-v6", action="store_true")
    ap.add_argument("--oof-s2026", default=os.path.join(REPO, "results/r55/vlm_qy8/p1_c233397_s2026_oof.npz"))
    ap.add_argument("--oof-s2027", default=os.path.join(REPO, "results/r55/vlm_qy8/p1_c233397_s2027_oof.npz"))
    a = ap.parse_args()
    if a.pre_register:
        run_pre_register()
    if a.eval_v6:
        run_eval_v6(a.oof_s2026, a.oof_s2027)


if __name__ == "__main__":
    main()
