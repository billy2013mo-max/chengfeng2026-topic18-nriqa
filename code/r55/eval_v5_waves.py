#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_v5_waves.py
=========================
Zero-leakage offline evaluation of Wave A and Wave B experiments against the exact
Round-2 `claude-fable-5-2` pre-registered statistical gates:

  Wave A:
    1) `--fast-fixed` equivalence check on folds 0-1 (`r55_ntg_fastcheck_s2026` vs `r55_ntg_s2026`)
    2) 8-fold `ntg_pe` epoch sweep (`E4` vs `E6` vs `E8`) inside `resident_v5` (`RoutedGateEUDecoder`)
       with 2,000x video-cluster bootstrap CI90 on `delta_RPS_all` and `delta_RPS_orig`.
       Pre-registered rule: keep `E8` unless `E4` or `E6` beats `E8` with CI90 excluding 0.

  Wave B (Folds 0-1 Controlled Screen):
    1) Multi-backbone native-tile candidates (`ntg_cradio_e4`, `ntg_dinov3_e6`, `ntg_convnext_e8`)
       evaluated on top of `resident_v5`, MINUS the matched same-slot seed control (`ntg_pe_ctrl_s2027`).
    2) `p12_pe_condb1` (2-seed mean: 2026, 2027) replacing `p12_pe` inside `resident_v5`,
       MINUS the matched same-teacher seed control (`p12_pe_ctrl` 2-seed mean: 2027, 2028).
    Pre-registered promotion gate: `delta_RPS <= -0.00015` vs same-slot seed control with CI90 excluding 0.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code/r55"))
sys.path.insert(0, os.path.join(REPO, "code/next55"))
sys.path.insert(0, os.path.join(REPO, "code"))

import common55 as c
import final_engine as fe
from eval_specialist_gate import cluster_bootstrap_rps_delta, load_oof_dir, rps_6x4
from routed_eu_decoder import RoutedGateEUDecoder, evaluate_nested_routed_gate_eu


def load_partial_oof(rel_dir: str, names: List[str], folds_req: Tuple[int, ...] = (0, 1)) -> Optional[np.ndarray]:
    arr, valid = load_oof_dir(rel_dir, names)
    names_arr, _, fold_arr, _ = c.load_split()
    mask_req = np.isin(fold_arr, folds_req)
    if not np.all(valid[mask_req]):
        return None
    return c.renorm(arr)


def eval_2fold_routed_pool(
    P_full_ref: Dict[str, np.ndarray],
    extra_member_name: Optional[str],
    extra_partial: Optional[np.ndarray],
    replace_member_name: Optional[str],
    replace_partial: Optional[np.ndarray],
    gold: np.ndarray,
    fold: np.ndarray,
    is_orig: np.ndarray,
) -> np.ndarray:
    """
    On folds k in (0, 1):
      - Fit `RoutedGateEUDecoder` on `tr = (fold != k)`?
        WAIT: for a 2-fold screen where `extra_partial` ONLY exists on `fold in (0, 1)`, we fit the router
        cross-validated between fold 0 and fold 1 (train on fold 1 -> eval fold 0; train on fold 0 -> eval fold 1)
        OR evaluate fixed-family-prior temperature-calibrated pool on `va = fold in (0, 1)` AND cross-fold `0 <-> 1`!
        Here we compute BOTH:
          (a) Exact zero-leakage cross-fold `fold 1 -> fold 0` and `fold 0 -> fold 1` `RoutedGateEUDecoder` pool `Q_cv01`
          (b) Parameter-free family-prior pool `Q_fam01` (immune to small-sample router variance on 1 fold).
    """
    m01 = np.isin(fold, (0, 1))
    P_sub = {m: v.copy() for m, v in P_full_ref.items()}
    base_members = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe"]
    p12_members = ["p12_pe"]

    if extra_member_name is not None and extra_partial is not None:
        P_sub[extra_member_name] = extra_partial
        base_members = base_members + [extra_member_name]
    if replace_member_name is not None and replace_partial is not None:
        P_sub[replace_member_name] = replace_partial

    base_prior = fe._base_family_weights(base_members) if all(m in fe.FAMILY for m in base_members) else []
    cfg = dict(
        base_members=base_members,
        base_prior=base_prior,
        p12_members=p12_members,
        p12_internal_weights=[1.0],
        p12_total_prior=0.18,
        l2_reg=0.03,  # Slightly stronger L2 since fitting on 1 fold (~436 rows)
        peg_rescue_thr=0.90,
        use_dirdiag=True,
        w_supp_cal=0.15,
        dirdiag_l2=0.05,
        t_max=3.80,
    )
    Q_cv = np.zeros((len(gold), 6, 4), dtype=np.float64)
    for k_ev, k_tr in ((0, 1), (1, 0)):
        tr, va = (fold == k_tr), (fold == k_ev)
        dec = RoutedGateEUDecoder(**cfg)
        streams_tr, prior = dec._build_streams({m: v[tr] for m, v in P_sub.items()})
        from eval_specialist_gate import apply_nested_routed_pool, fit_nested_routed_pool
        dec.temps, dec.W = fit_nested_routed_pool(streams_tr, gold[tr], prior, l2_reg=0.03, t_max=3.80)
        q_raw_tr = apply_nested_routed_pool(streams_tr, dec.temps, dec.W)
        dec.dirdiag_params = dec._fit_postpool_dirdiag(q_raw_tr, gold[tr], is_orig[tr], w_supp_cal=0.15, l2=0.05)
        Q_cv[va] = dec.pool_from_member_dict({m: v[va] for m, v in P_sub.items()})
    return Q_cv[m01]


def eval_2fold_prior_pool(
    P_full_ref: Dict[str, np.ndarray],
    extra_member_name: Optional[str],
    extra_partial: Optional[np.ndarray],
    replace_member_name: Optional[str],
    replace_partial: Optional[np.ndarray],
    fold: np.ndarray,
) -> np.ndarray:
    """Zero-fit family-balanced arithmetic pool on folds 0-1 (zero router estimation noise)."""
    m01 = np.isin(fold, (0, 1))
    P_sub = {m: v[m01].copy() for m, v in P_full_ref.items()}
    fam_map = {
        "cradio_v3": "t1", "pe_v3": "t1", "peg": "t1", "sig_v3": "t1",
        "v16": "v16", "p3_qy": "ph",
        "ntg_pe": "ntg", "ntg_extra": "ntg",
    }
    base_m = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe"]
    if extra_member_name is not None and extra_partial is not None:
        P_sub[extra_member_name] = extra_partial[m01]
        base_m.append(extra_member_name)
        fam_map[extra_member_name] = "ntg"
    if replace_member_name is not None and replace_partial is not None:
        P_sub[replace_member_name] = replace_partial[m01]

    eff_m = [m for m in base_m if m != "peg"]
    fams = [fam_map[m] for m in eff_m]
    ew = {m: 1.0 / (len(set(fams)) * fams.count(fam_map[m])) for m in eff_m}
    pe_w = ew["pe_v3"] * 0.5
    ew["pe_v3"] = pe_w
    ew["peg"] = pe_w
    bw = np.array([ew[m] for m in base_m], dtype=np.float64) * 0.82
    streams = [P_sub[m] for m in base_m] + [P_sub["p12_pe"]]
    ws = list(bw) + [0.18]
    return c.renorm(c.mean_pool(streams, ws))


def main():
    import csv
    os.chdir(REPO)
    names, gold, fold, groups = c.load_split()
    split_map = {row["name"]: row["split"] for row in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    is_orig = np.array([split_map[n] == "orig" for n in names])
    _, _, _, _, P_v5 = fe.oof_pool(fe.SETS["resident_v5"])
    _, _, _, _, P_v4 = fe.oof_pool(fe.SETS["resident_v4"])

    print("=" * 92)
    print("1. WAVE A1 — `--fast-fixed` Equivalence Check on Folds 0-1 (E8, seed 2026)")
    print("=" * 92)
    p_fast = load_partial_oof("results/next55/towers/r55_ntg_fastcheck_s2026/pe_core_large", names, (0, 1))
    m01 = np.isin(fold, (0, 1))
    if p_fast is None:
        print("  [PENDING] r55_ntg_fastcheck_s2026 folds 0-1 not yet complete.")
    else:
        p_orig_ntg = P_v5["ntg_pe"]
        rps_ref = rps_6x4(p_orig_ntg[m01], gold[m01])
        rps_fst = rps_6x4(p_fast[m01], gold[m01])
        dm, lo, hi = cluster_bootstrap_rps_delta(p_fast[m01], p_orig_ntg[m01], gold[m01], groups[m01])
        mad = float(np.mean(np.abs(p_fast[m01] - p_orig_ntg[m01])))
        print(f"  Standalone F0+F1 RPS: ref_ntg={rps_ref:.5f} vs fast_fixed={rps_fst:.5f} | "
              f"dRPS={dm:+.5f} [{lo:+.5f}, {hi:+.5f}] | mean|diff|={mad:.5f}")

    print("\n" + "=" * 92)
    print("2. WAVE A2 — 8-Fold `ntg_pe` Epoch Sweep (E4 vs E6 vs E8) inside `resident_v5`")
    print("=" * 92)
    base_v5 = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe"]
    cfg_v5 = dict(
        base_members=base_v5,
        base_prior=fe._base_family_weights(base_v5),
        p12_members=["p12_pe"],
        p12_internal_weights=[1.0],
        p12_total_prior=0.18,
        l2_reg=0.015,
        peg_rescue_thr=0.90,
        use_dirdiag=True,
        w_supp_cal=0.15,
        dirdiag_l2=0.02,
        t_max=3.80,
    )

    def get_oof_Q(P_in: Dict[str, np.ndarray]) -> Tuple[np.ndarray, dict]:
        Q_oof = np.zeros((len(gold), 6, 4), dtype=np.float64)
        pred_oof = np.zeros_like(gold)
        for k in range(8):
            tr, va = fold != k, fold == k
            dec_k = RoutedGateEUDecoder(**cfg_v5).fit(
                {m: v[tr] for m, v in P_in.items()}, gold[tr], is_orig=is_orig[tr], groups=groups[tr]
            )
            Q_oof[va] = dec_k.pool_from_member_dict({m: v[va] for m, v in P_in.items()})
            pred_oof[va] = dec_k.eu.predict(Q_oof[va])
        from routed_eu_decoder import _fast_boot654_pos_risk
        sc_all = c.score(pred_oof, gold)
        sc_orig = c.score(pred_oof[is_orig], gold[is_orig])
        perm_fail_15_o, _, _ = _fast_boot654_pos_risk(
            pred_oof[is_orig], gold[is_orig], groups[is_orig], n_boot=10000, seed=888, without_replacement=True, margin_buffer=0.0
        )
        res = dict(
            oof_rps_all=float(rps_6x4(Q_oof, gold)),
            oof_rps_orig=float(rps_6x4(Q_oof[is_orig], gold[is_orig])),
            oof_nested_doc2_all=float(sc_all["doc2_strict_total"]),
            oof_nested_doc2_orig=float(sc_orig["doc2_strict_total"]),
            perm654_orig_pos_lt15_pct=float(perm_fail_15_o),
        )
        return Q_oof, res

    Q_e8, res_e8 = get_oof_Q(P_v5)
    print(f"  [E8 baseline] standalone_RPS={rps_6x4(P_v5['ntg_pe'], gold):.5f} | "
          f"pool_RPS_all={res_e8['oof_rps_all']:.5f} orig={res_e8['oof_rps_orig']:.5f} | "
          f"Doc2_all={res_e8['oof_nested_doc2_all']:.3f} Doc2_orig={res_e8['oof_nested_doc2_orig']:.3f} | "
          f"Perm654<15={res_e8['perm654_orig_pos_lt15_pct']:.2f}%")

    for ep, rel in [(4, "results/next55/towers/r55_ntg_e4_s2026/pe_core_large"),
                    (6, "results/next55/towers/r55_ntg_e6_s2026/pe_core_large")]:
        arr, valid = load_oof_dir(rel, names)
        if not valid.all():
            if valid.any():
                r_part = rps_6x4(c.renorm(arr[valid]), gold[valid])
                r_e8_p = rps_6x4(P_v5["ntg_pe"][valid], gold[valid])
                done_f = sorted(set(int(x) for x in fold[valid]))
                print(f"  [E{ep}] PARTIAL folds={done_f} ({int(valid.sum())}/{len(names)} rows): "
                      f"standalone_RPS={r_part:.5f} vs E8_same_rows={r_e8_p:.5f} (diff={r_part - r_e8_p:+.5f})")
            else:
                print(f"  [E{ep}] PENDING (0/{len(names)} rows ready)")
            continue
        P_ep = dict(P_v5, ntg_pe=c.renorm(arr))
        Q_ep, res_ep = get_oof_Q(P_ep)
        dm_all, lo_all, hi_all = cluster_bootstrap_rps_delta(Q_ep, Q_e8, gold, groups)
        dm_org, lo_org, hi_org = cluster_bootstrap_rps_delta(Q_ep[is_orig], Q_e8[is_orig], gold[is_orig], groups[is_orig])
        win = (hi_all < 0.0) and (hi_org < 0.0)
        print(f"  [E{ep} vs E8]  standalone_RPS={rps_6x4(P_ep['ntg_pe'], gold):.5f} | "
              f"pool_RPS_all={res_ep['oof_rps_all']:.5f} (dRPS={dm_all:+.5f} [{lo_all:+.5f}, {hi_all:+.5f}]) | "
              f"orig={res_ep['oof_rps_orig']:.5f} (dRPS={dm_org:+.5f} [{lo_org:+.5f}, {hi_org:+.5f}]) | "
              f"Doc2_all={res_ep['oof_nested_doc2_all']:.3f} Doc2_orig={res_ep['oof_nested_doc2_orig']:.3f} | "
              f"Perm654<15={res_ep['perm654_orig_pos_lt15_pct']:.2f}% | BEAT_E8_GATE={win}")

    print("\n" + "=" * 92)
    print("3. WAVE B1 — Multi-Backbone Native-Tile 2-Fold Screen (Gate: dRPS <= -0.00015 vs Seed Ctrl)")
    print("=" * 92)
    p_ctrl_ntg = load_partial_oof("results/next55/towers/r55_ntg_pe_ctrl_s2027/pe_core_large", names, (0, 1))
    q_ref_fam = eval_2fold_prior_pool(P_v5, None, None, None, None, fold)
    q_ref_cv = eval_2fold_routed_pool(P_v5, None, None, None, None, gold, fold, is_orig)
    if p_ctrl_ntg is None:
        print("  [PENDING] Same-slot seed control `r55_ntg_pe_ctrl_s2027` folds 0-1 not yet complete.")
    else:
        q_ctrl_fam = eval_2fold_prior_pool(P_v5, "ntg_extra", p_ctrl_ntg, None, None, fold)
        q_ctrl_cv = eval_2fold_routed_pool(P_v5, "ntg_pe", p_ctrl_ntg, None, None, gold, fold, is_orig)
        dm_c, lo_c, hi_c = cluster_bootstrap_rps_delta(q_ctrl_fam, q_ref_fam, gold[m01], groups[m01])
        print(f"  [Seed Control: ntg_pe_s2027] standalone_RPS={rps_6x4(p_ctrl_ntg[m01], gold[m01]):.5f} | "
              f"prior_pool dRPS vs v5={dm_c:+.5f} [{lo_c:+.5f}, {hi_c:+.5f}]")
        for label, rel in [
            ("ntg_cradio_e4", "results/next55/towers/r55_ntg_cradio_e4_s2026/cradio_v4_h"),
            ("ntg_dinov3_e6", "results/next55/towers/r55_ntg_dinov3_e6_s2026/dinov3_vitl16"),
            ("ntg_convnext_e8", "results/next55/towers/r55_ntg_convnext_e8_s2026/convnextv2_base"),
        ]:
            p_cand = load_partial_oof(rel, names, (0, 1))
            if p_cand is None:
                print(f"  [{label:16s}] PENDING")
                continue
            q_cand_fam = eval_2fold_prior_pool(P_v5, "ntg_extra", p_cand, None, None, fold)
            dm_v5, lo_v5, hi_v5 = cluster_bootstrap_rps_delta(q_cand_fam, q_ref_fam, gold[m01], groups[m01])
            dm_ct, lo_ct, hi_ct = cluster_bootstrap_rps_delta(q_cand_fam, q_ctrl_fam, gold[m01], groups[m01])
            q_cand_cv = eval_2fold_routed_pool(P_v5, "ntg_pe", p_cand, None, None, gold, fold, is_orig)
            dm_cv, lo_cv, hi_cv = cluster_bootstrap_rps_delta(q_cand_cv, q_ctrl_cv, gold[m01], groups[m01])
            passed = (dm_ct <= -0.00015 and hi_ct < 0.0)
            print(f"  [{label:16s}] standalone_RPS={rps_6x4(p_cand[m01], gold[m01]):.5f} | "
                  f"vs_v5={dm_v5:+.5f} [{lo_v5:+.5f},{hi_v5:+.5f}] | "
                  f"vs_ctrl(fam)={dm_ct:+.5f} [{lo_ct:+.5f},{hi_ct:+.5f}] | "
                  f"vs_ctrl(cv)={dm_cv:+.5f} [{lo_cv:+.5f},{hi_cv:+.5f}] | PASS_GATE={passed}")

    print("\n" + "=" * 92)
    print("4. WAVE B2 — P12 Conditional-b1 (2 Seeds) vs Same-Teacher Controls (2 Seeds) on Folds 0-1")
    print("=" * 92)
    c26 = load_partial_oof("results/next55/towers/r55_p12_condb1_e5_s2026/pe_core_large", names, (0, 1))
    c27 = load_partial_oof("results/next55/towers/r55_p12_condb1_e5_s2027/pe_core_large", names, (0, 1))
    k27 = load_partial_oof("results/next55/towers/r55_p12_ctrl_e5_s2027/pe_core_large", names, (0, 1))
    k28 = load_partial_oof("results/next55/towers/r55_p12_ctrl_e5_s2028/pe_core_large", names, (0, 1))
    for nm, arr in [("condb1_s2026", c26), ("condb1_s2027", c27), ("ctrl_s2027", k27), ("ctrl_s2028", k28)]:
        st = f"standalone_RPS={rps_6x4(arr[m01], gold[m01]):.5f}" if arr is not None else "PENDING"
        print(f"  [{nm:14s}] {st}")
    if all(x is not None for x in (c26, c27, k27, k28)):
        p_cond_2s = c.renorm(0.5 * (c26 + c27))
        p_ctrl_2s = c.renorm(0.5 * (k27 + k28))
        dm_st, lo_st, hi_st = cluster_bootstrap_rps_delta(p_cond_2s[m01], p_ctrl_2s[m01], gold[m01], groups[m01])
        q_cond_fam = eval_2fold_prior_pool(P_v5, None, None, "p12_pe", p_cond_2s, fold)
        q_ctrl_fam = eval_2fold_prior_pool(P_v5, None, None, "p12_pe", p_ctrl_2s, fold)
        dm_pl, lo_pl, hi_pl = cluster_bootstrap_rps_delta(q_cond_fam, q_ctrl_fam, gold[m01], groups[m01])
        q_cond_cv = eval_2fold_routed_pool(P_v5, None, None, "p12_pe", p_cond_2s, gold, fold, is_orig)
        q_ctrl_cv = eval_2fold_routed_pool(P_v5, None, None, "p12_pe", p_ctrl_2s, gold, fold, is_orig)
        dm_cv, lo_cv, hi_cv = cluster_bootstrap_rps_delta(q_cond_cv, q_ctrl_cv, gold[m01], groups[m01])
        passed = (dm_pl <= -0.00015 and hi_pl < 0.0)
        print(f"  ==> [2-Seed Mean Comparison: condb1(2026,2027) vs ctrl(2027,2028)]")
        print(f"      Standalone dRPS = {dm_st:+.5f} [{lo_st:+.5f}, {hi_st:+.5f}]")
        print(f"      In-Pool (fam)   = {dm_pl:+.5f} [{lo_pl:+.5f}, {hi_pl:+.5f}] | In-Pool (cv) = {dm_cv:+.5f} [{lo_cv:+.5f}, {hi_cv:+.5f}] | PASS_GATE={passed}")


if __name__ == "__main__":
    main()
