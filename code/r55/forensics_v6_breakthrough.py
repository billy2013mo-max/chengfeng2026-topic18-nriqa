#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/forensics_v6_breakthrough.py
=====================================
Comprehensive, strictly zero-leakage OOF forensic & ceiling decomposition for the v6 Breakthrough Debate.
Strictly obeys AGENTS.md Rules 1-3:
  - Reads ZERO legacy test inference JSONs (`test_150` predictions are NOT touched).
  - 100% computed on labelled OOF (`3,573` total = `1,902` orig + `1,671` supp).
"""
import csv
import glob
import json
import os
import sys
import warnings
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import Ridge, LogisticRegression

warnings.filterwarnings("ignore")

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    sys.path.insert(0, os.path.join(REPO, _p))

import common55 as c
import decoder_eu as E
import final_engine as FE
from eval_specialist_gate import load_oof_dir, rps_6x4, per_dim_ap
from routed_eu_decoder import (
    RoutedGateEUDecoder,
    TrueGateConstrainedEU,
    _true_eu_predict,
)

DIMS = c.DIMS
GRADE_NAMES = ["Clean", "P2", "P1", "P0"]


def get_proto_cfg(members):
    p12_spec = FE.P12_SPEC
    base_m = [m for m in members if m not in p12_spec]
    base_prior = list(FE._base_family_weights(base_m))
    p12_m = [m for m in members if m in p12_spec]
    p12_int = [0.65, 0.35] if set(p12_m) == {"p12_pe", "p12_g14"} else [1.0 / max(1, len(p12_m))] * len(p12_m)
    p12_tot = 0.20 if len(p12_m) >= 2 else (0.18 if len(p12_m) == 1 else 0.0)
    return dict(
        base_members=base_m,
        base_prior=base_prior,
        p12_members=p12_m,
        p12_internal_weights=p12_int,
        p12_total_prior=p12_tot,
        l2_reg=0.015,
    )


def _fit_one_fold_gate(args):
    k, P_tr, P_va, g_tr, io_tr, grp_tr, cfg, use_dirdiag, t_max = args
    dec = RoutedGateEUDecoder(
        **cfg,
        use_dirdiag=use_dirdiag,
        w_supp_cal=0.15,
        dirdiag_l2=0.02,
        t_max=t_max,
    )
    if not use_dirdiag:
        dec.eu.w_supp = 0.25
    dec.fit(P_tr, g_tr, is_orig=io_tr, groups=grp_tr)
    return k, dec.pool_from_member_dict(P_va), dec.predict_from_member_dict(P_va)


def run_nested_routed_gate(P_dict, gold, fold, groups, is_orig, members, use_dirdiag=True, t_max=3.80):
    import concurrent.futures as cf
    cfg = get_proto_cfg(members)
    N = len(gold)
    Q_oof = np.zeros((N, 6, 4), dtype=np.float64)
    pred_oof = np.zeros((N, 6), dtype=np.int64)
    tasks = []
    for k in range(8):
        va = fold == k
        tr = ~va
        tasks.append((
            k,
            {m: P_dict[m][tr] for m in members},
            {m: P_dict[m][va] for m in members},
            gold[tr],
            is_orig[tr],
            groups[tr],
            cfg,
            use_dirdiag,
            t_max,
        ))
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        for k, q_va, p_va in ex.map(_fit_one_fold_gate, tasks):
            va = fold == k
            Q_oof[va] = q_va
            pred_oof[va] = p_va
    return Q_oof, pred_oof


def adj_auc(probs_4, y_col, g_lo, g_hi):
    m = (y_col == g_lo) | (y_col == g_hi)
    if m.sum() < 10 or (y_col[m] == g_hi).sum() < 3 or (y_col[m] == g_lo).sum() < 3:
        return np.nan
    # conditional score P(>= g_hi | >= g_lo) or log ratio p_hi / (p_lo + p_hi)
    s = probs_4[m, g_hi] / np.maximum(1e-9, probs_4[m, g_lo] + probs_4[m, g_hi])
    return float(roc_auc_score((y_col[m] == g_hi).astype(int), s))


def det_auc(probs_4, y_col):
    yb = (y_col > 0).astype(int)
    if yb.sum() < 5 or (1 - yb).sum() < 5:
        return np.nan
    return float(roc_auc_score(yb, 1.0 - probs_4[:, 0]))


def main():
    os.makedirs(os.path.join(REPO, "results/r55/v6_debate"), exist_ok=True)
    names, gold, fold, groups = c.load_split()
    N = len(names)
    name_to_idx = {n: i for i, n in enumerate(names)}
    manifest_rows = {r["name"]: r for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    is_orig = np.array([manifest_rows[n]["split"] == "orig" for n in names])

    # Load all candidate OOFs
    all_members = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe", "p12_pe", "p12_g14"]
    _, _, _, _, P_all = FE.oof_pool(all_members)

    print("=" * 100)
    print("SECTION 1: WHY ALL 3 RECENT SUBMISSIONS LANDED AT ~52.90 ONLINE (OOF DELTA + 150-IMAGE NOISE)")
    print("=" * 100)
    m_v4 = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "p12_pe"]
    m_v5 = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe", "p12_pe"]

    Q_v4, pr_v4 = run_nested_routed_gate(P_all, gold, fold, groups, is_orig, m_v4, use_dirdiag=False, t_max=1.60)
    Q_v4d, pr_v4d = run_nested_routed_gate(P_all, gold, fold, groups, is_orig, m_v4, use_dirdiag=True, t_max=3.80)
    Q_v5, pr_v5 = run_nested_routed_gate(P_all, gold, fold, groups, is_orig, m_v5, use_dirdiag=True, t_max=3.80)

    for tag, Q, pr in [("v4_routed_gate_eu", Q_v4, pr_v4), ("v4_dirdiag_gate_eu", Q_v4d, pr_v4d), ("v5_routed_gate_eu", Q_v5, pr_v5)]:
        sa = c.score(pr, gold)
        so = c.score(pr[is_orig], gold[is_orig])
        print(f"  [{tag:20s}] ALL: Doc2S={sa['doc2_strict_total']:.3f} (Pos={sa['doc2_S_pos']:.2f}, P0={sa['doc2_S_p0']:.2f}, Cln={sa['doc2_Clean2']:.2f}) RPS={rps_6x4(Q, gold):.5f}")
        print(f"  {'':22s} ORIG: Doc2S={so['doc2_strict_total']:.3f} (Pos={so['doc2_S_pos']:.2f}, P0={so['doc2_S_p0']:.2f}, Cln={so['doc2_Clean2']:.2f}) RPS={rps_6x4(Q[is_orig], gold[is_orig]):.5f}")

    for a_nm, a_pr, b_nm, b_pr in [("v4", pr_v4, "v4_dirdiag", pr_v4d), ("v4_dirdiag", pr_v4d, "v5", pr_v5), ("v4", pr_v4, "v5", pr_v5)]:
        diff_all = (a_pr != b_pr).mean() * 100
        diff_orig = (a_pr[is_orig] != b_pr[is_orig]).mean() * 100
        print(f"  Cell disagreement {a_nm} vs {b_nm}: ALL={diff_all:.2f}% ({int((a_pr!=b_pr).sum())}/{a_pr.size}), ORIG={diff_orig:.2f}% ({int((a_pr[is_orig]!=b_pr[is_orig]).sum())}/{a_pr[is_orig].size})")

    # 150-image subset Monte Carlo variance on ORIG (matching Clean=20 practice regime & general regime)
    rng = np.random.default_rng(2026)
    orig_idx = np.where(is_orig)[0]
    s_v4_150, s_v4d_150, s_v5_150 = [], [], []
    for _ in range(2000):
        sub = rng.choice(orig_idx, size=150, replace=False)
        s_v4_150.append(c.score(pr_v4[sub], gold[sub])["doc2_strict_total"])
        s_v4d_150.append(c.score(pr_v4d[sub], gold[sub])["doc2_strict_total"])
        s_v5_150.append(c.score(pr_v5[sub], gold[sub])["doc2_strict_total"])
    s_v4_150, s_v4d_150, s_v5_150 = map(np.array, (s_v4_150, s_v4d_150, s_v5_150))
    d_v5_v4 = s_v5_150 - s_v4_150
    print(f"  150-img ORIG draw SD: v4={s_v4_150.mean():.2f}±{s_v4_150.std():.2f} | v5={s_v5_150.mean():.2f}±{s_v5_150.std():.2f}")
    print(f"  150-img paired diff (v5 - v4): mean={d_v5_v4.mean():+.2f}, SD={d_v5_v4.std():.2f}, 90% interval=[{np.percentile(d_v5_v4,5):+.2f}, {np.percentile(d_v5_v4,95):+.2f}], P(v5 < v4 on a random 150-set)={(d_v5_v4<0).mean()*100:.1f}%")

    print("\n" + "=" * 100)
    print("SECTION 2: EXACT 14-CELL & P0 POINT-LOSS DECOMPOSITION OF resident_v5")
    print("=" * 100)
    for subset_tag, mask in [("ALL_3573", np.ones(N, dtype=bool)), ("ORIG_1902", is_orig)]:
        g_sub = gold[mask]
        p_sub = pr_v5[mask]
        q_sub = Q_v5[mask]
        ev = c.score(p_sub, g_sub)
        print(f"\n--- Subset: {subset_tag} (N={mask.sum()}) | Doc2S={ev['doc2_strict_total']:.3f} = S_pos {ev['doc2_S_pos']:.3f}/25 + S_p0 {ev['doc2_S_p0']:.3f}/20 + Clean2 {ev['doc2_Clean2']:.3f}/20 ---")
        print(f"{'Cell':14s} | {'Gold':>5s} {'Pred':>5s} {'TP':>4s} {'FP':>4s} {'FN':>4s} | {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'AP':>6s} | {'PosLost':>7s} {'P0Lost':>7s} {'TotLost':>7s} | {'FN_Adj%':>7s} {'FN_XDef%':>8s} {'FN_Cln%':>7s} | {'FP_Adj%':>7s} {'FP_XDef%':>8s} {'FP_Cln%':>7s}")
        print("-" * 155)
        tot_pos_lost = 0.0
        tot_p0_lost = 0.0
        by_def_lost = {d: 0.0 for d in range(6)}
        for d, g in E.LABELS:
            cname = f"{DIMS[d]}_{GRADE_NAMES[g]}"
            yt = (g_sub[:, d] == g)
            yp = (p_sub[:, d] == g)
            tp = int((yt & yp).sum())
            fp = int((~yt & yp).sum())
            fn = int((yt & ~yp).sum())
            prec = tp / max(1, tp + fp) if (tp + fp) > 0 else (1.0 if yt.sum() == 0 else 0.0)
            rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 1.0
            f1 = 2 * tp / max(1, 2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
            ap = float(average_precision_score(yt.astype(int), q_sub[:, d, g])) if yt.sum() > 0 else 1.0
            pos_lost = (25.0 / 14.0) * (1.0 - f1)
            p0_lost = (20.0 / 6.0) * (1.0 - (0.5 * rec + 0.2 * prec + 0.3 * f1)) if g == 3 else 0.0
            tot_lost = pos_lost + p0_lost
            tot_pos_lost += pos_lost
            tot_p0_lost += p0_lost
            by_def_lost[d] += tot_lost

            # FN breakdown: yt=True, yp=False
            fn_m = yt & ~yp
            if fn_m.sum() > 0:
                fn_adj = (p_sub[fn_m, d] > 0).mean() * 100.0  # predicted another >0 grade on same defect!
                fn_xdef = ((p_sub[fn_m, d] == 0) & (p_sub[fn_m] > 0).any(axis=1)).mean() * 100.0
                fn_cln = ((p_sub[fn_m] == 0).all(axis=1)).mean() * 100.0
            else:
                fn_adj = fn_xdef = fn_cln = 0.0

            # FP breakdown: yt=False, yp=True
            fp_m = ~yt & yp
            if fp_m.sum() > 0:
                fp_adj = (g_sub[fp_m, d] > 0).mean() * 100.0  # gold was another >0 grade on same defect!
                fp_xdef = ((g_sub[fp_m, d] == 0) & (g_sub[fp_m] > 0).any(axis=1)).mean() * 100.0
                fp_cln = ((g_sub[fp_m] == 0).all(axis=1)).mean() * 100.0
            else:
                fp_adj = fp_xdef = fp_cln = 0.0

            print(f"{cname:14s} | {int(yt.sum()):5d} {int(yp.sum()):5d} {tp:4d} {fp:4d} {fn:4d} | {prec:6.3f} {rec:6.3f} {f1:6.3f} {ap:6.3f} | {pos_lost:7.3f} {p0_lost:7.3f} {tot_lost:7.3f} | {fn_adj:6.1f}% {fn_xdef:7.1f}% {fn_cln:6.1f}% | {fp_adj:6.1f}% {fp_xdef:7.1f}% {fp_cln:6.1f}%")
        print("-" * 155)
        print(f"TOTAL LOST: S_pos_lost={tot_pos_lost:.3f}/25.000, S_p0_lost={tot_p0_lost:.3f}/20.000, Clean2_lost={20.0-ev['doc2_Clean2']:.3f}/20.000 => Sum={tot_pos_lost+tot_p0_lost+20.0-ev['doc2_Clean2']:.3f}")
        print("Points lost grouped by defect:", ", ".join(f"{DIMS[d]}={by_def_lost[d]:.2f}" for d in range(6)))

        # Adjacent-grade vs Detection AUC per defect
        print("Per-defect AUCs (Det = >0 vs 0 | P1|P2 = >=P1 given >0 | P0|P1 = P0 given >=P1):")
        for d in range(6):
            da = det_auc(q_sub[:, d], g_sub[:, d])
            if d in c.FULL_DIMS:
                a12 = adj_auc(q_sub[:, d], g_sub[:, d], 1, 2)
                a23 = adj_auc(q_sub[:, d], g_sub[:, d], 2, 3)
                # Also cumulative P(>=2 | >=1) and P(==3 | >=2)
                m_gt0 = g_sub[:, d] >= 1
                auc_ge2 = roc_auc_score((g_sub[m_gt0, d] >= 2).astype(int), (q_sub[m_gt0, d, 2] + q_sub[m_gt0, d, 3]) / np.maximum(1e-9, 1.0 - q_sub[m_gt0, d, 0]))
                m_ge2 = g_sub[:, d] >= 2
                auc_eq3 = roc_auc_score((g_sub[m_ge2, d] == 3).astype(int), q_sub[m_ge2, d, 3] / np.maximum(1e-9, q_sub[m_ge2, d, 2] + q_sub[m_ge2, d, 3]))
                print(f"  {DIMS[d]:8s}: DetAUC={da:.4f} | P1_vs_P2={a12:.4f} (cum >=P1|>0: {auc_ge2:.4f}) | P0_vs_P1={a23:.4f} (cum P0|>=P1: {auc_eq3:.4f})")
            else:
                print(f"  {DIMS[d]:8s}: DetAUC={da:.4f} (P0-only)")

        # Counterfactual Oracles: Detection-only Oracle vs Grading-only Oracle!
        # 1) Perfect Detection, Model Conditional Grade: where gold==0 -> pred=0; where gold>0 -> argmax_{g>=1} gain or Q[:,d,1:]
        pr_perf_det = np.zeros_like(p_sub)
        for d in range(6):
            m_pos = g_sub[:, d] > 0
            if d in c.FULL_DIMS:
                # conditional EU or conditional argmax
                pr_perf_det[m_pos, d] = np.where(p_sub[m_pos, d] > 0, p_sub[m_pos, d], 1 + np.argmax(q_sub[m_pos, d, 1:], axis=1))
            else:
                pr_perf_det[m_pos, d] = 3
        # 2) Model Detection, Perfect Conditional Grade: where p_sub>0 and gold>0 -> pred = gold; where p_sub>0 and gold==0 -> keep p_sub (FP); where p_sub==0 -> 0
        pr_perf_grd = p_sub.copy()
        both_pos = (p_sub > 0) & (g_sub > 0)
        pr_perf_grd[both_pos] = g_sub[both_pos]

        s_pd = c.score(pr_perf_det, g_sub)
        s_pg = c.score(pr_perf_grd, g_sub)
        print(f"  [Counterfactual 1: Perfect Detection + Current Grading] Doc2S={s_pd['doc2_strict_total']:.2f} (Δ={s_pd['doc2_strict_total']-ev['doc2_strict_total']:+.2f} | Pos={s_pd['doc2_S_pos']:.2f}, P0={s_pd['doc2_S_p0']:.2f}, Cln={s_pd['doc2_Clean2']:.2f})")
        print(f"  [Counterfactual 2: Current Detection + Perfect Grading] Doc2S={s_pg['doc2_strict_total']:.2f} (Δ={s_pg['doc2_strict_total']-ev['doc2_strict_total']:+.2f} | Pos={s_pg['doc2_S_pos']:.2f}, P0={s_pg['doc2_S_p0']:.2f}, Cln={s_pg['doc2_Clean2']:.2f})")

    print("\n" + "=" * 100)
    print("SECTION 3: BAYES POSTERIOR CEILING vs DECODER / 2ND-STAGE STACKER HEADROOM")
    print("=" * 100)
    # 3A: Calibration check & Bayes Self-Consistency Ceiling (Y_sim ~ Q_v5)
    # Check both calibrated Q_v5 (which has w_supp_cal=0.15 for orig) on ORIG_1902, AND an unweighted-calibrated Q on ALL_3573
    q_orig = c.renorm(Q_v5[is_orig])
    g_orig = gold[is_orig]
    # Reliability check on ORIG_1902 across all 14 cells
    p14 = q_orig[:, E.LD, E.LG].ravel()
    y14 = (g_orig[:, E.LD] == E.LG[None]).astype(np.float64).ravel()
    bins = np.linspace(0, 1, 11)
    ece_14 = 0.0
    print("  14-cell Reliability on ORIG_1902 (pred_prob vs empirical hit rate):")
    for b in range(10):
        m = (p14 >= bins[b]) & (p14 < bins[b + 1] if b < 9 else p14 <= bins[b + 1])
        if m.sum() > 0:
            mp, my = p14[m].mean(), y14[m].mean()
            ece_14 += (m.sum() / len(p14)) * abs(mp - my)
            print(f"    bin [{bins[b]:.1f}, {bins[b+1]:.1f}]: n={m.sum():5d} | mean_Q={mp:.3f} vs emp_gold={my:.3f} (diff={my-mp:+.3f})")
    print(f"  14-cell ECE on ORIG_1902 = {ece_14:.4f}")

    # Bayes Monte-Carlo ceiling if true data generator has posterior Q_orig
    rng = np.random.default_rng(42)
    sim_scores = []
    for _ in range(200):
        u = rng.random((len(q_orig), 6, 1))
        cdf = np.cumsum(q_orig, axis=-1)
        y_sim = (u > cdf).sum(axis=-1).astype(np.int64)
        sim_scores.append(c.score(pr_v5[is_orig], y_sim)["doc2_strict_total"])
    print(f"  Bayes Self-Consistency Score E_{'{Y~Q_orig}'}[Doc2(pr_v5, Y)] = {np.mean(sim_scores):.3f} ± {np.std(sim_scores):.3f} (vs actual gold score on ORIG: {c.score(pr_v5[is_orig], g_orig)['doc2_strict_total']:.3f}!)")

    # 3B: Decoder Headroom Test: Honest 8-Fold Coordinate Ascent of 14 Per-Cell Gain Multipliers + Clean Threshold on Q_v5!
    pr_ca = np.zeros_like(pr_v5)
    for k in range(8):
        va = fold == k
        tr = ~va
        tr_o = tr & is_orig
        # Fit base TrueGateConstrainedEU on tr
        eu = TrueGateConstrainedEU(w_supp=0.15).fit(Q_v5[tr], gold[tr], is_orig=is_orig[tr], groups=groups[tr])
        # Coordinate ascent on 14 cell multipliers alpha[14] around 1.0 on tr_o (with L1/grid regularization)
        alphas = np.ones(14, dtype=np.float64)
        def pred_with_alpha(Q_in, alp):
            Q_in = c.renorm(Q_in)
            st = eu.st
            D = np.maximum(1.0, st["npred"] + st["npos"])
            p = Q_in[:, E.LD, E.LG]
            w_f = np.where(E.LG == 3, 25.0 / 14.0 + eu.p0_scale * (20.0 * 0.3 / 6.0), 25.0 / 14.0)
            w_r = eu.p0_scale * E.W_R
            w_p = eu.p0_scale * E.W_P
            gain = alp[None, :] * (w_f * (2.0 * p - st["F"]) / D + w_r * p / np.maximum(1.0, st["npos"]) + w_p * (p - st["P"]) / np.maximum(1.0, st["npred"]))
            pred = np.zeros((len(Q_in), 6), dtype=np.int64)
            best_d = np.zeros((len(Q_in), 6), dtype=np.float64)
            for l, (d, g) in enumerate(E.LABELS):
                gl = gain[:, l]
                m = gl > best_d[:, d]
                pred[m, d] = g
                best_d[m, d] = gl[m]
            p_cl = np.prod(Q_in[:, :, 0], axis=1)
            ndef = max(1.0, st["N"] - st["ncl"])
            ncl = max(1.0, st["ncl"])
            clean_diff = 10.0 * (1.0 - p_cl) / ndef - eu.lam_cl * 10.0 * p_cl / ncl
            pred[(pred > 0).any(1) & ((best_d.sum(1) + clean_diff) < 0.0)] = 0
            return pred

        best_sc = c.score(pred_with_alpha(Q_v5[tr_o], alphas), gold[tr_o])["doc2_strict_total"]
        for _pass in range(2):
            for l in range(14):
                best_a = alphas[l]
                for cand in (0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35):
                    alp_try = alphas.copy()
                    alp_try[l] = cand
                    sc = c.score(pred_with_alpha(Q_v5[tr_o], alp_try), gold[tr_o])["doc2_strict_total"] - 0.15 * abs(cand - 1.0)
                    if sc > best_sc + 1e-6:
                        best_sc = sc
                        best_a = cand
                alphas[l] = best_a
        pr_ca[va] = pred_with_alpha(Q_v5[va], alphas)
    print(f"  [Nested 14-param Coordinate-Ascent Decoder on Q_v5] ALL={c.score(pr_ca, gold)['doc2_strict_total']:.3f} (vs {c.score(pr_v5, gold)['doc2_strict_total']:.3f}) | ORIG={c.score(pr_ca[is_orig], gold[is_orig])['doc2_strict_total']:.3f} (vs {c.score(pr_v5[is_orig], gold[is_orig])['doc2_strict_total']:.3f})")
    # Also check IN-SAMPLE 14-param coordinate ascent on ORIG just to know the absolute ceiling of tuning 14 thresholds even with oracle knowledge!
    eu_full = TrueGateConstrainedEU(w_supp=0.15).fit(Q_v5, gold, is_orig=is_orig, groups=groups)
    alphas_in = np.ones(14)
    def pred_in(Q_in, alp):
        Q_in = c.renorm(Q_in)
        st = eu_full.st
        D = np.maximum(1.0, st["npred"] + st["npos"])
        p = Q_in[:, E.LD, E.LG]
        w_f = np.where(E.LG == 3, 25.0 / 14.0 + eu_full.p0_scale * (20.0 * 0.3 / 6.0), 25.0 / 14.0)
        gain = alp[None, :] * (w_f * (2.0 * p - st["F"]) / D + eu_full.p0_scale * E.W_R * p / np.maximum(1.0, st["npos"]) + eu_full.p0_scale * E.W_P * (p - st["P"]) / np.maximum(1.0, st["npred"]))
        pred = np.zeros((len(Q_in), 6), dtype=np.int64)
        best_d = np.zeros((len(Q_in), 6), dtype=np.float64)
        for l, (d, g) in enumerate(E.LABELS):
            m = gain[:, l] > best_d[:, d]
            pred[m, d] = g
            best_d[m, d] = gain[m, l]
        p_cl = np.prod(Q_in[:, :, 0], axis=1)
        clean_diff = 10.0 * (1.0 - p_cl) / max(1.0, st["N"] - st["ncl"]) - eu_full.lam_cl * 10.0 * p_cl / max(1.0, st["ncl"])
        pred[(pred > 0).any(1) & ((best_d.sum(1) + clean_diff) < 0.0)] = 0
        return pred
    best_in = c.score(pred_in(Q_v5[is_orig], alphas_in), g_orig)["doc2_strict_total"]
    for _ in range(3):
        for l in range(14):
            for cand in np.linspace(0.6, 1.5, 19):
                at = alphas_in.copy(); at[l] = cand
                sc = c.score(pred_in(Q_v5[is_orig], at), g_orig)["doc2_strict_total"]
                if sc > best_in:
                    best_in = sc; alphas_in[l] = cand
    print(f"  [IN-SAMPLE Cheating 14-param Decoder Ceiling on ORIG Q_v5] ORIG={best_in:.3f} (only +{best_in - c.score(pr_v5[is_orig], g_orig)['doc2_strict_total']:.3f} even with in-sample oracle!)")

    # 3C: Cross-Defect + Multi-Tower 2nd-Stage Multinomial Stacker (8 towers x 24 log-probs = 192 features -> calibrated 6x4)
    X_stack = np.concatenate([np.log(c.renorm(P_all[m])).reshape(N, 24) for m in m_v5], axis=1)
    # Add cross-tower disagreement std per cell (24 features)
    std_stack = np.std(np.stack([c.renorm(P_all[m]) for m in m_v5], axis=0), axis=0).reshape(N, 24)
    X_full = np.concatenate([X_stack, std_stack], axis=1)
    Q_mlr = np.zeros((N, 6, 4), dtype=np.float64)
    for k in range(8):
        va = fold == k; tr = ~va
        for d in range(6):
            if d in c.FULL_DIMS:
                clf = LogisticRegression(C=0.05, max_iter=1000)
                clf.fit(X_full[tr], gold[tr, d])
                Q_mlr[va, d] = clf.predict_proba(X_full[va])
            else:
                clf = LogisticRegression(C=0.05, max_iter=1000)
                clf.fit(X_full[tr], (gold[tr, d] == 3).astype(int))
                p3 = clf.predict_proba(X_full[va])[:, 1]
                Q_mlr[va, d] = np.stack([1.0 - p3, np.full_like(p3, 1e-6), np.full_like(p3, 1e-6), p3], axis=1)
    Q_mlr = c.renorm(Q_mlr)
    Q_blend_mlr = c.renorm(0.75 * Q_v5 + 0.25 * Q_mlr)
    print(f"  [Cross-Defect + 8-Tower Multinomial Stacker (216 dims)] RPS_all: v5={rps_6x4(Q_v5, gold):.5f} vs stack={rps_6x4(Q_mlr, gold):.5f} vs 75/25_blend={rps_6x4(Q_blend_mlr, gold):.5f}")
    print(f"  [Cross-Defect + 8-Tower Multinomial Stacker (216 dims)] RPS_orig: v5={rps_6x4(Q_v5[is_orig], g_orig):.5f} vs stack={rps_6x4(Q_mlr[is_orig], g_orig):.5f} vs 75/25_blend={rps_6x4(Q_blend_mlr[is_orig], g_orig):.5f}")

    print("\n" + "=" * 100)
    print("SECTION 4: FULL TOWER AUDIT, PAIRWISE CORRELATIONS, LOO, AND UNTAPPED ASSETS ON DISK")
    print("=" * 100)
    print("  4A. Standalone & Leave-One-Out (LOO) of resident_v5 + p12_g14:")
    base_rps_all = rps_6x4(FE.pool(P_all, m_v5), gold)
    base_rps_orig = rps_6x4(FE.pool(P_all, m_v5)[is_orig], g_orig)
    for m in all_members:
        q_m = c.renorm(P_all[m])
        aps, map14_m = per_dim_ap(q_m, gold)
        if m in m_v5:
            rem = [x for x in m_v5 if x != m]
            q_rem = FE.pool(P_all, rem)
            loo_all = rps_6x4(q_rem, gold) - base_rps_all
            loo_orig = rps_6x4(q_rem[is_orig], g_orig) - base_rps_orig
            tag = f"LOO dRPS(drop): all={loo_all:+.5f}, orig={loo_orig:+.5f}"
        else:
            q_add = FE.pool(P_all, m_v5 + [m])
            add_all = rps_6x4(q_add, gold) - base_rps_all
            add_orig = rps_6x4(q_add[is_orig], g_orig) - base_rps_orig
            tag = f"ADD dRPS(+{m}): all={add_all:+.5f}, orig={add_orig:+.5f}"
        print(f"    {m:10s} | RPS_all={rps_6x4(q_m, gold):.5f} RPS_orig={rps_6x4(q_m[is_orig], g_orig):.5f} | mAP14={map14_m:.4f} OG_AP={aps['过度生成']:.4f} | {tag}")

    # Pairwise correlation of P(defect) logits across members
    print("\n  4B. Pairwise Correlation of Expected Severity E[grade] Across Towers:")
    sev = {m: (P_all[m] * np.array([0, 1, 2, 3])[None, None, :]).sum(-1).ravel() for m in all_members}
    hdr = "          " + " ".join(f"{m[:8]:>8s}" for m in all_members)
    print("   ", hdr)
    for m1 in all_members:
        row_s = " ".join(f"{np.corrcoef(sev[m1], sev[m2])[0,1]:8.4f}" for m2 in all_members)
        print(f"    {m1:10s} {row_s}")

    # 4C: Honest Non-Leaked Check of Phoenix Pillar-1 (Qwen3-VL-8B) and Pillar-2 (Clef-Flash-9B)!
    print("\n  4C. Honest Non-Leaked Audit of Fine-Tuned VLMs (Phoenix Pillar-1 Qwen3-VL-8B & Pillar-2 Clef-Flash-9B):")
    phx_split_csv = os.path.join(REPO, "data/phoenix_splits/phoenix_5fold.csv")
    if not os.path.exists(phx_split_csv):
        cands = glob.glob(os.path.join(REPO, "data/phoenix_splits/*.csv"))
        print("    phoenix_splits csvs:", cands)
    for p_idx, p_label in [(1, "Pillar1_Qwen3VL_8B"), (2, "Pillar2_ClefFlash_9B")]:
        oof_p = os.path.join(REPO, f"results/phoenix/pillar{p_idx}_stage2-rl2r_oof.npz")
        if os.path.exists(oof_p):
            z = np.load(oof_p, allow_pickle=True)
            mp = {str(n): q for n, q in zip(z["names"], z["probs_6x4"].astype(np.float64))}
            # Reconstruct exact 5 folds from fold0..fold4 files to identify ZERO-LEAKAGE validation rows!
            va_fold_of = {}
            for fk in range(5):
                zf = np.load(os.path.join(REPO, f"results/phoenix/pillar{p_idx}_stage2-rl2r_fold{fk}.npz"), allow_pickle=True)
                for n in zf["names"]:
                    va_fold_of[str(n)] = fk
            # A row i in fold fk is NON-LEAKED iff NO other row j with groups[j] == groups[i] had va_fold_of[j] != fk!
            grp_folds = {}
            for i, n in enumerate(names):
                grp_folds.setdefault(groups[i], set()).add(va_fold_of[n])
            non_leak = np.array([len(grp_folds[groups[i]]) == 1 for i in range(N)])
            Q_pil = c.renorm(np.stack([mp[n] for n in names]))
            q_v5_raw = FE.pool(P_all, m_v5)
            print(f"    [{p_label}] Non-leaked rows = {non_leak.sum()}/{N} ({non_leak.mean()*100:.1f}%), Non-leaked ORIG = {(non_leak & is_orig).sum()}/{is_orig.sum()}")
            for sub_nm, msk in [("NonLeak_ALL", non_leak), ("NonLeak_ORIG", non_leak & is_orig)]:
                r_pil = rps_6x4(Q_pil[msk], gold[msk])
                r_v5 = rps_6x4(q_v5_raw[msk], gold[msk])
                aps_pil, map14_pil = per_dim_ap(Q_pil[msk], gold[msk])
                aps_v5, map14_v5 = per_dim_ap(q_v5_raw[msk], gold[msk])
                # Blend test on non-leaked rows
                best_w, best_r = 0.0, r_v5
                for w in (0.05, 0.10, 0.15, 0.20, 0.25):
                    rb = rps_6x4(c.renorm((1 - w) * q_v5_raw[msk] + w * Q_pil[msk]), gold[msk])
                    if rb < best_r:
                        best_r, best_w = rb, w
                corr_v5 = np.corrcoef((Q_pil[msk] * np.arange(4)).sum(-1).ravel(), (q_v5_raw[msk] * np.arange(4)).sum(-1).ravel())[0, 1]
                print(f"      {sub_nm:13s}: standalone RPS={r_pil:.5f} (vs v5_pool={r_v5:.5f}) | mAP14={map14_pil:.4f} (vs {map14_v5:.4f}) | OG_AP={aps_pil['过度生成']:.4f} (vs {aps_v5['过度生成']:.4f}) | corr_w_v5={corr_v5:.4f} | best_blend(w={best_w:.2f}) dRPS={best_r - r_v5:+.5f}")

    # 4D: Zero-shot 27B Decision Models (clef.jsonl, jev-native.jsonl, qwen38-27b.jsonl) on ORIG_1902!
    print("\n  4D. Zero-Shot 27B VLM Probes on ORIG_1902 (results/*.jsonl):")
    opt_to_grade = {"无问题": 0, "P2": 1, "P1": 2, "P0": 3}
    dim_to_idx = {d: i for i, d in enumerate(DIMS)}
    for jf in ("clef.jsonl", "clef-flash.jsonl", "jev-native.jsonl", "qwen38-27b.jsonl", "nemotron-omni.jsonl"):
        jp = os.path.join(REPO, "results", jf)
        if os.path.exists(jp) and os.path.getsize(jp) > 0:
            Q_j = np.full((N, 6, 4), np.nan, dtype=np.float64)
            with open(jp) as f:
                for line in f:
                    obj = json.loads(line)
                    im = obj["image"]
                    if im in name_to_idx and obj["defect"] in dim_to_idx and obj.get("valid"):
                        ii = name_to_idx[im]
                        di = dim_to_idx[obj["defect"]]
                        opts = obj["options"]
                        pr_d = obj["probs"]
                        vec = np.zeros(4, dtype=np.float64)
                        for k_let, opt_nm in zip(["A", "B", "C", "D"], opts):
                            if opt_nm in opt_to_grade:
                                vec[opt_to_grade[opt_nm]] = float(pr_d.get(k_let, 0.0))
                        Q_j[ii, di] = vec
            valid_m = ~np.isnan(Q_j[:, :, 0]).any(axis=1)
            if valid_m.sum() > 100:
                qj = c.renorm(Q_j[valid_m])
                gj = gold[valid_m]
                qv5_j = FE.pool(P_all, m_v5)[valid_m]
                aps_j, map14_j = per_dim_ap(qj, gj)
                # Blend test after optimal temperature scaling
                best_r = rps_6x4(qv5_j, gj)
                best_cfg = (1.0, 0.0)
                for temp in (0.3, 0.5, 1.0, 2.0, 4.0):
                    lg = np.log(qj) / temp
                    lg -= lg.max(-1, keepdims=True)
                    qjt = c.renorm(np.exp(lg))
                    for w in (0.03, 0.06, 0.10, 0.15):
                        rb = rps_6x4(c.renorm((1 - w) * qv5_j + w * qjt), gj)
                        if rb < best_r:
                            best_r = rb; best_cfg = (temp, w)
                print(f"    {jf:22s} (n={valid_m.sum()}): RPS={rps_6x4(qj, gj):.5f} | mAP14={map14_j:.4f} | OG_AP={aps_j['过度生成']:.4f} | Blur_AP={aps_j['模糊']:.4f} | Sharp_AP={aps_j['过锐']:.4f} | +v5_blend(T={best_cfg[0]},w={best_cfg[1]}) dRPS={best_r - rps_6x4(qv5_j, gj):+.5f}")

    # 4E: Inspect InternViT-6B partial checkpoint metadata on disk
    iv_pt = os.path.join(REPO, "results/next55/towers/iv6b/internvit_6b/fold_0_best.pt")
    if os.path.exists(iv_pt):
        import torch
        ck = torch.load(iv_pt, map_location="cpu", weights_only=False)
        print(f"\n  4E. InternViT-6B checkpoint `fold_0_best.pt`: type={type(ck)}, len/keys={list(ck.keys())[:10] if isinstance(ck, dict) else 'state_dict'}")
        if isinstance(ck, dict):
            n_tensors = len(ck)
            tot_bytes = sum(v.numel() * v.element_size() for v in ck.values() if hasattr(v, "numel"))
            print(f"      tensors={n_tensors}, trainable/saved bytes={tot_bytes/1e6:.1f} MB, sample keys={list(ck.keys())[:6]}")

    print("\n" + "=" * 100)
    print("SECTION 5: SUBGROUP & CO-OCCURRENCE BOTTLENECK ANALYSIS ON resident_v5")
    print("=" * 100)
    # Co-occurrence matrix of gold > 0 on ORIG_1902 and ALL_3573
    pos_mat = (gold > 0).astype(np.float64)
    print("  5A. Defect Co-Occurrence P(col > 0 | row > 0) on ALL_3573:")
    print("            " + " ".join(f"{d:>8s}" for d in DIMS))
    for i in range(6):
        m = pos_mat[:, i] > 0
        vals = " ".join(f"{pos_mat[m, j].mean()*100:7.1f}%" for j in range(6))
        print(f"    {DIMS[i]:8s} {vals}  (n={int(m.sum())})")

    # Multi-frame video analysis on ALL_3573
    ug, cnts = np.unique(groups, return_counts=True)
    multi_grps = set(ug[cnts >= 2])
    is_multi = np.array([g in multi_grps for g in groups])
    print(f"\n  5B. Multi-frame vs Single-frame Videos: multi={is_multi.sum()} imgs ({len(multi_grps)} videos), single={(~is_multi).sum()} imgs")
    print(f"      Doc2S on Multi-frame={c.score(pr_v5[is_multi], gold[is_multi])['doc2_strict_total']:.3f} vs Single-frame={c.score(pr_v5[~is_multi], gold[~is_multi])['doc2_strict_total']:.3f}")

    # Confidence of errors: when resident_v5 misses a P1 or P2, how much probability did it put on the TRUE grade?
    print("\n  5C. Posterior Rank of True Grade When resident_v5 Makes an Error (ALL_3573):")
    for d in c.FULL_DIMS:
        err_m = pr_v5[:, d] != gold[:, d]
        q_err = Q_v5[err_m, d]
        y_err = gold[err_m, d]
        ranks = np.argsort(-q_err, axis=1)
        r1 = (ranks[:, 0] == y_err).mean() * 100  # argmax was actually right, EU shifted it!
        r2 = (ranks[:, 1] == y_err).mean() * 100  # true grade was #2 choice!
        r3 = (ranks[:, 2] == y_err).mean() * 100
        mean_true_p = q_err[np.arange(len(y_err)), y_err].mean()
        print(f"    {DIMS[d]:8s} (n_err={err_m.sum():4d}/{N}): True grade was Rank-1(argmax!)={r1:5.1f}%, Rank-2={r2:5.1f}%, Rank-3={r3:5.1f}% | mean Q(true_grade)={mean_true_p:.3f}")


if __name__ == "__main__":
    main()
