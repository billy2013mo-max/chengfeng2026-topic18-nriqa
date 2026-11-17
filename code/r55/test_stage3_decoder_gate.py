#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/test_stage3_decoder_gate.py
====================================
Stage 3 Verification (Claude-Fable-5.2 Round 3 §Stage 3):
  1. `peg` vs `pe_v3` vs `pe_v3 + dim3_peg` pooling audit (preserving `0.04252` OOF RPS AND `0.99986` unseen-source `花屏` AUC).
  2. Exact Lagrangian Gate-Constrained `DecoderEU`:
       - Fixed `beta = 1.0` (exact F1 threshold `F/2`; `VoteDecoder` dropped).
       - Prior-balanced statistics (`w_supp in {1.0, 0.5, 0.25, 0.0}`).
       - Lagrangian shadow price `alpha_pos >= 1.0` on `25 * S_pos` (`W_F_pos = alpha_pos * 25/14`)
         stopping `S_p0`'s `0.5 * Recall` bonus from cannibalizing borderline `P1/P2` into `P0`!
       - Pre-registered 10,000-draw 654-image video-cluster bootstrap constraint:
             P(25 * S_pos - 0.20 < 15.0) <= 0.5%
  3. 4-Condition Cross-Defect Meta-Calibration check (`C` by inner CV, no inter-tower std, 4 partitions).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    if os.path.join(REPO, _p) not in sys.path:
        sys.path.insert(0, os.path.join(REPO, _p))

import common55 as c
import decoder_eu as E
import final_engine as fe
from routed_eu_decoder import RoutedEUDecoder
from eval_specialist_gate import rps_6x4


class FastClusterBootstrap654:
    """Pre-indexes video groups once so 10,000 654-image draws run in ~8ms!"""
    def __init__(self, gold: np.ndarray, groups: np.ndarray, n_boot: int = 10000, seed: int = 2026):
        u_g, self.inv = np.unique(groups, return_inverse=True)
        self.G = len(u_g)
        self.gl = (gold[:, E.LD] == E.LG[None]).astype(np.float64)
        self.gold_cl = (gold == 0).all(axis=1)
        self.g_po = np.zeros((self.G, 14), dtype=np.float64)
        for j in range(14):
            self.g_po[:, j] = np.bincount(self.inv, weights=self.gl[:, j], minlength=self.G)
        self.g_cl_pos = np.bincount(self.inv, weights=self.gold_cl.astype(np.float64), minlength=self.G)
        self.g_cl_neg = np.bincount(self.inv, weights=(~self.gold_cl).astype(np.float64), minlength=self.G)

        mean_sz = len(gold) / float(self.G)
        n_pick = max(1, int(round(654.0 / mean_sz)))
        rng = np.random.default_rng(seed)
        # Represent 10,000 draws as a [n_boot, G] integer multiplicity matrix W_boot!
        picks = rng.integers(0, self.G, size=(n_boot, n_pick))
        W = np.zeros((n_boot, self.G), dtype=np.float64)
        rows = np.repeat(np.arange(n_boot), n_pick)
        np.add.at(W, (rows, picks.ravel()), 1.0)
        self.W = W  # [10000, G] -> matrix multiply takes 2 milliseconds!!
        self.po_boot = self.W @ self.g_po  # [10000, 14]
        self.cl_pos_boot = self.W @ self.g_cl_pos
        self.cl_neg_boot = self.W @ self.g_cl_neg

    def eval_pred(self, pred: np.ndarray, bias_allowance: float = 0.20) -> Tuple[float, float, float]:
        pl = (pred[:, E.LD] == E.LG[None]).astype(np.float64)
        tp_row = pl * self.gl
        g_tp = np.column_stack([np.bincount(self.inv, weights=tp_row[:, j], minlength=self.G) for j in range(14)])
        g_pr = np.column_stack([np.bincount(self.inv, weights=pl[:, j], minlength=self.G) for j in range(14)])

        tp = self.W @ g_tp
        pr = self.W @ g_pr
        po = self.po_boot
        f1 = np.where((pr + po) > 0, 2.0 * tp / np.maximum(1.0, pr + po), 1.0)
        s_pos = 25.0 * f1.mean(axis=1)

        cp0_row = (self.gold_cl & (pred == 3).any(axis=1)).astype(np.float64)
        g_cp0 = np.bincount(self.inv, weights=cp0_row, minlength=self.G)
        cp0 = self.W @ g_cp0
        cp0_fail = float(np.mean(cp0 > 0) * 100.0)

        pos_fail = float(np.mean((s_pos - bias_allowance) < 15.0) * 100.0)
        raw_fail = float(np.mean(s_pos < 15.0) * 100.0)
        return pos_fail, raw_fail, cp0_fail


def eu_predict_alpha(Q: np.ndarray, st: dict, lam: float, alpha_pos: float = 1.0) -> np.ndarray:
    """
    Exact Bayes plug-in EU with Lagrangian shadow price `alpha_pos >= 1.0` on `25 * S_pos`
    (`beta == 1.0` untouched: zero-crossing stays at exact `F / 2`!).
    """
    D = np.maximum(1.0, st["npred"] + st["npos"])
    p = Q[:, E.LD, E.LG]
    w_f = np.where(E.LG == 3, alpha_pos * (25.0 / 14.0) + 20.0 * 0.3 / 6.0, alpha_pos * (25.0 / 14.0))
    gain = w_f * (2.0 * p - st["F"]) / D + E.W_R * p / np.maximum(1.0, st["npos"]) + E.W_P * (p - st["P"]) / np.maximum(1.0, st["npred"])
    pred = np.zeros((len(Q), 6), dtype=np.int64)
    best = np.zeros((len(Q), 6), dtype=np.float64)
    for l, (d, g) in enumerate(E.LABELS):
        m = gain[:, l] > best[:, d]
        pred[m, d] = g
        best[m, d] = gain[m, l]
    if lam is not None:
        pred[best.sum(1) < lam * np.prod(Q[:, :, 0], 1) * 20.0 / max(1.0, st["ncl"])] = 0
    return pred


def fit_weighted_stats_alpha(Q: np.ndarray, gold: np.ndarray, row_w: np.ndarray, lam: float, alpha_pos: float = 1.0, iters: int = 15) -> dict:
    w = row_w[:, None]
    pred = E.Decoder55().predict(Q)
    st = {}
    for _ in range(iters):
        pl = (pred[:, E.LD] == E.LG[None]).astype(np.float64)
        gl = (gold[:, E.LD] == E.LG[None]).astype(np.float64)
        tp = (w * pl * gl).sum(0)
        npred = (w * pl).sum(0)
        npos = (w * gl).sum(0)
        ncl = float((row_w * (gold == 0).all(1).astype(np.float64)).sum())
        st = dict(
            F=2.0 * tp / np.maximum(1e-6, npred + npos),
            P=tp / np.maximum(1e-6, npred),
            npos=npos,
            npred=npred,
            ncl=max(1e-6, ncl),
            N=float(row_w.sum()),
        )
        new = eu_predict_alpha(Q, st, lam, alpha_pos=alpha_pos)
        if (new == pred).all():
            return st
        pred = new
    return st


def main() -> None:
    names, gold, fold, groups = c.load_split()
    man = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv"), encoding="utf-8"))}
    t_fold = np.array([int(r["fold"]) for r in csv.DictReader(open(os.path.join(REPO, "data/r55_tagsplit/folds.csv"), encoding="utf-8"))])
    is_orig = np.array([man[n] == "orig" for n in names], dtype=bool)
    is_supp = ~is_orig

    # 1) Compare `pe_v3` vs `peg` vs `pe_v3 (dims 0,1,2,4,5) + peg (dim 3)`
    members_all = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3", "p12_pe"]
    _, _, _, _, P_all = fe.oof_pool(members_all)

    P_pe_hybrid = P_all["pe_v3"].copy()
    P_pe_hybrid[:, 3, :] = P_all["peg"][:, 3, :]  # peg ONLY on dim 3 (花屏), keeping pe_v3's cleaner ECE on dims 0,1,2,4,5!
    P_all["pe_hybrid"] = P_pe_hybrid

    Q_variants: Dict[str, np.ndarray] = {}
    for tag, pe_key in [("pe_v3_only", "pe_v3"), ("peg_all6", "peg"), ("pe_hybrid_dim3_peg", "pe_hybrid")]:
        m_list = ["cradio_v3", pe_key, "sig_v3", "v16", "p3", "p12_pe"]
        cfg = dict(
            base_members=["cradio_v3", pe_key, "sig_v3", "v16", "p3"],
            base_prior=[1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 3],
            p12_members=["p12_pe"],
            p12_internal_weights=[1.0],
            p12_total_prior=0.18,
            l2_reg=0.015,
        )
        Q_oof = np.zeros((len(names), 6, 4), dtype=np.float64)
        for k in range(8):
            tr, va = fold != k, fold == k
            dec = RoutedEUDecoder(**cfg).fit({m: P_all[m][tr] for m in m_list}, gold[tr])
            Q_oof[va] = dec.pool_from_member_dict({m: P_all[m][va] for m in m_list})
        Q_variants[tag] = Q_oof
        print(f"[Pool Variant {tag:20s}] RPS_all={rps_6x4(Q_oof, gold):.5f} | RPS_orig={rps_6x4(Q_oof[is_orig], gold[is_orig]):.5f}")

    Q_best = Q_variants["pe_hybrid_dim3_peg"]

    # 2) Test Standard DecoderEU vs Lagrangian Gate-Constrained DecoderEU (alpha_pos, lam, w_supp)
    boot_all = FastClusterBootstrap654(gold, groups, n_boot=10000, seed=999)
    boot_orig = FastClusterBootstrap654(gold[is_orig], groups[is_orig], n_boot=10000, seed=888)

    # Baseline standard DecoderEU
    pred_std = np.zeros_like(gold)
    for k in range(8):
        tr, va = fold != k, fold == k
        eu0 = E.DecoderEU().fit(Q_best[tr], gold[tr])
        pred_std[va] = eu0.predict(Q_best[va])
    sa0 = c.score(pred_std, gold)
    so0 = c.score(pred_std[is_orig], gold[is_orig])
    ss0 = c.score(pred_std[is_supp], gold[is_supp])
    fa0, fra0, fc0 = boot_all.eval_pred(pred_std)
    fo0, fro0, fco0 = boot_orig.eval_pred(pred_std[is_orig])
    print(
        f"\n[Baseline DecoderEU] ALL={sa0['doc2_strict_total']:.3f} (Pos={sa0['doc2_S_pos']:.2f}, P0={sa0['doc2_S_p0']:.2f}, Cln={sa0['doc2_Clean2']:.2f}) "
        f"fail_all={fa0:.2f}% | ORIG={so0['doc2_strict_total']:.3f} (Pos={so0['doc2_S_pos']:.2f}, P0={so0['doc2_S_p0']:.2f}, Cln={so0['doc2_Clean2']:.2f}) "
        f"fail_orig={fo0:.2f}% (raw:{fro0:.2f}%) | SUPP={ss0['doc2_strict_total']:.3f}"
    )

    alphas_grid = (1.0, 1.35, 1.75, 2.25, 3.0, 4.0)
    lams_grid = (0.25, 0.50, 0.75, 1.0, 1.5, 2.0)

    report_rows = []
    for w_supp in (0.50, 0.25, 0.0):
        pred_oof = np.zeros_like(gold)
        chosen_params = []
        row_w_all = np.where(is_orig, 1.0, w_supp)
        for k in range(8):
            tr, va = fold != k, fold == k
            tr_o = tr & is_orig
            bt_tr_o = FastClusterBootstrap654(gold[tr_o], groups[tr_o], n_boot=10000, seed=10 + k)
            bt_tr_a = FastClusterBootstrap654(gold[tr], groups[tr], n_boot=10000, seed=20 + k)
            best = None
            for alp in alphas_grid:
                for lam in lams_grid:
                    st = fit_weighted_stats_alpha(Q_best[tr], gold[tr], row_w_all[tr], lam, alpha_pos=alp)
                    pr_tr = eu_predict_alpha(Q_best[tr], st, lam, alpha_pos=alp)
                    f_orig, _, cp0_o = bt_tr_o.eval_pred(pr_tr[is_orig[tr]], bias_allowance=0.20)
                    f_mix, _, _ = bt_tr_a.eval_pred(pr_tr, bias_allowance=0.20)
                    sc_o = float(c.score(pr_tr[is_orig[tr]], gold[tr_o])["doc2_strict_total"])
                    sc_a = float(c.score(pr_tr, gold[tr])["doc2_strict_total"])
                    obj = 0.5 * sc_o + 0.5 * sc_a
                    worst_f = max(f_orig, f_mix)
                    feasible = (worst_f <= 0.50) and (cp0_o <= 1.0)
                    key = (1 if feasible else 0, -round(worst_f, 3) if not feasible else 0.0, obj)
                    if best is None or key > best[0]:
                        best = (key, st, lam, alp)
            _, st_k, lam_k, alp_k = best
            chosen_params.append((alp_k, lam_k))
            pred_oof[va] = eu_predict_alpha(Q_best[va], st_k, lam_k, alpha_pos=alp_k)

        sa = c.score(pred_oof, gold)
        so = c.score(pred_oof[is_orig], gold[is_orig])
        ss = c.score(pred_oof[is_supp], gold[is_supp])
        fa, fra, fca = boot_all.eval_pred(pred_oof)
        fo, fro, fco = boot_orig.eval_pred(pred_oof[is_orig])
        print(
            f"[GateConstrainedEU w_supp={w_supp:.2f}] ALL={sa['doc2_strict_total']:.3f} (Pos={sa['doc2_S_pos']:.2f}, P0={sa['doc2_S_p0']:.2f}, Cln={sa['doc2_Clean2']:.2f}) "
            f"fail_all={fa:.2f}% | ORIG={so['doc2_strict_total']:.3f} (Pos={so['doc2_S_pos']:.2f}, P0={so['doc2_S_p0']:.2f}, Cln={so['doc2_Clean2']:.2f}) "
            f"fail_orig={fo:.2f}% (raw:{fro:.2f}%) | SUPP={ss['doc2_strict_total']:.3f} | params={chosen_params}"
        )
        report_rows.append(
            dict(
                w_supp=w_supp,
                all_total=float(sa["doc2_strict_total"]),
                all_pos=float(sa["doc2_S_pos"]),
                all_p0=float(sa["doc2_S_p0"]),
                all_cln=float(sa["doc2_Clean2"]),
                orig_total=float(so["doc2_strict_total"]),
                orig_pos=float(so["doc2_S_pos"]),
                orig_p0=float(so["doc2_S_p0"]),
                orig_cln=float(so["doc2_Clean2"]),
                supp_total=float(ss["doc2_strict_total"]),
                fail_all_pct=fa,
                fail_orig_pct=fo,
                raw_fail_orig_pct=fro,
                chosen_params=chosen_params,
            )
        )

    # 3) Also check 4-condition Cross-Defect Meta-Calibration (no inter-tower std!)
    eps = 1e-6
    X_meta = np.log(np.clip(Q_best, eps, 1.0 - eps)).reshape(len(names), 24)
    Q_meta = np.zeros_like(Q_best)
    for k in range(8):
        tr, va = fold != k, fold == k
        # Select C by inner 2-split CV on tr
        sub_f = (np.arange(tr.sum()) % 3 == 0)
        sc_x = StandardScaler().fit(X_meta[tr])
        X_tr = sc_x.transform(X_meta[tr])
        X_va = sc_x.transform(X_meta[va])
        best_c = 0.05
        best_r = 1e9
        for C_val in (0.01, 0.03, 0.10):
            q_sub = np.zeros((sub_f.sum(), 6, 4))
            for d in range(6):
                if d in (3, 5):
                    yb = (gold[tr][~sub_f, d] == 3).astype(int)
                    clf = LogisticRegression(C=C_val, max_iter=300).fit(X_tr[~sub_f], yb)
                    p3 = clf.predict_proba(X_tr[sub_f])[:, 1]
                    q_sub[:, d, 0] = 1.0 - p3
                    q_sub[:, d, 3] = p3
                else:
                    clf = LogisticRegression(C=C_val, max_iter=300).fit(X_tr[~sub_f], gold[tr][~sub_f, d])
                    q_sub[:, d, :] = clf.predict_proba(X_tr[sub_f])
            r = rps_6x4(0.5 * Q_best[tr][sub_f] + 0.5 * q_sub, gold[tr][sub_f])
            if r < best_r:
                best_r, best_c = r, C_val
        q_v = np.zeros((va.sum(), 6, 4))
        for d in range(6):
            if d in (3, 5):
                yb = (gold[tr, d] == 3).astype(int)
                clf = LogisticRegression(C=best_c, max_iter=300).fit(X_tr, yb)
                p3 = clf.predict_proba(X_va)[:, 1]
                q_v[:, d, 0] = 1.0 - p3
                q_v[:, d, 3] = p3
            else:
                clf = LogisticRegression(C=best_c, max_iter=300).fit(X_tr, gold[tr, d])
                q_v[:, d, :] = clf.predict_proba(X_va)
        Q_meta[va] = c.renorm(0.5 * Q_best[va] + 0.5 * q_v)
    print(
        f"\n[Meta-Cal Check (50% blend, no inter-tower std)] "
        f"RPS_all={rps_6x4(Q_meta, gold):.5f} (vs {rps_6x4(Q_best, gold):.5f}) | "
        f"RPS_orig={rps_6x4(Q_meta[is_orig], gold[is_orig]):.5f} (vs {rps_6x4(Q_best[is_orig], gold[is_orig]):.5f}) | "
        f"RPS_supp={rps_6x4(Q_meta[is_supp], gold[is_supp]):.5f} (vs {rps_6x4(Q_best[is_supp], gold[is_supp]):.5f})"
    )

    os.makedirs(os.path.join(REPO, "results/r55/stage3"), exist_ok=True)
    json.dump(dict(rows=report_rows), open(os.path.join(REPO, "results/r55/stage3/decoder_gate_audit.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
