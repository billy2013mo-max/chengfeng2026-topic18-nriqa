#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_step0a_spatial_gate.py
====================================
Pre-registered Step 0a Gate (Claude-Fable-5.2 Round 3 §Stage 2):
  - Nested 8-fold L2-regularized ordinal/logistic model on `过度生成` (and secondary check on `过锐`, `模糊`):
      Baseline: Pool OOF probabilities `[P(0), P(1), P(2), P(3)]` + `P(虚化突兀=P0)`
      Augmented: Baseline + strictly the 6 pre-registered spatial-evidence features from `defect_attribution_map.py`:
        `[og_peak_drop, og_area_frac, og_gini_conc, og_anat_text_ov, og_bokeh_cliff_ov, og_vs_bokeh_cos]`
  - Primary Pass Gate:
      `过度生成` adjacent-grade AUC at `P2|P1` (grade 1 vs 2) OR `P1|P0` (grade 2 vs 3) improves by `>= +0.030`
      with 90% video-cluster bootstrap CI lower bound `> 0`.
  - Secondary Endpoint:
      Same-exact-pipeline-tag cross-video `PosAdj` ordering accuracy on `过度生成` (`~0.637` baseline).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    if os.path.join(REPO, _p) not in sys.path:
        sys.path.insert(0, os.path.join(REPO, _p))

import common55 as c
import final_engine as fe
from routed_eu_decoder import RoutedEUDecoder
from eval_specialist_gate import rps_6x4

TAG_RE = re.compile(r"_(v\d+[A-Za-z0-9]*?)(?:\.mp4|_|\.)")


def _get_exact_tag(name: str) -> str:
    m = TAG_RE.search(name)
    return m.group(1) if m else ""


def _cluster_boot_auc_delta(y_bin: np.ndarray, s_new: np.ndarray, s_old: np.ndarray, grps: np.ndarray, n_boot: int = 2000, seed: int = 2026):
    auc_old = float(roc_auc_score(y_bin, s_old))
    auc_new = float(roc_auc_score(y_bin, s_new))
    delta = auc_new - auc_old

    rng = np.random.default_rng(seed)
    u_g = np.unique(grps)
    g_rows = [np.where(grps == g)[0] for g in u_g]
    diffs = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(u_g), size=len(u_g))
        ix = np.concatenate([g_rows[p] for p in picks])
        yb = y_bin[ix]
        if yb.min() == yb.max():
            continue
        diffs.append(float(roc_auc_score(yb, s_new[ix]) - roc_auc_score(yb, s_old[ix])))
    lo, hi = np.percentile(diffs, [5.0, 95.0])
    return auc_old, auc_new, delta, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="midblock_8x8")
    ap.add_argument("--attr-dir", default="results/r55/attr_maps")
    args = ap.parse_args()

    names, gold, fold, groups = c.load_split()
    idx_map = {nm: i for i, nm in enumerate(names)}
    N = len(names)

    # Load 8-fold OOF spatial evidence features (strictly first 6 pre-registered features for 过度生成!)
    spat = np.zeros((N, 15), dtype=np.float64)
    seen = np.zeros(N, dtype=bool)
    for fk in range(8):
        p = os.path.join(REPO, args.attr_dir, f"oof_attr_{args.mode}_fold{fk}.npz")
        assert os.path.exists(p), f"Missing {p}"
        z = np.load(p, allow_pickle=True)
        for nm, sf in zip(z["names"], z["spat_feats"]):
            ii = idx_map[str(nm)]
            spat[ii] = sf
            seen[ii] = True
    assert seen.all(), f"Missing OOF spatial features: {seen.sum()}/{N}"

    # Build clean nested OOF pool probabilities (compact 6-tower base: cradio_v3, peg, sig_v3, v16, p3, p12_pe)
    members = ["cradio_v3", "peg", "sig_v3", "v16", "p3", "p12_pe"]
    _, _, _, _, P_all = fe.oof_pool(members)
    cfg = dict(
        base_members=["cradio_v3", "peg", "sig_v3", "v16", "p3"],
        base_prior=[1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 3],
        p12_members=["p12_pe"],
        p12_internal_weights=[1.0],
        p12_total_prior=0.18,
        l2_reg=0.015,
    )
    Q_pool = np.zeros((N, 6, 4), dtype=np.float64)
    for k in range(8):
        tr, va = fold != k, fold == k
        dec = RoutedEUDecoder(**cfg).fit({m: P_all[m][tr] for m in members}, gold[tr])
        Q_pool[va] = dec.pool_from_member_dict({m: P_all[m][va] for m in members})

    # Nested 8-fold CV with inner-CV C selection on {0.02, 0.1, 0.5, 2.0}
    X_base = np.concatenate([Q_pool[:, 4, :], Q_pool[:, 5, 3:4]], axis=1)  # [N, 5]
    X_aug = np.concatenate([X_base, spat[:, :6]], axis=1)                  # [N, 11] (<= 6 map features!)

    Q4_base_cal = np.zeros((N, 4), dtype=np.float64)
    Q4_aug_cal = np.zeros((N, 4), dtype=np.float64)

    for k in range(8):
        tr, va = fold != k, fold == k
        y_tr = gold[tr, 4]
        f_tr = fold[tr]
        inner_folds = np.unique(f_tr)

        def _fit_nested_multinomial(X_all: np.ndarray) -> np.ndarray:
            best_c, best_rps = 0.1, 1e9
            for c_val in (0.02, 0.1, 0.5, 2.0):
                oof_in = np.zeros((tr.sum(), 4), dtype=np.float64)
                for ik in inner_folds:
                    itr = f_tr != ik
                    iva = f_tr == ik
                    sc = StandardScaler().fit(X_all[tr][itr])
                    clf = LogisticRegression(C=c_val, max_iter=1000)
                    clf.fit(sc.transform(X_all[tr][itr]), y_tr[itr])
                    oof_in[iva] = clf.predict_proba(sc.transform(X_all[tr][iva]))
                # 1-dim RPS
                rps_1 = float(np.mean(np.sum((np.cumsum(oof_in, 1)[:, :3] - (np.arange(3)[None, :] >= y_tr[:, None]).astype(float) ** 0) ** 2, 1)))
                # exact 1-dim ordinal RPS:
                cdf_p = np.cumsum(oof_in, axis=1)[:, :3]
                cdf_y = (y_tr[:, None] <= np.arange(3)[None, :]).astype(np.float64)
                rps_1 = float(np.mean((cdf_p - cdf_y) ** 2))
                if rps_1 < best_rps:
                    best_rps, best_c = rps_1, c_val
            sc_full = StandardScaler().fit(X_all[tr])
            clf_full = LogisticRegression(C=best_c, max_iter=1000)
            clf_full.fit(sc_full.transform(X_all[tr]), y_tr)
            return clf_full.predict_proba(sc_full.transform(X_all[va]))

        Q4_base_cal[va] = _fit_nested_multinomial(X_base)
        Q4_aug_cal[va] = _fit_nested_multinomial(X_aug)

    # Also evaluate direct binary conditional boundary classifiers: P(grade=2 | grade in {1,2}) and P(grade=3 | grade in {2,3})
    s_p2p1_base = Q_pool[:, 4, 2] + Q_pool[:, 4, 3]  # P(>= P1)
    s_p1p0_base = Q_pool[:, 4, 3]                    # P(P0)

    s_p2p1_aug_bin = np.zeros(N, dtype=np.float64)
    s_p1p0_aug_bin = np.zeros(N, dtype=np.float64)
    s_p2p1_ctl_bin = np.zeros(N, dtype=np.float64)
    s_p1p0_ctl_bin = np.zeros(N, dtype=np.float64)

    for k in range(8):
        tr, va = fold != k, fold == k
        # Boundary 1: P2 (1) vs P1 (2) among positives (or {1, 2})
        m12_tr = tr & np.isin(gold[:, 4], [1, 2])
        m23_tr = tr & np.isin(gold[:, 4], [2, 3])
        for X_in, out12, out23 in [(X_base, s_p2p1_ctl_bin, s_p1p0_ctl_bin), (X_aug, s_p2p1_aug_bin, s_p1p0_aug_bin)]:
            for m_sub, pos_g, out_arr in [(m12_tr, 2, out12), (m23_tr, 3, out23)]:
                X_s = X_in[m_sub]
                y_s = (gold[m_sub, 4] == pos_g).astype(int)
                f_s = fold[m_sub]
                best_c, best_ll = 0.1, 1e9
                for c_val in (0.02, 0.1, 0.5, 2.0):
                    oof_b = np.zeros(len(y_s), dtype=np.float64)
                    for ik in np.unique(f_s):
                        itr, iva = f_s != ik, f_s == ik
                        if y_s[itr].min() == y_s[itr].max():
                            continue
                        sc_i = StandardScaler().fit(X_s[itr])
                        cb = LogisticRegression(C=c_val, max_iter=1000).fit(sc_i.transform(X_s[itr]), y_s[itr])
                        oof_b[iva] = cb.predict_proba(sc_i.transform(X_s[iva]))[:, 1]
                    brier = float(np.mean((oof_b - y_s) ** 2))
                    if brier < best_ll:
                        best_ll, best_c = brier, c_val
                sc_f = StandardScaler().fit(X_s)
                cf = LogisticRegression(C=best_c, max_iter=1000).fit(sc_f.transform(X_s), y_s)
                out_arr[va] = cf.predict_proba(sc_f.transform(X_in[va]))[:, 1]

    y4 = gold[:, 4]
    m_12 = np.isin(y4, [1, 2])
    m_23 = np.isin(y4, [2, 3])

    s_mult_12_old = Q4_base_cal[:, 2] / np.maximum(1e-6, Q4_base_cal[:, 1] + Q4_base_cal[:, 2])
    s_mult_12_new = Q4_aug_cal[:, 2] / np.maximum(1e-6, Q4_aug_cal[:, 1] + Q4_aug_cal[:, 2])
    s_mult_23_old = Q4_base_cal[:, 3] / np.maximum(1e-6, Q4_base_cal[:, 2] + Q4_base_cal[:, 3])
    s_mult_23_new = Q4_aug_cal[:, 3] / np.maximum(1e-6, Q4_aug_cal[:, 2] + Q4_aug_cal[:, 3])

    r12_m = _cluster_boot_auc_delta((y4[m_12] == 2).astype(int), s_mult_12_new[m_12], s_mult_12_old[m_12], groups[m_12])
    r23_m = _cluster_boot_auc_delta((y4[m_23] == 3).astype(int), s_mult_23_new[m_23], s_mult_23_old[m_23], groups[m_23])
    r12_b = _cluster_boot_auc_delta((y4[m_12] == 2).astype(int), s_p2p1_aug_bin[m_12], s_p2p1_ctl_bin[m_12], groups[m_12])
    r23_b = _cluster_boot_auc_delta((y4[m_23] == 3).astype(int), s_p1p0_aug_bin[m_23], s_p1p0_ctl_bin[m_23], groups[m_23])

    # Secondary endpoint: Same-exact-pipeline-tag cross-video PosAdj ordering accuracy on 过度生成
    exact_tags = np.array([_get_exact_tag(n) for n in names])
    sev_old = Q4_base_cal[:, 1] + 2.0 * Q4_base_cal[:, 2] + 3.0 * Q4_base_cal[:, 3]
    sev_new = Q4_aug_cal[:, 1] + 2.0 * Q4_aug_cal[:, 2] + 3.0 * Q4_aug_cal[:, 3]
    rng = np.random.default_rng(2026)
    acc_old_l, acc_new_l = [], []
    for vt in np.unique(exact_tags[exact_tags != ""]):
        ix = np.where(exact_tags == vt)[0]
        if len(ix) < 2:
            continue
        ua = rng.integers(0, len(ix), size=40000)
        ub = rng.integers(0, len(ix), size=40000)
        u, v = ix[ua], ix[ub]
        m_pa = (groups[u] != groups[v]) & (y4[u] > 0) & (y4[v] > 0) & (np.abs(y4[u] - y4[v]) == 1)
        u, v = u[m_pa][:15000], v[m_pa][:15000]
        if len(u) == 0:
            continue
        sw = y4[u] > y4[v]
        lo_i, hi_i = np.where(sw, v, u), np.where(sw, u, v)
        acc_old_l.append((sev_old[hi_i] > sev_old[lo_i]).astype(float))
        acc_new_l.append((sev_new[hi_i] > sev_new[lo_i]).astype(float))
    st_old = float(np.mean(np.concatenate(acc_old_l)))
    st_new = float(np.mean(np.concatenate(acc_new_l)))

    passed = (
        (r12_m[2] >= 0.03 and r12_m[3] > 0)
        or (r23_m[2] >= 0.03 and r23_m[3] > 0)
        or (r12_b[2] >= 0.03 and r12_b[3] > 0)
        or (r23_b[2] >= 0.03 and r23_b[3] > 0)
    )

    rep = dict(
        mode=args.mode,
        multinomial_P2_vs_P1=dict(old=r12_m[0], new=r12_m[1], delta=r12_m[2], ci90=[r12_m[3], r12_m[4]]),
        multinomial_P1_vs_P0=dict(old=r23_m[0], new=r23_m[1], delta=r23_m[2], ci90=[r23_m[3], r23_m[4]]),
        binary_P2_vs_P1=dict(old=r12_b[0], new=r12_b[1], delta=r12_b[2], ci90=[r12_b[3], r12_b[4]]),
        binary_P1_vs_P0=dict(old=r23_b[0], new=r23_b[1], delta=r23_b[2], ci90=[r23_b[3], r23_b[4]]),
        same_exact_tag_posadj_ordering=dict(old=st_old, new=st_new, delta=st_new - st_old),
        step0a_gate_passed=bool(passed),
    )
    out_p = os.path.join(REPO, args.attr_dir, f"step0a_gate_{args.mode}.json")
    json.dump(rep, open(out_p, "w"), indent=2)
    print(json.dumps(rep, indent=2), flush=True)


if __name__ == "__main__":
    main()
