#!/usr/bin/env python3
"""Evaluate frozen-feature Top-3 encoder probes under strict 8-fold nested CV + resident_v6_core add-to-pool gate."""
import argparse
import csv
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

import common55 as c
import final_engine as fe
from eval_specialist_gate import cluster_bootstrap_rps_delta, per_dim_ap, rps_6x4
from eval_v6_gates import build_proto_cfg, fit_routed_gate_eu_lams, LAMS_V6


def fit_linear_probe_fold(X_tr, y_tr, X_va, wd=0.001, **_kwargs):
    sc = StandardScaler().fit(X_tr)
    xtr = sc.transform(X_tr)
    xva = sc.transform(X_va)
    p_va = np.zeros((len(X_va), 6, 4), dtype=np.float64)
    for d in range(6):
        clf = LogisticRegression(C=float(wd), max_iter=400).fit(xtr, y_tr[:, d])
        pr = clf.predict_proba(xva)
        for ci, cls in enumerate(clf.classes_):
            p_va[:, d, int(cls)] = pr[:, ci]
    return c.renorm(p_va)


def adj_aucs(probs, gold, dim=2):
    # Overgen = dim 2; P1 vs P2 (y in {1,2}) and P0 vs P1 (y in {2,3})
    out = {}
    for lo, hi, tag in [(1, 2, "P2_vs_P1"), (2, 3, "P1_vs_P0")]:
        m = (gold[:, dim] == lo) | (gold[:, dim] == hi)
        if m.sum() > 10 and (gold[m, dim] == hi).sum() > 2 and (gold[m, dim] == lo).sum() > 2:
            score = probs[m, dim, hi:].sum(axis=1)
            out[tag] = float(roc_auc_score((gold[m, dim] == hi).astype(int), score))
        else:
            out[tag] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True)
    ap.add_argument("--wd", type=float, default=0.05)
    args = ap.parse_args()

    names, gold, fold, grp = c.load_split()
    man = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    is_orig = np.array([man[str(n)] == "orig" for n in names], dtype=bool)
    idx_map = {str(n): i for i, n in enumerate(names)}

    core_m = fe.SETS["resident_v6_core"]
    _, _, _, _, P_v6 = fe.oof_pool(core_m)
    cfg_core = build_proto_cfg(core_m)

    N = len(names)
    p_base = np.zeros((N, 6, 4), dtype=np.float64)
    pred_base = np.zeros((N, 6), dtype=np.int64)
    for fk in range(8):
        m_tr, m_va = fold != fk, fold == fk
        dec = fit_routed_gate_eu_lams({m: P_v6[m][m_tr] for m in core_m}, gold[m_tr], is_orig[m_tr], grp[m_tr], cfg_core, LAMS_V6)
        p_base[m_va] = dec.pool_from_member_dict({m: P_v6[m][m_va] for m in core_m})
        pred_base[m_va] = dec.predict_from_member_dict({m: P_v6[m][m_va] for m in core_m})

    sc_b = c.score(pred_base[is_orig], gold[is_orig])
    print(f"[BASE resident_v6_core] RPS_all={rps_6x4(p_base, gold):.5f} RPS_orig={rps_6x4(p_base[is_orig], gold[is_orig]):.5f} "
          f"mAP14={per_dim_ap(p_base, gold)[1]:.4f} Doc2S_orig={sc_b['doc2_strict_total']:.3f}")

    for path in args.npz:
        z = np.load(path, allow_pickle=True)
        X_raw = z["X"]
        X = np.zeros((N, X_raw.shape[1]), dtype=np.float32)
        for nm, row in zip(z["names"], X_raw):
            X[idx_map[str(nm)]] = row

        p_probe = np.zeros((N, 6, 4), dtype=np.float64)
        for fk in range(8):
            m_tr, m_va = fold != fk, fold == fk
            p_probe[m_va] = fit_linear_probe_fold(X[m_tr], gold[m_tr], X[m_va], wd=args.wd, seed=2026 + fk)

        ra = rps_6x4(p_probe, gold)
        ro = rps_6x4(p_probe[is_orig], gold[is_orig])
        mp = per_dim_ap(p_probe, gold)[1]
        aucs = adj_aucs(p_probe[is_orig], gold[is_orig], dim=2)
        tag = os.path.basename(path)
        print(f"\n=== {tag} (dim={X.shape[1]}) ===")
        print(f"  Solo: RPS_all={ra:.5f} RPS_orig={ro:.5f} mAP14={mp:.4f} | Overgen adjAUC: {aucs}")
        for alpha in (0.05, 0.10, 0.15):
            blend = c.renorm((1.0 - alpha) * p_base + alpha * p_probe)
            ba_m, ba_lo, ba_hi = cluster_bootstrap_rps_delta(blend, p_base, gold, grp, n_boot=5000, seed=2026)
            bo_m, bo_lo, bo_hi = cluster_bootstrap_rps_delta(blend[is_orig], p_base[is_orig], gold[is_orig], grp[is_orig], n_boot=5000, seed=2026)
            print(f"  +core a={alpha:.2f}: dRPS_all={ba_m:+.5f} [{ba_lo:+.5f},{ba_hi:+.5f}] | dRPS_orig={bo_m:+.5f} [{bo_lo:+.5f},{bo_hi:+.5f}]")


if __name__ == "__main__":
    main()
