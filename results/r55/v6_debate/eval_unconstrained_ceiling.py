#!/usr/bin/env python3
"""Measure unconstrained OOF ceiling on existing 8-fold models + Folds 0/1 slice."""
import os
import sys
import numpy as np

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
from eval_specialist_gate import load_oof_dir, per_dim_ap, rps_6x4, cluster_bootstrap_rps_delta
from eval_v6_gates import build_proto_cfg, fit_routed_gate_eu_lams, LAMS_V6
from routed_eu_decoder import _fast_boot654_pos_risk


def eval_nested_8fold(P_dict, members, names, gold, fold, grp, is_orig):
    cfg = build_proto_cfg(members)
    N = len(names)
    oof_probs = np.zeros((N, 6, 4), dtype=np.float64)
    oof_pred = np.zeros((N, 6), dtype=np.int64)
    for fk in range(8):
        m_tr = fold != fk
        m_va = fold == fk
        P_tr = {m: P_dict[m][m_tr] for m in members}
        P_va = {m: P_dict[m][m_va] for m in members}
        dec = fit_routed_gate_eu_lams(P_tr, gold[m_tr], is_orig[m_tr], grp[m_tr], cfg, LAMS_V6)
        oof_probs[m_va] = dec.pool_from_member_dict(P_va)
        oof_pred[m_va] = dec.predict_from_member_dict(P_va)

    rps_all = rps_6x4(oof_probs, gold)
    rps_orig = rps_6x4(oof_probs[is_orig], gold[is_orig])
    map14 = per_dim_ap(oof_probs, gold)[1]
    sc_a = c.score(oof_pred, gold)
    sc_o = c.score(oof_pred[is_orig], gold[is_orig])
    return dict(
        rps_all=float(rps_all), rps_orig=float(rps_orig), map14=float(map14),
        doc2s_all=float(sc_a["doc2_strict_total"]), pos_all=float(sc_a["doc2_S_pos"]),
        p0_all=float(sc_a["doc2_S_p0"]), cln_all=float(sc_a["doc2_Clean2"]),
        doc2s_orig=float(sc_o["doc2_strict_total"]), pos_orig=float(sc_o["doc2_S_pos"]),
        p0_orig=float(sc_o["doc2_S_p0"]), cln_orig=float(sc_o["doc2_Clean2"]),
        oof_probs=oof_probs,
    )


def main():
    names, gold, fold, grp = c.load_split()
    _, _, _, _, P_v6 = fe.oof_pool(fe.SETS["resident_v6"])
    is_orig = np.array(["_g" not in str(n) for n in names], dtype=bool)

    p_g14_kd, _ = load_oof_dir(os.path.join(REPO, "results/next55/towers/r55_p12_kd_e4_s2026/pe_core_g14"), names)
    p_pe_kd, _ = load_oof_dir(os.path.join(REPO, "results/next55/towers/r55_p12_kd_e5_s2026/pe_core_large"), names)

    P_all = dict(P_v6)
    P_all["p12_g14"] = p_g14_kd
    P_all["p12_pe_kd"] = p_pe_kd
    # Register p12_pe_kd temporarily in fe.P12_SPEC / FAMILY for build_proto_cfg
    fe.P12_SPEC["p12_pe_kd"] = ("pe_core_large", 9)
    fe.FAMILY["p12_pe_kd"] = "p12"

    configs = [
        ("resident_v6_core (6m)", fe.SETS["resident_v6_core"]),
        ("resident_v6 (9m)", fe.SETS["resident_v6"]),
        ("v6 + p12_g14_kd (10m)", fe.SETS["resident_v6"] + ["p12_g14"]),
        ("v6 + p12_g14_kd + p12_pe_kd (11m)", fe.SETS["resident_v6"] + ["p12_g14", "p12_pe_kd"]),
    ]
    res = {}
    for label, members in configs:
        r = eval_nested_8fold(P_all, members, names, gold, fold, grp, is_orig)
        res[label] = r
        print(
            f"{label:38s} | RPS_all={r['rps_all']:.5f} RPS_orig={r['rps_orig']:.5f} mAP14={r['map14']:.4f} "
            f"| Doc2S_all={r['doc2s_all']:.3f} (Pos={r['pos_all']:.2f} P0={r['p0_all']:.2f} Cln={r['cln_all']:.2f}) "
            f"| Doc2S_orig={r['doc2s_orig']:.3f} (Pos={r['pos_orig']:.2f} P0={r['p0_orig']:.2f} Cln={r['cln_orig']:.2f})"
        )

    # Paired bootstrap vs resident_v6_core
    base = res["resident_v6_core (6m)"]["oof_probs"]
    for label, _ in configs[1:]:
        cand = res[label]["oof_probs"]
        d_all = cluster_bootstrap_rps_delta(cand, base, gold, grp, n_boot=5000, seed=2026)
        d_orig = cluster_bootstrap_rps_delta(cand[is_orig], base[is_orig], gold[is_orig], grp[is_orig], n_boot=5000, seed=2026)
        print(f"  vs core: {label:34s} | dRPS_all={d_all['mean']:+.5f} [{d_all['ci90'][0]:+.5f},{d_all['ci90'][1]:+.5f}] "
              f"| dRPS_orig={d_orig['mean']:+.5f} [{d_orig['ci90'][0]:+.5f},{d_orig['ci90'][1]:+.5f}]")


if __name__ == "__main__":
    main()
