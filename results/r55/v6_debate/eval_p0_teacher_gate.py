#!/usr/bin/env python3
"""
P0 Mega-Teacher Ceiling Probe Gate Evaluator (Folds 0 & 1, 2 seeds: 2026 & 2027).
Evaluates:
  1. Solo OOF RPS_all / RPS_orig / mAP14 of Qwen3-VL-32B and InternVL3_5-38B (each seed + 2-seed avg + dual blend) vs p1_qy (8B).
  2. Add-teacher (at alpha in 0.15, 0.25, 0.35) and Replace-p1_qy on top of resident_v6_core (frozen decoder + nested F0<->F1).
  3. Pre-registered Go/No-Go gate: add-teacher dRPS_ORIG <= -0.00020 with 10,000-draw cluster bootstrap CI90 < 0.
  4. Unconstrained ceiling on Folds 0 & 1 (all models + both 32B/38B teachers).
"""
import argparse
import csv
import json
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
from eval_specialist_gate import cluster_bootstrap_rps_delta, load_oof_dir, per_dim_ap, rps_6x4
from eval_v6_gates import build_proto_cfg, fit_routed_gate_eu_lams, LAMS_V6
from routed_eu_decoder import _fast_boot654_pos_risk


def load_teacher_f01(out_dir: str, names_all, seeds=(2026, 2027)):
    idx_map = {str(n): i for i, n in enumerate(names_all)}
    N = len(names_all)
    per_seed = {}
    for s in seeds:
        arr = np.zeros((N, 6, 4), dtype=np.float64)
        mask = np.zeros(N, dtype=bool)
        for fk in (0, 1):
            p = os.path.join(out_dir, f"p1_c233397_s{s}_fold{fk}.npz")
            if not os.path.isfile(p):
                print(f"[WARN] Missing {p}")
                continue
            z = np.load(p, allow_pickle=True)
            for nm, pr in zip(z["names"], z["probs_6x4"]):
                i = idx_map[str(nm)]
                arr[i] = np.asarray(pr, dtype=np.float64)
                mask[i] = True
        per_seed[s] = (c.renorm(arr), mask)
    return per_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-dir", default=os.path.join(REPO, "results/r55/teacher_p0_qwen3vl32b"))
    ap.add_argument("--intern-dir", default=os.path.join(REPO, "results/r55/teacher_p0_internvl35_38b"))
    ap.add_argument("--out-json", default=os.path.join(REPO, "results/r55/v6_debate/p0_teacher_gate_results.json"))
    args = ap.parse_args()

    names, gold, fold, grp = c.load_split()
    man = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    is_orig = np.array([man[str(n)] == "orig" for n in names], dtype=bool)
    m01 = (fold == 0) | (fold == 1)
    m01_orig = m01 & is_orig

    _, _, _, _, P_v6 = fe.oof_pool(fe.SETS["resident_v6"])
    p_g14_kd, _ = load_oof_dir(os.path.join(REPO, "results/next55/towers/r55_p12_kd_e4_s2026/pe_core_g14"), names)
    p1_s27 = load_teacher_f01(os.path.join(REPO, "results/r55/vlm_qy8"), names, seeds=(2027,))[2027]

    q_seeds = load_teacher_f01(args.qwen_dir, names)
    i_seeds = load_teacher_f01(args.intern_dir, names)

    def _avg_avail(d):
        arrs = [v[0] for v in d.values() if v[1][m01].all()]
        if not arrs:
            return None
        return c.renorm(np.mean(arrs, axis=0))

    teachers = {}
    if p1_s27[1][m01].all():
        teachers["ctrl_p1_8b_s2027"] = p1_s27[0]
    for s in (2026, 2027):
        if q_seeds[s][1][m01].all():
            teachers[f"qwen32b_s{s}"] = q_seeds[s][0]
        if i_seeds[s][1][m01].all():
            teachers[f"intern38b_s{s}"] = i_seeds[s][0]
    q_avg = _avg_avail(q_seeds)
    i_avg = _avg_avail(i_seeds)
    if q_avg is not None:
        teachers["qwen32b_2seed"] = q_avg
    if i_avg is not None:
        teachers["intern38b_2seed"] = i_avg
    if q_avg is not None and i_avg is not None:
        teachers["dual_32b_38b"] = c.renorm(0.5 * q_avg + 0.5 * i_avg)

    # Baseline solo p1_qy on Folds 0 & 1
    p1_qy = P_v6["p1_qy"]
    print("=== Solo OOF on Folds 0 & 1 (N_all=%d, N_orig=%d) ===" % (m01.sum(), m01_orig.sum()))
    solo_rep = {}
    for k, pr in [("p1_qy_8b", p1_qy), ("v16_6b", P_v6["v16"]), ("p12_g14_kd", p_g14_kd)] + list(teachers.items()):
        ra = rps_6x4(pr[m01], gold[m01])
        ro = rps_6x4(pr[m01_orig], gold[m01_orig])
        mp = per_dim_ap(pr[m01], gold[m01])[1]
        solo_rep[k] = dict(rps_all=float(ra), rps_orig=float(ro), map14=float(mp))
        print(f"  {k:18s} | RPS_all={ra:.5f} | RPS_orig={ro:.5f} | mAP14={mp:.4f}")

    # Baseline resident_v6_core & resident_v6 on Folds 0 & 1 (decoder fitted strictly on Folds 2..7!)
    m27 = ~m01
    core_m = fe.SETS["resident_v6_core"]
    cfg_core = build_proto_cfg(core_m)
    dec_core = fit_routed_gate_eu_lams({m: P_v6[m][m27] for m in core_m}, gold[m27], is_orig[m27], grp[m27], cfg_core, LAMS_V6)
    p_core_01 = dec_core.pool_from_member_dict({m: P_v6[m][m01] for m in core_m})
    pred_core_01 = dec_core.predict_from_member_dict({m: P_v6[m][m01] for m in core_m})
    d_core_all = float(c.score(pred_core_01, gold[m01])["doc2_strict_total"])
    d_core_orig = float(c.score(pred_core_01[is_orig[m01]], gold[m01_orig])["doc2_strict_total"])

    # Also resident_v6_core WITHOUT p1_qy (for replace-p1_qy test)
    core_nop1_m = [m for m in core_m if m != "p1_qy"]
    cfg_nop1 = build_proto_cfg(core_nop1_m)
    dec_nop1 = fit_routed_gate_eu_lams({m: P_v6[m][m27] for m in core_nop1_m}, gold[m27], is_orig[m27], grp[m27], cfg_nop1, LAMS_V6)
    p_nop1_01 = dec_nop1.pool_from_member_dict({m: P_v6[m][m01] for m in core_nop1_m})

    print(f"\n=== Base resident_v6_core on F0+F1 (decoder fitted on F2..7): RPS_all={rps_6x4(p_core_01, gold[m01]):.5f} "
          f"RPS_orig={rps_6x4(p_core_01[is_orig[m01]], gold[m01_orig]):.5f} Doc2S={d_core_all:.3f} | {d_core_orig:.3f} ===")

    gate_rep = {}
    for tname, t_pr in teachers.items():
        t_01 = t_pr[m01]
        t_res = {}
        for alpha in (0.15, 0.25, 0.35):
            blend = c.renorm((1.0 - alpha) * p_core_01 + alpha * t_01)
            ba_m, ba_lo, ba_hi = cluster_bootstrap_rps_delta(blend, p_core_01, gold[m01], grp[m01], n_boot=10000, seed=2026)
            bo_m, bo_lo, bo_hi = cluster_bootstrap_rps_delta(blend[is_orig[m01]], p_core_01[is_orig[m01]], gold[m01_orig], grp[m01_orig], n_boot=10000, seed=2026)
            passed = (bo_m <= -0.00020) and (bo_hi < 0.0)
            passed_kd = (bo_m <= -0.00050) and (bo_hi < 0.0)
            t_res[f"add_{alpha:.2f}"] = dict(
                d_all=dict(mean=ba_m, ci90=[ba_lo, ba_hi]),
                d_orig=dict(mean=bo_m, ci90=[bo_lo, bo_hi]),
                pass_gate=bool(passed),
                pass_kd_roi=bool(passed_kd),
            )
            print(f"  [ADD {tname:16s} a={alpha:.2f}] dRPS_all={ba_m:+.5f} [{ba_lo:+.5f},{ba_hi:+.5f}] "
                  f"| dRPS_orig={bo_m:+.5f} [{bo_lo:+.5f},{bo_hi:+.5f}] | P020={passed} PKD={passed_kd}")
        # Replace p1_qy at weight 0.18
        rep_blend = c.renorm(0.82 * p_nop1_01 + 0.18 * t_01)
        ba_rm, ba_rlo, ba_rhi = cluster_bootstrap_rps_delta(rep_blend, p_core_01, gold[m01], grp[m01], n_boot=10000, seed=2026)
        bo_rm, bo_rlo, bo_rhi = cluster_bootstrap_rps_delta(rep_blend[is_orig[m01]], p_core_01[is_orig[m01]], gold[m01_orig], grp[m01_orig], n_boot=10000, seed=2026)
        t_res["replace_p1_0.18"] = dict(
            d_all=dict(mean=ba_rm, ci90=[ba_rlo, ba_rhi]),
            d_orig=dict(mean=bo_rm, ci90=[bo_rlo, bo_rhi]),
        )
        print(f"  [REP {tname:16s} 0.18 ] dRPS_all={ba_rm:+.5f} [{ba_rlo:+.5f},{ba_rhi:+.5f}] "
              f"| dRPS_orig={bo_rm:+.5f} [{bo_rlo:+.5f},{bo_rhi:+.5f}]")
        gate_rep[tname] = t_res

    with open(args.out_json, "w") as f:
        json.dump(dict(solo=solo_rep, gates=gate_rep), f, indent=2)
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
