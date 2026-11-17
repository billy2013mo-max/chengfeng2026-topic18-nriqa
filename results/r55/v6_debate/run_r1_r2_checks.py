#!/usr/bin/env python3
import csv
import json
import os
import sys
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, "/wanqing-develop/luowenjing/18/code")
sys.path.insert(0, "/wanqing-develop/luowenjing/18/code/r55")
sys.path.insert(0, "/wanqing-develop/luowenjing/18/code/next55")

import common55 as c
import final_engine as fe
from routed_eu_decoder import RoutedGateEUDecoder, TrueGateConstrainedEU
from eval_specialist_gate import cluster_bootstrap_rps_delta, rps_6x4
from eval_v6_gates import build_proto_cfg

members = fe.SETS["resident_v6_core"]
names, gold, fold, _, P_dict = fe.oof_pool(members)
_, _, _, groups = c.load_split()
split_map = {r["name"]: r["split"] for r in csv.DictReader(open("/wanqing-develop/luowenjing/18/data/manifest_combined.csv"))}
is_orig = np.array([split_map[n] == "orig" for n in names])

# === R1: p1_qy s2026 vs s2027 vs 0.5*(s2026+s2027) on trained folds of s2027 ===
z27 = np.load("/wanqing-develop/luowenjing/18/results/r55/vlm_qy8/p1_c233397_s2027_oof.npz", allow_pickle=True)
mp27 = {str(n): q for n, q in zip(z27["names"], z27["probs_6x4"].astype(np.float64))}
p26 = P_dict["p1_qy"]
p27 = np.zeros_like(p26)
for i, n in enumerate(names):
    if str(n) in mp27:
        p27[i] = mp27[str(n)]
m6 = p27.sum(axis=(1, 2)) > 0.5
p_avg = p26.copy()
p_avg[m6] = c.renorm(0.5 * p26[m6] + 0.5 * p27[m6])

print(f"=== R1: p1_qy 2-seed check on {int(m6.sum())} images (folds {sorted(set(fold[m6].tolist()))}, orig={int((m6 & is_orig).sum())}) ===", flush=True)
print(f"  Solo p1_qy s2026 RPS (all|orig): {rps_6x4(p26[m6], gold[m6]):.5f} | {rps_6x4(p26[m6 & is_orig], gold[m6 & is_orig]):.5f}", flush=True)
print(f"  Solo p1_qy s2027 RPS (all|orig): {rps_6x4(p27[m6], gold[m6]):.5f} | {rps_6x4(p27[m6 & is_orig], gold[m6 & is_orig]):.5f}", flush=True)
print(f"  Solo p1_qy avg   RPS (all|orig): {rps_6x4(p_avg[m6], gold[m6]):.5f} | {rps_6x4(p_avg[m6 & is_orig], gold[m6 & is_orig]):.5f}", flush=True)

cfg = build_proto_cfg(members)
cfg["lams_cl"] = (1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25)
dec = RoutedGateEUDecoder(**cfg).fit(P_dict, gold, is_orig=is_orig, groups=groups)
q26 = dec.pool_from_member_dict(P_dict)
P_avg = dict(P_dict)
P_avg["p1_qy"] = p_avg
q_avg = dec.pool_from_member_dict(P_avg)

pred26 = dec.eu.predict(q26)
pred_avg = dec.eu.predict(q_avg)

d_all, lo_all, hi_all = cluster_bootstrap_rps_delta(q_avg[m6], q26[m6], gold[m6], groups[m6], n_boot=5000, seed=42)
m6o = m6 & is_orig
d_ori, lo_ori, hi_ori = cluster_bootstrap_rps_delta(q_avg[m6o], q26[m6o], gold[m6o], groups[m6o], n_boot=5000, seed=42)
sc26 = float(c.score(pred26[m6o], gold[m6o])["doc2_strict_total"])
scavg = float(c.score(pred_avg[m6o], gold[m6o])["doc2_strict_total"])
sc26_a = float(c.score(pred26[m6], gold[m6])["doc2_strict_total"])
scavg_a = float(c.score(pred_avg[m6], gold[m6])["doc2_strict_total"])
print(f"  In-pool frozen decoder dRPS_ALL  = {d_all:+.5f} CI90=[{lo_all:+.5f}, {hi_all:+.5f}]", flush=True)
print(f"  In-pool frozen decoder dRPS_ORIG = {d_ori:+.5f} CI90=[{lo_ori:+.5f}, {hi_ori:+.5f}]", flush=True)
print(f"  In-pool frozen Doc2S (all|orig): s2026={sc26_a:.3f}|{sc26:.3f} -> avg={scavg_a:.3f}|{scavg:.3f} (diff={scavg_a-sc26_a:+.3f}|{scavg-sc26:+.3f})", flush=True)

# === R2: Per-dimension decoder ceiling (in-sample optimistic upper bound) ===
print("\n=== R2: Per-dimension decoder in-sample ceiling on resident_v6_core ===", flush=True)
base_all = float(c.score(pred26, gold)["doc2_strict_total"])
base_ori = float(c.score(pred26[is_orig], gold[is_orig])["doc2_strict_total"])
print(f"  Global EU (lam_cl={dec.eu.lam_cl:.2f}, p0_scale={dec.eu.p0_scale:.2f}) insample Doc2S: all={base_all:.3f} | orig={base_ori:.3f}", flush=True)


def eval_per_dim(q_in, y_in, params):
    q_mod = q_in.copy()
    for d in range(6):
        q_mod[:, d, 2] *= params[d, 2]
        q_mod[:, d, 3] *= params[d, 1]
    q_mod = q_mod / np.clip(q_mod.sum(axis=-1, keepdims=True), 1e-12, None)
    pred_out = np.zeros((len(q_mod), 6), dtype=np.int64)
    st = dec.eu.st
    for d in range(6):
        eu_d = TrueGateConstrainedEU(w_supp=dec.eu.w_supp)
        eu_d.lam_cl = float(params[d, 0])
        eu_d.p0_scale = 1.0
        eu_d.st = st
        pred_out[:, d] = eu_d.predict(q_mod)[:, d]
    return float(c.score(pred_out, y_in)["doc2_strict_total"])


r2_res = {}
for target_name, idx_mask in [("orig", is_orig), ("all", np.ones(len(gold), dtype=bool))]:
    qt, yt = q26[idx_mask], gold[idx_mask]
    params = np.tile([dec.eu.lam_cl, dec.eu.p0_scale, 1.0], (6, 1))
    best_sc = eval_per_dim(qt, yt, params)
    for _pass in range(2):
        for d in range(6):
            for lam in (1.5, 2.0, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0):
                for p0s in (0.7, 0.85, 1.0, 1.15, 1.3):
                    for p1s in (0.7, 0.85, 1.0, 1.15, 1.3):
                        cand = params.copy()
                        cand[d] = [lam, p0s, p1s]
                        sc = eval_per_dim(qt, yt, cand)
                        if sc > best_sc + 1e-9:
                            best_sc = sc
                            params = cand
    base_t = base_ori if target_name == "orig" else base_all
    delta = best_sc - base_t
    r2_res[target_name] = dict(base=base_t, best=best_sc, delta=delta, params=params.tolist())
    print(f"  Optimistic 18-param per-dim ceiling ({target_name}): {best_sc:.3f} (delta={delta:+.3f})", flush=True)

out_json = "/wanqing-develop/luowenjing/18/results/r55/v6_debate/r1_r2_checks.json"
json.dump(
    dict(
        r1=dict(
            n_m6=int(m6.sum()),
            n_m6_orig=int(m6o.sum()),
            dRPS_ALL=float(d_all),
            CI90_ALL=[float(lo_all), float(hi_all)],
            dRPS_ORIG=float(d_ori),
            CI90_ORIG=[float(lo_ori), float(hi_ori)],
            doc2s_diff_all=float(scavg_a - sc26_a),
            doc2s_diff_orig=float(scavg - sc26),
        ),
        r2=r2_res,
    ),
    open(out_json, "w"),
    indent=2,
)
print(f"Saved {out_json}", flush=True)
