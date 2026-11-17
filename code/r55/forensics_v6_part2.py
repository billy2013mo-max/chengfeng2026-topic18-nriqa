#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    sys.path.insert(0, os.path.join(REPO, _p))

import common55 as c
import final_engine as FE
from eval_specialist_gate import rps_6x4, per_dim_ap

DIMS = c.DIMS
names, gold, fold, groups = c.load_split()
N = len(names)
name_to_idx = {n: i for i, n in enumerate(names)}
manifest_rows = {r["name"]: r for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
is_orig = np.array([manifest_rows[n]["split"] == "orig" for n in names])
g_orig = gold[is_orig]

all_members = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe", "p12_pe", "p12_g14"]
m_v5 = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe", "p12_pe"]
_, _, _, _, P_all = FE.oof_pool(all_members)

base_rps_all = rps_6x4(FE.pool(P_all, m_v5), gold)
base_rps_orig = rps_6x4(FE.pool(P_all, m_v5)[is_orig], g_orig)

for m in all_members:
    q_m = c.renorm(P_all[m])
    aps, map14_m = per_dim_ap(q_m, gold)
    og_ap = aps[4] if isinstance(aps, (list, tuple, np.ndarray)) else aps["过度生成"]
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
    print(f"    {m:10s} | RPS_all={rps_6x4(q_m, gold):.5f} RPS_orig={rps_6x4(q_m[is_orig], g_orig):.5f} | mAP14={map14_m:.4f} OG_AP={og_ap:.4f} | {tag}")

print("\n  4B. Pairwise Correlation of Expected Severity E[grade] Across Towers:")
sev = {m: (P_all[m] * np.array([0, 1, 2, 3])[None, None, :]).sum(-1).ravel() for m in all_members}
hdr = "          " + " ".join(f"{m[:8]:>8s}" for m in all_members)
print("   ", hdr)
for m1 in all_members:
    row_s = " ".join(f"{np.corrcoef(sev[m1], sev[m2])[0,1]:8.4f}" for m2 in all_members)
    print(f"    {m1:10s} {row_s}")

print("\n  4C. Honest Non-Leaked Audit of Fine-Tuned VLMs (Phoenix Pillar-1 Qwen3-VL-8B & Pillar-2 Clef-Flash-9B):")
for p_idx, p_label in [(1, "Pillar1_Qwen3VL_8B"), (2, "Pillar2_ClefFlash_9B")]:
    oof_p = os.path.join(REPO, f"results/phoenix/pillar{p_idx}_stage2-rl2r_oof.npz")
    if os.path.exists(oof_p):
        z = np.load(oof_p, allow_pickle=True)
        mp = {str(n): q for n, q in zip(z["names"], z["probs_6x4"].astype(np.float64))}
        va_fold_of = {}
        for fk in range(5):
            zf = np.load(os.path.join(REPO, f"results/phoenix/pillar{p_idx}_stage2-rl2r_fold{fk}.npz"), allow_pickle=True)
            for n in zf["names"]:
                va_fold_of[str(n)] = fk
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
            og_p = aps_pil[4] if isinstance(aps_pil, (list, tuple, np.ndarray)) else aps_pil["过度生成"]
            og_v = aps_v5[4] if isinstance(aps_v5, (list, tuple, np.ndarray)) else aps_v5["过度生成"]
            best_w, best_r = 0.0, r_v5
            for w in (0.05, 0.10, 0.15, 0.20, 0.25):
                rb = rps_6x4(c.renorm((1 - w) * q_v5_raw[msk] + w * Q_pil[msk]), gold[msk])
                if rb < best_r:
                    best_r, best_w = rb, w
            corr_v5 = np.corrcoef((Q_pil[msk] * np.arange(4)).sum(-1).ravel(), (q_v5_raw[msk] * np.arange(4)).sum(-1).ravel())[0, 1]
            print(f"      {sub_nm:13s}: standalone RPS={r_pil:.5f} (vs v5_pool={r_v5:.5f}) | mAP14={map14_pil:.4f} (vs {map14_v5:.4f}) | OG_AP={og_p:.4f} (vs {og_v:.4f}) | corr_w_v5={corr_v5:.4f} | best_blend(w={best_w:.2f}) dRPS={best_r - r_v5:+.5f}")

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
            og_j = aps_j[4] if isinstance(aps_j, (list, tuple, np.ndarray)) else aps_j["过度生成"]
            bl_j = aps_j[0] if isinstance(aps_j, (list, tuple, np.ndarray)) else aps_j["模糊"]
            sh_j = aps_j[2] if isinstance(aps_j, (list, tuple, np.ndarray)) else aps_j["过锐"]
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
            print(f"    {jf:22s} (n={valid_m.sum()}): RPS={rps_6x4(qj, gj):.5f} | mAP14={map14_j:.4f} | OG_AP={og_j:.4f} | Blur_AP={bl_j:.4f} | Sharp_AP={sh_j:.4f} | +v5_blend(T={best_cfg[0]},w={best_cfg[1]}) dRPS={best_r - rps_6x4(qv5_j, gj):+.5f}")

print("\n" + "=" * 100)
print("SECTION 5: SUBGROUP & CO-OCCURRENCE BOTTLENECK ANALYSIS")
print("=" * 100)
pos_mat = (gold > 0).astype(np.float64)
print("  5A. Defect Co-Occurrence P(col > 0 | row > 0) on ALL_3573:")
print("            " + " ".join(f"{d:>8s}" for d in DIMS))
for i in range(6):
    m = pos_mat[:, i] > 0
    vals = " ".join(f"{pos_mat[m, j].mean()*100:7.1f}%" for j in range(6))
    print(f"    {DIMS[i]:8s} {vals}  (n={int(m.sum())})")

q_v5 = FE.pool(P_all, m_v5)
pr_arg = np.argmax(q_v5, axis=-1)
print("\n  5B. Posterior Rank of True Grade When Ensemble Argmax Makes an Error (ALL_3573):")
for d in c.FULL_DIMS:
    err_m = pr_arg[:, d] != gold[:, d]
    q_err = q_v5[err_m, d]
    y_err = gold[err_m, d]
    ranks = np.argsort(-q_err, axis=1)
    r2 = (ranks[:, 1] == y_err).mean() * 100
    r3 = (ranks[:, 2] == y_err).mean() * 100
    mean_true_p = q_err[np.arange(len(y_err)), y_err].mean()
    print(f"    {DIMS[d]:8s} (n_err={err_m.sum():4d}/{N}): True grade was #2 choice={r2:5.1f}%, #3={r3:5.1f}% | mean Q(true_grade)={mean_true_p:.3f}")
