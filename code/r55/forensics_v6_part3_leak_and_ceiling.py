#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, glob, os, re, sys
import numpy as np

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    sys.path.insert(0, os.path.join(REPO, _p))

import common55 as c
import final_engine as FE
from eval_specialist_gate import rps_6x4, per_dim_ap, cluster_bootstrap_rps_delta

names, gold, fold, groups = c.load_split()
N = len(names)
manifest_rows = {r["name"]: r for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
is_orig = np.array([manifest_rows[n]["split"] == "orig" for n in names])
sources = np.array([manifest_rows[n]["source"] for n in names])

def video_stem(nm: str) -> str:
    # Strip frame/timestamp suffixes aggressively to catch ANY same-video sibling!
    s = nm.split(".mp4")[0]
    s = re.sub(r"_(frame_\d+|\d+\.\d+s|\d+s)$", "", os.path.splitext(s)[0])
    # Also take leading ID token before first underscore if numeric/hex >= 8 chars
    tok0 = s.split("_")[0]
    return tok0 if len(tok0) >= 8 else s

stems = np.array([video_stem(n) for n in names])
print(f"Unique qingyu groups={len(set(groups))}, unique sources={len(set(sources))}, unique aggressive stems={len(set(stems))}")

m_v5 = ["cradio_v3", "pe_v3", "peg", "sig_v3", "v16", "p3_qy", "ntg_pe", "p12_pe"]
_, _, _, _, P_all = FE.oof_pool(m_v5)
q_v5 = FE.pool(P_all, m_v5)

for stage in ("cv5", "stage2-rl2r"):
    print(f"\n=== Phoenix Stage: {stage} ===")
    Q_p = {}
    strict_clean_mask = np.ones(N, dtype=bool)
    for p_idx, p_nm in [(1, "p1_qwen3vl8b"), (2, "p2_clefflash9b")]:
        va_fold = {}
        for fk in range(5):
            zf = np.load(os.path.join(REPO, f"results/phoenix/pillar{p_idx}_{stage}_fold{fk}.npz"), allow_pickle=True)
            for n in zf["names"]:
                va_fold[str(n)] = fk
        # Strictest possible non-leak check: neither qingyu group, nor manifest source, nor aggressive stem appears in another fold!
        grp_f, src_f, stm_f = {}, {}, {}
        for i, n in enumerate(names):
            fk = va_fold[n]
            grp_f.setdefault(groups[i], set()).add(fk)
            src_f.setdefault(sources[i], set()).add(fk)
            stm_f.setdefault(stems[i], set()).add(fk)
        nl = np.array([len(grp_f[groups[i]]) == 1 and len(stm_f[stems[i]]) == 1 for i in range(N)])
        strict_clean_mask &= nl

        z = np.load(os.path.join(REPO, f"results/phoenix/pillar{p_idx}_{stage}_oof.npz"), allow_pickle=True)
        mp = {str(n): q for n, q in zip(z["names"], z["probs_6x4"].astype(np.float64))}
        Q_p[p_nm] = c.renorm(np.stack([mp[n] for n in names]))

        print(f"  [{p_nm}] Strict zero-sibling rows: ALL={nl.sum()}/{N}, ORIG={(nl & is_orig).sum()}/{is_orig.sum()}")
        for tag, msk in [("Strict_ALL", nl), ("Strict_ORIG", nl & is_orig), ("Leaked_Rows_Only", ~nl)]:
            if msk.sum() < 20:
                continue
            r_solo = rps_6x4(Q_p[p_nm][msk], gold[msk])
            r_base = rps_6x4(q_v5[msk], gold[msk])
            q_bl = c.renorm(0.82 * q_v5[msk] + 0.18 * Q_p[p_nm][msk])
            dm, lo, hi = cluster_bootstrap_rps_delta(q_bl, q_v5[msk], gold[msk], groups[msk])
            print(f"    {tag:16s} (n={msk.sum():4d}): solo_RPS={r_solo:.5f} | v5_RPS={r_base:.5f} | +18% blend dRPS={dm:+.5f} CI90=[{lo:+.5f}, {hi:+.5f}]")

    # Both p1 + p2 together on strict_clean_mask
    for tag, msk in [("Strict_ALL", strict_clean_mask), ("Strict_ORIG", strict_clean_mask & is_orig)]:
        q_p12 = c.renorm(0.5 * Q_p["p1_qwen3vl8b"][msk] + 0.5 * Q_p["p2_clefflash9b"][msk])
        q_bl = c.renorm(0.78 * q_v5[msk] + 0.22 * q_p12)
        dm, lo, hi = cluster_bootstrap_rps_delta(q_bl, q_v5[msk], gold[msk], groups[msk])
        # Per-defect RPS delta
        per_d = []
        for d in range(6):
            # single defect RPS
            g1 = np.zeros_like(gold[msk]); g1[:, 0] = gold[msk][:, d]
            # compute exact 1-dim RPS:
            yd = gold[msk][:, d]
            yt_cum = np.stack([(yd >= 1), (yd >= 2), (yd >= 3)], axis=-1).astype(np.float64)
            qp_b = q_v5[msk][:, d, :]
            qp_n = q_bl[:, d, :]
            cb = np.stack([qp_b[:,1:].sum(-1), qp_b[:,2:].sum(-1), qp_b[:,3]], -1)
            cn = np.stack([qp_n[:,1:].sum(-1), qp_n[:,2:].sum(-1), qp_n[:,3]], -1)
            rb_d = ((cb - yt_cum)**2).mean()
            rn_d = ((cn - yt_cum)**2).mean()
            per_d.append(f"{c.DIMS[d]}:{rn_d-rb_d:+.5f}")
        print(f"  [BOTH p1+p2 +22% blend on {tag} (n={msk.sum()})]: dRPS={dm:+.5f} CI90=[{lo:+.5f}, {hi:+.5f}] | " + " ".join(per_d))
