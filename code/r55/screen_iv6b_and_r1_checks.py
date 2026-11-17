#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/screen_iv6b_and_r1_checks.py
=====================================
Runs the exact empirical checks requested in Round 1 of the claude-fable-5-2 debate:
  1. 8-GPU parallel inference OOF extraction for the existing 8-fold `InternViT-6B` Epoch-2
     checkpoints (`results/next55/towers/iv6b/internvit_6b/fold_{0..7}_best.pt`, LoRA r=32)
     with runtime SDPA + BF16 autocast -> saves `results/r55/v6_debate/iv6b_ep2_8fold_oof.npz`
     and evaluates standalone RPS/mAP14 + in-pool ΔRPS + cluster bootstrap CI90 on ALL & ORIG!
  2. Sibling-label-as-prediction F1 noise ceiling per 14 cells + exact `phoenix_5fold.csv` group audit.
  3. Honest nested `RoutedGateEUDecoder` (`Doc2_Strict`) evaluation on `Strict_ALL (3,473)` &
     `Strict_ORIG (1,807)` comparing `resident_v5`, `v5_pruned (drop pe_v3, sig_v3)`,
     `v5_pruned + p1_cv5`, `v5_pruned + p2_cv5`, and `v5_pruned + p1_cv5 + p2_cv5`.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(REPO, "code"), os.path.join(REPO, "code/next55"), os.path.join(REPO, "code/r55")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _infer_iv6b_fold_worker(fold_id: int) -> str:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(fold_id)
    import torch
    from probe_v6_hw_budget import patch_internvit_sdpa
    from qingyu.backbones_qingyu import QingyuVisionQualityTower
    from qingyu.config_and_paths import LosslessNativeImageLoader, setup_sys_path_and_cwd
    from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM, PrecomputedDualEngineStore
    from qingyu.train_tier1_5backbones_8fold import load_fold_rows, make_tier1_collate_fn, predict_2x_tta

    setup_sys_path_and_cwd(chdir_repo=True)
    out_f = os.path.join(REPO, f"results/r55/v6_debate/iv6b_ep2_f{fold_id}.npz")
    if os.path.exists(out_f):
        return out_f

    dev = torch.device("cuda:0")
    _, va_rows = load_fold_rows(os.path.join(REPO, f"data/qingyu_splits/fold_{fold_id}.csv"))
    loader = LosslessNativeImageLoader(REPO)
    store = PrecomputedDualEngineStore(os.path.join(REPO, "results/qingyu/dual_engine_meta.npz"))

    model = QingyuVisionQualityTower(
        backbone_key="internvit_6b",
        phys_dim=PHYS_FEAT_DIM,
        use_lora_6b=True,
        lora_rank=32,
    )
    ck_path = os.path.join(REPO, f"results/next55/towers/iv6b/internvit_6b/fold_{fold_id}_best.pt")
    sd = torch.load(ck_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"Fold {fold_id} unexpected keys: {unexpected[:5]}"
    raw_m = model.backbone.base_model.model if hasattr(model.backbone, "base_model") else model.backbone
    patch_internvit_sdpa(raw_m)
    raw_m.encoder.gradient_checkpointing = False
    model.to(dev).eval()

    orig_single = model._encode_single_view
    def _bf16_single(x_in):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return orig_single(x_in)
    model._encode_single_view = _bf16_single

    collate = make_tier1_collate_fn("multi_crop_448")
    probs, logits, _, names_va = predict_2x_tta(
        model, va_rows, loader, store, collate, dev, batch_size=8, workers=2
    )
    np.savez_compressed(out_f, names=np.array(names_va, dtype=object), probs_6x4=probs.astype(np.float32), logits_6x4=logits.astype(np.float32))
    print(f"[IV6B Screen] Fold {fold_id} done ({len(names_va)} rows) -> {out_f}", flush=True)
    return out_f


def run_all() -> None:
    import common55 as c
    import decoder_eu as E
    from eval_specialist_gate import cluster_bootstrap_rps_delta, load_p0_fallback_base, per_dim_ap, rps_6x4
    from final_engine import P12_SPEC, SETS, _base_family_weights, oof_pool, pool
    from routed_eu_decoder import evaluate_nested_routed_gate_eu

    t0 = time.time()
    oof_agg = os.path.join(REPO, "results/r55/v6_debate/iv6b_ep2_8fold_oof.npz")
    names, gold, fold, groups = c.load_split()
    idx_map = {n: i for i, n in enumerate(names)}

    if not os.path.exists(oof_agg):
        import subprocess
        print("=== [1/3] Running 8-GPU parallel OOF screen on existing InternViT-6B Epoch-2 checkpoints ===", flush=True)
        procs = [
            subprocess.Popen([sys.executable, __file__, "--worker-fold", str(fid)])
            for fid in range(8)
        ]
        for fid, pr in enumerate(procs):
            rc = pr.wait()
            assert rc == 0, f"Fold {fid} worker exited with {rc}"
        part_files = [os.path.join(REPO, f"results/r55/v6_debate/iv6b_ep2_f{fid}.npz") for fid in range(8)]
        iv_probs = np.zeros((len(names), 6, 4), dtype=np.float32)
        seen = np.zeros(len(names), dtype=bool)
        for pf in part_files:
            z = np.load(pf, allow_pickle=True)
            for nm, pr in zip(z["names"], z["probs_6x4"]):
                ii = idx_map[str(nm)]
                iv_probs[ii] = pr
                seen[ii] = True
        assert seen.all(), f"Missing IV6B rows: {(~seen).sum()}"
        np.savez_compressed(oof_agg, names=np.array(names, dtype=object), probs_6x4=iv_probs)
        print(f"Saved full 8-fold IV6B Epoch-2 OOF ({time.time()-t0:.1f}s) -> {oof_agg}", flush=True)
    else:
        iv_probs = np.load(oof_agg, allow_pickle=True)["probs_6x4"]
    Q_iv6b = c.renorm(iv_probs.astype(np.float64))

    split_map = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO, "data/manifest_combined.csv")))}
    is_orig = np.array([split_map[n] == "orig" for n in names])

    # Load resident_v5 OOF dictionary
    _, _, _, _, P_v5 = oof_pool(SETS["resident_v5"])
    base_m = [m for m in SETS["resident_v5"] if m not in P12_SPEC]
    p12_m = [m for m in SETS["resident_v5"] if m in P12_SPEC]
    from routed_eu_decoder import RoutedGateEUDecoder
    dec_v5 = RoutedGateEUDecoder(
        base_members=base_m,
        base_prior=list(_base_family_weights(base_m)),
        p12_members=p12_m,
        p12_internal_weights=[1.0],
        p12_total_prior=0.18,
        l2_reg=0.015,
    )
    # Build honest nested Q_v5 and Q_v5_pruned (drop pe_v3, sig_v3)
    def build_honest_q(P_sub, m_list):
        bm = [m for m in m_list if m not in P12_SPEC]
        pm = [m for m in m_list if m in P12_SPEC]
        Q_out = np.zeros((len(names), 6, 4), dtype=np.float64)
        for k in range(8):
            tr, va = (fold != k), (fold == k)
            d_k = RoutedGateEUDecoder(
                base_members=bm,
                base_prior=list(_base_family_weights(bm)),
                p12_members=pm,
                p12_internal_weights=[1.0 / max(1, len(pm))] * len(pm),
                p12_total_prior=0.18 if pm else 0.0,
                l2_reg=0.015,
            ).fit({m: P_sub[m][tr] for m in m_list}, gold[tr], is_orig=is_orig[tr], groups=groups[tr])
            Q_out[va] = d_k.pool_from_member_dict({m: P_sub[m][va] for m in m_list})
        return Q_out

    Q_v5 = build_honest_q(P_v5, SETS["resident_v5"])
    pruned_members = ["cradio_v3", "peg", "v16", "p3_qy", "ntg_pe", "p12_pe"]
    Q_v5_pruned = build_honest_q(P_v5, pruned_members)

    aps_iv, map_iv = per_dim_ap(Q_iv6b, gold)
    print(f"\n=== [InternViT-6B Epoch-2 8-Fold OOF Screen] ===")
    print(f"Standalone IV6B Ep2: RPS_ALL={rps_6x4(Q_iv6b, gold):.5f} | RPS_ORIG={rps_6x4(Q_iv6b[is_orig], gold[is_orig]):.5f} | mAP14={map_iv:.4f} | OG_AP={aps_iv[4]:.4f}")
    for w in (0.08, 0.12, 0.16):
        Qb = c.renorm((1.0 - w) * Q_v5 + w * Q_iv6b)
        da, la, ha = cluster_bootstrap_rps_delta(Q_v5, Qb, gold, groups, n_boot=3000)
        do, lo, ho = cluster_bootstrap_rps_delta(Q_v5[is_orig], Qb[is_orig], gold[is_orig], groups[is_orig], n_boot=3000)
        print(f"  v5 + {w:.2f}*IV6B_Ep2: dRPS_ALL={da:+.5f} [{la:+.5f}, {ha:+.5f}] | dRPS_ORIG={do:+.5f} [{lo:+.5f}, {ho:+.5f}]")

    dp_a, lpa, hpa = cluster_bootstrap_rps_delta(Q_v5, Q_v5_pruned, gold, groups, n_boot=3000)
    dp_o, lpo, hpo = cluster_bootstrap_rps_delta(Q_v5[is_orig], Q_v5_pruned[is_orig], gold[is_orig], groups[is_orig], n_boot=3000)
    print(f"\n=== [G2 Pruning Check: drop pe_v3 + sig_v3] ===")
    print(f"  v5_pruned vs v5: dRPS_ALL={dp_a:+.5f} [{lpa:+.5f}, {hpa:+.5f}] | dRPS_ORIG={dp_o:+.5f} [{lpo:+.5f}, {hpo:+.5f}]")

    print(f"\n=== [2/3] Phoenix 5-Fold Group Audit & Sibling-Label Noise Ceiling ===")
    ph_rows = list(csv.DictReader(open(os.path.join(REPO, "data/phoenix_splits/phoenix_5fold.csv"))))
    ph_fold = {r["name"]: int(r["fold"]) for r in ph_rows}
    grp_to_ph_folds = {}
    grp_to_indices = {}
    for i, (nm, g) in enumerate(zip(names, groups)):
        grp_to_ph_folds.setdefault(g, set()).add(ph_fold[nm])
        grp_to_indices.setdefault(g, []).append(i)
    leaked_canon_groups = [g for g, fs in grp_to_ph_folds.items() if len(fs) > 1]
    strict_mask = np.array([len(grp_to_ph_folds[g]) == 1 for g in groups])
    print(f"phoenix_5fold.csv vs qingyu canon_group: {len(leaked_canon_groups)}/{len(grp_to_ph_folds)} multi-fold groups ({(~strict_mask).sum()}/3573 rows); strict zero-sibling rows={strict_mask.sum()} (orig={(strict_mask & is_orig).sum()})")

    # Sibling-label-as-prediction F1 noise ceiling on multi-frame videos
    sib_idx = []
    sib_pred = []
    rng = np.random.default_rng(2026)
    for g, idxs in grp_to_indices.items():
        if len(idxs) >= 2:
            for ii in idxs:
                others = [jj for jj in idxs if jj != ii]
                pick = others[int(rng.integers(0, len(others)))]
                sib_idx.append(ii)
                sib_pred.append(gold[pick])
    sib_idx = np.array(sib_idx)
    sib_pred = np.stack(sib_pred)
    sib_sc = c.score(sib_pred, gold[sib_idx])
    v5_on_sib = c.score(RoutedGateEUDecoder(base_members=base_m, base_prior=list(_base_family_weights(base_m)), p12_members=p12_m, p12_internal_weights=[1.0], p12_total_prior=0.18).fit(P_v5, gold, is_orig=is_orig, groups=groups).eu.predict(Q_v5[sib_idx]), gold[sib_idx])
    print(f"Multi-frame subset (N={len(sib_idx)}): Sibling-label-as-prediction Doc2_Strict={sib_sc['doc2_strict_total']:.2f} "
          f"(Pos={sib_sc['doc2_S_pos']:.2f}, P0={sib_sc['doc2_S_p0']:.2f}, Clean={sib_sc['doc2_Clean2']:.2f}) "
          f"vs Current v5={v5_on_sib['doc2_strict_total']:.2f} (Pos={v5_on_sib['doc2_S_pos']:.2f}, P0={v5_on_sib['doc2_S_p0']:.2f}, Clean={v5_on_sib['doc2_Clean2']:.2f})")

    print(f"\n=== [3/3] Honest Nested RoutedGateEUDecoder Check on Strict Zero-Sibling Rows (N={strict_mask.sum()}, ORIG={(strict_mask & is_orig).sum()}) ===")
    def load_ph_oof(fn):
        z = np.load(os.path.join(REPO, "results/phoenix", fn), allow_pickle=True)
        mp_d = {str(n): q for n, q in zip(z["names"], z["probs_6x4"].astype(np.float64))}
        return c.renorm(np.stack([mp_d[n] for n in names]))

    P_ext = dict(P_v5)
    P_ext["p1_cv5"] = load_ph_oof("pillar1_cv5_oof.npz")
    P_ext["p2_cv5"] = load_ph_oof("pillar2_cv5_oof.npz")

    P_p0 = load_p0_fallback_base(names)
    Q_p0 = pool(P_p0, ["cradio", "pe", "sig", "v16", "p3"])

    # Temporarily register family for _base_family_weights
    import final_engine as FE
    FE.FAMILY["p1_cv5"] = "vlm"
    FE.FAMILY["p2_cv5"] = "vlm"
    FE.FAMILY["iv6b"] = "tier2"
    P_ext["iv6b"] = Q_iv6b

    sm = strict_mask
    P_p0_s = {k: v[sm] for k, v in P_p0.items()}
    Q_p0_s = Q_p0[sm]

    for tag, mlist in [
        ("resident_v5 (8m)", SETS["resident_v5"]),
        ("v5_pruned (6m: -pe_v3 -sig_v3)", pruned_members),
        ("v5_pruned + p1_cv5 (7m)", pruned_members + ["p1_cv5"]),
        ("v5_pruned + p2_cv5 (7m)", pruned_members + ["p2_cv5"]),
        ("v5_pruned + p1_cv5 + p2_cv5 (8m)", pruned_members + ["p1_cv5", "p2_cv5"]),
        ("v5_pruned + p1_cv5 + iv6b_ep2 (8m)", pruned_members + ["p1_cv5", "iv6b"]),
    ]:
        bm = [m for m in mlist if m not in P12_SPEC]
        pm = [m for m in mlist if m in P12_SPEC]
        pcfg = dict(
            base_members=bm,
            base_prior=list(_base_family_weights(bm)),
            p12_members=pm,
            p12_internal_weights=[1.0 / max(1, len(pm))] * len(pm),
            p12_total_prior=0.18,
            l2_reg=0.015,
        )
        P_sub = {m: P_ext[m][sm] for m in mlist}
        rep = evaluate_nested_routed_gate_eu(P_sub, gold[sm], fold[sm], groups[sm], is_orig[sm], pcfg, Q_p0_s)
        print(
            f"  {tag:36s}: Nested Doc2S ALL={rep['oof_nested_doc2_all']:.3f} ORIG={rep['oof_nested_doc2_orig']:.3f} "
            f"(Pos={rep['oof_nested_pos_orig']:.2f}, P0={rep['oof_nested_p0_orig']:.2f}, Cln={rep['oof_nested_clean_orig']:.2f}) | "
            f"RPS_ALL={rep['oof_rps_all']:.5f} RPS_ORIG={rep['oof_rps_orig']:.5f} mAP14={rep['oof_mAP14']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-fold", type=int, default=-1)
    a = ap.parse_args()
    if a.worker_fold >= 0:
        _infer_iv6b_fold_worker(a.worker_fold)
    else:
        run_all()
