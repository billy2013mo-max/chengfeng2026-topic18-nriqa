#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/run_v8_boundary_fresh_infer.py
=======================================
100% Fresh, Zero-Legacy-Anchoring, Batch-Invariant (`N=1 == N=Batch`) Inference Runner
for `resident_v8_boundary` + `deconf_v6_boundary` on BOTH `test_150` (150 imgs) and
`final_644` (644 imgs), sharded across idle GPUs (default: GPU 0 for `test_150`,
GPUs 1..6 for `final_644`; NEVER touches GPU 7!).

Every single image is inferred from scratch using `final_engine.py` with the latest
fullfit weights (`--set resident_v8_boundary --decoder deconf_v6_boundary --half
--share-weights --embed-on-host`) and `R55_AUDIT_TRAIN_READS=1`.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/finetune", "code/r55"):
    _ap = os.path.join(REPO, _p)
    if os.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)

DIMS_ZH = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
GRADES_ZH = ["无问题", "P2", "P1", "P0"]
GRADE_STR_TO_INT = {"无问题": 0, "P2": 1, "P1": 2, "P0": 3}


def _load_json_grid(path: str, names: List[str]) -> np.ndarray:
    rows = json.load(open(path, "r", encoding="utf-8"))
    mp = {str(r["image"]): r["label"] for r in rows}
    g = np.zeros((len(names), 6), dtype=np.int64)
    for i, nm in enumerate(names):
        lbl = mp[str(nm)]
        for d, dn in enumerate(DIMS_ZH):
            g[i, d] = GRADE_STR_TO_INT[str(lbl[dn])]
    return g


def _audit_diff(tag: str, names: List[str], new_pred: np.ndarray, ref_path: str, ref_tag: str) -> None:
    if not os.path.exists(ref_path):
        print(f"  [audit] skip {ref_tag} (not found: {ref_path})", flush=True)
        return
    ref_pred = _load_json_grid(ref_path, names)
    ch = np.where(ref_pred != new_pred)
    n_cln_ref = int((ref_pred == 0).all(1).sum())
    n_cln_new = int((new_pred == 0).all(1).sum())
    same_cln = int(((ref_pred == 0).all(1) == (new_pred == 0).all(1)).sum())
    cnt = Counter((DIMS_ZH[d], f"{GRADES_ZH[ref_pred[i, d]]}->{GRADES_ZH[new_pred[i, d]]}") for i, d in zip(*ch))
    by_dim = Counter(DIMS_ZH[d] for d in ch[1])
    print(
        f"  [{tag} vs {ref_tag}] changed {len(ch[0])}/{new_pred.size} cells across "
        f"{len(set(ch[0].tolist()))}/{len(names)} imgs | Clean: {n_cln_ref} -> {n_cln_new} "
        f"(same_clean_mask={same_cln}/{len(names)})",
        flush=True,
    )
    print(f"    by_dim: {dict(by_dim)}", flush=True)
    print(f"    top transitions: {cnt.most_common(12)}", flush=True)


def _merge_and_verify_shards(
    tag: str,
    template_path: str,
    shard_outs: List[str],
    final_json_path: str,
    final_probs_path: str,
    members: List[str],
    set_name: str,
    decoder_name: str,
) -> Tuple[List[str], np.ndarray]:
    import importlib.util
    import common55 as c
    import final_engine as fe

    _vspec = importlib.util.spec_from_file_location("r55_validate_submission", os.path.join(REPO, "code/r55/validate_submission.py"))
    _vmod = importlib.util.module_from_spec(_vspec)
    _vspec.loader.exec_module(_vmod)
    validate = _vmod.validate

    tmpl_names = c.load_template_names(template_path)
    name_to_idx = {str(n): i for i, n in enumerate(tmpl_names)}
    N = len(tmpl_names)

    P_full: Dict[str, np.ndarray] = {m: np.zeros((N, 6, 4), dtype=np.float64) for m in members}
    spatial_full = np.zeros((N, 20), dtype=np.float32)
    tile_full = np.zeros((N, 6, 12), dtype=np.float32)
    shard_pred_full = np.full((N, 6), -1, dtype=np.int64)
    seen = np.zeros(N, dtype=bool)
    has_spatial, has_tile = False, False

    for sp in shard_outs:
        pz = os.path.splitext(sp)[0] + "_probs.npz"
        assert os.path.exists(sp), f"Missing shard JSON: {sp}"
        assert os.path.exists(pz), f"Missing shard probs: {pz}"
        z = np.load(pz, allow_pickle=True)
        s_names = [str(x) for x in z["names"]]
        s_grid = _load_json_grid(sp, s_names)
        for j, nm in enumerate(s_names):
            gi = name_to_idx[nm]
            assert not seen[gi], f"Duplicate image {nm} across shards!"
            seen[gi] = True
            shard_pred_full[gi] = s_grid[j]
            for m in members:
                P_full[m][gi] = z[f"p_{m}"][j].astype(np.float64)
            if "aux__spatial_20d" in z.files:
                spatial_full[gi] = z["aux__spatial_20d"][j].astype(np.float32)
                has_spatial = True
            if "aux__tile_peaks_6x12" in z.files:
                tile_full[gi] = z["aux__tile_peaks_6x12"][j].astype(np.float32)
                has_tile = True

    assert seen.all(), f"[{tag}] Missing {int((~seen).sum())} images after shard merge!"
    assert has_spatial and has_tile, f"[{tag}] Missing auxiliary spatial/tile features in shard npz!"

    P_dec = dict(P_full)
    P_dec["_spatial_20d"] = spatial_full
    P_dec["_tile_peaks_6x12"] = tile_full

    decode_fn, _ = fe.load_decoder(set_name, decoder_name)
    pred_batch, q_batch = decode_fn(P_dec)
    pred_batch = np.asarray(pred_batch, dtype=np.int64)
    q_batch = np.asarray(q_batch, dtype=np.float64)

    # Strict N=1 == N=Batch check across EVERY SINGLE IMAGE
    max_q_diff = 0.0
    n1_diffs = 0
    for i in range(N):
        p1, q1 = decode_fn({k: v[i : i + 1] for k, v in P_dec.items()})
        n1_diffs += int((np.asarray(p1)[0] != pred_batch[i]).sum())
        max_q_diff = max(max_q_diff, float(np.max(np.abs(np.asarray(q1)[0] - q_batch[i]))))

    shard_diffs = int((shard_pred_full != pred_batch).sum())
    print(
        f"[{tag}] Merged {len(shard_outs)} shards ({N} imgs): "
        f"N=1 vs N={N} diffs={n1_diffs}/{pred_batch.size} (max_q_diff={max_q_diff:.2e}) | "
        f"shard_json vs merged_batch diffs={shard_diffs}/{pred_batch.size}",
        flush=True,
    )
    assert n1_diffs == 0 and max_q_diff < 1e-11, f"[{tag}] N=1 invariance FAILED!"
    assert shard_diffs == 0, f"[{tag}] Shard vs merged decode mismatch!"

    os.makedirs(os.path.dirname(os.path.abspath(final_json_path)), exist_ok=True)
    sub_obj = c.format_official_submission(tmpl_names, pred_batch)[0]
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(sub_obj, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(final_probs_path)), exist_ok=True)
    pp = fe.pool(P_full, members)
    np.savez_compressed(
        final_probs_path,
        names=np.array(tmpl_names),
        pooled=pp.astype(np.float32),
        routed=q_batch.astype(np.float32),
        **{f"p_{m}": P_full[m].astype(np.float32) for m in members},
        aux__spatial_20d=spatial_full,
        aux__tile_peaks_6x12=tile_full,
    )

    validate(final_json_path, template_path)
    return tmpl_names, pred_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="resident_v8_boundary")
    ap.add_argument("--decoder", default="deconf_v6_boundary")
    ap.add_argument("--gpu-test150", type=int, default=0)
    ap.add_argument("--gpus-final644", default="1,2,3,4,5,6")
    ap.add_argument("--work-dir", default="results/r55/v8_boundary_fresh_shards")
    ap.add_argument("--merge-only", action="store_true", help="Skip GPU inference and merge/verify existing shard outputs")
    args = ap.parse_args()

    gpus_644 = [int(x.strip()) for x in args.gpus_final644.split(",") if x.strip()]
    all_gpus = [int(args.gpu_test150)] + gpus_644
    assert 7 not in all_gpus, "GPU 7 is strictly forbidden!"
    assert len(set(all_gpus)) == len(all_gpus), f"Duplicate GPUs: {all_gpus}"

    import final_engine as fe

    members = fe.SETS[args.set]
    work_dir = os.path.join(REPO, args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    tmpl_150 = os.path.join(REPO, "results/v2_full/template_150.json")
    dir_150 = os.path.join(REPO, "data/test_150")
    tmpl_644 = os.path.join(REPO, "results/final/template_final_644.json")
    dir_644 = "/opt/final_eval_644"

    # Prepare 1 shard for test_150 (full unsharded single-GPU run) + K shards for final_644
    rows_644 = json.load(open(tmpl_644, "r", encoding="utf-8"))
    K = len(gpus_644)
    shard_jobs = []

    out_150_shard = os.path.join(work_dir, "test150_gpu0.json")
    log_150_path = os.path.join(work_dir, "test150_gpu0.log")
    shard_jobs.append(("test_150", args.gpu_test150, tmpl_150, dir_150, out_150_shard, log_150_path))

    outs_644_shards = []
    chunk_sz = (len(rows_644) + K - 1) // K
    for si, gpu in enumerate(gpus_644):
        sub = rows_644[si * chunk_sz : (si + 1) * chunk_sz]
        st_path = os.path.join(work_dir, f"final644_shard{si}_tmpl.json")
        with open(st_path, "w", encoding="utf-8") as f:
            json.dump(sub, f, ensure_ascii=False, indent=2)
        so_path = os.path.join(work_dir, f"final644_shard{si}_gpu{gpu}.json")
        sl_path = os.path.join(work_dir, f"final644_shard{si}_gpu{gpu}.log")
        outs_644_shards.append(so_path)
        shard_jobs.append((f"final644_s{si}", gpu, st_path, dir_644, so_path, sl_path))

    if not args.merge_only:
        print(
            f"=== Launching 100% Fresh Inference ({args.set} + {args.decoder}) across {len(shard_jobs)} GPUs {all_gpus} ===",
            flush=True,
        )
        t0 = time.time()
        procs = []
        log_fhs = []
        for jtag, gpu, tp, idir, op, lp in shard_jobs:
            env = os.environ.copy()
            env["R55_AUDIT_TRAIN_READS"] = "1"
            cmd = [
                sys.executable,
                os.path.join(REPO, "code/r55/final_engine.py"),
                "--set",
                args.set,
                "--gpu",
                str(gpu),
                "--half",
                "--share-weights",
                "--embed-on-host",
                "--decoder",
                args.decoder,
                "--template",
                tp,
                "--img-dir",
                idir,
                "--out",
                op,
            ]
            fh = open(lp, "w", encoding="utf-8")
            log_fhs.append(fh)
            p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
            procs.append((jtag, gpu, p, lp))
            print(f"  [spawn] {jtag:12s} | GPU {gpu} | pid={p.pid} | log={lp}", flush=True)

        failed = []
        for jtag, gpu, p, lp in procs:
            rc = p.wait()
            if rc != 0:
                failed.append((jtag, gpu, rc, lp))
            else:
                tail = open(lp, "r", encoding="utf-8").read().splitlines()[-6:]
                audit_lines = [ln for ln in tail if "audit" in ln or "N=1 vs N=" in ln or "per-image" in ln]
                print(f"  [done]  {jtag:12s} | GPU {gpu} | rc=0 | " + " ; ".join(audit_lines), flush=True)
        for fh in log_fhs:
            fh.close()

        if failed:
            for jtag, gpu, rc, lp in failed:
                print(f"[ERROR] {jtag} on GPU {gpu} failed (rc={rc}); tail of {lp}:", file=sys.stderr)
                print("\n".join(open(lp, "r", encoding="utf-8").read().splitlines()[-30:]), file=sys.stderr)
            sys.exit(1)

        print(f"\n=== All {len(shard_jobs)} fresh inference workers completed in {(time.time() - t0) / 60.0:.2f} min ===", flush=True)

    # Verify zero training-data opens across all logs
    for jtag, gpu, _, _, _, lp in shard_jobs:
        txt = open(lp, "r", encoding="utf-8").read()
        assert "training-data file opens during this inference run: 0" in txt, (
            f"Audit check failed in {lp}!"
        )
    print("[AUDIT PASS] 0 training-data file opens confirmed across all workers!", flush=True)

    out_150 = os.path.join(REPO, f"results/final/pred_nriqa_dataset_test_r55_{args.set}_{args.decoder}.json")
    out_644 = os.path.join(REPO, f"results/final/pred_nriqa_dataset_final_644_r55_{args.set}_{args.decoder}.json")
    prb_150 = os.path.join(REPO, f"results/r55/final/test_150_{args.set}_probs.npz")
    prb_644 = os.path.join(REPO, f"results/r55/final/final_644_{args.set}_probs.npz")

    nm_150, pred_150 = _merge_and_verify_shards(
        "test_150", tmpl_150, [out_150_shard], out_150, prb_150, members, args.set, args.decoder
    )
    nm_644, pred_644 = _merge_and_verify_shards(
        "final_644", tmpl_644, outs_644_shards, out_644, prb_644, members, args.set, args.decoder
    )

    # Copy finalized JSONs & decoder artifacts into final_sprint_delivery/
    sprint_01 = os.path.join(REPO, "final_sprint_delivery/01_核心预测结果JSON")
    sprint_04 = os.path.join(REPO, "final_sprint_delivery/04_最终校准解码器与实测审计账本")
    if os.path.isdir(sprint_01):
        shutil.copy2(out_150, os.path.join(sprint_01, os.path.basename(out_150)))
        shutil.copy2(out_644, os.path.join(sprint_01, os.path.basename(out_644)))
    if os.path.isdir(sprint_04):
        for ext in (".pkl", ".json"):
            src = os.path.join(REPO, f"results/r55/final/{args.set}_decoder_{args.decoder}{ext}")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(sprint_04, os.path.basename(src)))

    # Strictly post-hoc comparison audit vs online v4_tile (54.68) and Phase 1 v5_boundary
    print("\n=== Post-Hoc Audit Diff (Pure Diagnostic After Independent Inference) ===", flush=True)
    _audit_diff(
        "test_150",
        nm_150,
        pred_150,
        os.path.join(REPO, "results/final/pred_nriqa_dataset_test_r55_resident_v7_deconf_v4_tile.json"),
        "deconf_v4_tile (54.68)",
    )
    _audit_diff(
        "test_150",
        nm_150,
        pred_150,
        os.path.join(REPO, "results/final/pred_nriqa_dataset_test_r55_resident_v7_deconf_v5_boundary.json"),
        "deconf_v5_boundary (Phase1)",
    )
    _audit_diff(
        "final_644",
        nm_644,
        pred_644,
        os.path.join(REPO, "results/final/pred_nriqa_dataset_final_644_r55_resident_v7_deconf_v4_tile.json"),
        "deconf_v4_tile (54.68)",
    )
    _audit_diff(
        "final_644",
        nm_644,
        pred_644,
        os.path.join(REPO, "results/final/pred_nriqa_dataset_final_644_r55_resident_v7_deconf_v5_boundary.json"),
        "deconf_v5_boundary (Phase1)",
    )
    print("\n=== ALL DONE! ===", flush=True)


if __name__ == "__main__":
    main()
