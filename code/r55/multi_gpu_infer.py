#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/multi_gpu_infer.py
===========================
Multi-GPU Data-Parallel Sharded Inference & Joint Method 1+2 Decoder Runner for `test_150` (or `final_644`).

Key guarantees (`AGENTS.md` compliant):
  1. 100% Fresh Weights Inference:
     - Splits `--template` evenly across `--gpus` into temporary shard templates, runs `code/r55/final_engine.py`
       at `N = 1` per image on each GPU concurrently, and merges member probability tensors (`p_<member>`)
       in the exact template order.
  2. Bit-Identical Single-GPU Equivalence & `N = 1 == N = 150` Verification:
     - Explicitly runs BOTH batch decoding (`N = len(template)`) AND single-image `N = 1` loop decoding,
       asserting `0 / (6 * N)` cell differences across BOTH:
         * Primary total-score optimal output (`--out`, `pred_primary_eu`), AND
         * Companion P1/P2 floor-boosted output (`<out_stem>_balanced.json`, `pred_balanced`).
  3. Built-in Official Submission Validation:
     - Invokes `validate_submission.validate()` on every emitted JSON against `--template`.
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)


def parse_gpu_list(gpus_str: str) -> List[int]:
    if gpus_str.strip():
        return [int(x.strip()) for x in gpus_str.split(",") if x.strip() != ""]
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        return [int(x.strip()) for x in vis.split(",") if x.strip().isdigit()]
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
        ids = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
        if ids:
            return ids
    except Exception:
        pass
    return [0]


def run_sharded_final_engine(
    set_name: str,
    gpus: List[int],
    template_path: str,
    img_dir: str,
    merged_probs_npz: str,
    half: bool = True,
) -> Dict[str, np.ndarray]:
    import common55 as c
    from final_engine import SETS

    members = SETS[set_name]
    names = c.load_template_names(template_path)
    num_shards = min(len(gpus), len(names))
    active_gpus = gpus[:num_shards]

    work_dir = tempfile.mkdtemp(prefix=f"r55_mgpu_{set_name}_", dir="/tmp")
    procs = []
    shard_specs = []
    t0 = time.time()

    print(
        f"=== [multi_gpu_infer] launching {num_shards} parallel FinalEngine workers on GPUs {active_gpus} "
        f"for {len(names)} images (set={set_name}) ===",
        flush=True,
    )
    try:
        for s_idx, gpu_id in enumerate(active_gpus):
            shard_names = [names[i] for i in range(len(names)) if (i % num_shards) == s_idx]
            shard_tmpl = os.path.join(work_dir, f"tmpl_shard_{s_idx:02d}.json")
            shard_out = os.path.join(work_dir, f"pred_shard_{s_idx:02d}.json")
            shard_log = os.path.join(work_dir, f"worker_gpu{gpu_id}_s{s_idx:02d}.log")
            with open(shard_tmpl, "w", encoding="utf-8") as f:
                json.dump([{"image": nm, "label": {}} for nm in shard_names], f, ensure_ascii=False)

            cmd = [
                sys.executable,
                "-u",
                os.path.join(REPO, "code/r55/final_engine.py"),
                "--set",
                set_name,
                "--gpu",
                str(gpu_id),
                "--template",
                shard_tmpl,
                "--img-dir",
                img_dir,
                "--out",
                shard_out,
                "--decoder",
                "routed_gate_eu",
            ]
            if half:
                cmd.append("--half")

            env = dict(os.environ)
            env.pop("CUDA_VISIBLE_DEVICES", None)  # let final_engine.py bind --gpu directly
            log_fh = open(shard_log, "w", encoding="utf-8")
            p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
            procs.append((s_idx, gpu_id, p, log_fh, shard_log))
            shard_specs.append((s_idx, shard_names, os.path.splitext(shard_out)[0] + "_probs.npz"))

        # Poll workers and report progress cleanly
        failed = []
        while True:
            alive = [it for it in procs if it[2].poll() is None]
            if not alive:
                break
            time.sleep(3.0)

        for s_idx, gpu_id, p, log_fh, shard_log in procs:
            log_fh.close()
            if p.returncode != 0:
                tail = ""
                if os.path.exists(shard_log):
                    with open(shard_log, "r", encoding="utf-8", errors="replace") as lf:
                        tail = "".join(lf.readlines()[-40:])
                failed.append((s_idx, gpu_id, p.returncode, tail))

        if failed:
            msg_lines = [f"[multi_gpu_infer] {len(failed)}/{num_shards} shard workers failed:"]
            for s_idx, gpu_id, rc, tail in failed:
                msg_lines.append(f"--- Shard {s_idx} (GPU {gpu_id}) exit={rc} ---\n{tail}")
            raise RuntimeError("\n".join(msg_lines))

        idx_map = {str(n): i for i, n in enumerate(names)}
        P_merged = {m: np.zeros((len(names), 6, 4), dtype=np.float64) for m in members}
        seen = np.zeros(len(names), dtype=bool)

        for _, _, npz_p in shard_specs:
            assert os.path.exists(npz_p), f"Missing shard probs output: {npz_p}"
            z = np.load(npz_p, allow_pickle=True)
            s_names = [str(x) for x in z["names"]]
            for loc_i, nm in enumerate(s_names):
                gi = idx_map[nm]
                seen[gi] = True
                for m in members:
                    P_merged[m][gi] = z[f"p_{m}"][loc_i]

        assert bool(seen.all()), f"Incomplete shard merge: {seen.sum()}/{len(names)}"
        os.makedirs(os.path.dirname(os.path.abspath(merged_probs_npz)), exist_ok=True)
        np.savez_compressed(
            merged_probs_npz,
            names=np.array(names),
            **{f"p_{m}": v for m, v in P_merged.items()},
        )
        el = time.time() - t0
        print(
            f"[multi_gpu_infer] merged {len(names)} images across {num_shards} GPUs in {el:.1f}s "
            f"(effective {el / len(names):.2f}s/img wall-clock) -> {merged_probs_npz}",
            flush=True,
        )
        return P_merged
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="resident_v7")
    ap.add_argument("--gpus", default="", help="Comma-separated GPU IDs, e.g. 0,1,2,3,4,5,6,7 (default: auto-detect all)")
    ap.add_argument("--template", default="results/v2_full/template_150.json")
    ap.add_argument("--img-dir", default="data/test_150")
    ap.add_argument("--bundle", default="", help="Path to <set>_os_og_bundle.pkl from deconfound_calibrator.py")
    ap.add_argument("--spatial-target", default="results/r55/os_og_breakthrough/spatial_cache/spatial_target.npz")
    ap.add_argument("--reuse-probs-npz", default="", help="Optional existing fresh member probs npz (default: run fresh multi-GPU inference)")
    ap.add_argument("--out", default="results/r55/os_og_breakthrough/pred_nriqa_dataset_test_os_og_breakthrough.json")
    ap.add_argument("--no-half", action="store_true")
    args = ap.parse_args()

    import common55 as c
    import deconfound_calibrator  # noqa: F401  (needed for pickle class resolution)
    from final_engine import SETS
    from validate_submission import validate

    tmpl_abs = args.template if os.path.isabs(args.template) else os.path.join(REPO, args.template)
    img_dir_abs = args.img_dir if os.path.isabs(args.img_dir) else os.path.join(REPO, args.img_dir)
    out_abs = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)

    bundle_abs = (
        args.bundle
        if args.bundle
        else os.path.join(REPO, "results/r55/os_og_breakthrough", f"{args.set}_os_og_bundle.pkl")
    )
    assert os.path.exists(bundle_abs), f"Missing bundle {bundle_abs}; run deconfound_calibrator.py first."
    payload = pickle.load(open(bundle_abs, "rb"))
    bundle = payload["bundle"]

    names = c.load_template_names(tmpl_abs)
    members = SETS[args.set]
    merged_npz = os.path.splitext(out_abs)[0] + "_member_probs.npz"

    if args.reuse_probs_npz and os.path.exists(args.reuse_probs_npz):
        print(f"[multi_gpu_infer] loading member probabilities from {args.reuse_probs_npz}", flush=True)
        z = np.load(args.reuse_probs_npz, allow_pickle=True)
        mp_idx = {str(nm): i for i, nm in enumerate(z["names"])}
        P_dict = {m: np.stack([z[f"p_{m}"][mp_idx[str(n)]] for n in names], axis=0) for m in members}
    else:
        gpus = parse_gpu_list(args.gpus)
        P_dict = run_sharded_final_engine(
            set_name=args.set,
            gpus=gpus,
            template_path=tmpl_abs,
            img_dir=img_dir_abs,
            merged_probs_npz=merged_npz,
            half=not args.no_half,
        )

    sp_target = None
    sp_abs = args.spatial_target if os.path.isabs(args.spatial_target) else os.path.join(REPO, args.spatial_target)
    if bundle.mode in ("spatial", "joint"):
        assert os.path.exists(sp_abs), f"Bundle requires spatial features ({bundle.mode}) but {sp_abs} is missing!"
    if os.path.exists(sp_abs):
        z_sp = np.load(sp_abs, allow_pickle=True)
        mp_sp = {str(nm): fv for nm, fv in zip(z_sp["names"], z_sp["feats"])}
        sp_target = np.stack([mp_sp[str(n)] for n in names], axis=0).astype(np.float64)

    # Batch decode (N = len(names))
    batch_res = bundle.predict_all(P_dict, spatial_feats=sp_target)

    # Strict N = 1 loop decode verification
    n1_eu, n1_bal = [], []
    for i in range(len(names)):
        p_i = {m: v[i : i + 1] for m, v in P_dict.items()}
        sp_i = sp_target[i : i + 1] if sp_target is not None else None
        r_i = bundle.predict_all(p_i, spatial_feats=sp_i)
        n1_eu.append(r_i["pred_primary_eu"][0])
        n1_bal.append(r_i["pred_balanced"][0])

    n1_eu_arr = np.stack(n1_eu, axis=0)
    n1_bal_arr = np.stack(n1_bal, axis=0)
    diff_eu = int(np.sum(n1_eu_arr != batch_res["pred_primary_eu"]))
    diff_bal = int(np.sum(n1_bal_arr != batch_res["pred_balanced"]))
    assert diff_eu == 0, f"CRITICAL: N=1 vs batch mismatch on primary EU ({diff_eu} cells)"
    assert diff_bal == 0, f"CRITICAL: N=1 vs batch mismatch on balanced ({diff_bal} cells)"
    print(
        f"[multi_gpu_infer] N=1 vs N={len(names)} batch-invariance check: "
        f"primary_eu={diff_eu}/{n1_eu_arr.size} diff, balanced={diff_bal}/{n1_bal_arr.size} diff [PASS]",
        flush=True,
    )

    stem, _ = os.path.splitext(out_abs)
    out_bal = stem + "_balanced.json"
    out_base = stem + "_v7_baseline_fresh.json"

    for path_j, pred_arr, tag in (
        (out_abs, batch_res["pred_primary_eu"], "Primary (Method 1+2 Calibrated + GateEU)"),
        (out_bal, batch_res["pred_balanced"], "Balanced (Method 1+2 + Clean-Locked P1/P2 Ordinal)"),
        (out_base, batch_res["pred_base"], "Fresh Baseline (RoutedGateEUDecoder)"),
    ):
        sub_json, _ = c.format_official_submission(names, pred_arr)
        with open(path_j, "w", encoding="utf-8") as f:
            json.dump(sub_json, f, ensure_ascii=False, indent=2)
        print(f"\n--- Validating {tag}: {path_j} ---", flush=True)
        validate(path_j, tmpl_abs)

    # Print concise comparison of changed cells vs fresh baseline on dim=2 (过锐) and dim=4 (过度生成)
    pb = batch_res["pred_base"]
    for label_t, pr in (("Primary_EU", batch_res["pred_primary_eu"]), ("Balanced_P1P2", batch_res["pred_balanced"])):
        d_os = int(np.sum(pr[:, 2] != pb[:, 2]))
        d_og = int(np.sum(pr[:, 4] != pb[:, 4]))
        d_tot = int(np.sum(pr != pb))
        print(
            f"[multi_gpu_infer] diff vs fresh baseline ({label_t}): total={d_tot}/{pb.size} cells "
            f"(过锐 changed={d_os}/{len(names)}, 过度生成 changed={d_og}/{len(names)})",
            flush=True,
        )

    np.savez_compressed(
        stem + "_calibrated_probs.npz",
        names=np.array(names),
        Q_base=batch_res["Q_base"],
        Q_cal=batch_res["Q_cal"],
        pred_base=batch_res["pred_base"],
        pred_primary_eu=batch_res["pred_primary_eu"],
        pred_balanced=batch_res["pred_balanced"],
    )


if __name__ == "__main__":
    main()
