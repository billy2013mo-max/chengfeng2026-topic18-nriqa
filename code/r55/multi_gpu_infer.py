#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/multi_gpu_infer.py
===========================
Multi-GPU Data-Parallel Sharded Inference & Joint Method 1+2 Decoder Runner for `test_150` (or `final_644`).

System & Competition Hardening:
  1. Free-VRAM & Host-RAM Aware Sharding:
     - Filters out GPUs with `< 26 GiB` free VRAM automatically so shared-cluster jobs on other cards
       never trigger `CUDA out of memory`.
     - Checks `/proc/meminfo` (`MemAvailable`) to bound concurrent `Qwen3-VL-8B` bf16->4bit loads if
       container host RAM is constrained, and staggers worker launches by `4.5s` to warm Linux page cache
       and prevent CephFS / HuggingFace file-lock contention.
  2. Live Heartbeat & Persistent Logs:
     - Saves per-shard worker logs under `<out_dir>/logs/` and prints live progress
       every `15s` so `tail -f` never appears frozen.
  3. Cross-Module Safe Unpickling + `N = 1 == N = Batch` Verification (`0` cell diff) + Schema Validation.
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


def _host_avail_ram_gb() -> float:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except Exception:
        pass
    return 512.0


def parse_healthy_gpus(gpus_str: str, min_free_mib: int = 26000) -> List[int]:
    cand: List[int] = []
    if gpus_str.strip():
        cand = [int(x.strip()) for x in gpus_str.split(",") if x.strip() != ""]
    else:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if vis:
            cand = [int(x.strip()) for x in vis.split(",") if x.strip().isdigit()]

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        free_map: Dict[int, int] = {}
        for ln in out.splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                free_map[int(parts[0])] = int(float(parts[1]))
        if not cand:
            cand = sorted(free_map.keys())
        healthy = [g for g in cand if free_map.get(g, min_free_mib) >= min_free_mib]
        if healthy:
            if len(healthy) < len(cand):
                skipped = {g: free_map.get(g, -1) for g in cand if g not in healthy}
                print(f"[multi_gpu_infer][WARN] skipping busy GPUs (<{min_free_mib} MiB free): {skipped}", flush=True)
            return healthy
        if cand and free_map:
            best_g = max(cand, key=lambda g: free_map.get(g, 0))
            print(f"[multi_gpu_infer][WARN] all GPUs below {min_free_mib} MiB; using freest GPU {best_g} ({free_map.get(best_g)} MiB)", flush=True)
            return [best_g]
    except Exception:
        pass
    return cand if cand else [0]


def run_sharded_final_engine(
    set_name: str,
    gpus: List[int],
    template_path: str,
    img_dir: str,
    merged_probs_npz: str,
    log_dir: str,
    half: bool = True,
    stagger_sec: float = 4.5,
) -> Dict[str, np.ndarray]:
    import common55 as c
    from final_engine import SETS

    members = SETS[set_name]
    names = c.load_template_names(template_path)
    num_shards = min(len(gpus), len(names))
    active_gpus = gpus[:num_shards]

    avail_ram = _host_avail_ram_gb()
    # Each FinalEngine worker peaks ~20 GB host RAM transiently while quantizing Qwen3-VL-8B; cap concurrency safely
    max_concurrent = max(1, min(num_shards, int(avail_ram // 22.0))) if avail_ram < 220.0 else num_shards

    os.makedirs(log_dir, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix=f"r55_mgpu_{set_name}_")
    shard_specs = []
    t0 = time.time()

    print(
        f"=== [multi_gpu_infer] {len(names)} images across {num_shards} shards on GPUs {active_gpus} "
        f"(max_concurrent={max_concurrent}, host_avail_ram={avail_ram:.0f} GB, set={set_name}) ===",
        flush=True,
    )
    try:
        for s_idx in range(num_shards):
            shard_names = [names[i] for i in range(len(names)) if (i % num_shards) == s_idx]
            shard_tmpl = os.path.join(work_dir, f"tmpl_shard_{s_idx:02d}.json")
            shard_out = os.path.join(work_dir, f"pred_shard_{s_idx:02d}.json")
            shard_log = os.path.join(log_dir, f"final_engine_s{s_idx:02d}_gpu{active_gpus[s_idx]}.log")
            with open(shard_tmpl, "w", encoding="utf-8") as f:
                json.dump([{"image": nm, "label": {}} for nm in shard_names], f, ensure_ascii=False)
            shard_specs.append((s_idx, active_gpus[s_idx], shard_names, shard_tmpl, shard_out, shard_log, os.path.splitext(shard_out)[0] + "_probs.npz"))

        # Run in waves of size `max_concurrent` (1 wave of 8 on standard H200 nodes)
        for w0 in range(0, num_shards, max_concurrent):
            wave = shard_specs[w0 : w0 + max_concurrent]
            procs = []
            for idx_in_w, (s_idx, gpu_id, shard_names, shard_tmpl, shard_out, shard_log, _) in enumerate(wave):
                if idx_in_w > 0 and stagger_sec > 0:
                    time.sleep(stagger_sec)
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
                env.pop("CUDA_VISIBLE_DEVICES", None)  # final_engine.py binds --gpu directly before importing torch
                if img_dir:
                    env["NEXT55_EXTRA_IMG_DIR"] = img_dir
                log_fh = open(shard_log, "w", encoding="utf-8")
                p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
                procs.append((s_idx, gpu_id, len(shard_names), p, log_fh, shard_log))
                print(f"  - Launched shard {s_idx}/{num_shards - 1} ({len(shard_names)} imgs) on GPU {gpu_id} -> {os.path.relpath(shard_log, REPO)}", flush=True)

            last_hb = time.time()
            while True:
                alive = [it for it in procs if it[3].poll() is None]
                if not alive:
                    break
                if time.time() - last_hb >= 15.0:
                    last_line = ""
                    try:
                        with open(procs[0][5], "r", encoding="utf-8", errors="replace") as lf:
                            lines = [ln.strip() for ln in lf.readlines() if ln.strip()]
                            if lines:
                                last_line = lines[-1][:95]
                    except Exception:
                        pass
                    print(
                        f"  [heartbeat +{time.time() - t0:5.0f}s] {len(alive)}/{len(procs)} GPU workers active | shard0: {last_line}",
                        flush=True,
                    )
                    last_hb = time.time()
                time.sleep(3.0)

            failed = []
            for s_idx, gpu_id, _, p, log_fh, shard_log in procs:
                log_fh.close()
                if p.returncode != 0:
                    tail = ""
                    if os.path.exists(shard_log):
                        with open(shard_log, "r", encoding="utf-8", errors="replace") as lf:
                            tail = "".join(lf.readlines()[-45:])
                    failed.append((s_idx, gpu_id, p.returncode, shard_log, tail))

            if failed:
                msg_lines = [f"[multi_gpu_infer] {len(failed)}/{len(procs)} shard workers failed:"]
                for s_idx, gpu_id, rc, slog, tail in failed:
                    msg_lines.append(f"--- Shard {s_idx} (GPU {gpu_id}, log={slog}) exit={rc} ---\n{tail}")
                raise RuntimeError("\n".join(msg_lines))

        idx_map = {str(n): i for i, n in enumerate(names)}
        P_merged = {m: np.zeros((len(names), 6, 4), dtype=np.float64) for m in members}
        seen = np.zeros(len(names), dtype=bool)

        for _, _, _, _, _, _, npz_p in shard_specs:
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
    ap.add_argument("--gpus", default="", help="Comma-separated GPU IDs, e.g. 0,1,2,3,4,5,6,7 (default: auto-detect healthy GPUs)")
    ap.add_argument("--template", default="results/v2_full/template_150.json")
    ap.add_argument("--img-dir", default="data/test_150")
    ap.add_argument("--bundle", default="", help="Path to <set>_os_og_bundle.pkl from deconfound_calibrator.py")
    ap.add_argument("--spatial-target", default="", help="Optional target spatial features npz (required only if bundle.mode in {spatial, joint})")
    ap.add_argument("--reuse-probs-npz", default="", help="Optional existing fresh member probs npz (default: run fresh multi-GPU inference)")
    ap.add_argument("--out", default="results/r55/os_og_breakthrough/pred_nriqa_dataset_test_os_og_breakthrough.json")
    ap.add_argument("--no-half", action="store_true")
    args = ap.parse_args()

    import common55 as c
    import deconfound_calibrator
    import routed_eu_decoder
    from final_engine import SETS
    from validate_submission import validate

    tmpl_abs = args.template if os.path.isabs(args.template) else os.path.join(REPO, args.template)
    img_dir_abs = args.img_dir if os.path.isabs(args.img_dir) else os.path.join(REPO, args.img_dir)
    out_abs = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)
    out_dir_abs = os.path.dirname(out_abs)
    os.makedirs(out_dir_abs, exist_ok=True)

    bundle_abs = (
        args.bundle
        if args.bundle
        else os.path.join(REPO, "results/r55/os_og_breakthrough", f"{args.set}_os_og_bundle.pkl")
    )
    if not os.path.isabs(bundle_abs):
        bundle_abs = os.path.join(REPO, bundle_abs)
    assert os.path.exists(bundle_abs), f"Missing bundle {bundle_abs}; run deconfound_calibrator.py first."

    class _CompatUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module == "__main__":
                for mod in (deconfound_calibrator, routed_eu_decoder):
                    if hasattr(mod, name):
                        return getattr(mod, name)
            return super().find_class(module, name)

    with open(bundle_abs, "rb") as _bf:
        payload = _CompatUnpickler(_bf).load()
    bundle = payload["bundle"]
    rep_meta = payload.get("report", {})

    names = c.load_template_names(tmpl_abs)
    members = SETS[args.set]
    merged_npz = os.path.splitext(out_abs)[0] + "_member_probs.npz"

    reuse_p = ""
    if args.reuse_probs_npz:
        reuse_p = args.reuse_probs_npz if os.path.isabs(args.reuse_probs_npz) else os.path.join(REPO, args.reuse_probs_npz)

    if reuse_p and os.path.isfile(reuse_p):
        print(f"[multi_gpu_infer] loading member probabilities from {reuse_p}", flush=True)
        z = np.load(reuse_p, allow_pickle=True)
        mp_idx = {str(nm): i for i, nm in enumerate(z["names"])}
        P_dict = {m: np.stack([z[f"p_{m}"][mp_idx[str(n)]] for n in names], axis=0) for m in members}
    else:
        gpus = parse_healthy_gpus(args.gpus)
        P_dict = run_sharded_final_engine(
            set_name=args.set,
            gpus=gpus,
            template_path=tmpl_abs,
            img_dir=img_dir_abs,
            merged_probs_npz=merged_npz,
            log_dir=os.path.join(out_dir_abs, "logs"),
            half=not args.no_half,
        )

    sp_target = None
    sp_arg = args.spatial_target.strip()
    sp_abs = (sp_arg if os.path.isabs(sp_arg) else os.path.join(REPO, sp_arg)) if sp_arg else ""
    if bundle.mode in ("spatial", "joint"):
        assert sp_abs and os.path.isfile(sp_abs), (
            f"Bundle mode={bundle.mode!r} requires --spatial-target for {tmpl_abs}, got {sp_abs!r}"
        )
    if sp_abs and os.path.isfile(sp_abs):
        z_sp = np.load(sp_abs, allow_pickle=True)
        mp_sp = {str(nm): fv for nm, fv in zip(z_sp["names"], z_sp["feats"])}
        if all(str(n) in mp_sp for n in names):
            sp_target = np.stack([mp_sp[str(n)] for n in names], axis=0).astype(np.float64)
        elif bundle.mode in ("spatial", "joint"):
            missing_sp = [str(n) for n in names if str(n) not in mp_sp]
            raise KeyError(f"spatial_target {sp_abs} missing {len(missing_sp)}/{len(names)} template IDs (e.g. {missing_sp[:3]})")
        else:
            print(f"[multi_gpu_infer] note: bundle.mode={bundle.mode!r} does not use spatial_target; ignoring mismatched cache {sp_abs}", flush=True)

    batch_res = bundle.predict_all(P_dict, spatial_feats=sp_target)

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
        (out_bal, batch_res["pred_balanced"], "Balanced (Method 1+2 + OOF-Gated Clean-Shielded Ordinal)"),
        (out_base, batch_res["pred_base"], "Fresh Baseline (RoutedGateEUDecoder)"),
    ):
        sub_json, _ = c.format_official_submission(names, pred_arr)
        with open(path_j, "w", encoding="utf-8") as f:
            json.dump(sub_json, f, ensure_ascii=False, indent=2)
        print(f"\n--- Validating {tag}: {path_j} ---", flush=True)
        validate(path_j, tmpl_abs)

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

    rec_file = out_bal if rep_meta.get("balanced_dominates_doc2", False) else out_abs
    print(f"[multi_gpu_infer] OOF-recommended top submission file: {rec_file}", flush=True)

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
