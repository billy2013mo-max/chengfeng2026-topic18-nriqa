#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/run_v5_8gpu_waves.py
=============================
Live-GPU-aware, VRAM-slot-aware 8 x NVIDIA H200 scheduler executing in exact
Round-2 `claude-fable-5-2` consensus priority order:

  Slot model on 140 GiB H200 (2 slots/GPU capacity):
    - `ntg` (`NativeTileTower`, bs=6, <=102 views): `slots=2` (exclusive GPU, 100% SM util, ~62 GiB with expandable_segments)
    - `p12` (`P12SpecialistTower`, bs=12, 7 views): `slots=1` (~57.7 GiB -> 2 jobs/GPU = 115.4 GiB < 139.8 GiB)
    - `infer` (`final_engine.py --half`):           `slots=1` (~18.5 GiB)

  Safe live backfill (`--backfill`):
    - Inspects `/proc/*/cmdline` + `/proc/*/environ` + `nvidia-smi` every tick so it NEVER duplicates
      or over-allocates any GPU already occupied by an earlier scheduler wave.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Set, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(REPO, "logs/r55/v5_waves")


def check_disk_gb() -> float:
    return shutil.disk_usage(REPO).free / (1024.0 ** 3)


def query_gpu_mem_mib() -> Dict[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    res: Dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            res[int(parts[0])] = int(parts[1])
    return res


def scan_live_external_processes(own_pids: Set[int]) -> Tuple[List[str], Dict[int, int], bool]:
    """
    Returns:
      - `ext_cmds`: command lines of live external python training/inference processes
      - `ext_slots`: {gpu_idx: occupied_slots} by external processes
      - `orig_sched_alive`: whether an earlier non-backfill `run_v5_8gpu_waves.py` is still alive
    """
    my_pid = os.getpid()
    ext_cmds: List[str] = []
    ext_slots: Dict[int, int] = {g: 0 for g in range(8)}
    orig_sched_alive = False

    # Map pid -> ppid so forked DataLoader worker children are never double-counted!
    ppid_map: Dict[int, int] = {}
    for spath in glob.glob("/proc/[0-9]*/stat"):
        try:
            txt = open(spath, "r", encoding="utf-8", errors="ignore").read()
            rparen = txt.rfind(")")
            fields = txt[rparen + 2:].split()
            ppid_map[int(spath.split("/")[2])] = int(fields[1])
        except Exception:
            continue

    cand_procs: Dict[int, Tuple[List[str], str]] = {}
    for cpath in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            pid = int(cpath.split("/")[2])
            if pid == my_pid or pid in own_pids:
                continue
            raw = open(cpath, "rb").read()
            if not raw:
                continue
            args = [a.decode("utf-8", errors="ignore") for a in raw.split(b"\x00") if a]
            if not args or not os.path.basename(args[0]).startswith("python"):
                continue
            cmd = " ".join(args)
            if "run_v5_8gpu_waves.py" in cmd and "--backfill" not in cmd and "--dry-run" not in cmd:
                orig_sched_alive = True
                continue
            if ("train_testbed.py" in cmd) or ("train_p12_specialist.py" in cmd) or ("final_engine.py" in cmd):
                cand_procs[pid] = (args, cmd)
        except Exception:
            continue

    for pid, (args, cmd) in cand_procs.items():
        # Skip forked DataLoader worker children whose parent is already in cand_procs
        if ppid_map.get(pid, -1) in cand_procs:
            continue
        try:
            is_ntg = ("train_testbed.py" in cmd)
            ext_cmds.append(cmd)
            env_raw = open(f"/proc/{pid}/environ", "rb").read()
            env_kv = {}
            for item in env_raw.split(b"\x00"):
                if b"=" in item:
                    k, v = item.split(b"=", 1)
                    env_kv[k.decode("utf-8", errors="ignore")] = v.decode("utf-8", errors="ignore")
            g_idx = None
            cvd = env_kv.get("CUDA_VISIBLE_DEVICES", "").strip()
            if cvd and cvd.split(",")[0].isdigit():
                g_idx = int(cvd.split(",")[0])
            elif "--gpu" in args:
                idx = args.index("--gpu")
                if idx + 1 < len(args) and args[idx + 1].isdigit():
                    g_idx = int(args[idx + 1])
            if g_idx is not None and g_idx in ext_slots:
                ext_slots[g_idx] += (2 if is_ntg else 1)
        except Exception:
            continue

    return ext_cmds, ext_slots, orig_sched_alive


def build_jobs() -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []

    # Priority 0: Fresh inference of resident_v4 with CPU DirDiag + unclip_T decoder
    out_v4_json = "results/final/pred_nriqa_dataset_test_r55_resident_v4_dirdiag_gate_eu.json"
    jobs.append({
        "id": "p0_infer_v4_dirdiag",
        "slots": 1,
        "match_keys": ["final_engine.py", "--set resident_v4", out_v4_json],
        "done_path": out_v4_json,
        "prereqs": ["results/r55/final/resident_v4_decoder_routed_gate_eu.pkl"],
        "shell": (
            f"python3 -u code/r55/final_engine.py --set resident_v4 --gpu {{gpu}} --half "
            f"--decoder routed_gate_eu --template results/v2_full/template_150.json "
            f"--img-dir data/test_150 --out {out_v4_json} && "
            f"python3 -u code/r55/validate_submission.py {out_v4_json} --template results/v2_full/template_150.json"
        ),
    })

    # Wave A1: 100%-data Full-Fit of ntg_pe (E8, fixed, tiles+global)
    jobs.append({
        "id": "waveA_fullfit_ntg_pe_e8",
        "slots": 2,
        "match_keys": ["train_testbed.py", "--out-dir results/r55/fullfit_v5_ntg", "--fold 0"],
        "done_path": "results/r55/fullfit_v5_ntg/pe_core_large/fold_0_best.pt",
        "prereqs": [],
        "shell": (
            "CUDA_VISIBLE_DEVICES={gpu} python3 -u code/r55/train_testbed.py "
            "--backbone pe_core_large --fold 0 --gpu 0 --seed 2026 "
            "--split-dir data/next55_fullfit/qingyu "
            "--out-dir results/r55/fullfit_v5_ntg "
            "--dual-engine-npz results/qingyu/dual_engine_meta.npz "
            "--epochs 8 --batch-size 6 --native-tiles tiles+global --tile-cap 16 "
            "--workers 4 --select fixed --fast-fixed --skip-t150"
        ),
    })

    # Wave A2: Fast-fixed equivalence check on folds 0-1 (E8, seed 2026)
    for f in (0, 1):
        jobs.append({
            "id": f"waveA_ntg_pe_fastcheck_f{f}",
            "slots": 2,
            "match_keys": ["train_testbed.py", "--out-dir results/next55/towers/r55_ntg_fastcheck_s2026", f"--fold {f}"],
            "done_path": f"results/next55/towers/r55_ntg_fastcheck_s2026/pe_core_large/fold_{f}_probs.npz",
            "prereqs": [],
            "shell": (
                f"CUDA_VISIBLE_DEVICES={{gpu}} python3 -u code/r55/train_testbed.py "
                f"--backbone pe_core_large --fold {f} --gpu 0 --seed 2026 "
                f"--out-dir results/next55/towers/r55_ntg_fastcheck_s2026 "
                f"--epochs 8 --batch-size 6 --native-tiles tiles+global --tile-cap 16 "
                f"--workers 4 --select fixed --fast-fixed --skip-t150 --no-save-ckpt"
            ),
        })

    # Wave A3: ntg_pe Fixed-E4 and Fixed-E6 8-fold OOF sweeps
    for ep in (4, 6):
        for f in range(8):
            jobs.append({
                "id": f"waveA_ntg_pe_e{ep}_f{f}",
                "slots": 2,
                "match_keys": ["train_testbed.py", f"--out-dir results/next55/towers/r55_ntg_e{ep}_s2026", f"--fold {f}"],
                "done_path": f"results/next55/towers/r55_ntg_e{ep}_s2026/pe_core_large/fold_{f}_probs.npz",
                "prereqs": [],
                "shell": (
                    f"CUDA_VISIBLE_DEVICES={{gpu}} python3 -u code/r55/train_testbed.py "
                    f"--backbone pe_core_large --fold {f} --gpu 0 --seed 2026 "
                    f"--out-dir results/next55/towers/r55_ntg_e{ep}_s2026 "
                    f"--epochs {ep} --batch-size 6 --native-tiles tiles+global --tile-cap 16 "
                    f"--workers 4 --select fixed --fast-fixed --skip-t150 --no-save-ckpt"
                ),
            })

    # Post-Wave-A: Bench + Fresh Inference of resident_v5 as soon as fullfit_ntg_pe_e8 finishes!
    out_v5_json = "results/final/pred_nriqa_dataset_test_r55_resident_v5_routed_gate_eu.json"
    jobs.append({
        "id": "waveA_bench_and_infer_v5",
        "slots": 1,
        "defer_if_orig_sched": True,
        "match_keys": ["final_engine.py", "--set resident_v5"],
        "done_path": out_v5_json,
        "prereqs": [
            "results/r55/fullfit_v5_ntg/pe_core_large/fold_0_best.pt",
            "results/r55/final/resident_v5_decoder_routed_gate_eu.pkl",
        ],
        "shell": (
            f"python3 -u code/r55/final_engine.py --set resident_v5 --gpu {{gpu}} --half --bench 40 && "
            f"python3 -u code/r55/final_engine.py --set resident_v5 --gpu {{gpu}} --half "
            f"--decoder routed_gate_eu --template results/v2_full/template_150.json "
            f"--img-dir data/test_150 --out {out_v5_json} && "
            f"python3 -u code/r55/validate_submission.py {out_v5_json} --template results/v2_full/template_150.json"
        ),
    })

    # Wave B1: Multi-Backbone Native-Tile 2-Fold Screen + Same-Slot Seed Control
    for tag, bb, ep, sd in [
        ("r55_ntg_cradio_e4_s2026", "cradio_v4_h", 4, 2026),
        ("r55_ntg_dinov3_e6_s2026", "dinov3_vitl16", 6, 2026),
        ("r55_ntg_convnext_e8_s2026", "convnextv2_base", 8, 2026),
        ("r55_ntg_pe_ctrl_s2027", "pe_core_large", 8, 2027),
    ]:
        for f in (0, 1):
            jobs.append({
                "id": f"waveB_{tag}_f{f}",
                "slots": 2,
                "match_keys": ["train_testbed.py", f"--out-dir results/next55/towers/{tag}", f"--fold {f}"],
                "done_path": f"results/next55/towers/{tag}/{bb}/fold_{f}_probs.npz",
                "prereqs": [],
                "shell": (
                    f"CUDA_VISIBLE_DEVICES={{gpu}} python3 -u code/r55/train_testbed.py "
                    f"--backbone {bb} --fold {f} --gpu 0 --seed {sd} "
                    f"--out-dir results/next55/towers/{tag} "
                    f"--epochs {ep} --batch-size 6 --native-tiles tiles+global --tile-cap 16 "
                    f"--workers 4 --select fixed --fast-fixed --skip-t150 --no-save-ckpt"
                ),
            })

    # Wave B2: P12 Conditional-b1 (2 seeds) vs Same-Teacher Seed Controls (2 seeds) on Folds 0-1
    for tag, sd, extra_flag in [
        ("r55_p12_condb1_e5_s2026", 2026, "--cond-b1"),
        ("r55_p12_condb1_e5_s2027", 2027, "--cond-b1"),
        ("r55_p12_ctrl_e5_s2027", 2027, ""),
        ("r55_p12_ctrl_e5_s2028", 2028, ""),
    ]:
        for f in (0, 1):
            jobs.append({
                "id": f"waveB_{tag}_f{f}",
                "slots": 1,
                "match_keys": ["train_p12_specialist.py", f"--out-dir results/next55/towers/{tag}", f"--fold {f}"],
                "done_path": f"results/next55/towers/{tag}/pe_core_large/fold_{f}_probs.npz",
                "prereqs": [],
                "shell": (
                    f"CUDA_VISIBLE_DEVICES={{gpu}} python3 -u code/r55/train_p12_specialist.py "
                    f"--backbone pe_core_large --fold {f} --gpu 0 --seed {sd} "
                    f"--out-dir results/next55/towers/{tag} "
                    f"--dual-engine-npz results/r55/semantic_roi_meta.npz "
                    f"--epochs 5 --batch-size 12 --lr-backbone 4e-5 --lr-head 5e-4 --unfreeze-blocks 9 "
                    f"--nested-kd-npz results/r55/nested_kd_teacher.npz --lambda-kd 0.35 "
                    f"--workers 4 --select fixed --fast-fixed --no-save-ckpt {extra_flag}"
                ).strip(),
            })

    return jobs


def is_externally_running(job: Dict[str, object], ext_cmds: List[str]) -> bool:
    keys = [str(k) for k in job.get("match_keys", [])]
    if not keys:
        return False
    return any(all(k in cmd for k in keys) for cmd in ext_cmds)


def collect_descendant_pids(root_pids: Set[int]) -> Set[int]:
    """Include full descendant tree (bash -> python -> DataLoader workers) of our own jobs."""
    all_pids = set(root_pids)
    ppid_map: Dict[int, int] = {}
    for spath in glob.glob("/proc/[0-9]*/stat"):
        try:
            txt = open(spath, "r", encoding="utf-8", errors="ignore").read()
            rparen = txt.rfind(")")
            fields = txt[rparen + 2:].split()
            ppid_map[int(spath.split("/")[2])] = int(fields[1])
        except Exception:
            continue
    changed = True
    while changed:
        changed = False
        for pid, ppid in ppid_map.items():
            if ppid in all_pids and pid not in all_pids:
                all_pids.add(pid)
                changed = True
    return all_pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true", help="Co-exist safely with an already-running wave scheduler")
    ap.add_argument("--dry-run", action="store_true", help="Print detected GPU states and runnable jobs without launching")
    args = ap.parse_args()

    os.chdir(REPO)
    os.makedirs(LOG_DIR, exist_ok=True)
    gpus = list(range(8))
    gpu_cap = 2
    todo = build_jobs()
    running: Dict[subprocess.Popen, Tuple[int, Dict[str, object], float, object]] = {}
    done_ids: List[str] = []
    failed_ids: List[str] = []

    print(f"[v5-8gpu] Starting smart queue (backfill={args.backfill}): {len(todo)} total jobs across GPUs {gpus} | "
          f"free disk = {check_disk_gb():.1f} GB", flush=True)

    if args.dry_run:
        ext_cmds, ext_slots, orig_alive = scan_live_external_processes(set())
        mem_mib = query_gpu_mem_mib()
        print(f"  orig_sched_alive={orig_alive} | ext_slots={ext_slots} | mem_mib={mem_mib}")
        for j in todo:
            done_ok = os.path.exists(os.path.join(REPO, str(j["done_path"])))
            ext_run = is_externally_running(j, ext_cmds)
            print(f"  job={j['id']:34s} slots={j['slots']} done={done_ok} ext_running={ext_run}")
        return

    while todo or running:
        for proc in list(running.keys()):
            rc = proc.poll()
            if rc is not None:
                g, job, t0, fh = running.pop(proc)
                fh.close()
                dt = time.time() - t0
                jid = str(job["id"])
                if rc == 0 and os.path.exists(os.path.join(REPO, str(job["done_path"]))):
                    done_ids.append(jid)
                    print(f"[v5-8gpu] DONE   {jid:34s} (gpu={g}, {dt:5.0f}s) | done={len(done_ids)} run={len(running)} wait={len(todo)}", flush=True)
                else:
                    failed_ids.append(jid)
                    print(f"[v5-8gpu] FAILED {jid:34s} (gpu={g}, rc={rc}, {dt:5.0f}s) -> logs/r55/v5_waves/{jid}.log", flush=True)

        # Skip any completed jobs (including ones completed externally!)
        for job in list(todo):
            if os.path.exists(os.path.join(REPO, str(job["done_path"]))):
                todo.remove(job)
                done_ids.append(str(job["id"]))
                print(f"[v5-8gpu] SKIP   {job['id']:34s} (done_path exists)", flush=True)

        if not todo and not running:
            break

        own_pids = collect_descendant_pids({p.pid for p in running.keys()})
        ext_cmds, ext_slots, orig_alive = scan_live_external_processes(own_pids)
        mem_mib = query_gpu_mem_mib()

        if todo:
            free_gb = check_disk_gb()
            if free_gb < 80.0:
                print(f"[v5-8gpu][ABORT] Free disk {free_gb:.1f} GB < 80 GB guard!", flush=True)
                break

            for g in gpus:
                own_used = sum(int(jb["slots"]) for gg, jb, _, _ in running.values() if gg == g)
                total_used = own_used + ext_slots.get(g, 0)
                avail_slots = gpu_cap - total_used
                if avail_slots <= 0:
                    continue

                cur_mem = mem_mib.get(g, 0)
                chosen = None
                for job in todo:
                    if is_externally_running(job, ext_cmds):
                        continue
                    if job.get("defer_if_orig_sched") and orig_alive:
                        continue
                    need_slots = int(job["slots"])
                    if need_slots > avail_slots:
                        continue
                    # Hardware VRAM safety gate (prevents OOM even before external mem settles)
                    if need_slots == 2 and cur_mem >= 3000:
                        continue
                    if need_slots == 1 and cur_mem >= 68000:
                        continue
                    prereqs = [str(p) for p in job.get("prereqs", [])]
                    if not all(os.path.exists(os.path.join(REPO, p)) for p in prereqs):
                        continue
                    chosen = job
                    break

                if chosen is not None:
                    todo.remove(chosen)
                    jid = str(chosen["id"])
                    need_slots = int(chosen["slots"])
                    cmd_str = str(chosen["shell"]).format(gpu=g)
                    log_fh = open(os.path.join(LOG_DIR, f"{jid}.log"), "w", encoding="utf-8")
                    env = dict(
                        os.environ,
                        PYTHONPATH=f"{REPO}/code:{REPO}/code/r55:{REPO}/code/next55",
                        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                        OMP_NUM_THREADS="4",
                        MKL_NUM_THREADS="4",
                    )
                    proc = subprocess.Popen(["bash", "-lc", cmd_str], cwd=REPO, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
                    running[proc] = (g, chosen, time.time(), log_fh)
                    print(
                        f"[v5-8gpu] START  {jid:34s} on gpu={g} "
                        f"(need_slots={need_slots}, used={total_used + need_slots}/{gpu_cap}, mem_before={cur_mem}MiB)",
                        flush=True,
                    )
                    # Stagger startup by 2.5s to avoid concurrent HF/filesystem metadata spikes
                    time.sleep(2.5)
                    mem_mib = query_gpu_mem_mib()

        if todo or running:
            time.sleep(8.0)

    print(f"[v5-8gpu] ALL FINISHED: ok={len(done_ids)}, failed={len(failed_ids)} {failed_ids}", flush=True)
    sys.exit(1 if failed_ids else 0)


if __name__ == "__main__":
    main()
