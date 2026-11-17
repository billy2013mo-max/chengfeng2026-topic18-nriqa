#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/queue_run.py — small GPU job queue for train_testbed.py experiments (several sessions share this machine).

Each job is one fold of one preset of run_r55.sh ("<preset>:<seed>:<folds>", folds like 0-7 or 2,3).  A job starts on
a GPU only when (a) that GPU has >= --need-gb free memory RIGHT NOW (so it never lands on top of another session's
jobs), (b) this queue runs fewer than --per-gpu jobs there, and (c) its prerequisite files exist (ssl/xd wait for their
teachers).  A lock file per job (logs/r55/locks) keeps two instances from starting the same fold; finished folds
(fold_k_probs.npz present) are skipped, so the queue is resumable.

Two instances must not launch on the same GPU within the allocation window (three jobs landed on one GPU and the
newest died with OOM on 2026-11-11): every launch takes a per-GPU file lock and a time stamp shared by all instances,
and no instance launches on a GPU whose stamp is younger than 150 s.  Prefer ONE instance for all GPUs.
A job lock whose trainer is no longer running and that left no result is stale (crashed or orphaned run) and is
taken over.
Logs and output directories are exactly those of `run_r55.sh exp`.

Usage:
  python code/r55/queue_run.py --gpus 2 3 4 5 --jobs fixed:2026:2-7 rcrop:2026:2-7 halfA:2026:0-7 ...
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOW = "results/next55/towers"
LOCKS = os.path.join(REPO, "logs/r55/locks")


def folds_of(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def free_gb():
    o = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"]).decode()
    return {int(i): (float(t) - float(u)) / 1024.0 for i, t, u in (l.split(",") for l in o.strip().splitlines())}


def gpu_gate(g, hold=150):
    """True (and the stamp is refreshed) if no instance launched on GPU g during the last `hold` seconds."""
    import fcntl
    stamp = os.path.join(LOCKS, f"gpu{g}.stamp")
    with open(os.path.join(LOCKS, f"gpu{g}.flock"), "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if os.path.exists(stamp) and time.time() - os.path.getmtime(stamp) < hold:
                return False
            open(stamp, "w").close()
            return True
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def trainer_running(out_dir, fold, bb):
    o = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any("train_testbed.py" in l and f"--backbone {bb} " in l and f"--out-dir {out_dir} " in l and f"--fold {fold} " in l
               for l in o.splitlines())


def needs(name, seed, fold, bb):
    def t(n, s):
        return os.path.join(REPO, TOW, f"r55_{n}_s{s}", bb, f"fold_{fold}_probs.npz")
    if name == "ssl":
        return [t("halfA", 2026), t("halfA", 2027)]
    if name == "xd":
        return [t("halfA", 2026), t("halfA", 2027), t("halfB", 2026), t("halfB", 2027)]
    if name.startswith("synth-") or name.startswith("shiftsynth-"):
        r = name.split("-", 1)[1]
        return [os.path.join(REPO, f"results/r55/dual_engine_meta_aux_{r}.npz"),
                os.path.join(REPO, f"results/r55/aux_teacher_{r}" + ("_shift" if name.startswith("shift") else "") + ".npz")]
    if name in ("glitch", "shiftglitch", "glitch2", "shiftglitch2"):
        return [os.path.join(REPO, "results/r55/dual_engine_meta_aux_glitch.npz"),
                os.path.join(REPO, "results/r55/aux_teacher_glitch" + ("_shift" if name.startswith("shiftglitch") else "") + ".npz")]
    if name == "shiftkvq":
        return [os.path.join(REPO, "results/r55/aux_teacher_kvq_shift.npz")]
    if name in ("kvq", "kvqclean", "kvqcert"):                    # in-domain pool: teacher soft labels (+ the certified-clean list)
        req = [os.path.join(REPO, "results/r55/aux_teacher_kvq.npz")]
        return req + ([os.path.join(REPO, "data/r55_aux/kvq_clean.csv")] if name != "kvq" else [])
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", required=True)
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--need-gb", type=float, default=68.0)
    ap.add_argument("--backbone", default="pe_core_large")
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--poll", type=int, default=30)
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(LOCKS, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "code"))

    todo = []
    for j in a.jobs:
        name, seed, fs = j.split(":")
        flags = subprocess.check_output(["bash", "run_r55.sh", "preset", name], env=dict(env, BACKBONE=a.backbone)).decode().split()
        for f in folds_of(fs):
            todo.append(dict(name=name, seed=int(seed), fold=f, flags=flags))
    # locks and per-job logs carry the backbone (except the default one, whose names other sessions already use):
    # two instances training the same preset/seed/fold for different backbones collided on one lock on 2026-11-12
    sfx = "" if a.backbone == "pe_core_large" else "." + a.backbone
    running, cool, failed, done = {}, {}, [], 0          # running: Popen -> (gpu, job)
    print(f"[queue] {len(todo)} jobs on GPUs {a.gpus} (<= {a.per_gpu}/GPU, need {a.need_gb:.0f} GB free)", flush=True)

    while todo or running:
        for p in list(running):
            if p.poll() is not None:
                g, job = running.pop(p)
                tag = f"{job['name']}_s{job['seed']}_f{job['fold']}"
                if p.returncode == 0:
                    done += 1
                    print(f"[queue] done   {tag} (gpu {g}) | {done} finished, {len(todo)} waiting, {len(running)} running", flush=True)
                else:
                    failed.append(tag)
                    print(f"[queue] FAILED {tag} (gpu {g}, rc={p.returncode}) see logs/r55/exp_{tag}{sfx}.log", flush=True)
        if todo:
            fg = free_gb()
            for g in a.gpus:
                n_here = sum(1 for gg, _ in running.values() if gg == g)
                if n_here >= a.per_gpu or time.time() < cool.get(g, 0) or fg.get(g, 0) < a.need_gb:
                    continue
                pick = None
                for job in todo:
                    out_dir = f"{TOW}/r55_{job['name']}_s{job['seed']}"
                    tag = f"{job['name']}_s{job['seed']}_f{job['fold']}"
                    if os.path.exists(os.path.join(REPO, out_dir, a.backbone, f"fold_{job['fold']}_probs.npz")):
                        todo.remove(job)
                        print(f"[queue] skip   {tag}: already finished", flush=True)
                        break
                    if not all(os.path.exists(x) for x in needs(job["name"], job["seed"], job["fold"], a.backbone)):
                        continue
                    lock = os.path.join(LOCKS, tag + sfx + ".lock")
                    if os.path.exists(lock):
                        if trainer_running(out_dir, job["fold"], a.backbone):
                            if job.get("seen_running") is None:
                                print(f"[queue] wait   {tag}: running under another launcher", flush=True)
                            job["seen_running"] = True
                            continue                            # keep it: if that run dies without a result we take over
                        try:
                            if time.time() - os.path.getmtime(lock) < 300:
                                continue                        # just claimed by someone, trainer not visible yet
                            print(f"[queue] stale  {tag}: lock without trainer or result, taking over", flush=True)
                            os.remove(lock)
                        except FileNotFoundError:               # another instance removed it between the two calls
                            continue
                    if not gpu_gate(g):
                        break
                    try:
                        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                    except FileExistsError:
                        continue
                    pick = (job, out_dir, tag)
                    break
                if pick:
                    job, out_dir, tag = pick
                    todo.remove(job)
                    cmd = ["python3", "-u", "code/r55/train_testbed.py", "--backbone", a.backbone, "--seed", str(job["seed"]),
                           "--out-dir", out_dir, "--fold", str(job["fold"]), "--gpu", str(g)] + job["flags"]
                    log = open(os.path.join(REPO, f"logs/r55/exp_{tag}{sfx}.log"), "w")
                    running[subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)] = (g, job)
                    cool[g] = time.time() + 150            # let the new job allocate before judging free memory again
                    print(f"[queue] start  {tag} on gpu {g} ({fg[g]:.0f} GB free)", flush=True)
        time.sleep(a.poll)
    print(f"[queue] finished: {done} ok, {len(failed)} failed {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
