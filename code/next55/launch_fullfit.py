#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/launch_fullfit.py — refits every tower of a frozen stack config on 100% of the data
(one model per tower/seed = the shippable, <=30B-param model).  One job per GPU, queued.

Reads results/next55/stack/<tag>/final.pkl -> towers, and for each tower key launches:
  qy_<bb> / <prefix>_<bb> (Tier-1)   train_tier1_5backbones_8fold.py --fold 0 --split-dir data/next55_fullfit/qingyu
  *_internvit_6b                      train_tier2_internvit6b_8fold.py  (same split-dir)
  *_qwen3_vl_8b                       train_tier3_qwen3vl8b_8fold.py    (same split-dir)
  v16_full8 / v16_s<seed>             launch_v16_seedbag.py --fullfit
with the SAME seed and hyper-parameters as its 8-fold OOF run, writing to results/next55/fullfit/...
If FINAL_TEMPLATE / FINAL_IMG_DIR are exported, the final set is predicted inside the same run.

Usage:
  python code/next55/make_fullfit_splits.py
  python code/next55/launch_fullfit.py --stack-tag s2 [--gpus 0 1 2 3 4 5 6 7] [--dry-run]
"""
import argparse
import os
import pickle
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common55 import FULLFIT_ROOT, OUT_DIR, QINGYU_TIER1, REPO, discover_towers  # noqa: E402


def seed_of(prefix: str, default: int) -> int:
    m = re.search(r"_s(\d+)$", prefix) or re.search(r"^v16_s(\d+)$", prefix)
    return int(m.group(1)) if m else default


def job_for(key: str):
    fmt, d = discover_towers()[key]
    ft, fe = os.environ.get("FINAL_TEMPLATE", ""), os.environ.get("FINAL_IMG_DIR", "")
    dual = os.environ.get("NEXT55_DUAL_NPZ", "results/qingyu/dual_engine_meta.npz")
    if fmt == "v16":
        seed = 600 if key == "v16_full8" else seed_of(key, 700)
        variant = 0 if key == "v16_full8" else 1
        done = os.path.join(FULLFIT_ROOT, key, "V16_Full_F0_AllGrade_Microscope_swa.pt")
        return done, ["python3", "-u", "code/next55/launch_v16_seedbag.py", "--fullfit", "--tower-key", key,
                      "--seed", str(seed), "--variant", str(variant)]
    d = d.rstrip("/")
    bb = os.path.basename(d)
    parent = os.path.dirname(d)
    prefix = "qy" if parent == os.path.join(REPO, "results/qingyu") else os.path.basename(parent)
    out_dir = os.path.relpath(os.path.join(FULLFIT_ROOT, prefix), REPO)
    common = ["--fold", "0", "--split-dir", "data/next55_fullfit/qingyu", "--out-dir", out_dir,
              "--seed", str(seed_of(prefix, 2026)), "--dual-engine-npz", dual,
              "--extra-test-template", ft, "--extra-test-dir", fe]
    done = os.path.join(FULLFIT_ROOT, prefix, bb, "fold_0_probs.npz")
    if bb in QINGYU_TIER1:
        return done, ["python3", "-u", "code/qingyu/train_tier1_5backbones_8fold.py", "--backbone", bb] + common
    if bb == "internvit_6b":
        return done, ["python3", "-u", "code/qingyu/train_tier2_internvit6b_8fold.py"] + common
    if bb == "qwen3_vl_8b":
        return done, ["python3", "-u", "code/qingyu/train_tier3_qwen3vl8b_8fold.py"] + common
    raise ValueError(f"no full-fit recipe for tower {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-tag", required=True)
    ap.add_argument("--gpus", type=int, nargs="*", default=list(range(8)))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)
    assert os.path.exists("data/next55_fullfit/qingyu/fold_0.csv"), "run code/next55/make_fullfit_splits.py first"
    towers = pickle.load(open(os.path.join(OUT_DIR, "stack", a.stack_tag, "final.pkl"), "rb"))["towers"]
    queue = []
    for t in towers:
        done, cmd = job_for(t)
        if os.path.exists(done):
            print(f"[skip] {t} (found {done})")
            continue
        queue.append((t, cmd))
        print(f"[plan] {t}: {' '.join(cmd)}")
    if a.dry_run or not queue:
        return
    os.makedirs(os.path.join(FULLFIT_ROOT, "logs"), exist_ok=True)
    env0 = dict(os.environ, PYTHONPATH=os.path.join(REPO, "code"), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    running, failed, t0 = {}, [], time.time()
    while queue or running:
        busy = {g for (_, g, _) in running.values()}
        for g in [g for g in a.gpus if g not in busy]:
            if not queue:
                break
            t, cmd = queue.pop(0)
            if cmd[2].endswith("launch_v16_seedbag.py"):
                cmd = cmd + ["--gpus", str(g)]
                env = env0
            else:
                cmd = cmd + ["--gpu", "0"]
                env = dict(env0, CUDA_VISIBLE_DEVICES=str(g))
            log = open(os.path.join(FULLFIT_ROOT, "logs", f"{t}.log"), "w")
            running[t] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), g, log)
            print(f"[{time.time()-t0:6.0f}s] launch {t} on GPU {g}", flush=True)
        time.sleep(30)
        for t in list(running):
            p, g, log = running[t]
            if p.poll() is None:
                continue
            log.close()
            (failed.append(t) if p.returncode else None)
            print(f"[{time.time()-t0:6.0f}s] {'FAIL' if p.returncode else 'DONE'} {t} (exit {p.returncode})", flush=True)
            del running[t]
    if failed:
        print(f"[fullfit] FAILED: {failed} -> results/next55/fullfit/logs/")
        sys.exit(1)
    print("[fullfit] all done")


if __name__ == "__main__":
    main()
