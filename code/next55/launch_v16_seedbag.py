#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/launch_v16_seedbag.py — 8 GPUs, ~1.6 h per seed (3 ep × ~27 min + TTA).

Re-runs the strongest existing tower (V16 AllGrade Microscope, OOF Doc2S 50.0) with a new seed,
the exact same 8-fold job table (backbones / synth manifests / hyper-params) as
code/finetune/launch_v16_fulldata_8gpus.py, but
  * writes to results/next55/towers/v16_s<seed>/   (auto-discovered by common55.discover_towers)
  * --save-ckpt  (the original V16 run kept no weights -> could never score the final set)
  * --extra-template <json> when FINAL_TEMPLATE is set (predicts the final set in the same run)

Usage:
  python code/next55/launch_v16_seedbag.py --seed 700 [--gpus 0 1 2 3 4 5 6 7]
  FINAL_TEMPLATE=... NEXT55_EXTRA_IMG_DIR=... python code/next55/launch_v16_seedbag.py --seed 700
  python code/next55/launch_v16_seedbag.py --fullfit --tower-key v16_s700 --seed 700 --variant 1 --gpus 3
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_jobs():
    spec = importlib.util.spec_from_file_location("lv16", os.path.join(REPO, "code/finetune/launch_v16_fulldata_8gpus.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.JOBS, mod.ALL6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="base seed; fold k uses seed+k (original run used 600+k)")
    ap.add_argument("--gpus", type=int, nargs="*", default=list(range(8)))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--fullfit", action="store_true", help="ONE 100%%-data refit -> results/next55/fullfit/<tower-key>/")
    ap.add_argument("--tower-key", default="", help="--fullfit: OOF tower this refit replaces (v16_full8 / v16_s700 ...)")
    ap.add_argument("--variant", type=int, default=1, help="--fullfit: copy backbones/hparams from JOBS[variant] (0=F0 style, 1=F1 style)")
    a = ap.parse_args()
    os.chdir(REPO)
    jobs, all6 = load_jobs()
    if a.fullfit:
        assert a.tower_key, "--fullfit needs --tower-key"
        j = list(jobs[a.variant])
        j[2] = 0
        j[6] = "data/next55_fullfit/v16/split_full_v16_cleanog.csv"
        j[7] = "data/next55_fullfit/v16/manifest_full_v16_cleanog.csv"
        assert os.path.exists(j[6]), "run code/next55/make_fullfit_splits.py first"
        jobs = [tuple(["V16_Full_F0_AllGrade_Microscope"] + j[1:])]
        out_dir = os.path.join(REPO, "results/next55/fullfit", a.tower_key)
    else:
        assert a.seed != 600, "seed 600 is the existing results/v16_fulldata run"
        out_dir = os.path.join(REPO, f"results/next55/towers/v16_s{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    final_tpl = os.environ.get("FINAL_TEMPLATE", "")

    # Pre-extract V16 physics cache single-process so 8 parallel GPU folds never compete or write concurrently
    sys.path.insert(0, os.path.join(REPO, "code/finetune"))
    from extract_native_phys_feats import extract_for_names
    import csv
    all_needed = set()
    for j in jobs:
        for p in (j[6], j[7]):
            if os.path.exists(p):
                with open(p) as fp:
                    all_needed.update(r['name'] for r in csv.DictReader(fp))
    if os.path.exists('results/v2_full/template_150.json'):
        all_needed.update(e['image'] for e in json.load(open('results/v2_full/template_150.json')))
    print(f"[v16-seedbag] Verifying physics cache for {len(all_needed)} images...", flush=True)
    extract_for_names(sorted(all_needed))
    print("[v16-seedbag] Physics cache is ready.", flush=True)

    queue = list(jobs)
    running = {}
    t0 = time.time()
    failed = []
    while queue or running:
        free = [g for g in a.gpus if g not in {v[1] for v in running.values()}]
        while queue and free:
            name, _, fold, bbs, sz, tiles, split_csv, man_csv, w_neg, p0_cln, lr_hd, cln_rep = queue.pop(0)
            gpu = free.pop(0)
            out_base = os.path.join(out_dir, name)
            if os.path.exists(out_base + "_probs.npz") and os.path.exists(out_base + "_swa.pt"):
                print(f"[skip] {name}", flush=True)
                continue
            cmd = ["python3", "-u", "-I", "code/finetune/train_v16_microscope_specialist.py",
                   "--backbones", bbs, "--focus-classes", all6, "--seed", str(a.seed + fold),
                   "--epochs", str(a.epochs), "--batch", "8", "--grad-accum", "1", "--workers", "8",
                   "--size", str(sz), "--tiles", str(tiles), "--clean-repeat", str(cln_rep),
                   "--p0-clean-weight", str(p0_cln), "--channel-neg-weight", str(w_neg),
                   "--lr-full", "7.5e-05", "--lr-last", "2.6e-05", "--lr-head", str(lr_hd),
                   "--wd", "0.012", "--dropout", "0.12", "--alpha-ex", "0.58",
                   "--split", split_csv, "--manifest", man_csv, "--out", out_base, "--save-ckpt"]
            if final_tpl:
                cmd += ["--extra-template", final_tpl]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            log = open(out_base + ".log", "w")
            running[name] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), gpu, log, out_base)
            print(f"[{time.time()-t0:6.0f}s] launch {name} fold={fold} gpu={gpu} seed={a.seed + fold}", flush=True)
        time.sleep(20)
        for name in list(running):
            p, gpu, log, out_base = running[name]
            if p.poll() is None:
                continue
            log.close()
            if p.returncode == 0 and os.path.exists(out_base + ".json"):
                d = json.load(open(out_base + ".json"))
                print(f"[{time.time()-t0:6.0f}s] DONE {name}: Doc1={d['doc1']:.2f} Doc2={d['doc2']:.2f}", flush=True)
            else:
                failed.append(name)
                print(f"[{time.time()-t0:6.0f}s] FAIL {name} (exit {p.returncode}) -> {out_base}.log", flush=True)
            del running[name]
    if failed:
        print(f"[v16-seedbag] FAILED: {failed}", flush=True)
        sys.exit(1)
    print(f"[v16-seedbag] all 8 folds done in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
