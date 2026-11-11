#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/prep_aux.py — make the auxiliary pool trainable (8 GPUs, ~20-40 min for ~5k images).

  1. template   data/r55_aux/template.json  (every manifest row whose file exists)
  2. meta       results/r55/dual_engine_meta_aux.npz  = physics vectors + guided crop boxes for
                train + t150 + aux (8 ranks + merge; no lazy CUDA work inside DataLoader workers)
  3. teachers   every Tier-1 tower of the accepted stack predicts the aux images with each of its 8 fold
                models  ->  <tower_dir>/ext_<tag>_probs.npz
  4. merge      results/r55/aux_teacher.npz : names, probs_folds (8, N, 6, 4) = log-linear mean over towers

Leak rule: the student of fold k uses probs_folds[k] only.  Those models never saw fold k's validation
images or labels, so the soft labels carry no validation information.

Usage:  python code/r55/prep_aux.py [--steps template meta teachers merge] [--gpus 8] [--tag r55aux]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import (AUX_IMG, AUX_META, AUX_TEACHER, AUX_TEMPLATE, R55_OUT, REPO, TEACHER_TOWERS,  # noqa: E402
                        read_manifest)


def run_parallel(cmds, logs):
    procs = []
    for cmd, log in zip(cmds, logs):
        os.makedirs(os.path.dirname(log), exist_ok=True)
        procs.append(subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                                      env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "code"))))
    rc = [p.wait() for p in procs]
    assert all(c == 0 for c in rc), f"jobs failed rc={rc}; see {logs}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", default=["template", "meta", "teachers", "merge"])
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--tag", default="r55aux")
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(R55_OUT, exist_ok=True)
    py = sys.executable

    if "template" in a.steps:
        names = [r["name"] for r in read_manifest() if os.path.exists(os.path.join(AUX_IMG, r["name"]))]
        assert names, "auxiliary manifest is empty — run build_rank_tuples.py / ingest_external_pool.py first"
        assert len(names) == len(set(names))
        json.dump([{"image": n} for n in names], open(AUX_TEMPLATE, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[prep_aux] template: {len(names)} images")

    if "meta" in a.steps:
        run_parallel(
            [[py, "-u", "code/qingyu/dual_engine_preprocess.py", "--rank", str(r), "--world-size", str(a.gpus),
              "--extra-template", AUX_TEMPLATE, "--extra-test-dir", AUX_IMG, "--out", AUX_META] for r in range(a.gpus)],
            [os.path.join(R55_OUT, f"logs/meta_shard{r}.log") for r in range(a.gpus)])
        subprocess.check_call([py, "-u", "code/qingyu/dual_engine_preprocess.py", "--merge-shards", str(a.gpus), "--out", AUX_META],
                              env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "code")))
        d = np.load(AUX_META, allow_pickle=True)
        have = set(map(str, d["names"]))
        miss = [x["image"] for x in json.load(open(AUX_TEMPLATE, encoding="utf-8")) if x["image"] not in have]
        assert not miss, f"{len(miss)} aux images missing from meta, e.g. {miss[:3]}"
        print(f"[prep_aux] meta: {len(have)} images -> {AUX_META}")

    if "teachers" in a.steps:
        cmds, logs = [], []
        for i, (bb, d) in enumerate(TEACHER_TOWERS):
            cmds.append([py, "-u", "code/next55/infer_tier1_ckpt.py", "--backbone", bb, "--tower-dir", d,
                         "--template", AUX_TEMPLATE, "--img-dir", AUX_IMG, "--tag", a.tag, "--gpu", str(i % a.gpus),
                         "--dual-engine-npz", AUX_META])
            logs.append(os.path.join(R55_OUT, f"logs/teacher_{i}_{bb}.log"))
        run_parallel(cmds, logs)
        print("[prep_aux] teachers done")

    if "merge" in a.steps:
        names = [x["image"] for x in json.load(open(AUX_TEMPLATE, encoding="utf-8"))]
        acc = []
        for bb, d in TEACHER_TOWERS:
            z = np.load(os.path.join(REPO, d, f"ext_{a.tag}_probs.npz"), allow_pickle=True)
            assert list(map(str, z["names"])) == names, f"{d}: teacher output is stale — rerun the teachers step"
            assert z["probs_6x4_folds"].shape[0] == 8
            acc.append(np.log(np.clip(z["probs_6x4_folds"].astype(np.float64), 1e-6, 1.0)))
        p = np.exp(np.mean(acc, axis=0))                      # (8, N, 6, 4) log-linear mean over towers
        p[:, :, [3, 5], 1:3] = 0.0                            # 花屏 / 虚化突兀 have only {无问题, P0}
        p = p / p.sum(-1, keepdims=True)
        np.savez_compressed(AUX_TEACHER, names=np.array(names), probs_folds=p.astype(np.float32),
                            towers=np.array([d for _, d in TEACHER_TOWERS]))
        print(f"[prep_aux] teacher soft labels: {p.shape} -> {AUX_TEACHER}")


if __name__ == "__main__":
    main()
