#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/synth_build.py — SynthLab stage 1: build one recipe (CPU for images, GPUs for the image metadata).

For recipe R (synth_recipes.RECIPES[R]) it writes
  data/r55_aux/images/r55syn_<R>_<suffix>__<seed>.png     the synthetic frames (PNG, same path for every variant)
  data/r55_aux/manifest_<R>.csv                           kind=synth, seed_fold = the seed's fold in data/qingyu_splits
  data/r55_aux/manifest_<R>_shift.csv                     the same with seed_fold = the seed's source family (data/r55_tagsplit)
  results/r55/aux_teacher_<R>.npz, ..._shift.npz          names, probs_folds (1,N,6,4) = labels by construction, mask (N,6), det (N,)
  results/r55/dual_engine_meta_aux_<R>.npz                crop boxes / physics vector, via prep_aux.py
  results/r55/synthlab/<R>/build.json                     counts and the recipe's doc string

Leak control: a synthetic frame inherits the fold of its seed, so the trainer never sees a frame derived from a
validation image of the fold it is training (train_testbed.build_aux_catalog drops rows with seed_fold == fold); the
shift manifest does the same for the held-out source family.  Labels come only from the seed's human labels and the
recipe's construction (synth_recipes.py); no model prediction is used anywhere in the build.

Usage:  python code/r55/synth_build.py --recipe benign1 [--workers 40] [--no-meta]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import AUX_DIR, AUX_IMG, FIELDS, REPO  # noqa: E402
import synth_recipes as sr  # noqa: E402

DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]


def find(name):
    for d in ("data/images", "data/images_supp"):
        p = os.path.join(REPO, d, name)
        if os.path.exists(p):
            return p


def job(arg):
    recipe, name = arg
    im = cv2.imdecode(np.fromfile(find(name), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return name, []
    out = []
    stem = os.path.splitext(name)[0]
    try:
        variants = sr.RECIPES[recipe]["fn"](im, sr.rng_for(recipe, name))
    except Exception:
        return name, []
    for suffix, arr, spec in variants:
        fn = f"r55syn_{recipe}_{suffix}__{stem}.png"
        cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(os.path.join(AUX_IMG, fn))
        out.append((fn, spec))
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--no-meta", action="store_true")
    a = ap.parse_args()
    R = sr.RECIPES[a.recipe]
    rows = list(csv.DictReader(open(os.path.join(REPO, "data/qingyu_splits/fold_0.csv"), encoding="utf-8")))
    rows = [r for r in rows if r["split"] in ("train", "val")]
    seen, lab = set(), {}
    for r in rows:
        if r["name"] not in seen:
            seen.add(r["name"])
            lab[r["name"]] = [int(r[d]) for d in DIMS]
    vfold = {}
    for f in range(8):
        for r in csv.DictReader(open(os.path.join(REPO, f"data/qingyu_splits/fold_{f}.csv"), encoding="utf-8")):
            if r["split"] == "val":
                vfold[r["name"]] = f
    sfold = {r["name"]: int(r["fold"]) for r in csv.DictReader(open(os.path.join(REPO, "data/r55_tagsplit/folds.csv"), encoding="utf-8"))}
    cand = sorted(n for n, g in lab.items() if sr.SEEDS[R["seeds"]](g) and n in vfold and n in sfold)
    rs = np.random.RandomState(0)
    pick = [cand[i] for i in sorted(rs.choice(len(cand), min(R["n_seeds"], len(cand)), replace=False))]
    os.makedirs(AUX_IMG, exist_ok=True)
    made = []
    with Pool(a.workers) as pool:
        for i, (seed, out) in enumerate(pool.imap_unordered(job, [(a.recipe, n) for n in pick], chunksize=4)):
            made += [(seed, fn, spec) for fn, spec in out]
            if (i + 1) % 300 == 0:
                print(f"[synth_build] {a.recipe}: {i + 1}/{len(pick)} seeds", flush=True)
    made.sort(key=lambda t: t[1])
    names, T, M, Dt = [], [], [], []
    man, man_s = [], []
    for seed, fn, spec in made:
        g = lab[seed]
        t = np.full((6, 4), 1e-4, dtype=np.float32)
        for d in range(6):
            t[d, g[d]] = 1.0
        mask = np.asarray(spec["mask"], dtype=np.float32).copy()
        tgt, lv = int(spec["target"]), int(spec["level"])
        if tgt >= 0:
            mask[tgt] = 1.0
            t[tgt] = 1e-4
            t[tgt, 3 if lv == 3 else 0] = 1.0                      # level 9: the row is replaced by the presence target in the loss
        names.append(fn)
        T.append(t / t.sum(-1, keepdims=True))
        M.append(mask)
        Dt.append(1 if lv == 9 else 0)
        base = dict(name=fn, kind="synth", seed=seed, tuple_id="", dim=max(tgt, 0), level=lv, op=a.recipe)
        man.append(dict(base, seed_fold=vfold[seed]))
        man_s.append(dict(base, seed_fold=sfold[seed]))
    for path, rr in ((os.path.join(AUX_DIR, f"manifest_{a.recipe}.csv"), man), (os.path.join(AUX_DIR, f"manifest_{a.recipe}_shift.csv"), man_s)):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for r in rr:
                w.writerow({k: r.get(k, "") for k in FIELDS})
    for suf in ("", "_shift"):
        np.savez_compressed(os.path.join(REPO, f"results/r55/aux_teacher_{a.recipe}{suf}.npz"), names=np.array(names),
                            probs_folds=np.stack(T)[None].astype(np.float32), mask=np.stack(M), det=np.array(Dt, dtype=np.int64),
                            towers=np.array(["labels by construction (seed's human labels + recipe)"]))
    od = os.path.join(REPO, f"results/r55/synthlab/{a.recipe}")
    os.makedirs(od, exist_ok=True)
    lv = np.array([int(s["level"]) for _, _, s in made])
    json.dump(dict(recipe=a.recipe, doc=" ".join(R["doc"].split()), seeds=len(pick), frames=len(made), negatives=int((lv == 0).sum()),
                   present_unknown_grade=int((lv == 9).sum()), p0=int((lv == 3).sum())), open(os.path.join(od, "build.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[synth_build] {a.recipe}: {len(pick)} seeds -> {len(made)} frames ({int((lv == 0).sum())} level-0, {int((lv == 9).sum())} present, {int((lv == 3).sum())} P0)")
    if not a.no_meta:
        subprocess.check_call([sys.executable, "code/r55/prep_aux.py", "--gpus", "8", "--steps", "template", "meta",
                               "--manifest", f"data/r55_aux/manifest_{a.recipe}.csv", "--suffix", f"_{a.recipe}", "--tag", f"r55{a.recipe}"],
                              cwd=REPO, env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "code")))


if __name__ == "__main__":
    main()
