#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/synth_fidelity.py — SynthLab stage 2: what do models trained on REAL data make of a recipe's frames? (1 GPU)

For every synthetic frame the prediction is taken from the fold model that did not train on its seed (the eight
`r55_fixed_s2026` pe_core_large checkpoints; for seeds outside the labelled set, fold 0).  Reported:

  recipes with a target defect   P(target present), share called >= P2 and expected grade on the positives and on their
                                 matched negatives; the same for the seeds' own out-of-fold predictions; cross-talk
  label-preserving recipes       how much the prediction moves when nothing that matters was changed: total-variation
                                 distance to the seed's out-of-fold prediction, share of frames whose grade changes in
                                 any defect, and how often a clean seed becomes "defective" (false alarm created by the
                                 operation alone)

Reading: a positive that real-trained models do not recognise is either a different phenomenon from the labelled defect
(the FLUX 过度生成 ladder: P 0.16-0.25) or a case the labelled data does not cover (花屏 from another source: recall 0.36).
The two are told apart by the screen on the unseen-source folds, not here.  A positive that is already recognised
with certainty carries no new information (blur / denoise / USM ladders).

Usage:  python code/r55/synth_fidelity.py --recipe benign1 --gpu 7
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402

DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
TOWER = "results/next55/towers/r55_fixed_s2026/pe_core_large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-infer", action="store_true")
    a = ap.parse_args()
    tag = f"r55syn{a.recipe}"
    out_npz = os.path.join(REPO, TOWER, f"ext_{tag}_probs.npz")
    if not (a.skip_infer and os.path.exists(out_npz)):
        subprocess.check_call([sys.executable, "-u", "code/next55/infer_tier1_ckpt.py", "--backbone", "pe_core_large", "--tower-dir", TOWER,
                               "--template", f"data/r55_aux/template_{a.recipe}.json", "--img-dir", "data/r55_aux/images", "--tag", tag,
                               "--gpu", str(a.gpu), "--dual-engine-npz", f"results/r55/dual_engine_meta_aux_{a.recipe}.npz"],
                              cwd=REPO, env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "code")))
    z = np.load(out_npz, allow_pickle=True)
    pos = {str(n): i for i, n in enumerate(z["names"])}
    PF = z["probs_6x4_folds"].astype(np.float64)
    man = list(csv.DictReader(open(os.path.join(REPO, f"data/r55_aux/manifest_{a.recipe}.csv"), encoding="utf-8")))
    names, gold, _, _ = c.load_split()
    gi = {n: i for i, n in enumerate(names)}
    seed_oof = {}
    for f in range(8):
        s = np.load(os.path.join(REPO, TOWER, f"fold_{f}_probs.npz"), allow_pickle=True)
        seed_oof.update({str(n): q.astype(np.float64) for n, q in zip(s["va_names"], s["va_probs_6x4"])})
    rn = lambda p: p / p.sum(-1, keepdims=True)                       # noqa: E731
    rows = []
    for r in man:
        if r["name"] not in pos:
            continue
        f = int(r["seed_fold"]) if 0 <= int(r["seed_fold"]) < 8 else 0
        rows.append((r, rn(PF[f, pos[r["name"]]]), rn(seed_oof[r["seed"]]) if r["seed"] in seed_oof else None))
    rep = dict(recipe=a.recipe, n=len(rows))
    print(f"=== SynthLab fidelity | recipe {a.recipe} | {len(rows)} frames | models: pe_core_large fold checkpoints that never saw the seed ===")
    levels = sorted({int(r["level"]) for r, _, _ in rows})
    if levels == [0] and all(int(r["dim"]) == 0 for r, _, _ in rows):
        q = np.stack([p for _, p, s in rows if s is not None])
        s = np.stack([s for _, p, s in rows if s is not None])
        g = np.stack([gold[gi[r["seed"]]] for r, p, sd in rows if sd is not None])
        tv = 0.5 * np.abs(q - s).sum(-1)
        flip = (q.argmax(-1) != s.argmax(-1))
        clean = (g == 0).all(1)
        fa_seed = (s[clean].argmax(-1) > 0).any(1).mean()
        fa_syn = (q[clean].argmax(-1) > 0).any(1).mean()
        acc_seed = (s.argmax(-1) == g).mean()
        acc_syn = (q.argmax(-1) == g).mean()
        print(f"  label-preserving recipe: mean TV to the seed's own prediction {tv.mean():.4f} (per defect " + " ".join(f"{DIMS[d]} {tv[:, d].mean():.3f}" for d in range(6)) + ")")
        print(f"  frames whose predicted grade changes in some defect: {flip.any(1).mean() * 100:.1f} % | grade accuracy vs human labels: seed {acc_seed:.4f} -> processed {acc_syn:.4f}")
        print(f"  clean seeds (n={int(clean.sum())}): flagged before {fa_seed * 100:.1f} % -> after {fa_syn * 100:.1f} %  | defect mass moved: " +
              " ".join(f"{DIMS[d]} {((1 - q[:, d, 0]) - (1 - s[:, d, 0])).mean():+.3f}" for d in range(6)))
        rep.update(kind="label_preserving", tv=float(tv.mean()), flip_any=float(flip.any(1).mean()), acc_seed=float(acc_seed), acc_syn=float(acc_syn),
                   clean_flagged_seed=float(fa_seed), clean_flagged_syn=float(fa_syn))
    else:
        print(f"  {'target':6s} level     n  P(present)  called>=P2  E[grade] | seed: P(present)  called | largest cross-talk")
        rep["levels"] = {}
        for d in sorted({int(r["dim"]) for r, _, _ in rows}):
            for lv in levels:
                sel = [(p, s) for r, p, s in rows if int(r["dim"]) == d and int(r["level"]) == lv]
                if not sel:
                    continue
                q = np.stack([p for p, _ in sel])
                s = np.stack([s for _, s in sel if s is not None]) if any(s is not None for _, s in sel) else None
                pres = 1 - q[:, d, 0]
                ct = None
                if s is not None and len(s) == len(q):
                    dp = (1 - q[:, :, 0]).mean(0) - (1 - s[:, :, 0]).mean(0)
                    o = max([k for k in range(6) if k != d], key=lambda k: abs(dp[k]))
                    ct = f"{DIMS[o]} {dp[o]:+.3f}"
                print(f"  {DIMS[d]:6s} {lv:5d} {len(q):5d}   {pres.mean():.3f}      {(q[:, d].argmax(-1) > 0).mean():.3f}     {(q[:, d] * np.arange(4)).sum(-1).mean():.2f}   | "
                      + (f"{(1 - s[:, d, 0]).mean():.3f}        {(s[:, d].argmax(-1) > 0).mean():.3f}" if s is not None else "   -            -  ") + f"  | {ct or '-'}")
                rep["levels"][f"{DIMS[d]}_L{lv}"] = dict(n=len(q), p_present=float(pres.mean()), called=float((q[:, d].argmax(-1) > 0).mean()))
    od = os.path.join(REPO, f"results/r55/synthlab/{a.recipe}")
    os.makedirs(od, exist_ok=True)
    json.dump(rep, open(os.path.join(od, "fidelity.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
