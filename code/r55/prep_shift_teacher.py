#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/prep_shift_teacher.py — soft labels for the KVQ pool from the SHIFT-fold models (1 GPU).

`shiftkvq` asks: does self-training on unlabelled frames from many processing pipelines shrink the gap to a pipeline
that is absent from the labelled training data?  For that test the teacher of fold f must itself never have seen
family f, so the 8-fold teachers of prep_aux.py cannot be used.  Here the teacher of fold f is the `shift` model of
fold f (results/next55/towers/r55_shift_s2026/<backbone>, trained on data/r55_tagsplit without family f).

Output: results/r55/aux_teacher_kvq_shift.npz  (names, probs_folds (7, N, 6, 4)) — same layout as aux_teacher_kvq.npz.

Usage:  python code/r55/prep_shift_teacher.py --gpu 1
"""
import argparse
import os
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--backbone", default="pe_core_large")
    ap.add_argument("--tower", default="results/next55/towers/r55_shift_s2026")
    ap.add_argument("--tag", default="r55kvqshift")
    a = ap.parse_args()
    tower = os.path.join(a.tower, a.backbone)
    assert all(os.path.exists(os.path.join(REPO, tower, f"fold_{f}_best.pt")) for f in range(7)), "shift folds 0-6 are not all trained"
    subprocess.check_call([sys.executable, "-u", "code/next55/infer_tier1_ckpt.py", "--backbone", a.backbone, "--tower-dir", tower,
                           "--template", "data/r55_aux/template_kvq.json", "--img-dir", "data/r55_aux/images", "--tag", a.tag,
                           "--gpu", str(a.gpu), "--folds", *[str(f) for f in range(7)],
                           "--dual-engine-npz", "results/r55/dual_engine_meta_aux_kvq.npz"],
                          cwd=REPO, env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "code")))
    z = np.load(os.path.join(REPO, tower, f"ext_{a.tag}_probs.npz"), allow_pickle=True)
    p = np.clip(z["probs_6x4_folds"].astype(np.float64), 1e-6, 1.0)
    assert p.shape[0] == 7
    p[:, :, [3, 5], 1:3] = 0.0                               # 花屏 / 虚化突兀 have only {无问题, P0}
    p = p / p.sum(-1, keepdims=True)
    out = os.path.join(REPO, "results/r55/aux_teacher_kvq_shift.npz")
    np.savez_compressed(out, names=z["names"], probs_folds=p.astype(np.float32), towers=np.array([tower]))
    print(f"[prep_shift_teacher] {p.shape} -> {out}")


if __name__ == "__main__":
    main()
