#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/build_submission55.py — CPU, seconds.

Builds an official submission JSON *only* from fresh tower outputs + the frozen fusion/decoder
in results/next55/stack/<stack-tag>/final.pkl.  Never reads any earlier submission JSON
(AGENTS.md rule 1) and never looks at test labels or test-set statistics (rule 2).

Usage:
  python code/next55/build_submission55.py --stack-tag s2 --target t150 \
      --template results/v2_full/template_150.json --out results/final/pred_nriqa_dataset_test_next55_s2.json
  python code/next55/build_submission55.py --stack-tag s2 --target final654 \
      --template <final.json> --out results/final/pred_final654_next55_s2.json
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common55 import DIMS, OUT_DIR, REPO, discover_towers, format_official_submission, fullfit_dir, load_target_folds, load_template_names, stack_maps  # noqa: E402
from decoder55 import Decoder55  # noqa: E402
from stack55 import LRStacker, fuse_apply  # noqa: E402,F401  (LRStacker needed for unpickling)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack-tag", required=True)
    ap.add_argument("--target", required=True, help="t150 or the ext tag used at inference time (e.g. final654)")
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold-agg", default="median_vote", choices=["median_vote", "mean_prob"],
                    help="median_vote: decode each fold-k ensemble with the OOF-fitted decoder (same calibration regime "
                         "as OOF: single model per tower) then take the per-cell ordinal median over k. "
                         "mean_prob: average the 8 fold models first (smoother -> under-predicts rare grades)")
    ap.add_argument("--deploy", default="8fold", choices=["8fold", "fullfit"],
                    help="8fold: the 8 fold models of every tower (research / t150 A-B only — exceeds the 30B cap). "
                         "fullfit: the single 100%%-data refit per tower (results/next55/fullfit) = the shippable model.")
    a = ap.parse_args()

    cfg = pickle.load(open(os.path.join(OUT_DIR, "stack", a.stack_tag, "final.pkl"), "rb"))
    names = load_template_names(a.template)
    disc = discover_towers()
    per_tower, missing = [], []
    for t in cfg["towers"]:
        if a.deploy == "fullfit":
            fmt, d = fullfit_dir(t)
            fl = load_target_folds(fmt, d, a.target, min_parts=1) if os.path.isdir(d) else None
        else:
            fl = load_target_folds(*disc[t], a.target) if t in disc else None
        if fl is None or any(n not in f for f in fl for n in names):
            missing.append(t)
        per_tower.append(fl)
    assert not missing, (f"towers without '{a.target}' predictions for every template image: {missing}. "
                         f"Run infer_tier1_ckpt.py / infer_v16_ckpt.py, or re-run stack55.py with --towers excluding them.")

    dec = Decoder55.from_state(cfg["decoder"])
    if a.fold_agg == "mean_prob":
        maps = [{n: np.mean([f[n] for f in fl], axis=0) for n in names} for fl in per_tower]
        pred = dec.predict(fuse_apply(cfg["fusion"], cfg["model"], stack_maps(maps, names)))
    else:
        K = min(len(fl) for fl in per_tower)
        votes = np.stack([dec.predict(fuse_apply(cfg["fusion"], cfg["model"], stack_maps([fl[k] for fl in per_tower], names)))
                          for k in range(K)])                       # (K, N, 6) ordinal labels 0..3
        pred = np.sort(votes, axis=0)[(K - 1) // 2]                  # lower median (ties -> milder grade)
    sub, _ = format_official_submission(names, pred)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(sub, open(a.out, "w"), ensure_ascii=False, indent=2)

    n_clean = int((pred == 0).all(1).sum())
    print(f"[build55] {a.out}\n  towers={cfg['towers']} fusion={cfg['fusion']} objective={cfg['decoder']['objective']} fold_agg={a.fold_agg} deploy={a.deploy}\n"
          f"  N={len(names)} predicted-clean={n_clean} any-P0={int((pred == 3).any(1).sum())}")
    for d, nm in enumerate(DIMS):
        c = np.bincount(pred[:, d], minlength=4)
        print(f"  {nm:6s} 无={c[0]:4d} P2={c[1]:4d} P1={c[2]:4d} P0={c[3]:4d}")
    r = subprocess.run([sys.executable, os.path.join(REPO, "code/qingyu/validate_submission.py"), "--pred", a.out, "--template", a.template],
                       cwd=REPO, env=dict(os.environ, PYTHONPATH=os.path.join(REPO, "code")))
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
