#!/usr/bin/env python3
"""Estimate Bayes RPS noise floor from near-duplicate sibling frames within video groups."""
import os
import sys
from collections import defaultdict
import numpy as np
from PIL import Image

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

import common55 as c
from vlm_c233397_qy8 import PureRealImageResolver


def pair_rps_6x4(y1: np.ndarray, y2: np.ndarray) -> float:
    # y1, y2: (6,) int grades in {0,1,2,3}
    tot = 0.0
    for d in range(6):
        k = np.arange(3)
        c1 = (y1[d] <= k).astype(np.float64)
        c2 = (y2[d] <= k).astype(np.float64)
        tot += np.mean((c1 - c2) ** 2)
    return tot / 6.0


def main():
    names, gold, fold, grp = c.load_split()
    resolver = PureRealImageResolver()

    by_grp = defaultdict(list)
    for i, (nm, g) in enumerate(zip(names, grp)):
        if "_g" not in str(nm):
            by_grp[str(g)].append(i)

    multi_grps = {g: idxs for g, idxs in by_grp.items() if len(idxs) >= 2}
    print(f"Total orig images: {sum(len(v) for v in by_grp.values())} | Groups with >=2 frames: {len(multi_grps)} "
          f"| Frames in multi-frame groups: {sum(len(v) for v in multi_grps.values())}")

    vecs = {}
    for g, idxs in multi_grps.items():
        for i in idxs:
            im = resolver.load_rgb_pil(str(names[i])).resize((128, 128), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32).ravel()
            arr = arr - arr.mean()
            norm = np.linalg.norm(arr) + 1e-8
            vecs[i] = arr / norm

    pairs = []
    for g, idxs in multi_grps.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                cos = float(np.dot(vecs[i], vecs[j]))
                rps = pair_rps_6x4(gold[i], gold[j])
                cln_i = int((gold[i] == 0).all())
                cln_j = int((gold[j] == 0).all())
                exact6 = int((gold[i] == gold[j]).all())
                exact_dim = float(np.mean(gold[i] == gold[j]))
                pairs.append((cos, rps, cln_i == cln_j, exact6, exact_dim))

    print(f"Total sibling pairs: {len(pairs)}")
    for thr_name, thr in [("all_siblings", -2.0), ("cos>=0.90", 0.90), ("cos>=0.95", 0.95), ("cos>=0.98", 0.98), ("cos>=0.99", 0.99)]:
        sub = [p for p in pairs if p[0] >= thr]
        if not sub:
            continue
        rps_m = float(np.mean([p[1] for p in sub]))
        bayes_rps = 0.5 * rps_m
        cln_ag = float(np.mean([p[2] for p in sub]))
        ex6 = float(np.mean([p[3] for p in sub]))
        exd = float(np.mean([p[4] for p in sub]))
        print(f"  {thr_name:14s} (n={len(sub):4d}) | pair_RPS={rps_m:.5f} | est_Bayes_RPS={bayes_rps:.5f} "
              f"| clean_agree={cln_ag:.3f} | exact6={ex6:.3f} | dim_exact={exd:.3f}")


if __name__ == "__main__":
    main()
