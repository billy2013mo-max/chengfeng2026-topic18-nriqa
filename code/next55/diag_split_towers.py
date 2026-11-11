#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/diag_split_towers.py  — CPU, ~5-15 min

The legacy InternViT-6B (v11 / v12) and Qwen3-VL-8B (v13) towers only exist on the s16..s20
random splits, not on the 8-fold.  Their `te_*` predictions are still genuinely out-of-train,
and the union of te sets covers ~2.9k / 3.57k images.  On that common subset this script
answers "is a 6B / VLM tower worth re-training on the 8-fold?" by measuring its marginal
gain on top of the 8-fold Tier-1 ensemble with the same nested decoder.

Usage: python code/next55/diag_split_towers.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common55 import REPO, build_bank, load_split, logit_pool, p012_to_6x4_simplex, renorm, score, score_line, stack_oof  # noqa: E402
from decoder55 import nested_oof_decode  # noqa: E402

FAMS = {
    "v12fix_iv6b_abmil": "results/v12_phaseB_fixed/v12_abmil_s{}_probs.npz",
    "v11_iv6b": "results/v11_pure_real/iv6b_s{}_probs.npz",
    "v13_qwen3vl8b": "results/v13_vlm_real/vlm_corn_s{}_probs.npz",
}


def load_te_union(pattern):
    acc = {}
    for s in range(16, 21):
        p = os.path.join(REPO, pattern.format(s))
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        S = p012_to_6x4_simplex(d["te_p0"], d["te_p1"], d["te_p2"])
        for n, v in zip(map(str, d["te_names"]), S):
            acc.setdefault(n, []).append(v)
    return {n: np.mean(v, 0) for n, v in acc.items()}


def main():
    names, gold, fold, _ = load_split()
    te_maps = {k: load_te_union(p) for k, p in FAMS.items()}
    idx = [i for i, n in enumerate(names) if all(n in m for m in te_maps.values())]
    nm = [names[i] for i in idx]
    g, f = gold[idx], fold[idx]
    print(f"[split-towers] common subset N={len(idx)} (clean={(g == 0).all(1).sum()})")

    bank = build_bank()
    keys = sorted(bank)
    S = stack_oof(bank, keys, nm)
    towers = {k: S[i] for i, k in enumerate(keys)}
    for k, m in te_maps.items():
        towers[k] = renorm(np.stack([m[n] for n in nm]))

    def rep(tag, s):
        e = score(nested_oof_decode(s, g, f, "blend", sweeps=2), g)
        print(f"  {tag:34s} {score_line(e)}")
        return e

    for k, s in towers.items():
        rep(k, s)
    base = logit_pool([towers[k] for k in keys])
    rep("8FOLD_ENS", base)
    for k in FAMS:
        rep(f"8FOLD_ENS + {k}", logit_pool([base, towers[k]], [len(keys), 2.0]))
    rep("8FOLD_ENS + v12fix + v13", logit_pool([base, towers["v12fix_iv6b_abmil"], towers["v13_qwen3vl8b"]], [len(keys), 2.0, 2.0]))


if __name__ == "__main__":
    main()
