#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/gate_r55.py — paired acceptance gate for one experiment against its baseline (CPU, ~2 min).

Both sides are lists of tower directories (one per seed).  Seeds are mean-pooled, both sides are decoded with the
same nested leave-one-fold-out decoder (objective Doc2_Strict, the metric the scoring site uses), and compared on
the SAME images with a paired bootstrap.  Works on any subset of folds present on both sides (>= 2).

Single seeds of the same recipe differ by ~0.4 Doc2S (pe_core_large: 49.30 vs 48.91), so:
  * a screen on one seed is only a hint;
  * the PASS verdict requires >= 2 seeds on each side;
  * `--noise` (>= 3 baseline seeds) measures the seed-noise floor of a two-seed baseline and prints the
    --min-delta every later gate must use.

PASS = delta Doc2S >= --min-delta (0.30)  and  P(delta <= 0) <= 0.20  and  delta S_p0 >= -0.10  and
       delta S_clean >= -0.10.            Per-label F1 deltas are printed so one dimension cannot hide another.

Usage:
  python code/r55/gate_r55.py --name rank \
     --base results/qingyu/pe_core_large results/next55/towers/qy_s2027/pe_core_large \
     --exp  results/next55/towers/r55_rank_s2026/pe_core_large results/next55/towers/r55_rank_s2027/pe_core_large
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "code/next55"))
import common55 as c  # noqa: E402
from decoder55 import nested_oof_decode  # noqa: E402

OBJ = "doc2_strict_total"
LABELS = [(0, "模糊"), (1, "纹理损失"), (2, "过锐"), (4, "过度生成")]


def load_dir(d, folds):
    m = {}
    for f in folds:
        z = np.load(os.path.join(REPO, d, f"fold_{f}_probs.npz"), allow_pickle=True)
        m.update({str(n): p for n, p in zip(z["va_names"], z["va_probs_6x4"])})
    return m


def folds_of(d):
    return {f for f in range(8) if os.path.exists(os.path.join(REPO, d, f"fold_{f}_probs.npz"))}


def per_label_f1(pred, gold):
    out = {}
    for j, name in LABELS:
        for g, tag in ((1, "P2"), (2, "P1"), (3, "P0")):
            tp = ((pred[:, j] == g) & (gold[:, j] == g)).sum()
            out[f"{name}_{tag}"] = 2 * tp / max(1, (pred[:, j] == g).sum() + (gold[:, j] == g).sum())
    for j, name in ((3, "花屏"), (5, "虚化突兀")):
        tp = ((pred[:, j] == 3) & (gold[:, j] == 3)).sum()
        out[f"{name}_P0"] = 2 * tp / max(1, (pred[:, j] == 3).sum() + (gold[:, j] == 3).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--exp", nargs="*", default=[])
    ap.add_argument("--min-delta", type=float, default=0.30)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--noise", action="store_true",
                    help="seed-noise floor: score every 2-seed subset of --base (needs >= 3 base seeds), then exit")
    a = ap.parse_args()

    folds = sorted(set.intersection(*[folds_of(d) for d in a.base + a.exp]))
    assert len(folds) >= 2, f"need >= 2 common folds, have {folds}"
    names, gold, fold, _ = c.load_split()
    keep = np.isin(fold, folds)
    names = [n for n, k in zip(names, keep) if k]
    gold, fold = gold[keep], fold[keep]

    def side(dirs):
        stack = [np.stack([load_dir(d, folds)[n] for n in names]) for d in dirs]
        s = c.mean_pool(stack)
        pred = nested_oof_decode(s, gold, fold, objective=OBJ)
        return pred[0] if isinstance(pred, tuple) else pred

    if a.noise:
        from itertools import combinations
        assert len(a.base) >= 3, "--noise needs >= 3 baseline seeds"
        sc = [c.score(side(list(p)), gold)[OBJ] for p in combinations(a.base, 2)]
        sd = float(np.std(sc, ddof=1))
        print(f"=== seed-noise floor from {len(sc)} two-seed baselines: " + " ".join(f"{x:.2f}" for x in sc))
        print(f"  mean {np.mean(sc):.2f}  sd {sd:.2f}  range {max(sc) - min(sc):.2f}")
        print(f"  => run every gate with --min-delta {max(0.30, 2 * sd):.2f}  (max of 0.30 and 2 x sd)")
        return

    pb, pe = side(a.base), side(a.exp)
    eb, ee = c.score(pb, gold), c.score(pe, gold)
    keys = [("doc2_strict_total", "Doc2S"), ("doc2_S_pos", "S_pos"), ("doc2_S_p0", "S_p0"), ("doc2_Clean2", "S_clean")]
    print(f"=== gate '{a.name}' | folds {folds} | N={len(names)} | seeds base={len(a.base)} exp={len(a.exp)} ===")
    for k, lab in keys:
        print(f"  {lab:8s} base {eb[k]:6.2f}   exp {ee[k]:6.2f}   delta {ee[k] - eb[k]:+.2f}")

    rs = np.random.RandomState(0)
    deltas = []
    for _ in range(a.bootstrap):
        i = rs.randint(0, len(names), len(names))
        deltas.append(c.score(pe[i], gold[i])[OBJ] - c.score(pb[i], gold[i])[OBJ])
    deltas = np.array(deltas)
    p_le0 = float((deltas <= 0).mean())
    lo, hi = np.quantile(deltas, [0.05, 0.95])
    print(f"  paired bootstrap: delta {deltas.mean():+.3f}  90% CI [{lo:+.3f}, {hi:+.3f}]  P(delta<=0)={p_le0:.2f}")

    fb, fe = per_label_f1(pb, gold), per_label_f1(pe, gold)
    print("  per-label F1 delta: " + "  ".join(f"{k} {fe[k] - fb[k]:+.3f}" for k in fb))

    d = ee[OBJ] - eb[OBJ]
    checks = {
        f"delta >= {a.min_delta:.2f}": d >= a.min_delta,
        "P(delta<=0) <= 0.20": p_le0 <= 0.20,
        "S_p0 not down > 0.10": ee["doc2_S_p0"] - eb["doc2_S_p0"] >= -0.10,
        "S_clean not down > 0.10": ee["doc2_Clean2"] - eb["doc2_Clean2"] >= -0.10,
        ">= 2 seeds per side": min(len(a.base), len(a.exp)) >= 2,
        "all 8 folds": len(folds) == 8,
    }
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'NO'}] {k}")
    verdict = "PASS" if all(checks.values()) else ("SCREEN-ONLY" if all(list(checks.values())[:4]) else "FAIL")
    print(f"  VERDICT: {verdict}")
    out = os.path.join(REPO, "results/r55/gates")
    os.makedirs(out, exist_ok=True)
    json.dump(dict(name=a.name, base=a.base, exp=a.exp, folds=folds, base_score={k: eb[k] for k, _ in keys},
                   exp_score={k: ee[k] for k, _ in keys}, delta=d, ci90=[float(lo), float(hi)], p_le0=p_le0,
                   per_label_f1_delta={k: float(fe[k] - fb[k]) for k in fb}, verdict=verdict),
              open(os.path.join(out, f"{a.name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
