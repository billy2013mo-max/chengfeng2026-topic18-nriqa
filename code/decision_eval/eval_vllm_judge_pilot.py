"""Score the VLM judge / in-context grading pilot (run_vllm_judge_pilot.py) against the ensemble's out-of-fold
predictions on exactly the same frames.  CPU.

Usage: python code/decision_eval/eval_vllm_judge_pilot.py --name qwen38-27b --pool <pool.npz with names, P (N,6,4)>
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
from probe_eval import DIMS  # noqa: E402

GN = {"无问题": 0, "P2": 1, "P1": 2, "P0": 3}
CN = {3: "P0 vs P1", 2: "P1 vs P2", 1: "P2 vs none"}


def ci(x, B=2000, seed=0):
    x = np.asarray(x, dtype=float)
    rs = np.random.RandomState(seed)
    m = np.array([x[rs.randint(0, len(x), len(x))].mean() for _ in range(B)])
    return x.mean(), np.quantile(m, .05), np.quantile(m, .95)


def order_auc(s, g):
    """P(score of the higher-grade frame > score of the lower-grade frame), ties count one half."""
    s, g = np.asarray(s, float), np.asarray(g)
    num = den = 0.0
    for a in np.unique(g):
        for b in np.unique(g):
            if a > b:
                x, y = s[g == a][:, None], s[g == b][None]
                num += (x > y).sum() + 0.5 * (x == y).sum()
                den += x.size * y.shape[1]
    return num / max(1.0, den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="qwen38-27b")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--dir", default=os.path.join(REPO, "results/r55/explore4/vlm_pilot"))
    a = ap.parse_args()
    z = np.load(a.pool, allow_pickle=True)
    pos = {str(n): i for i, n in enumerate(z["names"])}
    P = z["P"].astype(np.float64)
    P = P / P.sum(-1, keepdims=True)
    EG = (P * np.arange(4)[None, None]).sum(-1)                      # ensemble expected grade (out of fold), (N, 6)

    # ---------------- pairwise judge
    rows = [json.loads(l) for l in open(os.path.join(a.dir, f"pairs_{a.name}.jsonl"), encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault((r["defect"], r["hi"], r["img_hi"], r["img_lo"]), {})[r["first"]] = r
    print(f"=== pairwise judge ({a.name}): {len(by)} pairs, valid answers {np.mean([r.get('valid', True) for r in rows]) * 100:.1f} %, "
          f"position bias: mean P(answer = first image) {np.mean([r['probs']['A'] for r in rows]):.3f}")
    print(f"{'defect':8s} {'contrast':11s} {'n':>4s} | {'VLM, both orders':>24s} | {'VLM, single order':>17s} | {'ensemble (out of fold)':>24s} | VLM right where the ensemble is wrong")
    allv, alle = [], []
    for defect in ("模糊", "纹理损失", "过锐", "过度生成"):
        d = DIMS.index(defect)
        for hi in (3, 2, 1):
            v, v1, e = [], [], []
            for (df, h, ih, il), pr in by.items():
                if df != defect or h != hi or len(pr) < 2:
                    continue
                p_hi = 0.5 * (pr["hi"]["probs"]["A"] + pr["lo"]["probs"]["B"])
                v.append(float(p_hi > 0.5) + 0.5 * float(p_hi == 0.5))
                v1 += [float(pr["hi"]["probs"]["A"] > 0.5), float(pr["lo"]["probs"]["B"] > 0.5)]
                x, y = EG[pos[ih], d], EG[pos[il], d]
                e.append(float(x > y) + 0.5 * float(x == y))
            v, e = np.array(v), np.array(e)
            allv += list(v)
            alle += list(e)
            cv, ce = ci(v), ci(e)
            w = v[e < 0.5]
            print(f"{defect:8s} {CN[hi]:11s} {len(v):4d} | {cv[0]:.3f} [{cv[1]:.3f}, {cv[2]:.3f}]   | {np.mean(v1):17.3f} | {ce[0]:.3f} [{ce[1]:.3f}, {ce[2]:.3f}]   | {w.mean() if len(w) else float('nan'):.3f} (n={len(w)})")
    cv, ce = ci(allv), ci(alle)
    dd = ci(np.array(allv) - np.array(alle))
    print(f"{'all':8s} {'':11s} {len(allv):4d} | {cv[0]:.3f} [{cv[1]:.3f}, {cv[2]:.3f}]   | {'':17s} | {ce[0]:.3f} [{ce[1]:.3f}, {ce[2]:.3f}]   | VLM - ensemble {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]")

    # ---------------- grading: zero-shot vs four exemplars vs ensemble
    rows = [json.loads(l) for l in open(os.path.join(a.dir, f"grade_{a.name}.jsonl"), encoding="utf-8")]
    print(f"\n=== grade question on a grade-balanced sample ({a.name}); order AUC = P(higher-grade frame gets the higher expected grade)")
    print(f"{'defect':8s} {'n':>4s} | {'zero-shot':>32s} | {'four labelled exemplars':>32s} | {'ensemble (out of fold)':>32s}")
    print(f"{'':8s} {'':4s} | {'order AUC   Spearman  accuracy':>32s} | {'order AUC   Spearman  accuracy':>32s} | {'order AUC   Spearman  accuracy':>32s}")
    for defect in ("模糊", "纹理损失", "过锐", "过度生成"):
        d = DIMS.index(defect)
        out = []
        n = 0
        for mode in ("zero", "few"):
            rr = [r for r in rows if r["defect"] == defect and r["mode"] == mode]
            g = np.array([r["gold_grade"] for r in rr])
            pg = np.zeros((len(rr), 4))
            for k, r in enumerate(rr):
                for L, opt in zip("ABCD", r["options"]):
                    pg[k, GN[opt]] = r["probs"].get(L, 0.0)
            eg = (pg * np.arange(4)[None]).sum(1)
            out.append((order_auc(eg, g), spearmanr(eg, g).correlation, (pg.argmax(1) == g).mean()))
            n = len(rr)
        rr = [r for r in rows if r["defect"] == defect and r["mode"] == "zero"]
        g = np.array([r["gold_grade"] for r in rr])
        idx = np.array([pos[r["image"]] for r in rr])
        out.append((order_auc(EG[idx, d], g), spearmanr(EG[idx, d], g).correlation, (P[idx, d].argmax(1) == g).mean()))
        print(f"{defect:8s} {n:4d} | " + " | ".join(f"{x[0]:9.3f} {x[1]:10.3f} {x[2]:9.3f}  " for x in out))


if __name__ == "__main__":
    main()
