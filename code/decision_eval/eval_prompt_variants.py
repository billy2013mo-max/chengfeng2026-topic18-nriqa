"""Score the prompt-variant study (run_http_prompt_variants.py) against the ensemble on the same pairs.  CPU.
Usage: python code/decision_eval/eval_prompt_variants.py --stage dev --name gemini4argon --pool <pool.npz> [--variants v1 v2 v3 v4]
For heldout, the original-prompt run of the pilot (pairs_<name>.jsonl) is shown next to the picked variant.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
from probe_eval import DIMS  # noqa: E402

D = os.path.join(REPO, "results/r55/explore4/vlm_pilot")
NAMES = {3: "P0 vs P1", 2: "P1 vs P2", 1: "P2 vs none"}


def load(path):
    by = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        by.setdefault((r["hi"], r["img_hi"], r["img_lo"]), {})[r["first"]] = r
    return {k: v for k, v in by.items() if len(v) == 2}


def per_pair(by):
    out = {}
    for k, pr in by.items():
        p_hi = 0.5 * (pr["hi"]["probs"]["A"] + pr["lo"]["probs"]["B"])
        out[k] = (float(p_hi > .5) + .5 * float(p_hi == .5), float(p_hi in (0.0, 1.0)), pr["hi"].get("valid", True) and pr["lo"].get("valid", True))
    return out


def boot(x, B=2000):
    rs = np.random.RandomState(0)
    m = np.array([x[rs.randint(0, len(x), len(x))].mean() for _ in range(B)])
    return x.mean(), np.quantile(m, .05), np.quantile(m, .95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--name", default="gemini4argon")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--variants", nargs="+", default=["v1", "v2", "v3", "v4"])
    a = ap.parse_args()
    z = np.load(a.pool, allow_pickle=True)
    pos = {str(n): i for i, n in enumerate(z["names"])}
    P = z["P"].astype(np.float64)
    P = P / P.sum(-1, keepdims=True)
    EG = (P * np.arange(4)[None, None]).sum(-1)[:, DIMS.index("过度生成")]
    runs = {}
    if a.stage == "heldout":
        runs["original prompt"] = load(os.path.join(D, f"pairs_{a.name}.jsonl"))
    for v in a.variants:
        p = os.path.join(D, f"pv_{a.stage}_{v}_{a.name}.jsonl")
        if os.path.exists(p):
            runs[v] = load(p)
    keys = sorted(set.intersection(*[set(r) for r in runs.values()]))
    ens = {k: float(EG[pos[k[1]]] > EG[pos[k[2]]]) + .5 * float(EG[pos[k[1]]] == EG[pos[k[2]]]) for k in keys}
    print(f"stage {a.stage}: {len(keys)} pairs common to all runs ({', '.join(f'{NAMES[h]} {sum(1 for k in keys if k[0] == h)}' for h in (3, 2, 1))})")
    print(f"{'run':16s} " + " ".join(f"{NAMES[h]:>12s}" for h in (3, 2, 1)) + f" | {'mean':>6s} {'overall [CI90]':>22s} | order-consistent | invalid | right where ensemble wrong")
    rows = {}
    for nm, by in runs.items():
        pp = per_pair({k: by[k] for k in keys})
        cells = [np.mean([pp[k][0] for k in keys if k[0] == h]) for h in (3, 2, 1)]
        allx = np.array([pp[k][0] for k in keys])
        o, lo, hi = boot(allx)
        wrong = [pp[k][0] for k in keys if ens[k] < .5]
        rows[nm] = (np.mean(cells), allx)
        print(f"{nm:16s} " + " ".join(f"{c:12.3f}" for c in cells) + f" | {np.mean(cells):6.3f} {o:7.3f} [{lo:.3f}, {hi:.3f}] | {np.mean([pp[k][1] for k in keys]):16.2f} | {sum(not pp[k][2] for k in keys):7d} | {np.mean(wrong) if wrong else float('nan'):.3f} (n={len(wrong)})")
    ec = [np.mean([ens[k] for k in keys if k[0] == h]) for h in (3, 2, 1)]
    print(f"{'ensemble (OOF)':16s} " + " ".join(f"{c:12.3f}" for c in ec) + f" | {np.mean(ec):6.3f} {np.mean(list(ens.values())):7.3f}")
    best = max(rows, key=lambda r: rows[r][0])
    print(f"\nbest by mean of the three contrasts: {best}")


if __name__ == "__main__":
    main()
