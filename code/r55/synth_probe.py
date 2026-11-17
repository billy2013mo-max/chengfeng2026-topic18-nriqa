#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/synth_probe.py — SynthLab stage 0: behavioural tests.  What do the models NOT respond to?

Controlled stimuli: every operator of synth_ops.SUITE is applied, at graded strengths, to clean labelled frames.
A frame is only ever shown to models that did not train on it: the stimuli made from the clean validation images of
fold f are scored by the fold-f checkpoints (three Tier-1 towers; their mean is the response).  Nothing is trained.

Reported per condition (results/r55/synthlab/probe/<name>.md / .json):
  P(target)        mean probability that the target defect is present (1 - P(clean)) on the stimulus
  dP               the same minus the response to the untouched frame (how much the operator moves the model)
  called           share of stimuli whose argmax grade for the target is >= P2
  E[grade]         mean expected grade (0-3) of the target
  monotone         share of frames whose expected grade never decreases along the strength ladder of an operator
  cross-talk       mean dP on the five other defects (should be small, except 模糊 <-> 纹理损失)
  benign rows      P(any defect) on harmless operations — every point above the untouched level is a false alarm risk
A weak dP at visible strengths, a non-monotone ladder, or a strong response to a benign operation is a gap that
synthetic data can be aimed at; a strong, monotone response means the models already have that sensitivity and
more data of that kind cannot help (that is why the ranking ladders failed).

Usage:
  python code/r55/synth_probe.py --build --per-fold 30 --workers 32
  for f in 0..7: python code/r55/synth_probe.py --run --fold f --gpu g &
  python code/r55/synth_probe.py --report
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import AUX_DIR, REPO  # noqa: E402
from synth_ops import OPS, SUITE  # noqa: E402

PDIR = os.path.join(AUX_DIR, "probe")
OUT = os.path.join(REPO, "results/r55/synthlab/probe")
TOWERS = [("pe_core_large", "results/next55/towers/r55_fixed_s2026/pe_core_large"), ("cradio_v4_h", "results/qingyu/cradio_v4_h"),
          ("siglip2_so400m_naflex", "results/qingyu/siglip2_so400m_naflex")]
DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]


def find(name):
    for d in ("data/images", "data/images_supp"):
        p = os.path.join(REPO, d, name)
        if os.path.exists(p):
            return p


def seeds(per_fold, seed=0):
    rs = np.random.RandomState(seed)
    out = []
    for f in range(8):
        rows = [r for r in csv.DictReader(open(os.path.join(REPO, f"data/qingyu_splits/fold_{f}.csv"), encoding="utf-8")) if r["split"] == "val"]
        clean = sorted(r["name"] for r in rows if all(int(r[d]) == 0 for d in DIMS))
        pick = [clean[i] for i in sorted(rs.choice(len(clean), min(per_fold, len(clean)), replace=False))]
        out += [(f, n) for n in pick]
    return out


def make(job):
    f, n = job
    im = cv2.imdecode(np.fromfile(find(n), dtype=np.uint8), cv2.IMREAD_COLOR)
    stem = os.path.splitext(n)[0]
    rows = []
    for cond, tgt, op, kw in [("none", -1, None, {})] + SUITE:
        out = im if op is None else OPS[op](im, **kw)
        name = f"r55probe_{cond}__{stem}.png"
        cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tofile(os.path.join(PDIR, name))
        rows.append(dict(name=name, fold=f, seed=n, cond=cond, target=tgt))
    return rows


def build(a):
    os.makedirs(PDIR, exist_ok=True)
    jobs = seeds(a.per_fold)
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(make, jobs, chunksize=2):
            rows += r
    rows.sort(key=lambda r: (r["fold"], r["seed"], r["cond"]))
    with open(os.path.join(PDIR, "stimuli.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "fold", "seed", "cond", "target"])
        w.writeheader()
        w.writerows(rows)
    print(f"[probe] {len(jobs)} clean held-out frames x {len(SUITE) + 1} conditions = {len(rows)} stimuli in {PDIR}")


def run(a):
    import torch
    from qingyu.backbones_qingyu import QingyuVisionQualityTower
    from qingyu.config_and_paths import REPO_ROOT, LosslessNativeImageLoader, setup_sys_path_and_cwd
    from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM, DualEnginePhysicsExtractor, PrecomputedDualEngineStore
    from qingyu.train_tier1_5backbones_8fold import make_tier1_collate_fn, predict_2x_tta
    setup_sys_path_and_cwd(chdir_repo=True)
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)
    rows = [r for r in csv.DictReader(open(os.path.join(PDIR, "stimuli.csv"), encoding="utf-8")) if int(r["fold"]) == a.fold]
    items = [{"name": r["name"], "group_id": r["name"], "grades": [0] * 6} for r in rows]
    loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=[PDIR])
    store = PrecomputedDualEngineStore("__none__.npz", lazy_extractor=DualEnginePhysicsExtractor(device=str(dev), allow_depth_download=False))
    acc = []
    for bb, d in TOWERS:
        model = QingyuVisionQualityTower(backbone_key=bb, phys_dim=PHYS_FEAT_DIM).to(dev)
        _, unexpected = model.load_state_dict(torch.load(os.path.join(REPO, d, f"fold_{a.fold}_best.pt"), map_location="cpu"), strict=False)
        assert not unexpected, unexpected[:5]
        collate = make_tier1_collate_fn(model.mode_tag, naflex_processor=getattr(model, "naflex_processor", None), max_num_patches=784)
        p, _, _, pn = predict_2x_tta(model, items, loader, store, collate, dev, 8, 0)      # workers=0: the lazy meta extractor lives on the GPU
        assert list(map(str, pn)) == [r["name"] for r in rows]
        acc.append(p)
        del model
        torch.cuda.empty_cache()
        print(f"[probe] fold {a.fold}: {bb} done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"pred_fold{a.fold}.npz"), names=np.array([r["name"] for r in rows]),
                        probs=np.stack(acc).astype(np.float32), towers=np.array([t[0] for t in TOWERS]))


def report(a):
    rows = list(csv.DictReader(open(os.path.join(PDIR, "stimuli.csv"), encoding="utf-8")))
    P = {}
    for f in range(8):
        fp = os.path.join(OUT, f"pred_fold{f}.npz")
        if os.path.exists(fp):
            z = np.load(fp, allow_pickle=True)
            m = z["probs"].astype(np.float64).mean(0)
            m = m / m.sum(-1, keepdims=True)
            P.update({str(n): q for n, q in zip(z["names"], m)})
    rows = [r for r in rows if r["name"] in P]
    by = {}
    for r in rows:
        by.setdefault(r["seed"], {})[r["cond"]] = P[r["name"]]
    seeds_ = [s for s in by if "none" in by[s]]
    base = np.stack([by[s]["none"] for s in seeds_])
    pres = lambda p: 1.0 - p[..., 0]                                   # noqa: E731
    eg = lambda p: (p * np.arange(4)).sum(-1)                           # noqa: E731
    rep = {"n_frames": len(seeds_), "conditions": {}}
    lines = [f"# SynthLab behavioural probe — {len(seeds_)} clean held-out frames, mean of {len(TOWERS)} Tier-1 towers (fold models that never saw the frame)", "",
             f"Untouched frames: P(defect) " + "  ".join(f"{DIMS[d]} {pres(base)[:, d].mean():.3f}" for d in range(6))
             + f" | any defect called {(base.argmax(-1) > 0).any(1).mean():.3f}", "",
             "| condition | target | P(target) | dP | called >= P2 | E[grade] | largest cross-talk dP | any defect called |", "|---|---|---|---|---|---|---|---|"]
    for cond, tgt, op, kw in SUITE:
        q = np.stack([by[s][cond] for s in seeds_ if cond in by[s]])
        b = np.stack([by[s]["none"] for s in seeds_ if cond in by[s]])
        dp = pres(q).mean(0) - pres(b).mean(0)
        anyc = float((q.argmax(-1) > 0).any(1).mean())
        if tgt >= 0:
            others = [d for d in range(6) if d != tgt]
            ct = max(others, key=lambda d: dp[d])
            rec = dict(target=DIMS[tgt], p_target=float(pres(q)[:, tgt].mean()), dp=float(dp[tgt]), called=float((q[:, tgt].argmax(-1) > 0).mean()),
                       e_grade=float(eg(q)[:, tgt].mean()), cross=f"{DIMS[ct]} {dp[ct]:+.3f}", any_called=anyc)
            lines.append(f"| {cond} | {DIMS[tgt]} | {rec['p_target']:.3f} | {rec['dp']:+.3f} | {rec['called']:.3f} | {rec['e_grade']:.2f} | {rec['cross']} | {anyc:.3f} |")
        else:
            ct = int(np.argmax(dp))
            rec = dict(target="none", dp_max=float(dp[ct]), cross=f"{DIMS[ct]} {dp[ct]:+.3f}", any_called=anyc)
            lines.append(f"| {cond} | (benign) | - | - | - | - | {rec['cross']} | {anyc:.3f} |")
        rep["conditions"][cond] = rec
    # monotonicity along each ladder
    lines += ["", "| ladder | target | frames with non-decreasing E[grade] | mean E[grade] along the ladder |", "|---|---|---|---|"]
    ladders = {}
    for cond, tgt, op, kw in SUITE:
        if tgt >= 0:
            ladders.setdefault(cond.rsplit("_", 1)[0], (tgt, []))[1].append(cond)
    for lad, (tgt, conds) in ladders.items():
        ok, curve = [], []
        for s in seeds_:
            if all(cn in by[s] for cn in conds):
                v = [eg(by[s]["none"])[tgt]] + [eg(by[s][cn])[tgt] for cn in conds]
                ok.append(all(v[i + 1] >= v[i] - 0.02 for i in range(len(v) - 1)))
                curve.append(v)
        curve = np.mean(curve, 0)
        rep.setdefault("ladders", {})[lad] = dict(target=DIMS[tgt], monotone=float(np.mean(ok)), curve=[float(x) for x in curve])
        lines.append(f"| {lad} | {DIMS[tgt]} | {np.mean(ok):.3f} | " + " → ".join(f"{x:.2f}" for x in curve) + " |")
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, f"{a.name}.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(rep, open(os.path.join(OUT, f"{a.name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--per-fold", type=int, default=30)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--name", default="tier1_baseline")
    args = ap.parse_args()
    if args.build:
        build(args)
    if args.run:
        run(args)
    if args.report:
        report(args)
