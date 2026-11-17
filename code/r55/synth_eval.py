#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/synth_eval.py — SynthLab gates: one recipe against its control, same folds, same images (CPU).

  screen / confirm   towers r55_synth-<R>_s<seed> vs r55_fixed_s<seed> on the standard folds they share, and
                     r55_shiftsynth-<R>_s<seed> vs r55_shift_s<seed> on the source-held-out folds they share
  pool               (needs all 8 standard folds) the recipe tower in place of `pe` in the primary ensemble

Per defect: RPS (Brier for the P0-only defects), detection AUC, adjacent-grade AUC, mean label F1 at argmax; paired
bootstrap over videos for the RPS difference.  Clean-image false alarms are listed separately for frames with and
without a pipeline tag in the file name (the tag only forms the subgroup; it is never a model input).

Verdict (written to results/r55/synthlab/LEDGER.md, append-only):
  The reference is the MEAN of all control seeds (2026 / 2027 / 2028 where trained).  An effect counts when the paired
  bootstrap interval over videos excludes zero AND it exceeds twice the training-noise standard error estimated from
  the spread of the control seeds (se = sd * sqrt(1/n_ctrl + 1/n_recipe)).
  PASS      a targeted defect (or the mean) improves on the standard or the unseen-source folds, and nothing gets worse
  HARM      some defect or the mean gets worse on either set of folds, or clean false alarms rise by > 1.5 per 100 clean
  NULL      neither
A PASS on two folds is a licence to run all eight (`confirm`), nothing more; adoption needs the `pool` gate.

Usage:
  python code/r55/synth_eval.py --recipe benign1 --targets 0 1 2 --stage screen
  python code/r55/synth_eval.py --recipe benign1 --stage pool
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
from decoder55 import argmax_decode  # noqa: E402
from gate_plus import auc  # noqa: E402
from probe_eval import rps_dim  # noqa: E402

DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
FULL = (0, 1, 2, 4)
TOW = "results/next55/towers"
TAG_RE = re.compile(r"_(v\d+[A-Za-z0-9]*?)(?:\.mp4|_|\.)")
LEDGER = os.path.join(REPO, "results/r55/synthlab/LEDGER.md")


def load_folds(d, n_folds):
    m, have = {}, []
    for f in range(n_folds):
        p = os.path.join(REPO, d, f"fold_{f}_probs.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            for n, q in zip(z["va_names"], z["va_probs_6x4"]):
                m[str(n)] = (f, q)
            have.append(f)
    return m, have


def f1(yp, yt):
    tp = (yp & yt).sum()
    return 2 * tp / max(1, 2 * tp + (yp & ~yt).sum() + (~yp & yt).sum())


def compare(name_a, dir_a, name_b, dir_b, n_folds, names, gold, grp, boot=600):
    A, fa = load_folds(dir_a, n_folds)
    B, fb = load_folds(dir_b, n_folds)
    folds = sorted(set(fa) & set(fb))
    keep = [i for i, n in enumerate(names) if n in A and n in B and A[n][0] in folds]
    if not keep:
        return None
    g = gold[keep]
    pa = c.renorm(np.stack([A[names[i]][1] for i in keep]))
    pb = c.renorm(np.stack([B[names[i]][1] for i in keep]))
    gr = grp[keep]
    ug, inv = np.unique(gr, return_inverse=True)
    mem = [np.where(inv == i)[0] for i in range(len(ug))]
    rs = np.random.RandomState(0)
    boots = [np.concatenate([mem[i] for i in rs.randint(0, len(ug), len(ug))]) for _ in range(boot)]
    out = dict(folds=folds, n=len(keep), dims={})
    pra, prb = argmax_decode(pa), argmax_decode(pb)
    for d in range(6):
        ra, rb = rps_dim(pa[:, d], g[:, d], d), rps_dim(pb[:, d], g[:, d], d)
        dd = np.array([rb[b].mean() - ra[b].mean() for b in boots])
        grades = (1, 2, 3) if d in FULL else (3,)
        rec = dict(rps_ctrl=float(ra.mean()), rps_new=float(rb.mean()), d_rps=float(rb.mean() - ra.mean()),
                   ci90=[float(np.quantile(dd, .05)), float(np.quantile(dd, .95))],
                   auc_det_ctrl=float(auc(1 - pa[:, d, 0], g[:, d] > 0)), auc_det_new=float(auc(1 - pb[:, d, 0], g[:, d] > 0)),
                   f1_ctrl=float(np.mean([f1(pra[:, d] == k, g[:, d] == k) for k in grades])),
                   f1_new=float(np.mean([f1(prb[:, d] == k, g[:, d] == k) for k in grades])))
        out["dims"][DIMS[d]] = rec
    clean = (g == 0).all(1)
    nm = [names[i] for i in keep]
    proc = np.array([bool(TAG_RE.search(n)) for n in nm])
    for lab, m in (("tagged", clean & proc), ("untagged", clean & ~proc)):
        out[f"clean_{lab}"] = dict(n=int(m.sum()), flagged_ctrl=int((pra[m] > 0).any(1).sum()), flagged_new=int((prb[m] > 0).any(1).sum()))
    K = "doc2_strict_total"
    out["argmax"] = dict(ctrl=float(c.score(pra, g)[K]), new=float(c.score(prb, g)[K]))
    mra = np.mean([rps_dim(pa[:, d], g[:, d], d) for d in range(6)], 0)
    mrb = np.mean([rps_dim(pb[:, d], g[:, d], d) for d in range(6)], 0)
    dd = np.array([mrb[b].mean() - mra[b].mean() for b in boots])
    out["mean_rps"] = dict(ctrl=float(mra.mean()), new=float(mrb.mean()), d=float(mrb.mean() - mra.mean()), ci90=[float(np.quantile(dd, .05)), float(np.quantile(dd, .95))])
    return out


def compare_multi(ctrl_dirs, new_dirs, n_folds, names, gold, grp, boot=600):
    """Recipe (one or more seeds) against the control trained with SEVERAL seeds.  Per image the loss is averaged over
    the seeds of each arm; the paired bootstrap over videos gives the image-sampling interval, and the spread of the
    control seeds gives the training-noise standard error `se_seed` = sd(control seeds) * sqrt(1/n_ctrl + 1/n_new).
    A single control seed can be a lucky or unlucky draw (seed 2026 is 0.0025 better than 2027 / 2028 on 过度生成), so
    the reference is the mean of all control seeds."""
    L = [load_folds(d, n_folds) for d in list(ctrl_dirs) + list(new_dirs)]
    folds = sorted(set.intersection(*[set(h) for _, h in L]))
    keep = [i for i, n in enumerate(names) if all(n in m for m, _ in L) and L[0][0][n][0] in folds]
    if not keep or not folds:
        return None
    g = gold[keep]
    P = [c.renorm(np.stack([m[names[i]][1] for i in keep])) for m, _ in L]
    PA, PB = P[:len(ctrl_dirs)], P[len(ctrl_dirs):]
    gr = grp[keep]
    ug, inv = np.unique(gr, return_inverse=True)
    mem = [np.where(inv == i)[0] for i in range(len(ug))]
    rs = np.random.RandomState(0)
    boots = [np.concatenate([mem[i] for i in rs.randint(0, len(ug), len(ug))]) for _ in range(boot)]
    kk = np.sqrt(1.0 / len(PA) + 1.0 / len(PB))
    out = dict(folds=folds, n=len(keep), n_ctrl=len(PA), n_new=len(PB), dims={})
    DA, DB = [argmax_decode(p) for p in PA], [argmax_decode(p) for p in PB]

    def one(ra_s, rb_s):
        ra, rb = np.mean(ra_s, 0), np.mean(rb_s, 0)
        dd = np.array([rb[b].mean() - ra[b].mean() for b in boots])
        sd = float(np.std([r.mean() for r in ra_s], ddof=1)) if len(ra_s) > 1 else float("nan")
        return dict(ctrl=float(ra.mean()), new=float(rb.mean()), d=float(rb.mean() - ra.mean()), ci90=[float(np.quantile(dd, .05)), float(np.quantile(dd, .95))],
                    ctrl_seeds=[float(r.mean()) for r in ra_s], new_seeds=[float(r.mean()) for r in rb_s], se_seed=sd * kk)

    for d in range(6):
        grades = (1, 2, 3) if d in FULL else (3,)
        r = one([rps_dim(p[:, d], g[:, d], d) for p in PA], [rps_dim(p[:, d], g[:, d], d) for p in PB])
        r.update(rps_ctrl=r["ctrl"], rps_new=r["new"], d_rps=r["d"],
                 auc_det_ctrl=float(np.mean([auc(1 - p[:, d, 0], g[:, d] > 0) for p in PA])), auc_det_new=float(np.mean([auc(1 - p[:, d, 0], g[:, d] > 0) for p in PB])),
                 f1_ctrl=float(np.mean([[f1(q[:, d] == k, g[:, d] == k) for k in grades] for q in DA])),
                 f1_new=float(np.mean([[f1(q[:, d] == k, g[:, d] == k) for k in grades] for q in DB])))
        out["dims"][DIMS[d]] = r
    clean = (g == 0).all(1)
    nm = [names[i] for i in keep]
    proc = np.array([bool(TAG_RE.search(n)) for n in nm])
    for lab, m in (("tagged", clean & proc), ("untagged", clean & ~proc)):
        out[f"clean_{lab}"] = dict(n=int(m.sum()), flagged_ctrl=float(np.mean([(q[m] > 0).any(1).sum() for q in DA])), flagged_new=float(np.mean([(q[m] > 0).any(1).sum() for q in DB])))
    K = "doc2_strict_total"
    out["argmax"] = dict(ctrl=float(np.mean([c.score(q, g)[K] for q in DA])), new=float(np.mean([c.score(q, g)[K] for q in DB])),
                         ctrl_seeds=[float(c.score(q, g)[K]) for q in DA])
    out["mean_rps"] = one([np.mean([rps_dim(p[:, d], g[:, d], d) for d in range(6)], 0) for p in PA], [np.mean([rps_dim(p[:, d], g[:, d], d) for d in range(6)], 0) for p in PB])
    return out


def counts(v):
    """An effect counts if the image-bootstrap interval excludes zero AND it exceeds twice the training-noise SE."""
    se = v.get("se_seed", float("nan"))
    big = (abs(v["d"]) > 2.0 * se) if se == se else True
    return (v["ci90"][0] > 0 and big, v["ci90"][1] < 0 and big)          # (worse, better)


def show(title, r):
    ms = r["mean_rps"]
    extra = f" | 2 x seed SE {2 * ms['se_seed']:.5f}; control seeds {' '.join(f'{x:.5f}' for x in ms['ctrl_seeds'])}" if "se_seed" in ms else ""
    seeds = f" ({r['n_ctrl']} control seed(s) vs {r['n_new']} recipe seed(s))" if "n_ctrl" in r else ""
    print(f"--- {title}: folds {r['folds']}, {r['n']} images{seeds} | argmax Doc2S {r['argmax']['ctrl']:.2f} -> {r['argmax']['new']:.2f} | "
          f"mean RPS {ms['ctrl']:.5f} -> {ms['new']:.5f} (d {ms['d']:+.5f} CI90 [{ms['ci90'][0]:+.5f}, {ms['ci90'][1]:+.5f}]){extra}")
    for k, v in r["dims"].items():
        print(f"    {k:6s} RPS {v['rps_ctrl']:.5f} -> {v['rps_new']:.5f} d {v['d_rps']:+.5f} CI90 [{v['ci90'][0]:+.5f}, {v['ci90'][1]:+.5f}]"
              + (f" 2xSE {2 * v['se_seed']:.5f}" if "se_seed" in v else "") + " | "
              f"det AUC {v['auc_det_ctrl']:.4f} -> {v['auc_det_new']:.4f} | label F1 {v['f1_ctrl']:.3f} -> {v['f1_new']:.3f}")
    print(f"    clean frames flagged: tagged {r['clean_tagged']['flagged_ctrl']} -> {r['clean_tagged']['flagged_new']} of {r['clean_tagged']['n']} | "
          f"untagged {r['clean_untagged']['flagged_ctrl']} -> {r['clean_untagged']['flagged_new']} of {r['clean_untagged']['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--targets", type=int, nargs="*", default=[], help="defect indices the recipe is meant to improve")
    ap.add_argument("--stage", default="screen", choices=["screen", "confirm", "pool", "aa"])
    ap.add_argument("--aa-seeds", type=int, nargs="*", default=[2027, 2028])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="recipe seeds to pool (default: --seed)")
    ap.add_argument("--ctrl-seeds", type=int, nargs="*", default=[2026, 2027, 2028], help="control seeds; those that exist are pooled")
    ap.add_argument("--note", default="")
    ap.add_argument("--prefix", default=None, help='tower name prefix: "synth-" (extra rows), "tw-" (in-place twins), "" (trainer-side recipe)')
    a = ap.parse_args()
    names, gold, fold, grp = c.load_split()
    od = os.path.join(REPO, f"results/r55/synthlab/{a.recipe}")
    os.makedirs(od, exist_ok=True)
    rep = dict(recipe=a.recipe, stage=a.stage, targets=[DIMS[t] for t in a.targets], prefix=a.prefix)
    print(f"=== SynthLab {a.stage} | recipe {a.recipe} | targets {rep['targets']} ===")
    if a.stage == "aa":
        # control vs control: the same recipe (no synthetic data) trained with other seeds.  The spread of these
        # differences is what a recipe has to beat.
        fl = {"standard": {}, "shift": {}, "pairs": []}
        acc = {"standard": [], "shift": []}
        for sd in a.aa_seeds:
            for scope, ctrl, n in (("standard", "fixed", 8), ("shift", "shift", 7)):
                r = compare(ctrl, f"{TOW}/r55_{ctrl}_s{a.seed}/pe_core_large", ctrl, f"{TOW}/r55_{ctrl}_s{sd}/pe_core_large", n, names, gold, grp)
                if r:
                    show(f"A/A {scope}: seed {a.seed} vs {sd}", r)
                    acc[scope].append(dict({k: v["d_rps"] for k, v in r["dims"].items()}, mean=r["mean_rps"]["d"]))
                    fl["pairs"].append(dict(scope=scope, seed=sd, folds=r["folds"], mean=r["mean_rps"]["d"], ci90=r["mean_rps"]["ci90"],
                                            clean_flagged=[r["clean_tagged"]["flagged_ctrl"] + r["clean_untagged"]["flagged_ctrl"], r["clean_tagged"]["flagged_new"] + r["clean_untagged"]["flagged_new"]]))
        for scope, lst in acc.items():
            for k in (lst[0] if lst else {}):
                fl[scope][k] = float(np.sqrt(np.mean([x[k] ** 2 for x in lst])))          # RMS of control-vs-control differences
        json.dump(fl, open(os.path.join(REPO, "results/r55/synthlab/noise_floor.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("noise floor (RMS difference between two trainings of the control):")
        for scope in ("standard", "shift"):
            print(f"  {scope}: " + "  ".join(f"{k} {v:.5f}" for k, v in fl[scope].items()))
        open(LEDGER, "a", encoding="utf-8").write(f"| {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} | control vs control | aa | seeds {a.seed} vs {a.aa_seeds} | - | noise floor mean RPS: standard {fl['standard'].get('mean', float('nan')):.5f}, unseen sources {fl['shift'].get('mean', float('nan')):.5f} | - | effects must exceed twice these to count |\n")
        return
    if a.stage in ("screen", "confirm"):
        # recipes built by synth_build.py train as synth-<R> / shiftsynth-<R>; trainer-side recipes (e.g. baug) as <R> / shift<R>
        pre = a.prefix if a.prefix is not None else ("synth-" if os.path.isdir(os.path.join(REPO, f"{TOW}/r55_synth-{a.recipe}_s{a.seed}")) or os.path.isdir(os.path.join(REPO, f"{TOW}/r55_shiftsynth-{a.recipe}_s{a.seed}")) else "")
        rs_ = a.seeds or [a.seed]
        ex = lambda d: os.path.isdir(os.path.join(REPO, d))                 # noqa: E731
        cs = [f"{TOW}/r55_fixed_s{x}/pe_core_large" for x in a.ctrl_seeds if ex(f"{TOW}/r55_fixed_s{x}/pe_core_large")]
        css = [f"{TOW}/r55_shift_s{x}/pe_core_large" for x in a.ctrl_seeds if ex(f"{TOW}/r55_shift_s{x}/pe_core_large")]
        ns = [f"{TOW}/r55_{pre}{a.recipe}_s{x}/pe_core_large" for x in rs_ if ex(f"{TOW}/r55_{pre}{a.recipe}_s{x}/pe_core_large")]
        nss = [f"{TOW}/r55_shift{pre}{a.recipe}_s{x}/pe_core_large" for x in rs_ if ex(f"{TOW}/r55_shift{pre}{a.recipe}_s{x}/pe_core_large")]
        std = compare_multi(cs, ns, 8, names, gold, grp) if ns else None
        sh = compare_multi(css, nss, 7, names, gold, grp) if nss else None
        if std:
            show("standard folds (in distribution)", std)
        if sh:
            show("source-held-out folds", sh)
        rep.update(standard=std, shift=sh, recipe_seeds=rs_, ctrl_seeds=a.ctrl_seeds)
        harm, gain = [], []
        for scope, r, where in ((std, std, "in distribution"), (sh, sh, "on unseen sources")):
            if not r:
                continue
            for k, v in r["dims"].items():
                worse, better = counts(v)
                if worse:
                    harm.append(f"{k} worse {where} ({v['d']:+.5f})")
                if better and (not a.targets or k in rep["targets"]):
                    gain.append(f"{k} better {where} ({v['d']:+.5f})")
            worse, better = counts(r["mean_rps"])
            if worse:
                harm.append(f"mean RPS worse {where} ({r['mean_rps']['d']:+.5f})")
            if better:
                gain.append(f"mean RPS better {where} ({r['mean_rps']['d']:+.5f})")
        if std:
            cl = std["clean_tagged"]["n"] + std["clean_untagged"]["n"]
            up = (std["clean_tagged"]["flagged_new"] + std["clean_untagged"]["flagged_new"]) - (std["clean_tagged"]["flagged_ctrl"] + std["clean_untagged"]["flagged_ctrl"])
            if cl and up / cl > 0.015:
                harm.append(f"clean false alarms {up:+.1f} of {cl}")
            if cl and up / cl < -0.015:
                gain.append(f"clean false alarms {up:+.1f} of {cl}")
        verdict = "HARM" if harm else ("PASS" if gain else "NULL")
        rep.update(verdict=verdict, gain=gain, harm=harm)
        print(f"VERDICT {verdict} | gains: {gain or '-'} | harms: {harm or '-'}")
        row = (f"| {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} | {(a.prefix or '') + a.recipe} | {a.stage} | std folds {std['folds'] if std else '-'}, shift folds {sh['folds'] if sh else '-'}; recipe seeds {rs_} vs control seeds {a.ctrl_seeds} (mean) | "
               f"{verdict} | {'; '.join(gain) or '-'} | {'; '.join(harm) or '-'} | {a.note} |")
    else:
        from decoder55 import nested_oof_decode
        from final_engine import SETS, oof_pool, pool
        from gate_plus import per_image
        K = "doc2_strict_total"
        _, _, _, p0, P = oof_pool(SETS["primary"])
        pre = a.prefix if a.prefix is not None else ("synth-" if os.path.isdir(os.path.join(REPO, f"{TOW}/r55_synth-{a.recipe}_s{a.seed}")) else "")
        A, have = load_folds(f"{TOW}/r55_{pre}{a.recipe}_s{a.seed}/pe_core_large", 8)
        assert len(have) == 8, f"pool gate needs all 8 standard folds, have {have}"
        Q = dict(P)
        Q["pe"] = c.renorm(np.stack([A[n][1] for n in names]))
        p1 = pool(Q, SETS["primary"])
        r0, _ = per_image(p0, gold)
        r1, _ = per_image(p1, gold)
        ug, inv = np.unique(grp, return_inverse=True)
        mem = [np.where(inv == i)[0] for i in range(len(ug))]
        rs = np.random.RandomState(0)
        boots = [np.concatenate([mem[i] for i in rs.randint(0, len(ug), len(ug))]) for _ in range(500)]
        d = np.array([r1[b].mean() - r0[b].mean() for b in boots])
        n0, n1 = nested_oof_decode(p0, gold, fold, K), nested_oof_decode(p1, gold, fold, K)
        s = np.array([c.score(n1[b], gold[b])[K] - c.score(n0[b], gold[b])[K] for b in boots[:300]])
        e0, e1 = c.score(n0, gold), c.score(n1, gold)
        verdict = "PASS" if np.quantile(d, .95) < 0 else ("HARM" if np.quantile(d, .05) > 0 else "NULL")
        print(f"pool: RPS {r0.mean():.5f} -> {r1.mean():.5f} (d {r1.mean() - r0.mean():+.5f} CI90 [{np.quantile(d, .05):+.5f}, {np.quantile(d, .95):+.5f}]) | "
              f"nested total {e0[K]:.2f} -> {e1[K]:.2f} (d {s.mean():+.2f} CI90 [{np.quantile(s, .05):+.2f}, {np.quantile(s, .95):+.2f}]) | VERDICT {verdict}")
        rep.update(pool=dict(rps0=float(r0.mean()), rps1=float(r1.mean()), d=float(r1.mean() - r0.mean()), ci90=[float(np.quantile(d, .05)), float(np.quantile(d, .95))],
                             total0=float(e0[K]), total1=float(e1[K]), d_total=float(s.mean())), verdict=verdict)
        row = (f"| {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} | {a.recipe} | pool | pe -> recipe tower in the primary ensemble | {verdict} | "
               f"pool RPS {r1.mean() - r0.mean():+.5f} CI90 [{np.quantile(d, .05):+.5f}, {np.quantile(d, .95):+.5f}]; nested total {s.mean():+.2f} | - | {a.note} |")
    json.dump(rep, open(os.path.join(od, f"{a.stage}{'_' + a.prefix.strip('-') if a.prefix else ''}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if not os.path.exists(LEDGER):
        open(LEDGER, "w", encoding="utf-8").write("# SynthLab ledger (append-only)\n\nEvery recipe, every gate. Control = same trainer, same folds, same seed, no synthetic rows.\n\n"
                                                  "| Time (UTC) | Recipe | Gate | Scope | Verdict | Gains (CI90 excludes 0) | Harms | Note |\n|---|---|---|---|---|---|---|---|\n")
    open(LEDGER, "a", encoding="utf-8").write(row + "\n")


if __name__ == "__main__":
    main()
