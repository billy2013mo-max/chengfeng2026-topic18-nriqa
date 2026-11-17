#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/stack_plus.py — stack composition judged on PROBABILITIES, not on the thresholded score (CPU, ~3 min).

Why: the nested-decoder Doc2_Strict of a stack moves by about +-0.3 when one of eight members is swapped for a rerun
of the same recipe (s1 51.04 -> 51.36 with r55_ctrl_s2026 in place of qy_pe_core_large, RPS unchanged), so stack
decisions below ~0.4 Doc2S are not evidence.  The ranked probability score (RPS), NLL and adjacent-grade AUC of the
pooled out-of-fold probabilities do not go through a decoder and separate compositions cleanly.

Members: the eight s1 towers (three Tier-1 backbones x two seeds, V16 x two seeds) and the three Phoenix pillars
(results/phoenix/pillar{1,2,3}_<phoenix-stage>_oof.npz).  Families: tier1 / v16 / phoenix.
"family-balanced" = every family gets the same total weight, split evenly inside the family.  The weights are a
fixed prior, not fitted; `--fit` additionally fits temperatures and weights inside a leave-one-fold-out loop to
show that nothing is left on the table (it is not needed for the deliverable).

Differences are paired cluster bootstraps over videos.  Phoenix folds are not the 8 folds: 11 videos (100 images)
straddle Phoenix folds; the table is repeated without them.  Everything is out-of-fold; no test image is read.

Usage:  python code/r55/stack_plus.py [--phoenix-stage stage2-rl2r|cv5] [--fit]
Output: results/r55/gates/stack_plus_<stage>.json
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
from decoder55 import argmax_decode, nested_oof_decode  # noqa: E402
from gate_plus import aucs, per_image  # noqa: E402

S1 = {"radio": "qy_cradio_v4_h", "pe": "qy_pe_core_large", "sig": "qy_siglip2_so400m_naflex",
      "radio27": "qy_s2027_cradio_v4_h", "pe27": "qy_s2027_pe_core_large", "sig27": "qy_s2027_siglip2_so400m_naflex",
      "v16": "v16_full8", "v16b": "v16_s700"}
FAM = {"radio": 0, "pe": 0, "sig": 0, "radio27": 0, "pe27": 0, "sig27": 0, "v16": 1, "v16b": 1, "p1": 2, "p2": 2, "p3": 2}
K = "doc2_strict_total"


def balanced(keys):
    cnt = np.bincount([FAM[k] for k in keys], minlength=3)
    return [1.0 / cnt[FAM[k]] for k in keys]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phoenix-stage", default="stage2-rl2r")
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--fit", action="store_true", help="also fit family temperatures / weights in a nested loop")
    ap.add_argument("--decode", action="store_true", help="also print the (noisy) nested-decoder Doc2S")
    a = ap.parse_args()

    names, gold, fold, grp = c.load_split()
    bank = c.build_bank(only=list(S1.values()))
    B = dict(zip(S1, c.stack_oof(bank, list(S1.values()), names)))
    strad = np.zeros(len(names), bool)
    for p in (1, 2, 3):
        z = np.load(os.path.join(REPO, f"results/phoenix/pillar{p}_{a.phoenix_stage}_oof.npz"), allow_pickle=True)
        m = {str(n): q for n, q in zip(z["names"], z["probs_6x4"].astype(np.float64))}
        pf = {str(n): int(f) for n, f in zip(z["names"], z["folds"])}
        B[f"p{p}"] = c.renorm(np.stack([m[n] for n in names]))
        f = np.array([pf[n] for n in names])
        for g in np.unique(grp):
            i = grp == g
            if len(set(f[i])) > 1:
                strad |= i

    ug, inv = np.unique(grp, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ug))]
    rs = np.random.RandomState(0)
    idxs = [np.concatenate([members[i] for i in rs.randint(0, len(ug), len(ug))]) for _ in range(a.bootstrap)]

    def metrics(keys, w=None):
        p = c.renorm(c.mean_pool([B[k] for k in keys], w))
        rps, nll = per_image(p, gold)
        det, grd = aucs(p, gold)
        out = dict(keys=list(keys), w=None if w is None else [float(x) for x in w], rps=rps, nll=nll, p=p,
                   argmax=float(c.score(argmax_decode(p), gold)[K]), auc_det=float(np.nanmean(det)), auc_grade=float(np.nanmean(grd)))
        if a.decode:
            out["nested"] = float(c.score(nested_oof_decode(p, gold, fold, K), gold)[K])
        return out

    rows, report = {}, {}

    def line(lab, keys, w=None, ref="s1"):
        m = metrics(keys, w)
        rows[lab] = m
        s = (f"{lab:42s} n={len(keys):2d} argmax {m['argmax']:.2f}" + (f" nested {m['nested']:.2f}" if a.decode else "")
             + f" RPS {m['rps'].mean():.5f} NLL {m['nll'].mean():.4f} AUCdet {m['auc_det']:.4f} AUCgrade {m['auc_grade']:.4f}")
        rec = {k: m[k] for k in ("keys", "w", "argmax", "auc_det", "auc_grade")}
        rec.update(rps=float(m["rps"].mean()), nll=float(m["nll"].mean()), rps_no_straddle=float(m["rps"][~strad].mean()))
        if "nested" in m:
            rec["nested"] = m["nested"]
        if ref in rows and ref != lab:
            b = rows[ref]
            d = np.array([m["rps"][i].mean() - b["rps"][i].mean() for i in idxs])
            dn = np.array([m["nll"][i].mean() - b["nll"][i].mean() for i in idxs])
            s += (f" | vs {ref}: dRPS {d.mean():+.5f} CI90 [{np.quantile(d, .05):+.5f}, {np.quantile(d, .95):+.5f}] P(no gain)={(d >= 0).mean():.2f}"
                  f"  dNLL {dn.mean():+.4f}  (dRPS without straddling videos {m['rps'][~strad].mean() - b['rps'][~strad].mean():+.5f})")
            rec.update(ref=ref, d_rps=float(d.mean()), d_rps_ci90=[float(np.quantile(d, .05)), float(np.quantile(d, .95))],
                       p_no_gain=float((d >= 0).mean()), d_nll=float(dn.mean()))
        print(s, flush=True)
        report[lab] = rec

    s1 = ["radio", "pe", "radio27", "pe27", "sig27", "sig", "v16", "v16b"]
    ph = ["p1", "p2", "p3"]
    one = ["radio", "pe", "sig", "v16"]
    print(f"=== stack_plus | phoenix {a.phoenix_stage} | N={len(names)} | {int(strad.sum())} images in videos that straddle phoenix folds ===")
    print("--- single members")
    for k in s1 + ph:
        line(k, [k], ref="")
    print("--- s1 and leave-one-out (equal weights)")
    line("s1", s1)
    for k in s1:
        line(f"s1 - {k}", [x for x in s1 if x != k])
    print("--- equal weights")
    line("s1 + p1 p2 p3", s1 + ph)
    line("s1 + p1 p3", s1 + ["p1", "p3"])
    line("s1 + p3", s1 + ["p3"])
    line("one seed each + p1 p2 p3", one + ph)
    line("tier1 x6 + p1 p2 p3 (no v16)", s1[:6] + ph)
    line("p1 p2 p3", ph)
    print("--- family-balanced weights (tier1 / v16 / phoenix each sum to 1)")
    line("s1, balanced", s1, balanced(s1))
    line("s1 + p1 p2 p3, balanced", s1 + ph, balanced(s1 + ph))
    line("s1 + p1 p3, balanced", s1 + ["p1", "p3"], balanced(s1 + ["p1", "p3"]))
    line("one seed each + p1 p2 p3, balanced", one + ph, balanced(one + ph))
    line("one seed each + p1 p3, balanced", one + ["p1", "p3"], balanced(one + ["p1", "p3"]))
    line("one seed each + p3, balanced", one + ["p3"], balanced(one + ["p3"]))
    line("  same vs equal-weight s1+p123", s1 + ph, balanced(s1 + ph), ref="s1 + p1 p2 p3")
    line("  7 models balanced vs 11 balanced", one + ph, balanced(one + ph), ref="s1 + p1 p2 p3, balanced")

    if a.fit:
        import torch
        torch.set_num_threads(16)
        G = torch.tensor(gold)
        full, p0 = torch.tensor([0, 1, 2, 4]), torch.tensor([3, 5])

        def nll_t(p, g):
            x = -torch.log(p[:, full].gather(-1, g[:, full, None]).squeeze(-1).clamp_min(1e-6)).sum(1)
            q = p[:, p0, 3] / (p[:, p0, 0] + p[:, p0, 3]).clamp_min(1e-9)
            y = (g[:, p0] == 3).float()
            return ((x - torch.log(torch.where(y == 1, q, 1 - q).clamp_min(1e-6)).sum(1)) / 6).mean()

        def fitted(lab, keys, ref):
            L = torch.log(torch.tensor(np.stack([B[k] for k in keys])).clamp_min(1e-9))
            fam = torch.tensor([FAM[k] for k in keys])
            w0 = torch.log(torch.tensor(balanced(keys)))
            out, Ts = np.zeros((len(names), 6, 4)), []

            def pool(Lx, aT, aw):
                q = torch.softmax(Lx * torch.exp(-aT[fam])[:, None, None, None], -1)
                return (q * torch.softmax(w0 + aw[fam], 0)[:, None, None, None]).sum(0)

            for f in range(8):
                tr, te = torch.tensor(fold != f), torch.tensor(fold == f)
                aT, aw = torch.zeros(3, requires_grad=True), torch.zeros(3, requires_grad=True)
                opt = torch.optim.LBFGS([aT, aw], lr=0.5, max_iter=300, line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad()
                    loss = nll_t(pool(L[:, tr], aT, aw), G[tr])
                    loss.backward()
                    return loss
                opt.step(closure)
                with torch.no_grad():
                    out[fold == f] = pool(L[:, te], aT, aw).numpy()
                Ts.append(np.exp(aT.detach().numpy()))
            B["_fit"] = c.renorm(out)
            line(lab, ["_fit"], ref=ref)
            print(f"      fitted family temperatures (mean over folds) {np.round(np.mean(Ts, 0), 2)}")
        print("--- nested fit of 3 family temperatures + 3 family weights (start = family-balanced)")
        fitted("fit: s1 + p1 p2 p3", s1 + ph, "s1 + p1 p2 p3, balanced")
        fitted("fit: one seed each + p1 p2 p3", one + ph, "one seed each + p1 p2 p3, balanced")

    od = os.path.join(REPO, "results/r55/gates")
    os.makedirs(od, exist_ok=True)
    json.dump(report, open(os.path.join(od, f"stack_plus_{a.phoenix_stage}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[stack_plus] wrote results/r55/gates/stack_plus_{a.phoenix_stage}.json")


if __name__ == "__main__":
    main()
