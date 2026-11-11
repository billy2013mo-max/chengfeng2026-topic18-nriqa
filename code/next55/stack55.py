#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/stack55.py — CPU, ~5-20 min.  The fusion + decision layer of the next55 plan.

Two fusion candidates, evaluated under ONE strictly nested protocol (outer 8 folds; for every
outer fold the decoder — and for `lr` the stacker too — are fitted only on the other 7 folds):

  mean : log-linear pool of all towers                        (0 learned fusion params)
  lr   : per-dimension multinomial logistic regression on the concatenated log-simplices of all
         towers for ALL six dims (cross-defect evidence, e.g. 模糊 vs 纹理损失), L2, C small.
         Decoder for outer fold f is fitted on an inner 7-fold OOF of the stacker (no leakage).

Then a paired bootstrap (images resampled, metric recomputed) vs the reference fusion
(`--ref-towers`, default = the six towers that exist today → 50.7 Doc2S / 51.2 Doc1).

Outputs  results/next55/stack/<tag>/
  report.json      nested scores, per-fold scores, bootstrap CI, chosen fusion
  final.pkl        fitted on ALL OOF: towers, fusion, stacker coefs, decoder state
                   (used by build_submission55.py — test labels are never touched)

Usage:
  python code/next55/stack55.py --tag s1 [--towers k1 k2 ...] [--objective blend|doc1_total|doc2_strict_total]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common55 import FULL_DIMS, OUT_DIR, build_bank, load_split, logit_pool, renorm, score, score_line, stack_oof  # noqa: E402
from decoder55 import Decoder55  # noqa: E402

REF_TOWERS = ["qy_convnextv2_base", "qy_cradio_v4_h", "qy_dinov3_vitl16", "qy_pe_core_large",
              "qy_siglip2_so400m_naflex", "v16_full8"]


# ----------------------------------------------------------------------------------------------
class LRStacker:
    def __init__(self, C: float = 0.05):
        self.C = C
        self.models = {}
        self.mu = None
        self.sd = None

    @staticmethod
    def feats(S: np.ndarray) -> np.ndarray:  # S: (M, N, 6, 4)
        M, N = S.shape[:2]
        return np.log(np.clip(S, 1e-6, 1.0)).transpose(1, 0, 2, 3).reshape(N, M * 24)

    def fit(self, S, gold):
        X = self.feats(S)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = (X - self.mu) / self.sd
        for d in range(6):
            y = gold[:, d]
            m = LogisticRegression(C=self.C, max_iter=2000, class_weight=None)
            m.fit(X, y)
            self.models[d] = m
        return self

    def predict(self, S):
        X = (self.feats(S) - self.mu) / self.sd
        out = np.full((X.shape[0], 6, 4), 1e-6)
        for d, m in self.models.items():
            pr = m.predict_proba(X)
            for j, c in enumerate(m.classes_):
                out[:, d, int(c)] = pr[:, j]
        return renorm(out)


def fuse_fit(kind, S, gold, C):
    return LRStacker(C).fit(S, gold) if kind == "lr" else None


def fuse_apply(kind, model, S):
    return model.predict(S) if kind == "lr" else logit_pool(list(S))


def nested_eval(kind, S, gold, fold, objective, C):
    """Returns (pred labels OOF, fused OOF simplex) under the strict nested protocol."""
    pred = np.zeros_like(gold)
    fused_oof = np.zeros((gold.shape[0], 6, 4))
    for f in np.unique(fold):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        if kind == "lr":
            inner = np.zeros((len(tr), 6, 4))
            for g in np.unique(fold[tr]):
                a, b = fold[tr] != g, fold[tr] == g
                inner[b] = fuse_apply(kind, fuse_fit(kind, S[:, tr[a]], gold[tr[a]], C), S[:, tr[b]])
        else:
            inner = fuse_apply(kind, None, S[:, tr])
        dec = Decoder55(objective).fit(inner, gold[tr])
        fz = fuse_apply(kind, fuse_fit(kind, S[:, tr], gold[tr], C), S[:, te])
        fused_oof[te] = fz
        pred[te] = dec.predict(fz)
    return pred, fused_oof


def per_fold(pred, gold, fold):
    return {k: [float(score(pred[fold == f], gold[fold == f])[k]) for f in np.unique(fold)]
            for k in ("doc1_final", "doc1_total", "doc2_strict_total", "blend")}


def paired_bootstrap(pa, pb, gold, n=300, seed=0):
    """CI of score(pa) - score(pb) with images resampled (metric is non-decomposable)."""
    rng = np.random.default_rng(seed)
    N = gold.shape[0]
    out = {k: [] for k in ("doc1_final", "doc1_total", "doc2_strict_total", "blend")}
    for _ in range(n):
        ix = rng.integers(0, N, N)
        ea, eb = score(pa[ix], gold[ix]), score(pb[ix], gold[ix])
        for k in out:
            out[k].append(ea[k] - eb[k])
    return {k: {"mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                "p_le0": float(np.mean(np.array(v) <= 0))} for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--towers", nargs="*", default=None, help="default: every discovered tower with full OOF")
    ap.add_argument("--ref-towers", nargs="*", default=REF_TOWERS)
    ap.add_argument("--objective", default="doc1_final", choices=["doc1_final", "blend", "doc1_total", "doc2_strict_total"])
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--fusions", nargs="*", default=["mean"], help="'lr' lost to 'mean' at every C tried (handoff §4.3)")
    ap.add_argument("--bootstrap", type=int, default=300)
    args = ap.parse_args()

    out_dir = os.path.join(OUT_DIR, "stack", args.tag)
    os.makedirs(out_dir, exist_ok=True)
    names, gold, fold, _ = load_split()
    bank = build_bank()
    towers = sorted(args.towers or bank.keys())
    missing = [t for t in towers + args.ref_towers if t not in bank]
    assert not missing, f"towers without full OOF: {missing}"
    S = stack_oof(bank, towers, names)
    S_ref = stack_oof(bank, sorted(args.ref_towers), names)
    print(f"[stack55] towers({len(towers)})={towers}\n[stack55] objective={args.objective}")

    ref_pred, _ = nested_eval("mean", S_ref, gold, fold, args.objective, args.C)
    ev_ref = score(ref_pred, gold)
    print(f"  REF  mean{len(args.ref_towers)}  {score_line(ev_ref)}")

    report = {"towers": towers, "objective": args.objective, "ref": {"towers": args.ref_towers,
              "score": {k: ev_ref[k] for k in ("doc1_final", "doc1_total", "doc2_strict_total", "blend")},
              "per_fold": per_fold(ref_pred, gold, fold)}, "cands": {}}
    best = None
    for kind in args.fusions:
        pred, fused_oof = nested_eval(kind, S, gold, fold, args.objective, args.C)
        ev = score(pred, gold)
        pf = per_fold(pred, gold, fold)
        wins = int(sum(a > b for a, b in zip(pf[args.objective], report["ref"]["per_fold"][args.objective])))
        bs = paired_bootstrap(pred, ref_pred, gold, n=args.bootstrap)
        print(f"  {kind:4s} M={len(towers)}   {score_line(ev)}\n"
              f"        Δ vs REF ({args.objective}) = {bs[args.objective]['mean']:+.3f} "
              f"[{bs[args.objective]['lo']:+.3f}, {bs[args.objective]['hi']:+.3f}]  fold-wins {wins}/8  "
              f"per-fold Doc1 {np.mean(pf['doc1_total']):.2f}±{np.std(pf['doc1_total']):.2f} "
              f"Doc2S {np.mean(pf['doc2_strict_total']):.2f}±{np.std(pf['doc2_strict_total']):.2f}")
        report["cands"][kind] = {"score": {k: ev[k] for k in ("doc1_final", "doc1_total", "doc2_strict_total", "blend")},
                                 "per_fold": pf, "fold_wins_vs_ref": wins, "bootstrap_vs_ref": bs,
                                 "per_14": ev["per_14"]}
        if best is None or ev[args.objective] > best[1]:
            best = (kind, ev[args.objective], fused_oof)

    # `lr` must beat `mean` clearly, otherwise prefer the zero-parameter pool (Occam gate)
    chosen = best[0]
    if chosen == "lr" and "mean" in report["cands"]:
        if report["cands"]["lr"]["score"][args.objective] - report["cands"]["mean"]["score"][args.objective] < 0.15:
            chosen = "mean"
    report["chosen"] = chosen
    print(f"[stack55] chosen fusion = {chosen}")

    # ---- final artefact: fusion fitted on all OOF; decoder fitted on the (nested) fused OOF ----
    model = fuse_fit(chosen, S, gold, args.C)
    fused_oof = nested_eval(chosen, S, gold, fold, args.objective, args.C)[1] if chosen != best[0] else best[2]
    dec = Decoder55(args.objective).fit(fused_oof, gold)
    pickle.dump({"towers": towers, "fusion": chosen, "model": model, "decoder": dec.state(), "C": args.C},
                open(os.path.join(out_dir, "final.pkl"), "wb"))
    report["final_decoder"] = dec.state()
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[stack55] wrote {out_dir}/report.json and final.pkl")


if __name__ == "__main__":
    main()
