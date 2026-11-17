#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/eval_tta.py — inference-only: does averaging over several native-crop sets help? (1 GPU, no training)

For each fold k of a Tier-1-format tower it reloads fold_k_best.pt and predicts ONLY that fold's validation rows
(so every number stays out-of-fold) with
  std        the trainer's deterministic guided crops, 2x h-flip          (must reproduce the stored OOF)
  rand_i     pass i with the four crops re-drawn (CropJitterStore), 2x h-flip
and writes <tower-dir>/tta_<tag>.npz with va_names, gold, fold, probs_std, probs_rand (K, N, 6, 4).
`--score` then reports nested-decoder Doc2_Strict for std, mean of the first K random passes, and std+random.

Nothing here touches test150 or any test template.

Usage:
  python code/r55/eval_tta.py --tower-dir results/next55/towers/qy_s2028/pe_core_large --gpu 0 --k 8 --tag r50j96
  python code/r55/eval_tta.py --tower-dir ... --tag r50j96 --score
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import REPO  # noqa: E402,F401  (adds code/ to sys.path)
from qingyu.config_and_paths import REPO_ROOT, LosslessNativeImageLoader, setup_sys_path_and_cwd  # noqa: E402
from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM, DualEnginePhysicsExtractor, PrecomputedDualEngineStore  # noqa: E402
from qingyu.backbones_qingyu import QingyuVisionQualityTower  # noqa: E402
from qingyu.train_tier1_5backbones_8fold import load_fold_rows, make_tier1_collate_fn, predict_2x_tta  # noqa: E402


def run(a):
    from train_testbed import CropJitterStore
    setup_sys_path_and_cwd(chdir_repo=True)
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)
    tower = a.tower_dir if os.path.isabs(a.tower_dir) else os.path.join(REPO_ROOT, a.tower_dir)
    loader = LosslessNativeImageLoader(REPO_ROOT)
    store = PrecomputedDualEngineStore(a.dual_engine_npz, lazy_extractor=DualEnginePhysicsExtractor(device=str(dev), allow_depth_download=False))
    jit = CropJitterStore(store, a.p_rand, a.jitter)
    names, gold, folds, p_std, p_rand = [], [], [], [], []
    for f in a.folds:
        model = QingyuVisionQualityTower(backbone_key=a.backbone, phys_dim=PHYS_FEAT_DIM).to(dev)
        _, unexpected = model.load_state_dict(torch.load(os.path.join(tower, f"fold_{f}_best.pt"), map_location="cpu"), strict=False)
        assert not unexpected, unexpected[:5]
        collate = make_tier1_collate_fn(model.mode_tag, naflex_processor=getattr(model, "naflex_processor", None), max_num_patches=784)
        _, va = load_fold_rows(os.path.join(a.split_dir, f"fold_{f}.csv"))
        vp, _, g, vn = predict_2x_tta(model, va, loader, store, collate, dev, a.batch_size, a.workers)
        ref = np.load(os.path.join(tower, f"fold_{f}_probs.npz"), allow_pickle=True)
        pos = {str(n): i for i, n in enumerate(ref["va_names"])}
        d = float(np.max(np.abs(vp - ref["va_probs_6x4"][[pos[str(n)] for n in vn]])))
        print(f"  fold {f}: std reload max|d|={d:.4g}", flush=True)
        assert d < 2e-2, "reload does not reproduce the stored OOF"
        rs = []
        for i in range(a.k):
            torch.manual_seed(1000 * f + i)                 # DataLoader workers derive their `random` seed from this
            rp, _, _, rn = predict_2x_tta(model, va, loader, jit, collate, dev, a.batch_size, a.workers)
            assert list(rn) == list(vn)
            rs.append(rp)
        names += list(vn); gold.append(g); folds += [f] * len(vn); p_std.append(vp); p_rand.append(np.stack(rs))
        del model
        torch.cuda.empty_cache()
    out = os.path.join(tower, f"tta_{a.tag}.npz")
    np.savez_compressed(out, va_names=np.array(names), gold=np.concatenate(gold), fold=np.array(folds),
                        probs_std=np.concatenate(p_std), probs_rand=np.concatenate(p_rand, axis=1))
    print(f"[eval_tta] wrote {out}")


def score(a):
    sys.path.insert(0, os.path.join(REPO, "code/next55"))
    from common55 import renorm, score as sc
    from decoder55 import nested_oof_decode
    tower = a.tower_dir if os.path.isabs(a.tower_dir) else os.path.join(REPO_ROOT, a.tower_dir)
    z = np.load(os.path.join(tower, f"tta_{a.tag}.npz"), allow_pickle=True)
    gold, fold, std, rnd = z["gold"], z["fold"], z["probs_std"].astype(np.float64), z["probs_rand"].astype(np.float64)

    def line(lab, p):
        ev = sc(nested_oof_decode(renorm(p), gold, fold, "doc2_strict_total"), gold)
        print(f"  {lab:<22s} Doc2S {ev['doc2_strict_total']:6.2f} | pos {ev['doc2_S_pos']:5.2f} p0 {ev['doc2_S_p0']:5.2f} clean {ev['doc2_Clean2']:5.2f}")
        return ev["doc2_strict_total"]

    print(f"[eval_tta] {tower} | N={len(gold)} | K={rnd.shape[0]}")
    line("std (guided crops)", std)
    for k in (1, 2, 4, 8, 16):
        if k <= rnd.shape[0]:
            line(f"random x{k}", rnd[:k].mean(0))
    for k in (1, 2, 4, 8, 16):
        if k <= rnd.shape[0]:
            line(f"std + random x{k}", (std + rnd[:k].sum(0)) / (k + 1))
    # stability: how often the argmax grade changes between two independent random passes
    if rnd.shape[0] >= 2:
        a0, a1 = rnd[0].argmax(-1), rnd[1].argmax(-1)
        print("  argmax flip rate between two random passes, per dim:", np.round((a0 != a1).mean(0), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="pe_core_large")
    ap.add_argument("--tower-dir", required=True)
    ap.add_argument("--tag", default="r50j96")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(8)))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--p-rand", type=float, default=0.5)
    ap.add_argument("--jitter", type=int, default=96)
    ap.add_argument("--split-dir", default="data/qingyu_splits")
    ap.add_argument("--dual-engine-npz", default="results/qingyu/dual_engine_meta.npz")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    score(args) if args.score else run(args)
