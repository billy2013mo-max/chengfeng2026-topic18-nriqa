#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/infer_tier1_ckpt.py — inference-only for any qingyu-format Tier-1 tower (1 GPU).

The Tier-1 trainer only predicts an extra test set *during* training.  The hidden final
set (654 imgs) arrives later, so this script reloads the 8 saved `fold_{k}_best.pt`
(trainable_state_dict incl. head/physics-scaler/prototype buffers), runs the same 2x
hflip TTA, and writes <tower_dir>/ext_<tag>_probs.npz (per-fold stack + mean over the 8 folds).

Safety: `--verify-va` re-predicts each fold's own val set and asserts it reproduces the
stored `va_probs_6x4` (max |Δ| < 2e-2), i.e. the reload is bit-faithful before trusting test output.

Usage:
  python code/next55/infer_tier1_ckpt.py --backbone pe_core_large --tower-dir results/qingyu/pe_core_large \
      --template <final_template.json> --img-dir <final_img_dir> --tag final654 --gpu 0 [--verify-va]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qingyu.config_and_paths import REPO_ROOT, LosslessNativeImageLoader, load_test_template_names, setup_sys_path_and_cwd  # noqa: E402
from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM, DualEnginePhysicsExtractor, PrecomputedDualEngineStore  # noqa: E402
from qingyu.backbones_qingyu import QingyuVisionQualityTower  # noqa: E402
from qingyu.train_tier1_5backbones_8fold import load_fold_rows, make_tier1_collate_fn, predict_2x_tta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--tower-dir", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--img-dir", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(8)))
    ap.add_argument("--split-dir", default="data/qingyu_splits")
    ap.add_argument("--dual-engine-npz", default=os.environ.get("NEXT55_DUAL_NPZ", "results/qingyu/dual_engine_meta.npz"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify-va", action="store_true")
    a = ap.parse_args()

    setup_sys_path_and_cwd(chdir_repo=True)
    dev = torch.device(f"cuda:{a.gpu}")
    torch.cuda.set_device(dev)
    tower_dir = a.tower_dir if os.path.isabs(a.tower_dir) else os.path.join(REPO_ROOT, a.tower_dir)
    names = load_test_template_names(a.template)
    rows = [{"name": n, "group_id": n, "grades": [0] * 6} for n in names]
    img_loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=[a.img_dir] if a.img_dir else None)
    meta_store = PrecomputedDualEngineStore(a.dual_engine_npz, lazy_extractor=DualEnginePhysicsExtractor(device=str(dev), allow_depth_download=False))

    acc = []
    for f in a.folds:
        ck = os.path.join(tower_dir, f"fold_{f}_best.pt")
        assert os.path.exists(ck), ck
        model = QingyuVisionQualityTower(backbone_key=a.backbone, phys_dim=PHYS_FEAT_DIM).to(dev)
        missing, unexpected = model.load_state_dict(torch.load(ck, map_location="cpu"), strict=False)
        assert not unexpected, f"unexpected keys in {ck}: {unexpected[:5]}"
        print(f"  [params] {a.backbone}: {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B", flush=True)
        collate = make_tier1_collate_fn(model.mode_tag, naflex_processor=getattr(model, "naflex_processor", None), max_num_patches=784)
        if a.verify_va:
            _, va_rows = load_fold_rows(os.path.join(a.split_dir, f"fold_{f}.csv"))
            ref = np.load(os.path.join(tower_dir, f"fold_{f}_probs.npz"), allow_pickle=True)
            vp, _, _, vn = predict_2x_tta(model, va_rows, img_loader, meta_store, collate, dev, a.batch_size, a.workers)
            pos = {str(n): i for i, n in enumerate(ref["va_names"])}
            d = float(np.max(np.abs(vp - ref["va_probs_6x4"][[pos[str(n)] for n in vn]])))
            print(f"  [verify] fold {f}: max|Δ va_probs| = {d:.4g}", flush=True)
            assert d < 2e-2, f"fold {f} reload does not reproduce stored OOF (Δ={d})"
        p, _, _, pn = predict_2x_tta(model, rows, img_loader, meta_store, collate, dev, a.batch_size, a.workers)
        assert list(map(str, pn)) == names
        acc.append(p)
        print(f"  [infer] {a.backbone} fold {f} -> {p.shape}", flush=True)
        del model
        torch.cuda.empty_cache()

    out = os.path.join(tower_dir, f"ext_{a.tag}_probs.npz")
    np.savez_compressed(out, names=np.array(names), probs_6x4=np.mean(acc, 0).astype(np.float32),
                        probs_6x4_folds=np.stack(acc).astype(np.float32), n_folds=len(acc))
    print(f"[infer_tier1_ckpt] wrote {out}")


if __name__ == "__main__":
    main()
