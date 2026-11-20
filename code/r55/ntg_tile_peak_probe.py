#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/ntg_tile_peak_probe.py
===============================
Zero-overhead 1:1 native-tile per-grade peak extractor (`max_t`, `top2_t`, `max-mean`, `std_t`, `mil-mean`
for `P(>=P2)`, `P(>=P1)`, `P(P0)`, and `severity_s`) from `NativeTileTower` (`ntg_pe`).

- Uses the EXACT same `Tier1DualEngineDataset(mode_tag="native_tiles")` + 2x h-flip TTA + `NativeTileTower`
  tensors that `FinalEngine._ntg("ntg_pe", name)` ALREADY computes during normal inference (~0.3 ms overhead,
  0 extra models, 0 MB extra VRAM).
- Strictly zero-leakage paired extraction:
    * `--mode oof`: fold `k` uses held-out `results/next55/towers/r55_ntg_s2026/pe_core_large/fold_{k}_best.pt`
    * `--mode target`: uses matched full-fit `results/r55/fullfit_v5_ntg/pe_core_large/fold_0_best.pt`
"""
import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _preparse_gpu_env() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"].strip() != "":
        return
    argv = sys.argv[1:]
    for i, tok in enumerate(argv):
        if tok == "--gpu" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(argv[i + 1])
            return
        if tok.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = tok.split("=", 1)[1]
            return


_preparse_gpu_env()

import numpy as np  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

TILE_STAT_NAMES: Tuple[str, ...] = (
    "max_p_any",
    "gap_p_any",
    "max_p_ge1",
    "gap_p_ge1",
    "max_p_p0",
    "gap_p_p0",
    "top2_p_p0",
    "std_p_p0",
    "mil_gap_p0",
    "max_sev",
    "gap_sev",
    "top2_sev",
)
N_TILE_STATS = len(TILE_STAT_NAMES)


def summarize_per_tile_outputs(tile_probs: np.ndarray, tile_sev: np.ndarray, tile_mil_w: np.ndarray) -> np.ndarray:
    """
    Order-invariant per-tile summary `(6, 12)` from a single orientation's `(T, 6, 4)` tile probs,
    `(T, 6)` tile severities, and `(T,)` MIL attention weights. Strictly `N = 1` deterministic.
    """
    tp = np.clip(np.asarray(tile_probs, dtype=np.float64), 1e-7, 1.0)
    tp = tp / np.sum(tp, axis=-1, keepdims=True)
    ts = np.asarray(tile_sev, dtype=np.float64)
    mw = np.asarray(tile_mil_w, dtype=np.float64)
    mw = mw / max(1e-8, float(np.sum(mw)))

    T = tp.shape[0]
    k2 = min(2, T)
    out = np.zeros((6, N_TILE_STATS), dtype=np.float32)

    p_any = 1.0 - tp[:, :, 0]              # (T, 6)
    p_ge1 = tp[:, :, 2] + tp[:, :, 3]      # (T, 6)
    p_p0 = tp[:, :, 3]                     # (T, 6)

    for d in range(6):
        pa, pg1, pp0, sv = p_any[:, d], p_ge1[:, d], p_p0[:, d], ts[:, d]
        out[d, 0] = float(np.max(pa))
        out[d, 1] = float(np.max(pa) - np.mean(pa))
        out[d, 2] = float(np.max(pg1))
        out[d, 3] = float(np.max(pg1) - np.mean(pg1))
        out[d, 4] = float(np.max(pp0))
        out[d, 5] = float(np.max(pp0) - np.mean(pp0))
        out[d, 6] = float(np.mean(np.sort(pp0)[-k2:]))
        out[d, 7] = float(np.std(pp0))
        out[d, 8] = float(np.sum(mw * pp0) - np.mean(pp0))
        out[d, 9] = float(np.max(sv))
        out[d, 10] = float(np.max(sv) - np.mean(sv))
        out[d, 11] = float(np.mean(np.sort(sv)[-k2:]))
    return np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)


def infer_ntg_pe_with_tile_capture(model, batch_dict: dict, dev) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs a single-image (`B == 1`) `NativeTileTower` forward pass while capturing per-tile
    `(T, 6, 4)` probabilities, `(T, 6)` severities, and `(T,)` MIL weights from the EXACT
    same `f_tiles` and `slot0` tensors (zero redundant backbone encode).
    """
    import torch

    views_01 = batch_dict["tiles_u8"].to(dev, non_blocking=True).permute(0, 1, 4, 2, 3).float() / 255.0
    phys_feat = batch_dict["phys"].to(dev, non_blocking=True)
    tile_mask = batch_dict["tile_mask"].to(dev, non_blocking=True)
    global_01 = batch_dict["global_01"].to(dev, non_blocking=True) if "global_01" in batch_dict else None

    use_amp = dev.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.bfloat16, enabled=use_amp):
        B, T = views_01.shape[:2]
        assert B == 1
        valid = tile_mask.bool()
        flat = model._norm_img(views_01[valid])
        chunks = [model._encode_single_view(flat[s : s + 16]) for s in range(0, flat.shape[0], 16)]
        emb = torch.cat(chunks, dim=0)
        feats = emb.new_zeros((B, T, emb.shape[-1]))
        feats[valid] = emb
        g = model._encode_single_view(model._norm_img(global_01)) if (model.use_global and global_01 is not None) else None

    with torch.no_grad():
        f = feats.float()
        m = valid.float()
        mean = (f * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
        lg = model.view_agg.gate(f).squeeze(-1).masked_fill(~valid, -1e4)
        w = torch.softmax(lg, dim=1).unsqueeze(-1)
        attn = (w * f).sum(1)
        mx = f.masked_fill(~valid.unsqueeze(-1), -1e4).amax(1)
        slot0 = g.float() if g is not None else mean
        vis_img = torch.cat([slot0, attn, mx], dim=-1)
        out_img = model.head(vis_img, phys_feat=phys_feat.float())

        # Per-tile head pass on the exact same `f` and `slot0` (~0.3 ms!)
        ft = f[0, valid[0]]                                    # (Tv, D)
        Tv = ft.shape[0]
        s0_rep = slot0[0:1].expand(Tv, -1)
        vis_tile = torch.cat([s0_rep, ft, ft], dim=-1)         # (Tv, 3*D)
        out_tile = model.head(vis_tile, phys_feat=phys_feat[0:1].float().expand(Tv, -1))
        w_valid = w[0, valid[0], 0]
        w_valid = w_valid / w_valid.sum().clamp(min=1e-8)

        img_p = out_img["probs_6x4"][0].float().cpu().numpy()
        tile_p = out_tile["probs_6x4"].float().cpu().numpy()
        tile_s = out_tile["severity_s"].float().cpu().numpy()
        tile_w = w_valid.float().cpu().numpy()
    return img_p, tile_p, tile_s, tile_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["oof", "merge-oof", "target", "merge-target"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--template", default="")
    ap.add_argument("--img-dir", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default="results/r55/ntg_tile_peaks")
    ap.add_argument("--out-npz", default="")
    args = ap.parse_args()

    import common55 as c

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "merge-oof":
        names, _, fold_arr, _ = c.load_split()
        idx_map = {str(n): i for i, n in enumerate(names)}
        stats = np.zeros((len(names), 6, N_TILE_STATS), dtype=np.float32)
        valid = np.zeros(len(names), dtype=bool)
        for k in range(8):
            z = np.load(os.path.join(out_dir, f"oof_fold_{k}.npz"), allow_pickle=True)
            for nm, st in zip(z["names"], z["stats"]):
                ii = idx_map[str(nm)]
                assert int(fold_arr[ii]) == k
                stats[ii] = st
                valid[ii] = True
        assert bool(valid.all()), f"Incomplete: {valid.sum()}/{len(names)}"
        out_p = args.out_npz or os.path.join(out_dir, "ntg_tile_peaks_oof_3573.npz")
        np.savez_compressed(out_p, names=np.array(names), stats=stats, stat_names=np.array(TILE_STAT_NAMES))
        print(f"[ntg_tile_peak_probe] merged OOF {stats.shape} -> {out_p}")
        return

    if args.mode == "merge-target":
        names = c.load_template_names(args.template)
        idx_map = {str(n): i for i, n in enumerate(names)}
        stats = np.zeros((len(names), 6, N_TILE_STATS), dtype=np.float32)
        valid = np.zeros(len(names), dtype=bool)
        for fp in sorted(glob.glob(os.path.join(out_dir, f"target_shard_*_of_{args.num_shards:02d}.npz"))):
            z = np.load(fp, allow_pickle=True)
            for nm, st in zip(z["names"], z["stats"]):
                sn = str(nm)
                if sn in idx_map:
                    stats[idx_map[sn]] = st
                    valid[idx_map[sn]] = True
        assert bool(valid.all()), f"Incomplete: {valid.sum()}/{len(names)}"
        out_p = args.out_npz or os.path.join(out_dir, "ntg_tile_peaks_target.npz")
        np.savez_compressed(out_p, names=np.array(names), stats=stats, stat_names=np.array(TILE_STAT_NAMES))
        print(f"[ntg_tile_peak_probe] merged target {stats.shape} -> {out_p}")
        return

    import torch
    from torch.utils.data import DataLoader
    from native_tiles import NativeTileTower
    from qingyu.config_and_paths import REPO_ROOT, LosslessNativeImageLoader, setup_sys_path_and_cwd
    from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM, DualEnginePhysicsExtractor, PrecomputedDualEngineStore
    from qingyu.train_tier1_5backbones_8fold import Tier1DualEngineDataset, make_tier1_collate_fn

    setup_sys_path_and_cwd(chdir_repo=True)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "oof":
        names_all, _, fold_arr, _ = c.load_split()
        sub_names = [str(names_all[i]) for i in range(len(names_all)) if int(fold_arr[i]) == int(args.fold)]
        ckpt_path = os.path.join(REPO, f"results/next55/towers/r55_ntg_s2026/pe_core_large/fold_{args.fold}_best.pt")
        out_p = args.out_npz or os.path.join(out_dir, f"oof_fold_{args.fold}.npz")
        meta_npz = os.path.join(REPO, "results/qingyu/dual_engine_meta.npz")
    else:
        names_all = c.load_template_names(args.template)
        sub_names = [str(names_all[i]) for i in range(len(names_all)) if (i % args.num_shards) == args.shard]
        ckpt_path = os.path.join(REPO, "results/r55/fullfit_v5_ntg/pe_core_large/fold_0_best.pt")
        out_p = args.out_npz or os.path.join(out_dir, f"target_shard_{args.shard:02d}_of_{args.num_shards:02d}.npz")
        meta_npz = "__no_precomputed_meta__.npz"

    assert os.path.exists(ckpt_path), f"Missing {ckpt_path}"
    loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=[args.img_dir] if args.img_dir else None)
    store = PrecomputedDualEngineStore(meta_npz, lazy_extractor=DualEnginePhysicsExtractor(device=str(dev), allow_depth_download=False))

    model = NativeTileTower(backbone_key="pe_core_large", phys_dim=PHYS_FEAT_DIM, use_global=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    _, unexp = model.load_state_dict(sd, strict=False)
    assert not unexp, unexp[:5]
    if dev.type == "cuda":
        model.backbone.to(torch.bfloat16)
        for n, mod in model.backbone.named_modules():
            if any(t in n.split(".")[-1] for t in ("tap3", "dual_tap", "abmil", "spatial_proj")):
                mod.float()
    model.to(dev).eval()

    rows = [{"name": nm, "group_id": nm, "grades": [0] * 6} for nm in sub_names]
    collate = make_tier1_collate_fn("native_tiles", None)
    Tier1DualEngineDataset.TILE_GLOBAL = True

    t0 = time.time()
    stats_acc = np.zeros((len(sub_names), 6, N_TILE_STATS), dtype=np.float64)
    for flip in (False, True):
        ds = Tier1DualEngineDataset(rows, loader, store, "native_tiles", train=False, hflip_tta=flip)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
        for idx_i, batch in enumerate(dl):
            _, tp, ts, tw = infer_ntg_pe_with_tile_capture(model, batch, dev)
            stats_acc[idx_i] += 0.5 * summarize_per_tile_outputs(tp, ts, tw)
            if flip is False and ((idx_i + 1) % 100 == 0 or (idx_i + 1) == len(sub_names)):
                print(f"  [ntg_tile_peak:{args.mode}] flip0 ({idx_i+1}/{len(sub_names)}) {(time.time()-t0)/(idx_i+1):.2f}s/img", flush=True)

    np.savez_compressed(
        out_p,
        names=np.array(sub_names),
        stats=stats_acc.astype(np.float32),
        stat_names=np.array(TILE_STAT_NAMES),
    )
    print(f"[ntg_tile_peak_probe] saved {out_p} ({len(sub_names)} imgs, {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
