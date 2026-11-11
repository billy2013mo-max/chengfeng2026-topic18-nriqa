#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/infer_v16_ckpt.py — inference-only for a V16 seed-bag tower (1 GPU per fold, ~5 min/fold).

Rebuilds MultiNetV16 from models/ with the fold's exact --backbones string, loads the
*_swa.pt written by train_v16_microscope_specialist.py --save-ckpt (trainable params + buffers),
runs 2x hflip TTA, and APPENDS ext_p0/ext_p1/ext_p2/ext_gate/ext_names (tag-suffixed copy) to a
separate file  <tower_dir>/ext_<tag>_<fold-name>.npz.  common55.load_target() reads them.

Safety: --verify-va re-predicts the fold's val set and asserts it reproduces stored va_p0 (|Δ|<2e-2).

Usage:
  NEXT55_EXTRA_IMG_DIR=<final_img_dir> CUDA_VISIBLE_DEVICES=0 python -I code/next55/infer_v16_ckpt.py \
      --tower-dir results/next55/towers/v16_s700 --fold 0 --template <final.json> --tag final654 [--verify-va]
"""
import argparse
import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code/finetune"))
sys.path.insert(0, os.path.join(REPO, "code/decision_eval"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_v16_microscope_specialist as v16  # noqa: E402
from extract_native_phys_feats import extract_for_names  # noqa: E402


def run(net, rows, phys, mu, sd, args_tr, dev):
    outs = []
    for hf in (False, True):
        ds = v16.V16NativeMicroscopeDataset(rows, phys, mu, sd, args_tr["size"], False, hflip=hf, tiles=args_tr["tiles"])
        outs.append(v16.predict_v16(net, DataLoader(ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True), dev, args_tr["alpha_ex"]))
    return [0.5 * (outs[0][i] + outs[1][i]) for i in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower-dir", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--verify-va", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    ck_path = glob.glob(os.path.join(a.tower_dir, f"V16_Full_F{a.fold}_*_swa.pt"))
    assert len(ck_path) == 1, ck_path
    ck = torch.load(ck_path[0], map_location="cpu", weights_only=False)
    tr = ck["args"]

    specs = []
    for item in tr["backbones"].split(","):
        nm, _ = item.split(":")
        specs.append(v16.build_v9_backbone(nm))
    net = v16.MultiNetV16(specs, phys_dim=60, dropout=tr["dropout"], n_quad=tr["tiles"] * tr["tiles"]).to(dev)
    missing, unexpected = net.load_state_dict({k: v.to(dev) for k, v in ck["state_dict"].items()}, strict=False)
    assert not unexpected, unexpected[:5]
    net.eval()
    print(f"[infer_v16_ckpt] params={sum(p.numel() for p in net.parameters()) / 1e9:.3f}B backbones={tr['backbones']}", flush=True)

    base = ck_path[0][: -len("_swa.pt")]
    if a.verify_va:
        ref = np.load(base + "_probs.npz", allow_pickle=True)
        vn = [str(x) for x in ref["va_names"]]
        phys_va = extract_for_names(vn, tr["phys_cache"])
        rows = [{"name": n, **{c: 0 for c in v16.CLASSES}} for n in vn]
        p0 = run(net, rows, phys_va, ck["phys_mu"], ck["phys_sd"], tr, dev)[0]
        d = float(np.max(np.abs(p0 - ref["va_p0"])))
        print(f"  [verify] fold {a.fold}: max|Δ va_p0| = {d:.4g}", flush=True)
        assert d < 2e-2, f"reload does not reproduce stored OOF (Δ={d})"

    names = [e["image"] for e in json.load(open(a.template))]
    phys = extract_for_names(names, tr["phys_cache"])
    n_zero = sum(1 for n in names if not np.any(phys[n]))
    assert n_zero == 0, f"{n_zero} images not found -> set NEXT55_EXTRA_IMG_DIR"
    rows = [{"name": n, **{c: 0 for c in v16.CLASSES}} for n in names]
    p0, p1, p2, g = run(net, rows, phys, ck["phys_mu"], ck["phys_sd"], tr, dev)
    out = os.path.join(a.tower_dir, f"ext_{a.tag}_F{a.fold}.npz")
    np.savez(out, ext_p0=p0, ext_p1=p1, ext_p2=p2, ext_gate=g, ext_names=np.array(names))
    print(f"[infer_v16_ckpt] wrote {out}")


if __name__ == "__main__":
    main()
