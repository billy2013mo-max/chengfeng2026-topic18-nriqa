#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/probe_sr_idem.py — E4: public restoration models as a measuring instrument ("idempotence" features).
Run with `python3 -I` (loader library in third_party/pydeps_sr, weights in data/r55_aux/raw/sr_weights, tensors only).

Idea (from generated-image detection by reconstruction): a frame that already went through generative enhancement is
close to a fixed point of such an enhancer.  For every labelled image x and each restoration model G with scale s
    y = G(down_s(x))        b = bicubic_up(down_s(x))
and the features compare how well y and b reproduce x: a frame whose fine detail is "generated-looking" is reproduced
better by G than by interpolation.  Per model: error ratio, high-pass correlation and energy ratios, and the spread of
the error ratio over an 8 x 8 grid (localised generated regions).

This is a refutation test like E1 / E2: `probe_eval.py` decides whether the features add anything to the ensemble's
out-of-fold probabilities.  Only the labelled images are read.

Usage:
  for r in 0..7: CUDA_VISIBLE_DEVICES=$r python3 -I code/r55/probe_sr_idem.py --rank $r --world 8 &
  python3 -I code/r55/probe_sr_idem.py --merge 8
  python code/r55/probe_eval.py --name sr_idem --feats results/r55/probes/sr_idem_feats.npz
"""
import argparse
import csv
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "third_party/pydeps_sr"))
OUT = os.path.join(REPO, "results/r55/probes")
WEI = os.path.join(REPO, "data/r55_aux/raw/sr_weights")
MODELS = ["RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth", "realesr-general-x4v3.pth"]
FEATS = ["err_ratio", "err_sr", "err_bic", "hp_corr_sr", "hp_corr_bic", "hp_energy_sr", "hp_energy_bic", "hp_err_ratio",
         "grid_ratio_min", "grid_ratio_max", "grid_ratio_std"]


def find(name):
    for d in ("data/images", "data/images_supp"):
        p = os.path.join(REPO, d, name)
        if os.path.exists(p):
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--merge", type=int, default=0)
    a = ap.parse_args()
    import numpy as np
    os.makedirs(OUT, exist_ok=True)
    fn = [f"{m.split('.')[0]}:{f}" for m in MODELS for f in FEATS]
    if a.merge:
        N, X = [], []
        for r in range(a.merge):
            z = np.load(os.path.join(OUT, f"sr_idem_shard{r}.npz"), allow_pickle=True)
            N += [str(n) for n in z["names"]]
            X.append(z["X"])
        np.savez_compressed(os.path.join(OUT, "sr_idem_feats.npz"), names=np.array(N), X=np.concatenate(X).astype(np.float32), feat_names=np.array(fn))
        print(f"[sr_idem] merged {len(N)} images x {len(fn)} features")
        return
    import cv2
    import torch
    from spandrel import ModelLoader
    dev = torch.device("cuda")
    nets = []
    for f in MODELS:
        sd = torch.load(os.path.join(WEI, f), map_location="cpu", weights_only=True)
        for k in ("params_ema", "params", "state_dict"):
            if isinstance(sd, dict) and k in sd:
                sd = sd[k]
                break
        m = ModelLoader(device=dev).load_from_state_dict(sd)
        m.eval()
        nets.append(m)
    names = sorted({r["name"] for f in range(8) for r in csv.DictReader(open(os.path.join(REPO, f"data/qingyu_splits/fold_{f}.csv"), encoding="utf-8")) if r["split"] == "val"})[a.rank::a.world]
    hp = lambda t: t - cv2.GaussianBlur(t, (0, 0), 1.5)                  # noqa: E731
    X = np.full((len(names), len(fn)), np.nan, dtype=np.float32)
    for i, n in enumerate(names):
        im = cv2.imdecode(np.fromfile(find(n), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        h, w = h - h % 8, w - w % 8
        im = im[:h, :w]
        x = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hx = hp(x)
        for j, net in enumerate(nets):
            s = net.scale
            small = cv2.resize(im, (w // s, h // s), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(small[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().to(dev) / 255.0
            with torch.no_grad():
                y = net(t).clamp(0, 1)
            y = (y[0].permute(1, 2, 0).float().cpu().numpy()[:, :, ::-1] * 255.0).round().astype(np.uint8)
            if y.shape[:2] != (h, w):
                y = cv2.resize(y, (w, h), interpolation=cv2.INTER_AREA)
            b = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
            yg, bg = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
            es, eb = np.abs(x - yg), np.abs(x - bg)
            hy, hb = hp(yg), hp(bg)
            cc = lambda u, v: float((u * v).mean() / (np.sqrt((u * u).mean() * (v * v).mean()) + 1e-6))     # noqa: E731
            gs = es[:h // 8 * 8, :w // 8 * 8].reshape(8, h // 8, 8, w // 8).mean((1, 3))
            gb = eb[:h // 8 * 8, :w // 8 * 8].reshape(8, h // 8, 8, w // 8).mean((1, 3))
            gr = gs / (gb + 0.25)
            X[i, j * len(FEATS):(j + 1) * len(FEATS)] = [es.mean() / (eb.mean() + 1e-3), es.mean(), eb.mean(), cc(hx, hy), cc(hx, hb),
                                                         np.abs(hy).mean() / (np.abs(hx).mean() + 1e-3), np.abs(hb).mean() / (np.abs(hx).mean() + 1e-3),
                                                         np.abs(hx - hy).mean() / (np.abs(hx - hb).mean() + 1e-3), gr.min(), gr.max(), gr.std()]
        if (i + 1) % 100 == 0:
            print(f"[sr_idem] rank {a.rank}: {i + 1}/{len(names)}", flush=True)
    np.savez_compressed(os.path.join(OUT, f"sr_idem_shard{a.rank}.npz"), names=np.array(names), X=X)
    print(f"[sr_idem] rank {a.rank}: {len(names)} images")


if __name__ == "__main__":
    main()
