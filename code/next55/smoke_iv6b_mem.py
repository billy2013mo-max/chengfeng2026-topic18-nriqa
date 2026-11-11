#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/next55/smoke_iv6b_mem.py — 1-GPU, ~3 min.
Verifies the Tier-2 InternViT-6B OOM fix: one fwd/bwd at the production batch shape
(B=6, V=5, 448x448), reports peak memory and that LoRA weights actually receive gradients.

Usage: CUDA_VISIBLE_DEVICES=0 python code/next55/smoke_iv6b_mem.py [--batch 6]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qingyu.backbones_qingyu import QingyuVisionQualityTower  # noqa: E402
from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--views", type=int, default=5)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    m = QingyuVisionQualityTower(backbone_key="internvit_6b", phys_dim=PHYS_FEAT_DIM, use_lora_6b=True, lora_rank=32).to(dev)
    m.train()
    x = torch.rand(a.batch, a.views, 3, 448, 448, device=dev)
    ph = torch.randn(a.batch, PHYS_FEAT_DIM, device=dev)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = m(views_01=x, phys_feat=ph)
    t = next(v for v in out.values() if torch.is_tensor(v) and v.requires_grad)
    t.float().mean().backward()
    # lora_B is zero-initialised, so lora_A legitimately has zero grad on step 0 -> check lora_B only
    lora = [(n, p) for n, p in m.named_parameters() if "lora_B" in n and p.requires_grad]
    with_grad = sum(1 for _, p in lora if p.grad is not None and p.grad.abs().sum() > 0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[smoke-iv6b] B={a.batch} V={a.views} peak={peak:.1f} GiB | LoRA_B tensors with grad: {with_grad}/{len(lora)} "
          f"| grad_ckpt={os.environ.get('QY_IV6B_GRAD_CKPT', '1')}")
    ok = peak < 120 and with_grad == len(lora) and len(lora) > 0
    print("[smoke-iv6b] PASS" if ok else "[smoke-iv6b] FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
