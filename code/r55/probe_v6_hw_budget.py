#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/probe_v6_hw_budget.py
==============================
Empirical GPU verification probes for v6 breakthrough candidates:
  --mode vlm   : Verifies Pillar2 (Clef-Flash-9B) & Pillar1 (Qwen3-VL-8B) 4090 merged-DoRA NF4
                 quantization fidelity (TV vs BF16 reference), exact resident/peak VRAM, and speed.
  --mode iv6b  : Verifies InternViT-6B SDPA monkey-patch training speedup on H200 AND
                 4090 inference VRAM + BF16 vs NF4 quantization fidelity.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BNB_PATH = os.path.join(REPO, "third_party", "pydeps")
for _p in (BNB_PATH, os.path.join(REPO, "code"), os.path.join(REPO, "code/next55"), os.path.join(REPO, "code/r55")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


def patch_internvit_sdpa(model) -> int:
    """Monkey-patches InternAttention instances at runtime to use PyTorch SDPA instead of eager matmul."""
    import types
    import torch
    import torch.nn.functional as F

    def _sdpa_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        dp = float(self.attn_drop.p) if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, scale=float(self.scale))
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    n_patched = 0
    for mod in model.modules():
        if mod.__class__.__name__ == "InternAttention":
            mod._naive_attn = types.MethodType(_sdpa_attn, mod)
            mod.use_flash_attn = False
            n_patched += 1
    return n_patched


def run_vlm_probe(n_imgs: int = 20) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from phoenix.config import PHOENIX_CKPT_DIR, resolve_local_or_hf_model
    from phoenix.native_physics_and_views import (
        CoCDepthDefocusProbe,
        compute_1to1_wavelet_and_artifact_maps,
        resize_native_aspect_pil,
        select_scrutiny_1to1_patches,
    )
    from phoenix.train_pillar12_native_vlm import NativeVLMPillarModel, detect_lora_rank_from_state_dict
    from phoenix_pillar_4090 import build_pillar_for_4090

    dev = torch.device("cuda:0")
    z1 = np.load(os.path.join(REPO, "results/r55/phoenix_fidelity_pillar1.npz"), allow_pickle=True)
    names = [str(x) for x in z1["names"][:n_imgs]]
    img_dir = os.path.join(REPO, "data/r55_aux/images")

    print(f"[VLM Probe] Precomputing 3-view inputs for {len(names)} frames...", flush=True)
    probe = CoCDepthDefocusProbe(model_path=resolve_local_or_hf_model("pillar3_depth_anything_s"), device="cuda:0")
    views_cache = []
    t_prep = time.time()
    for nm in names:
        pil = Image.open(os.path.join(img_dir, nm)).convert("RGB")
        maps = compute_1to1_wavelet_and_artifact_maps(np.asarray(pil, dtype=np.uint8))
        coc = probe.compute_coc_jacobian_field(pil, maps)
        crops, _ = select_scrutiny_1to1_patches(pil_img=pil, phys_maps=maps, coc_field=coc, patch_size=384, num_patches=2)
        g = resize_native_aspect_pil(pil, max_pixels=784 * 784, grid_factor=28)
        views_cache.append([g, crops[0], crops[1]])
    prep_sec = (time.time() - t_prep) / len(names)
    del probe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[VLM Probe] View prep (Wavelet+CoC): {prep_sec:.3f} s/img", flush=True)

    for pillar in ("pillar2", "pillar1"):
        print(f"\n=== Testing {pillar} ===", flush=True)
        ckpt_path = os.path.join(str(PHOENIX_CKPT_DIR), f"{pillar}_fulldata_swa.pt")
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck.get("state_dict", ck)
        rank = detect_lora_rank_from_state_dict(ck, default_rank=32)

        # 1. Exact BF16 training-time reference
        m_ref = NativeVLMPillarModel(pillar_name=pillar, quantize_4bit_infer=False, apply_dora=True, lora_rank=rank).to(dev)
        res = m_ref.load_state_dict(sd, strict=False)
        bad = [k for k in res.missing_keys if "lora" in k.lower() or k.startswith("head")] + list(res.unexpected_keys)
        assert not bad, f"{pillar} missing/unexpected: {bad[:5]}"
        m_ref.eval()
        n_params = sum(p.numel() for p in m_ref.parameters())
        bf16_res_gb = torch.cuda.memory_allocated(dev) / (1024 ** 3)
        ref_ps = []
        with torch.no_grad():
            for vs in views_cache:
                o = m_ref([vs], batch_ids=None, apply_batch_bias=False, device=dev)
                ref_ps.append(o["probs_6x4"].float().cpu().numpy()[0])
        p_ref = np.stack(ref_ps, axis=0)
        del m_ref
        gc.collect()
        torch.cuda.empty_cache()

        # 2. Merged DoRA + NF4 4-bit LLM (4090 path)
        torch.cuda.reset_peak_memory_stats(dev)
        m_4090 = build_pillar_for_4090(pillar, ckpt_path=ckpt_path, device="cuda:0", bnb_path=BNB_PATH, four_bit=True, verbose=True)
        build_peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        resident_gb = torch.cuda.memory_allocated(dev) / (1024 ** 3)

        # Warmup 2
        with torch.no_grad():
            for vs in views_cache[:2]:
                _ = m_4090([vs], batch_ids=None, apply_batch_bias=False, device=dev)
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        q4_ps = []
        with torch.no_grad():
            for vs in views_cache:
                o = m_4090([vs], batch_ids=None, apply_batch_bias=False, device=dev)
                q4_ps.append(o["probs_6x4"].float().cpu().numpy()[0])
        torch.cuda.synchronize(dev)
        fwd_sec = (time.time() - t0) / len(views_cache)
        fwd_peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        p_q4 = np.stack(q4_ps, axis=0)

        diff = np.abs(p_q4 - p_ref)
        mean_tv = float(0.5 * diff.sum(axis=-1).mean())
        max_dp = float(diff.max())
        grade_agree = float((p_q4.argmax(-1) == p_ref.argmax(-1)).mean())
        print(
            f"[{pillar} SUMMARY] total_params={n_params/1e9:.3f}B | bf16_res={bf16_res_gb:.2f}GB | "
            f"4090_q4_resident={resident_gb:.2f}GB | 4090_fwd_peak={fwd_peak_gb:.2f}GB (build_peak={build_peak_gb:.2f}GB) | "
            f"1x_fwd={fwd_sec:.3f}s/img (2x_TTA={2*fwd_sec:.3f}s/img) | "
            f"mean_TV={mean_tv:.4f} | max|dp|={max_dp:.4f} | argmax_agree={grade_agree*100:.2f}%",
            flush=True,
        )
        del m_4090
        gc.collect()
        torch.cuda.empty_cache()


def quantize_iv6b_blocks_nf4(backbone_merged, dev) -> int:
    import bitsandbytes as bnb
    import torch

    n = 0
    for name, mod in list(backbone_merged.named_modules()):
        if "encoder.layers." not in name:
            continue
        for cname, child in list(mod.named_children()):
            if type(child) is torch.nn.Linear and cname in ("qkv", "proj", "fc1", "fc2"):
                q = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                    compress_statistics=True,
                )
                q.weight = bnb.nn.Params4bit(
                    child.weight.data.float().cpu(),
                    requires_grad=False,
                    quant_type="nf4",
                    compress_statistics=True,
                )
                if child.bias is not None:
                    q.bias = torch.nn.Parameter(child.bias.data.clone(), requires_grad=False)
                setattr(mod, cname, q.to(dev))
                n += 1
    return n


def run_iv6b_probe() -> None:
    import numpy as np
    import torch
    from train_p12_specialist import P12SemanticRoiTower

    dev = torch.device("cuda:0")
    print("[IV6B Probe] Building P12SemanticRoiTower('internvit_6b', lora_rank=24)...", flush=True)
    t0 = time.time()
    tower = P12SemanticRoiTower(backbone_key="internvit_6b", lora_rank=24).to(dev)
    raw_m = tower.backbone.base_model.model if hasattr(tower.backbone, "base_model") else tower.backbone
    npatch = patch_internvit_sdpa(raw_m)

    # Freeze LoRA in bottom 27 layers (0..26), keep LoRA active in top 18 layers (27..44), remove emb grad hook!
    raw_m.embeddings._forward_hooks.clear()
    for n, p in tower.backbone.named_parameters():
        if "lora_" in n:
            # Extract layer index
            parts = n.split(".")
            lyr = -1
            for i, pt in enumerate(parts):
                if pt == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    lyr = int(parts[i + 1])
                    break
            p.requires_grad = (lyr >= 27)
    raw_m.encoder.gradient_checkpointing = False

    # Wrap _encode_single_view in BF16 autocast + chunk_sz=20
    orig_single = tower._encode_single_view
    def _bf16_single(x_in):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return orig_single(x_in)
    tower._encode_single_view = _bf16_single
    tower._encode_chunked = lambda flat, _csz: (
        tower._encode_single_view(flat)
        if flat.shape[0] <= 20
        else torch.cat([tower._encode_single_view(flat[s : s + 20]) for s in range(0, flat.shape[0], 20)], dim=0)
    )

    n_total = sum(p.numel() for p in tower.parameters())
    n_train = sum(p.numel() for p in tower.parameters() if p.requires_grad)
    print(
        f"[IV6B Probe] Loaded in {time.time()-t0:.1f}s | patched_sdpa={npatch} | "
        f"total={n_total/1e9:.3f}B | trainable_top18={n_train/1e6:.2f}M",
        flush=True,
    )

    B, V = 4, 5
    x = torch.rand(B, V, 3, 448, 448, device=dev, dtype=torch.float32)
    phys = torch.randn(B, 48, device=dev, dtype=torch.float32)
    tower.train()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)

    tower.zero_grad(set_to_none=True)
    out = tower(views_01=x, phys_feat=phys)
    (out["logits_6x4"].sum() + out["z_norm"].sum()).backward()
    torch.cuda.synchronize(dev)

    torch.cuda.reset_peak_memory_stats(dev)
    t_s = time.time()
    n_steps = 5
    for _ in range(n_steps):
        tower.zero_grad(set_to_none=True)
        out = tower(views_01=x, phys_feat=phys)
        (out["logits_6x4"].sum() + out["z_norm"].sum()).backward()
    torch.cuda.synchronize(dev)
    step_s = (time.time() - t_s) / n_steps
    peak_gb = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    est_ep_min = (793 * step_s) / 60.0
    print(
        f"[IV6B Train Bench: sdpa_bf16_top18LoRA_gcOFF_chunk20] step={step_s:.3f}s (B=4,V=5) | "
        f"peak_VRAM={peak_gb:.2f}GB | est_1fold_epoch={est_ep_min:.1f} min",
        flush=True,
    )

    # Benchmark 2: Inference BF16 merged vs NF4 quantized (10 views = 5 views x 2x TTA, chunked by 5 views = 1 orientation)
    tower.eval()
    tower.backbone = tower.backbone.merge_and_unload()
    raw_m = tower.backbone
    patch_internvit_sdpa(raw_m)
    raw_m.encoder.gradient_checkpointing = False
    tower._encode_chunked = lambda flat, _csz: (
        tower._encode_single_view(flat)
        if flat.shape[0] <= 5
        else torch.cat([tower._encode_single_view(flat[s : s + 5]) for s in range(0, flat.shape[0], 5)], dim=0)
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    bf16_res = torch.cuda.memory_allocated(dev) / (1024 ** 3)

    x_inf = [torch.rand(2, 5, 3, 448, 448, device=dev) for _ in range(10)]
    p_inf = [torch.randn(2, 48, device=dev) for _ in range(10)]
    with torch.no_grad():
        _ = tower(views_01=x_inf[0], phys_feat=p_inf[0])
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    t_i = time.time()
    ref_probs, ref_zn = [], []
    with torch.no_grad():
        for xi, pi in zip(x_inf, p_inf):
            o = tower(views_01=xi, phys_feat=pi)
            ref_probs.append(o["probs_6x4"].mean(0).cpu().numpy())
            ref_zn.append(o["z_norm"].mean(0).cpu().numpy())
    torch.cuda.synchronize(dev)
    bf16_sec = (time.time() - t_i) / len(x_inf)
    bf16_peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    ref_probs = np.stack(ref_probs)
    ref_zn = np.stack(ref_zn)
    print(
        f"[IV6B Infer BF16] resident={bf16_res:.2f}GB | peak={bf16_peak:.2f}GB | 10-view(2xTTA)={bf16_sec:.3f}s/img",
        flush=True,
    )

    nq = quantize_iv6b_blocks_nf4(tower.backbone, dev)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    q4_res = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    with torch.no_grad():
        _ = tower(views_01=x_inf[0], phys_feat=p_inf[0])
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    t_q = time.time()
    q4_probs, q4_zn = [], []
    with torch.no_grad():
        for xi, pi in zip(x_inf, p_inf):
            o = tower(views_01=xi, phys_feat=pi)
            q4_probs.append(o["probs_6x4"].mean(0).cpu().numpy())
            q4_zn.append(o["z_norm"].mean(0).cpu().numpy())
    torch.cuda.synchronize(dev)
    q4_sec = (time.time() - t_q) / len(x_inf)
    q4_peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    q4_probs = np.stack(q4_probs)
    q4_zn = np.stack(q4_zn)
    cos = float((ref_zn * q4_zn).sum(axis=-1).mean())
    tv = float(0.5 * np.abs(ref_probs - q4_probs).sum(axis=-1).mean())
    print(
        f"[IV6B Infer NF4 ({nq} linears)] resident={q4_res:.2f}GB | peak={q4_peak:.2f}GB | "
        f"10-view(2xTTA)={q4_sec:.3f}s/img | z_norm_cos={cos:.5f} | untuned_head_TV={tv:.4f}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vlm", "iv6b"], required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    if args.mode == "vlm":
        run_vlm_probe(n_imgs=args.n)
    else:
        run_iv6b_probe()


if __name__ == "__main__":
    main()
