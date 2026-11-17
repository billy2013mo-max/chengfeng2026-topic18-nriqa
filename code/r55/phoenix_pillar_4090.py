#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/phoenix_pillar_4090.py — build a Phoenix VLM pillar (pillar1 / pillar2) for a 24 GB card WITHOUT changing
the function that was trained.

Why this exists: code/phoenix/deploy_single_4090_engine.py builds the pillars with `quantize_4bit_infer=True`.
That path quantises at load time, creates plain LoRA modules instead of DoRA, and cannot load the trained merger
weights; forced to load, 73 % of images change grade against the trained model (results/r55/phoenix_fidelity_pillar1.json).

Correct order, implemented here:
  1. build the pillar exactly as in training (bf16 base + DoRA adapters) and load the checkpoint (strict on adapters/head);
  2. merge the adapters into the base weights (`merge_and_unload`);
  3. replace every nn.Linear of the language model by a 4-bit NF4 layer holding the merged weight
     (vision tower, merger and the head stay in bf16 / fp32).
Measured on 120 unseen frames for pillar1: 5.6 GB, mean total-variation distance to the trained function 0.009
(bf16 merge alone: 0.003; dropping the DoRA magnitudes: 0.017; the engine's path: 0.16).

`bitsandbytes` is not in the shared environment: pass `bnb_path` (a `pip install --no-deps --target <dir> bitsandbytes`
directory) or install it.  Peak memory while building is the bf16 model (~16 GB), then it drops to ~6 GB.

Usage from an inference script:
    from phoenix_pillar_4090 import build_pillar_for_4090
    pillar = build_pillar_for_4090("pillar1", device="cuda:0", bnb_path="/path/to/pydeps")
    out = pillar([[global_view, crop0, crop1]], batch_ids=None, apply_batch_bias=False, device=torch.device("cuda:0"))

Self-check (compares with the stored fidelity run on the same frames):
    python code/r55/phoenix_pillar_4090.py --pillar pillar1 --gpu 1 --bnb-path <dir> --n 20
"""
import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code"))

SKIP = ("visual", "vision_tower", "vision_model", "mlp1", "merger")      # kept in bf16


def quantize_llm_inplace(model, device):
    """Replace every plain nn.Linear outside the vision tower / merger by a 4-bit NF4 layer with the same weights."""
    import bitsandbytes as bnb
    import torch
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if type(child) is torch.nn.Linear and not any(s in full for s in SKIP):
                q = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=child.bias is not None,
                                      compute_dtype=torch.bfloat16, quant_type="nf4", compress_statistics=True)
                q.weight = bnb.nn.Params4bit(child.weight.data.float().cpu(), requires_grad=False, quant_type="nf4", compress_statistics=True)
                if child.bias is not None:
                    q.bias = torch.nn.Parameter(child.bias.data.clone(), requires_grad=False)
                setattr(mod, cname, q.to(device))
                n += 1
    return n


def build_pillar_for_4090(pillar="pillar1", ckpt_path=None, device="cuda", bnb_path=None, four_bit=True, verbose=True):
    """Trained pillar -> merged -> (optionally) 4-bit language model.  Returns the module in eval mode on `device`."""
    import torch
    if bnb_path and bnb_path not in sys.path:
        sys.path.append(bnb_path)
    from phoenix.config import PHOENIX_CKPT_DIR
    from phoenix.train_pillar12_native_vlm import NativeVLMPillarModel, detect_lora_rank_from_state_dict

    dev = torch.device(device)
    ck = torch.load(ckpt_path or os.path.join(str(PHOENIX_CKPT_DIR), f"{pillar}_fulldata_swa.pt"), map_location="cpu")
    sd = ck.get("state_dict", ck)
    m = NativeVLMPillarModel(pillar_name=pillar, quantize_4bit_infer=False, apply_dora=True,
                             lora_rank=detect_lora_rank_from_state_dict(ck, default_rank=32)).to(dev)
    res = m.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if "lora" in k.lower() or k.startswith("head")] + list(res.unexpected_keys)
    assert not bad, f"checkpoint does not match the training-time model: {bad[:5]}"
    m.backbone = m.backbone.merge_and_unload()
    nq = quantize_llm_inplace(m.backbone, dev) if four_bit else 0
    torch.cuda.empty_cache()
    m.eval()
    if verbose:
        print(f"[phoenix_pillar_4090] {pillar}: adapters merged, {nq} linears in 4-bit, {torch.cuda.memory_allocated(dev) / 2 ** 30:.2f} GB on {dev}", flush=True)
    return m


def _self_check():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pillar", default="pillar1")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bnb-path", default="")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--img-dir", default="data/r55_aux/images")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.chdir(REPO)
    import time

    import numpy as np
    import torch
    from PIL import Image
    from phoenix.config import resolve_local_or_hf_model
    from phoenix.native_physics_and_views import (CoCDepthDefocusProbe, compute_1to1_wavelet_and_artifact_maps,
                                                  resize_native_aspect_pil, select_scrutiny_1to1_patches)
    z = np.load(os.path.join(REPO, f"results/r55/phoenix_fidelity_{a.pillar}.npz"), allow_pickle=True)
    names = [str(x) for x in z["names"][:a.n]]
    dev = torch.device("cuda")
    probe = CoCDepthDefocusProbe(model_path=resolve_local_or_hf_model("pillar3_depth_anything_s"), device=str(dev))
    m = build_pillar_for_4090(a.pillar, device="cuda", bnb_path=a.bnb_path or None)
    ps, t0 = [], time.time()
    with torch.no_grad():
        for n in names:
            pil = Image.open(os.path.join(a.img_dir, n)).convert("RGB")
            maps = compute_1to1_wavelet_and_artifact_maps(np.asarray(pil, dtype=np.uint8))
            crops, _ = select_scrutiny_1to1_patches(pil_img=pil, phys_maps=maps, coc_field=probe.compute_coc_jacobian_field(pil, maps),
                                                    patch_size=384, num_patches=2)
            o = m([[resize_native_aspect_pil(pil, max_pixels=784 * 784, grid_factor=28), crops[0], crops[1]]],
                  batch_ids=None, apply_batch_bias=False, device=dev)
            ps.append(o["probs_6x4"].float().cpu().numpy()[0])
    p = np.stack(ps)
    sec = (time.time() - t0) / len(names)
    for k in ("probs_q4merged", "probs_ref"):
        d = np.abs(p - z[k][:a.n])
        print(f"[self-check] vs stored {k[6:]:9s}: mean TV {0.5 * d.sum(-1).mean():.4f}  max|dp| {d.max():.3f}")
    print(f"[self-check] {sec:.2f} s per image for this pillar alone (views + one forward, no flip; shared H200, not a 4090), "
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB (includes the bf16 build)")


if __name__ == "__main__":
    _self_check()
