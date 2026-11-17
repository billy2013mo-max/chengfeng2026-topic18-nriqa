#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/phoenix_deploy_fidelity.py — is the Phoenix VLM pillar that deploy_single_4090_engine.py runs the same
function as the one that was trained and validated?  (1 GPU, label-free, no test image)

The pillars were trained and cross-validated in bf16 with DoRA adapters.  The single-4090 engine builds them with
`quantize_4bit_infer=True`, which (train_pillar12_native_vlm.py::_load_vlm_backbone)
  * loads the language model in 4-bit NF4, and
  * creates plain LoRA adapters (`use_dora = not quantize_4bit_infer`), so every `lora_magnitude_vector` of the
    checkpoint is dropped by `load_state_dict(strict=False)`.

This script loads one full-data pillar checkpoint in up to four configurations and compares the output
probabilities on the same unlabelled in-domain frames (KVQ pool, never used for Phoenix training):

  ref       bf16 + DoRA                       the trained function
  merged    ref with the adapters merged      sanity check (must equal ref up to rounding)
  q4merged  merged, then the language model   the correct way to get a 4-bit model of the trained function:
            quantised to 4-bit NF4 in place   only quantisation error remains
  lora      bf16 + LoRA                       isolates the dropped DoRA magnitudes
  deploy    4-bit + LoRA                      what the engine builds today; if the checkpoint does not load
                                              (size mismatch), tensors of the wrong shape are dropped and the
                                              mode is reported as "deploy (mismatched tensors dropped)"

Reported against `ref`: mean / max |dp|, share of images whose argmax grade changes in any dimension, per-dimension
argmax agreement, mean total-variation distance.  Views are built exactly as in the engine (global + 2 crops, no flip).

`bitsandbytes` is not in the shared environment; point --bnb-path at an isolated `pip install --target` directory.

Usage:
  python code/r55/phoenix_deploy_fidelity.py --pillar pillar1 --gpu 1 --n 120 --bnb-path <dir> [--modes ref merged q4merged lora deploy]
Output: results/r55/phoenix_fidelity_<pillar>.npz / .json
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "code"))

MODES = {"ref": (False, True), "deploy": (True, False), "lora": (False, False)}   # (4-bit at load, DoRA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pillar", default="pillar1", choices=["pillar1", "pillar2"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--modes", nargs="+", default=["ref", "merged", "q4merged", "lora", "deploy"])
    ap.add_argument("--bnb-path", default="")
    ap.add_argument("--frames-csv", default="data/r55_aux/kvq_frames.csv")
    ap.add_argument("--img-dir", default="data/r55_aux/images")
    a = ap.parse_args()
    if a.bnb_path:
        sys.path.append(a.bnb_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.chdir(REPO)

    import peft
    import torch
    from PIL import Image
    from phoenix.config import PHOENIX_CKPT_DIR, resolve_local_or_hf_model
    from phoenix.native_physics_and_views import (CoCDepthDefocusProbe, compute_1to1_wavelet_and_artifact_maps,
                                                  resize_native_aspect_pil, select_scrutiny_1to1_patches)
    from phoenix.train_pillar12_native_vlm import NativeVLMPillarModel, detect_lora_rank_from_state_dict

    dev = torch.device("cuda")
    rows = list(csv.DictReader(open(a.frames_csv, encoding="utf-8")))
    rs = np.random.RandomState(0)
    pick = [rows[i]["name"] for i in sorted(rs.choice(len(rows), a.n, replace=False))]

    # views exactly as Single4090PhoenixEngine._forward_one_orientation (built once, shared by every mode)
    probe = CoCDepthDefocusProbe(model_path=resolve_local_or_hf_model("pillar3_depth_anything_s"), device=str(dev))
    views = []
    for n in pick:
        pil = Image.open(os.path.join(a.img_dir, n)).convert("RGB")
        maps = compute_1to1_wavelet_and_artifact_maps(np.asarray(pil, dtype=np.uint8))
        coc = probe.compute_coc_jacobian_field(pil, maps)
        crops, _ = select_scrutiny_1to1_patches(pil_img=pil, phys_maps=maps, coc_field=coc, patch_size=384, num_patches=2)
        views.append([resize_native_aspect_pil(pil, max_pixels=784 * 784, grid_factor=28), crops[0], crops[1]])
    del probe
    print(f"[fidelity] {a.pillar}: {len(views)} frames prepared", flush=True)

    ck = torch.load(os.path.join(str(PHOENIX_CKPT_DIR), f"{a.pillar}_fulldata_swa.pt"), map_location="cpu")
    sd = ck.get("state_dict", ck)
    rank = detect_lora_rank_from_state_dict(ck, default_rank=32)
    n_mag = sum("lora_magnitude_vector" in k for k in sd)
    print(f"[fidelity] checkpoint: {len(sd)} tensors, LoRA rank {rank}, {n_mag} DoRA magnitude vectors", flush=True)

    real_cfg = peft.LoraConfig
    out, info = {}, {}

    def predict(m):
        m.eval()
        ps = []
        with torch.no_grad():
            for v in views:
                o = m([v], batch_ids=None, apply_batch_bias=False, device=dev)
                ps.append(o["probs_6x4"].float().cpu().numpy()[0])
        return np.stack(ps)

    def quantize_llm_inplace(model):
        """Replace every nn.Linear outside the vision tower / merger by a 4-bit NF4 layer holding the same weights."""
        import bitsandbytes as bnb
        n = 0
        for name, mod in list(model.named_modules()):
            for cname, child in list(mod.named_children()):
                full = f"{name}.{cname}" if name else cname
                if type(child) is torch.nn.Linear and not any(s in full for s in ("visual", "vision_tower", "vision_model", "mlp1", "merger")):
                    q = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=child.bias is not None,
                                          compute_dtype=torch.bfloat16, quant_type="nf4", compress_statistics=True)
                    q.weight = bnb.nn.Params4bit(child.weight.data.float().cpu(), requires_grad=False, quant_type="nf4", compress_statistics=True)
                    if child.bias is not None:
                        q.bias = torch.nn.Parameter(child.bias.data.clone(), requires_grad=False)
                    setattr(mod, cname, q.to(dev))
                    n += 1
        return n

    for mode in [x for x in a.modes if x in MODES]:
        q4, dora = MODES[mode]

        def forced(*args, _d=dora, **kw):                 # the builder hard-codes use_dora = not quantize_4bit_infer
            kw["use_dora"] = _d
            return real_cfg(*args, **kw)
        peft.LoraConfig = forced
        try:
            m = NativeVLMPillarModel(pillar_name=a.pillar, quantize_4bit_infer=q4, apply_dora=True, lora_rank=rank).to(dev)
        except Exception as e:
            print(f"[fidelity] mode {mode}: could not build ({type(e).__name__}: {str(e)[:200]})", flush=True)
            continue
        finally:
            peft.LoraConfig = real_cfg
        label, use_sd, dropped = mode, sd, 0
        try:
            res = m.load_state_dict(use_sd, strict=False)
        except RuntimeError as e:                         # what the engine would hit: checkpoint tensors of another shape
            cur = m.state_dict()
            bad = [k for k, v in sd.items() if k in cur and tuple(cur[k].shape) != tuple(v.shape)]
            print(f"[fidelity] mode {mode}: load_state_dict FAILS as in the engine ({len(bad)} size mismatches, e.g. {bad[0]}: "
                  f"checkpoint {tuple(sd[bad[0]].shape)} vs model {tuple(cur[bad[0]].shape)})", flush=True)
            use_sd = {k: v for k, v in sd.items() if k not in set(bad)}
            dropped, label = len(bad), mode + "_dropmismatch"
            res = m.load_state_dict(use_sd, strict=False)
        miss = [k for k in res.missing_keys if "lora" in k.lower() or k.startswith("head")]
        unexp = list(res.unexpected_keys)
        info[label] = dict(four_bit=q4, dora=dora, unexpected=len(unexp), unexpected_magnitude=sum("magnitude" in k for k in unexp),
                           missing_adapter_or_head=len(miss), dropped_size_mismatch=dropped, gpu_gb=round(torch.cuda.memory_allocated() / 2 ** 30, 2))
        print(f"[fidelity] mode {label}: 4bit={q4} dora={dora} | unexpected keys {len(unexp)} (magnitude {info[label]['unexpected_magnitude']}) "
              f"| missing adapter/head keys {len(miss)} | dropped for shape {dropped} | {info[label]['gpu_gb']} GB", flush=True)
        out[label] = predict(m)
        if mode == "ref" and ("merged" in a.modes or "q4merged" in a.modes):
            try:
                m.backbone = m.backbone.merge_and_unload()
                out["merged"] = predict(m)
                info["merged"] = dict(gpu_gb=round(torch.cuda.memory_allocated() / 2 ** 30, 2))
                print(f"[fidelity] mode merged: adapters merged into the base weights | {info['merged']['gpu_gb']} GB", flush=True)
                if "q4merged" in a.modes:
                    nq = quantize_llm_inplace(m.backbone)
                    torch.cuda.empty_cache()
                    out["q4merged"] = predict(m)
                    info["q4merged"] = dict(quantized_linears=nq, gpu_gb=round(torch.cuda.memory_allocated() / 2 ** 30, 2))
                    print(f"[fidelity] mode q4merged: {nq} language-model linears quantised to 4-bit NF4 | {info['q4merged']['gpu_gb']} GB", flush=True)
            except Exception as e:
                print(f"[fidelity] merged / q4merged failed ({type(e).__name__}: {str(e)[:300]})", flush=True)
        del m
        torch.cuda.empty_cache()

    rep = {"pillar": a.pillar, "n": len(pick), "modes": info, "vs_ref": {}}
    if "ref" in out:
        r = out["ref"]
        for mode, p in out.items():
            if mode == "ref":
                continue
            d = np.abs(p - r)
            flip = (p.argmax(-1) != r.argmax(-1))
            tv = 0.5 * d.sum(-1)
            rep["vs_ref"][mode] = dict(mean_abs=float(d.mean()), max_abs=float(d.max()), tv_mean=float(tv.mean()),
                                       img_any_flip=float(flip.any(1).mean()), flip_per_dim=[float(x) for x in flip.mean(0)])
            print(f"[fidelity] {mode:7s} vs ref: mean|dp| {d.mean():.4f}  max|dp| {d.max():.3f}  mean TV {tv.mean():.4f}  "
                  f"images with a changed grade {flip.any(1).mean() * 100:.1f}%  per-dim flips {np.round(flip.mean(0), 3)}", flush=True)
    os.makedirs(os.path.join(REPO, "results/r55"), exist_ok=True)
    np.savez_compressed(os.path.join(REPO, f"results/r55/phoenix_fidelity_{a.pillar}.npz"), names=np.array(pick), **{f"probs_{k}": v for k, v in out.items()})
    json.dump(rep, open(os.path.join(REPO, f"results/r55/phoenix_fidelity_{a.pillar}.json"), "w"), indent=2)
    print(f"[fidelity] wrote results/r55/phoenix_fidelity_{a.pillar}.json")


if __name__ == "__main__":
    main()
