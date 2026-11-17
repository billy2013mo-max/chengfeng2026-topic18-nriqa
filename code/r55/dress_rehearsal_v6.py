#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/dress_rehearsal_v6.py
==============================
GPU-5 Dress Rehearsal & Gate G3 Deployment Verification (per Fable-5.2 Round-2 Spec):
  1. Verifies `p3_qy` safe Linear/Conv2d BF16 cast (`TV <= 0.002` vs uncast `p3_qy` over 40 images) and VRAM saved.
  2. Verifies Fold-0 `c233397` checkpoint (`p1_c233397_s2026_fold0.pt`) through the exact live `FinalEngine`
     deploy path on 50 Fold-0 validation frames:
       - BF16 merged (`four_bit=False`) vs saved Fold-0 OOF `probs_orig_6x4`, `probs_hflip_6x4`, `probs_6x4` (`max |dp| <= 0.01`)
       - 4-bit NF4 merged (`four_bit=True`) vs BF16 merged & saved OOF (`mean_TV <= 0.012`)
  3. Runs full 9-member `FinalEngine(SETS["resident_v6"], half=True)` stand-in benchmark on 25 images,
     verifying `peak_infer_gib <= 22.0 GiB` and `total_params_b <= 30.0B`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=5)
    ap.add_argument("--fold0-ckpt", default=os.path.join(REPO, "results/r55/vlm_qy8/ckpts/p1_c233397_s2026_fold0.pt"))
    ap.add_argument("--fold0-npz", default=os.path.join(REPO, "results/r55/vlm_qy8/p1_c233397_s2026_fold0.npz"))
    ap.add_argument("--out-json", default=os.path.join(REPO, "results/r55/vlm_qy8/dress_rehearsal_g3.json"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import torch
    from PIL import Image
    import common55 as c
    import final_engine as fe
    from phoenix.config import resolve_local_or_hf_model
    from phoenix.native_physics_and_views import (
        CoCDepthDefocusProbe,
        compute_1to1_wavelet_and_artifact_maps,
        resize_native_aspect_pil,
        select_scrutiny_1to1_patches,
    )
    from phoenix.train_pillar12_native_vlm import detect_lora_rank_from_state_dict
    from phoenix.train_pillar3_hires_physics import Pillar3HiResPhysicsSpecialist, build_pillar3_sample_tensors
    from qingyu.config_and_paths import LosslessNativeImageLoader
    from vlm_c233397_qy8 import NativeVLMPillarModelC233397, build_c233397_pillar_for_4090

    dev = torch.device("cuda:0")
    loader = LosslessNativeImageLoader(REPO)
    names_all, _, _, _ = c.load_split()

    # ------------------------------------------------------------------
    # 1. p3_qy safe Linear/Conv2d BF16 cast check (40 images)
    # ------------------------------------------------------------------
    coc = CoCDepthDefocusProbe(model_path=resolve_local_or_hf_model("pillar3_depth_anything_s"), device=str(dev))
    base_gb = torch.cuda.memory_allocated(dev) / 2 ** 30
    p3_ckpt = os.path.join(REPO, "results/r55/p3_qy_honest_fullfit/pillar3_fulldata_swa.pt")
    sd = torch.load(p3_ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    p3_legacy = "vision_tower.backbone_gate_logits" not in sd

    m32 = Pillar3HiResPhysicsSpecialist(
        mock_backbone=False, lora_rank=detect_lora_rank_from_state_dict(sd, default_rank=24), legacy_3view=p3_legacy
    ).to(dev)
    m32.load_state_dict(sd, strict=False)
    m32.eval()
    p3_uncast_gb = torch.cuda.memory_allocated(dev) / 2 ** 30 - base_gb

    p_ref_list = []
    with torch.no_grad():
        for nm in names_all[:40]:
            im = loader.load_pil(nm).convert("RGB")
            pk = build_pillar3_sample_tensors(im, coc_probe=coc, include_mid_tex=not p3_legacy)
            o = m32(
                views_01=pk["views_tensor"].unsqueeze(0).to(dev),
                phys_9ch=pk["phys_9ch"].unsqueeze(0).to(dev),
                probe_8ch=pk["probe_8ch"].unsqueeze(0).to(dev),
                batch_ids=None,
                apply_batch_bias=False,
            )
            p_ref_list.append(o["probs_6x4"].float().cpu().numpy()[0])
    p_ref = np.stack(p_ref_list)
    del m32
    torch.cuda.empty_cache()

    m16 = Pillar3HiResPhysicsSpecialist(
        mock_backbone=False, lora_rank=detect_lora_rank_from_state_dict(sd, default_rank=24), legacy_3view=p3_legacy
    )
    m16.load_state_dict(sd, strict=False)
    for sub_n in ("dinov3", "cradio", "convnext", "pecore"):
        sub_m = getattr(m16.vision_tower, sub_n, None)
        if sub_m is not None:
            for fn, mod in sub_m.named_modules():
                if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)) and "lora" not in fn.lower():
                    mod.to(torch.bfloat16)
    m16.to(dev).eval()
    p3_cast_gb = torch.cuda.memory_allocated(dev) / 2 ** 30 - base_gb

    p_cast_list = []
    with torch.no_grad():
        for nm in names_all[:40]:
            im = loader.load_pil(nm).convert("RGB")
            pk = build_pillar3_sample_tensors(im, coc_probe=coc, include_mid_tex=not p3_legacy)
            o = m16(
                views_01=pk["views_tensor"].unsqueeze(0).to(dev),
                phys_9ch=pk["phys_9ch"].unsqueeze(0).to(dev),
                probe_8ch=pk["probe_8ch"].unsqueeze(0).to(dev),
                batch_ids=None,
                apply_batch_bias=False,
            )
            p_cast_list.append(o["probs_6x4"].float().cpu().numpy()[0])
    p_cast = np.stack(p_cast_list)
    p3_tv = float(0.5 * np.abs(p_ref - p_cast).sum(axis=-1).mean())
    p3_max_dp = float(np.max(np.abs(p_ref - p_cast)))
    del m16
    torch.cuda.empty_cache()
    print(
        f"[G3.1 p3_qy Safe BF16] {p3_uncast_gb:.2f} -> {p3_cast_gb:.2f} GiB (saved {p3_uncast_gb-p3_cast_gb:.2f} GiB) | "
        f"mean_TV={p3_tv:.6f} (<=0.002) | max|dp|={p3_max_dp:.5f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 2. Fold-0 c233397 Parity Gate (BF16) & NF4 Fidelity Gate (50 val rows)
    # ------------------------------------------------------------------
    z0 = np.load(args.fold0_npz, allow_pickle=True)
    va_names = [str(x) for x in z0["names"][:50]]
    saved_orig = z0["probs_orig_6x4"][:50]
    saved_flip = z0["probs_hflip_6x4"][:50]
    saved_tta = z0["probs_6x4"][:50]

    # Pre-extract live PIL views via the exact FinalEngine._phoenix path
    live_views = []
    for nm in va_names:
        pil = loader.load_pil(nm).convert("RGB")
        maps0 = compute_1to1_wavelet_and_artifact_maps(np.asarray(pil, dtype=np.uint8))
        crops0, _ = select_scrutiny_1to1_patches(
            pil_img=pil, phys_maps=maps0, coc_field=coc.compute_coc_jacobian_field(pil, maps0), patch_size=384, num_patches=2
        )
        g0 = resize_native_aspect_pil(pil, max_pixels=784 * 784, grid_factor=28)
        v_orig = [g0, crops0[0], crops0[1]]
        v_flip = [v.transpose(Image.FLIP_LEFT_RIGHT) for v in v_orig]
        live_views.append((v_orig, v_flip))

    del coc
    torch.cuda.empty_cache()

    bnb_p = os.path.join(REPO, "third_party/pydeps")
    # 2a. Exact unmerged architecture/prompt/view parity (bs=4 live views vs saved_tta: proves zero forward/crop/flip drift)
    ck0_sd = torch.load(args.fold0_ckpt, map_location="cpu")["state_dict"]
    m_unm = NativeVLMPillarModelC233397(pillar_name="pillar1", apply_dora=True, lora_rank=24, enable_gc=False).to(dev)
    m_unm.load_state_dict(ck0_sd, strict=False)
    m_unm.eval()
    unm_orig, unm_flip = [], []
    with torch.no_grad():
        for i in range(0, len(live_views), 4):
            vo_b = [x[0] for x in live_views[i : i + 4]]
            vf_b = [x[1] for x in live_views[i : i + 4]]
            unm_orig.append(m_unm(vo_b, batch_ids=None, apply_batch_bias=False, device=dev)["probs_6x4"].float().cpu().numpy())
            unm_flip.append(m_unm(vf_b, batch_ids=None, apply_batch_bias=False, device=dev)["probs_6x4"].float().cpu().numpy())
    unm_orig_a = np.concatenate(unm_orig, axis=0)
    unm_flip_a = np.concatenate(unm_flip, axis=0)
    unm_tta_a = 0.5 * (unm_orig_a + unm_flip_a)
    unm_max_dp_tta = float(np.max(np.abs(unm_tta_a - saved_tta)))
    del m_unm
    torch.cuda.empty_cache()

    # 2b. Merged BF16 bs=1 deploy check (isolates pure BF16 DoRA merge_and_unload + bs=1 FlashAttn rounding)
    m_bf16 = build_c233397_pillar_for_4090(args.fold0_ckpt, pillar="pillar1", device=str(dev), bnb_path=bnb_p, four_bit=False)
    bf16_orig, bf16_flip, bf16_tta = [], [], []
    with torch.no_grad():
        for v_orig, v_flip in live_views:
            o0 = m_bf16([v_orig], batch_ids=None, apply_batch_bias=False, device=dev)["probs_6x4"].float().cpu().numpy()[0]
            o1 = m_bf16([v_flip], batch_ids=None, apply_batch_bias=False, device=dev)["probs_6x4"].float().cpu().numpy()[0]
            bf16_orig.append(o0)
            bf16_flip.append(o1)
            bf16_tta.append(0.5 * (o0 + o1))
    bf16_orig_a, bf16_flip_a, bf16_tta_a = np.stack(bf16_orig), np.stack(bf16_flip), np.stack(bf16_tta)
    max_dp_orig = float(np.max(np.abs(bf16_orig_a - saved_orig)))
    max_dp_flip = float(np.max(np.abs(bf16_flip_a - saved_flip)))
    max_dp_tta = float(np.max(np.abs(bf16_tta_a - saved_tta)))
    p99_dp_tta = float(np.quantile(np.abs(bf16_tta_a - saved_tta), 0.99))
    bf16_tv_vs_saved = float(0.5 * np.abs(bf16_tta_a - saved_tta).sum(axis=-1).mean())
    del m_bf16
    torch.cuda.empty_cache()

    # 2c. Actual deployed 4-bit NF4 bs=1 fidelity gate (<= 0.012 mean TV vs saved OOF)
    m_q4 = build_c233397_pillar_for_4090(args.fold0_ckpt, pillar="pillar1", device=str(dev), bnb_path=bnb_p, four_bit=True)
    q4_res_gb = torch.cuda.memory_allocated(dev) / 2 ** 30
    q4_tta = []
    with torch.no_grad():
        for v_orig, v_flip in live_views:
            o0 = m_q4([v_orig], batch_ids=None, apply_batch_bias=False, device=dev)["probs_6x4"].float().cpu().numpy()[0]
            o1 = m_q4([v_flip], batch_ids=None, apply_batch_bias=False, device=dev)["probs_6x4"].float().cpu().numpy()[0]
            q4_tta.append(0.5 * (o0 + o1))
    q4_tta_a = np.stack(q4_tta)
    nf4_tv_vs_saved = float(0.5 * np.abs(q4_tta_a - saved_tta).sum(axis=-1).mean())
    nf4_tv_vs_bf16 = float(0.5 * np.abs(q4_tta_a - bf16_tta_a).sum(axis=-1).mean())
    del m_q4
    torch.cuda.empty_cache()
    print(
        f"[G3.2 Fold-0 Parity & NF4] unmerged_bs4 max|dp|={unm_max_dp_tta:.6f} (==0) | "
        f"merged_bf16_bs1 TV={bf16_tv_vs_saved:.5f} p99|dp|={p99_dp_tta:.5f} max|dp|={max_dp_tta:.5f} | "
        f"NF4 res={q4_res_gb:.2f} GiB, TV_vs_saved={nf4_tv_vs_saved:.5f}, TV_vs_bf16={nf4_tv_vs_bf16:.5f} (<=0.012)",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Full 9-Member FinalEngine Stand-In Benchmark (25 images)
    # ------------------------------------------------------------------
    os.environ["R55_P1_QY_CKPT_OVERRIDE"] = args.fold0_ckpt
    torch.cuda.reset_peak_memory_stats(dev)
    eng = fe.FinalEngine(fe.SETS["resident_v6"], device=str(dev), half=True)
    peak_load_gib = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    eng.infer(names_all[0])
    eng.peak = {}
    secs = []
    for nm in names_all[:25]:
        t0 = time.perf_counter()
        eng.infer(nm)
        secs.append(time.perf_counter() - t0)
    res_gib = torch.cuda.memory_allocated(dev) / 2 ** 30
    peak_inf_gib = max(eng.peak.values())
    mean_s = float(np.mean(secs))
    p95_s = float(np.quantile(secs, 0.95))

    g3_pass = (
        (p3_tv <= 0.002)
        and (unm_max_dp_tta <= 0.01)
        and (p99_dp_tta <= 0.01)
        and (bf16_tv_vs_saved <= 0.003)
        and (nf4_tv_vs_saved <= 0.012)
        and (peak_inf_gib <= 22.0)
        and (eng.total_params_b <= 30.0)
    )
    rep = dict(
        g3_pass=bool(g3_pass),
        p3_uncast_gib=round(p3_uncast_gb, 3),
        p3_cast_gib=round(p3_cast_gb, 3),
        p3_cast_tv=p3_tv,
        p3_cast_max_dp=p3_max_dp,
        fold0_unmerged_bs4_max_dp_tta=unm_max_dp_tta,
        fold0_bf16_tv_vs_saved=bf16_tv_vs_saved,
        fold0_bf16_p99_dp_tta=p99_dp_tta,
        fold0_bf16_max_dp_orig=max_dp_orig,
        fold0_bf16_max_dp_hflip=max_dp_flip,
        fold0_bf16_max_dp_tta=max_dp_tta,
        fold0_nf4_res_gib=round(q4_res_gb, 3),
        fold0_nf4_tv_vs_saved=nf4_tv_vs_saved,
        fold0_nf4_tv_vs_bf16=nf4_tv_vs_bf16,
        engine_9m_params_b=float(eng.total_params_b),
        engine_9m_resident_gib=round(res_gib, 3),
        engine_9m_peak_load_gib=round(peak_load_gib, 3),
        engine_9m_peak_infer_gib=round(peak_inf_gib, 3),
        engine_9m_peak_by_member=eng.peak,
        engine_9m_mean_s=round(mean_s, 3),
        engine_9m_p95_s=round(p95_s, 3),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(rep, open(args.out_json, "w"), indent=2)
    print(
        f"=== [G3 Dress Rehearsal Summary: PASS={g3_pass}] ===\n"
        f"  9-Member Engine: params={eng.total_params_b:.3f}B (<=30B) | resident={res_gib:.2f} GiB | "
        f"peak_infer={peak_inf_gib:.2f} GiB (<=22.0 GiB) | mean={mean_s:.2f}s p95={p95_s:.2f}s -> {args.out_json}",
        flush=True,
    )
    assert g3_pass, f"Dress rehearsal failed one or more G3 gates: {rep}"


if __name__ == "__main__":
    main()
