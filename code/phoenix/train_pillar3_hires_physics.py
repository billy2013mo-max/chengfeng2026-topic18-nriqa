#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/phoenix/train_pillar3_hires_physics.py
===========================================
Breakthrough 2 & 3: Pillar-3 1:1 High-Frequency & Optical Physics Specialist (`~1.555B`).

Server-Verified Architecture Components (100% Orthogonal to VLMs):
  1. `DINOv3-ViT-L/16` (`0.304B`, `models/dinov3-vitl16/model.safetensors` via `timm`
     `vit_large_patch16_dinov3`) with 4-Depth Intermediate Hypercolumn Hooks (`25%, 50%, 75%, 100%`)
     + Full-Depth `nn.Linear`-only DoRA!
  2. `C-RADIOv4-H` (`0.630B`, `models/c-radio-v4-h` via `AutoModel(..., trust_remote_code=True)`,
     taking `[0, 1]` RGB `summ, feat = model(x_01)`) with 4-Depth Intermediate Hypercolumn Hooks
     on `radio_model.model.blocks` (`dim=1280`, register-prefix safe)!
  3. `ConvNeXtV2` (`0.200B`, `models/convnextv2-base/model.safetensors` or `convnextv2-large`)
     + `1:1` 2D Haar Wavelet & Bipolar Overshoot ConvStem (`9ch`: Haar LH/HL/HH, Bipolar Halo,
     8x8 Macroblock Glitch, UI Stroke Mask, and 3ch CoC Depth-Defocus Jacobian Field).
  4. `PE-Core-Large` / `FGResQ` Perceptual-Degradation Branch (`0.350B`, auto-loads
     `models/pe-core-l14/model.safetensors` when available!) + `PAL4VST` Spatial Artifact Probe.
  5. Fork-safe main-process CoC/box caching + BF16 vision autocast + FP32 `PhoenixUnimodalDeQAHead`
     + Multi-GPU per-fold sharding (`--fold 0..4 --gpu 0..7` + `--aggregate-cv5`) & resume!
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(THIS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(THIS_DIR))

from phoenix.config import (
    PHOENIX_CKPT_DIR,
    PHOENIX_OOF_DIR,
    PHOENIX_SPLITS_DIR,
    PureRealImageResolver,
    evaluate_official_metrics,
    resolve_local_or_hf_model,
    set_deterministic_seed,
)
from phoenix.losses_and_heads import (
    BasinAlignedSWACollector,
    CrossBatchThurstoneRL2RLoss,
    DeQAOrdinalDistributionLoss,
    PhoenixUnimodalDeQAHead,
    extract_trainable_state_dict,
)
from phoenix.native_physics_and_views import (
    CoCDepthDefocusProbe,
    compute_1to1_wavelet_and_artifact_maps,
    resize_native_aspect_pil,
)
from phoenix.stage0_real_restoration_tapt import (
    FGResQDegradationProjector,
    PAL4VSTArtifactSaliencyProbe,
)
from phoenix.train_pillar12_native_vlm import (
    aggregate_5fold_oof,
    crop_1to1_from_box,
    get_or_build_scrutiny_box_cache,
    load_batch_shift_init,
    load_split_rows,
)


class _FourStageBlockHook:
    """Captures intermediate token outputs at 25%, 50%, 75%, and 100% block depths."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        n = len(blocks)
        self.indices = [
            max(0, n // 4 - 1),
            max(0, n // 2 - 1),
            max(0, (3 * n) // 4 - 1),
            max(0, n - 1),
        ]
        self.cache: Dict[int, Optional[torch.Tensor]] = {s: None for s in range(4)}
        for slot, b_idx in enumerate(self.indices):
            blocks[b_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot: int):
        def _hk(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.cache[slot] = t

        return _hk

    def pop_pooled_hypercolumn(self, fallback_tokens: torch.Tensor, grid_hw: Tuple[int, int] = (6, 6)) -> torch.Tensor:
        target_len = grid_hw[0] * grid_hw[1]
        ft = fallback_tokens.float()
        l_ref = ft.shape[1]
        chunks: List[torch.Tensor] = []
        for slot in range(4):
            c = self.cache.get(slot)
            self.cache[slot] = None
            if c is None:
                c_f = ft
            else:
                c_f = c.float()
                # Strip leading CLS / register summary tokens if present
                if c_f.shape[1] > l_ref and l_ref >= target_len:
                    c_f = c_f[:, -l_ref:, :]
            if c_f.ndim == 4:
                p = F.adaptive_avg_pool2d(c_f, output_size=grid_hw).flatten(2).transpose(1, 2)
            else:
                p = F.adaptive_avg_pool1d(c_f.transpose(1, 2), output_size=target_len).transpose(1, 2)
            chunks.append(p)
        return torch.cat(chunks, dim=-1)  # [B, 36, 4*D]


def _attach_linear_only_dora_or_unfreeze(
    model: nn.Module,
    blocks: Optional[nn.ModuleList],
    rank: int = 16,
    use_dora: bool = True,
) -> nn.Module:
    try:
        from peft import LoraConfig, get_peft_model

        exact_linear = [
            name
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear)
            and not any(s in name.lower() for s in ("patch_embed", "head", "stem"))
            and name.split(".")[-1] in {"qkv", "proj", "fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"}
        ]
        if exact_linear:
            for p in model.parameters():
                p.requires_grad = False
            cfg = LoraConfig(
                r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.05,
                target_modules=exact_linear,
                use_dora=use_dora,
                bias="none",
            )
            return get_peft_model(model, cfg)
    except Exception:
        pass

    for p in model.parameters():
        p.requires_grad = False
    if blocks is not None and len(blocks) > 0:
        for blk in list(blocks)[-min(8, len(blocks)) :]:
            for p in blk.parameters():
                p.requires_grad = True
    for n, p in model.named_parameters():
        if "norm" in n.lower():
            p.requires_grad = True
    return model


class HaarWaveletCoCPhysicsStem(nn.Module):
    def __init__(self, in_channels: int = 9, out_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.Conv2d(192, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, out_dim),
            nn.GELU(),
        )

    def forward(self, phys_9ch: torch.Tensor) -> torch.Tensor:
        amp_dev = "cuda" if phys_9ch.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            fmap = self.net(phys_9ch.float())
            pooled = F.adaptive_avg_pool2d(fmap, output_size=(6, 6))
            return pooled.flatten(2).transpose(1, 2)  # [B, 36, out_dim]


class MultiLayerHypercolumnVisionTower(nn.Module):
    """
    Natively loads `/wanqing-develop/luowenjing/18/models/` offline checkpoints:
      - `dinov3-vitl16` (`timm` `vit_large_patch16_dinov3` + `safetensors`)
      - `c-radio-v4-h` (`AutoModel(..., trust_remote_code=True)` taking `[0, 1]` RGB)
      - `convnextv2` (`timm` `convnextv2_base` / `convnextv2_large` + `safetensors`)
      - Optional `pe-core-l14` (`timm` `vit_pe_core_large_patch14_336` + `safetensors`)
    """

    def __init__(self, out_dim: int = 512, mock_backbone: bool = False) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.mock_backbone = mock_backbone
        self.has_pecore = False

        if mock_backbone:
            self.mock_conv = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=7, stride=4, padding=3),
                nn.GELU(),
                nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d((6, 6)),
            )
            return

        # 1. DINOv3-ViT-L/16 (`timm` safetensors or HF fallback)
        self.dinov3, self._dino_hook, d_dim = self._build_dinov3()
        self.proj_dino = nn.Sequential(nn.LayerNorm(d_dim * 4), nn.Linear(d_dim * 4, out_dim)).float()

        # 2. C-RADIOv4-H (`[0, 1]` input, `m.radio_model.model.blocks`)
        self.cradio, self._radio_hook, r_dim = self._build_cradio()
        self.proj_radio = nn.Sequential(nn.LayerNorm(r_dim * 4), nn.Linear(r_dim * 4, out_dim)).float()

        # 3. ConvNeXtV2 (`models/convnextv2-base` or `models/convnextv2-large`)
        self.convnext, c_dim = self._build_convnext()
        self.proj_convnext = nn.Sequential(nn.LayerNorm(c_dim), nn.Linear(c_dim, out_dim)).float()

        # 4. Optional PE-Core-L (`models/pe-core-l14/model.safetensors` under our 0.35B slot)
        pec_res = self._try_build_pecore()
        if pec_res is not None:
            self.pecore, self._pec_hook, p_dim = pec_res
            self.proj_pecore = nn.Sequential(nn.LayerNorm(p_dim * 4), nn.Linear(p_dim * 4, out_dim)).float()
            self.has_pecore = True

        self.register_buffer("imnet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imnet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pec_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pec_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1), persistent=False)

    @staticmethod
    def _build_dinov3():
        import timm
        from safetensors.torch import load_file

        p_cand = resolve_local_or_hf_model("pillar3_dinov3_l")
        w_file = p_cand if os.path.isfile(p_cand) else os.path.join(p_cand, "model.safetensors")
        if os.path.exists(w_file):
            m = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0, dynamic_img_size=True)
            m.load_state_dict(load_file(w_file), strict=False)
        else:
            m = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
        hook = _FourStageBlockHook(m.blocks)
        dim = int(m.embed_dim)
        if torch.cuda.is_available():
            m = m.to(torch.bfloat16)
        m = _attach_linear_only_dora_or_unfreeze(m, m.blocks, rank=16, use_dora=True)
        return m, hook, dim

    @staticmethod
    def _build_cradio():
        from transformers import AutoModel

        repo = resolve_local_or_hf_model("pillar3_cradio_v4_h")
        dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            m = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=dt)
        except TypeError:
            m = AutoModel.from_pretrained(repo, trust_remote_code=True, torch_dtype=dt)
        blocks = m.radio_model.model.blocks
        hook = _FourStageBlockHook(blocks)
        dim = 1280
        # Unfreeze top 8 blocks of C-RADIO cleanly (custom remote_code modules work safest with block unfreezing)
        for p in m.parameters():
            p.requires_grad = False
        for blk in list(blocks)[-8:]:
            for p in blk.parameters():
                p.requires_grad = True
        return m, hook, dim

    @staticmethod
    def _build_convnext():
        import timm
        from safetensors.torch import load_file

        p_cand = resolve_local_or_hf_model("pillar3_convnextv2_l")
        w_file = p_cand if os.path.isfile(p_cand) else os.path.join(p_cand, "model.safetensors")
        arch = "convnextv2_large.fcmae_ft_in22k_in1k_384" if "large" in p_cand.lower() else "convnextv2_base.fcmae_ft_in22k_in1k"
        if os.path.exists(w_file):
            m = timm.create_model(arch, pretrained=False, num_classes=0)
            m.load_state_dict(load_file(w_file), strict=False)
        else:
            m = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True, num_classes=0)
        if torch.cuda.is_available():
            m = m.to(torch.bfloat16)
        for p in m.parameters():
            p.requires_grad = True
        return m, int(m.num_features)

    @staticmethod
    def _try_build_pecore():
        p_cand = resolve_local_or_hf_model("pillar3_fgresq_adapter")
        w_file = p_cand if os.path.isfile(p_cand) else os.path.join(p_cand, "model.safetensors")
        if not os.path.exists(w_file) or "pe-core" not in w_file.lower():
            return None
        try:
            import timm
            from safetensors.torch import load_file

            m = timm.create_model("vit_pe_core_large_patch14_336", pretrained=False, num_classes=0, dynamic_img_size=True)
            m.load_state_dict(load_file(w_file), strict=False)
            hook = _FourStageBlockHook(m.blocks)
            dim = int(m.embed_dim)
            if torch.cuda.is_available():
                m = m.to(torch.bfloat16)
            m = _attach_linear_only_dora_or_unfreeze(m, m.blocks, rank=16, use_dora=True)
            return m, hook, dim
        except Exception:
            return None

    def forward(self, views_01: torch.Tensor) -> torch.Tensor:
        """
        Args:
          views_01: `[B, 3_views, 3, 384, 384]` RGB tensor in `[0, 1]`.
        Returns:
          `[B, 108, out_dim]` FP32 hypercolumn visual tokens.
        """
        bsz, n_v, c, h, w = views_01.shape
        flat_01 = views_01.reshape(bsz * n_v, c, h, w)

        if self.mock_backbone:
            fmap = self.mock_conv(flat_01)
            return fmap.flatten(2).transpose(1, 2).reshape(bsz, n_v * 36, self.out_dim)

        use_bf16 = flat_01.is_cuda
        imnet_x = (flat_01 - self.imnet_mean) / self.imnet_std

        with torch.autocast(device_type="cuda" if use_bf16 else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
            # 1. DINOv3 forward (`timm` unwrapped or PeftModel wrapped)
            dino_base = self.dinov3.get_base_model() if hasattr(self.dinov3, "get_base_model") else self.dinov3
            raw_dino = dino_base.forward_features(imnet_x)
            n_pref = int(getattr(dino_base, "num_prefix_tokens", 1))
            spatial_dino = raw_dino[:, n_pref:, :] if raw_dino.shape[1] > n_pref else raw_dino

            # 2. C-RADIOv4-H forward (expects `[0, 1]` input directly!)
            _, feat_radio = self.cradio(flat_01)

            # 3. ConvNeXtV2 spatial map forward
            cx_map = self.convnext.forward_features(imnet_x)

            # 4. Optional PE-Core-L forward (`336x336` patch-14 friendly resize)
            if self.has_pecore:
                pec_in = F.interpolate((flat_01 - self.pec_mean) / self.pec_std, size=(336, 336), mode="bilinear", align_corners=False)
                pec_base = self.pecore.get_base_model() if hasattr(self.pecore, "get_base_model") else self.pecore
                raw_pec = pec_base.forward_features(pec_in)
                p_pref = int(getattr(pec_base, "num_prefix_tokens", 1))
                spatial_pec = raw_pec[:, p_pref:, :] if raw_pec.shape[1] > p_pref else raw_pec
            else:
                spatial_pec = None

        amp_dev = "cuda" if flat_01.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            hc_dino = self.proj_dino(self._dino_hook.pop_pooled_hypercolumn(spatial_dino, (6, 6)))
            hc_radio = self.proj_radio(self._radio_hook.pop_pooled_hypercolumn(feat_radio, (6, 6)))

            if cx_map.ndim == 4:
                cx_tok = self.proj_convnext(
                    F.adaptive_avg_pool2d(cx_map.float(), output_size=(6, 6)).flatten(2).transpose(1, 2)
                )
            else:
                cx_tok = self.proj_convnext(
                    F.adaptive_avg_pool1d(cx_map.float().transpose(1, 2), output_size=36).transpose(1, 2)
                )

            if self.has_pecore and spatial_pec is not None:
                hc_pec = self.proj_pecore(self._pec_hook.pop_pooled_hypercolumn(spatial_pec, (6, 6)))
                fused = (hc_dino + hc_radio + cx_tok + hc_pec) / 4.0
            else:
                fused = (hc_dino + hc_radio + cx_tok) / 3.0

            return fused.reshape(bsz, n_v * 36, self.out_dim)


class Pillar3HiResPhysicsSpecialist(nn.Module):
    def __init__(
        self,
        init_batch_shift: Optional[np.ndarray] = None,
        stage0_ckpt_path: Optional[str] = None,
        mock_backbone: bool = False,
    ) -> None:
        super().__init__()
        token_dim = 512
        self.vision_tower = MultiLayerHypercolumnVisionTower(out_dim=token_dim, mock_backbone=mock_backbone)
        self.physics_stem = HaarWaveletCoCPhysicsStem(in_channels=9, out_dim=token_dim).float()
        self.fgresq_branch = FGResQDegradationProjector(in_channels=8, feat_dim=256).float()
        self.pal4vst_probe = PAL4VSTArtifactSaliencyProbe(in_channels=8, hidden_dim=192).float()
        self.probe_proj = nn.Linear(256, token_dim).float()

        if stage0_ckpt_path and os.path.exists(stage0_ckpt_path):
            ckpt = torch.load(stage0_ckpt_path, map_location="cpu")
            if "fgresq_state_dict" in ckpt:
                self.fgresq_branch.load_state_dict(ckpt["fgresq_state_dict"], strict=False)

        self.head = PhoenixUnimodalDeQAHead(
            in_dim=token_dim,
            hidden_dim=512,
            num_heads=8,
            dropout=0.10,
            init_batch_shift=init_batch_shift,
        ).float()

    def forward(
        self,
        views_01: torch.Tensor,
        phys_9ch: torch.Tensor,
        probe_8ch: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        apply_batch_bias: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        vis_tokens = self.vision_tower(views_01)
        amp_dev = "cuda" if views_01.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            phys_tokens = self.physics_stem(phys_9ch.float())
            fg_tokens = self.probe_proj(self.fgresq_branch(probe_8ch.float())["tokens"])
            pal_tokens = self.probe_proj(self.pal4vst_probe(probe_8ch.float())["artifact_tokens"])
            all_tokens = torch.cat([vis_tokens.float(), phys_tokens, fg_tokens, pal_tokens], dim=1)
            return self.head(all_tokens, batch_ids=batch_ids, apply_batch_bias=apply_batch_bias)


def build_pillar3_sample_tensors(
    pil_img,
    coc_probe: Optional[CoCDepthDefocusProbe] = None,
    precomputed_boxes: Optional[np.ndarray] = None,
    patch_size: int = 384,
) -> Dict[str, object]:
    """
    Constructs `[0, 1]` `views_01 [3, 3, 384, 384]` + `phys_9ch [9, 256, 256]` + `probe_8ch [8, 256, 256]`.
    Uses CPU-safe analytical CoC depth fallback inside DataLoader workers when `coc_probe` is None.
    """
    rgb_np = np.asarray(pil_img, dtype=np.uint8)
    phys_maps = compute_1to1_wavelet_and_artifact_maps(rgb_np)
    probe_obj = coc_probe if coc_probe is not None else CoCDepthDefocusProbe(model_path=None, device="cpu")
    coc_field = probe_obj.compute_coc_jacobian_field(pil_img, phys_maps)

    if precomputed_boxes is not None:
        crops = [
            crop_1to1_from_box(pil_img, precomputed_boxes[0], patch_size=patch_size),
            crop_1to1_from_box(pil_img, precomputed_boxes[1], patch_size=patch_size),
        ]
        crop_meta = [{"box_xyxy": precomputed_boxes[0].tolist()}, {"box_xyxy": precomputed_boxes[1].tolist()}]
    else:
        from phoenix.native_physics_and_views import select_scrutiny_1to1_patches

        crops, crop_meta = select_scrutiny_1to1_patches(
            pil_img=pil_img,
            phys_maps=phys_maps,
            coc_field=coc_field,
            patch_size=patch_size,
            num_patches=2,
        )

    g_pil = resize_native_aspect_pil(pil_img, max_pixels=patch_size * patch_size, grid_factor=16)
    g_arr = np.asarray(g_pil, dtype=np.uint8)
    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    g_clip = g_arr[:patch_size, :patch_size]
    ch_h, ch_w, _ = g_clip.shape
    y_off = (patch_size - ch_h) // 2
    x_off = (patch_size - ch_w) // 2
    canvas[y_off : y_off + ch_h, x_off : x_off + ch_w] = g_clip

    view_arrays = [
        (arr_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)
        for arr_u8 in (canvas, np.asarray(crops[0], dtype=np.uint8), np.asarray(crops[1], dtype=np.uint8))
    ]
    views_01 = torch.from_numpy(np.stack(view_arrays, axis=0)).float()  # [3, 3, 384, 384] in [0, 1]

    phys_9_np = np.concatenate(
        [
            phys_maps["haar_hf"],
            phys_maps["bipolar_halo"][:, :, None],
            phys_maps["macroblock_glitch"][:, :, None],
            phys_maps["ui_text_mask"][:, :, None],
            coc_field,
        ],
        axis=-1,
    )
    phys_9_t = F.interpolate(
        torch.from_numpy(phys_9_np.transpose(2, 0, 1)).unsqueeze(0).float(),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    probe_8_t = F.interpolate(
        PAL4VSTArtifactSaliencyProbe.build_input_tensor(rgb_np).unsqueeze(0),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return {
        "views_tensor": views_01,
        "phys_9ch": phys_9_t,
        "probe_8ch": probe_8_t,
        "crop_meta": crop_meta,
    }


class Pillar3Dataset(Dataset):
    """100% fork-safe dataset using precomputed 1:1 scrutiny boxes (zero CUDA inside workers!)."""

    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        resolver: PureRealImageResolver,
        box_cache: Dict[str, np.ndarray],
        is_train: bool = True,
        oof_teacher_probs: Optional[np.ndarray] = None,
    ) -> None:
        self.rows = list(rows)
        self.resolver = resolver
        self.box_cache = box_cache
        self.is_train = is_train
        self.oof_teacher_probs = oof_teacher_probs

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        from PIL import Image

        row = self.rows[idx]
        nm = str(row["name"])
        pil_img = self.resolver.load_rgb_pil(nm)
        orig_w, _ = pil_img.size
        boxes = self.box_cache[nm].copy()

        if self.is_train and np.random.rand() < 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            old_x0 = boxes[:, 0].copy()
            old_x1 = boxes[:, 2].copy()
            boxes[:, 0] = np.maximum(0, orig_w - old_x1)
            boxes[:, 2] = boxes[:, 0] + 384

        pack = build_pillar3_sample_tensors(pil_img, coc_probe=None, precomputed_boxes=boxes)
        item: Dict[str, object] = {
            "name": nm,
            "views_tensor": pack["views_tensor"],
            "phys_9ch": pack["phys_9ch"],
            "probe_8ch": pack["probe_8ch"],
            "batch_id": torch.tensor(int(row.get("batch_id", 0)), dtype=torch.long),
            "noisy_flag": torch.tensor(int(row.get("noisy_flag", 0)), dtype=torch.long),
            "grades": torch.tensor(row.get("grades", [0] * 6), dtype=torch.long),
        }
        if self.oof_teacher_probs is not None:
            item["teacher_probs"] = torch.from_numpy(self.oof_teacher_probs[idx]).float()
        return item


@torch.no_grad()
def predict_pillar3_dataset(
    model: Pillar3HiResPhysicsSpecialist,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    names_all: List[str] = []
    probs_l, logits_l, clean_l, p_clean_l, mu_l, var_l, emb_l, gold_l = [], [], [], [], [], [], [], []
    for batch in loader:
        names_all.extend(batch["name"])
        out = model(
            views_01=batch["views_tensor"].to(device, non_blocking=True),
            phys_9ch=batch["phys_9ch"].to(device, non_blocking=True),
            probe_8ch=batch["probe_8ch"].to(device, non_blocking=True),
            batch_ids=None,
            apply_batch_bias=False,
        )
        probs_l.append(out["probs_6x4"].cpu().numpy())
        logits_l.append(out["logits_6x4"].cpu().numpy())
        clean_l.append(out["clean_logit"].cpu().numpy())
        p_clean_l.append(out["p_clean_joint"].cpu().numpy())
        mu_l.append(out["mu_6d"].cpu().numpy())
        var_l.append(out["var_6d"].cpu().numpy())
        emb_l.append(out["quality_emb"].cpu().numpy())
        gold_l.append(batch["grades"].numpy())

    return {
        "names": np.array(names_all, dtype=object),
        "probs_6x4": np.concatenate(probs_l, axis=0).astype(np.float32),
        "logits_6x4": np.concatenate(logits_l, axis=0).astype(np.float32),
        "clean_logit": np.concatenate(clean_l, axis=0).astype(np.float32),
        "p_clean_joint": np.concatenate(p_clean_l, axis=0).astype(np.float32),
        "mu_6d": np.concatenate(mu_l, axis=0).astype(np.float32),
        "var_6d": np.concatenate(var_l, axis=0).astype(np.float32),
        "quality_emb": np.concatenate(emb_l, axis=0).astype(np.float32),
        "gold_6d": np.concatenate(gold_l, axis=0).astype(np.int64),
    }


def train_pillar3_run(
    model: Pillar3HiResPhysicsSpecialist,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr_head: float = 2.0e-4,
    lr_backbone: float = 4.0e-5,
    swa_start_epoch: Optional[int] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    tower_ids = {id(p) for p in model.vision_tower.parameters()}
    fast_params = [p for p in model.parameters() if p.requires_grad and id(p) not in tower_ids]
    slow_params = [p for p in model.vision_tower.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(
        [
            {"params": fast_params, "lr": lr_head, "weight_decay": 1e-2},
            {"params": slow_params, "lr": lr_backbone, "weight_decay": 1e-2},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * len(train_loader)), eta_min=lr_backbone * 0.1
    )
    deqa_loss_fn = DeQAOrdinalDistributionLoss().to(device)
    rl2r_loss_fn = CrossBatchThurstoneRL2RLoss(queue_size=1024, emb_dim=256).to(device)
    swa = BasinAlignedSWACollector() if swa_start_epoch is not None else None
    trainable_all = [p for p in model.parameters() if p.requires_grad]

    for ep in range(epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for batch in train_loader:
            y_6d = batch["grades"].to(device, non_blocking=True)
            b_ids = batch["batch_id"].to(device, non_blocking=True)
            noisy = batch["noisy_flag"].to(device, non_blocking=True)
            t_probs = batch.get("teacher_probs")
            if t_probs is not None:
                t_probs = t_probs.to(device, non_blocking=True)

            out = model(
                views_01=batch["views_tensor"].to(device, non_blocking=True),
                phys_9ch=batch["phys_9ch"].to(device, non_blocking=True),
                probe_8ch=batch["probe_8ch"].to(device, non_blocking=True),
                batch_ids=b_ids,
                apply_batch_bias=True,
            )
            loss_dict = deqa_loss_fn(
                head_out=out,
                y_6d=y_6d,
                noisy_flags=noisy,
                oof_teacher_probs=t_probs,
                supp_bias_param=model.head.supp_batch_bias,
            )
            loss = loss_dict["loss"] + 0.35 * rl2r_loss_fn(out, y_6d)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_all, max_norm=1.0)
            opt.step()
            sched.step()
            running += float(loss.item())

        if swa is not None and ep >= swa_start_epoch:
            swa.update(model, weight=1.0)
        print(
            f"  [pillar3] Epoch {ep+1}/{epochs} ({time.time()-t0:.0f}s) | loss={running / max(1, len(train_loader)):.4f}",
            flush=True,
        )

    return swa.export_state_dict() if (swa is not None and swa.count > 0) else None


def run_smoke_test() -> None:
    from PIL import Image

    set_deterministic_seed(2026)
    device = torch.device("cpu")
    samples = []
    for i in range(6):
        pil_img = Image.fromarray(np.random.randint(0, 255, (720, 405, 3), dtype=np.uint8))
        pack = build_pillar3_sample_tensors(pil_img)
        samples.append(
            {
                "name": f"p3_smoke_{i:02d}.png",
                "views_tensor": pack["views_tensor"],
                "phys_9ch": pack["phys_9ch"],
                "probe_8ch": pack["probe_8ch"],
                "batch_id": torch.tensor(i % 2, dtype=torch.long),
                "noisy_flag": torch.tensor(0, dtype=torch.long),
                "grades": torch.tensor([i % 4, 0, (i + 1) % 4, 0, 0, 3 if i == 4 else 0], dtype=torch.long),
            }
        )

    dl = DataLoader(samples, batch_size=3, shuffle=False)
    model = Pillar3HiResPhysicsSpecialist(mock_backbone=True).to(device)
    swa_sd = train_pillar3_run(model, dl, device, epochs=2, swa_start_epoch=0)
    assert swa_sd is not None
    model.load_state_dict(swa_sd, strict=False)
    pred = predict_pillar3_dataset(model, dl, device)
    assert pred["probs_6x4"].shape == (6, 6, 4)
    dl_one = DataLoader(samples[:1], batch_size=1, shuffle=False)
    pred_one = predict_pillar3_dataset(model, dl_one, device)
    diff = float(np.max(np.abs(pred_one["probs_6x4"][0] - pred["probs_6x4"][0])))
    assert diff < 1e-6, f"Pillar-3 single-image independence failed: diff={diff}"
    print(f"[Pillar-3 HiRes Physics Smoke Test PASS] probs={pred['probs_6x4'].shape} | single_img_diff={diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix Pillar-3 1:1 Hi-Res & Optical Physics Specialist")
    parser.add_argument("--mode", choices=["cv5", "stage2-rl2r", "fulldata-swa"], default="cv5")
    parser.add_argument("--split-csv", default=str(PHOENIX_SPLITS_DIR / "phoenix_5fold.csv"))
    parser.add_argument("--audit-json", default=str(PHOENIX_SPLITS_DIR / "batch_shift_and_audit.json"))
    parser.add_argument("--stage0-ckpt", default=str(PHOENIX_CKPT_DIR / "stage0_tapt" / "stage0_fgresq_tapt.pt"))
    parser.add_argument("--teacher-oof-npz", default="")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=2.0e-4)
    parser.add_argument("--lr-backbone", type=float, default=4.0e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--aggregate-cv5", action="store_true")
    parser.add_argument("--out-dir", default=str(PHOENIX_OOF_DIR))
    parser.add_argument("--ckpt-dir", default=str(PHOENIX_CKPT_DIR))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    rows = load_split_rows(args.split_csv)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.aggregate_cv5:
        aggregate_5fold_oof(rows, "pillar3", args.mode, args.out_dir)
        return

    set_deterministic_seed(2026 + max(0, args.fold))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    resolver = PureRealImageResolver()
    box_cache_path = os.path.join(args.out_dir, "scrutiny_boxes_384_cache.npz")
    box_cache = get_or_build_scrutiny_box_cache(
        names=[str(r["name"]) for r in rows],
        resolver=resolver,
        device_str=str(device),
        cache_npz_path=box_cache_path,
    )
    batch_shift = load_batch_shift_init(args.audit_json)

    teacher_probs_all = None
    if args.teacher_oof_npz and os.path.exists(args.teacher_oof_npz):
        t_npz = np.load(args.teacher_oof_npz, allow_pickle=True)
        t_map = {str(nm): i for i, nm in enumerate(t_npz["names"])}
        teacher_probs_all = np.stack([t_npz["probs_6x4"][t_map[str(r["name"])]] for r in rows], axis=0)

    if args.mode in ("cv5", "stage2-rl2r"):
        full_oof_path = os.path.join(args.out_dir, f"pillar3_{args.mode}_oof.npz")
        if args.fold < 0 and os.path.exists(full_oof_path):
            print(f"[pillar3 Resume] {full_oof_path} already exists; skipping.", flush=True)
            return

        target_folds = list(range(5)) if args.fold < 0 else [args.fold]
        gold_all = np.array([r["grades"] for r in rows], dtype=np.int64)

        for f_idx in target_folds:
            fold_npz = os.path.join(args.out_dir, f"pillar3_{args.mode}_fold{f_idx}.npz")
            if os.path.exists(fold_npz):
                print(f"[pillar3 Resume] Fold {f_idx} already completed ({fold_npz}); skipping.", flush=True)
                continue

            tr_idx = [i for i, r in enumerate(rows) if int(r["fold"]) != f_idx]
            va_idx = [i for i, r in enumerate(rows) if int(r["fold"]) == f_idx]
            tr_teacher = teacher_probs_all[tr_idx] if teacher_probs_all is not None else None

            ds_tr = Pillar3Dataset([rows[i] for i in tr_idx], resolver, box_cache, True, tr_teacher)
            ds_va = Pillar3Dataset([rows[i] for i in va_idx], resolver, box_cache, False)
            dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type == "cuda"))
            dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"))

            print(f"=== [pillar3 {args.mode}] Fold {f_idx}/5 | train={len(tr_idx)} val={len(va_idx)} | device={device} ===", flush=True)
            model = Pillar3HiResPhysicsSpecialist(
                init_batch_shift=batch_shift,
                stage0_ckpt_path=args.stage0_ckpt,
            ).to(device)
            swa_sd = train_pillar3_run(
                model=model,
                train_loader=dl_tr,
                device=device,
                epochs=args.epochs,
                lr_head=args.lr_head,
                lr_backbone=args.lr_backbone,
                swa_start_epoch=max(2, args.epochs - 3),
            )
            if swa_sd is not None:
                model.load_state_dict(swa_sd, strict=False)

            va_out = predict_pillar3_dataset(model, dl_va, device)
            np.savez_compressed(fold_npz, fold=np.int64(f_idx), **va_out)

            m = evaluate_official_metrics(np.argmax(va_out["probs_6x4"], axis=-1), gold_all[va_idx])
            print(f"[pillar3 Fold {f_idx}] Naive Argmax Strict={m['strict_total']:.2f} | Practice={m['practice_total']:.2f}", flush=True)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.fold < 0:
            aggregate_5fold_oof(rows, "pillar3", args.mode, args.out_dir)

    elif args.mode == "fulldata-swa":
        ckpt_path = os.path.join(args.ckpt_dir, "pillar3_fulldata_swa.pt")
        if os.path.exists(ckpt_path):
            print(f"[pillar3 Resume] Full-data SWA checkpoint exists ({ckpt_path}); skipping.", flush=True)
            return

        ds_full = Pillar3Dataset(rows, resolver, box_cache, True, teacher_probs_all)
        model = Pillar3HiResPhysicsSpecialist(
            init_batch_shift=batch_shift,
            stage0_ckpt_path=args.stage0_ckpt,
        ).to(device)
        dl_warm = DataLoader(ds_full, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type == "cuda"))
        warm_epochs = max(2, args.epochs - 3)
        train_pillar3_run(model, dl_warm, device, epochs=warm_epochs, lr_head=args.lr_head, lr_backbone=args.lr_backbone)
        warm_state = extract_trainable_state_dict(model)

        basin_swa = BasinAlignedSWACollector()
        for b_idx, seed in enumerate((2026, 2027, 2028)):
            set_deterministic_seed(seed)
            model.load_state_dict(warm_state, strict=False)
            dl_b = DataLoader(ds_full, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type == "cuda"))
            train_pillar3_run(
                model,
                dl_b,
                device,
                epochs=3,
                lr_head=args.lr_head * 0.30,
                lr_backbone=args.lr_backbone * 0.30,
            )
            basin_swa.update(model, weight=1.0)
            print(f"[pillar3 FullData Basin-SWA] Collected branch {b_idx+1}/3 (seed={seed})", flush=True)

        torch.save({"pillar": "pillar3", "state_dict": basin_swa.export_state_dict()}, ckpt_path)
        print(f"[pillar3] Exported SINGLE Basin-Aligned SWA checkpoint -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
