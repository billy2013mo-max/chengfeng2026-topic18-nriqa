#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/backbones_qingyu.py
===============================
Pillar 2 (Section 3.5) & Pillar 5 (25.22B Pool) Vision Backbone Builders:
  - Tier-1 Lightweight & Native-Aspect Specialists (5 Backbones, 1.96B total):
      1. `convnextv2_base` (`0.089B`): Global `448x448` + 4 deterministic 1:1 `448x448` physics crops
         (`focus_fg`, `meor_non_ui`, `cutout_boundary`, `fg_micro_tex`) with multi-stage feature tap + Gated MIL.
      2. `pe_core_large` (`0.350B`): Global `448x448` + 4 deterministic 1:1 `448x448` crops with
         **Intermediate Multi-Layer Gated Readout at 0-indexed blocks `[15, 19, 22]`** (layers 16, 20, 23,
         i.e. 67%/83%/96% depth — bypassing contrastive top-layer texture smoothing!).
      3. `dinov3_vitl16` (`0.330B`): **Native-Aspect 2D-RoPE Dynamic Grid** (`patch_mult=16`, ~784 patches,
         e.g. `352x576` for 9:16 portrait, zero square warping!) + 2 key 1:1 `448x448` boundary/focus crops,
         with 3-layer intermediate tap + `TokenGatedABMILPool`.
      4. `siglip2_so400m_naflex` (`0.430B`): **Native Variable-Aspect NaFlex Sequence** (`max_num_patches=784`
         with `pixel_attention_mask`-aware `MaskedTokenGatedABMILPool` + intermediate tap; seamlessly supports
         both `Siglip2VisionModel` NaFlex and interpolated `SiglipVisionModel` local checkpoints).
      5. `cradio_v4_h` (`0.670B`): **Native-Aspect Multi-Teacher Aggregator** (`patch_mult=16`, input `[0, 1]`,
         SAM + DINO + CLIP/SigLIP teacher representations) + 2 key 1:1 `448x448` halo/texture crops.
  - Tier-2 Visual Foundation Bridge (`6.08B`):
      6. `internvit_6b` (`InternViT-6B-448px-V2.5`): Global `448x448` + 4 deterministic 1:1 `448x448` physics crops,
         3-Layer Pyramid Tap (layers 32/40/48) + `TokenGatedABMILPool` + LoRA (`r=32`) / last-k unfreezing.
"""
import os
import sys
import glob
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.config_and_paths import REPO_ROOT, resolve_model_path, setup_sys_path_and_cwd
from qingyu.unimodal_xbm_losses import UnimodalQuadraticResidualHead

setup_sys_path_and_cwd(chdir_repo=False)
from abmil_layers import TokenGatedABMILPool, DualLayerTapWrapper


class IntermediateThreeLayerGatedTap(nn.Module):
    """
    Explicit 3-Layer Intermediate Readout Hook (Section 3.5):
    Hooks arbitrary specified block indices `tap_indices` (e.g. `[15, 19, 22]` on 24-block `PE-Core-Large`)
    and fuses them with final-layer tokens via LayerNorm + learned per-channel sigmoid gates (zero-init residual
    so step-0 representations remain smooth while immediately exposing mid-layer high-frequency texture gradients).
    """
    def __init__(self, blocks: nn.ModuleList, dim: int, tap_indices: Sequence[int]):
        super().__init__()
        n_blk = len(blocks)
        valid_idx = [max(0, min(n_blk - 1, int(i))) for i in tap_indices]
        self.tap_indices = valid_idx
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in valid_idx])
        self.proj = nn.Sequential(
            nn.Linear(dim * len(valid_idx), dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        self.gate_logit = nn.Parameter(torch.full((dim,), -1.2))
        self._caches: Dict[int, Optional[torch.Tensor]] = {k: None for k in range(len(valid_idx))}

        for slot, blk_idx in enumerate(valid_idx):
            blocks[blk_idx].register_forward_hook(self._make_hook(slot))

    def _make_hook(self, slot: int):
        def _hk(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self._caches[slot] = t
        return _hk

    def fuse(self, final_tokens: torch.Tensor) -> torch.Tensor:
        parts = []
        for slot, ln in enumerate(self.norms):
            c = self._caches.get(slot, None)
            if c is None or c.shape != final_tokens.shape:
                self.clear()
                return final_tokens
            parts.append(ln(c.float()))
        self.clear()
        cat_mid = torch.cat(parts, dim=-1)
        gate = torch.sigmoid(self.gate_logit).view(1, 1, -1)
        return final_tokens.float() + 0.50 * gate * self.proj(cat_mid)

    def clear(self) -> None:
        for k in list(self._caches.keys()):
            self._caches[k] = None


class MaskedTokenGatedABMILPool(nn.Module):
    """
    Padding-mask-aware `TokenGatedABMILPool` for variable-length `SigLIP2-SO400M-NaFlex` sequences (`max_num_patches=784`).
    Computes masked mean, masked spatial std, masked Top-5% anomaly pooling, and masked Gated ABMIL!
    """
    def __init__(self, dim: int, bottleneck: int = 256):
        super().__init__()
        self.dim = dim
        self.saliency_proj = nn.Linear(dim, 4)
        self.tau_raw = nn.Parameter(torch.zeros(1, 1, 4))
        self.anom_scorer = nn.Linear(dim, 1)
        self.attn_V = nn.Linear(dim, bottleneck)
        self.attn_U = nn.Linear(dim, bottleneck)
        self.attn_w = nn.Linear(bottleneck, 1)
        self.norm_stats = nn.LayerNorm(dim * 4)
        self.proj_res = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.proj_res[-1].weight)
        nn.init.zeros_(self.proj_res[-1].bias)
        self.gate_alpha = nn.Parameter(torch.tensor(0.25))

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`tokens`: `[B, L, D]`, `mask`: optional `[B, L]` float/bool (`1 = valid patch, 0 = pad`)."""
        t = tokens.float()
        B, L, D = t.shape
        if mask is None:
            mask_f = t.new_ones((B, L))
        else:
            mask_f = mask.float()
            if mask_f.shape[1] != L:
                mask_f = F.interpolate(mask_f.unsqueeze(1), size=L, mode="nearest").squeeze(1)

        valid_cnt = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        m_exp = mask_f.unsqueeze(-1)                                # [B, L, 1]
        mean_pool = (t * m_exp).sum(dim=1) / valid_cnt              # [B, D]

        diff_sq = ((t - mean_pool.unsqueeze(1)) ** 2) * m_exp
        spatial_std = torch.sqrt(diff_sq.sum(dim=1) / valid_cnt + 1e-5)

        neg_inf = -1e4
        tau = 0.10 + 0.15 * torch.sigmoid(self.tau_raw)
        sal_logits = (torch.tanh(self.saliency_proj(t)) / tau).masked_fill(m_exp == 0, neg_inf)
        sal_w = F.softmax(sal_logits, dim=1).mean(dim=-1, keepdim=True)
        sal_pool = (sal_w * t).sum(dim=1)

        scores = (self.anom_scorer(t).squeeze(-1) / math.sqrt(D)).masked_fill(mask_f == 0, neg_inf)
        k = max(1, min(L, int(math.ceil(L * 0.05))))
        top_v, top_i = torch.topk(scores, k=k, dim=1)
        top_w = F.softmax(top_v, dim=1).unsqueeze(-1)
        top_p = torch.gather(t, dim=1, index=top_i.unsqueeze(-1).expand(-1, -1, D))
        top5_pool = (top_w * top_p).sum(dim=1)

        mil_a = self.attn_w(torch.tanh(self.attn_V(t)) * torch.sigmoid(self.attn_U(t))).squeeze(-1)
        mil_a = mil_a.masked_fill(mask_f == 0, neg_inf)
        mil_w = F.softmax(mil_a, dim=1).unsqueeze(-1)
        mil_pool = (mil_w * t).sum(dim=1)

        cat = self.norm_stats(torch.cat([spatial_std, sal_pool, top5_pool, mil_pool], dim=-1))
        return mean_pool + torch.tanh(self.gate_alpha) * self.proj_res(cat)


class MultiViewAttentionAggregator(nn.Module):
    """
    Aggregates `[global_view, crop_1, ..., crop_M]` representations `[B, V, D]` into `[B, 3 * D]`
    using `concat([global_view, gated_attn_crops, amax_crops])` so sharp local crop anomalies
    are never diluted by clean crops.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, v_feats: torch.Tensor) -> torch.Tensor:
        # v_feats: [B, V, D]
        if v_feats.shape[1] == 1:
            return torch.cat([v_feats[:, 0], v_feats[:, 0], v_feats[:, 0]], dim=-1)
        g_feat = v_feats[:, 0]
        local_v = v_feats[:, 1:]  # [B, V-1, D]
        w = F.softmax(self.gate(local_v).squeeze(-1), dim=1).unsqueeze(-1)
        attn_local = (w * local_v).sum(dim=1)
        max_local = local_v.amax(dim=1)
        return torch.cat([g_feat, attn_local, max_local], dim=-1)


class QingyuVisionQualityTower(nn.Module):
    """
    Unified Pure-Real Vision Tower wrapping any of the 6 Tier-1 / Tier-2 backbones
    (`convnextv2_base`, `pe_core_large`, `dinov3_vitl16`, `siglip2_so400m_naflex`, `cradio_v4_h`, `internvit_6b`)
    with `UnimodalQuadraticResidualHead` + `TipAdapterPrototypeBank`.
    """
    def __init__(
        self,
        backbone_key: str,
        phys_dim: int = 36,
        unfreeze_last_k: int = 6,
        use_lora_6b: bool = True,
        lora_rank: int = 32,
    ):
        super().__init__()
        self.backbone_key = backbone_key
        self.is_naflex_dict_input = False
        self.is_native_aspect_pair = False

        (
            self.backbone,
            self._encode_single_view,
            mean,
            std,
            blocks,
            single_view_dim,
            mode_tag,
        ) = self._build_raw_backbone(backbone_key, unfreeze_last_k, use_lora_6b, lora_rank)

        self.mode_tag = mode_tag
        self.register_buffer("img_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        self.view_agg = MultiViewAttentionAggregator(single_view_dim)
        self.head = UnimodalQuadraticResidualHead(
            in_dim=single_view_dim * 3,
            phys_dim=phys_dim,
            hidden_dim=768 if single_view_dim <= 2048 else 1024,
            proj_dim=512,
            dropout=0.20,
            max_prototypes=128,
        )

    def _unfreeze_top_blocks(self, model: nn.Module, blocks: Optional[Sequence[nn.Module]], k: int) -> None:
        for p in model.parameters():
            p.requires_grad = False
        if blocks is not None and len(blocks) > 0:
            k_eff = min(len(blocks), max(1, k))
            for blk in list(blocks)[-k_eff:]:
                for p in blk.parameters():
                    p.requires_grad = True
        # Always unfreeze final norms & our added tap/ABMIL modules
        for n, p in model.named_parameters():
            if any(tag in n for tag in ("norm", "post_layernorm", "head_norm", "tap3", "dual_tap", "abmil", "stages.3", "stages.2")):
                p.requires_grad = True

    def _build_raw_backbone(
        self,
        key: str,
        unfreeze_last_k: int,
        use_lora_6b: bool,
        lora_rank: int,
    ):
        if key == "convnextv2_base":
            import timm
            p_cand = resolve_model_path("convnextv2_base")
            if os.path.isdir(p_cand) and os.path.exists(os.path.join(p_cand, "model.safetensors")):
                from safetensors.torch import load_file
                m = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=False, num_classes=0)
                m.load_state_dict(load_file(os.path.join(p_cand, "model.safetensors")), strict=False)
            else:
                m = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True, num_classes=0)
            dc = timm.data.resolve_data_config({}, model=m)
            # Full finetuning on 0.089B ConvNeXtV2-Base with spatial mean + std + max pooling (3 * 1024 -> 1024 proj)
            for p in m.parameters():
                p.requires_grad = True
            m.spatial_proj = nn.Sequential(
                nn.LayerNorm(1024 * 3),
                nn.Linear(1024 * 3, 1024),
                nn.GELU(),
            )

            def fwd_cnv(x: torch.Tensor) -> torch.Tensor:
                feat_map = m.forward_features(x)  # [N, 1024, H', W']
                if feat_map.ndim == 4:
                    f_mean = feat_map.float().mean(dim=(2, 3))
                    f_std = torch.sqrt(feat_map.float().var(dim=(2, 3), unbiased=False) + 1e-5)
                    f_max = feat_map.float().amax(dim=(2, 3))
                    return m.spatial_proj(torch.cat([f_mean, f_std, f_max], dim=-1))
                return feat_map.float()

            return m, fwd_cnv, dc["mean"], dc["std"], m.stages, 1024, "multi_crop_448"

        if key == "pe_core_large":
            import timm
            from safetensors.torch import load_file
            p_cand = resolve_model_path("pe_core_large")
            w_path = p_cand if os.path.isfile(p_cand) else os.path.join(p_cand, "model.safetensors")
            m = timm.create_model("vit_pe_core_large_patch14_336", pretrained=False, num_classes=0, dynamic_img_size=True)
            m.load_state_dict(load_file(w_path), strict=False)
            dc = timm.data.resolve_data_config({}, model=m)
            # Blueprint Section 3.5: Intermediate Readout at 0-indexed blocks [15, 19, 22] (1-indexed layers 16, 20, 23)
            m.tap3 = IntermediateThreeLayerGatedTap(m.blocks, m.embed_dim, tap_indices=[15, 19, 22])
            m.abmil = TokenGatedABMILPool(m.embed_dim)
            self._unfreeze_top_blocks(m, m.blocks, max(unfreeze_last_k, 9))

            def fwd_pec(x: torch.Tensor) -> torch.Tensor:
                t = m.forward_features(x).float()
                t = m.tap3.fuse(t)
                n_pref = getattr(m, "num_prefix_tokens", 1)
                if n_pref > 0:
                    pt = torch.cat([t[:, :1], t[:, n_pref:]], dim=1)
                    return torch.cat([t[:, 0], m.abmil(pt, has_cls=True)], dim=1)
                return torch.cat([t.mean(dim=1), m.abmil(t, has_cls=False)], dim=1)

            return m, fwd_pec, dc["mean"], dc["std"], m.blocks, m.embed_dim * 2, "multi_crop_448"

        if key == "dinov3_vitl16":
            import timm
            from safetensors.torch import load_file
            p_cand = resolve_model_path("dinov3_vitl16")
            w_path = p_cand if os.path.isfile(p_cand) else os.path.join(p_cand, "model.safetensors")
            m = timm.create_model("vit_large_patch16_dinov3", pretrained=False, num_classes=0, dynamic_img_size=True)
            m.load_state_dict(load_file(w_path), strict=True)
            dc = timm.data.resolve_data_config({}, model=m)
            m.tap3 = IntermediateThreeLayerGatedTap(m.blocks, m.embed_dim, tap_indices=[15, 19, 22])
            m.abmil = TokenGatedABMILPool(m.embed_dim)
            self._unfreeze_top_blocks(m, m.blocks, max(unfreeze_last_k, 8))
            self.is_native_aspect_pair = True

            def fwd_dino(x: torch.Tensor) -> torch.Tensor:
                t = m.forward_features(x).float()
                t = m.tap3.fuse(t)
                n_pref = getattr(m, "num_prefix_tokens", 1)
                pt = torch.cat([t[:, :1], t[:, n_pref:]], dim=1)
                return torch.cat([t[:, 0], m.abmil(pt, has_cls=True)], dim=1)

            return m, fwd_dino, dc["mean"], dc["std"], m.blocks, m.embed_dim * 2, "native_aspect_plus_crops"

        if key == "siglip2_so400m_naflex":
            from transformers import AutoModel, AutoImageProcessor, SiglipVisionModel
            repo = resolve_model_path("siglip2_so400m_naflex")
            vis = None
            proc = None
            is_true_naflex = False
            for r_try in [repo, "google/siglip2-so400m-patch16-naflex", "google/siglip2-so400m-patch14-384"]:
                for loc_only in (True, False):
                    try:
                        proc = AutoImageProcessor.from_pretrained(r_try, local_files_only=loc_only)
                        full_m = AutoModel.from_pretrained(r_try, attn_implementation="sdpa", local_files_only=loc_only)
                        vis = full_m.vision_model if hasattr(full_m, "vision_model") else full_m
                        is_true_naflex = "naflex" in r_try.lower() or "naflex" in type(vis).__name__.lower()
                        del full_m
                        break
                    except Exception:
                        continue
                if vis is not None:
                    break

            if vis is None:
                raise RuntimeError("Could not load SigLIP2-SO400M-NaFlex or SigLIP2-SO400M vision tower.")

            self.naflex_processor = proc
            blocks = vis.encoder.layers
            hid = vis.config.hidden_size  # 1152
            vis.tap3 = IntermediateThreeLayerGatedTap(blocks, hid, tap_indices=[max(0, len(blocks) - 9), max(0, len(blocks) - 5), len(blocks) - 2])
            vis.masked_abmil = MaskedTokenGatedABMILPool(hid)
            self._unfreeze_top_blocks(vis, blocks, max(unfreeze_last_k, 8))

            if is_true_naflex:
                self.is_naflex_dict_input = True

                def fwd_naflex(naflex_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
                    out = vis(
                        pixel_values=naflex_batch["pixel_values"],
                        pixel_attention_mask=naflex_batch.get("pixel_attention_mask", None),
                        spatial_shapes=naflex_batch.get("spatial_shapes", None),
                    )
                    t = vis.tap3.fuse(out.last_hidden_state.float())
                    pmask = naflex_batch.get("pixel_attention_mask", None)
                    pooled = vis.masked_abmil(t, mask=pmask)
                    if pmask is not None:
                        mf = pmask.unsqueeze(-1).float()
                        g_mean = (t * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                    else:
                        g_mean = t.mean(dim=1)
                    return torch.cat([g_mean, pooled], dim=1)

                return vis, fwd_naflex, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), blocks, hid * 2, "naflex_dict"
            else:
                # Interpolated dynamic-aspect fallback if server has `siglip2-so400m-patch14-384` cached offline
                emb = vis.embeddings
                orig_emb_fwd = emb.forward
                emb.forward = lambda pixel_values, **kw: orig_emb_fwd(pixel_values, interpolate_pos_encoding=True)
                self.is_native_aspect_pair = True

                def fwd_sig_interp(x: torch.Tensor) -> torch.Tensor:
                    t = vis.tap3.fuse(vis(pixel_values=x).last_hidden_state.float())
                    return torch.cat([t.mean(dim=1), vis.masked_abmil(t, mask=None)], dim=1)

                return vis, fwd_sig_interp, proc.image_mean, proc.image_std, blocks, hid * 2, "native_aspect_plus_crops"

        if key == "cradio_v4_h":
            from transformers import AutoModel
            repo = resolve_model_path("cradio_v4_h")
            m = AutoModel.from_pretrained(repo, trust_remote_code=True)
            blocks = m.radio_model.model.blocks
            m.tap3 = IntermediateThreeLayerGatedTap(blocks, 1280, tap_indices=[max(0, len(blocks) - 10), max(0, len(blocks) - 5), len(blocks) - 2])
            m.abmil = TokenGatedABMILPool(1280)
            self._unfreeze_top_blocks(m, blocks, max(unfreeze_last_k, 8))
            self.is_native_aspect_pair = True

            def fwd_radio(x_01: torch.Tensor) -> torch.Tensor:
                # Note: C-RADIO expects [0, 1] RGB inputs (mean=(0,0,0), std=(1,1,1))
                summ, feat = m(x_01)
                feat_f = m.tap3.fuse(feat.float())
                return torch.cat([summ.float(), m.abmil(feat_f, has_cls=False)], dim=1)

            return m, fwd_radio, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), blocks, int(2560 + 1280), "native_aspect_plus_crops"

        if key == "internvit_6b":
            from safetensors.torch import load_file
            sys.path.insert(0, "/tmp/race18")
            repo = resolve_model_path("internvit_6b")
            try:
                from internvit_pkg.configuration_intern_vit import InternVisionConfig
                from internvit_pkg.modeling_intern_vit import InternVisionModel
                cfg = InternVisionConfig.from_pretrained(repo)
                m = InternVisionModel(cfg)
                sd = {}
                for f in sorted(glob.glob(os.path.join(repo, "model-*.safetensors"))):
                    sd.update(load_file(f))
                m.load_state_dict(sd, strict=False)
                del sd
            except ImportError:
                from transformers import AutoModel
                m = AutoModel.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16)
                cfg = m.config

            m = m.to(torch.bfloat16)
            blocks = m.encoder.layers
            if use_lora_6b:
                from peft import LoraConfig, get_peft_model
                for p in m.parameters():
                    p.requires_grad = False
                l_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    lora_dropout=0.05,
                    target_modules=["qkv", "proj", "fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"],
                    bias="none",
                )
                m = get_peft_model(m, l_cfg)
            else:
                self._unfreeze_top_blocks(m, blocks, max(unfreeze_last_k, 6))

            raw_m = m.base_model.model if hasattr(m, "base_model") else m
            raw_m.encoder.gradient_checkpointing = False
            m.dual_tap = DualLayerTapWrapper(raw_m.encoder.layers, 3200).float()
            m.abmil = TokenGatedABMILPool(3200).float()
            _cur_grid = [int(getattr(cfg, "image_size", 448))]

            def _ensure_pos_448(s: int):
                if s == _cur_grid[0]:
                    return
                pe = raw_m.embeddings.position_embedding
                ps = int(getattr(cfg, "patch_size", 14))
                with torch.no_grad():
                    cls_t = pe[:, :1, :]
                    rest = pe[:, 1:, :].reshape(1, _cur_grid[0] // ps, _cur_grid[0] // ps, -1).permute(0, 3, 1, 2)
                    rest = F.interpolate(rest.float(), size=(s // ps, s // ps), mode="bicubic", align_corners=False)
                    rest = rest.to(pe.dtype).reshape(1, pe.shape[-1], -1).permute(0, 2, 1)
                    pe.data = torch.cat([cls_t, rest], dim=1).contiguous()
                _cur_grid[0] = s

            def fwd_iv6b(x: torch.Tensor) -> torch.Tensor:
                _ensure_pos_448(x.shape[-1])
                out = m(pixel_values=x.to(torch.bfloat16))
                t = out.last_hidden_state.float()
                if m.dual_tap._mid_cache is not None:
                    m.dual_tap._mid_cache = m.dual_tap._mid_cache.float()
                t = m.dual_tap.fuse(t)
                return torch.cat([t[:, 0], m.abmil(t, has_cls=True)], dim=1)

            return m, fwd_iv6b, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), raw_m.encoder.layers, 6400, "multi_crop_448"

        raise ValueError(f"Unsupported Qingyu backbone key: {key}")

    def _norm_img(self, x_01: torch.Tensor) -> torch.Tensor:
        return (x_01 - self.img_mean) / self.img_std

    def forward(
        self,
        views_01: Optional[torch.Tensor] = None,
        phys_feat: Optional[torch.Tensor] = None,
        native_global_list: Optional[List[torch.Tensor]] = None,
        naflex_views_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Supports all 3 input modes seamlessly:
          1. `multi_crop_448`: `views_01` of shape `[B, V, 3, 448, 448]` in `[0, 1]`.
          2. `native_aspect_plus_crops`: `native_global_list` (list of `B` variable-aspect `[1, 3, H_i, W_i]` tensors)
             + `views_01` (`[B, V_crop, 3, 448, 448]` 1:1 guided crops).
          3. `naflex_dict`: `naflex_views_list` (list of `V` batched processor dicts `[B, max_patches, ...]`).
        """
        if self.mode_tag == "naflex_dict" and naflex_views_list is not None:
            v_embs = [self._encode_single_view(vd) for vd in naflex_views_list]
            v_stack = torch.stack(v_embs, dim=1)  # [B, V, D]
        elif self.mode_tag == "native_aspect_plus_crops" and native_global_list is not None:
            # Group same-shape native-aspect tensors in the mini-batch to batch GPU calls cleanly!
            B = len(native_global_list)
            g_feats = [None] * B
            shape_groups: Dict[Tuple[int, int], List[int]] = {}
            for bi, gt in enumerate(native_global_list):
                hw = (int(gt.shape[-2]), int(gt.shape[-1]))
                shape_groups.setdefault(hw, []).append(bi)
            for _, idxs in shape_groups.items():
                gx = torch.cat([native_global_list[i] for i in idxs], dim=0)
                gf = self._encode_single_view(self._norm_img(gx))
                for loc_j, orig_i in enumerate(idxs):
                    g_feats[orig_i] = gf[loc_j]
            g_stack = torch.stack(g_feats, dim=0).unsqueeze(1)  # [B, 1, D]

            if views_01 is not None and views_01.shape[1] > 0:
                B_, Vc, C, H, W = views_01.shape
                cf = self._encode_single_view(self._norm_img(views_01.reshape(B_ * Vc, C, H, W))).view(B_, Vc, -1)
                v_stack = torch.cat([g_stack, cf], dim=1)
            else:
                v_stack = g_stack
        else:
            assert views_01 is not None, "views_01 must be provided for multi_crop_448 mode"
            B, V, C, H, W = views_01.shape
            flat = self._norm_img(views_01.reshape(B * V, C, H, W))
            v_stack = self._encode_single_view(flat).view(B, V, -1)

        vis_summary = self.view_agg(v_stack)
        return self.head(vis_summary, phys_feat=phys_feat)

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Saves ONLY trainable parameters + buffers (`phys_mu`, `phys_sd`, `proto_bank`) for lightweight Model Soup."""
        req = {n for n, p in self.named_parameters() if p.requires_grad}
        out = {}
        for k, v in self.state_dict().items():
            if (k in req) or k.startswith("head.") or k.startswith("view_agg.") or ("tap3" in k) or ("dual_tap" in k) or ("abmil" in k):
                out[k] = v.detach().cpu()
        return out
