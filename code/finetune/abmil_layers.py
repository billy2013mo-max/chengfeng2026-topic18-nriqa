#!/usr/bin/env python3
"""
code/finetune/abmil_layers.py
Token-Level Gated ABMIL + Top-5% Anomaly Pooling + Spatial Std + Intermediate Layer Tap
Resolves the 1024-patch spatial dilution bottleneck in NR-IQA local BadCase detection.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenGatedABMILPool(nn.Module):
    """
    Replaces naive `t[:, 1:].mean(1)` inside each 448x448 or 384x384 ViT view.
    Extracts 4 complementary statistics across the N=1024 spatial patch tokens:
      1. Global semantic anchor: (CLS + Mean) / 2
      2. Spatial standard deviation: captures bokeh boundary gradients & local texture loss
      3. Top-5% anomaly token pooling: captures tiny extra fingers, local text garbling, halo edges
      4. Gated Attention MIL (Ilse et al., ICML 2018): learned soft instance selection
    """
    def __init__(self, dim: int, topk_ratio: float = 0.05, bottleneck: int = 256):
        super().__init__()
        self.dim = dim
        self.topk_ratio = topk_ratio

        # Lightweight anomaly scorer for Top-k% patch selection
        self.anom_scorer = nn.Linear(dim, 1)

        # Gated ABMIL attention (Ilse et al., ICML 2018)
        self.attn_V = nn.Linear(dim, bottleneck)
        self.attn_U = nn.Linear(dim, bottleneck)
        self.attn_w = nn.Linear(bottleneck, 1)

        # Residual fusion projection back to `dim` (initialized near identity on global anchor)
        self.norm_stats = nn.LayerNorm(dim * 3)
        self.proj_residual = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # Zero-init final linear layer so pretrained backbone starts smoothly from mean pool
        nn.init.zeros_(self.proj_residual[-1].weight)
        nn.init.zeros_(self.proj_residual[-1].bias)
        self.gate_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, tokens: torch.Tensor, has_cls: bool = True) -> torch.Tensor:
        """
        Args:
            tokens: [B, L, D] sequence of visual tokens from a ViT backbone
            has_cls: whether tokens[:, 0] is a CLS token
        Returns:
            pooled: [B, D] spatially-enriched view representation
        """
        if tokens.ndim == 2:
            return tokens

        if has_cls and tokens.shape[1] > 1:
            cls_tok = tokens[:, 0]
            patch_toks = tokens[:, 1:]
            base_anchor = 0.5 * (cls_tok + patch_toks.mean(dim=1))
        else:
            patch_toks = tokens
            base_anchor = patch_toks.mean(dim=1)

        B, N, D = patch_toks.shape
        if N <= 4:
            return base_anchor

        # 1. Spatial Standard Deviation across patches [B, D]
        spatial_std = torch.sqrt(patch_toks.var(dim=1, unbiased=False) + 1e-5)

        # 2. Top-5% Anomaly Token Soft-Pooling [B, D]
        k = max(1, int(math.ceil(N * self.topk_ratio)))
        scores = self.anom_scorer(patch_toks).squeeze(-1) / math.sqrt(D)  # [B, N]
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=1)              # [B, k]
        topk_weights = F.softmax(topk_vals, dim=1).unsqueeze(-1)          # [B, k, 1]
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)           # [B, k, D]
        topk_patches = torch.gather(patch_toks, dim=1, index=idx_expanded)
        top5_pool = (topk_weights * topk_patches).sum(dim=1)              # [B, D]

        # 3. Gated ABMIL Attention Pooling [B, D]
        mil_logits = self.attn_w(
            torch.tanh(self.attn_V(patch_toks)) * torch.sigmoid(self.attn_U(patch_toks))
        ).squeeze(-1)                                                     # [B, N]
        mil_weights = F.softmax(mil_logits, dim=1).unsqueeze(-1)          # [B, N, 1]
        mil_pool = (mil_weights * patch_toks).sum(dim=1)                  # [B, D]

        # Combine complementary local statistics as residual enhancement over base_anchor
        concat_stats = self.norm_stats(torch.cat([spatial_std, top5_pool, mil_pool], dim=-1))
        delta = self.proj_residual(concat_stats)
        return base_anchor + torch.tanh(self.gate_alpha) * delta


class DualLayerTapWrapper(nn.Module):
    """
    Hooks into Layer ~75% of a Transformer block list (where high-frequency sharpness,
    blur, and texture loss peak) and fuses it with Layer 100% (where semantic overgen peaks).
    Reference: Perception Encoder (Bolya et al., CVPR 2025, arXiv:2504.13181).
    """
    def __init__(self, blocks: nn.ModuleList, dim: int):
        super().__init__()
        self.num_blocks = len(blocks)
        self.tap_idx = max(0, int(round(self.num_blocks * 0.75)) - 1)
        self.norm_mid = nn.LayerNorm(dim)
        self.norm_final = nn.LayerNorm(dim)
        self.mix_logit = nn.Parameter(torch.zeros(dim))
        self._mid_cache = None

        # Register forward hook on the 75% depth block
        blocks[self.tap_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        self._mid_cache = t

    def fuse(self, final_tokens: torch.Tensor) -> torch.Tensor:
        if self._mid_cache is None or self._mid_cache.shape != final_tokens.shape:
            return final_tokens
        w = torch.sigmoid(self.mix_logit).view(1, 1, -1)
        fused = (1.0 - 0.35 * w) * self.norm_final(final_tokens) + (0.35 * w) * self.norm_mid(self._mid_cache)
        self._mid_cache = None
        return fused
