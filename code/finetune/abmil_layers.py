#!/usr/bin/env python3
"""
code/finetune/abmil_layers.py
Complete implementation of:
1. TokenGatedABMILPool:
   - Learnable-Temperature Spatial Saliency Gate (tau_c in [0.10, 0.25], BREAKTHROUGH_SOLUTION_REPORT Sec 3.2.3)
   - Top-5% Hard-Instance Anomaly Token Pooling
   - Spatial Standard Deviation Pooling (captures bokeh boundary gradients & local texture smoothing)
   - Gated Attention MIL (Ilse et al., ICML 2018)
2. DualLayerTapWrapper (Upgraded to 3-Layer Shallow/Mid/Deep Pyramid Tap, Sec 3.2.1):
   - Simultaneously extracts Layer ~67% (e.g., Layer 32 in InternViT-6B for geometric edges/halos),
     Layer ~83% (e.g., Layer 40 for local texture/blur), and Layer 100% (Layer 48 for semantic overgen).
   - Zero-initialized residual adapters guarantee 100% preservation of pretrained representations at step 0.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenGatedABMILPool(nn.Module):
    """
    Replaces naive `t[:, 1:].mean(1)` inside each 448x448 or 384x384 ViT view.
    Combines:
      1. Global semantic anchor: (CLS + Mean) / 2
      2. Spatial standard deviation: captures bokeh boundary gradients & local texture loss
      3. Learnable-Temperature Spatial Saliency Gate (tau in [0.10, 0.25], Sec 3.2.3) + Top-5% Anomaly Pooling
      4. Gated Attention MIL (Ilse et al., ICML 2018): learned soft instance selection
    """
    def __init__(self, dim: int, topk_ratio: float = 0.05, bottleneck: int = 256, num_saliency_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.topk_ratio = topk_ratio
        self.num_heads = num_saliency_heads

        # Multi-head spatial saliency scorer with learnable temperature tau_c in [0.10, 0.25] (Sec 3.2.3)
        self.saliency_proj = nn.Linear(dim, num_saliency_heads)
        # Raw parameter mapped via sigmoid to [0.10, 0.25]
        self.tau_raw = nn.Parameter(torch.zeros(1, 1, num_saliency_heads))

        # Lightweight anomaly scorer for Top-5% patch selection
        self.anom_scorer = nn.Linear(dim, 1)

        # Gated ABMIL attention (Ilse et al., ICML 2018)
        self.attn_V = nn.Linear(dim, bottleneck)
        self.attn_U = nn.Linear(dim, bottleneck)
        self.attn_w = nn.Linear(bottleneck, 1)

        # Residual fusion projection back to `dim` (strictly zero-initialized at step 0)
        self.norm_stats = nn.LayerNorm(dim * 4)
        self.proj_residual = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.proj_residual[-1].weight)
        nn.init.zeros_(self.proj_residual[-1].bias)
        self.gate_alpha = nn.Parameter(torch.tensor(0.25))

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

        # 2. Learnable-Temperature Spatial Saliency Gate (tau_c in [0.10, 0.25], Sec 3.2.3)
        tau = 0.10 + 0.15 * torch.sigmoid(self.tau_raw)                      # [1, 1, H] in [0.10, 0.25]
        sal_logits = torch.tanh(self.saliency_proj(patch_toks)) / tau        # [B, N, H]
        sal_weights = F.softmax(sal_logits, dim=1).mean(dim=-1, keepdim=True) # [B, N, 1]
        saliency_pool = (sal_weights * patch_toks).sum(dim=1)                # [B, D]

        # 3. Top-5% Anomaly Token Soft-Pooling [B, D]
        k = max(1, int(math.ceil(N * self.topk_ratio)))
        scores = self.anom_scorer(patch_toks).squeeze(-1) / math.sqrt(D)     # [B, N]
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=1)                 # [B, k]
        topk_weights = F.softmax(topk_vals, dim=1).unsqueeze(-1)             # [B, k, 1]
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)              # [B, k, D]
        topk_patches = torch.gather(patch_toks, dim=1, index=idx_expanded)
        top5_pool = (topk_weights * topk_patches).sum(dim=1)                 # [B, D]

        # 4. Gated ABMIL Attention Pooling [B, D]
        mil_logits = self.attn_w(
            torch.tanh(self.attn_V(patch_toks)) * torch.sigmoid(self.attn_U(patch_toks))
        ).squeeze(-1)                                                        # [B, N]
        mil_weights = F.softmax(mil_logits, dim=1).unsqueeze(-1)             # [B, N, 1]
        mil_pool = (mil_weights * patch_toks).sum(dim=1)                     # [B, D]

        # Combine all 4 local anomaly streams as a zero-init residual enhancement over base_anchor
        concat_stats = self.norm_stats(
            torch.cat([spatial_std, saliency_pool, top5_pool, mil_pool], dim=-1)
        )
        delta = self.proj_residual(concat_stats)
        return base_anchor + torch.tanh(self.gate_alpha) * delta


class DualLayerTapWrapper(nn.Module):
    """
    Three-Layer Shallow/Mid/Deep Pyramid Tap (BREAKTHROUGH_SOLUTION_REPORT Sec 3.2.1):
    Simultaneously hooks:
      - Shallow-Mid Layer (~67% depth, e.g., Layer 32 in InternViT-6B): geometric edges & oversharpening halos
      - Mid-Deep Layer (~83% depth, e.g., Layer 40 in InternViT-6B): local texture loss & motion blur
      - Final Layer (100% depth, e.g., Layer 48 in InternViT-6B): semantic over-generation & bokeh depth
    Uses zero-initialized residual adapters so step-0 representations match the pretrained backbone 100%.
    """
    def __init__(self, blocks: nn.ModuleList, dim: int):
        super().__init__()
        self.num_blocks = len(blocks)
        self.tap_67_idx = max(0, int(round(self.num_blocks * (2.0 / 3.0))) - 1)
        self.tap_83_idx = max(0, int(round(self.num_blocks * (5.0 / 6.0))) - 1)

        self.norm_67 = nn.LayerNorm(dim)
        self.norm_83 = nn.LayerNorm(dim)
        self.pyramid_adapter = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.pyramid_adapter[-1].weight)
        nn.init.zeros_(self.pyramid_adapter[-1].bias)
        self.mix_logit = nn.Parameter(torch.full((dim,), -2.0))

        self._cache_67 = None
        self._cache_83 = None
        self._mid_cache = None  # backward compatibility alias

        blocks[self.tap_67_idx].register_forward_hook(self._hook_67)
        if self.tap_83_idx != self.tap_67_idx:
            blocks[self.tap_83_idx].register_forward_hook(self._hook_83)

    def _hook_67(self, module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        self._cache_67 = t
        self._mid_cache = t

    def _hook_83(self, module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        self._cache_83 = t

    def fuse(self, final_tokens: torch.Tensor) -> torch.Tensor:
        if self._cache_67 is None or self._cache_67.shape != final_tokens.shape:
            self._cache_67 = None
            self._cache_83 = None
            self._mid_cache = None
            return final_tokens

        t67 = self.norm_67(self._cache_67.float())
        if self._cache_83 is not None and self._cache_83.shape == final_tokens.shape:
            t83 = self.norm_83(self._cache_83.float())
        else:
            t83 = t67

        pyr_feat = self.pyramid_adapter(torch.cat([t67, t83], dim=-1))
        w = torch.sigmoid(self.mix_logit).view(1, 1, -1)
        fused = final_tokens + 0.35 * w * pyr_feat

        self._cache_67 = None
        self._cache_83 = None
        self._mid_cache = None
        return fused
