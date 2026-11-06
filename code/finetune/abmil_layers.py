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


class NativePixelPhysicsExtractor(nn.Module):
    """24-dim zero-parameter spatial-frequency physics descriptor.

    Operates on the 3 zero-resize 1:1 native-pixel patches produced by
    `data._native_patches` (peak Sobel, peak Laplacian, median texture).

    Groups (8 dims each):
      1. Signed-Laplacian bipole ringing / overshoot  (1-2 px over-sharpening)
      2. Log-PSD high-frequency band, spectral slope, edge-to-texture ratio
         (texture loss vs global blur)
      3. Foreground/background sharpness transition + tangential sharpness
         variance (abrupt bokeh / cutout boundary)

    No learnable parameters; features are detached so the backbone is not
    penalised by this branch.
    """
    def __init__(self):
        super().__init__()
        lap_k = torch.tensor([[0.0, -1.0, 0.0],
                              [-1.0, 4.0, -1.0],
                              [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer('lap_k', lap_k)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    @staticmethod
    def _radial_bins(h, w, nbins=16):
        fy = torch.fft.fftfreq(h).view(-1, 1)
        fx = torch.fft.rfftfreq(w).view(1, -1)
        r = torch.sqrt(fy ** 2 + fx ** 2)
        r = r / (r.max() + 1e-8)
        idx = torch.clamp((r * nbins).long(), 0, nbins - 1)
        return idx, nbins

    @torch.no_grad()
    def forward(self, native_rgb: torch.Tensor) -> torch.Tensor:
        """native_rgb: [B, V, 3, H, W] in [0, 1]. Returns [B, 24]."""
        B, V, C, H, W = native_rgb.shape
        x = native_rgb.reshape(B * V, C, H, W).float()
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        eps = 1e-6

        # ---- 1. signed Laplacian bipole ringing / overshoot (8) ----
        # Disable autocast for the physics branch: bf16 ops (quantile/FFT) are unstable.
        gx = F.conv2d(gray, self.sobel_x, padding=1).float()
        gy = F.conv2d(gray, self.sobel_y, padding=1).float()
        grad = torch.sqrt(gx * gx + gy * gy + eps)
        lap1 = F.conv2d(gray, self.lap_k, padding=1).float()
        lap2 = F.conv2d(gray, self.lap_k, padding=2, dilation=2).float()

        pos = F.relu(lap1)
        neg = F.relu(-lap1)
        pos_max3 = F.max_pool2d(pos, 3, 1, 1)
        neg_max3 = F.max_pool2d(neg, 3, 1, 1)
        bipole = torch.sqrt(pos_max3 * neg_max3 + eps)

        g_mean = grad.mean(dim=(2, 3), keepdim=True)
        g_std = grad.std(dim=(2, 3), keepdim=True)
        edge = (grad > (g_mean + g_std)).float()
        edge_n = edge.sum(dim=(2, 3)).clamp(min=1.0)

        def _mean(a):
            return a.mean(dim=(1, 2, 3))

        def _edge_mean(a):
            return (a * edge).sum(dim=(2, 3)).squeeze(1) / edge_n.squeeze(1)

        f1 = torch.stack([
            _mean(bipole),
            bipole.flatten(1).amax(1),
            _edge_mean(bipole),
            _mean(pos) / (_mean(grad) + eps),
            _mean(neg) / (_mean(grad) + eps),
            _mean(F.relu(lap2)) / (_mean(grad) + eps),
            _mean(torch.abs(lap1)) / (_mean(grad) + eps),
            _mean(torch.abs(lap2)) / (_mean(grad) + eps),
        ], dim=1)

        # ---- 2. log-PSD high-frequency band / slope / edge-texture (8) ----
        Fp = torch.fft.rfft2(gray, norm='ortho')
        power = Fp.real ** 2 + Fp.imag ** 2
        idx, nbins = self._radial_bins(H, W, nbins=16)
        idx = idx.to(power.device)
        # [B*V, nbins]
        band = torch.zeros(power.shape[0], nbins, device=power.device, dtype=power.dtype)
        band.scatter_add_(1, idx.view(1, -1).expand(power.shape[0], -1), power.flatten(1))
        counts = torch.zeros(nbins, device=power.device, dtype=power.dtype)
        counts.scatter_add_(0, idx.view(-1), torch.ones_like(idx.view(-1), dtype=power.dtype))
        band = band / counts.clamp(min=1.0)[None, :]
        total = band.sum(1, keepdim=True) + eps
        # high band corresponds to radial bins 0.20-0.48 of max frequency
        hi = band[:, 3:8].sum(1) / total.squeeze(1)
        lo = band[:, 1:3].sum(1) / total.squeeze(1)
        # spectral slope: linear fit of log(band) vs log(bin index)
        logb = torch.log(band + eps)
        k = torch.arange(nbins, device=power.device, dtype=power.dtype)
        kc = k - k.mean()
        slope = (logb * kc[None, :]).sum(1) / (kc ** 2).sum()
        # edge-to-texture ratio and local variance
        gflat = grad.flatten(1)
        g_p99 = torch.quantile(gflat, 0.99, dim=1)
        g_med = torch.quantile(gflat, 0.50, dim=1)
        edge_tex = g_p99 / (g_med + eps)
        gvar = grad.var(dim=(1, 2, 3))
        gvar_edge = ((grad - g_mean) ** 2 * edge).sum(dim=(2, 3)).squeeze(1) / edge_n.squeeze(1)
        f2 = torch.stack([hi, lo, slope, edge_tex, gvar, gvar_edge, _mean(gray), _mean(gray * gray)], dim=1)

        # ---- 3. foreground/background sharpness transition (8) ----
        yy = torch.linspace(-1, 1, H, device=gray.device).view(1, 1, H, 1)
        xx = torch.linspace(-1, 1, W, device=gray.device).view(1, 1, 1, W)
        r = torch.sqrt(yy * yy + xx * xx)
        center = (r < 0.5).float()
        ring_in = ((r >= 0.30) & (r < 0.55)).float()
        ring_out = ((r >= 0.70) & (r < 0.95)).float()
        border = (r >= 0.75).float()

        def _mask_mean(a, m):
            return (a * m).sum(dim=(2, 3)).squeeze(1) / m.sum().clamp(min=1.0)

        lap_abs = torch.abs(lap1)
        f3 = torch.stack([
            _mask_mean(lap_abs, center) / (_mask_mean(lap_abs, border) + eps),
            _mask_mean(grad, center) / (_mask_mean(grad, border) + eps),
            _mask_mean(lap_abs, ring_in) / (_mask_mean(lap_abs, ring_out) + eps),
            _mask_mean(grad * grad, center) / (_mask_mean(grad * grad, border) + eps),
            (ring_in * lap_abs).sum(dim=(2, 3)).squeeze(1) / (ring_in.sum().clamp(min=1.0)),
            (ring_out * lap_abs).sum(dim=(2, 3)).squeeze(1) / (ring_out.sum().clamp(min=1.0)),
            (center * grad).sum(dim=(2, 3)).squeeze(1) / (center.sum().clamp(min=1.0)),
            (border * grad).sum(dim=(2, 3)).squeeze(1) / (border.sum().clamp(min=1.0)),
        ], dim=1)

        feat = torch.cat([f1, f2, f3], dim=1)  # [B*V, 24]
        feat = feat.view(B, V, 24).mean(dim=1)  # aggregate the 3 patches
        return torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
