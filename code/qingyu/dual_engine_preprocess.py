#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/dual_engine_preprocess.py
=====================================
Pillar 2 (Section 3 of Blueprint):
Dual-Engine Native-Aspect Global + 1:1 Pixel-Exact Physics & Depth-Decoupled Preprocessor.

Strictly implements every mathematical operator in Section 3.1 – 3.4:
  1. Vector UI / Subtitle Stroke Mask `M_UI(x, y)` (Section 3.2):
     Detects high-contrast equal-width bimodal fill+outline subtitle/watermark strokes
     (e.g., white/yellow text with dark 1–3px border) and dilates by 5px so sharp vector
     subtitles NEVER trigger false `过锐` MEOR peaks or guided crops.
  2. 1:1 Morphological Envelope Overshoot-Undershoot Ratio `MEOR(x, y)` (Equation in Section 3.2):
       Y_tilde = MedianFilter_7x7(Y)
       MEOR(x, y) = [max(0, Y - delta_5(Y_tilde)) + max(0, eps_5(Y_tilde) - Y)]
                    / (delta_9(Y_tilde) - eps_9(Y_tilde) + 1e-4)
     Zero response to ideal clean step edges; strong sharp peaks on 1–3px USM/ISP overshoot halos.
  3. Monocular Depth `D(x, y)` (`Depth-Anything-V2-Small`, 0.025B, Apache-2.0) +
     1:1 Re-blurring Defocus Sharpness `S(x, y)` (Section 3.3):
       S(x, y) = ||grad(Y)||_2 / (||grad(G_sigma * Y)||_2 + 1e-4)
       S_fg: quantiles of S(x, y) strictly inside foreground {D(x, y) > tau_fg}
       A_cutout = Percentile_99( ||grad(S)||_2 * exp(-gamma * ||grad(D)||_2) | D > tau_fg )
     Physically separates natural optical shallow-DOF bokeh (`||grad D||` co-varies with `||grad S||`)
     from abrupt portrait segmentation / background blur artifacts (`||grad S||` jumps where `||grad D|| ~ 0`).
  4. Deterministic 1:1 Guided Crops & Semantic ROIs (Section 3.4):
     - Four `448x448` 1:1 native crops for Tier-1 (`ConvNeXtV2-Base`, `PE-Core-Large`):
         [0] Foreground Peak-Sharpness Crop (`模糊` verification)
         [1] Peak Non-UI `MEOR` Halo Crop (`过锐` verification)
         [2] Peak `A_cutout` Foreground-Boundary Crop (`虚化突兀` verification)
         [3] Foreground Low-Gradient Micro-Texture Crop (`纹理损失` verification)
     - Two `384x384` 1:1 semantic ROIs for Tier-3 Interleaved-MRoPE (`[img_global, roi_anatomy, roi_text_tex]`):
         [0] `roi_anatomy`: Foreground skin/face/hand contour salient region
         [1] `roi_text_tex`: Scene text / high-kurtosis synthetic plastic ridge region
  5. Single-Image Strict Independence:
     Every feature & crop box depends ONLY on the single input image `x_i` (zero batch/test statistics).
"""
import os
import sys
import math
import json
import time
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.config_and_paths import (
    REPO_ROOT,
    LosslessNativeImageLoader,
    enforce_deterministic_runtime,
    load_pure_real_manifest,
    load_test_template_names,
    resolve_model_path,
    setup_sys_path_and_cwd,
)

PHYS_FEAT_DIM: int = 36


def extract_1to1_crop_pil(im, box: Sequence[int], crop_size: int):
    """
    Extracts an exact 1:1 pixel-pitch `crop_size x crop_size` PIL RGB crop without resampling!
    If an image side is smaller than `crop_size`, pads symmetrically via edge reflection
    instead of upsampling, strictly preserving the native 1:1 pixel frequency spectrum.
    """
    import numpy as np
    from PIL import Image

    w, h = im.size
    x0, y0, x1, y1 = [int(v) for v in box]
    if w >= crop_size and h >= crop_size:
        x0 = max(0, min(w - crop_size, x0))
        y0 = max(0, min(h - crop_size, y0))
        return im.crop((x0, y0, x0 + crop_size, y0 + crop_size))

    arr = np.asarray(im, dtype=np.uint8)
    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)
    top = pad_h // 2
    bot = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    mode = "reflect" if (h > 1 and w > 1) else "edge"
    arr_pad = np.pad(arr, ((top, bot), (left, right), (0, 0)), mode=mode)
    hp, wp = arr_pad.shape[:2]
    x0 = max(0, min(wp - crop_size, x0))
    y0 = max(0, min(hp - crop_size, y0))
    return Image.fromarray(arr_pad[y0 : y0 + crop_size, x0 : x0 + crop_size], mode="RGB")


def resize_native_aspect_pil(
    im,
    max_pixels: int = 768 * 768,
    min_pixels: int = 256 * 256,
    patch_mult: int = 16,
):
    """Resizes `im` preserving native aspect ratio (9:16, 16:9, 4:3, 1:1) aligned to `patch_mult`."""
    from PIL import Image

    w, h = im.size
    area = float(w * h)
    scale = 1.0
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
    elif area < min_pixels:
        scale = math.sqrt(min_pixels / area)
    nw = max(patch_mult, int(round((w * scale) / patch_mult)) * patch_mult)
    nh = max(patch_mult, int(round((h * scale) / patch_mult)) * patch_mult)
    if (nw, nh) == (w, h):
        return im
    return im.resize((nw, nh), Image.BICUBIC)


class DepthAnythingV2Estimator:
    """
    Wraps `Depth-Anything-V2-Small` (24.8M params, Apache-2.0) for fast monocular relative depth `D(x, y) in [0, 1]`.
    Automatically falls back to an analytical multi-scale focus-saliency pseudo-depth prior ONLY if weights
    cannot be loaded (e.g. air-gapped CPU unit test without model directory).
    """
    def __init__(self, device: str = "cuda", allow_download: bool = True):
        self.device = device
        self.model = None
        self.processor = None
        self.available = False
        self._init_model(allow_download=allow_download)

    def _init_model(self, allow_download: bool = True) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError:
            return

        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        self.device = self.device if use_cuda else "cpu"
        cand = resolve_model_path("depth_anything_v2_s")
        for local_only in ([True, False] if allow_download else [True]):
            try:
                self.processor = AutoImageProcessor.from_pretrained(cand, local_files_only=local_only)
                dtype = torch.float16 if use_cuda else torch.float32
                self.model = AutoModelForDepthEstimation.from_pretrained(
                    cand, torch_dtype=dtype, local_files_only=local_only
                ).to(self.device).eval()
                self.available = True
                return
            except Exception:
                continue

    def predict_depth_01(self, pil_im, target_hw: Tuple[int, int]):
        """Returns float32 relative foreground closeness map `D in [0, 1]` of shape `target_hw` (higher = closer)."""
        import numpy as np
        import torch
        import torch.nn.functional as F

        th, tw = int(target_hw[0]), int(target_hw[1])
        if self.available and self.model is not None and self.processor is not None:
            with torch.inference_mode():
                inputs = self.processor(images=pil_im, return_tensors="pt")
                inputs = {
                    k: (v.to(self.device, dtype=next(self.model.parameters()).dtype) if v.is_floating_point() else v.to(self.device))
                    for k, v in inputs.items()
                }
                out = self.model(**inputs)
                pred = out.predicted_depth  # [1, h, w]
                if pred.ndim == 3:
                    pred = pred.unsqueeze(1)
                pred = F.interpolate(pred.float(), size=(th, tw), mode="bilinear", align_corners=False)[0, 0]
                d_min = float(pred.min().item())
                d_max = float(pred.max().item())
                d_norm = (pred - d_min) / max(1e-6, d_max - d_min)
                return d_norm.cpu().numpy().astype(np.float32)

        # Analytical center-weighted multi-scale sharpness fallback (only used if Depth-Anything weights absent)
        arr = np.asarray(pil_im.resize((tw, th)), dtype=np.float32) / 255.0
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        gy, gx = np.gradient(gray)
        grad = np.hypot(gx, gy)
        yy = np.linspace(-1.0, 1.0, th, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, tw, dtype=np.float32)[None, :]
        center_prior = np.exp(-0.65 * (xx * xx + 0.8 * yy * yy))
        # Smooth local gradient density
        t_g = torch.from_numpy(grad)[None, None]
        loc_sharp = F.avg_pool2d(t_g, kernel_size=15, stride=1, padding=7)[0, 0].numpy()
        loc_sharp = loc_sharp / (float(np.percentile(loc_sharp, 95)) + 1e-6)
        d_est = np.clip(0.55 * center_prior + 0.45 * np.clip(loc_sharp, 0.0, 1.2), 0.0, 1.0)
        d_est = (d_est - d_est.min()) / max(1e-6, float(d_est.max() - d_est.min()))
        return d_est.astype(np.float32)


def _gaussian_kernel_1d(sigma: float, device, dtype):
    import torch
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    k = k / k.sum()
    return k, radius


def _gaussian_blur_2d(x, sigma: float):
    """Separable 2D Gaussian blur on `[1, 1, H, W]` tensor with reflect padding."""
    import torch.nn.functional as F
    k1d, r = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    kh = k1d.view(1, 1, 1, -1)
    kv = k1d.view(1, 1, -1, 1)
    x_pad = F.pad(x, (r, r, r, r), mode="reflect")
    out = F.conv2d(F.conv2d(x_pad, kh), kv)
    return out


def _median_filter_7x7(y_11hw):
    """
    Exact 7x7 2D Median Filter `Y_tilde = MedianFilter_7x7(Y)` on `[1, 1, H, W]`.
    Uses chunked row unfolding so peak VRAM on 4K images stays under 120 MB!
    """
    import torch
    import torch.nn.functional as F

    _, _, H, W = y_11hw.shape
    padded = F.pad(y_11hw, (3, 3, 3, 3), mode="reflect")
    out = torch.empty_like(y_11hw)
    row_step = 256
    for r0 in range(0, H, row_step):
        r1 = min(H, r0 + row_step)
        sub = padded[:, :, r0 : r1 + 6, :]  # [1, 1, (r1-r0)+6, W+6]
        patches = sub.unfold(2, 7, 1).unfold(3, 7, 1)  # [1, 1, r1-r0, W, 7, 7]
        flat = patches.reshape(1, 1, r1 - r0, W, 49)
        out[:, :, r0:r1, :] = flat.median(dim=-1).values
    return out


def _morph_dilate_erode(y_11hw, ksize: int):
    import torch.nn.functional as F
    pad = ksize // 2
    yp = F.pad(y_11hw, (pad, pad, pad, pad), mode="reflect")
    dil = F.max_pool2d(yp, kernel_size=ksize, stride=1)
    ero = -F.max_pool2d(-yp, kernel_size=ksize, stride=1)
    return dil, ero


def _window_integral_sums_2d(energy_2d, win_h: int, win_w: int):
    """Fast 2D integral-image window sum over all valid `(H - win_h + 1, W - win_w + 1)` top-left positions."""
    import numpy as np
    e = np.asarray(energy_2d, dtype=np.float64)
    H, W = e.shape
    wh = min(win_h, H)
    ww = min(win_w, W)
    csum = np.pad(np.cumsum(np.cumsum(e, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    return csum[wh:, ww:] - csum[:-wh, ww:] - csum[wh:, :-ww] + csum[:-wh, :-ww]


def _pick_diverse_top_box(
    score_map_2d,
    crop_size: int,
    orig_h: int,
    orig_w: int,
    scale_y: float,
    scale_x: float,
    taken_centers: List[Tuple[float, float]],
    min_dist_frac: float = 0.55,
) -> Tuple[int, int, int, int]:
    """Selects highest-scoring `crop_size x crop_size` 1:1 native box while enforcing spatial diversity."""
    import numpy as np

    sh, sw = score_map_2d.shape
    wh = max(4, min(sh, int(round(crop_size / max(1.0, scale_y)))))
    ww = max(4, min(sw, int(round(crop_size / max(1.0, scale_x)))))
    wsum = _window_integral_sums_2d(score_map_2d, wh, ww)
    flat_order = np.argsort(wsum.ravel())[::-1]
    min_dist_sq = (crop_size * min_dist_frac) ** 2

    chosen_x0, chosen_y0 = max(0, (orig_w - crop_size) // 2), max(0, (orig_h - crop_size) // 2)
    # Inspect top candidates (stride if huge)
    step = max(1, len(flat_order) // 4096)
    for idx in flat_order[::step]:
        ry, rx = divmod(int(idx), wsum.shape[1])
        x0 = int(round(rx * scale_x))
        y0 = int(round(ry * scale_y))
        x0 = max(0, min(max(0, orig_w - crop_size), x0))
        y0 = max(0, min(max(0, orig_h - crop_size), y0))
        cx = x0 + 0.5 * min(orig_w, crop_size)
        cy = y0 + 0.5 * min(orig_h, crop_size)
        if all((cx - tx) ** 2 + (cy - ty) ** 2 >= min_dist_sq for tx, ty in taken_centers):
            chosen_x0, chosen_y0 = x0, y0
            break
    else:
        if len(flat_order) > 0:
            ry, rx = divmod(int(flat_order[0]), wsum.shape[1])
            chosen_x0 = max(0, min(max(0, orig_w - crop_size), int(round(rx * scale_x))))
            chosen_y0 = max(0, min(max(0, orig_h - crop_size), int(round(ry * scale_y))))

    taken_centers.append((chosen_x0 + 0.5 * min(orig_w, crop_size), chosen_y0 + 0.5 * min(orig_h, crop_size)))
    return (chosen_x0, chosen_y0, chosen_x0 + crop_size, chosen_y0 + crop_size)


class DualEnginePhysicsExtractor:
    """
    End-to-end Pillar-2 Single-Image Deterministic Dual-Engine Operator:
      - Computes 1:1 native `MEOR`, `M_UI` subtitle mask, `Depth-Anything-V2-Small` `D(x,y)`,
        1:1 defocus sharpness `S(x,y)`, foreground `S_fg`, and boundary discontinuity `A_cutout`.
      - Emits:
          * `phys_vec` (`float32[36]`): calibrated physical features for Tier-1/2/3 & Stacking;
          * `crops_448_boxes` (`int32[4, 4]`): `[focus_fg, meor_non_ui, cutout_boundary, fg_micro_tex]`;
          * `rois_384_boxes` (`int32[2, 4]`): `[roi_anatomy, roi_text_tex]`;
          * `norm_peak_boxes` (`float32[4, 4]`): `[0, 1000]` normalized peak coordinates for Grounded Attribution.
    """
    def __init__(self, device: str = "cuda", allow_depth_download: bool = True, max_1to1_side: int = 2048):
        self.device = device
        self.max_1to1_side = max_1to1_side
        self.depth_estimator = DepthAnythingV2Estimator(device=device, allow_download=allow_depth_download)

    def analyze_pil_image(self, pil_im) -> Dict[str, object]:
        import numpy as np
        import torch
        import torch.nn.functional as F

        orig_w, orig_h = pil_im.size
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        dev = torch.device(self.device if use_cuda else "cpu")

        # Keep 1:1 native pixels up to 2048px (if 4K 3840px, take full 2048px 1:1 analysis grid or native if <=2048)
        if max(orig_w, orig_h) > self.max_1to1_side:
            # Do NOT blur-downsample for MEOR! Use stride-1 center/full ROI or cap only for memory safety:
            # Even 3840x2160 float32 is only 33 MB, so on CUDA we run TRUE 1:1 up to 3840px!
            cap = 4096 if use_cuda else self.max_1to1_side
        else:
            cap = max(orig_w, orig_h)

        if max(orig_w, orig_h) > cap:
            s_cap = cap / float(max(orig_w, orig_h))
            work_w = max(64, int(round(orig_w * s_cap)))
            work_h = max(64, int(round(orig_h * s_cap)))
            work_im = pil_im.resize((work_w, work_h))
        else:
            work_w, work_h = orig_w, orig_h
            work_im = pil_im

        rgb_np = np.asarray(work_im, dtype=np.float32) / 255.0  # [H, W, 3]
        rgb_t = torch.from_numpy(rgb_np).to(dev).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, 3, H, W]
        r_ch, g_ch, b_ch = rgb_t[:, 0:1], rgb_t[:, 1:2], rgb_t[:, 2:3]
        y_t = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch  # [1, 1, H, W]

        with torch.inference_mode():
            # 1. Vector UI / Subtitle Bimodal Stroke Mask M_UI(x, y)
            # Subtitles/watermarks exhibit near-saturated bright fill (white Y>0.88 or bright yellow R>0.85,G>0.75,B<0.35)
            # immediately bordered within 3 pixels by a crisp dark stroke outline (Y < 0.16)
            bright_fill = (y_t > 0.88) | ((r_ch > 0.84) & (g_ch > 0.72) & (b_ch < 0.35))
            dark_stroke = y_t < 0.16
            near_dark = F.max_pool2d(dark_stroke.float(), kernel_size=7, stride=1, padding=3) > 0.5
            near_bright = F.max_pool2d(bright_fill.float(), kernel_size=7, stride=1, padding=3) > 0.5
            bimodal_stroke = (bright_fill & near_dark) | (dark_stroke & near_bright)
            # Subtitles & platform watermarks concentrate in top 18% and bottom 28% bands (still catch ultra-crisp strokes elsewhere)
            yy_norm = torch.linspace(0.0, 1.0, work_h, device=dev).view(1, 1, work_h, 1)
            ui_band = (yy_norm < 0.18) | (yy_norm > 0.72)
            # Require high local stroke density in 15x15 window to avoid masking natural specular highlights
            stroke_dens = F.avg_pool2d(bimodal_stroke.float(), kernel_size=15, stride=1, padding=7)
            m_ui_raw = bimodal_stroke & ((ui_band & (stroke_dens > 0.06)) | (stroke_dens > 0.16))
            # Dilate by 5 pixels (11x11 max pool) per Section 3.2
            m_ui = F.max_pool2d(m_ui_raw.float(), kernel_size=11, stride=1, padding=5)  # [1, 1, H, W] in {0, 1}
            non_ui_mask = 1.0 - m_ui

            # 2. 1:1 Morphological Envelope Overshoot-Undershoot Ratio MEOR(x, y) (Section 3.2)
            y_med7 = _median_filter_7x7(y_t)
            dil5, ero5 = _morph_dilate_erode(y_med7, 5)
            dil9, ero9 = _morph_dilate_erode(y_med7, 9)
            env_span9 = dil9 - ero9
            overshoot = F.relu(y_t - dil5) + F.relu(ero5 - y_t)
            meor_full = overshoot / (env_span9 + 1e-4)
            # Evaluate MEOR strictly on non-UI active contrast edges (env_span9 > 0.08)
            active_edge = (env_span9 > 0.08).float() * non_ui_mask
            meor_masked = meor_full * active_edge

            # 3. 1:1 Gaussian Re-Blurring Defocus Sharpness S(x, y) (Zhuo & Sim, Section 3.3)
            sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=dev).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(2, 3)
            gx_y = F.conv2d(F.pad(y_t, (1, 1, 1, 1), mode="reflect"), sobel_x)
            gy_y = F.conv2d(F.pad(y_t, (1, 1, 1, 1), mode="reflect"), sobel_y)
            grad_y = torch.sqrt(gx_y * gx_y + gy_y * gy_y + 1e-8)

            y_blur1 = _gaussian_blur_2d(y_t, sigma=2.0)
            gx_b1 = F.conv2d(F.pad(y_blur1, (1, 1, 1, 1), mode="reflect"), sobel_x)
            gy_b1 = F.conv2d(F.pad(y_blur1, (1, 1, 1, 1), mode="reflect"), sobel_y)
            grad_b1 = torch.sqrt(gx_b1 * gx_b1 + gy_b1 * gy_b1 + 1e-8)

            y_blur2 = _gaussian_blur_2d(y_t, sigma=4.0)
            gx_b2 = F.conv2d(F.pad(y_blur2, (1, 1, 1, 1), mode="reflect"), sobel_x)
            gy_b2 = F.conv2d(F.pad(y_blur2, (1, 1, 1, 1), mode="reflect"), sobel_y)
            grad_b2 = torch.sqrt(gx_b2 * gx_b2 + gy_b2 * gy_b2 + 1e-8)

            # Active structure weighting avoids flat sky noise inflating 0/0 gradient ratios
            s_defocus = (grad_y / (grad_b1 + 5e-3)) * (grad_b1 > 0.015).float()
            s_defocus_wide = (grad_y / (grad_b2 + 5e-3)) * (grad_b2 > 0.012).float()

            # Downsample spatial maps to ~384px max side for fast depth fusion & crop integral search
            ds_scale = min(1.0, 384.0 / float(max(work_h, work_w)))
            ds_h = max(32, int(round(work_h * ds_scale)))
            ds_w = max(32, int(round(work_w * ds_scale)))

            s_ds = F.interpolate(s_defocus, size=(ds_h, ds_w), mode="area")
            sw_ds = F.interpolate(s_defocus_wide, size=(ds_h, ds_w), mode="area")
            grad_ds = F.interpolate(grad_y, size=(ds_h, ds_w), mode="area")
            meor_ds = F.adaptive_max_pool2d(meor_masked, output_size=(ds_h, ds_w))
            mui_ds = F.interpolate(m_ui, size=(ds_h, ds_w), mode="area")

            # Monocular Depth D(x, y) in [0, 1] at (ds_h, ds_w)
            depth_np = self.depth_estimator.predict_depth_01(work_im, target_hw=(ds_h, ds_w))
            depth_t = torch.from_numpy(depth_np).to(dev).view(1, 1, ds_h, ds_w)

            # Foreground mask {D(x, y) > tau_fg} (tau_fg = 55th percentile of D)
            tau_fg = torch.quantile(depth_t.flatten(), 0.55)
            tau_bg = torch.quantile(depth_t.flatten(), 0.35)
            fg_mask = (depth_t >= tau_fg).float() * (1.0 - (mui_ds > 0.5).float())
            if float(fg_mask.sum().item()) < 16.0:
                fg_mask = (depth_t >= tau_fg).float()
            bg_mask = (depth_t <= tau_bg).float() * (1.0 - (mui_ds > 0.5).float())

            # Spatial gradients of S(x, y) and D(x, y) for Cutout Discontinuity Operator A_cutout (Section 3.3)
            gx_s = F.conv2d(F.pad(s_ds, (1, 1, 1, 1), mode="reflect"), sobel_x)
            gy_s = F.conv2d(F.pad(s_ds, (1, 1, 1, 1), mode="reflect"), sobel_y)
            grad_s_mag = torch.sqrt(gx_s * gx_s + gy_s * gy_s + 1e-8)

            gx_d = F.conv2d(F.pad(depth_t, (1, 1, 1, 1), mode="reflect"), sobel_x)
            gy_d = F.conv2d(F.pad(depth_t, (1, 1, 1, 1), mode="reflect"), sobel_y)
            grad_d_mag = torch.sqrt(gx_d * gx_d + gy_d * gy_d + 1e-8)

            gamma_cut = 8.0
            cutout_map = grad_s_mag * torch.exp(-gamma_cut * grad_d_mag) * fg_mask

            # 8x8 MPEG/H.264 Blockiness & Chroma Glitch Detector (for 花屏)
            # Compare boundary differences at x%8==0 / y%8==0 vs interior pixel differences
            if work_h >= 32 and work_w >= 32:
                dh_all = torch.abs(y_t[:, :, 1:, :] - y_t[:, :, :-1, :])
                dw_all = torch.abs(y_t[:, :, :, 1:] - y_t[:, :, :, :-1])
                b_rows = dh_all[:, :, 7::8, :].mean()
                i_rows = dh_all[:, :, 3::8, :].mean() + 1e-5
                b_cols = dw_all[:, :, :, 7::8].mean()
                i_cols = dw_all[:, :, :, 3::8].mean() + 1e-5
                blockiness_ratio = float(0.5 * (b_rows / i_rows + b_cols / i_cols).item())
                # Extreme chroma saturation jump (decoding macroblock corruption)
                uv_jump = torch.abs((r_ch - g_ch)[:, :, 1:, :] - (r_ch - g_ch)[:, :, :-1, :]) + torch.abs(
                    (b_ch - g_ch)[:, :, 1:, :] - (b_ch - g_ch)[:, :, :-1, :]
                )
                chroma_spike_p999 = float(torch.quantile(uv_jump.flatten()[:: max(1, uv_jump.numel() // 65536)], 0.999).item())
            else:
                blockiness_ratio = 1.0
                chroma_spike_p999 = 0.0

            # High-frequency Laplacian kurtosis & skin/anatomy salience map (for Tier-3 semantic ROIs)
            lap_k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=dev).view(1, 1, 3, 3)
            lap_abs = torch.abs(F.conv2d(F.pad(y_t, (1, 1, 1, 1), mode="reflect"), lap_k))
            lap_ds = F.interpolate(lap_abs, size=(ds_h, ds_w), mode="area")

            rgb_ds = F.interpolate(rgb_t, size=(ds_h, ds_w), mode="area")
            rd, gd, bd = rgb_ds[:, 0:1], rgb_ds[:, 1:2], rgb_ds[:, 2:3]
            # Smooth YCbCr/RGB skin-likeness & foreground contour salience for `roi_anatomy`
            cb = -0.1687 * rd - 0.3313 * gd + 0.5000 * bd + 0.5
            cr = 0.5000 * rd - 0.4187 * gd - 0.0813 * bd + 0.5
            skin_prob = torch.exp(-(((cb - 0.43) / 0.09) ** 2 + ((cr - 0.60) / 0.09) ** 2))
            anatomy_sal_ds = (0.55 * skin_prob + 0.45 * (grad_ds / (grad_ds.mean() + 1e-4)).clamp(0.0, 3.0) / 3.0) * fg_mask

            # Text / plastic synthetic ridge salience (`roi_text_tex`): high local edge kurtosis & medium-high freq ridges
            lap_sq_ds = F.interpolate(lap_abs * lap_abs, size=(ds_h, ds_w), mode="area")
            local_kurt_ds = lap_sq_ds / (lap_ds * lap_ds + 1e-5)
            text_tex_sal_ds = (0.55 * (grad_ds / (grad_ds.max() + 1e-5)) + 0.45 * (local_kurt_ds / (local_kurt_ds.max() + 1e-5))) * (
                0.7 * fg_mask + 0.3
            )

            # Extract scalar quantiles strictly from this single image
            def _safe_quantiles(vals_1d, qs=(0.50, 0.75, 0.90, 0.98, 0.995)):
                if vals_1d.numel() == 0:
                    return [0.0 for _ in qs]
                if vals_1d.numel() > 131072:
                    vals_1d = vals_1d[:: max(1, vals_1d.numel() // 131072)]
                q_t = torch.tensor(qs, device=vals_1d.device, dtype=torch.float32)
                return [float(x) for x in torch.quantile(vals_1d.float(), q_t).cpu().tolist()]

            edge_vals = meor_full[active_edge > 0.5]
            meor_q = _safe_quantiles(edge_vals, (0.50, 0.85, 0.95, 0.99, 0.995))
            meor_mean = float(edge_vals.mean().item()) if edge_vals.numel() > 0 else 0.0
            meor_gt035 = float((edge_vals > 0.035).float().mean().item()) if edge_vals.numel() > 0 else 0.0
            meor_gt110 = float((edge_vals > 0.110).float().mean().item()) if edge_vals.numel() > 0 else 0.0
            ui_ratio = float(m_ui.mean().item())
            active_edge_ratio = float(active_edge.mean().item())

            s_fg_vals = s_ds[fg_mask > 0.5]
            s_bg_vals = s_ds[bg_mask > 0.5]
            s_fg_q = _safe_quantiles(s_fg_vals, (0.25, 0.50, 0.75, 0.90, 0.98))
            sw_fg_p90 = _safe_quantiles(sw_ds[fg_mask > 0.5], (0.90,))[0]
            s_bg_p50, s_bg_p90 = _safe_quantiles(s_bg_vals, (0.50, 0.90))
            fg_bg_sharp_ratio = math.log((s_fg_q[3] + 1e-3) / (s_bg_p90 + 1e-3))

            cut_fg_vals = cutout_map[fg_mask > 0.5]
            cut_p90, cut_p95, cut_p99, cut_max = _safe_quantiles(cut_fg_vals, (0.90, 0.95, 0.99, 0.999))
            depth_std = float(depth_t.std().item())
            grad_d_p95 = _safe_quantiles(grad_d_mag.flatten(), (0.95,))[0]

            # Foreground micro-texture statistics (for 纹理损失 vs 模糊 decoupling)
            fg_grad_vals = grad_ds[fg_mask > 0.5]
            fg_lap_vals = lap_ds[fg_mask > 0.5]
            fg_g_p25, fg_g_p50, fg_g_p95 = _safe_quantiles(fg_grad_vals, (0.25, 0.50, 0.95))
            fg_l_p50, fg_l_p95 = _safe_quantiles(fg_lap_vals, (0.50, 0.95))
            edge_to_tex_fg = fg_g_p95 / (fg_g_p50 + 1e-4)
            lap_to_grad_fg = fg_l_p50 / (fg_g_p50 + 1e-4)

            phys_36 = np.array([
                # [0..9] MEOR & UI stroke decoupling (过锐)
                meor_q[0], meor_q[1], meor_q[2], meor_q[3], meor_q[4],
                meor_mean, meor_gt035, meor_gt110, ui_ratio, active_edge_ratio,
                # [10..18] Foreground Depth-Decoupled Defocus Sharpness S_fg (模糊)
                s_fg_q[0], s_fg_q[1], s_fg_q[2], s_fg_q[3], s_fg_q[4],
                sw_fg_p90, s_bg_p50, s_bg_p90, fg_bg_sharp_ratio,
                # [19..24] Cutout Discontinuity A_cutout & Depth Transition (虚化突兀)
                cut_p90, cut_p95, cut_p99, cut_max, depth_std, grad_d_p95,
                # [25..30] Foreground Micro-Texture vs Edge Retention (纹理损失)
                fg_g_p25, fg_g_p50, fg_g_p95, fg_l_p50, fg_l_p95, edge_to_tex_fg,
                # [31..35] Blockiness / Chroma Glitch (花屏) & Native Geometry
                lap_to_grad_fg,
                blockiness_ratio,
                chroma_spike_p999,
                math.log(max(1.0, float(orig_w)) / max(1.0, float(orig_h))),
                math.log10(max(1.0, float(orig_w * orig_h))),
            ], dtype=np.float32)
            phys_36 = np.nan_to_num(phys_36, nan=0.0, posinf=10.0, neginf=-10.0)

            # Convert score maps to NumPy for deterministic integral-window crop selection
            s_fg_map_np = (s_ds * fg_mask)[0, 0].cpu().numpy()
            meor_map_np = meor_ds[0, 0].cpu().numpy()
            cut_map_np = cutout_map[0, 0].cpu().numpy()
            # Micro-texture crop targets foreground interior with moderate-low macro gradient (skin/fabric surface, not background sky)
            g_med_fg = float(np.median(fg_grad_vals.cpu().numpy())) if fg_grad_vals.numel() > 0 else 0.02
            tex_map_np = (
                fg_mask[0, 0].cpu().numpy()
                * np.exp(-((grad_ds[0, 0].cpu().numpy() - g_med_fg) / max(0.01, g_med_fg + 0.01)) ** 2)
            )
            anat_map_np = anatomy_sal_ds[0, 0].cpu().numpy()
            text_map_np = text_tex_sal_ds[0, 0].cpu().numpy()

        scale_y = float(orig_h) / float(ds_h)
        scale_x = float(orig_w) / float(ds_w)

        taken_448: List[Tuple[float, float]] = []
        box_focus = _pick_diverse_top_box(s_fg_map_np, 448, orig_h, orig_w, scale_y, scale_x, taken_448, 0.50)
        box_meor = _pick_diverse_top_box(meor_map_np, 448, orig_h, orig_w, scale_y, scale_x, taken_448, 0.50)
        box_cutout = _pick_diverse_top_box(cut_map_np, 448, orig_h, orig_w, scale_y, scale_x, taken_448, 0.50)
        box_tex = _pick_diverse_top_box(tex_map_np, 448, orig_h, orig_w, scale_y, scale_x, taken_448, 0.50)
        crops_448_boxes = np.array([box_focus, box_meor, box_cutout, box_tex], dtype=np.int32)

        taken_384: List[Tuple[float, float]] = []
        roi_anat = _pick_diverse_top_box(anat_map_np, 384, orig_h, orig_w, scale_y, scale_x, taken_384, 0.55)
        roi_text = _pick_diverse_top_box(text_map_np, 384, orig_h, orig_w, scale_y, scale_x, taken_384, 0.55)
        rois_384_boxes = np.array([roi_anat, roi_text], dtype=np.int32)

        def _norm1000(b):
            x0, y0, x1, y1 = b
            return [
                int(round(1000.0 * max(0, x0) / max(1, orig_w))),
                int(round(1000.0 * max(0, y0) / max(1, orig_h))),
                int(round(1000.0 * min(orig_w, x1) / max(1, orig_w))),
                int(round(1000.0 * min(orig_h, y1) / max(1, orig_h))),
            ]

        norm_peak_boxes = np.array([
            _norm1000(box_focus),
            _norm1000(box_meor),
            _norm1000(box_cutout),
            _norm1000(roi_anat),
        ], dtype=np.int32)

        return {
            "orig_hw": np.array([orig_h, orig_w], dtype=np.int32),
            "phys_vec": phys_36,
            "crops_448_boxes": crops_448_boxes,
            "rois_384_boxes": rois_384_boxes,
            "norm_peak_boxes": norm_peak_boxes,
        }

    def prepare_tier3_multi_images(self, pil_im, precomputed: Optional[Dict[str, object]] = None, global_max_pixels: int = 768 * 768):
        """
        Returns the 3-image native list `[img_global_native, roi_anatomy_384, roi_text_tex_384]`
        for `Clef-Flash-9B` and `Qwen3-VL-8B` Interleaved-MRoPE (ZERO collage seams!).
        """
        meta = precomputed if precomputed is not None else self.analyze_pil_image(pil_im)
        img_global = resize_native_aspect_pil(pil_im, max_pixels=global_max_pixels, min_pixels=256 * 256, patch_mult=32)
        roi_boxes = meta["rois_384_boxes"]
        roi_anatomy = extract_1to1_crop_pil(pil_im, roi_boxes[0], crop_size=384)
        roi_text_tex = extract_1to1_crop_pil(pil_im, roi_boxes[1], crop_size=384)
        return [img_global, roi_anatomy, roi_text_tex], meta

    def prepare_tier1_4crops_and_global(self, pil_im, precomputed: Optional[Dict[str, object]] = None, global_size: int = 448):
        """
        Returns `[global_448, crop_focus_448, crop_meor_448, crop_cutout_448, crop_tex_448]` (5 views)
        plus `phys_vec` (`float32[36]`).
        """
        from PIL import Image
        meta = precomputed if precomputed is not None else self.analyze_pil_image(pil_im)
        g_view = pil_im.resize((global_size, global_size), Image.BICUBIC)
        c_boxes = meta["crops_448_boxes"]
        crops = [extract_1to1_crop_pil(pil_im, c_boxes[i], crop_size=448) for i in range(4)]
        if global_size != 448:
            crops = [c.resize((global_size, global_size), Image.BICUBIC) for c in crops]
        return [g_view] + crops, meta


class PrecomputedDualEngineStore:
    """Fast O(1) lookup wrapper over `results/qingyu/dual_engine_meta.npz` with on-the-fly fallback."""
    def __init__(self, npz_path: str = "results/qingyu/dual_engine_meta.npz", lazy_extractor: Optional[DualEnginePhysicsExtractor] = None):
        self.map: Dict[str, Dict[str, object]] = {}
        self.lazy_extractor = lazy_extractor
        full_p = npz_path if os.path.isabs(npz_path) else os.path.join(REPO_ROOT, npz_path)
        if os.path.exists(full_p):
            import numpy as np
            d = np.load(full_p, allow_pickle=True)
            names = [str(x) for x in d["names"]]
            orig_hw = d["orig_hw"]
            phys_vecs = d["phys_vecs"]
            c448 = d["crops_448_boxes"]
            r384 = d["rois_384_boxes"]
            nbox = d["norm_peak_boxes"]
            for i, nm in enumerate(names):
                self.map[nm] = {
                    "orig_hw": orig_hw[i],
                    "phys_vec": phys_vecs[i],
                    "crops_448_boxes": c448[i],
                    "rois_384_boxes": r384[i],
                    "norm_peak_boxes": nbox[i],
                }

    def get(self, name: str, pil_im=None) -> Dict[str, object]:
        s_nm = str(name)
        if s_nm in self.map:
            return self.map[s_nm]
        base = os.path.basename(s_nm)
        if base in self.map:
            return self.map[base]
        if pil_im is not None:
            if self.lazy_extractor is None:
                self.lazy_extractor = DualEnginePhysicsExtractor(device="cuda", allow_depth_download=False)
            meta = self.lazy_extractor.analyze_pil_image(pil_im)
            self.map[s_nm] = meta
            return meta
        raise KeyError(f"Image '{s_nm}' not in precomputed dual-engine cache and no PIL image supplied.")


def run_parallel_precompute(
    manifest_path: str = "data/manifest_combined.csv",
    include_test150: bool = True,
    extra_template: str = "",
    extra_test_dir: str = "",
    out_npz: str = "results/qingyu/dual_engine_meta.npz",
    rank: int = 0,
    world_size: int = 1,
    device: str = "cuda",
) -> str:
    """Precomputes Pillar-2 physics & 1:1 ROI boxes across all real train + test images (supports multi-GPU sharding)."""
    import numpy as np

    setup_sys_path_and_cwd(chdir_repo=True)
    enforce_deterministic_runtime(2026 + rank)

    names_ordered: List[str] = []
    seen = set()
    for r in load_pure_real_manifest(manifest_path):
        nm = str(r["name"])
        if nm not in seen:
            seen.add(nm)
            names_ordered.append(nm)

    if include_test150 and os.path.exists(os.path.join(REPO_ROOT, "results/v2_full/template_150.json")):
        for nm in load_test_template_names("results/v2_full/template_150.json"):
            if nm not in seen:
                seen.add(nm)
                names_ordered.append(nm)

    if extra_template and os.path.exists(extra_template):
        for nm in load_test_template_names(extra_template):
            if nm not in seen:
                seen.add(nm)
                names_ordered.append(nm)

    shard_names = names_ordered[rank::world_size]
    dev_str = f"cuda:{rank}" if (device == "cuda" and world_size > 1) else device
    loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=[extra_test_dir] if extra_test_dir else None)
    extractor = DualEnginePhysicsExtractor(device=dev_str, allow_depth_download=True)

    n_sh = len(shard_names)
    orig_hw = np.zeros((n_sh, 2), dtype=np.int32)
    phys_vecs = np.zeros((n_sh, PHYS_FEAT_DIM), dtype=np.float32)
    crops_448 = np.zeros((n_sh, 4, 4), dtype=np.int32)
    rois_384 = np.zeros((n_sh, 2, 4), dtype=np.int32)
    norm_peaks = np.zeros((n_sh, 4, 4), dtype=np.int32)

    t0 = time.time()
    for i, nm in enumerate(shard_names):
        im = loader.load_pil(nm)
        res = extractor.analyze_pil_image(im)
        orig_hw[i] = res["orig_hw"]
        phys_vecs[i] = res["phys_vec"]
        crops_448[i] = res["crops_448_boxes"]
        rois_384[i] = res["rois_384_boxes"]
        norm_peaks[i] = res["norm_peak_boxes"]
        if (i + 1) % 100 == 0 or (i + 1) == n_sh:
            el = max(1e-3, time.time() - t0)
            print(
                f"[DualEngine shard {rank}/{world_size}] {i+1}/{n_sh} images ({(i+1)/el:.2f} img/s) "
                f"| depth_model_active={extractor.depth_estimator.available}",
                flush=True,
            )

    abs_out = out_npz if os.path.isabs(out_npz) else os.path.join(REPO_ROOT, out_npz)
    os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    save_path = abs_out if world_size == 1 else abs_out.replace(".npz", f".part{rank}of{world_size}.npz")
    np.savez_compressed(
        save_path,
        names=np.array(shard_names),
        orig_hw=orig_hw,
        phys_vecs=phys_vecs,
        crops_448_boxes=crops_448,
        rois_384_boxes=rois_384,
        norm_peak_boxes=norm_peaks,
    )
    print(f"[DualEngine] Saved {n_sh} records -> {save_path}", flush=True)
    return save_path


def merge_precompute_shards(out_npz: str, world_size: int) -> str:
    import numpy as np
    abs_out = out_npz if os.path.isabs(out_npz) else os.path.join(REPO_ROOT, out_npz)
    parts = [np.load(abs_out.replace(".npz", f".part{r}of{world_size}.npz"), allow_pickle=True) for r in range(world_size)]
    np.savez_compressed(
        abs_out,
        names=np.concatenate([p["names"] for p in parts], axis=0),
        orig_hw=np.concatenate([p["orig_hw"] for p in parts], axis=0),
        phys_vecs=np.concatenate([p["phys_vecs"] for p in parts], axis=0),
        crops_448_boxes=np.concatenate([p["crops_448_boxes"] for p in parts], axis=0),
        rois_384_boxes=np.concatenate([p["rois_384_boxes"] for p in parts], axis=0),
        norm_peak_boxes=np.concatenate([p["norm_peak_boxes"] for p in parts], axis=0),
    )
    print(f"[DualEngine] Merged {world_size} shards into {abs_out}", flush=True)
    return abs_out


def main():
    ap = argparse.ArgumentParser(description="Qingyu Pillar-2 Dual-Engine 1:1 Physics & Depth Preprocessor")
    ap.add_argument("--manifest", default="data/manifest_combined.csv")
    ap.add_argument("--include-test150", action="store_true", default=True)
    ap.add_argument("--extra-template", default="")
    ap.add_argument("--extra-test-dir", default="")
    ap.add_argument("--out", default="results/qingyu/dual_engine_meta.npz")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--merge-shards", type=int, default=0, help="If >0, merge N shard files into --out")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.merge_shards > 0:
        merge_precompute_shards(args.out, args.merge_shards)
    else:
        run_parallel_precompute(
            manifest_path=args.manifest,
            include_test150=args.include_test150,
            extra_template=args.extra_template,
            extra_test_dir=args.extra_test_dir,
            out_npz=args.out,
            rank=args.rank,
            world_size=args.world_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
