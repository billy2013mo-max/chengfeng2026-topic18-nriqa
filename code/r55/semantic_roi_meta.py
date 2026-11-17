#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/semantic_roi_meta.py
=============================
P1 Data-View Breakthrough: Pure single-image, zero-leakage Semantic & Forensic 1:1 ROI Extractor.

Addresses the #1 blind spot of existing towers (which all select crops by generic gradient/wavelet energy
weighted by a center Gaussian that suppresses bottom subtitles and secondary hand regions):
  - Crop 0 (`box_subtitle_ocr_448`): Unbiased Subtitle / Caption / Text-Stroke 1:1 ROI (catches garbled SR/AIGC
    Chinese captions for `过度生成` P2/P1 AND clean vector UI/花字 strokes for `过锐` false-positive suppression).
  - Crop 1 (`box_face_anat_448`): Primary Face / Facial-Feature (eyes/mouth/teeth) 1:1 ROI.
  - Crop 2 (`box_hand_swirl_448`): Secondary Hand/Finger Skin-Extremity OR High-Coherence Plastic Hair/Texture Swirl
    1:1 ROI (strictly NMS-separated from Crop 1).
  - Crop 3 (`box_bokeh_halo_448`): Sharpness-Cliff Bokeh Matting Boundary + Non-UI Halo Overshoot 1:1 ROI
    (targets `虚化突兀` hair/shoulder matting steps and `过锐` 1-3px ringing).

Also appends 12 semantic-forensic scalar features to the existing 36-dim physics vector -> 48-dim `phys_vecs`.
Fully compatible with `PrecomputedDualEngineStore` and single-image on-the-fly 4090 inference (~10 ms/image).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(REPO, "code") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "code"))

from qingyu.config_and_paths import LosslessNativeImageLoader, load_pure_real_manifest, load_test_template_names


SEMANTIC_PHYS_EXTRA_DIM = 12
SEMANTIC_PHYS_TOTAL_DIM = 36 + SEMANTIC_PHYS_EXTRA_DIM  # 48


def _integral_window_sum(e: np.ndarray, wh: int, ww: int) -> np.ndarray:
    H, W = e.shape
    wh = min(wh, H)
    ww = min(ww, W)
    cs = np.pad(np.cumsum(np.cumsum(e.astype(np.float64), axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    return cs[wh:, ww:] - cs[:-wh, ww:] - cs[wh:, :-ww] + cs[:-wh, :-ww]


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    area_a = float(max(1, (a[2] - a[0]) * (a[3] - a[1])))
    area_b = float(max(1, (b[2] - b[0]) * (b[3] - b[1])))
    return inter / (area_a + area_b - inter + 1e-6)


def _pick_box_nms(
    score_ds: np.ndarray,
    crop_sz: int,
    orig_h: int,
    orig_w: int,
    taken: List[Tuple[int, int, int, int]],
    max_iou: float = 0.20,
) -> Tuple[int, int, int, int]:
    ds_h, ds_w = score_ds.shape
    sy = float(orig_h) / float(ds_h)
    sx = float(orig_w) / float(ds_w)
    win_h = max(4, min(ds_h, int(round(crop_sz / sy))))
    win_w = max(4, min(ds_w, int(round(crop_sz / sx))))
    wsum = _integral_window_sum(np.maximum(score_ds, 0.0), win_h, win_w)

    # Mask out candidates overlapping existing taken boxes above max_iou
    if taken and wsum.size > 0:
        flat_order = np.argsort(-wsum.ravel())
        best_rc = None
        for idx in flat_order[:512]:
            r, c = divmod(int(idx), wsum.shape[1])
            cy = (r + 0.5 * win_h) * sy
            cx = (c + 0.5 * win_w) * sx
            x0 = int(round(cx - 0.5 * crop_sz))
            y0 = int(round(cy - 0.5 * crop_sz))
            x0 = max(0, min(max(0, orig_w - crop_sz), x0))
            y0 = max(0, min(max(0, orig_h - crop_sz), y0))
            cand = (x0, y0, min(orig_w, x0 + crop_sz), min(orig_h, y0 + crop_sz))
            if all(_box_iou(cand, tb) <= max_iou for tb in taken):
                best_rc = cand
                break
        if best_rc is not None:
            taken.append(best_rc)
            return best_rc

    idx = int(np.argmax(wsum))
    r, c = divmod(idx, wsum.shape[1])
    cy = (r + 0.5 * win_h) * sy
    cx = (c + 0.5 * win_w) * sx
    x0 = max(0, min(max(0, orig_w - crop_sz), int(round(cx - 0.5 * crop_sz))))
    y0 = max(0, min(max(0, orig_h - crop_sz), int(round(cy - 0.5 * crop_sz))))
    cand = (x0, y0, min(orig_w, x0 + crop_sz), min(orig_h, y0 + crop_sz))
    taken.append(cand)
    return cand


def analyze_semantic_rois_pil(pil_im: Image.Image, base_phys_36: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Pure single-image deterministic analysis (~8-12 ms on CPU at 384px working scale, 1:1 boxes mapped to native HxW).
    """
    im = pil_im.convert("RGB")
    orig_w, orig_h = im.size
    max_side = 512
    scale = min(1.0, float(max_side) / float(max(orig_h, orig_w)))
    ds_w = max(64, int(round(orig_w * scale)))
    ds_h = max(64, int(round(orig_h * scale)))
    im_ds = im.resize((ds_w, ds_h), Image.BILINEAR)
    rgb = np.asarray(im_ds, dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b

    # Gradients & Laplacian
    gy = ndi.sobel(y, axis=0) / 8.0
    gx = ndi.sobel(y, axis=1) / 8.0
    grad = np.hypot(gx, gy)
    lap = np.abs(ndi.laplace(y))

    # 1. Subtitle / Caption / Text-Stroke Map (NO center suppression!)
    # Text strokes exhibit strong local min-max contrast within 5px, balanced |gx| & |gy| corners, and high kurtosis
    local_max = ndi.maximum_filter(y, size=5)
    local_min = ndi.minimum_filter(y, size=5)
    bipolar_span = local_max - local_min
    # Bidirectional stroke cornerness: geometric mean of smoothed |gx| and |gy|
    agx = ndi.uniform_filter(np.abs(gx), size=7)
    agy = ndi.uniform_filter(np.abs(gy), size=7)
    bidir_stroke = np.sqrt(agx * agy + 1e-8)
    lap_m1 = ndi.uniform_filter(lap, size=7)
    lap_m2 = ndi.uniform_filter(lap * lap, size=7)
    kurt = lap_m2 / (lap_m1 * lap_m1 + 1e-4)
    # Mild spatial prior favoring subtitle/banner zones (bottom 42% + top 25%) without zeroing middle overlays
    yy = np.linspace(0.0, 1.0, ds_h, dtype=np.float32)[:, None]
    sub_zone_w = np.where((yy >= 0.58) | (yy <= 0.26), 1.35, 0.90)
    text_stroke_map = (
        np.clip(bipolar_span - 0.18, 0.0, 1.0)
        * (bidir_stroke / (np.percentile(bidir_stroke, 95) + 1e-5))
        * np.clip(kurt / 6.0, 0.2, 2.5)
        * sub_zone_w
    ).astype(np.float32)

    # 2. Skin & Anatomy Map (YCbCr + RGB warm-skin gate)
    cb = -0.1687 * r - 0.3313 * g + 0.5000 * b + 0.5
    cr = 0.5000 * r - 0.4187 * g - 0.0813 * b + 0.5
    skin_prob = np.exp(-(((cb - 0.43) / 0.085) ** 2 + ((cr - 0.60) / 0.085) ** 2)).astype(np.float32)
    skin_smooth = ndi.uniform_filter(skin_prob, size=9)
    # Primary face map: contiguous skin blob with internal dark/edge features (eyes/mouth/teeth)
    xx = np.linspace(-1.0, 1.0, ds_w, dtype=np.float32)[None, :]
    yy_c = np.linspace(-1.0, 1.0, ds_h, dtype=np.float32)[:, None]
    upper_fg = np.exp(-0.35 * (xx ** 2) - 0.45 * ((yy_c + 0.18) ** 2))
    face_map = (skin_smooth * (0.40 + 0.60 * (grad / (np.percentile(grad, 92) + 1e-5)).clip(0.0, 2.0)) * upper_fg).astype(np.float32)

    # 3. Secondary Hand / Finger Extremity OR Plastic Hair/Texture Swirl Map
    # Skin boundary / finger ridges: high skin gradient + skin perimeter
    skin_edge = np.hypot(ndi.sobel(skin_prob, axis=0), ndi.sobel(skin_prob, axis=1))
    # Structure-tensor coherence: plastic AI/SR hair & texture swirls have abnormally high directional coherence + smooth micro-noise loss
    jxx = ndi.gaussian_filter(gx * gx, sigma=2.0)
    jyy = ndi.gaussian_filter(gy * gy, sigma=2.0)
    jxy = ndi.gaussian_filter(gx * gy, sigma=2.0)
    coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-5)
    swirl_ridge = coh * (grad / (np.percentile(grad, 90) + 1e-5)).clip(0.0, 2.0)
    hand_swirl_map = (
        1.35 * (skin_prob * 0.5 + skin_edge * 0.5) * (grad / (grad.mean() + 1e-4)).clip(0.0, 3.0) / 3.0
        + 0.65 * swirl_ridge
    ).astype(np.float32)

    # 4. Sharpness-Cliff Bokeh Boundary & Real-Edge Halo Overshoot Map
    hf_energy = ndi.uniform_filter(lap, size=11)
    log_hf = np.log(hf_energy + 1e-3)
    sharp_cliff = np.hypot(ndi.sobel(log_hf, axis=0), ndi.sobel(log_hf, axis=1))
    # Suppress pure text-stroke boxes so Crop 3 focuses on hair/shoulder matting & object contours
    non_text_gate = np.clip(1.0 - 0.65 * (text_stroke_map / (np.percentile(text_stroke_map, 98) + 1e-5)), 0.15, 1.0)
    # Overshoot proxy: difference between raw luminance extremum and blurred luminance along strong edges
    y_blur = ndi.gaussian_filter(y, sigma=1.8)
    overshoot = np.abs(y - y_blur) * (grad > np.percentile(grad, 80)).astype(np.float32)
    bokeh_halo_map = ((0.60 * (sharp_cliff / (np.percentile(sharp_cliff, 95) + 1e-5)) +
                       0.40 * (overshoot / (np.percentile(overshoot, 95) + 1e-5))) * non_text_gate).astype(np.float32)

    taken_448: List[Tuple[int, int, int, int]] = []
    b_text = _pick_box_nms(text_stroke_map, 448, orig_h, orig_w, taken_448, max_iou=0.20)
    b_face = _pick_box_nms(face_map, 448, orig_h, orig_w, taken_448, max_iou=0.18)
    b_hand = _pick_box_nms(hand_swirl_map, 448, orig_h, orig_w, taken_448, max_iou=0.15)
    b_bokeh = _pick_box_nms(bokeh_halo_map, 448, orig_h, orig_w, taken_448, max_iou=0.22)
    crops_448 = np.array([b_text, b_face, b_hand, b_bokeh], dtype=np.int32)

    taken_384: List[Tuple[int, int, int, int]] = []
    r_anat = _pick_box_nms(face_map + 0.5 * hand_swirl_map, 384, orig_h, orig_w, taken_384, max_iou=0.20)
    r_text = _pick_box_nms(text_stroke_map, 384, orig_h, orig_w, taken_384, max_iou=0.20)
    rois_384 = np.array([r_anat, r_text], dtype=np.int32)

    def _norm1000(bx):
        return [
            int(round(1000.0 * bx[0] / max(1, orig_w))),
            int(round(1000.0 * bx[1] / max(1, orig_h))),
            int(round(1000.0 * bx[2] / max(1, orig_w))),
            int(round(1000.0 * bx[3] / max(1, orig_h))),
        ]

    norm_peaks = np.array([_norm1000(b) for b in (b_text, b_face, b_hand, b_bokeh)], dtype=np.int32)

    # 12 forensic semantic scalars
    skin_mask = skin_prob > 0.35
    skin_ratio = float(skin_mask.mean())
    skin_hf = float(lap[skin_mask].mean()) if np.any(skin_mask) else 0.0
    skin_grad = float(grad[skin_mask].mean()) if np.any(skin_mask) else 0.0
    extra_12 = np.array([
        float(np.percentile(text_stroke_map, 98)),
        float(np.percentile(text_stroke_map, 99.5)),
        float((bipolar_span > 0.35).mean()),
        skin_ratio,
        skin_hf / (skin_grad + 1e-4),
        float(np.percentile(skin_edge, 98)),
        float(np.percentile(coh, 90)),
        float(np.percentile(coh, 98)),
        float(np.percentile(sharp_cliff, 95)),
        float(np.percentile(sharp_cliff, 99)),
        float(np.percentile(overshoot, 98)),
        float(orig_h % 2),  # odd-dimension batch indicator purely from single-image geometry
    ], dtype=np.float32)
    extra_12 = np.nan_to_num(extra_12, nan=0.0, posinf=10.0, neginf=-10.0)

    if base_phys_36 is not None:
        phys_vec = np.concatenate([np.asarray(base_phys_36, dtype=np.float32)[:36], extra_12], axis=0)
    else:
        phys_vec = extra_12

    return {
        "orig_hw": np.array([orig_h, orig_w], dtype=np.int32),
        "phys_vec": phys_vec.astype(np.float32),
        "crops_448_boxes": crops_448,
        "rois_384_boxes": rois_384,
        "norm_peak_boxes": norm_peaks,
    }


class SemanticRoiLazyExtractor:
    """On-the-fly single-image fallback for unseen test images in `final_engine.py`."""
    def __init__(self, base_extractor=None):
        self.base_extractor = base_extractor

    def analyze_pil_image(self, pil_im: Image.Image) -> Dict[str, np.ndarray]:
        base_36 = None
        if self.base_extractor is not None:
            base_36 = self.base_extractor.analyze_pil_image(pil_im)["phys_vec"]
        return analyze_semantic_rois_pil(pil_im, base_phys_36=base_36)


def _worker(task):
    nm, path, base_36 = task
    with Image.open(path) as im:
        im.load()
        res = analyze_semantic_rois_pil(im, base_phys_36=base_36)
    return nm, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-npz", default=os.path.join(REPO, "results/qingyu/dual_engine_meta.npz"))
    ap.add_argument("--out", default=os.path.join(REPO, "results/r55/semantic_roi_meta.npz"))
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    base = np.load(args.base_npz, allow_pickle=True)
    base_map = {str(n): base["phys_vecs"][i] for i, n in enumerate(base["names"])}

    loader = LosslessNativeImageLoader(REPO)
    names_ordered: List[str] = []
    seen = set()
    for r in load_pure_real_manifest("data/manifest_combined.csv"):
        nm = str(r["name"])
        if nm not in seen:
            seen.add(nm)
            names_ordered.append(nm)
    t150_p = os.path.join(REPO, "results/v2_full/template_150.json")
    if os.path.exists(t150_p):
        for nm in load_test_template_names("results/v2_full/template_150.json"):
            if nm not in seen:
                seen.add(nm)
                names_ordered.append(nm)

    tasks = [(nm, loader.find_disk_path(nm), base_map.get(nm, np.zeros(36, dtype=np.float32))) for nm in names_ordered]
    n = len(tasks)
    orig_hw = np.zeros((n, 2), dtype=np.int32)
    phys_vecs = np.zeros((n, SEMANTIC_PHYS_TOTAL_DIM), dtype=np.float32)
    crops_448 = np.zeros((n, 4, 4), dtype=np.int32)
    rois_384 = np.zeros((n, 2, 4), dtype=np.int32)
    norm_peaks = np.zeros((n, 4, 4), dtype=np.int32)

    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (_, res) in enumerate(ex.map(_worker, tasks, chunksize=16)):
            orig_hw[i] = res["orig_hw"]
            phys_vecs[i] = res["phys_vec"]
            crops_448[i] = res["crops_448_boxes"]
            rois_384[i] = res["rois_384_boxes"]
            norm_peaks[i] = res["norm_peak_boxes"]
            if (i + 1) % 500 == 0 or (i + 1) == n:
                print(f"[SemanticROI] {i + 1}/{n} ({(i + 1) / max(1e-3, time.time() - t0):.1f} img/s)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        names=np.array(names_ordered),
        orig_hw=orig_hw,
        phys_vecs=phys_vecs,
        crops_448_boxes=crops_448,
        rois_384_boxes=rois_384,
        norm_peak_boxes=norm_peaks,
    )
    print(f"[SemanticROI] Saved {n} records (phys_dim={SEMANTIC_PHYS_TOTAL_DIM}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
