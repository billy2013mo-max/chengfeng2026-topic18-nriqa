#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/measured_evidence.py — measured image-physics evidence for the per-image semantic attribution.

The decoded grades (`pred_6d`) are NEVER changed here.  For every dimension the decoder flagged, this module
measures one or two classical image-physics quantities on native pixels and compares them with the same quantity
on CLEAN TRAINING frames (percentiles precomputed offline into `<set>_clean_footprint.npz`):

  dim        metric (direction)                                   what it measures
  模糊       lapvar   (low)   variance of the luma Laplacian        loss of high-frequency edge energy
  纹理损失   tex_hf   (low)   median |Laplacian| on mid-gradient    micro-texture energy in non-edge textured areas
  过锐       overshoot(high)  |y - G_1.5(y)| / local range at edges ringing / halo amplitude relative to edge contrast
  花屏       blocky   (high)  8-px grid-boundary step / interior     macro-block discontinuity
             chroma   (high)  p99 chroma gradient / p99 luma grad    chroma-channel overflow / colour tearing
  过度生成   coh      (high)  p90 structure-tensor coherence         over-regular, "painted" texture strokes
             hf_ratio (high)  box mean|Lap| / global mean|Lap|       local HF anomaly vs the rest of the frame
  虚化突兀   cliff    (high)  p99 |grad log HF-energy|               abrupt sharp/blur matte boundary

All quantities are computed on 448x448 native-resolution tiles (no resampling, so 1-3 px ringing and the 8-px
block grid survive).  "global" = median over a fixed 3x3 tile grid; "box" = the 448 box from the attribution
map when the map is spatially concentrated.  A value is called *supporting* only if it lies beyond the clean
P90 (P10 for "low" metrics); otherwise the attribution says explicitly that the measured physics does NOT
confirm the cause (the grade then rests on the learned model only).  No causal claim is fabricated.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

TILE = 448
DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
GRADE_STR = {0: "无问题", 1: "P2", 2: "P1", 3: "P0"}

# metric key -> (Chinese short name, direction)
METRICS: Dict[str, Tuple[str, str]] = {
    "lapvar": ("拉普拉斯方差", "low"),
    "tex_hf": ("纹理高频能量", "low"),
    "overshoot": ("边缘过冲比", "high"),
    "blocky": ("8px块边界跳变比", "high"),
    "chroma": ("色度跳变比", "high"),
    "coh": ("结构相干度", "high"),
    "hf_ratio": ("局部/全局高频比", "high"),
    "cliff": ("清晰度断崖强度", "high"),
}
DIM_METRICS: Dict[str, List[str]] = {
    "模糊": ["lapvar"],
    "纹理损失": ["tex_hf"],
    "过锐": ["overshoot"],
    "花屏": ["blocky", "chroma"],
    "过度生成": ["coh", "hf_ratio"],
    "虚化突兀": ["cliff"],
}
SUPPORT_PCT = 90.0  # beyond clean P90 (or below P10 for "low") -> measured support


def _tile_boxes(w: int, h: int, n: int = 3) -> List[Tuple[int, int, int, int]]:
    t = min(TILE, w, h)
    xs = np.linspace(0, max(0, w - t), n).round().astype(int)
    ys = np.linspace(0, max(0, h - t), n).round().astype(int)
    return [(int(x), int(y), int(x) + t, int(y) + t) for y in ys for x in xs]


def _need(keys, *ks) -> bool:
    return keys is None or any(k in keys for k in ks)


def tile_metrics(rgb01: np.ndarray, global_lap_mean: Optional[float] = None, keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """rgb01: (H, W, 3) float32 in [0, 1], one native tile.  `keys` restricts the work to the metrics needed."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    y = np.ascontiguousarray(0.299 * r + 0.587 * g + 0.114 * b, dtype=np.float32)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = cv2.magnitude(gx, gy)
    lap = cv2.Laplacian(y, cv2.CV_32F, ksize=1)
    alap = np.abs(lap)
    g30, g80, g95, g99 = np.percentile(grad, [30, 80, 95, 99])

    out: Dict[str, float] = {"lap_mean": float(alap.mean())}
    if _need(keys, "lapvar"):
        out["lapvar"] = float(lap.var())
    if _need(keys, "tex_hf"):
        mid = (grad >= g30) & (grad <= g80)
        out["tex_hf"] = float(np.median(alap[mid])) if mid.any() else float(np.median(alap))
    if _need(keys, "overshoot"):
        strong = (grad >= max(float(g95), 0.03)).astype(np.uint8)
        if int(strong.sum()) >= 32:
            k7 = np.ones((7, 7), np.uint8)
            rng = cv2.dilate(y, k7) - cv2.erode(y, k7)
            ring = np.abs(y - cv2.GaussianBlur(y, (0, 0), 1.5)) / (rng + 0.02)
            out["overshoot"] = float(ring[cv2.dilate(strong, np.ones((3, 3), np.uint8)) > 0].mean())
        else:
            out["overshoot"] = 0.0
    if _need(keys, "blocky"):
        dh = np.abs(np.diff(y, axis=1))
        dv = np.abs(np.diff(y, axis=0))
        bh = float(dh[:, 7::8].mean()) / (float(dh.mean()) + 1e-4) if dh.shape[1] >= 16 else 1.0
        bv = float(dv[7::8, :].mean()) / (float(dv.mean()) + 1e-4) if dv.shape[0] >= 16 else 1.0
        out["blocky"] = 0.5 * (bh + bv)
    if _need(keys, "chroma"):
        cb = np.ascontiguousarray(-0.1687 * r - 0.3313 * g + 0.5 * b, dtype=np.float32)
        cr = np.ascontiguousarray(0.5 * r - 0.4187 * g - 0.0813 * b, dtype=np.float32)
        cg = (cv2.magnitude(cv2.Sobel(cb, cv2.CV_32F, 1, 0), cv2.Sobel(cb, cv2.CV_32F, 0, 1))
              + cv2.magnitude(cv2.Sobel(cr, cv2.CV_32F, 1, 0), cv2.Sobel(cr, cv2.CV_32F, 0, 1))) / 8.0
        out["chroma"] = float(np.percentile(cg, 99)) / (float(g99) + 1e-3)
    if _need(keys, "coh"):
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
        coh = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-5)
        textured = grad >= g30
        out["coh"] = float(np.percentile(coh[textured], 90)) if textured.any() else 0.0
    if _need(keys, "hf_ratio"):
        out["hf_ratio"] = out["lap_mean"] / (global_lap_mean + 1e-5) if global_lap_mean is not None else 1.0
    if _need(keys, "cliff"):
        log_hf = np.log(cv2.blur(alap, (11, 11)) + 1e-3)
        cliff = cv2.magnitude(cv2.Sobel(log_hf, cv2.CV_32F, 1, 0), cv2.Sobel(log_hf, cv2.CV_32F, 0, 1)) / 8.0
        out["cliff"] = float(np.percentile(cliff, 99))
    return out


def image_metrics(pil_rgb, box: Optional[Sequence[int]] = None, keys: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """-> {"global": median over the 3x3 native tile grid, "tiles": per tile, "box": metrics of the box (if given)}."""
    arr = np.asarray(pil_rgb.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    tiles = [tile_metrics(arr[y0:y1, x0:x1].astype(np.float32) / 255.0, keys=keys) for (x0, y0, x1, y1) in _tile_boxes(w, h)]
    glob = {k: float(np.median([t[k] for t in tiles])) for k in tiles[0]}
    glob["hf_ratio"] = 1.0
    res = {"global": glob, "tiles": tiles}
    if box is not None:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x0 + 16, x1)), min(h, max(y0 + 16, y1))
        res["box"] = tile_metrics(arr[y0:y1, x0:x1].astype(np.float32) / 255.0, global_lap_mean=glob["lap_mean"], keys=keys)
    return res


# ---------------------------------------------------------------- clean reference (offline, training frames only)
def build_clean_physics_reference(pils: Sequence) -> Dict[str, np.ndarray]:
    """Per metric: sorted per-tile values (reference for box values) and sorted per-image global values."""
    tile_vals: Dict[str, List[float]] = {k: [] for k in METRICS}
    glob_vals: Dict[str, List[float]] = {k: [] for k in METRICS}
    for im in pils:
        m = image_metrics(im)
        g_lap = m["global"]["lap_mean"]
        for t in m["tiles"]:
            t = dict(t, hf_ratio=t["lap_mean"] / (g_lap + 1e-5))
            for k in METRICS:
                tile_vals[k].append(t[k])
        for k in METRICS:
            glob_vals[k].append(m["global"][k])
    out = {}
    for k in METRICS:
        out[f"phys_tile__{k}"] = np.sort(np.asarray(tile_vals[k], dtype=np.float32))
        out[f"phys_glob__{k}"] = np.sort(np.asarray(glob_vals[k], dtype=np.float32))
    return out


def clean_rank(ref_sorted: np.ndarray, v: float) -> float:
    """Percentile rank (0..100) of v within the clean reference."""
    n = len(ref_sorted)
    if n == 0:
        return float("nan")
    lo = np.searchsorted(ref_sorted, v, side="left")
    hi = np.searchsorted(ref_sorted, v, side="right")
    return float(100.0 * 0.5 * (lo + hi) / n)


def _fmt(v: float) -> str:
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"


def measure_defect(dim: str, phys: Dict[str, Dict[str, float]], ref: Dict[str, np.ndarray], use_box: bool) -> Dict[str, object]:
    """Measured evidence for one flagged dimension."""
    scope = "box" if (use_box and "box" in phys) else "global"
    items = []
    for k in DIM_METRICS[dim]:
        if k == "hf_ratio" and scope != "box":
            continue                                   # box-vs-global ratio is only defined for a box
        cn, direction = METRICS[k]
        v = float(phys[scope][k])
        ref_key = f"phys_tile__{k}" if scope == "box" else f"phys_glob__{k}"
        rk = clean_rank(ref[ref_key], v) if ref_key in ref else float("nan")
        supp = bool(np.isfinite(rk) and (rk >= SUPPORT_PCT if direction == "high" else rk <= 100.0 - SUPPORT_PCT))
        contra = bool(np.isfinite(rk) and (rk <= 100.0 - SUPPORT_PCT if direction == "high" else rk >= SUPPORT_PCT))
        verdict = "supports" if supp else ("contradicts" if contra else "inconclusive")
        it = {"metric": k, "name": cn, "direction": direction, "value": round(v, 5), "clean_pct": round(rk, 1), "supports": supp,
              "verdict": verdict}
        if scope == "box":
            it["global_value"] = round(float(phys["global"][k]), 5)
        items.append(it)
    supported = [x for x in items if x["supports"]]
    if supported:
        verdict = "supports"
    elif items and all(x["verdict"] == "contradicts" for x in items):
        verdict = "contradicts"
    else:
        verdict = "inconclusive" if items else "not_measured"
    return {"scope": scope, "measurements": items, "measured_support": bool(supported), "verdict": verdict}


def _pct_tag(it: Dict[str, object]) -> str:
    q = float(it["clean_pct"])
    if q > 99.0:
        return "洁净P>99"
    if q < 1.0:
        return "洁净P<1"
    return f"洁净P{int(round(q))}"


def compose_reason(loc: str, sev: str, dim: str, ev: Dict[str, object], limit: int = 40) -> str:
    """<= `limit` characters by construction (shorter variants are chosen, never a blind slice)."""
    its = ev["measurements"]
    best = None
    if its:
        cand = [x for x in its if x["supports"]] or its
        best = max(cand, key=lambda x: abs(float(x["clean_pct"]) - 50.0))
    loc_s = loc.replace("画面", "") if loc != "全图" else "全图"
    variants = []
    if best is not None and ev["measured_support"]:
        variants += [
            f"{loc}{sev}级{dim}：实测{best['name']}{_fmt(best['value'])}（{_pct_tag(best)}）",
            f"{loc_s}{sev}{dim}：{best['name']}{_fmt(best['value'])}（{_pct_tag(best)}）",
            f"{loc_s}{sev}{dim}：{best['name']}达{_pct_tag(best)}",
        ]
    else:
        word = "反向" if ev.get("verdict") == "contradicts" else "未见异常"
        tag = f"{best['name']}{_pct_tag(best)}{word}" if best is not None else "未测得物理量"
        variants += [
            f"{loc}{sev}级{dim}：{tag}，物理证据不支持，待复核",
            f"{loc_s}{sev}{dim}：{tag}，待复核",
            f"{loc_s}{sev}{dim}：物理证据不支持，待复核",
        ]
    for v in variants:
        if len(v) <= limit:
            return v
    return f"{sev}{dim}待复核"
