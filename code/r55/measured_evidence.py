#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/measured_evidence.py — measured native-pixel & scene-normalized physics evidence for semantic attribution.

The decoded grades (`pred_6d`) are NEVER changed here. For every dimension flagged by the decoder, this module
measures both classical native-pixel quantities and scene-normalized dimensionless ratios on 448x448 native tiles,
comparing them against CLEAN TRAINING frames (`<set>_clean_footprint.npz`, precomputed offline so runtime reads
zero training files):

  dim        metrics (direction)                                       physical phenomenon measured
  模糊       acutance (low), lapvar (low)                              edge high-frequency roll-off & luma Laplacian energy
  纹理损失   tex_norm (low), tex_hf (low)                              contrast-normalized mid-tone micro-texture loss
  过锐       overshoot_p90 (high), overshoot (high)                    1-3 px strong-edge ringing / white halo overshoot
  花屏       blocky (high), chroma (high)                              8-px macro-block step & chroma channel tearing
  过度生成   coh (high), hf_ratio (high)                               over-regular synthetic strokes & local HF anomaly
  虚化突兀   cliff (high), dof_bimod (high)                            sharp/blur matte boundary cliff & bimodal DOF split

All `reason_40ch` strings are <= 40 Chinese characters by construction (zero blind slicing), 100% consistent with
the predicted `label` (Official Appendix 2 Criterion 1), grounded in measured pixel statistics when beyond Clean
P10/P90 or observable multi-tower visual phenomena when semantic, and never fabricate upstream pipeline causes.
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
    "acutance": ("边缘锐度比", "low"),
    "tex_hf": ("纹理高频能量", "low"),
    "tex_norm": ("归一化微纹理比", "low"),
    "overshoot": ("边缘过冲均值", "high"),
    "overshoot_p90": ("强边缘过冲比", "high"),
    "blocky": ("8px块边界跳变", "high"),
    "chroma": ("色度撕裂比", "high"),
    "coh": ("结构相干度", "high"),
    "hf_ratio": ("局部/全局高频比", "high"),
    "cliff": ("清晰度断崖", "high"),
    "dof_bimod": ("虚实跨度比", "high"),
}
DIM_METRICS: Dict[str, List[str]] = {
    "模糊": ["acutance", "lapvar"],
    "纹理损失": ["tex_norm", "tex_hf"],
    "过锐": ["overshoot_p90", "overshoot"],
    "花屏": ["blocky", "chroma"],
    "过度生成": ["coh", "hf_ratio"],
    "虚化突兀": ["cliff", "dof_bimod"],
}
VISUAL_PHENOMENON_CUE: Dict[str, Dict[str, str]] = {
    "模糊": {
        "P0": "主体轮廓与高频边缘显著失焦发虚",
        "P1": "轮廓边缘中度发虚、细节清晰度下降",
        "P2": "局部或边缘清晰度轻度衰减",
    },
    "纹理损失": {
        "P0": "表面微纹理严重涂抹、质感大面积丢失",
        "P1": "中高频表面纹理中度平滑涂抹",
        "P2": "暗部或皮肤等微纹理轻度平滑",
    },
    "过锐": {
        "P0": "高反差边缘出现明显生硬亮白光晕",
        "P1": "边缘锐化过度、可见轻微白边振铃",
        "P2": "局部轮廓边缘存在轻度锐化强化感",
    },
    "花屏": {
        "P0": "出现明显解码色块错乱或像素撕裂",
    },
    "过度生成": {
        "P0": "多塔聚焦局部结构畸变或违背常理伪影",
        "P1": "局部细节存在AI生成伪纹理或结构异常",
        "P2": "局部微小纹理或边缘呈轻度合成感",
    },
    "虚化突兀": {
        "P0": "主体与背景虚实过渡生硬、边缘割裂",
    },
}
SUPPORT_PCT = 90.0  # P0 strict threshold (P90/P10); P1/P2 use tiered Clean percentiles + box-vs-global relative contrast
TIERED_SUPPORT_PCT: Dict[str, float] = {"P0": 90.0, "P1": 82.0, "P2": 75.0}


def describe_box_visual_cue(rgb01: np.ndarray) -> str:
    """Deterministic, zero-hallucination visual region descriptor from native box pixels (<= 5 Chinese chars)."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
    # Standard YCrCb skin-chroma band (Cb in [77,127]/255, Cr in [133,173]/255)
    skin_frac = float(((cb >= 0.30) & (cb <= 0.50) & (cr >= 0.52) & (cr <= 0.68) & (y >= 0.20)).mean())
    mean_y = float(y.mean())
    gx = cv2.Sobel(y.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(y.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = cv2.magnitude(gx, gy)
    g_mean = float(grad.mean())
    g90 = float(np.percentile(grad, 90))
    if skin_frac >= 0.22:
        return "人像肤色区"
    if g90 >= 0.13 and float(y.std()) >= 0.18:
        return "高反差边缘"
    if mean_y < 0.27:
        return "暗部区域"
    if mean_y > 0.74:
        return "高亮区域"
    if g_mean < 0.026:
        return "平滑背景区"
    return "主体细节区"


def _tile_boxes(w: int, h: int, n: int = 3) -> List[Tuple[int, int, int, int]]:
    t = min(TILE, w, h)
    xs = np.linspace(0, max(0, w - t), n).round().astype(int)
    ys = np.linspace(0, max(0, h - t), n).round().astype(int)
    return [(int(x), int(y), int(x) + t, int(y) + t) for y in ys for x in xs]


def _need(keys, *ks) -> bool:
    return keys is None or any(k in keys for k in ks)


def tile_metrics(
    rgb01: np.ndarray,
    global_lap_mean: Optional[float] = None,
    global_dof_bimod: Optional[float] = None,
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """rgb01: (H, W, 3) float32 in [0, 1], one native tile."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    y = np.ascontiguousarray(0.299 * r + 0.587 * g + 0.114 * b, dtype=np.float32)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = cv2.magnitude(gx, gy)
    lap = cv2.Laplacian(y, cv2.CV_32F, ksize=1)
    alap = np.abs(lap)
    g20, g30, g80, g95, g99 = np.percentile(grad, [20, 30, 80, 95, 99])

    lap_mean = float(alap.mean())
    grad_mean = float(grad.mean())
    out: Dict[str, float] = {"lap_mean": lap_mean, "acutance_raw": lap_mean / (grad_mean + 1e-3)}

    if _need(keys, "lapvar"):
        out["lapvar"] = float(lap.var())
    if _need(keys, "acutance"):
        out["acutance"] = out["acutance_raw"]
    if _need(keys, "tex_hf", "tex_norm"):
        mid = (grad >= g30) & (grad <= g80)
        thf = float(np.median(alap[mid])) if mid.any() else float(np.median(alap))
        out["tex_hf"] = thf
        out["tex_norm"] = thf / (float(g80 - g20) + 0.015)
    if _need(keys, "overshoot", "overshoot_p90"):
        strong = (grad >= max(float(g95), 0.03)).astype(np.uint8)
        if int(strong.sum()) >= 32:
            k7 = np.ones((7, 7), np.uint8)
            rng = cv2.dilate(y, k7) - cv2.erode(y, k7)
            ring = np.abs(y - cv2.GaussianBlur(y, (0, 0), 1.5)) / (rng + 0.02)
            vals = ring[cv2.dilate(strong, np.ones((3, 3), np.uint8)) > 0]
            out["overshoot"] = float(vals.mean())
            out["overshoot_p90"] = float(np.percentile(vals, 90))
        else:
            out["overshoot"] = 0.0
            out["overshoot_p90"] = 0.0
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
        out["hf_ratio"] = lap_mean / (global_lap_mean + 1e-5) if global_lap_mean is not None else 1.0
    if _need(keys, "cliff", "dof_bimod"):
        log_hf = np.log(cv2.blur(alap, (11, 11)) + 1e-3)
        cliff = cv2.magnitude(cv2.Sobel(log_hf, cv2.CV_32F, 1, 0), cv2.Sobel(log_hf, cv2.CV_32F, 0, 1)) / 8.0
        out["cliff"] = float(np.percentile(cliff, 99))
        # Within-tile bimodal sharpness ratio (P90 vs P15 of smoothed local HF energy)
        hf_s = cv2.blur(alap, (15, 15))
        p15, p90 = np.percentile(hf_s, [15, 90])
        local_bimod = float((p90 + 1e-4) / (p15 + 1e-3))
        out["dof_bimod"] = max(local_bimod, float(global_dof_bimod or 0.0))
    return out


def image_metrics(pil_rgb, box: Optional[Sequence[int]] = None, keys: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    """-> {"global": median over active 3x3 native tiles (+ cross-tile dof_bimod), "tiles": per tile, "box": optional box}."""
    arr = np.asarray(pil_rgb.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    tiles = []
    tile_stds = []
    for (x0, y0, x1, y1) in _tile_boxes(w, h):
        patch = arr[y0:y1, x0:x1].astype(np.float32) / 255.0
        tile_stds.append(float(patch.std()))
        tiles.append(tile_metrics(patch, keys=keys))
    # Exclude black letterbox / flat border bars so global_value never collapses to 0.0
    active_idx = [i for i, s in enumerate(tile_stds) if s >= 0.015]
    if len(active_idx) < 3:
        active_idx = list(np.argsort(tile_stds)[-min(5, len(tiles)):])
    active_tiles = [tiles[i] for i in active_idx]
    tile_laps = [t["lap_mean"] for t in active_tiles]
    cross_dof_bimod = float((np.percentile(tile_laps, 85) + 1e-4) / (np.percentile(tile_laps, 15) + 1e-3))
    glob = {k: float(np.median([t[k] for t in active_tiles])) for k in active_tiles[0]}
    # Ensure non-zero denominator fallback if active tiles had near-zero median on a specific metric
    for k in list(glob.keys()):
        if glob[k] < 1e-5:
            nz = [t[k] for t in active_tiles if t[k] >= 1e-5]
            if not nz:
                nz = [t[k] for t in tiles if t[k] >= 1e-5]
            if nz:
                glob[k] = float(np.median(nz))
    glob["hf_ratio"] = 1.0
    if "dof_bimod" in glob:
        glob["dof_bimod"] = max(glob["dof_bimod"], cross_dof_bimod)
    res = {"global": glob, "tiles": tiles, "cross_dof_bimod": cross_dof_bimod}
    if box is not None:
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x0 + 16, x1)), min(h, max(y0 + 16, y1))
        res["box"] = tile_metrics(
            arr[y0:y1, x0:x1].astype(np.float32) / 255.0,
            global_lap_mean=glob["lap_mean"],
            global_dof_bimod=cross_dof_bimod,
            keys=keys,
        )
    return res


def build_clean_physics_reference(pils: Sequence) -> Dict[str, np.ndarray]:
    """Offline clean reference distributions from clean TRAINING frames only."""
    tile_vals: Dict[str, List[float]] = {k: [] for k in METRICS}
    glob_vals: Dict[str, List[float]] = {k: [] for k in METRICS}
    for im in pils:
        m = image_metrics(im)
        g_lap = m["global"]["lap_mean"]
        c_bimod = m["cross_dof_bimod"]
        for t in m["tiles"]:
            t = dict(t, hf_ratio=t["lap_mean"] / (g_lap + 1e-5), dof_bimod=max(t.get("dof_bimod", 0.0), c_bimod))
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
    n = len(ref_sorted)
    if n == 0:
        return float("nan")
    lo = np.searchsorted(ref_sorted, v, side="left")
    hi = np.searchsorted(ref_sorted, v, side="right")
    return float(100.0 * 0.5 * (lo + hi) / n)


def _fmt(v: float) -> str:
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"


def measure_defect(
    dim: str,
    phys: Dict[str, Dict[str, float]],
    ref: Dict[str, np.ndarray],
    use_box: bool,
    sev: str = "P0",
) -> Dict[str, object]:
    """Measured evidence for one flagged dimension (tiered Clean percentile + box-vs-global relative contrast)."""
    scope = "box" if (use_box and "box" in phys) else "global"
    pct_thr = float(TIERED_SUPPORT_PCT.get(sev, SUPPORT_PCT))
    items = []
    for k in DIM_METRICS[dim]:
        if k == "hf_ratio" and scope != "box":
            continue
        if f"phys_{'tile' if scope == 'box' else 'glob'}__{k}" not in ref:
            continue
        cn, direction = METRICS[k]
        v = float(phys[scope][k])
        ref_key = f"phys_tile__{k}" if scope == "box" else f"phys_glob__{k}"
        rk = clean_rank(ref[ref_key], v)
        pct_supp = bool(np.isfinite(rk) and (rk >= pct_thr if direction == "high" else rk <= 100.0 - pct_thr))
        rel_ratio = None
        rel_supp = False
        gv = None
        if scope == "box":
            gv = float(phys["global"][k])
            if gv >= 1e-5:
                rel_ratio = v / gv
                rel_supp = (rel_ratio >= 1.20) if direction == "high" else (rel_ratio <= 0.80)
        supp = bool(pct_supp or rel_supp)
        contra = bool(
            (not supp)
            and np.isfinite(rk)
            and (rk <= 100.0 - SUPPORT_PCT if direction == "high" else rk >= SUPPORT_PCT)
        )
        verdict = "supports" if supp else ("contradicts" if contra else "inconclusive")
        support_type = "clean_pct" if pct_supp else ("local_contrast" if rel_supp else "none")
        it = {
            "metric": k,
            "name": cn,
            "direction": direction,
            "value": round(v, 5),
            "clean_pct": round(rk, 1),
            "supports": supp,
            "support_type": support_type,
            "verdict": verdict,
        }
        if scope == "box":
            it["global_value"] = round(float(gv), 5)
            if rel_ratio is not None:
                it["rel_ratio"] = round(float(rel_ratio), 3)
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


def compose_reason(
    loc: str,
    sev: str,
    dim: str,
    ev: Dict[str, object],
    conf: Optional[float] = None,
    box_cue: Optional[str] = None,
    conc_ratio: Optional[float] = None,
    limit: int = 40,
) -> str:
    """
    <= `limit` Chinese characters by construction (zero truncation, de-templated):
      - When absolute Clean-percentile confirms (`support_type == 'clean_pct'`), cites measured pixel value & Clean percentile.
      - When local box-vs-global contrast confirms (`support_type == 'local_contrast'`), cites relative box/global contrast.
      - Otherwise cites observable box visual-content cue + ensemble confidence / spatial concentration (never fabricates causes).
    """
    its = ev.get("measurements", [])
    best = None
    if its:
        pct_cands = [x for x in its if x.get("support_type") == "clean_pct"]
        rel_cands = [x for x in its if x.get("support_type") == "local_contrast"]
        if pct_cands:
            best = max(pct_cands, key=lambda x: abs(float(x["clean_pct"]) - 50.0))
        elif rel_cands:
            best = max(rel_cands, key=lambda x: abs(float(x.get("rel_ratio", 1.0)) - 1.0))
        else:
            best = max(its, key=lambda x: abs(float(x["clean_pct"]) - 50.0))
    loc_s = loc.replace("画面", "") if loc != "全图" else "全图"
    loc_tag = f"{loc_s}({box_cue})" if (box_cue and loc != "全图") else loc
    loc_short = f"{loc_s}{box_cue}" if (box_cue and loc != "全图") else loc_s
    variants = []
    if best is not None and best.get("support_type") == "clean_pct":
        variants += [
            f"{loc_tag}{sev}级{dim}：实测{best['name']}{_fmt(best['value'])}（{_pct_tag(best)}）",
            f"{loc_short}{sev}{dim}：{best['name']}{_fmt(best['value'])}（{_pct_tag(best)}）",
            f"{loc_s}{sev}{dim}：{best['name']}{_fmt(best['value'])}（{_pct_tag(best)}）",
            f"{loc_s}{sev}{dim}：{best['name']}达{_pct_tag(best)}",
        ]
    elif best is not None and best.get("support_type") == "local_contrast":
        rr = float(best.get("rel_ratio", 1.0))
        pct_diff = int(round(abs(rr - 1.0) * 100.0))
        dir_txt = f"高{pct_diff}%" if rr >= 1.0 else f"低{pct_diff}%"
        m_nm = str(best["name"]).replace("局部/全局", "")
        m_nm_loc = m_nm if m_nm.startswith("局部") else f"局部{m_nm}"
        variants += [
            f"{loc_tag}{sev}级{dim}：{m_nm_loc}较全图{dir_txt}",
            f"{loc_short}{sev}{dim}：{m_nm}较全图{dir_txt}",
            f"{loc_s}{sev}{dim}：{m_nm_loc}较全图{dir_txt}",
        ]
    else:
        cue = VISUAL_PHENOMENON_CUE.get(dim, {}).get(sev, "检出视觉质量异常")
        stat_tag = ""
        conf_short = f"（置信{conf:.2f}）" if conf is not None else ""
        if conf is not None and conc_ratio is not None and loc != "全图":
            stat_tag = f"（集中{conc_ratio:.1f}x/置信{conf:.2f}）"
        elif conf is not None:
            stat_tag = f"（深度表征置信{conf:.2f}）"
        variants += [
            f"{loc_tag}{sev}级{dim}：{cue}{stat_tag}",
            f"{loc_tag}{sev}级{dim}：{cue}{conf_short}",
            f"{loc_short}{sev}{dim}：{cue}{stat_tag}",
            f"{loc_short}{sev}{dim}：{cue}{conf_short}",
            f"{loc_tag}{sev}级{dim}：{cue}",
            f"{loc_s}{sev}{dim}：{cue}{stat_tag}",
            f"{loc_s}{sev}{dim}：{cue}{conf_short}",
            f"{loc_s}{sev}{dim}：{cue}",
        ]
    for v in variants:
        if len(v) <= limit:
            return v
    return f"{loc_s}{sev}级{dim}"[:limit]
