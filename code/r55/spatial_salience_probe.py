#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/spatial_salience_probe.py
==================================
Method 1: Spatial Salience x Native 1:1 Tile Artifact Interaction Extractor
for `过锐` (dim=2) and `过度生成` (dim=4).

Key physical & statistical properties:
  1. Zero-Leakage Paired Tower Discovery:
     - Automatically verifies that BOTH all 8 held-out fold checkpoints (`fold_0_best.pt` .. `fold_7_best.pt`)
       AND the matched 100%-data full-fit checkpoint exist on disk before activating a tile backbone
       (`ntg_pe`, `ntg_convnext`, with automatic fallback to `pe_v3` / `qy` 8-fold checkpoints if needed).
     - Guarantees exact feature-space alignment between `--mode oof` (held-out fold-k weights only) and
       `--mode target` (full-fit weights on `test_150`).
  2. Native 1:1 Pixel Coordinate Alignment (Zero Square-Squash Distortion):
     - Evaluates every native `448x448` tile `(x0, y0, x0+448, y0+448)` from `tile_starts(W, 448) x tile_starts(H, 448)`
       with 2x horizontal-flip TTA (flipping tile + global context symmetrically so flipped tile `t` maps back
       to the exact physical region `(x0, y0, x0+448, y0+448)`).
     - Aligns per-tile neural severity `sev_t` & grade probabilities `probs_t` against:
         * Foreground normalized depth map `z_hat` (`CoCDepthDefocusProbe` / `Depth-Anything-S`),
         * Semantic 1:1 ROIs (`subtitle_ocr`, `face_anat`, `hand_swirl`, `bokeh_halo`) + YCrCb skin salience,
         * Native 1:1 signed bipolar overshoot (`bipolar_halo`) split strictly into outside-UI vs inside-UI text strokes.
  3. Strict `N = 1` Single-Image Independence:
     - Every image's 20-D spatial interaction vector is computed purely from its own pixels (`f([x_i]) == f(X)[i]`),
       never reading filenames or batch statistics.
"""
import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

TILE = 448

SPATIAL_FEAT_NAMES: Tuple[str, ...] = (
    # --- dim=2 过锐 (Oversharpening: foreground non-UI halo vs subtitle stroke false-alarm separation) ---
    "os_fg_p_any",            # 0: foreground & non-UI weighted P(过锐 >= P2)
    "os_fg_top2_sev",         # 1: top-2 continuous severity_s[2] on non-UI foreground tiles
    "os_ui_vs_nonui_gap",     # 2: P(过锐 >= P2 | UI/subtitle tiles) - P(过锐 >= P2 | non-UI fg tiles)
    "os_fg_vs_bg_sev_gap",    # 3: foreground vs background continuous severity_s[2] contrast
    "os_halo_nonui_x_tile",   # 4: 1:1 physical non-UI bipolar overshoot x neural tile P(过锐 >= P2)
    "os_halo_ui_ratio",       # 5: fraction of physical bipolar ringing energy located inside UI/subtitle mask
    "os_tile_max_p01",        # 6: peak non-UI tile P(过锐 >= P1)
    "os_tile_conc_gini",      # 7: spatial concentration max_t P(>=P2) - mean_t P(>=P2)
    "os_mil_nonui_mass",      # 8: tower MIL gate attention mass landing outside UI/subtitle regions
    "os_texloss_cotile",      # 9: same-tile co-activation of P(过锐 >= P2) * P(纹理损失 >= P2) on subject
    # --- dim=4 过度生成 (AI Over-generation: foreground subject + face/hand/skin anatomy peak pooling) ---
    "og_fg_p_any",            # 10: foreground depth-weighted P(过度生成 >= P2)
    "og_anat_p_any",          # 11: face/hand/skin anatomy-weighted P(过度生成 >= P2)
    "og_fg_top2_sev",         # 12: top-2 continuous severity_s[4] on foreground tiles
    "og_anat_max_sev",        # 13: peak continuous severity_s[4] on anatomy/skin tiles (or fg fallback)
    "og_has_anat_roi",        # 14: continuous anatomy/skin coverage indicator in [0, 1]
    "og_fg_vs_bg_gap",        # 15: foreground vs background continuous severity_s[4] contrast
    "og_local_peak_gap",      # 16: localized synthesis spike max_t(fg * P(>=P2)) - mean_t P(>=P2)
    "og_p01_fg_peak",         # 17: peak foreground tile P(过度生成 >= P1)
    "og_vs_blur_tex_margin",  # 18: max foreground tile (sev[4] - 0.5*(sev[0] + sev[1])) margin
    "og_meor_x_tile",         # 19: physical micro-edge regularity excess x foreground tile P(过度生成 >= P2)
)
SPATIAL_FEAT_DIM = len(SPATIAL_FEAT_NAMES)


def _resolve_ckpt(repo_rel: str, pack_rel: str = "") -> str:
    p_repo = os.path.join(REPO, repo_rel)
    if os.path.exists(p_repo):
        return p_repo
    if pack_rel:
        p_pack = os.path.join(os.path.dirname(REPO), "03_模型产物", pack_rel)
        if os.path.exists(p_pack):
            return p_pack
    return p_repo


def discover_paired_tile_towers() -> List[Dict[str, str]]:
    """
    Discovers tile-capable vision towers that have BOTH:
      (a) all 8 held-out fold checkpoints `fold_0_best.pt` .. `fold_7_best.pt`, AND
      (b) a matched 100%-data full-fit checkpoint for target inference.
    Selects up to 2 complementary families: 1 PE-Core-Large tower + 1 ConvNeXtV2-Base tower (if saved).
    """
    pe_candidates = [
        dict(
            tag="ntg_pe",
            tower_cls="ntg",
            backbone="pe_core_large",
            oof_dir="results/next55/towers/r55_ntg_s2026/pe_core_large",
            fullfit_ckpt=_resolve_ckpt("results/r55/fullfit_v5_ntg/pe_core_large/fold_0_best.pt", "checkpoints/ntg_pe/fold_0_best.pt"),
        ),
        dict(
            tag="pe_v3_tile",
            tower_cls="tier1",
            backbone="pe_core_large",
            oof_dir="results/next55/towers/r55_fixed_s2026/pe_core_large",
            fullfit_ckpt=_resolve_ckpt("results/r55/fullfit_v3_tier1/pe_core_large/fold_0_best.pt", "checkpoints/pe_v3/fold_0_best.pt"),
        ),
        dict(
            tag="pe_qy_tile",
            tower_cls="tier1",
            backbone="pe_core_large",
            oof_dir="results/next55/towers/qy/pe_core_large",
            fullfit_ckpt=_resolve_ckpt("results/next55/fullfit/qy/pe_core_large/fold_0_best.pt", "checkpoints/pe/fold_0_best.pt"),
        ),
    ]
    cnx_candidates = [
        dict(
            tag="ntg_convnext",
            tower_cls="ntg",
            backbone="convnextv2_base",
            oof_dir="results/next55/towers/r55_ntg_convnext_e8_s2026/convnextv2_base",
            fullfit_ckpt=_resolve_ckpt("results/r55/fullfit_v7_ntg_convnext/convnextv2_base/fold_0_best.pt", "checkpoints/ntg_convnext/fold_0_best.pt"),
        ),
        dict(
            tag="cnv_qy_tile",
            tower_cls="tier1",
            backbone="convnextv2_base",
            oof_dir="results/next55/towers/qy/convnextv2_base",
            fullfit_ckpt=_resolve_ckpt("results/next55/fullfit/qy/convnextv2_base/fold_0_best.pt", ""),
        ),
    ]

    def _has_complete_pair(spec: Dict[str, str]) -> bool:
        if not os.path.exists(spec["fullfit_ckpt"]):
            return False
        for k in range(8):
            fp = os.path.join(REPO, spec["oof_dir"], f"fold_{k}_best.pt")
            if not os.path.exists(fp):
                return False
        return True

    chosen: List[Dict[str, str]] = []
    for cand in pe_candidates:
        if _has_complete_pair(cand):
            chosen.append(cand)
            break
    for cand in cnx_candidates:
        if _has_complete_pair(cand):
            chosen.append(cand)
            break

    if not chosen:
        raise FileNotFoundError(
            "No paired 8-fold + fullfit tile tower checkpoints found under results/next55/towers/ and results/r55/."
        )
    return chosen


def _box_iou_area_frac(tile_box: Tuple[int, int, int, int], roi_box: Sequence[int]) -> float:
    tx0, ty0, tx1, ty1 = tile_box
    rx0, ry0, rx1, ry1 = [int(v) for v in roi_box]
    ix0, iy0 = max(tx0, rx0), max(ty0, ry0)
    ix1, iy1 = min(tx1, rx1), min(ty1, ry1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    t_area = float(max(1, (tx1 - tx0) * (ty1 - ty0)))
    return inter / t_area


def _compute_skin_prob_map(rgb_u8: np.ndarray) -> np.ndarray:
    """Fast deterministic YCrCb skin-chrominance probability map in [0, 1] at native (H, W)."""
    r = rgb_u8[:, :, 0].astype(np.float32)
    g = rgb_u8[:, :, 1].astype(np.float32)
    b = rgb_u8[:, :, 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 128.0 + 0.500 * r - 0.418688 * g - 0.081312 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.500 * b
    # Smooth Gaussian-like chrominance ellipsoid around natural human skin tones (Cr ~ 148, Cb ~ 112)
    d2 = ((cr - 148.0) / 16.0) ** 2 + ((cb - 112.0) / 16.0) ** 2
    lum_gate = np.clip((y - 24.0) / 36.0, 0.0, 1.0) * np.clip((248.0 - y) / 24.0, 0.0, 1.0)
    return (np.exp(-0.5 * d2) * lum_gate).astype(np.float32)


def aggregate_spatial_salience_features(
    tile_probs: np.ndarray,
    tile_sev: np.ndarray,
    tile_mil_w: np.ndarray,
    tile_boxes: List[Tuple[int, int, int, int]],
    z_hat: np.ndarray,
    phys_maps: Dict[str, np.ndarray],
    sem_meta: Dict[str, np.ndarray],
    skin_map: np.ndarray,
) -> np.ndarray:
    """
    Pure NumPy deterministic spatial interaction aggregator (`N = 1` per image).
    Inputs:
      - `tile_probs`: `(T, 6, 4)` calibrated simplex per native `448x448` tile
      - `tile_sev`:   `(T, 6)` continuous latent severity `s_d` per tile
      - `tile_mil_w`: `(T,)` MIL attention weight across tiles (`sum == 1`)
      - `tile_boxes`: list of `T` `(x0, y0, x1, y1)` native-pixel boxes
      - `z_hat`:      `(H, W)` normalized foreground depth map in `[0, 1]`
      - `phys_maps`:  dict from `compute_1to1_wavelet_and_artifact_maps`
      - `sem_meta`:   dict from `analyze_semantic_rois_pil` (`crops_448_boxes`, `phys_vec`)
      - `skin_map`:   `(H, W)` chrominance skin probability in `[0, 1]`
    Returns:
      - `feat_20d`: `(20,)` float32 vector matching `SPATIAL_FEAT_NAMES`.
    """
    T = len(tile_boxes)
    assert tile_probs.shape == (T, 6, 4) and tile_sev.shape == (T, 6)

    bipolar = np.asarray(phys_maps["bipolar_halo"], dtype=np.float32)
    ui_mask = np.asarray(phys_maps["ui_text_mask"], dtype=np.float32)
    meor_map = np.asarray(phys_maps.get("meor_excess", np.zeros_like(bipolar)), dtype=np.float32)

    z_med = float(np.median(z_hat))
    fg_pix = 1.0 / (1.0 + np.exp(-6.0 * (z_hat.astype(np.float32) - z_med)))
    ui_bin = (ui_mask > 0.35).astype(np.float32)
    non_ui_pix = 1.0 - ui_bin

    roi_boxes = np.asarray(sem_meta["crops_448_boxes"], dtype=np.int32)  # [b_text, b_face, b_hand, b_bokeh]
    sem_vec = np.asarray(sem_meta["phys_vec"], dtype=np.float32)
    # Semantic coverage indicators from analyze_semantic_rois_pil (indices 36..47 when base_phys_36 is prepended, or 0..11)
    sem_tail = sem_vec[-12:] if sem_vec.size >= 12 else np.zeros(12, dtype=np.float32)
    face_cov_global = float(np.clip(sem_tail[1] if sem_tail.size > 1 else 0.0, 0.0, 1.0))
    skin_cov_global = float(np.clip(np.mean(skin_map * fg_pix > 0.25), 0.0, 1.0))

    fg_w = np.zeros(T, dtype=np.float64)
    bg_w = np.zeros(T, dtype=np.float64)
    ui_cov = np.zeros(T, dtype=np.float64)
    non_ui_w = np.zeros(T, dtype=np.float64)
    anat_w = np.zeros(T, dtype=np.float64)
    halo_non_ui_t = np.zeros(T, dtype=np.float64)
    halo_ui_t = np.zeros(T, dtype=np.float64)
    meor_t = np.zeros(T, dtype=np.float64)

    for t, (x0, y0, x1, y1) in enumerate(tile_boxes):
        fg_sl = fg_pix[y0:y1, x0:x1]
        ui_sl = ui_bin[y0:y1, x0:x1]
        nui_sl = non_ui_pix[y0:y1, x0:x1]
        bp_sl = bipolar[y0:y1, x0:x1]
        sk_sl = skin_map[y0:y1, x0:x1]
        mr_sl = meor_map[y0:y1, x0:x1]

        fg_m = float(fg_sl.mean()) if fg_sl.size else 0.5
        fg_w[t] = fg_m
        bg_w[t] = max(0.0, 1.0 - fg_m)

        ocr_ov = _box_iou_area_frac((x0, y0, x1, y1), roi_boxes[0]) if roi_boxes.shape[0] >= 1 else 0.0
        ui_c = float(ui_sl.mean()) if ui_sl.size else 0.0
        ui_cov[t] = min(1.0, 0.70 * ui_c + 0.30 * ocr_ov)
        non_ui_w[t] = float(np.clip(1.0 - 2.4 * ui_cov[t], 0.05, 1.0))

        face_ov = _box_iou_area_frac((x0, y0, x1, y1), roi_boxes[1]) if roi_boxes.shape[0] >= 2 else 0.0
        hand_ov = _box_iou_area_frac((x0, y0, x1, y1), roi_boxes[2]) if roi_boxes.shape[0] >= 3 else 0.0
        sk_fg = float((sk_sl * fg_sl).mean()) if sk_sl.size else 0.0
        anat_w[t] = float(np.clip(0.45 * face_ov + 0.30 * hand_ov + 0.65 * sk_fg, 0.0, 1.0))

        bp_nui = bp_sl * nui_sl
        bp_ui = bp_sl * ui_sl
        halo_non_ui_t[t] = float(0.5 * bp_nui.mean() + 0.5 * np.percentile(bp_nui, 95)) if bp_nui.size else 0.0
        halo_ui_t[t] = float(0.5 * bp_ui.mean() + 0.5 * np.percentile(bp_ui, 95)) if bp_ui.size else 0.0
        meor_t[t] = float(mr_sl.mean()) if mr_sl.size else 0.0

    # --- Oversharpening (dim = 2) features ---
    p_os_any = 1.0 - np.clip(tile_probs[:, 2, 0], 0.0, 1.0)
    p_os_p01 = np.clip(tile_probs[:, 2, 2] + tile_probs[:, 2, 3], 0.0, 1.0)
    sev_os = tile_sev[:, 2].astype(np.float64)
    p_tex_any = 1.0 - np.clip(tile_probs[:, 1, 0], 0.0, 1.0)

    w_os_fg = fg_w * non_ui_w + 1e-4
    os_fg_p_any = float(np.sum(w_os_fg * p_os_any) / np.sum(w_os_fg))

    nui_adj_sev = sev_os * (0.5 + 0.5 * fg_w) * non_ui_w
    k_top = min(2, T)
    os_fg_top2_sev = float(np.mean(np.sort(nui_adj_sev)[-k_top:]))

    ui_w_sum = float(np.sum(ui_cov))
    if ui_w_sum > 0.08:
        os_ui_p_any = float(np.sum(ui_cov * p_os_any) / (ui_w_sum + 1e-6))
        os_ui_vs_nonui_gap = os_ui_p_any - os_fg_p_any
    else:
        os_ui_vs_nonui_gap = 0.0

    fg_sev_os = float(np.sum((fg_w + 1e-4) * sev_os) / np.sum(fg_w + 1e-4))
    bg_sev_os = float(np.sum((bg_w + 1e-4) * sev_os) / np.sum(bg_w + 1e-4))
    os_fg_vs_bg_sev_gap = fg_sev_os - bg_sev_os

    os_halo_nonui_x_tile = float(np.mean(halo_non_ui_t * p_os_any))
    tot_halo_nui = float(np.sum(halo_non_ui_t))
    tot_halo_ui = float(np.sum(halo_ui_t))
    os_halo_ui_ratio = tot_halo_ui / (tot_halo_nui + tot_halo_ui + 1e-5)

    os_tile_max_p01 = float(np.max(non_ui_w * p_os_p01))
    os_tile_conc_gini = float(np.max(p_os_any) - np.mean(p_os_any))
    mil_norm = tile_mil_w / max(1e-6, float(np.sum(tile_mil_w)))
    os_mil_nonui_mass = float(np.sum(mil_norm * non_ui_w))
    os_texloss_cotile = float(np.sum(w_os_fg * p_os_any * p_tex_any) / np.sum(w_os_fg))

    # --- AI Over-generation (dim = 4) features ---
    p_og_any = 1.0 - np.clip(tile_probs[:, 4, 0], 0.0, 1.0)
    p_og_p01 = np.clip(tile_probs[:, 4, 2] + tile_probs[:, 4, 3], 0.0, 1.0)
    sev_og = tile_sev[:, 4].astype(np.float64)
    sev_blur = tile_sev[:, 0].astype(np.float64)
    sev_tex = tile_sev[:, 1].astype(np.float64)

    w_og_fg = fg_w + 1e-4
    og_fg_p_any = float(np.sum(w_og_fg * p_og_any) / np.sum(w_og_fg))

    og_has_anat_roi = float(np.clip(max(face_cov_global + skin_cov_global, float(np.max(anat_w))), 0.0, 1.0))
    if float(np.sum(anat_w)) > 0.06:
        og_anat_p_any = float(np.sum(anat_w * p_og_any) / (np.sum(anat_w) + 1e-6))
        og_anat_max_sev = float(np.max(sev_og * (0.35 + 0.65 * (anat_w / (np.max(anat_w) + 1e-6)))))
    else:
        og_anat_p_any = og_fg_p_any
        og_anat_max_sev = float(np.mean(np.sort(sev_og * (0.5 + 0.5 * fg_w))[-k_top:]))

    og_fg_top2_sev = float(np.mean(np.sort(sev_og * (0.5 + 0.5 * fg_w))[-k_top:]))
    fg_sev_og = float(np.sum(w_og_fg * sev_og) / np.sum(w_og_fg))
    bg_sev_og = float(np.sum((bg_w + 1e-4) * sev_og) / np.sum(bg_w + 1e-4))
    og_fg_vs_bg_gap = fg_sev_og - bg_sev_og

    og_local_peak_gap = float(np.max(fg_w * p_og_any) - np.mean(p_og_any))
    og_p01_fg_peak = float(np.max(fg_w * p_og_p01))
    og_vs_blur_tex_margin = float(np.max((0.4 + 0.6 * fg_w) * (sev_og - 0.5 * (sev_blur + sev_tex))))
    og_meor_x_tile = float(np.mean(meor_t * fg_w * p_og_any))

    out = np.array(
        [
            os_fg_p_any,
            os_fg_top2_sev,
            os_ui_vs_nonui_gap,
            os_fg_vs_bg_sev_gap,
            os_halo_nonui_x_tile,
            os_halo_ui_ratio,
            os_tile_max_p01,
            os_tile_conc_gini,
            os_mil_nonui_mass,
            os_texloss_cotile,
            og_fg_p_any,
            og_anat_p_any,
            og_fg_top2_sev,
            og_anat_max_sev,
            og_has_anat_roi,
            og_fg_vs_bg_gap,
            og_local_peak_gap,
            og_p01_fg_peak,
            og_vs_blur_tex_margin,
            og_meor_x_tile,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)


class SpatialSalienceExtractor:
    """
    GPU-resident native 1:1 per-tile + depth + semantic ROI interaction extractor.
    Strictly single-image (`N = 1`) invariant.
    """

    def __init__(self, tower_specs: List[Dict[str, str]], mode: str, fold: int = -1, device: str = "cuda", img_dir: str = ""):
        import torch
        from native_tiles import NativeTileTower
        from phoenix.config import resolve_local_or_hf_model
        from phoenix.native_physics_and_views import CoCDepthDefocusProbe
        from qingyu.backbones_qingyu import QingyuVisionQualityTower
        from qingyu.config_and_paths import REPO_ROOT, LosslessNativeImageLoader, setup_sys_path_and_cwd
        from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM, DualEnginePhysicsExtractor

        setup_sys_path_and_cwd(chdir_repo=True)
        self.torch = torch
        self.dev = torch.device(device)
        self.mode = mode
        self.fold = int(fold)
        self.loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=[img_dir] if img_dir else None)
        self.phys_extractor = DualEnginePhysicsExtractor(device=str(self.dev), allow_depth_download=False)
        self.coc = CoCDepthDefocusProbe(model_path=resolve_local_or_hf_model("pillar3_depth_anything_s"), device=str(self.dev))

        # Optional precomputed caches for training images ONLY when mode == "oof" (target mode computes 100% on the fly)
        self.sem_cache: Dict[str, Dict[str, np.ndarray]] = {}
        if mode == "oof":
            sem_npz = os.path.join(REPO, "results/r55/semantic_roi_meta.npz")
            if os.path.exists(sem_npz):
                z = np.load(sem_npz, allow_pickle=True)
                for nm, bx, pv in zip(z["names"], z["crops_448_boxes"], z["phys_vecs"]):
                    self.sem_cache[str(nm)] = {"crops_448_boxes": bx, "phys_vec": pv}

        self.models = []
        for spec in tower_specs:
            ckpt_path = (
                os.path.join(REPO, spec["oof_dir"], f"fold_{self.fold}_best.pt")
                if mode == "oof"
                else spec["fullfit_ckpt"]
            )
            assert os.path.exists(ckpt_path), f"Missing tile checkpoint: {ckpt_path}"
            if spec["tower_cls"] == "ntg":
                m = NativeTileTower(backbone_key=spec["backbone"], phys_dim=PHYS_FEAT_DIM, use_global=True)
            else:
                m = QingyuVisionQualityTower(backbone_key=spec["backbone"], phys_dim=PHYS_FEAT_DIM)
            sd = torch.load(ckpt_path, map_location="cpu")
            sd = sd.get("state_dict", sd)
            _, unexpected = m.load_state_dict(sd, strict=False)
            assert not unexpected, f"Unexpected keys in {ckpt_path}: {unexpected[:5]}"
            m.backbone.to(torch.bfloat16)
            for n, mod in m.backbone.named_modules():
                if any(t in n.split(".")[-1] for t in ("tap3", "dual_tap", "abmil", "spatial_proj")):
                    mod.float()
            m.to(self.dev).eval()
            self.models.append((spec["tag"], m))
            print(f"  [spatial_probe] loaded {spec['tag']} ({mode}, ckpt={os.path.relpath(ckpt_path, REPO)})", flush=True)

    def _encode_tiles_2x_tta(self, model, tiles_t: "torch.Tensor", global_t: "torch.Tensor", phys_36: "torch.Tensor"):
        """
        Runs 2x horizontal-flip TTA on all `T` native `448x448` tiles + global `448x448` view.
        Notice flipping `tiles_t` along width (`dims=[-1]`) AND `global_t` along width (`dims=[-1]`)
        preserves tile index `t`'s correspondence to physical image box `tile_boxes[t]`!
        """
        torch = self.torch
        T = tiles_t.shape[0]
        probs_acc = np.zeros((T, 6, 4), dtype=np.float64)
        sev_acc = np.zeros((T, 6), dtype=np.float64)
        mil_acc = np.zeros(T, dtype=np.float64)

        phys_rep = phys_36.view(1, -1).expand(T, -1)
        for flip in (False, True):
            tv = torch.flip(tiles_t, dims=[-1]) if flip else tiles_t
            gv = torch.flip(global_t, dims=[-1]) if flip else global_t
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                norm_tv = model._norm_img(tv)
                chunks = [model._encode_single_view(norm_tv[s : s + 16]) for s in range(0, T, 16)]
                f_tiles = torch.cat(chunks, dim=0).float()                   # (T, D)
                g_emb = model._encode_single_view(model._norm_img(gv)).float()  # (1, D)
            with torch.no_grad():
                lg = model.view_agg.gate(f_tiles.unsqueeze(0)).squeeze(0).squeeze(-1)  # (T,)
                w_mil = torch.softmax(lg.float(), dim=0)
                g_rep = g_emb.expand(T, -1)
                vis_per_tile = torch.cat([g_rep, f_tiles, f_tiles], dim=-1)  # (T, 3*D)
                hout = model.head(vis_per_tile, phys_feat=phys_rep)
                probs_acc += 0.5 * hout["probs_6x4"].float().cpu().numpy()
                sev_acc += 0.5 * hout["severity_s"].float().cpu().numpy()
                mil_acc += 0.5 * w_mil.float().cpu().numpy()
        return probs_acc, sev_acc, mil_acc

    def extract_one(self, name: str) -> np.ndarray:
        from PIL import Image
        from native_tiles import tile_starts
        from phoenix.native_physics_and_views import compute_1to1_wavelet_and_artifact_maps
        from semantic_roi_meta import analyze_semantic_rois_pil

        torch = self.torch
        pil = self.loader.load_pil(name).convert("RGB")
        W, H = pil.size
        rgb_u8 = np.asarray(pil, dtype=np.uint8)

        phys_maps = compute_1to1_wavelet_and_artifact_maps(rgb_u8)
        z_hat = self.coc.predict_normalized_depth(pil)
        skin_map = _compute_skin_prob_map(rgb_u8)

        if self.mode == "oof" and str(name) in self.sem_cache:
            sem_meta = self.sem_cache[str(name)]
            base_phys_36 = np.asarray(sem_meta["phys_vec"][:36], dtype=np.float32)
        else:
            meta_base = self.phys_extractor.extract_from_pil(pil)
            base_phys_36 = np.asarray(meta_base["phys_vec"], dtype=np.float32)
            sem_meta = analyze_semantic_rois_pil(pil, base_phys_36=base_phys_36)

        xs = tile_starts(W, TILE)
        ys = tile_starts(H, TILE)
        tile_boxes: List[Tuple[int, int, int, int]] = []
        tile_arrs: List[np.ndarray] = []

        # Pad frame symmetrically only if W < 448 or H < 448
        if H < TILE or W < TILE:
            pad_h = max(0, TILE - H)
            pad_w = max(0, TILE - W)
            rgb_pad = np.pad(rgb_u8, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            rgb_pad = rgb_u8

        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(H, y0 + TILE), min(W, x0 + TILE)
                tile_boxes.append((x0, y0, x1, y1))
                crop = rgb_pad[y0 : y0 + TILE, x0 : x0 + TILE]
                tile_arrs.append(crop)

        tiles_np = np.stack(tile_arrs, axis=0).astype(np.float32) / 255.0  # (T, 448, 448, 3)
        tiles_t = torch.from_numpy(tiles_np).permute(0, 3, 1, 2).contiguous().to(self.dev)

        g_pil = pil.resize((TILE, TILE), Image.BICUBIC)
        g_np = np.asarray(g_pil, dtype=np.float32) / 255.0
        global_t = torch.from_numpy(g_np).permute(2, 0, 1).unsqueeze(0).contiguous().to(self.dev)
        phys_t = torch.from_numpy(base_phys_36).to(self.dev)

        T = len(tile_boxes)
        probs_ens = np.zeros((T, 6, 4), dtype=np.float64)
        sev_ens = np.zeros((T, 6), dtype=np.float64)
        mil_ens = np.zeros(T, dtype=np.float64)

        w_m = 1.0 / len(self.models)
        for _, m in self.models:
            pm, sm, mm = self._encode_tiles_2x_tta(m, tiles_t, global_t, phys_t)
            probs_ens += w_m * pm
            sev_ens += w_m * sm
            mil_ens += w_m * mm

        return aggregate_spatial_salience_features(
            tile_probs=probs_ens,
            tile_sev=sev_ens,
            tile_mil_w=mil_ens,
            tile_boxes=tile_boxes,
            z_hat=z_hat,
            phys_maps=phys_maps,
            sem_meta=sem_meta,
            skin_map=skin_map,
        )


def _run_self_test() -> None:
    """Synthetic deterministic check verifying `N = 1` invariance and finite outputs."""
    rng = np.random.RandomState(42)
    H, W = 720, 576
    boxes = [(0, 0, 448, 448), (128, 0, 576, 448), (0, 272, 448, 720), (128, 272, 576, 720)]
    T = len(boxes)
    tp = rng.dirichlet([3.0, 0.5, 0.3, 0.2], size=(T, 6))
    ts = rng.randn(T, 6)
    mw = rng.dirichlet(np.ones(T))
    zh = rng.uniform(0.0, 1.0, size=(H, W)).astype(np.float32)
    pm = {
        "bipolar_halo": rng.uniform(0.0, 0.4, size=(H, W)).astype(np.float32),
        "ui_text_mask": (rng.uniform(0.0, 1.0, size=(H, W)) > 0.85).astype(np.float32),
        "meor_excess": rng.uniform(0.0, 0.3, size=(H, W)).astype(np.float32),
    }
    sm = {
        "crops_448_boxes": np.array([[40, 520, 488, 700], [60, 80, 508, 528], [80, 200, 528, 648], [0, 0, 448, 448]], dtype=np.int32),
        "phys_vec": rng.randn(48).astype(np.float32),
    }
    sk = _compute_skin_prob_map(rng.randint(0, 256, size=(H, W, 3), dtype=np.uint8))
    f1 = aggregate_spatial_salience_features(tp, ts, mw, boxes, zh, pm, sm, sk)
    f2 = aggregate_spatial_salience_features(tp.copy(), ts.copy(), mw.copy(), list(boxes), zh.copy(), pm, sm, sk.copy())
    assert f1.shape == (SPATIAL_FEAT_DIM,)
    assert np.all(np.isfinite(f1))
    assert np.max(np.abs(f1 - f2)) == 0.0
    print(f"[spatial_salience_probe][self-test] PASS: dim={SPATIAL_FEAT_DIM}, max_diff=0.0, sample={np.round(f1[:6], 4)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["oof", "merge-oof", "target", "merge-target", "self-test"])
    ap.add_argument("--fold", type=int, default=0, help="0..7 for --mode oof")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--template", default="", help="Template JSON for --mode target / merge-target")
    ap.add_argument("--img-dir", default="", help="Image directory for --mode target")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default="results/r55/os_og_breakthrough/spatial_cache")
    ap.add_argument("--out-npz", default="")
    args = ap.parse_args()

    if args.mode == "self-test":
        return _run_self_test()

    import common55 as c

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.mode == "merge-oof":
        names, _, fold_arr, _ = c.load_split()
        idx_map = {str(n): i for i, n in enumerate(names)}
        feats = np.zeros((len(names), SPATIAL_FEAT_DIM), dtype=np.float32)
        valid = np.zeros(len(names), dtype=bool)
        towers_used = None
        for k in range(8):
            fp = os.path.join(out_dir, f"oof_fold_{k}.npz")
            assert os.path.exists(fp), f"Missing fold file {fp}"
            z = np.load(fp, allow_pickle=True)
            meta_k = json.loads(str(z["meta_json"]))
            if towers_used is None:
                towers_used = meta_k["towers"]
            else:
                assert towers_used == meta_k["towers"], f"Tower mismatch on fold {k}: {towers_used} vs {meta_k['towers']}"
            for nm, fv in zip(z["names"], z["feats"]):
                ii = idx_map[str(nm)]
                assert int(fold_arr[ii]) == k, f"Fold leak check failed for {nm}: expected {fold_arr[ii]}, got {k}"
                feats[ii] = fv
                valid[ii] = True
        assert bool(valid.all()), f"Incomplete OOF spatial features: {valid.sum()}/{len(names)}"
        out_p = args.out_npz or os.path.join(out_dir, "spatial_oof_3573.npz")
        meta = dict(mode="oof", n=len(names), feat_names=list(SPATIAL_FEAT_NAMES), towers=towers_used)
        np.savez_compressed(out_p, names=np.array(names), feats=feats, valid_mask=valid,
                            feat_names=np.array(SPATIAL_FEAT_NAMES), meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
        print(f"[spatial_salience_probe] merged 8-fold leak-free OOF ({len(names)} rows, dim={SPATIAL_FEAT_DIM}, towers={towers_used}) -> {out_p}")
        return

    if args.mode == "merge-target":
        assert args.template, "--template is required for --mode merge-target"
        names = c.load_template_names(args.template)
        idx_map = {str(n): i for i, n in enumerate(names)}
        feats = np.zeros((len(names), SPATIAL_FEAT_DIM), dtype=np.float32)
        valid = np.zeros(len(names), dtype=bool)
        towers_used = None
        shard_files = sorted(glob.glob(os.path.join(out_dir, "target_shard_*_of_*.npz")))
        assert shard_files, f"No target shard files found in {out_dir}"
        for fp in shard_files:
            z = np.load(fp, allow_pickle=True)
            meta_s = json.loads(str(z["meta_json"]))
            if int(meta_s.get("num_shards", -1)) != int(args.num_shards):
                continue
            if towers_used is None:
                towers_used = meta_s["towers"]
            for nm, fv in zip(z["names"], z["feats"]):
                sn = str(nm)
                if sn in idx_map:
                    feats[idx_map[sn]] = fv
                    valid[idx_map[sn]] = True
        assert bool(valid.all()), f"Incomplete target spatial features: {valid.sum()}/{len(names)}"
        out_p = args.out_npz or os.path.join(out_dir, "spatial_target.npz")
        meta = dict(mode="target", n=len(names), feat_names=list(SPATIAL_FEAT_NAMES), towers=towers_used, template=args.template)
        np.savez_compressed(out_p, names=np.array(names), feats=feats, valid_mask=valid,
                            feat_names=np.array(SPATIAL_FEAT_NAMES), meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
        print(f"[spatial_salience_probe] merged target spatial features ({len(names)} rows, dim={SPATIAL_FEAT_DIM}, towers={towers_used}) -> {out_p}")
        return

    # GPU worker modes: oof (fold k) or target (shard s / M)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    specs = discover_paired_tile_towers()
    tower_tags = [s["tag"] for s in specs]

    if args.mode == "oof":
        names_all, _, fold_arr, _ = c.load_split()
        sub_names = [str(names_all[i]) for i in range(len(names_all)) if int(fold_arr[i]) == int(args.fold)]
        out_p = args.out_npz or os.path.join(out_dir, f"oof_fold_{args.fold}.npz")
    else:
        assert args.template, "--template is required for --mode target"
        names_all = c.load_template_names(args.template)
        sub_names = [str(names_all[i]) for i in range(len(names_all)) if (i % args.num_shards) == args.shard]
        out_p = args.out_npz or os.path.join(out_dir, f"target_shard_{args.shard:02d}_of_{args.num_shards:02d}.npz")

    t0 = time.time()
    ext = SpatialSalienceExtractor(specs, mode=args.mode, fold=args.fold, device="cuda", img_dir=args.img_dir)
    rows_f = np.zeros((len(sub_names), SPATIAL_FEAT_DIM), dtype=np.float32)
    for idx_i, nm in enumerate(sub_names):
        rows_f[idx_i] = ext.extract_one(nm)
        if (idx_i + 1) % 25 == 0 or (idx_i + 1) == len(sub_names):
            el = time.time() - t0
            print(f"  [spatial_probe:{args.mode}] ({idx_i + 1}/{len(sub_names)}) {(el / (idx_i + 1)):.2f}s/img -> {os.path.basename(out_p)}", flush=True)

    meta = dict(
        mode=args.mode,
        fold=int(args.fold),
        shard=int(args.shard),
        num_shards=int(args.num_shards),
        n=len(sub_names),
        towers=tower_tags,
        feat_names=list(SPATIAL_FEAT_NAMES),
    )
    np.savez_compressed(
        out_p,
        names=np.array(sub_names),
        feats=rows_f,
        feat_names=np.array(SPATIAL_FEAT_NAMES),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    print(f"[spatial_salience_probe] saved {out_p} ({len(sub_names)} images, {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
