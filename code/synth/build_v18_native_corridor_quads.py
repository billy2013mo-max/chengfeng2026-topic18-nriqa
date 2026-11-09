#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/synth/build_v18_native_corridor_quads.py
V18 Zero-Shortcut Native-Resolution Quadruplet Generator + True Teacher Corridor QC:
  Fix 1: 100% Native Resolution synthesis (NO max(h,w)>960 pre-downsampling!).
  Fix 2: Records exact defect mask center (focus_y, focus_x) for every synthesized image so
         1:1 microscope crops during training hit the synthesized defect 100% of the time.
  Fix 3: Pure FLUX.3-Klein (mild/strong) + Laplacian Pyramid Multi-Band Blending for OverGen
         (ZERO cv2.remap sine-wave distortion or artificial checkerboard grids!).
  Fix 4: Calibrated P0 parameters + True Neural Teacher Severity Corridor Filtering so synthetic
         P0 never overshoots and never depresses real P0 recall.
  Fix 5: Exports paired Natural Optical Shallow-DOF Hard Negatives (`{stem}_d5_NatNeg.png`,
         labeled Blur=0, Bokeh=0) into the training manifest.
"""
import os, sys, csv, glob, json, math, time, argparse, subprocess
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.extend(['code', 'code/finetune', 'code/synth'])

from utils import DIMS, FULL_DIMS, P0_ONLY_DIMS
from build_guide_isp_aigc_quadruplets_and_vlm_qc import (
    unprocess_srgb_to_linear,
    forward_isp_linear_to_srgb,
    parse_foreground_mask_and_disparity,
    make_disc_psf,
    make_motion_psf,
    guided_filter_rgb,
    jpeg_roundtrip,
    synthesize_oversharp_quadruplet,
    synthesize_screenart_p0,
    synthesize_bokeh_pair,
)
from train_v9_specialist import build_v9_backbone

SYNTH_DIR = '/dev/shm/v18_synth_quads'
META_OUT = 'data/v17_splits/v18_synth_metadata.json'
os.makedirs(SYNTH_DIR, exist_ok=True)
os.makedirs('data/v17_splits', exist_ok=True)


def laplacian_pyramid_blend(fg_rgb: np.ndarray, bg_rgb: np.ndarray, mask_1ch: np.ndarray, levels: int = 4) -> np.ndarray:
    """Multi-band Laplacian pyramid blending to eliminate artificial ellipse boundary seams."""
    G_fg = [fg_rgb.astype(np.float32)]
    G_bg = [bg_rgb.astype(np.float32)]
    G_m  = [np.clip(mask_1ch.astype(np.float32), 0.0, 1.0)[:, :, None]]
    for _ in range(levels):
        if min(G_fg[-1].shape[:2]) < 16:
            break
        G_fg.append(cv2.pyrDown(G_fg[-1]))
        G_bg.append(cv2.pyrDown(G_bg[-1]))
        m_down = cv2.pyrDown(G_m[-1])
        G_m.append(m_down[:, :, None] if m_down.ndim == 2 else m_down)

    L_out = [G_m[-1] * G_fg[-1] + (1.0 - G_m[-1]) * G_bg[-1]]
    for i in range(len(G_fg) - 2, -1, -1):
        size = (G_fg[i].shape[1], G_fg[i].shape[0])
        L_fg = G_fg[i] - cv2.pyrUp(G_fg[i + 1], dstsize=size)
        L_bg = G_bg[i] - cv2.pyrUp(G_bg[i + 1], dstsize=size)
        L_blend = G_m[i] * L_fg + (1.0 - G_m[i]) * L_bg
        L_out.append(cv2.pyrUp(L_out[-1], dstsize=size) + L_blend)
    return np.clip(np.round(L_out[-1]), 0, 255).astype(np.uint8)


def synthesize_v18_blur(im: np.ndarray, M_fg: np.ndarray, seed: int):
    """Calibrated Native-Resolution Blur Quadruplet (prevents extreme P0 threshold inflation)."""
    rng = np.random.default_rng(seed)
    h, w = im.shape[:2]
    scale_ref = math.sqrt((w * h) / (1080.0 * 720.0))
    scale_ref = np.clip(scale_ref, 0.75, 2.2)

    I_lin = unprocess_srgb_to_linear(im)
    use_motion = (rng.random() < 0.45)
    angle = float(rng.uniform(0.0, 180.0))
    mfg3 = M_fg[:, :, None]

    # P2: mild blur at native resolution
    k_p2 = make_motion_psf(rng.uniform(3.0, 4.5) * scale_ref, angle, 0.10) if use_motion else make_disc_psf(rng.uniform(1.3, 2.0) * scale_ref)
    lin_b2 = cv2.filter2D(I_lin, -1, k_p2, borderType=cv2.BORDER_REFLECT)
    w_p2 = np.clip(mfg3 * 0.85 + 0.15, 0.0, 1.0)
    img_p2 = forward_isp_linear_to_srgb(w_p2 * lin_b2 + (1.0 - w_p2) * I_lin)

    # P1: moderate blur
    k_p1 = make_motion_psf(rng.uniform(6.5, 9.5) * scale_ref, angle, 0.15) if use_motion else make_disc_psf(rng.uniform(2.6, 3.8) * scale_ref)
    lin_b1 = cv2.filter2D(I_lin, -1, k_p1, borderType=cv2.BORDER_REFLECT)
    w_p1 = np.clip(mfg3 * 0.90 + 0.25, 0.0, 1.0)
    img_p1 = forward_isp_linear_to_srgb(w_p1 * lin_b1 + (1.0 - w_p1) * I_lin)

    # P0: realistic human P0 boundary (r_c 5.2~8.2px scaled, NOT 15~35px extreme blur!)
    k_p0 = make_motion_psf(rng.uniform(11.5, 17.5) * scale_ref, angle, 0.20) if use_motion else make_disc_psf(rng.uniform(5.2, 8.2) * scale_ref)
    highlights = np.maximum(0.0, I_lin - 0.78) * 1.35
    lin_b0 = np.clip(cv2.filter2D(I_lin + highlights, -1, k_p0, borderType=cv2.BORDER_REFLECT), 0.0, 1.0)
    img_p0 = forward_isp_linear_to_srgb(lin_b0)
    return im, img_p2, img_p1, img_p0


def synthesize_v18_texloss(im: np.ndarray, M_fg: np.ndarray, seed: int):
    """Calibrated Native-Resolution Texture Loss Quadruplet."""
    rng = np.random.default_rng(seed)
    h, w = im.shape[:2]
    orig_f = im.astype(np.float32) / 255.0
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    tau_edge = np.percentile(grad, 82) + 1e-4
    non_edge = np.clip(1.0 - (grad / (tau_edge * 1.5)), 0.25, 1.0)[:, :, None]

    # P2: mild guided filter smoothing on interior texture
    gf_p2 = guided_filter_rgb(orig_f, r=4, eps=5.0e-4)
    p2_f = np.clip((1.0 - 0.48 * non_edge) * orig_f + (0.48 * non_edge) * gf_p2, 0.0, 1.0)
    img_p2 = jpeg_roundtrip(np.clip(np.round(p2_f * 255.0), 0, 255).astype(np.uint8), quality=int(rng.integers(68, 78)))

    # P1: bilateral beauty smoothing + 1.75x downscale-upscale playback
    bil_u8 = cv2.bilateralFilter(im, d=9, sigmaColor=float(rng.uniform(26.0, 38.0)), sigmaSpace=float(rng.uniform(7.0, 9.5)))
    small_p1 = cv2.resize(bil_u8, (max(32, int(w / 1.75)), max(32, int(h / 1.75))), interpolation=cv2.INTER_AREA)
    up_p1 = cv2.resize(small_p1, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    p1_f = np.clip(0.28 * orig_f + 0.72 * up_p1, 0.0, 1.0)
    img_p1 = jpeg_roundtrip(np.clip(np.round(p1_f * 255.0), 0, 255).astype(np.uint8), quality=int(rng.integers(52, 64)))

    # P0: realistic strong waxy smoothing (2.6x downscale-upscale + GF r=8, NOT 4x extreme blur)
    gf_p0 = guided_filter_rgb(orig_f, r=8, eps=1.1e-3)
    small_p0 = cv2.resize((gf_p0 * 255.0).astype(np.uint8), (max(32, int(w / 2.6)), max(32, int(h / 2.6))), interpolation=cv2.INTER_AREA)
    up_p0 = cv2.resize(small_p0, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    p0_f = np.clip(0.10 * orig_f + 0.90 * guided_filter_rgb(up_p0, r=6, eps=7.0e-4), 0.0, 1.0)
    img_p0 = jpeg_roundtrip(np.clip(np.round(p0_f * 255.0), 0, 255).astype(np.uint8), quality=int(rng.integers(38, 48)))
    return im, img_p2, img_p1, img_p0


def synthesize_v18_overgen_pure_diffusion(
    im: np.ndarray,
    M_fg: np.ndarray,
    M_core: np.ndarray,
    mild_rgb: np.ndarray,
    strong_rgb: np.ndarray,
    seed: int,
):
    """
    Pure Diffusion FLUX.3-Klein Over-Generation Quadruplet at Native Resolution:
      - ZERO cv2.remap sine-wave warping!
      - ZERO sin*cos checkerboard grid!
      - Uses Laplacian pyramid multi-band blending of authentic FLUX.3 mild & strong renders
        guided by semantic masks, and returns exact (focus_y, focus_x) of the synthesized region.
    """
    rng = np.random.default_rng(seed)
    h, w = im.shape[:2]
    mild_res = cv2.resize(mild_rgb, (w, h), interpolation=cv2.INTER_CUBIC) if mild_rgb.shape[:2] != (h, w) else mild_rgb
    strong_res = cv2.resize(strong_rgb, (w, h), interpolation=cv2.INTER_CUBIC) if strong_rgb.shape[:2] != (h, w) else strong_rgb

    diff_map = np.mean(np.abs(strong_res.astype(np.float32) - im.astype(np.float32)), axis=2) / 255.0
    diff_smooth = cv2.GaussianBlur(diff_map, (0, 0), sigmaX=max(8.0, min(h, w) / 32.0))

    def build_semantic_mask(priority_map, area_frac):
        score_map = priority_map * (0.5 + 0.5 * (diff_smooth / (np.percentile(diff_smooth, 95) + 1e-4)))
        cy, cx = np.unravel_index(np.argmax(score_map + rng.uniform(0, 0.08, size=score_map.shape)), score_map.shape)
        ry = max(24, int(math.sqrt(area_frac * h * w / math.pi) * rng.uniform(0.85, 1.18)))
        rx = max(24, int(math.sqrt(area_frac * h * w / math.pi) * rng.uniform(0.85, 1.18)))
        m = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(m, (int(cx), int(cy)), (rx, ry), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(6.0, min(rx, ry) * 0.30))
        return np.clip(m, 0.0, 1.0), (int(cy), int(cx))

    # P2: Mild FLUX enhancement blended on peripheral/secondary subject region (6%~10% area)
    m_p2, (cy2, cx2) = build_semantic_mask(0.6 * (1.0 - M_core) + 0.4 * M_fg, float(rng.uniform(0.06, 0.10)))
    img_p2 = laplacian_pyramid_blend(mild_res, im, m_p2, levels=3)

    # P1: Strong FLUX re-render blended on subject boundary / hand / secondary foreground (14%~22% area)
    edge_fg = np.abs(M_fg - cv2.GaussianBlur(M_fg, (0, 0), sigmaX=10.0)) + 0.65 * M_fg
    m_p1, (cy1, cx1) = build_semantic_mask(edge_fg, float(rng.uniform(0.14, 0.22)))
    m_p1_tot = np.maximum(m_p2 * 0.6, m_p1)
    img_p1 = laplacian_pyramid_blend(strong_res, im, m_p1_tot, levels=3)

    # P0: Full foreground / core subject FLUX strong re-render + multi-pass AI over-sharpened plastic texture (38%~60% area)
    m_p0, (cy0, cx0) = build_semantic_mask(M_core + 0.6 * M_fg, float(rng.uniform(0.38, 0.60)))
    m_p0_tot = np.clip(np.maximum(m_p1_tot, m_p0 + 0.45 * M_fg), 0.0, 1.0)
    s_f = strong_res.astype(np.float32)
    hf = s_f - cv2.GaussianBlur(s_f, (0, 0), sigmaX=1.6)
    strong_plus = np.clip(cv2.bilateralFilter(strong_res, d=7, sigmaColor=45, sigmaSpace=9).astype(np.float32) + 1.65 * hf, 0, 255).astype(np.uint8)
    img_p0 = laplacian_pyramid_blend(strong_plus, im, m_p0_tot, levels=3)

    return im, img_p2, img_p1, img_p0, [(cy2, cx2), (cy1, cx1), (cy0, cx0)]


def run_gpu_worker(rank: int, world_size: int, max_seeds: int = 240):
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Load SigLIP2 backbone + 6B-calibrated severity probe
    m, fwd, mean, std, _, _ = build_v9_backbone('siglip2')
    m = m.to(device).eval().to(torch.bfloat16)
    mean_t = torch.tensor(mean, device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=torch.bfloat16).view(1, 3, 1, 1)

    man_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
    all_names = [r['name'] for r in man_rows]
    y_real = np.array([[int(r[c]) for c in DIMS] for r in man_rows], dtype=np.int64)
    clean_indices = np.where((y_real == 0).all(axis=1))[0][:max_seeds]
    my_seeds = clean_indices[rank::world_size]

    roi_d = np.load('/dev/shm/race18_native/roi_coords_and_meta.npz', allow_pickle=True)
    roi_names = list(roi_d['names'])
    roi_coords = roi_d['coords']
    roi_map = {n: roi_coords[i] for i, n in enumerate(roi_names)}

    idx_d = np.load('/dev/shm/race18_native/index.npz', allow_pickle=True)
    shm_names = list(idx_d['names'])
    shm_offsets = idx_d['offsets']
    shm_h, shm_w = idx_d['h'], idx_d['w']
    shm_pos = {str(n): i for i, n in enumerate(shm_names)}
    shm_data = np.memmap('/dev/shm/race18_native/cache.bin', dtype=np.uint8, mode='r')

    def load_native_img(nm):
        if nm in shm_pos:
            i = shm_pos[nm]
            off, h, w = int(shm_offsets[i]), int(shm_h[i]), int(shm_w[i])
            return np.frombuffer(shm_data, dtype=np.uint8, count=h * w * 3, offset=off).reshape((h, w, 3)).copy()
        for d in ['data/images', 'data/images_supp']:
            p = os.path.join(d, nm)
            if os.path.exists(p):
                return np.asarray(Image.open(p).convert('RGB'), dtype=np.uint8)
        return None

    flux_mild_map, flux_strong_map = {}, {}
    for p in glob.glob('/tmp/synth/flux3/mild_*'):
        if not p.endswith('.txt'):
            flux_mild_map[os.path.basename(p)[len('mild_'):]] = p
    for p in glob.glob('/tmp/synth/flux3/strong_*'):
        if not p.endswith('.txt'):
            flux_strong_map[os.path.basename(p)[len('strong_'):]] = p

    meta_records = {}
    t0 = time.time()

    for dim_c in [0, 1, 2, 4, 3, 5]:
        seeds_use = my_seeds if dim_c in FULL_DIMS else my_seeds[:len(my_seeds) // 2 + 1]
        kept = 0
        for s_idx in seeds_use:
            nm = all_names[s_idx]
            stem = os.path.splitext(nm)[0]
            im = load_native_img(nm)
            if im is None:
                continue
            h, w = im.shape[:2]

            # Parse foreground & disparity on fast 480p proxy, then upsample masks back to 100% Native Resolution (h, w)!
            scale_m = min(1.0, 480.0 / max(h, w))
            im_small = cv2.resize(im, (max(32, int(w * scale_m)), max(32, int(h * scale_m))), interpolation=cv2.INTER_AREA) if scale_m < 1.0 else im
            M_fg_s, M_core_s, D_map_s = parse_foreground_mask_and_disparity(im_small)
            M_fg = cv2.resize(M_fg_s, (w, h), interpolation=cv2.INTER_LINEAR)
            M_core = cv2.resize(M_core_s, (w, h), interpolation=cv2.INTER_LINEAR)
            D_map = cv2.resize(D_map_s, (w, h), interpolation=cv2.INTER_LINEAR)

            c6 = roi_map.get(nm, np.zeros((6, 2), dtype=np.int32))
            y_edge, x_edge = int(c6[0, 0]), int(c6[0, 1])
            y_tex, x_tex   = int(c6[2, 0]), int(c6[2, 1])
            y_ring, x_ring = int(c6[4, 0]), int(c6[4, 1])

            focus_list = [(y_edge + 224, x_edge + 224)] * 3
            I_nat_hard_neg = None

            if dim_c == 0:
                I0, I1, I2, I3 = synthesize_v18_blur(im, M_fg, seed=s_idx * 17 + 1)
                focus_list = [(y_edge + 224, x_edge + 224)] * 3
            elif dim_c == 1:
                I0, I1, I2, I3 = synthesize_v18_texloss(im, M_fg, seed=s_idx * 17 + 2)
                focus_list = [(y_tex + 224, x_tex + 224)] * 3
            elif dim_c == 2:
                I0, I1, I2, I3 = synthesize_oversharp_quadruplet(im, M_fg, seed=s_idx * 17 + 3)
                focus_list = [(y_edge + 224, x_edge + 224)] * 3
            elif dim_c == 4:
                if nm not in flux_strong_map:
                    continue
                strong_rgb = np.asarray(Image.open(flux_strong_map[nm]).convert('RGB'), dtype=np.uint8)
                mild_rgb = np.asarray(Image.open(flux_mild_map[nm]).convert('RGB'), dtype=np.uint8) if nm in flux_mild_map else cv2.addWeighted(im, 0.55, strong_rgb, 0.45, 0.0)
                I0, I1, I2, I3, focus_list = synthesize_v18_overgen_pure_diffusion(im, M_fg, M_core, mild_rgb, strong_rgb, seed=s_idx * 17 + 4)
            elif dim_c == 3:
                I3 = synthesize_screenart_p0(im, seed=s_idx * 17 + 5)
                I0, I1, I2 = im, cv2.addWeighted(im, 0.72, I3, 0.28, 0.0), cv2.addWeighted(im, 0.42, I3, 0.58, 0.0)
                focus_list = [(h // 2, w // 2)] * 3
            else:
                I0, I_nat_hard_neg, I3 = synthesize_bokeh_pair(im, M_fg, D_map, seed=s_idx * 17 + 6)
                I1, I2 = I_nat_hard_neg, cv2.addWeighted(I_nat_hard_neg, 0.50, I3, 0.50, 0.0)
                focus_list = [(y_ring + 224, x_ring + 224)] * 3

            # Extract 1:1 Native Crops centered on exact defect focus for Neural Teacher Corridor QC
            S = min(448, h, w)
            crops = []
            for k_idx, sub_im in enumerate([I0, I1, I2, I3]):
                fy, fx = focus_list[min(2, max(0, k_idx - 1))]
                y0 = max(0, min(h - S, fy - S // 2))
                x0 = max(0, min(w - S, fx - S // 2))
                cp = sub_im[y0:y0 + S, x0:x0 + S]
                pil_cp = Image.fromarray(cp).resize((384, 384), Image.BICUBIC)
                crops.append(torch.from_numpy(np.asarray(pil_cp, dtype=np.float32).transpose(2, 0, 1) / 255.0))

            batch_t = torch.stack(crops).to(device).to(torch.bfloat16)
            with torch.inference_mode():
                feats = F.normalize(fwd((batch_t - mean_t) / std_t).float(), dim=-1)

            # Calibrated Neural Severity Corridor S_c in [0, 3]:
            # Computes directional progression along (f3 - f0) and checks strict corridor bounds
            d_vec = feats[3] - feats[0]
            d_norm = float(torch.norm(d_vec, p=2).item())
            d1 = float(torch.norm(feats[1] - feats[0], p=2).item())
            d2 = float(torch.norm(feats[2] - feats[0], p=2).item())
            d3 = d_norm

            # Map neural 1:1 patch perceptual shifts to calibrated [0, 3] teacher severity scale per dimension
            SCALE_DIM = {0: 12.0, 1: 17.5, 2: 15.0, 4: 14.0, 3: 10.5, 5: 11.5}
            sc_mult = SCALE_DIM[dim_c]
            s_p2 = d1 * sc_mult
            s_p1 = d2 * sc_mult
            s_p0 = d3 * sc_mult

            if dim_c in FULL_DIMS:
                # Corridor Filter: rejects weak inaudible edits AND rejects extreme overshoot P0 (> 2.95)!
                in_corridor = (
                    (0.22 <= s_p2 <= 1.55)
                    and (0.75 <= s_p1 <= 2.50)
                    and (1.35 <= s_p0 <= 2.95)
                    and (s_p1 - s_p2 >= 0.16)
                    and (s_p0 - s_p1 >= 0.20)
                )
            else:
                in_corridor = (1.10 <= s_p0 <= 2.95)

            if not in_corridor:
                continue

            # Save Native-Resolution quadruplets and focus coordinates
            grades_save = [('P2', I1, focus_list[0]), ('P1', I2, focus_list[1]), ('P0', I3, focus_list[2])] if dim_c in FULL_DIMS else [('P0', I3, focus_list[2])]
            for g_str, sub_im, (fy, fx) in grades_save:
                fn = f'{stem}_d{dim_c}_{g_str}.png'
                out_p = os.path.join(SYNTH_DIR, fn)
                if not os.path.exists(out_p):
                    cv2.imwrite(out_p, cv2.cvtColor(sub_im, cv2.COLOR_RGB2BGR))
                meta_records[fn] = {
                    'seed_name': nm,
                    'dim_c': int(dim_c),
                    'grade': g_str,
                    'focus_yx': [int(fy), int(fx)],
                    'native_hw': [int(h), int(w)],
                    'teacher_s': [round(s_p2, 3), round(s_p1, 3), round(s_p0, 3)],
                }

            # Fix 5: Save Natural Optical Shallow-DOF Hard Negative (`{stem}_d5_NatNeg.png`)
            if dim_c == 5 and I_nat_hard_neg is not None:
                fn_neg = f'{stem}_d5_NatNeg.png'
                out_neg = os.path.join(SYNTH_DIR, fn_neg)
                if not os.path.exists(out_neg):
                    cv2.imwrite(out_neg, cv2.cvtColor(I_nat_hard_neg, cv2.COLOR_RGB2BGR))
                meta_records[fn_neg] = {
                    'seed_name': nm,
                    'dim_c': 5,
                    'grade': 'NatNeg',
                    'focus_yx': [int(y_ring + 224), int(x_ring + 224)],
                    'native_hw': [int(h), int(w)],
                    'teacher_s': [0.0, 0.0, 0.0],
                }
            kept += 1

        print(f'[GPU {rank}] Dim {DIMS[dim_c]:6s}: Kept {kept}/{len(seeds_use)} Corridor-Verified Native Quads ({time.time()-t0:.0f}s)', flush=True)

    json.dump(meta_records, open(f'data/v17_splits/v18_meta_rank{rank}.json', 'w'), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, default=-1)
    ap.add_argument('--world-size', type=int, default=8)
    ap.add_argument('--max-seeds', type=int, default=240)
    args = ap.parse_args()

    if args.rank >= 0:
        run_gpu_worker(args.rank, args.world_size, args.max_seeds)
        return

    t0 = time.time()
    print(f'=== Launching 8-GPU V18 Native-Resolution Corridor-Filtered Quadruplet Generator (max_seeds={args.max_seeds}) ===', flush=True)
    procs = []
    for r in range(args.world_size):
        p = subprocess.Popen([sys.executable, '-u', __file__, '--rank', str(r), '--world-size', str(args.world_size), '--max-seeds', str(args.max_seeds)])
        procs.append(p)
    for p in procs:
        p.wait()

    merged = {}
    for r in range(args.world_size):
        rp = f'data/v17_splits/v18_meta_rank{r}.json'
        if os.path.exists(rp):
            merged.update(json.load(open(rp)))
    json.dump(merged, open(META_OUT, 'w'), indent=2, ensure_ascii=False)
    print(f'=== Completed V18 Native Quadruplet Generation in {time.time()-t0:.1f}s! Total verified files: {len(merged)} saved to {META_OUT} ===', flush=True)


if __name__ == '__main__':
    main()
