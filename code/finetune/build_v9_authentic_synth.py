"""Build V9 Authentic Multi-Defect Synthetic Dataset (Strictly Split-Isolated, Zero Ripple Shortcut).

Key fixes over V6/V7/V8:
1. Completely removes the artificial `cv2.remap` sinusoidal wave ripple that polluted `过度生成` in V6/V7/V8.
2. Generates authentic `过度生成` (`1:P2`, `2:P1`, `3:P0`) via:
   - Real FLUX3-Klein (`strong_*`, `mild_*`) and 2-pass Swin2SR (`p1_*`, `p2_*`) neural regeneration + multi-region feathered compositing,
   - Split-isolated seamless patch grafting from real training-split `过度生成=3 (P0)` and `过度生成=2 (P1)` high-artifact regions onto clean/mild training images,
   - Non-periodic anatomical/structural clone & affine seam distortion (simulating AI extra digits/misaligned eyes/melted text strokes) + heavy bilateral waxy plastic skin/surface smoothing.
3. Generates calibrated `过锐` (`1:P2`, `2:P1`, `3:P0`), `模糊+纹理损失` (`1:P2`, `2:P1`, `3:P0`), and multi-contour `虚化突兀` (`3:P0`) strictly isolated to each split (`s16` train, `s17` train).
"""
import csv
import glob
import os
import time
from multiprocessing import Pool
import cv2
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')

OUT_IMG_DIR = '/tmp/v9_authentic_imgs'
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs('data/v9_ladders', exist_ok=True)

CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']


def find_orig_path(name):
    for d in ['data/images', 'data/images_supp']:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def soft_feathered_mask(h, w, rng, area_frac_range=(0.25, 0.45)):
    mask = np.zeros((h, w), dtype=np.float32)
    target_frac = rng.uniform(*area_frac_range)
    n_blobs = int(rng.integers(1, 3))
    for _ in range(n_blobs):
        cy = int(rng.uniform(0.20, 0.80) * h)
        cx = int(rng.uniform(0.20, 0.80) * w)
        ry = int(np.sqrt(target_frac / n_blobs) * 0.55 * h * rng.uniform(0.8, 1.25))
        rx = int(np.sqrt(target_frac / n_blobs) * 0.55 * w * rng.uniform(0.8, 1.25))
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(mask, (cx, cy), (max(12, rx), max(12, ry)), angle, 0, 360, 1.0, -1)
    k = max(15, int(min(h, w) * 0.08) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), k * 0.35)
    return np.clip(mask, 0.0, 1.0)[:, :, None]


def apply_structural_clone_distortion(img_f, severity_int, rng):
    """Non-periodic anatomical/structural clone & affine seam distortion (ZERO sinusoidal waves!).
    Mimics AI extra fingers, double eyes/mouths, melted text strokes, and local geometric dislocation.
    """
    out = img_f.copy()
    h, w = out.shape[:2]
    n_clones = 3 if severity_int == 3 else (2 if severity_int == 2 else 1)
    for _ in range(n_clones):
        rh = int(rng.integers(max(32, h // 8), max(48, h // 3)))
        rw = int(rng.integers(max(32, w // 8), max(48, w // 3)))
        sy = int(rng.integers(0, max(1, h - rh)))
        sx = int(rng.integers(0, max(1, w - rw)))
        patch = np.clip(out[sy:sy + rh, sx:sx + rw], 0, 255).astype(np.uint8)

        angle = float(rng.uniform(-16, 16) if severity_int == 3 else rng.uniform(-7, 7))
        scale = float(rng.uniform(0.88, 1.15))
        M = cv2.getRotationMatrix2D((rw / 2.0, rh / 2.0), angle, scale)
        warped = cv2.warpAffine(patch, M, (rw, rh), borderMode=cv2.BORDER_REFLECT_101)
        if severity_int >= 2:
            warped = cv2.bilateralFilter(warped, d=9, sigmaColor=75, sigmaSpace=75)
            gb = cv2.GaussianBlur(warped.astype(np.float32), (0, 0), 1.6)
            warped = np.clip(warped.astype(np.float32) * 1.45 - gb * 0.45, 0, 255)
        else:
            warped = warped.astype(np.float32)

        dy = int(rng.integers(-rh // 2, rh // 2 + 1))
        dx = int(rng.integers(-rw // 2, rw // 2 + 1))
        ty = min(max(0, sy + dy), h - rh)
        tx = min(max(0, sx + dx), w - rw)

        m = np.zeros((rh, rw), dtype=np.float32)
        cv2.ellipse(m, (rw // 2, rh // 2), (max(8, int(rw * 0.42)), max(8, int(rh * 0.42))), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (25, 25), 8.0)[:, :, None]
        alpha = 0.90 if severity_int == 3 else (0.72 if severity_int == 2 else 0.45)
        roi = out[ty:ty + rh, tx:tx + rw]
        out[ty:ty + rh, tx:tx + rw] = (1.0 - alpha * m) * roi + (alpha * m) * warped
    return out


def make_authentic_ai_version(img, gen_img_or_none, real_p0_img_or_none, rng):
    """Construct an authentic AI-hallucinated plastic + structural-grafted image aligned to `img`
    WITHOUT any sinusoidal `cv2.remap` wave ripple!
    """
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)

    # Heavy waxy plastic bilateral smoothing + non-linear ridge enhancement
    sm = cv2.bilateralFilter(img, d=9, sigmaColor=65, sigmaSpace=65)
    sm = cv2.bilateralFilter(sm, d=9, sigmaColor=65, sigmaSpace=65).astype(np.float32)
    gb = cv2.GaussianBlur(sm, (0, 0), 1.6)
    hp = sm - gb
    ridge = np.tanh(hp / 14.0) * 28.0
    waxy_base = np.clip(gb + ridge, 0, 255)

    if gen_img_or_none is not None:
        flx_base = cv2.resize(gen_img_or_none, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        ai_base = 0.78 * flx_base + 0.22 * waxy_base
    else:
        ai_base = waxy_base

    # Graft a high-gradient region from a strictly split-isolated real training `过度生成 >= 2` image
    if real_p0_img_or_none is not None:
        ph_h, ph_w = real_p0_img_or_none.shape[:2]
        if ph_h >= 120 and ph_w >= 120:
            # Crop a prominent central/high-activity patch from the real overgen image
            cy0 = int(rng.uniform(0.15, 0.45) * ph_h)
            cx0 = int(rng.uniform(0.15, 0.45) * ph_w)
            ch = int(rng.uniform(0.40, 0.65) * ph_h)
            cw = int(rng.uniform(0.40, 0.65) * ph_w)
            crop = real_p0_img_or_none[cy0:min(ph_h, cy0 + ch), cx0:min(ph_w, cx0 + cw)]
            th = int(rng.uniform(0.42, 0.68) * h)
            tw = int(rng.uniform(0.42, 0.68) * w)
            ty = int(rng.integers(0, max(1, h - th)))
            tx = int(rng.integers(0, max(1, w - tw)))
            crop_rsz = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            roi = ai_base[ty:ty + th, tx:tx + tw]
            # Match luminance mean softly while keeping authentic AI artifact structure
            for c in range(3):
                diff = roi[:, :, c].mean() - crop_rsz[:, :, c].mean()
                crop_rsz[:, :, c] = np.clip(crop_rsz[:, :, c] + 0.45 * diff, 0, 255)
            gm = np.zeros((th, tw), dtype=np.float32)
            cv2.ellipse(gm, (tw // 2, th // 2), (max(12, int(tw * 0.44)), max(12, int(th * 0.44))), 0, 0, 360, 1.0, -1)
            gm = cv2.GaussianBlur(gm, (35, 35), 11.0)[:, :, None]
            ai_base[ty:ty + th, tx:tx + tw] = (1.0 - 0.82 * gm) * roi + (0.82 * gm) * crop_rsz

    return np.clip(ai_base, 0, 255)


def worker_build_overgen_triplet(args):
    """Generate authentic (P2=1, P1=2, P0=3) counterfactual ladder triplet for 过度生成."""
    sp, idx, base_name, gen_path, patch_path = args
    rng = np.random.default_rng(sp * 100000 + idx * 19 + 5)
    p_base = find_orig_path(base_name)
    img = cv2.imread(p_base, cv2.IMREAD_COLOR) if p_base else None
    if img is None:
        return []
    h, w = img.shape[:2]
    gen_img = cv2.imread(gen_path, cv2.IMREAD_COLOR) if (gen_path and os.path.exists(gen_path)) else None
    patch_img = cv2.imread(patch_path, cv2.IMREAD_COLOR) if (patch_path and os.path.exists(patch_path)) else None

    ai_full = make_authentic_ai_version(img, gen_img, patch_img, rng)
    img_f = img.astype(np.float32)

    out_rows = []
    # Grade 1 (P2): localized (25-42% area) + moderate alpha (0.42-0.52), mild texture loss (1)
    m_p2 = soft_feathered_mask(h, w, rng, (0.25, 0.42))
    alpha_p2 = float(rng.uniform(0.42, 0.52))
    im_p2 = np.clip((1.0 - alpha_p2 * m_p2) * img_f + (alpha_p2 * m_p2) * ai_full, 0, 255).astype(np.uint8)
    nm_p2 = f'v9_og_s{sp}_{idx:03d}_P2.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p2), im_p2, [cv2.IMWRITE_JPEG_QUALITY, 94])
    out_rows.append({'name': nm_p2, '模糊': 0, '纹理损失': 1, '过锐': 0, '花屏': 0, '过度生成': 1, '虚化突兀': 0})

    # Grade 2 (P1): larger area (55-78%) + strong alpha (0.70-0.84) + structural clone distortion
    m_p1 = soft_feathered_mask(h, w, rng, (0.55, 0.78))
    alpha_p1 = float(rng.uniform(0.70, 0.84))
    im_p1_f = (1.0 - alpha_p1 * m_p1) * img_f + (alpha_p1 * m_p1) * ai_full
    im_p1_f = apply_structural_clone_distortion(im_p1_f, 2, rng)
    if idx % 2 == 0:
        im_p1_f = 0.78 * im_p1_f + 0.22 * cv2.GaussianBlur(im_p1_f, (0, 0), 1.6)
        blur_g, sharp_g = 1, 0
    else:
        gb = cv2.GaussianBlur(im_p1_f, (0, 0), 1.4)
        im_p1_f = im_p1_f + 0.60 * (im_p1_f - gb)
        blur_g, sharp_g = 0, 1
    im_p1 = np.clip(im_p1_f, 0, 255).astype(np.uint8)
    nm_p1 = f'v9_og_s{sp}_{idx:03d}_P1.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p1), im_p1, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out_rows.append({'name': nm_p1, '模糊': blur_g, '纹理损失': 2, '过锐': sharp_g, '花屏': 0, '过度生成': 2, '虚化突兀': 0})

    # Grade 3 (P0): dominant/full area (90-100%) + severe structural clone & grafted real P0 distortion (ZERO wave ripples!)
    ai_p0 = apply_structural_clone_distortion(ai_full, 3, rng)
    gb0 = cv2.GaussianBlur(ai_p0, (0, 0), 1.7)
    ai_p0 = np.clip(ai_p0 + 0.85 * np.tanh((ai_p0 - gb0) / 11.0) * 20.0, 0, 255)
    alpha_p0 = float(rng.uniform(0.90, 0.99))
    im_p0 = np.clip((1.0 - alpha_p0) * img_f + alpha_p0 * ai_p0, 0, 255).astype(np.uint8)
    nm_p0 = f'v9_og_s{sp}_{idx:03d}_P0.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p0), im_p0, [cv2.IMWRITE_JPEG_QUALITY, 90])
    out_rows.append({'name': nm_p0, '模糊': blur_g, '纹理损失': 2, '过锐': sharp_g, '花屏': 0, '过度生成': 3, '虚化突兀': 0})

    return out_rows


def worker_build_sharp_triplet(args):
    sp, idx, base_name = args
    p_base = find_orig_path(base_name)
    img = cv2.imread(p_base, cv2.IMREAD_COLOR) if p_base else None
    if img is None:
        return []
    out_rows = []

    # P2 (grade=1)
    sm1 = cv2.bilateralFilter(img, d=5, sigmaColor=22, sigmaSpace=22).astype(np.float32)
    gb1 = cv2.GaussianBlur(sm1, (0, 0), 1.3)
    hp1 = np.clip(sm1 - gb1, -22.0, 22.0)
    im_p2 = np.clip(sm1 + 1.18 * hp1, 0, 255).astype(np.uint8)
    nm_p2 = f'v9_sh_s{sp}_{idx:03d}_P2.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p2), im_p2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out_rows.append({'name': nm_p2, '模糊': 0, '纹理损失': 1, '过锐': 1, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    # P1 (grade=2)
    sm2 = cv2.bilateralFilter(img, d=7, sigmaColor=36, sigmaSpace=36).astype(np.float32)
    gb2a = cv2.GaussianBlur(sm2, (0, 0), 1.2)
    gb2b = cv2.GaussianBlur(sm2, (0, 0), 2.6)
    hp2 = 1.70 * np.clip(sm2 - gb2a, -38.0, 38.0) + 0.78 * np.clip(gb2a - gb2b, -30.0, 30.0)
    im_p1 = np.clip(sm2 + hp2, 0, 255).astype(np.uint8)
    nm_p1 = f'v9_sh_s{sp}_{idx:03d}_P1.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p1), im_p1, [cv2.IMWRITE_JPEG_QUALITY, 93])
    out_rows.append({'name': nm_p1, '模糊': 0, '纹理损失': 2, '过锐': 2, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    # P0 (grade=3)
    sm3 = cv2.bilateralFilter(img, d=9, sigmaColor=50, sigmaSpace=50).astype(np.float32)
    gb3a = cv2.GaussianBlur(sm3, (0, 0), 1.1)
    gb3b = cv2.GaussianBlur(sm3, (0, 0), 2.8)
    gb3c = cv2.GaussianBlur(sm3, (0, 0), 5.2)
    hp3 = 2.65 * (sm3 - gb3a) + 1.40 * (gb3a - gb3b) - 0.35 * (gb3b - gb3c)
    im_p0 = np.clip(sm3 + hp3, 0, 255).astype(np.uint8)
    nm_p0 = f'v9_sh_s{sp}_{idx:03d}_P0.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p0), im_p0, [cv2.IMWRITE_JPEG_QUALITY, 91])
    out_rows.append({'name': nm_p0, '模糊': 0, '纹理损失': 2, '过锐': 3, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    return out_rows


def motion_blur_kernel(size, angle):
    k = np.zeros((size, size), dtype=np.float32)
    k[(size - 1) // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (size, size))
    return k / max(1e-6, np.sum(k))


def worker_build_blur_tex_triplet(args):
    sp, idx, base_name = args
    rng = np.random.default_rng(sp * 200000 + idx * 31 + 7)
    p_base = find_orig_path(base_name)
    img = cv2.imread(p_base, cv2.IMREAD_COLOR) if p_base else None
    if img is None:
        return []
    h, w = img.shape[:2]
    angle = float(rng.uniform(0, 180))
    out_rows = []

    # P2 (grade=1)
    b_p2 = cv2.GaussianBlur(img, (0, 0), 1.45)
    b_p2 = cv2.bilateralFilter(b_p2, d=5, sigmaColor=24, sigmaSpace=24)
    nm_p2 = f'v9_bt_s{sp}_{idx:03d}_P2.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p2), b_p2, [cv2.IMWRITE_JPEG_QUALITY, 86])
    out_rows.append({'name': nm_p2, '模糊': 1, '纹理损失': 1, '过锐': 0, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    # P1 (grade=2)
    k_p1 = motion_blur_kernel(9, angle)
    b_p1 = cv2.filter2D(img, -1, k_p1)
    b_p1 = cv2.GaussianBlur(b_p1, (0, 0), 2.2)
    b_p1 = cv2.resize(cv2.resize(b_p1, (max(16, w // 2), max(16, h // 2)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    nm_p1 = f'v9_bt_s{sp}_{idx:03d}_P1.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p1), b_p1, [cv2.IMWRITE_JPEG_QUALITY, 72])
    out_rows.append({'name': nm_p1, '模糊': 2, '纹理损失': 2, '过锐': 0, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    # P0 (grade=3)
    k_p0 = motion_blur_kernel(19, angle)
    b_p0 = cv2.filter2D(img, -1, k_p0)
    b_p0 = cv2.GaussianBlur(b_p0, (0, 0), 4.5)
    b_p0 = cv2.resize(cv2.resize(b_p0, (max(16, w // 4), max(16, h // 4)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    nm_p0 = f'v9_bt_s{sp}_{idx:03d}_P0.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p0), b_p0, [cv2.IMWRITE_JPEG_QUALITY, 55])
    out_rows.append({'name': nm_p0, '模糊': 3, '纹理损失': 3, '过锐': 0, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    return out_rows


def worker_build_bokeh_cutout(args):
    sp, idx, base_name = args
    rng = np.random.default_rng(sp * 300000 + idx * 43 + 3)
    p_base = find_orig_path(base_name)
    img = cv2.imread(p_base, cv2.IMREAD_COLOR) if p_base else None
    if img is None:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    cx = int(rng.uniform(0.35, 0.65) * w)
    cy = int(rng.uniform(0.38, 0.68) * h)
    rx = int(rng.uniform(0.18, 0.32) * w)
    ry = int(rng.uniform(0.24, 0.40) * h)
    cv2.ellipse(mask, (cx, cy + int(ry * 0.35)), (rx, ry), 0, 0, 360, 1.0, -1)
    cv2.circle(mask, (cx, max(10, cy - int(ry * 0.55))), int(rx * 0.58), 1.0, -1)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    high_grad = (grad > np.percentile(grad, 75)).astype(np.float32)
    mask = np.clip(mask + 0.35 * high_grad * cv2.dilate(mask, np.ones((15, 15), np.uint8)), 0.0, 1.0)
    mask = (mask > 0.5).astype(np.float32)
    mask_2d = cv2.GaussianBlur(mask, (5, 5), 1.2)
    mask_3d = mask_2d[:, :, None]

    sigma_bg = float(rng.uniform(7.5, 14.0))
    bg_blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma_bg)
    edge_ring = np.abs(mask_2d - cv2.GaussianBlur(mask_2d, (15, 15), 5.0))[:, :, None]
    comp = mask_3d * img.astype(np.float32) + (1.0 - mask_3d) * bg_blur + edge_ring * 18.0
    im_out = np.clip(comp, 0, 255).astype(np.uint8)
    nm = f'v9_bk_s{sp}_{idx:03d}_P0.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm), im_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return [{'name': nm, '模糊': 1, '纹理损失': 1, '过锐': 0, '花屏': 0, '过度生成': 0, '虚化突兀': 3}]


def build_for_split(sp):
    t0 = time.time()
    split_csv = f'data/resplits/split_s{sp}.csv'
    with open(split_csv) as f:
        sp_rows = list(csv.DictReader(f))
    tr_names = {r['name'] for r in sp_rows if r['split'] == 'train'}

    with open('data/manifest_combined.csv') as f:
        man_rows = list(csv.DictReader(f))
    man_map = {r['name']: r for r in man_rows}

    tr_clean = [n for n in sorted(tr_names) if all(int(man_map[n][c]) == 0 for c in CLASSES)]
    tr_mild = [n for n in sorted(tr_names) if int(man_map[n]['过度生成']) == 0 and int(man_map[n]['花屏']) == 0 and int(man_map[n]['虚化突兀']) == 0]
    tr_og_pos = [n for n in sorted(tr_names) if int(man_map[n]['过度生成']) >= 2]

    gen_map = {}
    for p in sorted(glob.glob('/tmp/synth/flux3/*.png') + glob.glob('/tmp/synth/flux3/*.jpg') + glob.glob('/tmp/synth/out/*.png') + glob.glob('/tmp/synth/out/*.jpg')):
        bn = os.path.basename(p)
        for tn in tr_names:
            stem = os.path.splitext(tn)[0]
            if stem in bn:
                gen_map[tn] = p
                break

    og_candidates = list(gen_map.keys()) + [n for n in tr_clean + tr_mild if n not in gen_map]
    og_bases = og_candidates[:180]
    sh_bases = (tr_clean + tr_mild)[:140]
    bt_bases = tr_clean[50:190] if len(tr_clean) >= 190 else (tr_clean + tr_mild)[140:280]
    bk_bases = (tr_clean + tr_mild)[100:340]

    og_args = [
        (sp, i, bn, gen_map.get(bn, None), find_orig_path(tr_og_pos[i % len(tr_og_pos)]) if tr_og_pos else None)
        for i, bn in enumerate(og_bases)
    ]
    sh_args = [(sp, i, bn) for i, bn in enumerate(sh_bases)]
    bt_args = [(sp, i, bn) for i, bn in enumerate(bt_bases)]
    bk_args = [(sp, i, bn) for i, bn in enumerate(bk_bases)]

    with Pool(24) as pool:
        og_syn = [r for sub in pool.map(worker_build_overgen_triplet, og_args) for r in sub]
        sh_syn = [r for sub in pool.map(worker_build_sharp_triplet, sh_args) for r in sub]
        bt_syn = [r for sub in pool.map(worker_build_blur_tex_triplet, bt_args) for r in sub]
        bk_syn = [r for sub in pool.map(worker_build_bokeh_cutout, bk_args) for r in sub]

    all_syn = og_syn + sh_syn + bt_syn + bk_syn
    man_out = f'data/v9_ladders/manifest_s{sp}_v9.csv'
    with open(man_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name'] + CLASSES)
        w.writeheader()
        for r in man_rows:
            w.writerow({k: r[k] for k in ['name'] + CLASSES})
        for r in all_syn:
            w.writerow(r)

    suites = {
        'overgen':     og_syn,
        'sharp':       sh_syn,
        'blurtex':     bt_syn,
        'bokeh':       bk_syn,
        'overgen_bk':  og_syn + bk_syn,
        'sharptex':    sh_syn + bt_syn,
        'all':         all_syn,
    }
    for sname, srows in suites.items():
        sp_out = f'data/v9_ladders/split_s{sp}_{sname}.csv'
        with open(sp_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'split'])
            w.writeheader()
            for r in sp_rows:
                w.writerow({'name': r['name'], 'split': r['split']})
            for r in srows:
                w.writerow({'name': r['name'], 'split': 'train'})

    print(f"[{time.time()-t0:.1f}s] Built s{sp} V9 Authentic Ladders: overgen={len(og_syn)}, sharp={len(sh_syn)}, blurtex={len(bt_syn)}, bokeh={len(bk_syn)} (total +{len(all_syn)} synth rows)", flush=True)


if __name__ == '__main__':
    for sp in (16, 17):
        build_for_split(sp)
