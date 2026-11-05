"""Build V8 Counterfactual Severity Ladder Synthetic Datasets for s16 & s17 (Zero-Leakage):
Generates 4 Dedicated Single-Feature Training Suites per split (strictly from the 2,216 train images):
1. Expert A (`过度生成` Ladder): ~510 images per split (170 base images x [P2, P1, P0] severity ladder triplets)
2. Expert B (`过锐` Ladder):     ~420 images per split (140 base images x [P2, P1, P0] USM halo ladder triplets)
3. Expert C (`模糊+纹理损失` Ladder): ~420 images per split (140 base images x [P2, P1, P0] blur/denoise ladder triplets)
4. Expert D (`虚化突兀` Cutout): ~240 images per split (240 abrupt bokeh cutout P0 images)

Total: ~1,590 zero-leakage counterfactual ladder images per split (~3,180 images total),
generated in parallel via 24 CPU workers in ~20 seconds.
"""
import csv
import glob
import os
import time
from multiprocessing import Pool
import cv2
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')

OUT_IMG_DIR = '/tmp/v8_ladder_imgs'
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs('data/v8_ladders', exist_ok=True)

CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']


def find_orig_path(name):
    for d in ['data/images', 'data/images_supp']:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def soft_feathered_mask(h, w, rng, area_frac_range=(0.25, 0.45)):
    """Create a smooth elliptical/multi-blob spatial mask covering area_frac of the image."""
    mask = np.zeros((h, w), dtype=np.float32)
    target_frac = rng.uniform(*area_frac_range)
    n_blobs = rng.integers(1, 3)
    for _ in range(n_blobs):
        cy = int(rng.uniform(0.22, 0.78) * h)
        cx = int(rng.uniform(0.22, 0.78) * w)
        ry = int(np.sqrt(target_frac / n_blobs) * 0.55 * h * rng.uniform(0.8, 1.25))
        rx = int(np.sqrt(target_frac / n_blobs) * 0.55 * w * rng.uniform(0.8, 1.25))
        angle = float(rng.uniform(0, 180))
        cv2.ellipse(mask, (cx, cy), (max(12, rx), max(12, ry)), angle, 0, 360, 1.0, -1)
    k = max(15, int(min(h, w) * 0.08) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), k * 0.35)
    return np.clip(mask, 0.0, 1.0)[:, :, None]


def make_ai_hallucinated_version(img, gen_img_or_none, patch_img_or_none, rng):
    """Construct a strong AI-rendered plastic + high-freq ridge image aligned to `img`."""
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)

    if gen_img_or_none is not None:
        ai_base = cv2.resize(gen_img_or_none, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    else:
        # Construct synthetic AI-plastic base via heavy bilateral cartooning + non-linear ridge sharpening
        sm = cv2.bilateralFilter(img, d=9, sigmaColor=55, sigmaSpace=55)
        sm = cv2.bilateralFilter(sm, d=9, sigmaColor=55, sigmaSpace=55).astype(np.float32)
        gb = cv2.GaussianBlur(sm, (0, 0), 1.6)
        hp = sm - gb
        # Non-linear ridge compression (creates synthetic "drawn" plastic ridges without natural grain)
        ridge = np.tanh(hp / 14.0) * 28.0
        ai_base = np.clip(gb + ridge, 0, 255)

    # Inject localized distorted patch or frequency warping for severe structural hallucination
    if patch_img_or_none is not None:
        ph_img = cv2.resize(patch_img_or_none, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # Match LAB color statistics of ph_img to img_f
        ai_mix = 0.65 * ai_base + 0.35 * ph_img
    else:
        ai_mix = ai_base

    # Add sinusoidal micro-warping on fine edges (characteristic of AI finger/eye/text geometry warp)
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    freq_x = rng.uniform(0.04, 0.09)
    freq_y = rng.uniform(0.04, 0.09)
    amp = rng.uniform(1.8, 3.8)
    map_x = (xx + amp * np.sin(yy * freq_y)).astype(np.float32)
    map_y = (yy + amp * np.cos(xx * freq_x)).astype(np.float32)
    ai_warped = cv2.remap(ai_mix, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return ai_warped


def worker_build_overgen_triplet(args):
    """Generate (P2, P1, P0) counterfactual ladder triplet for 过度生成 from a single base training image."""
    sp, idx, base_name, gen_path, patch_path = args
    rng = np.random.default_rng(sp * 100000 + idx * 17 + 1)
    p_base = find_orig_path(base_name)
    img = cv2.imread(p_base, cv2.IMREAD_COLOR) if p_base else None
    if img is None:
        return []
    h, w = img.shape[:2]
    gen_img = cv2.imread(gen_path, cv2.IMREAD_COLOR) if (gen_path and os.path.exists(gen_path)) else None
    patch_img = cv2.imread(patch_path, cv2.IMREAD_COLOR) if (patch_path and os.path.exists(patch_path)) else None

    ai_full = make_ai_hallucinated_version(img, gen_img, patch_img, rng)
    img_f = img.astype(np.float32)

    out_rows = []
    # Grade 1 (P2): localized (25-40% area) + moderate alpha (0.38-0.48), mild texture loss (1)
    m_p2 = soft_feathered_mask(h, w, rng, (0.25, 0.42))
    alpha_p2 = rng.uniform(0.38, 0.48)
    im_p2 = np.clip((1.0 - alpha_p2 * m_p2) * img_f + (alpha_p2 * m_p2) * ai_full, 0, 255).astype(np.uint8)
    nm_p2 = f'v8_og_s{sp}_{idx:03d}_P2.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p2), im_p2, [cv2.IMWRITE_JPEG_QUALITY, 94])
    out_rows.append({'name': nm_p2, '模糊': 0, '纹理损失': 1, '过锐': 0, '花屏': 0, '过度生成': 1, '虚化突兀': 0})

    # Grade 2 (P1): larger area (55-75%) + strong alpha (0.65-0.78) + realistic co-occurrence with 纹理损失(1-2) and 模糊/过锐(1)
    m_p1 = soft_feathered_mask(h, w, rng, (0.55, 0.78))
    alpha_p1 = rng.uniform(0.65, 0.78)
    im_p1_f = (1.0 - alpha_p1 * m_p1) * img_f + (alpha_p1 * m_p1) * ai_full
    if idx % 2 == 0:
        # Co-occur with mild blur (Sub1)
        im_p1_f = 0.75 * im_p1_f + 0.25 * cv2.GaussianBlur(im_p1_f, (0, 0), 1.8)
        blur_g, sharp_g = 1, 0
    else:
        # Co-occur with mild USM overshoot (Sub2)
        gb = cv2.GaussianBlur(im_p1_f, (0, 0), 1.5)
        im_p1_f = im_p1_f + 0.65 * (im_p1_f - gb)
        blur_g, sharp_g = 0, 1
    im_p1 = np.clip(im_p1_f, 0, 255).astype(np.uint8)
    nm_p1 = f'v8_og_s{sp}_{idx:03d}_P1.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p1), im_p1, [cv2.IMWRITE_JPEG_QUALITY, 92])
    out_rows.append({'name': nm_p1, '模糊': blur_g, '纹理损失': 2, '过锐': sharp_g, '花屏': 0, '过度生成': 2, '虚化突兀': 0})

    # Grade 3 (P0): dominant/full area (85-100%) + severe alpha (0.88-1.0) + extra high-freq plastic distortion
    gb0 = cv2.GaussianBlur(ai_full, (0, 0), 1.8)
    ai_p0 = np.clip(ai_full + 0.95 * np.tanh((ai_full - gb0) / 10.0) * 22.0, 0, 255)
    alpha_p0 = rng.uniform(0.88, 0.98)
    im_p0 = np.clip((1.0 - alpha_p0) * img_f + alpha_p0 * ai_p0, 0, 255).astype(np.uint8)
    nm_p0 = f'v8_og_s{sp}_{idx:03d}_P0.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p0), im_p0, [cv2.IMWRITE_JPEG_QUALITY, 90])
    out_rows.append({'name': nm_p0, '模糊': blur_g, '纹理损失': 2, '过锐': sharp_g, '花屏': 0, '过度生成': 3, '虚化突兀': 0})

    return out_rows


def worker_build_sharp_triplet(args):
    """Generate (P2, P1, P0) counterfactual ladder triplet for 过锐 from a single base training image."""
    sp, idx, base_name = args
    p_base = find_orig_path(base_name)
    img = cv2.imread(p_base, cv2.IMREAD_COLOR) if p_base else None
    if img is None:
        return []
    img_f = img.astype(np.float32)
    out_rows = []

    # P2: Mild USM (overshoot ~1.15x, light bilateral denoise -> 纹理损失=1)
    sm1 = cv2.bilateralFilter(img, d=5, sigmaColor=22, sigmaSpace=22).astype(np.float32)
    gb1 = cv2.GaussianBlur(sm1, (0, 0), 1.3)
    hp1 = np.clip(sm1 - gb1, -22.0, 22.0)
    im_p2 = np.clip(sm1 + 1.15 * hp1, 0, 255).astype(np.uint8)
    nm_p2 = f'v8_sh_s{sp}_{idx:03d}_P2.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p2), im_p2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    out_rows.append({'name': nm_p2, '模糊': 0, '纹理损失': 1, '过锐': 1, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    # P1: Medium multi-scale USM (overshoot ~2.0x + visible 1-2px edge halo -> 纹理损失=1 or 2)
    sm2 = cv2.bilateralFilter(img, d=7, sigmaColor=35, sigmaSpace=35).astype(np.float32)
    gb2a = cv2.GaussianBlur(sm2, (0, 0), 1.2)
    gb2b = cv2.GaussianBlur(sm2, (0, 0), 2.6)
    hp2 = 1.65 * np.clip(sm2 - gb2a, -38.0, 38.0) + 0.75 * np.clip(gb2a - gb2b, -30.0, 30.0)
    im_p1 = np.clip(sm2 + hp2, 0, 255).astype(np.uint8)
    nm_p1 = f'v8_sh_s{sp}_{idx:03d}_P1.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p1), im_p1, [cv2.IMWRITE_JPEG_QUALITY, 93])
    out_rows.append({'name': nm_p1, '模糊': 0, '纹理损失': 2, '过锐': 2, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    # P0: Severe multi-radius USM + crunchy double halo ringing (overshoot ~3.2x -> 过锐=3, 纹理损失=2)
    sm3 = cv2.bilateralFilter(img, d=9, sigmaColor=48, sigmaSpace=48).astype(np.float32)
    gb3a = cv2.GaussianBlur(sm3, (0, 0), 1.1)
    gb3b = cv2.GaussianBlur(sm3, (0, 0), 2.8)
    gb3c = cv2.GaussianBlur(sm3, (0, 0), 5.2)
    hp3 = 2.60 * (sm3 - gb3a) + 1.35 * (gb3a - gb3b) - 0.35 * (gb3b - gb3c)
    im_p0 = np.clip(sm3 + hp3, 0, 255).astype(np.uint8)
    nm_p0 = f'v8_sh_s{sp}_{idx:03d}_P0.jpg'
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
    """Generate (P2, P1, P0) counterfactual ladder triplet for 模糊 & 纹理损失 from a single clean image."""
    sp, idx, base_name = args
    rng = np.random.default_rng(sp * 200000 + idx * 31 + 7)
    p_base = find_orig_path(base_name)
    img = cv2.imread(p_base, cv2.IMREAD_COLOR) if p_base else None
    if img is None:
        return []
    h, w = img.shape[:2]
    angle = float(rng.uniform(0, 180))
    out_rows = []

    # P2: Mild defocus/motion blur + mild texture smoothing
    b_p2 = cv2.GaussianBlur(img, (0, 0), 1.45)
    b_p2 = cv2.bilateralFilter(b_p2, d=5, sigmaColor=24, sigmaSpace=24)
    nm_p2 = f'v8_bt_s{sp}_{idx:03d}_P2.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p2), b_p2, [cv2.IMWRITE_JPEG_QUALITY, 86])
    out_rows.append({'name': nm_p2, '模糊': 1, '纹理损失': 1, '过锐': 0, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    # P1: Medium motion/defocus blur + 0.5x downsample-upsample texture loss
    k_p1 = motion_blur_kernel(9, angle)
    b_p1 = cv2.filter2D(img, -1, k_p1)
    b_p1 = cv2.GaussianBlur(b_p1, (0, 0), 2.2)
    b_p1 = cv2.resize(cv2.resize(b_p1, (max(16, w // 2), max(16, h // 2)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    nm_p1 = f'v8_bt_s{sp}_{idx:03d}_P1.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p1), b_p1, [cv2.IMWRITE_JPEG_QUALITY, 72])
    out_rows.append({'name': nm_p1, '模糊': 2, '纹理损失': 2, '过锐': 0, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    # P0: Severe defocus + motion blur + 0.25x downsample-upsample heavy texture wipe
    k_p0 = motion_blur_kernel(19, angle)
    b_p0 = cv2.filter2D(img, -1, k_p0)
    b_p0 = cv2.GaussianBlur(b_p0, (0, 0), 4.5)
    b_p0 = cv2.resize(cv2.resize(b_p0, (max(16, w // 4), max(16, h // 4)), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
    nm_p0 = f'v8_bt_s{sp}_{idx:03d}_P0.jpg'
    cv2.imwrite(os.path.join(OUT_IMG_DIR, nm_p0), b_p0, [cv2.IMWRITE_JPEG_QUALITY, 55])
    out_rows.append({'name': nm_p0, '模糊': 3, '纹理损失': 3, '过锐': 0, '花屏': 0, '过度生成': 0, '虚化突兀': 0})

    return out_rows


def worker_build_bokeh_cutout(args):
    """Generate abrupt portrait/object bokeh cutout (虚化突兀 = P0) with realistic texture co-occurrence."""
    sp, idx, base_name = args
    rng = np.random.default_rng(sp * 300000 + idx * 43 + 3)
    p_base = find_orig_path(base_name)
    img = cv2.imread(p_base, cv2.IMREAD_COLOR) if p_base else None
    if img is None:
        return []
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Build a realistic central subject silhouette with a sharp, slightly misaligned step cutout boundary
    mask = np.zeros((h, w), dtype=np.float32)
    cx = int(rng.uniform(0.35, 0.65) * w)
    cy = int(rng.uniform(0.38, 0.68) * h)
    rx = int(rng.uniform(0.18, 0.32) * w)
    ry = int(rng.uniform(0.24, 0.40) * h)
    # Head + torso silhouette
    cv2.ellipse(mask, (cx, cy + int(ry * 0.35)), (rx, ry), 0, 0, 360, 1.0, -1)
    cv2.circle(mask, (cx, max(10, cy - int(ry * 0.55))), int(rx * 0.58), 1.0, -1)

    # Perturb mask boundary using image gradients so it snaps partially to real edges but leaves abrupt gaps
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    high_grad = (grad > np.percentile(grad, 75)).astype(np.float32)
    mask = np.clip(mask + 0.35 * high_grad * cv2.dilate(mask, np.ones((15, 15), np.uint8)), 0.0, 1.0)
    mask = (mask > 0.5).astype(np.float32)
    # Very narrow 3-pixel transition (abrupt artificial bokeh step!)
    mask_2d = cv2.GaussianBlur(mask, (5, 5), 1.2)
    mask_3d = mask_2d[:, :, None]

    # Heavy background blur (faux aperture blur)
    sigma_bg = float(rng.uniform(7.5, 14.0))
    bg_blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma_bg)

    # Create slight boundary halo / background bleed on the cutout edge
    edge_ring = np.abs(mask_2d - cv2.GaussianBlur(mask_2d, (15, 15), 5.0))[:, :, None]
    comp = mask_3d * img.astype(np.float32) + (1.0 - mask_3d) * bg_blur + edge_ring * 18.0
    im_out = np.clip(comp, 0, 255).astype(np.uint8)
    nm = f'v8_bk_s{sp}_{idx:03d}_P0.jpg'
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

    # Strictly filter to train split images only (100% zero leakage!)
    tr_clean = [n for n in sorted(tr_names) if all(int(man_map[n][c]) == 0 for c in CLASSES)]
    tr_mild = [n for n in sorted(tr_names) if int(man_map[n]['过度生成']) == 0 and int(man_map[n]['花屏']) == 0 and int(man_map[n]['虚化突兀']) == 0]
    tr_og_pos = [n for n in sorted(tr_names) if int(man_map[n]['过度生成']) >= 2]

    # Available zero-leakage FLUX3 / Swin2SR files derived from train_names
    gen_map = {}
    for p in sorted(glob.glob('/tmp/synth/flux3/*.png') + glob.glob('/tmp/synth/flux3/*.jpg') + glob.glob('/tmp/synth/out/*.png') + glob.glob('/tmp/synth/out/*.jpg')):
        bn = os.path.basename(p)
        for tn in tr_names:
            stem = os.path.splitext(tn)[0]
            if stem in bn:
                gen_map[tn] = p
                break

    og_candidates = list(gen_map.keys()) + [n for n in tr_clean + tr_mild if n not in gen_map]
    og_bases = og_candidates[:170]
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

    # Save 4 Dedicated Expert Splits & a Combined Manifest
    all_syn = og_syn + sh_syn + bt_syn + bk_syn
    man_out = f'data/v8_ladders/manifest_s{sp}_v8.csv'
    with open(man_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name'] + CLASSES)
        w.writeheader()
        for r in man_rows:
            w.writerow({k: r[k] for k in ['name'] + CLASSES})
        for r in all_syn:
            w.writerow(r)

    suites = {
        'overgen': og_syn,
        'sharp':   sh_syn,
        'blurtex': bt_syn,
        'bokeh':   bk_syn,
        'all':     all_syn,
    }
    for sname, srows in suites.items():
        sp_out = f'data/v8_ladders/split_s{sp}_{sname}.csv'
        with open(sp_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'split'])
            w.writeheader()
            for r in sp_rows:
                w.writerow({'name': r['name'], 'split': r['split']})
            for r in srows:
                w.writerow({'name': r['name'], 'split': 'train'})

    print(f"[{time.time()-t0:.1f}s] Built s{sp} V8 Ladders: overgen={len(og_syn)}, sharp={len(sh_syn)}, blurtex={len(bt_syn)}, bokeh={len(bk_syn)} (total +{len(all_syn)} synth rows)", flush=True)


if __name__ == '__main__':
    for sp in (16, 17):
        build_for_split(sp)
