"""Build V6 Multi-Defect Synthetic Dataset (strictly zero-leakage per split).

Combines:
1. Dedicated generative/SR model re-renderings (FLUX3-Klein + Swin2SR) blended with
   original clean training images at controlled regional/alpha ratios for 过度生成 (P2/P1/P0).
2. ISP Bilateral-Denoise + Randomized LAB Multi-Scale USM & Ringing for 过锐 (P1/P0).
3. Abrupt Unfeathered Cutout + Disk Defocus Background for 虚化突兀 (P0).
4. Directional Motion Blur + Defocus with Co-occurring Texture Loss for 模糊 (P2/P1) & 纹理损失 (P1).

All synthetic labels strictly follow empirical ground-truth co-occurrence priors.
"""
import argparse
import csv
import glob
import os
import random
import cv2
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']


def find_orig_path(name):
    for d in ['data/images', 'data/images_supp']:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def save_with_realistic_compression(img_bgr, out_path, rng):
    """Apply mild randomized JPEG round-trip (Q=82..94) to eliminate compression shortcuts."""
    q = int(rng.randint(82, 95))
    ok, enc = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        img_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if out_path.lower().endswith('.jpg') or out_path.lower().endswith('.jpeg'):
        cv2.imwrite(out_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    else:
        cv2.imwrite(out_path, img_bgr)


def make_high_detail_mask(img_bgr, area_frac, feather_ratio, rng):
    """Create a soft mask targeting 1-2 high-gradient regions plus a random offset."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sigmaX=max(8, min(h, w) // 25))

    mask = np.zeros((h, w), dtype=np.float32)
    # Pick top energy center with slight random jitter
    flat_idx = int(np.argmax(energy))
    cy, cx = divmod(flat_idx, w)
    cy = int(np.clip(cy + rng.randint(-h // 8, h // 8 + 1), h // 6, 5 * h // 6))
    cx = int(np.clip(cx + rng.randint(-w // 8, w // 8 + 1), w // 6, 5 * w // 6))

    rx = int(w * np.sqrt(area_frac) * rng.uniform(0.48, 0.65))
    ry = int(h * np.sqrt(area_frac) * rng.uniform(0.48, 0.65))
    cv2.ellipse(mask, (cx, cy), (max(16, rx), max(16, ry)), rng.uniform(0, 180), 0, 360, 1.0, -1)

    if feather_ratio > 0:
        sigma = max(3.0, min(h, w) * feather_ratio)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(mask, 0.0, 1.0)[..., None]


def synth_overgen_from_pair(orig_bgr, gen_bgr, grade, rng):
    """Blend original clean image with FLUX3 / Swin2SR re-rendered image."""
    h, w = orig_bgr.shape[:2]
    if gen_bgr.shape[:2] != (h, w):
        gen_bgr = cv2.resize(gen_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    orig_f = orig_bgr.astype(np.float32)
    gen_f = gen_bgr.astype(np.float32)

    if grade == 'P2':
        # Localized 25-35% high-detail region + 0.30 global blend
        mask = make_high_detail_mask(orig_bgr, area_frac=rng.uniform(0.25, 0.38), feather_ratio=0.06, rng=rng)
        w_map = np.clip(0.30 + 0.55 * mask, 0.0, 0.85)
        out = orig_f * (1.0 - w_map) + gen_f * w_map
        labels = {'模糊': 0, '纹理损失': 1, '过锐': 0, '花屏': 0, '过度生成': 1, '虚化突兀': 0}
    elif grade == 'P1':
        # Larger 50-70% region + 0.55 global blend
        mask = make_high_detail_mask(orig_bgr, area_frac=rng.uniform(0.50, 0.70), feather_ratio=0.05, rng=rng)
        w_map = np.clip(0.55 + 0.40 * mask, 0.0, 0.95)
        out = orig_f * (1.0 - w_map) + gen_f * w_map
        labels = {'模糊': 0, '纹理损失': 2, '过锐': 0, '花屏': 0, '过度生成': 2, '虚化突兀': 0}
    else:  # P0: Full generative re-render + mild localized elastic micro-warp + plastic smoothing
        out = gen_f.copy()
        # Add subtle localized micro-warping in high-detail region to simulate severe AI structural distortion
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
        amp = rng.uniform(2.2, 4.0)
        freq = rng.uniform(28.0, 50.0)
        mask = make_high_detail_mask(orig_bgr, area_frac=rng.uniform(0.30, 0.50), feather_ratio=0.05, rng=rng)[..., 0]
        dx = (amp * np.sin(2.0 * np.pi * yy / freq) * mask).astype(np.float32)
        dy = (amp * np.cos(2.0 * np.pi * xx / freq) * mask).astype(np.float32)
        warped = cv2.remap(out, xx + dx, yy + dy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        out = 0.35 * out + 0.65 * warped
        labels = {'模糊': 1, '纹理损失': 2, '过锐': 0, '花屏': 0, '过度生成': 3, '虚化突兀': 0}

    return np.clip(out, 0, 255).astype(np.uint8), labels


def synth_isp_oversharpen_v6(orig_bgr, grade, rng, orig_tex=0):
    """ISP bilateral micro-denoise + LAB L-channel multi-scale USM & Laplacian ringing."""
    # Step 1: ISP pre-denoise (creates realistic micro-texture smoothing that co-occurs in 95%+ of real 过锐)
    if orig_tex == 0:
        d_work = cv2.bilateralFilter(orig_bgr, d=7, sigmaColor=float(rng.uniform(18, 30)), sigmaSpace=float(rng.uniform(18, 30)))
        tex_label = 1 if grade == 'P1' and rng.rand() < 0.55 else 2
    else:
        d_work = orig_bgr.copy()
        tex_label = max(orig_tex, 1 if grade == 'P1' else 2)

    lab = cv2.cvtColor(d_work, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_f = l_chan.astype(np.float32)

    if grade == 'P1':
        s1 = rng.uniform(1.0, 1.6)
        s2 = rng.uniform(2.0, 3.0)
        amt1 = rng.uniform(2.2, 3.2)
        amt2 = rng.uniform(1.0, 1.8)
        clip_lim = rng.uniform(32.0, 46.0)
        sharp_label = 2
    else:  # P0
        s1 = rng.uniform(1.3, 2.0)
        s2 = rng.uniform(2.6, 4.0)
        amt1 = rng.uniform(3.8, 5.2)
        amt2 = rng.uniform(2.0, 3.0)
        clip_lim = rng.uniform(55.0, 78.0)
        sharp_label = 3

    b1 = cv2.GaussianBlur(l_f, (0, 0), sigmaX=s1, sigmaY=s1)
    b2 = cv2.GaussianBlur(l_f, (0, 0), sigmaX=s2, sigmaY=s2)
    d1 = l_f - b1
    d2 = b1 - b2

    # Soft core coring threshold so flat sky noise doesn't explode, focusing halos on edges
    thr = rng.uniform(2.0, 5.0)
    m1 = np.clip((np.abs(d1) - thr) / 5.0, 0.0, 1.0)
    delta = m1 * (amt1 * d1 + amt2 * d2)
    delta = np.clip(delta, -clip_lim * 0.85, clip_lim)

    l_new = np.clip(l_f + delta, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([l_new, a_chan, b_chan]), cv2.COLOR_LAB2BGR)
    labels = {'模糊': 0, '纹理损失': tex_label, '过锐': sharp_label, '花屏': 0, '过度生成': 0, '虚化突兀': 0}
    return out, labels


def synth_abrupt_bokeh_p0(orig_bgr, rng, orig_sharp=0):
    """Synthesize 虚化突兀_P0: unfeathered / mis-segmented cutout + heavy disk-defocus surround."""
    h, w = orig_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    # Central subject ellipse + 2-3 abrupt polygonal "segmentation error" bites/protrusions
    cx = int(w * rng.uniform(0.38, 0.62))
    cy = int(h * rng.uniform(0.40, 0.60))
    rx = int(w * rng.uniform(0.22, 0.34))
    ry = int(h * rng.uniform(0.26, 0.40))
    cv2.ellipse(mask, (cx, cy), (rx, ry), rng.uniform(-15, 15), 0, 360, 1.0, -1)

    # Add 2 abrupt unfeathered satellite islands/bites along the boundary (classic portrait-mode error)
    for _ in range(2):
        ang = rng.uniform(0, 2 * np.pi)
        bx = int(np.clip(cx + rx * np.cos(ang), 10, w - 10))
        by = int(np.clip(cy + ry * np.sin(ang), 10, h - 10))
        brx = int(w * rng.uniform(0.04, 0.09))
        bry = int(h * rng.uniform(0.04, 0.09))
        val = 1.0 if rng.rand() < 0.5 else 0.0
        cv2.ellipse(mask, (bx, by), (brx, bry), rng.uniform(0, 180), 0, 360, val, -1)

    # Virtually ZERO feathering (0 or 1px) so the transition is harshly abrupt
    if rng.rand() < 0.4:
        mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=0.6)

    # Disk defocus kernel on background
    rad = int(rng.randint(9, 16))
    y_k, x_k = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    disk = ((x_k ** 2 + y_k ** 2) <= rad ** 2).astype(np.float32)
    disk /= disk.sum()
    bg_blur = cv2.filter2D(orig_bgr, -1, disk)

    mask3 = mask[..., None]
    out = (orig_bgr.astype(np.float32) * mask3 + bg_blur.astype(np.float32) * (1.0 - mask3)).astype(np.uint8)
    labels = {'模糊': 0, '纹理损失': 1, '过锐': orig_sharp, '花屏': 0, '过度生成': 0, '虚化突兀': 3}
    return out, labels


def synth_motion_defocus_blur(orig_bgr, grade, rng):
    """Synthesize realistic directional motion + slight defocus blur with co-occurring 纹理损失."""
    if grade == 'P2':
        klen = int(rng.randint(5, 9))
        sigma = rng.uniform(0.8, 1.4)
        blur_label, tex_label = 1, 1
    else:  # P1
        klen = int(rng.randint(10, 16))
        sigma = rng.uniform(1.5, 2.4)
        blur_label, tex_label = 2, 2

    # Build directional motion kernel
    kernel = np.zeros((klen, klen), dtype=np.float32)
    kernel[klen // 2, :] = 1.0
    angle = float(rng.uniform(0, 180))
    rot = cv2.getRotationMatrix2D((klen / 2.0 - 0.5, klen / 2.0 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (klen, klen))
    kernel /= max(kernel.sum(), 1e-6)

    out = cv2.filter2D(orig_bgr, -1, kernel)
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
    labels = {'模糊': blur_label, '纹理损失': tex_label, '过锐': 0, '花屏': 0, '过度生成': 0, '虚化突兀': 0}
    return out, labels


def build_for_split(split_csv, out_dir, out_split_csv, out_manifest_csv, seed=42, scale=1.0):
    rng = np.random.RandomState(seed)
    os.makedirs(out_dir, exist_ok=True)

    man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
    split_rows = list(csv.DictReader(open(split_csv)))
    train_names = {r['name'] for r in split_rows if r['split'] == 'train'}

    # Pure clean & pure mild-texture-loss images strictly in train split
    tr_clean = sorted([
        n for n in train_names
        if all(int(man[n][c]) == 0 for c in CLASSES)
    ])
    tr_tex1 = sorted([
        n for n in train_names
        if int(man[n]['纹理损失']) == 1 and all(int(man[n][c]) == 0 for c in CLASSES if c != '纹理损失')
    ])

    # Available FLUX3 and Swin2SR pairs whose source image is STRICTLY in train_names
    flx_map = {}
    for p in sorted(glob.glob('/tmp/synth/flux3/strong_*')):
        if p.endswith('.txt'):
            continue
        src = os.path.basename(p)[len('strong_'):]
        if src in train_names and all(int(man[src][c]) == 0 for c in CLASSES):
            flx_map[src] = p

    sr_map = {}
    for p in sorted(glob.glob('/tmp/synth/out/p2_*')):
        src = os.path.basename(p)[len('p2_'):]
        if src in train_names and all(int(man[src][c]) == 0 for c in CLASSES):
            sr_map[src] = p

    print(f"[{split_csv}] train_clean={len(tr_clean)}, train_tex1={len(tr_tex1)}, valid_flx={len(flx_map)}, valid_sr={len(sr_map)}")

    new_manifest = dict(man)
    new_split = list(split_rows)
    counts = {}

    def register(sname, img_bgr, labels):
        spath = os.path.join(out_dir, sname)
        if not os.path.exists(spath):
            save_with_realistic_compression(img_bgr, spath, rng)
        new_split.append({'name': sname, 'split': 'train', 'source': 'synthetic_v6'})
        row = {'name': sname}
        row.update(labels)
        new_manifest[sname] = row

    # 1. 过度生成 (P2, P1, P0) from FLUX3 + Swin2SR pairs
    flx_srcs = list(rng.permutation(sorted(flx_map.keys())))
    sr_srcs = list(rng.permutation([s for s in sorted(sr_map.keys()) if s not in flx_map]))
    gen_pool = [(s, flx_map[s], 'flx') for s in flx_srcs] + [(s, sr_map[s], 'sr') for s in sr_srcs]

    n_gen_p2 = int(round(35 * scale))
    n_gen_p1 = int(round(35 * scale))
    n_gen_p0 = int(round(20 * scale))

    for idx, (src, gpath, tag) in enumerate(gen_pool[:n_gen_p2 + n_gen_p1 + n_gen_p0]):
        opath = find_orig_path(src)
        if not opath:
            continue
        orig_bgr = cv2.imread(opath)
        gen_bgr = cv2.imread(gpath)
        if orig_bgr is None or gen_bgr is None:
            continue
        if idx < n_gen_p2:
            grade = 'P2'
        elif idx < n_gen_p2 + n_gen_p1:
            grade = 'P1'
        else:
            grade = 'P0'
        sname = f"v6_overgen_{grade}_{tag}_{src}"
        out_bgr, labels = synth_overgen_from_pair(orig_bgr, gen_bgr, grade, rng)
        register(sname, out_bgr, labels)
        counts[f'过度生成_{grade}'] = counts.get(f'过度生成_{grade}', 0) + 1

    # 2. 过锐 (P1, P0) on 60% clean + 40% tex1 pool
    sharp_pool_clean = list(rng.permutation(tr_clean))
    sharp_pool_tex1 = list(rng.permutation(tr_tex1))
    n_sharp_p1 = int(round(40 * scale))
    n_sharp_p0 = int(round(25 * scale))
    sharp_candidates = sharp_pool_tex1[:int(0.4 * (n_sharp_p1 + n_sharp_p0))] + sharp_pool_clean[:n_sharp_p1 + n_sharp_p0]
    sharp_candidates = sharp_candidates[:n_sharp_p1 + n_sharp_p0]

    for idx, src in enumerate(sharp_candidates):
        opath = find_orig_path(src)
        if not opath:
            continue
        orig_bgr = cv2.imread(opath)
        if orig_bgr is None:
            continue
        grade = 'P1' if idx < n_sharp_p1 else 'P0'
        orig_tex = int(man[src]['纹理损失'])
        sname = f"v6_sharp_{grade}_{src}"
        out_bgr, labels = synth_isp_oversharpen_v6(orig_bgr, grade, rng, orig_tex=orig_tex)
        register(sname, out_bgr, labels)
        counts[f'过锐_{grade}'] = counts.get(f'过锐_{grade}', 0) + 1

    # 3. 虚化突兀 (P0)
    bokeh_candidates = list(rng.permutation(tr_clean))[60:60 + int(round(30 * scale))]
    for src in bokeh_candidates:
        opath = find_orig_path(src)
        if not opath:
            continue
        orig_bgr = cv2.imread(opath)
        if orig_bgr is None:
            continue
        sname = f"v6_bokeh_P0_{src}"
        out_bgr, labels = synth_abrupt_bokeh_p0(orig_bgr, rng, orig_sharp=0)
        register(sname, out_bgr, labels)
        counts['虚化突兀_P0'] = counts.get('虚化突兀_P0', 0) + 1

    # 4. 模糊 (P2, P1) + 纹理损失 (P2, P1)
    blur_candidates = list(rng.permutation(tr_clean))[100:100 + int(round(40 * scale))]
    n_blur_p2 = int(round(20 * scale))
    for idx, src in enumerate(blur_candidates):
        opath = find_orig_path(src)
        if not opath:
            continue
        orig_bgr = cv2.imread(opath)
        if orig_bgr is None:
            continue
        grade = 'P2' if idx < n_blur_p2 else 'P1'
        sname = f"v6_blur_{grade}_{src}"
        out_bgr, labels = synth_motion_defocus_blur(orig_bgr, grade, rng)
        register(sname, out_bgr, labels)
        counts[f'模糊_{grade}'] = counts.get(f'模糊_{grade}', 0) + 1

    with open(out_split_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'split', 'source'])
        w.writeheader()
        w.writerows(new_split)

    fields = ['name'] + CLASSES
    with open(out_manifest_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k, v in new_manifest.items():
            w.writerow({c: v[c] for c in fields})

    print(f"Finished {out_split_csv}: added {sum(counts.values())} synthetic samples -> {counts}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['s16', 's17', 'full', 'all'], default='all')
    args = ap.parse_args()

    if args.target in ('s16', 'all'):
        build_for_split(
            'data/resplits/split_s16.csv',
            'data/synth_v6_s16',
            'data/resplits/split_s16_synth_v6.csv',
            'data/manifest_s16_synth_v6.csv',
            seed=1606, scale=1.0
        )
    if args.target in ('s17', 'all'):
        build_for_split(
            'data/resplits/split_s17.csv',
            'data/synth_v6_s17',
            'data/resplits/split_s17_synth_v6.csv',
            'data/manifest_s17_synth_v6.csv',
            seed=1706, scale=1.0
        )
    if args.target in ('full', 'all'):
        build_for_split(
            'data/split_all_full.csv',
            'data/synth_v6_full',
            'data/split_all_full_synth_v6.csv',
            'data/manifest_combined_synth_v6.csv',
            seed=2026, scale=1.25
        )
