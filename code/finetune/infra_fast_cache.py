#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/infra_fast_cache.py
Precomputes 1:1 Native Microscope ROI coordinates (Edge Top-2, Interior Texture Top-2,
Depth Boundary Top-2) and image geometry metadata (H, W, aspect_ratio, scale_factor)
for all 3,573 real images + 150 test images in parallel across 32 CPU workers.
"""
import os, csv, json, math, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2
from PIL import Image

os.chdir('/wanqing-develop/luowenjing/18')
CACHE_OUT = '/dev/shm/race18_native/roi_coords_and_meta.npz'

def find_img_path(name):
    for d in ['data/images', 'data/images_supp', 'data/test_150', '/dev/shm/v18_synth_quads', '/dev/shm/v17_synth_quads']:
        p = os.path.join(d, name) if not d.startswith('/') else os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

def compute_roi_for_image(name, size=448):
    p = find_img_path(name)
    if p is None:
        return name, np.zeros((6, 2), dtype=np.int32), np.array([448, 448, 1.0, 0.0, 1.0], dtype=np.float32)
    with Image.open(p) as im_pil:
        im = np.asarray(im_pil.convert('RGB'), dtype=np.uint8)
    h, w = im.shape[:2]
    ar = float(w) / float(max(1, h))
    log_ar = abs(math.log(max(ar, 1e-3)))
    scale_fac = math.sqrt((w * h) / float(size * size))
    meta = np.array([h, w, ar, log_ar, scale_fac], dtype=np.float32)

    if h <= size and w <= size:
        return name, np.zeros((6, 2), dtype=np.int32), meta

    # Downsample by 4x strictly for fast integral energy localization; crops are taken at 1:1 native pixels
    gray = cv2.cvtColor(im[::4, ::4], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gh, gw = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, gh, dtype=np.float32),
                         np.linspace(-1.0, 1.0, gw, dtype=np.float32), indexing='ij')
    center_p = np.exp(-0.55 * (xx ** 2 + (yy + 0.05) ** 2))
    sal = cv2.GaussianBlur(grad, (0, 0), sigmaX=max(1.5, min(gh, gw) / 24.0)) * (0.65 + 0.35 * center_p)
    thr = np.percentile(sal, 50)
    mfg = np.clip((sal - thr * 0.75) / (float(np.max(sal)) - thr * 0.75 + 1e-5), 0.0, 1.0)
    ring = np.abs(cv2.dilate(mfg, np.ones((5, 5), np.float32)) - cv2.erode(mfg, np.ones((5, 5), np.float32)))

    ys = np.unique(np.linspace(0, max(0, h - size), num=5, dtype=int))
    xs = np.unique(np.linspace(0, max(0, w - size), num=5, dtype=int))

    cands, s_edge, s_tex, s_ring = [], [], [], []
    for y0 in ys:
        for x0 in xs:
            sy0, sy1 = y0 // 4, max(y0 // 4 + 1, min(gh, (y0 + size) // 4))
            sx0, sx1 = x0 // 4, max(x0 // 4 + 1, min(gw, (x0 + size) // 4))
            g_sub = grad[sy0:sy1, sx0:sx1]
            l_sub = lap[sy0:sy1, sx0:sx1]
            m_sub = mfg[sy0:sy1, sx0:sx1]
            r_sub = ring[sy0:sy1, sx0:sx1]
            im_sub = gray[sy0:sy1, sx0:sx1]
            cands.append((int(y0), int(x0)))
            s_edge.append(float(np.mean((g_sub + 0.85 * l_sub) * (0.5 + 0.5 * m_sub))))
            std_i = float(np.std(im_sub))
            s_tex.append(float(np.mean(m_sub)) * math.exp(-((std_i - 0.12) ** 2) / 0.025) / (float(np.mean(g_sub)) + 0.025))
            s_ring.append(float(np.mean(r_sub * g_sub)))

    def top2_non_overlap(scores):
        order = np.argsort(scores)[::-1]
        picked = [cands[int(order[0])]]
        for idx in order[1:]:
            cy, cx = cands[int(idx)]
            if abs(cy - picked[0][0]) + abs(cx - picked[0][1]) >= size // 3:
                picked.append((cy, cx))
                break
        if len(picked) == 1:
            picked.append(picked[0])
        return picked

    coords = top2_non_overlap(s_edge) + top2_non_overlap(s_tex) + top2_non_overlap(s_ring)
    return name, np.array(coords, dtype=np.int32), meta

def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(CACHE_OUT), exist_ok=True)
    man = list(csv.DictReader(open('data/manifest_combined.csv')))
    tpl = json.load(open('results/v2_full/template_150.json'))
    names = sorted(set([r['name'] for r in man] + [e['image'] for e in tpl]))
    print(f'Precomputing 1:1 ROI coords & geometry meta for {len(names)} images across 32 CPU workers...', flush=True)
    with ProcessPoolExecutor(max_workers=32) as ex:
        res = list(ex.map(compute_roi_for_image, names))
    coords_map = {n: c for n, c, _ in res}
    meta_map = {n: m for n, _, m in res}
    np.savez_compressed(
        CACHE_OUT,
        names=np.array(names),
        coords=np.stack([coords_map[n] for n in names], axis=0),
        meta=np.stack([meta_map[n] for n in names], axis=0),
    )
    print(f'Saved {len(names)} ROI coords & meta to {CACHE_OUT} in {time.time()-t0:.1f}s!', flush=True)

if __name__ == '__main__':
    main()
