"""Extract 1:1 Native-Resolution Physical & Frequency-Domain Caliper Features (60 dims)
for all images in manifest_combined.csv (and synthetic manifests).

Why:
Standard ViT backbones resize images (e.g. 1920x1080 -> 384x384), which destroys:
- 1-pixel USM halos & overshoot ratios (过锐 P2 vs P1 vs P0)
- Mild defocus vs sharp micro-gradients (模糊 P2 vs P1 vs 无问题)
- Micro-texture variance in flat/mid-tone patches vs macro-edges (纹理损失 P2 vs P1 vs P0)
- High-frequency AI plastic ridges & directional ringing (过度生成 P2 vs P1 vs P0)
- Spatial sharpness bimodality & abrupt transition gradients (虚化突兀 P0)

These 60 native-resolution features act as a resolution-invariant "physical caliper".
"""
import csv
import os
import time
from multiprocessing import Pool
import cv2
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')

IMG_DIRS = [
    '/dev/shm/v17_synth_quads',
    'data/images',
    'data/images_supp',
    'data/train',
    '/tmp/v9_authentic_imgs',
    '/tmp/v8_ladder_imgs',
    '/tmp/v7_overgen_pro_imgs',
    '/tmp/v6_synth_imgs',
    'data/synth_overgen_pro_s16',
    'data/synth_overgen_pro_s17',
    'data/synth_v6_s16',
    'data/synth_v6_s17',
    'data/test_150',
    'data/test',
]
# next55: final-eval images live outside the repo tree; prepend so they are found (and never silently zero-filled)
if os.environ.get('NEXT55_EXTRA_IMG_DIR'):
    IMG_DIRS.insert(0, os.environ['NEXT55_EXTRA_IMG_DIR'])


def find_img_path(name):
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def compute_one_image_feats(name):
    p = find_img_path(name)
    if p is None:
        return name, np.zeros(60, dtype=np.float32)
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        return name, np.zeros(60, dtype=np.float32)

    h, w = img.shape[:2]
    # Cap max side to 1600 for speed while preserving 1:1 native pixel fidelity on 1080p images
    if max(h, w) > 1600:
        scale = 1600.0 / max(h, w)
        img_nat = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_nat = img

    hn, wn = img_nat.shape[:2]
    gray = cv2.cvtColor(img_nat, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_nat, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

    feats = []
    # 1. Geometry & basic luminance/color stats (8 dims)
    feats.extend([
        np.log1p(h * w) / 16.0,
        float(w) / max(1.0, float(h)),
        float(max(h, w)) / 1000.0,
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.mean(hsv[:, :, 1])),
        float(np.std(hsv[:, :, 1])),
        float(np.mean((gray < 0.02) | (gray > 0.98))),  # clipping fraction
    ])

    # 2. Multi-scale Laplacian & High-Pass Energy at 1:1, 1/2, 1/4 scales (12 dims)
    # Critical for 模糊 (P2 vs P1 vs P0) and 过锐 (P2 vs P1 vs P0)
    lap1 = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad1 = np.hypot(gx, gy)

    gray_d2 = cv2.resize(gray, (max(8, wn // 2), max(8, hn // 2)), interpolation=cv2.INTER_AREA)
    lap2 = np.abs(cv2.Laplacian(gray_d2, cv2.CV_32F, ksize=3))
    grad2 = np.hypot(cv2.Sobel(gray_d2, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray_d2, cv2.CV_32F, 0, 1, ksize=3))

    gray_d4 = cv2.resize(gray_d2, (max(8, wn // 4), max(8, hn // 4)), interpolation=cv2.INTER_AREA)
    lap4 = np.abs(cv2.Laplacian(gray_d4, cv2.CV_32F, ksize=3))
    grad4 = np.hypot(cv2.Sobel(gray_d4, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray_d4, cv2.CV_32F, 0, 1, ksize=3))

    m_lap1, m_lap2, m_lap4 = float(np.mean(lap1)), float(np.mean(lap2)), float(np.mean(lap4))
    m_grd1, m_grd2, m_grd4 = float(np.mean(grad1)), float(np.mean(grad2)), float(np.mean(grad4))
    feats.extend([
        m_lap1 * 10.0,
        m_lap2 * 10.0,
        m_lap4 * 10.0,
        m_lap1 / (m_lap2 + 1e-5),  # 1:1 vs 1/2 high-freq rolloff ratio!
        m_lap2 / (m_lap4 + 1e-5),  # 1/2 vs 1/4 rolloff ratio!
        m_grd1 * 5.0,
        m_grd1 / (m_grd2 + 1e-5),
        m_grd2 / (m_grd4 + 1e-5),
        float(np.percentile(lap1, 90)) * 5.0,
        float(np.percentile(lap1, 99)) * 2.0,
        float(np.percentile(lap1, 99.8)),
        float(np.percentile(grad1, 99)),
    ])

    # 3. USM / Over-Sharpening Halo Overshoot & Ringing Features (12 dims)
    # Critical for 过锐 (P2 vs P1 vs P0) and 过度生成 (Sub2 sharp plastic)
    blur_s1 = cv2.GaussianBlur(gray, (0, 0), 1.2)
    blur_s2 = cv2.GaussianBlur(gray, (0, 0), 2.5)
    blur_s4 = cv2.GaussianBlur(gray, (0, 0), 5.0)
    hp1 = gray - blur_s1
    hp2 = gray - blur_s2
    bp12 = blur_s1 - blur_s2
    bp24 = blur_s2 - blur_s4

    edge_thr = max(0.08, float(np.percentile(grad1, 90)))
    edge_mask = grad1 > edge_thr
    # Dilate edge mask to inspect immediate 2-3 pixel halo neighborhood around strong edges
    k5 = np.ones((5, 5), np.uint8)
    edge_dil = cv2.dilate(edge_mask.astype(np.uint8), k5) > 0
    halo_zone = edge_dil & (~edge_mask)

    if np.any(edge_mask):
        pos_overshoot = float(np.percentile(hp2[edge_mask], 95))
        neg_undershoot = float(np.percentile(-hp2[edge_mask], 95))
        extreme_pos = float(np.mean(hp1[edge_mask] > 0.08))
        extreme_neg = float(np.mean(hp1[edge_mask] < -0.08))
    else:
        pos_overshoot = neg_undershoot = extreme_pos = extreme_neg = 0.0

    if np.any(halo_zone):
        halo_ring = float(np.std(hp2[halo_zone])) * 10.0
        halo_p95 = float(np.percentile(np.abs(hp2[halo_zone]), 95)) * 10.0
    else:
        halo_ring = halo_p95 = 0.0

    feats.extend([
        pos_overshoot * 5.0,
        neg_undershoot * 5.0,
        (pos_overshoot + neg_undershoot) * 5.0,
        extreme_pos * 5.0,
        extreme_neg * 5.0,
        halo_ring,
        halo_p95,
        float(np.mean( np.abs(hp1) )) * 20.0,
        float(np.mean( np.abs(bp12) )) * 20.0,
        float(np.mean( np.abs(bp24) )) * 20.0,
        float(np.mean(np.abs(hp1))) / (float(np.mean(np.abs(bp12))) + 1e-5),
        float(np.mean(np.abs(bp12))) / (float(np.mean(np.abs(bp24))) + 1e-5),
    ])

    # 4. Patch-Grid Sharpness & Texture Distribution (16 dims)
    # Critical for 纹理损失 (P2 vs P1 vs P0), 模糊 (P2 vs P1 vs P0), and 虚化突兀 (P0)
    gh, gw = 8, 8
    ph, pw = max(1, hn // gh), max(1, wn // gw)
    patch_lap = np.zeros((gh, gw), dtype=np.float32)
    patch_grd = np.zeros((gh, gw), dtype=np.float32)
    patch_hp = np.zeros((gh, gw), dtype=np.float32)
    for r in range(gh):
        for c in range(gw):
            sl_r = slice(r * ph, min(hn, (r + 1) * ph))
            sl_c = slice(c * pw, min(wn, (c + 1) * pw))
            patch_lap[r, c] = np.mean(lap1[sl_r, sl_c])
            patch_grd[r, c] = np.mean(grad1[sl_r, sl_c])
            patch_hp[r, c] = np.std(hp1[sl_r, sl_c])

    pl_flat = np.sort(patch_lap.ravel())
    pg_flat = np.sort(patch_grd.ravel())
    ph_flat = np.sort(patch_hp.ravel())

    # Center 4x4 vs Border patches (for 虚化突兀 bokeh cutout detection!)
    center_lap = float(np.mean(patch_lap[2:6, 2:6]))
    border_mask = np.ones((gh, gw), dtype=bool)
    border_mask[2:6, 2:6] = False
    border_lap = float(np.mean(patch_lap[border_mask]))

    # Sharpness transition abruptness across adjacent grid cells
    adj_diff_r = np.abs(np.diff(patch_lap, axis=0))
    adj_diff_c = np.abs(np.diff(patch_lap, axis=1))
    max_abrupt_jump = float(max(np.max(adj_diff_r), np.max(adj_diff_c))) / (float(np.mean(patch_lap)) + 1e-5)

    # Bimodality of patch sharpness: fraction of patches that are ultra-flat vs sharp
    top20_lap = float(np.mean(pl_flat[-13:]))
    bot30_lap = float(np.mean(pl_flat[:20]))
    mid_lap = float(np.mean(pl_flat[20:45]))

    feats.extend([
        top20_lap * 10.0,
        mid_lap * 10.0,
        bot30_lap * 10.0,
        top20_lap / (mid_lap + 1e-5),        # High when background is waxy/blurred but foreground is sharp
        mid_lap / (bot30_lap + 1e-5),
        center_lap / (border_lap + 1e-5),    # Center-vs-border bokeh ratio
        max_abrupt_jump,                     # Abrupt bokeh transition jump (虚化突兀!)
        float(np.std(pl_flat)) / (float(np.mean(pl_flat)) + 1e-5),
        float(np.mean(ph_flat[-13:])) * 20.0,
        float(np.mean(ph_flat[20:45])) * 20.0,
        float(np.mean(ph_flat[:20])) * 20.0,
        float(np.mean(pg_flat[-13:])) * 5.0,
        float(np.mean(pg_flat[20:45])) * 5.0,
        float(np.mean(pg_flat[:20])) * 5.0,
        float(np.mean(pl_flat < 0.008)),     # Fraction of dead-flat/waxy patches (纹理损失!)
        float(np.mean(pl_flat > 0.045)),     # Fraction of hyper-sharp/noisy patches
    ])

    # 5. AI Plastic Ridge / Micro-Texture Kurtosis & 8x8 Blockiness (12 dims)
    # Critical for 过度生成 (AI synthetic ridges/stipples) and 纹理损失 (compression/denoise)
    # Non-edge mid-tone mask: where macro gradient is moderate/low, how much micro-texture exists?
    non_edge = (grad2_up := cv2.resize(grad2, (wn, hn))) < np.percentile(grad2_up, 65)
    if np.any(non_edge):
        micro_tex_mean = float(np.mean(np.abs(hp1[non_edge]))) * 30.0
        micro_tex_p90 = float(np.percentile(np.abs(hp1[non_edge]), 90)) * 20.0
        # Kurtosis of high-pass in non-edge regions (AI hallucinations have spiky ridges in smooth skin/surfaces)
        hp_ne = hp1[non_edge]
        s_ne = float(np.std(hp_ne)) + 1e-6
        kurt_ne = float(np.mean(((hp_ne - np.mean(hp_ne)) / s_ne) ** 4))
        kurt_ne = min(50.0, kurt_ne) / 10.0
    else:
        micro_tex_mean = micro_tex_p90 = kurt_ne = 0.0

    # Directional anisotropy (horizontal/vertical vs diagonal gradients - SR/GAN & compression artifact)
    g_hv = float(np.mean(np.abs(gx) + np.abs(gy)))
    g_diag1 = float(np.mean(np.abs(gray[1:, 1:] - gray[:-1, :-1])))
    g_diag2 = float(np.mean(np.abs(gray[1:, :-1] - gray[:-1, 1:])))
    aniso = g_hv / (g_diag1 + g_diag2 + 1e-5)

    # 8x8 blockiness metric
    if hn > 16 and wn > 16:
        b_h = float(np.mean(np.abs(gray[7:-1:8, :] - gray[8::8, :])))
        in_h = float(np.mean(np.abs(gray[3:-1:8, :] - gray[4::8, :])))
        block_ratio = b_h / (in_h + 1e-5)
    else:
        block_ratio = 1.0

    # Color channel high-freq correlation (chroma noise vs luma edge alignment)
    b_ch, g_ch, r_ch = [ img_nat[:, :, i].astype(np.float32) / 255.0 for i in range(3) ]
    rg_diff = np.abs(cv2.Laplacian(r_ch - g_ch, cv2.CV_32F, ksize=3))
    by_diff = np.abs(cv2.Laplacian(b_ch - g_ch, cv2.CV_32F, ksize=3))
    chroma_hf = float(np.mean(rg_diff + by_diff)) * 10.0

    feats.extend([
        micro_tex_mean,
        micro_tex_p90,
        kurt_ne,
        aniso,
        block_ratio,
        chroma_hf,
        chroma_hf / (m_lap1 * 10.0 + 1e-4),
        float(np.percentile(rg_diff, 99)) * 10.0,
        float(np.mean(grad1 > 0.15)),
        float(np.mean((grad1 > 0.04) & (grad1 <= 0.15))),
        float(np.mean((grad1 > 0.01) & (grad1 <= 0.04))),
        float(np.mean(grad1 <= 0.01)),
    ])

    arr = np.nan_to_num(np.array(feats[:60], dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
    return name, arr


def extract_for_names(names, cache_path='results/v8_experts/native_phys_feats.npz'):
    import fcntl
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    lock_path = cache_path + '.lock'

    existing = {}
    if os.path.exists(cache_path):
        try:
            d = np.load(cache_path, allow_pickle=True)
            if 'data' in d:
                existing.update(d['data'].item())
        except Exception as e:
            print(f'[warning] cache load failed ({e}), will re-extract', flush=True)

    missing = [n for n in names if n not in existing]
    if missing:
        t0 = time.time()
        n_workers = min(48, os.cpu_count() or 24)
        with Pool(n_workers) as pool:
            for nm, vec in pool.imap_unordered(compute_one_image_feats, missing, chunksize=32):
                existing[nm] = vec

        # Atomic lock + read latest + merge + atomic write
        with open(lock_path, 'w') as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if os.path.exists(cache_path):
                    try:
                        latest = np.load(cache_path, allow_pickle=True)
                        if 'data' in latest:
                            merged = latest['data'].item()
                            merged.update(existing)
                            existing = merged
                    except Exception:
                        pass
                tmp_path = f"{cache_path}.tmp.{os.getpid()}_{time.time()}.npz"
                np.savez_compressed(tmp_path, data=existing)
                os.replace(tmp_path, cache_path)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
        print(f'[{time.time()-t0:.1f}s] Extracted 60-dim native physical features for {len(missing)} images (total cached: {len(existing)})', flush=True)
    return existing


if __name__ == '__main__':
    import glob
    import json
    with open('data/manifest_combined.csv') as f:
        all_names = [r['name'] for r in csv.DictReader(f)]
    if os.path.exists('results/v2_full/template_150.json'):
        tpl = json.load(open('results/v2_full/template_150.json'))
        all_names.extend([e['image'] for e in tpl])
    for m_csv in glob.glob('data/v9_ladders/manifest_*.csv') + glob.glob('data/v8_ladders/manifest_*.csv'):
        with open(m_csv) as f:
            all_names.extend([r['name'] for r in csv.DictReader(f)])
    all_names = sorted(set(all_names))
    extract_for_names(all_names)
    extract_for_names(all_names)
