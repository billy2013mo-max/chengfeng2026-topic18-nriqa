import os, time, cv2, numpy as np

def compute_88d_phys_feats(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros(88, dtype=np.float32)
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        scale = 1600.0 / max(h, w)
        img_nat = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_nat = img

    hn, wn = img_nat.shape[:2]
    gray = cv2.cvtColor(img_nat, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_nat, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0

    feats = []
    # --- Part 1: Original 60 Dimensions ---
    # 1. Geometry & luminance/color stats (8 dims)
    feats.extend([
        np.log1p(h * w) / 16.0,
        float(w) / max(1.0, float(h)),
        float(max(h, w)) / 1000.0,
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.mean(hsv[:, :, 1])),
        float(np.std(hsv[:, :, 1])),
        float(np.mean((gray < 0.02) | (gray > 0.98))),
    ])

    # 2. Multi-scale Laplacian & High-Pass (12 dims)
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
        m_lap1 / (m_lap2 + 1e-5),
        m_lap2 / (m_lap4 + 1e-5),
        m_grd1 * 5.0,
        m_grd1 / (m_grd2 + 1e-5),
        m_grd2 / (m_grd4 + 1e-5),
        float(np.percentile(lap1, 90)) * 5.0,
        float(np.percentile(lap1, 99)) * 2.0,
        float(np.percentile(lap1, 99.8)),
        float(np.percentile(grad1, 99)),
    ])

    # 3. USM / Over-Sharpening Halo (12 dims)
    blur_s1 = cv2.GaussianBlur(gray, (0, 0), 1.2)
    blur_s2 = cv2.GaussianBlur(gray, (0, 0), 2.5)
    blur_s4 = cv2.GaussianBlur(gray, (0, 0), 5.0)
    hp1 = gray - blur_s1
    hp2 = gray - blur_s2
    bp12 = blur_s1 - blur_s2
    bp24 = blur_s2 - blur_s4

    edge_thr = max(0.08, float(np.percentile(grad1, 90)))
    edge_mask = grad1 > edge_thr
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
        float(np.mean(np.abs(hp1))) * 20.0,
        float(np.mean(np.abs(bp12))) * 20.0,
        float(np.mean(np.abs(bp24))) * 20.0,
        float(np.mean(np.abs(hp1))) / (float(np.mean(np.abs(bp12))) + 1e-5),
        float(np.mean(np.abs(bp12))) / (float(np.mean(np.abs(bp24))) + 1e-5),
    ])

    # 4. Patch-Grid Sharpness & Distribution (16 dims)
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

    center_lap = float(np.mean(patch_lap[2:6, 2:6]))
    border_mask = np.ones((gh, gw), dtype=bool)
    border_mask[2:6, 2:6] = False
    border_lap = float(np.mean(patch_lap[border_mask]))

    adj_diff_r = np.abs(np.diff(patch_lap, axis=0))
    adj_diff_c = np.abs(np.diff(patch_lap, axis=1))
    max_abrupt_jump = float(max(np.max(adj_diff_r), np.max(adj_diff_c))) / (float(np.mean(patch_lap)) + 1e-5)

    top20_lap = float(np.mean(pl_flat[-13:]))
    bot30_lap = float(np.mean(pl_flat[:20]))
    mid_lap = float(np.mean(pl_flat[20:45]))

    feats.extend([
        top20_lap * 10.0,
        mid_lap * 10.0,
        bot30_lap * 10.0,
        top20_lap / (mid_lap + 1e-5),
        mid_lap / (bot30_lap + 1e-5),
        center_lap / (border_lap + 1e-5),
        max_abrupt_jump,
        float(np.std(pl_flat)) / (float(np.mean(pl_flat)) + 1e-5),
        float(np.mean(ph_flat[-13:])) * 20.0,
        float(np.mean(ph_flat[20:45])) * 20.0,
        float(np.mean(ph_flat[:20])) * 20.0,
        float(np.mean(pg_flat[-13:])) * 5.0,
        float(np.mean(pg_flat[20:45])) * 5.0,
        float(np.mean(pg_flat[:20])) * 5.0,
        float(np.mean(pl_flat < 0.008)),
        float(np.mean(pl_flat > 0.045)),
    ])

    # 5. AI Plastic Ridge & 8x8 Blockiness (12 dims)
    grad2_up = cv2.resize(grad2, (wn, hn))
    non_edge = grad2_up < np.percentile(grad2_up, 65)
    if np.any(non_edge):
        micro_tex_mean = float(np.mean(np.abs(hp1[non_edge]))) * 30.0
        micro_tex_p90 = float(np.percentile(np.abs(hp1[non_edge]), 90)) * 20.0
        hp_ne = hp1[non_edge]
        s_ne = float(np.std(hp_ne)) + 1e-6
        kurt_ne = float(np.mean(((hp_ne - np.mean(hp_ne)) / s_ne) ** 4))
        kurt_ne = min(50.0, kurt_ne) / 10.0
    else:
        micro_tex_mean = micro_tex_p90 = kurt_ne = 0.0

    g_hv = float(np.mean(np.abs(gx) + np.abs(gy)))
    g_diag1 = float(np.mean(np.abs(gray[1:, 1:] - gray[:-1, :-1])))
    g_diag2 = float(np.mean(np.abs(gray[1:, :-1] - gray[:-1, 1:])))
    aniso = g_hv / (g_diag1 + g_diag2 + 1e-5)

    if hn > 16 and wn > 16:
        b_h = float(np.mean(np.abs(gray[7:-1:8, :] - gray[8::8, :])))
        in_h = float(np.mean(np.abs(gray[3:-1:8, :] - gray[4::8, :])))
        block_ratio = b_h / (in_h + 1e-5)
    else:
        block_ratio = 1.0

    b_ch, g_ch, r_ch = [img_nat[:, :, i].astype(np.float32) / 255.0 for i in range(3)]
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

    assert len(feats) == 60, f"Expected 60, got {len(feats)}"

    # --- Part 2: 28 New Dimensions (Total = 88 Dims) ---

    # Module 1: Edge-Normal Halo Overshoot Profile Operator (8 dims)
    # Extracts precise 1-3px USM overshoot peaks vs monotonic natural edges
    p85 = float(np.percentile(grad1, 85))
    p98 = float(np.percentile(grad1, 98))
    main_edge_mask = (grad1 >= p85) & (grad1 <= p98)
    
    # Subsample up to 2000 points on main edge for vectorised normal sampling
    edge_pts = np.argwhere(main_edge_mask)
    if len(edge_pts) > 2000:
        idx = np.random.RandomState(42).choice(len(edge_pts), 2000, replace=False)
        edge_pts = edge_pts[idx]

    if len(edge_pts) > 10:
        ys, xs = edge_pts[:, 0], edge_pts[:, 1]
        gxs = gx[ys, xs]
        gys = gy[ys, xs]
        mag = np.hypot(gxs, gys) + 1e-6
        nx = gxs / mag
        ny = gys / mag

        # Sample d in {1..6} along +n and -n
        # Bilinear sample using remap or grid
        # For speed: round to integer displacement
        os_pos_list = []
        us_neg_list = []
        for d in range(1, 4):
            yp_d = np.clip(np.round(ys + d * ny).astype(int), 0, hn - 1)
            xp_d = np.clip(np.round(xs + d * nx).astype(int), 0, wn - 1)
            ym_d = np.clip(np.round(ys - d * ny).astype(int), 0, hn - 1)
            xm_d = np.clip(np.round(xs - d * nx).astype(int), 0, wn - 1)
            os_pos_list.append(gray[yp_d, xp_d])
            us_neg_list.append(gray[ym_d, xm_d])
        
        yp_6 = np.clip(np.round(ys + 6 * ny).astype(int), 0, hn - 1)
        xp_6 = np.clip(np.round(xs + 6 * nx).astype(int), 0, wn - 1)
        ym_6 = np.clip(np.round(ys - 6 * ny).astype(int), 0, hn - 1)
        xm_6 = np.clip(np.round(xs - 6 * nx).astype(int), 0, wn - 1)
        i_pos_6 = gray[yp_6, xp_6]
        i_neg_6 = gray[ym_6, xm_6]

        max_pos_13 = np.maximum.reduce(os_pos_list)
        min_neg_13 = np.minimum.reduce(us_neg_list)
        step_height = np.maximum(np.abs(i_pos_6 - i_neg_6), 10.0 / 255.0)

        os_plus = np.maximum(0.0, (max_pos_13 - np.maximum(i_pos_6, i_neg_6))) / step_height
        us_minus = np.maximum(0.0, (np.minimum(i_pos_6, i_neg_6) - min_neg_13)) / step_height
        os_sum = os_plus + us_minus

        feats.extend([
            float(np.mean(os_plus)) * 5.0,
            float(np.percentile(os_plus, 90)) * 5.0,
            float(np.percentile(os_plus, 98)) * 3.0,
            float(np.mean(us_minus)) * 5.0,
            float(np.percentile(us_minus, 90)) * 5.0,
            float(np.percentile(us_minus, 98)) * 3.0,
            float(np.mean(os_plus > 0.08)),  # Overshoot area ratio
            float(np.percentile(os_sum, 95)) * 3.0,
        ])
    else:
        feats.extend([0.0] * 8)

    # Module 2: Edge-Spread vs. Interior Wavelet Entropy Operator (10 dims)
    # Decouples blur (edge spread) from texture loss (interior smear)
    p75_g = float(np.percentile(grad1, 75))
    p25_g = float(np.percentile(grad1, 25))
    mask_edge = grad1 > p75_g
    mask_flat = grad1 < p25_g
    mask_tex = (~mask_edge) & (~mask_flat)

    # Marziliano Edge Width W_edge
    # Measure width across edge normal: distance where grad drops below 50% max
    w_edge_mean = float(np.mean(grad1[mask_edge])) / (float(np.mean(lap1[mask_edge])) + 1e-4) if np.any(mask_edge) else 1.0
    w_edge_p90 = float(np.percentile(grad1[mask_edge], 90)) / (float(np.percentile(lap1[mask_edge], 90)) + 1e-4) if np.any(mask_edge) else 1.0

    # 2-level Haar Wavelet subbands
    # Level 1:
    h1 = gray[::2, :] - gray[1::2, :] if hn % 2 == 0 else gray[:-1:2, :] - gray[1::2, :]
    w1 = gray[:, ::2] - gray[:, 1::2] if wn % 2 == 0 else gray[:, :-1:2] - gray[:, 1::2]
    # Resize to common shape for subband energy
    e1_sub = cv2.resize(np.abs(h1), (wn, hn)) + cv2.resize(np.abs(w1), (wn, hn))
    
    # Texture energy in mask_tex
    e_tex = float(np.mean(e1_sub[mask_tex])) if np.any(mask_tex) else 0.0
    e_flat = float(np.mean(e1_sub[mask_flat])) if np.any(mask_flat) else 1e-4

    # Level 2 Haar energy:
    g_d2 = cv2.resize(gray, (max(4, wn // 2), max(4, hn // 2)), interpolation=cv2.INTER_AREA)
    h2 = g_d2[::2, :] - g_d2[1::2, :] if g_d2.shape[0] % 2 == 0 else g_d2[:-1:2, :] - g_d2[1::2, :]
    e2_tex = float(np.mean(np.abs(h2)))

    # Local Binary Pattern (LBP) entropy approximation in texture zone
    # Fast 4-neighbor sign difference
    c_g = gray[1:-1, 1:-1]
    lbp_code = ((c_g > gray[:-2, 1:-1]).astype(np.uint8) * 1 +
                (c_g > gray[2:, 1:-1]).astype(np.uint8) * 2 +
                (c_g > gray[1:-1, :-2]).astype(np.uint8) * 4 +
                (c_g > gray[1:-1, 2:]).astype(np.uint8) * 8)
    # LBP entropy
    lbp_hist, _ = np.histogram(lbp_code, bins=16, range=(0, 16), density=True)
    lbp_entropy = -float(np.sum(lbp_hist * np.log2(lbp_hist + 1e-7)))

    # Smear-to-sharpness decoupling ratio (SER)
    ser_ratio = float(np.log1p(100.0 * e_tex) / (np.log1p(1.0 / (w_edge_mean + 1e-3)) + 1e-5))

    tot_px = float(hn * wn)
    feats.extend([
        min(20.0, w_edge_mean),
        min(20.0, w_edge_p90),
        e_tex * 30.0,
        e2_tex * 30.0,
        lbp_entropy,
        ser_ratio,
        float(e_tex / (e_flat + 1e-4)),
        float(np.sum(mask_flat) / tot_px),
        float(np.sum(mask_tex) / tot_px),
        float(np.sum(mask_edge) / tot_px),
    ])

    # Module 3: Boundary Depth-Sharpness Coherence Operator (10 dims)
    # Evaluates whether blur changes smoothly along contour (natural optical bokeh) or has abrupt jumps (AI cutout error)
    # Foreground mask approximation: central elliptical Gaussian + adaptive threshold
    yy, xx = np.ogrid[:hn, :wn]
    cy, cx = hn / 2.0, wn / 2.0
    dist_sq = ((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2
    center_weight = np.exp(-dist_sq)
    fg_cue = blur_s4 * 0.5 + center_weight * 0.5
    fg_mask = fg_cue > np.percentile(fg_cue, 55)

    k21 = np.ones((21, 21), np.uint8)
    fg_dil = cv2.dilate(fg_mask.astype(np.uint8), k21)
    fg_ero = cv2.erode(fg_mask.astype(np.uint8), k21)
    boundary_ring = (fg_dil > 0) & (fg_ero == 0)

    # 16-sector radial partition around center
    angles = np.arctan2(yy - cy, xx - cx)  # [-pi, pi]
    sector_bins = np.linspace(-np.pi, np.pi, 17)

    delta_laps = []
    inner_mask = (fg_mask > 0) & boundary_ring
    outer_mask = (fg_mask == 0) & boundary_ring

    for s in range(16):
        sec_m = boundary_ring & (angles >= sector_bins[s]) & (angles < sector_bins[s+1])
        sec_in = sec_m & (fg_mask > 0)
        sec_out = sec_m & (fg_mask == 0)
        if np.any(sec_in) and np.any(sec_out):
            l_in = float(np.mean(lap1[sec_in]))
            l_out = float(np.mean(lap1[sec_out]))
            d_l = float(np.log1p(l_in * 100.0) - np.log1p(l_out * 100.0))
            delta_laps.append(d_l)
        else:
            delta_laps.append(0.0)

    delta_arr = np.array(delta_laps, dtype=np.float32)
    ring_lap = lap1[boundary_ring] if np.any(boundary_ring) else lap1
    fg_lap = float(np.mean(lap1[fg_mask])) if np.any(fg_mask) else 0.01
    bg_lap = float(np.mean(lap1[~fg_mask])) if np.any(~fg_mask) else 0.01

    diff_consec = np.abs(np.diff(np.append(delta_arr, delta_arr[0])))

    feats.extend([
        float(np.mean(delta_arr)),
        float(np.max(delta_arr)),
        float(np.min(delta_arr)),
        float(np.std(delta_arr)),  # Key: high std = abnormal cutout variance along boundary!
        float(np.percentile(delta_arr, 75) - np.percentile(delta_arr, 25)),  # IQR
        float(np.mean(np.abs(delta_arr) > 1.2)),  # Fraction of abrupt sectors
        float(np.mean(ring_lap)) * 10.0,
        float(np.std(ring_lap)) * 10.0,
        float(np.log1p(fg_lap * 100.0) / (np.log1p(bg_lap * 100.0) + 1e-4)),
        float(np.max(diff_consec)),  # Max consecutive sector jump
    ])

    assert len(feats) == 88, f"Expected 88, got {len(feats)}"
    arr = np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
    return arr

if __name__ == '__main__':
    t0 = time.time()
    test_imgs = os.listdir('data/test_150')[:5]
    for nm in test_imgs:
        p = os.path.join('data/test_150', nm)
        f = compute_88d_phys_feats(p)
        print(f'{nm}: shape={f.shape}, min={f.min():.3f}, max={f.max():.3f}, mean={f.mean():.3f}')
    print(f'Done 5 images in {time.time()-t0:.2f}s')
