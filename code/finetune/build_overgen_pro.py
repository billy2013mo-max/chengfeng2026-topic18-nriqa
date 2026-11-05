"""Build V7 OverGen-Pro Dataset (Dedicated 过度生成 Specialist Synthesis, Zero-Leakage).

Implements 3 domain-authentic 过度生成 synthesis mechanisms aligned with the 3 major
empirical co-occurrence subtypes in `manifest_combined.csv`:

1. Subtype 1 (Degrade-Blur-then-Hallucinate, ~55% of real 过度生成):
   - Downsample + motion/defocus blur + JPEG block degradation on original image,
     followed by compositing plastic surfaces & hallucinated high-frequency ridges
     from FLUX3-Klein and 2-pass Swin2SR (`p2_*`).
   - Produces classic "blurry frame + AI-hallucinated plastic ridges/melted details".
   - Labels: (模糊=1..2, 纹理损失=2, 过锐=0, 花屏=0, 过度生成=1..3, 虚化突兀=0).

2. Subtype 2 (AI Re-render + ISP Edge Oversharpening, ~59% of real 过度生成_P2):
   - Combines FLUX3/Swin2SR plastic texture re-synthesis with LAB L-channel USM edge haloing.
   - Matches the #1 real 过度生成_P2 tuple (0, 1, 1, 0, 1, 0) and (0, 2, 1, 0, 1, 0).
   - Labels: (模糊=0, 纹理损失=1..2, 过锐=1..2, 花屏=0, 过度生成=1..2, 虚化突兀=0).

3. Subtype 3 (Real 过度生成_P0/P1 BadCase Patch Seamless Grafting):
   - Extracts high-gradient AI-corrupted patches strictly from training-split real
     过度生成=P0/P1 images, color-matches them in LAB space, and grafts them onto
     training recipient images.
   - Labels inherit recipient base defects + 过度生成=1 (P2, 12-20% area), 2 (P1, 25-40% area), 3 (P0).
"""
import csv
import glob
import os
import shutil
import cv2
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']

from build_multi_synth import (
    find_orig_path,
    save_with_realistic_compression,
    make_high_detail_mask,
    synth_motion_defocus_blur,
)


def synth_subtype1_degrade_then_hallucinate(orig_bgr, gen_bgr, grade, rng):
    """Subtype 1: Blur/degrade base image -> composite neural hallucinated plastic ridges from FLUX3/Swin2SR."""
    h, w = orig_bgr.shape[:2]
    if gen_bgr.shape[:2] != (h, w):
        gen_bgr = cv2.resize(gen_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    blur_grade = 'P2' if grade == 'P2' or rng.rand() < 0.45 else 'P1'
    blurred_orig, b_labels = synth_motion_defocus_blur(orig_bgr, blur_grade, rng)

    # Extract plastic surface + hallucinated high-frequency ridges from FLUX3/Swin2SR
    gen_f = gen_bgr.astype(np.float32)
    gen_low = cv2.GaussianBlur(gen_f, (0, 0), sigmaX=2.2)
    gen_high = gen_f - gen_low
    plastic_base = cv2.bilateralFilter(gen_bgr, d=9, sigmaColor=42, sigmaSpace=42).astype(np.float32)
    halluc_bgr = np.clip(plastic_base + rng.uniform(1.45, 1.85) * gen_high, 0, 255)

    if grade == 'P2':
        mask = make_high_detail_mask(orig_bgr, area_frac=rng.uniform(0.28, 0.42), feather_ratio=0.06, rng=rng)
        w_map = np.clip(0.35 + 0.55 * mask, 0.0, 0.88)
        out = blurred_orig.astype(np.float32) * (1.0 - w_map) + halluc_bgr * w_map
        og_label = 1
    elif grade == 'P1':
        mask = make_high_detail_mask(orig_bgr, area_frac=rng.uniform(0.55, 0.75), feather_ratio=0.05, rng=rng)
        w_map = np.clip(0.55 + 0.40 * mask, 0.0, 0.95)
        out = blurred_orig.astype(np.float32) * (1.0 - w_map) + halluc_bgr * w_map
        og_label = 2
    else:  # P0
        out = (0.25 * blurred_orig.astype(np.float32) + 0.75 * halluc_bgr).astype(np.float32)
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
        mask = make_high_detail_mask(orig_bgr, area_frac=rng.uniform(0.35, 0.55), feather_ratio=0.05, rng=rng)[..., 0]
        amp = rng.uniform(2.5, 4.5)
        freq = rng.uniform(24.0, 44.0)
        dx = (amp * np.sin(2.0 * np.pi * yy / freq) * mask).astype(np.float32)
        dy = (amp * np.cos(2.0 * np.pi * xx / freq) * mask).astype(np.float32)
        out = cv2.remap(out, xx + dx, yy + dy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        og_label = 3

    labels = {
        '模糊': b_labels['模糊'],
        '纹理损失': 2,
        '过锐': 0,
        '花屏': 0,
        '过度生成': og_label,
        '虚化突兀': 0,
    }
    return np.clip(out, 0, 255).astype(np.uint8), labels


def synth_subtype2_ai_plus_isp_sharp(orig_bgr, gen_bgr, grade, rng):
    """Subtype 2: AI re-synthesis + ISP USM edge haloing (matches 59% of real 过度生成_P2)."""
    h, w = orig_bgr.shape[:2]
    if gen_bgr.shape[:2] != (h, w):
        gen_bgr = cv2.resize(gen_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    if grade == 'P2':
        mask = make_high_detail_mask(orig_bgr, area_frac=rng.uniform(0.30, 0.45), feather_ratio=0.06, rng=rng)
        w_map = np.clip(0.40 + 0.50 * mask, 0.0, 0.88)
        blended = (orig_bgr.astype(np.float32) * (1.0 - w_map) + gen_bgr.astype(np.float32) * w_map).astype(np.uint8)
        og_label = 1
        tex_label = 1 if rng.rand() < 0.55 else 2
    else:  # P1
        mask = make_high_detail_mask(orig_bgr, area_frac=rng.uniform(0.55, 0.78), feather_ratio=0.05, rng=rng)
        w_map = np.clip(0.65 + 0.30 * mask, 0.0, 0.95)
        blended = (orig_bgr.astype(np.float32) * (1.0 - w_map) + gen_bgr.astype(np.float32) * w_map).astype(np.uint8)
        og_label = 2
        tex_label = 2

    lab = cv2.cvtColor(blended, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_f = l_chan.astype(np.float32)
    sigma = rng.uniform(1.0, 1.8)
    amt = rng.uniform(1.6, 2.5) if rng.rand() < 0.75 else rng.uniform(2.6, 3.5)
    sharp_label = 1 if amt < 2.55 else 2

    blur_l = cv2.GaussianBlur(l_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    diff = np.clip(amt * (l_f - blur_l), -32.0, 36.0)
    l_new = np.clip(l_f + diff, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([l_new, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

    labels = {
        '模糊': 0,
        '纹理损失': tex_label,
        '过锐': sharp_label,
        '花屏': 0,
        '过度生成': og_label,
        '虚化突兀': 0,
    }
    return out, labels


def graft_real_overgen_patch(target_bgr, target_labels, donor_bgr, grade, rng):
    """Subtype 3: Graft high-detail AI-corrupted patch from a real training 过度生成_P0/P1 donor."""
    th, tw = target_bgr.shape[:2]
    dh, dw = donor_bgr.shape[:2]

    if grade == 'P2':
        area_frac = rng.uniform(0.12, 0.20)
        og_label = 1
    elif grade == 'P1':
        area_frac = rng.uniform(0.24, 0.36)
        og_label = 2
    else:
        area_frac = rng.uniform(0.38, 0.52)
        og_label = 3

    pw = int(np.clip(tw * np.sqrt(area_frac) * rng.uniform(0.85, 1.15), 64, tw - 16))
    ph = int(np.clip(th * np.sqrt(area_frac) * rng.uniform(0.85, 1.15), 64, th - 16))

    if dw <= pw + 8 or dh <= ph + 8:
        donor_bgr = cv2.resize(donor_bgr, (max(dw, pw + 32), max(dh, ph + 32)), interpolation=cv2.INTER_LANCZOS4)
        dh, dw = donor_bgr.shape[:2]

    d_gray = cv2.cvtColor(donor_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(d_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d_gray, cv2.CV_32F, 0, 1, ksize=3)
    d_energy = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), sigmaX=16)
    valid_e = d_energy[ph // 2:dh - ph // 2, pw // 2:dw - pw // 2]
    if valid_e.size > 0:
        ey, ex = divmod(int(np.argmax(valid_e)), valid_e.shape[1])
        x0 = int(np.clip(ex + rng.randint(-20, 21), 0, dw - pw))
        y0 = int(np.clip(ey + rng.randint(-20, 21), 0, dh - ph))
    else:
        x0 = rng.randint(0, max(1, dw - pw))
        y0 = rng.randint(0, max(1, dh - ph))

    patch = donor_bgr[y0:y0 + ph, x0:x0 + pw].copy()

    tx0 = int(rng.randint(8, max(9, tw - pw - 8)))
    ty0 = int(rng.randint(8, max(9, th - ph - 8)))
    target_roi = target_bgr[ty0:ty0 + ph, tx0:tx0 + pw]

    p_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_lab = cv2.cvtColor(target_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        pm, ps = p_lab[..., c].mean(), p_lab[..., c].std() + 1e-5
        tm, ts = t_lab[..., c].mean(), t_lab[..., c].std() + 1e-5
        p_lab[..., c] = (p_lab[..., c] - pm) * np.clip(ts / ps, 0.6, 1.5) + tm
    patch_matched = cv2.cvtColor(np.clip(p_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    mask = np.zeros((ph, pw), dtype=np.float32)
    cv2.ellipse(mask, (pw // 2, ph // 2), (int(pw * 0.44), int(ph * 0.44)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(6.0, min(pw, ph) * 0.10))[..., None]

    out = target_bgr.copy()
    roi_f = target_roi.astype(np.float32)
    pat_f = patch_matched.astype(np.float32)
    out[ty0:ty0 + ph, tx0:tx0 + pw] = np.clip(roi_f * (1.0 - mask) + pat_f * mask, 0, 255).astype(np.uint8)

    labels = dict(target_labels)
    labels['过度生成'] = og_label
    labels['纹理损失'] = max(int(labels['纹理损失']), 1 if grade == 'P2' else 2)
    return out, labels


def build_overgen_pro_for_split(split_csv, out_dir, out_split_csv, out_manifest_csv, seed=77, scale=1.0):
    rng = np.random.RandomState(seed)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
    split_rows = list(csv.DictReader(open(split_csv)))
    train_names = {r['name'] for r in split_rows if r['split'] == 'train'}

    tr_overgen_donors = sorted([
        n for n in train_names
        if int(man[n]['过度生成']) in (2, 3) and int(man[n]['花屏']) == 0 and int(man[n]['虚化突兀']) == 0
    ])
    tr_recipients = sorted([
        n for n in train_names
        if int(man[n]['过度生成']) == 0 and int(man[n]['花屏']) == 0 and int(man[n]['虚化突兀']) == 0
        and int(man[n]['纹理损失']) in (1, 2) and int(man[n]['模糊']) in (0, 1, 2) and int(man[n]['过锐']) in (0, 1)
    ])

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

    gen_pool = [(s, flx_map[s], 'flx') for s in sorted(flx_map)] + [(s, sr_map[s], 'sr') for s in sorted(sr_map)]
    rng.shuffle(gen_pool)

    new_manifest = dict(man)
    new_split = list(split_rows)
    counts = {}

    def register(sname, img_bgr, labels):
        spath = os.path.join(out_dir, sname)
        save_with_realistic_compression(img_bgr, spath, rng)
        new_split.append({'name': sname, 'split': 'train', 'source': 'synth_overgen_pro'})
        row = {'name': sname}
        row.update(labels)
        new_manifest[sname] = row

    # 1. Subtype 1: Degrade-Blur-then-Hallucinate (30 P2 + 35 P1 + 20 P0)
    n1_p2, n1_p1, n1_p0 = int(round(30 * scale)), int(round(35 * scale)), int(round(20 * scale))
    for i, (src, gpath, tag) in enumerate(gen_pool[:n1_p2 + n1_p1 + n1_p0]):
        opath = find_orig_path(src)
        if not opath:
            continue
        orig_bgr = cv2.imread(opath)
        gen_bgr = cv2.imread(gpath)
        if orig_bgr is None or gen_bgr is None:
            continue
        grade = 'P2' if i < n1_p2 else ('P1' if i < n1_p2 + n1_p1 else 'P0')
        sname = f"v7_og_sub1_{grade}_{tag}_{src}"
        out_bgr, labels = synth_subtype1_degrade_then_hallucinate(orig_bgr, gen_bgr, grade, rng)
        register(sname, out_bgr, labels)
        counts[f'Sub1_{grade}'] = counts.get(f'Sub1_{grade}', 0) + 1

    # 2. Subtype 2: AI Re-render + ISP Edge Oversharpening (30 P2 + 25 P1)
    rng.shuffle(gen_pool)
    n2_p2, n2_p1 = int(round(30 * scale)), int(round(25 * scale))
    for i, (src, gpath, tag) in enumerate(gen_pool[:n2_p2 + n2_p1]):
        opath = find_orig_path(src)
        if not opath:
            continue
        orig_bgr = cv2.imread(opath)
        gen_bgr = cv2.imread(gpath)
        if orig_bgr is None or gen_bgr is None:
            continue
        grade = 'P2' if i < n2_p2 else 'P1'
        sname = f"v7_og_sub2_{grade}_{tag}_{src}"
        out_bgr, labels = synth_subtype2_ai_plus_isp_sharp(orig_bgr, gen_bgr, grade, rng)
        register(sname, out_bgr, labels)
        counts[f'Sub2_{grade}'] = counts.get(f'Sub2_{grade}', 0) + 1

    # 3. Subtype 3: Real 过度生成_P0/P1 Patch Grafting onto recipients (25 P2 + 25 P1 + 15 P0)
    n3_p2, n3_p1, n3_p0 = int(round(25 * scale)), int(round(25 * scale)), int(round(15 * scale))
    rec_pool = list(rng.permutation(tr_recipients))[:n3_p2 + n3_p1 + n3_p0]
    don_pool = list(rng.permutation(tr_overgen_donors))
    for i, rname in enumerate(rec_pool):
        dname = don_pool[i % len(don_pool)]
        rpath, dpath = find_orig_path(rname), find_orig_path(dname)
        if not rpath or not dpath:
            continue
        rbgr, dbgr = cv2.imread(rpath), cv2.imread(dpath)
        if rbgr is None or dbgr is None:
            continue
        grade = 'P2' if i < n3_p2 else ('P1' if i < n3_p2 + n3_p1 else 'P0')
        r_labels = {c: int(man[rname][c]) for c in CLASSES}
        sname = f"v7_og_sub3_{grade}_{rname}"
        out_bgr, labels = graft_real_overgen_patch(rbgr, r_labels, dbgr, grade, rng)
        register(sname, out_bgr, labels)
        counts[f'Sub3_{grade}'] = counts.get(f'Sub3_{grade}', 0) + 1

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

    print(f"Built {out_split_csv}: total added={sum(counts.values())} -> {counts}", flush=True)


if __name__ == '__main__':
    build_overgen_pro_for_split(
        'data/resplits/split_s16.csv',
        'data/synth_overgen_pro_s16',
        'data/resplits/split_s16_overgen_pro.csv',
        'data/manifest_s16_overgen_pro.csv',
        seed=1607, scale=1.0
    )
    build_overgen_pro_for_split(
        'data/resplits/split_s17.csv',
        'data/synth_overgen_pro_s17',
        'data/resplits/split_s17_overgen_pro.csv',
        'data/manifest_s17_overgen_pro.csv',
        seed=1707, scale=1.0
    )
    build_overgen_pro_for_split(
        'data/split_all_full.csv',
        'data/synth_overgen_pro_full',
        'data/split_all_full_overgen_pro.csv',
        'data/manifest_combined_overgen_pro.csv',
        seed=2027, scale=1.25
    )
