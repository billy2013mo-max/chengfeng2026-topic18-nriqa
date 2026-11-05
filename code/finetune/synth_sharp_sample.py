"""Generate realistic ISP over-sharpening samples (P2 and P1) on verified Clean images.
Uses physical Unsharp Masking (USM) with edge overshoot clipping to simulate
camera ISP ringing/halo artifacts without corrupting color or overall exposure.
"""
import csv, os, sys
import cv2
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')

OUT_DIR = 'data/synth_sharp_sample'
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Identify verified Clean images from manifest_combined
man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
rows = list(csv.DictReader(open('data/split_all_full.csv')))
clean_names = [
    r['name'] for r in rows
    if all(int(man[r['name']][c]) == 0 for c in ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀'])
]
print(f"Found {len(clean_names)} clean images in train pool.")

def find_img_path(name):
    for d in ['data/images', 'data/images_supp', 'data/test_150']:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

def apply_isp_oversharpen(img_bgr, severity='P2'):
    """
    Simulate camera ISP edge over-sharpening in LAB/Luminance space.
    - P2 (slight ringing/halos around sharp edges): radius=1.2, amount=2.0, threshold=4
    - P1 (moderate/obvious white ringing halos): radius=2.2, amount=4.2, threshold=3
    """
    # Work on luminance channel to prevent chromatic aberration/color shift
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)

    if severity == 'P2':
        ksize = 5
        sigma = 1.2
        amount = 2.0
        threshold = 4.0
    elif severity == 'P1':
        ksize = 9
        sigma = 2.2
        amount = 4.2
        threshold = 3.0
    else:
        raise ValueError(f"Unknown severity: {severity}")

    blurred = cv2.GaussianBlur(l_float, (ksize, ksize), sigma)
    diff = l_float - blurred

    # Only amplify when difference exceeds noise threshold (edge ringing)
    mask = np.abs(diff) >= threshold
    sharpened = l_float.copy()
    sharpened[mask] = l_float[mask] + amount * diff[mask]

    # Clip luminance to [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    lab_sharp = cv2.merge([sharpened, a, b])
    res_bgr = cv2.cvtColor(lab_sharp, cv2.COLOR_LAB2BGR)
    return res_bgr

# 2. Generate 15 pairs (30 images total) for inspection
selected = clean_names[:15]
stats = []

for idx, name in enumerate(selected):
    path = find_img_path(name)
    if not path:
        continue
    img = cv2.imread(path)
    if img is None:
        continue

    # P2
    p2_img = apply_isp_oversharpen(img, 'P2')
    p2_name = f"synth_sharp_P2_{idx:03d}_{name}"
    cv2.imwrite(os.path.join(OUT_DIR, p2_name), p2_img)

    # P1
    p1_img = apply_isp_oversharpen(img, 'P1')
    p1_name = f"synth_sharp_P1_{idx:03d}_{name}"
    cv2.imwrite(os.path.join(OUT_DIR, p1_name), p1_img)

    # Difference metrics
    diff_p2 = np.abs(p2_img.astype(float) - img.astype(float)).mean()
    diff_p1 = np.abs(p1_img.astype(float) - img.astype(float)).mean()
    stats.append({
        'base': name,
        'p2_name': p2_name,
        'p1_name': p1_name,
        'diff_p2': diff_p2,
        'diff_p1': diff_p1,
    })

print(f"\nGenerated {len(stats)*2} synthetic over-sharpened images in: {OUT_DIR}")
print(f"Mean pixel deviation vs original: P2 = {np.mean([s['diff_p2'] for s in stats]):.2f}, P1 = {np.mean([s['diff_p1'] for s in stats]):.2f}")
for s in stats[:5]:
    print(f"  [{s['base'][:25]}...] P2 diff={s['diff_p2']:.2f} | P1 diff={s['diff_p1']:.2f}")
