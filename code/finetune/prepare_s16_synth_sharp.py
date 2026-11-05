"""Generate synthetic over-sharpened images (75 P2 + 75 P1 = 150 images)
STRICTLY from s16 train-set clean images, maintaining 100% zero-leakage into val or test.
"""
import csv, os
import cv2
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')

OUT_DIR = 'data/synth_sharp_s16'
os.makedirs(OUT_DIR, exist_ok=True)

from synth_sharp_sample import apply_isp_oversharpen, find_img_path

man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
s16_rows = list(csv.DictReader(open('data/resplits/split_s16.csv')))

# Only take clean images that strictly belong to s16 train
tr_clean = [
    r['name'] for r in s16_rows
    if r['split'] == 'train' and all(int(man[r['name']][c]) == 0 for c in ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀'])
]
print(f"Total clean images in s16 train: {len(tr_clean)}")

selected = tr_clean[:75]
new_manifest_rows = dict(man)
new_s16_rows = list(s16_rows)

count = 0
for name in selected:
    path = find_img_path(name)
    if not path:
        continue
    img = cv2.imread(path)
    if img is None:
        continue

    # P2
    p2_name = f"synth_sharp_P2_{name}"
    p2_path = os.path.join(OUT_DIR, p2_name)
    if not os.path.exists(p2_path):
        cv2.imwrite(p2_path, apply_isp_oversharpen(img, 'P2'))
    new_s16_rows.append({'name': p2_name, 'split': 'train', 'source': 'synthetic'})
    new_manifest_rows[p2_name] = {
        'name': p2_name, '模糊': 0, '纹理损失': 0, '过锐': 1, '花屏': 0, '过度生成': 0, '虚化突兀': 0
    }

    # P1
    p1_name = f"synth_sharp_P1_{name}"
    p1_path = os.path.join(OUT_DIR, p1_name)
    if not os.path.exists(p1_path):
        cv2.imwrite(p1_path, apply_isp_oversharpen(img, 'P1'))
    new_s16_rows.append({'name': p1_name, 'split': 'train', 'source': 'synthetic'})
    new_manifest_rows[p1_name] = {
        'name': p1_name, '模糊': 0, '纹理损失': 0, '过锐': 2, '花屏': 0, '过度生成': 0, '虚化突兀': 0
    }
    count += 2

print(f"Generated and registered {count} synthetic images ({count//2} P2, {count//2} P1)")

# Save new split and manifest
with open('data/resplits/split_s16_synth_sharp.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'split', 'source'])
    writer.writeheader()
    writer.writerows(new_s16_rows)

classes_fields = ['name', '模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
with open('data/manifest_s16_synth_sharp.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=classes_fields)
    writer.writeheader()
    for k, v in new_manifest_rows.items():
        writer.writerow({c: v[c] for c in classes_fields})

print(f"Saved: data/resplits/split_s16_synth_sharp.csv (total {len(new_s16_rows)} rows)")
print(f"Saved: data/manifest_s16_synth_sharp.csv")
