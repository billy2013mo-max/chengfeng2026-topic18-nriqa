"""Generate 200 synthetic over-sharpened images (100 P2, 100 P1)
from verified Clean images across the full training dataset.
"""
import csv, os, sys
import cv2
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')

from synth_sharp_sample import apply_isp_oversharpen, find_img_path

OUT_DIR = 'data/synth_sharp_full'
os.makedirs(OUT_DIR, exist_ok=True)

man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
full_rows = list(csv.DictReader(open('data/split_all_full.csv')))

# Find all clean images in full train set
clean_names = [
    r['name'] for r in full_rows
    if all(int(man[r['name']][c]) == 0 for c in ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀'])
]
print(f"Total clean images in full pool: {len(clean_names)}")

selected = clean_names[:100]
new_manifest_rows = dict(man)
new_full_rows = list(full_rows)

count = 0
for idx, name in enumerate(selected):
    path = find_img_path(name)
    if not path:
        continue
    img = cv2.imread(path)
    if img is None:
        continue

    # P2 (slight ringing)
    p2_name = f"full_synth_sharp_P2_{idx:03d}_{name}"
    p2_path = os.path.join(OUT_DIR, p2_name)
    if not os.path.exists(p2_path):
        cv2.imwrite(p2_path, apply_isp_oversharpen(img, 'P2'))
    new_full_rows.append({'name': p2_name, 'split': 'train', 'source': 'synthetic'})
    new_manifest_rows[p2_name] = {
        'name': p2_name, '模糊': 0, '纹理损失': 0, '过锐': 1, '花屏': 0, '过度生成': 0, '虚化突兀': 0
    }

    # P1 (obvious halo/ringing)
    p1_name = f"full_synth_sharp_P1_{idx:03d}_{name}"
    p1_path = os.path.join(OUT_DIR, p1_name)
    if not os.path.exists(p1_path):
        cv2.imwrite(p1_path, apply_isp_oversharpen(img, 'P1'))
    new_full_rows.append({'name': p1_name, 'split': 'train', 'source': 'synthetic'})
    new_manifest_rows[p1_name] = {
        'name': p1_name, '模糊': 0, '纹理损失': 0, '过锐': 2, '花屏': 0, '过度生成': 0, '虚化突兀': 0
    }
    count += 2

print(f"Generated {count} full synthetic over-sharpened images ({count//2} P2, {count//2} P1).")

# Save updated split and manifest
with open('data/split_all_full_synth.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'split', 'source'])
    writer.writeheader()
    writer.writerows(new_full_rows)

classes_fields = ['name', '模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
with open('data/manifest_combined_synth.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=classes_fields)
    writer.writeheader()
    for k, v in new_manifest_rows.items():
        writer.writerow({c: v[c] for c in classes_fields})

print(f"Saved: data/split_all_full_synth.csv (Total rows: {len(new_full_rows)})")
print(f"Saved: data/manifest_combined_synth.csv")
