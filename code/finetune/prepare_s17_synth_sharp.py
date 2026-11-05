import csv, os, cv2
os.chdir('/wanqing-develop/luowenjing/18')
OUT_DIR = 'data/synth_sharp_s17'
os.makedirs(OUT_DIR, exist_ok=True)
from synth_sharp_sample import apply_isp_oversharpen, find_img_path

man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
s17_rows = list(csv.DictReader(open('data/resplits/split_s17.csv')))
tr_clean = [
    r['name'] for r in s17_rows
    if r['split'] == 'train' and all(int(man[r['name']][c]) == 0 for c in ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀'])
]
selected = tr_clean[:75]
new_manifest_rows = dict(man)
new_s17_rows = list(s17_rows)

for name in selected:
    path = find_img_path(name)
    if not path: continue
    img = cv2.imread(path)
    if img is None: continue
    p2_name = f"synth_sharp_P2_{name}"
    cv2.imwrite(os.path.join(OUT_DIR, p2_name), apply_isp_oversharpen(img, 'P2'))
    new_s17_rows.append({'name': p2_name, 'split': 'train', 'source': 'synthetic'})
    new_manifest_rows[p2_name] = {'name': p2_name, '模糊': 0, '纹理损失': 0, '过锐': 1, '花屏': 0, '过度生成': 0, '虚化突兀': 0}

    p1_name = f"synth_sharp_P1_{name}"
    cv2.imwrite(os.path.join(OUT_DIR, p1_name), apply_isp_oversharpen(img, 'P1'))
    new_s17_rows.append({'name': p1_name, 'split': 'train', 'source': 'synthetic'})
    new_manifest_rows[p1_name] = {'name': p1_name, '模糊': 0, '纹理损失': 0, '过锐': 2, '花屏': 0, '过度生成': 0, '虚化突兀': 0}

with open('data/resplits/split_s17_synth_sharp.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'split', 'source'])
    writer.writeheader(); writer.writerows(new_s17_rows)

classes_fields = ['name', '模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
with open('data/manifest_s17_synth_sharp.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=classes_fields)
    writer.writeheader()
    for k, v in new_manifest_rows.items():
        writer.writerow({c: v[c] for c in classes_fields})
print("s17 synth sharp ready!")
