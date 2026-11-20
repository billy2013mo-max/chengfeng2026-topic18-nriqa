"""Build manifest.csv with labels, source group and resolution."""
import openpyxl, csv, os, sys, collections
from PIL import Image

xlsx, imgdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
CLASSES = ['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']
GRADE = {'无问题': 0, 'P2': 1, 'P1': 2, 'P0': 3}

def parse(v):
    v = str(v).strip()
    if v == '无问题': return 0
    return GRADE[v[:2]]

wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
ws = wb.active
rows = list(ws.iter_rows(values_only=True))
hdr = rows[0]
assert list(hdr[1:])[0] == '模糊', hdr
idx = {c: list(hdr).index(c) for c in CLASSES}

recs = []
for r in rows[1:]:
    name = r[0]
    path = os.path.join(imgdir, name)
    if not os.path.exists(path):
        print('MISSING', name); continue
    w, h = Image.open(path).size
    src = name.split('_')[0]
    rec = {'name': name, 'source': src, 'width': w, 'height': h}
    for c in CLASSES:
        rec[c] = parse(r[idx[c]])
    recs.append(rec)

os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
    w.writeheader(); w.writerows(recs)
print('wrote', out, len(recs))
for c in CLASSES:
    print(c, collections.Counter(x[c] for x in recs))
