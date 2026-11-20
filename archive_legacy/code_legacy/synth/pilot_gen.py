"""Pilot: synthesise 'pure over-generation' (AI upscaler) variants of clean images.

Same content, same size; only the texture is re-synthesised by an AI super-
resolution network, so the label should change only for 过度生成 (no blur, no
texture-loss degradation of the kind caused by downscale/compression).

Strength ladder:
  p1 = 2x SR -> resize back to original size          (mild)
  p2 = two passes of that                            (stronger)
"""
import argparse, csv, os, sys, numpy as np, torch
from PIL import Image
from torchvision import transforms
from transformers import Swin2SRForImageSuperResolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CL = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--out', default='/tmp/synth/out')
ap.add_argument('--src-list', default='/tmp/synth/src.csv')
ap.add_argument('--device', default='cuda:0')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
train = [r['name'] for r in csv.DictReader(open('data/split_final.csv')) if r['split'] == 'train']
clean = [n for n in train if all(int(man[n][c]) == 0 for c in CL)]

# stratify by resolution bucket
def rb(n):
    m = man[n]; x = min(int(m['width']), int(m['height']))
    return '<720' if x < 720 else ('720-1080' if x < 1080 else ('1080-2160' if x < 2160 else '>=2160'))
buckets = {}
for n in clean:
    buckets.setdefault(rb(n), []).append(n)
print({k: len(v) for k, v in buckets.items()}, flush=True)
rng = np.random.RandomState(0)
pick = []
frac = {'<720': .07, '720-1080': .61, '1080-2160': .29, '>=2160': .04}
for k, v in buckets.items():
    take = max(1, int(round(a.n * frac.get(k, 0))))
    pick += list(rng.permutation(sorted(v))[:min(take, len(v))])
pick = pick[:a.n]
print('picked', len(pick), flush=True)

with open(a.src_list, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['name', 'bucket', 'w', 'h'])
    for n in pick:
        m = man[n]; w.writerow([n, rb(n), m['width'], m['height']])

model = Swin2SRForImageSuperResolution.from_pretrained('caidas/swin2SR-classical-sr-x2-64').to(a.device).eval()


@torch.no_grad()
def sr2x(im):
    """Tiled 2x super-resolution with overlap; returns a 2x PIL image."""
    W, H = im.size
    tile, ov = 256, 16
    out = np.zeros((H * 2, W * 2, 3), np.float64)
    cnt = np.zeros((H * 2, W * 2, 1), np.float64)
    xs = list(range(0, max(1, W - ov), tile - ov)) or [0]
    ys = list(range(0, max(1, H - ov), tile - ov)) or [0]
    tiles, pos = [], []
    for y in ys:
        for x in xs:
            box = (x, y, min(x + tile, W), min(y + tile, H))
            if box[2] - box[0] < 32 or box[3] - box[1] < 32:
                continue
            t = np.asarray(im.crop(box), dtype=np.float32) / 255.
            th, tw = t.shape[:2]
            tt = torch.from_numpy(t).permute(2, 0, 1)
            tt = torch.nn.functional.pad(tt, (0, tile - tw, 0, tile - th), mode='replicate')
            tiles.append(tt); pos.append(box)
    for i in range(0, len(tiles), 6):
        batch = torch.stack(tiles[i:i + 6]).to(a.device)
        o = model(batch).reconstruction.data.clamp(0, 1).float().cpu().numpy()
        for j, box in enumerate(pos[i:i + 6]):
            x0, y0, x1, y1 = box
            oy0, oy1, ox0, ox1 = y0 * 2, y1 * 2, x0 * 2, x1 * 2
            out[oy0:oy1, ox0:ox1] += np.transpose(o[j], (1, 2, 0))[:oy1 - oy0, :ox1 - ox0]
            cnt[oy0:oy1, ox0:ox1] += 1
    out = out / np.maximum(cnt, 1)
    return Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8))


def load(n):
    p = f'data/images/{n}'
    if not os.path.exists(p):
        p = f'data/images_supp/{n}'
    return Image.open(p).convert('RGB')


for i, n in enumerate(pick):
    im = load(n)
    W, H = im.size
    x1 = sr2x(im).resize((W, H), Image.LANCZOS)          # pass 1
    x2 = sr2x(x1).resize((W, H), Image.LANCZOS)          # pass 2
    x1.save(os.path.join(a.out, 'p1_' + n.replace('/', '_')))
    x2.save(os.path.join(a.out, 'p2_' + n.replace('/', '_')))
    if (i + 1) % 10 == 0:
        print(f'{i+1}/{len(pick)}', flush=True)
print('done', len(pick), 'x 2 variants ->', a.out)
