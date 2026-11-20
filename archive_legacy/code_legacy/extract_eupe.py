"""EUPE feature extraction (FAIR Noncommercial Research License)."""
import argparse, csv, os, sys, numpy as np, torch
from PIL import Image
from torchvision.transforms import v2

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='/tmp/race18/eupe_repo')
ap.add_argument('--weights', required=True)
ap.add_argument('--hub', default='eupe_vitb16')
ap.add_argument('--manifest', required=True); ap.add_argument('--imgdir', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--size', type=int, default=256)
ap.add_argument('--batch', type=int, default=16); ap.add_argument('--gpu', default='cuda')
a = ap.parse_args()
m = torch.hub.load(a.repo, a.hub, source='local', weights=a.weights).eval().to(a.gpu)
tf = v2.Compose([v2.ToImage(), v2.Resize((a.size, a.size), antialias=True),
                 v2.ToDtype(torch.float32, scale=True),
                 v2.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])
rows = list(csv.DictReader(open(a.manifest)))
feats, names = [], []
for i in range(0, len(rows), a.batch):
    chunk = rows[i:i+a.batch]; views = []
    for r in chunk:
        im = Image.open(os.path.join(a.imgdir, r['name'])).convert('RGB'); w, h = im.size
        views.append(im)
        for dy in (0,1):
            for dx in (0,1):
                views.append(im.crop((dx*w//2, dy*h//2, (dx+1)*w//2, (dy+1)*h//2)))
    batch = torch.stack([tf(v) for v in views]).to(a.gpu)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        o = m.forward_features(batch)
    cls = o['x_norm_clstoken'].float(); pat = o['x_norm_patchtokens'].float()
    per = torch.cat([cls, pat.mean(1), pat.amax(1)], 1)           # Nv, 3D
    per = per.view(len(chunk), 5, -1)
    f = torch.cat([per[:,0], per[:,1:].mean(1), per[:,1:].amax(1)], 1)
    feats.append(f.cpu().numpy()); names += [r['name'] for r in chunk]
    if (i//a.batch) % 20 == 0: print(i, flush=True)
X = np.concatenate(feats, 0)
np.savez_compressed(a.out, X=X.astype(np.float32), names=np.array(names))
print('saved', a.out, X.shape)
