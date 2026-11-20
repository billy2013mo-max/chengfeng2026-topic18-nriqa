"""InternViT-6B feature extraction via direct instantiation (bypasses transformers 5.x remote-code issue)."""
import argparse, csv, glob, os, sys, numpy as np, torch
from PIL import Image
sys.path.insert(0, '/tmp/race18')
from internvit_pkg.configuration_intern_vit import InternVisionConfig
from internvit_pkg.modeling_intern_vit import InternVisionModel
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument('--model-dir', required=True); ap.add_argument('--manifest', required=True)
ap.add_argument('--imgdir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--size', type=int, default=448); ap.add_argument('--batch', type=int, default=4)
a = ap.parse_args()
cfg = InternVisionConfig.from_pretrained(a.model_dir)
m = InternVisionModel(cfg)
sd = {}
for f in sorted(glob.glob(os.path.join(a.model_dir, 'model-*.safetensors'))): sd.update(load_file(f))
r = m.load_state_dict(sd, strict=False); print('missing', len(r.missing_keys), 'unexpected', len(r.unexpected_keys))
m = m.to(torch.bfloat16).cuda().eval()
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda()
std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
def prep(im):
    im = im.resize((a.size, a.size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2,0,1)/255.
    return x
rows = list(csv.DictReader(open(a.manifest)))
feats, names = [], []
for i in range(0, len(rows), a.batch):
    chunk = rows[i:i+a.batch]; views = []
    for r_ in chunk:
        im = Image.open(os.path.join(a.imgdir, r_['name'])).convert('RGB'); w,h = im.size
        views.append(prep(im))
        for dy in (0,1):
            for dx in (0,1):
                views.append(prep(im.crop((dx*w//2, dy*h//2, (dx+1)*w//2, (dy+1)*h//2))))
    x = torch.stack(views).cuda()
    x = (x - mean)/std
    with torch.inference_mode():
        o = m(pixel_values=x.to(torch.bfloat16))
    t = o.last_hidden_state.float()          # B, 1+1024, D
    per = torch.cat([t[:,0], t[:,1:].mean(1), t[:,1:].amax(1)], 1).view(len(chunk), 5, -1)
    f = torch.cat([per[:,0], per[:,1:].mean(1), per[:,1:].amax(1)], 1)
    feats.append(f.cpu().numpy()); names += [r_['name'] for r_ in chunk]
    if (i//a.batch) % 20 == 0: print(i, flush=True)
X = np.concatenate(feats, 0)
np.savez_compressed(a.out, X=X.astype(np.float32), names=np.array(names))
print('saved', a.out, X.shape)
