"""SigLIP2 NaFlex feature extraction (native aspect ratio, variable patch count)."""
import argparse, csv, os, numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--manifest', required=True)
ap.add_argument('--imgdir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--patches', type=int, default=576); ap.add_argument('--batch', type=int, default=8)
a = ap.parse_args()
m = AutoModel.from_pretrained(a.model, attn_implementation='sdpa', dtype=torch.bfloat16).eval().cuda()
vis = m.vision_model if hasattr(m, 'vision_model') else m
proc = AutoProcessor.from_pretrained(a.model)
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
    inp = proc.image_processor(images=views, return_tensors='pt', max_num_patches=a.patches)
    inp = {k: v.cuda() for k, v in inp.items()}
    with torch.inference_mode():
        o = vis(**inp)
    t = o.last_hidden_state.float()
    mask = inp['pixel_attention_mask'].unsqueeze(-1).float()
    mean = (t * mask).sum(1) / mask.sum(1).clamp(min=1)
    mx = t.masked_fill(mask == 0, -1e4).amax(1)
    per = torch.cat([mean, mx], 1).view(len(chunk), 5, -1)
    f = torch.cat([per[:,0], per[:,1:].mean(1), per[:,1:].amax(1)], 1)
    feats.append(f.cpu().numpy()); names += [r['name'] for r in chunk]
    if (i//a.batch) % 20 == 0: print(i, flush=True)
X = np.concatenate(feats, 0)
np.savez_compressed(a.out, X=X.astype(np.float32), names=np.array(names))
print('saved', a.out, X.shape)
