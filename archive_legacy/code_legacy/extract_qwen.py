import argparse, csv, os, numpy as np, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='Qwen/Qwen3-VL-2B-Instruct')
ap.add_argument('--manifest', required=True); ap.add_argument('--imgdir', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--size', type=int, default=448); ap.add_argument('--gpu', default='cuda')
a = ap.parse_args()
proc = AutoProcessor.from_pretrained(a.repo)
m = AutoModelForImageTextToText.from_pretrained(a.repo, dtype=torch.bfloat16, device_map=a.gpu)
try:
    del m.model.language_model; torch.cuda.empty_cache()
except Exception as e:
    print('keep lm', e)
vis = m.model.visual.eval()
rows = list(csv.DictReader(open(a.manifest)))
feats, names = [], []
px = a.size * a.size
for i in range(0, len(rows), a.batch):
    chunk = rows[i:i+a.batch]; views = []
    for r in chunk:
        im = Image.open(os.path.join(a.imgdir, r['name'])).convert('RGB')
        w, h = im.size
        views.append(im.resize((a.size, a.size), Image.BILINEAR))
        for dy in (0,1):
            for dx in (0,1):
                views.append(im.crop((dx*w//2, dy*h//2, (dx+1)*w//2, (dy+1)*h//2)).resize((a.size, a.size), Image.BILINEAR))
    inp = proc.image_processor(images=views, return_tensors='pt', min_pixels=px, max_pixels=px)
    inp = {k: v.to(a.gpu) for k, v in inp.items()}
    with torch.no_grad():
        out = vis(hidden_states=inp['pixel_values'], grid_thw=inp['image_grid_thw'])
    tok = out.last_hidden_state.float()
    # split per image by token counts
    thw = inp['image_grid_thw'].cpu().numpy()
    counts = (thw[:,0]*thw[:,1]*thw[:,2]).astype(int)
    assert sum(counts)==tok.shape[0], (sum(counts), tok.shape)
    off=0; per=[]
    for c in counts:
        per.append(torch.cat([tok[off:off+c].mean(0), tok[off:off+c].amax(0)]).cpu().numpy()); off+=c
    per=np.stack(per).reshape(len(chunk),5,-1)
    feats.append(np.concatenate([per[:,0], per[:,1:].mean(1)],1))
    names += [r['name'] for r in chunk]
    if (i//a.batch)%10==0: print(i, flush=True)
X=np.concatenate(feats,0)
np.savez_compressed(a.out, X=X.astype(np.float32), names=np.array(names))
print('saved', a.out, X.shape)
