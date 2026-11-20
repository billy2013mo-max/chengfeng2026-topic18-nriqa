"""C-RADIOv4-H feature extraction (NVIDIA Open Model License)."""
import argparse, csv, os, sys, numpy as np, torch
from PIL import Image
from transformers import AutoModel
ap=argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--manifest', required=True)
ap.add_argument('--imgdir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--size', type=int, default=768); ap.add_argument('--batch', type=int, default=4)
a=ap.parse_args()
m=AutoModel.from_pretrained(a.model, trust_remote_code=True).eval().cuda()
rows=list(csv.DictReader(open(a.manifest)))
def prep(im):
    im=im.resize((a.size,a.size), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im,dtype=np.float32).copy()).permute(2,0,1)/255.
feats,names=[],[]
for i in range(0,len(rows),a.batch):
    chunk=rows[i:i+a.batch]; views=[]
    for r in chunk:
        im=Image.open(os.path.join(a.imgdir,r['name'])).convert('RGB'); w,h=im.size
        views.append(prep(im))
        for dy in (0,1):
            for dx in (0,1):
                views.append(prep(im.crop((dx*w//2,dy*h//2,(dx+1)*w//2,(dy+1)*h//2))))
    x=torch.stack(views).cuda()
    with torch.inference_mode():
        summ, feat = m(x)
    per=torch.cat([summ.float(), feat.float().mean(1), feat.float().amax(1)],1).view(len(chunk),5,-1)
    f=torch.cat([per[:,0], per[:,1:].mean(1), per[:,1:].amax(1)],1)
    feats.append(f.cpu().numpy()); names += [r['name'] for r in chunk]
    if (i//a.batch)%20==0: print(i, flush=True)
X=np.concatenate(feats,0)
np.savez_compressed(a.out, X=X.astype(np.float32), names=np.array(names))
print('saved', a.out, X.shape)
