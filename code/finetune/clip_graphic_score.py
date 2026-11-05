import argparse, csv, os, numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
ap=argparse.ArgumentParser(); ap.add_argument('--out',default='results/graphic_score.npz'); a=ap.parse_args()
os.chdir('/wanqing-develop/luowenjing/18')
m=CLIPModel.from_pretrained('openai/clip-vit-large-patch14').eval().cuda()
p=CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
photo=['a photograph taken by a camera','a real-world photo','a natural photograph','a photo of a real scene']
graphic=['an illustration or drawing','an anime or cartoon image','a mobile phone screenshot','a video game screenshot','a poster or advertisement with text','a 3D rendered computer graphic','a digital art image']
with torch.no_grad():
    def emb(texts):
        t=p(text=texts,return_tensors='pt',padding=True).to('cuda')
        o=m.text_model(input_ids=t['input_ids'],attention_mask=t['attention_mask'])
        f=m.text_projection(o.pooler_output); return f/f.norm(dim=-1,keepdim=True)
    tp=emb(photo).mean(0); tg=emb(graphic).mean(0)
    tp/=tp.norm(); tg/=tg.norm()
rows=list(csv.DictReader(open('data/manifest_combined.csv')))
orig=set(r['name'] for r in csv.DictReader(open('data/manifest.csv')))
names=[]; scores=[]
for i in range(0,len(rows),32):
    chunk=rows[i:i+32]; ims=[]
    for r in chunk:
        d='data/images' if r['name'] in orig else 'data/images_supp'
        ims.append(Image.open(os.path.join(d,r['name'])).convert('RGB'))
    inp=p(images=ims,return_tensors='pt',padding=True).to('cuda')
    with torch.no_grad():
        o=m.vision_model(pixel_values=inp['pixel_values']); f=m.visual_projection(o.pooler_output); f=f/f.norm(dim=-1,keepdim=True)
        sp=(f@tp); sg=(f@tg)
        pr=torch.softmax(torch.stack([sp,sg],1)/0.01,1)[:,1].cpu().numpy()
    names+= [r['name'] for r in chunk]; scores+=list(pr)
    if i%320==0: print(i,flush=True)
np.savez_compressed(a.out,names=np.array(names),score=np.array(scores))
print('saved',a.out,len(names))
