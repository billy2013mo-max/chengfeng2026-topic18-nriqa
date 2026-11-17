import sys, os, numpy as np, random
from PIL import Image, ImageDraw
REPO="/wanqing-develop/luowenjing/18"
z=np.load(sys.argv[1],allow_pickle=True); names=[str(x) for x in z["names"]]; gold=z["gold"]; P=z["P"]; nd=z["nested"]
def find(n):
    for d in ("data/images","data/images_supp"):
        p=os.path.join(REPO,d,n)
        if os.path.exists(p): return p
def sheet(idx,out,cols=4,cell=420):
    rows=(len(idx)+cols-1)//cols
    W=Image.new("RGB",(cols*cell,rows*(cell+34)),(0,0,0)); dr=ImageDraw.Draw(W)
    for k,i in enumerate(idx):
        im=Image.open(find(names[i])).convert("RGB"); w,h=im.size; s=cell/max(w,h); im=im.resize((int(w*s),int(h*s)),Image.LANCZOS)
        x,y=(k%cols)*cell,(k//cols)*(cell+34); W.paste(im,(x,y))
        g=gold[i]; dr.text((x+2,y+cell+2),"#%d %dx%d gold BL%d TX%d SH%d GL%d OG%d BK%d"%(i,w,h,*g),fill=(255,255,255))
        dr.text((x+2,y+cell+16),"pred OG %s BK %.2f | dec OG%d BK%d"%(" ".join("%.2f"%v for v in P[i,4]),P[i,5,3],nd[i,4],nd[i,5]),fill=(120,255,120))
    W.save(out,quality=88)
random.seed(int(sys.argv[3]))
odd=np.array([("虚化突兀" in n) for n in names])
sel={"bk_missed":[i for i in range(len(names)) if gold[i,5]==3 and nd[i,5]==0],
     "bk_fp":[i for i in range(len(names)) if gold[i,5]==0 and nd[i,5]==3],
     "bk_hit_nonbatch":[i for i in range(len(names)) if gold[i,5]==3 and nd[i,5]==3 and not odd[i]],
     "og_p0_fp":[i for i in range(len(names)) if gold[i,4]<3 and nd[i,4]==3]}
for k,v in sel.items():
    print(k,len(v)); random.shuffle(v); sheet(v[:12],os.path.join(sys.argv[2],k+".jpg"))
