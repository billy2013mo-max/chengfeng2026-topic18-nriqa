import argparse, csv, json, os, sys, time, copy
import numpy as np, torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
try: torch.set_float32_matmul_precision('high')
except Exception: pass
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','decision_eval'))
from data import IQADataset, CachedIQADataset, load_split, read_manifest, targets_from_grades, CLASSES
CACHE_OK=os.path.exists('/dev/shm/race18_native/index.npz')
from models import build_backbone
import comp_metrics as cm

class DualNet(nn.Module):
    def __init__(self, b1, b2, hidden=0, dropout=0.2):
        super().__init__()
        self.m1,f1,mu1,sd1,bl1,d1=b1; self.m2,f2,mu2,sd2,bl2,d2=b2
        self.f1,self.f2=f1,f2
        self.r1=torch.tensor(mu1).view(1,3,1,1); self.s1=torch.tensor(sd1).view(1,3,1,1)
        self.r2=torch.tensor(mu2).view(1,3,1,1); self.s2=torch.tensor(sd2).view(1,3,1,1)
        d=d1*3+d2*3
        self.head=nn.Sequential(nn.Dropout(dropout), nn.Linear(d,hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden,14)) if hidden else nn.Sequential(nn.Dropout(dropout), nn.Linear(d,14))
    def _agg(self,f): return torch.cat([f[:,0], f[:,1:].mean(1), f[:,1:].amax(1)],1)
    def forward(self,x):
        B,V,C,H,W=x.shape
        x1=((x.view(B*V,C,H,W)-self.r1.to(x.device))/self.s1.to(x.device))
        x2=((x.view(B*V,C,H,W)-self.r2.to(x.device))/self.s2.to(x.device))
        f1=self.f1(x1).view(B,V,-1); f2=self.f2(x2).view(B,V,-1)
        return self.head(torch.cat([self._agg(f1),self._agg(f2)],1))

def exact(P):
    P=P.float().cpu().numpy(); n=len(P); p0=np.zeros((n,6)); p1=np.zeros((n,4)); p2=np.zeros((n,4)); j=0
    for ci,c in enumerate(CLASSES):
        if c in ['模糊','纹理损失','过锐','过度生成']:
            c2,c1,c0=P[:,j],P[:,j+1],P[:,j+2]; j+=3; k=['模糊','纹理损失','过锐','过度生成'].index(c)
            s2,s1,s0=1/(1+np.exp(-c2)),1/(1+np.exp(-c1)),1/(1+np.exp(-c0))
            p2[:,k]=np.clip(s2-s1,0,None); p1[:,k]=np.clip(s1-s0,0,None); p0[:,ci]=s0
        else: p0[:,ci]=1/(1+np.exp(-P[:,j])); j+=1
    return p0,p1,p2

@torch.no_grad()
def predict(net,dl,dev):
    net.eval(); O=[]; G=[]
    for x,g,_ in dl:
        x=x.to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            O.append(net(x).float().cpu())
        G.append(g)
    return torch.cat(O), torch.cat(G)

ap=argparse.ArgumentParser()
ap.add_argument('--b1',default='eupe'); ap.add_argument('--mode1',default='full')
ap.add_argument('--b2',default='siglip2'); ap.add_argument('--mode2',default='lastk'); ap.add_argument('--lastk2',type=int,default=4)
ap.add_argument('--epochs',type=int,default=6); ap.add_argument('--batch',type=int,default=6); ap.add_argument('--size',type=int,default=384)
ap.add_argument('--lr-bb',type=float,default=2e-5); ap.add_argument('--lr-head',type=float,default=1e-3)
ap.add_argument('--wd',type=float,default=0.05); ap.add_argument('--workers',type=int,default=10)
ap.add_argument('--out',default='results/ft_dual.json'); ap.add_argument('--limit',type=int,default=0)
ap.add_argument('--tau',type=float,default=0.02)
a=ap.parse_args(); dev='cuda'
rows=load_split('data/split_trainval.csv'); man=read_manifest('data/manifest_combined.csv')
for r in rows: r.update({k:man[r['name']][k] for k in CLASSES})
if a.limit:
    tr=[r for r in rows if r['split']=='train'][:a.limit]; va=[r for r in rows if r['split']=='val'][:max(20,a.limit//4)]; te=[r for r in rows if r['split']=='test'][:max(20,a.limit//4)]
else:
    tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='val']; te=[r for r in rows if r['split']=='test']
orig=set(r['name'] for r in csv.DictReader(open('data/manifest.csv')))
def mk(rows,tr_): return CachedIQADataset(rows,a.size,tr_) if CACHE_OK else IQADataset(rows,'data/images','data/images_supp',orig,a.size,tr_)
dtr=mk(tr,True); dva=mk(va,False); dte=mk(te,False)
ltr=DataLoader(dtr,batch_size=a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,drop_last=True); lva=DataLoader(dva,batch_size=a.batch,shuffle=False,num_workers=a.workers); lte=DataLoader(dte,batch_size=a.batch,shuffle=False,num_workers=a.workers)
b1=build_backbone(a.b1); b2=build_backbone(a.b2)
net=DualNet(b1,b2).to(dev)
for p in b1[0].parameters(): p.requires_grad=True
for p in b2[0].parameters(): p.requires_grad=False
if b2[4] is not None:
    for bl in b2[4][-a.lastk2:]:
        for p in bl.parameters(): p.requires_grad=True
tr_p=[p for p in net.parameters() if p.requires_grad]
print('trainable %.1fM / %.1fM'%(sum(p.numel() for p in tr_p)/1e6, sum(p.numel() for p in net.parameters())/1e6),flush=True)
opt=torch.optim.AdamW([{'params':[p for p in b1[0].parameters() if p.requires_grad],'lr':a.lr_bb},
                       {'params':[p for p in b2[0].parameters() if p.requires_grad],'lr':a.lr_bb*0.5},
                       {'params':net.head.parameters(),'lr':a.lr_head}],weight_decay=a.wd)
Y=targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr]))
pw=((Y.numel()-Y.sum(0))/Y.sum(0).clamp(min=1)).clamp(1,20).to(dev)
lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
steps=(len(ltr)//1)*a.epochs
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[a.lr_bb,a.lr_bb*0.5,a.lr_head],total_steps=max(1,steps),pct_start=0.1)
best=None
for ep in range(a.epochs):
    net.train(); t0=time.time(); tot=0; nb=0
    for x,g,_ in ltr:
        x=x.to(dev); g=g.to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=net(x).float()
        loss=lossf(logits, targets_from_grades(g))
        loss.backward(); torch.nn.utils.clip_grad_norm_(tr_p,1.0); opt.step(); sch.step(); opt.zero_grad()
        tot+=loss.item(); nb+=1
    pv,gv=predict(net,lva,dev); p0,p1,p2=exact(pv); gv=gv.numpy().astype(int); b,th=cm.optimise(gv,p0,p1,p2,a.tau)
    print(f'ep{ep+1} loss={tot/max(1,nb):.3f} val_total={b["total"]:.1f} S_pos={b["S_pos"]:.3f} P0R={b["p0R"]:.3f} S_clean={b["S_clean"]:.3f} t={time.time()-t0:.0f}s',flush=True)
    if best is None or b['total']>best[0]: best=(b['total'],th,copy.deepcopy(net.state_dict()))
net.load_state_dict(best[2])
ptr,gtr=predict(net,ltr,dev); pva,gva=predict(net,lva,dev); pte,gte=predict(net,lte,dev)
a0,a1,a2=exact(pva); b0,b1_,b2_=exact(pte); q0,_,_=exact(ptr)
gva_n=gva.numpy().astype(int); gte_n=gte.numpy().astype(int); gtr_n=gtr.numpy().astype(int)
ct=(gtr_n==0).all(1); cv=(gva_n==0).all(1)
cs=np.concatenate([q0[ct].max(1),a0[cv].max(1)]) if (ct.any() or cv.any()) else np.array([0.5])
t0c=float(min(0.995,max(0.05,np.quantile(cs,0.995)+0.02)))
bs=None
for t1 in np.arange(0.1,0.91,0.1):
    for t2 in np.arange(0.1,0.91,0.1):
        s=cm.score(gva_n,cm.decode(a0,a1,a2,t0c,t1,t2),a.tau)
        if bs is None or s['S_pos']>bs[0]['S_pos']: bs=(s,(t0c,t1,t2))
st=cm.score(gte_n,cm.decode(b0,b1_,b2_,*bs[1]),a.tau)
so=cm.score(gte_n,cm.decode(b0,b1_,b2_,*best[1]),a.tau)
res={'b1':a.b1,'b2':a.b2,'mode2':a.mode2,'cal_t0':t0c,'test':{k:st[k] for k in ['total','S_pos','p0R','p0P','p0F','S_p0','S_clean','fpr']},'test_obj':{k:so[k] for k in ['total','S_pos','p0R','S_p0','S_clean','fpr']},'per_p0':{f'{x[0]}_P0':{'P':x[2],'R':x[3],'F1':x[4]} for x in st['per'] if x[1]=='P0'}}
json.dump(res,open(a.out,'w'),ensure_ascii=False,indent=1); print('TEST',json.dumps(res['test'],ensure_ascii=False)); print('saved',a.out)
