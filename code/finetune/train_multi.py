import argparse, csv, json, os, random, sys, time, copy
import numpy as np, torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
try: torch.set_float32_matmul_precision('high')
except Exception: pass
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','decision_eval'))
from data import IQADataset, CachedIQADataset, load_split, read_manifest, targets_from_grades, CLASSES, FULL
CACHE_OK=os.path.exists('/dev/shm/race18_native/index.npz')
from models import build_backbone
import comp_metrics as cm
# CLASSES order is 模糊,纹理损失,过锐,花屏,过度生成,虚化突兀; targets_from_grades appends
# 3 cols for each FULL class and 1 col for 花屏/虚化突兀, so the P0 columns are:
# 模糊 2, 纹理损失 5, 过锐 8, 花屏 9, 过度生成 12, 虚化突兀 13.
P0_IDX=[2,5,8,9,12,13]

class MultiNet(nn.Module):
    def __init__(self, specs, hidden=0, dropout=0.2, head='linear'):
        self.head_kind=head
        super().__init__()
        self.bs=nn.ModuleList(); self.fwds=[]
        for (m,fwd,mu,sd,bl,dim) in specs:
            self.bs.append(m); self.fwds.append(fwd)
            self.register_buffer('mu_%d'%len(self.fwds), torch.tensor(mu).view(1,3,1,1))
            self.register_buffer('sd_%d'%len(self.fwds), torch.tensor(sd).view(1,3,1,1))
        d=sum(3*x[5] for x in specs)
        if head=='perclass':
            self.head=None
            self.heads=nn.ModuleList([nn.Sequential(nn.Dropout(dropout), nn.Linear(d,3)) if c in FULL else nn.Sequential(nn.Dropout(dropout), nn.Linear(d,1)) for c in CLASSES])
        elif hidden:
            self.head=nn.Sequential(nn.Dropout(dropout), nn.Linear(d,hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden,14))
        else:
            self.head=nn.Sequential(nn.Dropout(dropout), nn.Linear(d,14))
        self.gate=nn.Linear(d,1)
    def _agg(self,f): return torch.cat([f[:,0],f[:,1:].mean(1),f[:,1:].amax(1)],1)
    def forward(self,x):
        B,V,C,H,W=x.shape; out=[]
        for i,(m,fwd) in enumerate(zip(self.bs,self.fwds)):
            mu=getattr(self,'mu_%d'%(i+1)); sd=getattr(self,'sd_%d'%(i+1))
            dt=next(m.parameters()).dtype
            z=((x.view(B*V,C,H,W)-mu)/sd).to(dt)
            out.append(self._agg(fwd(z).view(B,V,-1)).float())
        fused=torch.cat(out,1)
        if self.head_kind=='perclass':
            return torch.cat([h(fused) for h in self.heads],1), self.gate(fused).squeeze(1)
        return self.head(fused), self.gate(fused).squeeze(1)

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
        with torch.autocast('cuda',dtype=torch.bfloat16):
            lg,gt=net(x.to(dev))
        O.append(torch.cat([lg.float(),gt.float().unsqueeze(1)],1).cpu()); G.append(g)
    return torch.cat(O), torch.cat(G)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def ce_loss(logits,g):
    import torch.nn.functional as F
    tot=0.0; j=0
    for ci,c in enumerate(CLASSES):
        if c in FULL:
            l2,l1,l0=logits[:,j],logits[:,j+1],logits[:,j+2]; j+=3
            s2=torch.sigmoid(l2); s1=torch.sigmoid(l1); s0=torch.sigmoid(l0)
            p=torch.stack([1-s2, s2-s1, s1-s0, s0],1).clamp(min=1e-6)
            p=p/p.sum(1,keepdim=True); tgt=g[:,ci].long()
        else:
            s0=torch.sigmoid(logits[:,j]); j+=1
            p=torch.stack([1-s0,s0],1).clamp(min=1e-6); p=p/p.sum(1,keepdim=True)
            tgt=(g[:,ci]==3).long()
        tot=tot+F.nll_loss(torch.log(p),tgt)
    return tot/len(CLASSES)

ap=argparse.ArgumentParser()
ap.add_argument('--backbones', required=True, help='eupe:full,siglip2:lastk4,cradio:lastk4')
ap.add_argument('--epochs',type=int,default=6); ap.add_argument('--batch',type=int,default=6); ap.add_argument('--size',type=int,default=384)
ap.add_argument('--lr-full',type=float,default=2e-5); ap.add_argument('--lr-last',type=float,default=1e-5); ap.add_argument('--lr-head',type=float,default=1e-3)
ap.add_argument('--wd',type=float,default=0.05); ap.add_argument('--workers',type=int,default=10)
ap.add_argument('--seed',type=int,default=1); ap.add_argument('--clean-penalty',type=float,default=0.0)
ap.add_argument('--gate-weight',type=float,default=0.0)
ap.add_argument('--gate-th',type=float,default=0.0)
ap.add_argument('--dropout',type=float,default=0.2)
ap.add_argument('--clean-repeat',type=int,default=1)
ap.add_argument('--out',default='results/ft_multi.json'); ap.add_argument('--limit',type=int,default=0)
ap.add_argument('--split',default='data/split_trainval.csv')
ap.add_argument('--manifest',default='data/manifest_combined.csv')
ap.add_argument('--tau',type=float,default=0.02)
# --- nightly experiments (defaults keep the old behaviour) ---
ap.add_argument('--clean-penalty-class',type=int,default=-1,help='apply clean penalty to this class index only; -1 = max over all')
ap.add_argument('--clean-bokeh',type=float,default=0.0,help='prob of bokeh/beautify augmentation on clean train images')
ap.add_argument('--bokeh-radius',type=float,default=6.0)
ap.add_argument('--hard-clean-file',default='',help='txt of image names to repeat as hard clean negatives')
ap.add_argument('--hard-repeat',type=int,default=0)
ap.add_argument('--save-state',default='',help='path to save the best checkpoint')
ap.add_argument('--dump-train-clean',default='',help='csv path: score every train clean image with the best model')
ap.add_argument('--tiles',type=int,default=2,help='T x T native-resolution crops per image (default 2 -> 5 views)')
ap.add_argument('--loss',default='bce',choices=['bce','ce'])
ap.add_argument('--head',default='linear',choices=['linear','perclass'])
ap.add_argument('--hidden',type=int,default=0)
ap.add_argument('--mild-weight',type=float,default=1.0,help='extra loss weight for grade-1/2 samples')
a=ap.parse_args(); set_seed(a.seed); dev='cuda'
specs=[]
for s in a.backbones.split(','):
    name,mode=s.split(':')
    b=build_backbone(name); specs.append((b,name,mode))
rows=load_split(a.split); man=read_manifest(a.manifest)
for r in rows: r.update({k:man[r['name']][k] for k in CLASSES})
tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='val']; te=[r for r in rows if r['split']=='test']
if not te:
    te=va; print('WARNING: split has no test rows, scoring on val', flush=True)
tr_base=[r for r in tr if all(int(r[c])==0 for c in CLASSES)]   # clean train rows, before any repeats
if a.clean_repeat>1:
    clean=[r for r in tr if all(int(r[c])==0 for c in CLASSES)]
    tr=tr+clean*(a.clean_repeat-1)
if a.hard_repeat>0 and a.hard_clean_file:
    hard=set(x.strip() for x in open(a.hard_clean_file) if x.strip())
    extra=[r for r in tr_base if r['name'] in hard]
    tr=tr+extra*a.hard_repeat
    print(f'hard clean repeat: {len(extra)} images x{a.hard_repeat}', flush=True)
if a.limit: tr=tr[:a.limit]; va=va[:max(20,a.limit//4)]; te=te[:max(20,a.limit//4)]
orig=set(r['name'] for r in csv.DictReader(open('data/manifest.csv')))
def mk(rows,tr_): return CachedIQADataset(rows,a.size,tr_,bokeh=a.clean_bokeh,bokeh_r=a.bokeh_radius,tiles=a.tiles) if CACHE_OK else IQADataset(rows,'data/images','data/images_supp',orig,a.size,tr_,bokeh=a.clean_bokeh,bokeh_r=a.bokeh_radius,tiles=a.tiles)
dtr=mk(tr,True); dva=mk(va,False); dte=mk(te,False)
print('using cache' if CACHE_OK else 'using raw images', flush=True)
ltr=DataLoader(dtr,batch_size=a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,drop_last=True,worker_init_fn=lambda w:set_seed(a.seed+w))
lva=DataLoader(dva,batch_size=a.batch,shuffle=False,num_workers=a.workers); lte=DataLoader(dte,batch_size=a.batch,shuffle=False,num_workers=a.workers)
net=MultiNet([x[0] for x in specs], hidden=a.hidden, dropout=a.dropout, head=a.head).to(dev)
groups=[]
for i,(b,name,mode) in enumerate(specs):
    m=b[0]; blocks=b[4]
    for p in m.parameters(): p.requires_grad=(mode=='full')
    if mode.startswith('lastk') and blocks is not None:
        k=int(mode[len('lastk'):] or 4)
        for bl in blocks[-k:]:
            for p in bl.parameters(): p.requires_grad=True
    lr=a.lr_full if mode=='full' else a.lr_last
    groups.append({'params':[p for p in m.parameters() if p.requires_grad],'lr':lr})
groups.append({'params':(net.heads.parameters() if a.head=='perclass' else net.head.parameters()),'lr':a.lr_head})
tr_p=[p for g in groups for p in g['params']]
print('trainable %.1fM / %.1fM'%(sum(p.numel() for p in tr_p)/1e6,sum(p.numel() for p in net.parameters())/1e6),flush=True)
opt=torch.optim.AdamW(groups,weight_decay=a.wd)
Y=targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr]))
# Y is (N,14): the negative/positive ratio must use the sample count N, not Y.numel()=14N
# (the old form saturated every channel at the clamp ceiling 20 and caused cross-defect P2/P1 FPs).
pw=((Y.shape[0]-Y.sum(0))/Y.sum(0).clamp(min=1)).clamp(1,20).to(dev)
lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
lossf_nw=nn.BCEWithLogitsLoss(pos_weight=pw,reduction='none') if a.mild_weight!=1.0 else None
steps=len(ltr)*a.epochs
sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[g['lr'] for g in groups],total_steps=max(1,steps),pct_start=0.1)
best=None
for ep in range(a.epochs):
    net.train(); t0=time.time(); tot=0; nb=0
    for x,g,_ in ltr:
        x=x.to(dev); g=g.to(dev)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,gl=net(x)
        logits=logits.float(); gl=gl.float()
        if a.loss=='ce':
            loss = ce_loss(logits,g)
        elif lossf_nw is not None:
            l=lossf_nw(logits,targets_from_grades(g)).mean(1)
            mg=g.max(1).values
            w=torch.where((mg==1)|(mg==2), a.mild_weight, 1.0)
            loss=(l*w).mean()
        else:
            loss = lossf(logits,targets_from_grades(g))
        if a.gate_weight>0:
            cl=(g==0).all(1).float()
            loss=loss+a.gate_weight*nn.functional.binary_cross_entropy_with_logits(gl,cl,pos_weight=torch.tensor([4.0],device=gl.device))
        if a.clean_penalty>0:
            cmask=(g==0).all(1)
            if cmask.any():
                if a.clean_penalty_class>=0:
                    loss=loss+a.clean_penalty*torch.sigmoid(logits[:,P0_IDX[a.clean_penalty_class]])[cmask].mean()
                else:
                    loss=loss+a.clean_penalty*torch.sigmoid(logits[:,P0_IDX]).max(1)[0][cmask].mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(tr_p,1.0); opt.step(); sch.step(); opt.zero_grad()
        tot+=loss.item(); nb+=1
    pv,gv=predict(net,lva,dev); p0,p1,p2=exact(pv[:,:14]); gv=gv.numpy().astype(int)
    b,th=cm.optimise(gv,p0,p1,p2,a.tau)
    if a.gate_weight>0:
        gprob=pv[:,14].numpy()
        for tg in [0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.98]:
            p0v=p0.copy(); p0v[gprob>tg,:]=0.0
            bb,tt=cm.optimise(gv,p0v,p1,p2,a.tau)
            if bb['total']>b['total']: b,th=bb,tt; b['gate']=tg; a.gate_th=tg
    print(f'ep{ep+1} loss={tot/max(1,nb):.3f} val={b["total"]:.1f} S_pos={b["S_pos"]:.3f} P0R={b["p0R"]:.3f} S_clean={b["S_clean"]:.3f} t={time.time()-t0:.0f}s',flush=True)
    if best is None or b['total']>best[0]: best=(b['total'],th,copy.deepcopy(net.state_dict()))
net.load_state_dict(best[2])
if a.save_state:
    torch.save({'state':best[2],'backbones':a.backbones,'size':a.size}, a.save_state)
    print('state saved', a.save_state, flush=True)
pte,gte=predict(net,lte,dev); p0,p1,p2=exact(pte[:,:14]); gte_n=gte.numpy().astype(int)
pva,gva=predict(net,lva,dev); a0,a1,a2=exact(pva[:,:14]); gva_n=gva.numpy().astype(int)
tp0=p0.copy()
if a.gate_weight>0 and a.gate_th>0:
    tp0[pte[:,14].numpy()>a.gate_th,:]=0.0
st=cm.score(gte_n,cm.decode(tp0,p1,p2,*best[1]),a.tau)
res={'backbones':a.backbones,'seed':a.seed,'clean_penalty':a.clean_penalty,'clean_repeat':a.clean_repeat,
     'loss':a.loss,'val_total':best[0],'thresholds':[float(x) for x in best[1]],
     'test':{k:st[k] for k in ['total','S_pos','p0R','p0P','p0F','S_p0','S_clean','fpr']},
     'per_p0':{f'{x[0]}_P0':{'P':x[2],'R':x[3],'F1':x[4]} for x in st['per'] if x[1]=='P0'}}
json.dump(res,open(a.out,'w'),ensure_ascii=False,indent=1)
np.savez_compressed(a.out.replace('.json','_probs.npz'),
    va_p0=a0,va_p1=a1,va_p2=a2,va_gate=pva[:,14].numpy(),gva=gva_n,va_names=np.array([r['name'] for r in va]),
    te_p0=p0,te_p1=p1,te_p2=p2,te_gate=pte[:,14].numpy(),gte=gte_n,te_names=np.array([r['name'] for r in te]))
if a.dump_train_clean:
    dcl=mk(tr_base,False)
    lcl=DataLoader(dcl,batch_size=a.batch,shuffle=False,num_workers=a.workers)
    pcl,_=predict(net,lcl,dev); c0,_,_=exact(pcl[:,:14])
    with open(a.dump_train_clean,'w') as f:
        f.write('name,maxp0\n')
        for i,r in enumerate(tr_base): f.write(f"{r['name']},{float(c0[i].max()):.4f}\n")
    print('train clean scores ->', a.dump_train_clean, flush=True)
print('TEST',json.dumps(res['test'],ensure_ascii=False)); print('saved',a.out)
