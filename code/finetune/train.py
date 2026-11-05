import argparse, csv, json, os, sys, time, copy
import numpy as np, torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
try: torch.set_float32_matmul_precision('high')
except Exception: pass
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'decision_eval'))
from data import IQADataset, CachedIQADataset, load_split, read_manifest, targets_from_grades, CLASSES
CACHE_OK=os.path.exists('/dev/shm/race18_native/index.npz')
from models import build_backbone, Net
import comp_metrics as cm

def exact(p):
    P = p.float().cpu().numpy(); n = P.shape[0]
    p0 = np.zeros((n,6)); p1 = np.zeros((n,4)); p2 = np.zeros((n,4)); j=0
    for ci,c in enumerate(CLASSES):
        if c in ['模糊','纹理损失','过锐','过度生成']:
            c2,c1,c0 = P[:,j],P[:,j+1],P[:,j+2]; j+=3; k=['模糊','纹理损失','过锐','过度生成'].index(c)
            s2,s1,s0 = 1/(1+np.exp(-c2)),1/(1+np.exp(-c1)),1/(1+np.exp(-c0))
            p2[:,k]=np.clip(s2-s1,0,None); p1[:,k]=np.clip(s1-s0,0,None); p0[:,ci]=s0
        else:
            p0[:,ci]=1/(1+np.exp(-P[:,j])); j+=1
    return p0,p1,p2

@torch.no_grad()
def predict(net, dl, mean, std, device):
    net.eval(); outs=[]; gs=[]
    for x,g,_ in dl:
        x=x.to(device); x=(x-mean)/std
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outs.append(net(x).float().cpu())
        gs.append(g)
    return torch.cat(outs), torch.cat(gs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backbone', default='siglip2'); ap.add_argument('--mode', default='lastk')
    ap.add_argument('--lastk', type=int, default=4); ap.add_argument('--lora-r', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--accum', type=int, default=2); ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--lr-bb', type=float, default=1e-5); ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.05); ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--head-hidden', type=int, default=0); ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--split', default='data/split_trainval.csv'); ap.add_argument('--manifest', default='data/manifest_combined.csv')
    ap.add_argument('--out', default='results/ft_run.json'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tau', type=float, default=0.02)
    a=ap.parse_args()
    dev='cuda'
    rows=load_split(a.split); man=read_manifest(a.manifest)
    for r in rows: r.update({k:man[r['name']][k] for k in CLASSES})
    if a.limit:
        tr=[r for r in rows if r['split']=='train'][:a.limit]; va=[r for r in rows if r['split']=='val'][:max(20,a.limit//4)]; te=[r for r in rows if r['split']=='test'][:max(20,a.limit//4)]
    else:
        tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='val']; te=[r for r in rows if r['split']=='test']
    names_orig=set(r['name'] for r in csv.DictReader(open('data/manifest.csv')))
    def mk(rows,tr_): return CachedIQADataset(rows,a.size,tr_) if CACHE_OK else IQADataset(rows,'data/images','data/images_supp',names_orig,a.size,tr_)
    dtr=mk(tr,True); dva=mk(va,False); dte=mk(te,False)
    ltr=DataLoader(dtr,batch_size=a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,drop_last=True)
    lva=DataLoader(dva,batch_size=a.batch,shuffle=False,num_workers=a.workers)
    lte=DataLoader(dte,batch_size=a.batch,shuffle=False,num_workers=a.workers)
    m,fwd,mean,std,blocks,dim=build_backbone(a.backbone)
    net=Net(m,fwd,dim,a.head_hidden,a.dropout).to(dev)
    # freeze
    for p in m.parameters(): p.requires_grad=False
    if a.mode in ('lastk','full'):
        if a.mode=='full':
            for p in m.parameters(): p.requires_grad=True
        elif blocks is not None:
            for b in blocks[-a.lastk:]:
                for p in b.parameters(): p.requires_grad=True
            # unfreeze final norms
            for nm,mod in m.named_modules():
                if 'post_layernorm' in nm or nm.endswith('norm') or nm.endswith('norm1') or nm.endswith('norm2'):
                    for p in mod.parameters(): p.requires_grad=True
    elif a.mode=='lora':
        net.m=__import__('models').lora_wrap(m,a.lora_r); net.fwd=None
        # rebuild fwd for wrapped model
        if a.backbone=='siglip2':
            net.fwd=lambda x: net.m(pixel_values=x).last_hidden_state.mean(1)
        elif a.backbone=='eupe':
            def f(x):
                o=net.m.forward_features(x); return torch.cat([o['x_norm_clstoken'],o['x_norm_patchtokens'].mean(1)],1)
            net.fwd=f
        else: raise ValueError('lora only for siglip2/eupe')
    trainable=[p for p in net.parameters() if p.requires_grad]
    print('trainable params %.1fM / total %.1fM'%(sum(p.numel() for p in trainable)/1e6, sum(p.numel() for p in net.parameters())/1e6), flush=True)
    opt=torch.optim.AdamW([{'params':[p for p in net.m.parameters() if p.requires_grad],'lr':a.lr_bb},
                           {'params':[p for p in net.head.parameters() if p.requires_grad],'lr':a.lr_head}],weight_decay=a.wd)
    steps=len(ltr)//a.accum*a.epochs
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[a.lr_bb,a.lr_head],total_steps=max(1,steps),pct_start=0.1)
    mean_t=torch.tensor(mean,dtype=torch.float32,device=dev).view(1,1,3,1,1)
    std_t=torch.tensor(std,dtype=torch.float32,device=dev).view(1,1,3,1,1)
    # pos weights from train
    Y=targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr]))
    pw=(Y.numel()-Y.sum(0))/Y.sum(0).clamp(min=1); pw=pw.clamp(1,20).to(dev)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw)
    best=None; best_state=None
    for ep in range(a.epochs):
        net.train(); t0=time.time(); tot=0; nb=0; opt.zero_grad()
        for i,(x,g,_) in enumerate(ltr):
            x=x.to(dev); x=(x-mean_t)/std_t; g=g.to(dev)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=net(x).float()
            loss=lossf(logits, targets_from_grades(g))
            (loss/a.accum).backward()
            if (i+1)%a.accum==0:
                torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); sch.step(); opt.zero_grad()
            tot+=loss.item(); nb+=1
        pv,gv=predict(net,lva,mean_t,std_t,dev)
        p0,p1,p2=exact(pv); gv=gv.numpy().astype(int)
        b,th=cm.optimise(gv,p0,p1,p2,a.tau)
        print(f'ep{ep+1} loss={tot/max(1,nb):.3f} val total={b["total"]:.1f} S_pos={b["S_pos"]:.3f} P0R={b["p0R"]:.3f} S_clean={b["S_clean"]:.3f} t={time.time()-t0:.0f}s',flush=True)
        if best is None or b['total']>best[0]:
            best=(b['total'],th,copy.deepcopy(net.state_dict()))
    # evaluate test with val-tuned thresholds + quantile calibration
    net.load_state_dict(best[2])
    ptr,gtr=predict(net,ltr,mean_t,std_t,dev); pva,gva=predict(net,lva,mean_t,std_t,dev); pte,gte=predict(net,lte,mean_t,std_t,dev)
    q0,q1,q2=exact(ptr); q_va1,q_va2=exact(pva)[0],exact(pva)[1]
    a0,a1,a2=exact(pva); b0,b1,b2=exact(pte)
    gtr_n=gtr.numpy().astype(int); gva_n=gva.numpy().astype(int); gte_n=gte.numpy().astype(int)
    # conservative P0 threshold from train+val clean samples
    clean_tr=(gtr_n==0).all(1); clean_va=(gva_n==0).all(1)
    cs=np.concatenate([q0[clean_tr].max(1), a0[clean_va].max(1)]) if (clean_tr.any() or clean_va.any()) else np.array([0.5])
    t0_cal=float(min(0.995, max(0.05, np.quantile(cs,0.995)+0.02)))
    best_s=None
    for t1 in np.arange(0.1,0.91,0.1):
        for t2 in np.arange(0.1,0.91,0.1):
            s=cm.score(gva_n, cm.decode(a0,a1,a2,t0_cal,t1,t2), a.tau)
            if best_s is None or s['S_pos']>best_s[0]['S_pos']: best_s=(s,(t0_cal,t1,t2))
    th_cal=best_s[1]
    st=cm.score(gte_n, cm.decode(b0,b1,b2,*th_cal), a.tau)
    st_obj=cm.score(gte_n, cm.decode(b0,b1,b2,*best[1]), a.tau)
    np.savez_compressed(a.out.replace('.json','_probs.npz'), va_p0=a0, va_p1=a1, va_p2=a2, te_p0=b0, te_p1=b1, te_p2=b2, gva=gva_n, gte=gte_n,
                        tr_clean_max=q0[clean_tr].max(1) if clean_tr.any() else np.array([]))
    res={'backbone':a.backbone,'mode':a.mode,'best_val_total':best[0],'thresholds':[float(x) for x in best[1]],
         'cal_thresholds':[float(x) for x in th_cal],'cal_t0':t0_cal,
         'test':{'total':st['total'],'S_pos':st['S_pos'],'P0R':st['p0R'],'P0P':st['p0P'],'P0F1':st['p0F'],'S_p0':st['S_p0'],'S_clean':st['S_clean'],'fpr':st['fpr']},
         'test_obj_tuned':{'total':st_obj['total'],'S_pos':st_obj['S_pos'],'P0R':st_obj['p0R'],'S_p0':st_obj['S_p0'],'S_clean':st_obj['S_clean'],'fpr':st_obj['fpr']},
         'per_p0':{f'{x[0]}_P0':{'P':x[2],'R':x[3],'F1':x[4]} for x in st['per'] if x[1]=='P0'},'n_test':len(te)}
    json.dump(res,open(a.out,'w'),ensure_ascii=False,indent=1)
    print('TEST',json.dumps(res['test'],ensure_ascii=False),flush=True)
    print('saved',a.out)

if __name__=='__main__': main()
