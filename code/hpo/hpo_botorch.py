"""BoTorch HPO maximizing a tail-aware validation objective.

obj = val_total - 10 * max(0, val_clean_p99 - 0.15)
where val_clean_p99 is the 99th percentile of max-P0 on clean validation images.
This correlates with the test clean-FPR failure mode observed empirically.
"""
import argparse, csv, json, math, os, subprocess
import numpy as np, torch
from botorch.models import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

os.chdir('/wanqing-develop/luowenjing/18')
CONT=[('clean_penalty',0.0,2.0,'cont'),('lr_full',5e-6,1e-4,'log'),('lr_head',3e-4,3e-3,'log'),
      ('dropout',0.0,0.4,'cont'),('wd',0.01,0.2,'log')]
INT=[('clean_repeat',2,8)]
CAT=[('batch',[6,12,24]),('backbones',['dual','triple'])]
N=len(CONT)+len(INT)+len(CAT); CAT_DIMS=[len(CONT)+len(INT)+i for i in range(len(CAT))]
BOUNDS=torch.tensor([[0.0]*N,[1.0]*(len(CONT)+len(INT))+[len(v)-1 for _,v in CAT]],dtype=torch.double)

def encode(c):
    x=[]
    for name,lo,hi,kind in CONT:
        v=c[name]; v=(math.log(v)-math.log(lo))/(math.log(hi)-math.log(lo)) if kind=='log' else (v-lo)/(hi-lo)
        x.append(min(1.0,max(0.0,v)))
    for name,lo,hi in INT: x.append((c[name]-lo)/(hi-lo))
    for name,vals in CAT: x.append(float(vals.index(c[name])))
    return x

def decode(x):
    c={}
    for i,(name,lo,hi,kind) in enumerate(CONT):
        v=float(x[i]); v=math.exp(math.log(lo)+v*(math.log(hi)-math.log(lo))) if kind=='log' else lo+v*(hi-lo)
        c[name]=v
    for j,(name,lo,hi) in enumerate(INT): c[name]=int(round(lo+float(x[len(CONT)+j])*(hi-lo)))
    for k,(name,vals) in enumerate(CAT):
        idx=int(round(float(x[CAT_DIMS[k]]))); c[name]=vals[max(0,min(len(vals)-1,idx))]
    c['clean_penalty']=float(np.clip(c['clean_penalty'],0,2)); c['lr_full']=float(np.clip(c['lr_full'],5e-6,1e-4))
    c['lr_head']=float(np.clip(c['lr_head'],3e-4,3e-3)); c['dropout']=float(np.clip(c['dropout'],0,0.4))
    c['wd']=float(np.clip(c['wd'],0.01,0.2)); c['clean_repeat']=int(np.clip(c['clean_repeat'],2,8))
    if c['backbones']=='triple': c['batch']=min(c['batch'],6)
    return c

def key(c): return tuple(round(x,4) for x in encode(c))

def run_one(c,gpu,idx,seed,epochs,limit):
    out=f'results/hpo/run_{idx:03d}.json'; log=f'results/hpo/run_{idx:03d}.log'
    bbs={'dual':'eupe:full,siglip2:lastk4','triple':'eupe:full,siglip2:lastk4,cradio:lastk4'}[c['backbones']]
    cmd=['python3','-u','-I','code/finetune/train_multi.py','--backbones',bbs,'--seed',str(seed),
         '--epochs',str(epochs),'--batch',str(c['batch']),'--workers','8',
         '--clean-penalty',f"{c['clean_penalty']:.4f}",'--clean-repeat',str(c['clean_repeat']),
         '--lr-full',f"{c['lr_full']:.3e}",'--lr-head',f"{c['lr_head']:.3e}",
         '--wd',f"{c['wd']:.4f}",'--dropout',f"{c['dropout']:.3f}",'--out',out]
    if limit: cmd += ['--limit',str(limit)]
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
    f=open(log,'w'); p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    return p,out

def eval_batch(cfgs,gpus,start_idx,seed,epochs,limit):
    procs=[run_one(c,gpus[i%len(gpus)],start_idx+i,seed,epochs,limit) for i,c in enumerate(cfgs)]
    for p,_ in procs: p.wait()
    out_list=[]
    for p,out in procs:
        try:
            d=json.load(open(out)); pr=np.load(out.replace('.json','_probs.npz'),allow_pickle=True)
            clean=(pr['gva']==0).all(1)
            p99=float(np.percentile(pr['va_p0'][clean].max(1),99)) if clean.any() else 1.0
            v=float(d['val_total']); obj=v-10.0*max(0.0,p99-0.15)
            out_list.append((obj,v,p99,float(d['test']['total']),out))
        except Exception:
            out_list.append((0.0,0.0,1.0,0.0,out))
    return out_list

def save(hist):
    with open('results/hpo/history.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['idx','obj','val','p99','test','cfg','out'])
        for h in hist: w.writerow([h['idx'],h['obj'],h['val'],h['p99'],h['test'],json.dumps(h['cfg'],ensure_ascii=False),h['out']])
    json.dump(sorted(hist,key=lambda h:-h['obj'])[:10],open('results/hpo/best.json','w'),ensure_ascii=False,indent=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--init',type=int,default=20); ap.add_argument('--rounds',type=int,default=8)
    ap.add_argument('--q',type=int,default=7); ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--epochs',type=int,default=4); ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--gpus',default='0,1,2,3,4,6,7'); ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args(); gpus=[int(x) for x in a.gpus.split(',')]
    if a.smoke: a.init=2; a.rounds=0; a.q=2; a.epochs=1; a.limit=a.limit or 100
    os.makedirs('results/hpo',exist_ok=True); hist=[]; idx=0
    def add(cfgs,vals):
        nonlocal idx
        for c,(obj,v,p99,t,out) in zip(cfgs,vals):
            hist.append({'idx':idx,'cfg':c,'obj':obj,'val':v,'p99':p99,'test':t,'out':out}); idx+=1
            print(f"[{idx-1:03d}] obj={obj:.1f} val={v:.1f} p99={p99:.3f} test={t:.1f} {json.dumps(c,ensure_ascii=False)}",flush=True)
        save(hist)
    base=dict(clean_penalty=0.5,clean_repeat=3,lr_full=2e-5,lr_head=1e-3,dropout=0.2,wd=0.05,batch=6,backbones='dual')
    anchors=[dict(base),dict(base,backbones='triple'),dict(base,clean_penalty=1.9,clean_repeat=8,batch=12),
             dict(base,backbones='triple',lr_full=6.6e-6,lr_head=3.3e-4,dropout=0.1,wd=0.015,clean_penalty=0.47)]
    if not a.smoke: add(anchors,eval_batch(anchors,gpus,idx,a.seed,a.epochs,a.limit))
    if a.init>0:
        Xs=draw_sobol_samples(bounds=BOUNDS,n=a.init,q=1).squeeze(1)
        cfgs=[decode(x) for x in Xs]
        seen={key(h['cfg']) for h in hist}; cfgs=[c for c in cfgs if key(c) not in seen][:a.init]
        for i in range(0,len(cfgs),a.q):
            add(cfgs[i:i+a.q],eval_batch(cfgs[i:i+a.q],gpus,idx,a.seed,a.epochs,a.limit))
    for r in range(a.rounds):
        X=torch.tensor([encode(h['cfg']) for h in hist],dtype=torch.double); X[:,CAT_DIMS]=X[:,CAT_DIMS].round()
        Y=torch.tensor([[h['obj']] for h in hist],dtype=torch.double)
        ym,ys=Y.mean(),Y.std().clamp(min=1e-6); Yz=(Y-ym)/ys; noise=(4.0/ys.item())**2
        model=MixedSingleTaskGP(X,Yz,cat_dims=CAT_DIMS,train_Yvar=torch.full_like(Yz,noise))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model))
        acq=qLogNoisyExpectedImprovement(model=model,X_baseline=X,sampler=SobolQMCNormalSampler(torch.Size([512])))
        FFL=[{6:b,7:k} for b in range(len(CAT[0][1])) for k in range(len(CAT[1][1]))]
        cands,_=optimize_acqf_mixed(acq,bounds=BOUNDS,q=a.q,num_restarts=8,fixed_features_list=FFL,raw_samples=128)
        cfgs=[decode(c) for c in cands]
        seen={key(h['cfg']) for h in hist}; cfgs=[c for c in cfgs if key(c) not in seen] or cfgs
        add(cfgs,eval_batch(cfgs,gpus,idx,a.seed,a.epochs,a.limit))
        print(f'round {r+1} done, best obj={max(h["obj"] for h in hist):.1f} best test={max(h["test"] for h in hist):.1f}',flush=True)
    best=sorted(hist,key=lambda h:-h['obj'])[:5]
    print('TOP5 by obj:'); [print(f"  obj={h['obj']:.1f} val={h['val']:.1f} p99={h['p99']:.3f} test={h['test']:.1f} {json.dumps(h['cfg'],ensure_ascii=False)}") for h in best]
    print('TOP5 by test:'); [print(f"  test={h['test']:.1f} obj={h['obj']:.1f} {json.dumps(h['cfg'],ensure_ascii=False)}") for h in sorted(hist,key=lambda h:-h['test'])[:5]]
    best=sorted(hist,key=lambda h:-h['obj'])[:10]; json.dump(best,open('results/hpo/best.json','w'),ensure_ascii=False,indent=1)

if __name__=='__main__': main()
