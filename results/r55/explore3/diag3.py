# How much do file-name side facts (pipeline tag, batch, mining keyword) know beyond the pooled probabilities?
# ANALYSIS ONLY: these are not image content. Nested by fold (video-grouped).
import os, sys, re, csv, numpy as np, collections, warnings
warnings.filterwarnings("ignore")
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import argmax_decode, nested_oof_decode
from sklearn.linear_model import LogisticRegression
z=np.load(sys.argv[1],allow_pickle=True)
names=[str(x) for x in z["names"]]; gold=z["gold"]; fold=z["fold"]; grp=z["grp"]; P=z["P"]
DIMS=["模糊","纹理损失","过锐","花屏","过度生成","虚化突兀"]; K="doc2_strict_total"
sp={r['name']:r['split'] for r in csv.DictReader(open(REPO+'/data/manifest_combined.csv',encoding='utf-8'))}
def tag(n):
    m=re.search(r'_((?:v\d+|k\d+|sl\d+)[A-Za-z]*?\d*[A-Za-z]*?(?:V\d+|VE|hd\d+))(?=[._]|$)',n)
    return m.group(1) if m else "none"
def kw(n):
    for k in ["overgeneration","detail_loss","blur","abrupt_bokeh","oversharpening","过锐","模糊","虚化突兀","过度生成","花屏","细节丢失","sharp","sample","frame"]:
        if k in n: return k
    if re.search(r'\d+\.\d+s',n): return "ts"
    return "other"
T=np.array([tag(n) for n in names]); Kw=np.array([kw(n) for n in names]); Bt=np.array([sp[n] for n in names])
print(collections.Counter(T).most_common(30)); print(collections.Counter(Kw).most_common())
def onehot(v,minc=15):
    cnt=collections.Counter(v); keys=[k for k,n in cnt.items() if n>=minc]
    return np.stack([(v==k).astype(float) for k in keys],1), keys
def rps(p,g):
    cp=np.cumsum(p,-1)[...,:3]; cg=(g[...,None]<=np.arange(3)).astype(float); return ((cp-cg)**2).sum(-1)
def run(feats,label):
    Q=np.zeros_like(P)
    for d in range(6):
        X0=np.log(np.clip(P[:,d,:],1e-6,1))
        X=X0 if feats is None else np.concatenate([X0,feats],1)
        for f in range(8):
            tr,te=fold!=f,fold==f
            m=LogisticRegression(C=1.0,max_iter=2000).fit(X[tr],gold[tr,d])
            pr=np.zeros((te.sum(),4)); pr[:,m.classes_]=m.predict_proba(X[te]); Q[te,d]=pr
    Q=c.renorm(Q)
    r=rps(Q,gold)
    nd=nested_oof_decode(Q,gold,fold,K); sc=c.score(nd,gold); am=c.score(argmax_decode(Q),gold)[K]
    print("%-28s meanRPS %.5f | per-defect "%(label,r.mean())+" ".join("%.4f"%x for x in r.mean(0))+" | argmax %.2f nested %.2f (pos %.2f p0 %.2f clean %.2f)"%(am,sc[K],sc["doc2_S_pos"],sc["doc2_S_p0"],sc["doc2_Clean2"]))
    return r
r0=rps(P,gold); print("%-28s meanRPS %.5f | per-defect "%("raw pool",r0.mean())+" ".join("%.4f"%x for x in r0.mean(0)))
ra=run(None,"pool via LR (control)")
ft,_=onehot(T); rt=run(ft,"+ pipeline tag")
fb=(Bt=='supp').astype(float)[:,None]; rb=run(fb,"+ batch flag")
fk,_=onehot(Kw); rk=run(fk,"+ filename keyword")
rall=run(np.concatenate([ft,fb,fk],1),"+ tag + batch + keyword")
# cluster bootstrap for the biggest
ug,inv=np.unique(grp,return_inverse=True); mem=[np.where(inv==i)[0] for i in range(len(ug))]
rs=np.random.RandomState(0)
for lab,r in [("tag",rt),("batch",rb),("keyword",rk),("all",rall)]:
    ds=[]
    for _ in range(300):
        ii=np.concatenate([mem[i] for i in rs.randint(0,len(ug),len(ug))]); ds.append(r[ii].mean()-ra[ii].mean())
    ds=np.array(ds); print("  %s vs control: dRPS %+.5f CI90 [%+.5f,%+.5f]"%(lab,ds.mean(),np.quantile(ds,.05),np.quantile(ds,.95)))
