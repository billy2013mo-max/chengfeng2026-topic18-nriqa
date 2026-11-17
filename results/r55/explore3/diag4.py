# (a) linear cross-defect second stage (low-variance version of the boosting test);
# (b) is the grade residual autocorrelated along spreadsheet row order (annotation-session effect)?
import os, sys, re, csv, numpy as np, collections, warnings, openpyxl
warnings.filterwarnings("ignore")
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import argmax_decode, nested_oof_decode
from sklearn.linear_model import LogisticRegression
z=np.load(sys.argv[1],allow_pickle=True)
names=[str(x) for x in z["names"]]; gold=z["gold"]; fold=z["fold"]; grp=z["grp"]; P=z["P"]
K="doc2_strict_total"
def rps(p,g):
    cp=np.cumsum(p,-1)[...,:3]; cg=(g[...,None]<=np.arange(3)).astype(float); return ((cp-cg)**2).sum(-1)
def run(mode,label,C=1.0):
    Q=np.zeros_like(P); L=np.log(np.clip(P,1e-6,1))
    for d in range(6):
        if mode=="own": X=L[:,d,:]
        else: X=L.reshape(len(P),-1)
        for f in range(8):
            tr,te=fold!=f,fold==f
            m=LogisticRegression(C=C,max_iter=3000).fit(X[tr],gold[tr,d])
            pr=np.zeros((te.sum(),4)); pr[:,m.classes_]=m.predict_proba(X[te]); Q[te,d]=pr
    Q=c.renorm(Q); r=rps(Q,gold)
    nd=nested_oof_decode(Q,gold,fold,K); sc=c.score(nd,gold)
    print("%-30s meanRPS %.5f | "%(label,r.mean())+" ".join("%.4f"%x for x in r.mean(0))+" | argmax %.2f nested %.2f"%(c.score(argmax_decode(Q),gold)[K],sc[K]))
    return r
r0=rps(P,gold); print("%-30s meanRPS %.5f | "%("raw pool",r0.mean())+" ".join("%.4f"%x for x in r0.mean(0)))
ra=run("own","own-defect LR (control)")
for C in (0.03,0.3,3.0):
    rx=run("all","cross-defect LR C=%g"%C,C)
ug,inv=np.unique(grp,return_inverse=True); mem=[np.where(inv==i)[0] for i in range(len(ug))]
rs=np.random.RandomState(0); ds=[]
for _ in range(300):
    ii=np.concatenate([mem[i] for i in rs.randint(0,len(ug),len(ug))]); ds.append(rx[ii].mean()-ra[ii].mean())
print("  cross (last C) vs control dRPS %+.5f CI90 [%+.5f,%+.5f]"%(np.mean(ds),np.quantile(ds,.05),np.quantile(ds,.95)))
# (b) row-order autocorrelation of residual expected grade
order={}
k=0
for f in ["dataset/总表_训练集结果.xlsx","dataset/总表_训练集补充结果.xlsx"]:
    ws=openpyxl.load_workbook(os.path.join(REPO,f),read_only=True).worksheets[0]
    for i,row in enumerate(ws.iter_rows(values_only=True)):
        if i==0 or row[0] is None: continue
        order[str(row[0]).strip()]=k; k+=1
miss=[n for n in names if n not in order]; print("names not found in sheets:",len(miss), miss[:3])
idx=np.array([order.get(n,-1) for n in names]); ok=idx>=0
E=(P*np.arange(4)).sum(-1); res=gold-E          # signed residual per defect
o=np.argsort(idx[ok]); R=res[ok][o]; G=grp[ok][o]; I=idx[ok][o]
DIMS=["模糊","纹理损失","过锐","花屏","过度生成","虚化突兀"]
rs=np.random.RandomState(1)
for lag in (1,5,20,50):
    out=[]
    for d in range(6):
        a,b=R[:-lag,d],R[lag:,d]; diff=G[:-lag]!=G[lag:]          # only pairs from different videos
        near=(I[lag:]-I[:-lag])==lag
        m=diff&near
        rr=np.corrcoef(a[m],b[m])[0,1]
        out.append("%s %+.3f"%(DIMS[d],rr))
    print("lag %2d (different videos, n=%d): "%(lag,m.sum())+"  ".join(out))
# block effect: variance of residual explained by 50-row blocks vs random partition
for d in (0,1,2,4):
    blk=I//50
    def eta(b):
        tot=((R[:,d]-R[:,d].mean())**2).sum(); 
        bw=sum(((R[b==u,d].mean()-R[:,d].mean())**2)*(b==u).sum() for u in np.unique(b)); return bw/tot
    nul=[eta(rs.permutation(blk)) for _ in range(200)]
    print("%s: residual variance explained by 50-row blocks %.3f  (random blocks %.3f ± %.3f)"%(DIMS[d],eta(blk),np.mean(nul),np.std(nul)))
