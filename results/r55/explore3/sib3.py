# On top of a tower that HAS trained on the sibling frames (sibling-exposed folds), do the siblings' human labels,
# given explicitly, add anything?  pe tower, variants A and B pooled (620 held-out frames of multi-frame videos).
import os, sys, re, numpy as np, collections, warnings
warnings.filterwarnings("ignore")
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from sklearn.linear_model import LogisticRegression
from probe_eval import rps_dim
from eval_sib_study import load, vid
names,gold,fold,grp=c.load_split(); N=len(names)
V=np.array([vid(n) for n in names]); byv=collections.defaultdict(list)
for i,v in enumerate(V): byv[v].append(i)
multi=np.array([len(byv[v])>1 for v in V])
OH=np.eye(4)[gold]
Q=np.zeros((N,6,4))
for i in range(N):
    s=[j for j in byv[V[i]] if j!=i]
    if s: Q[i]=OH[s].mean(0)
G=c.renorm(load("fixed",2026,names))
rows=[]
for var in "AB":
    E=load(f"sibx{var}",2026,names); ok=np.isfinite(E).all(axis=(1,2)); E=c.renorm(np.where(ok[:,None,None],E,0.25))
    # training table for the combiner: every frame with an exposed prediction (multi hold-outs + single-frame videos)
    L=np.log(np.clip(E,1e-6,1)); h=(multi&ok).astype(float)[:,None]
    out={}
    for lab,fn in (("exposed tower via LR",lambda d: L[:,d,:]),("exposed tower + sibling labels",lambda d: np.concatenate([L[:,d,:],Q[:,d,:]*h,h],1))):
        R=np.zeros((N,6,4))
        for d in range(6):
            X=fn(d)
            for f in range(8):
                tr,te=(fold!=f)&ok,(fold==f)&ok
                m=LogisticRegression(C=1.0,max_iter=3000).fit(X[tr],gold[tr,d]); pr=np.zeros((te.sum(),4)); pr[:,m.classes_]=m.predict_proba(X[te]); R[te,d]=pr
        out[lab]=c.renorm(np.where(ok[:,None,None],R,0.25))
    m=ok&multi
    rp=lambda P: np.stack([rps_dim(P[:,d],gold[:,d],d) for d in range(6)],1)
    rows.append((V[m],rp(G)[m],rp(E)[m],rp(out["exposed tower via LR"])[m],rp(out["exposed tower + sibling labels"])[m]))
vids=np.concatenate([r[0] for r in rows]); R=[np.concatenate([r[k] for r in rows]) for k in range(1,5)]
ug,inv=np.unique(vids,return_inverse=True); mem=[np.where(inv==k)[0] for k in range(len(ug))]; rs=np.random.RandomState(0)
boots=[np.concatenate([mem[j] for j in rs.randint(0,len(ug),len(ug))]) for _ in range(1000)]
labs=["grouped tower (video unseen)","exposed tower (siblings trained on)","exposed tower via LR (control)","exposed tower + sibling labels"]
for lab,r in zip(labs,R): print("%-40s mean RPS %.5f | per defect %s"%(lab,r.mean()," ".join("%.4f"%x for x in r.mean(0))))
x=R[3].mean(1)-R[2].mean(1); dd=np.array([x[b].mean() for b in boots])
print("sibling labels on top of the exposed tower: dRPS %+.5f (%+.1f%%) CI90 [%+.5f,%+.5f]"%(x.mean(),100*x.mean()/R[2].mean(),*np.quantile(dd,[.05,.95])))
for d,nm in enumerate(["模糊","纹理损失","过锐","花屏","过度生成","虚化突兀"]):
    x=R[3][:,d]-R[2][:,d]; dd=np.array([x[b].mean() for b in boots]); print("   %s %+.5f [%+.5f,%+.5f]"%(nm,x.mean(),*np.quantile(dd,[.05,.95])))
