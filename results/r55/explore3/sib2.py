# Video-level correction: for a frame t whose video has other labelled frames s, use the RESIDUAL of the siblings
# (human label minus the out-of-fold prediction of the sibling) as extra evidence.  Simulated inside the labelled data
# (grouped folds: the pool never saw the video of t or s); combiner fitted on the other folds.
import os, sys, re, numpy as np, collections, warnings
warnings.filterwarnings("ignore")
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import argmax_decode, nested_oof_decode
from sklearn.linear_model import LogisticRegression
from probe_eval import rps_dim
z=np.load(sys.argv[1],allow_pickle=True)
names=[str(x) for x in z["names"]]; gold=z["gold"]; fold=z["fold"]; P=c.renorm(z["P"])
K="doc2_strict_total"; D=["模糊","纹理损失","过锐","花屏","过度生成","虚化突兀"]
def vid(n):
    m=re.match(r'^(?:\d{3}_)?([0-9a-f]{8,}|\d{9,})',n); return m.group(1) if m else n
V=np.array([vid(n) for n in names]); N=len(names)
byv=collections.defaultdict(list)
for i,v in enumerate(V): byv[v].append(i)
has=np.array([len(byv[v])>1 for v in V])
OH=np.eye(4)[gold]                                   # (N,6,4)
Q=np.zeros((N,6,4)); R=np.zeros((N,6,4)); PS=np.zeros((N,6,4)); E=np.zeros((N,6))
G=np.arange(4)
for i in range(N):
    s=[j for j in byv[V[i]] if j!=i]
    if not s: continue
    Q[i]=OH[s].mean(0); PS[i]=P[s].mean(0); R[i]=(OH[s]-P[s]).mean(0); E[i]=((OH[s]-P[s])*G).sum(-1).mean(0)   # E: signed grade residual
L=np.log(np.clip(P,1e-6,1)); h=has.astype(float)[:,None]
def combine(feat_fn,C=1.0):
    X0=np.zeros_like(P)
    for d in range(6):
        X=feat_fn(d)
        for f in range(8):
            tr,te=fold!=f,fold==f
            m=LogisticRegression(C=C,max_iter=3000).fit(X[tr],gold[tr,d])
            pr=np.zeros((te.sum(),4)); pr[:,m.classes_]=m.predict_proba(X[te]); X0[te,d]=pr
    return c.renorm(X0)
def rps(Pp): return np.stack([rps_dim(Pp[:,d],gold[:,d],d) for d in range(6)],1)
variants={
 "control (own log-probs)": lambda d: L[:,d,:],
 "+ sibling labels": lambda d: np.concatenate([L[:,d,:],Q[:,d,:],h],1),
 "+ sibling residual (label - prediction)": lambda d: np.concatenate([L[:,d,:],R[:,d,:],h],1),
 "+ sibling labels + sibling predictions": lambda d: np.concatenate([L[:,d,:],Q[:,d,:],PS[:,d,:],h],1),
 "+ signed grade residual (1 number)": lambda d: np.concatenate([L[:,d,:],E[:,[d]],h],1),
}
out={}
ref=None
ug,inv=np.unique(V[has],return_inverse=True); idx=np.where(has)[0]; rs=np.random.RandomState(0)
mem=[idx[inv==k] for k in range(len(ug))]; boots=[np.concatenate([mem[j] for j in rs.randint(0,len(ug),len(ug))]) for _ in range(400)]
for lab,fn in variants.items():
    Pp=combine(fn); r=rps(Pp); out[lab]=Pp
    if ref is None: ref=r
    d=np.array([r[b].mean()-ref[b].mean() for b in boots])
    print("%-42s sibling frames RPS %.5f (%+.1f%%, CI90 [%+.5f,%+.5f]) per defect %s | no-sibling frames %.5f"%(lab,r[has].mean(),100*(r[has].mean()-ref[has].mean())/ref[has].mean(),np.quantile(d,.05),np.quantile(d,.95)," ".join("%.4f"%x for x in r[has].mean(0)),r[~has].mean()))
print("--- decoded on sibling frames (decoder nested by fold, fitted on all frames of the other folds) ---")
dec={k:nested_oof_decode(v,gold,fold,K) for k,v in out.items()}
base=dec["control (own log-probs)"]
for k,v in dec.items():
    sc=c.score(v[has],gold[has]); dd=np.array([c.score(v[b],gold[b])[K]-c.score(base[b],gold[b])[K] for b in boots[:300]])
    print("%-42s total %.2f = %.2f + %.2f + %.2f | vs control %+.2f CI90 [%+.2f,%+.2f] | F1 %s"%(k,sc[K],sc["doc2_S_pos"],sc["doc2_S_p0"],sc["doc2_Clean2"],dd.mean(),np.quantile(dd,.05),np.quantile(dd,.95)," ".join("%.2f"%r["F1"] for r in sc["per_14"])))
