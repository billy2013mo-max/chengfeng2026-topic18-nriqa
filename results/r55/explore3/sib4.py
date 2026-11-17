# Pool-level view of the sibling-exposure effect with what exists: only the pe member has sibling-exposed runs.
# On the held-out frames of multi-frame videos: primary pool with the pe member grouped (as in all OOF numbers) vs
# the same pool with the pe member exposed.  Other members stay grouped, so this is a LOWER bound for a fully
# exposed pool and an indication of how much of one member's gain survives pooling.
import os, sys, numpy as np, collections
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import argmax_decode
from probe_eval import rps_dim
from eval_sib_study import load, vid
z=np.load(sys.argv[1],allow_pickle=True); names=[str(x) for x in z["names"]]; gold=z["gold"]
keys=["radio","pe","sig","v16","p1","p3"]; M={k:c.renorm(z["m_"+k]) for k in keys}
w={"radio":1/3,"pe":1/3,"sig":1/3,"v16":1.0,"p1":0.5,"p3":0.5}
G=c.renorm(load("fixed",2026,names)); V=np.array([vid(n) for n in names]); cnt=collections.Counter(V); multi=np.array([cnt[v]>1 for v in V])
def pool(pe): return c.renorm((w["radio"]*M["radio"]+w["pe"]*pe+w["sig"]*M["sig"]+w["v16"]*M["v16"]+w["p1"]*M["p1"]+w["p3"]*M["p3"])/3.0)
rp=lambda P: np.stack([rps_dim(P[:,d],gold[:,d],d) for d in range(6)],1).mean(1)
K="doc2_strict_total"; rows=[]
for var in "AB":
    E=load(f"sibx{var}",2026,names); ok=np.isfinite(E).all(axis=(1,2)); E=c.renorm(np.where(ok[:,None,None],E,0.25)); m=ok&multi
    Pg,Pe=pool(G),pool(E)
    rows.append((V[m],rp(G)[m],rp(E)[m],rp(Pg)[m],rp(Pe)[m]))
    print("variant %s (%d frames): pe alone argmax %.2f -> %.2f | pool argmax %.2f -> %.2f (only the pe member exposed)"%(var,m.sum(),c.score(argmax_decode(G[m]),gold[m])[K],c.score(argmax_decode(E[m]),gold[m])[K],c.score(argmax_decode(Pg[m]),gold[m])[K],c.score(argmax_decode(Pe[m]),gold[m])[K]))
vids=np.concatenate([r[0] for r in rows]); R=[np.concatenate([r[k] for r in rows]) for k in range(1,5)]
ug,inv=np.unique(vids,return_inverse=True); mem=[np.where(inv==k)[0] for k in range(len(ug))]; rs=np.random.RandomState(0)
boots=[np.concatenate([mem[j] for j in rs.randint(0,len(ug),len(ug))]) for _ in range(1000)]
for lab,a,b in (("pe tower alone",R[0],R[1]),("primary pool, pe member swapped",R[2],R[3])):
    x=b-a; dd=np.array([x[i].mean() for i in boots]); print("%-34s RPS %.5f -> %.5f  (%+.1f%%) CI90 [%+.5f,%+.5f]"%(lab,a.mean(),b.mean(),100*x.mean()/a.mean(),*np.quantile(dd,[.05,.95])))
