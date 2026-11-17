# Is the DECODER sensitive to sibling exposure?  pe tower only.  Thresholds fitted (nested by fold) on
#   (g) grouped out-of-fold probabilities (what the deliverable's decoder was fitted on), or
#   (e) sibling-exposed probabilities with the sibling hold-outs up-weighted to 63 % of the fitting set
# and both applied to the exposed probabilities of the held-out frames of multi-frame videos.
import os, sys, numpy as np, collections
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import Decoder55, argmax_decode
from eval_sib_study import load, vid
names,gold,fold,grp=c.load_split(); K="doc2_strict_total"
V=np.array([vid(n) for n in names]); cnt=collections.Counter(V); multi=np.array([cnt[v]>1 for v in V])
G=c.renorm(load("fixed",2026,names))
res=collections.defaultdict(list); keep=[]
for var in "AB":
    E=load(f"sibx{var}",2026,names); ok=np.isfinite(E).all(axis=(1,2)); E=c.renorm(np.where(ok[:,None,None],E,0.25))
    pg=np.zeros_like(gold); pe=np.zeros_like(gold); pge=np.zeros_like(gold)
    for f in range(8):
        tr=fold!=f; te=(fold==f)&ok&multi
        dg=Decoder55(K).fit(G[tr],gold[tr])
        # exposed fitting set: sibling hold-outs replicated so that they are ~63 % of the rows
        a=np.where(tr&ok&multi)[0]; b=np.where(tr&ok&~multi)[0]; rep=max(1,int(round(0.63/0.37*len(b)/len(a))))
        ii=np.concatenate([np.repeat(a,rep),b]); de=Decoder55(K).fit(E[ii],gold[ii])
        pg[te]=dg.predict(E[te]); pe[te]=de.predict(E[te]); pge[te]=dg.predict(G[te])
    m=ok&multi
    for lab,p in (("grouped probs + grouped decoder",pge),("exposed probs + grouped decoder",pg),("exposed probs + exposed decoder",pe),("exposed probs, argmax",argmax_decode(E))):
        s=c.score(p[m],gold[m]); res[lab].append((s[K],s["doc2_S_pos"],s["doc2_S_p0"],s["doc2_Clean2"]))
    keep.append((V[m],pg[m],pe[m],gold[m]))
for lab,v in res.items():
    v=np.array(v); print("%-34s variants A/B: %.2f / %.2f | mean %.2f = pos %.2f + p0 %.2f + clean %.2f"%(lab,v[0,0],v[1,0],*v.mean(0)))
vids=np.concatenate([k[0] for k in keep]); pg=np.concatenate([k[1] for k in keep]); pe=np.concatenate([k[2] for k in keep]); gd=np.concatenate([k[3] for k in keep])
ug,inv=np.unique(vids,return_inverse=True); mem=[np.where(inv==k)[0] for k in range(len(ug))]; rs=np.random.RandomState(0); d=[]
for _ in range(500):
    b=np.concatenate([mem[j] for j in rs.randint(0,len(ug),len(ug))]); d.append(c.score(pe[b],gd[b])[K]-c.score(pg[b],gd[b])[K])
print("exposed decoder - grouped decoder on exposed probabilities: %+.2f CI90 [%+.2f,%+.2f]"%(np.mean(d),*np.quantile(d,[.05,.95])))
