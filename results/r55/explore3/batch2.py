# Does a tower trained on the first batch only add something to the ENSEMBLE on first-batch images?
import sys,os,csv,numpy as np
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import argmax_decode, nested_oof_decode
from probe_eval import rps_dim, DIMS
from eval_batch_study import load
z=np.load(sys.argv[1],allow_pickle=True); names=[str(x) for x in z["names"]]; gold=z["gold"]; grp=z["grp"]; fold=z["fold"]
sp={r['name']:r['split'] for r in csv.DictReader(open(REPO+'/data/manifest_combined.csv',encoding='utf-8'))}
o=np.array([sp[n]=='orig' for n in names]); K="doc2_strict_total"
keys=["radio","pe","sig","v16","p1","p3"]; M={k:c.renorm(z["m_"+k]) for k in keys}
M["fx"]=c.renorm(load("fixed",2026,names)); M["oo"]=c.renorm(load("origonly",2026,names)); M["so"]=c.renorm(load("supponly",2026,names))
def rpsd(P): return np.stack([rps_dim(P[:,d],gold[:,d],d) for d in range(6)],1)
def pool(w): return c.renorm(sum(v*M[k] for k,v in w.items())/sum(w.values()))
ug,inv=np.unique(grp,return_inverse=True); rs=np.random.RandomState(0)
def boots_for(mask):
    mem=[np.where((inv==i)&mask)[0] for i in range(len(ug))]; mem=[m for m in mem if len(m)]
    return [np.concatenate([mem[j] for j in rs.randint(0,len(mem),len(mem))]) for _ in range(400)]
bo=boots_for(o); bs=boots_for(~o)
bal={"radio":1/3,"pe":1/3,"sig":1/3,"v16":1.0,"p1":0.5,"p3":0.5}
ref=rpsd(pool(bal))
def line(lab,w,refr=ref):
    P=pool(w); r=rpsd(P)
    d=np.array([r[b].mean()-refr[b].mean() for b in bo]); ds=np.array([r[b].mean()-refr[b].mean() for b in bs])
    am=c.score(argmax_decode(P[o]),gold[o])[K]
    print("%-52s first batch: RPS %.5f  d %+.5f [%+.5f,%+.5f] (%+.1f%%) | per defect d: %s | argmax %.2f || supp: d %+.5f [%+.5f,%+.5f]"%(lab,r[o].mean(),r[o].mean()-refr[o].mean(),np.quantile(d,.05),np.quantile(d,.95),100*(r[o].mean()-refr[o].mean())/refr[o].mean()," ".join("%+.4f"%x for x in (r[o].mean(0)-refr[o].mean(0))),am,r[~o].mean()-refr[~o].mean(),np.quantile(ds,.05),np.quantile(ds,.95)))
    return P
print("single towers on first-batch images: fixed %.5f  origonly %.5f  mean(fixed,origonly) %.5f | the original pe member %.5f"%(rpsd(M["fx"])[o].mean(),rpsd(M["oo"])[o].mean(),rpsd(pool({"fx":1,"oo":1}))[o].mean(),rpsd(M["pe"])[o].mean()))
print("reference = primary pool: first-batch RPS %.5f, argmax %.2f"%(ref[o].mean(),c.score(argmax_decode(pool(bal)[o]),gold[o])[K]))
line("primary with pe -> fixed (same recipe, sanity)",{**bal,"pe":0,"fx":1/3})
line("primary with pe -> origonly",{**bal,"pe":0,"oo":1/3})
line("primary + origonly as 4th tier-1 member",{"radio":.25,"pe":.25,"sig":.25,"oo":.25,"v16":1.0,"p1":0.5,"p3":0.5})
line("primary + fixed as 4th tier-1 member (control)",{"radio":.25,"pe":.25,"sig":.25,"fx":.25,"v16":1.0,"p1":0.5,"p3":0.5})
P4=line("primary + origonly as its own family (weight 1)",{**bal,"oo":1.0})
line("primary + fixed as its own family (control)",{**bal,"fx":1.0})
line("primary + origonly, family weight 0.5",{**bal,"oo":0.5})
line("primary + origonly, family weight 2",{**bal,"oo":2.0})
np.savez(sys.argv[2],P4=P4)
