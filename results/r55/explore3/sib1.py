# How much do the HUMAN LABELS of other frames of the same video (which would be TRAINING rows when the test frame
# comes from a video that is also in the training set) add to the ensemble?  Simulation inside the labelled data:
# grouped CV keeps all frames of a video in one fold, so for frame i the pool never saw its video, and the labels of
# its siblings play the role of "training labels of the same video".  The combiner is fitted on the other folds only.
import os, sys, re, numpy as np, collections, warnings
warnings.filterwarnings("ignore")
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import argmax_decode, nested_oof_decode, Decoder55
from sklearn.linear_model import LogisticRegression
z=np.load(sys.argv[1],allow_pickle=True)
names=[str(x) for x in z["names"]]; gold=z["gold"]; fold=z["fold"]; grp=z["grp"]; P=c.renorm(z["P"]); nd=z["nested"]
T=np.load(sys.argv[2])["T"]
K="doc2_strict_total"; D=["模糊","纹理损失","过锐","花屏","过度生成","虚化突兀"]
def vid(n):
    m=re.match(r'^(?:\d{3}_)?([0-9a-f]{8,}|\d{9,})',n); return m.group(1) if m else n
V=np.array([vid(n) for n in names]); N=len(names)
byv=collections.defaultdict(list)
for i,v in enumerate(V): byv[v].append(i)
sibs=[[j for j in byv[V[i]] if j!=i] for i in range(N)]
has=np.array([len(s)>0 for s in sibs])
Q=np.zeros((N,6,4)); sim=np.zeros(N); Qn=np.zeros((N,6,4))     # Q: mean one-hot of sibling labels; Qn: nearest sibling only
for i in range(N):
    if not sibs[i]: continue
    for j in sibs[i]:
        for d in range(6): Q[i,d,gold[j,d]]+=1.0/len(sibs[i])
    s=[float(T[i]@T[j]) for j in sibs[i]]; j=sibs[i][int(np.argmax(s))]; sim[i]=max(s)
    for d in range(6): Qn[i,d,gold[j,d]]=1.0
print("frames with a sibling: %d of %d | thumbnail correlation to nearest sibling: median %.2f, quartiles %.2f / %.2f"%(has.sum(),N,np.median(sim[has]),*np.quantile(sim[has],[.25,.75])))
def rps(p,g):
    cp=np.cumsum(p,-1)[...,:3]; cg=(g[...,None]<=np.arange(3)).astype(float); return ((cp-cg)**2).sum(-1)/3
def show(tag,pred,m):
    sc=c.score(pred[m],gold[m]); print("  %-46s total %.2f = pos %.2f + p0 %.2f + clean %.2f | F1 "%(tag,sc[K],sc["doc2_S_pos"],sc["doc2_S_p0"],sc["doc2_Clean2"])+" ".join("%.2f"%r["F1"] for r in sc["per_14"]))
    return sc[K]
# agreement of sibling labels
print("label agreement with the nearest sibling (exact grade): "+" ".join("%s %.2f"%(D[d],(Qn[has,d].argmax(-1)==gold[has,d]).mean()) for d in range(6)), "| all six equal: %.2f"%((Qn[has].argmax(-1)==gold[has]).all(1).mean()))
print("ensemble argmax agreement with the label on the same frames:   "+" ".join("%s %.2f"%(D[d],(P[has,d].argmax(-1)==gold[has,d]).mean()) for d in range(6)), "| all six equal: %.2f"%((P[has].argmax(-1)==gold[has]).all(1).mean()))
print("--- frames WITH a sibling (n=%d) ---"%has.sum())
show("ensemble, nested decoder (never saw the video)",nd,has)
show("copy the nearest sibling's labels",Qn.argmax(-1),has)
# combiner
def combine(feat_fn,label):
    R=np.zeros_like(P)
    for d in range(6):
        X=feat_fn(d)
        for f in range(8):
            tr,te=fold!=f,fold==f
            m=LogisticRegression(C=1.0,max_iter=3000).fit(X[tr],gold[tr,d])
            pr=np.zeros((te.sum(),4)); pr[:,m.classes_]=m.predict_proba(X[te]); R[te,d]=pr
    return c.renorm(R)
L=np.log(np.clip(P,1e-6,1))
h=has.astype(float)[:,None]
f_ctrl=lambda d: L[:,d,:]
f_sib =lambda d: np.concatenate([L[:,d,:], Q[:,d,:], h],1)
f_sib2=lambda d: np.concatenate([L[:,d,:], Q[:,d,:], h, Qn[:,d,:]*sim[:,None], (sim*has)[:,None]],1)
Rc=combine(f_ctrl,"ctrl"); Rs=combine(f_sib,"sib"); Rs2=combine(f_sib2,"sib2")
for lab,R in (("pool via LR (control)",Rc),("pool + sibling labels",Rs),("pool + sibling labels + similarity",Rs2)):
    r=rps(R,gold); 
    print("  %-36s RPS on sibling frames %.5f (per defect %s) | on frames without sibling %.5f"%(lab,r[has].mean()," ".join("%.4f"%x for x in r[has].mean(0)),r[~has].mean()))
ndc=nested_oof_decode(Rc,gold,fold,K); nds=nested_oof_decode(Rs,gold,fold,K); nds2=nested_oof_decode(Rs2,gold,fold,K)
print("--- decoded (decoder nested by fold on the same probabilities) ---")
for m,lab in ((has,"frames with a sibling"),(~has,"frames without a sibling"),(np.ones(N,bool),"all frames")):
    print(lab,"n=%d"%m.sum())
    a=show("control",ndc,m); b=show("pool + sibling labels",nds,m); b2=show("pool + sibling labels + similarity",nds2,m)
np.savez(sys.argv[3],has=has,Rc=Rc,Rs=Rs,Rs2=Rs2,ndc=ndc,nds=nds,nds2=nds2,Q=Q,Qn=Qn,sim=sim)
# bootstrap over videos for the difference on sibling frames, and for a practice-like mix (63 % sibling frames)
ug,inv=np.unique(V,return_inverse=True); rs=np.random.RandomState(0)
hv=[np.where((inv==i))[0] for i in range(len(ug))]; hs=[x for x in hv if has[x[0]]]; hn=[x for x in hv if not has[x[0]]]
d1=[];d2=[]
for _ in range(300):
    a=np.concatenate([hs[i] for i in rs.randint(0,len(hs),len(hs))])
    d1.append(c.score(nds[a],gold[a])[K]-c.score(ndc[a],gold[a])[K])
    # practice-like mix: 412 sibling frames + 242 others = 654
    a2=rs.choice(np.where(has)[0],412,replace=True); b2=rs.choice(np.where(~has)[0],242,replace=True); ii=np.concatenate([a2,b2])
    d2.append((c.score(nds[ii],gold[ii])[K],c.score(ndc[ii],gold[ii])[K]))
d1=np.array(d1); d2=np.array(d2)
print("sibling frames: decoded gain %+.2f  CI90 [%+.2f, %+.2f]"%(d1.mean(),np.quantile(d1,.05),np.quantile(d1,.95)))
print("654-image mix with 63 %% sibling frames: control %.2f ± %.2f -> with sibling labels %.2f ± %.2f | gain %+.2f CI90 [%+.2f, %+.2f]"%(d2[:,1].mean(),d2[:,1].std(),d2[:,0].mean(),d2[:,0].std(),(d2[:,0]-d2[:,1]).mean(),*np.quantile(d2[:,0]-d2[:,1],[.05,.95])))
