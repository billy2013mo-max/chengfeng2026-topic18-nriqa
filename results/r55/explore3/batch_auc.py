# Release-batch study, discrimination only: detection / adjacent-grade AUC of the pe tower on first-batch validation
# images when trained on both batches (fixed) or on the first batch only (origonly); paired bootstrap over videos.
import sys,os,csv,numpy as np
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from probe_eval import skills, DIMS
from eval_batch_study import load
from decoder55 import nested_oof_decode
names,gold,fold,grp=c.load_split()
sp={r['name']:r['split'] for r in csv.DictReader(open(REPO+'/data/manifest_combined.csv',encoding='utf-8'))}
o=np.array([sp[n]=='orig' for n in names]); K="doc2_strict_total"
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 2026
BBN=sys.argv[2] if len(sys.argv)>2 else "pe_core_large"
A=c.renorm(load("fixed",SEED,names,BBN)); B=c.renorm(load("origonly",SEED,names,BBN))
print("backbone",BBN,"seed",SEED)
def summ(P,idx):
    det=[];gr=[]
    for d in range(6):
        s=skills(P[idx,d],gold[idx,d],d); det.append(s[0])
        if d in (0,1,2,4): gr.append(np.nanmean(s[1:]))
    return np.mean(det),np.mean(gr)
idx=np.where(o)[0]; ug,inv=np.unique(grp[idx],return_inverse=True); mem=[idx[inv==k] for k in range(len(ug))]; rs=np.random.RandomState(0)
a=summ(A,idx); b=summ(B,idx); dd=[]
for _ in range(300):
    bi=np.concatenate([mem[j] for j in rs.randint(0,len(ug),len(ug))]); x=summ(A,bi); y=summ(B,bi); dd.append((y[0]-x[0],y[1]-x[1]))
dd=np.array(dd)
print("first-batch validation images (n=%d): mean detection AUC both batches %.4f, first batch only %.4f, difference %+.4f CI90 [%+.4f,%+.4f]"%(len(idx),a[0],b[0],b[0]-a[0],*np.quantile(dd[:,0],[.05,.95])))
print("                                        mean adjacent-grade AUC both %.4f, first only %.4f, difference %+.4f CI90 [%+.4f,%+.4f]"%(a[1],b[1],b[1]-a[1],*np.quantile(dd[:,1],[.05,.95])))
# decoded with a decoder nested by fold and fitted on first-batch images only
da=nested_oof_decode(A[o],gold[o],fold[o],K); db=nested_oof_decode(B[o],gold[o],fold[o],K)
sa=c.score(da,gold[o]); sb=c.score(db,gold[o])
go=gold[o]; g2=grp[o]; ug,inv=np.unique(g2,return_inverse=True); mem=[np.where(inv==k)[0] for k in range(len(ug))]; d=[]
for _ in range(400):
    bi=np.concatenate([mem[j] for j in rs.randint(0,len(ug),len(ug))]); d.append(c.score(db[bi],go[bi])[K]-c.score(da[bi],go[bi])[K])
print("decoded on first-batch images (decoder nested, fitted on first-batch images): both batches %.2f (%.2f/%.2f/%.2f), first batch only %.2f (%.2f/%.2f/%.2f), difference %+.2f CI90 [%+.2f,%+.2f]"%(sa[K],sa["doc2_S_pos"],sa["doc2_S_p0"],sa["doc2_Clean2"],sb[K],sb["doc2_S_pos"],sb["doc2_S_p0"],sb["doc2_Clean2"],sb[K]-sa[K],*np.quantile(d,[.05,.95])))
