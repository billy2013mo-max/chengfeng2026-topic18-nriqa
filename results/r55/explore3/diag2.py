import os, sys, re, csv, numpy as np, collections
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import argmax_decode, nested_oof_decode, Decoder55
z=np.load(sys.argv[1],allow_pickle=True)
names=[str(x) for x in z["names"]]; gold=z["gold"]; fold=z["fold"]; grp=z["grp"]; P=z["P"]; nd=z["nested"]
DIMS=["模糊","纹理损失","过锐","花屏","过度生成","虚化突兀"]
sp={r['name']:r['split'] for r in csv.DictReader(open(REPO+'/data/manifest_combined.csv',encoding='utf-8'))}
B=np.array([sp[n] for n in names]); o=B=='orig'; s=B=='supp'
K="doc2_strict_total"
def show(tag,pred,g):
    sc=c.score(pred,g); print("%-34s n=%4d total %.2f pos %.2f p0 %.2f clean %.2f | "%(tag,len(g),sc[K],sc["doc2_S_pos"],sc["doc2_S_p0"],sc["doc2_Clean2"])+" ".join("%.2f"%r["F1"] for r in sc["per_14"]))
print("prevalence (rows: grade none/P2/P1/P0) orig | supp")
for d in range(6):
    print(DIMS[d], " orig:", np.round(np.bincount(gold[o,d],minlength=4)/o.sum(),3), " supp:", np.round(np.bincount(gold[s,d],minlength=4)/s.sum(),3))
print("clean frac orig %.3f supp %.3f"%((gold[o]==0).all(1).mean(), (gold[s]==0).all(1).mean()))
print("labels: 模糊P2 P1 P0 | 纹理 | 过锐 | 花屏 | 过度生成 | 虚化")
show("all, pooled nested decoder", nd, gold)
show("orig subset of pooled nested", nd[o], gold[o])
show("supp subset of pooled nested", nd[s], gold[s])
show("orig subset, argmax", argmax_decode(P)[o], gold[o])
show("supp subset, argmax", argmax_decode(P)[s], gold[s])
# decoder fitted on orig only (nested by fold), evaluated on orig
ndo=nested_oof_decode(P[o],gold[o],fold[o],K); show("orig, nested decoder fit on orig", ndo, gold[o])
nds=nested_oof_decode(P[s],gold[s],fold[s],K); show("supp, nested decoder fit on supp", nds, gold[s])
np.savez(sys.argv[2], orig=o, ndo=ndo)
# bootstrap 140-image and 654-image draws from orig with pooled decoder vs orig decoder
rs=np.random.RandomState(0)
io=np.where(o)[0]
for nn in (654,):
    a=[];b=[]
    for _ in range(300):
        j=rs.choice(len(io),nn,replace=False)
        a.append(c.score(nd[io[j]],gold[io[j]])[K]); b.append(c.score(ndo[j],gold[io[j]])[K])
    a=np.array(a);b=np.array(b)
    print("orig draws n=%d: pooled-decoder %.2f±%.2f  orig-decoder %.2f±%.2f  diff %.2f [%.2f,%.2f]"%(nn,a.mean(),a.std(),b.mean(),b.std(),(b-a).mean(),np.quantile(b-a,.05),np.quantile(b-a,.95)))
