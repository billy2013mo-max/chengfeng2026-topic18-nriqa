import os, sys, re, numpy as np, collections
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import argmax_decode
z=np.load(sys.argv[1],allow_pickle=True)
names=[str(x) for x in z["names"]]; gold=z["gold"]; fold=z["fold"]; grp=z["grp"]; P=z["P"]; nd=z["nested"]
DIMS=["模糊","纹理损失","过锐","花屏","过度生成","虚化突兀"]
sc=c.score(nd,gold)
print("nested total %.2f pos %.2f p0 %.2f clean %.2f"%(sc["doc2_strict_total"],sc["doc2_S_pos"],sc["doc2_S_p0"],sc["doc2_Clean2"]))
for r in sc["per_14"]:
    print("  %-12s P %.3f R %.3f F1 %.3f tp %4d fp %4d fn %4d  npos %4d"%(r["channel"],r["P"],r["R"],r["F1"],r["tp"],r["fp"],r["fn"],r["tp"]+r["fn"]))
# source typing by filename
def typ(n):
    b=os.path.splitext(n)[0]
    m=re.search(r'_((?:v\d+|k\d+)[A-Za-z0-9]*?(?:V\d+|High\w*|Hi\w*))(?:[._]|$)',b)
    if m: return "tag:"+m.group(1)
    if re.search(r'[一-鿿]',b): return "cn"
    return "other"
T=[typ(n) for n in names]
cnt=collections.Counter(T)
print(cnt.most_common(40))
# print samples of 'other' and 'cn'
import random
random.seed(0)
for t in ["other","cn"]:
    xs=[n for n,tt in zip(names,T) if tt==t]
    print(t,len(xs),random.sample(xs,min(25,len(xs))))
