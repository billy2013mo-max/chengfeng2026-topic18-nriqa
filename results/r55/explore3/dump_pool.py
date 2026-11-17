import os, sys, numpy as np
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import argmax_decode, nested_oof_decode
S1 = {"radio": "qy_cradio_v4_h", "pe": "qy_pe_core_large", "sig": "qy_siglip2_so400m_naflex","v16": "v16_full8"}
FAM = {"radio":0,"pe":0,"sig":0,"v16":1,"p1":2,"p3":2}
names, gold, fold, grp = c.load_split()
bank = c.build_bank(only=list(S1.values()))
B = dict(zip(S1, c.stack_oof(bank, list(S1.values()), names)))
for p in (1,3):
    z=np.load(os.path.join(REPO,f"results/phoenix/pillar{p}_stage2-rl2r_oof.npz"),allow_pickle=True)
    m={str(n):q for n,q in zip(z["names"],z["probs_6x4"].astype(np.float64))}
    B[f"p{p}"]=c.renorm(np.stack([m[n] for n in names]))
keys=["radio","pe","sig","v16","p1","p3"]
cnt=np.bincount([FAM[k] for k in keys],minlength=3)
w=[1.0/cnt[FAM[k]] for k in keys]
P=c.renorm(c.mean_pool([B[k] for k in keys],w))
print("defects order:", getattr(c,'DEFECTS',None))
print(P.shape, gold.shape, np.unique(fold), len(np.unique(grp)))
sc=c.score(argmax_decode(P),gold); print({k:(round(float(v),3) if np.isscalar(v) else v) for k,v in sc.items() if 'doc2' in k or 'S_' in k or 's_' in k})
nd=nested_oof_decode(P,gold,fold,"doc2_strict_total")
sc2=c.score(nd,gold); print("nested", {k:(round(float(v),3) if np.isscalar(v) else v) for k,v in sc2.items() if 'doc2' in k})
np.savez(sys.argv[1], names=np.array(names), gold=gold, fold=fold, grp=np.array(grp), P=P, nested=nd, **{f"m_{k}":B[k] for k in keys})
