# Decoder variance: in-sample fit vs nested; and a model-based decoder (optimise the EXPECTED score under the pooled
# probabilities instead of the realised labels), both evaluated on real labels with nesting.
import os, sys, numpy as np
REPO="/wanqing-develop/luowenjing/18"
for _p in ("code","code/next55","code/r55"): sys.path.insert(0, os.path.join(REPO,_p))
import common55 as c
from decoder55 import argmax_decode, nested_oof_decode, Decoder55
z=np.load(sys.argv[1],allow_pickle=True)
gold=z["gold"]; fold=z["fold"]; P=c.renorm(z["P"]); K="doc2_strict_total"
d=Decoder55(K).fit(P,gold); print("in-sample fit on all 3573: %.2f   (nested 51.45, argmax 51.28)"%c.score(d.predict(P),gold)[K])
# calibration check of pooled probs: reliability by bins for P(present) and P(P0)
for nm,idx in (("P(present)",None),("P(P0)",3)):
    for dd in range(6):
        p=(1-P[:,dd,0]) if idx is None else P[:,dd,3]; y=(gold[:,dd]>0) if idx is None else (gold[:,dd]==3)
        bins=[0,.05,.2,.4,.6,.8,.95,1.0001]; out=[]
        for a,b in zip(bins[:-1],bins[1:]):
            m=(p>=a)&(p<b)
            if m.sum()>20: out.append("%.2f/%.2f"%(p[m].mean(),y[m].mean()))
        print("  calib %s defect %d: "%(nm,dd)+"  ".join(out))
