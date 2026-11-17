# Direct look at label repeatability: pairs of labelled frames that are near-identical in pixels.
import os, sys, numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
REPO="/wanqing-develop/luowenjing/18"
z=np.load(sys.argv[1],allow_pickle=True); names=[str(x) for x in z["names"]]; gold=z["gold"]; grp=z["grp"]; P=z["P"]
def find(n):
    for d in ("data/images","data/images_supp"):
        p=os.path.join(REPO,d,n)
        if os.path.exists(p): return p
def thumb(n):
    im=Image.open(find(n)).convert("L"); w,h=im.size
    a=np.asarray(im.resize((48,48),Image.BILINEAR),dtype=np.float32).ravel()
    a=(a-a.mean())/(a.std()+1e-6)
    return a/np.linalg.norm(a), w, h
if __name__=="__main__":
    with ProcessPoolExecutor(48) as ex: out=list(ex.map(thumb,names,chunksize=16))
    T=np.stack([o[0] for o in out]); W=np.array([o[1] for o in out]); H=np.array([o[2] for o in out])
    C=T@T.T; np.fill_diagonal(C,0); iu=np.triu_indices(len(names),1)
    np.savez(sys.argv[2],T=T,W=W,H=H)
    D=["模糊","纹理损失","过锐","花屏","过度生成","虚化突兀"]
    for lo,hi in ((0.995,1.01),(0.985,0.995),(0.97,0.985),(0.95,0.97)):
        m=(C[iu]>=lo)&(C[iu]<hi)&(W[iu[0]]==W[iu[1]])&(H[iu[0]]==H[iu[1]])
        a,b=iu[0][m],iu[1][m]
        if len(a)==0: print("corr [%.3f,%.3f): no pairs"%(lo,hi)); continue
        same_v=(grp[a]==grp[b])
        ex=(gold[a]==gold[b]); pres=((gold[a]>0)==(gold[b]>0))
        both=((gold[a]>0)&(gold[b]>0)); 
        print("corr [%.3f,%.3f): %d pairs (%d same video) | exact-grade agreement per defect: %s | presence agreement: %s | all six equal: %.2f"%(lo,hi,len(a),same_v.sum()," ".join("%.2f"%x for x in ex.mean(0))," ".join("%.2f"%x for x in pres.mean(0)),ex.all(1).mean()))
        # grade agreement given both positive, and model's own agreement (argmax) on the same pairs
        am=P.argmax(-1)
        g=[ (gold[a,d][both[:,d]]==gold[b,d][both[:,d]]).mean() if both[:,d].sum()>0 else np.nan for d in range(6)]
        print("      grade agreement when both frames carry the defect: "+" ".join("%s %.2f (n=%d)"%(D[d],g[d],both[:,d].sum()) for d in (0,1,2,4)))
        print("      ensemble argmax agrees between the two frames: "+" ".join("%.2f"%x for x in (am[a]==am[b]).mean(0)))
