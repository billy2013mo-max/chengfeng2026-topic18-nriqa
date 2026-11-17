import numpy as np, sys, os, warnings; warnings.filterwarnings("ignore")
from harness import *
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
R="/wanqing-develop/luowenjing/18/"; idx={n:i for i,n in enumerate(names)}
def load_dir(d):
    arr=np.zeros((len(names),6,4)); ok=np.zeros(len(names),bool)
    for k in range(8):
        f=f"{R}{d}/fold_{k}_probs.npz"
        if not os.path.exists(f): continue
        z=np.load(f,allow_pickle=True)
        for n,q in zip(z["va_names"],z["va_probs_6x4"]): arr[idx[str(n)]]=q; ok[idx[str(n)]]=True
    return arr,ok
def stats(A,m,d):
    y=g[m,d]; q=A[m,d]; s=(q*np.arange(4)).sum(1); r=[AP(y==lv,q[:,lv]) for lv in (3,2,1)]
    for a,b in [(1,2),(2,3)]:
        k=(y==a)|(y==b); r.append(AUC(y[k]==b,s[k]))
    return np.round(r,3).tolist()
if __name__=="__main__":
  dirs=sys.argv[1:]; A={d:load_dir(d) for d in dirs}
  m=np.all([A[d][1] for d in dirs],0); print("common rows",m.sum(),"folds",sorted(set(fold[m])))
  for d in dirs:
    print(f"{d.split('/')[-2][:34]:34s}", "过锐",stats(A[d][0],m,2)," 过度生成",stats(A[d][0],m,4))
  print(f"{'POOL v7':34s}", "过锐",stats(D['Q'],m,2)," 过度生成",stats(D['Q'],m,4))
