import sys, numpy as np
sys.path.insert(0,'code/finetune'); sys.path.insert(0,'code/decision_eval')
from train_phase2_corn_tower import decode_roll_down, compute_competition_metrics
from phase3_split_worker import TH_ANY, TH_ROLL, TH_P0
S=['split_s16','split_s17','split_s18','split_s19','split_s20']
D={s:np.load(f'results/phase3/{s}.npz') for s in S}
def pr(n,s): return decode_roll_down(*[D[s][f'{n}_{j}'] for j in range(3)],TH_ANY,TH_ROLL,TH_P0)
G=np.concatenate([D[s]['gold'] for s in S]); P={n:np.concatenate([pr(n,s) for s in S]) for n in('p2','p3','fused')}
f=lambda g,p:(lambda m:25*m[1]+20*m[2])(compute_competition_metrics(g,p))
# cluster bootstrap by split-test image identity is not available here; resample images, and also per-split paired diffs
for a in('fused','p3'):
    per=[f(D[s]['gold'],pr(a,s))-f(D[s]['gold'],pr('p2',s)) for s in S]
    rng=np.random.default_rng(0);N=len(G);d=[]
    for _ in range(1000):
        i=rng.integers(0,N,N);d.append(f(G[i],P[a][i])-f(G[i],P['p2'][i]))
    d=np.array(d)
    print(f'{a}-p2 on pos+p0 only: per-split diffs={np.round(per,2)}  mean={np.mean(per):+.3f}  boot95%CI=[{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}] P(<=0)={(d<=0).mean():.3f}')
