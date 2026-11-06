"""Per-channel F1 check (过度生成 P0/P1/P2, 虚化突兀 P0, plus all others) on pooled s16~s20 blind tests."""
import sys, numpy as np
sys.path.insert(0,'code/finetune'); sys.path.insert(0,'code/decision_eval')
from train_phase2_corn_tower import decode_roll_down, CLASSES, FULL_IDX
from phase3_split_worker import TH_ANY, TH_ROLL, TH_P0
S=['split_s16','split_s17','split_s18','split_s19','split_s20']
D={s:np.load(f'results/phase3/{s}.npz') for s in S}
pr=lambda n,s: decode_roll_down(*[D[s][f'{n}_{j}'] for j in range(3)],TH_ANY,TH_ROLL,TH_P0)
G=np.concatenate([D[s]['gold'] for s in S]); P={n:np.concatenate([pr(n,s) for s in S]) for n in('p2','p3','fused')}
def f1(g,p,gr):
    tp=((g==gr)&(p==gr)).sum(); fp=((g!=gr)&(p==gr)).sum(); fn=((g==gr)&(p!=gr)).sum()
    return 2*tp/(2*tp+fp+fn) if tp+fp+fn else 0.0
print(f"{'channel/grade':16s} {'n_pos':>6s} {'p2':>7s} {'p3':>7s} {'fused':>7s}  (F1)")
for ci,c in enumerate(CLASSES):
    for gr in ([1,2,3] if ci in FULL_IDX else [3]):
        print(f"{c}_{['','P2','P1','P0'][gr]:3s}".ljust(16), f"{(G[:,ci]==gr).sum():6d}", *[f"{f1(G[:,ci],P[n][:,ci],gr):7.3f}" for n in('p2','p3','fused')])
