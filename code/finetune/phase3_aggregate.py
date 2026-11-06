"""Phase 3 gate (doc section 5): paired bootstrap on the pooled s16~s20 blind-test set.
Compares  fused (Phase2 + 22% aligned Phase3)  vs  p2 (Phase-2 tower)  and  p3 alone vs p2.
Promotion rule from the doc: gain must be positive with paired-bootstrap p < 0.05.
"""
import sys, numpy as np
sys.path.insert(0, 'code/finetune'); sys.path.insert(0, 'code/decision_eval')
from train_phase2_corn_tower import decode_roll_down, compute_competition_metrics
from phase3_split_worker import TH_ANY, TH_ROLL, TH_P0

SPLITS = ['split_s16', 'split_s17', 'split_s18', 'split_s19', 'split_s20']
D = {s: np.load(f'results/phase3/{s}.npz') for s in SPLITS}
pred = {n: {s: decode_roll_down(*[D[s][f'{n}_{j}'] for j in range(3)], TH_ANY, TH_ROLL, TH_P0) for s in SPLITS} for n in ('p2', 'p3', 'fused')}

print('=== per-split blind-test total (official 65-pt formula) ===')
print(f"{'split':10s} {'p2':>7s} {'p3':>7s} {'fused':>7s}  clean(p2/p3/fused)")
tot = {n: [] for n in pred}
for s in SPLITS:
    g = D[s]['gold']; row = []
    for n in pred:
        m = compute_competition_metrics(g, pred[n][s]); tot[n].append(m); row.append(m)
    print(f"{s:10s} {row[0][0]:7.2f} {row[1][0]:7.2f} {row[2][0]:7.2f}  {row[0][3]*20:.1f}/{row[1][3]*20:.1f}/{row[2][3]*20:.1f}")
for n in pred:
    a = np.array([[m[0], m[1] * 25, m[2] * 20, m[3] * 20] for m in tot[n]]).mean(0)
    print(f'MEAN {n:6s}: total={a[0]:.2f} pos={a[1]:.2f} p0={a[2]:.2f} clean={a[3]:.2f}')

G = np.concatenate([D[s]['gold'] for s in SPLITS])
P = {n: np.concatenate([pred[n][s] for s in SPLITS]) for n in pred}
N = len(G); rng = np.random.default_rng(0); B = 1000
print(f'\npooled blind samples N={N}')
for a, b in [('fused', 'p2'), ('p3', 'p2')]:
    d = []
    for _ in range(B):
        i = rng.integers(0, N, N)
        d.append(compute_competition_metrics(G[i], P[a][i])[0] - compute_competition_metrics(G[i], P[b][i])[0])
    d = np.array(d)
    obs = compute_competition_metrics(G, P[a])[0] - compute_competition_metrics(G, P[b])[0]
    p = (d <= 0).mean()
    print(f'{a} - {b}: observed={obs:+.3f}  boot mean={d.mean():+.3f}  95%CI=[{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]  P(diff<=0)={p:.3f}  -> {"PASS" if (obs>0 and p<0.05) else "NOT SIGNIFICANT"}')
