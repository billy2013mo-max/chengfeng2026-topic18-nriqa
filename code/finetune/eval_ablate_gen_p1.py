"""Evaluate ablation variants on held-out test rows with ONE fixed decoder (HAND) - no per-variant tuning."""
import sys, numpy as np
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/finetune')
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/decision_eval')
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from data import CLASSES

R = 'results/v3_ablate'
names = ['A_base', 'E_synth', 'F_v5']
want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0']
print(f"{'variant':10s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} | " + ' '.join(f'{w[:6]+w[-3:]:>9s}' for w in want))
agg = {}
for n in names:
    rows = []
    for sp in (16, 17):
        try:
            d = np.load(f'{R}/{n}_s{sp}_probs.npz')
        except Exception:
            continue
        pred = decode_pc(d['te_p0'], d['te_p1'], d['te_p2'], HAND)
        s = cm.score(d['gte'], pred, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
        rows.append((s, f1))
        print(f"{n+'_s'+str(sp):10s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} | " + ' '.join(f'{f1[w]:9.3f}' for w in want))
    if rows:
        m = lambda k: np.mean([r[0][k] for r in rows])
        mf = [np.mean([r[1][w] for r in rows]) for w in want]
        print(f"{n+' MEAN':10s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} | " + ' '.join(f'{x:9.3f}' for x in mf))
        print()
