"""Compare Qwen3-VL-8B LoRA classifier vs V9_SpecA_OverGen_Giant on leak-free s16/s17 test (1000 imgs each)."""
import os, sys, json
import numpy as np
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune'); sys.path.insert(0, 'code/decision_eval')
import comp_metrics as cm
from train_v2 import optimise_scaled, decode_scaled_argmax

K, C = 3, 4  # FULL index / CLASSES index of 过度生成


def ap(y, s):
    o = np.argsort(-s); y = y[o].astype(float)
    if y.sum() == 0: return float('nan')
    return float(((np.cumsum(y) / np.arange(1, len(y) + 1)) * y).sum() / y.sum())


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def align(ref_names, d, pre):
    idx = {str(n): i for i, n in enumerate(d[f'{pre}_names'])}
    o = np.array([idx[str(n)] for n in ref_names])
    return d[f'{pre}_p0'][o], d[f'{pre}_p1'][o], d[f'{pre}_p2'][o], (d['gva'] if pre == 'va' else d['gte'])[o]


def evaluate(va, te):
    _, th = optimise_scaled(va[3], va[0], va[1], va[2], 0.02)
    st = cm.score(te[3], decode_scaled_argmax(te[0], te[1], te[2], *th), 0.02)
    f1 = {f'{x[1]}': x[4] for x in st['per'] if x[0] == '过度生成'}
    g = te[3][:, C]
    aps = {'P2': ap(g == 1, te[2][:, K]), 'P1': ap(g == 2, te[1][:, K]), 'P0': ap(g == 3, te[0][:, C])}
    return st, f1, aps


rows = []
summ = {}
for sp in (16, 17):
    v = load(f'results/vlm_cls/qwen3vl8b_lora_s{sp}_probs.npz')
    b = load(f'results/v9_specialists/V9_SpecA_OverGen_Giant_s{sp}_probs.npz')
    va_n, te_n = v['va_names'], v['te_names']
    V = (align(va_n, v, 'va'), align(te_n, v, 'te'))
    B = (align(va_n, b, 'va'), align(te_n, b, 'te'))
    E = tuple(tuple(0.5 * (x + y) if i < 3 else x for i, (x, y) in enumerate(zip(V[j], B[j]))) for j in range(2))
    for name, M in [('V9_SpecA (baseline)', B), ('Qwen3-VL-8B LoRA', V), ('avg(V9_SpecA, VLM)', E)]:
        st, f1, aps = evaluate(*M)
        summ.setdefault(name, []).append((st, f1, aps))

print(f'{"model":22s} | {"OG_P2 F1":>8s} {"OG_P1 F1":>8s} {"OG_P0 F1":>8s} | {"AP_P2":>6s} {"AP_P1":>6s} {"AP_P0":>6s} | {"total(doc1)":>11s} {"S_pos":>6s} {"S_p0":>6s}')
for name, L in summ.items():
    f = lambda key, g: np.mean([x[1 if key == 'f1' else 2][g] for x in L])
    tot = np.mean([x[0]['total'] for x in L]); sp_ = np.mean([x[0]['S_pos'] for x in L]); s0 = np.mean([x[0]['S_p0'] for x in L])
    print(f'{name:22s} | {f("f1","P2"):8.3f} {f("f1","P1"):8.3f} {f("f1","P0"):8.3f} | {f("ap","P2"):6.3f} {f("ap","P1"):6.3f} {f("ap","P0"):6.3f} | {tot:11.2f} {sp_:6.3f} {s0:6.3f}')
print('\n(mean over s16 & s17 test, 1000 imgs each, source-grouped => no leakage; thresholds fit on each split\'s val_500)')
for name, L in summ.items():
    print(name, 'per-split OG F1:', [{k: round(v, 3) for k, v in x[1].items()} for x in L])
