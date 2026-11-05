import os, sys, json
from collections import Counter
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
from data import CLASSES, FULL
from optimize_v8_isolated_test import odds_scale

GRADE = {0: '无问题', 1: 'P2', 2: 'P1', 3: 'P0'}

def self_align_mean(q, target_mean):
    lo, hi = 0.35, 2.8
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if odds_scale(q, mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return odds_scale(q, 0.5 * (lo + hi))

tpl = json.load(open('results/v2_full/template_150.json'))
ordered_names = [e['image'] for e in tpl]

v2_probs = np.load('results/v2_super/test_150_all_models_probs.npz', allow_pickle=True)['data'].item()
v2_fam_w = {'pcl': 1.3, 'd3': 1.3, 'eupe': 1.1, 'base': 0.8}
v3_probs = np.load('results/v3_super/test_150_v3_models_probs.npz', allow_pickle=True)['data'].item()
v5_probs = np.load('results/v5_clean/test_150_v5_models_probs.npz', allow_pickle=True)['data'].item()

cln_p0, cln_p1, cln_p2, cln_w = 0.0, 0.0, 0.0, 0.0
for name, (p0_i, p1_i, p2_i, fam, is_iso) in v2_probs.items():
    wi = v2_fam_w.get(fam, 1.0) * (0.70 if is_iso else 1.0)
    cln_p0 += wi * p0_i
    cln_p1 += wi * p1_i
    cln_p2 += wi * p2_i
    cln_w += wi
for name, val in v3_probs.items():
    p0_i, p1_i, p2_i = val[0], val[1], val[2]
    wi = float(val[-1])
    cln_p0 += wi * p0_i
    cln_p1 += wi * p1_i
    cln_p2 += wi * p2_i
    cln_w += wi
cln_p0 /= cln_w
cln_p1 /= cln_w
cln_p2 /= cln_w

def avg_ckpt_dict(pdict, w_neg_corr=1.0):
    p0_acc, p1_acc, p2_acc, w_tot = 0.0, 0.0, 0.0, 0.0
    for k, val in pdict.items():
        p0_i, p1_i, p2_i = val[0], val[1], val[2]
        wi = float(val[-1]) if isinstance(val[-1], (int, float, np.floating)) else 1.0
        p0_acc += wi * p0_i
        p1_acc += wi * p1_i
        p2_acc += wi * p2_i
        w_tot += wi
    p0_m, p1_m, p2_m = p0_acc / w_tot, p1_acc / w_tot, p2_acc / w_tot
    if w_neg_corr != 1.0:
        def corr(q):
            q = np.clip(q, 1e-6, 1.0 - 1e-6)
            return (w_neg_corr * q) / (w_neg_corr * q + (1.0 - q))
        c0 = corr(p0_m)
        c01_full = corr(p0_m[:, [0, 1, 2, 4]] + p1_m)
        c012_full = corr(p0_m[:, [0, 1, 2, 4]] + p1_m + p2_m)
        p0_m = c0
        p1_m = np.maximum(0.0, c01_full - c0[:, [0, 1, 2, 4]])
        p2_m = np.maximum(0.0, c012_full - c01_full)
    return p0_m, p1_m, p2_m

s5_p0, s5_p1, s5_p2 = avg_ckpt_dict(v5_probs, w_neg_corr=1.0)
base_p0 = 0.65 * cln_p0 + 0.35 * s5_p0
base_p1 = 0.65 * cln_p1 + 0.35 * s5_p1
base_p2 = 0.65 * cln_p2 + 0.35 * s5_p2

v4_probs = np.load('results/v4_synth/test_150_v4_models_probs.npz', allow_pickle=True)['data'].item()
v6_probs = np.load('results/v6_synth/test_150_v6_models_probs.npz', allow_pickle=True)['data'].item()
v7_probs = np.load('results/v7_overgen_pro/test_150_v7_models_probs.npz', allow_pickle=True)['data'].item()
s4_p0, s4_p1, s4_p2 = avg_ckpt_dict(v4_probs, w_neg_corr=1.0)
s6_p0, s6_p1, s6_p2 = avg_ckpt_dict(v6_probs, w_neg_corr=1.35)
s7_p0, s7_p1, s7_p2 = avg_ckpt_dict(v7_probs, w_neg_corr=1.35)

def load_v8_w2_pair(prefix):
    p0_l, p1_l, p2_l = [], [], []
    for sp in (16, 17):
        f = f'results/v8_experts/{prefix}_s{sp}_probs.npz'
        if os.path.exists(f):
            d = np.load(f)
            if 't150_p0' in d.files:
                p0_l.append(d['t150_p0'])
                p1_l.append(d['t150_p1'])
                p2_l.append(d['t150_p2'])
    return np.mean(p0_l, axis=0), np.mean(p1_l, axis=0), np.mean(p2_l, axis=0)

v8A_p0, v8A_p1, v8A_p2 = load_v8_w2_pair('V8_W2_ExpA_overgen')
v8B_p0, v8B_p1, v8B_p2 = load_v8_w2_pair('V8_W2_ExpB_sharp')
v8C_p0, v8C_p1, v8C_p2 = load_v8_w2_pair('V8_W2_ExpC_blurtex')
v8D_p0, v8D_p1, v8D_p2 = load_v8_w2_pair('V8_W2_ExpD_bokeh')

# Joint P1-Protected + Higher V8 Single-Feature Dedicated Expert Routing (Backup Model, 53.25 / S_pos=0.658)
p0_150 = np.zeros((150, 6))
p1_150 = np.zeros((150, 4))
p2p_150 = np.zeros((150, 4))
p2g_150 = np.zeros((150, 4))

for k, ci in [(0, 0), (1, 1)]:
    p0_150[:, ci] = 0.45 * base_p0[:, ci] + 0.35 * v8C_p0[:, ci] + 0.12 * s6_p0[:, ci] + 0.08 * s4_p0[:, ci]
    p1_150[:, k]  = 0.35 * base_p1[:, k]  + 0.45 * v8C_p1[:, k]  + 0.12 * s6_p1[:, k]  + 0.08 * s7_p1[:, k]
    p2p_150[:, k] = 0.35 * base_p2[:, k]  + 0.45 * v8C_p2[:, k]  + 0.12 * s6_p2[:, k]  + 0.08 * s7_p2[:, k]
    p2g_150[:, k] = p2p_150[:, k].copy()

p0_150[:, 2] = 0.38 * base_p0[:, 2] + 0.36 * v8B_p0[:, 2] + 0.14 * s4_p0[:, 2] + 0.10 * s6_p0[:, 2] + 0.14 * v8B_p1[:, 2]
p1_150[:, 2]  = 0.32 * base_p1[:, 2] + 0.44 * v8B_p1[:, 2] + 0.14 * s4_p1[:, 2] + 0.10 * s6_p1[:, 2]
p2p_150[:, 2] = 0.32 * base_p2[:, 2] + 0.44 * v8B_p2[:, 2] + 0.14 * s4_p2[:, 2] + 0.10 * s6_p2[:, 2]
p2g_150[:, 2] = p2p_150[:, 2].copy()

p0_150[:, 3] = cln_p0[:, 3].copy()

p0_150[:, 4] = 0.42 * base_p0[:, 4] + 0.24 * s7_p0[:, 4] + 0.24 * v8A_p0[:, 4] + 0.10 * s6_p0[:, 4]
p2p_150[:, 3] = 0.26 * base_p2[:, 3] + 0.28 * s6_p2[:, 3] + 0.28 * v8A_p2[:, 3] + 0.18 * s7_p2[:, 3]
p1_150[:, 3]  = 0.28 * base_p1[:, 3] + 0.42 * v8A_p1[:, 3] + 0.30 * s7_p1[:, 3]
p2g_150[:, 3] = 0.28 * base_p2[:, 3] + 0.42 * v8A_p2[:, 3] + 0.30 * s7_p2[:, 3]

p0_150[:, 5] = 0.35 * base_p0[:, 5] + 0.45 * v8D_p0[:, 5] + 0.20 * s6_p0[:, 5]

# Slightly boosted target mean (+5% relative recall boost on P1/P0 for hard tail features)
boost_p0 = [1.02, 1.06, 1.06, 1.00, 1.05, 1.06]
boost_p1 = [1.03, 1.06, 1.06, 1.08]
boost_p2 = [1.02, 1.04, 1.04, 1.05]
for ci in range(6):
    p0_150[:, ci] = self_align_mean(p0_150[:, ci], cln_p0[:, ci].mean() * boost_p0[ci])
for k in range(4):
    p1_150[:, k] = self_align_mean(p1_150[:, k], cln_p1[:, k].mean() * boost_p1[k])
    p2p_150[:, k] = self_align_mean(p2p_150[:, k], cln_p2[:, k].mean() * boost_p2[k])
    p2g_150[:, k] = self_align_mean(p2g_150[:, k], cln_p2[:, k].mean())

b5243 = {e['image']: e['label'] for e in json.load(open('results/v3_super/pred_nriqa_dataset_test.json'))}
main_v8 = {e['image']: e['label'] for e in json.load(open('pred_nriqa_dataset_test.json'))}

# Joint P1-Protected + P0-Recall Arbitrator Thresholds (task-1894: S_pos=0.658, 过度生成_P1=0.452)
T0_150   = [0.39, 0.39, 0.34, 0.40, 0.33, 0.33]
T2_150   = [0.26, 0.24, 0.22, 0.21]
T1p_150  = [0.24, 0.25, 0.24, 0.21]
Tcum_150 = [0.43, 0.43, 0.38, 0.36]
T1_150   = [0.23, 0.24, 0.22, 0.195]
R12_150  = [0.78, 0.82, 0.75, 0.68]
W0_150   = [0.20, 0.25, 0.20, 0.22]

pred_mat = np.zeros((150, 6), dtype=int)
for ci, c in enumerate(CLASSES):
    if c not in FULL:
        pred_mat[:, ci] = np.where(p0_150[:, ci] > T0_150[ci], 3, 0)
    else:
        k = FULL.index(c)
        q0 = p0_150[:, ci]
        q1 = p1_150[:, k]
        q2p = p2p_150[:, k]
        q2g = p2g_150[:, k]
        cum = q0 + q1 + q2p
        c0 = q0 > T0_150[ci]
        if k == 3:
            # Stronger P1 protection for 过度生成_P1 (lifts Overgen_P1 F1 to 0.452 on isolated 1000-test)
            c0 = c0 & ((q0 > 0.43) | (q0 >= 0.80 * q1))
        present = (q2p > T2_150[k]) | (q1 > T1p_150[k]) | (cum > Tcum_150[k])
        s1 = q1 + W0_150[k] * q0
        is_p1 = (s1 > T1_150[k]) & (s1 >= R12_150[k] * q2g)
        pr = np.where(present, 1, 0)
        pr = np.where(present & is_p1, 2, pr)
        pr = np.where(c0, 3, pr)
        pred_mat[:, ci] = pr

out_entries = []
for i, name in enumerate(ordered_names):
    old_lbl = b5243[name]
    if all(v == '无问题' for v in old_lbl.values()):
        lbl = old_lbl.copy()
    else:
        lbl = {k: GRADE[int(pred_mat[i, CLASSES.index(k)])] for k in ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']}
        if all(v == '无问题' for v in lbl.values()):
            lbl = old_lbl.copy()
    out_entries.append({'image': name, 'label': lbl})

out1 = 'pred_nriqa_dataset_test_v8_backup_p1p0boost.json'
out2 = 'results/v8_experts/pred_nriqa_dataset_test_v8_backup_p1p0boost.json'
for p in [out1, out2]:
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out_entries, f, ensure_ascii=False, indent=2)

diffs_base = []
diffs_main = []
trans_base = Counter()
for e in out_entries:
    im = e['image']
    for c in CLASSES:
        ov = b5243[im][c]
        mv = main_v8[im][c]
        nv = e['label'][c]
        if ov != nv:
            diffs_base.append((im, c, ov, nv))
            trans_base[(c, f'{ov}->{nv}')] += 1
        if mv != nv:
            diffs_main.append((im, c, mv, nv))

print(f"Saved {out1} and {out2}!")
print(f"Diff vs 52.43 baseline: {len(diffs_base)} cells across {len(set(x[0] for x in diffs_base))}/150 images")
print(f"Diff vs Main V8 (53.35): {len(diffs_main)} cells across {len(set(x[0] for x in diffs_main))}/150 images")
for (c, tr), cnt in sorted(trans_base.items()):
    print(f"  {c:8s}: {tr:12s} x {cnt}")
print("\nDifferences between Backup V8 and Main V8:")
for im, c, mv, nv in diffs_main:
    print(f"  {im:24s} | {c:6s}: MainV8={mv} -> BackupV8={nv} (52.43={b5243[im][c]})")
