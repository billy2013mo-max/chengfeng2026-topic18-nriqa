import sys, os, pickle
import numpy as np

sys.path.extend(['code', 'code/finetune', 'code/decision_eval'])
from utils import competition_score
from cv_decoder_calib import decode_pc

with open('results/v16_fulldata/fast_stack5553_cache.pkl', 'rb') as f:
    C = pickle.load(f)

prm = C['prm']
stack5457 = C['stack5457']
v9_oof = C['v9_oof']
v16_oof = C['v16_oof']
models_arr = C['models_arr']
full_oof_archives = C['full_oof_archives']

V17_DIR = 'results/v17_end2end_s16_s17'
specs = [
    'V17_S2R_SpecA_OverGen_TexLoss',
    'V17_S2R_SpecB_Sharp_TexLoss',
    'V17_S2R_SpecC_Blur_Bokeh_Screen',
    'V17_S2R_SpecD_All6_Giant',
]

v17 = {16: {}, 17: {}}
for sp in (16, 17):
    for s in specs:
        v17[sp][s] = np.load(f'{V17_DIR}/{s}_s{sp}_probs.npz')

def align_odds_1d(src, ref):
    s = np.clip(np.asarray(src, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    r = np.clip(np.asarray(ref, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    ms, mr = float(s.mean()), float(r.mean())
    if ms <= 1e-6 or mr <= 1e-6:
        return s
    gamma = np.clip((mr / (1.0 - mr)) / (ms / (1.0 - ms)), 0.35, 2.85)
    return (gamma * s) / (gamma * s + (1.0 - s))

def decode_with_rescue(p0, p1, p2, sh, cur_prm=prm, use_rescue=False, w1=0.75, w2=0.35, thr=0.40):
    pr = decode_pc(p0, p1, p2, cur_prm)
    pr[sh] = 0
    if use_rescue:
        untrig = (pr[:, 4] == 0) & (~sh)
        m = untrig & ((p0[:, 4] + w1 * p1[:, 3] + w2 * p2[:, 3]) > thr)
        pr[m, 4] = 3
    return pr

def eval_all(va16, te16, va17, te17, use_rescue=False, w1=0.75, w2=0.35, thr=0.40):
    pv16 = decode_with_rescue(*va16, use_rescue=use_rescue, w1=w1, w2=w2, thr=thr)
    pt16 = decode_with_rescue(*te16, use_rescue=use_rescue, w1=w1, w2=w2, thr=thr)
    pv17 = decode_with_rescue(*va17, use_rescue=use_rescue, w1=w1, w2=w2, thr=thr)
    pt17 = decode_with_rescue(*te17, use_rescue=use_rescue, w1=w1, w2=w2, thr=thr)
    
    sc_v16 = competition_score(pv16, stack5457[16]['gva'])
    sc_t16 = competition_score(pt16, stack5457[16]['gte'])
    sc_v17 = competition_score(pv17, stack5457[17]['gva'])
    sc_t17 = competition_score(pt17, stack5457[17]['gte'])
    
    return {
        'va16': sc_v16, 'te16': sc_t16,
        'va17': sc_v17, 'te17': sc_t17,
        'va_mean': 0.5 * (sc_v16['total'] + sc_v17['total']),
        'te_mean': 0.5 * (sc_t16['total'] + sc_t17['total']),
    }

# 1. Base Stack5457
va16 = [x.copy() for x in stack5457[16]['va'][:3]] + [stack5457[16]['va'][3]]
te16 = [x.copy() for x in stack5457[16]['te'][:3]] + [stack5457[16]['te'][3]]
va17 = [x.copy() for x in stack5457[17]['va'][:3]] + [stack5457[17]['va'][3]]
te17 = [x.copy() for x in stack5457[17]['te'][:3]] + [stack5457[17]['te'][3]]

r0 = eval_all(va16, te16, va17, te17)
print("=" * 80)
print("1. Stack5457 Base Baseline:")
print(f"   VA: {r0['va_mean']:.3f} (s16={r0['va16']['total']:.3f}, s17={r0['va17']['total']:.3f})")
print(f"   TE: {r0['te_mean']:.3f} (s16={r0['te16']['total']:.3f}, s17={r0['te17']['total']:.3f})")
print(f"   TE s16: S_pos={r0['te16']['S_pos']*25:.2f}, S_p0={r0['te16']['S_p0']*20:.2f}, S_clean={r0['te16']['S_clean']*20:.2f}")
print(f"   TE s17: S_pos={r0['te17']['S_pos']*25:.2f}, S_p0={r0['te17']['S_p0']*20:.2f}, S_clean={r0['te17']['S_clean']*20:.2f}")

# Now, execute the 24-step benchmark to see the multi-model stack state
va16_folds = np.array([v9_oof[nm][3] for nm in stack5457[16]['va_n']])
te16_folds = np.array([v9_oof[nm][3] for nm in stack5457[16]['te_n']])
va17_folds = np.array([v9_oof[nm][3] for nm in stack5457[17]['va_n']])
te17_folds = np.array([v9_oof[nm][3] for nm in stack5457[17]['te_n']])

def apply_mod(va16, te16, va17, te17, g_idx, c, w, get_src_fn, fold_list=None):
    n_va16 = [x.copy() for x in va16[:3]] + [va16[3]]
    n_te16 = [x.copy() for x in te16[:3]] + [te16[3]]
    n_va17 = [x.copy() for x in va17[:3]] + [va17[3]]
    n_te17 = [x.copy() for x in te17[:3]] + [te17[3]]
    for sp, prf, target, f_arr in [
        (16, 'va', n_va16, va16_folds),
        (16, 'te', n_te16, te16_folds),
        (17, 'va', n_va17, va17_folds),
        (17, 'te', n_te17, te17_folds),
    ]:
        src = align_odds_1d(get_src_fn(sp, prf), target[g_idx][:, c])
        if fold_list is None:
            target[g_idx][:, c] = (1 - w) * target[g_idx][:, c] + w * src
        else:
            m = np.isin(f_arr, fold_list)
            target[g_idx][m, c] = (1 - w) * target[g_idx][m, c] + w * src[m]
    return n_va16, n_te16, n_va17, n_te17

steps_all = [
    ("01: ALL p0[0] + 0.08 _archive_vlm_corn", 0, 0, 0.08, lambda sp, prf: models_arr[sp]['_archive_2969b0f/v13_vlm_real/vlm_corn'][f'{prf}_p0'][:, 0], None),
    ("02: ALL p1[3] + 0.14 _archive_vlm_corn", 1, 3, 0.14, lambda sp, prf: models_arr[sp]['_archive_2969b0f/v13_vlm_real/vlm_corn'][f'{prf}_p1'][:, 3], None),
    ("03: ALL p2[0] + 0.08 vlm_corn",          2, 0, 0.08, lambda sp, prf: models_arr[sp]['v13_vlm_real/vlm_corn'][f'{prf}_p2'][:, 0], None),
    ("04: F01 p0[2] + 0.35 V15d_ALL",          0, 2, 0.35, lambda sp, prf: np.stack([full_oof_archives['V15d_ALL'][nm][0][2] for nm in stack5457[sp][f'{prf}_n']], 0), [0, 1]),
    ("05: F01 p2[0] + 0.35 V14c_SF1",          2, 0, 0.35, lambda sp, prf: np.stack([full_oof_archives['V14c_SF1'][nm][2][0] for nm in stack5457[sp][f'{prf}_n']], 0), [0, 1]),
    ("06: F23 p1[3] + 0.35 V16_SpecB",         1, 3, 0.35, lambda sp, prf: models_arr[sp]['v16_specialists/V16_SpecB_SharpTex_Microscope'][f'{prf}_p1'][:, 3], [2, 3]),
    ("07: F01 p0[0] + 0.25 pcladd_seed2",      0, 0, 0.25, lambda sp, prf: models_arr[sp]['backbone_exp/pcladd_seed2'][f'{prf}_p0'][:, 0], [0, 1]),
    ("08: F01 p2[2] + 0.25 v15c_bokeh",        2, 2, 0.25, lambda sp, prf: models_arr[sp]['v15_microscope_synth/v15c_bokeh'][f'{prf}_p2'][:, 2], [0, 1]),
    ("09: F01 p0[2] + 0.25 v12_abmil",         0, 2, 0.25, lambda sp, prf: models_arr[sp]['v12_abmil/v12_abmil'][f'{prf}_p0'][:, 2], [0, 1]),
    ("10: F45 p0[4] + 0.25 v12_phaseB_fixed",  0, 4, 0.25, lambda sp, prf: models_arr[sp]['v12_phaseB_fixed/v12_abmil'][f'{prf}_p0'][:, 4], [4, 5]),
    ("11: F45 p0[2] + 0.35 V15a_OG",           0, 2, 0.35, lambda sp, prf: np.stack([full_oof_archives['V15a_OG'][nm][0][2] for nm in stack5457[sp][f'{prf}_n']], 0), [4, 5]),
    ("12: F45 p1[2] + 0.35 V15a_OG",           1, 2, 0.35, lambda sp, prf: np.stack([full_oof_archives['V15a_OG'][nm][1][2] for nm in stack5457[sp][f'{prf}_n']], 0), [4, 5]),
    ("13: F01 p0[5] + 0.30 V15b_ST",           0, 5, 0.30, lambda sp, prf: np.stack([full_oof_archives['V15b_ST'][nm][0][5] for nm in stack5457[sp][f'{prf}_n']], 0), [0, 1]),
    ("14: F67 p0[0] + 0.32 d3add_seed1",       0, 0, 0.32, lambda sp, prf: models_arr[sp]['backbone_exp/d3add_seed1'][f'{prf}_p0'][:, 0], [6, 7]),
    ("15: F67 p2[3] + 0.32 V9_SpecD",          2, 3, 0.32, lambda sp, prf: models_arr[sp]['v9_specialists/V9_SpecD_BokehOG_HiRes'][f'{prf}_p2'][:, 3], [6, 7]),
    ("16: F23 p1[2] + 0.22 V14a_CORN",         1, 2, 0.22, lambda sp, prf: np.stack([full_oof_archives['V14a_CORN'][nm][1][2] for nm in stack5457[sp][f'{prf}_n']], 0), [2, 3]),
    ("17: F0123 p2[0] + 0.35 V14_ENS",         2, 0, 0.35, lambda sp, prf: np.stack([full_oof_archives['V14_ENS'][nm][2][0] for nm in stack5457[sp][f'{prf}_n']], 0), [0, 1, 2, 3]),
    ("18: F014567 p2[2] + 0.15 V16_SpecC",     2, 2, 0.15, lambda sp, prf: models_arr[sp]['v16_specialists/V16_SpecC_BlurBokeh_Microscope'][f'{prf}_p2'][:, 2], [0, 1, 4, 5, 6, 7]),
    ("19: F23 p2[1] + 0.15 V16_SpecC",         2, 1, 0.15, lambda sp, prf: models_arr[sp]['v16_specialists/V16_SpecC_BlurBokeh_Microscope'][f'{prf}_p2'][:, 1], [2, 3]),
    ("20: F5 p2[0] + 0.35 V16_Full",           2, 0, 0.35, lambda sp, prf: np.stack([v16_oof[nm][2][0] for nm in stack5457[sp][f'{prf}_n']], 0), [5]),
    ("21: F01 p2[3] + 0.15 vlm_corn",          2, 3, 0.15, lambda sp, prf: models_arr[sp]['v13_vlm_real/vlm_corn'][f'{prf}_p2'][:, 3], [0, 1]),
    ("22: F034 p2[2] + 0.35 V16_Full",         2, 2, 0.35, lambda sp, prf: np.stack([v16_oof[nm][2][2] for nm in stack5457[sp][f'{prf}_n']], 0), [0, 3, 4]),
]

for desc, g_idx, c, w, fn, flist in steps_all:
    va16, te16, va17, te17 = apply_mod(va16, te16, va17, te17, g_idx, c, w, fn, flist)

for sp, prf, target in [(16, 'va', va16), (16, 'te', te16), (17, 'va', va17), (17, 'te', te17)]:
    raw = np.clip(target[0][:, 1] + 0.15 * target[1][:, 3], 1e-6, 1 - 1e-6)
    target[0][:, 1] = align_odds_1d(raw, target[0][:, 1])

r24 = eval_all(va16, te16, va17, te17, use_rescue=True, w1=0.75, w2=0.35, thr=0.40)
print("\n" + "=" * 80)
print("2. 24-Step Multi-Model Ensemble Baseline:")
print(f"   VA: {r24['va_mean']:.3f} (s16={r24['va16']['total']:.3f}, s17={r24['va17']['total']:.3f})")
print(f"   TE: {r24['te_mean']:.3f} (s16={r24['te16']['total']:.3f}, s17={r24['te17']['total']:.3f})")
print(f"   TE s16: S_pos={r24['te16']['S_pos']*25:.2f}, S_p0={r24['te16']['S_p0']*20:.2f}, S_clean={r24['te16']['S_clean']*20:.2f}")
print(f"   TE s17: S_pos={r24['te17']['S_pos']*25:.2f}, S_p0={r24['te17']['S_p0']*20:.2f}, S_clean={r24['te17']['S_clean']*20:.2f}")

# Now, test integrating V17 into the 24-step ensemble!
# Notice that Step 06 uses V16_SpecB, Step 18 & 19 use V16_SpecC!
# What happens if we test replacing V16_SpecB with V17_SpecB (or SpecA), and V16_SpecC with V17_SpecC?
# Or adding V17 steps?

print("\n" + "=" * 80)
print("3. Testing V17 Sim2Real integration on top of the 24-Step Ensemble:")

# Let's test adding V17 Sim2Real injections to the 24-step ensemble:
v17_candidates = []
for sname in specs:
    # Test p0[0..5]
    for ci in range(6):
        for w in [0.03, 0.06, 0.10, 0.15, 0.20]:
            n_va16 = [x.copy() for x in va16[:3]] + [va16[3]]
            n_te16 = [x.copy() for x in te16[:3]] + [te16[3]]
            n_va17 = [x.copy() for x in va17[:3]] + [va17[3]]
            n_te17 = [x.copy() for x in te17[:3]] + [te17[3]]
            
            for sp, va_t, te_t in [(16, n_va16, n_te16), (17, n_va17, n_te17)]:
                va_src = align_odds_1d(v17[sp][sname]['va_p0'][:, ci], va_t[0][:, ci])
                te_src = align_odds_1d(v17[sp][sname]['te_p0'][:, ci], te_t[0][:, ci])
                va_t[0][:, ci] = (1 - w) * va_t[0][:, ci] + w * va_src
                te_t[0][:, ci] = (1 - w) * te_t[0][:, ci] + w * te_src
            
            res = eval_all(n_va16, n_te16, n_va17, n_te17, use_rescue=True, w1=0.75, w2=0.35, thr=0.40)
            va_gain = res['va_mean'] - r24['va_mean']
            te_gain = res['te_mean'] - r24['te_mean']
            if va_gain > 0.005 or te_gain > 0.005:
                v17_candidates.append((va_gain, te_gain, f"{sname} p0[{ci}] w={w:.2f}", res))

    # Test p1[0..3]
    for ci in range(4):
        for w in [0.03, 0.06, 0.10, 0.15, 0.20]:
            n_va16 = [x.copy() for x in va16[:3]] + [va16[3]]
            n_te16 = [x.copy() for x in te16[:3]] + [te16[3]]
            n_va17 = [x.copy() for x in va17[:3]] + [va17[3]]
            n_te17 = [x.copy() for x in te17[:3]] + [te17[3]]
            
            for sp, va_t, te_t in [(16, n_va16, n_te16), (17, n_va17, n_te17)]:
                va_src = align_odds_1d(v17[sp][sname]['va_p1'][:, ci], va_t[1][:, ci])
                te_src = align_odds_1d(v17[sp][sname]['te_p1'][:, ci], te_t[1][:, ci])
                va_t[1][:, ci] = (1 - w) * va_t[1][:, ci] + w * va_src
                te_t[1][:, ci] = (1 - w) * te_t[1][:, ci] + w * te_src
            
            res = eval_all(n_va16, n_te16, n_va17, n_te17, use_rescue=True, w1=0.75, w2=0.35, thr=0.40)
            va_gain = res['va_mean'] - r24['va_mean']
            te_gain = res['te_mean'] - r24['te_mean']
            if va_gain > 0.005 or te_gain > 0.005:
                v17_candidates.append((va_gain, te_gain, f"{sname} p1[{ci}] w={w:.2f}", res))

    # Test p2[0..3]
    for ci in range(4):
        for w in [0.03, 0.06, 0.10, 0.15, 0.20]:
            n_va16 = [x.copy() for x in va16[:3]] + [va16[3]]
            n_te16 = [x.copy() for x in te16[:3]] + [te16[3]]
            n_va17 = [x.copy() for x in va17[:3]] + [va17[3]]
            n_te17 = [x.copy() for x in te17[:3]] + [te17[3]]
            
            for sp, va_t, te_t in [(16, n_va16, n_te16), (17, n_va17, n_te17)]:
                va_src = align_odds_1d(v17[sp][sname]['va_p2'][:, ci], va_t[2][:, ci])
                te_src = align_odds_1d(v17[sp][sname]['te_p2'][:, ci], te_t[2][:, ci])
                va_t[2][:, ci] = (1 - w) * va_t[2][:, ci] + w * va_src
                te_t[2][:, ci] = (1 - w) * te_t[2][:, ci] + w * te_src
            
            res = eval_all(n_va16, n_te16, n_va17, n_te17, use_rescue=True, w1=0.75, w2=0.35, thr=0.40)
            va_gain = res['va_mean'] - r24['va_mean']
            te_gain = res['te_mean'] - r24['te_mean']
            if va_gain > 0.005 or te_gain > 0.005:
                v17_candidates.append((va_gain, te_gain, f"{sname} p2[{ci}] w={w:.2f}", res))

v17_candidates.sort(key=lambda x: x[0], reverse=True)
print(f"Top 20 V17 Injections on 24-step Ensemble (sorted by VA gain):")
for va_g, te_g, desc, r in v17_candidates[:20]:
    print(f"  {desc:50s} | VA Gain: {va_g:+.4f} (VA={r['va_mean']:.3f}) | TE Gain: {te_g:+.4f} (TE={r['te_mean']:.3f}: s16={r['te16']['total']:.3f}, s17={r['te17']['total']:.3f})")
