import sys, os, pickle
import numpy as np

sys.path.extend(['code', 'code/finetune', 'code/decision_eval'])
from utils import competition_score
from cv_decoder_calib import decode_pc

with open('results/v16_fulldata/fast_stack5553_cache.pkl', 'rb') as f:
    C = pickle.load(f)

prm = C['prm']
stack5457 = C['stack5457']

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

def eval_preds(va16, te16, va17, te17):
    pv16 = decode_pc(*va16[:3], prm); pv16[va16[3]] = 0
    pt16 = decode_pc(*te16[:3], prm); pt16[te16[3]] = 0
    pv17 = decode_pc(*va17[:3], prm); pv17[va17[3]] = 0
    pt17 = decode_pc(*te17[:3], prm); pt17[te17[3]] = 0
    
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

base_res = eval_preds(
    stack5457[16]['va'], stack5457[16]['te'],
    stack5457[17]['va'], stack5457[17]['te']
)
print("=== Base Stack5457 Baseline ===")
print(f"VA Mean: {base_res['va_mean']:.4f} (s16={base_res['va16']['total']:.4f}, s17={base_res['va17']['total']:.4f})")
print(f"TE Mean: {base_res['te_mean']:.4f} (s16={base_res['te16']['total']:.4f}, s17={base_res['te17']['total']:.4f})")

# Test single channel injections across all specialists
candidates = []
for sname in specs:
    # Test p0 channels 0..5
    for ci in range(6):
        for w in [0.02, 0.05, 0.08, 0.12, 0.18]:
            # copy
            va16 = [x.copy() for x in stack5457[16]['va'][:3]] + [stack5457[16]['va'][3]]
            te16 = [x.copy() for x in stack5457[16]['te'][:3]] + [stack5457[16]['te'][3]]
            va17 = [x.copy() for x in stack5457[17]['va'][:3]] + [stack5457[17]['va'][3]]
            te17 = [x.copy() for x in stack5457[17]['te'][:3]] + [stack5457[17]['te'][3]]
            
            for sp, va_t, te_t in [(16, va16, te16), (17, va17, te17)]:
                va_src = align_odds_1d(v17[sp][sname]['va_p0'][:, ci], va_t[0][:, ci])
                te_src = align_odds_1d(v17[sp][sname]['te_p0'][:, ci], te_t[0][:, ci])
                va_t[0][:, ci] = (1 - w) * va_t[0][:, ci] + w * va_src
                te_t[0][:, ci] = (1 - w) * te_t[0][:, ci] + w * te_src
            
            res = eval_preds(va16, te16, va17, te17)
            va_gain = res['va_mean'] - base_res['va_mean']
            te_gain = res['te_mean'] - base_res['te_mean']
            if va_gain > 0.005:
                candidates.append((va_gain, te_gain, f"{sname} p0[{ci}] w={w:.2f}", res))

    # Test p1 channels 0..3
    for ci in range(4):
        for w in [0.02, 0.05, 0.08, 0.12, 0.18]:
            va16 = [x.copy() for x in stack5457[16]['va'][:3]] + [stack5457[16]['va'][3]]
            te16 = [x.copy() for x in stack5457[16]['te'][:3]] + [stack5457[16]['te'][3]]
            va17 = [x.copy() for x in stack5457[17]['va'][:3]] + [stack5457[17]['va'][3]]
            te17 = [x.copy() for x in stack5457[17]['te'][:3]] + [stack5457[17]['te'][3]]
            
            for sp, va_t, te_t in [(16, va16, te16), (17, va17, te17)]:
                va_src = align_odds_1d(v17[sp][sname]['va_p1'][:, ci], va_t[1][:, ci])
                te_src = align_odds_1d(v17[sp][sname]['te_p1'][:, ci], te_t[1][:, ci])
                va_t[1][:, ci] = (1 - w) * va_t[1][:, ci] + w * va_src
                te_t[1][:, ci] = (1 - w) * te_t[1][:, ci] + w * te_src
            
            res = eval_preds(va16, te16, va17, te17)
            va_gain = res['va_mean'] - base_res['va_mean']
            te_gain = res['te_mean'] - base_res['te_mean']
            if va_gain > 0.005:
                candidates.append((va_gain, te_gain, f"{sname} p1[{ci}] w={w:.2f}", res))

    # Test p2 channels 0..3
    for ci in range(4):
        for w in [0.02, 0.05, 0.08, 0.12, 0.18]:
            va16 = [x.copy() for x in stack5457[16]['va'][:3]] + [stack5457[16]['va'][3]]
            te16 = [x.copy() for x in stack5457[16]['te'][:3]] + [stack5457[16]['te'][3]]
            va17 = [x.copy() for x in stack5457[17]['va'][:3]] + [stack5457[17]['va'][3]]
            te17 = [x.copy() for x in stack5457[17]['te'][:3]] + [stack5457[17]['te'][3]]
            
            for sp, va_t, te_t in [(16, va16, te16), (17, va17, te17)]:
                va_src = align_odds_1d(v17[sp][sname]['va_p2'][:, ci], va_t[2][:, ci])
                te_src = align_odds_1d(v17[sp][sname]['te_p2'][:, ci], te_t[2][:, ci])
                va_t[2][:, ci] = (1 - w) * va_t[2][:, ci] + w * va_src
                te_t[2][:, ci] = (1 - w) * te_t[2][:, ci] + w * te_src
            
            res = eval_preds(va16, te16, va17, te17)
            va_gain = res['va_mean'] - base_res['va_mean']
            te_gain = res['te_mean'] - base_res['te_mean']
            if va_gain > 0.005:
                candidates.append((va_gain, te_gain, f"{sname} p2[{ci}] w={w:.2f}", res))

candidates.sort(key=lambda x: x[0], reverse=True)
print(f"\nTop 20 Single-Channel Injections (sorted by VA gain):")
for va_g, te_g, desc, r in candidates[:20]:
    print(f"  {desc:50s} | VA Gain: {va_g:+.4f} (VA={r['va_mean']:.3f}) | TE Gain: {te_g:+.4f} (TE={r['te_mean']:.3f}: s16={r['te16']['total']:.3f}, s17={r['te17']['total']:.3f})")
