import sys, os, pickle
import numpy as np
from scipy.optimize import minimize

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

# Baseline evaluation
for sp in (16, 17):
    gte = stack5457[sp]['gte']
    p0, p1, p2 = stack5457[sp]['te'][:3]
    sh = stack5457[sp]['te'][3]
    pr = decode_pc(p0, p1, p2, prm)
    pr[sh] = 0
    sc = competition_score(pr, gte)
    print(f"Base Stack5457 s{sp}: Total={sc['total']:.3f} (S_pos={sc['S_pos']*25:.2f}, S_p0={sc['S_p0']*20:.2f}, S_clean={sc['S_clean']*20:.2f})")

# Let's test a clean, interpretable selective routing for V17 into Stack5457
# We want to find a set of weights W_p0 (6,), W_p1 (4,), W_p2 (4,) for each specialist
# that strictly improves VA mean while preserving clean shield.

def evaluate_blend(weights_dict, base_type='stack5457'):
    # weights_dict: {spec_name: {'p0': (6,), 'p1': (4,), 'p2': (4,)}}
    res = {}
    for split_type in ['va', 'te']:
        split_scores = {}
        for sp in (16, 17):
            gt = stack5457[sp][f'g{split_type}']
            p0, p1, p2 = [x.copy() for x in stack5457[sp][split_type][:3]]
            sh = stack5457[sp][split_type][3]
            
            for sname, w in weights_dict.items():
                m = v17[sp][sname]
                for ci in range(6):
                    w_val = w['p0'][ci]
                    if w_val > 0:
                        src = align_odds_1d(m[f'{split_type}_p0'][:, ci], p0[:, ci])
                        p0[:, ci] = (1 - w_val) * p0[:, ci] + w_val * src
                for ci in range(4):
                    w_val = w['p1'][ci]
                    if w_val > 0:
                        src = align_odds_1d(m[f'{split_type}_p1'][:, ci], p1[:, ci])
                        p1[:, ci] = (1 - w_val) * p1[:, ci] + w_val * src
                    w_val2 = w['p2'][ci]
                    if w_val2 > 0:
                        src2 = align_odds_1d(m[f'{split_type}_p2'][:, ci], p2[:, ci])
                        p2[:, ci] = (1 - w_val2) * p2[:, ci] + w_val2 * src2
            
            pr = decode_pc(p0, p1, p2, prm)
            pr[sh] = 0
            sc = competition_score(pr, gt)
            split_scores[sp] = sc
        res[split_type] = split_scores
        res[f'{split_type}_mean'] = 0.5 * (split_scores[16]['total'] + split_scores[17]['total'])
    return res

# Coordinate search on VA
best_weights = {
    s: {'p0': np.zeros(6), 'p1': np.zeros(4), 'p2': np.zeros(4)} for s in specs
}

base_res = evaluate_blend(best_weights)
print(f"\nInitial Base: VA={base_res['va_mean']:.4f} | TE={base_res['te_mean']:.4f} (s16={base_res['te'][16]['total']:.3f}, s17={base_res['te'][17]['total']:.3f})")

current_best_va = base_res['va_mean']
step_options = [0.02, 0.04, 0.06, 0.08, 0.12]

# Greedy coordinate search purely on VA
improved = True
pass_num = 0
while improved and pass_num < 3:
    improved = False
    pass_num += 1
    print(f"\n--- Pass {pass_num} ---")
    for sname in specs:
        for head in ['p0', 'p1', 'p2']:
            n_c = 6 if head == 'p0' else 4
            for ci in range(n_c):
                best_w = best_weights[sname][head][ci]
                best_gain = 0.0
                best_cand_w = best_w
                
                for cand_w in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
                    if cand_w == best_w:
                        continue
                    best_weights[sname][head][ci] = cand_w
                    r = evaluate_blend(best_weights)
                    va_sc = r['va_mean']
                    # Require that BOTH va16 and va17 do not regress by more than 0.05
                    if va_sc > current_best_va and r['va'][16]['total'] >= base_res['va'][16]['total'] - 0.05 and r['va'][17]['total'] >= base_res['va'][17]['total'] - 0.05:
                        gain = va_sc - current_best_va
                        if gain > best_gain:
                            best_gain = gain
                            best_cand_w = cand_w
                    best_weights[sname][head][ci] = best_w
                
                if best_gain > 0.002:
                    best_weights[sname][head][ci] = best_cand_w
                    current_best_va += best_gain
                    improved = True
                    r = evaluate_blend(best_weights)
                    print(f"  Applied {sname} {head}[{ci}] = {best_cand_w:.2f} -> VA={r['va_mean']:.4f} (+{best_gain:.4f}) | TE={r['te_mean']:.4f} (s16={r['te'][16]['total']:.3f}, s17={r['te'][17]['total']:.3f})")

final_res = evaluate_blend(best_weights)
print("\n" + "=" * 80)
print("FINAL CALIBRATED ENSEMBLE EVALUATION:")
print(f"  VA Baseline Mean : {base_res['va_mean']:.4f} (s16={base_res['va'][16]['total']:.4f}, s17={base_res['va'][17]['total']:.4f})")
print(f"  VA Final Mean    : {final_res['va_mean']:.4f} (s16={final_res['va'][16]['total']:.4f}, s17={final_res['va'][17]['total']:.4f}) | Gain: {final_res['va_mean']-base_res['va_mean']:+.4f}")
print("-" * 80)
print(f"  TE Baseline Mean : {base_res['te_mean']:.4f} (s16={base_res['te'][16]['total']:.4f}, s17={base_res['te'][17]['total']:.4f})")
print(f"  TE Final Mean    : {final_res['te_mean']:.4f} (s16={final_res['te'][16]['total']:.4f}, s17={final_res['te'][17]['total']:.4f}) | Gain: {final_res['te_mean']-base_res['te_mean']:+.4f}")
print("=" * 80)

print("\nNon-zero routing weights:")
for sname in specs:
    for head in ['p0', 'p1', 'p2']:
        for ci, w in enumerate(best_weights[sname][head]):
            if w > 0:
                print(f"  {sname} {head}[{ci}] = {w:.2f}")
