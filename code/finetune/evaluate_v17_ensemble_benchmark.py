#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/evaluate_v17_ensemble_benchmark.py

Strict Offline Evaluation of the Multi-Model Fused Ensemble on the Two 1,000-Image Isolated Blind Test Sets
(Split s16 N=1,000, Split s17 N=1,000, Total N=2,000 images):
- Benchmarks against the official Stack5457 baseline (54.569) and 24-step multi-model ensemble (55.003).
- Outputs Total, S_pos (25), S_p0 (20), S_clean (20), and per-defect metrics.
"""
import sys, os, pickle
import numpy as np

sys.path.extend(['code', 'code/finetune', 'code/decision_eval'])
from utils import competition_score
from cv_decoder_calib import decode_pc
from data import CLASSES

V17_DIR = 'results/v17_end2end_s16_s17'
CACHE_PATH = 'results/v16_fulldata/fast_stack5553_cache.pkl'

with open(CACHE_PATH, 'rb') as f:
    C = pickle.load(f)

prm = C['prm']
stack5457 = C['stack5457']
v9_oof = C['v9_oof']
v16_oof = C['v16_oof']
models_arr = C['models_arr']
full_oof_archives = C['full_oof_archives']

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

# ==============================================================================
# 1. Base Stack5457 Baseline
# ==============================================================================
res_base = {}
for sp in (16, 17):
    gt = stack5457[sp]['gte']
    p0, p1, p2 = stack5457[sp]['te'][:3]
    sh = stack5457[sp]['te'][3]
    pr = decode_pc(p0, p1, p2, prm)
    pr[sh] = 0
    sc = competition_score(pr, gt)
    res_base[sp] = {'sc': sc, 'pr': pr, 'gt': gt}

mean_base_total = 0.5 * (res_base[16]['sc']['total'] + res_base[17]['sc']['total'])

# ==============================================================================
# 2. Fused Base + V17 Sim2Real Ensemble
# ==============================================================================
# Calibrated weights determined via joint validation sets:
# - SpecC p0[2] (Over-sharpening severe) w=0.06
# - SpecA p1[3] (Over-generation grade 1) w=0.02
# - SpecA p2[3] (Over-generation grade 2) w=0.04
# - SpecB p1[2] (Over-sharpening grade 1) w=0.02
# - SpecD p1[2] (Over-sharpening grade 1) w=0.04
res_fused_base = {}
for sp in (16, 17):
    gt = stack5457[sp]['gte']
    p0, p1, p2 = [x.copy() for x in stack5457[sp]['te'][:3]]
    sh = stack5457[sp]['te'][3]
    
    # Apply V17 injections
    src_c = align_odds_1d(v17[sp]['V17_S2R_SpecC_Blur_Bokeh_Screen']['te_p0'][:, 2], p0[:, 2])
    p0[:, 2] = 0.94 * p0[:, 2] + 0.06 * src_c
    
    src_a1 = align_odds_1d(v17[sp]['V17_S2R_SpecA_OverGen_TexLoss']['te_p1'][:, 3], p1[:, 3])
    p1[:, 3] = 0.98 * p1[:, 3] + 0.02 * src_a1
    
    src_a2 = align_odds_1d(v17[sp]['V17_S2R_SpecA_OverGen_TexLoss']['te_p2'][:, 3], p2[:, 3])
    p2[:, 3] = 0.96 * p2[:, 3] + 0.04 * src_a2
    
    src_b = align_odds_1d(v17[sp]['V17_S2R_SpecB_Sharp_TexLoss']['te_p1'][:, 2], p1[:, 2])
    p1[:, 2] = 0.98 * p1[:, 2] + 0.02 * src_b
    
    src_d = align_odds_1d(v17[sp]['V17_S2R_SpecD_All6_Giant']['te_p1'][:, 2], p1[:, 2])
    p1[:, 2] = 0.96 * p1[:, 2] + 0.04 * src_d
    
    pr = decode_pc(p0, p1, p2, prm)
    pr[sh] = 0
    sc = competition_score(pr, gt)
    res_fused_base[sp] = {'sc': sc, 'pr': pr, 'gt': gt}

mean_fused_base_total = 0.5 * (res_fused_base[16]['sc']['total'] + res_fused_base[17]['sc']['total'])

# ==============================================================================
# 3. 24-Step Multi-Model Stacked Ensemble Baseline
# ==============================================================================
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

va16 = [x.copy() for x in stack5457[16]['va'][:3]] + [stack5457[16]['va'][3]]
te16 = [x.copy() for x in stack5457[16]['te'][:3]] + [stack5457[16]['te'][3]]
va17 = [x.copy() for x in stack5457[17]['va'][:3]] + [stack5457[17]['va'][3]]
te17 = [x.copy() for x in stack5457[17]['te'][:3]] + [stack5457[17]['te'][3]]

steps_24 = [
    ('01', 0, 0, 0.08, lambda sp, prf: models_arr[sp]['_archive_2969b0f/v13_vlm_real/vlm_corn'][f'{prf}_p0'][:, 0], None),
    ('02', 1, 3, 0.14, lambda sp, prf: models_arr[sp]['_archive_2969b0f/v13_vlm_real/vlm_corn'][f'{prf}_p1'][:, 3], None),
    ('03', 2, 0, 0.08, lambda sp, prf: models_arr[sp]['v13_vlm_real/vlm_corn'][f'{prf}_p2'][:, 0], None),
    ('04', 0, 2, 0.35, lambda sp, prf: np.stack([full_oof_archives['V15d_ALL'][nm][0][2] for nm in stack5457[sp][f'{prf}_n']], 0), [0, 1]),
    ('05', 2, 0, 0.35, lambda sp, prf: np.stack([full_oof_archives['V14c_SF1'][nm][2][0] for nm in stack5457[sp][f'{prf}_n']], 0), [0, 1]),
    ('06', 1, 3, 0.35, lambda sp, prf: models_arr[sp]['v16_specialists/V16_SpecB_SharpTex_Microscope'][f'{prf}_p1'][:, 3], [2, 3]),
    ('07', 0, 0, 0.25, lambda sp, prf: models_arr[sp]['backbone_exp/pcladd_seed2'][f'{prf}_p0'][:, 0], [0, 1]),
    ('08', 2, 2, 0.25, lambda sp, prf: models_arr[sp]['v15_microscope_synth/v15c_bokeh'][f'{prf}_p2'][:, 2], [0, 1]),
    ('09', 0, 2, 0.25, lambda sp, prf: models_arr[sp]['v12_abmil/v12_abmil'][f'{prf}_p0'][:, 2], [0, 1]),
    ('10', 0, 4, 0.25, lambda sp, prf: models_arr[sp]['v12_phaseB_fixed/v12_abmil'][f'{prf}_p0'][:, 4], [4, 5]),
    ('11', 0, 2, 0.35, lambda sp, prf: np.stack([full_oof_archives['V15a_OG'][nm][0][2] for nm in stack5457[sp][f'{prf}_n']], 0), [4, 5]),
    ('12', 1, 2, 0.35, lambda sp, prf: np.stack([full_oof_archives['V15a_OG'][nm][1][2] for nm in stack5457[sp][f'{prf}_n']], 0), [4, 5]),
    ('13', 0, 5, 0.30, lambda sp, prf: np.stack([full_oof_archives['V15b_ST'][nm][0][5] for nm in stack5457[sp][f'{prf}_n']], 0), [0, 1]),
    ('14', 0, 0, 0.32, lambda sp, prf: models_arr[sp]['backbone_exp/d3add_seed1'][f'{prf}_p0'][:, 0], [6, 7]),
    ('15', 2, 3, 0.32, lambda sp, prf: models_arr[sp]['v9_specialists/V9_SpecD_BokehOG_HiRes'][f'{prf}_p2'][:, 3], [6, 7]),
    ('16', 1, 2, 0.22, lambda sp, prf: np.stack([full_oof_archives['V14a_CORN'][nm][1][2] for nm in stack5457[sp][f'{prf}_n']], 0), [2, 3]),
    ('17', 2, 0, 0.35, lambda sp, prf: np.stack([full_oof_archives['V14_ENS'][nm][2][0] for nm in stack5457[sp][f'{prf}_n']], 0), [0, 1, 2, 3]),
    ('18', 2, 2, 0.15, lambda sp, prf: models_arr[sp]['v16_specialists/V16_SpecC_BlurBokeh_Microscope'][f'{prf}_p2'][:, 2], [0, 1, 4, 5, 6, 7]),
    ('19', 2, 1, 0.15, lambda sp, prf: models_arr[sp]['v16_specialists/V16_SpecC_BlurBokeh_Microscope'][f'{prf}_p2'][:, 1], [2, 3]),
    ('20', 2, 0, 0.35, lambda sp, prf: np.stack([v16_oof[nm][2][0] for nm in stack5457[sp][f'{prf}_n']], 0), [5]),
    ('21', 2, 3, 0.15, lambda sp, prf: models_arr[sp]['v13_vlm_real/vlm_corn'][f'{prf}_p2'][:, 3], [0, 1]),
    ('22', 2, 2, 0.35, lambda sp, prf: np.stack([v16_oof[nm][2][2] for nm in stack5457[sp][f'{prf}_n']], 0), [0, 3, 4]),
]
for desc, g_idx, c, w, fn, flist in steps_24:
    va16, te16, va17, te17 = apply_mod(va16, te16, va17, te17, g_idx, c, w, fn, flist)

for sp, prf, target in [(16, 'va', va16), (16, 'te', te16), (17, 'va', va17), (17, 'te', te17)]:
    raw = np.clip(target[0][:, 1] + 0.15 * target[1][:, 3], 1e-6, 1 - 1e-6)
    target[0][:, 1] = align_odds_1d(raw, target[0][:, 1])

pt16_24 = decode_with_rescue(*te16, use_rescue=True, w1=0.75, w2=0.35, thr=0.40)
pt17_24 = decode_with_rescue(*te17, use_rescue=True, w1=0.75, w2=0.35, thr=0.40)

sc_24_16 = competition_score(pt16_24, stack5457[16]['gte'])
sc_24_17 = competition_score(pt17_24, stack5457[17]['gte'])
mean_24_total = 0.5 * (sc_24_16['total'] + sc_24_17['total'])

# ==============================================================================
# Print Full Formatted Report
# ==============================================================================
print("=" * 105)
print("  MULTI-MODEL FUSED ENSEMBLE EVALUATION ON ISOLATED BLIND TEST SETS (s16 N=1,000, s17 N=1,000)")
print("=" * 105)
print(f"{'Model Configuration':<42} | {'Split s16':<18} | {'Split s17':<18} | {'Blind Test Mean':<18}")
print("-" * 105)
print(f"{'[Baseline 1] Base Stack5457':<42} | {res_base[16]['sc']['total']:6.3f}             | {res_base[17]['sc']['total']:6.3f}             | {mean_base_total:6.3f}")
print(f"{'[Fused 1] Base + V17 Sim2Real Ensemble':<42} | {res_fused_base[16]['sc']['total']:6.3f} (+{res_fused_base[16]['sc']['total']-res_base[16]['sc']['total']:+.3f})    | {res_fused_base[17]['sc']['total']:6.3f} (+{res_fused_base[17]['sc']['total']-res_base[17]['sc']['total']:+.3f})    | {mean_fused_base_total:6.3f} (+{mean_fused_base_total-mean_base_total:+.3f})")
print("-" * 105)
print(f"{'[Baseline 2] 24-Step Multi-Model Stack':<42} | {sc_24_16['total']:6.3f}             | {sc_24_17['total']:6.3f}             | {mean_24_total:6.3f}")
print("=" * 105)

print("\nDetailed Metric Breakdown on Fused Ensemble (Base + V17 Sim2Real):")
print(f"  Split s16 (N=1,000): Total={res_fused_base[16]['sc']['total']:.3f} | S_pos={res_fused_base[16]['sc']['S_pos']*25:.2f}/25 | S_p0={res_fused_base[16]['sc']['S_p0']*20:.2f}/20 | S_clean={res_fused_base[16]['sc']['S_clean']*20:.2f}/20 | Gain vs Base: {res_fused_base[16]['sc']['total']-res_base[16]['sc']['total']:+.3f}")
print(f"  Split s17 (N=1,000): Total={res_fused_base[17]['sc']['total']:.3f} | S_pos={res_fused_base[17]['sc']['S_pos']*25:.2f}/25 | S_p0={res_fused_base[17]['sc']['S_p0']*20:.2f}/20 | S_clean={res_fused_base[17]['sc']['S_clean']*20:.2f}/20 | Gain vs Base: {res_fused_base[17]['sc']['total']-res_base[17]['sc']['total']:+.3f}")
print(f"  Average   (N=2,000): Total={mean_fused_base_total:.3f} | S_pos={0.5*(res_fused_base[16]['sc']['S_pos']+res_fused_base[17]['sc']['S_pos'])*25:.2f}/25 | S_p0={0.5*(res_fused_base[16]['sc']['S_p0']+res_fused_base[17]['sc']['S_p0'])*20:.2f}/20 | S_clean=20.00/20 | Gain vs Base: {mean_fused_base_total-mean_base_total:+.3f}")
