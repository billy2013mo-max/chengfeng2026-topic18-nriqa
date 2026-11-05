"""Audit exact theoretical ceilings, cross-defect conditional probabilities, and multi-signal
non-linear boundary rules on s16 & s17 (1,000 test images each).
"""
import os, sys, csv
import numpy as np
from sklearn.metrics import average_precision_score

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from build_v9_meta_stacker import load_aux_context_dict, build_meta_features_for_split

aux_dict = load_aux_context_dict()
splits = {}
for sp in (16, 17):
    models, gva, gte = load_healthy_split_models(sp)
    for pref in ['pcl_448t3_seed1', 'd3_448t3_seed1', 'pcl_384t2_seed2', 'd3_384t2_seed1']:
        fep = f'results/v6_val/{pref}_s{sp}_epochs.npz'
        if os.path.exists(fep):
            dep = np.load(fep)
            for ep in (1, 2, 3, 4):
                if f'ep{ep}_p0' in dep.files:
                    models[f'v6ep/{pref}_ep{ep}'] = {
                        'te_p0': dep[f'ep{ep}_p0'],
                        'te_p1': dep[f'ep{ep}_p1'],
                        'te_p2': dep[f'ep{ep}_p2'],
                    }
    ref = np.load(f'results/v3_ablate/A_base_s{sp}_probs.npz', allow_pickle=True)
    va_names = [str(x) for x in ref['va_names']]
    te_names = [str(x) for x in ref['te_names']]
    splits[sp] = (models, gva, gte, va_names, te_names)

common_keys = sorted(set(splits[16][0].keys()) & set(splits[17][0].keys()))
print(f"Total aligned models: {len(common_keys)}")

# Let's check for each FULL class (模糊, 纹理损失, 过锐, 过度生成) and each BIN class (花屏, 虚化突兀):
# 1. What is the exact point breakdown currently?
# 2. If we know the exact presence (gt >= 1) vs current presence (pr >= 1), how many points are lost in:
#    (a) False Negatives / False Positives at Presence boundary (0 vs >=1)
#    (b) Grade confusion between P2 (1) and P1 (2)
#    (c) Grade confusion between P1 (2) and P0 (3)
for sp in (16, 17):
    models, gva, gte, va_names, te_names = splits[sp]
    te_p0, te_p1, te_p2, cln0, cln1, cln2 = build_decoupled_14ch_probs(models, 'te')
    cum_cln = np.zeros((1000, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum_cln[:, ci] = cln0[:, ci] + cln1[:, k] + cln2[:, k]
        else:
            cum_cln[:, ci] = cln0[:, ci]
    sh = cum_cln.max(1) < 0.28
    # Check how many true defects are killed by `sh` (Clean Shield)!
    killed_by_sh = np.sum((gte[sh] > 0))
    print(f"[s{sp}] Clean shield active on {sh.sum()}/1000 images; true defect cells inside shield = {killed_by_sh}")
