"""Explore Zero-Leakage Per-Channel Specialist Optimization + Two-Stage Conditional Grade Calibration
+ 60-Dim Native Physical Caliper Refinement on s16 & s17 (1,000-Image Isolated Test Sets).
"""
import csv
import glob
import os
import sys
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from extract_native_phys_feats import extract_for_names
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale, decode_shielded_pc


def evaluate_per_channel_val_selection():
    with open('data/manifest_combined.csv') as f:
        all_names = [r['name'] for r in csv.DictReader(f)]
    phys_map = extract_for_names(all_names)

    # Check what other npz directories exist in results/
    all_npz = sorted(glob.glob('results/**/*probs.npz', recursive=True))
    dirs = {}
    for p in all_npz:
        d = os.path.dirname(p)
        dirs[d] = dirs.get(d, 0) + 1
    print("NPZ counts by directory:", dirs)

    splits_data = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        with open(f'data/resplits/split_s{sp}.csv') as f:
            sp_rows = list(csv.DictReader(f))
        va_names = [r['name'] for r in sp_rows if r['split'] == 'val']
        te_names = [r['name'] for r in sp_rows if r['split'] == 'test']
        X_phys_va = np.stack([phys_map[nm] for nm in va_names], axis=0)
        X_phys_te = np.stack([phys_map[nm] for nm in te_names], axis=0)
        splits_data[sp] = (models, gva, gte, X_phys_va, X_phys_te)

    # Find common model keys across s16 and s17
    common_keys = sorted(set(splits_data[16][0].keys()) & set(splits_data[17][0].keys()))
    print(f"\nFound {len(common_keys)} common healthy zero-leakage model families across s16 & s17:")
    for k in common_keys:
        print(f"  - {k}")

    # For each class ci (0..5) and grade channel (p0, p1, p2), evaluate every model on val (s16 + s17)
    # and rank them by Val F1 / AUC so we pick the strongest specialists strictly via Val!
    print("\n=== Per-Channel Val Performance Ranking (Strictly on 357 Val Images per Split) ===")
    for ci, c in enumerate(CLASSES):
        print(f"\n--- Class {ci}: {c} ---")
        rank_list = []
        for k in common_keys:
            va_f1s, te_f1s = [], []
            va_p0s, te_p0s = [], []
            for sp in (16, 17):
                models, gva, gte, _, _ = splits_data[sp]
                d = models[k]
                pr_va = decode_pc(d['va_p0'], d['va_p1'], d['va_p2'], HAND)
                pr_te = decode_pc(d['te_p0'], d['te_p1'], d['te_p2'], HAND)
                sc_va = cm.score(gva, pr_va, TAU)
                sc_te = cm.score(gte, pr_te, TAU)
                f_va = [x[4] for x in sc_va['per'] if x[0] == c]
                f_te = [x[4] for x in sc_te['per'] if x[0] == c]
                va_f1s.append(np.mean(f_va))
                te_f1s.append(np.mean(f_te))
                va_p0s.append(f_va[-1])
                te_p0s.append(f_te[-1])
            rank_list.append((np.mean(va_f1s), np.mean(te_f1s), np.mean(va_p0s), np.mean(te_p0s), k))
        rank_list.sort(reverse=True)
        for v_f1, t_f1, v_p0, t_p0, k in rank_list[:10]:
            print(f"  {k:36s} | Val_MeanF1={v_f1:.3f} (P0={v_p0:.3f}) | Test_MeanF1={t_f1:.3f} (P0={t_p0:.3f})")


if __name__ == '__main__':
    evaluate_per_channel_val_selection()
