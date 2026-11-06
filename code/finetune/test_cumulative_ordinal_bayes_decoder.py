"""Verify the 'Split-Probability Mass Bug' in `decode_pc` (HAND) and test
Cumulative-Consistent Ordinal Decoding + QualiRAG Scene-Calibrated Baseline Subtraction + Stage-2A Clean Gate!
"""
import os, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from run_scheme1_scheme2_breakthrough import FULL_IDX, BIN_IDX, score_both
from fast_test_qualirag_preserved_p0 import CACHE_PATH, align_prob_mean, build_p0_preserved_v9_14ch

cache = torch.load(CACHE_PATH, weights_only=False)

for sp in (16, 17):
    models, gva, gte = load_healthy_split_models(sp)
    p0_te, p1_te, p2_te, cln0_te, cln1_te, cln2_te = build_p0_preserved_v9_14ch(models, 'te', None)
    pred_hand = decode_pc(p0_te, p1_te, p2_te, HAND)

    print(f"=== Split s{sp}: Checking Split-Probability Mass Bug in `decode_pc(HAND)` ===")
    for k, ci in enumerate(FULL_IDX):
        cname = CLASSES[ci]
        q0, q1, q2 = p0_te[:, ci], p1_te[:, k], p2_te[:, k]
        q_cum = q0 + q1 + q2
        q_01 = q0 + q1
        # Cases where decode_pc output 0, even though q_cum > 0.42 or q_01 > 0.34
        missed_cum = (pred_hand[:, ci] == 0) & (q_cum > 0.42)
        missed_01  = (pred_hand[:, ci] <= 1) & (q_01 > 0.36)
        true_pos_in_missed_cum = (gte[missed_cum, ci] >= 1).mean() if missed_cum.sum() > 0 else 0.0
        true_ge2_in_missed_01  = (gte[missed_01, ci] >= 2).mean() if missed_01.sum() > 0 else 0.0
        print(f"  {cname:6s}: missed with cum>0.42: {missed_cum.sum():3d} imgs (true defect rate = {true_pos_in_missed_cum:.1%}) | "
              f"missed P1/P0 with (p0+p1)>0.36: {missed_01.sum():3d} imgs (true >=P1 rate = {true_ge2_in_missed_01:.1%})")
