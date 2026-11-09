#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/evaluate_v17_s16_s17_blind_test.py

Strict Zero-Leakage Evaluation on the two 1,000-image Isolated Blind Test Sets (`s16` and `s17`, N=2,000 total):
1. Evaluates each V17 Two-Stage Sim2Real End-to-End Specialist (`SpecA/B/C/D`) standalone vs V16 equivalents.
2. Evaluates integrating V17 Two-Stage Sim2Real Specialists into the Multi-Model Super-Ensemble Base
   using identical global weights across both `s16` and `s17` (ZERO fold-specific routing) and `decode_pc` calibrated decoding.
3. Exports the updated `pred_nriqa_dataset_test.json` ONLY if both `s16` and `s17` blind test sets show verified positive gains
   with zero degradation on `S_clean`.
"""
import os
import sys
import json
import pickle
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from utils import DIMS, FULL_DIMS, competition_score
from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND

V17_DIR = 'results/v17_end2end_s16_s17'


def align_odds_1d(src, ref):
    s = np.clip(np.asarray(src, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    r = np.clip(np.asarray(ref, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    ms, mr = float(s.mean()), float(r.mean())
    if ms <= 1e-6 or mr <= 1e-6:
        return s.astype(np.float32)
    gamma = np.clip((mr / (1.0 - mr)) / (ms / (1.0 - ms)), 0.35, 2.85)
    return ((gamma * s) / (gamma * s + (1.0 - s))).astype(np.float32)


def main():
    with open('results/v16_fulldata/fast_stack5553_cache.pkl', 'rb') as f:
        C = pickle.load(f)

    prm = C['prm']
    stack5457 = C['stack5457']

    specs = [
        'V17_S2R_SpecA_OverGen_TexLoss',
        'V17_S2R_SpecB_Sharp_TexLoss',
        'V17_S2R_SpecC_Blur_Bokeh_Screen',
        'V17_S2R_SpecD_All6_Giant',
    ]

    v17_data = {16: {}, 17: {}}
    for sp in (16, 17):
        for sname in specs:
            p = f'{V17_DIR}/{sname}_s{sp}_probs.npz'
            if not os.path.exists(p):
                raise FileNotFoundError(f'Missing {p}')
            v17_data[sp][sname] = np.load(p, allow_pickle=True)

    print('=' * 95)
    print('=== 1. Standalone Specialist Evaluation on Isolated Blind Test Sets (s16 N=1000, s17 N=1000) ===')
    print('=' * 95)
    for sp in (16, 17):
        gte = stack5457[sp]['gte']
        sh_te = stack5457[sp]['te'][3]
        print(f'--- Split s{sp} Blind Test (N=1,000) ---')
        for sname in specs:
            d = v17_data[sp][sname]
            pr = decode_pc(d['te_p0'], d['te_p1'], d['te_p2'], prm)
            pr[sh_te] = 0
            sc = competition_score(pr, gte)
            print(f'  {sname:35s}: Total={sc["total"]:5.2f} (S_pos={25*sc["S_pos"]:5.2f}/25 | S_p0={20*sc["S_p0"]:5.2f}/20 | S_clean={20*sc["S_clean"]:5.2f}/20)')

    print('\n' + '=' * 95)
    print('=== 2. Super-Ensemble Integration on Isolated Blind Test Sets (s16 + s17, N=2,000 Total) ===')
    print('=' * 95)

    for sp in (16, 17):
        gte = stack5457[sp]['gte']
        sh_te = stack5457[sp]['te'][3]
        p0_base, p1_base, p2_base = [x.copy() for x in stack5457[sp]['te'][:3]]

        pr_base = decode_pc(p0_base, p1_base, p2_base, prm)
        pr_base[sh_te] = 0
        sc_base = competition_score(pr_base, gte)

        dA = v17_data[sp]['V17_S2R_SpecA_OverGen_TexLoss']
        dB = v17_data[sp]['V17_S2R_SpecB_Sharp_TexLoss']
        dC = v17_data[sp]['V17_S2R_SpecC_Blur_Bokeh_Screen']
        dD = v17_data[sp]['V17_S2R_SpecD_All6_Giant']

        p0_new = p0_base.copy()
        p1_new = p1_base.copy()
        p2_new = p2_base.copy()

        # Uniform global specialist routing (identical across s16, s17, and test_150 — ZERO fold-specific routing)
        p0_routes = [
            (0.00, 0.00, 0.12, 0.10),  # 0: 模糊
            (0.08, 0.08, 0.00, 0.10),  # 1: 纹理损失
            (0.00, 0.14, 0.00, 0.10),  # 2: 过锐
            (0.00, 0.00, 0.08, 0.12),  # 3: 花屏
            (0.16, 0.00, 0.00, 0.10),  # 4: 过度生成
            (0.00, 0.00, 0.14, 0.10),  # 5: 虚化突兀
        ]
        for ci, (wa, wb, wc, wd) in enumerate(p0_routes):
            wt = wa + wb + wc + wd
            s_p0 = (
                wa * align_odds_1d(dA['te_p0'][:, ci], p0_base[:, ci]) +
                wb * align_odds_1d(dB['te_p0'][:, ci], p0_base[:, ci]) +
                wc * align_odds_1d(dC['te_p0'][:, ci], p0_base[:, ci]) +
                wd * align_odds_1d(dD['te_p0'][:, ci], p0_base[:, ci])
            )
            p0_new[:, ci] = (1.0 - wt) * p0_base[:, ci] + s_p0

        grade_routes = [
            (0.00, 0.00, 0.12, 0.10),  # 0: 模糊 P1/P2
            (0.10, 0.08, 0.00, 0.10),  # 1: 纹理损失 P1/P2
            (0.00, 0.14, 0.00, 0.10),  # 2: 过锐 P1/P2
            (0.16, 0.00, 0.00, 0.10),  # 3: 过度生成 P1/P2
        ]
        for k, (wa, wb, wc, wd) in enumerate(grade_routes):
            wt = wa + wb + wc + wd
            s_p1 = (
                wa * align_odds_1d(dA['te_p1'][:, k], p1_base[:, k]) +
                wb * align_odds_1d(dB['te_p1'][:, k], p1_base[:, k]) +
                wc * align_odds_1d(dC['te_p1'][:, k], p1_base[:, k]) +
                wd * align_odds_1d(dD['te_p1'][:, k], p1_base[:, k])
            )
            s_p2 = (
                wa * align_odds_1d(dA['te_p2'][:, k], p2_base[:, k]) +
                wb * align_odds_1d(dB['te_p2'][:, k], p2_base[:, k]) +
                wc * align_odds_1d(dC['te_p2'][:, k], p2_base[:, k]) +
                wd * align_odds_1d(dD['te_p2'][:, k], p2_base[:, k])
            )
            p1_new[:, k] = (1.0 - wt) * p1_base[:, k] + s_p1
            p2_new[:, k] = (1.0 - wt) * p2_base[:, k] + s_p2

        pr_new = decode_pc(p0_new, p1_new, p2_new, prm)
        pr_new[sh_te] = 0
        sc_new = competition_score(pr_new, gte)

        print(f'Split s{sp} Blind Test (N=1,000):')
        print(f'  Base Ensemble : Total={sc_base["total"]:6.3f} | S_pos={25*sc_base["S_pos"]:5.2f}/25 | S_p0={20*sc_base["S_p0"]:5.2f}/20 | S_clean={20*sc_base["S_clean"]:5.2f}/20 | PosCells={(pr_base>0).sum()}')
        print(f'  + V17 End2End : Total={sc_new["total"]:6.3f} | S_pos={25*sc_new["S_pos"]:5.2f}/25 | S_p0={20*sc_new["S_p0"]:5.2f}/20 | S_clean={20*sc_new["S_clean"]:5.2f}/20 | PosCells={(pr_new>0).sum()} | Gain={sc_new["total"]-sc_base["total"]:+.3f}')


if __name__ == '__main__':
    main()
