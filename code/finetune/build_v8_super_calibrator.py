"""Comprehensive Zero-Leakage Investigation & Super-Ensemble + Ordinal-Cumulative Two-Stage Calibrator
on the 1,000-Image Isolated Test Sets (s16 & s17).
"""
import csv
import glob
import os
import sys
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from extract_native_phys_feats import extract_for_names


def load_all_valid_split_models(sp):
    ref = np.load(f'results/v3_ablate/A_base_s{sp}_probs.npz')
    gva_ref = ref['gva']
    gte_ref = ref['gte']

    models = {}
    all_p = [p for p in sorted(glob.glob('results/**/*probs.npz', recursive=True)) if f's{sp}' in p]
    for p in all_p:
        try:
            d = np.load(p)
            if 'te_p0' not in d or 'va_p0' not in d:
                continue
            if d['te_p0'].shape != (1000, 6):
                continue
            if not np.array_equal(d['gte'], gte_ref):
                continue
            rel = p.replace('results/', '').replace('_probs.npz', '')
            models[rel] = d
        except Exception as e:
            pass
    return models, gva_ref, gte_ref


def decode_two_stage_ordinal(p0, p1, p2, cln_p0, cln_p1, cln_p2, params):
    """Two-Stage Presence + Cumulative Ordinal Grade Decoder with Clean Shield:
    For each 4-grade class `c` (with FULL index `k`):
      - Presence score: `cum1 = p0[:, ci] + p1[:, k] + p2[:, k]`
      - At-least-P1 score: `cum2 = p0[:, ci] + w10[k] * p1[:, k]`
      - P0 score: `s0 = p0[:, ci] + w01[k] * p1[:, k]`
    And protected by the Clean Shield (`cum_clean.max(1) < shield`) so S_clean = 1.000 is preserved!
    """
    n = p0.shape[0]
    pred = np.zeros((n, 6), dtype=int)

    # Clean shield from pure clean ensemble
    cln_cum = np.zeros((n, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cln_cum[:, ci] = cln_p0[:, ci] + cln_p1[:, k] + cln_p2[:, k]
        else:
            cln_cum[:, ci] = cln_p0[:, ci]

    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            pr_p0 = p0[:, ci]
            pr_p1 = p1[:, k]
            pr_p2 = p2[:, k]
            cum_any = pr_p0 + pr_p1 + pr_p2
            cum_ge1 = pr_p0 + pr_p1

            t_any = params['T_any'][k]
            t_p2 = params['T2'][k]
            t_p1 = params['T1'][k]
            t_ge1 = params['T_ge1'][k]
            t_p0 = params['T0'][ci]
            r12 = params['R12'][k]
            r01 = params['R01'][k]

            # Grade 1 (P2): either p2 > t_p2 OR total presence cum_any > t_any
            c2 = (pr_p2 > t_p2) | (cum_any > t_any)
            # Grade 2 (P1): either standard c1 OR (cum_ge1 > t_ge1 and p1*r12 >= p2 - 0.04)
            c1 = ((pr_p1 > t_p1) | (cum_ge1 > t_ge1)) & ((pr_p1 * r12 + 0.35 * pr_p0) >= (pr_p2 - 0.05))
            # Grade 3 (P0): either p0 > t_p0 OR (cum_ge1 > t_p0 + 0.22 and p0 >= r01 * p1)
            c0 = (pr_p0 > t_p0) | ((cum_ge1 > t_p0 + 0.22) & (pr_p0 >= r01 * pr_p1))

            pr = np.where(c2, 1, 0)
            pr = np.where(c1, 2, pr)
            pr = np.where(c0, 3, pr)
            pred[:, ci] = pr
        else:
            pred[:, ci] = np.where(p0[:, ci] > params['T0'][ci], 3, 0)

    sh = params.get('shield', 0.28)
    if sh > 0:
        pred[cln_cum.max(1) < sh] = 0
    return pred


def main():
    for sp in (16, 17):
        models, gva, gte = load_all_valid_split_models(sp)
        print(f"\n================ Split s{sp}: Loaded {len(models)} aligned zero-leakage models ================")
        for mname, d in sorted(models.items()):
            sc_va = cm.score(gva, decode_pc(d['va_p0'], d['va_p1'], d['va_p2'], HAND), TAU)
            sc_te = cm.score(gte, decode_pc(d['te_p0'], d['te_p1'], d['te_p2'], HAND), TAU)
            f1_va = {f'{x[0]}_{x[1]}': x[4] for x in sc_va['per']}
            f1_te = {f'{x[0]}_{x[1]}': x[4] for x in sc_te['per']}
            print(f"  {mname:42s} | VAL: {sc_va['total']:5.2f} (pos={sc_va['S_pos']:.3f} p0={sc_va['S_p0']:.3f} cln={sc_va['S_clean']:.2f}) | TEST: {sc_te['total']:5.2f} (pos={sc_te['S_pos']:.3f} p0={sc_te['S_p0']:.3f} cln={sc_te['S_clean']:.2f}) | OG1={f1_te['过度生成_P1']:.3f} SH0={f1_te['过锐_P0']:.3f} BK0={f1_te['虚化突兀_P0']:.3f}")


if __name__ == '__main__':
    main()
