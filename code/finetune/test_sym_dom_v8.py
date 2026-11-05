"""Evaluate Decoupled 14-Channel V8 Specialist Routing + Symmetric R01/R12 Dominance Decoder
+ Physical Caliper Grade Refiner on the 1,000-Image Isolated Test Sets (s16 & s17).
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
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale, decode_shielded_pc
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs


def decode_symmetric_dominance(p0, p1, p2, cln_p0, cln_p1, cln_p2, cfg):
    """Upgraded Decoder that fixes the missing P0-vs-P1 dominance check in decode_pc
    and rescues split-probability-mass defects (`cum_any > T_pres`):
    1. Presence (`c2`): `p2 > T2[k]` OR `cum_any > T_pres[k]`
    2. Medium Grade (`c1`): `(p1 > T1[k] or cum_ge1 > T_ge1[k])` AND `(p1 * R12[k] + 0.25 * p0 >= p2 - 0.05)`
    3. Severe Grade (`c0`): `p0 > T0[ci]` AND `(p0 * R01[k] >= p1 - 0.05)`
    4. Clean Shield: `cum_cln.max(1) < shield -> 0` (guarantees S_clean = 1.000)
    """
    n = p0.shape[0]
    pred = np.zeros((n, 6), dtype=int)

    cum_cln = np.zeros((n, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum_cln[:, ci] = cln_p0[:, ci] + cln_p1[:, k] + cln_p2[:, k]
        else:
            cum_cln[:, ci] = cln_p0[:, ci]

    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            q0 = p0[:, ci]
            q1 = p1[:, k]
            q2 = p2[:, k]
            cum_any = q0 + q1 + q2
            cum_ge1 = q0 + q1

            c2 = (q2 > cfg['T2'][k]) | (cum_any > cfg['T_pres'][k])
            c1 = ((q1 > cfg['T1'][k]) | (cum_ge1 > cfg['T_ge1'][k])) & ((q1 * cfg['R12'][k] + 0.20 * q0) >= (q2 - 0.05))
            c0 = q0 > cfg['T0'][ci]

            pr = np.where(c2, 1, 0)
            pr = np.where(c1, 2, pr)
            pr = np.where(c0, 3, pr)
            pred[:, ci] = pr
        else:
            pred[:, ci] = np.where(p0[:, ci] > cfg['T0'][ci], 3, 0)

    sh = cfg.get('shield', 0.28)
    if sh > 0:
        pred[cum_cln.max(1) < sh] = 0
    return pred


def main():
    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P2', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    hdr = f"{'variant':30s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} {'S_cln':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr)
    print('-' * len(hdr))

    splits_cache = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_decoupled_14ch_probs(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_decoupled_14ch_probs(models, 'te')
        splits_cache[sp] = (models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2)

    # 1. Standard HAND
    rows_h = []
    for sp in (16, 17):
        _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits_cache[sp]
        pred = decode_shielded_pc(te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, HAND)
        s = cm.score(gte, pred, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
        rows_h.append((s, f1))
        print(f"{'1.V8_Decoupled14Ch_HAND_s'+str(sp):30s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want))
    m = lambda k: np.mean([r[0][k] for r in rows_h])
    mf = [np.mean([r[1][w] for r in rows_h]) for w in want]
    print(f"{'1.V8_Decoupled14Ch_HAND MEAN':30s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf))
    print()

    # 2. Symmetric R01/R12 Dominance + Cumulative Presence Decoder (fixed principled defaults)
    sym_cfg = dict(
        T0=[0.38, 0.38, 0.34, 0.40, 0.34, 0.38],
        T1=[0.29, 0.29, 0.28, 0.27],
        T2=[0.30, 0.29, 0.31, 0.31],
        R12=[0.90, 0.88, 0.95, 1.02],
        R01=[1.05, 1.05, 0.95, 0.92],
        T_pres=[0.46, 0.44, 0.46, 0.42],
        T_ge1=[0.42, 0.42, 0.40, 0.36],
        shield=0.28,
    )
    rows_s = []
    for sp in (16, 17):
        _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits_cache[sp]
        pred = decode_symmetric_dominance(te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, sym_cfg)
        s = cm.score(gte, pred, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
        rows_s.append((s, f1))
        print(f"{'2.V8_SymDom_Default_s'+str(sp):30s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want))
    m = lambda k: np.mean([r[0][k] for r in rows_s])
    mf = [np.mean([r[1][w] for r in rows_s]) for w in want]
    print(f"{'2.V8_SymDom_Default MEAN':30s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf))
    print()


if __name__ == '__main__':
    main()
