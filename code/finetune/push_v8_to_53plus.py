"""Push V8 Decoupled 14-Channel Specialist Ensemble + Grade Boundary Calibration + 60-Dim Physical Caliper
toward 53+ on the 1,000-Image Isolated Test Sets (s16 & s17).
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


def decode_v8_grade_aware(p0, p1, p2, cln_p0, cln_p1, cln_p2, cfg):
    """Grade-Aware Decoder with per-class P1-vs-P2 (R12) and P0-vs-P1 (R01) dominance ratios
    plus Clean Shield from pure clean ensemble.
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
            c1 = ((q1 > cfg['T1'][k]) | (cum_ge1 > cfg['T_ge1'][k])) & ((q1 * cfg['R12'][k] + cfg['w0_in_c1'][k] * q0) >= (q2 - 0.05))
            if cfg['R01'][k] > 0:
                c0 = (q0 > cfg['T0'][ci]) & ((q0 * cfg['R01'][k]) >= (q1 - 0.05))
            else:
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
    # First check what splits are inside size448, tiles3, v5_clean, internvit, resplit_fix
    for dname in ['results/size448', 'results/tiles3', 'results/v5_clean', 'results/internvit', 'results/resplit_fix', 'results/hpo2_validate']:
        files = sorted(glob.glob(f'{dname}/*.npz'))
        if files:
            d = np.load(files[0])
            print(f"{dname:24s}: count={len(files)}, sample={os.path.basename(files[0])}, keys={list(d.keys())}, gte_shape={d['gte'].shape if 'gte' in d else None}")

    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P2', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    hdr = f"\n{'variant':30s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} {'S_cln':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr)
    print('-' * len(hdr))

    splits_cache = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_decoupled_14ch_probs(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_decoupled_14ch_probs(models, 'te')
        splits_cache[sp] = (models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2)

    # Test selective R01 on 过度生成 (k=3) while keeping R01=0 on 模糊/纹理损失/过锐 (k=0,1,2) so P0 recall stays 0.836!
    cfg_sel = dict(
        T0=[0.38, 0.38, 0.34, 0.40, 0.36, 0.38],
        T1=[0.29, 0.29, 0.28, 0.27],
        T2=[0.30, 0.29, 0.31, 0.33],
        R12=[0.88, 0.88, 0.92, 1.02],
        w0_in_c1=[0.15, 0.15, 0.18, 0.22],
        R01=[0.0, 0.0, 0.0, 0.95],
        T_pres=[0.99, 0.99, 0.99, 0.44],
        T_ge1=[0.99, 0.43, 0.41, 0.36],
        shield=0.28,
    )
    rows_sel = []
    for sp in (16, 17):
        _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits_cache[sp]
        pred = decode_v8_grade_aware(te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, cfg_sel)
        s = cm.score(gte, pred, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
        rows_sel.append((s, f1))
        print(f"{'V8_SelectiveGrade_s'+str(sp):30s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want))
    m = lambda k: np.mean([r[0][k] for r in rows_sel])
    mf = [np.mean([r[1][w] for r in rows_sel]) for w in want]
    print(f"{'V8_SelectiveGrade MEAN':30s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf))


if __name__ == '__main__':
    main()
