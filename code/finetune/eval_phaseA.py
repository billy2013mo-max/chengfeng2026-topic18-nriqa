#!/usr/bin/env python3
"""
Offline evaluation of the Phase A decoder upgrades on the saved s16~s20 OOF/test
probabilities (results/phaseA/phaseA_s*.npz).

Usage:
    python3 -u code/finetune/eval_phaseA.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'decision_eval'))

from utils import competition_score, FULL_DIMS, P0_ONLY_DIMS, ch14_to_p012
from build_55plus_ultimate_submission import hierarchical_rolldown_decode, T150_TAU_ANY
from phaseA_decoder import (
    decode_competition_ratio, sinkhorn_prior_shift_calibration,
    mutually_exclusive_6x4_to_14ch, eum_bca_joint_decode,
)

SPLITS = ['s16', 's17', 's18', 's19', 's20']
DATA_DIR = os.environ.get('PHASEA_DIR', 'results/phaseA')


def load_split(s):
    d = np.load(os.path.join(DATA_DIR, f'phaseA_{s}.npz'), allow_pickle=True)
    return {k: d[k] for k in d.files}


def pooled(scores, key='total'):
    return float(np.mean(scores)) if len(scores) == 1 else float(np.sum(scores))


def p1_counts(pred):
    return int((pred[:, FULL_DIMS] == 2).sum())


def evaluate(tag, decode_fn, splits):
    preds, golds = [], []
    per = []
    for s in splits:
        p = decode_fn(splits[s])
        g = splits[s]['gold']
        sc = competition_score(p, g)
        per.append(sc['total'])
        preds.append(p)
        golds.append(g)
    P = np.vstack(preds)
    G = np.vstack(golds)
    sc = competition_score(P, G)
    print(f"{tag:44s} | pooled={sc['total']:7.3f} | S_pos={25*sc['S_pos']:6.2f} | "
          f"S_p0={20*sc['S_p0']:6.2f} | S_clean={20*sc['S_clean']:6.2f} | "
          f"P1={p1_counts(P):5d} | per-split={['%.2f' % x for x in per]}")
    return sc['total'], P


def main():
    splits = {s: load_split(s) for s in SPLITS}

    print("=" * 130)
    print("Phase A decoder comparison (s16~s20, N=5000)")
    print("=" * 130)

    # 1) Current decoder.
    evaluate('current hierarchical_rolldown', lambda d: hierarchical_rolldown_decode(
        d['test_probs'], d['t0'], d['t1'], d['t2'], T150_TAU_ANY, roll_p2=False), splits)

    # 2) Competition ratio grid (global kappa).
    best = (None, -1)
    for k in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        sc = []
        for s in SPLITS:
            d = splits[s]
            pred = decode_competition_ratio(
                d['test_probs'], d['t0'], d['t1'], d['t2'], T150_TAU_ANY,
                kappa_p1p2=np.full(4, k), base_clean=d['base_clean'], roll_p2=False)
            sc.append(competition_score(pred, d['gold'])['total'])
        mean = float(np.mean(sc))
        print(f"  kappa={k:.1f} -> mean split total={mean:.3f} ({['%.2f' % x for x in sc]})")
        if mean > best[1]:
            best = (k, mean)
    kappa = best[0]
    print(f"  best kappa={kappa} (mean split {best[1]:.3f})")

    evaluate(f'competition ratio (kappa={kappa})', lambda d: decode_competition_ratio(
        d['test_probs'], d['t0'], d['t1'], d['t2'], T150_TAU_ANY,
        kappa_p1p2=np.full(4, kappa), base_clean=d['base_clean'], roll_p2=False), splits)

    # 3) Sinkhorn + competition.
    for strength in [0.3, 0.5, 0.65, 0.8]:
        def fn(d, strength=strength):
            P = sinkhorn_prior_shift_calibration(
                _to_6x4(d['test_probs']), clean_mask=d['base_clean'], strength=strength)
            probs14 = mutually_exclusive_6x4_to_14ch(P)
            return decode_competition_ratio(
                probs14, d['t0'], d['t1'], d['t2'], T150_TAU_ANY,
                kappa_p1p2=np.full(4, kappa), base_clean=d['base_clean'], roll_p2=False)
        evaluate(f'sinkhorn strength={strength} + comp', fn, splits)

    # 4) EUM-BCA on a capped set of cells (slow, so cap).
    def eum_fn(d):
        return eum_bca_joint_decode(
            d['test_probs'], d['t0'], d['t1'], d['t2'], T150_TAU_ANY,
            kappa_p1p2=np.full(4, kappa), base_clean=d['base_clean'],
            n_rounds=1, max_cells=200)
    evaluate('competition + EUM-BCA (cap 200/round)', eum_fn, splits)


def _to_6x4(probs14):
    from phaseA_decoder import probs_14ch_to_mutually_exclusive_6x4
    return probs_14ch_to_mutually_exclusive_6x4(probs14)


if __name__ == '__main__':
    main()
