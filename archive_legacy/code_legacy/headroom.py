"""Offline headroom analysis on saved probabilities (no GPU).

Honest protocol: decode thresholds are tuned on each split's VAL, scored on TEST.
Usage: python3 -I code/headroom.py
"""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decision_eval'))
import comp_metrics as cm
from common import CLASSES, FULL

TAU = 0.02
FULL_IDX = cm.FULL_IDX; BIN_IDX = cm.BIN_IDX


def load(f):
    p = np.load(f, allow_pickle=True)
    return {k: p[k] for k in p.files}


def severity(p0, p1, p2):
    """expected grade = s2+s1+s0 for 4-level classes, 3*p0 for binary ones."""
    n = p0.shape[0]; sev = np.zeros((n, 6))
    for j, ci in enumerate(FULL_IDX):
        s0 = p0[:, ci]; s1 = p1[:, j] + s0; s2 = p2[:, j] + s1
        sev[:, ci] = s0 + s1 + s2
    for ci in BIN_IDX:
        sev[:, ci] = 3 * p0[:, ci]
    return sev


def decode_sev(sev, u0, u1, u2):
    pred = np.where(sev > u0, 3, np.where(sev > u1, 2, np.where(sev > u2, 1, 0)))
    pred[:, BIN_IDX] = np.where(sev[:, BIN_IDX] > u0, 3, 0)
    return pred


def tune(g, sev, tau=TAU):
    best = None
    for u0 in np.arange(1.5, 2.95, 0.1):
        for u1 in np.arange(0.8, 2.3, 0.1):
            for u2 in np.arange(0.2, 1.6, 0.1):
                if not (u2 < u1 < u0):
                    continue
                s = cm.score(g, decode_sev(sev, u0, u1, u2), tau)
                if best is None or s['total'] > best[0]['total']:
                    best = (s, (u0, u1, u2))
    return best


def parts(s):
    return dict(S_pos=25 * s['S_pos'], S_p0=20 * s['S_p0'], S_clean=20 * s['S_clean'], total=s['total'])


def splits_runs():
    """final recipe runs on the untouched splits s16-s20."""
    out = {}
    for sp in range(16, 21):
        fs = sorted(glob.glob(f'results/night/4/final_s{sp}_seed*_probs.npz'))
        out[sp] = fs
    return out


def evaluate(fs, tau=TAU):
    runs = [load(f) for f in fs]
    mean = lambda key: np.mean([r[key] for r in runs], 0)
    va = severity(mean('va_p0'), mean('va_p1'), mean('va_p2'))
    te = severity(mean('te_p0'), mean('te_p1'), mean('te_p2'))
    gva = runs[0]['gva'].astype(int); gte = runs[0]['gte'].astype(int)
    s, u = tune(gva, va, tau)
    st = cm.score(gte, decode_sev(te, *u), tau)
    return st, u, gte, te


def main():
    rows = []
    for sp, fs in splits_runs().items():
        st, u, gte, te = evaluate(fs)
        p = parts(st); rows.append(p)
        nclean = int((gte == 0).all(1).sum())
        print(f's{sp} n_runs={len(fs)} clean_test={nclean} u={tuple(round(x,2) for x in u)} '
              f'S_pos={p["S_pos"]:.2f} S_p0={p["S_p0"]:.2f} S_clean={p["S_clean"]:.2f} total={p["total"]:.2f}')
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print('MEAN', {k: round(v, 2) for k, v in m.items()})

    # arithmetic: what is needed for 55
    need = 55 - m['S_clean']
    print(f'\nTo reach 55 with S_clean={m["S_clean"]:.1f}: need S_pos+S_p0 = {need:.1f} '
          f'(now {m["S_pos"]+m["S_p0"]:.1f}, gap {need-m["S_pos"]-m["S_p0"]:.1f})')
    for p0r in (0.75, 0.80, 0.85, 0.90):
        # S_p0 ~ proportional; pick S_p0 value directly
        pass
    for sp0 in (15.0, 16.0, 17.0, 18.0):
        print(f'  if S_p0={sp0:.0f}/20 -> S_pos must be {need-sp0:.1f}/25 (F1 macro {(need-sp0)/25:.3f}; now {m["S_pos"]/25:.3f})')

    # per-label F1 on pooled test predictions (s16-s20)
    from collections import defaultdict
    acc = defaultdict(lambda: [0, 0, 0])
    for sp, fs in splits_runs().items():
        st, u, gte, te = evaluate(fs)
        for c, g, P, R, F, tp, fp, fn in st['per']:
            a = acc[(c, g)]; a[0] += tp; a[1] += fp; a[2] += fn
    print('\nper-label pooled F1 (s16-s20, test):')
    f1s = []
    for (c, g), (tp, fp, fn) in acc.items():
        f1 = 2 * tp / max(1, 2 * tp + fp + fn); f1s.append(f1)
        print(f'  {c}_{g}: F1={f1:.3f} P={tp/max(1,tp+fp):.3f} R={tp/max(1,tp+fn):.3f} (tp={tp} fp={fp} fn={fn})')
    print('  macro F1 =', round(float(np.mean(f1s)), 3))


if __name__ == '__main__':
    main()
