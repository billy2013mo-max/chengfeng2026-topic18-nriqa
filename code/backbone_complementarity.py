"""Do the frozen backbones specialise on non-overlapping cases?

Uses the 1902-image OOF probe predictions (5-fold GroupKFold, identical folds
for every backbone) so every model is scored on exactly the same images with
the same split.

Outputs
  1. per-label AP matrix and the best backbone per label
  2. per-scenario (resolution / photo-vs-graphic) AP matrix
  3. error-set overlap between backbones and an oracle-union F1 upper bound
"""
import csv, itertools, os, sys
import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'decision_eval'))
CL = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
FULL = ['模糊', '纹理损失', '过锐', '过度生成']
LBL = [(i, c, g) for i, c in enumerate(CL)
       for g in (['P2', 'P1', 'P0'] if c in FULL else ['P0'])]

MODELS = ['siglip2-so400m', 'siglip2-giant', 'siglip2-naflex', 'c-radio-v4-h',
          'eupe-vit-b', 'internvit-6b', 'pe-core-l14', 'pe-core-g14',
          'metaclip2-h14', 'convnextv2-base', 'dinov2-large', 'eva02-base',
          'clip-vit-large', 'qwen3vl-2b']


def cum_y(grades):
    y, names = [], []
    for ci, c in enumerate(CL):
        if c in FULL:
            for need, nm in [(1, 'P2'), (2, 'P1'), (3, 'P0')]:
                y.append((grades[:, ci] >= need).astype(int)); names.append(f'{c}_{nm}')
        else:
            y.append((grades[:, ci] == 3).astype(int)); names.append(f'{c}_P0')
    return np.stack(y, 1), names


def load(model, man):
    f = f'feats/{model}_oof.npz'
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    feat = np.load(f'feats/{model}.npz', allow_pickle=True)
    names = list(feat['names'])
    P = d['P']; grades = d['grades']; pnames = [str(x) for x in d['names']]
    # P columns are cumulative in pnames order
    Y, ynames = cum_y(grades)
    assert ynames == pnames, (model, pnames[:3], ynames[:3])
    return dict(model=model, names=names, P=P, grades=grades, Y=Y, ln=ynames)


def best_f1_thr(y, s):
    order = np.argsort(-s); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    P = len(y); T = y.sum()
    f1 = 2 * tp / np.maximum(tp + fp + T, 1e-9)
    i = int(np.argmax(f1))
    return float(s[order][i]), float(f1[i])


def main():
    man = {r['name']: r for r in csv.DictReader(open('data/manifest.csv'))}
    gs = np.load('results/graphic_score.npz', allow_pickle=True)
    gmap = dict(zip([str(x) for x in gs['names']], gs['score']))

    ds = {}
    for m in MODELS:
        d = load(m, man)
        if d:
            ds[m] = d
    ref = ds['siglip2-so400m']
    names = ref['names']
    Y = ref['Y']; ln = ref['ln']
    n = len(names)

    def resb(nm):
        r = man[nm]; x = min(int(r['width']), int(r['height']))
        return 'lt720' if x < 720 else ('720-1080' if x < 1080 else ('1080-2160' if x < 2160 else 'ge2160'))

    rbk = np.array([resb(x) for x in names])
    gsc = np.array([gmap.get(x, np.nan) for x in names])
    gq = np.nanpercentile(gsc, 80)
    scen = {'all': np.ones(n, bool)}
    for b in ['lt720', '720-1080', '1080-2160', 'ge2160']:
        scen[b] = rbk == b
    scen['photo(80%)'] = gsc <= gq
    scen['graphic(top20%)'] = gsc > gq

    # 1. per-label AP
    AP = {}
    for m, d in ds.items():
        assert d['names'] == names
        AP[m] = np.array([average_precision_score(Y[:, j], d['P'][:, j]) for j in range(14)])
    print('== per-label AP ==')
    print('label'.ljust(14) + ''.join(m[:9].rjust(10) for m in ds) + '   best')
    for j, nm in enumerate(ln):
        vals = {m: AP[m][j] for m in ds}
        b = max(vals, key=vals.get)
        print(nm.ljust(14) + ''.join(f'{vals[m]:10.3f}' for m in ds) + f'   {b} ({vals[b]:.3f})')
    print('mAP14'.ljust(14) + ''.join(f'{AP[m].mean():10.3f}' for m in ds))
    p0j = [j for j, x in enumerate(ln) if x.endswith('P0')]
    print('mAP_P0'.ljust(14) + ''.join(f'{AP[m][p0j].mean():10.3f}' for m in ds))

    # 2. per-scenario AP
    print('\n== per-scenario mAP14 ==')
    hdr = ''.join(k.rjust(16) for k in scen)
    print('model'.ljust(16) + hdr)
    SC = {}
    for m, d in ds.items():
        row = []
        SC[m] = {}
        for k, mask in scen.items():
            aps = [average_precision_score(Y[mask, j], d['P'][mask, j]) if 0 < Y[mask, j].sum() < mask.sum() else np.nan
                   for j in range(14)]
            SC[m][k] = np.nanmean(aps); row.append(SC[m][k])
        print(m[:15].ljust(16) + ''.join(f'{v:16.3f}' for v in row))
    for k in scen:
        b = max(ds, key=lambda m: SC[m][k])
        print(f'  best on {k}: {b} ({SC[b][k]:.3f})')

    # 3. error overlap at per-label best-F1 operating point
    print('\n== pairwise complementarity (best-F1 operating point) ==')
    tp = {}; fp = {}
    for m, d in ds.items():
        TM = np.zeros(Y.shape, bool); FM = np.zeros(Y.shape, bool)
        for j in range(14):
            th, _ = best_f1_thr(Y[:, j], d['P'][:, j])
            pred = d['P'][:, j] >= th
            TM[:, j] = pred & (Y[:, j] == 1); FM[:, j] = pred & (Y[:, j] == 0)
        tp[m] = TM; fp[m] = FM

    def f1_from(TM, FM):
        out = []
        for j in range(14):
            t = TM[:, j].sum(); p = t + FM[:, j].sum(); r = t / max(1, Y[:, j].sum())
            out.append(2 * t / max(1, 2 * t + FM[:, j].sum() + (Y[:, j].sum() - t)))
        return np.array(out)

    single = {m: f1_from(tp[m], fp[m]) for m in ds}
    print('model'.ljust(16) + 'macroF1   uniqueTP%(vs ensemble)')
    for m in ds:
        others = [o for o in ds if o != m]
        utp = []
        for j in range(14):
            oth = np.zeros(n, bool)
            for o in others:
                oth |= tp[o][:, j]
            utp.append((tp[m][:, j] & ~oth).sum() / max(1, tp[m][:, j].sum()))
        print(m[:15].ljust(16) + f'{single[m].mean():8.3f}   {100*np.mean(utp):6.1f}%')

    best = np.zeros(14); oracle_tp = np.zeros(Y.shape, bool); oracle_fp = np.zeros(Y.shape, bool)
    for j in range(14):
        bm = max(ds, key=lambda m: single[m][j]); best[j] = single[bm][j]
        oracle_tp[:, j] = tp[bm][:, j]; oracle_fp[:, j] = fp[bm][:, j]
    union_tp = np.zeros(Y.shape, bool); union_fp = np.zeros(Y.shape, bool)
    for j in range(14):
        for m in ds:
            union_tp[:, j] |= tp[m][:, j]; union_fp[:, j] |= fp[m][:, j]
    print(f'\nbest single model macro-F1 : {max(single[m].mean() for m in ds):.3f} '
          f'({max(ds, key=lambda m: single[m].mean())})')
    print(f'oracle per-label routing  : {best.mean():.3f}  (+{best.mean()-max(single[m].mean() for m in ds):.3f})')
    print(f'oracle union of TP        : {f1_from(union_tp, union_fp).mean():.3f} (upper bound, unrealisable)')

    # 4. is routing learnable? per-scenario best model vs global best
    print('\n== per-scenario best backbone (oracle routing by scenario) ==')
    glob = max(ds, key=lambda m: SC[m]['all'])
    for k, mask in scen.items():
        b = max(ds, key=lambda m: SC[m][k])
        print(f'{k:16s} n={mask.sum():4d}  best={b:15s} mAP={SC[b][k]:.3f}  global-best={SC[glob][k]:.3f}')


if __name__ == '__main__':
    main()
