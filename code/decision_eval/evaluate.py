"""Evaluate decision-model JSONL outputs with competition-aligned metrics."""
import argparse, json, collections, os, sys
import numpy as np
from sklearn.metrics import average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from common import CLASSES, FULL

LET = 'ABCD'

def prob_of(r, grade):
    o = r['options']
    if grade not in o: return 0.0
    return float(r['probs'].get(LET[o.index(grade)], 0.0))

def pred_grade(r, t0=None):
    o = r['options']
    if t0 is not None and 'P0' in o and prob_of(r, 'P0') > t0:
        return 'P0'
    best, bg = -1.0, o[0]
    for i, g in enumerate(o):
        if t0 is not None and g == 'P0':
            continue
        p = float(r['probs'].get(LET[i], 0.0))
        if p > best: best, bg = p, g
    return bg

def load(path):
    by = collections.defaultdict(dict)
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        by[r['image']][r['defect']] = r
    imgs = [i for i, d in by.items() if len(d) == 6]
    return rows, by, imgs

def metrics(path, tau=0.02):
    rows, by, imgs = load(path)
    if len(imgs) < 50:
        return {'n_images': len(imgs), 'error': 'too_few_images'}
    valid = float(np.mean([r['valid'] for r in rows]))
    names, P, Y = [], [], []
    for c in CLASSES:
        glist = ['P2','P1','P0'] if c in FULL else ['P0']
        for g in glist:
            names.append(f'{c}_{g}')
            P.append(np.array([prob_of(by[i][c], g) for i in imgs]))
            Y.append(np.array([int(by[i][c]['gold'] == g) for i in imgs]))
    P = np.stack(P,1); Y = np.stack(Y,1)
    with np.errstate(all='ignore'):
        ap = [average_precision_score(Y[:,j], P[:,j]) if Y[:,j].sum() > 0 else float('nan') for j in range(len(names))]
    mAP14 = float(np.nanmean(ap))
    mAP_P0 = float(np.nanmean([ap[j] for j,n in enumerate(names) if n.endswith('P0')]))
    per_p0 = {names[j]: (None if np.isnan(ap[j]) else round(float(ap[j]),3))
              for j,n in enumerate(names) if n.endswith('P0')}
    p0score = np.array([max(prob_of(by[i][c], 'P0') for c in CLASSES) for i in imgs])
    clean = np.array([all(by[i][c]['gold'] == '无问题' for c in CLASSES) for i in imgs])
    out = {'n_images': len(imgs), 'valid': round(valid,3), 'mAP14': round(mAP14,3),
           'mAP_P0': round(mAP_P0,3), 'per_p0_ap': per_p0}
    for tgt in (0.01, 0.02):
        t0 = float(np.quantile(p0score[clean], 1-tgt)) if clean.sum() else 0.5
        f1s, recs, precs = [], [], []
        for c in CLASSES:
            for g in (['P2','P1','P0'] if c in FULL else ['P0']):
                yt = np.array([int(by[i][c]['gold'] == g) for i in imgs], dtype=int)
                yp = np.array([int(pred_grade(by[i][c], t0) == g) for i in imgs], dtype=int)
                tp=int((yt&yp).sum()); fp=int(((1-yt)&yp).sum()); fn=int((yt&(1-yp)).sum())
                f1s.append(2*tp/max(1,2*tp+fp+fn))
                if g == 'P0':
                    recs.append(tp/max(1,tp+fn)); precs.append(tp/max(1,tp+fp))
        fpr = float((p0score[clean] > t0).mean()) if clean.sum() else 0.0
        out[f'FPR<={tgt:.0%}'] = {'S_pos':round(float(np.mean(f1s)),3),
                                  'P0R':round(float(np.mean(recs)),3),
                                  'P0P':round(float(np.mean(precs)),3),
                                  'FPR':round(fpr,4),'t0':round(t0,3)}
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    for f in ap.parse_args().files:
        print('==', os.path.basename(f))
        print(json.dumps(metrics(f), ensure_ascii=False))
