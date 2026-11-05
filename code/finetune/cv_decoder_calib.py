"""Leave-one-split-out CV of decoder calibration on the 5 strictly-unseen OOF splits (s16-s20).
Compares: hand-set V3 decoder / global tuned / per-class coordinate ascent (+ shield).
Only adopt a variant if it wins on HELD-OUT splits. No test-set information is used.
"""
import sys, json, itertools
import numpy as np
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/finetune')
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/decision_eval')
import comp_metrics as cm
from data import CLASSES, FULL

FULL_IDX = [i for i, c in enumerate(CLASSES) if c in FULL]
TAU = 0.02


def load_oof():
    S = []
    for s in range(16, 21):
        d = np.load(f'/wanqing-develop/luowenjing/18/results/v2/v2_pcl_s{s}_seed1_probs.npz')
        S.append(tuple(np.concatenate([d['va_' + k], d['te_' + k]], 0) for k in ['p0', 'p1', 'p2']) +
                 (np.concatenate([d['gva'], d['gte']], 0),))
    return S


def decode_pc(p0, p1, p2, P):
    T0, T1, T2, R12, sh = P['T0'], P['T1'], P['T2'], P['R12'], P['shield']
    n = p0.shape[0]
    pred = np.zeros((n, 6), int)
    cum = np.zeros((n, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum[:, ci] = p0[:, ci] + p1[:, k] + p2[:, k]
            c2 = p2[:, k] > T2[k]
            c1 = (p1[:, k] > T1[k]) & ((p1[:, k] * R12[k]) >= (p2[:, k] - 0.05))
            c0 = p0[:, ci] > T0[ci]
            pr = np.where(c2, 1, 0); pr = np.where(c1, 2, pr); pr = np.where(c0, 3, pr)
            pred[:, ci] = pr
        else:
            cum[:, ci] = p0[:, ci]
            pred[:, ci] = np.where(p0[:, ci] > T0[ci], 3, 0)
    if sh > 0:
        pred[cum.max(1) < sh] = 0
    return pred


def sc(S, P):
    return float(np.mean([cm.score(g, decode_pc(a, b, c, P), TAU)['total'] for a, b, c, g in S]))


HAND = dict(T0=[0.38, 0.38, 0.34, 0.40, 0.34, 0.38], T1=[0.30] * 4, T2=[0.30, 0.29, 0.31, 0.34], R12=[0.85] * 4, shield=0.28)
GLOB = dict(T0=[0.40] * 6, T1=[0.30] * 4, T2=[0.30] * 4, R12=[0.85] * 4, shield=0.0)

GRID = dict(T0=np.arange(0.25, 0.65, 0.05), T1=np.arange(0.20, 0.55, 0.05), T2=np.arange(0.20, 0.55, 0.05),
            R12=[0.7, 0.85, 1.0, 1.2], shield=[0.0, 0.2, 0.28])


def coord_ascent(S, init, rounds=2, shrink=0.0):
    P = json.loads(json.dumps(init, default=float))
    best = sc(S, P)
    for _ in range(rounds):
        for key in ['T0', 'T1', 'T2', 'R12']:
            for i in range(len(P[key])):
                for v in GRID[key]:
                    Q = json.loads(json.dumps(P, default=float)); Q[key][i] = float(v)
                    s = sc(S, Q)
                    if s > best + 1e-9:
                        best, P = s, Q
        for v in GRID['shield']:
            Q = json.loads(json.dumps(P, default=float)); Q['shield'] = float(v)
            s = sc(S, Q)
            if s > best + 1e-9:
                best, P = s, Q
    return P, best


if __name__ == '__main__':
    S = load_oof()
    res = {'hand': [], 'glob': [], 'pc': []}
    for ho in range(5):
        tr = [S[i] for i in range(5) if i != ho]; te = [S[ho]]
        P, b = coord_ascent(tr, GLOB)
        res['hand'].append(sc(te, HAND)); res['glob'].append(sc(te, GLOB)); res['pc'].append(sc(te, P))
        print(f'fold s{16+ho}: hand={res["hand"][-1]:.2f} glob={res["glob"][-1]:.2f} perclass={res["pc"][-1]:.2f} (train={b:.2f})', flush=True)
    for k, v in res.items():
        print(k, 'held-out mean', round(float(np.mean(v)), 3))
    Pfull, b = coord_ascent(S, GLOB)
    print('FINAL per-class params (fit on all 5 splits, in-sample %.2f):' % b, json.dumps(Pfull))
    json.dump(Pfull, open('/wanqing-develop/luowenjing/18/results/v3_super/cv_calibrated_params.json', 'w'), indent=1)
