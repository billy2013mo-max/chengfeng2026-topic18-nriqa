"""Diagnostics: (1) source-video leakage in our CV splits, (2) near-duplicate leakage,
(3) bootstrap noise of the official score at n=150 / n=654, (4) per-label support."""
import os, sys, csv, glob, json
from collections import Counter, defaultdict
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/decision_eval')
import comp_metrics as cm

CL = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
man = {}
for r in csv.DictReader(open('data/manifest_combined.csv', encoding='utf-8')):
    man[r['name'].strip()] = r
src = {k: v['source'] for k, v in man.items()}
print(f'manifest images: {len(man)}, unique sources: {len(set(src.values()))}')
per_src = Counter(src.values())
print('images per source: ', Counter(per_src.values()).most_common(10))


def split_leak(path):
    tr, te = set(), []
    for r in csv.DictReader(open(path, encoding='utf-8')):
        n = r['name'].strip()
        if n not in src:
            continue
        if r['split'] == 'train':
            tr.add(src[n])
        elif r['split'] in ('test', 'val'):
            te.append(n)
    leak = sum(1 for n in te if src[n] in tr)
    return leak, len(te)


print('\n=== (1) Source-video leakage (eval image whose source video also appears in train) ===')
for p in sorted(glob.glob('data/v9_fulldata/split_f*.csv')) + sorted(glob.glob('data/resplits/*s16*.csv')) + sorted(glob.glob('data/resplits/*s17*.csv')):
    try:
        l, n = split_leak(p)
        print(f'  {p:55s}: {l}/{n} = {l/max(1,n):.1%}')
    except Exception as e:
        print('  skip', p, e)

print('\n=== (2) Near-duplicate check via C-RADIO embeddings ===')
fe = None
for cand in ['feats/c-radio-v4-h_combined.npz', 'feats/c-radio-v4-h.npz']:
    if os.path.exists(cand):
        d = np.load(cand, allow_pickle=True)
        print('  keys', cand, d.files)
        X = None
        for k in ['X', 'feats', 'F', 'emb']:
            if k in d.files:
                X = d[k]
        nm = None
        for k in ['names', 'name', 'images']:
            if k in d.files and len(d[k]) == (len(X) if X is not None else -1):
                nm = [str(x) for x in d[k]]
        if X is not None and nm is not None:
            fe = (cand, X.reshape(len(X), -1).astype(np.float32), nm)
            break
if fe:
    cand, X, nm = fe
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    S = X @ X.T
    np.fill_diagonal(S, -1)
    mx = S.max(1)
    arg = S.argmax(1)
    for th in (0.98, 0.95, 0.92, 0.90):
        idx = np.where(mx > th)[0]
        diff_src = sum(1 for i in idx if src.get(nm[i]) != src.get(nm[arg[i]]))
        print(f'  cos>{th}: {len(idx)} images have a near-twin ({diff_src} of them with a DIFFERENT source id)')
    # label agreement between near-twins
    idx = np.where(mx > 0.95)[0]
    agree = []
    for i in idx:
        a, b = man.get(nm[i]), man.get(nm[arg[i]])
        if a and b:
            agree.append(np.mean([a[c] == b[c] for c in CL]))
    if agree:
        print(f'  near-twin (cos>0.95) per-class label agreement: {np.mean(agree):.3f}  -> label noise ceiling hint')
else:
    print('  no usable embedding file found')

print('\n=== (3) Bootstrap noise of official score (doc2 S_clean) on stitched 8-fold OOF ===')
P0, P1, P2, G = [], [], [], []
for f in sorted(glob.glob('results/v9_fulldata/V9_Full_F*_probs.npz')):
    d = np.load(f, allow_pickle=True)
    P0.append(d['te_p0']); P1.append(d['te_p1']); P2.append(d['te_p2']); G.append(d['gte'])
P0, P1, P2, G = map(np.concatenate, (P0, P1, P2, G))
pred = cm.decode(P0, P1, P2, 0.42, 0.32, 0.28)


def doc2(g, p):
    s = cm.score(g, p, 0.02)
    cl = (g == 0).all(1)
    sc = float(((p == 0).all(1) & cl).sum() / max(1, cl.sum()))
    return 25 * s['S_pos'] + 20 * s['S_p0'] + 20 * sc, s['S_pos'], s['S_p0'], sc


full = doc2(G, pred)
print(f'  full OOF n={len(G)}: total={full[0]:.2f} S_pos={full[1]:.3f} S_p0={full[2]:.3f} S_clean_doc2={full[3]:.3f}')
rng = np.random.default_rng(0)
for n in (150, 654, 1000):
    tots = []
    for _ in range(400):
        ii = rng.choice(len(G), n, replace=False)
        tots.append(doc2(G[ii], pred[ii])[0])
    tots = np.array(tots)
    print(f'  n={n:4d}: mean={tots.mean():.2f}  std={tots.std():.2f}  90% range=[{np.percentile(tots,5):.2f}, {np.percentile(tots,95):.2f}]')

# paired noise: two predictors differing slightly -> how often does a truly-better one lose at n=150?
pred_b = cm.decode(P0, P1, P2, 0.40, 0.30, 0.26)
db = doc2(G, pred_b)[0] - full[0]
wins = []
for _ in range(400):
    ii = rng.choice(len(G), 150, replace=False)
    wins.append(doc2(G[ii], pred_b[ii])[0] - doc2(G[ii], pred[ii])[0])
wins = np.array(wins)
print(f'  paired diff (true full-set delta={db:+.2f}): at n=150 std of delta={wins.std():.2f}, sign flips {np.mean(np.sign(wins)!=np.sign(db)):.0%}')

print('\n=== (4) Per-label support & OOF F1 (where the macro points are) ===')
s = cm.score(G, pred, 0.02)
for c, g, P, R, F, tp, fp, fn in s['per']:
    print(f'  {c}_{g:3s}: support={tp+fn:4d}  F1={F:.3f}  (each label = {25/14:.2f} pts of S_pos)')
cl = (G == 0).all(1)
print(f'  all-clean images: {cl.sum()} / {len(G)}; clean images with any pred defect: {((pred>0).any(1)&cl).sum()}')
