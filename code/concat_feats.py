import sys, numpy as np
out = sys.argv[1]
blocks = []
ref = None
for p in sys.argv[2:]:
    d = np.load(p, allow_pickle=True)
    names = list(d['names'])
    if ref is None: ref = names; Xs = [d['X']]
    else:
        idx = {n:i for i,n in enumerate(names)}
        assert set(ref) == set(idx), 'name mismatch'
        Xs.append(d['X'][[idx[n] for n in ref]])
X = np.concatenate(Xs, 1)
np.savez_compressed(out, X=X.astype(np.float32), names=np.array(ref))
print(out, X.shape)
