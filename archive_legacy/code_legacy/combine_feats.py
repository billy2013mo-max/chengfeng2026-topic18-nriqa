"""Merge original + supplemental feature files in combined-manifest order."""
import csv, sys, numpy as np
orig, supp, man, out = sys.argv[1:5]
def load(p):
    d = np.load(p, allow_pickle=True); return {n: i for i, n in enumerate(d['names'])}, d['X']
io, Xo = load(orig); isu, Xs = load(supp)
names = [r['name'] for r in csv.DictReader(open(man))]
X = np.stack([Xo[io[n]] if n in io else Xs[isu[n]] for n in names])
np.savez_compressed(out, X=X.astype(np.float32), names=np.array(names))
print(out, X.shape, 'missing', sum(1 for n in names if n not in io and n not in isu))
