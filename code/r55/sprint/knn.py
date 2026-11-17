import os; os.environ["OMP_NUM_THREADS"]="8"; os.environ["OPENBLAS_NUM_THREADS"]="8"; os.environ["MKL_NUM_THREADS"]="8"
import numpy as np, sys, pickle, torch
torch.set_num_threads(8)
from harness import *
from probe import load, ordm
def knn_probs(X, K=30, T=0.05):
    X = torch.tensor(X); X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = torch.nn.functional.normalize(X, dim=1)
    out = {d: np.zeros((len(names), 4)) for d in (2, 4)}
    for k in range(8):
        tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
        S = X[va] @ X[tr].T; v, i = S.topk(K, dim=1); w = torch.softmax(v / T, 1).numpy(); i = i.numpy()
        for d in (2, 4):
            Y = np.eye(4)[g[tr, d]]; out[d][va] = (w[:, :, None] * Y[i]).sum(1) * 0.9 + 0.1 * Y.mean(0)
    return out
if __name__ == "__main__":
    res = {}
    for f in sys.argv[1:]:
        X, ok = load(f); assert ok.all()
        o = knn_probs(X); res[f] = o
        print(f"{os.path.basename(f):32s}", {d: np.round(ordm(o[d], g[:, d]), 3).tolist() for d in (2, 4)}, flush=True)
    pickle.dump(res, open(f"{OUT}/knn_feats.pkl", "wb"))
