import numpy as np, sys, os, pickle, warnings; warnings.filterwarnings("ignore")
from harness import *
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from joblib import Parallel, delayed
R = "/wanqing-develop/luowenjing/18/"
idx = {n: i for i, n in enumerate(names)}
def load(path):
    z = np.load(R + path, allow_pickle=True); X = z["X"].astype(np.float32)
    out = np.zeros((len(names), X.shape[1]), np.float32); ok = np.zeros(len(names), bool)
    for n, r in zip(z["names"], X):
        if str(n) in idx: out[idx[str(n)]] = r; ok[idx[str(n)]] = True
    return out, ok
def one_fold(X, k, C, npc):
    tr, va = fold != k, fold == k
    sc = StandardScaler().fit(X[tr]); Xt, Xv = sc.transform(X[tr]), sc.transform(X[va])
    if npc and X.shape[1] > npc:
        p = PCA(npc, random_state=0).fit(Xt); Xt, Xv = p.transform(Xt), p.transform(Xv)
    res = {}
    for d in (2, 4):
        m = LogisticRegression(C=C, max_iter=3000).fit(Xt, g[tr, d]); pr = np.zeros((va.sum(), 4)); pr[:, m.classes_] = m.predict_proba(Xv); res[d] = pr
    return k, res
def probe(X, C=0.01, npc=512):
    out = {d: np.zeros((len(names), 4)) for d in (2, 4)}
    for k, res in Parallel(n_jobs=8)(delayed(one_fold)(X, k, C, npc) for k in range(8)):
        for d in (2, 4): out[d][fold == k] = res[d]
    return out
def ordm(Qd, y):
    s = (Qd * np.arange(4)).sum(1); r = []
    for a, b in [(1, 2), (2, 3)]:
        m = (y == a) | (y == b); r.append(AUC(y[m] == b, s[m]))
    return [AP(y == lv, Qd[:, lv]) for lv in (3, 2, 1)] + r
if __name__ == "__main__":
    Q = D["Q"]
    print("POOL v7      ", {d: np.round(ordm(Q[:, d], g[:, d]), 3).tolist() for d in (2, 4)}, flush=True)
    feats = sys.argv[1:]
    store = {}
    for f in feats:
        X, ok = load(f)
        if not ok.all(): print(f, "coverage", ok.sum()); continue
        o = probe(X); store[f] = o
        print(f"{os.path.basename(f):32s}", {d: np.round(ordm(o[d], g[:, d]), 3).tolist() for d in (2, 4)}, flush=True)
    pickle.dump(store, open(f"{OUT}/probe_{len(feats)}_{os.getpid()}.pkl", "wb"))
