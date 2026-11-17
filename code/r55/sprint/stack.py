import numpy as np, sys, warnings; warnings.filterwarnings("ignore")
from harness import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
P = D["P"]; mem = list(P)
def feat_members(dims=None):
    X = []
    for m in mem:
        v = np.log(np.clip(P[m], 1e-5, 1))
        X.append((v if dims is None else v[:, dims]).reshape(len(v), -1))
    return np.concatenate(X, 1)
def oof_stack(X, y, model):
    out = np.zeros((len(y), 4))
    for k in range(8):
        tr, va = fold != k, fold == k
        sc = StandardScaler().fit(X[tr]); mdl = model()
        mdl.fit(sc.transform(X[tr]), y[tr]); pr = mdl.predict_proba(sc.transform(X[va]))
        full = np.zeros((va.sum(), 4)); full[:, mdl.classes_] = pr; out[va] = full
    return out
if __name__ == "__main__":
    LR = lambda C: (lambda: LogisticRegression(C=C, max_iter=2000))
    for name, X in [("own-dim 11x4", None), ("all-dims 11x24", feat_members())]:
        for C in (0.01, 0.1):
            res = {}
            for d in (2, 4):
                Xd = feat_members([d]) if X is None else X
                Qs = oof_stack(Xd, g[:, d], LR(C)); res[d] = nested_bias_decode(Qs, g[:, d], fold)
            report(f"stackLR {name} C={C} + bias", res)
    res = {}
    for d in (2, 4):
        Qs = oof_stack(feat_members(), g[:, d], lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0))
        res[d] = nested_bias_decode(Qs, g[:, d], fold)
    report("stackHGB all-dims 11x24 + bias", res)
