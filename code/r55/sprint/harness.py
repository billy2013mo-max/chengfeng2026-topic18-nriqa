"""Sprint harness: 过锐 (dim 2) / 过度生成 (dim 4) only. Nested group 8-fold.
All decision params fitted on training folds only."""
import pickle, numpy as np, itertools
OUT = "/wanqing-develop/luowenjing/18/results/r55/sprint_os_og"
D = pickle.load(open(f"{OUT}/v7_oof.pkl", "rb"))
g, fold, io, groups, names = D["gold"], D["fold"], D["is_orig"], D["groups"], D["names"]
TD = {2: "过锐", 4: "过度生成"}
LV = {3: "P0", 2: "P1", 1: "P2"}

def f1s(y, yp):
    out = []
    for lv in (3, 2, 1):
        a = y == lv; b = yp == lv; tp = (a & b).sum()
        out.append(2 * tp / max(1, a.sum() + b.sum()))
    return np.array(out)

BGRID = np.linspace(-3, 3, 25)
def fit_bias(logq, y, iters=3):
    """maximize sum F1(P0,P1,P2) via argmax(logq + b), coordinate ascent; b[0]=0"""
    b = np.zeros(4)
    best = f1s(y, (logq + b).argmax(1)).sum()
    for _ in range(iters):
        for c in (1, 2, 3):
            for v in BGRID:
                bb = b.copy(); bb[c] = v
                s = f1s(y, (logq + bb).argmax(1)).sum()
                if s > best + 1e-9: best, b = s, bb
    return b

def nested_bias_decode(Qd, y, fold):
    """Qd: (N,4) probs for one dim, already OOF. returns preds"""
    lq = np.log(np.clip(Qd, 1e-6, 1)); pred = np.zeros(len(y), int)
    for k in range(8):
        tr, va = fold != k, fold == k
        b = fit_bias(lq[tr], y[tr]); pred[va] = (lq[va] + b).argmax(1)
    return pred

def report(tag, preds_by_dim, file=None):
    lines = [f"### {tag}"]
    for d, pr in preds_by_dim.items():
        y = g[:, d]
        a = f1s(y, pr); o = f1s(y[io], pr[io]); s = f1s(y[~io], pr[~io])
        lines.append(f"  {TD[d]:4s} ALL P0/P1/P2 {a[0]:.3f}/{a[1]:.3f}/{a[2]:.3f} mean {a.mean():.3f} | ORIG {o[0]:.3f}/{o[1]:.3f}/{o[2]:.3f} | SUPP {s[0]:.3f}/{s[1]:.3f}/{s[2]:.3f}")
    txt = "\n".join(lines); print(txt, flush=True)
    with open(f"{OUT}/LEDGER.md", "a") as f: f.write(txt + "\n")
    return txt
