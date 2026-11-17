# Shared helpers for the decoding / calibration study (CPU, out-of-fold only; nothing reads test_150 or legacy JSONs).
#   fast Doc2_Strict scorer (checked against common55.score), nested runner, video-cluster bootstrap, decoders.
import os
import sys

import numpy as np

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
import decoder55 as d55  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K = "doc2_strict_total"
FULL = [0, 1, 2, 4]
P0ONLY = [3, 5]
LABELS = [(d, g) for d in FULL for g in (1, 2, 3)] + [(3, 3), (5, 3)]   # the 14 positive labels
LD = np.array([l[0] for l in LABELS]); LG = np.array([l[1] for l in LABELS])


def load():
    z = np.load(os.path.join(HERE, "dec_pool.npz"), allow_pickle=True)
    mem = {str(k): z[f"m_{k}"].astype(np.float64) for k in z["mem_keys"]}
    return dict(names=z["names"], gold=z["gold"], fold=z["fold"], grp=z["grp"], vid=z["vid"], orig=z["orig"].astype(bool),
                P=z["P"].astype(np.float64), mem=mem)


def onehot(a):
    return np.eye(4, dtype=np.int32)[a]


def counts(pred, gold):
    Po, Go = onehot(pred), onehot(gold)
    tp = (Po * Go).sum(0); npred = Po.sum(0); npos = Go.sum(0)
    clean = (gold == 0).all(1)
    return tp, npred, npos, int(clean.sum()), int(((pred == 0).all(1) & clean).sum())


def score_from_counts(tp, npred, npos, ncl, okcl):
    tp, npred, npos = tp[LD, LG].astype(float), npred[LD, LG].astype(float), npos[LD, LG].astype(float)
    fp, fn = npred - tp, npos - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    pr = tp / np.maximum(1, npred); rc = tp / np.maximum(1, npos)
    p0 = LG == 3
    s_pos = f1.mean()
    s_p0 = 0.5 * rc[p0].mean() + 0.3 * f1[p0].mean() + 0.2 * pr[p0].mean()
    s_cl = okcl / max(1, ncl) if ncl > 0 else 1.0
    return 25 * s_pos + 20 * s_p0 + 20 * s_cl, (25 * s_pos, 20 * s_p0, 20 * s_cl)


def fscore(pred, gold):
    return score_from_counts(*counts(pred, gold))[0]


def fparts(pred, gold):
    return score_from_counts(*counts(pred, gold))[1]


# make Decoder55 use the fast scorer (identical numbers, checked in dec_check.py)
d55.score = lambda p, g: {K: fscore(p, g)}


def d55_multistart(p, g, starts=(0.0, 0.3, 0.6, 0.9), cls=None, **kw):
    cls = cls or d55.Decoder55
    best = None
    for b in starts:
        d = cls(K, **kw) if kw else cls(K)
        d.bias[:, 1:] += b
        d.fit(p, g)
        o = d._obj(c.renorm(p), g)
        if best is None or o > best[0]:
            best = (o, d)
    return best[1]


def nested(fit_predict, data, P=None, rows_fit=None):
    """fit_predict(P_tr, gold_tr, P_te, tr_idx, te_idx) -> pred_te.  rows_fit: optional bool mask restricting the fit rows."""
    P = data["P"] if P is None else P
    gold, fold = data["gold"], data["fold"]
    pred = np.zeros_like(gold)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        if rows_fit is not None:
            tr = tr & rows_fit
        pred[te] = fit_predict(P[tr], gold[tr], P[te], np.where(tr)[0], np.where(te)[0])
    return pred


def cluster_boot(preds, base, data, mask=None, B=1000, seed=0):
    """Paired video-cluster bootstrap of score(pred) - score(base). Returns dict name -> (point, lo5, hi95, P(delta>0))."""
    gold, vid = data["gold"], data["vid"]
    idx_all = np.arange(len(gold)) if mask is None else np.where(mask)[0]
    uv, inv = np.unique(vid[idx_all], return_inverse=True)
    order = np.argsort(inv); bounds = np.searchsorted(inv[order], np.arange(len(uv) + 1))
    members = [idx_all[order[bounds[k]:bounds[k + 1]]] for k in range(len(uv))]
    rs = np.random.RandomState(seed)
    draws = [np.concatenate([members[k] for k in rs.randint(0, len(uv), len(uv))]) for _ in range(B)]
    out = {}
    b0 = np.array([fscore(base[ix], gold[ix]) for ix in draws])
    s0 = fscore(base[idx_all], gold[idx_all])
    for k, pr in preds.items():
        s = fscore(pr[idx_all], gold[idx_all])
        dd = np.array([fscore(pr[ix], gold[ix]) for ix in draws]) - b0
        out[k] = (s, s - s0, np.quantile(dd, 0.05), np.quantile(dd, 0.95), (dd > 0).mean())
    return out


def report(tag, preds, base_key, data, log):
    for mtag, mask in (("all rows", None), ("first batch", data["orig"])):
        res = cluster_boot(preds, preds[base_key], data, mask)
        log(f"\n[{tag}] {mtag}: score, delta vs '{base_key}' [CI90], P(delta>0)")
        for k, (s, d, lo, hi, pp) in res.items():
            pts = fparts(preds[k] if mask is None else preds[k][mask], data["gold"] if mask is None else data["gold"][mask])
            log(f"  {k:42s} {s:6.2f}  {d:+.2f} [{lo:+.2f}, {hi:+.2f}]  {pp:4.2f}   (pos {pts[0]:5.2f} p0 {pts[1]:5.2f} cl {pts[2]:5.2f})")


def logger(path):
    fh = open(path, "w")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        fh.write(s + "\n"); fh.flush()
    return log


# ------------------------------------------------------------------ calibration helpers
def nll_dim(p, y):
    return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, None)))


def temp_scale(p, T):
    """p (N,4) simplex -> tempered simplex p^(1/T) renormalised (temperature on log-probs)."""
    lg = np.log(np.clip(p, 1e-9, None)) / T
    lg -= lg.max(-1, keepdims=True)
    q = np.exp(lg)
    return q / q.sum(-1, keepdims=True)


T_GRID = np.exp(np.linspace(np.log(0.4), np.log(3.0), 41))


def fit_T(p, y, dimidx):
    """best temperature for one dimension by NLL (grid)."""
    pp = p.copy()
    if dimidx in P0ONLY:   # binary dims: only columns 0 and 3 carry mass
        pp = pp[:, [0, 3]]; y = (y == 3).astype(int)
    return T_GRID[int(np.argmin([nll_dim(temp_scale(pp, T), y) for T in T_GRID]))]


def apply_T(P, T):
    Q = P.copy()
    for d in range(6):
        if d in P0ONLY:
            q = temp_scale(P[:, d][:, [0, 3]], T[d])
            Q[:, d, 0], Q[:, d, 3] = q[:, 0], q[:, 1]
        else:
            Q[:, d] = temp_scale(P[:, d], T[d])
    return c.renorm(Q)
