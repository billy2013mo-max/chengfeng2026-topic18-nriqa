# Calibration / pooling variants for the fallback members, each decoded by the shipped rule (Decoder55, 4 starts).
# Everything fitted per outer fold on the seven training folds (temperatures, weights, rank maps), then the decoder is
# fitted on the transformed training-fold probabilities and applied to the held-out fold.  CPU only.
# Usage: python dec_exp_pool.py
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

import dec_lib as L

D = L.load()
GOLD, FOLD = D["gold"], D["fold"]
MEM = D["mem"]
KEYS = list(MEM)                                   # cradio pe sig v16 p3
FAM = {"cradio": 0, "pe": 0, "sig": 0, "v16": 1, "p3": 2}
W_FAM = np.array([1.0 / (3 * sum(FAM[x] == FAM[k] for x in KEYS)) for k in KEYS])
STACK = np.stack([L.c.renorm(MEM[k]) for k in KEYS])          # (M,N,6,4)
c = L.c


def arith(S, w):
    return c.renorm(np.tensordot(w / w.sum(), S, 1))


def geo(S, w):
    lg = np.tensordot(w / w.sum(), np.log(np.clip(S, 1e-6, 1)), 1)
    lg -= lg.max(-1, keepdims=True)
    return c.renorm(np.exp(lg))


def nll_all(P, g):
    return np.mean([L.nll_dim(P[:, d][:, [0, 3]] / P[:, d][:, [0, 3]].sum(1, keepdims=True), (g[:, d] == 3).astype(int))
                    if d in L.P0ONLY else L.nll_dim(P[:, d], g[:, d]) for d in range(6)])


def ece(P, g, bins=15):
    """top-label ECE over all (row, dim) pairs."""
    conf = P.max(-1).ravel(); acc = (P.argmax(-1) == g).ravel()
    e = 0.0
    for a, b in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - acc[m].mean())
    return e


# ------------------------------------------------------------------ transforms fitted on training rows
def t_member_temp(tr):
    Ts = [np.array([L.fit_T(STACK[m][tr][:, d], GOLD[tr][:, d], d) for d in range(6)]) for m in range(len(KEYS))]
    S = np.stack([L.apply_T(STACK[m], Ts[m]) for m in range(len(KEYS))])
    return arith(S, W_FAM)


def t_pool_temp(tr):
    P = arith(STACK, W_FAM)
    T = np.array([L.fit_T(P[tr][:, d], GOLD[tr][:, d], d) for d in range(6)])
    return L.apply_T(P, T)


def t_geo_temp(tr):
    P = geo(STACK, W_FAM)
    T = np.array([L.fit_T(P[tr][:, d], GOLD[tr][:, d], d) for d in range(6)])
    return L.apply_T(P, T)


def dim_weights(tr, shrink):
    """per-dimension member weights minimising pooled NLL on the training rows (softmax-parametrised), shrunk toward
    the family-balanced prior: w = (1-shrink) w_nll + shrink w_fam."""
    W = np.zeros((6, len(KEYS)))
    for d in range(6):
        S = STACK[:, tr, d]                                   # (M,n,4)
        y = GOLD[tr, d]

        def f(z):
            w = np.exp(z - z.max()); w /= w.sum()
            p = np.tensordot(w, S, 1)
            return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, None)))
        z = minimize(f, np.log(W_FAM), method="Nelder-Mead", options=dict(maxiter=600, xatol=1e-3, fatol=1e-6)).x
        w = np.exp(z - z.max()); w /= w.sum()
        W[d] = (1 - shrink) * w + shrink * W_FAM / W_FAM.sum()
    return W


def t_dim_weights(shrink):
    def f(tr):
        W = dim_weights(tr, shrink)
        P = np.zeros(STACK.shape[1:])
        for d in range(6):
            P[:, d] = np.tensordot(W[d], STACK[:, :, d], 1)
        return c.renorm(P)
    return f


def t_rank(tr):
    """rank-average blending: each member's P(grade) is replaced by its empirical CDF among the training rows; ranks are
    averaged with the family weights and mapped back to a probability by quantile matching onto the family-balanced
    arithmetic pool of the training rows (per dimension and grade); then renormalised."""
    P0 = arith(STACK, W_FAM)
    Q = P0.copy()
    w = W_FAM / W_FAM.sum()
    for d in range(6):
        for k in ((1, 2, 3) if d in L.FULL else (3,)):
            R = np.zeros(STACK.shape[1])
            for m in range(len(KEYS)):
                ref = np.sort(STACK[m, tr, d, k])
                R += w[m] * np.searchsorted(ref, STACK[m, :, d, k], side="right") / len(ref)
            rtr = np.sort(R[tr]); ptr = np.sort(P0[tr, d, k])
            q = np.searchsorted(rtr, R, side="right") / len(rtr)
            Q[:, d, k] = np.quantile(ptr, np.clip(q, 0, 1))
        Q[:, d, 0] = np.clip(1 - Q[:, d, 1:].sum(-1), 1e-6, None)
    return c.renorm(Q)


VARIANTS = {
    "arith family-balanced (shipped)": lambda tr: arith(STACK, W_FAM),
    "arith equal member weights": lambda tr: arith(STACK, np.ones(len(KEYS))),
    "geometric family-balanced": lambda tr: geo(STACK, W_FAM),
    "per-member per-dim temperature, arith": t_member_temp,
    "pooled per-dim temperature": t_pool_temp,
    "geometric + pooled temperature": t_geo_temp,
    "per-dim NLL weights, shrink 0.5": t_dim_weights(0.5),
    "per-dim NLL weights, no shrink": t_dim_weights(0.0),
    "rank-average (quantile-mapped)": t_rank,
}


def run(k):
    tf = VARIANTS[k]
    pred = np.zeros_like(GOLD)
    oofP = np.zeros(STACK.shape[1:])
    for f in range(8):
        tr, te = FOLD != f, FOLD == f
        P = tf(tr)
        oofP[te] = P[te]
        pred[te] = L.d55_multistart(P[tr], GOLD[tr]).predict(P[te])
    return k, pred, oofP


if __name__ == "__main__":
    log = L.logger("/wanqing-develop/luowenjing/18/logs/r55/explore4/decoding/pool.log")
    with Pool(len(VARIANTS)) as pool:
        res = pool.map(run, list(VARIANTS))
    preds = {k: p for k, p, _ in res}
    log("calibration of the held-out probabilities (transform fitted on other folds): mean per-dim NLL, top-label ECE, argmax score")
    for m, k in enumerate(KEYS):
        log(f"  member {k:8s} NLL {nll_all(STACK[m], GOLD):.4f}  ECE {ece(STACK[m], GOLD):.4f}  argmax {L.fscore(L.d55.argmax_decode(STACK[m]), GOLD):.2f}")
    for k, _, P in res:
        log(f"  {k:40s} NLL {nll_all(P, GOLD):.4f}  ECE {ece(P, GOLD):.4f}  argmax {L.fscore(L.d55.argmax_decode(P), GOLD):.2f}")
    np.savez(L.HERE + "/dec_pool_preds.npz", **{f"v{n}": preds[k] for n, k in enumerate(VARIANTS)}, keys=np.array(list(VARIANTS)))
    L.report("pooling / calibration, nested, decoded by Decoder55 4 starts", preds, "arith family-balanced (shipped)", D, log)
