# Decision-rule variants on the SAME pooled fallback OOF probabilities, all nested over the 8 folds
# (every threshold / calibration / statistic of fold f is computed on the other seven folds only).
# Paired video-cluster bootstrap vs the shipped rule (Decoder55, four starts).  CPU only.
# Usage: python dec_exp_decoders.py
import sys
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LogisticRegression

import dec_lib as L

D = L.load()
GOLD, FOLD, ORIG = D["gold"], D["fold"], D["orig"]
c, d55 = L.c, L.d55


# ---------------------------------------------------------------- 1. ordinal cumulative-threshold decoder
class OrdDec(d55.Decoder55):
    """grade = highest k with P(grade >= k) >= t[d,k] (k = P2, P1, P0); binary dims: P(P0) >= t; + Decoder55 clean gate."""
    TG = np.round(np.arange(0.05, 0.96, 0.05), 3)

    def __init__(self, objective=L.K, t0=0.5):
        super().__init__(objective)
        self.t = np.full((6, 4), t0)

    def predict(self, s):
        s = c.renorm(s)
        cum = np.cumsum(s[:, :, ::-1], -1)[:, :, ::-1]          # cum[..., k] = P(grade >= k)
        pred = np.zeros(s.shape[:2], dtype=np.int64)
        for d in range(6):
            ks = (1, 2, 3) if d in L.FULL else (3,)
            for k in ks:                                        # ascending: later (higher) k overrides
                pred[cum[:, d, k] >= self.t[d, k], d] = k
        if self.t_clean > 0:
            pred[np.prod(s[:, :, 0], 1) >= self.t_clean] = 0
        return pred

    def fit(self, s, gold, sweeps=3, verbose=False):
        s = c.renorm(s)
        best = self._obj(s, gold)
        for _ in range(sweeps):
            imp = False
            for d in range(6):
                for k in ((1, 2, 3) if d in L.FULL else (3,)):
                    keep = self.t[d, k]
                    for v in self.TG:
                        self.t[d, k] = v
                        o = self._obj(s, gold)
                        if o > best + 1e-9:
                            best, keep, imp = o, v, True
                    self.t[d, k] = keep
            keep = self.t_clean
            for v in d55.GATE_GRID:
                self.t_clean = v
                o = self._obj(s, gold)
                if o > best + 1e-9:
                    best, keep, imp = o, v, True
            self.t_clean = keep
            if not imp:
                break
        return self


def v_ordinal(Ptr, gtr, Pte, i, j):
    best = None
    for t0 in (0.5, 0.4, 0.3, 0.2):
        d = OrdDec(t0=t0).fit(Ptr, gtr)
        o = d._obj(c.renorm(Ptr), gtr)
        if best is None or o > best[0]:
            best = (o, d)
    return best[1].predict(Pte)


# ---------------------------------------------------------------- 2. expected-utility (plug-in) decoder
W_F = np.where(L.LG == 3, 25 / 14 + 20 * 0.3 / 6, 25 / 14)
W_R = np.where(L.LG == 3, 20 * 0.5 / 6, 0.0)
W_P = np.where(L.LG == 3, 20 * 0.2 / 6, 0.0)


def label_stats(pred, gold):
    tp, npred, npos, ncl, okcl = L.counts(pred, gold)
    tp, npred, npos = (x[L.LD, L.LG].astype(float) for x in (tp, npred, npos))
    F = 2 * tp / np.maximum(1, npred + npos)
    Pr = tp / np.maximum(1, npred)
    return dict(F=F, P=Pr, npos=npos, npred=npred, ncl=ncl, N=len(gold))


def label_stats_expected(pred, Q):
    """same statistics with gold replaced by its expectation under Q (no labels; transductive)."""
    pq = Q[:, L.LD, L.LG]                                       # (N,14) P(label)
    pl = pred[:, L.LD] == L.LG[None]
    tp = (pq * pl).sum(0); npred = pl.sum(0).astype(float); npos = pq.sum(0)
    return dict(F=2 * tp / np.maximum(1, npred + npos), P=tp / np.maximum(1, npred), npos=npos, npred=npred,
                ncl=float(np.prod(Q[:, :, 0], 1).sum()), N=len(Q))


def eu_predict(Q, st, qclean=None, clean_w=1.0):
    """Marginal-gain rule.  Gain of adding one positive prediction with probability p for label l (first-order in the
    corpus counts, scaled to the size of the set being decoded):
        w_F (2p - F)/D + w_R p/npos + w_P (p - P)/npred,   D = npred + npos   (F-part = Lipton et al.'s F*/2 rule)
    Per dimension the grade with the largest positive gain is predicted.  A row is left fully clean when the sum of its
    positive gains is below the expected Clean loss  q_clean * 20 / n_clean."""
    N = len(Q); sc = N / st["N"]
    D = np.maximum(1, (st["npred"] + st["npos"]) * sc); npos = np.maximum(1, st["npos"] * sc); npred = np.maximum(1, st["npred"] * sc)
    p = Q[:, L.LD, L.LG]
    gain = W_F * (2 * p - st["F"]) / D + W_R * p / npos + W_P * (p - st["P"]) / npred      # (N,14)
    pred = np.zeros((N, 6), dtype=np.int64)
    best = np.zeros((N, 6))
    for l, (d, g) in enumerate(L.LABELS):
        m = gain[:, l] > best[:, d]
        pred[m, d] = g; best[m, d] = gain[m, l]
    if qclean is not None:
        ncl = max(1.0, st["ncl"] * sc)
        pen = clean_w * qclean * 20.0 / ncl
        pred[best.sum(1) < pen] = 0
    return pred


def eu_fit_stats(Q, gold, qclean=None, iters=15, clean_w=1.0):
    pred = d55.argmax_decode(Q)
    for _ in range(iters):
        st = label_stats(pred, gold)
        new = eu_predict(Q, st, qclean, clean_w)
        if (new == pred).all():
            break
        pred = new
    return label_stats(pred, gold)


def nested_T(Ptr, gtr):
    return np.array([L.fit_T(Ptr[:, d], gtr[:, d], d) for d in range(6)])


def q_prod(Q):
    return np.prod(Q[:, :, 0], 1)


def v_eu(calib=False, clean=True, trans=False):
    def f(Ptr, gtr, Pte, i, j):
        if calib:
            T = nested_T(Ptr, gtr); Ptr, Pte = L.apply_T(Ptr, T), L.apply_T(Pte, T)
        qtr = q_prod(Ptr) if clean else None
        qte = q_prod(Pte) if clean else None
        st = eu_fit_stats(Ptr, gtr, qtr)
        if trans:                                               # transductive: statistics re-estimated on the decoded set
            pred = eu_predict(Pte, st, qte)
            for _ in range(15):
                st2 = label_stats_expected(pred, Pte)
                st2["ncl"] = st["ncl"] * len(Pte) / st["N"]
                new = eu_predict(Pte, st2, qte)
                if (new == pred).all():
                    break
                pred = new
            return pred
        return eu_predict(Pte, st, qte)
    return f


# ---------------------------------------------------------------- 3. Decoder55 with a learned clean detector as gate
def clean_feats(Q):
    lq = np.log(np.clip(Q, 1e-6, 1))
    return np.concatenate([lq[:, :, 0], lq[:, :, 3], np.log(np.clip(1 - Q[:, :, 0], 1e-6, 1)), np.log(q_prod(Q))[:, None]], 1)


class GateDec(d55.Decoder55):
    """Decoder55 whose clean gate thresholds q from a logistic clean detector (self.q, one value per row) instead of
    prod_d P(无问题)."""
    q = None

    def predict(self, s):
        t, self.t_clean = self.t_clean, 0.0
        pred = super().predict(s)
        self.t_clean = t
        if t > 0:
            pred[self.q >= t] = 0
        return pred


def v_learned_gate(Ptr, gtr, Pte, i, j):
    ftr, fte = clean_feats(Ptr), clean_feats(Pte)
    ytr = (gtr == 0).all(1).astype(int)
    # q for the decoder-fit rows must itself be out-of-fold (inner cross-fit over the 7 training folds)
    qtr = np.zeros(len(gtr)); ftr_fold = FOLD[i]
    for f in np.unique(ftr_fold):
        a, b = ftr_fold != f, ftr_fold == f
        qtr[b] = LogisticRegression(C=0.3, max_iter=2000).fit(ftr[a], ytr[a]).predict_proba(ftr[b])[:, 1]
    qte = LogisticRegression(C=0.3, max_iter=2000).fit(ftr, ytr).predict_proba(fte)[:, 1]
    best = None
    for b0 in (0.0, 0.3, 0.6, 0.9):
        d = GateDec(L.K); d.q = qtr; d.bias[:, 1:] += b0
        d.fit(Ptr, gtr)
        o = d._obj(c.renorm(Ptr), gtr)
        if best is None or o > best[0]:
            best = (o, d)
    best[1].q = qte
    return best[1].predict(Pte)


# ---------------------------------------------------------------- 4. structural (pairwise co-occurrence) constraint
def forbidden_pairs(gtr, min_count=1, min_expected=8.0):
    """(i, gi, j, gj) combinations never (< min_count) seen together in the training gold although independence would
    predict >= min_expected of them."""
    out = []
    n = len(gtr)
    for i in range(6):
        for j in range(i + 1, 6):
            for gi in (1, 2, 3):
                for gj in (1, 2, 3):
                    a, b = gtr[:, i] == gi, gtr[:, j] == gj
                    if a.sum() == 0 or b.sum() == 0:
                        continue
                    if (a & b).sum() < min_count and a.mean() * b.mean() * n >= min_expected:
                        out.append((i, gi, j, gj))
    return out


class ConsDec(d55.Decoder55):
    pairs = []

    def predict(self, s):
        pred = super().predict(s)
        s = c.renorm(s)
        for (i, gi, j, gj) in self.pairs:
            m = (pred[:, i] == gi) & (pred[:, j] == gj)
            if m.any():
                drop_i = s[m, i, gi] < s[m, j, gj]               # keep the more confident defect, drop the other to 无问题
                rows = np.where(m)[0]
                pred[rows[drop_i], i] = 0
                pred[rows[~drop_i], j] = 0
        return pred


def v_constraint(Ptr, gtr, Pte, i, j):
    pairs = forbidden_pairs(gtr)

    class C_(ConsDec):
        pass
    C_.pairs = pairs
    return L.d55_multistart(Ptr, gtr, cls=C_).predict(Pte)


def v_constraint_post(Ptr, gtr, Pte, i, j):
    """constraint applied after a normally fitted Decoder55 (thresholds not re-fitted)."""
    dec = L.d55_multistart(Ptr, gtr)

    class C_(ConsDec):
        pass
    C_.pairs = forbidden_pairs(gtr)
    cd = C_(L.K); cd.bias, cd.floor, cd.t_clean = dec.bias, dec.floor, dec.t_clean
    return cd.predict(Pte)


# ---------------------------------------------------------------- 5. first batch: fit rows, prior shift, weighting
def v_shipped(Ptr, gtr, Pte, i, j):
    return L.d55_multistart(Ptr, gtr).predict(Pte)


def v_orig_dup(Ptr, gtr, Pte, i, j):
    """fit rows = all training rows, first-batch rows counted twice."""
    o = ORIG[i]
    P2 = np.concatenate([Ptr, Ptr[o]]); g2 = np.concatenate([gtr, gtr[o]])
    return L.d55_multistart(P2, g2).predict(Pte)


def prior_shift(P, pi_from, pi_to):
    Q = P * (pi_to / pi_from)[None]
    return c.renorm(Q)


def v_prior_shift(Ptr, gtr, Pte, i, j):
    """decoder fitted on all training rows; probabilities of rows to be decoded multiplied by pi_orig/pi_all
    (both priors from the training folds' gold; Saerens-style label-shift correction toward the first batch)."""
    o = ORIG[i]
    pa = np.stack([np.bincount(gtr[:, d], minlength=4) for d in range(6)]).astype(float) + 1
    po = np.stack([np.bincount(gtr[o, d], minlength=4) for d in range(6)]).astype(float) + 1
    pa /= pa.sum(1, keepdims=True); po /= po.sum(1, keepdims=True)
    dec = L.d55_multistart(Ptr, gtr)
    return dec.predict(prior_shift(Pte, pa, po))


VARIANTS = {
    "shipped: Decoder55 4 starts": (v_shipped, None),
    "Decoder55 single start": (lambda a, b, x, i, j: d55.Decoder55(L.K).fit(a, b).predict(x), None),
    "argmax (no fit)": (lambda a, b, x, i, j: d55.argmax_decode(x), None),
    "ordinal cumulative thresholds": (v_ordinal, None),
    "EU plug-in, no clean term": (v_eu(clean=False), None),
    "EU plug-in": (v_eu(), None),
    "EU plug-in + nested temperature": (v_eu(calib=True), None),
    "EU plug-in, transductive stats": (v_eu(trans=True), None),
    "EU plug-in + temp, transductive": (v_eu(calib=True, trans=True), None),
    "Decoder55 + learned clean gate": (v_learned_gate, None),
    "Decoder55 + pair constraints (fitted in)": (v_constraint, None),
    "Decoder55 + pair constraints (post hoc)": (v_constraint_post, None),
    "Decoder55 fit on first-batch rows only": (v_shipped, ORIG),
    "Decoder55 fit, first batch x2": (v_orig_dup, None),
    "Decoder55 + prior shift to first batch": (v_prior_shift, None),
}


def run(k):
    fn, rows = VARIANTS[k]
    return k, L.nested(fn, D, rows_fit=rows)


if __name__ == "__main__":
    log = L.logger("/wanqing-develop/luowenjing/18/logs/r55/explore4/decoding/decoders.log")
    with Pool(len(VARIANTS)) as pool:
        res = dict(pool.map(run, list(VARIANTS)))
    preds = {k: res[k] for k in VARIANTS}
    np.savez(L.HERE + "/dec_decoder_preds.npz", **{f"v{n}": preds[k] for n, k in enumerate(VARIANTS)}, keys=np.array(list(VARIANTS)))
    log("forbidden pairs on all rows:", forbidden_pairs(GOLD))
    L.report("decision rules, nested", preds, "shipped: Decoder55 4 starts", D, log)
