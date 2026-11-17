# Robustness of the two decision rules that looked positive (ordinal cumulative thresholds; EU plug-in):
#   (a) three extra random 8-way partitions of the rows for the nesting (video/group components kept together)
#   (b) other probability pools: primary / primary_v2 sets and every fallback member on its own
#   (c) ordinal variants: single start, finer grid, ordinal thresholds + EU-style clean rule
# All nested; paired video-cluster bootstrap of (variant - shipped Decoder55 4 starts).  CPU only.
import os
from multiprocessing import Pool

import numpy as np

import dec_lib as L
from dec_exp_decoders import OrdDec, v_ordinal, v_eu, eu_fit_stats, eu_predict, q_prod

D = L.load()
GOLD = D["gold"]
c = L.c


def partitions(n_extra=3):
    """connected components over (video id, group id), randomly dealt to 8 folds."""
    vid, grp = D["vid"], D["grp"]
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for v, g in zip(vid, grp):
        a, b = find("v" + v), find("g" + g)
        if a != b:
            parent[a] = b
    comp = np.array([find("v" + v) for v in vid])
    uc, inv = np.unique(comp, return_inverse=True)
    out = {"canonical": D["fold"]}
    for s in range(n_extra):
        rs = np.random.RandomState(100 + s)
        perm = rs.permutation(len(uc))
        sizes = np.bincount(inv, minlength=len(uc))
        fold_of = np.zeros(len(uc), int); load = np.zeros(8)
        for k in perm:                                           # greedy balanced deal in random order
            f = int(np.argmin(load)); fold_of[k] = f; load[f] += sizes[k]
        out[f"random{s}"] = fold_of[inv]
    return out


def get_pools():
    P = {"fallback": D["P"]}
    for k, v in D["mem"].items():
        P[f"member {k}"] = c.renorm(v)
    cache = os.path.join(L.HERE, "dec_pools_extra.npz")
    if not os.path.exists(cache):
        import final_engine as fe
        ex = {}
        for s in ("primary", "primary_v2"):
            nm, g, f, p, _ = fe.oof_pool(fe.SETS[s])
            assert list(nm) == list(D["names"])
            ex[s] = np.asarray(p)
        np.savez(cache, **ex)
    z = np.load(cache)
    for s in z.files:
        P[s] = z[s]
    return P


POOLS = get_pools()
PARTS = partitions()


def v_ord_single(Ptr, gtr, Pte, i, j):
    return OrdDec(t0=0.5).fit(Ptr, gtr).predict(Pte)


class OrdFine(OrdDec):
    TG = np.round(np.arange(0.025, 0.976, 0.025), 3)


def v_ord_fine(Ptr, gtr, Pte, i, j):
    best = None
    for t0 in (0.5, 0.4, 0.3, 0.2):
        d = OrdFine(t0=t0).fit(Ptr, gtr)
        o = d._obj(c.renorm(Ptr), gtr)
        if best is None or o > best[0]:
            best = (o, d)
    return best[1].predict(Pte)


def v_ord_euclean(Ptr, gtr, Pte, i, j):
    """ordinal thresholds fitted as usual; then rows are blanked by the EU clean rule (sum of positive EU gains of the
    predicted labels below q_clean * 20 / n_clean, statistics from the training folds)."""
    best = None
    for t0 in (0.5, 0.4, 0.3, 0.2):
        d = OrdDec(t0=t0).fit(Ptr, gtr)
        o = d._obj(c.renorm(Ptr), gtr)
        if best is None or o > best[0]:
            best = (o, d)
    dec = best[1]
    st = eu_fit_stats(Ptr, gtr, q_prod(Ptr))
    pred = dec.predict(Pte)
    eu = eu_predict(Pte, st, q_prod(Pte))
    pred[(eu == 0).all(1)] = 0
    return pred


DECS = {"shipped": lambda a, b, x, i, j: L.d55_multistart(a, b).predict(x), "ordinal": v_ordinal, "EU": v_eu(),
        "ordinal single start": v_ord_single, "ordinal fine grid": v_ord_fine, "ordinal + EU clean rule": v_ord_euclean}

JOBS = ([("fallback", p, d) for p in PARTS for d in ("shipped", "ordinal", "EU")]
        + [(pl, "canonical", d) for pl in POOLS if pl != "fallback" for d in ("shipped", "ordinal", "EU")]
        + [("fallback", "canonical", d) for d in ("ordinal single start", "ordinal fine grid", "ordinal + EU clean rule")])


def run(job):
    pl, part, dn = job
    DD = dict(D); DD["fold"] = PARTS[part]
    return job, L.nested(DECS[dn], DD, P=POOLS[pl])


if __name__ == "__main__":
    log = L.logger("/wanqing-develop/luowenjing/18/logs/r55/explore4/decoding/robust.log")
    with Pool(min(len(JOBS), 48)) as pool:
        res = dict(pool.map(run, JOBS))
    np.savez(L.HERE + "/dec_robust_preds.npz", **{"|".join(k): v for k, v in res.items()})
    groups = {}
    for (pl, part, dn), pr in res.items():
        groups.setdefault((pl, part), {})[dn] = pr
    for (pl, part), preds in groups.items():
        L.report(f"pool={pl} partition={part}", {"shipped": preds["shipped"], **{k: v for k, v in preds.items() if k != "shipped"}}
                 if "shipped" in preds else preds, "shipped", D, log) if "shipped" in preds else None
    extra = {k: res[("fallback", "canonical", k)] for k in ("shipped", "ordinal single start", "ordinal fine grid", "ordinal + EU clean rule")}
    L.report("fallback canonical, ordinal variants", extra, "shipped", D, log)
