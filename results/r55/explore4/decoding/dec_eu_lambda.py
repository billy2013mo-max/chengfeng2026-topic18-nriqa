# EU plug-in with ONE fitted parameter (clean-penalty scale lambda, chosen on the training folds), and a per-cell
# majority vote of the three nested decoders (shipped / ordinal / EU), over the four nesting partitions.  CPU only.
from multiprocessing import Pool
import numpy as np
import dec_lib as L
from dec_exp_decoders import eu_fit_stats, eu_predict, q_prod
from dec_robust import PARTS, D
LAMS = [0.25, 0.5, 1.0, 2.0, 4.0]

def v_eu_lam(Ptr, gtr, Pte, i, j):
    best = None
    for lam in LAMS:
        st = eu_fit_stats(Ptr, gtr, q_prod(Ptr), clean_w=lam)
        o = L.fscore(eu_predict(Ptr, st, q_prod(Ptr), clean_w=lam), gtr)
        if best is None or o > best[0]:
            best = (o, st, lam)
    return eu_predict(Pte, best[1], q_prod(Pte), clean_w=best[2])

def run(part):
    DD = dict(D); DD["fold"] = PARTS[part]
    return part, L.nested(v_eu_lam, DD)

if __name__ == "__main__":
    log = L.logger("/wanqing-develop/luowenjing/18/logs/r55/explore4/decoding/eu_lambda.log")
    R = np.load(L.HERE + "/dec_robust_preds.npz")
    with Pool(4) as pool:
        lam = dict(pool.map(run, list(PARTS)))
    for part in PARTS:
        s, o, e = (R[f"fallback|{part}|{k}"] for k in ("shipped", "ordinal", "EU"))
        vote = np.where(o == e, o, np.where(s == o, s, np.where(s == e, s, e)))
        L.report(f"fallback partition={part}", {"shipped": s, "EU": e, "EU + fitted clean lambda": lam[part], "vote(shipped,ordinal,EU)": vote}, "shipped", D, log)
