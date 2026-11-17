import os; os.environ["OMP_NUM_THREADS"]="2"
import sys, pickle
from ordinal_dec import *
from joblib import Parallel, delayed
srcs = {"v7": D["Q"], "v7+cnxfocus": pickle.load(open(OUT + "/pred_v7+cnxfocus.pkl", "rb"))["Q"]}
jobs = [(s, d) for s in srcs for d in (2, 4)]
out = Parallel(n_jobs=4)(delayed(nested_cum_bag)(srcs[s][:, d], g[:, d]) for s, d in jobs)
R = {}
for (s, d), pr in zip(jobs, out): R.setdefault(s, {})[d] = pr
for s in srcs: report(f"{s} + bagged cum-decode (B=20)", R[s])
