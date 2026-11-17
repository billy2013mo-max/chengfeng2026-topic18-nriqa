import os; os.environ["OMP_NUM_THREADS"]="2"
import sys, numpy as np
from harness import *
from dimpool import nested_dimpool
from ordinal_dec import nested_cum_bag
from joblib import Parallel, delayed
sys.path.insert(0,"/wanqing-develop/luowenjing/18/code/r55"); sys.path.insert(0,"/wanqing-develop/luowenjing/18/code"); sys.path.insert(0,"/wanqing-develop/luowenjing/18/code/next55")
from eval_specialist_gate import load_oof_dir
import common55 as c
P13 = dict(D["P"])
for nm, d_ in [("cnxf", "results/next55/towers/r55_sp_ntgcnx_focus_s2026/convnextv2_base"), ("pef", "results/next55/towers/r55_sp_ntgpe_focus_s2026/pe_core_large")]:
    a, ok = load_oof_dir("/wanqing-develop/luowenjing/18/" + d_, names); P13[nm] = c.renorm(a)
pools = {"v7(11)": nested_dimpool(D["P"], 2)[0], "v7+2(13)": nested_dimpool(P13, 2)[0]}
jobs = [(pn, w) for pn in pools for w in [(1, 1, 1), (1, 1.3, 1), (1, 1.6, 1), (1, 2.0, 1)]]
out = Parallel(n_jobs=8)(delayed(nested_cum_bag)(pools[pn], g[:, 2], 20, 0, w) for pn, w in jobs)
for (pn, w), pr in zip(jobs, out):
    a = f1s(g[:, 2], pr); o = f1s(g[io, 2], pr[io]); s = f1s(g[~io, 2], pr[~io])
    line = f"过锐 pool={pn:9s} w={w} ALL {a.round(3)} ORIG {o.round(3)} SUPP {s.round(3)}"
    print(line, flush=True); open(OUT + "/LEDGER.md", "a").write(line + "\n")
