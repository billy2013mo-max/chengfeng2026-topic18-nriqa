"""final_eval.py TAG [name=tower_dir ...]: v7 members (+extra towers) -> per-dim RPS-optimal convex pool (nested)
-> bagged cumulative-threshold ordinal decode (nested) -> F1 + video-cluster bootstrap CI.  Only dims 2 (过锐) / 4 (过度生成)."""
import os; os.environ["OMP_NUM_THREADS"]="2"
import sys, pickle, numpy as np, collections
from harness import *
from dimpool import nested_dimpool
from ordinal_dec import nested_cum_bag
sys.path.insert(0, "/wanqing-develop/luowenjing/18/code/r55"); sys.path.insert(0, "/wanqing-develop/luowenjing/18/code")
from eval_specialist_gate import load_oof_dir
sys.path.insert(0, "/wanqing-develop/luowenjing/18/code/next55"); import common55 as c
tag = sys.argv[1]; P = dict(D["P"])
for e in sys.argv[2:]:
    nm, d_ = e.split("="); arr, ok = load_oof_dir(d_, names); assert ok.all(), nm; P[nm] = c.renorm(arr)
res, W = {}, {}
for d in (2, 4):
    Qd, mem, w = nested_dimpool(P, d); W[d] = dict(zip(mem, np.round(w, 3).tolist())); res[d] = nested_cum_bag(Qd, g[:, d])
report(f"FINAL[{tag}] {len(P)} members, per-dim convex pool + bagged cum decode", res)
by = collections.defaultdict(list)
for i, x in enumerate(groups): by[x].append(i)
keys = list(by); rng = np.random.default_rng(0); B = 500
for d in (2, 4):
    bs = []
    for _ in range(B):
        ii = np.concatenate([by[k] for k in rng.choice(keys, len(keys))]); bs.append(f1s(g[ii, d], res[d][ii]))
    bs = np.array(bs); lo, hi = np.percentile(bs, 5, 0), np.percentile(bs, 95, 0)
    line = f"  {TD[d]} 90% CI P0 [{lo[0]:.3f},{hi[0]:.3f}] P1 [{lo[1]:.3f},{hi[1]:.3f}] P2 [{lo[2]:.3f},{hi[2]:.3f}] | P(F1>=0.5) P0 {np.mean(bs[:,0]>=.5):.2f} P1 {np.mean(bs[:,1]>=.5):.2f} P2 {np.mean(bs[:,2]>=.5):.2f}"
    print(line); open(f"{OUT}/LEDGER.md", "a").write(line + "\n")
pickle.dump(dict(pred=res, weights=W, members=list(P)), open(f"{OUT}/final_{tag}.pkl", "wb"))
