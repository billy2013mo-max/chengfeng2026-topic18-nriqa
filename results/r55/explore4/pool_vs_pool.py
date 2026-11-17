# primary (with the VLM) vs fallback under each decision rule, nested over the canonical folds, paired video bootstrap.
import csv, os, sys
import numpy as np
REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c
import decoder_eu as M
import final_engine as fe
K = M.K
pred = {}
for s in ("fallback", "primary"):
    names, gold, fold, P, _ = fe.oof_pool(fe.SETS[s])
    P = np.asarray(P, dtype=np.float64)
    pred[s] = {k: np.zeros_like(gold) for k in ("d55", "eu", "vote")}
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        v = M.VoteDecoder().fit(P[tr], gold[tr]); e = M.DecoderEU().fit(P[tr], gold[tr])
        pred[s]["d55"][te], pred[s]["eu"][te], pred[s]["vote"][te] = v.d55.predict(P[te]), e.predict(P[te]), v.predict(P[te])
names = [str(n) for n in names]
batch = {r["name"]: r["split"] for r in csv.DictReader(open(REPO + "/data/manifest_combined.csv", encoding="utf-8"))}
o = np.array([batch[n] == "orig" for n in names])
_, _, _, grp = c.load_split()
ug, inv = np.unique(np.asarray(grp), return_inverse=True); rs = np.random.RandomState(0)
np.savez_compressed(REPO + "/results/r55/explore4/pool_vs_pool_preds.npz", names=np.array(names), **{f"{s}_{k}": v for s in pred for k, v in pred[s].items()})
for tag, mask in (("all rows", np.ones(len(names), bool)), ("first-batch rows", o)):
    mem = [np.where((inv == i) & mask)[0] for i in range(len(ug))]; mem = [m for m in mem if len(m)]
    draws = [np.concatenate([mem[j] for j in rs.randint(0, len(mem), len(mem))]) for _ in range(1000)]
    sc = lambda p, i: c.score(p[i], gold[i])
    print(f"--- {tag}")
    for k in ("d55", "eu", "vote"):
        a, b = pred["primary"][k], pred["fallback"][k]
        sa, sb = sc(a, mask), sc(b, mask)
        dd = np.array([sc(a, i)[K] - sc(b, i)[K] for i in draws])
        print(f"  rule {k:4s}: primary {sa[K]:.2f} (pos {sa['doc2_S_pos']:.2f}, P0 {sa['doc2_S_p0']:.2f}) | fallback {sb[K]:.2f} (pos {sb['doc2_S_pos']:.2f}, P0 {sb['doc2_S_p0']:.2f})"
              f" | primary - fallback {sa[K] - sb[K]:+.2f} [{np.quantile(dd, .05):+.2f}, {np.quantile(dd, .95):+.2f}], P(>0) {(dd > 0).mean():.2f}", flush=True)
    base = pred["fallback"]["eu"]
    for s, k in (("primary", "eu"), ("primary", "vote"), ("fallback", "vote")):
        dd = np.array([sc(pred[s][k], i)[K] - sc(base, i)[K] for i in draws])
        print(f"  {s}+{k} minus fallback+eu (the uploaded system): {sc(pred[s][k], mask)[K] - sc(base, mask)[K]:+.2f} [{np.quantile(dd, .05):+.2f}, {np.quantile(dd, .95):+.2f}], P(>0) {(dd > 0).mean():.2f}", flush=True)
