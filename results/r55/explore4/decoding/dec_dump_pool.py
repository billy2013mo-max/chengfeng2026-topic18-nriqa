# Dump the 'fallback' OOF pool + members to a local npz cache (CPU, OOF only). Usage: python dec_dump_pool.py
import csv, os, re, sys, numpy as np
REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"): sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c, final_engine as fe
names, gold, fold, P, M = fe.oof_pool(fe.SETS["fallback"])
_, _, _, grp = c.load_split()
sp = {r["name"]: r["split"] for r in csv.DictReader(open(REPO + "/data/manifest_combined.csv", encoding="utf-8"))}
orig = np.array([sp[n] == "orig" for n in names])
def vid(n):
    m = re.match(r"^(?:\d{3}_)?([0-9a-f]{8,}|\d{9,})", n); return m.group(1) if m else n
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dec_pool.npz"), names=np.array(names), gold=gold, fold=fold,
         grp=np.array(grp), vid=np.array([vid(n) for n in names]), orig=orig, P=P, mem_keys=np.array(list(M)),
         **{f"m_{k}": np.asarray(v) for k, v in M.items()})
