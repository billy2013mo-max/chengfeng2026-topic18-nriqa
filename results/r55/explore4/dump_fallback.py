#!/usr/bin/env python3
# Dump the FALLBACK ensemble OOF pool + decodings for the error analysis (CPU only, OOF only).
#   pooled probs  : final_engine.oof_pool(SETS["fallback"])  (family-balanced mean)
#   nested        : Decoder55 nested over the 8 folds (objective doc2_strict_total) -> the reference 51.35
#   per-fold decoders kept, so oracle analyses can re-use the exact decision layer of each held-out fold
#   insample      : Decoder55 multistart fitted on ALL OOF rows (optimistic upper bound of decoding)
# Usage: python3 dump_fallback.py <out.npz>
import os
import pickle
import sys

import numpy as np

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c  # noqa: E402
from decoder55 import Decoder55, argmax_decode  # noqa: E402
from decoder_select import fit_multistart  # noqa: E402
from final_engine import SETS, oof_pool  # noqa: E402

K = "doc2_strict_total"
names, gold, fold, P, mem = oof_pool(SETS["fallback"])
_, _, _, grp = c.load_split()
nested = np.zeros_like(gold)
decs = {}
for f in np.unique(fold):
    tr, te = fold != f, fold == f
    d = Decoder55(K).fit(P[tr], gold[tr], sweeps=3)
    nested[te] = d.predict(P[te])
    decs[int(f)] = d.state()
ins = fit_multistart(P, gold)
pin = ins.predict(P)
for tag, pr in (("argmax", argmax_decode(P)), ("nested", nested), ("insample", pin)):
    print(tag, c.score_line(c.score(pr, gold)))
np.savez(sys.argv[1], names=np.array(names), gold=gold, fold=fold, grp=np.array(grp), P=P, nested=nested,
         insample=pin, argmax=argmax_decode(P), **{f"m_{k}": v for k, v in mem.items()})
pickle.dump(dict(fold_decoders=decs, insample=ins.state()), open(sys.argv[1].replace(".npz", "_decoders.pkl"), "wb"))
