# Decode the fallback engine's saved practice-set probabilities with the new decision rules (CPU).
# The probabilities are the engine's own output of 2026-11-11 (same full-fit weights as today; nothing was retrained),
# NOT a legacy prediction file: no label of any earlier model enters.  Integrity check first: decoding them with the
# shipped Decoder55 must reproduce the fallback JSON written by that run.  A fresh GPU inference pass replaces these
# files as soon as a GPU is free (final_engine.py --decoder eu vote).
import json, os, pickle, sys
import numpy as np
REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, _p))
import common55 as c
import decoder_eu  # noqa: F401
F = os.path.join(REPO, "results/final/pred_nriqa_dataset_test_r55_fallback")
z = np.load(F + "_probs.npz", allow_pickle=True)
names, pp = [str(n) for n in z["names"]], z["pooled"]
assert names == list(c.load_template_names()) and pp.shape == (150, 6, 4), (len(names), pp.shape)
def dump(pred): return c.format_official_submission(names, pred)[0]
dec = {k: pickle.load(open(os.path.join(REPO, f"results/r55/final/fallback_decoder{s}.pkl"), "rb"))["decoder"] for k, s in (("d55", ""), ("eu", "_eu"), ("vote", "_vote"))}
P = {k: d.predict(pp) for k, d in dec.items()}
old = json.load(open(F + ".json", encoding="utf-8"))
same = json.dumps(dump(P["d55"]), ensure_ascii=False, sort_keys=True) == json.dumps(old, ensure_ascii=False, sort_keys=True)
print("integrity: Decoder55 on the saved probabilities reproduces the fallback JSON of 2026-11-11:", same)
assert same
DIMS = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
for k in ("eu", "vote"):
    json.dump(dump(P[k]), open(F + f"_{k}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ch = (P[k] != P["d55"])
    print(f"{k:5s} -> {F}_{k}.json | images with a changed label {int(ch.any(1).sum())}/150, changed cells {int(ch.sum())} ("
          + ", ".join(f"{DIMS[d]} {int(ch[:, d].sum())}" for d in range(6)) + f") | fully clean images {int((P[k] == 0).all(1).sum())} (Decoder55: {int((P['d55'] == 0).all(1).sum())})"
          + f" | P0 cells {int((P[k] == 3).sum())} (Decoder55: {int((P['d55'] == 3).sum())})")
