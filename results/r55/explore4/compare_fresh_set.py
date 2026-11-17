# Compare the fresh practice-set inference pass with the files decoded from the saved probabilities of 2026-11-11.
import json, sys, numpy as np
SET = sys.argv[1] if len(sys.argv) > 1 else "fallback"
R = "/wanqing-develop/luowenjing/18/results/final/"
a = np.load(R + "pred_nriqa_dataset_test_r55_" + SET + "_probs.npz", allow_pickle=True)
b = np.load(R + "fresh_20261113/pred_nriqa_dataset_test_r55_" + SET + "_probs.npz", allow_pickle=True)
assert list(a["names"]) == list(b["names"])
d = np.abs(a["pooled"] - b["pooled"])
print(f"pooled probabilities, fresh vs saved: max |dp| {d.max():.5f}, mean total variation {0.5 * d.sum(-1).mean():.6f}")
for rule, old in (("d55", "pred_nriqa_dataset_test_r55_" + SET + ".json"), ("eu", "pred_nriqa_dataset_test_r55_" + SET + "_eu.json"), ("vote", "pred_nriqa_dataset_test_r55_" + SET + "_vote.json")):
    x = json.load(open(R + old, encoding="utf-8")); y = json.load(open(R + "fresh_20261113/pred_nriqa_dataset_test_r55_" + SET + f"_{rule}.json", encoding="utf-8"))
    n = sum(json.dumps(p, ensure_ascii=False, sort_keys=True) != json.dumps(q, ensure_ascii=False, sort_keys=True) for p, q in zip(x, y))
    print(f"rule {rule:4s}: images whose labels differ between the fresh pass and the delivered file: {n}/150")
