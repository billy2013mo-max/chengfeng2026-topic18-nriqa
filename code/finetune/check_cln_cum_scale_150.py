import os, sys, json
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from optimize_v8_isolated_test import load_healthy_split_models
from fast_test_qualirag_preserved_p0 import build_p0_preserved_v9_14ch
from generate_v9_150_submission import load_150_context
from run_scheme1_scheme2_breakthrough import FULL_IDX

for sp in (16, 17):
    models, gva, gte = load_healthy_split_models(sp)
    p0, p1, p2, cln0, cln1, cln2 = build_p0_preserved_v9_14ch(models, 'te', None)
    cln_cum = cln0.copy()
    for k, ci in enumerate(FULL_IDX):
        cln_cum[:, ci] += cln1[:, k] + cln2[:, k]
    print(f"[s{sp}_te] cln_cum.max(1) quantiles (10%, 15%, 20%, 50%):", np.quantile(cln_cum.max(1), [0.10, 0.15, 0.20, 0.50]).round(3))
    print(f"         frac(cln_cum.max(1) < 0.28) = {(cln_cum.max(1) < 0.28).mean():.3f}, frac(< 0.38) = {(cln_cum.max(1) < 0.38).mean():.3f}")

ctx_150 = load_150_context()
cln0_150, cln1_150, cln2_150 = ctx_150['cln_p0'], ctx_150['cln_p1'], ctx_150['cln_p2']
cln_cum_150 = cln0_150.copy()
for k, ci in enumerate(FULL_IDX):
    cln_cum_150[:, ci] += cln1_150[:, k] + cln2_150[:, k]
print(f"[test_150] cln_cum_150.max(1) quantiles (10%, 15%, 20%, 50%):", np.quantile(cln_cum_150.max(1), [0.10, 0.15, 0.20, 0.50]).round(3))
print(f"           frac(cln_cum_150.max(1) < 0.28) = {(cln_cum_150.max(1) < 0.28).mean():.3f}, frac(< 0.38) = {(cln_cum_150.max(1) < 0.38).mean():.3f}")
