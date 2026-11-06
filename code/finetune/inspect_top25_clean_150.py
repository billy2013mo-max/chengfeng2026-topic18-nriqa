import os, sys, json
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from generate_v9_150_submission import load_150_context
from run_scheme1_scheme2_breakthrough import FULL_IDX

ctx_150 = load_150_context()
names = ctx_150['ordered_names']
cln0, cln1, cln2 = ctx_150['cln_p0'], ctx_150['cln_p1'], ctx_150['cln_p2']
cln_cum = cln0.copy()
for k, ci in enumerate(FULL_IDX):
    cln_cum[:, ci] += cln1[:, k] + cln2[:, k]

s2_sub = {e['image']: e['label'] for e in json.load(open('results/scheme2_sub/pred_nriqa_dataset_test_scheme2.json'))}
b5243 = ctx_150['b5243']

mx = cln_cum.max(1)
order = np.argsort(mx)
print("Rank | Image Name                     | cln_cum.max | 52.43 Clean? | Scheme2 Clean? | Scheme2 Labels")
for rank, idx in enumerate(order[:25], 1):
    n = names[idx]
    c_old = all(v == '无问题' for v in b5243[n].values())
    c_new = all(v == '无问题' for v in s2_sub[n].values())
    active = {k: v for k, v in s2_sub[n].items() if v != '无问题'}
    print(f"{rank:4d} | {n:30s} | {mx[idx]:11.4f} | {str(c_old):12s} | {str(c_new):14s} | {active}")
