import os, sys, glob, json, csv
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

print("=== 1. Checking .jsonl files in results/ ===")
for root, dirs, files in os.walk('results'):
    for f in sorted(files):
        if f.endswith('.jsonl'):
            fp = os.path.join(root, f)
            n_lines = sum(1 for _ in open(fp))
            print(f"  {fp:45s}: {n_lines} lines")

print("\n=== 2. Checking alignment of feats/*_combined.npz with s16 and s17 ===")
man = [r['name'] for r in csv.DictReader(open('data/manifest_combined.csv'))]
print(f"manifest_combined.csv count: {len(man)}")
for sp in (16, 17):
    sp_rows = list(csv.DictReader(open(f'data/resplits/split_s{sp}.csv')))
    by_s = {}
    for r in sp_rows:
        by_s.setdefault(r['split'], []).append(r['name'])
    print(f"split_s{sp}:", {k: len(v) for k, v in by_s.items()})
