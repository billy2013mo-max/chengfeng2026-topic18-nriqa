import os, sys, glob
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

print("=== 1. Models in /wanqing-develop/luowenjing/18/models ===")
if os.path.exists('models'):
    for d in sorted(os.listdir('models')):
        p = os.path.join('models', d)
        if os.path.isdir(p):
            files = os.listdir(p)
            sz = sum(os.path.getsize(os.path.join(p, f)) for f in files if os.path.isfile(os.path.join(p, f))) / (1024**2)
            print(f"  models/{d:25s}: {sz:8.1f} MB | files={files[:5]}")

from optimize_v8_isolated_test import load_healthy_split_models
m16, gva16, gte16 = load_healthy_split_models(16)
m17, gva17, gte17 = load_healthy_split_models(17)
common = sorted(set(m16.keys()) & set(m17.keys()))
print(f"\n=== 2. Current Common Models in s16 & s17 ({len(common)}) ===")
for k in common:
    print(f"  {k}")

print("\n=== 3. Check all results/ subdirectories for any other s16/s17 or giant models ===")
for root, dirs, files in os.walk('results'):
    for f in sorted(files):
        if f.endswith('.npz'):
            print(f"  {os.path.join(root, f)}")
