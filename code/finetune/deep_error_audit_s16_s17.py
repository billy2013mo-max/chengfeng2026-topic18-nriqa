import os, sys, glob, json
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from build_v8_final_submission_and_eval import run_isolated_1000_eval

print("=== 1. Listing code/ and data/ directories ===")
for d in ['code', 'code/finetune', 'code/decision_eval', 'data']:
    if os.path.exists(d):
        print(f"\n[{d}]:")
        for f in sorted(os.listdir(d)):
            print(f"  {f}")

print("\n=== 2. Running Detailed Per-Channel Error & Confusion Audit on Isolated s16 & s17 ===")
data_sp, p0_rec, p1p2_rec = run_isolated_1000_eval()
