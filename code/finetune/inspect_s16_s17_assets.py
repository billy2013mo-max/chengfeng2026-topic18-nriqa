import subprocess
import os, glob
import numpy as np
import pandas as pd

# Clean up any lingering VLM training processes first
subprocess.run(['pkill', '-f', 'train_vlm_cls.py'], check=False)

print("=== Checking available probability & feature caches for s16 / s17 / 3573 / test150 ===")
for pat in [
    'results/*/*.npz',
    'results/*/*/*.npz',
    'data/*.npz',
    'data/*/*.npz',
    'features/*.npz',
    'results/*.npz'
]:
    files = sorted(glob.glob(pat))
    if files:
        print(fPattern := pat, f"({len(files)} files):", files[:8], "..." if len(files) > 8 else "")

for p in ['data/resplits/split_s16.csv', 'data/resplits/split_s17.csv', 'data/manifest_combined.csv']:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(p, len(df), df.columns.tolist()[:12])
        if 'split' in df.columns:
            print("  splits:", df['split'].value_counts().to_dict())
