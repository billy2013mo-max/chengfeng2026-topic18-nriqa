import glob, os
import numpy as np

for root, dirs, files in os.walk('.'):
    if '.git' in root or 'models' in root:
        continue
    for f in files:
        if f.endswith('.npz') or f.endswith('.pt') or f.endswith('.npy'):
            p = os.path.join(root, f)
            if any(k in p for k in ['feat', 'emb', 'phys', 'scene', 'graphic', 'cache', 'repr', 'dinov', 'siglip', 'clip']):
                sz = os.path.getsize(p) / (1024 * 1024)
                print(f"{p:65s} {sz:8.2f} MB")
