import glob, os
import numpy as np

for root, dirs, files in os.walk('.'):
    if '.git' in root or 'models' in root:
        continue
    for f in files:
        if f.endswith('.npz'):
            p = os.path.join(root, f)
            try:
                d = np.load(p, allow_pickle=True)
                for k in d.files:
                    arr = d[k]
                    if hasattr(arr, 'shape') and any(s in (150, 3723) for s in arr.shape):
                        print(f"{p:60s} key={k:15s} shape={arr.shape}")
            except Exception:
                pass
