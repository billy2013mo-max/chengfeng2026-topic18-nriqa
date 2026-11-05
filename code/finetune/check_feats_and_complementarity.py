import os, sys, glob
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
print("=== Files in feats/ ===")
if os.path.exists('feats'):
    for f in sorted(os.listdir('feats')):
        fp = os.path.join('feats', f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp) / (1024**2)
            try:
                d = np.load(fp, allow_pickle=True)
                shapes = {k: d[k].shape for k in d.files}
            except Exception as e:
                shapes = str(e)
            print(f"  {f:35s} ({sz:7.1f} MB): {shapes}")
else:
    print("No feats/ directory")

print("\n=== Top-level directories in /wanqing-develop/luowenjing/18 ===")
for x in sorted(os.listdir('.')):
    print(f"  {x}")
