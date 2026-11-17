import os; os.environ["OMP_NUM_THREADS"]="1"
import sys, numpy as np, importlib.util
from multiprocessing import Pool
spec = importlib.util.spec_from_file_location("op", "/root/.gemini/antigravity-cli/brain/9090e9e1-1206-441a-8178-f8744f6243b1/scratch/p1_oversharp_probe.py")
from harness import *
def path(x):
    for d in ("data/images", "data/images_supp"):
        p = f"/wanqing-develop/luowenjing/18/{d}/{x}"
        if os.path.exists(p): return p
def feat(n):
    import cv2; cv2.setNumThreads(1)
    op = importlib.util.module_from_spec(spec); spec.loader.exec_module(op)
    return op.extract_oversharp_feats(path(n))
if __name__ == "__main__":
    with Pool(16) as p: X = np.stack(p.map(feat, names, chunksize=16))
    np.save(OUT + "/halo14.npy", X); print(X.shape)
