import os, sys, numpy as np, cv2
REPO = "/wanqing-develop/luowenjing/18"
for p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
from final_engine import SETS, oof_pool
OUT = sys.argv[1]
names, gold, fold, pp, _ = oof_pool(SETS["primary"]); gold = np.asarray(gold); pp = np.asarray(pp)
AB = ["BL", "TX", "SH", "GL", "OG", "BK"]
def find(n):
    for d in ("data/images", "data/images_supp"):
        p = os.path.join(REPO, d, n)
        if os.path.exists(p): return p
def big(i, H=1400):
    im = cv2.imdecode(np.fromfile(find(names[i]), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = im.shape[:2]; s = min(H / h, 800 / w)
    t = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    pad = np.zeros((H + 30, 800, 3), np.uint8); pad[:t.shape[0], :t.shape[1]] = t
    g = " ".join(f"{a}{int(gold[i, k])}" for k, a in enumerate(AB) if gold[i, k] > 0) or "clean"
    cv2.putText(pad, f"#{i} {w}x{h} gold[{g}] OGp=" + " ".join(f"{v:.2f}" for v in pp[i, 4]), (4, H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return pad
y = gold[:, 4]
m = np.where(y == int(sys.argv[2]))[0]
o = m[np.argsort(-pp[m, 4, 0])]
k0 = int(sys.argv[3])
sel = o[k0:k0 + 2]
cv2.imwrite(os.path.join(OUT, f"big_g{sys.argv[2]}_{k0}.jpg"), np.concatenate([big(i) for i in sel], 1), [cv2.IMWRITE_JPEG_QUALITY, 90])
print([int(i) for i in sel], [names[i][-28:] for i in sel])
