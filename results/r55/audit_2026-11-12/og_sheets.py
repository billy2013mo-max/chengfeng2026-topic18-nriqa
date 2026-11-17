import os, sys, numpy as np, cv2
REPO = "/wanqing-develop/luowenjing/18"
for p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
from final_engine import SETS, oof_pool
OUT = sys.argv[1]
names, gold, fold, pp, _ = oof_pool(SETS["primary"])
gold = np.asarray(gold); pp = np.asarray(pp)
print("pool", pp.shape, gold.shape)
D = 4
eg = (pp[:, D] * np.arange(4)).sum(1)
def find(n):
    for d in ("data/images", "data/images_supp"):
        p = os.path.join(REPO, d, n)
        if os.path.exists(p): return p
AB = ["BL", "TX", "SH", "GL", "OG", "BK"]
def tile(i, H=720):
    im = cv2.imdecode(np.fromfile(find(names[i]), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    s = H / h
    t = cv2.resize(im, (max(1, int(w * s)), H), interpolation=cv2.INTER_AREA)
    if t.shape[1] > 560:
        x0 = (t.shape[1] - 560) // 2; t = t[:, x0:x0 + 560]
    pad = np.zeros((H + 44, 560, 3), np.uint8); pad[:H, :t.shape[1]] = t
    g = " ".join(f"{a}{int(gold[i, k])}" for k, a in enumerate(AB) if gold[i, k] > 0) or "clean"
    cv2.putText(pad, f"#{i} {w}x{h} gold[{g}]", (4, H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(pad, "OG p=" + " ".join(f"{v:.2f}" for v in pp[i, D]), (4, H + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 255, 120), 1)
    return pad
def sheet(tag, idx):
    rows = [np.concatenate([tile(i) for i in idx[r:r + 3]], 1) for r in range(0, len(idx), 3)]
    cv2.imwrite(os.path.join(OUT, f"{tag}.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(tag, [int(i) for i in idx])
y = gold[:, D]
rs = np.random.RandomState(0)
m = np.where(y == 3)[0]; sheet("p0_missed", m[np.argsort(eg[m])[:6]])
sheet("p0_hit", m[np.argsort(-eg[m])[:6]])
m = np.where(y == 0)[0]; sheet("none_falsealarm", m[np.argsort(-eg[m])[:6]])
m = np.where(y == 1)[0]; sheet("p2_called_high", m[np.argsort(-eg[m])[:6]])
m = np.where(y == 2)[0]; sheet("p1_random", rs.choice(m, 6, replace=False))
m = np.where(y == 1)[0]; sheet("p2_random", rs.choice(m, 6, replace=False))
np.savez(os.path.join(OUT, "idx.npz"), names=np.array(names), eg=eg, y=y)
