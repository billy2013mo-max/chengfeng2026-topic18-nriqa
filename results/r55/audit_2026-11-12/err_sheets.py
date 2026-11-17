import os, sys, numpy as np, cv2
REPO = "/wanqing-develop/luowenjing/18"
for p in ("code", "code/next55", "code/r55"):
    sys.path.insert(0, os.path.join(REPO, p))
from final_engine import SETS, oof_pool
OUT = sys.argv[1]; D = int(sys.argv[2])
names, gold, fold, pp, _ = oof_pool(SETS["primary"]); gold = np.asarray(gold); pp = np.asarray(pp)
AB = ["BL", "TX", "SH", "GL", "OG", "BK"]
def find(n):
    for d in ("data/images", "data/images_supp"):
        p = os.path.join(REPO, d, n)
        if os.path.exists(p): return p
def tile(i, S=560):
    im = cv2.imdecode(np.fromfile(find(names[i]), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    # left: whole frame fitted to S x S ; right: native centre crop S x S
    s = min(S / h, S / w)
    t = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    a = np.zeros((S, S, 3), np.uint8); a[:t.shape[0], :t.shape[1]] = t
    c = min(S, h, w); y0 = (h - c) // 2; x0 = (w - c) // 2
    b = np.zeros((S, S, 3), np.uint8); b[:c, :c] = im[y0:y0 + c, x0:x0 + c]
    pad = np.zeros((S + 24, 2 * S, 3), np.uint8); pad[:S, :S] = a; pad[:S, S:] = b
    g = " ".join(f"{k}{int(gold[i, j])}" for j, k in enumerate(AB) if gold[i, j] > 0) or "clean"
    cv2.putText(pad, f"#{i} {w}x{h} gold[{g}] {AB[D]} p=" + " ".join(f"{v:.2f}" for v in pp[i, D]), (4, S + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return pad
def sheet(tag, idx):
    cv2.imwrite(os.path.join(OUT, f"{AB[D]}_{tag}.jpg"), np.concatenate([tile(i) for i in idx], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(tag, [int(i) for i in idx])
y = gold[:, D]; rs = np.random.RandomState(1)
m = np.where(y == 1)[0]; o = m[np.argsort(-pp[m, D, 0])]; sheet("p2_called_none", rs.choice(o[:60], 3, replace=False))
m = np.where(y == 0)[0]; o = m[np.argsort(pp[m, D, 0])]; sheet("none_called_defect", rs.choice(o[:60], 3, replace=False))
m = np.where(y == 1)[0]; o = m[np.argsort(pp[m, D, 0])]; sheet("p2_hit", rs.choice(o[:60], 3, replace=False))
