import os, sys, numpy as np, cv2
REPO = "/wanqing-develop/luowenjing/18"
OUT = sys.argv[1]; tag = sys.argv[2]
z = np.load(os.path.join(OUT, "idx.npz"), allow_pickle=True); names = [str(n) for n in z["names"]]
def find(n):
    for d in ("data/images", "data/images_supp"):
        p = os.path.join(REPO, d, n)
        if os.path.exists(p): return p
tiles = []
for spec in sys.argv[3:]:
    i, fx, fy, sz = spec.split(":"); i = int(i); fx = float(fx); fy = float(fy); sz = int(sz)
    im = cv2.imdecode(np.fromfile(find(names[i]), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = im.shape[:2]; sz = min(sz, h, w)
    x0 = int(np.clip(fx * w - sz / 2, 0, w - sz)); y0 = int(np.clip(fy * h - sz / 2, 0, h - sz))
    c = im[y0:y0 + sz, x0:x0 + sz]
    c = cv2.resize(c, (560, 560), interpolation=cv2.INTER_CUBIC if sz < 560 else cv2.INTER_AREA)
    cv2.putText(c, f"#{i} crop {sz}px", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    tiles.append(c)
rows = [np.concatenate(tiles[r:r + 3], 1) for r in range(0, len(tiles), 3)]
cv2.imwrite(os.path.join(OUT, f"zoom_{tag}.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
