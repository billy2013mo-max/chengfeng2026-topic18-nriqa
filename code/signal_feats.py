"""Cheap NR signal features (no deep model) as a low-level reference baseline."""
import argparse, csv, os, numpy as np, cv2

def feat(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        from PIL import Image
        img = cv2.cvtColor(np.asarray(Image.open(path).convert('RGB')), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    s = 512.0 / min(h, w)
    img = cv2.resize(img, (max(1,int(w*s)), max(1,int(h*s))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = []
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    f += [lap.var(), np.abs(lap).mean(), np.percentile(np.abs(lap), 90), np.percentile(np.abs(lap), 99)]
    for p in (50, 75, 90, 99): f.append(np.percentile(mag, p))
    f += [mag.mean() / (g.std()+1e-3), (mag > 32).mean(), (mag > 64).mean()]
    # local sharpness map 32x32 -> variance of log-sharpness (foreground/background split)
    H, W = (g.shape[0]//32)*32, (g.shape[1]//32)*32
    if H >= 64 and W >= 64:
        lv = lap[:H, :W].reshape(H//32, 32, W//32, 32).var(axis=(1,3))
        ls = np.log(lv + 1e-3)
        f += [ls.std(), ls.max()-np.median(ls), np.percentile(ls, 95)-np.percentile(ls, 5)]
        gxs = cv2.Sobel(ls.astype(np.float32), cv2.CV_32F, 1, 0); gys = cv2.Sobel(ls.astype(np.float32), cv2.CV_32F, 0, 1)
        dm = np.hypot(gxs, gys); f += [np.percentile(dm, 95), np.percentile(dm, 99), dm.max()]
    else:
        f += [0,0,0,0,0,0]
    # spectrum slope / high-freq ratio
    gf = np.fft.fftshift(np.abs(np.fft.fft2(g)))
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    r = np.hypot(yy-g.shape[0]/2, xx-g.shape[1]/2); rmax = r.max()
    tot = gf.sum()+1e-6
    f += [gf[r/rmax < .05].sum()/tot, gf[(r/rmax >= .05)&(r/rmax < .2)].sum()/tot,
          gf[(r/rmax >= .2)&(r/rmax < .5)].sum()/tot, gf[r/rmax >= .5].sum()/tot]
    # blockiness (8px grid)
    d = np.abs(np.diff(g, axis=1)); col = d.mean(0)
    bnd = col[7::8].mean() if col[7::8].size else 0; inn = col.mean()+1e-3
    f += [bnd/inn]
    d2 = np.abs(np.diff(g, axis=0)); row = d2.mean(1)
    bnd2 = row[7::8].mean() if row[7::8].size else 0
    f += [bnd2/(row.mean()+1e-3)]
    # overshoot / ringing near strong edges
    strong = mag > np.percentile(mag, 99)
    if strong.any():
        f += [np.abs(lap[strong]).mean(), np.abs(lap[strong]).max()]
    else: f += [0,0]
    # color stats
    b,gc,rr = img[:,:,0].astype(np.float32), img[:,:,1].astype(np.float32), img[:,:,2].astype(np.float32)
    f += [((rr>250)|(gc>250)|(b>250)).mean(), ((rr<5)|(gc<5)|(b<5)).mean()]
    f += [np.sqrt(rr.var()+gc.var()+b.var() + 0.3*((rr-gc)**2+(rr-b)**2+(gc-b)**2).mean())]
    f += [img.shape[0]/img.shape[1], np.log(w*h)]
    return np.array(f, dtype=np.float32)

ap = argparse.ArgumentParser()
ap.add_argument('--manifest'); ap.add_argument('--imgdir'); ap.add_argument('--out')
a = ap.parse_args()
rows = list(csv.DictReader(open(a.manifest)))
X = []
for i, r in enumerate(rows):
    X.append(feat(os.path.join(a.imgdir, r['name'])))
    if i % 400 == 0: print(i, flush=True)
X = np.stack(X)
np.savez_compressed(a.out, X=X, names=np.array([r['name'] for r in rows]))
print('saved', a.out, X.shape)
