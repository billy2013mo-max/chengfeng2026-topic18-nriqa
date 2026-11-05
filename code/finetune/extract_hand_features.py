"""Hand-crafted physical NR-IQA descriptors (blur / over-sharpening / texture loss / blockiness / bokeh uniformity).
These capture high-frequency behaviour that ViTs lose when the image is resized to 384/448.
Extracted for every labelled image and for the 150 test images (images only, no labels).
"""
import csv, os, sys
import numpy as np, cv2
from multiprocessing import Pool
os.chdir('/wanqing-develop/luowenjing/18')
cv2.setNumThreads(1)
DIRS = ['data/images', 'data/images_supp', 'data/test_150']
MAXSIDE = 1280


def find(name):
    for d in DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p): return p


def feats(name):
    p = find(name)
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if im is None: return name, None
    h, w = im.shape[:2]
    if max(h, w) > MAXSIDE:
        s = MAXSIDE / max(h, w); im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = g.shape
    f = []
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    f += [np.log1p(lap.var()), np.log1p(gm.mean())] + [np.log1p(np.percentile(gm, q)) for q in (50, 90, 99, 99.9)]
    # multi-scale gradient ratios (blur indicator): fine vs coarse
    b1 = cv2.GaussianBlur(g, (0, 0), 1.0); b2 = cv2.GaussianBlur(g, (0, 0), 2.0); b4 = cv2.GaussianBlur(g, (0, 0), 4.0)
    for a, b in ((g, b1), (b1, b2), (b2, b4)):
        d = np.abs(a - b); f += [np.log1p(d.mean()), np.log1p(np.percentile(d, 99))]
    # unsharp-mask residual / halo: signed residual statistics at edges
    res = g - b2
    edge = gm > np.percentile(gm, 90)
    r_e = res[edge]
    f += [np.log1p(np.percentile(np.abs(r_e), 99)) if r_e.size else 0, float((np.abs(r_e) > 25).mean()) if r_e.size else 0,
          float(np.percentile(r_e, 99) / (abs(np.percentile(r_e, 1)) + 1e-3)) if r_e.size else 1,
          float(np.log1p(np.percentile(np.abs(res), 99.9)))]
    # overshoot: pixels beyond the local min/max envelope (ringing)
    mx = cv2.dilate(g, np.ones((5, 5), np.uint8)); mn = cv2.erode(g, np.ones((5, 5), np.uint8))
    near = cv2.dilate((gm > np.percentile(gm, 95)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ring = ((g >= 254) | (g <= 1)) & near
    f += [float(ring.mean()), float((g >= 250).mean()), float((g <= 5).mean())]
    # FFT radial energy bands
    sub = g[:min(H, 512), :min(W, 512)]
    F = np.abs(np.fft.fftshift(np.fft.fft2(sub - sub.mean()))) ** 2
    yy, xx = np.indices(F.shape); rr = np.sqrt((yy - F.shape[0] / 2) ** 2 + (xx - F.shape[1] / 2) ** 2) / (min(F.shape) / 2)
    tot = F.sum() + 1e-6
    f += [float(np.log((F[(rr >= a) & (rr < b)].sum() + 1e-6) / tot)) for a, b in ((0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.0))]
    # noise: MAD of Laplacian in flat regions
    flat = gm < np.percentile(gm, 30)
    f += [float(np.log1p(1.4826 * np.median(np.abs(lap[flat] - np.median(lap[flat]))))) if flat.any() else 0]
    # JPEG-like blockiness (8x8 grid discontinuity)
    dx = np.abs(np.diff(g, axis=1)); dy = np.abs(np.diff(g, axis=0))
    bx = dx[:, 7::8].mean() / (dx.mean() + 1e-3) if dx.shape[1] > 8 else 1.0
    by = dy[7::8, :].mean() / (dy.mean() + 1e-3) if dy.shape[0] > 8 else 1.0
    f += [float(bx), float(by)]
    # spatial uniformity of sharpness (defocus / bokeh): tile Laplacian variance on 6x6 grid
    tv = []
    for i in range(6):
        for j in range(6):
            t = lap[i * H // 6:(i + 1) * H // 6, j * W // 6:(j + 1) * W // 6]
            tv.append(np.log1p(t.var()))
    tv = np.array(tv)
    f += [tv.max(), tv.min(), tv.std(), tv.max() - tv.min(), float(np.mean(tv[14:22])) - float(np.mean(np.r_[tv[:6], tv[30:]])), float((tv < tv.max() - 2.5).mean())]
    # intensity / colour stats
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    f += [g.mean() / 255, g.std() / 255, hsv[..., 1].mean() / 255, float(np.std(im.reshape(-1, 3).mean(1)) / 255)]
    # texture entropy: local std histogram entropy
    ls = np.sqrt(np.maximum(cv2.blur(g * g, (7, 7)) - cv2.blur(g, (7, 7)) ** 2, 0))
    hist, _ = np.histogram(ls, bins=32, range=(0, 64)); pr = hist / hist.sum() + 1e-9
    f += [float(-(pr * np.log(pr)).sum()), float((ls < 2).mean()), float((ls > 30).mean())]
    f += [np.log(h * w) / 10, w / h]  # source resolution / aspect (kept: grouped CV checks it does not hurt)
    return name, np.array(f, dtype=np.float32)


if __name__ == '__main__':
    names = [r['name'] for r in csv.DictReader(open('data/split_all_full.csv'))]
    names += os.listdir('data/test_150')
    names = list(dict.fromkeys(names))
    with Pool(96) as pool:
        out = pool.map(feats, names, chunksize=8)
    ok = [(n, v) for n, v in out if v is not None]
    print('extracted', len(ok), 'of', len(names), 'dim', ok[0][1].shape[0])
    np.savez_compressed('results/hand_features.npz', names=np.array([n for n, _ in ok]), X=np.stack([v for _, v in ok]))
