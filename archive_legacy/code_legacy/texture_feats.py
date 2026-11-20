"""Texture / photorealism statistics at native resolution.

Motivation: global pooled ViT features are nearly invariant to AI re-rendering
(plastic skin, smeared micro-texture), which is exactly the low-severity
"过度生成" signal. These hand-crafted statistics measure the high-frequency
content directly: noise floor, Laplacian kurtosis, spectral slope, local
variance distribution, high-frequency energy ratio.
"""
import argparse, csv, os, numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='feats/texture.npz')
ap.add_argument('--crop', type=int, default=512)
a = ap.parse_args()

idx = np.load('/dev/shm/race18_native/index.npz', allow_pickle=True)
names = list(idx['names']); offs = idx['offsets']; hs = idx['h']; ws = idx['w']
buf = np.memmap('/dev/shm/race18_native/cache.bin', dtype=np.uint8, mode='r')


def feats(g):
    f = []
    g = g.astype(np.float32)
    # multi-scale Gaussian residuals (high-pass)
    for sig in (1.0, 2.5):
        b = cv2.GaussianBlur(g, (0, 0), sig)
        r = g - b
        f += [r.std(), np.abs(r).mean(), np.percentile(np.abs(r), 99),
              float(((r - r.mean()) ** 4).mean() / (r.var() ** 2 + 1e-9)),   # kurtosis
              float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-9))]   # skew
    # Laplacian
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    f += [lap.std(), np.abs(lap).mean(), np.percentile(np.abs(lap), 99),
          float(((lap - lap.mean()) ** 4).mean() / (lap.var() ** 2 + 1e-9))]
    # noise estimate (MAD of high-pass / 0.6745)
    r1 = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    f += [float(np.median(np.abs(r1 - np.median(r1))) / 0.6745)]
    # local variance distribution on 8x8 blocks
    h, w = g.shape
    H, W = (h // 8) * 8, (w // 8) * 8
    blk = g[:H, :W].reshape(H // 8, 8, W // 8, 8).var(axis=(1, 3)).ravel()
    lb = np.log(blk + 1e-3)
    f += [lb.std(), np.percentile(lb, 5), np.percentile(lb, 95), np.percentile(lb, 95) - np.percentile(lb, 5)]
    # spectrum: radial energy ratio + log-log slope
    z = g - g.mean()
    win = np.outer(np.hanning(z.shape[0]), np.hanning(z.shape[1]))
    F = np.abs(np.fft.fftshift(np.fft.fft2(z * win))) ** 2
    cy, cx = np.array(F.shape) // 2
    yy, xx = np.mgrid[0:F.shape[0], 0:F.shape[1]]
    rad = np.hypot(yy - cy, xx - cx) / min(cy, cx)
    tot = F.sum() + 1e-9
    for lo, hi in [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]:
        m = (rad >= lo) & (rad < hi)
        f.append(F[m].sum() / tot)
    m = (rad > 0.05) & (rad < 0.9)
    lr = np.log(rad[m] + 1e-6); lf = np.log(F[m] + 1e-9)
    f.append(float(np.polyfit(lr, lf, 1)[0]))
    return f


X, out_names = [], []
for i, n in enumerate(names):
    h, w = int(hs[i]), int(ws[i])
    arr = np.asarray(buf[int(offs[i]):int(offs[i]) + h * w * 3]).reshape(h, w, 3)
    c = min(h, w, a.crop)
    y0 = (h - c) // 2; x0 = (w - c) // 2
    crop = arr[y0:y0 + c, x0:x0 + c]
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    X.append(feats(g)); out_names.append(n)
    if (i + 1) % 500 == 0:
        print(i + 1, flush=True)
X = np.array(X, dtype=np.float32)
np.savez_compressed(a.out, X=X, names=np.array(out_names))
print('saved', a.out, X.shape)
