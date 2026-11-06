import csv, os, random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

CLASSES = ['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']


def _bokeh_view(im, r):
    """Simulate a wide-aperture / beauty-camera look: sharp centre, blurred surround."""
    w, h = im.size
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).ellipse((w * 0.15, h * 0.08, w * 0.85, h * 0.92), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1.0, r * 2)))
    return Image.composite(im, im.filter(ImageFilter.GaussianBlur(r)), mask)


def _window_sums(energy, ws):
    """Sum over every ws x ws window (valid positions) using an integral image."""
    c = np.cumsum(np.cumsum(energy, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return c[ws:, ws:] - c[:-ws, ws:] - c[ws:, :-ws] + c[:-ws, :-ws]


def _native_patches(im, size=448, n=3):
    """Zero-resize 1:1 native-pixel microscope patches.

    Returns n crops at native resolution:
      - peak Sobel-gradient window (sharp edges / 1-2px ringing)
      - peak Laplacian window (overshoot / halo)
      - median-texture window (texture-loss / over-smoothing)
    The search is done on a ~256px copy with an integral image; the crops
    themselves are taken from the original pixels (no downsampling).
    """
    w, h = im.size
    if w < size or h < size:
        p = im.resize((size, size), Image.BILINEAR)
        return [p] * n

    gray = np.asarray(im.convert('L'), dtype=np.float32) / 255.0
    scale = max(1, int(np.ceil(max(w, h) / 256.0)))
    small = gray[::scale, ::scale]
    if small.shape[0] <= 8 or small.shape[1] <= 8:
        p = im.resize((size, size), Image.BILINEAR)
        return [p] * n

    gy, gx = np.gradient(small)
    grad = np.hypot(gx, gy)
    lap = np.abs(
        np.gradient(np.gradient(small, axis=0), axis=0)
        + np.gradient(np.gradient(small, axis=1), axis=1)
    )
    ws = max(4, min(int(round(size / scale)), min(small.shape) - 1))
    Sg = _window_sums(grad, ws)
    Sl = _window_sums(lap, ws)
    if Sg.size == 0 or Sl.size == 0:
        p = im.resize((size, size), Image.BILINEAR)
        return [p] * n

    taken = []

    def pick(E, taken_):
        for idx in np.argsort(E.ravel())[::-1]:
            y, x = divmod(int(idx), E.shape[1])
            cy, cx = y + ws / 2.0, x + ws / 2.0
            if all((cy - ty) ** 2 + (cx - tx) ** 2 >= (ws * 0.7) ** 2 for ty, tx in taken_):
                return y, x, cy, cx
        y, x = divmod(int(np.argmax(E)), E.shape[1])
        return y, x, y + ws / 2.0, x + ws / 2.0

    p1 = pick(Sg, taken); taken.append((p1[2], p1[3]))
    p2 = pick(Sl, taken); taken.append((p2[2], p2[3]))

    med = float(np.median(Sg))
    p3 = None
    for idx in np.argsort(np.abs(Sg.ravel() - med)):
        y, x = divmod(int(idx), Sg.shape[1])
        cy, cx = y + ws / 2.0, x + ws / 2.0
        if all((cy - ty) ** 2 + (cx - tx) ** 2 >= (ws * 0.7) ** 2 for ty, tx in taken):
            p3 = (y, x)
            break
    if p3 is None:
        p3 = p1[:2]

    patches = []
    for sy, sx in [p1[:2], p2[:2], p3]:
        x0, y0 = int(sx * scale), int(sy * scale)
        x1, y1 = min(w, x0 + size), min(h, y0 + size)
        x0, y0 = max(0, x1 - size), max(0, y1 - size)
        patches.append(im.crop((x0, y0, x1, y1)).resize((size, size), Image.BILINEAR))
    return patches
FULL = ['模糊','纹理损失','过锐','过度生成']
G2I = {'无问题':0,'P2':1,'P1':2,'P0':3}

class IQADataset(Dataset):
    def __init__(self, rows, imgdir_orig, imgdir_supp, names_orig, size=384, train=False, nviews=5,
                 bokeh=0.0, bokeh_r=6.0, tiles=2, native=0):
        self.rows = rows; self.o = imgdir_orig; self.s = imgdir_supp
        self.names_orig = names_orig; self.size = size; self.train = train; self.nviews = nviews
        self.bokeh = bokeh; self.bokeh_r = bokeh_r; self.tiles = tiles; self.native = native
    def __len__(self): return len(self.rows)
    def _path(self, name):
        return os.path.join(self.o, name) if name in self.names_orig else os.path.join(self.s, name)
    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(self._path(r['name'])).convert('RGB')
        if self.train:
            if random.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                im = ImageEnhance.Brightness(im).enhance(1.0 + random.uniform(-0.08, 0.08))
            if random.random() < 0.5:
                im = ImageEnhance.Contrast(im).enhance(1.0 + random.uniform(-0.08, 0.08))
        w, h = im.size
        views = [im.resize((self.size, self.size), Image.BILINEAR)]
        T = self.tiles
        for dy in range(T):
            for dx in range(T):
                views.append(im.crop((dx*w//T, dy*h//T, (dx+1)*w//T, (dy+1)*h//T)).resize((self.size, self.size), Image.BILINEAR))
        if self.native > 0:
            views.extend(_native_patches(im, self.size, self.native))
        if self.train and self.bokeh > 0 and random.random() < self.bokeh and all(int(r[c]) == 0 for c in CLASSES):
            views = [_bokeh_view(v, self.bokeh_r) for v in views]
        x = np.transpose(np.stack([np.asarray(v, dtype=np.float32) for v in views]) / 255.0, (0,3,1,2))   # V,3,S,S
        g = np.array([int(r[c]) for c in CLASSES], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(g), r['name']

def load_split(path):
    rows = list(csv.DictReader(open(path)))
    return rows

def read_manifest(path):
    return {r['name']: r for r in csv.DictReader(open(path))}

def targets_from_grades(g):
    """g: (B,6) int -> (B,14) cumulative targets in comp_metrics order."""
    out = []
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            # order must be P2cum, P1cum, P0 (comp_metrics.LABELS order):
            # exactly-P2 = P(>=1) - P(>=2), exactly-P1 = P(>=2) - P(==3)
            out += [(g[:,ci] >= 1).float(), (g[:,ci] >= 2).float(), (g[:,ci] == 3).float()]
        else:
            out += [(g[:,ci] == 3).float()]
    return torch.stack(out, 1)

class CachedIQADataset(Dataset):
    """Same view pipeline as IQADataset, but reads native-resolution pixels from a lossless memmap cache."""
    def __init__(self, rows, size=384, train=False,
                 cache_bin='/dev/shm/race18_native/cache.bin', index='/dev/shm/race18_native/index.npz',
                 bokeh=0.0, bokeh_r=6.0, tiles=2, native=0):
        self.rows = rows; self.size = size; self.train = train
        self.bokeh = bokeh; self.bokeh_r = bokeh_r; self.tiles = tiles; self.native = native
        d = np.load(index, allow_pickle=True)
        self.names = list(d['names']); self.offsets = d['offsets']; self.h = d['h']; self.w = d['w']
        self.pos = {n: i for i, n in enumerate(self.names)}
        self.data = np.memmap(cache_bin, dtype=np.uint8, mode='r')
    def __len__(self): return len(self.rows)
    def _load(self, name):
        if name in self.pos:
            j = self.pos[name]; h = int(self.h[j]); w = int(self.w[j]); off = int(self.offsets[j])
            arr = np.asarray(self.data[off:off + h * w * 3]).reshape(h, w, 3)
            return Image.fromarray(arr)
        for d in ['data/synth_overgen_pro_full', 'data/synth_overgen_pro_s16', 'data/synth_overgen_pro_s17', 'data/synth_v6_full', 'data/synth_v6_s16', 'data/synth_v6_s17', 'data/synth_sharp_full', 'data/synth_sharp_s16', 'data/synth_sharp_s17', 'data/images', 'data/images_supp', 'data/synth_sharp_sample']:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
        raise FileNotFoundError(f"Cannot find {name}")
    def __getitem__(self, i):
        r = self.rows[i]
        im = self._load(r['name'])
        if self.train:
            if random.random() < 0.5: im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                im = ImageEnhance.Brightness(im).enhance(1.0 + random.uniform(-0.08, 0.08))
            if random.random() < 0.5:
                im = ImageEnhance.Contrast(im).enhance(1.0 + random.uniform(-0.08, 0.08))
        w, h = im.size
        views = [im.resize((self.size, self.size), Image.BILINEAR)]
        T = self.tiles
        for dy in range(T):
            for dx in range(T):
                views.append(im.crop((dx*w//T, dy*h//T, (dx+1)*w//T, (dy+1)*h//T)).resize((self.size, self.size), Image.BILINEAR))
        if self.native > 0:
            views.extend(_native_patches(im, self.size, self.native))
        if self.train and self.bokeh > 0 and random.random() < self.bokeh and all(int(r[c]) == 0 for c in CLASSES):
            views = [_bokeh_view(v, self.bokeh_r) for v in views]
        x = np.transpose(np.stack([np.asarray(v, dtype=np.float32) for v in views]) / 255.0, (0,3,1,2))
        g = np.array([int(r[c]) for c in CLASSES], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(g), r['name']
