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
FULL = ['模糊','纹理损失','过锐','过度生成']
G2I = {'无问题':0,'P2':1,'P1':2,'P0':3}

class IQADataset(Dataset):
    def __init__(self, rows, imgdir_orig, imgdir_supp, names_orig, size=384, train=False, nviews=5,
                 bokeh=0.0, bokeh_r=6.0, tiles=2):
        self.rows = rows; self.o = imgdir_orig; self.s = imgdir_supp
        self.names_orig = names_orig; self.size = size; self.train = train; self.nviews = nviews
        self.bokeh = bokeh; self.bokeh_r = bokeh_r; self.tiles = tiles
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
                 bokeh=0.0, bokeh_r=6.0, tiles=2):
        self.rows = rows; self.size = size; self.train = train
        self.bokeh = bokeh; self.bokeh_r = bokeh_r; self.tiles = tiles
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
        if self.train and self.bokeh > 0 and random.random() < self.bokeh and all(int(r[c]) == 0 for c in CLASSES):
            views = [_bokeh_view(v, self.bokeh_r) for v in views]
        x = np.transpose(np.stack([np.asarray(v, dtype=np.float32) for v in views]) / 255.0, (0,3,1,2))
        g = np.array([int(r[c]) for c in CLASSES], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(g), r['name']
