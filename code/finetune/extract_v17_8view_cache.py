#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/extract_v17_8view_cache.py

Extracts 8-view native microscope features (1 global + 4 tiles + 3 native crops)
for a given backbone across all unique images in s16 and s17 manifests.
Saves to /dev/shm/v17_cache_{backbone}.npz.
"""
import os
import sys
import glob
import time
import argparse
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
from models import build_backbone

IMG_DIRS = [
    '/dev/shm/v17_synth_quads',
    '/tmp/v9_authentic_imgs',
    '/tmp/v8_ladder_imgs',
    '/tmp/v7_overgen_pro_imgs',
    '/tmp/v6_synth_imgs',
    'data/train',
    'data/images',
    'data/images_supp',
    'data/test_150',
    'data/test',
]

def find_image(name):
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

class EightViewDataset(Dataset):
    def __init__(self, names, size=448, tiles=2):
        self.names = names
        self.size = size
        self.tiles = tiles

    def __len__(self):
        return len(self.names)

    def _extract_crops(self, im):
        S = self.size
        w, h = im.size
        if w < S or h < S:
            im = im.resize((max(w, S), max(h, S)), Image.BICUBIC)
            w, h = im.size

        xs = [0, max(0, (w - S) // 2), max(0, w - S)] if w > S else [0]
        ys = [0, max(0, (h - S) // 2), max(0, h - S)] if h > S else [0]
        cands = []
        energies = []
        for y0 in ys:
            for x0 in xs:
                patch = im.crop((x0, y0, x0 + S, y0 + S))
                arr = np.asarray(patch, dtype=np.uint8)[::4, ::4, 1].astype(np.float32)
                gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
                gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
                lx = np.abs(2.0 * arr[:, 1:-1] - arr[:, :-2] - arr[:, 2:]).mean()
                ly = np.abs(2.0 * arr[1:-1, :] - arr[:-2, :] - arr[2:, :]).mean()
                cands.append(patch)
                energies.append(float(gx + gy + 0.85 * (lx + ly)))

        if len(cands) == 1:
            return [cands[0], cands[0], cands[0]]

        order = np.argsort(energies)
        p_peak1 = cands[int(order[-1])]
        p_peak2 = cands[int(order[-2])]
        p_mid   = cands[int(order[len(order) // 2])]
        return [p_peak1, p_peak2, p_mid]

    def __getitem__(self, idx):
        nm = self.names[idx]
        p = find_image(nm)
        if p is None:
            im = Image.new('RGB', (self.size, self.size), (128, 128, 128))
        else:
            try:
                im = Image.open(p).convert('RGB')
            except Exception:
                im = Image.new('RGB', (self.size, self.size), (128, 128, 128))

        w, h = im.size
        views = [im.resize((self.size, self.size), Image.BILINEAR)]
        T = self.tiles
        for dy in range(T):
            for dx in range(T):
                views.append(im.crop((dx * w // T, dy * h // T, (dx + 1) * w // T, (dy + 1) * h // T)).resize((self.size, self.size), Image.BILINEAR))
        views.extend(self._extract_crops(im))

        # views shape: (8, 3, 448, 448)
        x = np.stack([np.asarray(v, dtype=np.uint8) for v in views])
        x = np.transpose(x, (0, 3, 1, 2))  # (8, 3, 448, 448)
        return torch.from_numpy(x), nm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    dev = torch.device('cuda:0')
    out_file = args.out or f'/dev/shm/v17_cache_{args.backbone}.npz'
    if os.path.exists(out_file):
        print(f'[Skip] {out_file} already exists.')
        return

    # Gather all unique image names
    df16 = pd.read_csv('data/v17_splits/manifest_s16_v17.csv')
    df17 = pd.read_csv('data/v17_splits/manifest_s17_v17.csv')
    all_names = sorted(list(set(df16['name']).union(set(df17['name']))))
    print(f'[{args.backbone}] Extracting 8-view features for {len(all_names)} unique images...')

    m, fwd, mu, sd, _, dim = build_backbone(args.backbone, device=dev)
    m.eval()
    m.to(dev)

    mu_t = torch.tensor(mu, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=dev).view(1, 3, 1, 1)

    ds = EightViewDataset(all_names, size=448, tiles=2)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    feats_dict = {}
    t0 = time.time()
    with torch.no_grad():
        for step, (x_batch, names_batch) in enumerate(loader):
            # x_batch: (B, 8, 3, 448, 448)
            B, V, C, H, W = x_batch.shape
            x_flat = x_batch.view(B * V, C, H, W).to(dev, non_blocking=True).float() * (1.0 / 255.0)
            x_norm = (x_flat - mu_t) / sd_t
            with torch.autocast('cuda', dtype=torch.bfloat16):
                f_flat = fwd(x_norm)  # (B*8, dim)
            f_batch = f_flat.view(B, V, dim).float().cpu().numpy().astype(np.float32)  # (B, 8, dim)
            for nm, f_arr in zip(names_batch, f_batch):
                feats_dict[nm] = f_arr

            if (step + 1) % 50 == 0 or (step + 1) == len(loader):
                elapsed = time.time() - t0
                fps = (step + 1) * args.batch / elapsed
                print(f'[{args.backbone}] {min((step+1)*args.batch, len(all_names))}/{len(all_names)} done ({fps:.1f} imgs/s, {elapsed:.0f}s)', flush=True)

    print(f'[{args.backbone}] Saving {len(feats_dict)} cached features to {out_file}...')
    names_arr = np.array(list(feats_dict.keys()))
    feats_arr = np.stack([feats_dict[k] for k in names_arr])  # (N, 8, dim)
    np.savez_compressed(out_file, names=names_arr, feats=feats_arr)
    print(f'[{args.backbone}] Successfully cached! File size: {os.path.getsize(out_file)/(1024*1024):.1f} MB in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
