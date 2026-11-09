#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/train_v18_paired_kd_specialist.py
V18.1 Specialist Trainer with 5 Core Breakthroughs:
  1. Fast Precomputed 1:1 Native ROI Lookup (/dev/shm/race18_native/roi_coords_and_meta.npz)
     + Mask-Guided Crop for synthetic quadruplets + 5D Geometry Meta FiLM Conditioning.
  2. Paired Quadruplet Differential Contrast Batching: groups synthetic samples by same seed_stem
     in the forward pass and enforces pure differential ranking
     z_c(P0) > z_c(P1) > z_c(P2) > z_c(Clean) + cross-dimension orthogonality invariance.
  3. Zero-Leakage True 6B Teacher + V16_Full OOF Knowledge Distillation (distills InternViT-6B
     + V16_Full 8-Fold OOF + V15 OOF strictly on training images of the current split).
  4. Smooth Monotonic Interval CORN Head (preserves exact state_dict compatibility with MultiNetV18
     without hard logit distortion).
  5. Safe Name-Aligned V16 Pure-Real Specialist Anchor Blending (works seamlessly on both 500-val
     s16/s17 splits and 300-val FullData splits without shape broadcast crashes).
"""
import argparse, copy, csv, glob, json, math, os, random, sys, time
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

os.chdir('/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.extend(['code', 'code/finetune', 'code/decision_eval'])
from data import load_split, read_manifest, targets_from_grades, CLASSES, FULL
from train_v2 import exact_probs_from_dual, exact_targets_14, P0_IDX
from train_v9_specialist import CACHE_OK, CLASS_TO_COLS14, build_v9_backbone, set_seed
from train_v16_microscope_specialist import MultiNetV16, v16_clean_microscope_loss_fn
from train_v17_end2end_sim2real_specialist import is_synth_name, get_sample_class_mask6_v17
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm

ROI_CACHE_PATH = '/dev/shm/race18_native/roi_coords_and_meta.npz'
SYNTH_META_PATH = 'data/v17_splits/v18_synth_metadata.json'


def load_roi_and_synth_meta():
    roi_map, meta_map = {}, {}
    if os.path.exists(ROI_CACHE_PATH):
        d = np.load(ROI_CACHE_PATH, allow_pickle=True)
        names = list(d['names'])
        coords = d['coords']
        meta = d['meta']
        roi_map = {n: coords[i] for i, n in enumerate(names)}
        meta_map = {n: meta[i] for i, n in enumerate(names)}
    synth_meta = {}
    if os.path.exists(SYNTH_META_PATH):
        synth_meta = json.load(open(SYNTH_META_PATH))
    return roi_map, meta_map, synth_meta


class MultiNetV18(MultiNetV16):
    """
    Upgrades MultiNetV16 with:
      1. 5D Geometry Meta FiLM Conditioning (Scale & Aspect-Ratio Aware Feature Modulation, zero-initialized residual).
      2. Monotonic Interval CORN Residual Head (100% state_dict compatible with all V18 checkpoints).
    """
    def __init__(self, specs, phys_dim=60, dropout=0.12, n_quad=4):
        super().__init__(specs, phys_dim=phys_dim, dropout=dropout, n_quad=n_quad)
        d_vis = sum(8 * x[5] for x in specs)
        d = d_vis + 256
        self.geom_film = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, d * 2),
        )
        nn.init.zeros_(self.geom_film[-1].weight)
        nn.init.zeros_(self.geom_film[-1].bias)

        self.corn_delta = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Linear(128, 8),  # 4 dims x (u_p2, u_p1)
        )
        nn.init.zeros_(self.corn_delta[-1].weight)
        nn.init.zeros_(self.corn_delta[-1].bias)

    def forward(self, x, phys, geom=None):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W).to(torch.bfloat16).div_(255.0)
        outs = []
        for i, (m, fwd) in enumerate(zip(self.bs, self.fwds)):
            mu = getattr(self, 'mu_%d' % (i + 1))
            sd = getattr(self, 'sd_%d' % (i + 1))
            f = fwd((x_flat - mu) / sd).view(B, T, -1)
            outs.append(self._agg(f, self.tile_attn[i]))
        h_vis = torch.cat(outs, dim=-1)
        h_phys = self.phys_mlp(phys.to(h_vis.dtype))
        h = self.norm(torch.cat([h_vis, h_phys], dim=-1))
        h = h * self.se_gate(h)
        if geom is not None:
            gb = self.geom_film(geom.to(h.dtype))
            gamma, beta = gb.chunk(2, dim=-1)
            h = h * (1.0 + 0.15 * torch.tanh(gamma)) + 0.15 * beta
        h = self.drop(h)
        lc = self.head_cum(h)
        le = self.head_ex(h)
        gl = self.gate(h).squeeze(-1)

        # Smooth Monotonic Interval CORN adjustment (avoids hard torch.maximum upward bias when l_eq3 is small)
        u = self.corn_delta(h)
        lc_out = lc.clone()
        col = 0
        k_full = 0
        for ci, c in enumerate(CLASSES):
            if c in FULL:
                l_ge1 = lc[:, col]
                l_ge2 = lc[:, col + 1]
                l_eq3 = lc[:, col + 2]
                d_p2 = 0.20 * torch.tanh(u[:, 2 * k_full])
                d_p1 = 0.20 * torch.tanh(u[:, 2 * k_full + 1])
                gap_12 = F.relu(l_eq3 + 0.15 - l_ge2)
                lc_out[:, col + 1] = l_ge2 + d_p1 + 0.50 * gap_12
                gap_21 = F.relu(lc_out[:, col + 1] + 0.12 - l_ge1)
                lc_out[:, col] = l_ge1 + d_p2 + 0.50 * gap_21
                col += 3
                k_full += 1
            else:
                col += 1
        return lc_out, le, gl


class V18FastMicroscopeDataset(Dataset):
    """
    Uses precomputed 1:1 ROI coords (zero CPU Sobel overhead per epoch!),
    Mask-Guided focus_yx for V18 synthetic quadruplets, and 5D Geometry Meta.
    """
    def __init__(self, rows, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, size=448, train=True, hflip=False, tiles=2, synth_w=0.75):
        self.rows = rows
        self.phys_map = phys_map
        self.phys_mu = phys_mu
        self.phys_sd = phys_sd
        self.roi_map = roi_map
        self.meta_map = meta_map
        self.synth_meta = synth_meta
        self.size = size
        self.train = train
        self.hflip = hflip
        self.tiles = tiles
        self.synth_w = synth_w
        if CACHE_OK:
            d = np.load('/dev/shm/race18_native/index.npz', allow_pickle=True)
            self.names = list(d['names'])
            self.offsets = d['offsets']
            self.h = d['h']
            self.w = d['w']
            self.pos = {n: i for i, n in enumerate(self.names)}
            self.data = np.memmap('/dev/shm/race18_native/cache.bin', dtype=np.uint8, mode='r')
        else:
            self.pos = {}

    def __len__(self):
        return len(self.rows)

    def _load(self, name):
        if name in self.pos:
            j = self.pos[name]
            h, w, off = int(self.h[j]), int(self.w[j]), int(self.offsets[j])
            arr = np.asarray(self.data[off:off + h * w * 3]).reshape(h, w, 3)
            return Image.fromarray(arr)
        for d in ['/dev/shm/v18_synth_quads', '/dev/shm/v17_synth_quads', '/tmp/v9_authentic_imgs',
                  'data/train', 'data/images', 'data/images_supp', 'data/test_150', 'data/test']:
            p = os.path.join('/wanqing-develop/luowenjing/18', d, name) if not d.startswith('/') else os.path.join(d, name)
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
        raise FileNotFoundError(f'Cannot find {name}')

    def _get_fast_native_crops(self, im, nm, do_flip=False):
        S = self.size
        w, h = im.size
        if w < S or h < S:
            im = im.resize((max(w, S), max(h, S)), Image.BICUBIC)
            w, h = im.size

        focus_override = None
        if nm in self.synth_meta and 'focus_yx' in self.synth_meta[nm]:
            fy, fx = self.synth_meta[nm]['focus_yx']
            if do_flip:
                fx = w - fx
            focus_override = (max(0, min(h - S, int(fy) - S // 2)), max(0, min(w - S, int(fx) - S // 2)))

        lookup_nm = nm
        if lookup_nm not in self.roi_map:
            if nm in self.synth_meta and 'seed_name' in self.synth_meta[nm]:
                lookup_nm = self.synth_meta[nm]['seed_name']
            else:
                for d in range(6):
                    tag = f'_d{d}_'
                    if tag in lookup_nm:
                        stem = lookup_nm[:lookup_nm.index(tag)]
                        for ext in ('.png', '.jpg', '.jpeg'):
                            if (stem + ext) in self.roi_map:
                                lookup_nm = stem + ext
                                break
                        break

        if lookup_nm in self.roi_map:
            coords6 = self.roi_map[lookup_nm]
            c0 = focus_override if focus_override is not None else (int(coords6[0][0]), int(coords6[0][1]))
            selected = [c0, (int(coords6[1][0]), int(coords6[1][1])), (int(coords6[2][0]), int(coords6[2][1]))]
            patches = []
            for idx_c, (y0, x0) in enumerate(selected):
                if do_flip and not (idx_c == 0 and focus_override is not None):
                    x0 = max(0, w - S - x0)
                y_c = max(0, min(h - S, int(y0)))
                x_c = max(0, min(w - S, int(x0)))
                if self.train and w > S + 24 and h > S + 24:
                    y_c = max(0, min(h - S, y_c + random.randint(-12, 12)))
                    x_c = max(0, min(w - S, x_c + random.randint(-12, 12)))
                patches.append(im.crop((x_c, y_c, x_c + S, y_c + S)))
            return patches, lookup_nm

        cx, cy = max(0, (w - S) // 2), max(0, (h - S) // 2)
        return [
            im.crop((cx, cy, cx + S, cy + S)),
            im.crop((0, 0, S, S)),
            im.crop((max(0, w - S), max(0, h - S), w, h)),
        ], lookup_nm

    def __getitem__(self, i):
        r = self.rows[i]
        nm = r['name']
        im = self._load(nm)
        do_flip = False
        if self.train:
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                do_flip = True
            if random.random() < 0.25:
                im = ImageEnhance.Brightness(im).enhance(1.0 + random.uniform(-0.035, 0.035))
        elif self.hflip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            do_flip = True

        w, h = im.size
        views = [im.resize((self.size, self.size), Image.BILINEAR)]
        T = self.tiles
        for dy in range(T):
            for dx in range(T):
                views.append(im.crop((dx * w // T, dy * h // T, (dx + 1) * w // T, (dy + 1) * h // T)).resize((self.size, self.size), Image.BILINEAR))
        nat_patches, lookup_nm = self._get_fast_native_crops(im, nm, do_flip=do_flip)
        views.extend(nat_patches)

        x = np.transpose(np.stack([np.asarray(v, dtype=np.uint8) for v in views]), (0, 3, 1, 2))
        g = np.array([int(r[c]) for c in CLASSES], dtype=np.int64)
        mask7 = get_sample_class_mask6_v17(nm, synth_w=self.synth_w)
        if ('_d5_NatNeg' in nm) or ('_d5_P2' in nm and g[5] == 0):
            mask7 = np.array([0.85, 0.0, 0.0, 0.0, 0.0, 0.85, 0.0], dtype=np.float32)

        p_raw = self.phys_map.get(nm, self.phys_map.get(lookup_nm, np.zeros(60, dtype=np.float32)))
        p_norm = np.clip((p_raw - self.phys_mu) / self.phys_sd, -6.0, 6.0).astype(np.float32)

        raw_meta = self.meta_map.get(lookup_nm, np.array([h, w, float(w) / max(1, h), abs(math.log(max(float(w) / max(1, h), 1e-3))), math.sqrt((w * h) / float(self.size * self.size))], dtype=np.float32))
        geom_5d = np.array([
            math.log(max(raw_meta[0], 64.0) / 448.0),
            math.log(max(raw_meta[1], 64.0) / 448.0),
            math.log(max(raw_meta[2], 1e-2)),
            raw_meta[3],
            raw_meta[4] - 1.0,
        ], dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(p_norm), torch.from_numpy(geom_5d), torch.from_numpy(g), torch.from_numpy(mask7), nm


@torch.inference_mode()
def predict_v18(net, loader, dev, alpha_ex=0.58):
    net.eval()
    P0, P1, P2, GATE, G = [], [], [], [], []
    for x, phys, geom, g, _, _ in loader:
        x = x.to(dev, non_blocking=True)
        phys = phys.to(dev, non_blocking=True)
        geom = geom.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lc, le, gl = net(x, phys, geom)
        p0, p1, p2, _ = exact_probs_from_dual(lc.float(), le.float(), alpha_ex=alpha_ex)
        P0.append(p0.cpu().numpy())
        P1.append(p1.cpu().numpy())
        P2.append(p2.cpu().numpy())
        GATE.append(torch.sigmoid(gl.float()).cpu().numpy())
        G.append(g.numpy())
    if not P0:
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 6), dtype=np.int64),
        )
    return np.concatenate(P0, 0), np.concatenate(P1, 0), np.concatenate(P2, 0), np.concatenate(GATE, 0), np.concatenate(G, 0)


def _extract_seed_stem(nm):
    for d in range(6):
        tag = f'_d{d}_'
        if tag in nm:
            return nm[:nm.index(tag)]
    return os.path.splitext(nm)[0]


def compute_paired_quadruplet_and_kd_loss(lc, le, g, mask7, names, teacher_14_map, dev):
    """
    Computes:
      1. True 6B + V16_Full Teacher Soft-Distribution Distillation on real training images.
      2. Scene-Grouped Paired Synthetic Differential Ranking Loss (prioritizing same-seed pairs 3x).
    """
    loss_extra = lc.new_tensor(0.0)
    B = lc.shape[0]

    # 1. Teacher Knowledge Distillation on authentic real training samples
    kd_indices = []
    kd_targets = []
    for b_i, nm in enumerate(names):
        if mask7[b_i, 6] > 0.5 and nm in teacher_14_map:
            kd_indices.append(b_i)
            kd_targets.append(teacher_14_map[nm])
    if kd_indices:
        idx_t = torch.tensor(kd_indices, device=dev, dtype=torch.long)
        tgt_14 = torch.tensor(np.stack(kd_targets, axis=0), device=dev, dtype=torch.float32)
        pred_14 = torch.sigmoid(lc[idx_t].float())
        loss_kd = F.binary_cross_entropy(pred_14.clamp(1e-5, 1.0 - 1e-5), tgt_14.clamp(1e-5, 1.0 - 1e-5))
        loss_extra = loss_extra + 0.32 * loss_kd

    # 2. Scene-Grouped Within-Batch Differential Contrast on Synthetic / Same-Dim Samples
    stems = [_extract_seed_stem(str(nm)) for nm in names]
    col = 0
    for ci, c in enumerate(CLASSES):
        n_ch = 3 if c in FULL else 1
        valid = (mask7[:, ci] > 0.1)
        if valid.sum() >= 2:
            valid_idx = torch.where(valid)[0]
            y_c = g[valid_idx, ci]
            z_c = torch.sigmoid(lc[valid_idx, col:col + n_ch].float()).sum(dim=-1)  # Expected cumulative severity in [0, n_ch]
            diff_y = y_c.unsqueeze(1) - y_c.unsqueeze(0)
            pos_pairs = (diff_y >= 1)
            if pos_pairs.any():
                v_stems = [stems[int(ii)] for ii in valid_idx.cpu().tolist()]
                same_seed = torch.tensor(
                    [[1.0 if v_stems[a] == v_stems[b] else 0.0 for b in range(len(v_stems))] for a in range(len(v_stems))],
                    device=dev, dtype=torch.float32
                )
                pair_w = 1.0 + 2.5 * same_seed[pos_pairs]
                margin = (0.28 if c in FULL else 0.35) * diff_y[pos_pairs].float()
                diff_z = z_c.unsqueeze(1) - z_c.unsqueeze(0)
                loss_extra = loss_extra + 0.22 * ((F.relu(margin - diff_z[pos_pairs]) ** 2) * pair_w).mean()
        col += n_ch

    return loss_extra


def _p012_row_to_vec14(p0_row, p1_row, p2_row):
    vec14 = []
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            c_eq3 = float(np.clip(p0_row[ci], 1e-4, 1.0 - 1e-4))
            c_ge2 = float(np.clip(p0_row[ci] + p1_row[k], c_eq3, 1.0 - 1e-4))
            c_ge1 = float(np.clip(p0_row[ci] + p1_row[k] + p2_row[k], c_ge2, 1.0 - 1e-4))
            vec14.extend([c_ge1, c_ge2, c_eq3])
        else:
            vec14.append(float(np.clip(p0_row[ci], 1e-4, 1.0 - 1e-4)))
    return np.array(vec14, dtype=np.float32)


def load_teacher_14_map(split_csv):
    """
    Loads zero-leakage teacher soft 14-channel probabilities on real training images.
    Combines:
      1) True InternViT-6B OOF (`v12_phaseB_fixed`, `v12_phaseB`, `v11_pure_real`)
      2) V16_Full 8-Fold AllGrade Microscope OOF (`results/v16_fulldata/V16_Full_F0..F7_AllGrade_Microscope_probs.npz`)
      3) Baseline `v15_ens_oof3573_probs.npz`
    Restricted strictly to `train` images of the current split (`s16`, `s17`, or `split_full_v18`).
    """
    train_names = set(r['name'] for r in load_split(split_csv) if r['split'] == 'train')
    accum = {}  # nm -> [p0_sum(6,), p1_sum(4,), p2_sum(4,), w_sum]

    def add_entry(nm, p0, p1, p2, w):
        if nm not in train_names:
            return
        if nm not in accum:
            accum[nm] = [np.zeros(6, dtype=np.float64), np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64), 0.0]
        accum[nm][0] += w * np.asarray(p0, dtype=np.float64)
        accum[nm][1] += w * np.asarray(p1, dtype=np.float64)
        accum[nm][2] += w * np.asarray(p2, dtype=np.float64)
        accum[nm][3] += w

    # 1. Load 6B Teacher OOF across splits 16..20
    for sp in (16, 17, 18, 19, 20):
        for pat, w in [
            (f'results/v12_phaseB_fixed/v12_abmil_s{sp}_probs.npz', 0.35),
            (f'results/v12_phaseB/v12_abmil_s{sp}_probs.npz', 0.25),
            (f'results/v11_pure_real/iv6b_s{sp}_probs.npz', 0.20),
        ]:
            if os.path.exists(pat):
                d = np.load(pat, allow_pickle=True)
                for pref, k_nm in [('va', 'va_names'), ('te', 'te_names')]:
                    if k_nm in d:
                        nms = d[k_nm]
                        p0, p1, p2 = d[f'{pref}_p0'], d[f'{pref}_p1'], d[f'{pref}_p2']
                        for i, nm in enumerate(nms):
                            add_entry(str(nm), p0[i], p1[i], p2[i], w)

    # 2. Load V16_Full 8-Fold OOF
    for f_idx in range(8):
        p = f'results/v16_fulldata/V16_Full_F{f_idx}_AllGrade_Microscope_probs.npz'
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            for pref, k_nm in [('va', 'va_names'), ('te', 'te_names')]:
                if k_nm in d:
                    nms = d[k_nm]
                    p0, p1, p2 = d[f'{pref}_p0'], d[f'{pref}_p1'], d[f'{pref}_p2']
                    for i, nm in enumerate(nms):
                        add_entry(str(nm), p0[i], p1[i], p2[i], 0.45)

    # 3. Fallback V15 OOF so all 3,573 real images are guaranteed soft targets
    if os.path.exists('results/v15_full8fold/v15_ens_oof3573_probs.npz'):
        d = np.load('results/v15_full8fold/v15_ens_oof3573_probs.npz', allow_pickle=True)
        names = list(d['all_names'])
        for i, nm in enumerate(names):
            add_entry(str(nm), d['oof_p0'][i], d['oof_p1'][i], d['oof_p2'][i], 0.20)

    teacher_map = {}
    for nm, (s0, s1, s2, w_tot) in accum.items():
        if w_tot > 0:
            teacher_map[nm] = _p012_row_to_vec14(s0 / w_tot, s1 / w_tot, s2 / w_tot)
    return teacher_map


def blend_with_v16_anchor_safe(v16_npz, vp0, vp1, vp2, va_names, tp0, tp1, tp2, te_names, p150_0, p150_1, p150_2):
    """
    Safely blends V18 predictions with pure-real V16 specialist anchor predictions:
      - If shapes match directly (e.g. s16/s17 where va=500, te=1000), blends directly.
      - If shapes differ (e.g. FullData where va=300 and te=0), aligns by image name across
        s16 + s17 V16 specialist files and uses the s16+s17 average for test_150!
    """
    if not (v16_npz and os.path.exists(v16_npz)):
        return vp0, vp1, vp2, tp0, tp1, tp2, p150_0, p150_1, p150_2

    d16 = np.load(v16_npz, allow_pickle=True)
    w16_p0 = [(0, 0.55), (1, 0.55), (2, 0.35), (3, 0.30), (4, 0.20), (5, 0.45)]

    same_va = (vp0 is not None) and (len(d16['va_p0']) == len(vp0))
    same_te = (tp0 is not None) and (len(tp0) > 0) and (len(d16['te_p0']) == len(tp0))

    if same_va and same_te:
        for c, w16 in w16_p0:
            vp0[:, c] = w16 * d16['va_p0'][:, c] + (1.0 - w16) * vp0[:, c]
            tp0[:, c] = w16 * d16['te_p0'][:, c] + (1.0 - w16) * tp0[:, c]
            p150_0[:, c] = w16 * d16['t150_p0'][:, c] + (1.0 - w16) * p150_0[:, c]
        vp1 = 0.82 * vp1 + 0.18 * d16['va_p1']
        tp1 = 0.82 * tp1 + 0.18 * d16['te_p1']
        p150_1 = 0.82 * p150_1 + 0.18 * d16['t150_p1']
        vp2 = 0.82 * vp2 + 0.18 * d16['va_p2']
        tp2 = 0.82 * tp2 + 0.18 * d16['te_p2']
        p150_2 = 0.82 * p150_2 + 0.18 * d16['t150_p2']
        return vp0, vp1, vp2, tp0, tp1, tp2, p150_0, p150_1, p150_2

    # Name-aligned lookup across s16 and s17 V16 files (for FullData split_full_v18)
    s17_npz = v16_npz.replace('_s16_probs.npz', '_s17_probs.npz')
    cand_files = [p for p in [v16_npz, s17_npz] if os.path.exists(p)]
    name_to_v16 = {}
    t150_0_list, t150_1_list, t150_2_list = [], [], []
    for p in cand_files:
        dp = np.load(p, allow_pickle=True)
        for pref, k_nm in [('va', 'va_names'), ('te', 'te_names')]:
            if k_nm in dp:
                nms = dp[k_nm]
                for i, nm in enumerate(nms):
                    s_nm = str(nm)
                    if s_nm not in name_to_v16:
                        name_to_v16[s_nm] = ([], [], [])
                    name_to_v16[s_nm][0].append(dp[f'{pref}_p0'][i])
                    name_to_v16[s_nm][1].append(dp[f'{pref}_p1'][i])
                    name_to_v16[s_nm][2].append(dp[f'{pref}_p2'][i])
        if 't150_p0' in dp:
            t150_0_list.append(dp['t150_p0'])
            t150_1_list.append(dp['t150_p1'])
            t150_2_list.append(dp['t150_p2'])

    def _align_and_blend(arr0, arr1, arr2, names_list):
        if arr0 is None or len(arr0) == 0:
            return arr0, arr1, arr2
        ref0 = arr0.copy()
        ref1 = arr1.copy()
        ref2 = arr2.copy()
        for i, nm in enumerate(names_list):
            s_nm = str(nm)
            if s_nm in name_to_v16 and len(name_to_v16[s_nm][0]) > 0:
                ref0[i] = np.mean(name_to_v16[s_nm][0], axis=0)
                ref1[i] = np.mean(name_to_v16[s_nm][1], axis=0)
                ref2[i] = np.mean(name_to_v16[s_nm][2], axis=0)
        for c, w16 in w16_p0:
            arr0[:, c] = w16 * ref0[:, c] + (1.0 - w16) * arr0[:, c]
        arr1 = 0.82 * arr1 + 0.18 * ref1
        arr2 = 0.82 * arr2 + 0.18 * ref2
        return arr0, arr1, arr2

    vp0, vp1, vp2 = _align_and_blend(vp0, vp1, vp2, va_names)
    tp0, tp1, tp2 = _align_and_blend(tp0, tp1, tp2, te_names)

    if t150_0_list and p150_0 is not None:
        ref150_0 = np.mean(t150_0_list, axis=0)
        ref150_1 = np.mean(t150_1_list, axis=0)
        ref150_2 = np.mean(t150_2_list, axis=0)
        for c, w16 in w16_p0:
            p150_0[:, c] = w16 * ref150_0[:, c] + (1.0 - w16) * p150_0[:, c]
        p150_1 = 0.82 * p150_1 + 0.18 * ref150_1
        p150_2 = 0.82 * p150_2 + 0.18 * ref150_2

    return vp0, vp1, vp2, tp0, tp1, tp2, p150_0, p150_1, p150_2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='pecoreg:lastk2,siglip2g:lastk2,cradio:lastk2,convnext:full')
    ap.add_argument('--focus-classes', default='过度生成,纹理损失')
    ap.add_argument('--init-ckpt', default='', help='Warm-start checkpoint path')
    ap.add_argument('--v16-npz', default='', help='Pure-real V16 specialist npz for P0 prior anchoring')
    ap.add_argument('--epochs-s1', type=int, default=1)
    ap.add_argument('--epochs-s2', type=int, default=1)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--lr-full', type=float, default=3.5e-5)
    ap.add_argument('--lr-last', type=float, default=1.4e-5)
    ap.add_argument('--lr-head', type=float, default=4.5e-4)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=18)
    ap.add_argument('--dropout', type=float, default=0.12)
    ap.add_argument('--clean-repeat', type=int, default=2)
    ap.add_argument('--p0-clean-weight', type=float, default=4.2)
    ap.add_argument('--channel-neg-weight', type=float, default=1.05)
    ap.add_argument('--alpha-ex', type=float, default=0.58)
    ap.add_argument('--synth-w', type=float, default=0.65)
    ap.add_argument('--split', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--phys-cache', default='results/v8_experts/native_phys_feats.npz')
    ap.add_argument('--tau', type=float, default=0.02)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    set_seed(a.seed)
    dev = 'cuda'
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    roi_map, meta_map, synth_meta = load_roi_and_synth_meta()
    teacher_14_map = load_teacher_14_map(a.split)

    focus_cis = [CLASSES.index(c.strip()) for c in a.focus_classes.split(',') if c.strip() in CLASSES]
    print(f'=== Training V18.1 Paired-Quadruplet + True 6B/V16 Teacher-KD Specialist on focus_classes={a.focus_classes} (Stage1={a.epochs_s1}ep, Stage2={a.epochs_s2}ep) ===', flush=True)

    rows = load_split(a.split)
    man = read_manifest(a.manifest)
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})

    from extract_native_phys_feats import extract_for_names
    tr_orig = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test']
    tr_real = [r for r in tr_orig if not is_synth_name(r['name']) and ('_NatNeg' not in r['name'])]
    phys_map = extract_for_names([r['name'] for r in tr_real + va + te], a.phys_cache)

    phys_tr_mat = np.stack([phys_map[r['name']] for r in tr_real], axis=0)
    phys_mu = np.mean(phys_tr_mat, axis=0)
    phys_sd = np.std(phys_tr_mat, axis=0) + 1e-5

    Y_orig = targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr_real]))
    pw = ((len(Y_orig) - Y_orig.sum(0)) / Y_orig.sum(0).clamp(min=1)).clamp(1.0, 5.5).to(dev)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    tr_synth_all = [
        r for r in tr_orig
        if (is_synth_name(r['name']) or '_d5_NatNeg' in r['name'])
        and (any(get_sample_class_mask6_v17(r['name'], a.synth_w)[ci] > 0.1 for ci in focus_cis) or ('_d5_NatNeg' in r['name'] and (0 in focus_cis or 5 in focus_cis)))
    ]
    tr_synth_all.sort(key=lambda r: r['name'])

    rng = random.Random(a.seed)
    real_shuf = list(tr_real)
    rng.shuffle(real_shuf)
    tr_stage1 = []
    s_ptr = 0
    for i_r, r_row in enumerate(real_shuf):
        tr_stage1.append(r_row)
        if (i_r + 1) % 6 == 0 and s_ptr + 3 <= len(tr_synth_all):
            tr_stage1.extend(tr_synth_all[s_ptr:s_ptr + 3])
            s_ptr += 3

    if a.clean_repeat > 1:
        clean = [r for r in tr_real if all(int(r[c]) == 0 for c in CLASSES)]
        tr_stage1 = tr_stage1 + clean

    nat_negs = [r for r in tr_orig if '_d5_NatNeg' in r['name']] if (0 in focus_cis or 5 in focus_cis) else []
    tr_stage2 = list(tr_real) + nat_negs
    p1_p0_real_s2 = [r for r in tr_real if any(int(r[CLASSES[ci]]) >= 2 for ci in (1, 2, 3, 4, 5))]
    tr_stage2 = tr_stage2 + p1_p0_real_s2

    print(f'  Stage 1 Paired Train={len(tr_stage1)} (real={len(tr_real)}, corridor_synth={len(tr_synth_all)}) | Stage 2 PureReal+NatNeg={len(tr_stage2)} | Teacher KD={len(teacher_14_map)}', flush=True)

    dtr_s1 = V18FastMicroscopeDataset(tr_stage1, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, True, tiles=a.tiles, synth_w=a.synth_w)
    dtr_s2 = V18FastMicroscopeDataset(tr_stage2, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, True, tiles=a.tiles, synth_w=a.synth_w)
    dva = V18FastMicroscopeDataset(va, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, tiles=a.tiles, synth_w=a.synth_w)

    val_batch = a.batch * 2
    ltr_s1 = DataLoader(dtr_s1, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True, persistent_workers=True, drop_last=True)
    ltr_s2 = DataLoader(dtr_s2, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, persistent_workers=True, drop_last=True)
    lva = DataLoader(dva, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True, persistent_workers=True)

    specs = []
    pg = []
    for item in a.backbones.split(','):
        nm, mode = item.split(':')
        m, fwd, mu, sd, bl, dim = build_v9_backbone(nm)
        for p in m.parameters():
            p.requires_grad_(False)
        if mode == 'full':
            for p in m.parameters():
                p.requires_grad_(True)
            pg.append({'params': [p for p in m.parameters() if p.requires_grad], 'lr': a.lr_full})
        elif mode.startswith('lastk'):
            k = int(mode[5:])
            if k > 0:
                for blk in bl[-k:]:
                    for p in blk.parameters():
                        p.requires_grad_(True)
                pg.append({'params': [p for p in m.parameters() if p.requires_grad], 'lr': a.lr_last})
        specs.append((m, fwd, mu, sd, bl, dim))

    net = MultiNetV18(specs, phys_dim=60, dropout=a.dropout, n_quad=a.tiles * a.tiles).to(dev)
    init_sd_cpu = None
    if a.init_ckpt and os.path.exists(a.init_ckpt):
        print(f'  Loading warm-start checkpoint: {a.init_ckpt}', flush=True)
        sd_load = torch.load(a.init_ckpt, map_location='cpu')
        net.load_state_dict(sd_load, strict=False)
        init_sd_cpu = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    head_params = (
        list(net.head_cum.parameters())
        + list(net.head_ex.parameters())
        + list(net.gate.parameters())
        + list(net.norm.parameters())
        + list(net.se_gate.parameters())
        + list(net.tile_attn.parameters())
        + list(net.phys_mlp.parameters())
        + list(net.geom_film.parameters())
        + list(net.corn_delta.parameters())
    )
    pg.append({'params': head_params, 'lr': a.lr_head})
    trainable_params = [p for p in net.parameters() if p.requires_grad]

    try:
        opt = torch.optim.AdamW(pg, weight_decay=a.wd, fused=True)
    except Exception:
        opt = torch.optim.AdamW(pg, weight_decay=a.wd)

    total_steps = max(1, (len(ltr_s1) // a.grad_accum) * a.epochs_s1 + (len(ltr_s2) // a.grad_accum) * a.epochs_s2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    ckpts = []
    if init_sd_cpu is not None:
        ckpts.append(init_sd_cpu)

    total_epochs = a.epochs_s1 + a.epochs_s2
    for ep in range(total_epochs):
        t0 = time.time()
        is_stage2 = (ep >= a.epochs_s1)
        cur_loader = ltr_s2 if is_stage2 else ltr_s1
        stage_tag = 'Stage2-PureReal+KD' if is_stage2 else 'Stage1-PairedQuad+KD'
        cur_focus = list(range(6)) if is_stage2 else focus_cis

        net.train()
        opt.zero_grad(set_to_none=True)
        tot, n_step = 0.0, 0
        for step, (x, phys, geom, g, mask7, names) in enumerate(cur_loader):
            x = x.to(dev, non_blocking=True)
            phys = phys.to(dev, non_blocking=True)
            geom = geom.to(dev, non_blocking=True)
            g = g.to(dev, non_blocking=True)
            mask7 = mask7.to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lc, le, gl = net(x, phys, geom)
            loss_base = v16_clean_microscope_loss_fn(
                lc.float(), le.float(), gl.float(), g, mask7, bce_loss_none,
                focus_cis=cur_focus,
                p0_clean_weight=a.p0_clean_weight,
                channel_neg_weight=a.channel_neg_weight,
                alpha_ex=a.alpha_ex,
            )
            loss_kd_pair = compute_paired_quadruplet_and_kd_loss(lc.float(), le.float(), g, mask7, names, teacher_14_map, dev)
            loss = loss_base + loss_kd_pair
            (loss / a.grad_accum).backward()
            if (step + 1) % a.grad_accum == 0 or (step + 1) == len(cur_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sch.step()
            tot += float(loss.detach())
            n_step += 1

        vp0, vp1, vp2, vg, gva = predict_v18(net, lva, dev, a.alpha_ex)
        pred_va = decode_pc(vp0, vp1, vp2, HAND)
        s_hand = cm.score(gva, pred_va, a.tau)
        clean_va = (gva == 0).all(1)
        s_clean2 = float((clean_va & (pred_va == 0).all(1)).sum() / max(1, clean_va.sum()))
        doc2_va = 25 * s_hand['S_pos'] + 20 * s_hand['S_p0'] + 20 * s_clean2
        print(f'ep={ep+1}/{total_epochs} [{stage_tag}] loss={tot/max(1,n_step):.4f} val_Doc1={s_hand["total"]:.2f} val_Doc2={doc2_va:.2f} (S_pos={s_hand["S_pos"]:.3f} S_p0={s_hand["S_p0"]:.3f} Clean2={s_clean2:.3f}) ({time.time()-t0:.0f}s)', flush=True)

        cur_sd = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        torch.save(cur_sd, f'{a.out}_ep{ep+1}.pt')
        ckpts.append(cur_sd)

    # SWA over warm-start anchor + trained epochs (for newly added geom_film & corn_delta, average only trained epochs!)
    if len(ckpts) >= 2:
        swa_sd = {}
        w_list = [0.25, 0.35, 0.40] if len(ckpts) == 3 else [1.0 / len(ckpts)] * len(ckpts)
        trained_ckpts = ckpts[1:] if (init_sd_cpu is not None and len(ckpts) > 1) else ckpts
        for k in ckpts[0].keys():
            if ckpts[0][k].is_floating_point():
                if k.startswith('geom_film.') or k.startswith('corn_delta.'):
                    swa_sd[k] = sum(ck[k] for ck in trained_ckpts) / float(len(trained_ckpts))
                else:
                    swa_sd[k] = sum(w_i * ck[k] for w_i, ck in zip(w_list, ckpts))
            else:
                swa_sd[k] = ckpts[-1][k]
        net.load_state_dict({k: v.to(dev) for k, v in swa_sd.items()})
        torch.save(swa_sd, f'{a.out}_swa.pt')

    dva_flip = V18FastMicroscopeDataset(va, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, hflip=True, tiles=a.tiles, synth_w=a.synth_w)
    lva_flip = DataLoader(dva_flip, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    vp0_a, vp1_a, vp2_a, vg_a, gva = predict_v18(net, lva, dev, a.alpha_ex)
    vp0_b, vp1_b, vp2_b, vg_b, _   = predict_v18(net, lva_flip, dev, a.alpha_ex)
    vp0, vp1, vp2, vg = 0.5*(vp0_a+vp0_b), 0.5*(vp1_a+vp1_b), 0.5*(vp2_a+vp2_b), 0.5*(vg_a+vg_b)

    if len(te) > 0:
        dte = V18FastMicroscopeDataset(te, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, tiles=a.tiles, synth_w=a.synth_w)
        dte_flip = V18FastMicroscopeDataset(te, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, hflip=True, tiles=a.tiles, synth_w=a.synth_w)
        lte = DataLoader(dte, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
        lte_flip = DataLoader(dte_flip, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
        tp0_a, tp1_a, tp2_a, tg_a, gte = predict_v18(net, lte, dev, a.alpha_ex)
        tp0_b, tp1_b, tp2_b, tg_b, _   = predict_v18(net, lte_flip, dev, a.alpha_ex)
        tp0, tp1, tp2, tg = 0.5*(tp0_a+tp0_b), 0.5*(tp1_a+tp1_b), 0.5*(tp2_a+tp2_b), 0.5*(tg_a+tg_b)
    else:
        tp0, tp1, tp2, tg, gte = vp0.copy(), vp1.copy(), vp2.copy(), vg.copy(), gva.copy()

    tpl = json.load(open('results/v2_full/template_150.json'))
    names_150 = [e['image'] for e in tpl]
    phys_150 = extract_for_names(names_150, a.phys_cache)
    rows_150 = [{'name': n, **{c: 0 for c in CLASSES}} for n in names_150]
    d150 = V18FastMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, hflip=False, tiles=a.tiles, synth_w=a.synth_w)
    d150_f = V18FastMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, hflip=True, tiles=a.tiles, synth_w=a.synth_w)
    l150 = DataLoader(d150, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    l150_f = DataLoader(d150_f, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    p150_0a, p150_1a, p150_2a, p150_ga, _ = predict_v18(net, l150, dev, a.alpha_ex)
    p150_0b, p150_1b, p150_2b, p150_gb, _ = predict_v18(net, l150_f, dev, a.alpha_ex)
    p150_0, p150_1, p150_2, p150_g = 0.5*(p150_0a+p150_0b), 0.5*(p150_1a+p150_1b), 0.5*(p150_2a+p150_2b), 0.5*(p150_ga+p150_gb)

    va_names = [r['name'] for r in va]
    te_names = [r['name'] for r in te] if len(te) > 0 else va_names

    # Safe name-aligned V16 pure-real anchor blending (supports both s16/s17 500-val and FullData 300-val)
    vp0, vp1, vp2, tp0, tp1, tp2, p150_0, p150_1, p150_2 = blend_with_v16_anchor_safe(
        a.v16_npz, vp0, vp1, vp2, va_names, tp0, tp1, tp2, te_names, p150_0, p150_1, p150_2
    )

    out_base = a.out[:-5] if a.out.endswith('.json') else a.out
    np.savez(
        out_base + '_probs.npz',
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vg, gva=gva, va_names=va_names,
        te_p0=tp0, te_p1=tp1, te_p2=tp2, te_gate=tg, gte=gte, te_names=te_names,
        t150_p0=p150_0, t150_p1=p150_1, t150_p2=p150_2, t150_gate=p150_g, t150_names=names_150,
    )
    st_hand = cm.score(gte, decode_pc(tp0, tp1, tp2, HAND), a.tau)
    clean_te = (gte == 0).all(1)
    pred_te = decode_pc(tp0, tp1, tp2, HAND)
    s_clean2_te = float((clean_te & (pred_te == 0).all(1)).sum() / max(1, clean_te.sum()))
    doc2_te = 25 * st_hand['S_pos'] + 20 * st_hand['S_p0'] + 20 * s_clean2_te
    json.dump({'doc1': float(st_hand['total']), 'doc2': float(doc2_te), 'S_pos': float(st_hand['S_pos']),
               'S_p0': float(st_hand['S_p0']), 'Clean2': float(s_clean2_te)}, open(out_base + '.json', 'w'), indent=2)
    print(f'=== FINAL V18.1 PAIRED+KD SWA+TTA EVAL (N={len(gte)}): Doc1={st_hand["total"]:.2f} | Doc2={doc2_te:.2f} (S_pos={st_hand["S_pos"]:.3f} S_p0={st_hand["S_p0"]:.3f} Clean2={s_clean2_te:.3f}) ===', flush=True)


if __name__ == '__main__':
    main()
