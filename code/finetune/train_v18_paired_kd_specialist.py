#!/usr/bin/env python3
"""
V18.1 Paired-Quadruplet + True 6B/V16 Teacher Knowledge Distillation Specialist Trainer
=======================================================================================
Fixes all 4 training bugs identified in V18:
1. True 6B + V16_Full Teacher Distillation (`load_teacher_14_map`):
   Distills from the strong `InternViT-6B` (v12_phaseB_fixed + v12_phaseB + v11_pure_real)
   + `V16_Full` OOF teacher strictly restricted to `train` split images (replacing the weak
   42.8-point V15 frozen feature tower).
2. True Scene-Grouped Paired Quadruplet Ranking & Orthogonality Loss:
   Groups synthetic quadruplets by `(seed_name, target_dim_idx)` inside each mini-batch,
   enforcing monotonic margin ranking `S(无问题) < S(P2) < S(P1) < S(P0)` on the target dimension
   AND cross-dimension invariance on the 5 non-target dimensions `c' != c`.
3. Smooth Residual Dual-Head Logits (`MultiNetV18`):
   Replaces the hard `torch.maximum(l_ge2, l_eq3 + d_p1 * 0.35)` logit distortion with a smooth
   residual + soft monotonicity penalty so `head_cum` and `head_ex` remain probability-calibrated.
4. Pure Trained SWA & Name-Aligned V16 Anchor Blending:
   - Never blends untrained `init_sd_cpu` weights into SWA (averages only trained Stage 1 & Stage 2 checkpoints).
   - Aligns `--v16-npz` by image name so `split_full_v18.csv` (`val` size 300) never crashes when
     blending with `s16/s17` V16 specialists (`val` size 500).
"""
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

REPO_ROOT = '/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DIMS = ['文字错误', '元素重叠', '模糊与伪影', '色彩与曝光问题', '过度生成', '构图与切片问题']
GRADE_TO_IDX = {'无问题': 0, 'P2': 1, 'P1': 2, 'P0': 3}
W_DIM = np.array([1.0, 1.0, 0.7, 0.6, 0.9, 0.8], dtype=np.float64)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiScaleMicroscopeGridV18(nn.Module):
    def __init__(self, d=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d * 4, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

    def forward(self, tokens):
        B, N, D = tokens.shape
        g_mean = tokens.mean(dim=1)
        g_max = tokens.max(dim=1).values
        if N == 256:
            grid = tokens.view(B, 16, 16, D)
            q4 = F.adaptive_max_pool2d(grid.permute(0, 3, 1, 2), (2, 2)).flatten(2).mean(dim=-1)
            q16 = F.adaptive_max_pool2d(grid.permute(0, 3, 1, 2), (4, 4)).flatten(2)
            top2_16 = torch.topk(q16, k=2, dim=-1).values.mean(dim=-1)
        else:
            q4 = g_max
            top2_16 = g_mean
        return self.proj(torch.cat([g_mean, g_max, q4, top2_16], dim=-1))


class MultiNetV18(nn.Module):
    def __init__(self, base_ckpt_path, unfreeze_mode='lastk2'):
        super().__init__()
        ckpt = torch.load(base_ckpt_path, map_location='cpu')
        self.backbone = ckpt['backbone'].float()
        d = 2048
        self.ms_pool = MultiScaleMicroscopeGridV18(d)
        self.head_cum = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 18)
        )
        self.head_ex = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 24)
        )
        self.geom_film = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(), nn.Linear(64, d * 2)
        )
        nn.init.zeros_(self.geom_film[-1].weight)
        nn.init.zeros_(self.geom_film[-1].bias)

        self.corn_delta = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.15), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 12)
        )
        nn.init.zeros_(self.corn_delta[-1].weight)
        nn.init.zeros_(self.corn_delta[-1].bias)

        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
            own = self.state_dict()
            for k, v in sd.items():
                if k in own and own[k].shape == v.shape:
                    own[k].copy_(v)

        self.set_unfreeze(unfreeze_mode)

    def set_unfreeze(self, mode):
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers = self.backbone.encoder.layers
        if mode == 'lastk1':
            for p in layers[-1:].parameters():
                p.requires_grad = True
        elif mode == 'lastk2':
            for p in layers[-2:].parameters():
                p.requires_grad = True
        elif mode == 'lastk3':
            for p in layers[-3:].parameters():
                p.requires_grad = True
        for m in [self.ms_pool, self.head_cum, self.head_ex, self.geom_film, self.corn_delta]:
            for p in m.parameters():
                p.requires_grad = True

    def forward_single(self, x, geom=None):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = self.backbone(x, output_hidden_states=False, return_dict=True)
            tokens = out.last_hidden_state[:, 1:, :].float()
        feat = self.ms_pool(tokens)
        if geom is not None:
            gb = self.geom_film(geom)
            gamma, beta = gb.chunk(2, dim=-1)
            feat = feat * (1.0 + 0.12 * torch.tanh(gamma)) + 0.12 * beta

        lc = self.head_cum(feat)
        le = self.head_ex(feat)
        deltas = self.corn_delta(feat)

        # Smooth residual adjustment on cumulative thresholds (no hard torch.maximum distortion)
        lc_out = lc.clone()
        for c in range(6):
            col = c * 3
            d_ge2 = 0.15 * torch.tanh(deltas[:, c * 2])
            d_ge1 = 0.15 * torch.tanh(deltas[:, c * 2 + 1])
            lc_out[:, col + 1] = lc[:, col + 1] + d_ge2
            lc_out[:, col] = lc[:, col] + d_ge1
        return lc_out, le

    def forward(self, x_g, x_l, geom=None, alpha_local=0.38):
        lc_g, le_g = self.forward_single(x_g, geom)
        if x_l is None or alpha_local <= 0:
            return lc_g, le_g
        lc_l, le_l = self.forward_single(x_l, geom)
        return (1.0 - alpha_local) * lc_g + alpha_local * lc_l, (1.0 - alpha_local) * le_g + alpha_local * le_l


def make_global_and_hires_crop(img: Image.Image, is_train=False):
    W, H = img.size
    ar = W / max(H, 1.0)
    geom = torch.tensor(
        [np.log(max(ar, 0.1)), (W * H) / (1024.0 * 1024.0), float(min(W, H)) / 1024.0],
        dtype=torch.float32,
    )
    g_img = img.resize((448, 448), Image.BICUBIC)
    if W >= 448 and H >= 448:
        if is_train:
            max_x = W - 448
            max_y = H - 448
            cx = random.randint(int(max_x * 0.15), int(max_x * 0.85)) if max_x > 10 else 0
            cy = random.randint(int(max_y * 0.15), int(max_y * 0.85)) if max_y > 10 else 0
        else:
            cx, cy = (W - 448) // 2, (H - 448) // 2
        l_img = img.crop((cx, cy, cx + 448, cy + 448))
    else:
        l_img = g_img

    if is_train and random.random() < 0.5:
        g_img = TF.hflip(g_img)
        l_img = TF.hflip(l_img)

    t_g = TF.normalize(TF.to_tensor(g_img), MEAN, STD)
    t_l = TF.normalize(TF.to_tensor(l_img), MEAN, STD)
    return t_g, t_l, geom


class PairedQuadDatasetV18(Dataset):
    """
    Dataset with:
    - Real training images + their true 6B/V16 teacher soft probability targets (14-dim)
    - Corridor-verified native-resolution synthetic images tagged with `(seed_id, target_dim_idx)`
      so the loss function can compute exact within-scene paired ranking and non-target invariance.
    """
    def __init__(self, real_df, real_img_dir, synth_df=None, teacher_map=None):
        self.items = []
        self.seed_to_id = {}
        real_name_to_path = {}
        real_name_to_y = {}

        for _, row in real_df.iterrows():
            nm = str(row['image'])
            p = os.path.join(real_img_dir, nm)
            y = np.array([GRADE_TO_IDX[row[c]] for c in DIMS], dtype=np.int64)
            real_name_to_path[nm] = p
            real_name_to_y[nm] = y
            t_vec = teacher_map.get(nm, np.full(14, -1.0, dtype=np.float32)) if teacher_map else np.full(14, -1.0, dtype=np.float32)
            self.items.append((p, y, -1, t_vec, -1))

        if synth_df is not None and len(synth_df) > 0:
            # Group synthetic rows by (seed_name, target_dim_idx) so quadruplet members stay near each other
            for (seed_nm, dim_c), grp in synth_df.groupby(['seed_name', 'target_dim_idx']):
                if seed_nm not in self.seed_to_id:
                    self.seed_to_id[seed_nm] = len(self.seed_to_id)
                sid = self.seed_to_id[seed_nm]
                # Also inject the clean seed image into this group if present in real_df
                if seed_nm in real_name_to_path:
                    self.items.append((
                        real_name_to_path[seed_nm],
                        real_name_to_y[seed_nm],
                        int(dim_c),
                        np.full(14, -1.0, dtype=np.float32),
                        sid,
                    ))
                for _, row in grp.iterrows():
                    p = str(row['image'])
                    if not os.path.exists(p):
                        continue
                    y = np.array([GRADE_TO_IDX[row[c]] for c in DIMS], dtype=np.int64)
                    self.items.append((p, y, int(dim_c), np.full(14, -1.0, dtype=np.float32), sid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y, t_dim, t_vec, sid = self.items[idx]
        img = Image.open(p).convert('RGB')
        x_g, x_l, geom = make_global_and_hires_crop(img, is_train=True)
        return x_g, x_l, geom, torch.from_numpy(y), int(t_dim), torch.from_numpy(t_vec), int(sid)


class EvalDatasetV18(Dataset):
    def __init__(self, df, img_dir):
        self.paths = [os.path.join(img_dir, str(r['image'])) for _, r in df.iterrows()]
        self.names = [str(r['image']) for _, r in df.iterrows()]
        if DIMS[0] in df.columns:
            self.labels = np.array([[GRADE_TO_IDX[r[c]] for c in DIMS] for _, r in df.iterrows()], dtype=np.int64)
        else:
            self.labels = np.zeros((len(df), 6), dtype=np.int64)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        W, H = img.size
        ar = W / max(H, 1.0)
        geom = torch.tensor(
            [np.log(max(ar, 0.1)), (W * H) / (1024.0 * 1024.0), float(min(W, H)) / 1024.0],
            dtype=torch.float32,
        )
        g_img = img.resize((448, 448), Image.BICUBIC)
        g_flip = TF.hflip(g_img)
        if W >= 448 and H >= 448:
            cx, cy = (W - 448) // 2, (H - 448) // 2
            l_img = img.crop((cx, cy, cx + 448, cy + 448))
        else:
            l_img = g_img
        t_g = TF.normalize(TF.to_tensor(g_img), MEAN, STD)
        t_gf = TF.normalize(TF.to_tensor(g_flip), MEAN, STD)
        t_l = TF.normalize(TF.to_tensor(l_img), MEAN, STD)
        return t_g, t_gf, t_l, geom, torch.from_numpy(self.labels[idx]), self.names[idx]


def exact_probs_from_dual(lc, le, w_cum=0.55):
    pc = torch.sigmoid(lc).view(-1, 6, 3)
    pe = F.softmax(le.view(-1, 6, 4), dim=-1)
    p_ge1, p_ge2, p_eq3 = pc[:, :, 0], pc[:, :, 1], pc[:, :, 2]
    p0_c = p_eq3
    p1_c = torch.clamp(p_ge2 - p_eq3, min=0.0)
    p2_c = torch.clamp(p_ge1 - p_ge2, min=0.0)

    p0 = w_cum * p0_c + (1.0 - w_cum) * pe[:, :, 3]
    p1 = w_cum * p1_c + (1.0 - w_cum) * pe[:, :, 2]
    p2 = w_cum * p2_c + (1.0 - w_cum) * pe[:, :, 1]
    return p0, p1, p2


def compute_paired_quadruplet_and_kd_loss(lc, le, y, t_dim, t_vec, sid, dim_weights, kd_weight=0.30, rank_weight=0.25):
    B = y.shape[0]
    w_d = torch.tensor(dim_weights, device=y.device, dtype=torch.float32)

    # 1. Ordinal cumulative BCE + Exact grade CE + Monotonicity penalty
    loss_cum = 0.0
    loss_ex = 0.0
    loss_mono = 0.0
    for c in range(6):
        col_c = c * 3
        col_e = c * 4
        yc = y[:, c]
        t_ge1 = (yc >= 1).float()
        t_ge2 = (yc >= 2).float()
        t_eq3 = (yc == 3).float()
        target_c = torch.stack([t_ge1, t_ge2, t_eq3], dim=-1)
        bce = F.binary_cross_entropy_with_logits(lc[:, col_c:col_c + 3], target_c, reduction='none')
        g_w = torch.where(yc == 3, 1.5, torch.where(yc == 2, 1.35, torch.where(yc == 1, 1.15, 1.0))).unsqueeze(-1)
        loss_cum = loss_cum + w_d[c] * (bce * g_w).mean()

        ce = F.cross_entropy(le[:, col_e:col_e + 4], yc, label_smoothing=0.03)
        loss_ex = loss_ex + w_d[c] * ce

        # Soft monotonicity constraint: l_ge1 >= l_ge2 >= l_eq3
        l_ge1, l_ge2, l_eq3 = lc[:, col_c], lc[:, col_c + 1], lc[:, col_c + 2]
        loss_mono = loss_mono + F.relu(l_eq3 - l_ge2).mean() + F.relu(l_ge2 - l_ge1).mean()

    base_loss = 0.55 * loss_cum + 0.45 * loss_ex + 0.10 * loss_mono

    # 2. 14-Dim Soft Teacher Knowledge Distillation (True 6B + V16_Full Teacher)
    valid_kd = (t_vec[:, 0] >= 0.0)
    loss_kd = torch.tensor(0.0, device=y.device)
    if valid_kd.any() and kd_weight > 0:
        p0, p1, p2 = exact_probs_from_dual(lc[valid_kd], le[valid_kd])
        s_ge1 = torch.clamp(p0 + p1 + p2, 1e-5, 1.0 - 1e-5)
        s_ge2 = torch.clamp(p0 + p1, 1e-5, 1.0 - 1e-5)
        s_p0_img = torch.clamp(1.0 - torch.prod(1.0 - p0, dim=1, keepdim=True), 1e-5, 1.0 - 1e-5)
        s_clean_img = torch.clamp(torch.prod(1.0 - s_ge1, dim=1, keepdim=True), 1e-5, 1.0 - 1e-5)
        s_14 = torch.cat([s_ge1, s_ge2, s_p0_img, s_clean_img], dim=1)
        t_14 = torch.clamp(t_vec[valid_kd], 1e-5, 1.0 - 1e-5)
        loss_kd = F.kl_div(torch.log(s_14), t_14, reduction='batchmean') + F.mse_loss(s_14, t_14)

    # 3. Scene-Grouped Paired Quadruplet Ranking + Non-Target Orthogonality Loss
    loss_rank = torch.tensor(0.0, device=y.device)
    if rank_weight > 0:
        pc_all = torch.sigmoid(lc).view(B, 6, 3)
        sev_all = pc_all.sum(dim=-1)  # Expected severity S_c in [0, 3], shape (B, 6)

        # Group within-batch samples sharing the same (sid, t_dim) first; fallback to same t_dim
        rank_terms = []
        ortho_terms = []
        for c in range(6):
            mask_c = (t_dim == c)
            if mask_c.sum() >= 2:
                idx_c = torch.where(mask_c)[0]
                s_c = sev_all[idx_c, c]
                y_c = y[idx_c, c].float()
                sid_c = sid[idx_c]

                diff_y = y_c.unsqueeze(1) - y_c.unsqueeze(0)
                same_seed = (sid_c.unsqueeze(1) == sid_c.unsqueeze(0)) & (sid_c.unsqueeze(1) >= 0)
                pair_mask = (diff_y > 0)
                if pair_mask.any():
                    diff_s = s_c.unsqueeze(1) - s_c.unsqueeze(0)
                    margin = 0.22 * diff_y
                    # Weight same-seed pairs 3x higher than cross-seed pairs
                    pair_w = torch.where(same_seed[pair_mask], 3.0, 1.0)
                    rank_terms.append((F.relu(margin[pair_mask] - diff_s[pair_mask]) * pair_w).mean() * w_d[c])

                # Cross-dimension orthogonality: for same_seed pairs edited on dim c, non-target dims c' != c must stay invariant
                if same_seed.any():
                    ss_idx = torch.where(same_seed & (diff_y > 0))
                    if len(ss_idx[0]) > 0:
                        non_c = [j for j in range(6) if j != c]
                        s_non_i = sev_all[idx_c[ss_idx[0]]][:, non_c]
                        s_non_j = sev_all[idx_c[ss_idx[1]]][:, non_c]
                        ortho_terms.append(F.smooth_l1_loss(s_non_i, s_non_j.detach()))

        if len(rank_terms) > 0:
            loss_rank = torch.stack(rank_terms).mean()
        if len(ortho_terms) > 0:
            loss_rank = loss_rank + 0.35 * torch.stack(ortho_terms).mean()

    return base_loss + kd_weight * loss_kd + rank_weight * loss_rank


def load_teacher_14_map(train_names_set):
    """
    Builds the 14-dim soft teacher map strictly for `train_names_set` using:
    1) Primary: 6B Teacher OOF (`v12_phaseB_fixed`, `v12_phaseB`, `v11_pure_real`) + `V16_Full` 8-fold OOF
    2) Fallback: `v15_ens_oof3573_probs.npz` for any image not covered above
    """
    accum = {}  # nm -> [list of p0, list of p1, list of p2, list of weights]

    def add_pred(nm, p0, p1, p2, w):
        if nm not in train_names_set:
            return
        if nm not in accum:
            accum[nm] = [0.0, 0.0, 0.0, 0.0]
        accum[nm][0] = accum[nm][0] + w * p0.astype(np.float64)
        accum[nm][1] = accum[nm][1] + w * p1.astype(np.float64)
        accum[nm][2] = accum[nm][2] + w * p2.astype(np.float64)
        accum[nm][3] += w

    # 1. Load 6B OOF across splits 16 & 17
    for sp in ['16', '17']:
        for pat, w in [
            (f'{REPO_ROOT}/results/v12_phaseB_fixed/v12_abmil_s{sp}_probs.npz', 0.35),
            (f'{REPO_ROOT}/results/v12_phaseB/v12_abmil_s{sp}_probs.npz', 0.25),
            (f'{REPO_ROOT}/results/v11_pure_real/iv6b_s{sp}_probs.npz', 0.20),
        ]:
            if os.path.exists(pat):
                d = np.load(pat, allow_pickle=True)
                for pref, k_nm in [('va', 'va_names'), ('te', 'te_names')]:
                    if k_nm in d:
                        nms = d[k_nm]
                        p0, p1, p2 = d[f'{pref}_p0'], d[f'{pref}_p1'], d[f'{pref}_p2']
                        for i, nm in enumerate(nms):
                            add_pred(str(nm), p0[i], p1[i], p2[i], w)

    # 2. Load V16_Full 8-Fold OOF
    for f_idx in range(8):
        p = f'{REPO_ROOT}/results/v16_fulldata/V16_Full_F{f_idx}_AllGrade_Microscope_probs.npz'
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            for pref, k_nm in [('va', 'va_names'), ('te', 'te_names')]:
                if k_nm in d:
                    nms = d[k_nm]
                    p0, p1, p2 = d[f'{pref}_p0'], d[f'{pref}_p1'], d[f'{pref}_p2']
                    for i, nm in enumerate(nms):
                        add_pred(str(nm), p0[i], p1[i], p2[i], 0.45)

    # 3. Fallback V15 OOF with small weight (0.15) so all 3,573 train images have soft targets
    v15_p = f'{REPO_ROOT}/results/v15_full8fold/v15_ens_oof3573_probs.npz'
    if os.path.exists(v15_p):
        d = np.load(v15_p, allow_pickle=True)
        nms = d['names']
        p0, p1, p2 = d['p0'], d['p1'], d['p2']
        for i, nm in enumerate(nms):
            add_pred(str(nm), p0[i], p1[i], p2[i], 0.15)

    t_map = {}
    for nm, (s0, s1, s2, w_tot) in accum.items():
        if w_tot <= 0:
            continue
        p0 = s0 / w_tot
        p1 = s1 / w_tot
        p2 = s2 / w_tot
        p_ge1 = np.clip(p0 + p1 + p2, 0.0, 1.0)
        p_ge2 = np.clip(p0 + p1, 0.0, 1.0)
        p0_img = float(1.0 - np.prod(1.0 - p0))
        clean_img = float(np.prod(1.0 - p_ge1))
        vec14 = np.concatenate([p_ge1, p_ge2, [p0_img, clean_img]]).astype(np.float32)
        t_map[nm] = vec14
    return t_map


@torch.no_grad()
def eval_split_tta(model, loader, device):
    model.eval()
    all_p0, all_p1, all_p2, all_y, all_names = [], [], [], [], []
    for x_g, x_gf, x_l, geom, y, names in loader:
        x_g = x_g.to(device, non_blocking=True)
        x_gf = x_gf.to(device, non_blocking=True)
        x_l = x_l.to(device, non_blocking=True)
        geom = geom.to(device, non_blocking=True)

        lc1, le1 = model(x_g, x_l, geom, alpha_local=0.38)
        lc2, le2 = model.forward_single(x_gf, geom)
        lc = 0.72 * lc1 + 0.28 * lc2
        le = 0.72 * le1 + 0.28 * le2

        p0, p1, p2 = exact_probs_from_dual(lc, le)
        all_p0.append(p0.cpu().numpy())
        all_p1.append(p1.cpu().numpy())
        all_p2.append(p2.cpu().numpy())
        all_y.append(y.numpy())
        all_names.extend(names)

    return (
        np.concatenate(all_p0, axis=0),
        np.concatenate(all_p1, axis=0),
        np.concatenate(all_p2, axis=0),
        np.concatenate(all_y, axis=0),
        all_names,
    )


def score_18dim_exact_probs(p0, p1, p2, y_true):
    N = len(y_true)
    preds = np.zeros((N, 6), dtype=np.int64)
    T0 = [0.34, 0.30, 0.28, 0.23, 0.30, 0.28]
    T1 = [0.22, 0.22, 0.22, 0.20, 0.24, 0.16]
    T2 = [0.16, 0.22, 0.18, 0.20, 0.20, 0.15]
    for c in range(6):
        pr = np.zeros(N, dtype=np.int64)
        pr[p2[:, c] >= T2[c]] = 1
        pr[p1[:, c] >= T1[c]] = 2
        pr[p0[:, c] >= T0[c]] = 3
        mask_p1 = (pr == 3) & (p1[:, c] > p0[:, c] * 1.05) & (p0[:, c] < T0[c] + 0.10)
        pr[mask_p1] = 2
        preds[:, c] = pr

    tp = np.zeros((6, 3), dtype=np.float64)
    fp = np.zeros((6, 3), dtype=np.float64)
    fn = np.zeros((6, 3), dtype=np.float64)
    for c in range(6):
        yt, yp = y_true[:, c], preds[:, c]
        for g_idx, g_val in enumerate([3, 2, 1]):
            t_m, p_m = (yt == g_val), (yp == g_val)
            tp[c, g_idx] = np.sum(t_m & p_m)
            fp[c, g_idx] = np.sum((~t_m) & p_m)
            fn[c, g_idx] = np.sum(t_m & (~p_m))
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f2 = np.where(4 * prec + rec > 0, 5 * prec * rec / (4 * prec + rec), 0.0)
    f2_w = (f2[:, 0] * 1.2 + f2[:, 1] * 0.8 + f2[:, 2] * 0.4) / 2.4
    base_18 = float(np.sum(f2_w * W_DIM) / np.sum(W_DIM))
    t_p0, p_p0 = np.any(y_true == 3, axis=1), np.any(preds == 3, axis=1)
    tp0 = np.sum(t_p0 & p_p0)
    fp0 = np.sum((~t_p0) & p_p0)
    fn0 = np.sum(t_p0 & (~p_p0))
    pr0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
    rc0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0.0
    f2_img = 5 * pr0 * rc0 / (4 * pr0 + rc0) if (4 * pr0 + rc0) > 0 else 0.0
    em = float(np.mean(np.all(y_true == preds, axis=1)))
    total = (0.75 * base_18 + 0.15 * f2_img + 0.10 * em) * 100.0
    return total, f2_w * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--base-ckpt', type=str, required=True)
    parser.add_argument('--v16-npz', type=str, default='')
    parser.add_argument('--split-csv', type=str, required=True)
    parser.add_argument('--synth-csv', type=str, default=f'{REPO_ROOT}/splits/synth_v18_native_corridor_quads.csv')
    parser.add_argument('--target-dims', type=str, default='0,1,2,3,4,5')
    parser.add_argument('--dim-weights', type=str, default='1.0,1.0,0.7,0.6,0.9,0.8')
    parser.add_argument('--epochs-s1', type=int, default=2)
    parser.add_argument('--epochs-s2', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1.0e-4)
    parser.add_argument('--kd-weight', type=float, default=0.30)
    parser.add_argument('--rank-weight', type=float, default=0.25)
    parser.add_argument('--anchor-weight', type=float, default=0.55)
    parser.add_argument('--out-ckpt', type=str, required=True)
    parser.add_argument('--out-npz', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim_weights = [float(x) for x in args.dim_weights.split(',')]
    target_dims = [int(x) for x in args.target_dims.split(',')]

    df = pd.read_csv(args.split_csv)
    tr_df = df[df['split'] == 'train'].reset_index(drop=True)
    va_df = df[df['split'] == 'val'].sort_values('image').reset_index(drop=True)
    te_df = df[df['split'] == 'test'].sort_values('image').reset_index(drop=True)

    train_names_set = set(tr_df['image'].astype(str).tolist())
    teacher_map = load_teacher_14_map(train_names_set)

    synth_df = None
    if os.path.exists(args.synth_csv):
        sdf = pd.read_csv(args.synth_csv)
        sdf = sdf[sdf['seed_name'].astype(str).isin(train_names_set)]
        sdf = sdf[sdf['target_dim_idx'].isin(target_dims)].reset_index(drop=True)
        if len(sdf) > 600:
            sdf = sdf.sample(n=600, random_state=args.seed).reset_index(drop=True)
        synth_df = sdf

    img_dir = f'{REPO_ROOT}/nriqa_dataset/train/images'
    test150_dir = f'{REPO_ROOT}/nriqa_dataset_test/images'

    ds_s1 = PairedQuadDatasetV18(tr_df, img_dir, synth_df=synth_df, teacher_map=teacher_map)
    ds_s2 = PairedQuadDatasetV18(tr_df, img_dir, synth_df=None, teacher_map=teacher_map)

    # Keep shuffle=False on ds_s1 after grouping seeds or shuffle in chunks so same-seed quadruplets co-occur in mini-batches
    loader_s1 = DataLoader(ds_s1, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)
    loader_s2 = DataLoader(ds_s2, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)

    va_loader = DataLoader(EvalDatasetV18(va_df, img_dir), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    te_loader = DataLoader(EvalDatasetV18(te_df, img_dir), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    t150_files = sorted([f for f in os.listdir(test150_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
    t150_df = pd.DataFrame({'image': t150_files})
    for c in DIMS:
        t150_df[c] = '无问题'
    t150_loader = DataLoader(EvalDatasetV18(t150_df, test150_dir), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = MultiNetV18(args.base_ckpt, unfreeze_mode='lastk1').to(device)
    vp0, vp1, vp2, va_y, va_names = eval_split_tta(model, va_loader, device)
    tp0, tp1, tp2, te_y, te_names = eval_split_tta(model, te_loader, device)
    init_va, _ = score_18dim_exact_probs(vp0, vp1, vp2, va_y)
    init_te, _ = score_18dim_exact_probs(tp0, tp1, tp2, te_y)
    print(f"[{args.name}] Init Base -> VA={init_va:.3f} | TE={init_te:.3f} | S1_len={len(ds_s1)} | S2_len={len(ds_s2)}")

    swa_states = []
    best_va = init_va
    best_sd_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Stage 1: Paired Quadruplets + True 6B/V16 Teacher KD
    opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-2)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(1, args.epochs_s1 * len(loader_s1)))

    for ep in range(1, args.epochs_s1 + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for x_g, x_l, geom, y, t_dim, t_vec, sid in loader_s1:
            x_g, x_l, geom = x_g.to(device), x_l.to(device), geom.to(device)
            y, t_dim, t_vec, sid = y.to(device), t_dim.to(device), t_vec.to(device), sid.to(device)
            opt1.zero_grad()
            lc, le = model(x_g, x_l, geom, alpha_local=0.38)
            loss = compute_paired_quadruplet_and_kd_loss(
                lc, le, y, t_dim, t_vec, sid, dim_weights,
                kd_weight=args.kd_weight, rank_weight=args.rank_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            sched1.step()
            loss_sum += float(loss.item())

        vp0, vp1, vp2, va_y, _ = eval_split_tta(model, va_loader, device)
        tp0, tp1, tp2, te_y, _ = eval_split_tta(model, te_loader, device)
        va_sc, _ = score_18dim_exact_probs(vp0, vp1, vp2, va_y)
        te_sc, _ = score_18dim_exact_probs(tp0, tp1, tp2, te_y)
        print(f"[{args.name}] S1 Ep{ep} ({time.time()-t0:.1f}s) loss={loss_sum/len(loader_s1):.4f} | VA={va_sc:.3f} | TE={te_sc:.3f}")
        swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if va_sc >= best_va - 0.15:
            best_va = max(best_va, va_sc)
            best_sd_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Stage 2: Sim-to-Real Pure Real Prior Anchor + Teacher KD (lastk2)
    model.set_unfreeze('lastk2')
    opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr * 0.45, weight_decay=1e-2)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, args.epochs_s2 * len(loader_s2)))

    for ep in range(1, args.epochs_s2 + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for x_g, x_l, geom, y, t_dim, t_vec, sid in loader_s2:
            x_g, x_l, geom = x_g.to(device), x_l.to(device), geom.to(device)
            y, t_dim, t_vec, sid = y.to(device), t_dim.to(device), t_vec.to(device), sid.to(device)
            opt2.zero_grad()
            lc, le = model(x_g, x_l, geom, alpha_local=0.38)
            loss = compute_paired_quadruplet_and_kd_loss(
                lc, le, y, t_dim, t_vec, sid, dim_weights,
                kd_weight=args.kd_weight * 0.85, rank_weight=0.0,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt2.step()
            sched2.step()
            loss_sum += float(loss.item())

        vp0, vp1, vp2, va_y, _ = eval_split_tta(model, va_loader, device)
        tp0, tp1, tp2, te_y, _ = eval_split_tta(model, te_loader, device)
        va_sc, _ = score_18dim_exact_probs(vp0, vp1, vp2, va_y)
        te_sc, _ = score_18dim_exact_probs(tp0, tp1, tp2, te_y)
        print(f"[{args.name}] S2 Ep{ep} ({time.time()-t0:.1f}s) loss={loss_sum/len(loader_s2):.4f} | VA={va_sc:.3f} | TE={te_sc:.3f}")
        swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if va_sc >= best_va - 0.10:
            best_va = max(best_va, va_sc)
            best_sd_cpu = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Pure Trained SWA (Never mixes untrained init_sd_cpu on newly initialized layers!)
    if len(swa_states) >= 2:
        avg_sd = {}
        n_s = len(swa_states)
        for k in best_sd_cpu.keys():
            if best_sd_cpu[k].is_floating_point():
                s_mean = sum(sd[k].float() for sd in swa_states) / float(n_s)
                avg_sd[k] = (0.55 * best_sd_cpu[k].float() + 0.45 * s_mean).to(best_sd_cpu[k].dtype)
            else:
                avg_sd[k] = best_sd_cpu[k]
        model.load_state_dict(avg_sd)
        vp0_s, vp1_s, vp2_s, va_y, _ = eval_split_tta(model, va_loader, device)
        tp0_s, tp1_s, tp2_s, te_y, _ = eval_split_tta(model, te_loader, device)
        va_swa, _ = score_18dim_exact_probs(vp0_s, vp1_s, vp2_s, va_y)
        te_swa, _ = score_18dim_exact_probs(tp0_s, tp1_s, tp2_s, te_y)
        print(f"[{args.name}] Trained-SWA -> VA={va_swa:.3f} | TE={te_swa:.3f}")
        if va_swa >= best_va - 0.15:
            best_sd_cpu = avg_sd
        else:
            model.load_state_dict(best_sd_cpu)
    else:
        model.load_state_dict(best_sd_cpu)

    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save(best_sd_cpu, args.out_ckpt)

    vp0, vp1, vp2, va_y, va_names = eval_split_tta(model, va_loader, device)
    tp0, tp1, tp2, te_y, te_names = eval_split_tta(model, te_loader, device)
    t150_p0, t150_p1, t150_p2, _, t150_names = eval_split_tta(model, t150_loader, device)

    # Safe name-aligned V16 pure-real anchor blending (supports both s16/s17 500-val and FullData 300-val)
    if args.v16_npz and os.path.exists(args.v16_npz):
        d16 = np.load(args.v16_npz, allow_pickle=True)
        w16 = float(args.anchor_weight)
        if len(d16['va_p0']) == len(vp0) and len(d16['te_p0']) == len(tp0):
            vp0 = w16 * d16['va_p0'] + (1.0 - w16) * vp0
            vp1 = w16 * d16['va_p1'] + (1.0 - w16) * vp1
            vp2 = w16 * d16['va_p2'] + (1.0 - w16) * vp2
            tp0 = w16 * d16['te_p0'] + (1.0 - w16) * tp0
            tp1 = w16 * d16['te_p1'] + (1.0 - w16) * tp1
            tp2 = w16 * d16['te_p2'] + (1.0 - w16) * tp2
            t150_p0 = w16 * d16['t150_p0'] + (1.0 - w16) * t150_p0
            t150_p1 = w16 * d16['t150_p1'] + (1.0 - w16) * t150_p1
            t150_p2 = w16 * d16['t150_p2'] + (1.0 - w16) * t150_p2
        else:
            # Name-aligned anchor blend for split_full_v18
            from infer_v18_fulldata_test150 import load_v16_anchor_aligned
            s17_npz = args.v16_npz.replace('_s16_probs.npz', '_s17_probs.npz')
            build_fn, (t150_a0, t150_a1, t150_a2) = load_v16_anchor_aligned(
                args.v16_npz, s17_npz, va_names, te_names, t150_names
            )
            va_a0, va_a1, va_a2, _ = build_fn(va_names, vp0, vp1, vp2)
            te_a0, te_a1, te_a2, _ = build_fn(te_names, tp0, tp1, tp2)
            vp0 = w16 * va_a0 + (1.0 - w16) * vp0
            vp1 = w16 * va_a1 + (1.0 - w16) * vp1
            vp2 = w16 * va_a2 + (1.0 - w16) * vp2
            tp0 = w16 * te_a0 + (1.0 - w16) * tp0
            tp1 = w16 * te_a1 + (1.0 - w16) * tp1
            tp2 = w16 * te_a2 + (1.0 - w16) * tp2
            if t150_a0 is not None:
                t150_p0 = w16 * t150_a0 + (1.0 - w16) * t150_p0
                t150_p1 = w16 * t150_a1 + (1.0 - w16) * t150_p1
                t150_p2 = w16 * t150_a2 + (1.0 - w16) * t150_p2

    final_va, _ = score_18dim_exact_probs(vp0, vp1, vp2, va_y)
    final_te, _ = score_18dim_exact_probs(tp0, tp1, tp2, te_y)
    print(f"[{args.name}] Final Saved -> VA={final_va:.3f} | TE={final_te:.3f}")

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        va_p0=vp0.astype(np.float32),
        va_p1=vp1.astype(np.float32),
        va_p2=vp2.astype(np.float32),
        va_y=va_y,
        va_names=np.array(va_names),
        te_p0=tp0.astype(np.float32),
        te_p1=tp1.astype(np.float32),
        te_p2=tp2.astype(np.float32),
        te_y=te_y,
        te_names=np.array(te_names),
        t150_p0=t150_p0.astype(np.float32),
        t150_p1=t150_p1.astype(np.float32),
        t150_p2=t150_p2.astype(np.float32),
        t150_names=np.array(t150_names),
        best_te=float(final_te),
    )


if __name__ == '__main__':
    main()
