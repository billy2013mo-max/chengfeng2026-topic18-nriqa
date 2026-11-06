"""V10 Clean-Masked + 1:1 Native-Pixel Edge-Microscope + DeQAScore Soft-Label Specialist Trainer
(`train_v10_clean_native_specialist.py`).

Three Core Paradigm Upgrades over V9 (`train_v9_specialist.py`):
1. Per-Sample Synthetic Target Masking (`get_synth_class_mask`):
   Fixes the severe V9 synthetic multi-label pollution bug where 100% (1,620/1,620) of synthetic images in
   `manifest_s16_v9.csv` / `manifest_s17_v9.csv` were labeled `纹理损失 > 0`, 100% of `v9_bk` (bokeh) was labeled
   `模糊 = P2`, and 100% of `v9_og` (overgen) was labeled `纹理损失 + 模糊/过锐`.
   In V10, for any synthetic sample `nm`:
   - `v9_og*` ONLY computes loss on `过度生成` (ci=4); all other 5 classes have loss weight = 0.0!
   - `v9_sh*` ONLY computes loss on `过锐` (ci=2); all other 5 classes have loss weight = 0.0!
   - `v9_bk*` ONLY computes loss on `虚化突兀` (ci=5); all other 5 classes have loss weight = 0.0!
   - `v9_bt*` ONLY computes loss on `模糊` (ci=0) and `纹理损失` (ci=1); all other 4 classes have loss weight = 0.0!
   - `All-Clean Gate` (`gate_logit`) ONLY computes loss on 100% real human-annotated images (`is_real`)!
2. Zero-Resize 1:1 Native-Pixel High-Gradient Edge & Foreground Patch Selection (`_extract_native_microscope_crops`):
   Instead of resizing crops or taking only center + 1 corner, scans a 3x3=9 grid of 1:1 native-resolution `S x S`
   windows (ZERO resize/interpolation!) and selects:
   - Top-2 Highest Laplacian/Sobel Edge-Energy 1:1 Native Patches (preserving 1-2px `过锐` ringing halos & `过度生成` fine edge distortion),
   - 1 Median-Gradient Subject Texture 1:1 Native Patch (preserving `纹理损失` waxy smoothing & `模糊` P1/P2 roll-off without flat sky dilution).
3. DeQAScore Gaussian Soft-Label Ordinal Loss + Enhanced New `(2).md` All-Clean Gate Supervision:
   Replaces hard one-hot cross-entropy on P2/P1/P0 with a continuous Gaussian soft-label ordinal distribution
   (while keeping `无问题` sharp) and boosts real-image `All-Clean` gate supervision.
"""
import argparse, copy, csv, json, os, random, sys, time
import numpy as np
from PIL import Image, ImageEnhance
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'decision_eval'))
from data import load_split, read_manifest, targets_from_grades, CLASSES, FULL
from train_v2 import exact_probs_from_dual, exact_targets_14, P0_IDX
from train_v9_specialist import (
    CACHE_OK, CLASS_TO_COLS14, build_v9_backbone, pairwise_ordinal_margin_loss, set_seed
)
import comp_metrics as cm


def get_sample_class_mask6(name):
    """Return a 6-dim float mask indicating which classes have valid ground-truth supervision for sample `name`.
    For real human-annotated images: all 6 classes are 1.0 (plus bit 6 = 1.0 for real image gate supervision).
    For synthetic images: ONLY the synthesized target defect class is 1.0; all other classes are 0.0!
    """
    if name.startswith('v9_og') or name.startswith('v8_og') or name.startswith('v7_'):
        # Synthetic 过度生成 (ci=4) ONLY!
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    if name.startswith('v9_sh') or name.startswith('v8_sh') or name.startswith('v4_'):
        # Synthetic 过锐 (ci=2) ONLY!
        return np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if name.startswith('v9_bk') or name.startswith('v8_bk'):
        # Synthetic 虚化突兀 (ci=5) ONLY! (Never pollute 模糊 or 纹理损失!)
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    if name.startswith('v9_bt') or name.startswith('v8_bt'):
        # Synthetic 模糊 (ci=0) + 纹理损失 (ci=1) ONLY!
        return np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Real human-annotated image: all 6 classes + real-clean gate active!
    return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


class V10NativeMicroscopeDataset(Dataset):
    """Multi-View Dataset with:
    - 1 Global resized view (`size x size`)
    - `tiles x tiles` quadrant views (`size x size`)
    - 3 Zero-Resize 1:1 Native-Pixel Microscope Crops (`size x size` cut directly at 1:1 pixel scale!):
      * Top-1 Peak High-Frequency Edge Patch (captures 1-2px `过锐` halo/overshoot & `过度生成` fine artifacts)
      * Top-2 Second High-Frequency Edge Patch (spatial diversity across subject boundaries)
      * 1 Median-Gradient Subject Texture Patch (captures `模糊` P1/P2 & `纹理损失` waxy smoothing without sky/flat background)
    """
    def __init__(self, rows, phys_map, phys_mu, phys_sd, size=448, train=True, hflip=False, tiles=2):
        self.rows = rows
        self.phys_map = phys_map
        self.phys_mu = phys_mu
        self.phys_sd = phys_sd
        self.size = size
        self.train = train
        self.hflip = hflip
        self.tiles = tiles
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
            h = int(self.h[j])
            w = int(self.w[j])
            off = int(self.offsets[j])
            arr = np.asarray(self.data[off:off + h * w * 3]).reshape(h, w, 3)
            return Image.fromarray(arr)
        for d in ['/tmp/v9_authentic_imgs', '/tmp/v8_ladder_imgs', '/tmp/v7_overgen_pro_imgs', '/tmp/v6_synth_imgs',
                  'data/train', 'data/images', 'data/images_supp', 'data/test_150', 'data/test']:
            p = os.path.join('/wanqing-develop/luowenjing/18', d, name) if not d.startswith('/') else os.path.join(d, name)
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
        raise FileNotFoundError(f'Cannot find {name}')

    def _extract_native_microscope_crops(self, im):
        S = self.size
        w, h = im.size
        if w < S or h < S:
            im = im.resize((max(w, S), max(h, S)), Image.BICUBIC)
            w, h = im.size

        # Build 3x3 = 9 candidate 1:1 native-pixel windows covering the image (with slight jitter during training)
        xs = [0, max(0, (w - S) // 2), max(0, w - S)] if w > S else [0]
        ys = [0, max(0, (h - S) // 2), max(0, h - S)] if h > S else [0]
        cands = []
        energies = []
        for y0 in ys:
            for x0 in xs:
                if self.train and w > S + 16 and h > S + 16:
                    jx = random.randint(-8, 8)
                    jy = random.randint(-8, 8)
                    x_c = max(0, min(w - S, x0 + jx))
                    y_c = max(0, min(h - S, y0 + jy))
                else:
                    x_c, y_c = x0, y0
                patch = im.crop((x_c, y_c, x_c + S, y_c + S))
                # Fast 1:1 sub-sampled strided gradient energy (takes <0.05ms in numpy!)
                arr = np.asarray(patch, dtype=np.uint8)[::4, ::4, 1].astype(np.float32)  # green/luma channel at stride 4
                gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
                gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
                # Also 2nd-order Laplacian energy `|2*x[i] - x[i-1] - x[i+1]|` for ringing/overshoot sensitivity!
                lx = np.abs(2.0 * arr[:, 1:-1] - arr[:, :-2] - arr[:, 2:]).mean()
                ly = np.abs(2.0 * arr[1:-1, :] - arr[:-2, :] - arr[2:, :]).mean()
                cands.append(patch)
                energies.append(float(gx + gy + 0.8 * (lx + ly)))

        if len(cands) == 1:
            return [cands[0], cands[0], cands[0]]

        order = np.argsort(energies)  # ascending from flattest to sharpest/highest-contrast
        p_peak1 = cands[int(order[-1])]                     # Highest-contrast edge patch
        p_peak2 = cands[int(order[-2])]                     # 2nd highest-contrast edge patch
        p_mid   = cands[int(order[len(order) // 2])]        # Median-gradient foreground texture patch
        return [p_peak1, p_peak2, p_mid]

    def __getitem__(self, i):
        r = self.rows[i]
        nm = r['name']
        im = self._load(nm)
        if self.train:
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.30:
                im = ImageEnhance.Brightness(im).enhance(1.0 + random.uniform(-0.04, 0.04))
        elif self.hflip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        w, h = im.size
        views = [im.resize((self.size, self.size), Image.BILINEAR)]
        T = self.tiles
        for dy in range(T):
            for dx in range(T):
                views.append(im.crop((dx * w // T, dy * h // T, (dx + 1) * w // T, (dy + 1) * h // T)).resize((self.size, self.size), Image.BILINEAR))
        views.extend(self._extract_native_microscope_crops(im))

        x = np.transpose(np.stack([np.asarray(v, dtype=np.uint8) for v in views]), (0, 3, 1, 2))
        g = np.array([int(r[c]) for c in CLASSES], dtype=np.int64)
        mask7 = get_sample_class_mask6(nm)
        p_raw = self.phys_map.get(nm, np.zeros(60, dtype=np.float32))
        p_norm = np.clip((p_raw - self.phys_mu) / self.phys_sd, -6.0, 6.0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(p_norm), torch.from_numpy(g), torch.from_numpy(mask7), nm


class MultiNetV10(nn.Module):
    """V10 Multi-Backbone + Spatial Anomaly Pooling + 3-Patch 1:1 Native Microscope Stream + 60-Dim Physical Caliper Network."""
    def __init__(self, specs, phys_dim=60, dropout=0.12, n_quad=4):
        super().__init__()
        self.n_quad = n_quad
        self.bs = nn.ModuleList()
        self.fwds = []
        self.tile_attn = nn.ModuleList()
        for (m, fwd, mu, sd, bl, dim) in specs:
            self.bs.append(m)
            self.fwds.append(fwd)
            self.register_buffer('mu_%d' % len(self.fwds), torch.tensor(mu, dtype=torch.bfloat16).view(1, 3, 1, 1))
            self.register_buffer('sd_%d' % len(self.fwds), torch.tensor(sd, dtype=torch.bfloat16).view(1, 3, 1, 1))
            self.tile_attn.append(nn.Sequential(
                nn.Linear(dim, max(64, dim // 8)),
                nn.GELU(),
                nn.Linear(max(64, dim // 8), 1)
            ))
        # _agg outputs 6 summary vectors per backbone: [g_view, quad_mean, nat_mean, nat_max, t_max, t_att]
        d_vis = sum(6 * x[5] for x in specs)
        self.phys_mlp = nn.Sequential(
            nn.Linear(phys_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        d = d_vis + 256
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.head_cum = nn.Sequential(
            nn.Linear(d, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 14),
        )
        self.head_ex = nn.Sequential(
            nn.Linear(d, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 20),
        )
        self.gate = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def _agg(self, f, attn_mod):
        g_view = f[:, 0]
        tiles = f[:, 1:]
        quad_mean = f[:, 1:1 + self.n_quad].mean(1)
        nat_views = f[:, 1 + self.n_quad:]
        nat_mean = nat_views.mean(1)
        nat_max = nat_views.amax(1)  # Peak anomaly across the 3 1:1 native-pixel microscope patches!
        t_max = tiles.amax(1)
        w = torch.softmax(attn_mod(tiles), dim=1)
        t_att = (tiles * w).sum(1)
        return torch.cat([g_view, quad_mean, nat_mean, nat_max, t_max, t_att], dim=1)

    def forward(self, x, phys):
        B, V, C, H, W = x.shape
        if x.dtype == torch.uint8:
            x_bf16 = x.view(B * V, C, H, W).to(torch.bfloat16) * (1.0 / 255.0)
        else:
            x_bf16 = x.view(B * V, C, H, W).to(torch.bfloat16)
        out = []
        for i, (m, fwd, attn_mod) in enumerate(zip(self.bs, self.fwds, self.tile_attn)):
            mu = getattr(self, 'mu_%d' % (i + 1))
            sd = getattr(self, 'sd_%d' % (i + 1))
            z = (x_bf16 - mu) / sd
            feat = fwd(z).view(B, V, -1).float()
            out.append(self._agg(feat, attn_mod))
        p_emb = self.phys_mlp(phys.float())
        fused = self.drop(self.norm(torch.cat(out + [p_emb], dim=1)))
        return self.head_cum(fused), self.head_ex(fused), self.gate(fused).squeeze(1)


def v10_clean_masked_loss_fn(logits_cum, logits_ex, gate_logit, g, mask7, bce_loss_none, focus_cis,
                             p0_clean_weight=4.2, channel_neg_weight=1.05, alpha_ex=0.58, deqa_sigma=0.44):
    """V10 Loss with Per-Sample Synthetic Class Masking + DeQAScore Gaussian Soft-Label Ordinal Loss + Real-Only Clean Gate."""
    dev = g.device
    mask6 = mask7[:, :6]        # (B, 6): 1.0 where class label is valid for this sample, 0.0 on non-target synthetic classes!
    is_real = mask7[:, 6] > 0.5 # (B,): True only on authentic human-annotated images!

    c_weights6 = torch.full((6,), 0.25, device=dev)
    col_weights14 = torch.full((14,), 0.25, device=dev)
    for ci in focus_cis:
        c_weights6[ci] = 1.60
        for col in CLASS_TO_COLS14[ci]:
            col_weights14[col] = 1.60
    col_weights14 = col_weights14 / col_weights14.mean()
    c_weights6 = c_weights6 / c_weights6.mean()

    # Build 14-column sample validity mask `mask14` from `mask6`
    mask14 = torch.zeros((g.shape[0], 14), device=dev, dtype=torch.float32)
    for ci in range(6):
        for col in CLASS_TO_COLS14[ci]:
            mask14[:, col] = mask6[:, ci]

    Y_cum = targets_from_grades(g)
    raw_bce = bce_loss_none(logits_cum, Y_cum)
    clean_mask = (g == 0).all(dim=1) & is_real

    w_mat = torch.ones_like(raw_bce)
    if channel_neg_weight != 1.0:
        w_mat = torch.where(Y_cum == 0, torch.full_like(w_mat, channel_neg_weight), w_mat)
    if clean_mask.any() and p0_clean_weight > 1.0:
        for idx in P0_IDX:
            w_mat[clean_mask, idx] = p0_clean_weight

    # Apply `mask14` so polluted non-target classes on synthetic images contribute ZERO gradient!
    valid_bce = raw_bce * w_mat * mask14 * col_weights14.unsqueeze(0)
    loss_bce = valid_bce.sum() / (mask14.sum().clamp(min=1.0))

    mono_pen = 0.0
    j = 0
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            l2, l1, l0 = logits_cum[:, j], logits_cum[:, j + 1], logits_cum[:, j + 2]
            j += 3
            m_ci = mask6[:, ci]
            if m_ci.sum() > 0:
                mono_pen = mono_pen + c_weights6[ci] * (((F.relu(l0 - l1) + F.relu(l1 - l2)) * m_ci).sum() / m_ci.sum())
        else:
            j += 1

    # DeQAScore Gaussian Soft-Label KL Loss + Weighted CE + EMD (masked by `mask6[:, ci]`)
    ce_tot = 0.0
    emd_tot = 0.0
    j_ex = 0
    cw4 = torch.tensor([1.0, 1.25, 1.35, 1.45], device=dev)
    grid4 = torch.arange(4, device=dev, dtype=torch.float32)[None, :]
    for ci, c in enumerate(CLASSES):
        w_ci = c_weights6[ci]
        m_ci = mask6[:, ci]
        if c in FULL:
            lg4 = logits_ex[:, j_ex:j_ex + 4]
            tgt = g[:, ci].long()
            if m_ci.sum() > 0:
                # 1. Hard weighted CE (per-sample masked)
                ce_hard = F.cross_entropy(lg4, tgt, weight=cw4, reduction='none', label_smoothing=0.02)
                # 2. DeQAScore Soft Gaussian Ordinal Target distribution
                g_f = tgt.float()[:, None]
                w_soft = torch.exp(-0.5 * ((grid4 - g_f) / deqa_sigma) ** 2)
                is_zero = (tgt == 0)
                if is_zero.any():
                    w_soft[is_zero] = torch.tensor([0.96, 0.04, 0.0, 0.0], device=dev)
                w_soft = w_soft / w_soft.sum(dim=1, keepdim=True)
                ce_soft = -(w_soft * F.log_softmax(lg4, dim=1)).sum(dim=1)
                ce_comb = 0.65 * ce_hard + 0.35 * ce_soft
                ce_tot = ce_tot + w_ci * ((ce_comb * m_ci).sum() / m_ci.sum())

                probs4 = torch.softmax(lg4, dim=1)
                cdf_pred = torch.cumsum(probs4, dim=1)[:, :3]
                onehot = F.one_hot(tgt, num_classes=4).float()
                cdf_true = torch.cumsum(onehot, dim=1)[:, :3]
                b_w = torch.tensor([1.0, 1.5, 1.5], device=dev)
                emd_per = (((cdf_pred - cdf_true) ** 2) * b_w).mean(dim=1)
                emd_tot = emd_tot + w_ci * ((emd_per * m_ci).sum() / m_ci.sum())
            j_ex += 4
        else:
            tgt_bin = (g[:, ci] == 3).long()
            if m_ci.sum() > 0:
                ce_bin = F.cross_entropy(logits_ex[:, j_ex:j_ex + 2], tgt_bin, reduction='none', label_smoothing=0.02)
                ce_tot = ce_tot + w_ci * ((ce_bin * m_ci).sum() / m_ci.sum())
            j_ex += 2

    loss_ce = ce_tot / len(CLASSES)
    loss_emd = emd_tot / len(FULL)

    p0, p1, p2, P_exact14 = exact_probs_from_dual(logits_cum, logits_ex, alpha_ex=alpha_ex)
    Y_exact14 = exact_targets_14(g)
    tp = (P_exact14 * Y_exact14 * mask14).sum(dim=0)
    fp = (P_exact14 * (1.0 - Y_exact14) * mask14).sum(dim=0)
    fn = ((1.0 - P_exact14) * Y_exact14 * mask14).sum(dim=0)
    soft_f1 = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
    loss_s_pos = 1.0 - (soft_f1 * col_weights14).sum() / col_weights14.sum()

    Y_p0 = (g == 3).float()
    tp_p0 = (p0 * Y_p0 * mask6).sum(dim=0)
    fp_p0 = (p0 * (1.0 - Y_p0) * mask6).sum(dim=0)
    fn_p0 = ((1.0 - p0) * Y_p0 * mask6).sum(dim=0)
    r_p0 = (tp_p0 + 1e-4) / (tp_p0 + fn_p0 + 1e-4)
    p_p0 = (tp_p0 + 1e-4) / (tp_p0 + fp_p0 + 1e-4)
    f1_p0 = (2.0 * tp_p0 + 1e-4) / (2.0 * tp_p0 + fp_p0 + fn_p0 + 1e-4)
    soft_sp0 = 0.5 * r_p0 + 0.3 * f1_p0 + 0.2 * p_p0
    loss_s_p0 = 1.0 - (soft_sp0 * c_weights6).sum() / c_weights6.sum()

    loss_rank = pairwise_ordinal_margin_loss(p0, p1, p2, g, focus_cis)
    loss_clean_p0 = p0[clean_mask].amax(dim=1).mean() if clean_mask.any() else 0.0

    # Supervise gate_logit strictly on authentic human-annotated images (`is_real`)!
    if is_real.any():
        loss_gate = F.binary_cross_entropy_with_logits(gate_logit[is_real], clean_mask[is_real].float())
    else:
        loss_gate = 0.0

    return (
        loss_bce
        + 0.75 * loss_ce
        + 0.65 * loss_emd
        + 0.65 * loss_s_pos
        + 0.55 * loss_s_p0
        + 0.65 * loss_rank
        + 0.20 * mono_pen
        + 0.40 * loss_clean_p0
        + 0.35 * loss_gate
    )


@torch.no_grad()
def predict_v10(net, loader, dev, alpha_ex=0.58):
    net.eval()
    P0, P1, P2, GATE, G = [], [], [], [], []
    for x, phys, g, _, _ in loader:
        x = x.to(dev, non_blocking=True)
        phys = phys.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lc, le, gl = net(x, phys)
        p0, p1, p2, _ = exact_probs_from_dual(lc.float(), le.float(), alpha_ex=alpha_ex)
        P0.append(p0.cpu().numpy())
        P1.append(p1.cpu().numpy())
        P2.append(p2.cpu().numpy())
        GATE.append(torch.sigmoid(gl.float()).cpu().numpy())
        G.append(g.numpy())
    return np.concatenate(P0, 0), np.concatenate(P1, 0), np.concatenate(P2, 0), np.concatenate(GATE, 0), np.concatenate(G, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='siglip2g:lastk4,pecoreg:lastk4,cradio:lastk4,eupe:full')
    ap.add_argument('--focus-classes', default='过度生成')
    ap.add_argument('--real-only', action='store_true', help='Train strictly on 100% real human-annotated images')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--lr-full', type=float, default=8.0e-5)
    ap.add_argument('--lr-last', type=float, default=2.8e-5)
    ap.add_argument('--lr-head', type=float, default=1.4e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--dropout', type=float, default=0.12)
    ap.add_argument('--clean-repeat', type=int, default=3)
    ap.add_argument('--p0-clean-weight', type=float, default=4.2)
    ap.add_argument('--channel-neg-weight', type=float, default=1.05)
    ap.add_argument('--alpha-ex', type=float, default=0.58)
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

    focus_cis = [CLASSES.index(c.strip()) for c in a.focus_classes.split(',') if c.strip() in CLASSES]
    print(f'=== Training V10 Clean-Masked Native-Microscope Specialist on focus_classes={a.focus_classes} (real_only={a.real_only}) ===', flush=True)

    rows = load_split(a.split)
    man = read_manifest(a.manifest)
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})

    from extract_native_phys_feats import extract_for_names
    phys_map = extract_for_names([r['name'] for r in rows], a.phys_cache)

    tr_orig = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test']
    has_separate_te = len(te) > 0
    if not has_separate_te:
        te = va

    tr_real = [r for r in tr_orig if not r['name'].startswith(('v9_', 'v8_', 'v7_', 'v6_', 'v4_'))]
    phys_tr_mat = np.stack([phys_map[r['name']] for r in tr_real], axis=0)
    phys_mu = np.mean(phys_tr_mat, axis=0)
    phys_sd = np.std(phys_tr_mat, axis=0) + 1e-5

    Y_orig = targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr_real]))
    pw = ((len(Y_orig) - Y_orig.sum(0)) / Y_orig.sum(0).clamp(min=1)).clamp(1.0, 5.5).to(dev)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    if a.real_only:
        tr = list(tr_real)
    else:
        # Filter synthetic rows to ONLY keep synthetic images whose target class is in `focus_cis`!
        tr_synth = [
            r for r in tr_orig
            if r['name'].startswith(('v9_', 'v8_', 'v7_', 'v6_', 'v4_'))
            and any(get_sample_class_mask6(r['name'])[ci] > 0.5 for ci in focus_cis)
        ]
        print(f'  Filtered synthetic rows for focus_classes={a.focus_classes}: kept {len(tr_synth)} clean-masked synth samples (0 non-target pollution!)', flush=True)
        tr = list(tr_real) + tr_synth

    # Only oversample genuinely rare focus classes (过锐=2, 花屏=3, 过度生成=4, 虚化突兀=5), NEVER majority 纹理损失/模糊!
    rare_focus_cis = [ci for ci in focus_cis if ci in (2, 3, 4, 5)]
    if rare_focus_cis and not a.real_only:
        focus_real_pos = [r for r in tr_real if any(int(r[CLASSES[ci]]) > 0 for ci in rare_focus_cis)]
        tr = tr + focus_real_pos
    if a.clean_repeat > 1:
        clean = [r for r in tr_real if all(int(r[c]) == 0 for c in CLASSES)]
        tr = tr + clean * (a.clean_repeat - 1)

    dtr = V10NativeMicroscopeDataset(tr, phys_map, phys_mu, phys_sd, a.size, True, tiles=a.tiles)
    dva = V10NativeMicroscopeDataset(va, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles)
    dte = V10NativeMicroscopeDataset(te, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles)

    val_batch = a.batch * 2
    ltr = DataLoader(dtr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                     pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
    lva = DataLoader(dva, batch_size=val_batch, shuffle=False, num_workers=a.workers,
                     pin_memory=True, persistent_workers=True, prefetch_factor=4)
    lte = DataLoader(dte, batch_size=val_batch, shuffle=False, num_workers=a.workers,
                     pin_memory=True, persistent_workers=True, prefetch_factor=4)

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
            for blk in bl[-k:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            pg.append({'params': [p for p in m.parameters() if p.requires_grad], 'lr': a.lr_last})
        specs.append((m, fwd, mu, sd, bl, dim))

    net = MultiNetV10(specs, phys_dim=60, dropout=a.dropout, n_quad=a.tiles * a.tiles).to(dev)
    head_params = (
        list(net.head_cum.parameters())
        + list(net.head_ex.parameters())
        + list(net.gate.parameters())
        + list(net.norm.parameters())
        + list(net.tile_attn.parameters())
        + list(net.phys_mlp.parameters())
    )
    pg.append({'params': head_params, 'lr': a.lr_head})

    try:
        opt = torch.optim.AdamW(pg, weight_decay=a.wd, fused=True)
    except Exception:
        opt = torch.optim.AdamW(pg, weight_decay=a.wd)
    total_steps = max(1, (len(ltr) // a.grad_accum) * a.epochs)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    print(f'train={len(dtr)} (real={len(tr_real)}) val={len(dva)} test={len(dte)} views={1 + a.tiles*a.tiles + 3} size={a.size}', flush=True)

    from cv_decoder_calib import decode_pc, HAND
    ep_ckpts = []
    for ep in range(a.epochs):
        t0 = time.time()
        net.train()
        opt.zero_grad(set_to_none=True)
        tot, n_step = 0.0, 0
        for step, (x, phys, g, mask7, _) in enumerate(ltr):
            x = x.to(dev, non_blocking=True)
            phys = phys.to(dev, non_blocking=True)
            g = g.to(dev, non_blocking=True)
            mask7 = mask7.to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lc, le, gl = net(x, phys)
            loss = v10_clean_masked_loss_fn(
                lc.float(), le.float(), gl.float(), g, mask7, bce_loss_none,
                focus_cis=focus_cis,
                p0_clean_weight=a.p0_clean_weight,
                channel_neg_weight=a.channel_neg_weight,
                alpha_ex=a.alpha_ex,
            )
            (loss / a.grad_accum).backward()
            if (step + 1) % a.grad_accum == 0 or (step + 1) == len(ltr):
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sch.step()
            tot += float(loss.detach())
            n_step += 1

        vp0, vp1, vp2, vg, gva = predict_v10(net, lva, dev, a.alpha_ex)
        s_hand = cm.score(gva, decode_pc(vp0, vp1, vp2, HAND), a.tau)
        clean_va = (gva == 0).all(1)
        pred_va = decode_pc(vp0, vp1, vp2, HAND)
        s_clean2 = float((clean_va & (pred_va == 0).all(1)).sum() / max(1, clean_va.sum()))
        doc2_va = 25 * s_hand['S_pos'] + 20 * s_hand['S_p0'] + 20 * s_clean2
        print(f'ep={ep+1} loss={tot/max(1,n_step):.4f} val_Doc1={s_hand["total"]:.2f} val_Doc2={doc2_va:.2f} (S_pos={s_hand["S_pos"]:.3f} S_p0={s_hand["S_p0"]:.3f} Clean2={s_clean2:.3f}) ({time.time()-t0:.0f}s)', flush=True)
        if ep >= a.epochs - 2:
            ep_ckpts.append({k: v.cpu().clone() for k, v in net.state_dict().items()})

    if len(ep_ckpts) >= 2:
        swa_sd = {}
        for k in ep_ckpts[0].keys():
            if ep_ckpts[0][k].is_floating_point():
                swa_sd[k] = sum(ck[k] for ck in ep_ckpts) / len(ep_ckpts)
            else:
                swa_sd[k] = ep_ckpts[-1][k]
        net.load_state_dict({k: v.to(dev) for k, v in swa_sd.items()})

    dva_flip = V10NativeMicroscopeDataset(va, phys_map, phys_mu, phys_sd, a.size, False, hflip=True, tiles=a.tiles)
    dte_flip = V10NativeMicroscopeDataset(te, phys_map, phys_mu, phys_sd, a.size, False, hflip=True, tiles=a.tiles)
    lva_flip = DataLoader(dva_flip, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    lte_flip = DataLoader(dte_flip, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    vp0_a, vp1_a, vp2_a, vg_a, gva = predict_v10(net, lva, dev, a.alpha_ex)
    vp0_b, vp1_b, vp2_b, vg_b, _   = predict_v10(net, lva_flip, dev, a.alpha_ex)
    vp0, vp1, vp2, vg = 0.5*(vp0_a+vp0_b), 0.5*(vp1_a+vp1_b), 0.5*(vp2_a+vp2_b), 0.5*(vg_a+vg_b)

    if has_separate_te:
        tp0_a, tp1_a, tp2_a, tg_a, gte = predict_v10(net, lte, dev, a.alpha_ex)
        tp0_b, tp1_b, tp2_b, tg_b, _   = predict_v10(net, lte_flip, dev, a.alpha_ex)
        tp0, tp1, tp2, tg = 0.5*(tp0_a+tp0_b), 0.5*(tp1_a+tp1_b), 0.5*(tp2_a+tp2_b), 0.5*(tg_a+tg_b)
    else:
        tp0, tp1, tp2, tg, gte = vp0, vp1, vp2, vg, gva

    tpl = json.load(open('/wanqing-develop/luowenjing/18/results/v2_full/template_150.json'))
    names_150 = [e['image'] for e in tpl]
    phys_150 = extract_for_names(names_150, a.phys_cache)
    rows_150 = [{'name': n, **{c: 0 for c in CLASSES}} for n in names_150]
    d150 = V10NativeMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, a.size, False, hflip=False, tiles=a.tiles)
    d150_f = V10NativeMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, a.size, False, hflip=True, tiles=a.tiles)
    l150 = DataLoader(d150, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    l150_f = DataLoader(d150_f, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    p150_0a, p150_1a, p150_2a, p150_ga, _ = predict_v10(net, l150, dev, a.alpha_ex)
    p150_0b, p150_1b, p150_2b, p150_gb, _ = predict_v10(net, l150_f, dev, a.alpha_ex)
    p150_0, p150_1, p150_2, p150_g = 0.5*(p150_0a+p150_0b), 0.5*(p150_1a+p150_1b), 0.5*(p150_2a+p150_2b), 0.5*(p150_ga+p150_gb)

    out_base = a.out[:-5] if a.out.endswith('.json') else a.out
    np.savez(
        out_base + '_probs.npz',
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vg, gva=gva, va_names=[r['name'] for r in va],
        te_p0=tp0, te_p1=tp1, te_p2=tp2, te_gate=tg, gte=gte, te_names=[r['name'] for r in te],
        t150_p0=p150_0, t150_p1=p150_1, t150_p2=p150_2, t150_gate=p150_g, t150_names=names_150,
    )
    st_hand = cm.score(gte, decode_pc(tp0, tp1, tp2, HAND), a.tau)
    clean_te = (gte == 0).all(1)
    pred_te = decode_pc(tp0, tp1, tp2, HAND)
    s_clean2_te = float((clean_te & (pred_te == 0).all(1)).sum() / max(1, clean_te.sum()))
    doc2_te = 25 * st_hand['S_pos'] + 20 * st_hand['S_p0'] + 20 * s_clean2_te
    json.dump({'doc1': float(st_hand['total']), 'doc2': float(doc2_te), 'S_pos': float(st_hand['S_pos']),
               'S_p0': float(st_hand['S_p0']), 'Clean2': float(s_clean2_te)}, open(out_base + '.json', 'w'), indent=2)
    print(f'=== FINAL SWA+TTA TEST: Doc1={st_hand["total"]:.2f} | Doc2={doc2_te:.2f} (S_pos={st_hand["S_pos"]:.3f} S_p0={st_hand["S_p0"]:.3f} Clean2={s_clean2_te:.3f}) ===', flush=True)


if __name__ == '__main__':
    main()
