"""V16 Clean-Masked + 1:1 Native-Pixel Edge/Texture Contrast-Residual Microscope + Sharp P1-Boosted Specialist Trainer
(`train_v16_microscope_specialist.py`).

Four Core Training-Side Upgrades over V9 (`train_v9_specialist.py`) & V10 (`train_v10_clean_native_specialist.py`):
1. Per-Sample Synthetic Target Masking + P1/P0-Only Oversampling (Zero Non-Target Pollution & Zero P2 Inflation):
   - Fixes V9's synthetic multi-label pollution (`get_sample_class_mask6`: synthetic samples ONLY supervise their target
     synthesized defect class with weight 0.75, and `All-Clean Gate` ONLY supervises authentic human-annotated images).
   - Fixes V9/V10's `rare_focus_pos` bug where duplicating all `grade > 0` images duplicated 414 `过锐=P2` images and
     flooded `P2` over `P1`. In V16, real oversampling ONLY repeats `P1` (`grade=2`) and `P0` (`grade=3`) samples,
     and synthetic sampling keeps 100% of `P1/P0` counterfactuals and 60% of `P2` counterfactuals.
2. End-to-End 1:1 Native-Pixel Edge vs. Texture Microscope Contrast Residuals (`MultiNetV16`):
   - Separates the 3 Zero-Resize 1:1 Native-Pixel Microscope Crops (`[p_peak1, p_peak2, p_mid]`) into:
     * `edge_max = max(p_peak1, p_peak2)` (Peak 1:1 Laplacian/Sobel high-frequency edge representation)
     * `tex_view = p_mid` (1:1 Median-gradient subject interior texture representation)
     * `r_edge_glob = edge_max - g_view` (isolates localized 1-2px `过锐` halos, `过度生成` hand/text flaws, `虚化突兀` boundary cutouts)
     * `r_edge_tex  = edge_max - tex_view` (isolates edge overshoot vs interior waxy texture smoothing)
   - Fuses 8 physical-microscope summary vectors per backbone with 60D Native Physical Calipers via Squeeze-and-Excitation gating.
3. Sharp P1-Boosted Multinomial CE + Explicit P1/P2 Logit Margin + Masked Pairwise Ranking (`v16_clean_microscope_loss_fn`):
   - Removes V10's `deqa_sigma=0.44` Gaussian soft-label smearing (which blurred `P1` into the majority `P2` grade).
   - Uses `cw4 = [1.0, 1.20, 1.65, 1.50]` and CDF EMD weights `b_w = [1.0, 1.85, 1.55]` plus an explicit `P1 vs P2`
     direct logit separation margin (`loss_p1_sep`) and sample-validity-masked pairwise ranking loss.
4. Official Competition Metric Alignment:
   - Aligns differentiable `soft_sp0 = 0.60 * r_p0 + 0.40 * f1_p0` with `competition_score` (`code/utils.py`).
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
    CACHE_OK, CLASS_TO_COLS14, build_v9_backbone, set_seed
)
import comp_metrics as cm


def get_sample_class_mask6(name, synth_w=0.75):
    """Return a 7-dim float mask:
    - indices 0..5: class supervision weight for each of the 6 defect classes
    - index 6: 1.0 if authentic human-annotated real image (for All-Clean Gate supervision), 0.0 if synthetic
    """
    if name.startswith('v9_og') or name.startswith('v8_og') or name.startswith('v7_'):
        return np.array([0.0, 0.0, 0.0, 0.0, synth_w, 0.0, 0.0], dtype=np.float32)
    if name.startswith('v9_sh') or name.startswith('v8_sh') or name.startswith('v4_'):
        return np.array([0.0, 0.0, synth_w, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if name.startswith('v9_bk') or name.startswith('v8_bk'):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, synth_w, 0.0], dtype=np.float32)
    if name.startswith('v9_bt') or name.startswith('v8_bt'):
        return np.array([synth_w, synth_w, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


class V16NativeMicroscopeDataset(Dataset):
    """Multi-View Dataset with:
    - 1 Global resized view (`size x size`)
    - `tiles x tiles` quadrant views (`size x size`)
    - 3 Zero-Resize 1:1 Native-Pixel Microscope Crops (`size x size` cut directly at 1:1 pixel scale!):
      * Top-1 Peak Laplacian/Sobel High-Frequency Edge Patch (`p_peak1`)
      * Top-2 Second High-Frequency Edge Patch (`p_peak2`)
      * 1 Median-Gradient Subject Interior Texture Patch (`p_mid`)
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
        extra = [os.environ['NEXT55_EXTRA_IMG_DIR']] if os.environ.get('NEXT55_EXTRA_IMG_DIR') else []
        for d in extra + ['/tmp/v9_authentic_imgs', '/tmp/v8_ladder_imgs', '/tmp/v7_overgen_pro_imgs', '/tmp/v6_synth_imgs',
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

    def __getitem__(self, i):
        r = self.rows[i]
        nm = r['name']
        im = self._load(nm)
        if self.train:
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.25:
                im = ImageEnhance.Brightness(im).enhance(1.0 + random.uniform(-0.035, 0.035))
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


class MultiNetV16(nn.Module):
    """V16 Multi-Backbone + Spatial Anomaly Pooling + 1:1 Native Edge/Texture Contrast-Residual Microscope + 60D Physical Caliper Network."""
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
        # _agg outputs 8 physical-microscope summary vectors per backbone:
        # [g_view, quad_mean, edge_max, tex_view, r_edge_glob, r_edge_tex, t_max, t_att]
        d_vis = sum(8 * x[5] for x in specs)
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
        self.se_gate = nn.Sequential(
            nn.Linear(d, 256),
            nn.GELU(),
            nn.Linear(256, d),
            nn.Sigmoid(),
        )
        self.drop = nn.Dropout(dropout)
        self.head_cum = nn.Sequential(
            nn.Linear(d, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 14),
        )
        self.head_ex = nn.Sequential(
            nn.Linear(d, 512),
            nn.LayerNorm(512),
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
        nat_views = f[:, 1 + self.n_quad:]  # [p_peak1, p_peak2, p_mid]
        if nat_views.shape[1] >= 3:
            edge_max = nat_views[:, :2].amax(1)
            tex_view = nat_views[:, 2]
        else:
            edge_max = nat_views.amax(1)
            tex_view = nat_views.mean(1)
        # Explicit 1:1 Native Microscope Contrast Residuals:
        r_edge_glob = edge_max - g_view   # Isolates localized 1-2px edge ringing/overshoot & local AI hand/text flaws vs global scene
        r_edge_tex  = edge_max - tex_view # Isolates edge sharpness/overshoot vs subject interior texture smoothing
        t_max = tiles.amax(1)
        w = torch.softmax(attn_mod(tiles), dim=1)
        t_att = (tiles * w).sum(1)
        return torch.cat([g_view, quad_mean, edge_max, tex_view, r_edge_glob, r_edge_tex, t_max, t_att], dim=1)

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
        z_norm = self.norm(torch.cat(out + [p_emb], dim=1))
        fused = self.drop(z_norm * (0.5 + 0.5 * self.se_gate(z_norm)))
        return self.head_cum(fused), self.head_ex(fused), self.gate(fused).squeeze(1)


def masked_pairwise_ordinal_margin_loss(p0, p1, p2, g, mask6, focus_cis):
    """Sample-validity-masked pairwise ordinal severity ranking loss (`Clean < P2 < P1 < P0`).
    Fixes V10 bug where non-target synthetic classes were not masked inside pairwise ranking!
    """
    tot_rank = torch.tensor(0.0, device=g.device)
    n_active = 0
    for ci in focus_cis:
        c = CLASSES[ci]
        y = g[:, ci].float()
        m_valid = mask6[:, ci] > 0.1
        if m_valid.sum() < 2:
            continue
        if c in FULL:
            k = FULL.index(c)
            sev = 3.0 * p0[:, ci] + 2.0 * p1[:, k] + 1.0 * p2[:, k]
        else:
            sev = 3.0 * p0[:, ci]

        diff_y = y.unsqueeze(1) - y.unsqueeze(0)
        valid_pairs = m_valid.unsqueeze(1) & m_valid.unsqueeze(0)
        mask = (diff_y > 0) & valid_pairs
        if not mask.any():
            continue
        diff_s = sev.unsqueeze(1) - sev.unsqueeze(0)
        req_margin = 0.60 * diff_y
        viol = F.relu(req_margin - diff_s) * mask.float()

        # Strongly boost positive-grade separation (P1 vs P2, P0 vs P1, P0 vs P2)
        both_pos = (y.unsqueeze(1) >= 1.0) & (y.unsqueeze(0) >= 1.0) & mask
        pair_w = torch.where(both_pos, torch.full_like(viol, 3.0), torch.ones_like(viol))
        tot_rank = tot_rank + (viol * pair_w).sum() / (mask.float().sum() + 1e-5)
        n_active += 1
    return tot_rank / max(1, n_active)


def v16_clean_microscope_loss_fn(logits_cum, logits_ex, gate_logit, g, mask7, bce_loss_none, focus_cis,
                                 p0_clean_weight=4.2, channel_neg_weight=1.05, alpha_ex=0.58):
    """V16 Clean-Masked + Sharp P1-Boosted + Explicit P1/P2 Margin + Official-Metric-Aligned Loss."""
    dev = g.device
    mask6 = mask7[:, :6]        # (B, 6): 1.0 on real, 0.75 on target synthetic class, 0.0 on non-target synthetic classes
    is_real = mask7[:, 6] > 0.5 # (B,): True strictly on authentic human-annotated images

    c_weights6 = torch.full((6,), 0.25, device=dev)
    col_weights14 = torch.full((14,), 0.25, device=dev)
    for ci in focus_cis:
        c_weights6[ci] = 1.65
        for col in CLASS_TO_COLS14[ci]:
            col_weights14[col] = 1.65
    col_weights14 = col_weights14 / col_weights14.mean()
    c_weights6 = c_weights6 / c_weights6.mean()

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
    # Boost >=P1 cumulative threshold channel (middle column of each 3-col graded class) by 1.20x
    for ci in (0, 1, 2, 4):
        col_ge_p1 = CLASS_TO_COLS14[ci][1]
        w_mat[:, col_ge_p1] = w_mat[:, col_ge_p1] * 1.20
    if clean_mask.any() and p0_clean_weight > 1.0:
        for idx in P0_IDX:
            w_mat[clean_mask, idx] = p0_clean_weight

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

    # Sharp P1-Boosted Multinomial CE (`cw4 = [1.0, 1.20, 1.65, 1.50]`, ZERO Gaussian P1-into-P2 smearing!)
    # + P1-Boundary-Weighted CDF EMD (`b_w = [1.0, 1.85, 1.55]`)
    # + Explicit Direct Logit Separation Margin (`loss_p1_sep`) between P1 (tgt=2) and P2 (tgt=1)
    ce_tot = 0.0
    emd_tot = 0.0
    p1_sep_tot = 0.0
    j_ex = 0
    cw4 = torch.tensor([1.0, 1.20, 1.65, 1.50], device=dev)
    b_w = torch.tensor([1.0, 1.85, 1.55], device=dev)
    for ci, c in enumerate(CLASSES):
        w_ci = c_weights6[ci]
        m_ci = mask6[:, ci]
        if c in FULL:
            lg4 = logits_ex[:, j_ex:j_ex + 4]
            tgt = g[:, ci].long()
            if m_ci.sum() > 0:
                ce_hard = F.cross_entropy(lg4, tgt, weight=cw4, reduction='none', label_smoothing=0.015)
                ce_tot = ce_tot + w_ci * ((ce_hard * m_ci).sum() / m_ci.sum())

                probs4 = torch.softmax(lg4, dim=1)
                cdf_pred = torch.cumsum(probs4, dim=1)[:, :3]
                onehot = F.one_hot(tgt, num_classes=4).float()
                cdf_true = torch.cumsum(onehot, dim=1)[:, :3]
                emd_per = (((cdf_pred - cdf_true) ** 2) * b_w).mean(dim=1)
                emd_tot = emd_tot + w_ci * ((emd_per * m_ci).sum() / m_ci.sum())

                # Direct P1 vs P2 & P0 vs P1 logit separation margin
                is_p1 = (tgt == 2).float() * m_ci
                is_p2 = (tgt == 1).float() * m_ci
                is_p0 = (tgt == 3).float() * m_ci
                sep_loss = (
                    is_p1 * (F.relu(0.40 - (lg4[:, 2] - lg4[:, 1])) + 0.5 * F.relu(0.25 - (lg4[:, 2] - lg4[:, 3])))
                    + is_p2 * F.relu(0.25 - (lg4[:, 1] - lg4[:, 2]))
                    + is_p0 * F.relu(0.35 - (lg4[:, 3] - lg4[:, 2]))
                )
                p1_sep_tot = p1_sep_tot + w_ci * (sep_loss.sum() / (m_ci.sum() + 1e-5))
            j_ex += 4
        else:
            tgt_bin = (g[:, ci] == 3).long()
            if m_ci.sum() > 0:
                cw2 = torch.tensor([1.0, 1.45], device=dev)
                ce_bin = F.cross_entropy(logits_ex[:, j_ex:j_ex + 2], tgt_bin, weight=cw2, reduction='none', label_smoothing=0.015)
                ce_tot = ce_tot + w_ci * ((ce_bin * m_ci).sum() / m_ci.sum())
            j_ex += 2

    loss_ce = ce_tot / len(CLASSES)
    loss_emd = emd_tot / len(FULL)
    loss_p1_sep = p1_sep_tot / len(FULL)

    p0, p1, p2, P_exact14 = exact_probs_from_dual(logits_cum, logits_ex, alpha_ex=alpha_ex)
    Y_exact14 = exact_targets_14(g)
    m14_bin = (mask14 > 0.1).float()
    tp = (P_exact14 * Y_exact14 * m14_bin).sum(dim=0)
    fp = (P_exact14 * (1.0 - Y_exact14) * m14_bin).sum(dim=0)
    fn = ((1.0 - P_exact14) * Y_exact14 * m14_bin).sum(dim=0)
    soft_f1 = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
    loss_s_pos = 1.0 - (soft_f1 * col_weights14).sum() / col_weights14.sum()

    # Exact competition_score S_p0 alignment: 0.60 * R_p0 + 0.40 * F1_p0
    Y_p0 = (g == 3).float()
    m6_bin = (mask6 > 0.1).float()
    tp_p0 = (p0 * Y_p0 * m6_bin).sum(dim=0)
    fp_p0 = (p0 * (1.0 - Y_p0) * m6_bin).sum(dim=0)
    fn_p0 = ((1.0 - p0) * Y_p0 * m6_bin).sum(dim=0)
    r_p0 = (tp_p0 + 1e-4) / (tp_p0 + fn_p0 + 1e-4)
    f1_p0 = (2.0 * tp_p0 + 1e-4) / (2.0 * tp_p0 + fp_p0 + fn_p0 + 1e-4)
    soft_sp0 = 0.60 * r_p0 + 0.40 * f1_p0
    loss_s_p0 = 1.0 - (soft_sp0 * c_weights6).sum() / c_weights6.sum()

    loss_rank = masked_pairwise_ordinal_margin_loss(p0, p1, p2, g, mask6, focus_cis)
    loss_clean_p0 = p0[clean_mask].amax(dim=1).mean() if clean_mask.any() else 0.0

    if is_real.any():
        loss_gate = F.binary_cross_entropy_with_logits(gate_logit[is_real], clean_mask[is_real].float())
    else:
        loss_gate = 0.0

    return (
        loss_bce
        + 0.80 * loss_ce
        + 0.70 * loss_emd
        + 0.45 * loss_p1_sep
        + 0.70 * loss_s_pos
        + 0.60 * loss_s_p0
        + 0.65 * loss_rank
        + 0.20 * mono_pen
        + 0.40 * loss_clean_p0
        + 0.35 * loss_gate
    )


@torch.no_grad()
def predict_v16(net, loader, dev, alpha_ex=0.58):
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
    ap.add_argument('--backbones', default='pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,convnext:full')
    ap.add_argument('--focus-classes', default='过度生成')
    ap.add_argument('--real-only', action='store_true', help='Train strictly on 100% real human-annotated images')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--lr-full', type=float, default=8.0e-5)
    ap.add_argument('--lr-last', type=float, default=2.8e-5)
    ap.add_argument('--lr-head', type=float, default=1.3e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.12)
    ap.add_argument('--clean-repeat', type=int, default=2)
    ap.add_argument('--p0-clean-weight', type=float, default=4.2)
    ap.add_argument('--channel-neg-weight', type=float, default=1.05)
    ap.add_argument('--alpha-ex', type=float, default=0.58)
    ap.add_argument('--split', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--phys-cache', default='results/v8_experts/native_phys_feats.npz')
    ap.add_argument('--tau', type=float, default=0.02)
    ap.add_argument('--out', required=True)
    # next55: V16 never saved weights, so it could not score the hidden final set. These two flags fix that.
    ap.add_argument('--save-ckpt', action='store_true', help='save SWA state_dict + phys scaler next to --out')
    ap.add_argument('--extra-template', default='', help='extra test template json (e.g. final 654 set) -> *_probs.npz ext_*')
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    set_seed(a.seed)
    dev = 'cuda'
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    focus_cis = [CLASSES.index(c.strip()) for c in a.focus_classes.split(',') if c.strip() in CLASSES]
    print(f'=== Training V16 Clean-Masked Microscope Contrast Specialist on focus_classes={a.focus_classes} (real_only={a.real_only}) ===', flush=True)

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

    rng = random.Random(a.seed)
    if a.real_only:
        tr = list(tr_real)
        # Even in real_only mode, boost rare P1/P0 real samples (NEVER P2!) so P1/P0 boundaries are sharp!
        p1_p0_real = [
            r for r in tr_real
            if any(int(r[CLASSES[ci]]) >= 2 for ci in (2, 3, 4, 5))
        ]
        tr = tr + p1_p0_real
    else:
        # Filter synthetic rows to ONLY keep synthetic images whose target class is in `focus_cis`,
        # AND keep 100% of P1/P0 synthetic images + 60% of P2 synthetic images so P2 is never inflated over P1!
        tr_synth_all = [
            r for r in tr_orig
            if r['name'].startswith(('v9_', 'v8_', 'v7_', 'v6_', 'v4_'))
            and any(get_sample_class_mask6(r['name'])[ci] > 0.1 for ci in focus_cis)
        ]
        tr_synth = []
        for r in tr_synth_all:
            m6 = get_sample_class_mask6(r['name'])
            active_cis = [ci for ci in range(6) if m6[ci] > 0.1]
            max_g = max(int(r[CLASSES[ci]]) for ci in active_cis) if active_cis else 0
            if max_g >= 2 or rng.random() < 0.60:
                tr_synth.append(r)
        print(f'  Filtered P1/P0-enriched clean-masked synth rows for {a.focus_classes}: kept {len(tr_synth)}/{len(tr_synth_all)} samples', flush=True)
        tr = list(tr_real) + tr_synth

        # Strictly oversample ONLY P1 (grade=2) and P0 (grade=3) real samples in rare focus classes (NEVER P2 grade=1!)
        rare_focus_cis = [ci for ci in focus_cis if ci in (2, 3, 4, 5)]
        if rare_focus_cis:
            focus_real_p1_p0 = [
                r for r in tr_real
                if any(int(r[CLASSES[ci]]) >= 2 for ci in rare_focus_cis)
            ]
            tr = tr + focus_real_p1_p0
            print(f'  Oversampled P1/P0 real samples for rare_focus_cis={rare_focus_cis}: +{len(focus_real_p1_p0)} samples (0 P2 inflation!)', flush=True)

    if a.clean_repeat > 1:
        clean = [r for r in tr_real if all(int(r[c]) == 0 for c in CLASSES)]
        tr = tr + clean * (a.clean_repeat - 1)

    dtr = V16NativeMicroscopeDataset(tr, phys_map, phys_mu, phys_sd, a.size, True, tiles=a.tiles)
    dva = V16NativeMicroscopeDataset(va, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles)
    dte = V16NativeMicroscopeDataset(te, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles)

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

    net = MultiNetV16(specs, phys_dim=60, dropout=a.dropout, n_quad=a.tiles * a.tiles).to(dev)
    head_params = (
        list(net.head_cum.parameters())
        + list(net.head_ex.parameters())
        + list(net.gate.parameters())
        + list(net.norm.parameters())
        + list(net.se_gate.parameters())
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
            loss = v16_clean_microscope_loss_fn(
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

        vp0, vp1, vp2, vg, gva = predict_v16(net, lva, dev, a.alpha_ex)
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

    out_base = a.out[:-5] if a.out.endswith('.json') else a.out
    if a.save_ckpt:
        keep = {n for n, p in net.named_parameters() if p.requires_grad} | {n for n, _ in net.named_buffers()}
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items() if k in keep},
                    'phys_mu': phys_mu, 'phys_sd': phys_sd, 'args': vars(a)}, out_base + '_swa.pt')
        print(f'[v16] saved SWA checkpoint -> {out_base}_swa.pt', flush=True)

    dva_flip = V16NativeMicroscopeDataset(va, phys_map, phys_mu, phys_sd, a.size, False, hflip=True, tiles=a.tiles)
    dte_flip = V16NativeMicroscopeDataset(te, phys_map, phys_mu, phys_sd, a.size, False, hflip=True, tiles=a.tiles)
    lva_flip = DataLoader(dva_flip, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    lte_flip = DataLoader(dte_flip, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    vp0_a, vp1_a, vp2_a, vg_a, gva = predict_v16(net, lva, dev, a.alpha_ex)
    vp0_b, vp1_b, vp2_b, vg_b, _   = predict_v16(net, lva_flip, dev, a.alpha_ex)
    vp0, vp1, vp2, vg = 0.5*(vp0_a+vp0_b), 0.5*(vp1_a+vp1_b), 0.5*(vp2_a+vp2_b), 0.5*(vg_a+vg_b)

    if has_separate_te:
        tp0_a, tp1_a, tp2_a, tg_a, gte = predict_v16(net, lte, dev, a.alpha_ex)
        tp0_b, tp1_b, tp2_b, tg_b, _   = predict_v16(net, lte_flip, dev, a.alpha_ex)
        tp0, tp1, tp2, tg = 0.5*(tp0_a+tp0_b), 0.5*(tp1_a+tp1_b), 0.5*(tp2_a+tp2_b), 0.5*(tg_a+tg_b)
    else:
        tp0, tp1, tp2, tg, gte = vp0, vp1, vp2, vg, gva

    tpl = json.load(open('/wanqing-develop/luowenjing/18/results/v2_full/template_150.json'))
    names_150 = [e['image'] for e in tpl]
    phys_150 = extract_for_names(names_150, a.phys_cache)
    rows_150 = [{'name': n, **{c: 0 for c in CLASSES}} for n in names_150]
    d150 = V16NativeMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, a.size, False, hflip=False, tiles=a.tiles)
    d150_f = V16NativeMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, a.size, False, hflip=True, tiles=a.tiles)
    l150 = DataLoader(d150, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    l150_f = DataLoader(d150_f, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    p150_0a, p150_1a, p150_2a, p150_ga, _ = predict_v16(net, l150, dev, a.alpha_ex)
    p150_0b, p150_1b, p150_2b, p150_gb, _ = predict_v16(net, l150_f, dev, a.alpha_ex)
    p150_0, p150_1, p150_2, p150_g = 0.5*(p150_0a+p150_0b), 0.5*(p150_1a+p150_1b), 0.5*(p150_2a+p150_2b), 0.5*(p150_ga+p150_gb)

    out_base = a.out[:-5] if a.out.endswith('.json') else a.out
    ext = {}
    if a.extra_template:
        names_ext = [e['image'] for e in json.load(open(a.extra_template))]
        phys_ext = extract_for_names(names_ext, a.phys_cache)
        n_zero = sum(1 for n in names_ext if not np.any(phys_ext[n]))
        assert n_zero == 0, f'{n_zero} extra-template images not found (set NEXT55_EXTRA_IMG_DIR)'
        rows_ext = [{'name': n, **{c: 0 for c in CLASSES}} for n in names_ext]
        outs = []
        for hf in (False, True):
            dx = V16NativeMicroscopeDataset(rows_ext, phys_ext, phys_mu, phys_sd, a.size, False, hflip=hf, tiles=a.tiles)
            outs.append(predict_v16(net, DataLoader(dx, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True), dev, a.alpha_ex))
        ext = dict(ext_p0=0.5*(outs[0][0]+outs[1][0]), ext_p1=0.5*(outs[0][1]+outs[1][1]), ext_p2=0.5*(outs[0][2]+outs[1][2]),
                   ext_gate=0.5*(outs[0][3]+outs[1][3]), ext_names=names_ext)
    np.savez(
        out_base + '_probs.npz',
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vg, gva=gva, va_names=[r['name'] for r in va],
        te_p0=tp0, te_p1=tp1, te_p2=tp2, te_gate=tg, gte=gte, te_names=[r['name'] for r in te],
        t150_p0=p150_0, t150_p1=p150_1, t150_p2=p150_2, t150_gate=p150_g, t150_names=names_150,
        **ext,
    )
    st_hand = cm.score(gte, decode_pc(tp0, tp1, tp2, HAND), a.tau)
    clean_te = (gte == 0).all(1)
    pred_te = decode_pc(tp0, tp1, tp2, HAND)
    s_clean2_te = float((clean_te & (pred_te == 0).all(1)).sum() / max(1, clean_te.sum()))
    doc2_te = 25 * st_hand['S_pos'] + 20 * st_hand['S_p0'] + 20 * s_clean2_te
    json.dump({'doc1': float(st_hand['total']), 'doc2': float(doc2_te), 'S_pos': float(st_hand['S_pos']),
               'S_p0': float(st_hand['S_p0']), 'Clean2': float(s_clean2_te)}, open(out_base + '.json', 'w'), indent=2)
    print(f'=== FINAL V16 SWA+TTA TEST: Doc1={st_hand["total"]:.2f} | Doc2={doc2_te:.2f} (S_pos={st_hand["S_pos"]:.3f} S_p0={st_hand["S_p0"]:.3f} Clean2={s_clean2_te:.3f}) ===', flush=True)


if __name__ == '__main__':
    main()
