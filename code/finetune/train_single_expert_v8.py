"""V8 Single-Feature Dedicated Expert Trainer (train_single_expert_v8.py):
1. True Single-Feature Specialization (`--focus-classes`):
   Concentrates 85% of gradient & attention pooling supervision on the dedicated target defect(s)
   (e.g., `过度生成`, `过锐`, `模糊,纹理损失`, or `虚化突兀`), eliminating multi-task gradient dilution.
2. Native 1:1 Zero-Resize Pixel Crops (`native_crops=2`):
   Preserves 100% native 1-pixel USM halos, mild defocus gradients, and fine AI hallucinated textures
   by adding two 1:1 native-resolution crops (center subject crop + highest-gradient patch crop)
   alongside the global + 2x2 quadrant views.
3. 60-Dim Native Physical Caliper Feature Fusion:
   Injects standardized 1:1 multi-scale Laplacian, USM overshoot, patch sharpness bimodality, and
   micro-texture kurtosis features directly into the Dual Ordinal + Multinomial Head.
4. Pairwise Ordinal Severity Margin Ranking Loss (`pairwise_margin_loss`):
   Directly attacks the #1 bottleneck (30%-48% P2 <-> P1 <-> P0 grade confusion among detected defects)
   by enforcing strict margin ordering s(P0) > s(P1) > s(P2) > s(Clean) across all pairs in each batch.
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
from models import build_backbone
from train_v2 import (exact_probs_from_dual, exact_targets_14,
                      decode_scaled_argmax, optimise_scaled, P0_IDX)
import comp_metrics as cm

CACHE_OK = os.path.exists('/dev/shm/race18_native/index.npz')

# Map each class index (0..5) to its columns in the 14-dim cumulative/exact target tensor:
# 0:模糊 -> [0,1,2], 1:纹理损失 -> [3,4,5], 2:过锐 -> [6,7,8], 3:花屏 -> [9], 4:过度生成 -> [10,11,12], 5:虚化突兀 -> [13]
CLASS_TO_COLS14 = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
    3: [9],
    4: [10, 11, 12],
    5: [13],
}


class NativeCaliperDataset(Dataset):
    """Loads Global + 2x2 Quadrant views (5 views) PLUS 2 Native 1:1 Zero-Resize Pixel Crops (total 7 views),
    plus the 60-dim 1:1 Native Physical Caliper feature vector.
    """
    def __init__(self, rows, phys_map, phys_mu, phys_sd, size=384, train=False, tiles=2, hflip=False,
                 cache_bin='/dev/shm/race18_native/cache.bin', index='/dev/shm/race18_native/index.npz'):
        self.rows = rows
        self.phys_map = phys_map
        self.phys_mu = phys_mu
        self.phys_sd = phys_sd
        self.size = size
        self.train = train
        self.tiles = tiles
        self.hflip = hflip
        if CACHE_OK:
            d = np.load(index, allow_pickle=True)
            self.names = list(d['names'])
            self.offsets = d['offsets']
            self.h = d['h']
            self.w = d['w']
            self.pos = {n: i for i, n in enumerate(self.names)}
            self.data = np.memmap(cache_bin, dtype=np.uint8, mode='r')
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
        for d in ['/tmp/v8_ladder_imgs', '/tmp/v7_overgen_pro_imgs', '/tmp/v6_synth_imgs', 'data/train', 'data/images', 'data/images_supp', 'data/test_150', 'data/test']:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
        raise FileNotFoundError(f'Cannot find {name}')

    def _extract_native_crops(self, im):
        """Extract 2 Native 1:1 Zero-Resize Crops (size x size):
        1. Center subject 1:1 crop (or random jittered near-center during training).
        2. Highest-detail / off-center 1:1 crop from the sharpest non-center region.
        """
        w, h = im.size
        S = self.size
        if w <= S or h <= S:
            # Image is smaller than S; resize minimum side to S first
            scale = max(S / max(1, w), S / max(1, h))
            im_pad = im.resize((max(S, int(round(w * scale))), max(S, int(round(h * scale)))), Image.BILINEAR)
            pw, ph = im_pad.size
            cx = (pw - S) // 2
            cy = (ph - S) // 2
            c1 = im_pad.crop((cx, cy, cx + S, cy + S))
            c2 = im_pad.crop((0, 0, S, S))
            return [c1, c2]

        # 1:1 Native Center Crop
        cx = (w - S) // 2
        cy = (h - S) // 2
        if self.train:
            jx = random.randint(-min(cx, S // 4), min(w - S - cx, S // 4))
            jy = random.randint(-min(cy, S // 4), min(h - S - cy, S // 4))
            c1 = im.crop((cx + jx, cy + jy, cx + jx + S, cy + jy + S))
        else:
            c1 = im.crop((cx, cy, cx + S, cy + S))

        # 1:1 Native High-Frequency / Detail Crop from 4 candidate positions
        cand_coords = [
            (w // 6, h // 6),
            (max(0, w - S - w // 6), h // 6),
            (w // 6, max(0, h - S - h // 6)),
            (max(0, w - S - w // 6), max(0, h - S - h // 6)),
        ]
        best_c2 = None
        best_var = -1.0
        for x0, y0 in cand_coords:
            x0 = max(0, min(w - S, x0))
            y0 = max(0, min(h - S, y0))
            patch = im.crop((x0, y0, x0 + S, y0 + S))
            # Fast local gradient energy proxy on 64x64 thumbnail
            sm = np.asarray(patch.resize((64, 64), Image.NEAREST), dtype=np.float32)
            v = float(np.mean(np.abs(np.diff(sm, axis=0))) + np.mean(np.abs(np.diff(sm, axis=1))))
            if v > best_var:
                best_var = v
                best_c2 = patch
        return [c1, best_c2]

    def __getitem__(self, i):
        r = self.rows[i]
        nm = r['name']
        im = self._load(nm)
        if self.train:
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.35:
                im = ImageEnhance.Brightness(im).enhance(1.0 + random.uniform(-0.05, 0.05))
        elif self.hflip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = im.size
        views = [im.resize((self.size, self.size), Image.BILINEAR)]
        T = self.tiles
        for dy in range(T):
            for dx in range(T):
                views.append(im.crop((dx * w // T, dy * h // T, (dx + 1) * w // T, (dy + 1) * h // T)).resize((self.size, self.size), Image.BILINEAR))
        # Append 2 Native 1:1 Zero-Resize Crops!
        views.extend(self._extract_native_crops(im))

        x = np.transpose(np.stack([np.asarray(v, dtype=np.float32) for v in views]) / 255.0, (0, 3, 1, 2))
        g = np.array([int(r[c]) for c in CLASSES], dtype=np.int64)
        p_raw = self.phys_map.get(nm, np.zeros(60, dtype=np.float32))
        p_norm = np.clip((p_raw - self.phys_mu) / self.phys_sd, -6.0, 6.0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(p_norm), torch.from_numpy(g), nm


class MultiNetV8(nn.Module):
    """Multi-Backbone + Native 1:1 Stream + 60-Dim Physical Caliper Fusion Network."""
    def __init__(self, specs, phys_dim=60, dropout=0.1, n_quad=4):
        super().__init__()
        self.n_quad = n_quad
        self.bs = nn.ModuleList()
        self.fwds = []
        self.tile_attn = nn.ModuleList()
        for (m, fwd, mu, sd, bl, dim) in specs:
            self.bs.append(m)
            self.fwds.append(fwd)
            self.register_buffer('mu_%d' % len(self.fwds), torch.tensor(mu).view(1, 3, 1, 1))
            self.register_buffer('sd_%d' % len(self.fwds), torch.tensor(sd).view(1, 3, 1, 1))
            self.tile_attn.append(nn.Sequential(
                nn.Linear(dim, max(64, dim // 8)),
                nn.GELU(),
                nn.Linear(max(64, dim // 8), 1)
            ))
        # Each backbone outputs 5 * dim: [global, quad_mean, native1to1_mean, all_tile_max, tile_attn_pool]
        d_vis = sum(5 * x[5] for x in specs)
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
        self.gate = nn.Linear(d, 1)

    def _agg(self, f, attn_mod):
        # f: (B, V, D) where V = 1 (global) + n_quad (quadrants) + 2 (native 1:1 crops)
        g_view = f[:, 0]
        tiles = f[:, 1:]
        quad_mean = f[:, 1:1 + self.n_quad].mean(1)
        nat_mean = f[:, 1 + self.n_quad:].mean(1)
        t_max = tiles.amax(1)
        w = torch.softmax(attn_mod(tiles), dim=1)
        t_att = (tiles * w).sum(1)
        return torch.cat([g_view, quad_mean, nat_mean, t_max, t_att], dim=1)

    def forward(self, x, phys):
        B, V, C, H, W = x.shape
        out = []
        for i, (m, fwd, attn_mod) in enumerate(zip(self.bs, self.fwds, self.tile_attn)):
            mu = getattr(self, 'mu_%d' % (i + 1))
            sd = getattr(self, 'sd_%d' % (i + 1))
            dt = next(m.parameters()).dtype
            z = ((x.view(B * V, C, H, W) - mu) / sd).to(dt)
            feat = fwd(z).view(B, V, -1).float()
            out.append(self._agg(feat, attn_mod))
        p_emb = self.phys_mlp(phys.float())
        fused = self.drop(self.norm(torch.cat(out + [p_emb], dim=1)))
        return self.head_cum(fused), self.head_ex(fused), self.gate(fused).squeeze(1)


def pairwise_ordinal_margin_loss(p0, p1, p2, g, focus_cis):
    """Pairwise Severity Margin Ranking Loss on the continuous expected severity score
    S_c = p2_cum + p1_cum + p0 = (p0+p1+p2) + (p0+p1) + p0 = 3*p0 + 2*p1 + p2 in [0, 3].
    For every pair (i, j) in the batch with g[i, c] > g[j, c], enforce:
        S_c[i] - S_c[j] >= margin * (g[i, c] - g[j, c]),
    with 2.5x extra weight when both g[i, c] >= 1 and g[j, c] >= 1 (P2 <-> P1 <-> P0 boundary!).
    """
    tot_rank = torch.tensor(0.0, device=g.device)
    n_active = 0
    for ci in focus_cis:
        c = CLASSES[ci]
        y = g[:, ci].float()  # (B,) in {0, 1, 2, 3}
        if c in FULL:
            k = FULL.index(c)
            sev = 3.0 * p0[:, ci] + 2.0 * p1[:, k] + 1.0 * p2[:, k]  # (B,) in [0, 3]
        else:
            sev = 3.0 * p0[:, ci]

        diff_y = y.unsqueeze(1) - y.unsqueeze(0)  # (B, B)
        mask = diff_y > 0
        if not mask.any():
            continue
        diff_s = sev.unsqueeze(1) - sev.unsqueeze(0)
        # Required margin: 0.55 per grade step
        req_margin = 0.55 * diff_y
        viol = F.relu(req_margin - diff_s) * mask.float()

        # Boost weight by 2.5x for positive-vs-positive grade pairs (P2 vs P1, P1 vs P0, P2 vs P0)!
        both_pos = (y.unsqueeze(1) >= 1.0) & (y.unsqueeze(0) >= 1.0) & mask
        pair_w = torch.where(both_pos, torch.full_like(viol, 2.5), torch.ones_like(viol))
        tot_rank = tot_rank + (viol * pair_w).sum() / (mask.float().sum() + 1e-5)
        n_active += 1
    return tot_rank / max(1, n_active)


def v8_expert_loss_fn(logits_cum, logits_ex, gate_logit, g, bce_loss_none, focus_cis,
                      p0_clean_weight=4.2, channel_neg_weight=1.35, alpha_ex=0.55):
    # Build per-class and per-channel focus weights (85% focused on target class(es), 15% context)
    c_weights6 = torch.full((6,), 0.15, device=g.device)
    col_weights14 = torch.full((14,), 0.15, device=g.device)
    for ci in focus_cis:
        c_weights6[ci] = 1.50
        for col in CLASS_TO_COLS14[ci]:
            col_weights14[col] = 1.50
    col_weights14 = col_weights14 / col_weights14.mean()
    c_weights6 = c_weights6 / c_weights6.mean()

    Y_cum = targets_from_grades(g)
    raw_bce = bce_loss_none(logits_cum, Y_cum)
    clean_mask = (g == 0).all(dim=1)

    w_mat = torch.ones_like(raw_bce)
    if channel_neg_weight != 1.0:
        w_mat = torch.where(Y_cum == 0, torch.full_like(w_mat, channel_neg_weight), w_mat)
    if clean_mask.any() and p0_clean_weight > 1.0:
        for idx in P0_IDX:
            w_mat[clean_mask, idx] = p0_clean_weight
    loss_bce = (raw_bce * w_mat * col_weights14.unsqueeze(0)).mean()

    # Monotonicity penalty
    mono_pen = 0.0
    j = 0
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            l2, l1, l0 = logits_cum[:, j], logits_cum[:, j + 1], logits_cum[:, j + 2]
            j += 3
            mono_pen = mono_pen + c_weights6[ci] * (F.relu(l0 - l1).mean() + F.relu(l1 - l2).mean())
        else:
            j += 1

    # 4-way / 2-way Cross-Entropy + Ordinal EMD with extra weight on P2 (1.20) & P1 (1.30) to prevent grade squeezing!
    ce_tot = 0.0
    emd_tot = 0.0
    j_ex = 0
    cw4 = torch.tensor([1.0, 1.20, 1.30, 1.15], device=logits_ex.device)
    for ci, c in enumerate(CLASSES):
        w_ci = c_weights6[ci]
        if c in FULL:
            lg4 = logits_ex[:, j_ex:j_ex + 4]
            tgt = g[:, ci].long()
            ce_tot = ce_tot + w_ci * F.cross_entropy(lg4, tgt, weight=cw4, label_smoothing=0.02)
            probs4 = torch.softmax(lg4, dim=1)
            cdf_pred = torch.cumsum(probs4, dim=1)[:, :3]
            onehot = F.one_hot(tgt, num_classes=4).float()
            cdf_true = torch.cumsum(onehot, dim=1)[:, :3]
            # Higher weight on the P2-vs-P1 (cdf[:, 1]) boundary!
            b_w = torch.tensor([1.0, 1.5, 1.2], device=logits_ex.device)
            emd_tot = emd_tot + w_ci * (((cdf_pred - cdf_true) ** 2) * b_w).mean()
            j_ex += 4
        else:
            tgt_bin = (g[:, ci] == 3).long()
            ce_tot = ce_tot + w_ci * F.cross_entropy(logits_ex[:, j_ex:j_ex + 2], tgt_bin, label_smoothing=0.02)
            j_ex += 2
    loss_ce = ce_tot / len(CLASSES)
    loss_emd = emd_tot / len(FULL)

    # Differentiable Soft Macro-F1 weighted heavily toward focus_cis
    p0, p1, p2, P_exact14 = exact_probs_from_dual(logits_cum, logits_ex, alpha_ex=alpha_ex)
    Y_exact14 = exact_targets_14(g)
    tp = (P_exact14 * Y_exact14).sum(dim=0)
    fp = (P_exact14 * (1.0 - Y_exact14)).sum(dim=0)
    fn = ((1.0 - P_exact14) * Y_exact14).sum(dim=0)
    soft_f1 = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
    loss_s_pos = 1.0 - (soft_f1 * col_weights14).sum() / col_weights14.sum()

    # Pairwise Ordinal Severity Margin Ranking Loss on focus_cis
    loss_rank = pairwise_ordinal_margin_loss(p0, p1, p2, g, focus_cis)

    loss_clean_p0 = p0[clean_mask].amax(dim=1).mean() if clean_mask.any() else 0.0
    loss_gate = F.binary_cross_entropy_with_logits(gate_logit, clean_mask.float())

    return (
        loss_bce
        + 0.75 * loss_ce
        + 0.65 * loss_emd
        + 0.65 * loss_s_pos
        + 0.70 * loss_rank
        + 0.20 * mono_pen
        + 0.40 * loss_clean_p0
        + 0.10 * loss_gate
    )


@torch.no_grad()
def predict_v8(net, loader, dev, alpha_ex=0.55):
    net.eval()
    P0, P1, P2, GATE, G = [], [], [], [], []
    for x, phys, g, _ in loader:
        x = x.to(dev)
        phys = phys.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lc, le, gl = net(x, phys)
        p0, p1, p2, _ = exact_probs_from_dual(lc.float(), le.float(), alpha_ex=alpha_ex)
        P0.append(p0.cpu().numpy())
        P1.append(p1.cpu().numpy())
        P2.append(p2.cpu().numpy())
        GATE.append(torch.sigmoid(gl.float()).cpu().numpy())
        G.append(g.numpy())
    return np.concatenate(P0, 0), np.concatenate(P1, 0), np.concatenate(P2, 0), np.concatenate(GATE, 0), np.concatenate(G, 0)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='siglip2:lastk4,cradio:lastk4,pecorel:lastk4')
    ap.add_argument('--focus-classes', default='过度生成', help='Comma-separated class names to specialize on')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--lr-full', type=float, default=8.5e-5)
    ap.add_argument('--lr-last', type=float, default=3.2e-5)
    ap.add_argument('--lr-head', type=float, default=1.5e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--clean-repeat', type=int, default=3)
    ap.add_argument('--p0-clean-weight', type=float, default=4.2)
    ap.add_argument('--channel-neg-weight', type=float, default=1.20)
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

    focus_cis = [CLASSES.index(c.strip()) for c in a.focus_classes.split(',') if c.strip() in CLASSES]
    print(f'=== Training V8 Dedicated Expert on focus_classes={a.focus_classes} (indices={focus_cis}) ===', flush=True)

    rows = load_split(a.split)
    man = read_manifest(a.manifest)
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})

    # Ensure 60-dim native physical features are extracted for all rows in split
    from extract_native_phys_feats import extract_for_names
    phys_map = extract_for_names([r['name'] for r in rows], a.phys_cache)

    tr_orig = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test']

    # Compute mean/std of physical features strictly on real training rows
    tr_real = [r for r in tr_orig if not r['name'].startswith('v8_') and not r['name'].startswith('v6_') and not r['name'].startswith('v7_')]
    phys_tr_mat = np.stack([phys_map[r['name']] for r in tr_real], axis=0)
    phys_mu = np.mean(phys_tr_mat, axis=0)
    phys_sd = np.std(phys_tr_mat, axis=0) + 1e-5

    # Compute true pos_weight strictly on real training rows
    Y_orig = targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr_real]))
    pw = ((len(Y_orig) - Y_orig.sum(0)) / Y_orig.sum(0).clamp(min=1)).clamp(1.0, 5.5).to(dev)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    # Oversample positive real training examples of focus_cis by 2x so the batch is rich in real P2/P1/P0 pairs!
    tr = list(tr_orig)
    focus_real_pos = [r for r in tr_real if any(int(r[CLASSES[ci]]) > 0 for ci in focus_cis)]
    tr = tr + focus_real_pos
    if a.clean_repeat > 1:
        clean = [r for r in tr_real if all(int(r[c]) == 0 for c in CLASSES)]
        tr = tr + clean * (a.clean_repeat - 1)

    dtr = NativeCaliperDataset(tr, phys_map, phys_mu, phys_sd, a.size, True, tiles=a.tiles)
    dva = NativeCaliperDataset(va, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles)
    dte = NativeCaliperDataset(te, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles)

    ltr = DataLoader(dtr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                     pin_memory=True, drop_last=True, worker_init_fn=lambda w: set_seed(a.seed + w))
    lva = DataLoader(dva, batch_size=a.batch, shuffle=False, num_workers=a.workers)
    lte = DataLoader(dte, batch_size=a.batch, shuffle=False, num_workers=a.workers)

    specs = []
    for s in a.backbones.split(','):
        name, mode = s.split(':')
        b = build_backbone(name)
        specs.append((b, name, mode))

    net = MultiNetV8([x[0] for x in specs], phys_dim=60, dropout=a.dropout, n_quad=a.tiles * a.tiles).to(dev)
    groups = []
    for (b, name, mode) in specs:
        m = b[0]
        blocks = b[4]
        for p in m.parameters():
            p.requires_grad = (mode == 'full')
        if mode.startswith('lastk') and blocks is not None:
            k = int(mode[len('lastk'):] or 4)
            for bl in blocks[-k:]:
                for p in bl.parameters():
                    p.requires_grad = True
        lr = a.lr_full if mode == 'full' else a.lr_last
        groups.append({'params': [p for p in m.parameters() if p.requires_grad], 'lr': lr})

    head_params = (
        list(net.tile_attn.parameters()) +
        list(net.phys_mlp.parameters()) +
        list(net.norm.parameters()) +
        list(net.head_cum.parameters()) +
        list(net.head_ex.parameters()) +
        list(net.gate.parameters())
    )
    groups.append({'params': head_params, 'lr': a.lr_head})
    tr_p = [p for g in groups for p in g['params']]
    print(f'trainable {sum(p.numel() for p in tr_p)/1e6:.1f}M / {sum(p.numel() for p in net.parameters())/1e6:.1f}M '
          f'(size={a.size}, views={1 + a.tiles*a.tiles + 2} [incl. 2 Native 1:1 Crops + 60d Phys Caliper])', flush=True)

    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    accum = max(1, a.grad_accum)
    steps_per_ep = max(1, len(ltr) // accum)
    total_steps = steps_per_ep * a.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=total_steps, pct_start=0.1)

    best = None
    ep3_state = None
    for ep in range(a.epochs):
        net.train()
        t0 = time.time()
        tot = 0.0
        nb = 0
        opt.zero_grad()
        for step_idx, (x, phys, g, _) in enumerate(ltr):
            x = x.to(dev)
            phys = phys.to(dev)
            g = g.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lc, le, gl = net(x, phys)
            loss = v8_expert_loss_fn(lc.float(), le.float(), gl.float(), g, bce_loss_none, focus_cis,
                                     p0_clean_weight=a.p0_clean_weight,
                                     channel_neg_weight=a.channel_neg_weight,
                                     alpha_ex=a.alpha_ex)
            (loss / accum).backward()
            tot += loss.item()
            nb += 1
            if (step_idx + 1) % accum == 0 or (step_idx + 1) == len(ltr):
                torch.nn.utils.clip_grad_norm_(tr_p, 1.0)
                opt.step()
                if sch.last_epoch < total_steps - 1:
                    sch.step()
                opt.zero_grad()

        vp0, vp1, vp2, vgt, gv = predict_v8(net, lva, dev, alpha_ex=a.alpha_ex)
        b_cas, th_cas = cm.optimise(gv, vp0, vp1, vp2, a.tau)
        b_sca, th_sca = optimise_scaled(gv, vp0, vp1, vp2, a.tau)
        # Compute mean F1 specifically on the focused classes on val!
        f_per = [x[4] for x in b_sca['per'] if CLASSES.index(x[0]) in focus_cis]
        focus_val_f1 = float(np.mean(f_per)) if f_per else b_sca['S_pos']
        print(f'ep{ep+1} loss={tot/max(1,nb):.3f} focus_val_F1={focus_val_f1:.3f} val_sca={b_sca["total"]:.2f} '
              f'S_pos={b_sca["S_pos"]:.3f} P0R={b_sca["p0R"]:.3f} P0P={b_sca["p0P"]:.3f} S_clean={b_sca["S_clean"]:.3f} t={time.time()-t0:.0f}s', flush=True)

        cur_state = copy.deepcopy(net.state_dict())
        if (ep + 1) == 3:
            ep3_state = cur_state
        if best is None or focus_val_f1 > best[0]:
            best = (focus_val_f1, th_cas, th_sca, cur_state)

    # SWA average of ep3 and best epoch for rock-solid generalization
    if ep3_state is not None:
        swa_state = {}
        for k in best[3].keys():
            if best[3][k].is_floating_point():
                swa_state[k] = 0.45 * ep3_state[k] + 0.55 * best[3][k]
            else:
                swa_state[k] = best[3][k]
        net.load_state_dict(swa_state)
    else:
        net.load_state_dict(best[3])

    vp0, vp1, vp2, vgt, gva_n = predict_v8(net, lva, dev, alpha_ex=a.alpha_ex)
    tp0, tp1, tp2, tgt, gte_n = predict_v8(net, lte, dev, alpha_ex=a.alpha_ex)
    # 2-way HFlip TTA on val & test
    lva_f = DataLoader(NativeCaliperDataset(va, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles, hflip=True),
                       batch_size=a.batch, shuffle=False, num_workers=a.workers)
    lte_f = DataLoader(NativeCaliperDataset(te, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles, hflip=True),
                       batch_size=a.batch, shuffle=False, num_workers=a.workers)
    vp0_f, vp1_f, vp2_f, vgt_f, _ = predict_v8(net, lva_f, dev, alpha_ex=a.alpha_ex)
    tp0_f, tp1_f, tp2_f, tgt_f, _ = predict_v8(net, lte_f, dev, alpha_ex=a.alpha_ex)
    vp0, vp1, vp2, vgt = 0.5 * (vp0 + vp0_f), 0.5 * (vp1 + vp1_f), 0.5 * (vp2 + vp2_f), 0.5 * (vgt + vgt_f)
    tp0, tp1, tp2, tgt = 0.5 * (tp0 + tp0_f), 0.5 * (tp1 + tp1_f), 0.5 * (tp2 + tp2_f), 0.5 * (tgt + tgt_f)

    # Also predict 2-way HFlip TTA on the 150 blind evaluation images (data/test_150)
    tpl_150 = json.load(open('results/v2_full/template_150.json'))
    rows_150 = [{'name': e['image'], **{c: 0 for c in CLASSES}} for e in tpl_150]
    phys_map = extract_for_names([r['name'] for r in rows_150], a.phys_cache)
    l150_a = DataLoader(NativeCaliperDataset(rows_150, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles, hflip=False),
                        batch_size=a.batch, shuffle=False, num_workers=a.workers)
    l150_b = DataLoader(NativeCaliperDataset(rows_150, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles, hflip=True),
                        batch_size=a.batch, shuffle=False, num_workers=a.workers)
    p150_0a, p150_1a, p150_2a, g150_a, _ = predict_v8(net, l150_a, dev, alpha_ex=a.alpha_ex)
    p150_0b, p150_1b, p150_2b, g150_b, _ = predict_v8(net, l150_b, dev, alpha_ex=a.alpha_ex)
    t150_p0 = 0.5 * (p150_0a + p150_0b)
    t150_p1 = 0.5 * (p150_1a + p150_1b)
    t150_p2 = 0.5 * (p150_2a + p150_2b)
    t150_gate = 0.5 * (g150_a + g150_b)

    st_cas = cm.score(gte_n, cm.decode(tp0, tp1, tp2, *best[1]), a.tau)
    st_sca = cm.score(gte_n, decode_scaled_argmax(tp0, tp1, tp2, *best[2]), a.tau)

    res = {
        'focus_classes': a.focus_classes,
        'backbones': a.backbones,
        'seed': a.seed,
        'size': a.size,
        'tiles': a.tiles,
        'val_focus_f1': best[0],
        'test': {k: st_sca[k] for k in ['total', 'S_pos', 'p0R', 'p0P', 'p0F', 'S_p0', 'S_clean', 'fpr']},
        'test_cascade': {k: st_cas[k] for k in ['total', 'S_pos', 'p0R', 'p0P', 'p0F', 'S_p0', 'S_clean', 'fpr']},
        'per_label': {f'{x[0]}_{x[1]}': {'P': x[2], 'R': x[3], 'F1': x[4]} for x in st_sca['per']},
    }
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
    np.savez_compressed(
        a.out.replace('.json', '_probs.npz'),
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vgt, gva=gva_n, va_names=np.array([r['name'] for r in va]),
        te_p0=tp0, te_p1=tp1, te_p2=tp2, te_gate=tgt, gte=gte_n, te_names=np.array([r['name'] for r in te]),
        t150_p0=t150_p0, t150_p1=t150_p1, t150_p2=t150_p2, t150_gate=t150_gate, t150_names=np.array([r['name'] for r in rows_150])
    )
    print('TEST_SCALED', json.dumps(res['test'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
