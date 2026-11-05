"""V9 High-Resolution Spatial-Anomaly-Pooled Specialist Trainer (train_v9_specialist.py).

Key architectural & supervision upgrades over V8:
1. Spatial Anomaly Patch Token Pooling (`build_v9_backbone`):
   Instead of collapsing all 576~1024 spatial patch tokens with uniform `.mean(1)` (which dilutes
   localized 5%~15% defects like AI distorted hands/text in `过度生成`, boundary cutout halos in `虚化突兀`,
   and edge halos in `过锐`), extracts:
   - CLS / Summary token
   - Spatial Patch Mean (`patches.mean(1)`)
   - Spatial Patch Std (`patches.std(1)`) -> captures intra-view foreground-vs-background variance!
   - Spatial Top-K Anomaly Patch Mean (`patches.topk(k=32, dim=1).values.mean(1)`) -> preserves peak local defects!
2. New Diverse Backbones:
   Adds `eva02` (`eva02_base_patch14_448.mim_in22k_ft_in1k`) alongside `pecoreg`, `siglip2g`, `dinov3l`, `cradio`, `convnext`, `eupe`, `pecorel`, `siglip2`.
3. P0-Protected Competition-Aligned Loss (`v9_specialist_loss_fn`):
   - `cw4 = [1.0, 1.25, 1.35, 1.45]` (boosts `P0` class weight to 1.45 instead of 1.15!)
   - Adds explicit differentiable Soft `S_p0` (`0.5 * R_p0 + 0.3 * F1_p0 + 0.2 * P_p0`) loss term alongside Soft `S_pos` and Pairwise Severity Margin Ranking Loss!
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
from models import _enable_pos_interp
from train_v2 import (exact_probs_from_dual, exact_targets_14,
                      decode_scaled_argmax, optimise_scaled, P0_IDX)
import comp_metrics as cm

CACHE_OK = os.path.exists('/dev/shm/race18_native/index.npz')

CLASS_TO_COLS14 = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
    3: [9],
    4: [10, 11, 12],
    5: [13],
}


def pool_spatial_tokens(patches, k=32):
    """Given patch tokens (N, L, D), return [mean, std, topk_mean] of shape (N, 3*D)."""
    kk = min(k, patches.shape[1])
    p_mean = patches.mean(dim=1)
    p_std = patches.std(dim=1)
    p_topk = patches.topk(kk, dim=1).values.mean(dim=1)
    return torch.cat([p_mean, p_std, p_topk], dim=1)


def build_v9_backbone(name):
    """Returns (module, fwd, mean, std, block_list, out_dim) with Spatial Anomaly Patch Token Pooling."""
    if name == 'siglip2':
        from transformers import SiglipVisionModel, AutoImageProcessor
        repo = 'google/siglip2-so400m-patch14-384'
        try:
            m = SiglipVisionModel.from_pretrained(repo, attn_implementation='sdpa', local_files_only=True)
            proc = AutoImageProcessor.from_pretrained(repo, local_files_only=True)
        except Exception:
            m = SiglipVisionModel.from_pretrained(repo, attn_implementation='sdpa')
            proc = AutoImageProcessor.from_pretrained(repo)
        _enable_pos_interp(m)
        def fwd(x):
            t = m(pixel_values=x).last_hidden_state
            return pool_spatial_tokens(t, k=32)
        return m, fwd, proc.image_mean, proc.image_std, m.encoder.layers, 1152 * 3

    if name == 'siglip2g':
        from transformers import SiglipVisionModel, AutoImageProcessor
        repo = 'models/siglip2-giant'
        m = SiglipVisionModel.from_pretrained(repo, attn_implementation='sdpa')
        proc = AutoImageProcessor.from_pretrained(repo)
        _enable_pos_interp(m)
        def fwd(x):
            t = m(pixel_values=x).last_hidden_state
            return pool_spatial_tokens(t, k=32)
        return m, fwd, proc.image_mean, proc.image_std, m.encoder.layers, 1536 * 3

    if name == 'eupe':
        m = torch.hub.load('/tmp/race18/eupe_repo', 'eupe_vitb16', source='local',
                           weights='models/eupe-vit-b/EUPE-ViT-B.pt')
        def fwd(x):
            o = m.forward_features(x)
            return torch.cat([o['x_norm_clstoken'], pool_spatial_tokens(o['x_norm_patchtokens'], k=32)], dim=1)
        return m, fwd, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), m.blocks, 768 * 4

    if name == 'convnext':
        import timm
        m = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0)
        dc = timm.data.resolve_data_config({}, model=m)
        def fwd(x):
            y = m.forward_features(x)  # (N, C, H, W)
            if y.ndim == 4:
                N, C, H, W = y.shape
                tokens = y.flatten(2).transpose(1, 2)  # (N, H*W, C)
                return pool_spatial_tokens(tokens, k=16)
            return y
        return m, fwd, dc['mean'], dc['std'], m.stages, 1024 * 3

    if name == 'eva02':
        import timm
        m = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=True, num_classes=0, dynamic_img_size=True)
        dc = timm.data.resolve_data_config({}, model=m)
        def fwd(x):
            t = m.forward_features(x)
            p = m.num_prefix_tokens
            return torch.cat([t[:, 0], pool_spatial_tokens(t[:, p:], k=32)], dim=1)
        return m, fwd, dc['mean'], dc['std'], m.blocks, m.embed_dim * 4

    if name == 'cradio':
        from transformers import AutoModel
        m = AutoModel.from_pretrained('models/c-radio-v4-h', trust_remote_code=True)
        def fwd(x):
            summ, feat = m(x)
            return torch.cat([summ, pool_spatial_tokens(feat, k=32)], dim=1)
        return m, fwd, (0, 0, 0), (1, 1, 1), m.radio_model.model.blocks, 2560 + 1280 * 3

    if name in ('pecorel', 'pecoreg'):
        import timm
        from safetensors.torch import load_file
        arch, path, size = {
            'pecorel': ('vit_pe_core_large_patch14_336', 'models/pe-core-l14/model.safetensors', 448),
            'pecoreg': ('vit_pe_core_gigantic_patch14_448', 'models/pe-core-g14/model.safetensors', 448)
        }[name]
        m = timm.create_model(arch, pretrained=False, num_classes=0, dynamic_img_size=True)
        m.load_state_dict(load_file(path), strict=False)
        dc = timm.data.resolve_data_config({}, model=m)
        def fwd(x):
            if x.shape[-1] != size:
                x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False)
            t = m.forward_features(x)
            p = m.num_prefix_tokens
            if p:
                return torch.cat([t[:, 0], pool_spatial_tokens(t[:, p:], k=32)], dim=1)
            return pool_spatial_tokens(t, k=32)
        d = (4 if m.num_prefix_tokens else 3) * m.embed_dim
        return m, fwd, dc['mean'], dc['std'], m.blocks, d

    if name == 'dinov3l':
        import timm
        from safetensors.torch import load_file
        m = timm.create_model('vit_large_patch16_dinov3', pretrained=False, num_classes=0, dynamic_img_size=True)
        m.load_state_dict(load_file('models/dinov3-vitl16/model.safetensors'), strict=True)
        dc = timm.data.resolve_data_config({}, model=m)
        def fwd(x):
            t = m.forward_features(x)
            p = m.num_prefix_tokens
            return torch.cat([t[:, 0], pool_spatial_tokens(t[:, p:], k=32)], dim=1)
        return m, fwd, dc['mean'], dc['std'], m.blocks, 4 * m.embed_dim

    raise ValueError(f'Unknown backbone: {name}')


class V9CaliperDataset(Dataset):
    """Loads Global + T x T Tiles + 2 Native 1:1 Zero-Resize Crops, plus 60-dim Physical Caliper features."""
    def __init__(self, rows, phys_map, phys_mu, phys_sd, size=448, train=False, tiles=2, hflip=False,
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
        for d in ['/tmp/v9_authentic_imgs', '/tmp/v8_ladder_imgs', '/tmp/v7_overgen_pro_imgs', '/tmp/v6_synth_imgs',
                  'data/train', 'data/images', 'data/images_supp', 'data/test_150', 'data/test']:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
        raise FileNotFoundError(f'Cannot find {name}')

    def _extract_native_crops(self, im):
        w, h = im.size
        S = self.size
        if w <= S or h <= S:
            scale = max(S / max(1, w), S / max(1, h))
            im_pad = im.resize((max(S, int(round(w * scale))), max(S, int(round(h * scale)))), Image.BILINEAR)
            pw, ph = im_pad.size
            cx = (pw - S) // 2
            cy = (ph - S) // 2
            return [im_pad.crop((cx, cy, cx + S, cy + S)), im_pad.crop((0, 0, S, S))]

        cx = (w - S) // 2
        cy = (h - S) // 2
        if self.train:
            jx = random.randint(-min(cx, S // 4), min(w - S - cx, S // 4))
            jy = random.randint(-min(cy, S // 4), min(h - S - cy, S // 4))
            c1 = im.crop((cx + jx, cy + jy, cx + jx + S, cy + jy + S))
        else:
            c1 = im.crop((cx, cy, cx + S, cy + S))

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
        views.extend(self._extract_native_crops(im))

        x = np.transpose(np.stack([np.asarray(v, dtype=np.uint8) for v in views]), (0, 3, 1, 2))
        g = np.array([int(r[c]) for c in CLASSES], dtype=np.int64)
        p_raw = self.phys_map.get(nm, np.zeros(60, dtype=np.float32))
        p_norm = np.clip((p_raw - self.phys_mu) / self.phys_sd, -6.0, 6.0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(p_norm), torch.from_numpy(g), nm


class MultiNetV9(nn.Module):
    """V9 Multi-Backbone + Spatial Anomaly Token Pooling + Native 1:1 Stream + 60-Dim Physical Caliper Network."""
    def __init__(self, specs, phys_dim=60, dropout=0.1, n_quad=4):
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


def pairwise_ordinal_margin_loss(p0, p1, p2, g, focus_cis):
    tot_rank = torch.tensor(0.0, device=g.device)
    n_active = 0
    for ci in focus_cis:
        c = CLASSES[ci]
        y = g[:, ci].float()
        if c in FULL:
            k = FULL.index(c)
            sev = 3.0 * p0[:, ci] + 2.0 * p1[:, k] + 1.0 * p2[:, k]
        else:
            sev = 3.0 * p0[:, ci]

        diff_y = y.unsqueeze(1) - y.unsqueeze(0)
        mask = diff_y > 0
        if not mask.any():
            continue
        diff_s = sev.unsqueeze(1) - sev.unsqueeze(0)
        req_margin = 0.58 * diff_y
        viol = F.relu(req_margin - diff_s) * mask.float()

        both_pos = (y.unsqueeze(1) >= 1.0) & (y.unsqueeze(0) >= 1.0) & mask
        pair_w = torch.where(both_pos, torch.full_like(viol, 2.6), torch.ones_like(viol))
        tot_rank = tot_rank + (viol * pair_w).sum() / (mask.float().sum() + 1e-5)
        n_active += 1
    return tot_rank / max(1, n_active)


def v9_specialist_loss_fn(logits_cum, logits_ex, gate_logit, g, bce_loss_none, focus_cis,
                          p0_clean_weight=4.2, channel_neg_weight=1.10, alpha_ex=0.58):
    c_weights6 = torch.full((6,), 0.20, device=g.device)
    col_weights14 = torch.full((14,), 0.20, device=g.device)
    for ci in focus_cis:
        c_weights6[ci] = 1.60
        for col in CLASS_TO_COLS14[ci]:
            col_weights14[col] = 1.60
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

    mono_pen = 0.0
    j = 0
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            l2, l1, l0 = logits_cum[:, j], logits_cum[:, j + 1], logits_cum[:, j + 2]
            j += 3
            mono_pen = mono_pen + c_weights6[ci] * (F.relu(l0 - l1).mean() + F.relu(l1 - l2).mean())
        else:
            j += 1

    # Upgrade cw4 to strongly protect P0 (1.45) as well as P1 (1.35) and P2 (1.25)!
    ce_tot = 0.0
    emd_tot = 0.0
    j_ex = 0
    cw4 = torch.tensor([1.0, 1.25, 1.35, 1.45], device=logits_ex.device)
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
            b_w = torch.tensor([1.0, 1.5, 1.5], device=logits_ex.device)
            emd_tot = emd_tot + w_ci * (((cdf_pred - cdf_true) ** 2) * b_w).mean()
            j_ex += 4
        else:
            tgt_bin = (g[:, ci] == 3).long()
            ce_tot = ce_tot + w_ci * F.cross_entropy(logits_ex[:, j_ex:j_ex + 2], tgt_bin, label_smoothing=0.02)
            j_ex += 2
    loss_ce = ce_tot / len(CLASSES)
    loss_emd = emd_tot / len(FULL)

    p0, p1, p2, P_exact14 = exact_probs_from_dual(logits_cum, logits_ex, alpha_ex=alpha_ex)
    Y_exact14 = exact_targets_14(g)
    tp = (P_exact14 * Y_exact14).sum(dim=0)
    fp = (P_exact14 * (1.0 - Y_exact14)).sum(dim=0)
    fn = ((1.0 - P_exact14) * Y_exact14).sum(dim=0)
    soft_f1 = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
    loss_s_pos = 1.0 - (soft_f1 * col_weights14).sum() / col_weights14.sum()

    # Differentiable Soft S_p0 Loss (0.5 * R_p0 + 0.3 * F1_p0 + 0.2 * P_p0) to directly boost P0 recall & precision!
    Y_p0 = (g == 3).float()
    tp_p0 = (p0 * Y_p0).sum(dim=0)
    fp_p0 = (p0 * (1.0 - Y_p0)).sum(dim=0)
    fn_p0 = ((1.0 - p0) * Y_p0).sum(dim=0)
    r_p0 = (tp_p0 + 1e-4) / (tp_p0 + fn_p0 + 1e-4)
    p_p0 = (tp_p0 + 1e-4) / (tp_p0 + fp_p0 + 1e-4)
    f1_p0 = (2.0 * tp_p0 + 1e-4) / (2.0 * tp_p0 + fp_p0 + fn_p0 + 1e-4)
    soft_sp0 = 0.5 * r_p0 + 0.3 * f1_p0 + 0.2 * p_p0
    loss_s_p0 = 1.0 - (soft_sp0 * c_weights6).sum() / c_weights6.sum()

    loss_rank = pairwise_ordinal_margin_loss(p0, p1, p2, g, focus_cis)
    loss_clean_p0 = p0[clean_mask].amax(dim=1).mean() if clean_mask.any() else 0.0
    loss_gate = F.binary_cross_entropy_with_logits(gate_logit, clean_mask.float())

    return (
        loss_bce
        + 0.75 * loss_ce
        + 0.65 * loss_emd
        + 0.65 * loss_s_pos
        + 0.55 * loss_s_p0
        + 0.70 * loss_rank
        + 0.20 * mono_pen
        + 0.40 * loss_clean_p0
        + 0.10 * loss_gate
    )


@torch.no_grad()
def predict_v9(net, loader, dev, alpha_ex=0.58):
    net.eval()
    P0, P1, P2, GATE, G = [], [], [], [], []
    for x, phys, g, _ in loader:
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


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,convnext:full')
    ap.add_argument('--focus-classes', default='过度生成', help='Comma-separated class names to specialize on')
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
    ap.add_argument('--seed', type=int, default=9)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--clean-repeat', type=int, default=3)
    ap.add_argument('--p0-clean-weight', type=float, default=4.2)
    ap.add_argument('--channel-neg-weight', type=float, default=1.00)
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
    print(f'=== Training V9 HiRes Specialist on focus_classes={a.focus_classes} (indices={focus_cis}) ===', flush=True)

    rows = load_split(a.split)
    man = read_manifest(a.manifest)
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})

    from extract_native_phys_feats import extract_for_names
    phys_map = extract_for_names([r['name'] for r in rows], a.phys_cache)

    tr_orig = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test']

    tr_real = [r for r in tr_orig if not r['name'].startswith(('v9_', 'v8_', 'v7_', 'v6_'))]
    phys_tr_mat = np.stack([phys_map[r['name']] for r in tr_real], axis=0)
    phys_mu = np.mean(phys_tr_mat, axis=0)
    phys_sd = np.std(phys_tr_mat, axis=0) + 1e-5

    Y_orig = targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr_real]))
    pw = ((len(Y_orig) - Y_orig.sum(0)) / Y_orig.sum(0).clamp(min=1)).clamp(1.0, 5.5).to(dev)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    tr = list(tr_orig)
    focus_real_pos = [r for r in tr_real if any(int(r[CLASSES[ci]]) > 0 for ci in focus_cis)]
    tr = tr + focus_real_pos
    if a.clean_repeat > 1:
        clean = [r for r in tr_real if all(int(r[c]) == 0 for c in CLASSES)]
        tr = tr + clean * (a.clean_repeat - 1)

    dtr = V9CaliperDataset(tr, phys_map, phys_mu, phys_sd, a.size, True, tiles=a.tiles)
    dva = V9CaliperDataset(va, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles)
    dte = V9CaliperDataset(te, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles)

    val_batch = a.batch * 2
    ltr = DataLoader(dtr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                     pin_memory=True, persistent_workers=True, prefetch_factor=4,
                     drop_last=True, worker_init_fn=lambda w: set_seed(a.seed + w))
    lva = DataLoader(dva, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True, persistent_workers=True)
    lte = DataLoader(dte, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    specs = []
    for s in a.backbones.split(','):
        name, mode = s.split(':')
        b = build_v9_backbone(name)
        specs.append((b, name, mode))

    net = MultiNetV9([x[0] for x in specs], phys_dim=60, dropout=a.dropout, n_quad=a.tiles * a.tiles).to(dev)
    groups = []
    frozen_modules = []
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
            for bl in blocks[:-k]:
                frozen_modules.append(bl)
        # Pre-cast all frozen backbone weights to bfloat16 once so autocast never casts 3.35B fp32 weights per step!
        for p in m.parameters():
            if not p.requires_grad and p.is_floating_point():
                p.data = p.data.to(torch.bfloat16)
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
          f'(size={a.size}, batch={a.batch}, accum={a.grad_accum}, views={1 + a.tiles*a.tiles + 2} [Fused-BF16 Accelerated])', flush=True)

    opt = torch.optim.AdamW(groups, weight_decay=a.wd, fused=True)
    accum = max(1, a.grad_accum)
    steps_per_ep = max(1, len(ltr) // accum)
    total_steps = steps_per_ep * a.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=total_steps, pct_start=0.1)

    best = None
    ep3_state = None
    for ep in range(a.epochs):
        net.train()
        for fm in frozen_modules:
            fm.eval()
        t0 = time.time()
        tot = 0.0
        nb = 0
        opt.zero_grad(set_to_none=True)
        for step_idx, (x, phys, g, _) in enumerate(ltr):
            x = x.to(dev, non_blocking=True)
            phys = phys.to(dev, non_blocking=True)
            g = g.to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lc, le, gl = net(x, phys)
            loss = v9_specialist_loss_fn(lc.float(), le.float(), gl.float(), g, bce_loss_none, focus_cis,
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
                opt.zero_grad(set_to_none=True)

        vp0, vp1, vp2, vgt, gv = predict_v9(net, lva, dev, alpha_ex=a.alpha_ex)
        b_sca, th_sca = optimise_scaled(gv, vp0, vp1, vp2, a.tau)
        f_per = [x[4] for x in b_sca['per'] if CLASSES.index(x[0]) in focus_cis]
        focus_val_f1 = float(np.mean(f_per)) if f_per else b_sca['S_pos']
        print(f'ep{ep+1} loss={tot/max(1,nb):.3f} focus_val_F1={focus_val_f1:.3f} val_sca={b_sca["total"]:.2f} '
              f'S_pos={b_sca["S_pos"]:.3f} P0R={b_sca["p0R"]:.3f} P0P={b_sca["p0P"]:.3f} S_clean={b_sca["S_clean"]:.3f} t={time.time()-t0:.0f}s', flush=True)

        cur_state = {k: v.cpu() for k, v in net.state_dict().items()}
        if (ep + 1) == 3:
            ep3_state = cur_state
        if best is None or focus_val_f1 > best[0]:
            best = (focus_val_f1, None, th_sca, cur_state)

    if ep3_state is not None:
        swa_state = {}
        for k in best[3].keys():
            if best[3][k].is_floating_point():
                swa_state[k] = (0.45 * ep3_state[k].float() + 0.55 * best[3][k].float()).to(best[3][k].dtype)
            else:
                swa_state[k] = best[3][k]
        net.load_state_dict(swa_state)
    else:
        net.load_state_dict(best[3])

    vp0, vp1, vp2, vgt, gva_n = predict_v9(net, lva, dev, alpha_ex=a.alpha_ex)
    tp0, tp1, tp2, tgt, gte_n = predict_v9(net, lte, dev, alpha_ex=a.alpha_ex)
    lva_f = DataLoader(V9CaliperDataset(va, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles, hflip=True),
                       batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    lte_f = DataLoader(V9CaliperDataset(te, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles, hflip=True),
                       batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    vp0_f, vp1_f, vp2_f, vgt_f, _ = predict_v9(net, lva_f, dev, alpha_ex=a.alpha_ex)
    tp0_f, tp1_f, tp2_f, tgt_f, _ = predict_v9(net, lte_f, dev, alpha_ex=a.alpha_ex)
    vp0, vp1, vp2, vgt = 0.5 * (vp0 + vp0_f), 0.5 * (vp1 + vp1_f), 0.5 * (vp2 + vp2_f), 0.5 * (vgt + vgt_f)
    tp0, tp1, tp2, tgt = 0.5 * (tp0 + tp0_f), 0.5 * (tp1 + tp1_f), 0.5 * (tp2 + tp2_f), 0.5 * (tgt + tgt_f)

    _, th_cas = cm.optimise(gva_n, vp0, vp1, vp2, a.tau)
    _, th_sca = optimise_scaled(gva_n, vp0, vp1, vp2, a.tau)

    tpl_150 = json.load(open('results/v2_full/template_150.json'))
    rows_150 = [{'name': e['image'], **{c: 0 for c in CLASSES}} for e in tpl_150]
    phys_map = extract_for_names([r['name'] for r in rows_150], a.phys_cache)
    l150_a = DataLoader(V9CaliperDataset(rows_150, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles, hflip=False),
                        batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    l150_b = DataLoader(V9CaliperDataset(rows_150, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles, hflip=True),
                        batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    p150_0a, p150_1a, p150_2a, g150_a, _ = predict_v9(net, l150_a, dev, alpha_ex=a.alpha_ex)
    p150_0b, p150_1b, p150_2b, g150_b, _ = predict_v9(net, l150_b, dev, alpha_ex=a.alpha_ex)
    t150_p0 = 0.5 * (p150_0a + p150_0b)
    t150_p1 = 0.5 * (p150_1a + p150_1b)
    t150_p2 = 0.5 * (p150_2a + p150_2b)
    t150_gate = 0.5 * (g150_a + g150_b)

    st_cas = cm.score(gte_n, cm.decode(tp0, tp1, tp2, *th_cas), a.tau)
    st_sca = cm.score(gte_n, decode_scaled_argmax(tp0, tp1, tp2, *th_sca), a.tau)

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
