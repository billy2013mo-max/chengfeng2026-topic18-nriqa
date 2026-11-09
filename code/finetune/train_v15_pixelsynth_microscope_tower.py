"""V15 Clean-Masked 1:1 Native-Microscope + Pixel-Space Counterfactual Multi-Specialist Trainer
(`train_v15_pixelsynth_microscope_tower.py`).

Key Innovations:
1. Consumes the 5-Backbone (`dinov3l`, `eva02`, `convnext`, `siglip2`, `pecoreg`) 448px Global + 3x 1:1 Native-Pixel
   Edge/Texture Microscope + Spatial Anomaly Token (`[CLS, mean, std, top32]`) feature cache (`/dev/shm/v15_microscope_cache`)
   AND the 60D 1:1 Native Physical & Frequency-Domain Calipers (`results/v8_experts/native_phys_feats.npz`).
2. Computes Explicit Physical Microscope Contrast Residuals per backbone:
   - `r_edge_glob = f_emax - f_glob` (isolates localized 1-2px edge ringing/overshoot in `过锐`, localized AI finger/eye/text
     distortions in `过度生成`, and portrait cutout boundary halos in `虚化突兀` against global context)
   - `r_edge_tex  = f_emax - f_tex`  (isolates edge sharpness/overshoot vs interior subject texture)
   - `r_tex_glob  = f_tex  - f_glob` (isolates interior subject texture waxy smoothing in `纹理损失` & defocus rolloff in `模糊`)
3. Trains 4 Dedicated Clean-Masked Specialist Models per Split (`s16, s17, s18, s19, s20` + `full`):
   - `v15a_overgen`:  Real + Clean-Masked `v9_og` (`过度生成` P2/P1/P0 counterfactual triplets)
   - `v15b_sharptex`: Real + Clean-Masked `v9_sh` (`过锐` P2/P1/P0 triplets) + `v9_bt` (`模糊+纹理损失` P2/P1/P0 triplets)
   - `v15c_bokeh`:    Real + Clean-Masked `v9_bk` (`虚化突兀` P0 cutouts) + `v9_bt` (`模糊+纹理损失` triplets)
   - `v15d_all`:      Real + All 1,620 Clean-Masked Pixel-Space Counterfactuals with Direct Official-Metric Soft-F1 Loss
4. Intra-Scene Counterfactual Severity Ladder Margin Loss:
   Because each synthetic triplet `(P2, P1, P0)` is synthesized from a single real base image, enforces strict
   intra-triplet severity ordering (`sev(P2) < sev(P1) < sev(P0)`) and marginal grade separation (`P1` vs `P2`).
"""
import argparse, csv, glob, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from utils import competition_score, DIMS, FULL_DIMS, P0_ONLY_DIMS
from cv_decoder_calib import decode_pc, HAND
from extract_native_phys_feats import extract_for_names

CACHE_DIR = '/dev/shm/v15_microscope_cache'
OUT_DIR = 'results/v15_microscope_synth'
os.makedirs(OUT_DIR, exist_ok=True)
BACKBONES = ['dinov3l', 'eva02', 'convnext', 'siglip2', 'pecoreg']
CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
FULL = ['模糊', '纹理损失', '过锐', '过度生成']
FULL_IDX = [0, 1, 2, 4]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_sample_class_mask6(name):
    """Per-sample 6-class loss mask (and bit 6 = real clean gate mask).
    Fixes V9 synthetic multi-label pollution: synthetic samples ONLY supervise their target synthesized class!
    """
    if name.startswith('v9_og'):
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    if name.startswith('v9_sh'):
        return np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if name.startswith('v9_bk'):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    if name.startswith('v9_bt'):
        return np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def load_v15_shared_cache():
    t0 = time.time()
    names = list(np.load(os.path.join(CACHE_DIR, 'names.npy'), allow_pickle=True))
    name_to_idx = {str(n): i for i, n in enumerate(names)}
    glob_list, emax_list, emean_list, tex_list = [], [], [], []
    dims = []
    for bname in BACKBONES:
        d = np.load(os.path.join(CACHE_DIR, f'{bname}_merged.npz'))
        glob_list.append(d['glob'])
        emax_list.append(d['emax'])
        emean_list.append(d['emean'])
        tex_list.append(d['tex'])
        dims.append(d['glob'].shape[1])
    X_glob = np.concatenate(glob_list, axis=1)
    X_emax = np.concatenate(emax_list, axis=1)
    X_emean = np.concatenate(emean_list, axis=1)
    X_tex = np.concatenate(tex_list, axis=1)
    phys_map = extract_for_names(names)
    X_phys = np.stack([phys_map[str(n)] for n in names], axis=0).astype(np.float32)
    print(f'[V15 Cache] Loaded {len(names)} imgs x {X_glob.shape[1]}D (4 microscope views + 60D phys) in {time.time()-t0:.1f}s', flush=True)
    return {
        'names': names,
        'name_to_idx': name_to_idx,
        'X_glob': X_glob,
        'X_emax': X_emax,
        'X_emean': X_emean,
        'X_tex': X_tex,
        'X_phys': X_phys,
        'dims': dims,
    }


class V15MicroscopeSpecialistNet(nn.Module):
    """Multi-Backbone 448px + 1:1 Native-Pixel Microscope Residual Gated Specialist Network.
    Combines:
    - Per-backbone projection of `glob`, `emax`, `emean`, `tex`, and explicit microscope contrast residuals
      (`emax - glob`, `emax - tex`, `tex - glob`)
    - 60D Native Physical Caliper MLP + SE-Gating
    - Dual-Stream Ordinal CORN (14-ch cumulative) + 4-State Marginal DeQAScore Soft-Label Head (20-ch) + All-Clean Gate
    """
    def __init__(self, dims, phys_dim=60, hidden=768, drop=0.30):
        super().__init__()
        self.dims = dims
        self.offsets = [0] + list(np.cumsum(dims))
        self.n_b = len(dims)
        per_b_dim = 256

        self.b_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d * 6, per_b_dim),
                nn.LayerNorm(per_b_dim),
                nn.GELU(),
                nn.Dropout(drop),
            )
            for d in dims
        ])
        self.phys_mlp = nn.Sequential(
            nn.Linear(phys_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )
        in_total = self.n_b * per_b_dim + 192
        self.se_gate = nn.Sequential(
            nn.Linear(in_total, in_total // 4),
            nn.GELU(),
            nn.Linear(in_total // 4, in_total),
            nn.Sigmoid(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(in_total, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(drop * 0.7),
        )
        # Per-class specialized sub-heads (6 classes)
        # Full classes (0,1,2,4) output: 3 cumulative CORN logits + 4 marginal grade logits = 7 logits
        # Binary classes (3,5) output: 1 P0 logit + 2 marginal logits = 3 logits
        self.head_cum = nn.Linear(hidden, 14)   # [blur(3), tex(3), sharp(3), banding(1), overgen(3), bokeh(1)]
        self.head_marg4 = nn.Linear(hidden, 16) # 4 FULL classes x 4 states (Clean, P2, P1, P0)
        self.head_gate = nn.Linear(hidden, 1)   # Real All-Clean gate logit

    def forward(self, g, emax, emean, tex, phys):
        b_feats = []
        for bi in range(self.n_b):
            s, e = self.offsets[bi], self.offsets[bi + 1]
            gb = g[:, s:e]
            emb = emax[:, s:e]
            emnb = emean[:, s:e]
            txb = tex[:, s:e]
            r_eg = emb - gb
            r_et = emb - txb
            cat_b = torch.cat([gb, emb, emnb, txb, r_eg, r_et], dim=-1)
            b_feats.append(self.b_projs[bi](cat_b))
        pf = self.phys_mlp(phys)
        z = torch.cat(b_feats + [pf], dim=-1)
        z = z * self.se_gate(z)
        h = self.trunk(z)
        lc = self.head_cum(h)
        lm = self.head_marg4(h).view(-1, 4, 4)  # (N, 4_full, 4_grades: 0=Clean, 1=P2, 2=P1, 3=P0)
        lg = self.head_gate(h).squeeze(-1)
        return lc, lm, lg


def decode_v15_outputs(lc, lm, alpha_marg=0.42):
    """Combine cumulative CORN probabilities and 4-state marginal softmax probabilities."""
    # Cumulative probabilities
    # lc layout:
    # 0,1,2: blur (>=P2, >=P1, P0)
    # 3,4,5: tex  (>=P2, >=P1, P0)
    # 6,7,8: sharp(>=P2, >=P1, P0)
    # 9:     banding (P0)
    # 10,11,12: overgen (>=P2, >=P1, P0)
    # 13:    bokeh (P0)
    pc = torch.sigmoid(lc)
    pm = F.softmax(lm, dim=-1)  # (N, 4, 4): [:, k, 0=Clean, 1=P2, 2=P1, 3=P0]

    p0_list, p1_list, p2_list = [], [], []
    full_map = {0: (0, 0), 1: (3, 1), 2: (6, 2), 4: (10, 3)}
    for ci in range(6):
        if ci in full_map:
            base_c, k = full_map[ci]
            c_ge1 = pc[:, base_c]
            c_ge2 = pc[:, base_c + 1]
            c_p0  = pc[:, base_c + 2]
            # Unconditional CORN chain rule + direct sigmoid blend
            p0_corn = 0.55 * c_p0 + 0.45 * (c_ge1 * c_ge2 * c_p0)
            p1_corn = 0.55 * torch.clamp(c_ge2 - c_p0, 0.0, 1.0) + 0.45 * (c_ge1 * c_ge2 * (1.0 - c_p0))
            p2_corn = 0.55 * torch.clamp(c_ge1 - c_ge2, 0.0, 1.0) + 0.45 * (c_ge1 * (1.0 - c_ge2))

            p0_m = pm[:, k, 3]
            p1_m = pm[:, k, 2]
            p2_m = pm[:, k, 1]

            p0_list.append((1.0 - alpha_marg) * p0_corn + alpha_marg * p0_m)
            p1_list.append((1.0 - alpha_marg) * p1_corn + alpha_marg * p1_m)
            p2_list.append((1.0 - alpha_marg) * p2_corn + alpha_marg * p2_m)
        elif ci == 3:
            p0_list.append(pc[:, 9])
        elif ci == 5:
            p0_list.append(pc[:, 13])

    p0 = torch.stack(p0_list, dim=1)  # (N, 6)
    p1 = torch.stack(p1_list, dim=1)  # (N, 4)
    p2 = torch.stack(p2_list, dim=1)  # (N, 4)
    return p0, p1, p2


def compute_v15_loss(lc, lm, lg, g_target, mask7, focus_cis, pos_weight_14, p0_boost=1.35):
    """Clean-masked multi-task loss:
    1. Masked Cumulative/CORN BCE loss on 14 channels
    2. Masked DeQAScore Soft-Ordinal 4-State KL loss on the 4 FULL classes
    3. Direct Official-Metric Differentiable Soft-F1 + Soft-S_p0 loss (`25/18` FULL grade weight, `25/6` Binary weight, `20/6` S_p0 weight)
    4. Pairwise Ordinal Severity Margin Ranking loss (`Clean < P2 < P1 < P0`)
    5. Real-image All-Clean Gate loss (strictly masked to real human-annotated images `mask7[:, 6] == 1`)
    """
    dev = lc.device
    N = lc.shape[0]
    # Build 14-ch targets and masks
    t14 = torch.zeros((N, 14), device=dev, dtype=torch.float32)
    m14 = torch.zeros((N, 14), device=dev, dtype=torch.float32)
    full_cols = {0: (0, 0), 1: (3, 1), 2: (6, 2), 4: (10, 3)}
    for ci in range(6):
        w_c = mask7[:, ci] * (1.65 if ci in focus_cis else 1.0)
        gc = g_target[:, ci]
        if ci in full_cols:
            bcol, _ = full_cols[ci]
            t14[:, bcol]     = (gc >= 1).float()
            t14[:, bcol + 1] = (gc >= 2).float()
            t14[:, bcol + 2] = (gc == 3).float()
            m14[:, bcol]     = w_c
            m14[:, bcol + 1] = w_c * 1.15
            m14[:, bcol + 2] = w_c * p0_boost
        elif ci == 3:
            t14[:, 9] = (gc == 3).float()
            m14[:, 9] = w_c * p0_boost
        elif ci == 5:
            t14[:, 13] = (gc == 3).float()
            m14[:, 13] = w_c * p0_boost

    bce_raw = F.binary_cross_entropy_with_logits(lc, t14, pos_weight=pos_weight_14, reduction='none')
    loss_corn = (bce_raw * m14).sum() / (m14.sum() + 1e-6)

    # 2. DeQAScore Soft-Ordinal 4-State Marginal Loss on FULL classes
    # State index: 0=Clean(g=0), 1=P2(g=1), 2=P1(g=2), 3=P0(g=3)
    log_pm = F.log_softmax(lm, dim=-1)
    soft_templates = torch.tensor([
        [0.94, 0.05, 0.01, 0.00],  # g=0 (Clean: sharp zero boundary)
        [0.05, 0.78, 0.16, 0.01],  # g=1 (P2: mild defect)
        [0.01, 0.15, 0.74, 0.10],  # g=2 (P1: moderate defect)
        [0.00, 0.02, 0.12, 0.86],  # g=3 (P0: severe defect)
    ], device=dev, dtype=torch.float32)
    grade_cw = torch.tensor([1.0, 1.35, 1.50, 1.45], device=dev, dtype=torch.float32)
    loss_deqa = 0.0
    den_deqa = 0.0
    for k, ci in enumerate(FULL_IDX):
        w_c = mask7[:, ci] * (1.75 if ci in focus_cis else 1.0)
        gk = g_target[:, ci].long().clamp(0, 3)
        tgt_dist = soft_templates[gk]  # (N, 4)
        sample_cw = grade_cw[gk] * w_c
        kl_k = -(tgt_dist * log_pm[:, k, :]).sum(dim=-1)
        loss_deqa = loss_deqa + (kl_k * sample_cw).sum()
        den_deqa = den_deqa + sample_cw.sum()
    loss_deqa = loss_deqa / (den_deqa + 1e-6)

    # 3. Direct Differentiable Official Competition Metric Surrogate (`25*S_pos + 20*S_p0`)
    p0, p1, p2 = decode_v15_outputs(lc, lm, alpha_marg=0.42)
    soft_f1_loss = 0.0
    f1_w_sum = 0.0
    for ci in range(6):
        mc = (mask7[:, ci] > 0.5).float()
        if mc.sum() < 4:
            continue
        f_mult = 1.65 if ci in focus_cis else 1.0
        # Official P0 weight: 25/18 (FULL) or 25/6 (Binary) + 20/6 * (0.6*R + 0.4*F1)
        y0 = (g_target[:, ci] == 3).float() * mc
        pr0 = p0[:, ci] * mc
        tp0 = (pr0 * y0).sum()
        fp0 = (pr0 * (1.0 - y0) * mc).sum()
        fn0 = ((1.0 - pr0) * y0).sum()
        f1_0 = (2.0 * tp0 + 1e-4) / (2.0 * tp0 + fp0 + fn0 + 1e-4)
        r_0  = (tp0 + 1e-4) / (tp0 + fn0 + 1e-4)
        w0 = (3.8 if ci in (3, 5) else 2.2) * f_mult
        soft_f1_loss = soft_f1_loss + w0 * (1.0 - (0.65 * f1_0 + 0.35 * r_0))
        f1_w_sum += w0

        if ci in full_cols:
            _, k = full_cols[ci]
            for g_val, pr_g in [(2, p1[:, k]), (1, p2[:, k])]:
                yg = (g_target[:, ci] == g_val).float() * mc
                prg = pr_g * mc
                tpg = (prg * yg).sum()
                fpg = (prg * (1.0 - yg) * mc).sum()
                fng = ((1.0 - prg) * yg).sum()
                f1_g = (2.0 * tpg + 1e-4) / (2.0 * tpg + fpg + fng + 1e-4)
                wg = 1.45 * f_mult
                soft_f1_loss = soft_f1_loss + wg * (1.0 - f1_g)
                f1_w_sum += wg
    soft_f1_loss = soft_f1_loss / max(1e-6, f1_w_sum)

    # 4. Pairwise Ordinal Severity Margin Ranking Loss
    rank_loss = 0.0
    rank_cnt = 0
    for ci in range(6):
        mc_idx = torch.where(mask7[:, ci] > 0.5)[0]
        if len(mc_idx) < 8:
            continue
        gc = g_target[mc_idx, ci]
        if ci in full_cols:
            _, k = full_cols[ci]
            sev = p2[mc_idx, k] + 2.0 * p1[mc_idx, k] + 3.0 * p0[mc_idx, ci]
        else:
            sev = 3.0 * p0[mc_idx, ci]
        diff_g = gc[:, None] - gc[None, :]
        valid_pairs = (diff_g > 0)
        if valid_pairs.any():
            margin = 0.28 * diff_g.float()
            s_diff = sev[:, None] - sev[None, :]
            rl = F.relu(margin - s_diff)[valid_pairs].mean()
            rank_loss = rank_loss + (1.6 if ci in focus_cis else 1.0) * rl
            rank_cnt += 1
    if rank_cnt > 0:
        rank_loss = rank_loss / rank_cnt

    # 5. Real-Image All-Clean Gate Loss + Clean P0 Penalty
    real_m = mask7[:, 6]
    is_clean = (g_target == 0).all(dim=1).float()
    clean_real_m = (is_clean > 0.5) & (real_m > 0.5)
    loss_clean_p0 = p0[clean_real_m].amax(dim=1).mean() if clean_real_m.any() else 0.0
    loss_gate = (F.binary_cross_entropy_with_logits(lg, is_clean, reduction='none') * real_m).sum() / (real_m.sum() + 1e-6)

    return loss_corn + 0.45 * loss_deqa + 0.55 * soft_f1_loss + 0.25 * rank_loss + 0.30 * loss_clean_p0 + 0.15 * loss_gate


def load_split_rows_for_suite(sp, suite_name):
    """Load train/val/test rows for split `sp` (`16..20` or `'full'`) and specialist suite `suite_name`."""
    if sp == 'full':
        with open('data/manifest_combined.csv') as f:
            man_map = {r['name']: r for r in csv.DictReader(f)}
        with open('data/v9_fulldata/split_f0_real.csv') as f:
            sp_rows = list(csv.DictReader(f))
        tr_real = [man_map[r['name']] for r in sp_rows if r['split'] == 'train' and r['name'] in man_map]
        va_real = [man_map[r['name']] for r in sp_rows if r['split'] == 'val' and r['name'] in man_map]
        te_real = va_real
        if suite_name == 'real':
            return tr_real, va_real, te_real

        va_stems = {os.path.splitext(r['name'])[0] for r in va_real}
        syn_rows = []
        for f_idx in (0, 2, 4):
            m_path = f'data/v9_fulldata/manifest_f{f_idx}_v9.csv'
            if os.path.exists(m_path):
                with open(m_path) as f:
                    for r in csv.DictReader(f):
                        nm = r['name']
                        if not nm.startswith('v9_'):
                            continue
                        if f_idx != 0 and any(st in nm for st in va_stems):
                            continue
                        if suite_name == 'overgen' and not nm.startswith('v9_og'):
                            continue
                        if suite_name == 'sharptex' and not (nm.startswith('v9_sh') or nm.startswith('v9_bt')):
                            continue
                        if suite_name == 'bokeh' and not (nm.startswith('v9_bk') or nm.startswith('v9_bt')):
                            continue
                        syn_rows.append(r)
        return tr_real + syn_rows, va_real, te_real
    else:
        man_path = f'data/v9_ladders/manifest_s{sp}_v9.csv'
        with open(man_path) as f:
            man_map = {r['name']: r for r in csv.DictReader(f)}
        sp_col = {
            'overgen': 'overgen',
            'sharptex': 'sharptex',
            'bokeh': 'bokeh',
            'all': 'all',
            'real': 'all',
        }[suite_name]
        sp_path = f'data/v9_ladders/split_s{sp}_{sp_col}.csv'
        tr, va, te = [], [], []
        with open(sp_path) as f:
            for r in csv.DictReader(f):
                nm, s = r['name'], r['split']
                if nm not in man_map:
                    continue
                row = man_map[nm]
                if s == 'train':
                    tr.append(row)
                elif s == 'val':
                    va.append(row)
                elif s == 'test':
                    te.append(row)
        # If bokeh suite, also add blurtex synth rows to train
        if suite_name == 'bokeh':
            with open(f'data/v9_ladders/split_s{sp}_blurtex.csv') as f:
                for r in csv.DictReader(f):
                    if r['name'].startswith('v9_bt') and r['name'] in man_map:
                        tr.append(man_map[r['name']])
        if suite_name == 'real':
            tr = [r for r in tr if not r['name'].startswith('v9_')]
        return tr, va, te


def rows_to_tensors(rows, cache, phys_med, phys_sd, dev):
    name_to_idx = cache['name_to_idx']
    idxs = [name_to_idx[r['name']] for r in rows]
    g_np = np.array([[int(r[c]) for c in CLASSES] for r in rows], dtype=np.int64)
    m_np = np.stack([get_sample_class_mask6(r['name']) for r in rows], axis=0)
    xg = torch.from_numpy(cache['X_glob'][idxs].astype(np.float32)).to(dev)
    xem = torch.from_numpy(cache['X_emax'][idxs].astype(np.float32)).to(dev)
    xen = torch.from_numpy(cache['X_emean'][idxs].astype(np.float32)).to(dev)
    xt = torch.from_numpy(cache['X_tex'][idxs].astype(np.float32)).to(dev)
    xp_raw = cache['X_phys'][idxs]
    xp = torch.from_numpy(np.clip((xp_raw - phys_med) / phys_sd, -6.0, 6.0).astype(np.float32)).to(dev)
    gt = torch.from_numpy(g_np).to(dev)
    mt = torch.from_numpy(m_np).to(dev)
    return xg, xem, xen, xt, xp, gt, mt, [r['name'] for r in rows], g_np


def train_one_specialist(sp, suite_Tag, suite_name, focus_cis, cache, tpl_names, dev, seed=42):
    t0 = time.time()
    set_seed(seed)
    tr_rows, va_rows, te_rows = load_split_rows_for_suite(sp, suite_name)

    # Compute physical feature standardization strictly on real training images of this split!
    name_to_idx = cache['name_to_idx']
    real_tr_idxs = [name_to_idx[r['name']] for r in tr_rows if not r['name'].startswith('v9_')]
    phys_tr = cache['X_phys'][real_tr_idxs]
    phys_med = np.median(phys_tr, axis=0, keepdims=True)
    iqr = np.percentile(phys_tr, 75, axis=0, keepdims=True) - np.percentile(phys_tr, 25, axis=0, keepdims=True)
    phys_sd = np.maximum(iqr / 1.349, phys_tr.std(axis=0, keepdims=True) + 1e-5)

    xg_tr, xem_tr, xen_tr, xt_tr, xp_tr, g_tr, m_tr, _, g_tr_np = rows_to_tensors(tr_rows, cache, phys_med, phys_sd, dev)
    xg_va, xem_va, xen_va, xt_va, xp_va, g_va, m_va, va_names, g_va_np = rows_to_tensors(va_rows, cache, phys_med, phys_sd, dev)
    xg_te, xem_te, xen_te, xt_te, xp_te, g_te, m_te, te_names, g_te_np = rows_to_tensors(te_rows, cache, phys_med, phys_sd, dev)

    t150_rows = [{'name': n, **{c: 0 for c in CLASSES}} for n in tpl_names]
    xg_150, xem_150, xen_150, xt_150, xp_150, _, _, _, _ = rows_to_tensors(t150_rows, cache, phys_med, phys_sd, dev)

    # Compute masked positive class weights on real training rows
    real_g = g_tr_np[:len(real_tr_idxs)]
    pw_list = []
    for ci in range(6):
        gc = real_g[:, ci]
        if ci in FULL_IDX:
            for th in (1, 2, 3):
                pos = float((gc >= th).sum()) if th < 3 else float((gc == 3).sum())
                neg = float(len(gc) - pos)
                pw_list.append(min(6.0, max(1.0, math.sqrt(neg / max(1.0, pos)))))
        else:
            pos = float((gc == 3).sum())
            neg = float(len(gc) - pos)
            pw_list.append(min(6.0, max(1.0, math.sqrt(neg / max(1.0, pos)))))
    pos_weight_14 = torch.tensor(pw_list, device=dev, dtype=torch.float32)

    # Train 2 seeds and average for low-variance specialist predictions
    va_p0_seeds, va_p1_seeds, va_p2_seeds = [], [], []
    te_p0_seeds, te_p1_seeds, te_p2_seeds = [], [], []
    t150_p0_seeds, t150_p1_seeds, t150_p2_seeds = [], [], []

    N_tr = len(tr_rows)
    batch_size = 192
    epochs = 28

    for s_idx in range(2):
        set_seed(seed + s_idx * 101)
        net = V15MicroscopeSpecialistNet(cache['dims'], phys_dim=60, hidden=768, drop=0.30).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=4.5e-4, weight_decay=2e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        best_val_sc = -1.0
        best_preds = None
        swa_preds = []

        for ep in range(epochs):
            net.train()
            perm = torch.randperm(N_tr, device=dev)
            for bi in range(0, N_tr, batch_size):
                idx = perm[bi:bi + batch_size]
                # Feature-level Gaussian jitter + view dropout regularization
                noise = 0.015 * torch.randn_like(xg_tr[idx])
                opt.zero_grad(set_to_none=True)
                lc, lm, lg = net(
                    xg_tr[idx] + noise,
                    xem_tr[idx] + noise,
                    xen_tr[idx] + noise,
                    xt_tr[idx] + noise,
                    xp_tr[idx],
                )
                loss = compute_v15_loss(lc, lm, lg, g_tr[idx], m_tr[idx], focus_cis, pos_weight_14)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sch.step()

            if ep >= 14 and (ep % 2 == 1 or ep == epochs - 1):
                net.eval()
                with torch.inference_mode():
                    vp0, vp1, vp2 = decode_v15_outputs(*net(xg_va, xem_va, xen_va, xt_va, xp_va)[:2])
                    tp0, tp1, tp2 = decode_v15_outputs(*net(xg_te, xem_te, xen_te, xt_te, xp_te)[:2])
                    p150_0, p150_1, p150_2 = decode_v15_outputs(*net(xg_150, xem_150, xen_150, xt_150, xp_150)[:2])

                vp0_np, vp1_np, vp2_np = vp0.cpu().numpy(), vp1.cpu().numpy(), vp2.cpu().numpy()
                tp0_np, tp1_np, tp2_np = tp0.cpu().numpy(), tp1.cpu().numpy(), tp2.cpu().numpy()
                p150_0_np, p150_1_np, p150_2_np = p150_0.cpu().numpy(), p150_1.cpu().numpy(), p150_2.cpu().numpy()

                if ep >= 18:
                    swa_preds.append((vp0_np, vp1_np, vp2_np, tp0_np, tp1_np, tp2_np, p150_0_np, p150_1_np, p150_2_np))

                prm = dict(HAND)
                prm['T0'] = [0.33, 0.31, 0.27, 0.42, 0.27, 0.39]
                pr_va = decode_pc(vp0_np, vp1_np, vp2_np, prm)
                sc_va = competition_score(pr_va, g_va_np)
                # Score focus classes + overall total on val
                focus_f1 = np.mean([x[4] for x in sc_va['per'] if CLASSES.index(x[0]) in focus_cis])
                val_metric = 25.0 * focus_f1 + 0.35 * sc_va['total']
                if val_metric > best_val_sc:
                    best_val_sc = val_metric
                    best_preds = (vp0_np, vp1_np, vp2_np, tp0_np, tp1_np, tp2_np, p150_0_np, p150_1_np, p150_2_np)

        # Blend best checkpoint (0.55) + SWA tail average (0.45)
        if swa_preds:
            swa_avg = [np.mean([x[i] for x in swa_preds], axis=0) for i in range(9)]
            fin = [0.55 * best_preds[i] + 0.45 * swa_avg[i] for i in range(9)]
        else:
            fin = best_preds

        va_p0_seeds.append(fin[0]); va_p1_seeds.append(fin[1]); va_p2_seeds.append(fin[2])
        te_p0_seeds.append(fin[3]); te_p1_seeds.append(fin[4]); te_p2_seeds.append(fin[5])
        t150_p0_seeds.append(fin[6]); t150_p1_seeds.append(fin[7]); t150_p2_seeds.append(fin[8])

    vp0 = np.mean(va_p0_seeds, axis=0)
    vp1 = np.mean(va_p1_seeds, axis=0)
    vp2 = np.mean(va_p2_seeds, axis=0)
    tp0 = np.mean(te_p0_seeds, axis=0)
    tp1 = np.mean(te_p1_seeds, axis=0)
    tp2 = np.mean(te_p2_seeds, axis=0)
    t150_p0 = np.mean(t150_p0_seeds, axis=0)
    t150_p1 = np.mean(t150_p1_seeds, axis=0)
    t150_p2 = np.mean(t150_p2_seeds, axis=0)

    sp_str = f's{sp}' if sp != 'full' else 'full'
    out_npz = os.path.join(OUT_DIR, f'{suite_Tag}_{sp_str}_probs.npz')
    np.savez_compressed(
        out_npz,
        va_p0=vp0, va_p1=vp1, va_p2=vp2, gva=g_va_np, va_names=np.array(va_names),
        te_p0=tp0, te_p1=tp1, te_p2=tp2, gte=g_te_np, te_names=np.array(te_names),
        t150_p0=t150_p0, t150_p1=t150_p1, t150_p2=t150_p2, t150_names=np.array(tpl_names),
    )
    prm = dict(HAND); prm['T0'] = [0.33, 0.31, 0.27, 0.42, 0.27, 0.39]
    sc_te = competition_score(decode_pc(tp0, tp1, tp2, prm), g_te_np)
    print(f'[{sp_str} | {suite_Tag}] Done in {time.time()-t0:.1f}s | Standalone TE={sc_te["total"]:.2f} (S_pos={25*sc_te["S_pos"]:.2f}, S_p0={20*sc_te["S_p0"]:.2f})', flush=True)
    return vp0, vp1, vp2, tp0, tp1, tp2, t150_p0, t150_p1, t150_p2, g_va_np, g_te_np, va_names, te_names


def run_split_worker(sp_arg, gpu_id):
    dev = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(dev)
    cache = load_v15_shared_cache()
    tpl = json.load(open('results/v2_full/template_150.json'))
    tpl_names = [e['image'] for e in tpl]

    sp = 'full' if sp_arg == 'full' else int(sp_arg)
    suites = [
        ('v15a_overgen',  'overgen',  [4],          20261501),
        ('v15b_sharptex', 'sharptex', [0, 1, 2],    20261502),
        ('v15c_bokeh',    'bokeh',    [0, 1, 5],    20261503),
        ('v15d_all',      'all',      [0, 1, 2, 3, 4, 5], 20261504),
        ('v15e_real',     'real',     [0, 1, 2, 3, 4, 5], 20261505),
    ]
    res = {}
    for s_tag, s_name, f_cis, sd in suites:
        res[s_tag] = train_one_specialist(sp, s_tag, s_name, f_cis, cache, tpl_names, dev, seed=sd)

    # Build Decoupled Per-Channel V15 Specialist Ensemble (`v15_ens`)
    # Route each specialist to its clean-masked target channels!
    A = res['v15a_overgen']
    B = res['v15b_sharptex']
    C = res['v15c_bokeh']
    D = res['v15d_all']

    def route_v15(idx_offset):
        # idx_offset: 0 -> va (p0,p1,p2), 3 -> te (p0,p1,p2), 6 -> t150 (p0,p1,p2)
        p0_A, p1_A, p2_A = A[idx_offset], A[idx_offset + 1], A[idx_offset + 2]
        p0_B, p1_B, p2_B = B[idx_offset], B[idx_offset + 1], B[idx_offset + 2]
        p0_C, p1_C, p2_C = C[idx_offset], C[idx_offset + 1], C[idx_offset + 2]
        p0_D, p1_D, p2_D = D[idx_offset], D[idx_offset + 1], D[idx_offset + 2]

        p0 = np.zeros_like(p0_D)
        p1 = np.zeros_like(p1_D)
        p2 = np.zeros_like(p2_D)

        # Class 0: 模糊 (ci=0, k=0) -> B (sharptex), C (bokeh+blurtex), D (all)
        p0[:, 0] = 0.38 * p0_B[:, 0] + 0.32 * p0_C[:, 0] + 0.30 * p0_D[:, 0]
        p1[:, 0] = 0.40 * p1_B[:, 0] + 0.30 * p1_C[:, 0] + 0.30 * p1_D[:, 0]
        p2[:, 0] = 0.40 * p2_B[:, 0] + 0.30 * p2_C[:, 0] + 0.30 * p2_D[:, 0]

        # Class 1: 纹理损失 (ci=1, k=1) -> B (sharptex), C (bokeh+blurtex), D (all)
        p0[:, 1] = 0.40 * p0_B[:, 1] + 0.30 * p0_C[:, 1] + 0.30 * p0_D[:, 1]
        p1[:, 1] = 0.42 * p1_B[:, 1] + 0.28 * p1_C[:, 1] + 0.30 * p1_D[:, 1]
        p2[:, 1] = 0.42 * p2_B[:, 1] + 0.28 * p2_C[:, 1] + 0.30 * p2_D[:, 1]

        # Class 2: 过锐 (ci=2, k=2) -> B (sharptex), D (all), A (real on sharp)
        p0[:, 2] = 0.50 * p0_B[:, 2] + 0.35 * p0_D[:, 2] + 0.15 * p0_A[:, 2]
        p1[:, 2] = 0.52 * p1_B[:, 2] + 0.33 * p1_D[:, 2] + 0.15 * p1_A[:, 2]
        p2[:, 2] = 0.52 * p2_B[:, 2] + 0.33 * p2_D[:, 2] + 0.15 * p2_A[:, 2]

        # Class 3: 花屏 (ci=3) -> Equal across all 4 (100% pure real supervision on all 4!)
        p0[:, 3] = 0.25 * (p0_A[:, 3] + p0_B[:, 3] + p0_C[:, 3] + p0_D[:, 3])

        # Class 4: 过度生成 (ci=4, k=3) -> A (overgen), D (all), B (real on overgen)
        p0[:, 4] = 0.52 * p0_A[:, 4] + 0.33 * p0_D[:, 4] + 0.15 * p0_B[:, 4]
        p1[:, 4 - 1] = 0.55 * p1_A[:, 3] + 0.30 * p1_D[:, 3] + 0.15 * p1_B[:, 3]
        p2[:, 4 - 1] = 0.55 * p2_A[:, 3] + 0.30 * p2_D[:, 3] + 0.15 * p2_B[:, 3]

        # Class 5: 虚化突兀 (ci=5) -> C (bokeh specialist), D (all), A (real on bokeh)
        p0[:, 5] = 0.52 * p0_C[:, 5] + 0.33 * p0_D[:, 5] + 0.15 * p0_A[:, 5]
        return p0, p1, p2

    vp0, vp1, vp2 = route_v15(0)
    tp0, tp1, tp2 = route_v15(3)
    t150_p0, t150_p1, t150_p2 = route_v15(6)
    g_va_np, g_te_np, va_names, te_names = D[9], D[10], D[11], D[12]

    sp_str = f's{sp}' if sp != 'full' else 'full'
    out_ens = os.path.join(OUT_DIR, f'v15_ens_{sp_str}_probs.npz')
    np.savez_compressed(
        out_ens,
        va_p0=vp0, va_p1=vp1, va_p2=vp2, gva=g_va_np, va_names=np.array(va_names),
        te_p0=tp0, te_p1=tp1, te_p2=tp2, gte=g_te_np, te_names=np.array(te_names),
        t150_p0=t150_p0, t150_p1=t150_p1, t150_p2=t150_p2, t150_names=np.array(tpl_names),
    )
    prm = dict(HAND); prm['T0'] = [0.33, 0.31, 0.27, 0.42, 0.27, 0.39]
    sc_te = competition_score(decode_pc(tp0, tp1, tp2, prm), g_te_np)
    print(f'*** [{sp_str} | v15_ens] Routed V15 Ensemble TE={sc_te["total"]:.2f} (S_pos={25*sc_te["S_pos"]:.2f}, S_p0={20*sc_te["S_p0"]:.2f}, S_cln={20*sc_te["S_clean"]:.2f}) ***', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=str, default='all')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    if args.split == 'all':
        import subprocess
        splits = ['16', '17', '18', '19', '20', 'full']
        procs = [
            subprocess.Popen([sys.executable, __file__, '--split', sp_s, '--gpu', str(i)])
            for i, sp_s in enumerate(splits)
        ]
        for p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f'Split worker failed with {rc}')
    else:
        run_split_worker(args.split, args.gpu)
