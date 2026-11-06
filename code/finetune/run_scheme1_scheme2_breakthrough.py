""" Rigorous Zero-Leakage Implementation & Ablation of:
- Scheme 1: QualiRAG Scene-Matched Reference Residual IQA + Multi-Foundation k-NN Posteriors + DeQAScore Soft-Label Ordinal Tower
- Scheme 2: Official (2).md Two-Stage Decoupled All-Clean Gate + Cross-Defect Out-Of-Fold Stacking & Conditional Expected-Utility Decoder

Strict Zero-Leakage Protocol:
For each split `sp` in (16, 17) independently:
- `tr` (2,073 images): Reference Bank for QualiRAG + training set for DeQAScore Soft-Label Foundation Tower.
- `va` (500 images): 5-fold source-grouped OOF used to fit the Stage-2 All-Clean Gate, Cross-Defect Calibrator, and thresholds.
- `te` (1,000 images): 100% blind evaluation set (never touched during any training, scaling, or threshold selection).
"""
import os, sys, csv, glob, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL, targets_from_grades
from train_v2 import exact_probs_from_dual
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from extract_native_phys_feats import extract_for_names

FULL_IDX = [CLASSES.index(c) for c in FULL]
BIN_IDX = [i for i, c in enumerate(CLASSES) if c not in FULL]


def score_both(gold, pred, tau=0.02):
    s1 = cm.score(gold, pred, tau)
    clean = (gold == 0).all(1)
    pred_clean = (pred == 0).all(1)
    s_clean_doc2 = float((clean & pred_clean).sum() / max(1, clean.sum()))
    tot_doc2 = 25.0 * s1['S_pos'] + 20.0 * s1['S_p0'] + 20.0 * s_clean_doc2
    s1['S_clean_doc2'] = s_clean_doc2
    s1['total_doc2'] = tot_doc2
    return s1


def load_global_assets():
    t0 = time.time()
    man_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
    man_map = {r['name']: r for r in man_rows}
    all_names = [r['name'] for r in man_rows]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    all_sources = np.array([r['source'] for r in man_rows])
    all_grades = np.array([[int(man_map[n][c]) for c in CLASSES] for n in all_names], dtype=np.int64)

    print("1. Loading 6 high-res foundation feature banks for 3,573 images...")
    feat_files = [
        ('internvit6b', 'feats/internvit-6b_combined.npz'),
        ('cradio768',   'feats/c-radio-v4-h_combined.npz'),
        ('eupe',        'feats/eupe-vit-b_combined.npz'),
        ('convnextv2',  'feats/convnextv2-base_combined.npz'),
        ('pecorel',     'feats/pe-core-l14_combined.npz'),
        ('siglip2',     'feats/siglip2-so400m_combined.npz'),
    ]
    blocks = []
    block_dims = []
    for tag, fp in feat_files:
        d = np.load(fp, allow_pickle=True)
        fns = [str(x) for x in d['names']]
        idx_map = {n: i for i, n in enumerate(fns)}
        X = d['X'][[idx_map[n] for n in all_names]].astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        blocks.append(X)
        block_dims.append(X.shape[1])

    # Load physical, texture, hand, graphic_score, resolution features (133 dims)
    tex_d = np.load('feats/texture.npz', allow_pickle=True)
    tex_map = {str(n): i for i, n in enumerate(tex_d['names'])}
    X_tex = np.stack([tex_d['X'][tex_map[n]] for n in all_names]).astype(np.float32)

    phys_dict = extract_for_names(all_names)
    X_phys = np.stack([phys_dict[n] for n in all_names]).astype(np.float32)

    hand_d = np.load('results/hand_features.npz', allow_pickle=True)
    hand_map = {str(n): i for i, n in enumerate(hand_d['names'])}
    X_hand = np.stack([hand_d['X'][hand_map[n]] for n in all_names]).astype(np.float32)

    gs_d = np.load('results/graphic_score.npz', allow_pickle=True)
    gs_map = {str(n): float(s) for n, s in zip(gs_d['names'], gs_d['score'])}
    X_gs = np.array([[gs_map.get(n, 0.5)] for n in all_names], dtype=np.float32)

    X_res = []
    for n in all_names:
        r = man_map[n]
        w, h = float(r['width']), float(r['height'])
        mn, mx = min(w, h), max(w, h)
        X_res.append([np.log(w * h), w / h, mn / 1080.0, mx / 1920.0, float(mn < 720), float(mn >= 1080)])
    X_res = np.array(X_res, dtype=np.float32)

    X_aux_raw = np.nan_to_num(np.concatenate([X_tex, X_phys, X_hand, X_gs, X_res], axis=1), nan=0.0, posinf=10.0, neginf=-10.0)
    print(f"   Loaded 6 foundation banks ({sum(block_dims)} dims) + {X_aux_raw.shape[1]}-dim physical/scene calipers in {time.time()-t0:.1f}s")
    return dict(
        all_names=all_names, name_to_idx=name_to_idx, all_sources=all_sources,
        all_grades=all_grades, blocks=blocks, block_dims=block_dims,
        X_aux_raw=X_aux_raw, X_gs=X_gs[:, 0], X_res=X_res
    )


def compute_qualirag_features(assets, tr_idx, va_idx, te_idx, dev):
    """Scheme 1: Compute QualiRAG Scene-Matched Reference Residual & Multi-Foundation k-NN features
    strictly querying the 2,073-image training bank `tr_idx`!
    For `tr_idx` itself, masks out same-source images (Leave-One-Source-Out) so training & inference distributions match!
    """
    t0 = time.time()
    blocks = assets['blocks']
    X_aux_raw = assets['X_aux_raw']
    all_grades = assets['all_grades']
    all_sources = assets['all_sources']

    # Standardize X_aux_raw strictly on tr_idx
    mu_a = X_aux_raw[tr_idx].mean(axis=0, keepdims=True)
    sd_a = X_aux_raw[tr_idx].std(axis=0, keepdims=True) + 1e-6
    X_aux = np.clip((X_aux_raw - mu_a) / sd_a, -5.0, 5.0).astype(np.float32)

    # Move normalized foundation blocks to GPU for ultra-fast similarity computation
    # Wait: 3573 x 2073 similarity matrix is tiny (only 7.4M floats = 30MB)!
    N_all = len(all_grades)
    N_tr = len(tr_idx)
    g_tr = all_grades[tr_idx]  # (2073, 6)
    src_all = all_sources
    src_tr = all_sources[tr_idx]

    # Same-source mask between all 3573 images and 2073 tr images: True where source matches (must exclude!)
    same_src_mask = (src_all[:, None] == src_tr[None, :])  # (3573, 2073)
    same_src_t = torch.from_numpy(same_src_mask).to(dev)

    # Precompute binary target indicators on tr_idx (2073, ...)
    # 1. All-Clean indicator (1 dim)
    y_clean_tr = (g_tr == 0).all(axis=1).astype(np.float32)[:, None]  # (2073, 1)
    # 2. 14 exact channels: P0 (6), P1 (4), P2 (4)
    y_p0_tr = (g_tr == 3).astype(np.float32)  # (2073, 6)
    y_p1_tr = (g_tr[:, FULL_IDX] == 2).astype(np.float32)  # (2073, 4)
    y_p2_tr = (g_tr[:, FULL_IDX] == 1).astype(np.float32)  # (2073, 4)
    # 3. Cumulative defect indicators: g >= 1 (6), g >= 2 (4)
    y_ge1_tr = (g_tr >= 1).astype(np.float32)  # (2073, 6)
    y_ge2_tr = (g_tr[:, FULL_IDX] >= 2).astype(np.float32)  # (2073, 4)
    # 4. Continuous severity score: 0, 1, 2, 3 (6)
    y_sev_tr = g_tr.astype(np.float32) / 3.0  # (2073, 6)

    Y_bank = np.concatenate([y_clean_tr, y_p0_tr, y_p1_tr, y_p2_tr, y_ge1_tr, y_ge2_tr, y_sev_tr], axis=1)  # (2073, 31)
    Y_bank_t = torch.from_numpy(Y_bank).to(dev)

    # Compute individual backbone similarity matrices S_b: (3573, 2073)
    sim_mats = []
    for b in blocks:
        B_all = torch.from_numpy(b).to(dev)
        B_tr = B_all[tr_idx]
        S = B_all @ B_tr.T  # (3573, 2073), cosine similarity since rows are L2-normalized
        sim_mats.append(S)

    # Combined semantic + visual similarity (weighted average of cosine similarities)
    # Semantic scene similarity emphasizes global semantic backbones (siglip2, pecorel, eupe, internvit6b)
    S_comb = (0.22 * sim_mats[0] + 0.20 * sim_mats[1] + 0.16 * sim_mats[2] +
              0.14 * sim_mats[3] + 0.14 * sim_mats[4] + 0.14 * sim_mats[5])
    S_sem = (0.35 * sim_mats[5] + 0.25 * sim_mats[4] + 0.20 * sim_mats[2] + 0.20 * sim_mats[0])

    # Domain/Resolution compatibility penalty: images with similar graphic_score & log_wh are better scene references!
    gs_all = torch.from_numpy(assets['X_gs']).to(dev)
    logwh_all = torch.from_numpy(assets['X_res'][:, 0]).to(dev)
    asp_all = torch.from_numpy(assets['X_res'][:, 1]).to(dev)

    dom_pen = (0.15 * torch.abs(gs_all[:, None] - gs_all[tr_idx][None, :]) +
               0.04 * torch.abs(logwh_all[:, None] - logwh_all[tr_idx][None, :]) +
               0.03 * torch.clamp(torch.abs(asp_all[:, None] - asp_all[tr_idx][None, :]), max=2.0))

    S_scene = S_sem - dom_pen
    S_scene_masked = S_scene.masked_fill(same_src_t, -1e4)
    S_comb_masked = S_comb.masked_fill(same_src_t, -1e4)

    # --- MODULE 1A: Scene-Matched Clean Reference Physical Residual ---
    clean_tr_mask = torch.from_numpy((g_tr == 0).all(axis=1)).to(dev)  # (2073,)
    S_clean_only = S_scene_masked.masked_fill(~clean_tr_mask[None, :], -1e4)
    # Retrieve Top-20 most similar All-Clean references
    top_vals_c, top_idx_c = torch.topk(S_clean_only, k=20, dim=1)  # (3573, 20)
    w_c = F.softmax(top_vals_c / 0.05, dim=1)  # (3573, 20)

    X_aux_t = torch.from_numpy(X_aux).to(dev)
    X_aux_tr = X_aux_t[tr_idx]  # (2073, 133)
    ref_feats = X_aux_tr[top_idx_c]  # (3573, 20, 133)
    mu_clean = (w_c.unsqueeze(-1) * ref_feats).sum(dim=1)  # (3573, 133)
    var_clean = (w_c.unsqueeze(-1) * (ref_feats - mu_clean.unsqueeze(1)) ** 2).sum(dim=1)
    sd_clean = torch.sqrt(var_clean + 0.15)

    delta_phys = (X_aux_t - mu_clean).cpu().numpy()  # (3573, 133)
    z_phys = ((X_aux_t - mu_clean) / sd_clean).cpu().numpy()  # (3573, 133)

    # Summary anomaly norms of Z_phys (how far does image x deviate from scene-matched clean references?)
    z_abs = np.abs(z_phys)
    z_summary = np.column_stack([
        np.mean(z_abs, axis=1),
        np.max(z_abs, axis=1),
        np.percentile(z_abs, 90, axis=1),
        np.percentile(z_abs, 75, axis=1),
        np.mean(z_abs[:, :24], axis=1),     # texture deviation from scene clean
        np.mean(z_abs[:, 24:84], axis=1),   # native 60d physical deviation from scene clean
        np.max(z_abs[:, 24:84], axis=1),
        np.mean(z_abs[:, 84:126], axis=1),  # hand structural deviation from scene clean
        top_vals_c[:, 0].cpu().numpy(),     # max similarity to a clean image
        top_vals_c[:, :5].mean(dim=1).cpu().numpy(),
    ])

    # --- MODULE 1B: Multi-Scale k-NN Posteriors & Contrastive Prototype Margins ---
    knn_feats_list = []
    # 1. Global combined similarity at (K=15, tau=0.04), (K=35, tau=0.06), (K=80, tau=0.08)
    for kk, tau in [(15, 0.04), (35, 0.06), (80, 0.08)]:
        tv, ti = torch.topk(S_comb_masked, k=kk, dim=1)
        w = F.softmax(tv / tau, dim=1)  # (3573, kk)
        post = (w.unsqueeze(-1) * Y_bank_t[ti]).sum(dim=1)  # (3573, 31)
        knn_feats_list.append(post.cpu().numpy())

    # 2. Per-backbone k-NN posteriors at (K=25, tau=0.05) for each of the 6 foundation models
    for S_b in sim_mats:
        S_bm = S_b.masked_fill(same_src_t, -1e4)
        tv, ti = torch.topk(S_bm, k=25, dim=1)
        w = F.softmax(tv / 0.05, dim=1)
        post_b = (w.unsqueeze(-1) * Y_bank_t[ti]).sum(dim=1)  # (3573, 31)
        knn_feats_list.append(post_b.cpu().numpy())

    # 3. Contrastive Prototype Margins: Top-5 similarity to class (c, g) minus Top-5 similarity to All-Clean
    sim_clean_top5 = top_vals_c[:, :5].mean(dim=1)  # (3573,)
    margins = []
    for col in range(1, 21):  # 14 exact channels + 6 ge1 channels
        pos_mask = (Y_bank_t[:, col] > 0.5)
        if pos_mask.sum() >= 5:
            S_pos = S_comb_masked.masked_fill(~pos_mask[None, :], -1e4)
            tv_p, _ = torch.topk(S_pos, k=5, dim=1)
            m_col = tv_p.mean(dim=1) - sim_clean_top5
            margins.append(m_col.cpu().numpy())
        else:
            margins.append(np.zeros(N_all, dtype=np.float32))
    margins = np.column_stack(margins)  # (3573, 20)

    X_knn = np.concatenate(knn_feats_list + [margins, z_summary], axis=1).astype(np.float32)
    print(f"   Computed QualiRAG features: delta_phys={delta_phys.shape}, z_phys={z_phys.shape}, X_knn={X_knn.shape} ({time.time()-t0:.1f}s)")
    return dict(X_aux=X_aux, delta_phys=delta_phys, z_phys=z_phys, X_knn=X_knn)


class DeQAScoreQualiRAGTower(nn.Module):
    """Scheme 1C: DeQAScore Soft-Label Ordinal + QualiRAG Residual Neural Tower
    Takes the 6 foundation feature blocks + X_aux + QualiRAG (delta_phys, z_phys, X_knn)
    and outputs:
    - Continuous DeQAScore soft-label ordinal distribution (6 classes x 4 grades)
    - Cumulative ordinal logits (14 dims)
    - Dedicated All-Clean logit (1 dim)
    """
    def __init__(self, block_dims, aux_dim, knn_dim, proj_dim=320, dropout=0.35):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
            ) for d in block_dims
        ])
        self.phys_proj = nn.Sequential(
            nn.Linear(aux_dim * 3, 192),  # [X_aux, delta_phys, z_phys]
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.knn_proj = nn.Sequential(
            nn.Linear(knn_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        tot_d = len(block_dims) * proj_dim + 128 + 128
        raw_d = sum(block_dims)
        self.linear_skip_cum = nn.Linear(raw_d, 14)
        self.linear_skip_ex = nn.Linear(raw_d, 20)

        self.head_cum = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(tot_d, 128), nn.GELU(), nn.Linear(128, 3 if c in FULL else 1))
            for c in CLASSES
        ])
        self.head_ex = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(tot_d, 128), nn.GELU(), nn.Linear(128, 4 if c in FULL else 2))
            for c in CLASSES
        ])
        self.clean_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(tot_d, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, xs, x_phys3, x_knn):
        zs = [proj(x) for proj, x in zip(self.projs, xs)]
        zp = self.phys_proj(x_phys3)
        zk = self.knn_proj(x_knn)
        h = torch.cat(zs + [zp, zk], dim=1)
        x_cat = torch.cat(xs, dim=1)
        lc = torch.cat([m(h) for m in self.head_cum], dim=1) + 0.45 * self.linear_skip_cum(x_cat)
        le = torch.cat([m(h) for m in self.head_ex], dim=1) + 0.45 * self.linear_skip_ex(x_cat)
        l_clean = self.clean_head(h).squeeze(1)
        return lc, le, l_clean


def deqascore_soft_ordinal_loss(lc, le, l_clean, grades, bce_none, sigma=0.48):
    """DeQAScore Gaussian Soft-Label Distribution Loss + Cumulative Ordinal Loss + All-Clean Gate Loss."""
    from train_v5 import v5_loss_fn
    base_l = v5_loss_fn(lc, le, -l_clean, grades, bce_none, p0_clean_weight=4.0, channel_neg_weight=1.25, emd_weight=0.55, alpha_ex=0.55)

    # Add DeQAScore Gaussian Soft-Label KL Loss on the 4-grade exact heads
    soft_loss = 0.0
    off = 0
    device = le.device
    grid4 = torch.arange(4, device=device, dtype=torch.float32)[None, :]  # (1, 4)
    for ci, c in enumerate(CLASSES):
        k_dim = 4 if c in FULL else 2
        logits_c = le[:, off:off + k_dim]
        off += k_dim
        if k_dim == 4:
            g_c = grades[:, ci].float()[:, None]  # (B, 1)
            # Gaussian soft label centered at true grade g_c
            w_soft = torch.exp(-0.5 * ((grid4 - g_c) / sigma) ** 2)
            # Keep clean (0) sharp so clean images don't leak into P2!
            is_zero = (grades[:, ci] == 0)
            if is_zero.any():
                w_soft[is_zero] = torch.tensor([0.94, 0.06, 0.0, 0.0], device=device)
            w_soft = w_soft / w_soft.sum(dim=1, keepdim=True)
            log_p = F.log_softmax(logits_c, dim=1)
            soft_loss = soft_loss - (w_soft * log_p).sum(dim=1).mean()
        else:
            y_bin = (grades[:, ci] == 3).long()
            soft_loss = soft_loss + F.cross_entropy(logits_c, y_bin, label_smoothing=0.03)

    # Dedicated All-Clean Binary Cross-Entropy Loss ( directly aligned with S_clean_doc2! )
    y_clean = (grades == 0).all(dim=1).float()
    clean_loss = F.binary_cross_entropy_with_logits(l_clean, y_clean)
    return base_l + 0.25 * (soft_loss / 6.0) + 0.60 * clean_loss


def train_deqascore_qualirag_tower(assets, q_feats, tr_idx, va_idx, te_idx, sp, dev, seeds=(1, 2, 3, 4)):
    t0 = time.time()
    blocks = assets['blocks']
    block_dims = assets['block_dims']
    all_grades = assets['all_grades']

    X_phys3 = np.concatenate([q_feats['X_aux'], q_feats['delta_phys'], q_feats['z_phys']], axis=1).astype(np.float32)
    X_knn = q_feats['X_knn']
    # Standardize X_knn on tr_idx
    mu_k = X_knn[tr_idx].mean(axis=0, keepdims=True)
    sd_k = X_knn[tr_idx].std(axis=0, keepdims=True) + 1e-6
    X_knn_s = np.clip((X_knn - mu_k) / sd_k, -5.0, 5.0).astype(np.float32)

    xs_tr = [torch.from_numpy(b[tr_idx]).to(dev) for b in blocks]
    xs_va = [torch.from_numpy(b[va_idx]).to(dev) for b in blocks]
    xs_te = [torch.from_numpy(b[te_idx]).to(dev) for b in blocks]

    xp_tr = torch.from_numpy(X_phys3[tr_idx]).to(dev)
    xp_va = torch.from_numpy(X_phys3[va_idx]).to(dev)
    xp_te = torch.from_numpy(X_phys3[te_idx]).to(dev)

    xk_tr = torch.from_numpy(X_knn_s[tr_idx]).to(dev)
    xk_va = torch.from_numpy(X_knn_s[va_idx]).to(dev)
    xk_te = torch.from_numpy(X_knn_s[te_idx]).to(dev)

    g_tr = torch.from_numpy(all_grades[tr_idx]).to(dev)
    Y_tr = targets_from_grades(g_tr)
    pos = Y_tr.sum(0).clamp(min=1.0)
    neg = (len(Y_tr) - pos).clamp(min=1.0)
    pw = (neg / pos).clamp(min=1.0, max=5.5)
    bce_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    va_p0, va_p1, va_p2, va_cln = 0.0, 0.0, 0.0, 0.0
    te_p0, te_p1, te_p2, te_cln = 0.0, 0.0, 0.0, 0.0

    n_tr = len(tr_idx)
    bs = 128
    for seed in seeds:
        torch.manual_seed(seed + sp * 100)
        np.random.seed(seed + sp * 100)
        net = DeQAScoreQualiRAGTower(block_dims, q_feats['X_aux'].shape[1], X_knn_s.shape[1], proj_dim=320, dropout=0.35).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=7e-4, weight_decay=2.5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=42, eta_min=2e-5)

        swa_va = [0.0, 0.0, 0.0, 0.0]
        swa_te = [0.0, 0.0, 0.0, 0.0]
        cnt = 0

        for ep in range(1, 43):
            net.train()
            perm = torch.randperm(n_tr, device=dev)
            for i in range(0, n_tr, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                lc, le, l_cln = net([x[idx] for x in xs_tr], xp_tr[idx], xk_tr[idx])
                loss = deqascore_soft_ordinal_loss(lc, le, l_cln, g_tr[idx], bce_none)
                loss.backward()
                opt.step()
            sched.step()

            if ep >= 24 and ep % 3 == 0:
                net.eval()
                with torch.no_grad():
                    lc_v, le_v, lcln_v = net(xs_va, xp_va, xk_va)
                    p0_v, p1_v, p2_v, _ = exact_probs_from_dual(lc_v, le_v, alpha_ex=0.55)
                    pcln_v = torch.sigmoid(lcln_v)

                    lc_t, le_t, lcln_t = net(xs_te, xp_te, xk_te)
                    p0_t, p1_t, p2_t, _ = exact_probs_from_dual(lc_t, le_t, alpha_ex=0.55)
                    pcln_t = torch.sigmoid(lcln_t)

                swa_va[0] += p0_v.cpu().numpy()
                swa_va[1] += p1_v.cpu().numpy()
                swa_va[2] += p2_v.cpu().numpy()
                swa_va[3] += pcln_v.cpu().numpy()

                swa_te[0] += p0_t.cpu().numpy()
                swa_te[1] += p1_t.cpu().numpy()
                swa_te[2] += p2_t.cpu().numpy()
                swa_te[3] += pcln_t.cpu().numpy()
                cnt += 1

        va_p0 += swa_va[0] / cnt / len(seeds)
        va_p1 += swa_va[1] / cnt / len(seeds)
        va_p2 += swa_va[2] / cnt / len(seeds)
        va_cln += swa_va[3] / cnt / len(seeds)

        te_p0 += swa_te[0] / cnt / len(seeds)
        te_p1 += swa_te[1] / cnt / len(seeds)
        te_p2 += swa_te[2] / cnt / len(seeds)
        te_cln += swa_te[3] / cnt / len(seeds)

    print(f"   Trained DeQAScore+QualiRAG 6-Tower on s{sp}_tr ({n_tr} imgs) in {time.time()-t0:.1f}s")
    return dict(
        va_p0=va_p0, va_p1=va_p1, va_p2=va_p2, va_cln=va_cln,
        te_p0=te_p0, te_p1=te_p1, te_p2=te_p2, te_cln=te_cln,
        X_knn_s=X_knn_s
    )


def build_v9_enhanced_14ch_probs(models, split_prefix, deqa_res=None):
    """Integrate the V2..V8 Decoupled 14-Channel Specialist Pool WITH the V9 High-Res Specialists
    (V9_SpecA..E, V8_W2_ExpA..D) and optional DeQAScore+QualiRAG 6-Tower predictions!
    """
    p0, p1, p2, cln0, cln1, cln2 = build_decoupled_14ch_probs(models, split_prefix)

    def get_m(k, arr):
        return models[k][f'{split_prefix}_{arr}'] if k in models else None

    # Incorporate V9 Clean RealHiRes Giant into Clean Shield Base
    v9_e0 = get_m('v9_specialists/V9_SpecE_RealHiRes_Giant', 'p0')
    v9_e1 = get_m('v9_specialists/V9_SpecE_RealHiRes_Giant', 'p1')
    v9_e2 = get_m('v9_specialists/V9_SpecE_RealHiRes_Giant', 'p2')
    if v9_e0 is not None:
        cln0 = 0.82 * cln0 + 0.18 * v9_e0
        cln1 = 0.82 * cln1 + 0.18 * v9_e1
        cln2 = 0.82 * cln2 + 0.18 * v9_e2

    # Incorporate V9 Specialists into each defect channel
    v9_a0 = get_m('v9_specialists/V9_SpecA_OverGen_Giant', 'p0')
    v9_a1 = get_m('v9_specialists/V9_SpecA_OverGen_Giant', 'p1')
    v9_a2 = get_m('v9_specialists/V9_SpecA_OverGen_Giant', 'p2')

    v9_b0 = get_m('v9_specialists/V9_SpecB_SharpTex_HiRes', 'p0')
    v9_b1 = get_m('v9_specialists/V9_SpecB_SharpTex_HiRes', 'p1')
    v9_b2 = get_m('v9_specialists/V9_SpecB_SharpTex_HiRes', 'p2')

    v9_c0 = get_m('v9_specialists/V9_SpecC_AllP0_Tile3', 'p0')
    v9_c1 = get_m('v9_specialists/V9_SpecC_AllP0_Tile3', 'p1')
    v9_c2 = get_m('v9_specialists/V9_SpecC_AllP0_Tile3', 'p2')

    v9_d0 = get_m('v9_specialists/V9_SpecD_BokehOG_HiRes', 'p0')
    v9_d1 = get_m('v9_specialists/V9_SpecD_BokehOG_HiRes', 'p1')
    v9_d2 = get_m('v9_specialists/V9_SpecD_BokehOG_HiRes', 'p2')

    if v9_c0 is not None:
        # P0 channels
        p0[:, 0] = 0.78 * p0[:, 0] + 0.12 * v9_c0[:, 0] + 0.10 * v9_e0[:, 0]
        p0[:, 1] = 0.76 * p0[:, 1] + 0.14 * v9_b0[:, 1] + 0.10 * v9_c0[:, 1]
        p0[:, 2] = 0.72 * p0[:, 2] + 0.16 * v9_b0[:, 2] + 0.12 * v9_c0[:, 2]
        p0[:, 3] = 0.80 * p0[:, 3] + 0.12 * v9_c0[:, 3] + 0.08 * v9_e0[:, 3]
        p0[:, 4] = 0.65 * p0[:, 4] + 0.20 * v9_a0[:, 4] + 0.15 * v9_d0[:, 4]
        p0[:, 5] = 0.70 * p0[:, 5] + 0.18 * v9_d0[:, 5] + 0.12 * v9_c0[:, 5]

        # P1 channels (0:模糊, 1:纹理损失, 2:过锐, 3:过度生成)
        p1[:, 0] = 0.78 * p1[:, 0] + 0.12 * v9_c1[:, 0] + 0.10 * v9_e1[:, 0]
        p1[:, 1] = 0.74 * p1[:, 1] + 0.14 * v9_b1[:, 1] + 0.12 * v9_c1[:, 1]
        p1[:, 2] = 0.72 * p1[:, 2] + 0.18 * v9_b1[:, 2] + 0.10 * v9_c1[:, 2]
        p1[:, 3] = 0.62 * p1[:, 3] + 0.22 * v9_a1[:, 3] + 0.16 * v9_d1[:, 3]

        # P2 channels
        p2[:, 0] = 0.78 * p2[:, 0] + 0.12 * v9_c2[:, 0] + 0.10 * v9_e2[:, 0]
        p2[:, 1] = 0.76 * p2[:, 1] + 0.12 * v9_b2[:, 1] + 0.12 * v9_e2[:, 1]
        p2[:, 2] = 0.74 * p2[:, 2] + 0.16 * v9_b2[:, 2] + 0.10 * v9_e2[:, 2]
        p2[:, 3] = 0.65 * p2[:, 3] + 0.20 * v9_a2[:, 3] + 0.15 * v9_d2[:, 3]

    if deqa_res is not None:
        dq0 = deqa_res[f'{split_prefix}_p0']
        dq1 = deqa_res[f'{split_prefix}_p1']
        dq2 = deqa_res[f'{split_prefix}_p2']
        # Blend 12% DeQAScore+QualiRAG 6-Tower (trained on 2,073 tr images with InternViT-6B + C-RADIO + QualiRAG!)
        p0 = 0.88 * p0 + 0.12 * dq0
        p1 = 0.86 * p1 + 0.14 * dq1
        p2 = 0.86 * p2 + 0.14 * dq2
        cln0 = 0.88 * cln0 + 0.12 * dq0
        cln1 = 0.88 * cln1 + 0.12 * dq1
        cln2 = 0.88 * cln2 + 0.12 * dq2

    return p0, p1, p2, cln0, cln1, cln2


def compute_scene_model_bias_correction(assets, va_idx, te_idx, p0_va, p1_va, p2_va, gva, dev):
    """Module 1D: Scene-Conditional Model Bias Correction!
    Measures whether the base ensemble systematically over-predicts or under-predicts each defect
    on scenes visually/semantically similar to query image x!
    - For `va_idx`: uses Leave-One-Source-Out within `va_idx` (zero target leakage!).
    - For `te_idx`: queries `va_idx` (zero target leakage!).
    """
    blocks = assets['blocks']
    src_va = assets['all_sources'][va_idx]
    same_src_va = torch.from_numpy(src_va[:, None] == src_va[None, :]).to(dev)

    # Semantic similarity using SigLIP2 + PE-Core + EUPE + InternViT6B
    S_vv = 0.0
    S_tv = 0.0
    for idx_b, w_b in [(5, 0.35), (4, 0.25), (2, 0.20), (0, 0.20)]:
        B_va = torch.from_numpy(blocks[idx_b][va_idx]).to(dev)
        B_te = torch.from_numpy(blocks[idx_b][te_idx]).to(dev)
        S_vv = S_vv + w_b * (B_va @ B_va.T)
        S_tv = S_tv + w_b * (B_te @ B_va.T)

    gs_va = torch.from_numpy(assets['X_gs'][va_idx]).to(dev)
    gs_te = torch.from_numpy(assets['X_gs'][te_idx]).to(dev)
    S_vv = S_vv - 0.15 * torch.abs(gs_va[:, None] - gs_va[None, :])
    S_tv = S_tv - 0.15 * torch.abs(gs_te[:, None] - gs_va[None, :])
    S_vv = S_vv.masked_fill(same_src_va, -1e4)

    # True indicator matrix on va: [is_clean (1), y_p0 (6), y_p1 (4), y_p2 (4), y_ge1 (6)] = 21 cols
    pred_clean_va = np.prod(1.0 - np.clip(p0_va, 0, 0.99), axis=1, keepdims=True)
    cum_pred_va = p0_va.copy()
    for k, ci in enumerate(FULL_IDX):
        cum_pred_va[:, ci] = np.clip(p0_va[:, ci] + p1_va[:, k] + p2_va[:, k], 0.0, 1.0)
    pred_clean_va = np.prod(1.0 - 0.7 * cum_pred_va, axis=1, keepdims=True)

    Y_true_va = np.concatenate([
        (gva == 0).all(axis=1, keepdims=True).astype(np.float32),
        (gva == 3).astype(np.float32),
        (gva[:, FULL_IDX] == 2).astype(np.float32),
        (gva[:, FULL_IDX] == 1).astype(np.float32),
        (gva >= 1).astype(np.float32),
    ], axis=1)

    Y_pred_va = np.concatenate([
        pred_clean_va.astype(np.float32),
        p0_va.astype(np.float32),
        p1_va.astype(np.float32),
        p2_va.astype(np.float32),
        cum_pred_va.astype(np.float32),
    ], axis=1)

    # Residual R_va = Y_true_va - Y_pred_va: positive means human annotators label MORE than model predicts on this scene;
    # negative means model OVER-PREDICTS on this scene!
    R_va_t = torch.from_numpy(Y_true_va - Y_pred_va).to(dev)
    Y_true_va_t = torch.from_numpy(Y_true_va).to(dev)

    tv_vv, ti_vv = torch.topk(S_vv, k=20, dim=1)
    w_vv = F.softmax(tv_vv / 0.05, dim=1)
    bias_va = (w_vv.unsqueeze(-1) * R_va_t[ti_vv]).sum(dim=1).cpu().numpy()
    post_va = (w_vv.unsqueeze(-1) * Y_true_va_t[ti_vv]).sum(dim=1).cpu().numpy()

    tv_tv, ti_tv = torch.topk(S_tv, k=20, dim=1)
    w_tv = F.softmax(tv_tv / 0.05, dim=1)
    bias_te = (w_tv.unsqueeze(-1) * R_va_t[ti_tv]).sum(dim=1).cpu().numpy()
    post_te = (w_tv.unsqueeze(-1) * Y_true_va_t[ti_tv]).sum(dim=1).cpu().numpy()

    return np.concatenate([bias_va, post_va], axis=1), np.concatenate([bias_te, post_te], axis=1)


def build_rich_image_and_channel_features(models, split_prefix, idx_list, q_feats, deqa_res, scene_bias, include_qualirag=True):
    """Build feature matrix for Stage 2A (Image-Level All-Clean Classifier) and Stage 2B (14-Channel Cross-Defect Calibrator)."""
    p0, p1, p2, cln0, cln1, cln2 = build_v9_enhanced_14ch_probs(
        models, split_prefix, deqa_res if include_qualirag else None
    )
    N = len(idx_list)

    # 1. Cumulative & severity per class
    cum6 = p0.copy()
    cln_cum6 = cln0.copy()
    sev4 = np.zeros((N, 4), dtype=np.float32)
    for k, ci in enumerate(FULL_IDX):
        cum6[:, ci] = p0[:, ci] + p1[:, k] + p2[:, k]
        cln_cum6[:, ci] = cln0[:, ci] + cln1[:, k] + cln2[:, k]
        sev4[:, k] = 3.0 * p0[:, ci] + 2.0 * p1[:, k] + 1.0 * p2[:, k]

    # 2. Image-level joint Clean/Defect summary features
    p_all_clean_prod = np.prod(np.clip(1.0 - cln_cum6, 1e-4, 1.0), axis=1, keepdims=True)
    p_all_clean_ens  = np.prod(np.clip(1.0 - cum6, 1e-4, 1.0), axis=1, keepdims=True)
    cln_cum_sorted = np.sort(cln_cum6, axis=1)
    cum_sorted = np.sort(cum6, axis=1)

    # Collect neural gate predictions across all models that have va_gate / te_gate
    gate_list = []
    for k, m in sorted(models.items()):
        gk = f'{split_prefix}_gate'
        if gk in m and m[gk].shape == (N,):
            gate_list.append(m[gk])
    if gate_list:
        gate_stack = np.stack(gate_list, axis=1)
        gate_feats = np.column_stack([
            gate_stack.mean(axis=1),
            gate_stack.std(axis=1),
            np.percentile(gate_stack, 85, axis=1),
            np.percentile(gate_stack, 15, axis=1),
        ])
    else:
        gate_feats = np.zeros((N, 4), dtype=np.float32)

    # Collect family statistics across all models
    all_keys = sorted(models.keys())
    def get_14(m):
        return np.concatenate([m[f'{split_prefix}_p0'], m[f'{split_prefix}_p1'], m[f'{split_prefix}_p2']], axis=1)

    all_stack = np.stack([get_14(models[k]) for k in all_keys], axis=0)  # (M, N, 14)
    clean_keys = [k for k in all_keys if any(x in k for x in ['A_base', 'F_v5', 'C_p1', 'v2_', 'pcl_448', 'd3_448', 'pcg_seed', 'pcl_seed', 'SpecE'])]
    synth_keys = [k for k in all_keys if any(x in k for x in ['Exp', 'SpecA', 'SpecB', 'SpecC', 'SpecD', 'G_v6', 'H_og', 'E_synth'])]

    cln_stack = np.stack([get_14(models[k]) for k in clean_keys], axis=0)
    syn_stack = np.stack([get_14(models[k]) for k in synth_keys], axis=0)

    # Disagreement between synthetic specialists and pure clean models!
    syn_minus_cln = syn_stack.mean(axis=0) - cln_stack.mean(axis=0)  # (N, 14)

    # Cross-defect interaction & conditional grade ratio features
    ratios = []
    for k, ci in enumerate(FULL_IDX):
        q0, q1, q2 = p0[:, ci], p1[:, k], p2[:, k]
        tot = q0 + q1 + q2 + 1e-6
        ratios.extend([q0 / tot, q1 / tot, q2 / tot, (q0 + q1) / tot, q1 / (q1 + q2 + 1e-6), q0 / (q0 + q1 + 1e-6)])
    ratios = np.column_stack(ratios)

    inter = np.column_stack([
        p_all_clean_prod, p_all_clean_ens,
        cln_cum_sorted[:, -1], cln_cum_sorted[:, -2], cln_cum6.sum(axis=1),
        cum_sorted[:, -1], cum_sorted[:, -2], cum6.sum(axis=1),
        p0.max(axis=1), p0.sum(axis=1),
        p1.max(axis=1), p2.max(axis=1),
        cum6[:, 0] * cum6[:, 1],  # blur * tex
        cum6[:, 2] * cum6[:, 1],  # sharp * tex
        cum6[:, 4] * cum6[:, 1],  # overgen * tex
        cum6[:, 0] - p0[:, 5],    # blur - bokeh_p0
        cum6[:, 2] - cum6[:, 0],  # sharp - blur
    ])

    base_feats = [
        p0, p1, p2, cum6, cln0, cln1, cln2, cln_cum6, sev4,
        ratios, inter, gate_feats,
        all_stack.mean(axis=0), all_stack.std(axis=0), np.percentile(all_stack, 85, axis=0), np.percentile(all_stack, 15, axis=0),
        cln_stack.mean(axis=0), cln_stack.max(axis=0),
        syn_stack.mean(axis=0), syn_minus_cln,
        q_feats['X_aux'][idx_list],
    ]

    if include_qualirag:
        dq_p0 = deqa_res[f'{split_prefix}_p0']
        dq_p1 = deqa_res[f'{split_prefix}_p1']
        dq_p2 = deqa_res[f'{split_prefix}_p2']
        dq_cln = deqa_res[f'{split_prefix}_cln'][:, None]
        base_feats.extend([
            dq_p0, dq_p1, dq_p2, dq_cln,
            q_feats['delta_phys'][idx_list],
            q_feats['z_phys'][idx_list],
            q_feats['X_knn'][idx_list],
            scene_bias,
        ])

    X = np.concatenate(base_feats, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    return X, (p0, p1, p2, cln0, cln1, cln2, cum6, cln_cum6)


def fit_oof_and_predict_stage2(X_va, gva, src_va, base_va, X_te, base_te):
    """Scheme 2: Fit Stage-2A (Image-Level All-Clean Gate Classifier) and Stage-2B (14-Channel Cross-Defect Calibrator)
    using 5-fold GroupKFold (grouped by video `source`) on `va` (500 images) and predict on `te` (1,000 images)!
    Returns:
      - `p_clean_va` (500,) OOF and `p_clean_te` (1000,)
      - Calibrated `(p0_va, p1_va, p2_va)` OOF and `(p0_te, p1_te, p2_te)`
    """
    gkf = GroupKFold(n_splits=5)
    N_va = len(gva)
    N_te = len(X_te)

    scaler = StandardScaler()
    X_va_s = np.clip(scaler.fit_transform(X_va), -5.0, 5.0)
    X_te_s = np.clip(scaler.transform(X_te), -5.0, 5.0)

    # --- STAGE 2A: Dedicated Image-Level All-Clean Classifier P(All-Clean | x) ---
    y_clean_va = (gva == 0).all(axis=1).astype(int)
    oof_clean_va = np.zeros(N_va, dtype=np.float64)
    pred_clean_te = np.zeros(N_te, dtype=np.float64)

    for tr_f, vl_f in gkf.split(X_va_s, y_clean_va, groups=src_va):
        lr_c = LogisticRegression(C=0.03, class_weight='balanced', max_iter=500, random_state=42)
        lr_c.fit(X_va_s[tr_f], y_clean_va[tr_f])

        et_c = ExtraTreesClassifier(n_estimators=400, max_depth=7, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1)
        et_c.fit(X_va_s[tr_f], y_clean_va[tr_f])

        rf_c = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_c.fit(X_va_s[tr_f], y_clean_va[tr_f])

        oof_clean_va[vl_f] = (
            0.35 * lr_c.predict_proba(X_va_s[vl_f])[:, 1] +
            0.40 * et_c.predict_proba(X_va_s[vl_f])[:, 1] +
            0.25 * rf_c.predict_proba(X_va_s[vl_f])[:, 1]
        )
        pred_clean_te += (
            0.35 * lr_c.predict_proba(X_te_s)[:, 1] +
            0.40 * et_c.predict_proba(X_te_s)[:, 1] +
            0.25 * rf_c.predict_proba(X_te_s)[:, 1]
        ) / 5.0

    # Also fuse with analytical clean shield product probability `prod(1 - cln_cum6)`
    p_prod_va = np.prod(np.clip(1.0 - base_va[7], 1e-4, 1.0), axis=1)
    p_prod_te = np.prod(np.clip(1.0 - base_te[7], 1e-4, 1.0), axis=1)
    p_clean_va = 0.65 * oof_clean_va + 0.35 * p_prod_va
    p_clean_te = 0.65 * pred_clean_te + 0.35 * p_prod_te

    # --- STAGE 2B: Cross-Defect + QualiRAG Calibrator for 14 Channels & Cumulative Channels ---
    meta_p0_va = np.zeros((N_va, 6), dtype=np.float64)
    meta_p1_va = np.zeros((N_va, 4), dtype=np.float64)
    meta_p2_va = np.zeros((N_va, 4), dtype=np.float64)

    meta_p0_te = np.zeros((N_te, 6), dtype=np.float64)
    meta_p1_te = np.zeros((N_te, 4), dtype=np.float64)
    meta_p2_te = np.zeros((N_te, 4), dtype=np.float64)

    def fit_binary_oof(y_target, seed_offset=0):
        oof_p = np.zeros(N_va, dtype=np.float64)
        te_p = np.zeros(N_te, dtype=np.float64)
        if y_target.sum() < 4:
            return oof_p, te_p
        for tr_f, vl_f in gkf.split(X_va_s, y_target, groups=src_va):
            if y_target[tr_f].sum() < 2:
                continue
            lr = LogisticRegression(C=0.025, max_iter=500, random_state=42 + seed_offset)
            lr.fit(X_va_s[tr_f], y_target[tr_f])
            et = ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42 + seed_offset, n_jobs=-1)
            et.fit(X_va_s[tr_f], y_target[tr_f])
            oof_p[vl_f] = 0.45 * lr.predict_proba(X_va_s[vl_f])[:, 1] + 0.55 * et.predict_proba(X_va_s[vl_f])[:, 1]
            te_p += (0.45 * lr.predict_proba(X_te_s)[:, 1] + 0.55 * et.predict_proba(X_te_s)[:, 1]) / 5.0
        return oof_p, te_p

    for ci, cname in enumerate(CLASSES):
        y0 = (gva[:, ci] == 3).astype(int)
        oof0, te0 = fit_binary_oof(y0, seed_offset=ci * 10)
        meta_p0_va[:, ci] = oof0
        meta_p0_te[:, ci] = te0

        if cname in FULL:
            k = FULL.index(cname)
            y1 = (gva[:, ci] == 2).astype(int)
            oof1, te1 = fit_binary_oof(y1, seed_offset=ci * 10 + 1)
            y2 = (gva[:, ci] == 1).astype(int)
            oof2, te2 = fit_binary_oof(y2, seed_offset=ci * 10 + 2)
            meta_p1_va[:, k] = oof1
            meta_p1_te[:, k] = te1
            meta_p2_va[:, k] = oof2
            meta_p2_te[:, k] = te2

    # Blend base specialist probabilities with OOF-calibrated cross-defect + QualiRAG meta-probabilities
    # (alpha=0.28 meta + 0.72 specialist ensemble preserves sharp specialist dynamic range while injecting QualiRAG & cross-defect priors!)
    p0_va_cal = 0.72 * base_va[0] + 0.28 * meta_p0_va
    p1_va_cal = 0.70 * base_va[1] + 0.30 * meta_p1_va
    p2_va_cal = 0.70 * base_va[2] + 0.30 * meta_p2_va

    p0_te_cal = 0.72 * base_te[0] + 0.28 * meta_p0_te
    p1_te_cal = 0.70 * base_te[1] + 0.30 * meta_p1_te
    p2_te_cal = 0.70 * base_te[2] + 0.30 * meta_p2_te

    return dict(
        p_clean_va=p_clean_va, p_clean_te=p_clean_te,
        p0_va=p0_va_cal, p1_va=p1_va_cal, p2_va=p2_va_cal,
        p0_te=p0_te_cal, p1_te=p1_te_cal, p2_te=p2_te_cal,
        cln_cum_va=base_va[7], cln_cum_te=base_te[7],
    )


def decode_two_stage_doc2(p0, p1, p2, p_clean, cln_cum, params):
    """Scheme 2 Two-Stage Decoupled Decoder aligned with Official (2).md AND (1).md:
    Stage A (Image-Level All-Clean Gate & Rescue):
      - Any image with `p_clean >= tau_clean` AND `cln_cum.max(1) < tau_cln_max` is predicted as All-Clean (`0` across all 6 classes)!
      - Also protects Doc1 S_clean by blocking P0 on images with `p_clean >= tau_p0_shield`!
    Stage B (Conditional Grade Assignment on Confirmed Non-Clean Images):
      - Once an image is NOT gated as All-Clean, predicting P2 / P1 / P0 on it incurs ZERO penalty on `S_clean_doc2`!
      - Decodes each class using per-class `(t0, t1, t2, r12, t_cum)` parameters.
      - Rescue rule: if an image has `p_clean < tau_rescue` (strongly defective image) yet all 6 classes decoded to 0,
        assigns P2 (1) to its highest normalized cumulative-probability FULL class!
    """
    N = len(p0)
    pred = np.zeros((N, 6), dtype=int)

    tau_clean = params['tau_clean']
    tau_cln_max = params['tau_cln_max']
    tau_p0_shield = params['tau_p0_shield']
    tau_rescue = params['tau_rescue']

    is_gated_clean = (p_clean >= tau_clean) & (cln_cum.max(axis=1) < tau_cln_max)
    is_p0_shielded = (p_clean >= tau_p0_shield) | (cln_cum.max(axis=1) < 0.27)

    # Decode 4 multi-grade FULL classes: 模糊 (0), 纹理损失 (1), 过锐 (2), 过度生成 (4)
    for k, ci in enumerate(FULL_IDX):
        t0, t1, t2, r12, t_cum = params['full'][k]
        q0, q1, q2 = p0[:, ci], p1[:, k], p2[:, k]
        q_cum = q0 + q1 + q2

        # Defect presence condition (cumulative OR grade-specific)
        has_pos = (q2 >= t2) | (q1 >= t1) | (q_cum >= t_cum)
        c_pred = np.where(has_pos, 1, 0)

        # P1 vs P2 conditional competition
        s1 = q1 + 0.35 * q0
        is_p1 = has_pos & (s1 >= t1) & (s1 >= r12 * q2)
        c_pred = np.where(is_p1, 2, c_pred)

        # P0 assignment (protected by P0 shield for Doc1 S_clean=1.000!)
        is_p0 = (q0 >= t0) & (~is_p0_shielded)
        c_pred = np.where(is_p0, 3, c_pred)
        pred[:, ci] = c_pred

    # Decode 2 binary P0-only classes: 花屏 (3), 虚化突兀 (5)
    for b_i, ci in enumerate(BIN_IDX):
        t0 = params['bin'][b_i]
        is_p0 = (p0[:, ci] >= t0) & (~is_p0_shielded)
        pred[:, ci] = np.where(is_p0, 3, 0)

    # Apply Stage-A All-Clean Gate
    pred[is_gated_clean] = 0

    # Apply Stage-A Defective Image Rescue (if strongly non-clean image decoded to all-0, rescue its strongest defect as P2)
    still_all_zero = (pred == 0).all(axis=1) & (~is_gated_clean) & (p_clean <= tau_rescue)
    if still_all_zero.any():
        cum4 = np.column_stack([p0[:, ci] + p1[:, k] + p2[:, k] for k, ci in enumerate(FULL_IDX)])
        # Normalize by class mean so rare defects can also win
        cum4_norm = cum4 / (cum4.mean(axis=0, keepdims=True) + 0.05)
        best_k = np.argmax(cum4_norm[still_all_zero], axis=1)
        idx_z = np.where(still_all_zero)[0]
        for ii, kk in zip(idx_z, best_k):
            pred[ii, FULL_IDX[kk]] = 1

    return pred


def optimize_two_stage_decoder_on_val(p0_va, p1_va, p2_va, p_clean_va, cln_cum_va, gva):
    """Optimize the Two-Stage Decoder strictly on `va` (500 images) to maximize Official Doc2 Total Score
    while simultaneously keeping Doc1 S_clean >= 0.98!
    Because there are only 22 continuous threshold parameters (and per-class F1 is nearly separable once the image-level Clean Gate is fixed),
    coordinate ascent on `va` takes < 2 seconds and cannot overfit like multi-model boolean clause search!
    """
    # Initial sensible defaults from HAND
    params = {
        'tau_clean': 0.52,
        'tau_cln_max': 0.42,
        'tau_p0_shield': 0.62,
        'tau_rescue': 0.22,
        'full': [
            [0.42, 0.25, 0.24, 0.85, 0.42],  # 模糊
            [0.37, 0.27, 0.25, 0.85, 0.42],  # 纹理损失
            [0.40, 0.23, 0.22, 0.85, 0.38],  # 过锐
            [0.37, 0.23, 0.19, 0.80, 0.34],  # 过度生成
        ],
        'bin': [0.42, 0.40],                 # 花屏, 虚化突兀
    }

    def obj_fn(prm):
        pr = decode_two_stage_doc2(p0_va, p1_va, p2_va, p_clean_va, cln_cum_va, prm)
        s = score_both(gva, pr)
        # Jointly maximize Doc2 Total + 0.25 * Doc1 Total (ensuring both Doc2 and Doc1 are top-tier!)
        return s['total_doc2'] + 0.25 * s['total']

    best_val = obj_fn(params)

    for _round in range(3):
        # 1. Optimize Image-Level Clean Gate parameters (tau_clean, tau_cln_max, tau_rescue, tau_p0_shield)
        for tc in np.linspace(0.32, 0.72, 21):
            for tcm in [0.34, 0.40, 0.46, 0.52, 0.60]:
                cand = {**params, 'tau_clean': float(tc), 'tau_cln_max': float(tcm)}
                v = obj_fn(cand)
                if v > best_val + 1e-5:
                    best_val = v
                    params = cand

        for trs in np.linspace(0.05, 0.40, 15):
            for tps in [0.48, 0.55, 0.62, 0.70, 0.78]:
                cand = {**params, 'tau_rescue': float(trs), 'tau_p0_shield': float(tps)}
                v = obj_fn(cand)
                if v > best_val + 1e-5:
                    best_val = v
                    params = cand

        # 2. Optimize per-class thresholds on FULL classes
        for k in range(4):
            # Step 2a: t0 (P0 threshold)
            for t0 in np.linspace(0.28, 0.52, 25):
                import copy
                cand = copy.deepcopy(params)
                cand['full'][k][0] = float(t0)
                v = obj_fn(cand)
                if v > best_val + 1e-5:
                    best_val = v
                    params = cand

            # Step 2b: (t2, t_cum) presence thresholds
            for t2 in np.linspace(0.14, 0.36, 23):
                for tcum in np.linspace(0.26, 0.54, 15):
                    cand = copy.deepcopy(params)
                    cand['full'][k][2] = float(t2)
                    cand['full'][k][4] = float(tcum)
                    v = obj_fn(cand)
                    if v > best_val + 1e-5:
                        best_val = v
                        params = cand

            # Step 2c: (t1, r12) P1 vs P2 boundary
            for t1 in np.linspace(0.15, 0.36, 22):
                for r12 in np.linspace(0.55, 1.25, 15):
                    cand = copy.deepcopy(params)
                    cand['full'][k][1] = float(t1)
                    cand['full'][k][3] = float(r12)
                    v = obj_fn(cand)
                    if v > best_val + 1e-5:
                        best_val = v
                        params = cand

        # 3. Optimize binary P0 classes
        for b_i in range(2):
            for t0 in np.linspace(0.28, 0.54, 27):
                import copy
                cand = copy.deepcopy(params)
                cand['bin'][b_i] = float(t0)
                v = obj_fn(cand)
                if v > best_val + 1e-5:
                    best_val = v
                    params = cand

    return params


def run_full_ablation():
    t0_all = time.time()
    assets = load_global_assets()
    all_names = assets['all_names']
    name_to_idx = assets['name_to_idx']
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    summary_rows = []

    for sp in (16, 17):
        print(f"\n=====================================================================================")
        print(f"🚀 RUNNING STRICT ZERO-LEAKAGE ABLATION ON SPLIT s{sp} (tr=2073, va=500, te=1000)")
        print(f"=====================================================================================")
        models, gva, gte = load_healthy_split_models(sp)
        ref = np.load(f'results/v3_ablate/A_base_s{sp}_probs.npz', allow_pickle=True)
        va_names = [str(x) for x in ref['va_names']]
        te_names = [str(x) for x in ref['te_names']]
        sp_csv = {r['name']: r['split'] for r in csv.DictReader(open(f'data/resplits/split_s{sp}.csv'))}
        tr_names = [n for n in all_names if sp_csv.get(n) == 'train']

        tr_idx = np.array([name_to_idx[n] for n in tr_names])
        va_idx = np.array([name_to_idx[n] for n in va_names])
        te_idx = np.array([name_to_idx[n] for n in te_names])
        src_va = assets['all_sources'][va_idx]

        # Verify 0% source overlap between tr, va, te
        s_tr = set(assets['all_sources'][tr_idx])
        s_va = set(assets['all_sources'][va_idx])
        s_te = set(assets['all_sources'][te_idx])
        assert len(s_tr & s_va) == 0 and len(s_tr & s_te) == 0 and len(s_va & s_te) == 0, "Source leakage!"
        print(f"   [Verified] 0.0% video source overlap across tr ({len(s_tr)} srcs), va ({len(s_va)} srcs), te ({len(s_te)} srcs).")

        # --- Baseline 0: Decoupled 14ch V8 Ensemble + HAND decoder ---
        p0_v8_va, p1_v8_va, p2_v8_va, cln0_v8_va, cln1_v8_va, cln2_v8_va = build_decoupled_14ch_probs(models, 'va')
        p0_v8_te, p1_v8_te, p2_v8_te, cln0_v8_te, cln1_v8_te, cln2_v8_te = build_decoupled_14ch_probs(models, 'te')
        res_b0_va = score_both(gva, decode_pc(p0_v8_va, p1_v8_va, p2_v8_va, HAND))
        res_b0_te = score_both(gte, decode_pc(p0_v8_te, p1_v8_te, p2_v8_te, HAND))

        # --- Baseline 1: V9 Enhanced 14ch Ensemble + HAND decoder ---
        p0_v9_va, p1_v9_va, p2_v9_va, _, _, _ = build_v9_enhanced_14ch_probs(models, 'va', deqa_res=None)
        p0_v9_te, p1_v9_te, p2_v9_te, _, _, _ = build_v9_enhanced_14ch_probs(models, 'te', deqa_res=None)
        res_b1_va = score_both(gva, decode_pc(p0_v9_va, p1_v9_va, p2_v9_va, HAND))
        res_b1_te = score_both(gte, decode_pc(p0_v9_te, p1_v9_te, p2_v9_te, HAND))

        # --- Scheme 1: Compute QualiRAG Reference Residuals + Train DeQAScore+QualiRAG 6-Tower on tr_idx ---
        q_feats = compute_qualirag_features(assets, tr_idx, va_idx, te_idx, dev)
        deqa_res = train_deqascore_qualirag_tower(assets, q_feats, tr_idx, va_idx, te_idx, sp, dev)
        sb_va, sb_te = compute_scene_model_bias_correction(assets, va_idx, te_idx, p0_v9_va, p1_v9_va, p2_v9_va, gva, dev)

        # --- Ablation A: Scheme 2 ONLY (Two-Stage All-Clean Gate + Cross-Defect Calibrator WITHOUT QualiRAG) ---
        X_va_noq, base_va_noq = build_rich_image_and_channel_features(models, 'va', va_idx, q_feats, deqa_res, sb_va, include_qualirag=False)
        X_te_noq, base_te_noq = build_rich_image_and_channel_features(models, 'te', te_idx, q_feats, deqa_res, sb_te, include_qualirag=False)
        s2_noq = fit_oof_and_predict_stage2(X_va_noq, gva, src_va, base_va_noq, X_te_noq, base_te_noq)
        prm_noq = optimize_two_stage_decoder_on_val(s2_noq['p0_va'], s2_noq['p1_va'], s2_noq['p2_va'], s2_noq['p_clean_va'], s2_noq['cln_cum_va'], gva)
        res_s2_va = score_both(gva, decode_two_stage_doc2(s2_noq['p0_va'], s2_noq['p1_va'], s2_noq['p2_va'], s2_noq['p_clean_va'], s2_noq['cln_cum_va'], prm_noq))
        res_s2_te = score_both(gte, decode_two_stage_doc2(s2_noq['p0_te'], s2_noq['p1_te'], s2_noq['p2_te'], s2_noq['p_clean_te'], s2_noq['cln_cum_te'], prm_noq))

        # --- Ablation B: Scheme 1 + Scheme 2 FULL (QualiRAG Scene Reference Residual + DeQAScore 6-Tower + Two-Stage Decoder) ---
        X_va_full, base_va_full = build_rich_image_and_channel_features(models, 'va', va_idx, q_feats, deqa_res, sb_va, include_qualirag=True)
        X_te_full, base_te_full = build_rich_image_and_channel_features(models, 'te', te_idx, q_feats, deqa_res, sb_te, include_qualirag=True)
        s12_full = fit_oof_and_predict_stage2(X_va_full, gva, src_va, base_va_full, X_te_full, base_te_full)
        # Also blend DeQAScore+QualiRAG 6-Tower's dedicated clean head into p_clean!
        p_cln_va_full = 0.80 * s12_full['p_clean_va'] + 0.20 * deqa_res['va_cln']
        p_cln_te_full = 0.80 * s12_full['p_clean_te'] + 0.20 * deqa_res['te_cln']
        prm_full = optimize_two_stage_decoder_on_val(s12_full['p0_va'], s12_full['p1_va'], s12_full['p2_va'], p_cln_va_full, s12_full['cln_cum_va'], gva)
        pred_va_full = decode_two_stage_doc2(s12_full['p0_va'], s12_full['p1_va'], s12_full['p2_va'], p_cln_va_full, s12_full['cln_cum_va'], prm_full)
        pred_te_full = decode_two_stage_doc2(s12_full['p0_te'], s12_full['p1_te'], s12_full['p2_te'], p_cln_te_full, s12_full['cln_cum_te'], prm_full)
        res_s12_va = score_both(gva, pred_va_full)
        res_s12_te = score_both(gte, pred_te_full)

        print(f"\n--- Split s{sp} Blind Test Set (1,000 Images, Zero Leakage) Results ---")
        for tag, r_va, r_te in [
            ("Baseline 0: V8 Decoupled 14ch (HAND)", res_b0_va, res_b0_te),
            ("Baseline 1: V9 Enhanced 14ch (HAND)",  res_b1_va, res_b1_te),
            ("Scheme 2 Only (Two-Stage Clean Gate + Stack)", res_s2_va, res_s2_te),
            ("Scheme 1 + Scheme 2 (QualiRAG + Two-Stage)",   res_s12_va, res_s12_te),
        ]:
            print(f"  {tag:46s} | [TEST 1000] Doc2={r_te['total_doc2']:5.2f} (S_pos={r_te['S_pos']:.3f}, S_p0={r_te['S_p0']:.3f}, Clean2={r_te['S_clean_doc2']:.3f}) | Doc1={r_te['total']:5.2f} (Clean1={r_te['S_clean']:.3f}) || [VAL OOF] Doc2={r_va['total_doc2']:5.2f}")

        print(f"\n  Per-Label F1 Breakdown on s{sp} Blind Test (Baseline 1 vs Scheme 1+2):")
        for (c, g, P0, R0, F0, _, _, _), (_, _, P1, R1, F1, _, _, _) in zip(res_b1_te['per'], res_s12_te['per']):
            diff = F1 - F0
            print(f"    {c}_{g:4s}: {F0:.3f} -> {F1:.3f} ({diff:+.3f})  [P: {P0:.3f}->{P1:.3f}, R: {R0:.3f}->{R1:.3f}]")

        summary_rows.append((sp, res_b0_te, res_b1_te, res_s2_te, res_s12_te))

    print("\n=====================================================================================")
    print("🏆 2-SPLIT MEAN (s16_test + s17_test = 2,000 Blind Evaluations) SUMMARY")
    print("=====================================================================================")
    for idx_m, tag in [
        (1, "Baseline 0: V8 Decoupled 14ch (HAND)"),
        (2, "Baseline 1: V9 Enhanced 14ch (HAND)"),
        (3, "Scheme 2 Only (Two-Stage Clean Gate + Stack)"),
        (4, "Scheme 1 + Scheme 2 (QualiRAG + Two-Stage)"),
    ]:
        r16 = summary_rows[0][idx_m]
        r17 = summary_rows[1][idx_m]
        m_doc2 = 0.5 * (r16['total_doc2'] + r17['total_doc2'])
        m_doc1 = 0.5 * (r16['total'] + r17['total'])
        m_pos  = 0.5 * (r16['S_pos'] + r17['S_pos'])
        m_p0   = 0.5 * (r16['S_p0'] + r17['S_p0'])
        m_cln2 = 0.5 * (r16['S_clean_doc2'] + r17['S_clean_doc2'])
        m_cln1 = 0.5 * (r16['S_clean'] + r17['S_clean'])
        print(f"  {tag:46s} | Mean Doc2 = {m_doc2:5.2f} (S_pos={m_pos:.3f}, S_p0={m_p0:.3f}, Clean2={m_cln2:.3f}) | Mean Doc1 = {m_doc1:5.2f} (Clean1={m_cln1:.3f})")


if __name__ == '__main__':
    run_full_ablation()
