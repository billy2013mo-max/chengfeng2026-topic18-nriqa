"""V14 Manifold-Synthesized 8-Foundation + 190D Physical Caliper + QualiRAG Residual Specialist Towers
(`code/finetune/train_v14_manifold_synth_tower.py`).

Implements the Training-Side Breakthrough from `docs/SYNTHETIC_DATA_BREAKTHROUGH_PLAN.md`:
1. 8 High-Resolution Foundation Feature Banks (86,400D total, each [u_global || u_local_mean || u_local_max]):
   - InternViT-6B-448px-V2.5 (28,800D)
   - C-RADIOv4-H-768px (15,360D)
   - PE-Core-G14-448px (9,216D)
   - SigLIP2-SO400M-384px (6,912D)
   - SigLIP2-NaFlex-Native (6,912D)
   - EUPE-ViT-B (6,912D)
   - ConvNeXtV2-Base (6,144D)
   - PE-Core-L14 (6,144D)
2. 190D Native Physical & Structural Calipers (88D phys_88d + 60D native_phys + 42D hand_features)
   + QualiRAG Scene-Matched Clean Reference Physical Residuals (delta_phys, z_phys)
   + Multi-Scale Leave-One-Source-Out k-NN Posteriors & Contrastive Prototype Margins (X_knn).
3. Training-Side Synthetic Manifold Augmentation + Two-Stage Sim2Real Curriculum:
   - Mechanism A: Latent Tile-Embedding Grafting (`u_local_max` CutMix) for localized defects.
   - Mechanism B: Scene-Matched Ordinal Manifold Mixup along same-scene neighbors (`cos > 0.65`).
   - Mechanism C: Two-Stage Sim2Real Annealing (Stage A: mixed manifold synthesis; Stage B: 100% pure real data annealing).
4. Three Architecturally Diverse Sub-Towers (saved both individually and as an ensemble):
   - `V14A_HybridCORN`: Local-Max Anomaly Gated Attention + CORN Conditional Ordinal + Thurstone Ranking.
   - `V14B_DeQAScore`: Continuous Gaussian Soft-Label KL + Dual Cumulative/Exact + Linear Skip Tower.
   - `V14C_SoftF1_CrossAttn`: Differentiable Competition Metric Surrogate (Soft-Macro-F1 + Soft-S_p0 + Clean Shield) Tower.
"""
import argparse, csv, json, os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL, targets_from_grades
from train_v2 import exact_probs_from_dual
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm

FULL_IDX = [0, 1, 2, 4]
BIN_IDX = [3, 5]


def load_all_8foundation_and_calipers():
    """Load all 8 foundation feature banks (86,400D) and 190D physical/structural calipers
    for both the 3,573 combined labeled dataset and the 150 test_150 dataset.
    """
    t0 = time.time()
    man_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
    man_map = {r['name']: r for r in man_rows}
    all_names = [r['name'] for r in man_rows]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    all_sources = np.array([r['source'] for r in man_rows])
    all_grades = np.array([[int(man_map[n][c]) for c in CLASSES] for n in all_names], dtype=np.int64)

    tpl = json.load(open('results/v2_full/template_150.json'))
    t150_names = [e['image'] for e in tpl]

    def load_comb_and_t150(comb_path, t150_path):
        dc = np.load(comb_path, allow_pickle=True)
        mc = {str(n): i for i, n in enumerate(dc['names'])}
        Xc = dc['X'][[mc[n] for n in all_names]].astype(np.float32)

        dt = np.load(t150_path, allow_pickle=True)
        mt = {str(n): i for i, n in enumerate(dt['names'])}
        Xt = dt['X'][[mt[n] for n in t150_names]].astype(np.float32)
        return Xc, Xt

    def load_p3_comb_and_t150(orig_path, supp_path, t150_path):
        do = np.load(orig_path, allow_pickle=True)
        ds = np.load(supp_path, allow_pickle=True)
        X_os = np.concatenate([do['X'], ds['X']], axis=0)
        nm_os = [str(n) for n in do['names']] + [str(n) for n in ds['names']]
        mos = {n: i for i, n in enumerate(nm_os)}
        Xc = X_os[[mos[n] for n in all_names]].astype(np.float32)

        dt = np.load(t150_path, allow_pickle=True)
        mt = {str(n): i for i, n in enumerate(dt['names'])}
        Xt = dt['X'][[mt[n] for n in t150_names]].astype(np.float32)
        return Xc, Xt

    specs = [
        ('internvit6b', lambda: load_comb_and_t150('feats/internvit-6b_combined.npz', 'feats/internvit-6b_test150.npz')),
        ('cradio768',   lambda: load_comb_and_t150('feats/c-radio-v4-h_combined.npz', 'feats/c-radio-v4-h_test150.npz')),
        ('pecoreg14',   lambda: load_p3_comb_and_t150('feats/p3/peg_orig.npz', 'feats/p3/peg_supp.npz', 'feats/p3/peg_test150.npz')),
        ('siglip2_so',  lambda: load_comb_and_t150('feats/siglip2-so400m_combined.npz', 'feats/siglip2-so400m_test150.npz')),
        ('siglip2_naf', lambda: load_p3_comb_and_t150('feats/p3/naflex_orig.npz', 'feats/p3/naflex_supp.npz', 'feats/p3/naflex_test150.npz')),
        ('eupe_vitb',   lambda: load_comb_and_t150('feats/eupe-vit-b_combined.npz', 'feats/eupe-vit-b_test150.npz')),
        ('convnextv2',  lambda: load_comb_and_t150('feats/convnextv2-base_combined.npz', 'feats/convnextv2-base_test150.npz')),
        ('pecorel14',   lambda: load_comb_and_t150('feats/pe-core-l14_combined.npz', 'feats/pe-core-l14_test150.npz')),
    ]

    blocks_c, blocks_t, block_dims = [], [], []
    for tag, fn in specs:
        Xc, Xt = fn()
        D3 = Xc.shape[1]
        d_slice = D3 // 3
        for s_i in range(3):
            sl = slice(s_i * d_slice, (s_i + 1) * d_slice)
            Xc[:, sl] = Xc[:, sl] / (np.linalg.norm(Xc[:, sl], axis=1, keepdims=True) + 1e-8)
            Xt[:, sl] = Xt[:, sl] / (np.linalg.norm(Xt[:, sl], axis=1, keepdims=True) + 1e-8)
        Xc = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-8)
        Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-8)
        blocks_c.append(Xc)
        blocks_t.append(Xt)
        block_dims.append(D3)

    # Load 190D Physical & Structural Calipers (88D phys_88d + 60D native_phys + 42D hand)
    p88_c, p88_t = load_comb_and_t150('feats/phys_88d_combined.npz', 'feats/phys_88d_test150.npz')

    npf = np.load('results/v8_experts/native_phys_feats.npz', allow_pickle=True)['data'].item()
    p60_c = np.stack([npf[n] for n in all_names]).astype(np.float32)
    p60_t = np.stack([npf[n] for n in t150_names]).astype(np.float32)

    hd = np.load('results/hand_features.npz', allow_pickle=True)
    hmap = {str(n): i for i, n in enumerate(hd['names'])}
    p42_c = np.stack([hd['X'][hmap[n]] for n in all_names]).astype(np.float32)
    p42_t = np.stack([hd['X'][hmap[n]] for n in t150_names]).astype(np.float32)

    X_aux_c = np.nan_to_num(np.concatenate([p88_c, p60_c, p42_c], axis=1), nan=0.0, posinf=10.0, neginf=-10.0)
    X_aux_t = np.nan_to_num(np.concatenate([p88_t, p60_t, p42_t], axis=1), nan=0.0, posinf=10.0, neginf=-10.0)

    print(f"[Assets] Loaded 8 foundation banks ({sum(block_dims)}D) + {X_aux_c.shape[1]}D calipers in {time.time()-t0:.1f}s", flush=True)
    return dict(
        all_names=all_names, name_to_idx=name_to_idx, all_sources=all_sources, all_grades=all_grades,
        t150_names=t150_names, blocks_c=blocks_c, blocks_t=blocks_t, block_dims=block_dims,
        X_aux_c=X_aux_c, X_aux_t=X_aux_t
    )


def compute_qualirag_and_neighbors(assets, tr_idx, dev):
    """Compute QualiRAG Scene-Matched Clean Reference Physical Residuals (delta_phys, z_phys)
    and Multi-Scale k-NN Posteriors (X_knn) for BOTH the 3,573 combined images and the 150 test_150 images,
    strictly querying `tr_idx` (with Leave-One-Source-Out masking on `tr_idx`)!
    Also returns same-scene neighbor indices `nbr_idx_tr` on `tr_idx` for Scene-Matched Ordinal Manifold Mixup!
    """
    t0 = time.time()
    blocks_c = assets['blocks_c']
    blocks_t = assets['blocks_t']
    X_aux_c_raw = assets['X_aux_c']
    X_aux_t_raw = assets['X_aux_t']
    all_grades = assets['all_grades']
    all_sources = assets['all_sources']

    med_a = np.median(X_aux_c_raw[tr_idx], axis=0, keepdims=True)
    iqr_a = (np.percentile(X_aux_c_raw[tr_idx], 75, axis=0, keepdims=True) -
             np.percentile(X_aux_c_raw[tr_idx], 25, axis=0, keepdims=True))
    sd_a = np.maximum(iqr_a / 1.349, X_aux_c_raw[tr_idx].std(axis=0, keepdims=True) + 1e-5)
    X_aux_c = np.clip((X_aux_c_raw - med_a) / sd_a, -6.0, 6.0).astype(np.float32)
    X_aux_t = np.clip((X_aux_t_raw - med_a) / sd_a, -6.0, 6.0).astype(np.float32)

    X_aux_all = np.concatenate([X_aux_c, X_aux_t], axis=0)
    N_c = len(all_grades)
    N_t = len(X_aux_t)
    N_tot = N_c + N_t

    g_tr = all_grades[tr_idx]
    src_tr = all_sources[tr_idx]

    same_src_c = (all_sources[:, None] == src_tr[None, :])
    same_src_t = np.zeros((N_t, len(tr_idx)), dtype=bool)
    same_src_all = torch.from_numpy(np.concatenate([same_src_c, same_src_t], axis=0)).to(dev)

    y_clean_tr = (g_tr == 0).all(axis=1).astype(np.float32)[:, None]
    y_p0_tr = (g_tr == 3).astype(np.float32)
    y_p1_tr = (g_tr[:, FULL_IDX] == 2).astype(np.float32)
    y_p2_tr = (g_tr[:, FULL_IDX] == 1).astype(np.float32)
    y_ge1_tr = (g_tr >= 1).astype(np.float32)
    y_ge2_tr = (g_tr[:, FULL_IDX] >= 2).astype(np.float32)
    y_sev_tr = g_tr.astype(np.float32) / 3.0
    Y_bank_t = torch.from_numpy(
        np.concatenate([y_clean_tr, y_p0_tr, y_p1_tr, y_p2_tr, y_ge1_tr, y_ge2_tr, y_sev_tr], axis=1)
    ).to(dev)

    sim_mats = []
    sim_global_mats = []
    for bc, bt in zip(blocks_c, blocks_t):
        B_all = torch.from_numpy(np.concatenate([bc, bt], axis=0)).to(dev)
        B_tr = B_all[tr_idx]
        S = B_all @ B_tr.T
        sim_mats.append(S)

        d_gl = bc.shape[1] // 3
        Bg_all = F.normalize(B_all[:, :d_gl], dim=1)
        Bg_tr = Bg_all[tr_idx]
        sim_global_mats.append(Bg_all @ Bg_tr.T)

    w_8 = [0.20, 0.18, 0.14, 0.12, 0.12, 0.08, 0.08, 0.08]
    S_comb = sum(w * S for w, S in zip(w_8, sim_mats))
    S_sem = sum(w * Sg for w, Sg in zip([0.16, 0.12, 0.18, 0.18, 0.14, 0.08, 0.04, 0.10], sim_global_mats))

    log_area = torch.from_numpy(X_aux_all[:, 0]).to(dev)
    aspect = torch.from_numpy(X_aux_all[:, 1]).to(dev)
    luma_std = torch.from_numpy(X_aux_all[:, 4]).to(dev)
    dom_pen = (0.04 * torch.abs(log_area[:, None] - log_area[tr_idx][None, :]) +
               0.03 * torch.clamp(torch.abs(aspect[:, None] - aspect[tr_idx][None, :]), max=2.0) +
               0.03 * torch.abs(luma_std[:, None] - luma_std[tr_idx][None, :]))

    S_scene_masked = (S_sem - dom_pen).masked_fill(same_src_all, -1e4)
    S_comb_masked = S_comb.masked_fill(same_src_all, -1e4)

    # 1. Scene-Matched Clean Reference Physical Residual
    clean_tr_mask = torch.from_numpy((g_tr == 0).all(axis=1)).to(dev)
    S_clean_only = S_scene_masked.masked_fill(~clean_tr_mask[None, :], -1e4)
    top_vals_c, top_idx_c = torch.topk(S_clean_only, k=20, dim=1)
    w_c = F.softmax(top_vals_c / 0.05, dim=1)

    X_aux_t_gpu = torch.from_numpy(X_aux_all).to(dev)
    X_aux_tr_gpu = X_aux_t_gpu[tr_idx]
    ref_feats = X_aux_tr_gpu[top_idx_c]
    mu_clean = (w_c.unsqueeze(-1) * ref_feats).sum(dim=1)
    var_clean = (w_c.unsqueeze(-1) * (ref_feats - mu_clean.unsqueeze(1)) ** 2).sum(dim=1)
    sd_clean = torch.sqrt(var_clean + 0.15)

    delta_phys = (X_aux_t_gpu - mu_clean).cpu().numpy()
    z_phys = ((X_aux_t_gpu - mu_clean) / sd_clean).cpu().numpy()

    z_abs = np.abs(z_phys)
    z_summary = np.column_stack([
        np.mean(z_abs, axis=1),
        np.max(z_abs, axis=1),
        np.percentile(z_abs, 90, axis=1),
        np.percentile(z_abs, 75, axis=1),
        np.mean(z_abs[:, :88], axis=1),
        np.max(z_abs[:, :88], axis=1),
        np.mean(z_abs[:, 88:148], axis=1),
        np.mean(z_abs[:, 148:], axis=1),
        top_vals_c[:, 0].cpu().numpy(),
        top_vals_c[:, :5].mean(dim=1).cpu().numpy(),
    ])

    # 2. Multi-Scale k-NN Posteriors (Global + 8 Individual Foundation Backbones)
    knn_list = []
    for kk, tau in [(12, 0.035), (30, 0.055), (75, 0.08)]:
        tv, ti = torch.topk(S_comb_masked, k=kk, dim=1)
        w = F.softmax(tv / tau, dim=1)
        post = (w.unsqueeze(-1) * Y_bank_t[ti]).sum(dim=1)
        knn_list.append(post.cpu().numpy())

    for S_b in sim_mats:
        S_bm = S_b.masked_fill(same_src_all, -1e4)
        tv, ti = torch.topk(S_bm, k=25, dim=1)
        w = F.softmax(tv / 0.05, dim=1)
        post_b = (w.unsqueeze(-1) * Y_bank_t[ti]).sum(dim=1)
        knn_list.append(post_b.cpu().numpy())

    # 3. Contrastive Prototype Margins
    sim_clean_top5 = top_vals_c[:, :5].mean(dim=1)
    margins = []
    for col in range(1, 21):
        pos_mask = (Y_bank_t[:, col] > 0.5)
        if pos_mask.sum() >= 5:
            S_pos = S_comb_masked.masked_fill(~pos_mask[None, :], -1e4)
            tv_p, _ = torch.topk(S_pos, k=5, dim=1)
            margins.append((tv_p.mean(dim=1) - sim_clean_top5).cpu().numpy())
        else:
            margins.append(np.zeros(N_tot, dtype=np.float32))
    margins = np.column_stack(margins)

    X_knn_raw = np.concatenate(knn_list + [margins, z_summary], axis=1).astype(np.float32)
    mu_k = X_knn_raw[tr_idx].mean(axis=0, keepdims=True)
    sd_k = X_knn_raw[tr_idx].std(axis=0, keepdims=True) + 1e-5
    X_knn = np.clip((X_knn_raw - mu_k) / sd_k, -5.0, 5.0).astype(np.float32)

    S_tr_scene = S_scene_masked[tr_idx]
    _, nbr_idx_tr = torch.topk(S_tr_scene, k=12, dim=1)

    print(f"   [QualiRAG] Computed delta_phys={delta_phys.shape}, X_knn={X_knn.shape} in {time.time()-t0:.1f}s", flush=True)
    return dict(
        X_aux_c=X_aux_c, X_aux_t=X_aux_t,
        delta_c=delta_phys[:N_c], delta_t=delta_phys[N_c:],
        z_c=z_phys[:N_c], z_t=z_phys[N_c:],
        knn_c=X_knn[:N_c], knn_t=X_knn[N_c:],
        knn_raw_c=X_knn_raw[:N_c], knn_raw_t=X_knn_raw[N_c:],
        nbr_idx_tr=nbr_idx_tr
    )


def apply_latent_grafting_and_manifold_mixup(xs_b, xp_b, xk_b, g_b, xs_tr, xp_tr, xk_tr, g_tr_float,
                                             nbr_idx_tr, batch_tr_indices, donor_all_idx, p_graft=0.28, p_mix=0.32):
    """100% Vectorized GPU Implementation of Level 1 of `docs/SYNTHETIC_DATA_BREAKTHROUGH_PLAN.md`:
    1. Latent Tile-Embedding Grafting (`u_local_max` CutMix)
    2. Scene-Matched Ordinal Manifold Mixup
    Zero Python loops over samples, zero CPU-GPU syncs!
    """
    B = g_b.shape[0]
    dev = g_b.device
    xs_out = [x.clone() for x in xs_b]
    xp_out = xp_b.clone()
    xk_out = xk_b.clone()
    g_out = g_b.float().clone()

    u_rand = torch.rand(B, device=dev)

    # --- A. Vectorized Latent Tile-Embedding Grafting (`u_local_max` CutMix) ---
    graft_mask = (u_rand < p_graft)
    m_g = torch.where(graft_mask)[0]
    n_g = m_g.shape[0]
    if n_g > 0 and donor_all_idx.shape[0] > 0:
        r_idx = torch.randint(0, donor_all_idx.shape[0], (n_g,), device=dev)
        don_rows = donor_all_idx[r_idx]  # (n_g,)
        alpha = 0.35 + 0.40 * torch.rand((n_g, 1), device=dev)  # (n_g, 1) in [0.35, 0.75]

        for b_i, x_mod in enumerate(xs_out):
            D3 = x_mod.shape[1]
            ds = D3 // 3
            x_rec = x_mod[m_g]            # (n_g, D3)
            x_don = xs_tr[b_i][don_rows]  # (n_g, D3)
            gl = x_rec[:, :ds]
            mn = F.normalize((1.0 - 0.25 * alpha) * x_rec[:, ds:2*ds] + 0.25 * alpha * x_don[:, ds:2*ds], dim=1)
            mx = F.normalize((1.0 - alpha) * x_rec[:, 2*ds:] + alpha * x_don[:, 2*ds:], dim=1)
            x_mod[m_g] = F.normalize(torch.cat([gl, mn, mx], dim=1), dim=1)

        xp_out[m_g] = (1.0 - 0.5 * alpha) * xp_out[m_g] + 0.5 * alpha * xp_tr[don_rows]
        xk_out[m_g] = (1.0 - 0.5 * alpha) * xk_out[m_g] + 0.5 * alpha * xk_tr[don_rows]

        # Graft severe defect grades from donor (where donor has grade >= 2)
        g_don = g_tr_float[don_rows]  # (n_g, 6)
        sev_mask = (g_don >= 2.0).float()
        g_grafted = torch.clamp((1.0 - alpha) * g_out[m_g] + alpha * g_don, 0.0, 3.0)
        g_out[m_g] = sev_mask * g_grafted + (1.0 - sev_mask) * g_out[m_g]

    # --- B. Vectorized Scene-Matched Ordinal Manifold Mixup ---
    mix_mask = (u_rand >= p_graft) & (u_rand < (p_graft + p_mix))
    m_indices = torch.where(mix_mask)[0]
    n_mix = m_indices.shape[0]
    if n_mix > 0:
        tr_rows = batch_tr_indices[m_indices]
        k_choice = torch.randint(0, nbr_idx_tr.shape[1], (n_mix,), device=dev)
        nbr_rows = nbr_idx_tr[tr_rows, k_choice]

        u1 = torch.rand((n_mix, 1), device=dev)
        u2 = torch.rand((n_mix, 1), device=dev)
        lam = torch.clamp(0.5 * (u1 + u2), 0.20, 0.80)

        for b_i, x_mod in enumerate(xs_out):
            x_nbr = xs_tr[b_i][nbr_rows]
            x_mod[m_indices] = F.normalize(lam * x_mod[m_indices] + (1.0 - lam) * x_nbr, dim=1)

        xp_out[m_indices] = lam * xp_out[m_indices] + (1.0 - lam) * xp_tr[nbr_rows]
        xk_out[m_indices] = lam * xk_out[m_indices] + (1.0 - lam) * xk_tr[nbr_rows]
        g_out[m_indices] = lam * g_out[m_indices] + (1.0 - lam) * g_tr_float[nbr_rows]

    return xs_out, xp_out, xk_out, g_out


# =================================================================================================
# SUB-TOWER 1: V14A_HybridCORN (Local-Max Anomaly Gated Attention + CORN + Thurstone Ranking)
# =================================================================================================
class LocalMaxAnomalyBlockAdapter(nn.Module):
    """Explicitly decomposes each 3D_b foundation block into:
    - u_global (D_b)
    - u_local_mean (D_b)
    - u_local_max (D_b)
    - u_anomaly_diff = u_local_max - u_local_mean (D_b) -> highlights localized artifacts!
    """
    def __init__(self, d3, d_out=256, dropout=0.35):
        super().__init__()
        self.ds = d3 // 3
        self.proj_gl = nn.Linear(self.ds, d_out // 2)
        self.proj_loc = nn.Linear(self.ds * 3, d_out // 2)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(d_out, d_out), nn.Sigmoid())

    def forward(self, x):
        ds = self.ds
        u_gl = x[:, :ds]
        u_mn = x[:, ds:2*ds]
        u_mx = x[:, 2*ds:]
        u_diff = u_mx - u_mn
        h = torch.cat([self.proj_gl(self.drop(u_gl)), self.proj_loc(self.drop(torch.cat([u_mn, u_mx, u_diff], dim=1)))], dim=1)
        h = F.gelu(self.norm(h))
        return h * self.gate(h)


class V14A_HybridCORNTower(nn.Module):
    def __init__(self, block_dims, phys3_dim, knn_dim, d_hid=256, dropout=0.35):
        super().__init__()
        self.adapters = nn.ModuleList([LocalMaxAnomalyBlockAdapter(d, d_hid, dropout) for d in block_dims])
        self.phys_proj = nn.Sequential(
            nn.Linear(phys3_dim, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.knn_proj = nn.Sequential(
            nn.Linear(knn_dim, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.GELU()
        )
        tot_d = len(block_dims) * d_hid + 128 + 128
        self.fusion = nn.Sequential(
            nn.Linear(tot_d, 768), nn.LayerNorm(768), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout)
        )
        self.heads_full = nn.ModuleList([nn.Linear(512, 3) for _ in range(4)])
        self.heads_bin = nn.ModuleList([nn.Linear(512, 1) for _ in range(2)])
        self.clean_head = nn.Linear(512, 1)

    def forward(self, xs, xp, xk):
        hs = [ad(x) for ad, x in zip(self.adapters, xs)]
        hp = self.phys_proj(xp)
        hk = self.knn_proj(xk)
        f = self.fusion(torch.cat(hs + [hp, hk], dim=1))
        return [h(f) for h in self.heads_full], [h(f) for h in self.heads_bin], self.clean_head(f).squeeze(1)

    def predict_probs(self, xs, xp, xk):
        lf, lb, l_cln = self.forward(xs, xp, xk)
        B = xs[0].shape[0]
        dev = xs[0].device
        p0 = torch.zeros((B, 6), device=dev)
        p1 = torch.zeros((B, 4), device=dev)
        p2 = torch.zeros((B, 4), device=dev)
        for j, ci in enumerate(FULL_IDX):
            fe = torch.sigmoid(lf[j][:, 0])
            fm = torch.sigmoid(lf[j][:, 1])
            fp = torch.sigmoid(lf[j][:, 2])
            p0[:, ci] = fe * fm * fp
            p1[:, j] = fe * fm * (1.0 - fp)
            p2[:, j] = fe * (1.0 - fm)
        for j, ci in enumerate(BIN_IDX):
            p0[:, ci] = torch.sigmoid(lb[j][:, 0])
        return p0, p1, p2, torch.sigmoid(l_cln)


def hybrid_corn_continuous_loss(lf, lb, l_cln, g_cont):
    """Continuous-grade-aware CORN Conditional Ordinal Loss + Thurstone Pairwise Ranking + Clean Shield."""
    dev = g_cont.device
    B = g_cont.shape[0]
    is_clean = (g_cont < 0.15).all(dim=1)
    loss = 0.0

    for j, ci in enumerate(FULL_IDX):
        logits = lf[j]
        y = g_cont[:, ci]  # continuous in [0, 3]

        # 1. Existence (y >= 0.5) with continuous soft target clamp(y, 0, 1)
        t_exist = torch.clamp(y, 0.0, 1.0)
        w_exist = torch.ones(B, device=dev)
        w_exist[is_clean] = 2.2
        w_exist[y >= 0.8] = 1.5
        l_exist = F.binary_cross_entropy_with_logits(logits[:, 0], t_exist, weight=w_exist)

        # 2. Mid-severe conditional (y >= 1.5 | y >= 0.5)
        mask_pos = (y >= 0.5)
        if mask_pos.sum() > 0:
            t_mid = torch.clamp(y[mask_pos] - 1.0, 0.0, 1.0)
            l_mid = F.binary_cross_entropy_with_logits(logits[mask_pos, 1], t_mid)
        else:
            l_mid = 0.0
        l_mid_reg = 0.02 * F.binary_cross_entropy_with_logits(
            logits[~mask_pos, 1], torch.zeros((~mask_pos).sum(), device=dev)
        ) if (~mask_pos).sum() > 0 else 0.0

        # 3. P0 conditional (y >= 2.5 | y >= 1.5)
        mask_mid = (y >= 1.5)
        if mask_mid.sum() > 0:
            t_p0 = torch.clamp(y[mask_mid] - 2.0, 0.0, 1.0)
            l_p0 = F.binary_cross_entropy_with_logits(logits[mask_mid, 2], t_p0)
        else:
            l_p0 = 0.0

        l_cln_p0 = 3.5 * F.binary_cross_entropy_with_logits(
            logits[is_clean, 2], torch.zeros(is_clean.sum(), device=dev)
        ) if is_clean.sum() > 0 else 0.0

        # 4. Thurstone pairwise severity ranking loss
        fe, fm, fp = torch.sigmoid(logits[:, 0]), torch.sigmoid(logits[:, 1]), torch.sigmoid(logits[:, 2])
        mu_hat = fe * (1.0 - fm) + 2.0 * fe * fm * (1.0 - fp) + 3.0 * fe * fm * fp
        diff_y = y.unsqueeze(1) - y.unsqueeze(0)
        diff_mu = mu_hat.unsqueeze(1) - mu_hat.unsqueeze(0)
        pmask = (diff_y >= 0.75)
        l_rank = torch.mean(F.softplus(-2.5 * (diff_mu[pmask] - 0.45 * diff_y[pmask]))) if pmask.sum() > 0 else 0.0

        loss += l_exist + 1.25 * l_mid + l_mid_reg + 1.25 * l_p0 + l_cln_p0 + 0.30 * l_rank

    for j, ci in enumerate(BIN_IDX):
        y = g_cont[:, ci]
        t_bin = torch.clamp(y / 3.0, 0.0, 1.0)
        w_bin = torch.ones(B, device=dev)
        w_bin[is_clean] = 3.5
        w_bin[y >= 1.5] = 3.8
        loss += 1.5 * F.binary_cross_entropy_with_logits(lb[j][:, 0], t_bin, weight=w_bin)

    t_clean = torch.clamp(1.0 - g_cont.max(dim=1)[0], 0.0, 1.0)
    loss += 0.65 * F.binary_cross_entropy_with_logits(l_cln, t_clean)
    return loss


# =================================================================================================
# SUB-TOWER 2: V14B_DeQAScore (Continuous Gaussian Soft-Label KL + Dual Cumulative/Exact + Skip)
# =================================================================================================
class V14B_DeQAScoreTower(nn.Module):
    def __init__(self, block_dims, phys3_dim, knn_dim, proj_dim=288, dropout=0.35):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
            for d in block_dims
        ])
        self.phys_proj = nn.Sequential(
            nn.Linear(phys3_dim, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.knn_proj = nn.Sequential(
            nn.Linear(knn_dim, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.GELU()
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
        self.clean_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(tot_d, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, xs, xp, xk):
        zs = [p(x) for p, x in zip(self.projs, xs)]
        h = torch.cat(zs + [self.phys_proj(xp), self.knn_proj(xk)], dim=1)
        x_cat = torch.cat(xs, dim=1)
        lc = torch.cat([m(h) for m in self.head_cum], dim=1) + 0.42 * self.linear_skip_cum(x_cat)
        le = torch.cat([m(h) for m in self.head_ex], dim=1) + 0.42 * self.linear_skip_ex(x_cat)
        l_cln = self.clean_head(h).squeeze(1)
        return lc, le, l_cln

    def predict_probs(self, xs, xp, xk):
        lc, le, l_cln = self.forward(xs, xp, xk)
        p0, p1, p2, _ = exact_probs_from_dual(lc, le, alpha_ex=0.55)
        return p0, p1, p2, torch.sigmoid(l_cln)


def deqascore_continuous_loss(lc, le, l_cln, g_cont, sigma=0.46):
    """Continuous-grade-compatible DeQAScore Soft-Label KL + Cumulative Ordinal + EMD Loss."""
    dev = g_cont.device
    B = g_cont.shape[0]
    grid4 = torch.arange(4, device=dev, dtype=torch.float32)[None, :]
    is_clean = (g_cont < 0.15).all(dim=1)

    soft_loss = 0.0
    cum_loss = 0.0
    emd_loss = 0.0
    off_e = 0
    off_c = 0

    for ci, c in enumerate(CLASSES):
        y = g_cont[:, ci]
        if c in FULL:
            logits_e = le[:, off_e:off_e + 4]
            logits_c = lc[:, off_c:off_c + 3]
            off_e += 4
            off_c += 3

            # Continuous Gaussian soft label centered at y in [0, 3]
            w_soft = torch.exp(-0.5 * ((grid4 - y[:, None]) / sigma) ** 2)
            zero_m = (y < 0.10)
            if zero_m.any():
                w_soft[zero_m] = torch.tensor([0.95, 0.05, 0.0, 0.0], device=dev)
            w_soft = w_soft / w_soft.sum(dim=1, keepdim=True)

            log_p = F.log_softmax(logits_e, dim=1)
            soft_loss += -(w_soft * log_p).sum(dim=1).mean()

            # EMD (Earth Mover's Distance) on cumulative distribution
            p_ex = torch.softmax(logits_e, dim=1)
            cdf_pred = torch.cumsum(p_ex, dim=1)[:, :3]
            cdf_true = torch.cumsum(w_soft, dim=1)[:, :3]
            emd_loss += ((cdf_pred - cdf_true) ** 2).mean()

            # Cumulative targets: y >= 1, y >= 2, y == 3
            t_c1 = torch.clamp(y, 0.0, 1.0)
            t_c2 = torch.clamp(y - 1.0, 0.0, 1.0)
            t_c3 = torch.clamp(y - 2.0, 0.0, 1.0)
            t_cum = torch.stack([t_c1, t_c2, t_c3], dim=1)
            w_c = torch.ones_like(t_cum)
            w_c[is_clean, 2] = 4.0
            w_c[:, 0] = torch.where(y >= 0.5, 1.6, 1.1)
            w_c[:, 1] = torch.where(y >= 1.5, 2.0, 1.0)
            w_c[:, 2] = torch.where(y >= 2.5, 2.5, 1.0)
            cum_loss += F.binary_cross_entropy_with_logits(logits_c, t_cum, weight=w_c)
        else:
            logits_e = le[:, off_e:off_e + 2]
            logits_c = lc[:, off_c:off_c + 1]
            off_e += 2
            off_c += 1

            t_bin = torch.clamp(y / 3.0, 0.0, 1.0)
            w_soft2 = torch.stack([(1.0 - t_bin) * 0.97 + 0.015, t_bin * 0.97 + 0.015], dim=1)
            log_p2 = F.log_softmax(logits_e, dim=1)
            soft_loss += -(w_soft2 * log_p2).sum(dim=1).mean()

            w_b = torch.ones(B, device=dev)
            w_b[is_clean] = 4.0
            w_b[y >= 1.5] = 4.0
            cum_loss += 1.4 * F.binary_cross_entropy_with_logits(logits_c[:, 0], t_bin, weight=w_b)

    t_clean = torch.clamp(1.0 - g_cont.max(dim=1)[0], 0.0, 1.0)
    clean_l = F.binary_cross_entropy_with_logits(l_cln, t_clean)
    return cum_loss + 0.55 * soft_loss + 0.65 * emd_loss + 0.65 * clean_l


# =================================================================================================
# SUB-TOWER 3: V14C_SoftF1_CrossAttn (Cross-Modal Attention + Direct Soft-Macro-F1 & Soft-S_p0 Loss)
# =================================================================================================
class V14C_SoftF1CrossAttnTower(nn.Module):
    """Uses Multi-Head Cross-Attention where Physical/QualiRAG/k-NN tokens query the 8 Foundation Tokens,
    and directly optimizes a differentiable surrogate of the 65-point competition objective!
    """
    def __init__(self, block_dims, phys3_dim, knn_dim, d_tok=256, dropout=0.35):
        super().__init__()
        self.tok_projs = nn.ModuleList([
            nn.Sequential(nn.Dropout(dropout), nn.Linear(d, d_tok), nn.LayerNorm(d_tok), nn.GELU())
            for d in block_dims
        ])
        self.phys_tok = nn.Sequential(nn.Linear(phys3_dim, d_tok), nn.LayerNorm(d_tok), nn.GELU())
        self.knn_tok = nn.Sequential(nn.Linear(knn_dim, d_tok), nn.LayerNorm(d_tok), nn.GELU())

        self.mha = nn.MultiheadAttention(embed_dim=d_tok, num_heads=4, dropout=0.20, batch_first=True)
        self.norm1 = nn.LayerNorm(d_tok)

        tot_d = (len(block_dims) + 2) * d_tok
        self.mlp = nn.Sequential(
            nn.Linear(tot_d, 640), nn.LayerNorm(640), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(640, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout)
        )
        # Output both 4-state softmax logits per FULL class (4x4=16) + 2 binary logits + 14 direct skip logits
        self.head_full4 = nn.ModuleList([nn.Linear(384, 4) for _ in range(4)])
        self.head_bin1 = nn.ModuleList([nn.Linear(384, 1) for _ in range(2)])
        self.clean_head = nn.Linear(384, 1)

    def forward(self, xs, xp, xk):
        toks = [p(x) for p, x in zip(self.tok_projs, xs)] + [self.phys_tok(xp), self.knn_tok(xk)]
        T = torch.stack(toks, dim=1)  # (B, 10, d_tok)
        attn_out, _ = self.mha(T, T, T)
        T = self.norm1(T + 0.5 * attn_out)
        h = self.mlp(T.reshape(T.shape[0], -1))
        lf4 = [m(h) for m in self.head_full4]
        lb1 = [m(h) for m in self.head_bin1]
        l_cln = self.clean_head(h).squeeze(1)
        return lf4, lb1, l_cln

    def predict_probs(self, xs, xp, xk):
        lf4, lb1, l_cln = self.forward(xs, xp, xk)
        B = xs[0].shape[0]
        dev = xs[0].device
        p0 = torch.zeros((B, 6), device=dev)
        p1 = torch.zeros((B, 4), device=dev)
        p2 = torch.zeros((B, 4), device=dev)
        for j, ci in enumerate(FULL_IDX):
            pr = torch.softmax(lf4[j], dim=1)  # [P(0), P(P2=1), P(P1=2), P(P0=3)]
            p2[:, j] = pr[:, 1]
            p1[:, j] = pr[:, 2]
            p0[:, ci] = pr[:, 3]
        for j, ci in enumerate(BIN_IDX):
            p0[:, ci] = torch.sigmoid(lb1[j][:, 0])
        return p0, p1, p2, torch.sigmoid(l_cln)


def soft_competition_surrogate_loss(lf4, lb1, l_cln, g_cont, sigma=0.45):
    """Differentiable Competition Metric Surrogate Loss:
    1. Soft Gaussian Ordinal Cross-Entropy on the 4-state distributions
    2. Direct Batch Soft-F1 Loss across all 14 exact channels (25-pt S_pos surrogate!)
    3. Direct Batch Soft-(0.6*R_p0 + 0.4*F1_p0) Loss across the 6 P0 channels (20-pt S_p0 surrogate!)
    4. Clean P0 False-Positive Shield Penalty (20-pt S_clean surrogate!)
    """
    dev = g_cont.device
    B = g_cont.shape[0]
    grid4 = torch.arange(4, device=dev, dtype=torch.float32)[None, :]
    is_clean = (g_cont < 0.15).all(dim=1)

    ce_loss = 0.0
    p0_list, p1_list, p2_list = [], [], []
    t0_list, t1_list, t2_list = [], [], []

    for j, ci in enumerate(FULL_IDX):
        y = g_cont[:, ci]
        w_soft = torch.exp(-0.5 * ((grid4 - y[:, None]) / sigma) ** 2)
        zero_m = (y < 0.10)
        if zero_m.any():
            w_soft[zero_m] = torch.tensor([0.95, 0.05, 0.0, 0.0], device=dev)
        w_soft = w_soft / w_soft.sum(dim=1, keepdim=True)

        log_p = F.log_softmax(lf4[j], dim=1)
        ce_loss += -(w_soft * log_p).sum(dim=1).mean()

        pr = torch.softmax(lf4[j], dim=1)
        p2_list.append(pr[:, 1])
        p1_list.append(pr[:, 2])
        p0_list.append((ci, pr[:, 3]))

        t2_list.append(w_soft[:, 1])
        t1_list.append(w_soft[:, 2])
        t0_list.append((ci, w_soft[:, 3]))

    for j, ci in enumerate(BIN_IDX):
        y = g_cont[:, ci]
        t_bin = torch.clamp(y / 3.0, 0.0, 1.0)
        w_b = torch.ones(B, device=dev)
        w_b[is_clean] = 4.0
        w_b[y >= 1.5] = 4.0
        ce_loss += F.binary_cross_entropy_with_logits(lb1[j][:, 0], t_bin, weight=w_b)
        pr_b = torch.sigmoid(lb1[j][:, 0])
        p0_list.append((ci, pr_b))
        t0_list.append((ci, t_bin))

    # Assemble 14 exact channels
    p0_dict = dict(p0_list)
    t0_dict = dict(t0_list)
    P0 = torch.stack([p0_dict[ci] for ci in range(6)], dim=1)
    T0 = torch.stack([t0_dict[ci] for ci in range(6)], dim=1)
    P1 = torch.stack(p1_list, dim=1)
    T1 = torch.stack(t1_list, dim=1)
    P2 = torch.stack(p2_list, dim=1)
    T2 = torch.stack(t2_list, dim=1)

    P14 = torch.cat([P0, P1, P2], dim=1)
    T14 = torch.cat([T0, T1, T2], dim=1)

    # Differentiable Soft Macro-F1 across 14 channels
    tp14 = (P14 * T14).sum(dim=0)
    fp14 = (P14 * (1.0 - T14)).sum(dim=0)
    fn14 = ((1.0 - P14) * T14).sum(dim=0)
    soft_f1_14 = (2.0 * tp14 + 1e-2) / (2.0 * tp14 + fp14 + fn14 + 1e-2)
    loss_s_pos = 1.0 - soft_f1_14.mean()

    # Differentiable Soft S_p0 (0.6 * R_p0 + 0.4 * F1_p0) across 6 P0 channels
    tp0 = (P0 * T0).sum(dim=0)
    fp0 = (P0 * (1.0 - T0)).sum(dim=0)
    fn0 = ((1.0 - P0) * T0).sum(dim=0)
    soft_rec0 = (tp0 + 1e-2) / (tp0 + fn0 + 1e-2)
    soft_f1_0 = (2.0 * tp0 + 1e-2) / (2.0 * tp0 + fp0 + fn0 + 1e-2)
    loss_s_p0 = 1.0 - (0.6 * soft_rec0.mean() + 0.4 * soft_f1_0.mean())

    # Clean P0 suppression penalty
    loss_clean_p0 = P0[is_clean].mean() * 3.0 if is_clean.any() else 0.0
    t_clean = torch.clamp(1.0 - g_cont.max(dim=1)[0], 0.0, 1.0)
    loss_cln_gate = F.binary_cross_entropy_with_logits(l_cln, t_clean)

    return ce_loss + 1.8 * loss_s_pos + 1.5 * loss_s_p0 + loss_clean_p0 + 0.6 * loss_cln_gate


# =================================================================================================
# TWO-STAGE SIM2REAL CURRICULUM TRAINER
# =================================================================================================
def build_donor_pools(g_tr_np, dev):
    """Build a single weighted 1D GPU index tensor of severe (P0 or P1) training samples."""
    idx_list = []
    for ci, repeat_w in [(4, 3), (2, 3), (5, 2), (3, 2), (0, 1), (1, 1)]:
        idx = np.where(g_tr_np[:, ci] >= 2)[0]
        if len(idx) > 0:
            idx_list.extend([idx] * repeat_w)
    if not idx_list:
        return torch.zeros(0, dtype=torch.long, device=dev)
    return torch.from_numpy(np.concatenate(idx_list)).long().to(dev)


def train_three_diverse_v14_towers(assets, q_data, tr_idx, va_idx, te_idx, dev, seeds=(1, 2, 3), tag='s16'):
    t0 = time.time()
    blocks_c = assets['blocks_c']
    blocks_t = assets['blocks_t']
    block_dims = assets['block_dims']
    all_grades = assets['all_grades']

    X_phys3_c = np.concatenate([q_data['X_aux_c'], q_data['delta_c'], q_data['z_c']], axis=1).astype(np.float32)
    X_phys3_t = np.concatenate([q_data['X_aux_t'], q_data['delta_t'], q_data['z_t']], axis=1).astype(np.float32)
    X_knn_c = q_data['knn_c']
    X_knn_t = q_data['knn_t']
    nbr_idx_tr = q_data['nbr_idx_tr']

    xs_tr = [torch.from_numpy(b[tr_idx]).to(dev) for b in blocks_c]
    xs_va = [torch.from_numpy(b[va_idx]).to(dev) for b in blocks_c]
    xs_te = [torch.from_numpy(b[te_idx]).to(dev) for b in blocks_c]
    xs_t150 = [torch.from_numpy(b).to(dev) for b in blocks_t]

    xp_tr = torch.from_numpy(X_phys3_c[tr_idx]).to(dev)
    xp_va = torch.from_numpy(X_phys3_c[va_idx]).to(dev)
    xp_te = torch.from_numpy(X_phys3_c[te_idx]).to(dev)
    xp_t150 = torch.from_numpy(X_phys3_t).to(dev)

    xk_tr = torch.from_numpy(X_knn_c[tr_idx]).to(dev)
    xk_va = torch.from_numpy(X_knn_c[va_idx]).to(dev)
    xk_te = torch.from_numpy(X_knn_c[te_idx]).to(dev)
    xk_t150 = torch.from_numpy(X_knn_t).to(dev)

    g_tr_np = all_grades[tr_idx]
    g_tr = torch.from_numpy(g_tr_np).to(dev)
    g_tr_float = g_tr.float()
    donor_pools = build_donor_pools(g_tr_np, dev)

    phys3_dim = X_phys3_c.shape[1]
    knn_dim = X_knn_c.shape[1]
    n_tr = len(tr_idx)
    bs = 128

    tower_specs = [
        ('v14a_corn',    lambda: V14A_HybridCORNTower(block_dims, phys3_dim, knn_dim).to(dev), hybrid_corn_continuous_loss),
        ('v14b_deqa',    lambda: V14B_DeQAScoreTower(block_dims, phys3_dim, knn_dim).to(dev),  deqascore_continuous_loss),
        ('v14c_softf1',  lambda: V14C_SoftF1CrossAttnTower(block_dims, phys3_dim, knn_dim).to(dev), soft_competition_surrogate_loss),
    ]

    results_by_tower = {}
    for t_name, build_fn, loss_fn in tower_specs:
        va_acc = [0.0, 0.0, 0.0, 0.0]
        te_acc = [0.0, 0.0, 0.0, 0.0]
        t150_acc = [0.0, 0.0, 0.0, 0.0]

        for s_i, seed in enumerate(seeds):
            torch.manual_seed(seed * 1000 + hash(t_name) % 997)
            np.random.seed(seed * 1000 + hash(t_name) % 997)
            net = build_fn()
            opt = torch.optim.AdamW(net.parameters(), lr=6.5e-4, weight_decay=2.5e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=45, eta_min=1.5e-5)

            swa_v = [0.0, 0.0, 0.0, 0.0]
            swa_t = [0.0, 0.0, 0.0, 0.0]
            swa_150 = [0.0, 0.0, 0.0, 0.0]
            cnt = 0

            for ep in range(1, 46):
                net.train()
                perm = torch.randperm(n_tr, device=dev)

                # Two-Stage Sim2Real Curriculum:
                # Stage A (ep 1..28): Latent Tile Grafting + Scene-Matched Manifold Mixup
                # Stage B (ep 29..45): 100% Pure Real Data Annealing (p_graft=0, p_mix=0)
                if ep <= 22:
                    p_g, p_m = 0.28, 0.32
                elif ep <= 28:
                    decay = (28 - ep) / 6.0
                    p_g, p_m = 0.28 * decay, 0.32 * decay
                else:
                    p_g, p_m = 0.0, 0.0

                for i in range(0, n_tr, bs):
                    b_idx = perm[i:i + bs]
                    if len(b_idx) < 8:
                        continue
                    xs_b = [x[b_idx] for x in xs_tr]
                    xp_b = xp_tr[b_idx]
                    xk_b = xk_tr[b_idx]
                    g_b = g_tr_float[b_idx]

                    if p_g > 0.0 or p_m > 0.0:
                        xs_b, xp_b, xk_b, g_b = apply_latent_grafting_and_manifold_mixup(
                            xs_b, xp_b, xk_b, g_b, xs_tr, xp_tr, xk_tr, g_tr_float,
                            nbr_idx_tr, b_idx, donor_pools, p_graft=p_g, p_mix=p_m
                        )

                    opt.zero_grad()
                    out = net(xs_b, xp_b, xk_b)
                    loss = loss_fn(*out, g_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                    opt.step()
                sched.step()

                # Collect SWA snapshots strictly during Stage B (pure real annealed phase, ep 30..45)
                if ep >= 30 and ep % 3 == 0:
                    net.eval()
                    with torch.no_grad():
                        pv = net.predict_probs(xs_va, xp_va, xk_va)
                        pt = net.predict_probs(xs_te, xp_te, xk_te)
                        p150 = net.predict_probs(xs_t150, xp_t150, xk_t150)
                    for k_i in range(4):
                        swa_v[k_i] += pv[k_i].cpu().numpy()
                        swa_t[k_i] += pt[k_i].cpu().numpy()
                        swa_150[k_i] += p150[k_i].cpu().numpy()
                    cnt += 1

            for k_i in range(4):
                va_acc[k_i] += swa_v[k_i] / cnt / len(seeds)
                te_acc[k_i] += swa_t[k_i] / cnt / len(seeds)
                t150_acc[k_i] += swa_150[k_i] / cnt / len(seeds)

        results_by_tower[t_name] = dict(
            va_p0=va_acc[0], va_p1=va_acc[1], va_p2=va_acc[2], va_gate=1.0 - va_acc[3],
            te_p0=te_acc[0], te_p1=te_acc[1], te_p2=te_acc[2], te_gate=1.0 - te_acc[3],
            t150_p0=t150_acc[0], t150_p1=t150_acc[1], t150_p2=t150_acc[2], t150_gate=1.0 - t150_acc[3],
        )

    # Also build the fused V14 ensemble (0.36 * V14A + 0.36 * V14B + 0.28 * V14C)
    w_t = {'v14a_corn': 0.36, 'v14b_deqa': 0.36, 'v14c_softf1': 0.28}
    ens = {}
    for key in ['va_p0', 'va_p1', 'va_p2', 'va_gate', 'te_p0', 'te_p1', 'te_p2', 'te_gate',
                't150_p0', 't150_p1', 't150_p2', 't150_gate']:
        ens[key] = sum(w * results_by_tower[tn][key] for tn, w in w_t.items())
    results_by_tower['v14_ens'] = ens

    print(f"   [{tag}] Trained 3x{len(seeds)} V14 Manifold-Synthesized Towers in {time.time()-t0:.1f}s", flush=True)
    return results_by_tower


def run_split_mode(split_name, seeds=(1, 2, 3, 4), assets=None, dev=None):
    if dev is None:
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if assets is None:
        assets = load_all_8foundation_and_calipers()
    all_names = assets['all_names']
    name_to_idx = assets['name_to_idx']
    all_grades = assets['all_grades']

    os.makedirs('results/v14_manifold_synth', exist_ok=True)

    if split_name == 'full':
        # For 'full', run 5-fold GroupKFold OOF over all 3,573 images + full-bank test_150 inference!
        # Dispatch the 5 folds across available GPUs in parallel!
        import concurrent.futures
        gkf = GroupKFold(n_splits=5)
        N_all = len(all_names)
        oof_by_tower = {
            tn: {
                'oof_p0': np.zeros((N_all, 6), dtype=np.float32),
                'oof_p1': np.zeros((N_all, 4), dtype=np.float32),
                'oof_p2': np.zeros((N_all, 4), dtype=np.float32),
                'oof_gate': np.zeros(N_all, dtype=np.float32),
                't150_p0': 0.0, 't150_p1': 0.0, 't150_p2': 0.0, 't150_gate': 0.0
            }
            for tn in ['v14a_corn', 'v14b_deqa', 'v14c_softf1', 'v14_ens']
        }

        folds = list(enumerate(gkf.split(np.arange(N_all), all_grades[:, 0], groups=assets['all_sources'])))
        n_gpus = max(1, torch.cuda.device_count())

        def _run_full_fold(fold_i, tr_idx, vl_idx):
            f_dev = torch.device(f'cuda:{(fold_i + 3) % n_gpus}' if torch.cuda.is_available() else 'cpu')
            q_data = compute_qualirag_and_neighbors(assets, tr_idx, f_dev)
            res = train_three_diverse_v14_towers(
                assets, q_data, tr_idx, vl_idx, vl_idx, f_dev, seeds=seeds, tag=f'full_f{fold_i}'
            )
            return fold_i, vl_idx, res

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(_run_full_fold, fi, tr_i, vl_i) for fi, (tr_i, vl_i) in folds]
            for fut in concurrent.futures.as_completed(futs):
                fold_i, vl_idx, res = fut.result()
                for tn, d in res.items():
                    oof_by_tower[tn]['oof_p0'][vl_idx] = d['va_p0']
                    oof_by_tower[tn]['oof_p1'][vl_idx] = d['va_p1']
                    oof_by_tower[tn]['oof_p2'][vl_idx] = d['va_p2']
                    oof_by_tower[tn]['oof_gate'][vl_idx] = d['va_gate']
                    oof_by_tower[tn]['t150_p0'] += d['t150_p0'] / 5.0
                    oof_by_tower[tn]['t150_p1'] += d['t150_p1'] / 5.0
                    oof_by_tower[tn]['t150_p2'] += d['t150_p2'] / 5.0
                    oof_by_tower[tn]['t150_gate'] += d['t150_gate'] / 5.0

        for tn, d in oof_by_tower.items():
            sc = cm.score(all_grades, decode_pc(d['oof_p0'], d['oof_p1'], d['oof_p2'], HAND), 0.02)
            print(f"[FULL 3573 OOF] {tn:12s} | HAND Total={sc['total']:5.2f} (S_pos={25*sc['S_pos']:.2f}, S_p0={20*sc['S_p0']:.2f}, S_clean={20*sc['S_clean']:.2f})", flush=True)
            out_path = f'results/v14_manifold_synth/{tn}_full_probs.npz'
            np.savez_compressed(
                out_path,
                oof_p0=d['oof_p0'], oof_p1=d['oof_p1'], oof_p2=d['oof_p2'], oof_gate=d['oof_gate'],
                all_grades=all_grades, all_names=np.array(all_names),
                t150_p0=d['t150_p0'], t150_p1=d['t150_p1'], t150_p2=d['t150_p2'], t150_gate=d['t150_gate'],
                names_150=np.array(assets['t150_names'])
            )
        return

    # Standard split s16..s20: match exact ordering of va_names and te_names in backbone_exp/d3add_{split}_seed1_probs.npz
    ref = np.load(f'results/backbone_exp/d3add_{split_name}_seed1_probs.npz', allow_pickle=True)
    va_names = [str(x) for x in ref['va_names']]
    te_names = [str(x) for x in ref['te_names']]
    sp_csv = {r['name']: r['split'] for r in csv.DictReader(open(f'data/resplits/split_{split_name}.csv'))}
    tr_names = [n for n in all_names if sp_csv.get(n) == 'train']

    tr_idx = np.array([name_to_idx[n] for n in tr_names])
    va_idx = np.array([name_to_idx[n] for n in va_names])
    te_idx = np.array([name_to_idx[n] for n in te_names])

    # Zero-leakage sanity check
    s_tr = set(assets['all_sources'][tr_idx])
    s_va = set(assets['all_sources'][va_idx])
    s_te = set(assets['all_sources'][te_idx])
    assert len(s_tr & s_va) == 0 and len(s_tr & s_te) == 0 and len(s_va & s_te) == 0, "Source overlap detected!"

    q_data = compute_qualirag_and_neighbors(assets, tr_idx, dev)
    # Save QualiRAG features for this split so Stage-2 Cross-Defect + QualiRAG Calibrator can also use them!
    np.savez_compressed(
        f'results/v14_manifold_synth/qualirag_{split_name}_feats.npz',
        va_aux=q_data['X_aux_c'][va_idx], te_aux=q_data['X_aux_c'][te_idx], t150_aux=q_data['X_aux_t'],
        va_delta=q_data['delta_c'][va_idx], te_delta=q_data['delta_c'][te_idx], t150_delta=q_data['delta_t'],
        va_z=q_data['z_c'][va_idx], te_z=q_data['z_c'][te_idx], t150_z=q_data['z_t'],
        va_knn=q_data['knn_raw_c'][va_idx], te_knn=q_data['knn_raw_c'][te_idx], t150_knn=q_data['knn_raw_t'],
        va_names=np.array(va_names), te_names=np.array(te_names), names_150=np.array(assets['t150_names'])
    )

    res = train_three_diverse_v14_towers(assets, q_data, tr_idx, va_idx, te_idx, dev, seeds=seeds, tag=split_name)
    gva = all_grades[va_idx]
    gte = all_grades[te_idx]

    for tn, d in res.items():
        sv = cm.score(gva, decode_pc(d['va_p0'], d['va_p1'], d['va_p2'], HAND), 0.02)
        st = cm.score(gte, decode_pc(d['te_p0'], d['te_p1'], d['te_p2'], HAND), 0.02)
        print(f"[{split_name}] {tn:12s} | VAL={sv['total']:5.2f} | TEST={st['total']:5.2f} (S_pos={25*st['S_pos']:.2f}, S_p0={20*st['S_p0']:.2f}, S_clean={20*st['S_clean']:.2f})", flush=True)
        out_path = f'results/v14_manifold_synth/{tn}_{split_name}_probs.npz'
        np.savez_compressed(
            out_path,
            va_p0=d['va_p0'], va_p1=d['va_p1'], va_p2=d['va_p2'], va_gate=d['va_gate'],
            gva=gva, va_names=np.array(va_names),
            te_p0=d['te_p0'], te_p1=d['te_p1'], te_p2=d['te_p2'], te_gate=d['te_gate'],
            gte=gte, te_names=np.array(te_names),
            t150_p0=d['t150_p0'], t150_p1=d['t150_p1'], t150_p2=d['t150_p2'], t150_gate=d['t150_gate'],
            names_150=np.array(assets['t150_names'])
        )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['s16', 's17', 's18', 's19', 's20', 'full', 'all'])
    args = ap.parse_args()
    if args.split == 'all':
        import concurrent.futures
        assets = load_all_8foundation_and_calipers()
        n_gpus = max(1, torch.cuda.device_count())
        splits_5 = ['s16', 's17', 's18', 's19', 's20']
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futs = [
                ex.submit(run_split_mode, sp, (1, 2, 3, 4), assets, torch.device(f'cuda:{i % n_gpus}'))
                for i, sp in enumerate(splits_5)
            ]
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
        run_split_mode('full', (1, 2, 3, 4), assets)
    else:
        run_split_mode(args.split)
