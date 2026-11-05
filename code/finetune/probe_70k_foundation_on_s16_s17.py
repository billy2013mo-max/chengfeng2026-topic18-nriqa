"""Zero-Leakage Evaluation of Cached 70,400-Dim Multi-Foundation High-Res 5-Crop Features
(InternViT-6B 448x448 5-crop + C-RADIOv4-H 768x768 5-crop + PE-Core-L14 + EUPE-ViT-B + ConvNeXtV2-Base + SigLIP2-SO400M
+ Native Physical Calipers + Texture + Graphic Score + Resolution Metadata)
trained strictly on s16_train / s17_train, validated on s16_val / s17_val, and tested on s16_test / s17_test!
"""
import os, sys, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL, targets_from_grades
from train_v2 import exact_probs_from_dual, exact_targets_14, P0_IDX
from train_v5 import v5_loss_fn
import comp_metrics as cm

t0 = time.time()
man_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
man_map = {r['name']: r for r in man_rows}
all_names = [r['name'] for r in man_rows]
name_to_idx = {n: i for i, n in enumerate(all_names)}
all_grades = np.array([[int(man_map[n][c]) for c in CLASSES] for n in all_names], dtype=np.int64)

print("Loading cached _combined.npz foundation features for all 3,573 human-annotated images...")
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
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X = X / norms
    blocks.append(X)
    block_dims.append(X.shape[1])
    print(f"  Loaded {tag:12s}: {X.shape} ({time.time()-t0:.1f}s)")

tex_d = np.load('feats/texture.npz', allow_pickle=True)
tex_map = {str(n): i for i, n in enumerate(tex_d['names'])}
X_tex = np.stack([tex_d['X'][tex_map[n]] for n in all_names]).astype(np.float32)

from extract_native_phys_feats import extract_for_names
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
    X_res.append([np.log(w*h), w/h, mn/1080.0, mx/1920.0, float(mn < 720), float(mn >= 1080)])
X_res = np.array(X_res, dtype=np.float32)

X_aux_raw = np.nan_to_num(np.concatenate([X_tex, X_phys, X_hand, X_gs, X_res], axis=1), 0.0)
print(f"  Loaded aux calipers: {X_aux_raw.shape}, Total foundation dim = {sum(block_dims)}")


class MultiTowerFoundationHead(nn.Module):
    """Per-backbone projection + Aux Caliper Tower + Cross-Tower Gated Fusion + Per-Class Dual Head (Cum + Exact4)."""
    def __init__(self, block_dims, aux_dim, proj_dim=384, dropout=0.35):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d, proj_dim),
                nn.LayerNorm(proj_dim),
                nn.GELU(),
            ) for d in block_dims
        ])
        self.aux_proj = nn.Sequential(
            nn.Linear(aux_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        tot_d = len(block_dims) * proj_dim + 128
        # Direct linear skip from concatenated normalized blocks for fast linear probe signal
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
        self.gate = nn.Linear(tot_d, 1)

    def forward(self, xs, x_aux):
        zs = [proj(x) for proj, x in zip(self.projs, xs)]
        za = self.aux_proj(x_aux)
        h = torch.cat(zs + [za], dim=1)
        x_cat = torch.cat(xs, dim=1)
        lc = torch.cat([m(h) for m in self.head_cum], dim=1) + 0.5 * self.linear_skip_cum(x_cat)
        le = torch.cat([m(h) for m in self.head_ex], dim=1) + 0.5 * self.linear_skip_ex(x_cat)
        gt = self.gate(h).squeeze(1)
        return lc, le, gt


def train_and_eval_split(sp, seeds=(1, 2, 3, 4, 5)):
    ref = np.load(f'results/v2/v2_pcl_s{sp}_seed1_probs.npz', allow_pickle=True)
    va_names = [str(x) for x in ref['va_names']]
    te_names = [str(x) for x in ref['te_names']]
    sp_csv = {r['name']: r['split'] for r in csv.DictReader(open(f'data/resplits/split_s{sp}.csv'))}
    tr_names = [n for n in all_names if sp_csv.get(n) == 'train']

    tr_idx = [name_to_idx[n] for n in tr_names]
    va_idx = [name_to_idx[n] for n in va_names]
    te_idx = [name_to_idx[n] for n in te_names]

    # Fit aux standardization strictly on tr_idx
    mu_a = X_aux_raw[tr_idx].mean(axis=0, keepdims=True)
    sd_a = X_aux_raw[tr_idx].std(axis=0, keepdims=True) + 1e-6
    X_aux = np.clip((X_aux_raw - mu_a) / sd_a, -5.0, 5.0)

    dev = torch.device('cuda:0')
    xs_tr = [torch.from_numpy(b[tr_idx]).to(dev) for b in blocks]
    xs_va = [torch.from_numpy(b[va_idx]).to(dev) for b in blocks]
    xs_te = [torch.from_numpy(b[te_idx]).to(dev) for b in blocks]
    xa_tr = torch.from_numpy(X_aux[tr_idx]).to(dev)
    xa_va = torch.from_numpy(X_aux[va_idx]).to(dev)
    xa_te = torch.from_numpy(X_aux[te_idx]).to(dev)

    g_tr = torch.from_numpy(all_grades[tr_idx]).to(dev)
    g_va = all_grades[va_idx]
    g_te = all_grades[te_idx]
    assert np.array_equal(g_va, ref['gva']) and np.array_equal(g_te, ref['gte'])

    Y_tr = targets_from_grades(g_tr)
    pos = Y_tr.sum(0).clamp(min=1.0)
    neg = (len(Y_tr) - pos).clamp(min=1.0)
    pw = (neg / pos).clamp(min=1.0, max=5.5)
    bce_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    va_p0_ens, va_p1_ens, va_p2_ens = 0.0, 0.0, 0.0
    te_p0_ens, te_p1_ens, te_p2_ens = 0.0, 0.0, 0.0

    for seed in seeds:
        torch.manual_seed(seed + sp * 100)
        np.random.seed(seed + sp * 100)
        net = MultiTowerFoundationHead(block_dims, X_aux.shape[1], proj_dim=384, dropout=0.35).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=2e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=45, eta_min=2e-5)

        best_va_loss = 1e9
        best_va_probs = None
        best_te_probs = None
        bs = 128
        n_tr = len(tr_idx)

        for ep in range(1, 46):
            net.train()
            perm = torch.randperm(n_tr, device=dev)
            for i in range(0, n_tr, bs):
                idx = perm[i:i+bs]
                b_xs = [x[idx] for x in xs_tr]
                b_xa = xa_tr[idx]
                b_g = g_tr[idx]
                opt.zero_grad()
                lc, le, gt = net(b_xs, b_xa)
                loss = v5_loss_fn(lc, le, gt, b_g, bce_none, p0_clean_weight=4.2, channel_neg_weight=1.30, emd_weight=0.50, alpha_ex=0.55)
                loss.backward()
                opt.step()
            sched.step()

            if ep >= 15 and ep % 3 == 0:
                net.eval()
                with torch.no_grad():
                    lc_v, le_v, _ = net(xs_va, xa_va)
                    p0_v, p1_v, p2_v, _ = exact_probs_from_dual(lc_v, le_v, alpha_ex=0.55)
                    lc_t, le_t, _ = net(xs_te, xa_te)
                    p0_t, p1_t, p2_t, _ = exact_probs_from_dual(lc_t, le_t, alpha_ex=0.55)
                # SWA over epochs 24..45
                if ep >= 24:
                    if best_va_probs is None:
                        best_va_probs = [p0_v.cpu().numpy(), p1_v.cpu().numpy(), p2_v.cpu().numpy()]
                        best_te_probs = [p0_t.cpu().numpy(), p1_t.cpu().numpy(), p2_t.cpu().numpy()]
                        cnt_swa = 1
                    else:
                        best_va_probs[0] += p0_v.cpu().numpy()
                        best_va_probs[1] += p1_v.cpu().numpy()
                        best_va_probs[2] += p2_v.cpu().numpy()
                        best_te_probs[0] += p0_t.cpu().numpy()
                        best_te_probs[1] += p1_t.cpu().numpy()
                        best_te_probs[2] += p2_t.cpu().numpy()
                        cnt_swa += 1

        va_p0_ens += best_va_probs[0] / cnt_swa / len(seeds)
        va_p1_ens += best_va_probs[1] / cnt_swa / len(seeds)
        va_p2_ens += best_va_probs[2] / cnt_swa / len(seeds)
        te_p0_ens += best_te_probs[0] / cnt_swa / len(seeds)
        te_p1_ens += best_te_probs[1] / cnt_swa / len(seeds)
        te_p2_ens += best_te_probs[2] / cnt_swa / len(seeds)

    os.makedirs('results/v9_foundation', exist_ok=True)
    out_p = f'results/v9_foundation/v9_6tower_70k_s{sp}_probs.npz'
    np.savez_compressed(
        out_p,
        va_p0=va_p0_ens, va_p1=va_p1_ens, va_p2=va_p2_ens, gva=g_va,
        te_p0=te_p0_ens, te_p1=te_p1_ens, te_p2=te_p2_ens, gte=g_te,
        va_names=np.array(va_names), te_names=np.array(te_names)
    )
    from cv_decoder_calib import decode_pc, HAND
    sv = cm.score(g_va, decode_pc(va_p0_ens, va_p1_ens, va_p2_ens, HAND), 0.02)
    st = cm.score(g_te, decode_pc(te_p0_ens, te_p1_ens, te_p2_ens, HAND), 0.02)
    print(f"Split s{sp} Standalone V9_6Tower_70K (HAND decoder): VAL={sv['total']:.2f} (S_pos={sv['S_pos']:.3f}, S_p0={sv['S_p0']:.3f}, S_clean={sv['S_clean']:.3f}) | TEST={st['total']:.2f} (S_pos={st['S_pos']:.3f}, S_p0={st['S_p0']:.3f}, S_clean={st['S_clean']:.3f})")
    for x in st['per']:
        print(f"  s{sp} {x[0]}_{x[1]}: F1={x[4]:.3f} (P={x[2]:.3f}, R={x[3]:.3f})")

for sp in (16, 17):
    train_and_eval_split(sp)

print("\n=== Evaluating Full Pool + V9_6Tower_70K on Isolated s16 & s17 ===")
from build_v8_final_submission_and_eval import run_isolated_1000_eval
run_isolated_1000_eval()

