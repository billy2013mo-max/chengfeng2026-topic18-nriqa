"""Phase 2: Train Frozen-Backbone CORN Conditional Ordinal Neural Network Tower
with 88D Physical Calipers on 100% Pure Real Data.

Key Principles:
1. 100% Frozen Backbones: InternViT-6B (28800D) + C-RADIOv4-H (15360D) + SigLIP2-SO400M (6912D) + 88D Physical Calipers.
2. CORN (Conditional Ordinal Regression):
   - z_{c,1}: P(y_c >= 1) on all samples.
   - z_{c,2}: P(y_c >= 2 | y_c >= 1) strictly on non-zero defect samples (M_{c, >=1} = 1).
   - z_{c,3}: P(y_c = 3 | y_c >= 2) on P1/P0 samples (M_{c, >=2} = 1).
   - Guarantees rank-consistency: P(y_c >= 1) >= P(y_c >= 2) >= P(y_c = 3).
3. Evaluates across all 5 independent blind splits (s16 ~ s20).
4. Produces test_150 inference predictions for Phase 2 ensemble.
"""

import os, sys, time, csv, json, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, 'code/decision_eval')
from comp_metrics import FULL_IDX, BIN_IDX

CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
FULL = ['模糊', '纹理损失', '过锐', '过度生成']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_competition_metrics(gold, pred):
    """Accurately computes official competition score (Max 65 pts)."""
    n = len(gold)
    # 1. Macro F1 across 6 classes (25 pts)
    f1_list = []
    for ci in range(6):
        yg = gold[:, ci]
        yp = pred[:, ci]
        grades = [1, 2, 3] if ci in FULL_IDX else [3]
        f1s = []
        for g in grades:
            tp = np.sum((yg == g) & (yp == g))
            fp = np.sum((yg != g) & (yp == g))
            fn = np.sum((yg == g) & (yp != g))
            denom = 2 * tp + fp + fn
            f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
        f1_list.append(np.mean(f1s))
    S_pos = np.mean(f1_list)

    # 2. P0 score (20 pts)
    p0_recs, p0_f1s = [], []
    for ci in range(6):
        yg0 = (gold[:, ci] == 3)
        yp0 = (pred[:, ci] == 3)
        tp = np.sum(yg0 & yp0)
        fp = np.sum((~yg0) & yp0)
        fn = np.sum(yg0 & (~yp0))
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        denom = 2 * tp + fp + fn
        f1 = 2.0 * tp / denom if denom > 0 else 0.0
        p0_recs.append(rec)
        p0_f1s.append(f1)
    S_p0 = 0.6 * np.mean(p0_recs) + 0.4 * np.mean(p0_f1s)

    # 3. Clean score (20 pts)
    # Doc1 rule: P0 FPR on genuinely clean images <= 2%
    is_gold_clean = np.all(gold == 0, axis=1)
    has_p0_pred = np.any(pred == 3, axis=1)
    if np.sum(is_gold_clean) > 0:
        p0_fpr = np.mean(has_p0_pred[is_gold_clean])
        S_clean = max(0.0, 1.0 - p0_fpr / 0.02)
    else:
        S_clean = 1.0

    total = 25.0 * S_pos + 20.0 * S_p0 + 20.0 * S_clean
    return total, S_pos, S_p0, S_clean


class CORNAttentionTower(nn.Module):
    def __init__(self, d_intern=28800, d_radio=15360, d_siglip=6912, d_phys=88, d_hid=512, dropout=0.35):
        super().__init__()
        # Backbone projection adapters
        self.proj_intern = nn.Sequential(
            nn.Linear(d_intern, d_hid),
            nn.LayerNorm(d_hid),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_radio = nn.Sequential(
            nn.Linear(d_radio, d_hid),
            nn.LayerNorm(d_hid),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_siglip = nn.Sequential(
            nn.Linear(d_siglip, d_hid),
            nn.LayerNorm(d_hid),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_phys = nn.Sequential(
            nn.Linear(d_phys, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Cross-Modal Fusion MLP
        d_fused = d_hid * 3 + 128
        self.fusion = nn.Sequential(
            nn.Linear(d_fused, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # CORN Output Heads for 6 classes:
        # Full classes (4): (z_exist, z_mid, z_p0)
        # Binary classes (2): z_p0
        self.heads_full = nn.ModuleList([
            nn.Linear(512, 3) for _ in range(4)
        ])
        self.heads_bin = nn.ModuleList([
            nn.Linear(512, 1) for _ in range(2)
        ])

    def forward(self, x_intern, x_radio, x_siglip, x_phys):
        h_intern = self.proj_intern(x_intern)
        h_radio = self.proj_radio(x_radio)
        h_siglip = self.proj_siglip(x_siglip)
        h_phys = self.proj_phys(x_phys)

        fused = torch.cat([h_intern, h_radio, h_siglip, h_phys], dim=-1)
        feat = self.fusion(fused)

        logits_full = [h(feat) for h in self.heads_full]  # 4 x (B, 3)
        logits_bin = [h(feat) for h in self.heads_bin]    # 2 x (B, 1)
        return logits_full, logits_bin

    def predict_probs(self, x_intern, x_radio, x_siglip, x_phys):
        """Converts CORN conditional logits into grade probabilities (p0, p1, p2)."""
        logits_full, logits_bin = self.forward(x_intern, x_radio, x_siglip, x_phys)
        B = x_intern.shape[0]
        p0 = torch.zeros((B, 6), device=x_intern.device)
        p1 = torch.zeros((B, 4), device=x_intern.device)
        p2 = torch.zeros((B, 4), device=x_intern.device)

        for j, ci in enumerate(FULL_IDX):
            lf = logits_full[j]  # (B, 3): [z_exist, z_mid, z_p0]
            f_exist = torch.sigmoid(lf[:, 0])
            f_mid = torch.sigmoid(lf[:, 1])
            f_p0 = torch.sigmoid(lf[:, 2])

            p0[:, ci] = f_exist * f_mid * f_p0
            p1[:, j] = f_exist * f_mid * (1.0 - f_p0)
            p2[:, j] = f_exist * (1.0 - f_mid)

        for j, ci in enumerate(BIN_IDX):
            lb = logits_bin[j]  # (B, 1)
            p0[:, ci] = torch.sigmoid(lb[:, 0])

        return p0, p1, p2


def corn_loss_fn(logits_full, logits_bin, gold):
    """CORN Conditional Ordinal Loss with Thurstone ranking & clean shield penalty."""
    device = gold.device
    B = gold.shape[0]
    is_clean = torch.all(gold == 0, dim=1)  # All-clean images

    loss = 0.0

    # 1. Full classes (模糊, 纹理损失, 过锐, 过度生成)
    for j, ci in enumerate(FULL_IDX):
        lf = logits_full[j]  # (B, 3): [z_exist, z_mid, z_p0]
        y = gold[:, ci]      # 0, 1, 2, 3

        # Channel 1: Existence (y >= 1) on all samples
        t_exist = (y >= 1).float()
        w_exist = torch.ones(B, device=device)
        w_exist[is_clean] = 2.5  # Suppress false alarms on clean images
        w_exist[y >= 1] = 1.5    # Pos weight
        loss_exist = F.binary_cross_entropy_with_logits(lf[:, 0], t_exist, weight=w_exist)

        # Channel 2: Mid-severe vs mild (y >= 2 | y >= 1) strictly masked to positive samples!
        mask_pos = (y >= 1)
        if mask_pos.sum() > 0:
            t_mid = (y[mask_pos] >= 2).float()
            loss_mid = F.binary_cross_entropy_with_logits(lf[mask_pos, 1], t_mid)
        else:
            loss_mid = 0.0
        # Weak regularization on non-pos to prevent divergence
        loss_mid_reg = 0.02 * F.binary_cross_entropy_with_logits(lf[~mask_pos, 1], torch.zeros((~mask_pos).sum(), device=device))

        # Channel 3: P0 vs P1 (y == 3 | y >= 2) strictly masked to mid-severe samples!
        mask_mid = (y >= 2)
        if mask_mid.sum() > 0:
            t_p0 = (y[mask_mid] == 3).float()
            loss_p0 = F.binary_cross_entropy_with_logits(lf[mask_mid, 2], t_p0)
        else:
            loss_p0 = 0.0
        # Clean shield penalty: heavily penalize any P0 logit activation on clean images!
        if is_clean.sum() > 0:
            loss_clean_p0 = 4.0 * F.binary_cross_entropy_with_logits(lf[is_clean, 2], torch.zeros(is_clean.sum(), device=device))
        else:
            loss_clean_p0 = 0.0

        # Thurstone pairwise severity ranking loss within batch
        f_exist = torch.sigmoid(lf[:, 0])
        f_mid = torch.sigmoid(lf[:, 1])
        f_p0 = torch.sigmoid(lf[:, 2])
        mu_hat = f_exist * (1.0 - f_mid) + 2.0 * f_exist * f_mid * (1.0 - f_p0) + 3.0 * f_exist * f_mid * f_p0
        y_f = y.float()

        # Vectorized pairwise ranking:
        diff_y = y_f.unsqueeze(1) - y_f.unsqueeze(0)  # y_i - y_j
        diff_mu = mu_hat.unsqueeze(1) - mu_hat.unsqueeze(0)
        pair_mask = (diff_y > 0)
        if pair_mask.sum() > 0:
            loss_rank = torch.mean(F.softplus(-2.5 * (diff_mu[pair_mask] - 0.5 * diff_y[pair_mask])))
        else:
            loss_rank = 0.0

        loss += loss_exist + 1.2 * loss_mid + loss_mid_reg + 1.2 * loss_p0 + loss_clean_p0 + 0.3 * loss_rank

    # 2. Binary classes (花屏, 虚化突兀)
    for j, ci in enumerate(BIN_IDX):
        lb = logits_bin[j]  # (B, 1)
        y = gold[:, ci]     # 0 or 3
        t_bin = (y == 3).float()
        w_bin = torch.ones(B, device=device)
        w_bin[is_clean] = 4.0   # Critical: prevent P0 false alarms on clean images!
        w_bin[y == 3] = 4.0     # P0 pos weight
        loss_bin = F.binary_cross_entropy_with_logits(lb[:, 0], t_bin, weight=w_bin)
        loss += 1.5 * loss_bin

    return loss


def decode_roll_down(p0, p1, p2, t_any, t_roll, t_p0):
    """Two-stage Roll-Down Bayesian Decoder."""
    n = p0.shape[0]
    pred = np.zeros((n, 6), dtype=int)
    for ci in range(6):
        p0_c = p0[:, ci]
        if ci in FULL_IDX:
            j = FULL_IDX.index(ci)
            p1_c = p1[:, j]
            p2_c = p2[:, j]
            p_any = p0_c + p1_c + p2_c

            pass_any = (p_any >= t_any[ci])
            s1_roll = p1_c + 0.45 * p0_c
            is_p0 = pass_any & (p0_c >= t_p0[ci])
            is_p1 = pass_any & (~is_p0) & (s1_roll >= t_roll[ci])
            is_p2 = pass_any & (~is_p0) & (~is_p1)

            pred[is_p0, ci] = 3
            pred[is_p1, ci] = 2
            pred[is_p2, ci] = 1
        else:
            is_p0 = (p0_c >= t_p0[ci])
            pred[is_p0, ci] = 3
    return pred


def train_corn_on_split(split_name, feat_dict, manifest_rows, epochs=35, batch_size=64, lr=1.5e-3):
    split_csv = f'data/resplits/{split_name}.csv'
    name_to_split = {r['name']: r['split'] for r in csv.DictReader(open(split_csv))}
    name_to_row = {r['name']: r for r in manifest_rows}

    # Filter indices
    train_idx = [i for i, n in enumerate(feat_dict['names']) if name_to_split.get(n) == 'train']
    val_idx = [i for i, n in enumerate(feat_dict['names']) if name_to_split.get(n) == 'val']
    test_idx = [i for i, n in enumerate(feat_dict['names']) if name_to_split.get(n) == 'test']

    print(f"[{split_name}] Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Extract tensors
    gold_all = np.array([[int(name_to_row[n][c]) for c in CLASSES] for n in feat_dict['names']])

    x_intern = torch.from_numpy(feat_dict['internvit'][train_idx]).float()
    x_radio = torch.from_numpy(feat_dict['cradio'][train_idx]).float()
    x_siglip = torch.from_numpy(feat_dict['siglip'][train_idx]).float()
    x_phys = torch.from_numpy(feat_dict['phys'][train_idx]).float()
    y_train = torch.from_numpy(gold_all[train_idx]).long()

    train_ds = TensorDataset(x_intern, x_radio, x_siglip, x_phys, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Validation & Test inputs
    val_tensors = [
        torch.from_numpy(feat_dict['internvit'][val_idx]).float().to(DEVICE),
        torch.from_numpy(feat_dict['cradio'][val_idx]).float().to(DEVICE),
        torch.from_numpy(feat_dict['siglip'][val_idx]).float().to(DEVICE),
        torch.from_numpy(feat_dict['phys'][val_idx]).float().to(DEVICE),
    ]
    test_tensors = [
        torch.from_numpy(feat_dict['internvit'][test_idx]).float().to(DEVICE),
        torch.from_numpy(feat_dict['cradio'][test_idx]).float().to(DEVICE),
        torch.from_numpy(feat_dict['siglip'][test_idx]).float().to(DEVICE),
        torch.from_numpy(feat_dict['phys'][test_idx]).float().to(DEVICE),
    ]

    model = CORNAttentionTower().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_score = 0.0
    best_test_metrics = None
    best_test_probs = None

    thresh_any = [0.26, 0.28, 0.27, 0.20, 0.26, 0.20]
    thresh_roll = [0.28, 0.30, 0.30, 0.25, 0.28, 0.25]
    thresh_p0 = [0.35, 0.35, 0.38, 0.35, 0.38, 0.35]

    for epoch in range(epochs):
        model.train()
        tot_loss = 0.0
        for bi, (b_in, b_ra, b_si, b_ph, b_y) in enumerate(train_loader):
            b_in, b_ra, b_si, b_ph, b_y = (
                b_in.to(DEVICE), b_ra.to(DEVICE), b_si.to(DEVICE), b_ph.to(DEVICE), b_y.to(DEVICE)
            )
            optimizer.zero_grad()
            lf, lb = model(b_in, b_ra, b_si, b_ph)
            loss = corn_loss_fn(lf, lb, b_y)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                # Val evaluation
                vp0, vp1, vp2 = model.predict_probs(*val_tensors)
                vp0, vp1, vp2 = vp0.cpu().numpy(), vp1.cpu().numpy(), vp2.cpu().numpy()
                v_pred = decode_roll_down(vp0, vp1, vp2, thresh_any, thresh_roll, thresh_p0)
                v_tot, v_pos, v_sp0, v_cln = compute_competition_metrics(gold_all[val_idx], v_pred)

                # Test evaluation
                tp0, tp1, tp2 = model.predict_probs(*test_tensors)
                tp0, tp1, tp2 = tp0.cpu().numpy(), tp1.cpu().numpy(), tp2.cpu().numpy()
                t_pred = decode_roll_down(tp0, tp1, tp2, thresh_any, thresh_roll, thresh_p0)
                t_tot, t_pos, t_sp0, t_cln = compute_competition_metrics(gold_all[test_idx], t_pred)

                if v_tot > best_val_score:
                    best_val_score = v_tot
                    best_test_metrics = (t_tot, t_pos, t_sp0, t_cln)
                    best_test_probs = (tp0, tp1, tp2)

    print(f"[{split_name}] Best Val={best_val_score:.2f} -> Blind Test={best_test_metrics[0]:.2f} "
          f"(Pos={best_test_metrics[1]*25:.2f}, P0={best_test_metrics[2]*20:.2f}, Clean={best_test_metrics[3]*20:.2f})")
    return best_test_metrics, best_test_probs


def train_full_corn_ensemble(feat_dict, test150_dict, manifest_rows, num_seeds=5, epochs=35, batch_size=64, lr=1.5e-3):
    """Trains 5 independent full-data CORN models across 3,573 real images and infers on test_150."""
    print(f"\n=======================================================")
    print(f"   Training 5-Seed CORN Ensemble on All 3,573 Real Images   ")
    print(f"=======================================================")

    name_to_row = {r['name']: r for r in manifest_rows}
    gold_all = np.array([[int(name_to_row[n][c]) for c in CLASSES] for n in feat_dict['names']])

    x_intern = torch.from_numpy(feat_dict['internvit']).float()
    x_radio = torch.from_numpy(feat_dict['cradio']).float()
    x_siglip = torch.from_numpy(feat_dict['siglip']).float()
    x_phys = torch.from_numpy(feat_dict['phys']).float()
    y_all = torch.from_numpy(gold_all).long()

    train_ds = TensorDataset(x_intern, x_radio, x_siglip, x_phys, y_all)

    t150_intern = torch.from_numpy(test150_dict['internvit']).float().to(DEVICE)
    t150_radio = torch.from_numpy(test150_dict['cradio']).float().to(DEVICE)
    t150_siglip = torch.from_numpy(test150_dict['siglip']).float().to(DEVICE)
    t150_phys = torch.from_numpy(test150_dict['phys']).float().to(DEVICE)

    t150_p0_accum = np.zeros((150, 6))
    t150_p1_accum = np.zeros((150, 4))
    t150_p2_accum = np.zeros((150, 4))

    for seed in range(num_seeds):
        torch.manual_seed(42 + seed * 10)
        np.random.seed(42 + seed * 10)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        model = CORNAttentionTower().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        for epoch in range(epochs):
            model.train()
            for b_in, b_ra, b_si, b_ph, b_y in train_loader:
                b_in, b_ra, b_si, b_ph, b_y = (
                    b_in.to(DEVICE), b_ra.to(DEVICE), b_si.to(DEVICE), b_ph.to(DEVICE), b_y.to(DEVICE)
                )
                optimizer.zero_grad()
                lf, lb = model(b_in, b_ra, b_si, b_ph)
                loss = corn_loss_fn(lf, lb, b_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            s_p0, s_p1, s_p2 = model.predict_probs(t150_intern, t150_radio, t150_siglip, t150_phys)
            t150_p0_accum += s_p0.cpu().numpy() / num_seeds
            t150_p1_accum += s_p1.cpu().numpy() / num_seeds
            t150_p2_accum += s_p2.cpu().numpy() / num_seeds
            print(f"  [Seed {seed+1}/{num_seeds}] done. Mean P0={s_p0.mean().item():.4f}, P1={s_p1.mean().item():.4f}, P2={s_p2.mean().item():.4f}")

    return t150_p0_accum, t150_p1_accum, t150_p2_accum


def main():
    print("=== Loading All Multi-Scale Deep & 88D Physical Features ===", flush=True)
    t0 = time.time()
    
    # Combined (3,573 real images)
    intern_c = np.load('feats/internvit-6b_combined.npz')
    radio_c = np.load('feats/c-radio-v4-h_combined.npz')
    siglip_c = np.load('feats/siglip2-so400m_combined.npz')
    phys_c = np.load('feats/phys_88d_combined.npz')

    # Normalize 88D physical features with RobustScaler
    phys_c_X = phys_c['X']
    med = np.median(phys_c_X, axis=0)
    iqr = np.percentile(phys_c_X, 75, axis=0) - np.percentile(phys_c_X, 25, axis=0)
    iqr[iqr < 1e-4] = 1.0
    phys_c_norm = np.clip((phys_c_X - med) / iqr, -10.0, 10.0)

    # Test 150
    intern_t = np.load('feats/internvit-6b_test150.npz')
    radio_t = np.load('feats/c-radio-v4-h_test150.npz')
    siglip_t = np.load('feats/siglip2-so400m_test150.npz')
    phys_t = np.load('feats/phys_88d_test150.npz')
    phys_t_norm = np.clip((phys_t['X'] - med) / iqr, -10.0, 10.0)

    feat_dict = {
        'names': list(intern_c['names']),
        'internvit': intern_c['X'],
        'cradio': radio_c['X'],
        'siglip': siglip_c['X'],
        'phys': phys_c_norm,
    }
    test150_dict = {
        'names': list(intern_t['names']),
        'internvit': intern_t['X'],
        'cradio': radio_t['X'],
        'siglip': siglip_t['X'],
        'phys': phys_t_norm,
    }
    manifest_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
    print(f"Features loaded in {time.time()-t0:.1f}s.")

    # 1. 5-Split CV Validation
    print("\n=== Validating CORN Attention Tower Across All 5 Splits (s16 ~ s20) ===", flush=True)
    all_metrics = []
    for s in ['split_s16', 'split_s17', 'split_s18', 'split_s19', 'split_s20']:
        m, _ = train_corn_on_split(s, feat_dict, manifest_rows, epochs=35)
        all_metrics.append(m)

    mean_tot = np.mean([m[0] for m in all_metrics])
    mean_pos = np.mean([m[1] * 25.0 for m in all_metrics])
    mean_p0 = np.mean([m[2] * 20.0 for m in all_metrics])
    mean_cln = np.mean([m[3] * 20.0 for m in all_metrics])
    print(f"\n=======================================================")
    print(f"5-Split Blind Test Mean Score: {mean_tot:.2f} / 65")
    print(f"  Pos:   {mean_pos:.2f} / 25")
    print(f"  P0:    {mean_p0:.2f} / 20")
    print(f"  Clean: {mean_cln:.2f} / 20")
    print(f"=======================================================")

    # 2. Train Full-Data CORN Ensemble and Infer on test_150
    t150_p0, t150_p1, t150_p2 = train_full_corn_ensemble(feat_dict, test150_dict, manifest_rows, num_seeds=5, epochs=35)

    # Save inferred probabilities
    out_corn_npz = 'results/corn_phase2/test_150_corn_probs.npz'
    os.makedirs(os.path.dirname(out_corn_npz), exist_ok=True)
    np.savez_compressed(out_corn_npz, p0=t150_p0, p1=t150_p1, p2=t150_p2, names=np.array(test150_dict['names']))
    print(f"\nSaved test_150 CORN probabilities to {out_corn_npz}!")

if __name__ == '__main__':
    main()
