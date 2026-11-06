"""Worker script for 8-GPU parallel training of 70k Full-Spectrum CORN Models.
Executed on an isolated GPU:
1. Trains Fold K on pure real data, validates on Val K -> saves OOF predictions.
2. Trains Full-Data Model on all 3,573 real images with unique seed -> infers on test_150.
3. Saves outputs to results/8gpu_parallel/result_gpu_{gpu}.npz.
"""

import os
import sys
import time
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL

FULL_IDX = [CLASSES.index(c) for c in FULL]
BIN_IDX = [CLASSES.index(c) for c in CLASSES if c not in FULL]


class FullSpectrumCORNTower(nn.Module):
    def __init__(self, d_hid=512, dropout=0.3):
        super().__init__()
        # Adapters for each of the 6 heterogeneous backbones + 88D physical
        self.ad_intern = nn.Sequential(nn.Linear(28800, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(dropout))
        self.ad_radio = nn.Sequential(nn.Linear(15360, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(dropout))
        self.ad_siglip = nn.Sequential(nn.Linear(6912, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout))
        self.ad_eupe = nn.Sequential(nn.Linear(6912, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout))
        self.ad_conv = nn.Sequential(nn.Linear(6144, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout))
        self.ad_pe = nn.Sequential(nn.Linear(6144, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(dropout))
        self.ad_phys = nn.Sequential(nn.Linear(88, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15))

        d_total_fused = d_hid * 2 + 384 * 4 + 128  # 1024 + 1536 + 128 = 2688
        self.fusion = nn.Sequential(
            nn.Linear(d_total_fused, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.heads_full = nn.ModuleList([nn.Linear(512, 3) for _ in range(4)])
        self.heads_bin = nn.ModuleList([nn.Linear(512, 1) for _ in range(2)])

    def forward(self, x_in, x_ra, x_si, x_eu, x_co, x_pe, x_ph):
        h = torch.cat([
            self.ad_intern(x_in),
            self.ad_radio(x_ra),
            self.ad_siglip(x_si),
            self.ad_eupe(x_eu),
            self.ad_conv(x_co),
            self.ad_pe(x_pe),
            self.ad_phys(x_ph),
        ], dim=-1)
        feat = self.fusion(h)
        return [h_f(feat) for h_f in self.heads_full], [h_b(feat) for h_b in self.heads_bin]

    def predict_probs(self, x_in, x_ra, x_si, x_eu, x_co, x_pe, x_ph):
        lf, lb = self.forward(x_in, x_ra, x_si, x_eu, x_co, x_pe, x_ph)
        B = x_in.shape[0]
        p0 = torch.zeros((B, 6), device=x_in.device)
        p1 = torch.zeros((B, 4), device=x_in.device)
        p2 = torch.zeros((B, 4), device=x_in.device)

        for j, ci in enumerate(FULL_IDX):
            f_exist = torch.sigmoid(lf[j][:, 0])
            f_mid = torch.sigmoid(lf[j][:, 1])
            f_p0 = torch.sigmoid(lf[j][:, 2])

            p0[:, ci] = f_exist * f_mid * f_p0
            p1[:, j] = f_exist * f_mid * (1.0 - f_p0)
            p2[:, j] = f_exist * (1.0 - f_mid)

        for j, ci in enumerate(BIN_IDX):
            p0[:, ci] = torch.sigmoid(lb[j][:, 0])

        return p0, p1, p2


def corn_loss_fn(logits_full, logits_bin, gold):
    device = gold.device
    B = gold.shape[0]
    is_clean = torch.all(gold == 0, dim=1)

    loss = 0.0
    for j, ci in enumerate(FULL_IDX):
        lf = logits_full[j]
        y = gold[:, ci]

        t_exist = (y >= 1).float()
        w_exist = torch.ones(B, device=device)
        w_exist[is_clean] = 2.5
        w_exist[y >= 1] = 1.5
        loss_exist = F.binary_cross_entropy_with_logits(lf[:, 0], t_exist, weight=w_exist)

        mask_pos = (y >= 1)
        if mask_pos.sum() > 0:
            loss_mid = F.binary_cross_entropy_with_logits(lf[mask_pos, 1], (y[mask_pos] >= 2).float())
        else:
            loss_mid = 0.0
        loss_mid_reg = 0.02 * F.binary_cross_entropy_with_logits(lf[~mask_pos, 1], torch.zeros((~mask_pos).sum(), device=device))

        mask_mid = (y >= 2)
        if mask_mid.sum() > 0:
            loss_p0 = F.binary_cross_entropy_with_logits(lf[mask_mid, 2], (y[mask_mid] == 3).float())
        else:
            loss_p0 = 0.0
        if is_clean.sum() > 0:
            loss_clean_p0 = 4.0 * F.binary_cross_entropy_with_logits(lf[is_clean, 2], torch.zeros(is_clean.sum(), device=device))
        else:
            loss_clean_p0 = 0.0

        f_exist = torch.sigmoid(lf[:, 0])
        f_mid = torch.sigmoid(lf[:, 1])
        f_p0 = torch.sigmoid(lf[:, 2])
        mu_hat = f_exist * (1.0 - f_mid) + 2.0 * f_exist * f_mid * (1.0 - f_p0) + 3.0 * f_exist * f_mid * f_p0
        y_f = y.float()

        diff_y = y_f.unsqueeze(1) - y_f.unsqueeze(0)
        diff_mu = mu_hat.unsqueeze(1) - mu_hat.unsqueeze(0)
        pair_mask = (diff_y > 0)
        if pair_mask.sum() > 0:
            loss_rank = torch.mean(F.softplus(-2.5 * (diff_mu[pair_mask] - 0.5 * diff_y[pair_mask])))
        else:
            loss_rank = 0.0

        loss += loss_exist + 1.2 * loss_mid + loss_mid_reg + 1.2 * loss_p0 + loss_clean_p0 + 0.3 * loss_rank

    for j, ci in enumerate(BIN_IDX):
        lb = logits_bin[j]
        y = gold[:, ci]
        w_bin = torch.ones(B, device=device)
        w_bin[is_clean] = 4.0
        w_bin[y == 3] = 4.0
        loss += 1.5 * F.binary_cross_entropy_with_logits(lb[:, 0], (y == 3).float(), weight=w_bin)

    return loss


def train_and_eval(model, train_loader, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(epochs):
        model.train()
        for b_in, b_ra, b_si, b_eu, b_co, b_pe, b_ph, b_y in train_loader:
            b_in, b_ra, b_si, b_eu, b_co, b_pe, b_ph, b_y = (
                b_in.to(device), b_ra.to(device), b_si.to(device), b_eu.to(device),
                b_co.to(device), b_pe.to(device), b_ph.to(device), b_y.to(device)
            )
            optimizer.zero_grad()
            lf, lb = model(b_in, b_ra, b_si, b_eu, b_co, b_pe, b_ph)
            loss = corn_loss_fn(lf, lb, b_y)
            loss.backward()
            optimizer.step()
        scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1.5e-3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--out_npz', type=str, required=True)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[GPU {args.gpu}] Initializing Fold {args.fold}, Seed {args.seed} on {device}...", flush=True)

    # 1. Load features
    models = ['internvit-6b', 'c-radio-v4-h', 'siglip2-so400m', 'eupe-vit-b', 'convnextv2-base', 'pe-core-l14']
    keys = ['internvit', 'cradio', 'siglip', 'eupe', 'convnext', 'pecore']
    c_dict = {}
    t_dict = {}
    for m, k in zip(models, keys):
        fc = np.load(f'feats/{m}_combined.npz')
        ft = np.load(f'feats/{m}_test150.npz')
        c_dict[k] = fc['X']
        t_dict[k] = ft['X']

    phys_c = np.load('feats/phys_88d_combined.npz')
    phys_t = np.load('feats/phys_88d_test150.npz')
    med = np.median(phys_c['X'], axis=0)
    iqr = np.percentile(phys_c['X'], 75, axis=0) - np.percentile(phys_c['X'], 25, axis=0)
    iqr[iqr < 1e-4] = 1.0
    c_dict['phys'] = np.clip((phys_c['X'] - med) / iqr, -10.0, 10.0)
    t_dict['phys'] = np.clip((phys_t['X'] - med) / iqr, -10.0, 10.0)

    c_names = list(phys_c['names'])
    t_names = list(phys_t['names'])

    # Labels
    manifest_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
    name_to_row = {r['name']: r for r in manifest_rows}
    gold_all = np.array([[int(name_to_row[n][c]) for c in CLASSES] for n in c_names])

    # 2. Split into Train & Val for Fold K
    split_path = f'data/v9_fulldata/split_f{args.fold}_real.csv'
    split_map = {r['name']: r['split'] for r in csv.DictReader(open(split_path))}

    train_indices = [i for i, n in enumerate(c_names) if split_map.get(n) == 'train']
    val_indices = [i for i, n in enumerate(c_names) if split_map.get(n) == 'val']

    print(f"[GPU {args.gpu}] Fold {args.fold}: Train={len(train_indices)}, Val={len(val_indices)}", flush=True)

    def make_dataset(indices):
        return TensorDataset(
            torch.from_numpy(c_dict['internvit'][indices]).float(),
            torch.from_numpy(c_dict['cradio'][indices]).float(),
            torch.from_numpy(c_dict['siglip'][indices]).float(),
            torch.from_numpy(c_dict['eupe'][indices]).float(),
            torch.from_numpy(c_dict['convnext'][indices]).float(),
            torch.from_numpy(c_dict['pecore'][indices]).float(),
            torch.from_numpy(c_dict['phys'][indices]).float(),
            torch.from_numpy(gold_all[indices]).long()
        )

    # Train Fold model
    ds_train = make_dataset(train_indices)
    loader_train = DataLoader(ds_train, batch_size=64, shuffle=True, drop_last=True)
    model_fold = FullSpectrumCORNTower(d_hid=512, dropout=args.dropout).to(device)

    train_and_eval(model_fold, loader_train, args.epochs, args.lr, device)

    # Predict on Validation Fold (OOF)
    model_fold.eval()
    val_tensors = [
        torch.from_numpy(c_dict['internvit'][val_indices]).float().to(device),
        torch.from_numpy(c_dict['cradio'][val_indices]).float().to(device),
        torch.from_numpy(c_dict['siglip'][val_indices]).float().to(device),
        torch.from_numpy(c_dict['eupe'][val_indices]).float().to(device),
        torch.from_numpy(c_dict['convnext'][val_indices]).float().to(device),
        torch.from_numpy(c_dict['pecore'][val_indices]).float().to(device),
        torch.from_numpy(c_dict['phys'][val_indices]).float().to(device),
    ]
    with torch.no_grad():
        oof_p0, oof_p1, oof_p2 = model_fold.predict_probs(*val_tensors)

    # 3. Train Full-Data Model
    print(f"[GPU {args.gpu}] Training Full-Data Model on all {len(c_names)} images...", flush=True)
    all_indices = list(range(len(c_names)))
    ds_all = make_dataset(all_indices)
    loader_all = DataLoader(ds_all, batch_size=64, shuffle=True, drop_last=True)
    model_full = FullSpectrumCORNTower(d_hid=512, dropout=args.dropout).to(device)

    train_and_eval(model_full, loader_all, args.epochs, args.lr, device)

    # Predict on test_150
    t150_tensors = [
        torch.from_numpy(t_dict['internvit']).float().to(device),
        torch.from_numpy(t_dict['cradio']).float().to(device),
        torch.from_numpy(t_dict['siglip']).float().to(device),
        torch.from_numpy(t_dict['eupe']).float().to(device),
        torch.from_numpy(t_dict['convnext']).float().to(device),
        torch.from_numpy(t_dict['pecore']).float().to(device),
        torch.from_numpy(t_dict['phys']).float().to(device),
    ]
    with torch.no_grad():
        t150_p0, t150_p1, t150_p2 = model_full.predict_probs(*t150_tensors)

    # Save results
    np.savez_compressed(
        args.out_npz,
        val_names=np.array([c_names[i] for i in val_indices]),
        val_gold=gold_all[val_indices],
        oof_p0=oof_p0.cpu().numpy(),
        oof_p1=oof_p1.cpu().numpy(),
        oof_p2=oof_p2.cpu().numpy(),
        t150_names=np.array(t_names),
        t150_p0=t150_p0.cpu().numpy(),
        t150_p1=t150_p1.cpu().numpy(),
        t150_p2=t150_p2.cpu().numpy()
    )
    print(f"[GPU {args.gpu}] Finished! Saved to {args.out_npz}", flush=True)


if __name__ == '__main__':
    main()
