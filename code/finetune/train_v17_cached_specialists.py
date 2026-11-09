#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/train_v17_cached_specialists.py

High-speed training of V17 Two-Stage Sim2Real Specialists using 8-view cached features
from /dev/shm/v17_cache_{backbone}.npz.

Because the backbones are frozen and all 8-view features are pre-extracted in memory,
training a full 5-epoch specialist (with SWA and pure-real alignment) takes ~5-10 seconds!
"""
import os
import sys
import time
import copy
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import load_split, read_manifest, targets_from_grades, CLASSES, FULL
from train_v2 import exact_probs_from_dual, exact_targets_14, P0_IDX
from train_v9_specialist import set_seed, CLASS_TO_COLS14
from train_v16_microscope_specialist import v16_clean_microscope_loss_fn
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm
from extract_native_phys_feats import extract_for_names

OUT_DIR = 'results/v17_end2end_s16_s17'
SPLIT_DIR = 'data/v17_splits'
os.makedirs(OUT_DIR, exist_ok=True)


def is_synth_name(name: str) -> bool:
    if name.startswith(('v9_', 'v8_', 'v7_', 'v6_', 'v4_')):
        return True
    for d in range(6):
        if f'_d{d}_P' in name:
            return True
    return False


def get_sample_class_mask6_v17(name: str, synth_w: float = 0.75) -> np.ndarray:
    for d in range(6):
        if f'_d{d}_P' in name:
            m = np.zeros(7, dtype=np.float32)
            m[d] = synth_w
            return m
    if name.startswith('v9_og') or name.startswith('v8_og') or name.startswith('v7_'):
        return np.array([0.0, 0.0, 0.0, 0.0, synth_w, 0.0, 0.0], dtype=np.float32)
    if name.startswith('v9_sh') or name.startswith('v8_sh') or name.startswith('v4_'):
        return np.array([0.0, 0.0, synth_w, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if name.startswith('v9_bk') or name.startswith('v8_bk'):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, synth_w, 0.0], dtype=np.float32)
    if name.startswith('v9_bt') or name.startswith('v8_bt'):
        return np.array([synth_w, synth_w, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


class CachedDataset(Dataset):
    def __init__(self, rows, feat_dicts, bb_names, phys_map, phys_mu, phys_sd, synth_w=0.75):
        self.rows = rows
        self.feat_dicts = feat_dicts
        self.bb_names = bb_names
        self.phys_map = phys_map
        self.phys_mu = phys_mu
        self.phys_sd = phys_sd
        self.synth_w = synth_w

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        nm = r['name']
        feats = []
        for b in self.bb_names:
            if nm in self.feat_dicts[b]:
                feats.append(self.feat_dicts[b][nm])
            else:
                dim = self.feat_dicts[b][next(iter(self.feat_dicts[b]))].shape[-1]
                feats.append(np.zeros((8, dim), dtype=np.float32))
        g = np.array([int(r[c]) for c in CLASSES], dtype=np.int64)
        mask7 = get_sample_class_mask6_v17(nm, synth_w=self.synth_w)
        p_raw = self.phys_map.get(nm, np.zeros(60, dtype=np.float32))
        p_norm = np.clip((p_raw - self.phys_mu) / self.phys_sd, -6.0, 6.0).astype(np.float32)
        return feats, torch.from_numpy(p_norm), torch.from_numpy(g), torch.from_numpy(mask7), nm


def collate_cached(batch):
    # batch is list of (feats, p_norm, g, mask7, nm)
    num_bbs = len(batch[0][0])
    feats_by_bb = []
    for bi in range(num_bbs):
        bb_tensors = torch.stack([torch.from_numpy(item[0][bi]) for item in batch], dim=0) # (B, 8, dim)
        feats_by_bb.append(bb_tensors)
    phys = torch.stack([item[1] for item in batch], dim=0)
    g = torch.stack([item[2] for item in batch], dim=0)
    mask7 = torch.stack([item[3] for item in batch], dim=0)
    names = [item[4] for item in batch]
    return feats_by_bb, phys, g, mask7, names


class MultiNetCached(nn.Module):
    def __init__(self, bb_dims, phys_dim=60, dropout=0.12, n_quad=4):
        super().__init__()
        self.n_quad = n_quad
        self.bb_dims = bb_dims
        self.tile_attn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, max(64, dim // 8)),
                nn.GELU(),
                nn.Linear(max(64, dim // 8), 1)
            ) for dim in bb_dims
        ])
        d_vis = sum(8 * dim for dim in bb_dims)
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
        # f: (B, 8, dim)
        g_view = f[:, 0]
        tiles = f[:, 1:1 + self.n_quad]
        quad_mean = tiles.mean(1)
        nat_views = f[:, 1 + self.n_quad:]  # [p_peak1, p_peak2, p_mid]
        if nat_views.shape[1] >= 3:
            edge_max = nat_views[:, :2].amax(1)
            tex_view = nat_views[:, 2]
        else:
            edge_max = nat_views.amax(1)
            tex_view = nat_views.mean(1)
        r_edge_glob = edge_max - g_view
        r_edge_tex  = edge_max - tex_view
        t_max = tiles.amax(1)
        w = torch.softmax(attn_mod(tiles), dim=1)
        t_att = (tiles * w).sum(1)
        return torch.cat([g_view, quad_mean, edge_max, tex_view, r_edge_glob, r_edge_tex, t_max, t_att], dim=1)

    def forward(self, feats_by_bb, phys):
        out = []
        for f, attn_mod in zip(feats_by_bb, self.tile_attn):
            out.append(self._agg(f, attn_mod))
        p_emb = self.phys_mlp(phys.float())
        z_norm = self.norm(torch.cat(out + [p_emb], dim=1))
        fused = self.drop(z_norm * (0.5 + 0.5 * self.se_gate(z_norm)))
        lc = self.head_cum(fused)
        le = self.head_ex(fused)
        gl = self.gate(fused).squeeze(1)
        return lc, le, gl


@torch.no_grad()
def predict_cached(net, loader, dev, alpha_ex=0.58):
    net.eval()
    P0, P1, P2, GATE, G = [], [], [], [], []
    for feats_by_bb, phys, g, _, _ in loader:
        feats_dev = [f.to(dev, non_blocking=True).float() for f in feats_by_bb]
        phys = phys.to(dev, non_blocking=True).float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lc, le, gl = net(feats_dev, phys)
        pc0, pc1, pc2, _ = exact_probs_from_dual(lc.float(), le.float(), alpha_ex=alpha_ex)
        gt = torch.sigmoid(gl).view(-1).float()
        P0.append(pc0.cpu().numpy())
        P1.append(pc1.cpu().numpy())
        P2.append(pc2.cpu().numpy())
        GATE.append(gt.cpu().numpy())
        G.append(g.numpy())
    p0 = np.concatenate(P0, axis=0)
    p1 = np.concatenate(P1, axis=0)
    p2 = np.concatenate(P2, axis=0)
    gate = np.concatenate(GATE, axis=0)
    g_all = np.concatenate(G, axis=0)
    return p0, p1, p2, gate, g_all


def train_single_specialist(name, gpu_id, sp, focus_classes, bb_names, feat_dicts, phys_map,
                            ep_s1=3, ep_s2=2, batch=32, lr=1.2e-3, p0_clean_w=4.2, w_neg=1.05,
                            alpha_ex=0.58, synth_w=0.75, clean_rep=2, seed=42):
    dev = torch.device(f'cuda:{gpu_id}')
    set_seed(seed)
    focus_cis = [CLASSES.index(c.strip()) for c in focus_classes.split(',') if c.strip() in CLASSES]
    print(f'[{name}] Training on GPU {gpu_id} (Split s{sp}, focus={focus_classes}, backbones={bb_names})...', flush=True)

    split_file = f'{SPLIT_DIR}/split_s{sp}_v17.csv'
    man_file = f'{SPLIT_DIR}/manifest_s{sp}_v17.csv'
    rows = load_split(split_file)
    man = read_manifest(man_file)
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})

    tr_orig = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test']
    if not te: te = va

    tr_real = [r for r in tr_orig if not is_synth_name(r['name'])]
    phys_tr_mat = np.stack([phys_map[r['name']] for r in tr_real], axis=0)
    phys_mu = np.mean(phys_tr_mat, axis=0)
    phys_sd = np.std(phys_tr_mat, axis=0) + 1e-5

    Y_orig = targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr_real]))
    pw = ((len(Y_orig) - Y_orig.sum(0)) / Y_orig.sum(0).clamp(min=1)).clamp(1.0, 5.5).to(dev)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    rng = random.Random(seed)
    tr_synth_all = [
        r for r in tr_orig
        if is_synth_name(r['name'])
        and any(get_sample_class_mask6_v17(r['name'], synth_w)[ci] > 0.1 for ci in focus_cis)
    ]
    tr_synth = []
    for r in tr_synth_all:
        m6 = get_sample_class_mask6_v17(r['name'], synth_w)
        active_cis = [ci for ci in range(6) if m6[ci] > 0.1]
        max_g = max(int(r[CLASSES[ci]]) for ci in active_cis) if active_cis else 0
        if max_g >= 2 or rng.random() < 0.65:
            tr_synth.append(r)

    tr_stage1 = list(tr_real) + tr_synth
    rare_focus_cis = [ci for ci in focus_cis if ci in (1, 2, 3, 4, 5)]
    if rare_focus_cis:
        focus_real_p1_p0 = [r for r in tr_real if any(int(r[CLASSES[ci]]) >= 2 for ci in rare_focus_cis)]
        tr_stage1 = tr_stage1 + focus_real_p1_p0
    if clean_rep > 1:
        clean = [r for r in tr_real if all(int(r[c]) == 0 for c in CLASSES)]
        tr_stage1 = tr_stage1 + clean * (clean_rep - 1)

    tr_stage2 = list(tr_real)
    p1_p0_real_s2 = [r for r in tr_real if any(int(r[CLASSES[ci]]) >= 2 for ci in (1, 2, 3, 4, 5))]
    tr_stage2 = tr_stage2 + p1_p0_real_s2
    if clean_rep > 1:
        clean = [r for r in tr_real if all(int(r[c]) == 0 for c in CLASSES)]
        tr_stage2 = tr_stage2 + clean * (clean_rep - 1)

    dtr_s1 = CachedDataset(tr_stage1, feat_dicts, bb_names, phys_map, phys_mu, phys_sd, synth_w)
    dtr_s2 = CachedDataset(tr_stage2, feat_dicts, bb_names, phys_map, phys_mu, phys_sd, synth_w)
    dva = CachedDataset(va, feat_dicts, bb_names, phys_map, phys_mu, phys_sd, synth_w)
    dte = CachedDataset(te, feat_dicts, bb_names, phys_map, phys_mu, phys_sd, synth_w)

    ltr_s1 = DataLoader(dtr_s1, batch_size=batch, shuffle=True, collate_fn=collate_cached, drop_last=True)
    ltr_s2 = DataLoader(dtr_s2, batch_size=batch, shuffle=True, collate_fn=collate_cached, drop_last=True)
    lva = DataLoader(dva, batch_size=batch * 2, shuffle=False, collate_fn=collate_cached)
    lte = DataLoader(dte, batch_size=batch * 2, shuffle=False, collate_fn=collate_cached)

    bb_dims = [feat_dicts[b][rows[0]['name']].shape[-1] for b in bb_names]
    net = MultiNetCached(bb_dims, phys_dim=60, dropout=0.12).to(dev)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(ltr_s1) * ep_s1 + len(ltr_s2) * ep_s2
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))

    s2_ckpts = []
    best_val_doc2 = -1.0
    best_sd = None
    total_epochs = ep_s1 + ep_s2

    for ep in range(total_epochs):
        t0 = time.time()
        is_stage2 = (ep >= ep_s1)
        cur_loader = ltr_s2 if is_stage2 else ltr_s1
        stage_tag = "Stage2-PureReal" if is_stage2 else "Stage1-Sim2Real"
        cur_focus = list(range(6)) if is_stage2 else focus_cis

        net.train()
        opt.zero_grad(set_to_none=True)
        tot, n_step = 0.0, 0
        for feats_by_bb, phys, g, mask7, _ in cur_loader:
            feats_dev = [f.to(dev, non_blocking=True).float() for f in feats_by_bb]
            phys = phys.to(dev, non_blocking=True).float()
            g = g.to(dev, non_blocking=True)
            mask7 = mask7.to(dev, non_blocking=True)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                lc, le, gl = net(feats_dev, phys)

            loss = v16_clean_microscope_loss_fn(
                lc.float(), le.float(), gl.float(), g, mask7, bce_loss_none,
                focus_cis=cur_focus,
                p0_clean_weight=p0_clean_w,
                channel_neg_weight=w_neg,
                alpha_ex=alpha_ex,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sch.step()
            tot += float(loss.detach())
            n_step += 1

        vp0, vp1, vp2, vg, gva = predict_cached(net, lva, dev, alpha_ex)
        pred_va = decode_pc(vp0, vp1, vp2, HAND)
        s_hand = cm.score(gva, pred_va, 0.02)
        clean_va = (gva == 0).all(1)
        s_clean2 = float((clean_va & (pred_va == 0).all(1)).sum() / max(1, clean_va.sum()))
        doc2_va = 25 * s_hand['S_pos'] + 20 * s_hand['S_p0'] + 20 * s_clean2
        print(f'[{name}] ep={ep+1}/{total_epochs} [{stage_tag}] loss={tot/max(1,n_step):.4f} val_Doc1={s_hand["total"]:.2f} val_Doc2={doc2_va:.2f} (S_pos={s_hand["S_pos"]:.3f} S_p0={s_hand["S_p0"]:.3f} Clean2={s_clean2:.3f}) ({time.time()-t0:.2f}s)', flush=True)

        cur_sd = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if is_stage2:
            s2_ckpts.append(cur_sd)
        if doc2_va > best_val_doc2:
            best_val_doc2 = doc2_va
            best_sd = cur_sd

    if not s2_ckpts and best_sd is not None:
        s2_ckpts.append(best_sd)

    if len(s2_ckpts) >= 2:
        swa_sd = {}
        for k in s2_ckpts[0].keys():
            if s2_ckpts[0][k].is_floating_point():
                swa_sd[k] = torch.stack([sd[k] for sd in s2_ckpts], dim=0).mean(0)
            else:
                swa_sd[k] = s2_ckpts[-1][k]
        net.load_state_dict({k: v.to(dev) for k, v in swa_sd.items()})
    elif len(s2_ckpts) == 1:
        net.load_state_dict({k: v.to(dev) for k, v in s2_ckpts[0].items()})

    vp0, vp1, vp2, vg, gva = predict_cached(net, lva, dev, alpha_ex)
    tp0, tp1, tp2, tg, gte = predict_cached(net, lte, dev, alpha_ex)

    # test_150 predictions
    tpl = json.load(open('results/v2_full/template_150.json'))
    names_150 = [e['image'] for e in tpl]
    rows_150 = [{'name': n, **{c: 0 for c in CLASSES}} for n in names_150]
    d150 = CachedDataset(rows_150, feat_dicts, bb_names, phys_map, phys_mu, phys_sd, synth_w)
    l150 = DataLoader(d150, batch_size=batch * 2, shuffle=False, collate_fn=collate_cached)
    p150_0, p150_1, p150_2, p150_g, _ = predict_cached(net, l150, dev, alpha_ex)

    out_base = f'{OUT_DIR}/{name}'
    np.savez(
        out_base + '_probs.npz',
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vg, gva=gva, va_names=[r['name'] for r in va],
        te_p0=tp0, te_p1=tp1, te_p2=tp2, te_gate=tg, gte=gte, te_names=[r['name'] for r in te],
        t150_p0=p150_0, t150_p1=p150_1, t150_p2=p150_2, t150_gate=p150_g, t150_names=names_150,
    )
    st_hand = cm.score(gte, decode_pc(tp0, tp1, tp2, HAND), 0.02)
    clean_te = (gte == 0).all(1)
    pred_te = decode_pc(tp0, tp1, tp2, HAND)
    s_clean2_te = float((clean_te & (pred_te == 0).all(1)).sum() / max(1, clean_te.sum()))
    doc2_te = 25 * st_hand['S_pos'] + 20 * st_hand['S_p0'] + 20 * s_clean2_te
    json.dump({'doc1': float(st_hand['total']), 'doc2': float(doc2_te), 'S_pos': float(st_hand['S_pos']),
               'S_p0': float(st_hand['S_p0']), 'Clean2': float(s_clean2_te)}, open(out_base + '.json', 'w'), indent=2)
    print(f'[{name}] Finished! Blind Test (N={len(te)}): Doc1={st_hand["total"]:.2f} | Doc2={doc2_te:.2f} (S_pos={st_hand["S_pos"]:.3f} S_p0={st_hand["S_p0"]:.3f} Clean2={s_clean2_te:.3f})', flush=True)


def main():
    print('=== Preloading 8-View Feature Caches from /dev/shm into Host RAM ===', flush=True)
    t0 = time.time()
    all_bbs = ['cradio', 'dinov3l', 'pecoreg', 'pecorel', 'siglip2g', 'siglip2', 'convnext', 'eupe']
    feat_dicts = {}
    for b in all_bbs:
        npz_p = f'/dev/shm/v17_cache_{b}.npz'
        d = np.load(npz_p)
        names = d['names']
        feats = d['feats']
        feat_dicts[b] = {nm: feats[i] for i, nm in enumerate(names)}
        print(f'  Loaded {b:10s}: {len(names)} images, dim={feats.shape[-1]} ({time.time()-t0:.1f}s)', flush=True)

    print('Preloading 60D Native Physical Features...', flush=True)
    man16 = pd.read_csv(f'{SPLIT_DIR}/manifest_s16_v17.csv')
    man17 = pd.read_csv(f'{SPLIT_DIR}/manifest_s17_v17.csv')
    all_unique_names = sorted(list(set(man16['name']).union(set(man17['name']))))
    phys_map = extract_for_names(all_unique_names, 'results/v8_experts/native_phys_feats.npz')
    print(f'Physical features loaded for {len(phys_map)} images.', flush=True)

    # Specialist jobs configuration (SpecA, SpecB, SpecC, SpecD across s16 and s17)
    JOBS = [
        ('V17_S2R_SpecA_OverGen_TexLoss_s16', 0, 16, '过度生成,纹理损失', ['pecoreg', 'siglip2g', 'cradio', 'convnext'], 3, 2, 1.05, 4.2),
        ('V17_S2R_SpecA_OverGen_TexLoss_s17', 1, 17, '过度生成,纹理损失', ['pecoreg', 'siglip2g', 'cradio', 'convnext'], 3, 2, 1.05, 4.2),
        ('V17_S2R_SpecB_Sharp_TexLoss_s16',   2, 16, '过锐,纹理损失',     ['dinov3l', 'cradio', 'siglip2', 'convnext'], 3, 2, 1.05, 4.3),
        ('V17_S2R_SpecB_Sharp_TexLoss_s17',   3, 17, '过锐,纹理损失',     ['dinov3l', 'cradio', 'siglip2', 'convnext'], 3, 2, 1.05, 4.3),
        ('V17_S2R_SpecC_Blur_Bokeh_Screen_s16', 4, 16, '模糊,纹理损失,虚化突兀,花屏', ['pecorel', 'dinov3l', 'siglip2', 'eupe'], 3, 2, 1.02, 4.2),
        ('V17_S2R_SpecC_Blur_Bokeh_Screen_s17', 5, 17, '模糊,纹理损失,虚化突兀,花屏', ['pecorel', 'dinov3l', 'siglip2', 'eupe'], 3, 2, 1.02, 4.2),
        ('V17_S2R_SpecD_All6_Giant_s16',      6, 16, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀', ['pecoreg', 'dinov3l', 'cradio', 'pecorel'], 3, 2, 1.02, 4.2),
        ('V17_S2R_SpecD_All6_Giant_s17',      7, 17, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀', ['pecoreg', 'dinov3l', 'cradio', 'pecorel'], 3, 2, 1.02, 4.2),
    ]

    import concurrent.futures
    print('\n=== Launching 8 Cached Specialists Concurrently across GPUs 0..7 ===', flush=True)
    t_train_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                train_single_specialist,
                name, gpu, sp, focus, bbs, feat_dicts, phys_map,
                ep_s1=ep_s1, ep_s2=ep_s2, batch=32, lr=1.3e-3, p0_clean_w=p0_cln, w_neg=w_neg,
                seed=500 + gpu
            )
            for (name, gpu, sp, focus, bbs, ep_s1, ep_s2, w_neg, p0_cln) in JOBS
        ]
        concurrent.futures.wait(futures)
        for f in futures:
            if f.exception() is not None:
                raise f.exception()

    print(f'\n=== All 8 Specialists Completed in {time.time()-t_train_start:.1f}s! ===', flush=True)


if __name__ == '__main__':
    main()
