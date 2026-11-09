#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/prism/train_prism_quad.py
==============================
PRISM Tower 3: `Prism-Quad` Full-Spectrum 4-Backbone Deep Native Microscope Trainer.
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

REPO_ROOT = '/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(REPO_ROOT)
for p in [REPO_ROOT, os.path.join(REPO_ROOT, 'code'), os.path.join(REPO_ROOT, 'code/finetune'), os.path.join(REPO_ROOT, 'code/decision_eval'), os.path.dirname(os.path.abspath(__file__))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data import load_split, read_manifest, CLASSES, FULL
from train_v9_specialist import build_v9_backbone, set_seed
from train_v17_end2end_sim2real_specialist import is_synth_name
from train_v18_paired_kd_specialist import (
    load_roi_and_synth_meta,
    V18FastMicroscopeDataset,
    _extract_seed_stem,
)
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm

try:
    from code.prism.heads_and_losses import (
        ConditionalHazardOrdinalHead,
        PrismExactGradeLoss,
        PrismDifferentialCalibratorLoss,
    )
except ImportError:
    from heads_and_losses import (
        ConditionalHazardOrdinalHead,
        PrismExactGradeLoss,
        PrismDifferentialCalibratorLoss,
    )


class PrismQuadMicroscopeNet(nn.Module):
    def __init__(self, specs, phys_dim: int = 60, dropout: float = 0.12, n_quad: int = 4):
        super().__init__()
        self.n_quad = n_quad
        self.bs = nn.ModuleList()
        self.fwds = []
        self.tile_attn = nn.ModuleList()

        for (m, fwd, mu, sd, bl, dim) in specs:
            self.bs.append(m)
            self.fwds.append(fwd)
            self.register_buffer(f'mu_{len(self.fwds)}', torch.tensor(mu, dtype=torch.bfloat16).view(1, 3, 1, 1))
            self.register_buffer(f'sd_{len(self.fwds)}', torch.tensor(sd, dtype=torch.bfloat16).view(1, 3, 1, 1))
            self.tile_attn.append(nn.Sequential(
                nn.Linear(dim, max(64, dim // 8)),
                nn.GELU(),
                nn.Linear(max(64, dim // 8), 1),
            ))

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
        d_total = d_vis + 256
        self.geom_film = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, d_total * 2),
        )
        nn.init.zeros_(self.geom_film[-1].weight)
        nn.init.zeros_(self.geom_film[-1].bias)

        self.prism_head = ConditionalHazardOrdinalHead(
            in_dim=d_total,
            hidden_dim=512,
            proj_dim=128,
            dropout=dropout,
        )

    def _agg(self, f: torch.Tensor, attn_mod: nn.Module) -> torch.Tensor:
        g_view = f[:, 0]
        tiles = f[:, 1:]
        quad_mean = f[:, 1:1 + self.n_quad].mean(1)
        nat_views = f[:, 1 + self.n_quad:]
        if nat_views.shape[1] >= 3:
            edge_max = nat_views[:, :2].amax(1)
            tex_view = nat_views[:, 2]
        else:
            edge_max = nat_views.amax(1)
            tex_view = nat_views.mean(1)

        r_edge_glob = edge_max - g_view
        r_edge_tex = edge_max - tex_view
        t_max = tiles.amax(1)
        w = torch.softmax(attn_mod(tiles), dim=1)
        t_att = (tiles * w).sum(1)
        return torch.cat([g_view, quad_mean, edge_max, tex_view, r_edge_glob, r_edge_tex, t_max, t_att], dim=1)

    def forward(self, x: torch.Tensor, phys: torch.Tensor, geom: torch.Tensor = None, is_real_mask: torch.Tensor = None):
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W).to(torch.bfloat16).div_(255.0)
        outs = []
        for i, (m, fwd) in enumerate(zip(self.bs, self.fwds)):
            mu = getattr(self, f'mu_{i + 1}')
            sd = getattr(self, f'sd_{i + 1}')
            f = fwd((x_flat - mu) / sd).view(B, T, -1).float()
            outs.append(self._agg(f, self.tile_attn[i]))

        h_vis = torch.cat(outs, dim=-1)
        h_phys = self.phys_mlp(phys.float())
        h = torch.cat([h_vis, h_phys], dim=-1)
        if geom is not None:
            gb = self.geom_film(geom.float())
            gamma, beta = gb.chunk(2, dim=-1)
            h = h * (1.0 + 0.15 * torch.tanh(gamma)) + 0.15 * beta

        return self.prism_head(h, is_real_mask=is_real_mask)


def build_layerwise_param_groups(specs, lr_full: float, lr_last: float, decay: float = 0.85):
    pg = []
    for (m, fwd, mu, sd, bl, dim), mode in specs:
        for p in m.parameters():
            p.requires_grad_(False)
        if mode == 'full':
            for p in m.parameters():
                p.requires_grad_(True)
            pg.append({'params': [p for p in m.parameters() if p.requires_grad], 'lr': lr_full})
        elif mode.startswith('lastk') and bl is not None:
            k = int(mode[5:])
            n_blocks = len(bl)
            start_idx = max(0, n_blocks - k)
            for idx_b in range(start_idx, n_blocks):
                depth_from_top = (n_blocks - 1) - idx_b
                blk_lr = lr_last * (decay ** depth_from_top)
                for p in bl[idx_b].parameters():
                    p.requires_grad_(True)
                pg.append({'params': list(bl[idx_b].parameters()), 'lr': blk_lr})
        for p in m.parameters():
            if not p.requires_grad and p.is_floating_point():
                p.data = p.data.to(torch.bfloat16)
    return pg


@torch.inference_mode()
def predict_prism_quad(net: PrismQuadMicroscopeNet, loader: DataLoader, dev: str, alpha_simplex: float = 0.35):
    net.eval()
    P0, P1, P2, GATE, G = [], [], [], [], []
    for x, phys, geom, g, _, _ in loader:
        x = x.to(dev, non_blocking=True)
        phys = phys.to(dev, non_blocking=True)
        geom = geom.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lh, ls, gl, _, _ = net(x, phys, geom, is_real_mask=None)
        p0, p1, p2, _ = net.prism_head.exact_probabilities(lh.float(), ls.float(), alpha_simplex=alpha_simplex)
        P0.append(p0.cpu().numpy())
        P1.append(p1.cpu().numpy())
        P2.append(p2.cpu().numpy())
        GATE.append(torch.sigmoid(gl.float()).cpu().numpy())
        G.append(g.numpy())
    return np.concatenate(P0, 0), np.concatenate(P1, 0), np.concatenate(P2, 0), np.concatenate(GATE, 0), np.concatenate(G, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='dinov3l:lastk6,pecoreg:lastk6,siglip2g:lastk6,convnext:full')
    ap.add_argument('--init-ckpt', default='results/v17_end2end_s16_s17/V17_S2R_SpecD_All6_Giant_s16_ep5.pt')
    ap.add_argument('--epochs-s1', type=int, default=3)
    ap.add_argument('--epochs-s2', type=int, default=2)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--lr-full', type=float, default=6.5e-5)
    ap.add_argument('--lr-last', type=float, default=3.2e-5)
    ap.add_argument('--lr-head', type=float, default=8.0e-4)
    ap.add_argument('--layer-decay', type=float, default=0.85)
    ap.add_argument('--wd', type=float, default=0.015)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dropout', type=float, default=0.12)
    ap.add_argument('--alpha-simplex', type=float, default=0.35)
    ap.add_argument('--split', required=True)
    ap.add_argument('--manifest', default='data/manifest_combined.csv')
    ap.add_argument('--phys-cache', default='results/v8_experts/native_phys_feats.npz')
    ap.add_argument('--tau', type=float, default=0.02)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    set_seed(a.seed)
    dev = 'cuda'
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True

    roi_map, meta_map, synth_meta = load_roi_and_synth_meta()
    prism_meta_path = 'data/v17_splits/prism_synth_metadata.json'
    if os.path.exists(prism_meta_path):
        synth_meta.update(json.load(open(prism_meta_path, encoding='utf-8')))

    rows = load_split(a.split)
    man = read_manifest(a.manifest)
    rows = [r for r in rows if r['name'] in man]
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})

    from extract_native_phys_feats import extract_for_names
    tr_orig = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test'] or va

    tr_real = [r for r in tr_orig if not is_synth_name(r['name']) and ('_NatNeg' not in r['name']) and ('_prism_' not in r['name'])]
    phys_map = extract_for_names([r['name'] for r in tr_real + va + te], a.phys_cache)
    phys_tr_mat = np.stack([phys_map[r['name']] for r in tr_real], axis=0)
    phys_mu = np.mean(phys_tr_mat, axis=0)
    phys_sd = np.std(phys_tr_mat, axis=0) + 1e-5

    tr_synth_all = [r for r in tr_orig if is_synth_name(r['name']) or ('_prism_' in r['name'])]
    tr_synth_all.sort(key=lambda r: r['name'])
    rng = random.Random(a.seed)
    real_shuf = list(tr_real)
    rng.shuffle(real_shuf)
    tr_stage1 = []
    s_ptr = 0
    for i_r, r_row in enumerate(real_shuf):
        tr_stage1.append(r_row)
        if (i_r + 1) % 5 == 0 and s_ptr + 3 <= len(tr_synth_all):
            tr_stage1.extend(tr_synth_all[s_ptr:s_ptr + 3])
            s_ptr += 3

    tr_stage2 = list(tr_real)
    rare_p1_p0 = [r for r in tr_real if any(int(r[CLASSES[ci]]) >= 2 for ci in (2, 3, 4, 5))]
    clean_real = [r for r in tr_real if all(int(r[c]) == 0 for c in CLASSES)]
    tr_stage2 = tr_stage2 + rare_p1_p0 + clean_real

    print(
        f'=== [PRISM Tower 3: Prism-Quad] All-6 Full-Spectrum Deep Microscope Trainer ===\n'
        f'  Backbones={a.backbones} | Stage1={len(tr_stage1)} ({a.epochs_s1}ep) | Stage2={len(tr_stage2)} ({a.epochs_s2}ep) | Val={len(va)} | Test={len(te)}',
        flush=True,
    )

    dtr_s1 = V18FastMicroscopeDataset(tr_stage1, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, True, tiles=a.tiles)
    dtr_s2 = V18FastMicroscopeDataset(tr_stage2, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, True, tiles=a.tiles)
    dva = V18FastMicroscopeDataset(va, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, tiles=a.tiles)
    dte = V18FastMicroscopeDataset(te, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, tiles=a.tiles)

    val_batch = a.batch * 2
    ltr_s1 = DataLoader(dtr_s1, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True, persistent_workers=True, drop_last=True)
    ltr_s2 = DataLoader(dtr_s2, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, persistent_workers=True, drop_last=True)
    lva = DataLoader(dva, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True, persistent_workers=True)
    lte = DataLoader(dte, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True, persistent_workers=True)

    raw_specs = []
    specs_with_mode = []
    for item in a.backbones.split(','):
        nm, mode = item.split(':')
        b_tuple = build_v9_backbone(nm)
        raw_specs.append(b_tuple)
        specs_with_mode.append((b_tuple, mode))

    pg = build_layerwise_param_groups(specs_with_mode, a.lr_full, a.lr_last, decay=a.layer_decay)
    net = PrismQuadMicroscopeNet(raw_specs, phys_dim=60, dropout=a.dropout, n_quad=a.tiles * a.tiles).to(dev)

    if a.init_ckpt and os.path.exists(a.init_ckpt):
        print(f'  Warm-starting compatible backbone & microscope layers from {a.init_ckpt}', flush=True)
        sd = torch.load(a.init_ckpt, map_location='cpu')
        cur_sd = net.state_dict()
        compat_sd = {k: v for k, v in sd.items() if k in cur_sd and cur_sd[k].shape == v.shape}
        net.load_state_dict(compat_sd, strict=False)

    head_params = (
        list(net.tile_attn.parameters())
        + list(net.phys_mlp.parameters())
        + list(net.geom_film.parameters())
        + list(net.prism_head.parameters())
    )
    pg.append({'params': head_params, 'lr': a.lr_head})
    trainable_params = [p for p in net.parameters() if p.requires_grad]

    try:
        opt = torch.optim.AdamW(pg, weight_decay=a.wd, fused=True)
    except Exception:
        opt = torch.optim.AdamW(pg, weight_decay=a.wd)

    total_steps = max(1, (len(ltr_s1) // a.grad_accum) * a.epochs_s1 + (len(ltr_s2) // a.grad_accum) * a.epochs_s2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    loss_exact_fn = PrismExactGradeLoss(p0_clean_weight=4.5, alpha_simplex=a.alpha_simplex)
    loss_diff_fn = PrismDifferentialCalibratorLoss(base_margin=0.28, domain_weight=0.20)

    s2_ckpts = []
    total_epochs = a.epochs_s1 + a.epochs_s2
    for ep in range(total_epochs):
        t0 = time.time()
        is_stage2 = (ep >= a.epochs_s1)
        cur_loader = ltr_s2 if is_stage2 else ltr_s1
        stage_tag = 'Stage2-PureReal-Hazard' if is_stage2 else 'Stage1-DiffCalib+Real'

        net.train()
        opt.zero_grad(set_to_none=True)
        tot_loss, n_step = 0.0, 0
        for step, (x, phys, geom, g, mask7, names) in enumerate(cur_loader):
            x = x.to(dev, non_blocking=True)
            phys = phys.to(dev, non_blocking=True)
            geom = geom.to(dev, non_blocking=True)
            g = g.to(dev, non_blocking=True)
            mask7 = mask7.to(dev, non_blocking=True)
            is_real_mask = (mask7[:, 6] > 0.5)
            stems = [_extract_seed_stem(str(nm)) for nm in names]

            with torch.autocast('cuda', dtype=torch.bfloat16):
                lh, ls, gl, zm, dl = net(x, phys, geom, is_real_mask=is_real_mask)

            l_ex = loss_exact_fn(net.prism_head, lh.float(), ls.float(), gl.float(), g, mask7)
            l_df = loss_diff_fn(net.prism_head, zm.float(), dl.float(), g, mask7, stems)
            loss = l_ex + (0.35 if is_stage2 else 0.65) * l_df

            (loss / a.grad_accum).backward()
            if (step + 1) % a.grad_accum == 0 or (step + 1) == len(cur_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sch.step()
            tot_loss += float(loss.detach())
            n_step += 1

        vp0, vp1, vp2, vg, gva = predict_prism_quad(net, lva, dev, a.alpha_simplex)
        pred_va = decode_pc(vp0, vp1, vp2, HAND)
        s_hand = cm.score(gva, pred_va, a.tau)
        print(
            f'ep={ep+1}/{total_epochs} [{stage_tag}] loss={tot_loss/max(1,n_step):.4f} '
            f'val_Doc1={s_hand["total"]:.2f} (S_pos={s_hand["S_pos"]:.3f} S_p0={s_hand["S_p0"]:.3f} S_clean={s_hand["S_clean"]:.3f}) ({time.time()-t0:.0f}s)',
            flush=True,
        )
        cur_sd = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if is_stage2 or ep == total_epochs - 1:
            s2_ckpts.append(cur_sd)

    if len(s2_ckpts) >= 2:
        swa_sd = {
            k: (sum(ck[k] for ck in s2_ckpts) / float(len(s2_ckpts))) if s2_ckpts[0][k].is_floating_point() else s2_ckpts[-1][k]
            for k in s2_ckpts[0].keys()
        }
        net.load_state_dict({k: v.to(dev) for k, v in swa_sd.items()})
        out_base_pt = a.out[:-5] if a.out.endswith('.json') else a.out
        torch.save(swa_sd, f'{out_base_pt}_swa.pt')

    dva_f = V18FastMicroscopeDataset(va, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, hflip=True, tiles=a.tiles)
    dte_f = V18FastMicroscopeDataset(te, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, hflip=True, tiles=a.tiles)
    lva_f = DataLoader(dva_f, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    lte_f = DataLoader(dte_f, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    vp0_a, vp1_a, vp2_a, vg_a, gva = predict_prism_quad(net, lva, dev, a.alpha_simplex)
    vp0_b, vp1_b, vp2_b, vg_b, _ = predict_prism_quad(net, lva_f, dev, a.alpha_simplex)
    vp0, vp1, vp2, vg = 0.5 * (vp0_a + vp0_b), 0.5 * (vp1_a + vp1_b), 0.5 * (vp2_a + vp2_b), 0.5 * (vg_a + vg_b)

    tp0_a, tp1_a, tp2_a, tg_a, gte = predict_prism_quad(net, lte, dev, a.alpha_simplex)
    tp0_b, tp1_b, tp2_b, tg_b, _ = predict_prism_quad(net, lte_f, dev, a.alpha_simplex)
    tp0, tp1, tp2, tg = 0.5 * (tp0_a + tp0_b), 0.5 * (tp1_a + tp1_b), 0.5 * (tp2_a + tp2_b), 0.5 * (tg_a + tg_b)

    tpl = json.load(open('results/v2_full/template_150.json'))
    names_150 = [e['image'] for e in tpl]
    phys_150 = extract_for_names(names_150, a.phys_cache)
    rows_150 = [{'name': n, **{c: 0 for c in CLASSES}} for n in names_150]
    d150 = V18FastMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, hflip=False, tiles=a.tiles)
    d150_f = V18FastMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, hflip=True, tiles=a.tiles)
    l150 = DataLoader(d150, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    l150_f = DataLoader(d150_f, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    p150_0a, p150_1a, p150_2a, p150_ga, _ = predict_prism_quad(net, l150, dev, a.alpha_simplex)
    p150_0b, p150_1b, p150_2b, p150_gb, _ = predict_prism_quad(net, l150_f, dev, a.alpha_simplex)
    p150_0, p150_1, p150_2, p150_g = 0.5 * (p150_0a + p150_0b), 0.5 * (p150_1a + p150_1b), 0.5 * (p150_2a + p150_2b), 0.5 * (p150_ga + p150_gb)

    out_base = a.out[:-5] if a.out.endswith('.json') else a.out
    np.savez_compressed(
        out_base + '_probs.npz',
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vg, gva=gva, va_names=np.array([r['name'] for r in va]),
        te_p0=tp0, te_p1=tp1, te_p2=tp2, te_gate=tg, gte=gte, te_names=np.array([r['name'] for r in te]),
        t150_p0=p150_0, t150_p1=p150_1, t150_p2=p150_2, t150_gate=p150_g, t150_names=np.array(names_150), names_150=np.array(names_150),
    )
    st_hand = cm.score(gte, decode_pc(tp0, tp1, tp2, HAND), a.tau)
    json.dump({'doc1': float(st_hand['total']), 'S_pos': float(st_hand['S_pos']), 'S_p0': float(st_hand['S_p0']), 'S_clean': float(st_hand['S_clean'])}, open(out_base + '.json', 'w'), indent=2)
    print(f'=== FINAL PRISM-QUAD TEST (N={len(gte)}): Doc1={st_hand["total"]:.2f} (S_pos={st_hand["S_pos"]:.3f} S_p0={st_hand["S_p0"]:.3f} S_clean={st_hand["S_clean"]:.3f}) ===', flush=True)


if __name__ == '__main__':
    main()
