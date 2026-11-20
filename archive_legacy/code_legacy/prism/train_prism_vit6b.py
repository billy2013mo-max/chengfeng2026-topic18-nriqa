#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/prism/train_prism_vit6b.py
===============================
PRISM Tower 1: `Prism-ViT6B` Full-Spectrum 1:1 Microscope + ABMIL 6B Foundation Trainer.
"""
import argparse
import csv
import json
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

PRISM_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = '/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(PRISM_SRC_DIR, '../..'))
os.chdir(REPO_ROOT)
for p in [REPO_ROOT, os.path.join(REPO_ROOT, 'code'), os.path.join(REPO_ROOT, 'code/finetune'), os.path.join(REPO_ROOT, 'code/decision_eval'), PRISM_SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data import load_split, read_manifest, CLASSES
from models import build_backbone
from train_v9_specialist import set_seed
from train_v17_end2end_sim2real_specialist import is_synth_name
from train_v18_paired_kd_specialist import (
    load_roi_and_synth_meta,
    V18FastMicroscopeDataset,
    _extract_seed_stem,
)
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm

from heads_and_losses import (
    ConditionalHazardOrdinalHead,
    PrismExactGradeLoss,
    PrismDifferentialCalibratorLoss,
)
from train_prism_quad import PrismQuadMicroscopeNet, predict_prism_quad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='internvit_abmil:lastk4,pecoreg_abmil:lastk4,cradio_abmil:lastk4,dinov3l_abmil:lastk4')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--lr-last', type=float, default=1.2e-4)
    ap.add_argument('--lr-head', type=float, default=8.5e-4)
    ap.add_argument('--layer-decay', type=float, default=0.85)
    ap.add_argument('--wd', type=float, default=0.015)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.12)
    ap.add_argument('--alpha-simplex', type=float, default=0.35)
    ap.add_argument('--split', default='data/resplits/split_s16.csv')
    ap.add_argument('--manifest', default='data/manifest_combined.csv')
    ap.add_argument('--phys-cache', default='results/v8_experts/native_phys_feats.npz')
    ap.add_argument('--tau', type=float, default=0.02)
    ap.add_argument('--out', default='results/prism/prism_vit6b_s16.json')
    a = ap.parse_args()

    os.environ['INTERNVIT_SIZE'] = str(a.size)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    set_seed(a.seed)
    dev = 'cuda'
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    roi_map, meta_map, synth_meta = load_roi_and_synth_meta()
    rows = load_split(a.split)
    man = read_manifest(a.manifest)
    rows = [r for r in rows if r['name'] in man and not is_synth_name(r['name']) and ('_prism_' not in r['name'])]
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})

    tr = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test'] or va

    va_names_set = set(r['name'] for r in va)
    tr = [r for r in tr if r['name'] not in va_names_set]

    from extract_native_phys_feats import extract_for_names
    phys_map = extract_for_names([r['name'] for r in tr + va + te], a.phys_cache)
    phys_tr_mat = np.stack([phys_map[r['name']] for r in tr], axis=0)
    phys_mu = np.mean(phys_tr_mat, axis=0)
    phys_sd = np.std(phys_tr_mat, axis=0) + 1e-5

    rare_p1_p0 = [r for r in tr if any(int(r[CLASSES[ci]]) >= 2 for ci in (2, 3, 4, 5))]
    clean_tr = [r for r in tr if all(int(r[c]) == 0 for c in CLASSES)]
    tr_aug = list(tr) + rare_p1_p0 + clean_tr * 2

    dtr = V18FastMicroscopeDataset(tr_aug, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, True, tiles=a.tiles)
    dva = V18FastMicroscopeDataset(va, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, tiles=a.tiles)
    dte = V18FastMicroscopeDataset(te, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, tiles=a.tiles)

    ltr = DataLoader(dtr, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    lva = DataLoader(dva, batch_size=a.batch * 2, shuffle=False, num_workers=a.workers, pin_memory=True)
    lte = DataLoader(dte, batch_size=a.batch * 2, shuffle=False, num_workers=a.workers, pin_memory=True)

    specs = []
    for s in a.backbones.split(','):
        name, mode = s.split(':')
        b = build_backbone(name)
        specs.append((b, name, mode))

    net = PrismQuadMicroscopeNet([x[0] for x in specs], phys_dim=60, dropout=a.dropout, n_quad=a.tiles * a.tiles).to(dev)

    pg = []
    abmil_params = []
    for (b, name, mode) in specs:
        m = b[0]
        blocks = b[4]
        for p in m.parameters():
            p.requires_grad_(False)
        abmil_ids = set()
        for attr in ('abmil_pool', 'dual_tap'):
            if hasattr(m, attr):
                subm = getattr(m, attr).float()
                for p in subm.parameters():
                    p.requires_grad_(True)
                    abmil_params.append(p)
                    abmil_ids.add(id(p))
        if mode.startswith('lastk') and blocks is not None:
            k = int(mode[5:] or 4)
            n_blk = len(blocks)
            for idx_b in range(max(0, n_blk - k), n_blk):
                depth = (n_blk - 1) - idx_b
                blk_params = []
                for p in blocks[idx_b].parameters():
                    if id(p) not in abmil_ids:
                        p.requires_grad_(True)
                        blk_params.append(p)
                if blk_params:
                    pg.append({'params': blk_params, 'lr': a.lr_last * (a.layer_decay ** depth)})

    head_params = (
        abmil_params
        + list(net.tile_attn.parameters())
        + list(net.phys_mlp.parameters())
        + list(net.geom_film.parameters())
        + list(net.prism_head.parameters())
    )
    pg.append({'params': head_params, 'lr': a.lr_head})
    tr_p = [p for g in pg for p in g['params']]

    print(
        f'=== [PRISM Tower 1: Prism-ViT6B] 1:1 Microscope + ABMIL + Conditional Hazard Head ===\n'
        f'  Trainable={sum(p.numel() for p in tr_p)/1e6:.1f}M | Train={len(dtr)} | Val={len(dva)} | Test={len(dte)}',
        flush=True,
    )

    opt = torch.optim.AdamW(pg, weight_decay=a.wd)
    total_steps = max(1, (len(ltr) // a.grad_accum) * a.epochs)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    loss_exact_fn = PrismExactGradeLoss(p0_clean_weight=4.5, alpha_simplex=a.alpha_simplex)
    loss_diff_fn = PrismDifferentialCalibratorLoss(base_margin=0.28, domain_weight=0.10)

    ckpts = []
    for ep in range(a.epochs):
        t0 = time.time()
        net.train()
        opt.zero_grad(set_to_none=True)
        tot, nb = 0.0, 0
        for step, (x, phys, geom, g, mask7, names) in enumerate(ltr):
            x = x.to(dev, non_blocking=True)
            phys = phys.to(dev, non_blocking=True)
            geom = geom.to(dev, non_blocking=True)
            g = g.to(dev, non_blocking=True)
            mask7 = mask7.to(dev, non_blocking=True)
            stems = [_extract_seed_stem(str(nm)) for nm in names]

            with torch.autocast('cuda', dtype=torch.bfloat16):
                lh, ls, gl, zm, dl = net(x, phys, geom, is_real_mask=(mask7[:, 6] > 0.5))

            loss = (
                loss_exact_fn(net.prism_head, lh.float(), ls.float(), gl.float(), g, mask7)
                + 0.35 * loss_diff_fn(net.prism_head, zm.float(), dl.float(), g, mask7, stems)
            )
            (loss / a.grad_accum).backward()
            if (step + 1) % a.grad_accum == 0 or (step + 1) == len(ltr):
                torch.nn.utils.clip_grad_norm_(tr_p, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sch.step()
            tot += float(loss.detach())
            nb += 1

        vp0, vp1, vp2, vg, gva = predict_prism_quad(net, lva, dev, a.alpha_simplex)
        s_hand = cm.score(gva, decode_pc(vp0, vp1, vp2, HAND), a.tau)
        print(
            f'ep={ep+1}/{a.epochs} loss={tot/max(1,nb):.4f} val_Doc1={s_hand["total"]:.2f} '
            f'(S_pos={s_hand["S_pos"]:.3f} S_p0={s_hand["S_p0"]:.3f} S_clean={s_hand["S_clean"]:.3f}) ({time.time()-t0:.0f}s)',
            flush=True,
        )
        if ep >= max(1, a.epochs - 2):
            ckpts.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})

    if len(ckpts) >= 2:
        swa_sd = {
            k: (sum(ck[k].float() for ck in ckpts) / float(len(ckpts))).to(ckpts[-1][k].dtype)
            if ckpts[-1][k].is_floating_point() else ckpts[-1][k]
            for k in ckpts[-1].keys()
        }
        net.load_state_dict({k: v.to(dev) for k, v in swa_sd.items()})

    dva_f = V18FastMicroscopeDataset(va, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, hflip=True, tiles=a.tiles)
    dte_f = V18FastMicroscopeDataset(te, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, a.size, False, hflip=True, tiles=a.tiles)
    lva_f = DataLoader(dva_f, batch_size=a.batch * 2, shuffle=False, num_workers=a.workers, pin_memory=True)
    lte_f = DataLoader(dte_f, batch_size=a.batch * 2, shuffle=False, num_workers=a.workers, pin_memory=True)

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
    l150 = DataLoader(d150, batch_size=a.batch * 2, shuffle=False, num_workers=a.workers, pin_memory=True)
    l150_f = DataLoader(d150_f, batch_size=a.batch * 2, shuffle=False, num_workers=a.workers, pin_memory=True)
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
    print(f'=== FINAL PRISM-VIT6B TEST (N={len(gte)}): Doc1={st_hand["total"]:.2f} ===', flush=True)


if __name__ == '__main__':
    main()
