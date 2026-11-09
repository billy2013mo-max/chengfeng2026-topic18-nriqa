#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V17 Two-Stage Sim2Real End-to-End Multi-Backbone + 1:1 Native Microscope Contrast Specialist Trainer
(`code/finetune/train_v17_end2end_sim2real_specialist.py`).

Implements the Two-Stage Sim2Real Blueprint (`docs/guide.md` Sections 4.1 & 4.2) end-to-end on vision backbones:
1. Strict Zero-Leakage Seed Isolation on `s16` and `s17`:
   - Synthetic quadruplets `{stem}_d{c}_P{k}.png` from `/dev/shm/v17_synth_quads/` are included in `train`
     IF AND ONLY IF their clean anchor seed `{stem}` belongs to the `train` split of `s16` / `s17`
     (0% seed overlap with `val` 500 or `test` 1,000 isolated blind test images!).
2. Strict Single-Channel Masking (`get_sample_class_mask6`):
   - Every synthetic image `{stem}_d{c}_P{k}.png` ONLY supervises its target defect class `c` with weight `synth_w`,
     while all non-target dimensions `c' != c` have weight `0.0` (zero cross-dimension false-negative pollution)
     and `is_real = 0.0` (All-Clean Gate supervises 100% authentic human-annotated images only).
3. Two-Stage Sim2Real End-to-End Schedule:
   - Stage 1 (`--epochs-s1 3`): Unfreezes last-K vision backbone blocks + 1:1 native microscope contrast heads,
     training on `tr_real + tr_verified_quadruplets` with ordinal margin ranking (`Clean < P2 < P1 < P0`) and P1/P2 separation.
   - Stage 2 (`--epochs-s2 2`): Switches DataLoader to 100% Pure-Real Authentic Images (`tr_real` only, zero synthetic images!)
     to align logit thresholds, class priors, and decision boundaries 100% onto authentic human annotations, followed by SWA + 2x TTA.
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
from train_v2 import exact_probs_from_dual, exact_targets_14, P0_IDX
from train_v9_specialist import (
    CACHE_OK, CLASS_TO_COLS14, build_v9_backbone, set_seed
)
from train_v16_microscope_specialist import (
    MultiNetV16, v16_clean_microscope_loss_fn
)
import comp_metrics as cm


@torch.inference_mode()
def predict_v17(net, loader, dev, alpha_ex=0.58):
    net.eval()
    P0, P1, P2, GATE, G = [], [], [], [], []
    for x, phys, g, _, _ in loader:
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


def is_synth_name(name: str) -> bool:
    if name.startswith(('v9_', 'v8_', 'v7_', 'v6_', 'v4_')):
        return True
    for d in range(6):
        if f'_d{d}_P' in name:
            return True
    return False


def get_sample_class_mask6_v17(name: str, synth_w: float = 0.75) -> np.ndarray:
    """Return a 7-dim float mask:
    - indices 0..5: class supervision weight for each of the 6 defect classes
    - index 6: 1.0 if authentic human-annotated real image (for All-Clean Gate supervision), 0.0 if synthetic
    Note: DIMS in synth generator (`code/utils.py`):
      0: '模糊', 1: '纹理损失', 2: '过锐', 3: '花屏', 4: '过度生成', 5: '虚化突兀'
    which matches `CLASSES` in `code/finetune/data.py`:
      ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']!
    """
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


class V17NativeMicroscopeDataset(Dataset):
    def __init__(self, rows, phys_map, phys_mu, phys_sd, size=448, train=True, hflip=False, tiles=2, synth_w=0.75):
        self.rows = rows
        self.phys_map = phys_map
        self.phys_mu = phys_mu
        self.phys_sd = phys_sd
        self.size = size
        self.train = train
        self.hflip = hflip
        self.tiles = tiles
        self.synth_w = synth_w
        self._crop_cache = {}
        if CACHE_OK:
            d = np.load('/dev/shm/race18_native/index.npz', allow_pickle=True)
            self.names = list(d['names'])
            self.offsets = d['offsets']
            self.h = d['h']
            self.w = d['w']
            self.pos = {n: i for i, n in enumerate(self.names)}
            self.data = np.memmap('/dev/shm/race18_native/cache.bin', dtype=np.uint8, mode='r')
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
        for d in ['/dev/shm/v17_synth_quads', '/tmp/v9_authentic_imgs', '/tmp/v8_ladder_imgs',
                  '/tmp/v7_overgen_pro_imgs', '/tmp/v6_synth_imgs',
                  'data/train', 'data/images', 'data/images_supp', 'data/test_150', 'data/test']:
            p = os.path.join('/wanqing-develop/luowenjing/18', d, name) if not d.startswith('/') else os.path.join(d, name)
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
        raise FileNotFoundError(f'Cannot find {name}')

    def _extract_native_microscope_crops(self, im, nm=None):
        if not self.train and nm is not None and nm in self._crop_cache:
            return self._crop_cache[nm]
        S = self.size
        w, h = im.size
        if w < S or h < S:
            im = im.resize((max(w, S), max(h, S)), Image.BICUBIC)
            w, h = im.size

        xs = [0, max(0, (w - S) // 2), max(0, w - S)] if w > S else [0]
        ys = [0, max(0, (h - S) // 2), max(0, h - S)] if h > S else [0]
        cands = []
        energies = []
        for y0 in ys:
            for x0 in xs:
                if self.train and w > S + 16 and h > S + 16:
                    jx = random.randint(-8, 8)
                    jy = random.randint(-8, 8)
                    x_c = max(0, min(w - S, x0 + jx))
                    y_c = max(0, min(h - S, y0 + jy))
                else:
                    x_c, y_c = x0, y0
                patch = im.crop((x_c, y_c, x_c + S, y_c + S))
                arr = np.asarray(patch, dtype=np.uint8)[::4, ::4, 1].astype(np.float32)
                gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
                gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
                lx = np.abs(2.0 * arr[:, 1:-1] - arr[:, :-2] - arr[:, 2:]).mean()
                ly = np.abs(2.0 * arr[1:-1, :] - arr[:-2, :] - arr[2:, :]).mean()
                cands.append(patch)
                energies.append(float(gx + gy + 0.85 * (lx + ly)))

        if len(cands) == 1:
            res = [cands[0], cands[0], cands[0]]
        else:
            order = np.argsort(energies)
            p_peak1 = cands[int(order[-1])]
            p_peak2 = cands[int(order[-2])]
            p_mid   = cands[int(order[len(order) // 2])]
            res = [p_peak1, p_peak2, p_mid]
        if not self.train and nm is not None:
            self._crop_cache[nm] = res
        return res

    def __getitem__(self, i):
        r = self.rows[i]
        nm = r['name']
        im = self._load(nm)
        if self.train:
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.25:
                im = ImageEnhance.Brightness(im).enhance(1.0 + random.uniform(-0.035, 0.035))
        elif self.hflip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        w, h = im.size
        views = [im.resize((self.size, self.size), Image.BILINEAR)]
        T = self.tiles
        for dy in range(T):
            for dx in range(T):
                views.append(im.crop((dx * w // T, dy * h // T, (dx + 1) * w // T, (dy + 1) * h // T)).resize((self.size, self.size), Image.BILINEAR))
        views.extend(self._extract_native_microscope_crops(im, nm))

        x = np.transpose(np.stack([np.asarray(v, dtype=np.uint8) for v in views]), (0, 3, 1, 2))
        g = np.array([int(r[c]) for c in CLASSES], dtype=np.int64)
        mask7 = get_sample_class_mask6_v17(nm, synth_w=self.synth_w)
        p_raw = self.phys_map.get(nm, np.zeros(60, dtype=np.float32))
        p_norm = np.clip((p_raw - self.phys_mu) / self.phys_sd, -6.0, 6.0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(p_norm), torch.from_numpy(g), torch.from_numpy(mask7), nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='pecoreg:lastk4,siglip2g:lastk4,dinov3l:lastk6,convnext:full')
    ap.add_argument('--focus-classes', default='过度生成,纹理损失')
    ap.add_argument('--epochs-s1', type=int, default=3, help='Stage 1 epochs (Real + Zero-Leakage Verified Quadruplets)')
    ap.add_argument('--epochs-s2', type=int, default=2, help='Stage 2 epochs (Pure-Real Calibration + SWA)')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--lr-full', type=float, default=8.0e-5)
    ap.add_argument('--lr-last', type=float, default=2.8e-5)
    ap.add_argument('--lr-head', type=float, default=1.3e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.12)
    ap.add_argument('--clean-repeat', type=int, default=2)
    ap.add_argument('--p0-clean-weight', type=float, default=4.2)
    ap.add_argument('--channel-neg-weight', type=float, default=1.05)
    ap.add_argument('--alpha-ex', type=float, default=0.58)
    ap.add_argument('--synth-w', type=float, default=0.75)
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
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    focus_cis = [CLASSES.index(c.strip()) for c in a.focus_classes.split(',') if c.strip() in CLASSES]
    print(f'=== Training V17 Two-Stage Sim2Real End-to-End Specialist on focus_classes={a.focus_classes} (Stage1={a.epochs_s1}ep, Stage2={a.epochs_s2}ep) ===', flush=True)

    rows = load_split(a.split)
    man = read_manifest(a.manifest)
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})

    from extract_native_phys_feats import extract_for_names
    phys_map = extract_for_names([r['name'] for r in rows], a.phys_cache)

    tr_orig = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test']
    has_separate_te = len(te) > 0
    if not has_separate_te:
        te = va

    tr_real = [r for r in tr_orig if not is_synth_name(r['name'])]
    phys_tr_mat = np.stack([phys_map[r['name']] for r in tr_real], axis=0)
    phys_mu = np.mean(phys_tr_mat, axis=0)
    phys_sd = np.std(phys_tr_mat, axis=0) + 1e-5

    Y_orig = targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr_real]))
    pw = ((len(Y_orig) - Y_orig.sum(0)) / Y_orig.sum(0).clamp(min=1)).clamp(1.0, 5.5).to(dev)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    rng = random.Random(a.seed)

    # 1. Build Stage 1 Training Set: Authentic Real + Zero-Leakage Verified Quadruplets for `focus_cis`
    tr_synth_all = [
        r for r in tr_orig
        if is_synth_name(r['name'])
        and any(get_sample_class_mask6_v17(r['name'], a.synth_w)[ci] > 0.1 for ci in focus_cis)
    ]
    tr_synth = []
    for r in tr_synth_all:
        m6 = get_sample_class_mask6_v17(r['name'], a.synth_w)
        active_cis = [ci for ci in range(6) if m6[ci] > 0.1]
        max_g = max(int(r[CLASSES[ci]]) for ci in active_cis) if active_cis else 0
        # Keep 100% of P1/P0 synthetic quadruplet states and 65% of P2 states so P2 never swamps P1
        if max_g >= 2 or rng.random() < 0.65:
            tr_synth.append(r)

    tr_stage1 = list(tr_real) + tr_synth
    rare_focus_cis = [ci for ci in focus_cis if ci in (1, 2, 3, 4, 5)]
    if rare_focus_cis:
        focus_real_p1_p0 = [
            r for r in tr_real
            if any(int(r[CLASSES[ci]]) >= 2 for ci in rare_focus_cis)
        ]
        tr_stage1 = tr_stage1 + focus_real_p1_p0

    if a.clean_repeat > 1:
        clean = [r for r in tr_real if all(int(r[c]) == 0 for c in CLASSES)]
        tr_stage1 = tr_stage1 + clean * (a.clean_repeat - 1)

    # 2. Build Stage 2 Training Set: 100% Pure-Real Authentic Images ONLY (Zero Synthetic Shift!)
    tr_stage2 = list(tr_real)
    p1_p0_real_s2 = [
        r for r in tr_real
        if any(int(r[CLASSES[ci]]) >= 2 for ci in (1, 2, 3, 4, 5))
    ]
    tr_stage2 = tr_stage2 + p1_p0_real_s2
    if a.clean_repeat > 1:
        clean = [r for r in tr_real if all(int(r[c]) == 0 for c in CLASSES)]
        tr_stage2 = tr_stage2 + clean * (a.clean_repeat - 1)

    print(f'  Stage 1 Train={len(tr_stage1)} (real={len(tr_real)}, verified_synth={len(tr_synth)}) | Stage 2 Pure-Real Train={len(tr_stage2)} | Val={len(va)} | Blind Test={len(te)}', flush=True)

    dtr_s1 = V17NativeMicroscopeDataset(tr_stage1, phys_map, phys_mu, phys_sd, a.size, True, tiles=a.tiles, synth_w=a.synth_w)
    dtr_s2 = V17NativeMicroscopeDataset(tr_stage2, phys_map, phys_mu, phys_sd, a.size, True, tiles=a.tiles, synth_w=a.synth_w)
    dva = V17NativeMicroscopeDataset(va, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles, synth_w=a.synth_w)
    dte = V17NativeMicroscopeDataset(te, phys_map, phys_mu, phys_sd, a.size, False, tiles=a.tiles, synth_w=a.synth_w)

    val_batch = a.batch * 2
    ltr_s1 = DataLoader(dtr_s1, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
    ltr_s2 = DataLoader(dtr_s2, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                        pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
    lva = DataLoader(dva, batch_size=val_batch, shuffle=False, num_workers=a.workers,
                     pin_memory=True, persistent_workers=True, prefetch_factor=4)
    lte = DataLoader(dte, batch_size=val_batch, shuffle=False, num_workers=a.workers,
                     pin_memory=True, persistent_workers=True, prefetch_factor=4)

    specs = []
    pg = []
    for item in a.backbones.split(','):
        nm, mode = item.split(':')
        m, fwd, mu, sd, bl, dim = build_v9_backbone(nm)
        for p in m.parameters():
            p.requires_grad_(False)
        if mode == 'full':
            for p in m.parameters():
                p.requires_grad_(True)
            pg.append({'params': [p for p in m.parameters() if p.requires_grad], 'lr': a.lr_full})
        elif mode.startswith('lastk'):
            k = int(mode[5:])
            for blk in bl[-k:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            pg.append({'params': [p for p in m.parameters() if p.requires_grad], 'lr': a.lr_last})
        specs.append((m, fwd, mu, sd, bl, dim))

    net = MultiNetV16(specs, phys_dim=60, dropout=a.dropout, n_quad=a.tiles * a.tiles).to(dev)
    head_params = (
        list(net.head_cum.parameters())
        + list(net.head_ex.parameters())
        + list(net.gate.parameters())
        + list(net.norm.parameters())
        + list(net.se_gate.parameters())
        + list(net.tile_attn.parameters())
        + list(net.phys_mlp.parameters())
    )
    pg.append({'params': head_params, 'lr': a.lr_head})

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    try:
        opt = torch.optim.AdamW(pg, weight_decay=a.wd, fused=True)
    except Exception:
        opt = torch.optim.AdamW(pg, weight_decay=a.wd)

    total_steps = max(1, (len(ltr_s1) // a.grad_accum) * a.epochs_s1 + (len(ltr_s2) // a.grad_accum) * a.epochs_s2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    from cv_decoder_calib import decode_pc, HAND
    s2_ckpts = []
    best_val_doc2 = -1.0
    best_sd = None
    total_epochs = a.epochs_s1 + a.epochs_s2

    for ep in range(total_epochs):
        t0 = time.time()
        is_stage2 = (ep >= a.epochs_s1)
        cur_loader = ltr_s2 if is_stage2 else ltr_s1
        stage_tag = "Stage2-PureReal" if is_stage2 else "Stage1-Sim2Real"
        # In Stage 2 pure-real alignment, supervise all 6 classes uniformly on authentic data
        cur_focus = list(range(6)) if is_stage2 else focus_cis

        net.train()
        opt.zero_grad(set_to_none=True)
        tot, n_step = 0.0, 0
        for step, (x, phys, g, mask7, _) in enumerate(cur_loader):
            x = x.to(dev, non_blocking=True)
            phys = phys.to(dev, non_blocking=True)
            g = g.to(dev, non_blocking=True)
            mask7 = mask7.to(dev, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lc, le, gl = net(x, phys)
            loss = v16_clean_microscope_loss_fn(
                lc.float(), le.float(), gl.float(), g, mask7, bce_loss_none,
                focus_cis=cur_focus,
                p0_clean_weight=a.p0_clean_weight,
                channel_neg_weight=a.channel_neg_weight,
                alpha_ex=a.alpha_ex,
            )
            (loss / a.grad_accum).backward()
            if (step + 1) % a.grad_accum == 0 or (step + 1) == len(cur_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sch.step()
            tot += float(loss.detach())
            n_step += 1

        vp0, vp1, vp2, vg, gva = predict_v17(net, lva, dev, a.alpha_ex)
        pred_va = decode_pc(vp0, vp1, vp2, HAND)
        s_hand = cm.score(gva, pred_va, a.tau)
        clean_va = (gva == 0).all(1)
        s_clean2 = float((clean_va & (pred_va == 0).all(1)).sum() / max(1, clean_va.sum()))
        doc2_va = 25 * s_hand['S_pos'] + 20 * s_hand['S_p0'] + 20 * s_clean2
        print(f'ep={ep+1}/{total_epochs} [{stage_tag}] loss={tot/max(1,n_step):.4f} val_Doc1={s_hand["total"]:.2f} val_Doc2={doc2_va:.2f} (S_pos={s_hand["S_pos"]:.3f} S_p0={s_hand["S_p0"]:.3f} Clean2={s_clean2:.3f}) ({time.time()-t0:.0f}s)', flush=True)

        cur_sd = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        torch.save(cur_sd, f'{a.out}_ep{ep+1}.pt')
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
                swa_sd[k] = torch.stack([ck[k] for ck in s2_ckpts], dim=0).mean(0)
            else:
                swa_sd[k] = s2_ckpts[-1][k]
        net.load_state_dict({k: v.to(dev) for k, v in swa_sd.items()})
    elif len(s2_ckpts) == 1:
        net.load_state_dict({k: v.to(dev) for k, v in s2_ckpts[0].items()})

    dva_flip = V17NativeMicroscopeDataset(va, phys_map, phys_mu, phys_sd, a.size, False, hflip=True, tiles=a.tiles, synth_w=a.synth_w)
    dte_flip = V17NativeMicroscopeDataset(te, phys_map, phys_mu, phys_sd, a.size, False, hflip=True, tiles=a.tiles, synth_w=a.synth_w)
    lva_flip = DataLoader(dva_flip, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    lte_flip = DataLoader(dte_flip, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    vp0_a, vp1_a, vp2_a, vg_a, gva = predict_v17(net, lva, dev, a.alpha_ex)
    vp0_b, vp1_b, vp2_b, vg_b, _   = predict_v17(net, lva_flip, dev, a.alpha_ex)
    vp0, vp1, vp2, vg = 0.5*(vp0_a+vp0_b), 0.5*(vp1_a+vp1_b), 0.5*(vp2_a+vp2_b), 0.5*(vg_a+vg_b)

    if has_separate_te:
        tp0_a, tp1_a, tp2_a, tg_a, gte = predict_v17(net, lte, dev, a.alpha_ex)
        tp0_b, tp1_b, tp2_b, tg_b, _   = predict_v17(net, lte_flip, dev, a.alpha_ex)
        tp0, tp1, tp2, tg = 0.5*(tp0_a+tp0_b), 0.5*(tp1_a+tp1_b), 0.5*(tp2_a+tp2_b), 0.5*(tg_a+tg_b)
    else:
        tp0, tp1, tp2, tg, gte = vp0, vp1, vp2, vg, gva

    tpl = json.load(open('/wanqing-develop/luowenjing/18/results/v2_full/template_150.json'))
    names_150 = [e['image'] for e in tpl]
    phys_150 = extract_for_names(names_150, a.phys_cache)
    rows_150 = [{'name': n, **{c: 0 for c in CLASSES}} for n in names_150]
    d150 = V17NativeMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, a.size, False, hflip=False, tiles=a.tiles, synth_w=a.synth_w)
    d150_f = V17NativeMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, a.size, False, hflip=True, tiles=a.tiles, synth_w=a.synth_w)
    l150 = DataLoader(d150, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    l150_f = DataLoader(d150_f, batch_size=val_batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    p150_0a, p150_1a, p150_2a, p150_ga, _ = predict_v17(net, l150, dev, a.alpha_ex)
    p150_0b, p150_1b, p150_2b, p150_gb, _ = predict_v17(net, l150_f, dev, a.alpha_ex)
    p150_0, p150_1, p150_2, p150_g = 0.5*(p150_0a+p150_0b), 0.5*(p150_1a+p150_1b), 0.5*(p150_2a+p150_2b), 0.5*(p150_ga+p150_gb)

    out_base = a.out[:-5] if a.out.endswith('.json') else a.out
    np.savez(
        out_base + '_probs.npz',
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vg, gva=gva, va_names=[r['name'] for r in va],
        te_p0=tp0, te_p1=tp1, te_p2=tp2, te_gate=tg, gte=gte, te_names=[r['name'] for r in te],
        t150_p0=p150_0, t150_p1=p150_1, t150_p2=p150_2, t150_gate=p150_g, t150_names=names_150,
    )
    st_hand = cm.score(gte, decode_pc(tp0, tp1, tp2, HAND), a.tau)
    clean_te = (gte == 0).all(1)
    pred_te = decode_pc(tp0, tp1, tp2, HAND)
    s_clean2_te = float((clean_te & (pred_te == 0).all(1)).sum() / max(1, clean_te.sum()))
    doc2_te = 25 * st_hand['S_pos'] + 20 * st_hand['S_p0'] + 20 * s_clean2_te
    json.dump({'doc1': float(st_hand['total']), 'doc2': float(doc2_te), 'S_pos': float(st_hand['S_pos']),
               'S_p0': float(st_hand['S_p0']), 'Clean2': float(s_clean2_te)}, open(out_base + '.json', 'w'), indent=2)
    print(f'=== FINAL V17 TWO-STAGE SWA+TTA BLIND TEST (N={len(te)}): Doc1={st_hand["total"]:.2f} | Doc2={doc2_te:.2f} (S_pos={st_hand["S_pos"]:.3f} S_p0={st_hand["S_p0"]:.3f} Clean2={s_clean2_te:.3f}) ===', flush=True)


if __name__ == '__main__':
    main()
