#!/usr/bin/env python3
"""
code/finetune/train_v12_abmil_pure_real.py
Stage C Engine: End-to-End 10-View ViT Trainer with:
1. Token-Gated ABMIL + Top-5% Anomaly Token Pooling + Spatial Std (via `*_abmil` backbones).
2. Dual-Layer Tap (Layer 75% high-frequency sharpness/texture + Layer 100% semantic overgen).
3. Real-to-Real Latent Tile CutMix (recombines genuine defect tiles across real images in latent space).
4. 100% Pure Real Human-Annotated Images Only + Automatic 2x Flip TTA on test_150.
"""
import argparse, copy, csv, json, os, random, sys, time
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

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'decision_eval'))
from data import IQADataset, CachedIQADataset, load_split, read_manifest, targets_from_grades, CLASSES, FULL
from models import build_backbone
from train_v2 import (MultiNetV2, exact_probs_from_dual, exact_targets_14,
                      predict_v2, decode_scaled_argmax, optimise_scaled, P0_IDX)
from train_v3 import v3_loss_fn
import comp_metrics as cm

CACHE_OK = os.path.exists('/dev/shm/race18_native/index.npz')
TEST_IMG_DIR = 'data/test_150'


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class MultiNetV12_ABMIL(MultiNetV2):
    """
    Extends MultiNetV2 with Real-to-Real Latent Tile CutMix during training.
    """
    def forward_with_tile_cutmix(self, x, g, p_cutmix: float = 0.35):
        B, V, C, H, W = x.shape
        feats_per_bb = []
        for i, (m, fwd) in enumerate(zip(self.bs, self.fwds)):
            mu = getattr(self, 'mu_%d' % (i + 1))
            sd = getattr(self, 'sd_%d' % (i + 1))
            dt = next(m.parameters()).dtype
            z = ((x.view(B * V, C, H, W) - mu) / sd).to(dt)
            feat = fwd(z).view(B, V, -1).float()
            feats_per_bb.append(feat)

        if self.training and B >= 2 and V >= 5 and random.random() < p_cutmix:
            perm = torch.randperm(B, device=x.device)
            # Identify top-2 most salient local tiles per image using the first backbone's tile_attn
            with torch.no_grad():
                tile_scores = self.tile_attn[0](feats_per_bb[0][:, 1:]).squeeze(-1)  # [B, V-1]
                _, top2_tiles = torch.topk(tile_scores[perm], k=min(2, V - 1), dim=1) # [B, 2]
            for b_idx in range( len(feats_per_bb) ):
                f_b = feats_per_bb[b_idx].clone()
                f_perm = feats_per_bb[b_idx][perm]
                for b_row in range(B):
                    for t_col in top2_tiles[b_row]:
                        f_b[b_row, 1 + t_col] = f_perm[b_row, 1 + t_col]
                feats_per_bb[b_idx] = f_b
            g_mixed = torch.maximum(g, g[perm])
        else:
            g_mixed = g

        out = [self._agg(feat, attn_mod) for feat, attn_mod in zip(feats_per_bb, self.tile_attn)]
        fused = self.drop(self.norm(torch.cat(out, dim=1)))
        return self.head_cum(fused), self.head_ex(fused), self.gate(fused).squeeze(1), g_mixed


@torch.no_grad()
def infer_150_tta(net, ordered_names, dev, size, tiles, alpha_ex=0.45):
    net.eval()
    test_rows = [{'name': nm, **{c: 0 for c in CLASSES}} for nm in ordered_names]
    ds_150 = IQADataset(test_rows, TEST_IMG_DIR, TEST_IMG_DIR, set(ordered_names), size=size, train=False, tiles=tiles)
    dl_150 = DataLoader(ds_150, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    P0, P1, P2 = [], [], []
    for x, _, _ in dl_150:
        x = x.to(dev)
        x_flip = torch.flip(x, dims=[4])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lc1, le1, _ = net(x)
            lc2, le2, _ = net(x_flip)
        p0_1, p1_1, p2_1, _ = exact_probs_from_dual(lc1.float(), le1.float(), alpha_ex=alpha_ex)
        p0_2, p1_2, p2_2, _ = exact_probs_from_dual(lc2.float(), le2.float(), alpha_ex=alpha_ex)
        P0.append((0.5 * (p0_1 + p0_2)).cpu().numpy())
        P1.append((0.5 * (p1_1 + p1_2)).cpu().numpy())
        P2.append((0.5 * (p2_1 + p2_2)).cpu().numpy())
    return np.concatenate(P0, 0), np.concatenate(P1, 0), np.concatenate(P2, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='internvit_abmil:lastk4,pecoreg_abmil:lastk4,dinov3l_abmil:lastk4')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--lr-full', type=float, default=8.875e-5)
    ap.add_argument('--lr-last', type=float, default=1.5e-4)
    ap.add_argument('--lr-head', type=float, default=1.2e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--clean-repeat', type=int, default=4)
    ap.add_argument('--p0-clean-weight', type=float, default=4.5)
    ap.add_argument('--alpha-ex', type=float, default=0.45)
    ap.add_argument('--gen-repeat', type=int, default=2)
    ap.add_argument('--emd-weight', type=float, default=0.45)
    ap.add_argument('--p1-ce-weight', type=float, default=1.15)
    ap.add_argument('--split', default='data/resplits/split_s16.csv')
    ap.add_argument('--manifest', default='data/manifest_combined.csv')
    ap.add_argument('--tau', type=float, default=0.02)
    ap.add_argument('--out', default='results/v12_abmil/v12_abmil_s16.json')
    a = ap.parse_args()

    os.environ['INTERNVIT_SIZE'] = str(a.size)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    set_seed(a.seed)
    dev = 'cuda'

    specs = []
    for s in a.backbones.split(','):
        name, mode = s.split(':')
        b = build_backbone(name)
        specs.append((b, name, mode))

    rows = load_split(a.split)
    man = read_manifest(a.manifest)
    rows = [r for r in rows if r['name'] in man and not r['name'].startswith(('v9_', 'v8_', 'v7_', 'v6_'))]
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})
    tr = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test']
    if not te:
        te = va

    if a.clean_repeat > 1:
        clean = [r for r in tr if all(int(r[c]) == 0 for c in CLASSES)]
        tr = tr + clean * (a.clean_repeat - 1)

    if a.gen_repeat > 1:
        gen = [r for r in tr if int(r['过度生成']) > 0 or int(r['虚化突兀']) > 0]
        tr = tr + gen * (a.gen_repeat - 1)

    orig = set(r['name'] for r in csv.DictReader(open('data/manifest.csv')))
    def mk(rows_, tr_):
        return (CachedIQADataset(rows_, a.size, tr_, tiles=a.tiles)
                if CACHE_OK else IQADataset(rows_, 'data/images', 'data/images_supp', orig, a.size, tr_, tiles=a.tiles))

    dtr = mk(tr, True)
    dva = mk(va, False)
    dte = mk(te, False)
    ltr = DataLoader(dtr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                     pin_memory=True, drop_last=True, worker_init_fn=lambda w: set_seed(a.seed + w))
    lva = DataLoader(dva, batch_size=a.batch, shuffle=False, num_workers=a.workers)
    lte = DataLoader(dte, batch_size=a.batch, shuffle=False, num_workers=a.workers)

    net = MultiNetV12_ABMIL([x[0] for x in specs], dropout=a.dropout).to(dev)
    groups = []
    abmil_params = []
    for (b, name, mode) in specs:
        m = b[0]
        blocks = b[4]
        for p in m.parameters():
            p.requires_grad = (mode == 'full')
        if mode.startswith('lastk') and blocks is not None:
            k = int(mode[len('lastk'):] or 4)
            for bl in blocks[-k:]:
                for p in bl.parameters():
                    p.requires_grad = True
        # Always keep TokenGatedABMILPool & DualLayerTapWrapper trainable at head LR
        abmil_ids = set()
        for attr in ('abmil_pool', 'dual_tap'):
            if hasattr(m, attr):
                subm = getattr(m, attr)
                subm.float()
                for p in subm.parameters():
                    p.requires_grad = True
                    abmil_params.append(p)
                    abmil_ids.add(id(p))
        bb_params = [p for p in m.parameters() if p.requires_grad and id(p) not in abmil_ids]
        lr = a.lr_full if mode == 'full' else a.lr_last
        if bb_params:
            groups.append({'params': bb_params, 'lr': lr})

    head_params = (
        abmil_params +
        list(net.tile_attn.parameters()) +
        list(net.norm.parameters()) +
        list(net.head_cum.parameters()) +
        list(net.head_ex.parameters()) +
        list(net.gate.parameters())
    )
    groups.append({'params': head_params, 'lr': a.lr_head})
    tr_p = [p for g in groups for p in g['params']]
    print(f'[V12 ABMIL+DualLayer] trainable {sum(p.numel() for p in tr_p)/1e6:.1f}M / {sum(p.numel() for p in net.parameters())/1e6:.1f}M '
          f'(size={a.size}, tiles={a.tiles}, views={1 + a.tiles*a.tiles})', flush=True)

    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    Y = targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr]))
    pw = ((len(Y) - Y.sum(0)) / Y.sum(0).clamp(min=1)).clamp(1.0, 8.0).to(dev)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    accum = max(1, a.grad_accum)
    steps_per_ep = max(1, len(ltr) // accum)
    total_steps = steps_per_ep * a.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=total_steps, pct_start=0.1)

    best = None
    for ep in range(a.epochs):
        net.train()
        t0 = time.time()
        tot = 0.0
        nb = 0
        opt.zero_grad()
        for step_idx, (x, g, _) in enumerate(ltr):
            x = x.to(dev)
            g = g.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lc, le, gl, g_mix = net.forward_with_tile_cutmix(x, g, p_cutmix=0.30)
            loss = v3_loss_fn(lc.float(), le.float(), gl.float(), g_mix, bce_loss_none,
                              p0_clean_weight=a.p0_clean_weight, emd_weight=a.emd_weight, p1_ce_weight=a.p1_ce_weight)
            (loss / accum).backward()
            tot += loss.item()
            nb += 1
            if (step_idx + 1) % accum == 0 or (step_idx + 1) == len(ltr):
                torch.nn.utils.clip_grad_norm_(tr_p, 1.0)
                opt.step()
                if sch.last_epoch < total_steps - 1:
                    sch.step()
                opt.zero_grad()

        vp0, vp1, vp2, vgt, gv = predict_v2(net, lva, dev, alpha_ex=a.alpha_ex)
        b_cas, th_cas = cm.optimise(gv, vp0, vp1, vp2, a.tau)
        b_sca, th_sca = optimise_scaled(gv, vp0, vp1, vp2, a.tau)
        print(f'ep{ep+1} loss={tot/max(1,nb):.3f} val_cas={b_cas["total"]:.2f} val_sca={b_sca["total"]:.2f} '
              f'S_pos={b_sca["S_pos"]:.3f} P0R={b_sca["p0R"]:.3f} S_clean={b_sca["S_clean"]:.3f} t={time.time()-t0:.0f}s', flush=True)
        score_ep = max(b_cas['total'], b_sca['total'])
        cur_state = copy.deepcopy(net.state_dict())
        if best is None or score_ep > best[0]:
            best = (score_ep, th_cas, th_sca, cur_state)

    net.load_state_dict(best[3])
    vp0, vp1, vp2, vgt, gva_n = predict_v2(net, lva, dev, alpha_ex=a.alpha_ex)
    tp0, tp1, tp2, tgt, gte_n = predict_v2(net, lte, dev, alpha_ex=a.alpha_ex)

    st_cas = cm.score(gte_n, cm.decode(tp0, tp1, tp2, *best[1]), a.tau)
    st_sca = cm.score(gte_n, decode_scaled_argmax(tp0, tp1, tp2, *best[2]), a.tau)

    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names_150 = [e['image'] for e in tpl]
    t150_p0, t150_p1, t150_p2 = infer_150_tta(net, ordered_names_150, dev, a.size, a.tiles, alpha_ex=a.alpha_ex)
    print(f'Inferred test_150 (2x Flip TTA): p0_mean={t150_p0.mean():.4f}, p1_mean={t150_p1.mean():.4f}, p2_mean={t150_p2.mean():.4f}', flush=True)

    res = {
        'backbones': a.backbones,
        'seed': a.seed,
        'size': a.size,
        'tiles': a.tiles,
        'val_total': best[0],
        'test': {k: st_sca[k] for k in ['total', 'S_pos', 'p0R', 'p0P', 'p0F', 'S_p0', 'S_clean', 'fpr']},
        'test_cascade': {k: st_cas[k] for k in ['total', 'S_pos', 'p0R', 'p0P', 'p0F', 'S_p0', 'S_clean', 'fpr']},
    }
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
    np.savez_compressed(
        a.out.replace('.json', '_probs.npz'),
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vgt, gva=gva_n, va_names=np.array([r['name'] for r in va]),
        te_p0=tp0, te_p1=tp1, te_p2=tp2, te_gate=tgt, gte=gte_n, te_names=np.array([r['name'] for r in te]),
        t150_p0=t150_p0, t150_p1=t150_p1, t150_p2=t150_p2, names_150=np.array(ordered_names_150)
    )
    print('SAVED V12 ABMIL:', a.out, flush=True)


if __name__ == '__main__':
    main()
