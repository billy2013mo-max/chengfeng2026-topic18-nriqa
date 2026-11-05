"""Improved multi-backbone trainer (train_v2.py):
1. Fixes the Y.numel() pos_weight bug (which set pos_weight=20 on all 14 channels and caused
   massive P2/P1 cross-defect false positives on non-clean images) and fixes P0_IDX=[2,5,8,9,12,13].
2. Dual-Head Architecture: combines 14-dim Cumulative Ordinal BCE with 20-dim Direct 4-Way
   Multinomial Cross-Entropy + Differentiable Soft Macro-F1 Loss on the 14 exact competition labels.
3. P0-Specific Clean Suppression: penalizes Clean images specifically on the 6 P0 channels
   instead of crushing P2/P1 recall across the board.
4. Gated Attention Tile Pooling: augments [global, tile_mean, tile_max] with learned attention
   pooling over native-resolution tiles for local defects (over-sharpening, over-generation, bokeh).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'decision_eval'))
from data import IQADataset, CachedIQADataset, load_split, read_manifest, targets_from_grades, CLASSES, FULL
from models import build_backbone
import comp_metrics as cm

CACHE_OK = os.path.exists('/dev/shm/race18_native/index.npz')
# Correct P0 indices in targets_from_grades (14 columns):
# 0,1,2 (模糊 P2,P1,P0) | 3,4,5 (纹理损失 P2,P1,P0) | 6,7,8 (过锐 P2,P1,P0) | 9 (花屏 P0) | 10,11,12 (过度生成 P2,P1,P0) | 13 (虚化突兀 P0)
P0_IDX = [2, 5, 8, 9, 12, 13]
FULL_IDX = cm.FULL_IDX
BIN_IDX = cm.BIN_IDX


class MultiNetV2(nn.Module):
    def __init__(self, specs, dropout=0.1):
        super().__init__()
        self.bs = nn.ModuleList()
        self.fwds = []
        self.tile_attn = nn.ModuleList()
        for (m, fwd, mu, sd, bl, dim) in specs:
            self.bs.append(m)
            self.fwds.append(fwd)
            self.register_buffer('mu_%d' % len(self.fwds), torch.tensor(mu).view(1, 3, 1, 1))
            self.register_buffer('sd_%d' % len(self.fwds), torch.tensor(sd).view(1, 3, 1, 1))
            self.tile_attn.append(nn.Sequential(
                nn.Linear(dim, max(64, dim // 8)),
                nn.GELU(),
                nn.Linear(max(64, dim // 8), 1)
            ))
        # Each backbone produces 4 * dim: [global, tile_mean, tile_max, tile_attn_pool]
        d = sum(4 * x[5] for x in specs)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        # Head 1: 14 cumulative ordinal logits
        self.head_cum = nn.Linear(d, 14)
        # Head 2: 20 exact-grade multinomial logits (4*4 for FULL + 2*2 for BIN)
        self.head_ex = nn.Sequential(
            nn.Linear(d, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 20)
        )
        self.gate = nn.Linear(d, 1)

    def _agg(self, f, attn_mod):
        # f: (B, V, D) where V >= 2
        g_view = f[:, 0]
        tiles = f[:, 1:]
        t_mean = tiles.mean(1)
        t_max = tiles.amax(1)
        w = torch.softmax(attn_mod(tiles), dim=1)  # (B, V-1, 1)
        t_att = (tiles * w).sum(1)
        return torch.cat([g_view, t_mean, t_max, t_att], dim=1)

    def forward(self, x):
        B, V, C, H, W = x.shape
        out = []
        for i, (m, fwd, attn_mod) in enumerate(zip(self.bs, self.fwds, self.tile_attn)):
            mu = getattr(self, 'mu_%d' % (i + 1))
            sd = getattr(self, 'sd_%d' % (i + 1))
            dt = next(m.parameters()).dtype
            z = ((x.view(B * V, C, H, W) - mu) / sd).to(dt)
            feat = fwd(z).view(B, V, -1).float()
            out.append(self._agg(feat, attn_mod))
        fused = self.drop(self.norm(torch.cat(out, dim=1)))
        return self.head_cum(fused), self.head_ex(fused), self.gate(fused).squeeze(1)


def exact_probs_from_dual(logits_cum, logits_ex, alpha_ex=0.45):
    """Combines cumulative ordinal probabilities with direct multinomial 4-way probabilities
    into the 14 exact competition label probabilities (and returns p0, p1, p2 tensors)."""
    s_cum = torch.sigmoid(logits_cum)
    p0_list, p1_list, p2_list = [], [], []
    exact14 = []
    j_cum = 0
    j_ex = 0
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            s2, s1, s0 = s_cum[:, j_cum], s_cum[:, j_cum + 1], s_cum[:, j_cum + 2]
            j_cum += 3
            c_p2 = (s2 - s1).clamp(min=0.0)
            c_p1 = (s1 - s0).clamp(min=0.0)
            c_p0 = s0

            ex_probs = torch.softmax(logits_ex[:, j_ex:j_ex + 4], dim=1)
            j_ex += 4
            e_p2, e_p1, e_p0 = ex_probs[:, 1], ex_probs[:, 2], ex_probs[:, 3]

            p2 = (1.0 - alpha_ex) * c_p2 + alpha_ex * e_p2
            p1 = (1.0 - alpha_ex) * c_p1 + alpha_ex * e_p1
            p0 = (1.0 - alpha_ex) * c_p0 + alpha_ex * e_p0

            p0_list.append(p0)
            p1_list.append(p1)
            p2_list.append(p2)
            exact14.extend([p2, p1, p0])
        else:
            c_p0 = s_cum[:, j_cum]
            j_cum += 1
            ex_probs = torch.softmax(logits_ex[:, j_ex:j_ex + 2], dim=1)
            j_ex += 2
            e_p0 = ex_probs[:, 1]
            p0 = (1.0 - alpha_ex) * c_p0 + alpha_ex * e_p0
            p0_list.append(p0)
            exact14.append(p0)
    return (
        torch.stack(p0_list, dim=1),
        torch.stack(p1_list, dim=1),
        torch.stack(p2_list, dim=1),
        torch.stack(exact14, dim=1),
    )


def exact_targets_14(g):
    """g: (B, 6) int -> (B, 14) exact binary targets matching exact14 order."""
    out = []
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            out += [(g[:, ci] == 1).float(), (g[:, ci] == 2).float(), (g[:, ci] == 3).float()]
        else:
            out += [(g[:, ci] == 3).float()]
    return torch.stack(out, dim=1)


def dual_loss_fn(logits_cum, logits_ex, gate_logit, g, bce_loss_none, p0_clean_weight=5.0):
    # 1. Cumulative BCE loss with P0-specific clean weighting
    Y_cum = targets_from_grades(g)
    raw_bce = bce_loss_none(logits_cum, Y_cum)  # (B, 14)
    clean_mask = (g == 0).all(dim=1)            # (B,)
    if clean_mask.any() and p0_clean_weight > 1.0:
        w_mat = torch.ones_like(raw_bce)
        for idx in P0_IDX:
            w_mat[clean_mask, idx] = p0_clean_weight
        loss_bce = (raw_bce * w_mat).mean()
    else:
        loss_bce = raw_bce.mean()

    # 2. Monotonicity penalty on cumulative logits: s0 <= s1 <= s2
    mono_pen = 0.0
    j = 0
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            l2, l1, l0 = logits_cum[:, j], logits_cum[:, j + 1], logits_cum[:, j + 2]
            j += 3
            mono_pen = mono_pen + F.relu(l0 - l1).mean() + F.relu(l1 - l2).mean()
        else:
            j += 1

    # 3. Direct 4-way / 2-way Cross-Entropy loss on logits_ex
    ce_tot = 0.0
    j_ex = 0
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            ce_tot = ce_tot + F.cross_entropy(logits_ex[:, j_ex:j_ex + 4], g[:, ci].long(), label_smoothing=0.03)
            j_ex += 4
        else:
            tgt_bin = (g[:, ci] == 3).long()
            ce_tot = ce_tot + F.cross_entropy(logits_ex[:, j_ex:j_ex + 2], tgt_bin, label_smoothing=0.02)
            j_ex += 2
    loss_ce = ce_tot / len(CLASSES)

    # 4. Differentiable Soft Macro-F1 Loss on the 14 exact competition labels
    p0, p1, p2, P_exact14 = exact_probs_from_dual(logits_cum, logits_ex, alpha_ex=0.45)
    Y_exact14 = exact_targets_14(g)
    tp = (P_exact14 * Y_exact14).sum(dim=0)
    fp = (P_exact14 * (1.0 - Y_exact14)).sum(dim=0)
    fn = ((1.0 - P_exact14) * Y_exact14).sum(dim=0)
    soft_f1 = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
    loss_f1 = 1.0 - soft_f1.mean()

    # 5. Clean P0 max-probability suppression
    loss_clean_p0 = 0.0
    if clean_mask.any():
        loss_clean_p0 = p0[clean_mask].amax(dim=1).mean()

    # 6. Clean gate auxiliary loss
    cl_t = clean_mask.float()
    loss_gate = F.binary_cross_entropy_with_logits(gate_logit, cl_t)

    return loss_bce + 0.6 * loss_ce + 0.5 * loss_f1 + 0.2 * mono_pen + 0.4 * loss_clean_p0 + 0.1 * loss_gate


@torch.no_grad()
def predict_v2(net, dl, dev, alpha_ex=0.45):
    net.eval()
    P0, P1, P2, GT, G = [], [], [], [], []
    for x, g, _ in dl:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lc, le, gt = net(x.to(dev))
        p0, p1, p2, _ = exact_probs_from_dual(lc.float(), le.float(), alpha_ex=alpha_ex)
        P0.append(p0.cpu().numpy())
        P1.append(p1.cpu().numpy())
        P2.append(p2.cpu().numpy())
        GT.append(gt.float().cpu().numpy())
        G.append(g.numpy().astype(int))
    return (
        np.concatenate(P0, 0),
        np.concatenate(P1, 0),
        np.concatenate(P2, 0),
        np.concatenate(GT, 0),
        np.concatenate(G, 0),
    )


def decode_scaled_argmax(p0, p1, p2, t0, t1, t2, r12=1.0):
    n = p0.shape[0]
    pred = np.zeros((n, 6), int)
    cond_p2 = p2 > t2
    cond_p1 = (p1 > t1) & ((p1 * r12) >= (p2 - 0.05))
    g_full = np.where(cond_p2, 1, 0)
    g_full = np.where(cond_p1, 2, g_full)
    g_full = np.where(p0[:, FULL_IDX] > t0, 3, g_full)
    pred[:, FULL_IDX] = g_full
    pred[:, BIN_IDX] = np.where(p0[:, BIN_IDX] > t0, 3, 0)
    return pred


def optimise_scaled(gold, p0, p1, p2, tau=0.02):
    best = None
    for t0 in np.arange(0.25, 0.85, 0.05):
        for t1 in np.arange(0.15, 0.70, 0.05):
            for t2 in np.arange(0.10, 0.65, 0.05):
                for r12 in (0.85, 1.0, 1.2):
                    s = cm.score(gold, decode_scaled_argmax(p0, p1, p2, t0, t1, t2, r12), tau)
                    if best is None or s['total'] > best[0]['total']:
                        best = (s, (t0, t1, t2, r12))
    return best


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='siglip2:lastk4,cradio:lastk4,pecorel:lastk4')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--lr-full', type=float, default=8.875e-5)
    ap.add_argument('--lr-last', type=float, default=3e-5)
    ap.add_argument('--lr-head', type=float, default=1.5e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--clean-repeat', type=int, default=4)
    ap.add_argument('--p0-clean-weight', type=float, default=4.0)
    ap.add_argument('--alpha-ex', type=float, default=0.45)
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--split', default='data/resplits/split_s16.csv')
    ap.add_argument('--manifest', default='data/manifest_combined.csv')
    ap.add_argument('--tau', type=float, default=0.02)
    ap.add_argument('--out', default='results/v2/run.json')
    ap.add_argument('--save-state', default='', help='path to save checkpoint')
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    set_seed(a.seed)
    dev = 'cuda'

    specs = []
    for s in a.backbones.split(','):
        name, mode = s.split(':')
        b = build_backbone(name)
        specs.append((b, name, mode))

    rows = load_split(a.split)
    man = read_manifest(a.manifest)
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})
    tr = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test']
    if not te:
        te = va
        print('WARNING: split has no test rows, scoring on val', flush=True)

    if a.clean_repeat > 1:
        clean = [r for r in tr if all(int(r[c]) == 0 for c in CLASSES)]
        tr = tr + clean * (a.clean_repeat - 1)

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

    net = MultiNetV2([x[0] for x in specs], dropout=a.dropout).to(dev)
    groups = []
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
        lr = a.lr_full if mode == 'full' else a.lr_last
        groups.append({'params': [p for p in m.parameters() if p.requires_grad], 'lr': lr})

    head_params = (
        list(net.tile_attn.parameters()) +
        list(net.norm.parameters()) +
        list(net.head_cum.parameters()) +
        list(net.head_ex.parameters()) +
        list(net.gate.parameters())
    )
    groups.append({'params': head_params, 'lr': a.lr_head})
    tr_p = [p for g in groups for p in g['params']]
    print(f'trainable {sum(p.numel() for p in tr_p)/1e6:.1f}M / {sum(p.numel() for p in net.parameters())/1e6:.1f}M', flush=True)

    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    Y = targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr]))
    # TRUE positive weight using len(Y), not Y.numel()!
    pw = ((len(Y) - Y.sum(0)) / Y.sum(0).clamp(min=1)).clamp(1.0, 8.0).to(dev)
    print('pos_weight per channel:', [round(float(x), 2) for x in pw], flush=True)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    steps = len(ltr) * a.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=max(1, steps), pct_start=0.1)

    best = None
    for ep in range(a.epochs):
        net.train()
        t0 = time.time()
        tot = 0.0
        nb = 0
        for x, g, _ in ltr:
            x = x.to(dev)
            g = g.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lc, le, gl = net(x)
            loss = dual_loss_fn(lc.float(), le.float(), gl.float(), g, bce_loss_none, p0_clean_weight=a.p0_clean_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tr_p, 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            tot += loss.item()
            nb += 1

        vp0, vp1, vp2, vgt, gv = predict_v2(net, lva, dev, alpha_ex=a.alpha_ex)
        b_cas, th_cas = cm.optimise(gv, vp0, vp1, vp2, a.tau)
        b_sca, th_sca = optimise_scaled(gv, vp0, vp1, vp2, a.tau)
        print(f'ep{ep+1} loss={tot/max(1,nb):.3f} val_cas={b_cas["total"]:.2f} val_sca={b_sca["total"]:.2f} '
              f'S_pos={b_sca["S_pos"]:.3f} P0R={b_sca["p0R"]:.3f} S_clean={b_sca["S_clean"]:.3f} t={time.time()-t0:.0f}s', flush=True)
        score_ep = max(b_cas['total'], b_sca['total'])
        if best is None or score_ep > best[0]:
            best = (score_ep, th_cas, th_sca, copy.deepcopy(net.state_dict()))

    net.load_state_dict(best[3])
    if a.save_state:
        os.makedirs(os.path.dirname(a.save_state), exist_ok=True)
        torch.save({
            'state': best[3],
            'backbones': a.backbones,
            'size': a.size,
            'alpha_ex': a.alpha_ex,
            'thresholds_cascade': [float(x) for x in best[1]],
            'thresholds_scaled': [float(x) for x in best[2]],
        }, a.save_state)
        print('state saved', a.save_state, flush=True)
    vp0, vp1, vp2, vgt, gva_n = predict_v2(net, lva, dev, alpha_ex=a.alpha_ex)
    tp0, tp1, tp2, tgt, gte_n = predict_v2(net, lte, dev, alpha_ex=a.alpha_ex)

    st_cas = cm.score(gte_n, cm.decode(tp0, tp1, tp2, *best[1]), a.tau)
    st_sca = cm.score(gte_n, decode_scaled_argmax(tp0, tp1, tp2, *best[2]), a.tau)

    res = {
        'backbones': a.backbones,
        'seed': a.seed,
        'clean_repeat': a.clean_repeat,
        'p0_clean_weight': a.p0_clean_weight,
        'val_total': best[0],
        'thresholds_cascade': [float(x) for x in best[1]],
        'thresholds_scaled': [float(x) for x in best[2]],
        'test': {k: st_sca[k] for k in ['total', 'S_pos', 'p0R', 'p0P', 'p0F', 'S_p0', 'S_clean', 'fpr']},
        'test_cascade': {k: st_cas[k] for k in ['total', 'S_pos', 'p0R', 'p0P', 'p0F', 'S_p0', 'S_clean', 'fpr']},
        'per_label': {f'{x[0]}_{x[1]}': {'P': x[2], 'R': x[3], 'F1': x[4]} for x in st_sca['per']},
    }
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
    np.savez_compressed(
        a.out.replace('.json', '_probs.npz'),
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vgt, gva=gva_n, va_names=np.array([r['name'] for r in va]),
        te_p0=tp0, te_p1=tp1, te_p2=tp2, te_gate=tgt, gte=gte_n, te_names=np.array([r['name'] for r in te])
    )
    print('TEST_SCALED', json.dumps(res['test'], ensure_ascii=False), flush=True)
    print('TEST_CASCADE', json.dumps(res['test_cascade'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
