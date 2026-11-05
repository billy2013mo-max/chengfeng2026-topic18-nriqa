"""V5 Root-Cause Fixed Multi-Backbone Trainer (train_v5.py):
1. Pre-Repeat Positive Weighting: computes `pos_weight` BEFORE `clean_repeat` (and clamps at 5.5
   instead of 8.0) so that repeating all-clean images does not artificially inflate `pos_weight`
   and cause cross-defect P2/P1 false positives on partially-defective images.
2. Per-Channel Negative Suppression (`channel_neg_weight=1.35`): penalizes false positive
   activations on any negative channel (`g[:, ci] == 0`), even when other defects exist on the
   same image, eliminating the 96%-99% cross-defect P2 false positive bottleneck.
3. Exact Competition S_p0 Formula (`0.5 * Recall + 0.3 * F1 + 0.2 * Precision`): directly aligns
   differentiable P0 loss with the official evaluation metric (including the 0.2 * Precision term)
   to prevent P1 -> P0 over-triggering on `过锐` and `过度生成`.
4. Higher Exact 4-Way Multinomial Blending (`alpha_ex=0.55`): gives greater weight to the
   4-way Cross-Entropy + Ordinal EMD head to prevent middle-grade `P1` from being squeezed
   between `P2` and `P0`.
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
from train_v2 import (MultiNetV2, exact_probs_from_dual, exact_targets_14,
                      predict_v2, decode_scaled_argmax, optimise_scaled, P0_IDX)
import comp_metrics as cm

CACHE_OK = os.path.exists('/dev/shm/race18_native/index.npz')


def v5_loss_fn(logits_cum, logits_ex, gate_logit, g, bce_loss_none,
               p0_clean_weight=4.2, channel_neg_weight=1.35,
               emd_weight=0.50, alpha_ex=0.55):
    # 1. Cumulative BCE loss with both per-channel negative suppression and all-clean P0 shield
    Y_cum = targets_from_grades(g)
    raw_bce = bce_loss_none(logits_cum, Y_cum)  # (B, 14)
    clean_mask = (g == 0).all(dim=1)            # (B,)

    w_mat = torch.ones_like(raw_bce)
    if channel_neg_weight != 1.0:
        # Penalize false positives on any negative channel (even when other defects are present)
        w_mat = torch.where(Y_cum == 0, torch.full_like(w_mat, channel_neg_weight), w_mat)
    if clean_mask.any() and p0_clean_weight > 1.0:
        for idx in P0_IDX:
            w_mat[clean_mask, idx] = p0_clean_weight
    loss_bce = (raw_bce * w_mat).mean()

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

    # 3. Direct 4-way / 2-way Cross-Entropy + Ordinal EMD (Ranked Probability Score) on logits_ex
    ce_tot = 0.0
    emd_tot = 0.0
    j_ex = 0
    # Slight boost (1.15x) on P1 (grade 2) to counter middle-grade squeezing without distorting priors
    cw4 = torch.tensor([1.0, 1.0, 1.15, 1.0], device=logits_ex.device)
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            lg4 = logits_ex[:, j_ex:j_ex + 4]
            tgt = g[:, ci].long()
            ce_tot = ce_tot + F.cross_entropy(lg4, tgt, weight=cw4, label_smoothing=0.025)
            probs4 = torch.softmax(lg4, dim=1)
            cdf_pred = torch.cumsum(probs4, dim=1)[:, :3]
            onehot = F.one_hot(tgt, num_classes=4).float()
            cdf_true = torch.cumsum(onehot, dim=1)[:, :3]
            emd_tot = emd_tot + ((cdf_pred - cdf_true) ** 2).mean()
            j_ex += 4
        else:
            tgt_bin = (g[:, ci] == 3).long()
            ce_tot = ce_tot + F.cross_entropy(logits_ex[:, j_ex:j_ex + 2], tgt_bin, label_smoothing=0.02)
            j_ex += 2
    loss_ce = ce_tot / len(CLASSES)
    loss_emd = emd_tot / len(FULL)

    # 4. Differentiable Soft Macro-F1 (S_pos) + Exact Competition S_p0 (0.5*R + 0.3*F1 + 0.2*P)
    p0, p1, p2, P_exact14 = exact_probs_from_dual(logits_cum, logits_ex, alpha_ex=alpha_ex)
    Y_exact14 = exact_targets_14(g)
    tp = (P_exact14 * Y_exact14).sum(dim=0)
    fp = (P_exact14 * (1.0 - Y_exact14)).sum(dim=0)
    fn = ((1.0 - P_exact14) * Y_exact14).sum(dim=0)

    soft_f1 = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
    loss_s_pos = 1.0 - soft_f1.mean()

    # Exact Competition S_p0 = 0.5 * R_p0 + 0.3 * F1_p0 + 0.2 * P_p0 on the 6 P0 channels
    tp_p0 = tp[P0_IDX]
    fp_p0 = fp[P0_IDX]
    fn_p0 = fn[P0_IDX]
    soft_r_p0  = (tp_p0 + 1e-4) / (tp_p0 + fn_p0 + 1e-4)
    soft_p_p0  = (tp_p0 + 1e-4) / (tp_p0 + fp_p0 + 1e-4)
    soft_f1_p0 = (2.0 * tp_p0 + 1e-4) / (2.0 * tp_p0 + fp_p0 + fn_p0 + 1e-4)
    soft_s_p0_vec = 0.5 * soft_r_p0 + 0.3 * soft_f1_p0 + 0.2 * soft_p_p0
    p0_w = torch.tensor([1.0, 1.0, 1.15, 0.9, 1.15, 1.05], device=logits_cum.device)
    loss_s_p0 = 1.0 - (soft_s_p0_vec * p0_w).sum() / p0_w.sum()

    # 5. Clean P0 max-probability suppression
    loss_clean_p0 = 0.0
    if clean_mask.any():
        loss_clean_p0 = p0[clean_mask].amax(dim=1).mean()

    # 6. Clean gate auxiliary loss
    cl_t = clean_mask.float()
    loss_gate = F.binary_cross_entropy_with_logits(gate_logit, cl_t)

    return (
        loss_bce
        + 0.65 * loss_ce
        + emd_weight * loss_emd
        + 0.50 * loss_s_pos
        + 0.45 * loss_s_p0
        + 0.20 * mono_pen
        + 0.45 * loss_clean_p0
        + 0.10 * loss_gate
    )


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='siglip2:lastk4,cradio:lastk4,dinov3l:lastk4')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--tiles', type=int, default=3)
    ap.add_argument('--lr-full', type=float, default=8.875e-5)
    ap.add_argument('--lr-last', type=float, default=3e-5)
    ap.add_argument('--lr-head', type=float, default=1.5e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--clean-repeat', type=int, default=4)
    ap.add_argument('--p0-clean-weight', type=float, default=4.2)
    ap.add_argument('--channel-neg-weight', type=float, default=1.35)
    ap.add_argument('--alpha-ex', type=float, default=0.55)
    ap.add_argument('--emd-weight', type=float, default=0.50)
    ap.add_argument('--split', default='data/split_all_full.csv')
    ap.add_argument('--manifest', default='data/manifest_combined.csv')
    ap.add_argument('--tau', type=float, default=0.02)
    ap.add_argument('--out', default='results/v5_clean/run.json')
    ap.add_argument('--save-state', default='', help='path to save checkpoint')
    a = ap.parse_args()

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
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})
    tr_orig = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test']
    if not te:
        te = va
        print('WARNING: split has no test rows, scoring on val', flush=True)

    # CRITICAL FIX: Compute pos_weight on the TRUE un-repeated REAL training distribution (clamped at 5.5)
    tr_real = [r for r in tr_orig if not str(r.get('source', '')).startswith('synth')] or tr_orig
    Y_orig = targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr_real]))
    pw = ((len(Y_orig) - Y_orig.sum(0)) / Y_orig.sum(0).clamp(min=1)).clamp(1.0, 5.5).to(dev)
    print('Pre-repeat true pos_weight per channel:', [round(float(x), 2) for x in pw], flush=True)
    bce_loss_none = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    tr = list(tr_orig)
    if a.clean_repeat > 1:
        clean = [r for r in tr_orig if all(int(r[c]) == 0 for c in CLASSES)]
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
    print(f'trainable {sum(p.numel() for p in tr_p)/1e6:.1f}M / {sum(p.numel() for p in net.parameters())/1e6:.1f}M '
          f'(size={a.size}, tiles={a.tiles}, views={1 + a.tiles*a.tiles})', flush=True)

    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    accum = max(1, a.grad_accum)
    steps_per_ep = max(1, len(ltr) // accum)
    total_steps = steps_per_ep * a.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=total_steps, pct_start=0.1)

    best = None
    ep3_state = None
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
                lc, le, gl = net(x)
            loss = v5_loss_fn(lc.float(), le.float(), gl.float(), g, bce_loss_none,
                              p0_clean_weight=a.p0_clean_weight,
                              channel_neg_weight=a.channel_neg_weight,
                              emd_weight=a.emd_weight,
                              alpha_ex=a.alpha_ex)
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
              f'S_pos={b_sca["S_pos"]:.3f} P0R={b_sca["p0R"]:.3f} P0P={b_sca["p0P"]:.3f} S_clean={b_sca["S_clean"]:.3f} t={time.time()-t0:.0f}s', flush=True)
        score_ep = max(b_cas['total'], b_sca['total'])
        cur_state = copy.deepcopy(net.state_dict())
        if (ep + 1) == 3:
            ep3_state = cur_state
            if a.save_state:
                ep3_path = a.save_state.replace('.pt', '_ep3.pt')
                os.makedirs(os.path.dirname(os.path.abspath(ep3_path)), exist_ok=True)
                torch.save({
                    'state': ep3_state,
                    'backbones': a.backbones,
                    'size': a.size,
                    'tiles': a.tiles,
                    'alpha_ex': a.alpha_ex,
                    'thresholds_cascade': [float(x) for x in th_cas],
                    'thresholds_scaled': [float(x) for x in th_sca],
                }, ep3_path)
                print('ep3 state saved', ep3_path, flush=True)
        if best is None or score_ep > best[0]:
            best = (score_ep, th_cas, th_sca, cur_state)

    net.load_state_dict(best[3])
    if a.save_state:
        os.makedirs(os.path.dirname(os.path.abspath(a.save_state)), exist_ok=True)
        torch.save({
            'state': best[3],
            'backbones': a.backbones,
            'size': a.size,
            'tiles': a.tiles,
            'alpha_ex': a.alpha_ex,
            'thresholds_cascade': [float(x) for x in best[1]],
            'thresholds_scaled': [float(x) for x in best[2]],
        }, a.save_state)
        print('state saved', a.save_state, flush=True)
        if ep3_state is not None:
            swa_state = {}
            for k in best[3].keys():
                if best[3][k].is_floating_point():
                    swa_state[k] = 0.5 * ep3_state[k] + 0.5 * best[3][k]
                else:
                    swa_state[k] = best[3][k]
            swa_path = a.save_state.replace('.pt', '_swa.pt')
            torch.save({
                'state': swa_state,
                'backbones': a.backbones,
                'size': a.size,
                'tiles': a.tiles,
                'alpha_ex': a.alpha_ex,
                'thresholds_cascade': [float(x) for x in best[1]],
                'thresholds_scaled': [float(x) for x in best[2]],
            }, swa_path)
            print('swa state saved', swa_path, flush=True)

    vp0, vp1, vp2, vgt, gva_n = predict_v2(net, lva, dev, alpha_ex=a.alpha_ex)
    tp0, tp1, tp2, tgt, gte_n = predict_v2(net, lte, dev, alpha_ex=a.alpha_ex)

    st_cas = cm.score(gte_n, cm.decode(tp0, tp1, tp2, *best[1]), a.tau)
    st_sca = cm.score(gte_n, decode_scaled_argmax(tp0, tp1, tp2, *best[2]), a.tau)

    res = {
        'backbones': a.backbones,
        'seed': a.seed,
        'size': a.size,
        'tiles': a.tiles,
        'clean_repeat': a.clean_repeat,
        'p0_clean_weight': a.p0_clean_weight,
        'channel_neg_weight': a.channel_neg_weight,
        'alpha_ex': a.alpha_ex,
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
