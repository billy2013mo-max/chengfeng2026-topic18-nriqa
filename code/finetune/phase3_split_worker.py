"""Phase 3 (per BREAKTHROUGH_SOLUTION_REPORT.md section 5) split worker.

Phase 3 family: C-RADIOv4-H + SigLIP2-NaFlex + PE-Core-g14 (+ 88D physical), frozen features, CORN tower.
For one official blind split (data/resplits/split_sXX.csv):
  1. Train Phase-2 tower (InternViT+C-RADIO+SigLIP2+Phys)  -> baseline probs on blind test
  2. Train Phase-3 tower (C-RADIO+NaFlex+PE-g14+Phys)       -> probs on blind test
  3. Doc 3.5.2: align Phase-3 logit moments/quantiles to Phase-2 on the (unlabelled) test set,
     then fuse with 22% weight (doc: 20%~25%).
Saves gold + the three prob sets for paired bootstrap in phase3_aggregate.py.
Epoch selection uses the split's *val* part only; test part stays blind.
"""
import os, sys, csv, argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, 'code/finetune'); sys.path.insert(0, 'code/decision_eval')
import train_phase2_corn_tower as P2
from train_phase2_corn_tower import (CLASSES, FULL_IDX, BIN_IDX, DEVICE, corn_loss_fn, decode_roll_down,
                                     compute_competition_metrics)

W_P3 = 0.22
TH_ANY = [0.26, 0.28, 0.27, 0.20, 0.26, 0.20]
TH_ROLL = [0.28, 0.30, 0.30, 0.25, 0.28, 0.25]
TH_P0 = [0.35, 0.35, 0.38, 0.35, 0.38, 0.35]


def logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -15, 15)))


def align(p_new, p_ref):
    """Doc 3.5.2: logit moment matching (0.65) + rank-preserving quantile blend (0.35)."""
    ln, lr = logit(p_new), logit(p_ref)
    lm = lr.mean(0, keepdims=True) + lr.std(0, keepdims=True) / (ln.std(0, keepdims=True) + 1e-6) * (ln - ln.mean(0, keepdims=True))
    lq = np.zeros_like(ln)
    for c in range(ln.shape[1]):
        lq[:, c] = np.sort(lr[:, c])[np.argsort(np.argsort(ln[:, c]))]
    return sigmoid(0.65 * lm + 0.35 * lq)


class Tower3(nn.Module):
    def __init__(self, dims, d_hid=512, dropout=0.35):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(d, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(dropout)) for d in dims[:-1]])
        self.proj_phys = nn.Sequential(nn.Linear(dims[-1], 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2))
        self.fusion = nn.Sequential(nn.Linear(d_hid * (len(dims) - 1) + 128, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout))
        self.heads_full = nn.ModuleList([nn.Linear(512, 3) for _ in range(4)])
        self.heads_bin = nn.ModuleList([nn.Linear(512, 1) for _ in range(2)])

    def forward(self, *xs):
        h = torch.cat([p(x) for p, x in zip(self.proj, xs[:-1])] + [self.proj_phys(xs[-1])], -1)
        f = self.fusion(h)
        return [h_(f) for h_ in self.heads_full], [h_(f) for h_ in self.heads_bin]

    def predict_probs(self, *xs):
        lf, lb = self.forward(*xs)
        B = xs[0].shape[0]
        p0 = torch.zeros((B, 6), device=xs[0].device); p1 = torch.zeros((B, 4), device=xs[0].device); p2 = torch.zeros((B, 4), device=xs[0].device)
        for j, ci in enumerate(FULL_IDX):
            fe, fm, fp = torch.sigmoid(lf[j][:, 0]), torch.sigmoid(lf[j][:, 1]), torch.sigmoid(lf[j][:, 2])
            p0[:, ci] = fe * fm * fp; p1[:, j] = fe * fm * (1 - fp); p2[:, j] = fe * (1 - fm)
        for j, ci in enumerate(BIN_IDX):
            p0[:, ci] = torch.sigmoid(lb[j][:, 0])
        return p0, p1, p2


def train_tower(model, xs_tr, y_tr, xs_va, y_va, epochs=35, bs=64, lr=1.5e-3):
    """Returns blind-selection-safe model: epoch picked on val; returns state dict of best-val epoch."""
    ds = TensorDataset(*[torch.from_numpy(x).float() for x in xs_tr], torch.from_numpy(y_tr).long())
    ld = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    va_t = [torch.from_numpy(x).float().to(DEVICE) for x in xs_va]
    best, best_sd = -1, None
    for ep in range(epochs):
        model.train()
        for *bx, by in ld:
            bx = [b.to(DEVICE) for b in bx]; by = by.to(DEVICE)
            opt.zero_grad(); lf, lb = model(*bx); corn_loss_fn(lf, lb, by).backward(); opt.step()
        sch.step()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                p0, p1, p2 = [t.cpu().numpy() for t in model.predict_probs(*va_t)]
            s = compute_competition_metrics(y_va, decode_roll_down(p0, p1, p2, TH_ANY, TH_ROLL, TH_P0))[0]
            if s > best:
                best = s; best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd); model.eval()
    return best


def infer(model, xs):
    with torch.no_grad():
        return [t.cpu().numpy() for t in model.predict_probs(*[torch.from_numpy(x).float().to(DEVICE) for x in xs])]


def load_feats():
    def merged(orig, supp):
        a, b = np.load(orig), np.load(supp)
        return np.concatenate([a['X'], b['X']]), list(a['names']) + list(b['names'])
    F = {}
    for k, f in [('intern', 'feats/internvit-6b_combined.npz'), ('radio', 'feats/c-radio-v4-h_combined.npz'), ('siglip', 'feats/siglip2-so400m_combined.npz')]:
        d = np.load(f); F[k] = d['X']; F['names'] = list(d['names'])
    for k, o, s in [('naflex', 'feats/p3/naflex_orig.npz', 'feats/p3/naflex_supp.npz'), ('peg', 'feats/p3/peg_orig.npz', 'feats/p3/peg_supp.npz')]:
        X, nm = merged(o, s); idx = {n: i for i, n in enumerate(nm)}
        F[k] = X[[idx[n] for n in F['names']]]            # re-order into combined order
    ph = np.load('feats/phys_88d_combined.npz')['X']
    med = np.median(ph, 0); iqr = np.percentile(ph, 75, 0) - np.percentile(ph, 25, 0); iqr[iqr < 1e-4] = 1
    F['phys'] = np.clip((ph - med) / iqr, -10, 10)
    return F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--split', required=True); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    F = load_feats()
    rows = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
    gold = np.array([[int(rows[n][c]) for c in CLASSES] for n in F['names']])
    sp = {r['name']: r['split'] for r in csv.DictReader(open(f'data/resplits/{a.split}.csv'))}
    idx = {s: [i for i, n in enumerate(F['names']) if sp.get(n) == s] for s in ('train', 'val', 'test')}
    print(a.split, {k: len(v) for k, v in idx.items()}, flush=True)
    assert min(len(v) for v in idx.values()) > 0, 'empty split part!'
    # no overlap sanity
    assert not (set(idx['train']) & set(idx['test'])) and not (set(idx['val']) & set(idx['test']))

    def sel(keys, part):
        return [F[k][idx[part]] for k in keys]

    k2, k3 = ['intern', 'radio', 'siglip', 'phys'], ['radio', 'naflex', 'peg', 'phys']
    m2 = P2.CORNAttentionTower().to(DEVICE)
    # P2 tower forward signature is (intern, radio, siglip, phys)
    v2 = train_tower(m2, sel(k2, 'train'), gold[idx['train']], sel(k2, 'val'), gold[idx['val']])
    pr2 = infer(m2, sel(k2, 'test'))
    m3 = Tower3([F[k].shape[1] for k in k3]).to(DEVICE)
    v3 = train_tower(m3, sel(k3, 'train'), gold[idx['train']], sel(k3, 'val'), gold[idx['val']])
    pr3 = infer(m3, sel(k3, 'test'))

    pr3a = [align(pr3[i], pr2[i]) for i in range(3)]
    fused = [(1 - W_P3) * pr2[i] + W_P3 * pr3a[i] for i in range(3)]
    g = gold[idx['test']]
    out = {}
    for name, pr in [('p2', pr2), ('p3', pr3), ('fused', fused)]:
        out[name] = compute_competition_metrics(g, decode_roll_down(*pr, TH_ANY, TH_ROLL, TH_P0))
        print(f'[{a.split}] {name:6s} total={out[name][0]:.2f} pos={out[name][1]*25:.2f} p0={out[name][2]*20:.2f} clean={out[name][3]*20:.2f}', flush=True)
    os.makedirs('results/phase3', exist_ok=True)
    np.savez_compressed(f'results/phase3/{a.split}.npz', gold=g, **{f'{n}_{j}': pr[j] for n, pr in [('p2', pr2), ('p3', pr3), ('fused', fused)] for j in range(3)})


if __name__ == '__main__':
    main()
