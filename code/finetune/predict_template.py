"""Fill a competition label template with model predictions.

Input template: [{"image": "<name>", "label": {"模糊": "", ...}}, ...]
Output: same list/order/keys, with label values in {无问题, P2, P1, P0}.

The label key order in the template differs from CLASSES order, so predictions
are mapped by key name, never by position.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import CachedIQADataset, IQADataset, CLASSES, FULL   # noqa: E402
from models import build_backbone                             # noqa: E402

GRADE = {0: '无问题', 1: 'P2', 2: 'P1', 3: 'P0'}


class MultiNet(nn.Module):
    def __init__(self, specs, hidden=0, dropout=0.2):
        super().__init__()
        self.bs = nn.ModuleList(); self.fwds = []
        for (m, fwd, mu, sd, bl, dim) in specs:
            self.bs.append(m); self.fwds.append(fwd)
            self.register_buffer('mu_%d' % len(self.fwds), torch.tensor(mu).view(1, 3, 1, 1))
            self.register_buffer('sd_%d' % len(self.fwds), torch.tensor(sd).view(1, 3, 1, 1))
        d = sum(3 * x[5] for x in specs)
        self.head = (nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, 14))
                     if hidden else nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 14)))
        self.gate = nn.Linear(d, 1)

    def _agg(self, f):
        return torch.cat([f[:, 0], f[:, 1:].mean(1), f[:, 1:].amax(1)], 1)

    def forward(self, x):
        B, V, C, H, W = x.shape; out = []
        for i, (m, fwd) in enumerate(zip(self.bs, self.fwds)):
            mu = getattr(self, 'mu_%d' % (i + 1)); sd = getattr(self, 'sd_%d' % (i + 1))
            dt = next(m.parameters()).dtype
            z = ((x.view(B * V, C, H, W) - mu) / sd).to(dt)
            out.append(self._agg(fwd(z).view(B, V, -1)).float())
        fused = torch.cat(out, 1)
        return self.head(fused), self.gate(fused).squeeze(1)


def exact(P):
    P = P.float().cpu().numpy(); n = len(P)
    p0 = np.zeros((n, 6)); p1 = np.zeros((n, 4)); p2 = np.zeros((n, 4)); j = 0
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            c2, c1, c0 = P[:, j], P[:, j + 1], P[:, j + 2]; j += 3
            k = FULL.index(c)
            s2, s1, s0 = 1 / (1 + np.exp(-c2)), 1 / (1 + np.exp(-c1)), 1 / (1 + np.exp(-c0))
            p2[:, k] = np.clip(s2 - s1, 0, None); p1[:, k] = np.clip(s1 - s0, 0, None); p0[:, ci] = s0
        else:
            p0[:, ci] = 1 / (1 + np.exp(-P[:, j])); j += 1
    return p0, p1, p2


def decode(p0, p1, p2, t0, t1, t2):
    n = p0.shape[0]; pred = np.zeros((n, 6), int)
    FULL_IDX = [i for i, c in enumerate(CLASSES) if c in FULL]
    BIN_IDX = [i for i, c in enumerate(CLASSES) if c not in FULL]
    pred[:, FULL_IDX] = np.where(p2 > t2, 1, 0)
    pred[:, FULL_IDX] = np.where(p1 > t1, 2, pred[:, FULL_IDX])
    pred[:, FULL_IDX] = np.where(p0[:, FULL_IDX] > t0, 3, pred[:, FULL_IDX])
    pred[:, BIN_IDX] = np.where(p0[:, BIN_IDX] > t0, 3, 0)
    return pred


@torch.no_grad()
def predict(net, dl, dev):
    net.eval(); O = []
    for x, g, _ in dl:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg, _ = net(x.to(dev))
        O.append(lg.float().cpu())
    return torch.cat(O)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/final/model.pt')
    ap.add_argument('--template', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--manifest', default='data/manifest_combined.csv')
    ap.add_argument('--t0', type=float, default=0.55)
    ap.add_argument('--t1', type=float, default=0.45)
    ap.add_argument('--t2', type=float, default=0.30)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--tiles', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--imgdir', default='', help='load images from this dir instead of the native cache')
    ap.add_argument('--decode', default='cascade', choices=['cascade','severity'])
    ap.add_argument('--u0', type=float, default=2.70)
    ap.add_argument('--u1', type=float, default=1.75)
    ap.add_argument('--u2', type=float, default=0.60)
    a = ap.parse_args()
    dev = 'cuda'

    tpl = json.load(open(a.template))
    if a.limit:
        tpl = tpl[:a.limit]
    man = {r['name']: r for r in __import__('csv').DictReader(open(a.manifest))}
    rows = []
    for e in tpl:
        r = {'name': e['image']}
        m = man.get(e['image'], {})
        for c in CLASSES:
            r[c] = m.get(c, 0)
        rows.append(r)

    ck = torch.load(a.ckpt, map_location='cpu')
    specs = []
    for s in ck['backbones'].split(','):
        name, mode = s.split(':')
        specs.append((build_backbone(name), name, mode))
    net = MultiNet([x[0] for x in specs], dropout=0.0).to(dev)
    net.load_state_dict(ck['state'])
    print(f"loaded {a.ckpt}  backbones={ck['backbones']}  images={len(rows)}", flush=True)

    cache_ok = os.path.exists('/dev/shm/race18_native/index.npz')
    if a.imgdir:
        ds = IQADataset(rows, a.imgdir, a.imgdir, set(r['name'] for r in rows), a.size, False, tiles=a.tiles)
    elif cache_ok:
        ds = CachedIQADataset(rows, a.size, False, tiles=a.tiles)
    else:
        orig = set(man)
        ds = IQADataset(rows, 'data/images', 'data/images_supp', orig, a.size, False, tiles=a.tiles)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    import time
    t0 = time.time()
    O = predict(net, dl, dev)
    dt = time.time() - t0
    p0, p1, p2 = exact(O[:, :14])
    if a.decode == 'severity':
        sev = np.zeros_like(p0)
        for ci, c in enumerate(CLASSES):
            if c in FULL:
                j = FULL.index(c)
                cum2 = p0[:, ci] + p1[:, j] + p2[:, j]
                cum1 = p0[:, ci] + p1[:, j]
                sev[:, ci] = cum2 + cum1 + p0[:, ci]
            else:
                sev[:, ci] = 3 * p0[:, ci]
        pred = np.where(sev > a.u0, 3, np.where(sev > a.u1, 2, np.where(sev > a.u2, 1, 0)))
        for ci, c in enumerate(CLASSES):          # binary classes: only P0 / none
            if c not in FULL:
                pred[:, ci] = np.where(sev[:, ci] > a.u0, 3, 0)
    else:
        pred = decode(p0, p1, p2, a.t0, a.t1, a.t2)
    print(f'inference {dt:.1f}s for {len(rows)} images ({1000*dt/len(rows):.1f} ms/img)', flush=True)

    out = []
    for i, e in enumerate(tpl):
        lab = {}
        for k in e['label']:
            ci = CLASSES.index(k)
            lab[k] = GRADE[int(pred[i, ci])]
        out.append({'image': e['image'], 'label': lab})
    json.dump(out, open(a.out, 'w'), ensure_ascii=False)

    # distribution
    print('\nper-class grade distribution:')
    for ci, c in enumerate(CLASSES):
        vals = [GRADE[int(v)] for v in pred[:, ci]]
        uniq = {v: vals.count(v) for v in ['无问题', 'P2', 'P1', 'P0'] if v in vals}
        print(f'  {c:6s} {uniq}')
    clean = int((pred == 0).all(1).sum())
    print(f'\nall-clean images: {clean}/{len(pred)} ({100*clean/len(pred):.1f}%)')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
