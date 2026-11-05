"""Transductive test-time adaptation (TENT-style) for the fused IQA model.

Only the affine parameters of the normalisation layers are updated, by
minimising the entropy of the predicted grade distribution. No labels are used,
so this can be applied to the unlabelled evaluation set at submission time.
"""
import argparse, csv, json, os, sys
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'decision_eval'))
from data import CachedIQADataset, CLASSES, FULL          # noqa: E402
from models import build_backbone                         # noqa: E402
from predict_template import MultiNet, exact              # noqa: E402
import comp_metrics as cm                                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='results/tta/model_s1.pt')
ap.add_argument('--split', default='data/split_trainval.csv')
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--steps', type=int, default=1)
ap.add_argument('--entropy-weight', type=float, default=1.0)
a = ap.parse_args()
dev = 'cuda'

rows = list(csv.DictReader(open(a.split)))
man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
for r in rows:
    r.update({k: man[r['name']][k] for k in CLASSES})
va = [r for r in rows if r['split'] == 'val']
te = [r for r in rows if r['split'] == 'test']

ck = torch.load(a.ckpt, map_location='cpu')
specs = []
for s in ck['backbones'].split(','):
    name, mode = s.split(':')
    specs.append((build_backbone(name), name, mode))
net = MultiNet([x[0] for x in specs], dropout=0.0).to(dev)
net.load_state_dict(ck['state']); net.eval()
for p in net.parameters():
    p.requires_grad_(False)
params = []
for m in net.modules():
    if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        if m.weight is not None:
            m.weight.requires_grad_(True); params.append(m.weight)
        if m.bias is not None:
            m.bias.requires_grad_(True); params.append(m.bias)
opt = torch.optim.SGD(params, lr=a.lr)


def grade_logits_to_entropy(logits):
    """logits: (B,14) cumulative. Build a 4-way distribution per class and sum entropy."""
    ent = 0.0
    j = 0
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            l2, l1, l0 = logits[:, j], logits[:, j + 1], logits[:, j + 2]; j += 3
            s2, s1, s0 = torch.sigmoid(l2), torch.sigmoid(l1), torch.sigmoid(l0)
            p = torch.stack([1 - s2, s2 - s1, s1 - s0, s0], 1).clamp(min=1e-6)
        else:
            s0 = torch.sigmoid(logits[:, j]); j += 1
            p = torch.stack([1 - s0, torch.zeros_like(s0), torch.zeros_like(s0), s0], 1).clamp(min=1e-6)
        p = p / p.sum(1, keepdim=True)
        ent = ent - (p * p.log()).sum(1).mean()
    return ent / len(CLASSES)


@torch.no_grad()
def infer(loader):
    O, G = [], []
    for x, g, _ in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg, _ = net(x.to(dev))
        O.append(lg.float().cpu()); G.append(g)
    return torch.cat(O), torch.cat(G)


# thresholds from the un-adapted val set
vO, vG = infer(DataLoader(CachedIQADataset(va, 384, False), batch_size=a.batch, num_workers=a.workers))
vp = exact(vO[:, :14]); vgv = vG.numpy().astype(int)
_, th = cm.optimise(vgv, *vp, 0.02)

# baseline on test
tO, tG = infer(DataLoader(CachedIQADataset(te, 384, False), batch_size=a.batch, num_workers=a.workers))
tp = exact(tO[:, :14]); tg = tG.numpy().astype(int)
base = cm.score(tg, cm.decode(*tp, *th), 0.02)
print(f'baseline   total={base["total"]:.2f} S_pos={base["S_pos"]:.3f} S_p0={base["S_p0"]:.3f} S_clean={base["S_clean"]:.3f}', flush=True)

# TENT adaptation on the unlabelled test set
dl = DataLoader(CachedIQADataset(te, 384, False), batch_size=a.batch, shuffle=True, num_workers=a.workers)
outs = []
for x, _, _ in dl:
    x = x.to(dev)
    for _ in range(a.steps):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg, _ = net(x)
        loss = grade_logits_to_entropy(lg.float())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        lg, _ = net(x)
    outs.append(lg.float().cpu())
adapt = torch.cat(outs)
# order is shuffled: recover original order is not needed for scoring, but names must match
# rebuild in the dataset order
O2, G2 = [], []
dl2 = DataLoader(CachedIQADataset(te, 384, False), batch_size=a.batch, shuffle=False, num_workers=a.workers)
with torch.no_grad():
    for x, g, _ in dl2:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg, _ = net(x.to(dev))
        O2.append(lg.float().cpu()); G2.append(g)
A = torch.cat(O2); tg2 = torch.cat(G2).numpy().astype(int)
ap_ = exact(A[:, :14])
st = cm.score(tg2, cm.decode(*ap_, *th), 0.02)
print(f'TENT(steps={a.steps},lr={a.lr}) total={st["total"]:.2f} S_pos={st["S_pos"]:.3f} S_p0={st["S_p0"]:.3f} S_clean={st["S_clean"]:.3f}', flush=True)
