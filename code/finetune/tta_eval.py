"""Test-time augmentation for the fused model.

Averages the 14 cumulative logits over {identity, hflip, vflip, rot180} subsets.
Thresholds are re-tuned on val for each variant, then applied to test.
"""
import argparse, csv, json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'decision_eval'))
from data import CachedIQADataset, CLASSES                       # noqa: E402
from models import build_backbone                                # noqa: E402
from predict_template import MultiNet, exact                     # noqa: E402
import comp_metrics as cm                                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='results/tta/model_s1.pt')
ap.add_argument('--split', default='data/split_trainval.csv')
ap.add_argument('--batch', type=int, default=12)
ap.add_argument('--workers', type=int, default=8)
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
print('loaded', a.ckpt, ck['backbones'], flush=True)


@torch.no_grad()
def logits_for(rows_, flips):
    ds = CachedIQADataset(rows_, 384, False)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers)
    outs, gs = [], []
    for x, g, _ in dl:
        x = x.to(dev); acc = 0.0
        for fl in flips:
            xx = x
            if fl == 'h': xx = torch.flip(x, dims=[4])
            elif fl == 'v': xx = torch.flip(x, dims=[3])
            elif fl == 'r': xx = torch.flip(x, dims=[3, 4])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg, _ = net(xx)
            acc = acc + lg.float()
        outs.append((acc / len(flips)).cpu()); gs.append(g)
    return torch.cat(outs), torch.cat(gs)


variants = [('none', ['id']), ('hflip', ['id', 'h']), ('hv', ['id', 'h', 'v']), ('4way', ['id', 'h', 'v', 'r'])]
print(f"{'variant':8s} {'val_total':>9s} {'test_total':>10s} {'S_pos':>7s} {'p0R':>6s} {'p0P':>6s} {'S_clean':>8s}")
for name, fl in variants:
    vO, vG = logits_for(va, fl); tO, tG = logits_for(te, fl)
    vp = exact(vO[:, :14]); tp = exact(tO[:, :14])
    vg = vG.numpy().astype(int); tg = tG.numpy().astype(int)
    b, th = cm.optimise(vg, *vp, 0.02)
    st = cm.score(tg, cm.decode(*tp, *th), 0.02)
    print(f'{name:8s} {b["total"]:9.2f} {st["total"]:10.2f} {st["S_pos"]:7.3f} '
          f'{st["p0R"]:6.3f} {st["p0P"]:6.3f} {st["S_clean"]:8.3f}', flush=True)
