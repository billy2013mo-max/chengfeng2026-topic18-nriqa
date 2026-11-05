"""BoTorch HPO re-run on the fixed label order (the original 80-trial search was
done while the P2/P1 cumulative targets were swapped).

Search space around the current best two-backbone recipe.
Objective: val total on data/split_trainval.csv (single seed per trial).
"""
import argparse, csv, json, math, os, subprocess, time
import numpy as np, torch
from botorch.models import MixedSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

os.chdir('/wanqing-develop/luowenjing/18')
CONT = [('clean_repeat', 3, 10, 'int'), ('lr_full', 3e-5, 1.5e-4, 'log'),
        ('lr_last', 1e-5, 1e-4, 'log'), ('lr_head', 5e-4, 4e-3, 'log'),
        ('dropout', 0.0, 0.3, 'cont'), ('wd', 0.01, 0.2, 'log')]
CAT = [('batch', [6, 12])]
N = len(CONT) + len(CAT); CAT_DIMS = [len(CONT)]
BOUNDS = torch.tensor([[0.0] * N, [1.0] * len(CONT) + [len(CAT[0][1]) - 1]], dtype=torch.double)
BASE = ['--backbones', 'siglip2:lastk4,cradio:lastk4', '--seed', '1', '--epochs', '4',
        '--workers', '8', '--clean-penalty', '0', '--split', 'data/split_trainval.csv']


def encode(c):
    x = []
    for name, lo, hi, kind in CONT:
        v = c[name]
        v = (math.log(max(v, 1e-9)) - math.log(lo)) / (math.log(hi) - math.log(lo)) if kind == 'log' else (v - lo) / (hi - lo)
        x.append(min(1.0, max(0.0, v)))
    for name, vals in CAT:
        x.append(float(vals.index(c[name])))
    return x


def decode(x):
    c = {}
    for i, (name, lo, hi, kind) in enumerate(CONT):
        v = float(x[i])
        if kind == 'log':
            v = math.exp(math.log(lo) + v * (math.log(hi) - math.log(lo)))
        else:
            v = lo + v * (hi - lo)
        c[name] = int(round(v)) if kind == 'int' else v
    c['batch'] = CAT[0][1][int(round(float(x[len(CONT)])))]
    c['clean_repeat'] = int(np.clip(c['clean_repeat'], 3, 10))
    return c


def key(c):
    return tuple(round(v, 4) for v in encode(c))


def run_one(c, gpu, idx):
    out = f'results/hpo2/run_{idx:03d}.json'; log = f'results/hpo2/run_{idx:03d}.log'
    cmd = ['python3', '-u', '-I', 'code/finetune/train_multi.py'] + BASE + [
        '--batch', str(c['batch']), '--clean-repeat', str(c['clean_repeat']),
        '--lr-full', f"{c['lr_full']:.3e}", '--lr-last', f"{c['lr_last']:.3e}",
        '--lr-head', f"{c['lr_head']:.3e}", '--wd', f"{c['wd']:.4f}",
        '--dropout', f"{c['dropout']:.3f}", '--out', out]
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    f = open(log, 'w')
    return subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT), out


def gpu_free():
    o = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True).stdout
    fr = []
    for line in o.strip().splitlines():
        i, m = line.split(',')
        if int(m) < 2000:
            fr.append(int(i))
    return sorted(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', type=int, default=12); ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--q', type=int, default=7)
    a = ap.parse_args()
    os.makedirs('results/hpo2', exist_ok=True)
    hist = []; idx = 0

    def run_batch(cfgs):
        nonlocal idx
        pending = list(cfgs); running = {}
        res = {}
        while pending or running:
            for g in gpu_free():
                if g in running or not pending:
                    continue
                c = pending.pop(0)
                p, out = run_one(c, g, idx); running[g] = (p, c, out)
                print(f'  gpu{g} start #{idx} {json.dumps(c)}', flush=True); idx += 1
            time.sleep(10)
            for g in list(running):
                p, c, out = running[g]
                if p.poll() is not None:
                    del running[g]
                    try:
                        d = json.load(open(out)); v = float(d['val_total'])
                        p99 = None
                        try:
                            pr = np.load(out.replace('.json', '_probs.npz'), allow_pickle=True)
                            clean = (pr['gva'] == 0).all(1)
                            p99 = float(np.percentile(pr['va_p0'][clean].max(1), 99)) if clean.any() else None
                        except Exception:
                            pass
                        obj = v - 10.0 * max(0.0, (p99 or 0) - 0.15)
                        res[key(c)] = (obj, v, p99, out)
                        hist.append(dict(cfg=c, obj=obj, val=v, p99=p99, out=out))
                        print(f'  gpu{g} done #{idx-1} val={v:.2f} obj={obj:.2f}', flush=True)
                    except Exception as e:
                        res[key(c)] = (0.0, 0.0, 1.0, out)
                        print(f'  gpu{g} FAIL {e}', flush=True)
        with open('results/hpo2/history.json', 'w') as f:
            json.dump(hist, f, ensure_ascii=False, indent=1)
        return res

    anchors = [
        dict(clean_repeat=7, lr_full=8.875e-5, lr_last=3e-5, lr_head=2.2754e-3, dropout=0.0, wd=0.01, batch=6),
        dict(clean_repeat=5, lr_full=6e-5, lr_last=3e-5, lr_head=1.5e-3, dropout=0.1, wd=0.05, batch=6),
        dict(clean_repeat=9, lr_full=1.1e-4, lr_last=6e-5, lr_head=3e-3, dropout=0.05, wd=0.02, batch=6),
    ]
    seen = {key(c) for c in anchors}
    res = run_batch(anchors)
    Xs = draw_sobol_samples(bounds=BOUNDS, n=a.init, q=1).squeeze(1)
    cfgs = [decode(x) for x in Xs]
    cfgs = [c for c in cfgs if key(c) not in seen][:a.init]
    res.update(run_batch(cfgs))

    for r in range(a.rounds):
        data = [h for h in hist if h['out']]
        X = torch.tensor([encode(h['cfg']) for h in data], dtype=torch.double)
        X[:, CAT_DIMS] = X[:, CAT_DIMS].round()
        Y = torch.tensor([[h['obj']] for h in data], dtype=torch.double)
        ym, ys = Y.mean(), Y.std().clamp(min=1e-6)
        Yz = (Y - ym) / ys
        model = MixedSingleTaskGP(X, Yz, cat_dims=CAT_DIMS, train_Yvar=torch.full_like(Yz, (4.0 / ys.item()) ** 2))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        acq = qLogNoisyExpectedImprovement(model=model, X_baseline=X, sampler=SobolQMCNormalSampler(torch.Size([512])))
        cands, _ = optimize_acqf_mixed(acq, bounds=BOUNDS, q=a.q, num_restarts=8,
                                       fixed_features_list=[{len(CONT): k} for k in range(len(CAT[0][1]))],
                                       raw_samples=128)
        cfgs = [decode(c) for c in cands]
        seen = {key(h['cfg']) for h in hist}
        cfgs = [c for c in cfgs if key(c) not in seen] or cfgs
        res.update(run_batch(cfgs))
        best = max(hist, key=lambda h: h['obj'])
        print(f'round {r+1}: best val={best["val"]:.2f} {json.dumps(best["cfg"])}', flush=True)
    hist.sort(key=lambda h: -h['obj'])
    json.dump(hist[:10], open('results/hpo2/best.json', 'w'), ensure_ascii=False, indent=1)
    print('TOP5:')
    for h in hist[:5]:
        print(f'  val={h["val"]:.2f} obj={h["obj"]:.2f} {json.dumps(h["cfg"])}')


if __name__ == '__main__':
    main()
