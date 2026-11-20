"""Re-run one fixed hyperparameter configuration on several alternative splits.

Goal: is the HPO-selected config tied to one particular 2073/500/1000 split, or
does it reproduce the same score when the split is redrawn?

Jobs: best config on 5 new splits x 2 training seeds, plus a control (the old
dual baseline anchor) on 2 of those splits, for a reference point.
"""
import json, os, subprocess, sys, time, csv
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
GPUS = [0, 1, 2, 3, 4, 6, 7]
OUT = os.environ.get('RESPLIT_OUT', 'results/resplit')
BEST_ONLY = os.environ.get('RESPLIT_BEST_ONLY') == '1'
os.makedirs(OUT, exist_ok=True)

BEST = dict(backbones='eupe:full,siglip2:lastk4,cradio:lastk4', clean_penalty=0.0,
            clean_repeat=7, lr_full=8.875358285474408e-05, lr_head=0.0022753949862994462,
            wd=0.01, dropout=0.0, batch=6)
# old HPO anchor #0, as a control: same objective protocol, weaker config
CTRL = dict(backbones='eupe:full,siglip2:lastk4', clean_penalty=0.5,
            clean_repeat=3, lr_full=2e-05, lr_head=1e-03, wd=0.05, dropout=0.2, batch=6)

SPLITS = [11, 12, 13, 14, 15]
SEEDS = [1, 2]
EPOCHS = 4


def cmd(cfg, split_seed, train_seed, out):
    args = ['python3', '-u', '-I', 'code/finetune/train_multi.py',
            '--backbones', cfg['backbones'], '--seed', str(train_seed), '--epochs', str(EPOCHS),
            '--batch', str(cfg['batch']), '--workers', '8',
            '--clean-penalty', f"{cfg['clean_penalty']:.4f}", '--clean-repeat', str(cfg['clean_repeat']),
            '--lr-full', f"{cfg['lr_full']:.3e}", '--lr-head', f"{cfg['lr_head']:.3e}",
            '--wd', f"{cfg['wd']:.4f}", '--dropout', f"{cfg['dropout']:.3f}",
            '--split', f'data/resplits/split_s{split_seed}.csv', '--out', out]
    return args


def main():
    jobs = []
    for sp in SPLITS:
        for sd in SEEDS:
            name = f'best_s{sp}_seed{sd}'
            jobs.append((name, BEST, sp, sd))
    if not BEST_ONLY:
        for sp in SPLITS[:2]:
            for sd in SEEDS:
                name = f'ctrl_s{sp}_seed{sd}'
                jobs.append((name, CTRL, sp, sd))

    pending = list(jobs)
    running = {}  # gpu -> (proc, name, out, log)
    t0 = time.time()
    while pending or running:
        free = [g for g in GPUS if g not in running]
        for g in free:
            if not pending:
                break
            name, cfg, sp, sd = pending.pop(0)
            out = f'{OUT}/{name}.json'
            log = f'{OUT}/{name}.log'
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
            f = open(log, 'w')
            p = subprocess.Popen(cmd(cfg, sp, sd, out), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, name, out, f)
            print(f'[{time.time()-t0:6.0f}s] gpu{g} start {name}', flush=True)
        time.sleep(10)
        for g in list(running):
            p, name, out, f = running[g]
            if p.poll() is not None:
                f.close(); del running[g]
                try:
                    d = json.load(open(out))
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} done  {name} test={d["test"]["total"]:.2f} '
                          f'val={d["val_total"]:.2f}', flush=True)
                except Exception as e:
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} FAIL  {name} ({e})', flush=True)
    summarize()


def summarize():
    rows = []
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith('.json'):
            continue
        d = json.load(open(f'{OUT}/{fn}'))
        name = fn[:-5]
        cfg, sp, sd = name.split('_')
        s = d['test']
        clean = None
        try:
            pr = np.load(f'{OUT}/{fn[:-5]}_probs.npz', allow_pickle=True)
            clean = (pr['gva'] == 0).all(1)
            p99 = float(np.percentile(pr['va_p0'][clean].max(1), 99)) if clean is not None and clean.any() else None
            te_clean = (pr['gte'] == 0).all(1)
            test_p99 = float(np.percentile(pr['te_p0'][te_clean].max(1), 99)) if te_clean.any() else None
        except Exception:
            p99 = test_p99 = None
        rows.append(dict(cfg=cfg, split=int(sp[1:]), seed=int(sd[4:]), total=s['total'],
                         S_pos=s['S_pos'], p0R=s['p0R'], p0F=s['p0F'], p0P=s['p0P'],
                         S_clean=s['S_clean'], fpr=s['fpr'], val=d['val_total'],
                         val_p99=p99, test_p99=test_p99))
    if not rows:
        return
    cols = ['cfg', 'split', 'seed', 'val', 'total', 'S_pos', 'p0R', 'p0F', 'p0P', 'S_clean', 'fpr', 'val_p99', 'test_p99']
    with open(f'{OUT}/summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(rows, key=lambda r: (r['cfg'], r['split'], r['seed'])):
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in cols})
    print('\n==== summary ====', flush=True)
    for cfg in ['best', 'ctrl']:
        sub = [r for r in rows if r['cfg'] == cfg]
        if not sub:
            continue
        t = np.array([r['total'] for r in sub])
        print(f'{cfg}: n={len(sub)} mean={t.mean():.2f} std={t.std(ddof=1):.2f} '
              f'min={t.min():.2f} max={t.max():.2f}')
        for sp in sorted({r['split'] for r in sub}):
            ss = [r for r in sub if r['split'] == sp]
            tt = np.array([r['total'] for r in ss])
            print(f'   split {sp}: {tt.mean():.2f} (n={len(tt)}) '
                  f'S_pos={np.mean([r["S_pos"] for r in ss]):.3f} '
                  f'p0R={np.mean([r["p0R"] for r in ss]):.3f} '
                  f'S_clean={np.mean([r["S_clean"] for r in ss]):.3f} '
                  f'fpr={np.mean([r["fpr"] for r in ss]):.4f}')
        print(f'   components: S_pos={np.mean([r["S_pos"] for r in sub]):.3f} '
              f'p0R={np.mean([r["p0R"] for r in sub]):.3f} '
              f'p0F={np.mean([r["p0F"] for r in sub]):.3f} '
              f'p0P={np.mean([r["p0P"] for r in sub]):.3f} '
              f'S_clean={np.mean([r["S_clean"] for r in sub]):.3f} '
              f'fpr={np.mean([r["fpr"] for r in sub]):.4f}')


if __name__ == '__main__':
    main()
