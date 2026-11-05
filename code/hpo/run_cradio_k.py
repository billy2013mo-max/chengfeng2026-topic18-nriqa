"""C-RADIO single-backbone: how many blocks should be unfrozen?

Baseline already measured: cradio:lastk4 with lr_last=1e-5 -> test 48.23.
Here we fix lr_last=3e-5 (the new optimum found for the triple) and sweep k.
For 'full' the pipeline uses --lr-full instead of --lr-last, so that row is a
different regime and is labelled as such.
"""
import json, os, subprocess, sys, time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/cradio_k'
os.makedirs(OUT, exist_ok=True)
GPUS = [2, 3, 4, 6, 7]             # GPU 0/1 are busy with the InternViT experiments

BASE = ['--seed', '1', '--epochs', '4', '--batch', '6', '--workers', '8',
        '--clean-penalty', '0', '--clean-repeat', '7',
        '--lr-full', '8.875e-05', '--lr-head', '2.2754e-03',
        '--wd', '0.01', '--dropout', '0',
        '--split', 'data/split_trainval.csv']

JOBS = []
for k in [4, 8, 12]:
    for seed in [1, 2]:
        JOBS.append((f'lastk{k}', f'cradio:lastk{k}', ['--lr-last', '3e-05'], seed))
for seed in [1, 2]:
    JOBS.append(('full', 'cradio:full', [], seed))


def cmd(spec, extra, seed, out):
    a = ['python3', '-u', '-I', 'code/finetune/train_multi.py', '--backbones', spec, '--out', out]
    b = list(BASE)
    i = b.index('--seed'); b[i + 1] = str(seed)
    return a + b + extra


def main():
    pending = list(JOBS)
    running = {}
    t0 = time.time()
    while pending or running:
        for g in [x for x in GPUS if x not in running]:
            if not pending:
                break
            tag, spec, extra, seed = pending.pop(0)
            out = f'{OUT}/cradio_{tag}_s{seed}.json'
            log = f'{OUT}/cradio_{tag}_s{seed}.log'
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
            f = open(log, 'w')
            p = subprocess.Popen(cmd(spec, extra, seed, out), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, tag, seed, out, f)
            print(f'[{time.time()-t0:6.0f}s] gpu{g} start {tag} seed{seed}', flush=True)
        time.sleep(10)
        for g in list(running):
            p, tag, seed, out, f = running[g]
            if p.poll() is not None:
                f.close(); del running[g]
                try:
                    d = json.load(open(out))
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} done {tag} seed{seed} '
                          f'test={d["test"]["total"]:.2f} val={d["val_total"]:.2f}', flush=True)
                except Exception as e:
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} FAIL {tag} seed{seed} {e}', flush=True)
    summarize()


def summarize():
    print('\n==== summary ====', flush=True)
    for tag in ['lastk4', 'lastk8', 'lastk12', 'full']:
        rows = []
        for seed in [1, 2]:
            f = f'{OUT}/cradio_{tag}_s{seed}.json'
            if os.path.exists(f):
                d = json.load(open(f)); rows.append(d)
        if not rows:
            continue
        t = np.array([d['test']['total'] for d in rows])
        print(f'{tag:8s} n={len(rows)} test={t.mean():.2f}±{t.std(ddof=1) if len(rows)>1 else 0:.2f} '
              f'runs={[round(x,2) for x in t]} '
              f'S_pos={np.mean([d["test"]["S_pos"] for d in rows]):.3f} '
              f'p0R={np.mean([d["test"]["p0R"] for d in rows]):.3f} '
              f'S_clean={np.mean([d["test"]["S_clean"] for d in rows]):.3f}', flush=True)


if __name__ == '__main__':
    main()
