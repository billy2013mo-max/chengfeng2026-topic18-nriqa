"""Single backbone (C-RADIO lastk4) vs triple, at lr_last=3e-5, across splits.

C-RADIO alone scored 49.17 on the original split, basically tied with the
triple's 49.23. If that holds across splits, the expensive 3-backbone fusion is
not worth its inference cost.

Already available from earlier runs (lr_last=3e-5):
    triple  orig: 49.205 / 49.260      s11: 49.585 / 49.256      s13: 48.728 / 49.645
Missing here:
    cradio x 5 splits x 2 seeds = 10 runs
    triple s14, s15 x 2 seeds   =  4 runs
"""
import json, os, subprocess, time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/tiles3'
os.makedirs(OUT, exist_ok=True)
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

CFG = {
    'cradio': 'cradio:lastk4',
    'triple': 'eupe:full,siglip2:lastk4,cradio:lastk4',
}
SPLITS = [11, 12, 13, 14, 15]
SEEDS = [1, 2]

JOBS = []
for sp in SPLITS:
    for sd in SEEDS:
        JOBS.append(('triple', sp, sd))



def cmd(tag, sp, sd, out):
    return ['python3', '-u', '-I', 'code/finetune/train_multi.py',
            '--backbones', CFG[tag], '--seed', str(sd), '--epochs', '4', '--batch', '6',
            '--workers', '8', '--clean-penalty', '0', '--clean-repeat', '7',
            '--lr-full', '8.875e-05', '--lr-last', '3e-05', '--lr-head', '2.2754e-03',
            '--wd', '0.01', '--dropout', '0', '--tiles', '3',
            '--split', f'data/resplits/split_s{sp}.csv', '--out', out]


def main():
    pending = list(JOBS); running = {}; t0 = time.time()
    while pending or running:
        for g in [x for x in GPUS if x not in running]:
            if not pending:
                break
            tag, sp, sd = pending.pop(0)
            out = f'{OUT}/{tag}_s{sp}_seed{sd}.json'
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
            f = open(f'{OUT}/{tag}_s{sp}_seed{sd}.log', 'w')
            p = subprocess.Popen(cmd(tag, sp, sd, out), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, tag, sp, sd, out, f)
            print(f'[{time.time()-t0:6.0f}s] gpu{g} start {tag} s{sp} seed{sd}', flush=True)
        time.sleep(10)
        for g in list(running):
            p, tag, sp, sd, out, f = running[g]
            if p.poll() is not None:
                f.close(); del running[g]
                try:
                    d = json.load(open(out))
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} done {tag} s{sp} seed{sd} '
                          f'test={d["test"]["total"]:.2f} val={d["val_total"]:.2f}', flush=True)
                except Exception as e:
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} FAIL {tag} s{sp} seed{sd} {e}', flush=True)
    summarize()


def summarize():
    print('\n==== summary ====', flush=True)
    for tag in ['triple']:
        per_split = {}
        for sp in SPLITS:
            ts = []
            for sd in SEEDS:
                f = f'{OUT}/{tag}_s{sp}_seed{sd}.json'
                if not os.path.exists(f):
                    continue
                d = json.load(open(f)); ts.append(d['test']['total'])
                per_split.setdefault(sp, []).append(d)
            if ts:
                print(f'{tag:7s} s{sp}: {np.mean(ts):.2f} ({[round(x, 2) for x in ts]}) '
                      f'S_pos={np.mean([d["test"]["S_pos"] for d in per_split[sp]]):.3f} '
                      f'S_clean={np.mean([d["test"]["S_clean"] for d in per_split[sp]]):.3f}', flush=True)
        allv = [v for sp in per_split for d in per_split[sp] for v in [d['test']['total']]]
        if allv:
            print(f'{tag:7s} ALL n={len(allv)} mean={np.mean(allv):.2f} '
                  f'median={np.median(allv):.2f} min={np.min(allv):.2f}\n', flush=True)


if __name__ == '__main__':
    main()
