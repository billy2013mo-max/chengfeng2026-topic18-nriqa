"""C-RADIO: unfreeze-depth x learning-rate screen.

The earlier k sweep held lr_last fixed at 3e-5, which confounds depth with
effective learning rate: unfreezing more blocks with the same lr behaves like a
much larger update. 'full' additionally used lr_full=8.875e-5 and collapsed.
This screen crosses k with a smaller lr on a single seed (screening only).
"""
import json, os, subprocess, time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/cradio_lrk'
os.makedirs(OUT, exist_ok=True)
GPUS = [0, 1]

BASE = ['--seed', '1', '--epochs', '4', '--batch', '6', '--workers', '8',
        '--clean-penalty', '0', '--clean-repeat', '7',
        '--lr-full', '8.875e-05', '--lr-head', '2.2754e-03',
        '--wd', '0.01', '--dropout', '0', '--split', 'data/split_trainval.csv']

JOBS = [
    ('lastk8_lr1e5', 'cradio:lastk8', ['--lr-last', '1e-05']),
    ('lastk12_lr1e5', 'cradio:lastk12', ['--lr-last', '1e-05']),
    ('lastk8_lr3e6', 'cradio:lastk8', ['--lr-last', '3e-06']),
    ('full_lr1e5', 'cradio:full', ['--lr-full', '1e-05']),
    ('full_lr3e6', 'cradio:full', ['--lr-full', '3e-06']),
]


def cmd(spec, extra, out):
    return (['python3', '-u', '-I', 'code/finetune/train_multi.py', '--backbones', spec, '--out', out]
            + BASE + extra)


def main():
    pending = list(JOBS); running = {}; t0 = time.time()
    while pending or running:
        for g in [x for x in GPUS if x not in running]:
            if not pending:
                break
            tag, spec, extra = pending.pop(0)
            out = f'{OUT}/{tag}.json'
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
            f = open(f'{OUT}/{tag}.log', 'w')
            p = subprocess.Popen(cmd(spec, extra, out), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, tag, out, f)
            print(f'[{time.time()-t0:6.0f}s] gpu{g} start {tag}', flush=True)
        time.sleep(10)
        for g in list(running):
            p, tag, out, f = running[g]
            if p.poll() is not None:
                f.close(); del running[g]
                try:
                    d = json.load(open(out))
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} done {tag} '
                          f'test={d["test"]["total"]:.2f} val={d["val_total"]:.2f} '
                          f'S_pos={d["test"]["S_pos"]:.3f} S_clean={d["test"]["S_clean"]:.3f}', flush=True)
                except Exception as e:
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} FAIL {tag} {e}', flush=True)
    print('\n==== summary ====', flush=True)
    for tag, _, _ in JOBS:
        f = f'{OUT}/{tag}.json'
        if os.path.exists(f):
            d = json.load(open(f)); t = d['test']
            print(f'{tag:16s} test={t["total"]:.2f} val={d["val_total"]:.2f} '
                  f'S_pos={t["S_pos"]:.3f} p0R={t["p0R"]:.3f} p0P={t["p0P"]:.3f} '
                  f'S_clean={t["S_clean"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
