"""Two-backbone unfreeze-depth screen.

SigLIP2 and C-RADIO each get last-k blocks unfrozen, k in {4, 8, 12}, at
lr_last in {1e-5, 3e-5}. Neither backbone's k was ever swept in the joint
two-backbone recipe (k=4 was inherited from the original triple).

Screening: original split + the hard split s12, 1 seed each, 36 runs.
Survivors get promoted to 5 splits x 2 seeds by run_ksweep_confirm.py.
"""
import json, os, subprocess, time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/k_sweep'
os.makedirs(OUT, exist_ok=True)
GPUS = list(range(8))

KS = [4, 8, 12]
LRS = ['1e-05', '3e-05']
SPLITS = [(0, 'data/split_trainval.csv'), (12, 'data/resplits/split_s12.csv')]

JOBS = []
for ks in KS:
    for kc in KS:
        for lr in LRS:
            for sp, path in SPLITS:
                JOBS.append((ks, kc, lr, sp, path))


def name(ks, kc, lr, sp):
    return f'sig{ks}_rad{kc}_lr{lr.replace("e-05", "e5")}_s{sp}'


def cmd(ks, kc, lr, path, out):
    return ['python3', '-u', '-I', 'code/finetune/train_multi.py',
            '--backbones', f'siglip2:lastk{ks},cradio:lastk{kc}',
            '--seed', '1', '--epochs', '4', '--batch', '6', '--workers', '8',
            '--clean-penalty', '0', '--clean-repeat', '7',
            '--lr-full', '8.875e-05', '--lr-last', lr, '--lr-head', '2.2754e-03',
            '--wd', '0.01', '--dropout', '0', '--split', path, '--out', out]


def gpu_free():
    o = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True).stdout
    fr = []
    for line in o.strip().splitlines():
        i, m = line.split(',')
        if int(m) < 2000:
            fr.append(int(i))
    return [g for g in GPUS if g in fr]


def main():
    pending = list(JOBS); running = {}; t0 = time.time()
    while pending or running:
        for g in [x for x in gpu_free() if x not in running]:
            if not pending:
                break
            ks, kc, lr, sp, path = pending.pop(0)
            nm = name(ks, kc, lr, sp)
            out = f'{OUT}/{nm}.json'
            if os.path.exists(out):
                continue
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
            f = open(out.replace('.json', '.log'), 'w')
            p = subprocess.Popen(cmd(ks, kc, lr, path, out), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, nm, out, f)
            print(f'[{time.time()-t0:6.0f}s] gpu{g} start {nm}', flush=True)
        time.sleep(15)
        for g in list(running):
            p, nm, out, f = running[g]
            if p.poll() is not None:
                f.close(); del running[g]
                try:
                    d = json.load(open(out)); t = d['test']
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} done {nm} test={t["total"]:.2f} '
                          f'S_pos={t["S_pos"]:.3f} S_clean={t["S_clean"]:.3f}', flush=True)
                except Exception as e:
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} FAIL {nm} {e}', flush=True)
    summarize()


def summarize():
    print('\n==== k sweep (orig + s12, seed1) ====', flush=True)
    rows = []
    for ks in KS:
        for kc in KS:
            for lr in LRS:
                v, cl, sp_ = [], [], []
                for sp, _ in SPLITS:
                    f = f'{OUT}/{name(ks, kc, lr, sp)}.json'
                    if os.path.exists(f):
                        d = json.load(open(f)); v.append(d['test']['total'])
                        cl.append(d['test']['S_clean']); sp_.append(d['test']['S_pos'])
                if v:
                    rows.append((np.mean(v), ks, kc, lr, v[0], v[1], np.mean(sp_), np.mean(cl)))
    for r in sorted(rows, reverse=True):
        print(f'sig{r[1]:2d} rad{r[2]:2d} lr={r[3]:7s} mean={r[0]:6.2f} '
              f'orig={r[4]:6.2f} s12={r[5]:6.2f} S_pos={r[6]:.3f} S_clean={r[7]:.3f}', flush=True)


if __name__ == '__main__':
    main()
