"""Phase 5: two-backbone (SigLIP2 + C-RADIO) x class-5 clean penalty.

Two-backbone was the most stable variant overnight (s12: 45.89 vs triple 35.69).
A small class-5 clean penalty (lambda=0.3) also helped the same tail. This runs
the combination on the hard splits (s11-s15, incl. s12) and on the never-used
splits (s16-s20), with a plain two-backbone control on s16-s20.
"""
import json, os, subprocess, time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/phase5'
os.makedirs(OUT, exist_ok=True)
GPUS = list(range(8))
TWO = 'siglip2:lastk4,cradio:lastk4'

JOBS = []
for sp in [11, 12, 13, 14, 15]:
    for sd in [1, 2]:
        JOBS.append(('tb_l03', sp, sd))
for sp in [16, 17, 18, 19, 20]:
    for sd in [1, 2]:
        JOBS.append(('tb_l03', sp, sd))
        JOBS.append(('tb_plain', sp, sd))


def cmd(group, sp, sd, out):
    extra = ['--clean-penalty', '0.3', '--clean-penalty-class', '5'] if group == 'tb_l03' else ['--clean-penalty', '0']
    return ['python3', '-u', '-I', 'code/finetune/train_multi.py',
            '--backbones', TWO, '--seed', str(sd), '--epochs', '4', '--batch', '6', '--workers', '8',
            '--clean-repeat', '7', '--lr-full', '8.875e-05', '--lr-last', '3e-05', '--lr-head', '2.2754e-03',
            '--wd', '0.01', '--dropout', '0', '--split', f'data/resplits/split_s{sp}.csv',
            '--out', out] + extra


def gpu_free():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    fr = []
    for line in out.strip().splitlines():
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
            group, sp, sd = pending.pop(0)
            out = f'{OUT}/{group}_s{sp}_seed{sd}.json'
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
            f = open(out.replace('.json', '.log'), 'w')
            p = subprocess.Popen(cmd(group, sp, sd, out), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, group, sp, sd, out, f)
            print(f'[{time.time()-t0:6.0f}s] gpu{g} start {group} s{sp} seed{sd}', flush=True)
        time.sleep(15)
        for g in list(running):
            p, group, sp, sd, out, f = running[g]
            if p.poll() is not None:
                f.close(); del running[g]
                try:
                    d = json.load(open(out)); t = d['test']
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} done {group} s{sp} seed{sd} '
                          f'test={t["total"]:.2f} S_pos={t["S_pos"]:.3f} S_clean={t["S_clean"]:.3f}', flush=True)
                except Exception as e:
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} FAIL {group} s{sp} seed{sd} {e}', flush=True)
    summarize()


def summarize():
    print('\n==== phase5 summary ====', flush=True)
    for group in ['tb_l03', 'tb_plain']:
        rows = []
        for sp in [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            for sd in [1, 2]:
                f = f'{OUT}/{group}_s{sp}_seed{sd}.json'
                if os.path.exists(f):
                    rows.append((sp, sd, json.load(open(f))))
        if not rows:
            continue
        v = np.array([d['test']['total'] for _, _, d in rows])
        print(f'{group}: n={len(v)} mean={v.mean():.2f} median={np.median(v):.2f} min={v.min():.2f} '
              f'fails<45={int((v<45).sum())} S_pos={np.mean([d["test"]["S_pos"] for _,_,d in rows]):.3f}')
        for sp in sorted({r[0] for r in rows}):
            vv = [d['test']['total'] for s, _, d in rows if s == sp]
            print(f'   s{sp}: {np.mean(vv):.2f} {[round(x,2) for x in vv]}', flush=True)


if __name__ == '__main__':
    main()
