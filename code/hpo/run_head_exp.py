"""Model-side experiments: (a) 4-way CE loss, (b) SigLIP2-giant backbone.

Both use the frozen recipe otherwise, evaluated on the same 5 splits x 2 seeds.
"""
import json, os, subprocess, time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/head_exp'
os.makedirs(OUT, exist_ok=True)
GPUS = list(range(8))
SPLITS = [11, 12, 13, 14, 15]
SEEDS = [1, 2]

CFG = {
    'perclass': dict(backbones='siglip2:lastk4,cradio:lastk4', extra=['--head', 'perclass']),
    'mild2': dict(backbones='siglip2:lastk4,cradio:lastk4', extra=['--mild-weight', '2.0']),
    'hidden': dict(backbones='siglip2:lastk4,cradio:lastk4', extra=['--hidden', '512']),
}

JOBS = [(g, sp, sd) for g in CFG for sp in SPLITS for sd in SEEDS]


def cmd(g, sp, sd, out):
    c = CFG[g]
    return ['python3', '-u', '-I', 'code/finetune/train_multi.py',
            '--backbones', c['backbones'], '--seed', str(sd), '--epochs', '4', '--batch', '6',
            '--workers', '8', '--clean-penalty', '0', '--clean-repeat', '7',
            '--lr-full', '8.875e-05', '--lr-last', '3e-05', '--lr-head', '2.2754e-03',
            '--wd', '0.01', '--dropout', '0',
            '--split', f'data/resplits/split_s{sp}.csv', '--out', out] + c['extra']


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
            grp, sp, sd = pending.pop(0)
            out = f'{OUT}/{grp}_s{sp}_seed{sd}.json'
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
            f = open(out.replace('.json', '.log'), 'w')
            p = subprocess.Popen(cmd(grp, sp, sd, out), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, grp, sp, sd, out, f)
            print(f'[{time.time()-t0:6.0f}s] gpu{g} start {grp} s{sp} seed{sd}', flush=True)
        time.sleep(15)
        for g in list(running):
            p, grp, sp, sd, out, f = running[g]
            if p.poll() is not None:
                f.close(); del running[g]
                try:
                    d = json.load(open(out)); t = d['test']
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} done {grp} s{sp} seed{sd} '
                          f'test={t["total"]:.2f} S_pos={t["S_pos"]:.3f} S_clean={t["S_clean"]:.3f}', flush=True)
                except Exception as e:
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} FAIL {grp} s{sp} seed{sd} {e}', flush=True)
    summarize()


def summarize():
    print('\n==== model experiments ====', flush=True)
    for grp in CFG:
        v = []
        for sp in SPLITS:
            for sd in SEEDS:
                f = f'{OUT}/{grp}_s{sp}_seed{sd}.json'
                if os.path.exists(f):
                    d = json.load(open(f)); v.append((sp, d['test']['total'], d['test']['S_pos'], d['test']['S_clean']))
        if not v:
            continue
        a = np.array([[x[1], x[2], x[3]] for x in v])
        print(f'{grp:6s} n={len(v)} total={a[:,0].mean():.2f} min={a[:,0].min():.2f} '
              f'S_pos={a[:,1].mean():.3f} S_clean={a[:,2].mean():.3f} fails<45={int((a[:,0]<45).sum())}', flush=True)
        for sp in SPLITS:
            vv = [x[1] for x in v if x[0] == sp]
            if vv:
                print(f'   s{sp}: {np.mean(vv):.2f} {[round(x,2) for x in vv]}', flush=True)


if __name__ == '__main__':
    main()
