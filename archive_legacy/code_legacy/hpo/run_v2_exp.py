"""Launch train_v2.py in parallel across s16..s20 and compare with v1 baselines."""
import glob, json, os, subprocess, sys, time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/v2'
os.makedirs(OUT, exist_ok=True)

# 7 parallel jobs fitting into one single wave across the 7 GPUs with >100GB free memory
JOBS = [
    ('v2_pcl', 16, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4'),
    ('v2_pcl', 17, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4'),
    ('v2_pcl', 18, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4'),
    ('v2_pcl', 19, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4'),
    ('v2_pcl', 20, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4'),
    ('v2_eupe', 16, 1, 'eupe:full,siglip2:lastk4,cradio:lastk4'),
    ('v2_eupe', 17, 1, 'eupe:full,siglip2:lastk4,cradio:lastk4'),
]


def free_gpus(max_used_mib=60000):
    o = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    ).stdout
    avail = []
    for line in o.strip().splitlines():
        i, m = line.split(',')
        if int(m) < max_used_mib:
            avail.append((int(m), int(i)))
    avail.sort()
    return [i for _, i in avail]


def build_cmd(grp, sp, sd, bb, out):
    return [
        'python3', '-u', '-I', 'code/finetune/train_v2.py',
        '--backbones', bb,
        '--seed', str(sd),
        '--epochs', '4',
        '--batch', '6',
        '--workers', '6',
        '--size', '384',
        '--clean-repeat', '4',
        '--p0-clean-weight', '4.0',
        '--lr-full', '8.875e-05',
        '--lr-last', '3e-05',
        '--lr-head', '1.5e-03',
        '--wd', '0.01',
        '--dropout', '0.1',
        '--split', f'data/resplits/split_s{sp}.csv',
        '--out', out,
    ]


def main():
    pending = [j for j in JOBS if not os.path.exists(f'{OUT}/{j[0]}_s{j[1]}_seed{j[2]}.json')]
    running = {}
    t0 = time.time()
    print(f'Starting {len(pending)} v2 training jobs in parallel...', flush=True)

    while pending or running:
        avail = [g for g in free_gpus(60000) if g not in running]
        for g in avail:
            if not pending:
                break
            grp, sp, sd, bb = pending.pop(0)
            out = f'{OUT}/{grp}_s{sp}_seed{sd}.json'
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(g)
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            lf = open(out.replace('.json', '.log'), 'w')
            p = subprocess.Popen(build_cmd(grp, sp, sd, bb, out), env=env, stdout=lf, stderr=subprocess.STDOUT)
            running[g] = (p, grp, sp, sd, out, lf)
            print(f'[{time.time()-t0:5.0f}s] gpu{g} START {grp} s{sp} seed{sd}', flush=True)

        time.sleep(15)
        for g in list(running):
            p, grp, sp, sd, out, lf = running[g]
            if p.poll() is not None:
                lf.close()
                del running[g]
                try:
                    d = json.load(open(out))
                    ts = d['test']
                    tc = d['test_cascade']
                    print(f'[{time.time()-t0:5.0f}s] gpu{g} DONE  {grp} s{sp} seed{sd} | '
                          f'scaled={ts["total"]:.2f} (S_pos={ts["S_pos"]:.3f}, S_p0={ts["S_p0"]:.3f}, S_clean={ts["S_clean"]:.3f}) | '
                          f'cascade={tc["total"]:.2f}', flush=True)
                except Exception as e:
                    print(f'[{time.time()-t0:5.0f}s] gpu{g} FAIL  {grp} s{sp} seed{sd}: {e}', flush=True)

    print('\n=== V2 SINGLE-SEED SUMMARY ON s16..s20 ===', flush=True)
    for grp in ('v2_pcl', 'v2_eupe'):
        fs = sorted(glob.glob(f'{OUT}/{grp}_s*_seed*.json'))
        if not fs:
            continue
        rows = [json.load(open(f))['test'] for f in fs]
        print(f'{grp:10s} n={len(rows)} | mean_total={np.mean([r["total"] for r in rows]):5.2f} '
              f'| S_pos={np.mean([r["S_pos"] for r in rows]):.3f} '
              f'| S_p0={np.mean([r["S_p0"] for r in rows]):.3f} '
              f'| S_clean={np.mean([r["S_clean"] for r in rows]):.3f}', flush=True)


if __name__ == '__main__':
    main()
