"""Backbone comparison after fixing the two training bugs in train_multi.py:
  * pos_weight used Y.numel() (=14N) instead of the sample count N -> every channel hit the clamp 20
  * P0_IDX was [2,5,8,11,12,13]; the correct P0 columns are [2,5,8,9,12,13]

Same recipe as results/backbone_exp (4 epochs, batch 6, size 384, clean_repeat 7,
clean_penalty 0, lr_full 8.875e-5, lr_last 3e-5, lr_head 2.2754e-3, wd 0.01, dropout 0),
s16-s20 x seed1/2. Configs: current two-backbone baseline, +PE-Core-L, +PE-Core-G.
"""
import json, os, subprocess, time

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/backbone_fixed'
os.makedirs(OUT, exist_ok=True)
GPUS = list(range(8))
SPLITS = [16, 17, 18, 19, 20]
SEEDS = [1, 2]

BASE = 'siglip2:lastk4,cradio:lastk4'
CFG = {
    'base': BASE,
    'pcl':  f'{BASE},pecorel:lastk4',
    'pcg':  f'{BASE},pecoreg:lastk4',
}

JOBS = [(g, sp, sd) for g in CFG for sp in SPLITS for sd in SEEDS
        if not os.path.exists(f'{OUT}/{g}_s{sp}_seed{sd}.json')]


def cmd(g, sp, sd, out):
    return ['python3', '-u', '-I', 'code/finetune/train_multi.py',
            '--backbones', CFG[g], '--seed', str(sd), '--epochs', '4', '--batch', '6',
            '--workers', '8', '--size', '384', '--clean-penalty', '0', '--clean-repeat', '7',
            '--lr-full', '8.875e-05', '--lr-last', '3e-05', '--lr-head', '2.2754e-03',
            '--wd', '0.01', '--dropout', '0',
            '--split', f'data/resplits/split_s{sp}.csv', '--out', out]


def gpu_free():
    o = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True).stdout
    fr = []
    for line in o.strip().splitlines():
        i, m = line.split(',')
        if int(m) < 4000:
            fr.append(int(i))
    return [g for g in GPUS if g in fr]


def main():
    pending = list(JOBS); running = {}; t0 = time.time()
    print(f'{len(pending)} jobs', flush=True)
    while pending or running:
        for g in [x for x in gpu_free() if x not in running]:
            if not pending:
                break
            grp, sp, sd = pending.pop(0)
            out = f'{OUT}/{grp}_s{sp}_seed{sd}.json'
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
            env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
            f = open(out.replace('.json', '.log'), 'w')
            p = subprocess.Popen(cmd(grp, sp, sd, out), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, grp, sp, sd, out, f)
            print(f'[{time.time()-t0:6.0f}s] gpu{g} start {grp} s{sp} seed{sd}', flush=True)
        time.sleep(20)
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
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
