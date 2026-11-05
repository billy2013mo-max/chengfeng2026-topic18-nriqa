"""Backbone screen: can PE-Core / DINOv3 improve the current two-backbone recipe?

Two-backbone baseline (SigLIP2 + C-RADIO) and the plain triple (EUPE + SigLIP2 +
C-RADIO) already exist on s16-s20 (results/phase5/tb_plain, results/night/4/final).
This runs the same frozen recipe on the same splits, adding each new backbone to
the two-backbone base ("add") and swapping it in for C-RADIO ("swap").

All runs: 4 epochs, batch 6, size 384, lr_full 8.875e-5, lr_last 3e-5,
lr_head 2.2754e-3, wd 0.01, dropout 0, clean_repeat 7, clean_penalty 0.
"""
import json, os, subprocess, time

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/backbone_exp'
os.makedirs(OUT, exist_ok=True)
GPUS = list(range(8))
SPLITS = [16, 17, 18, 19, 20]
SEEDS = [1, 2]

BASE = 'siglip2:lastk4,cradio:lastk4'
CFG = {
    'd3add':  f'{BASE},dinov3l:lastk4',
    'pcladd': f'{BASE},pecorel:lastk4',
    'pcgadd': f'{BASE},pecoreg:lastk4',
    'd3swap': 'siglip2:lastk4,dinov3l:lastk4',
    'pclswap': 'siglip2:lastk4,pecorel:lastk4',
    'pcgswap': 'siglip2:lastk4,pecoreg:lastk4',
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
