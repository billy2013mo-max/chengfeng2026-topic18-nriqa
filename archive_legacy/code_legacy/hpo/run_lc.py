"""Learning-curve runs: two-backbone recipe trained on 25/50/75% of the s16-s18 train splits.

Goal (see goal: objective score >= 55): quantify how much extra labelled data could buy.
Test/val are the untouched splits' own 1000/500 images; only the train rows are subsampled
(by source). The 100% points already exist in results/phase5/tb_plain_s{16,17,18}_seed1.
"""
import os, subprocess, time

os.chdir('/wanqing-develop/luowenjing/18')
ROOT = 'results/lc'
os.makedirs(ROOT, exist_ok=True)
BB = 'siglip2:lastk4,cradio:lastk4'
JOBS = [(sp, f) for f in (25, 50, 75) for sp in (16, 17, 18)]


def free_gpus():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return [int(l.split(',')[0]) for l in out.strip().splitlines() if int(l.split(',')[1]) < 2000]


running = {}
pending = list(JOBS)
t0 = time.time()
while pending or running:
    for g in [x for x in free_gpus() if x not in running]:
        if not pending:
            break
        sp, f = pending.pop(0)
        out = f'{ROOT}/lc_s{sp}_f{f}_seed1.json'
        if os.path.exists(out):
            continue
        cmd = ['python3', '-u', '-I', 'code/finetune/train_multi.py', '--backbones', BB, '--seed', '1',
               '--epochs', '4', '--batch', '6', '--workers', '8', '--size', '384', '--clean-penalty', '0',
               '--clean-repeat', '7', '--lr-full', '8.875e-05', '--lr-last', '3e-05', '--lr-head', '2.2754e-03',
               '--wd', '0.01', '--dropout', '0', '--split', f'data/lc/split_s{sp}_f{f}.csv', '--out', out]
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
        lf = open(out.replace('.json', '.log'), 'w')
        running[g] = (subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT), out, lf)
        print(f'[{time.time()-t0:6.0f}s] gpu{g} start s{sp} f{f}', flush=True)
        time.sleep(20)
    for g in list(running):
        p, out, lf = running[g]
        if p.poll() is not None:
            lf.close(); print(f'[{time.time()-t0:6.0f}s] gpu{g} done {out} rc={p.returncode}', flush=True)
            del running[g]
    time.sleep(10)
print('ALL DONE', flush=True)
