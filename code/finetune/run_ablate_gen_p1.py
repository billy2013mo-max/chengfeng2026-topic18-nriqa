"""Small-scale held-out ablation: 过度生成 oversampling and P1-grading losses.
4 variants x 2 strictly unseen splits (s16, s17) on 8 GPUs, pcl 384/2x2 config.
"""
import os, subprocess, sys, time
os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/v3_ablate'
os.makedirs(OUT, exist_ok=True)
VARIANTS = {
    'A_base':   [],
    'B_gen3':   ['--gen-repeat', '3'],
    'C_p1':     ['--emd-weight', '1.2', '--p1-ce-weight', '2.0'],
    'D_gen_p1': ['--gen-repeat', '3', '--emd-weight', '1.2', '--p1-ce-weight', '2.0'],
}
procs = []
gpu = 0
for sp in (16, 17):
    for name, extra in VARIANTS.items():
        tag = f'{name}_s{sp}'
        cmd = ['python3', '-u', '-I', 'code/finetune/train_v3.py',
               '--backbones', 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',
               '--seed', '1', '--epochs', '4', '--batch', '4', '--grad-accum', '2', '--workers', '6',
               '--size', '384', '--tiles', '2', '--clean-repeat', '4', '--p0-clean-weight', '4.5',
               '--split', f'data/resplits/split_s{sp}.csv', '--out', f'{OUT}/{tag}.json'] + extra
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        lf = open(f'{OUT}/{tag}.log', 'w')
        procs.append((tag, subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)))
        print('launched', tag, 'GPU', gpu, flush=True)
        gpu += 1
t0 = time.time()
for tag, p in procs:
    p.wait()
    print(f'[{time.time()-t0:.0f}s] done {tag} rc={p.returncode}', flush=True)
