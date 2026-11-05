"""Held-out validation of the two biggest UNVERIFIED assumptions, on strictly unseen folds s16 & s17:
  (a) does 448 / 3x3 tiles really beat 384 / 2x2 (for two different backbone sets)?
  (b) epoch 3 vs epoch 4 vs their average (per-epoch probs are dumped)
Also gives a same-config re-run with a different seed => noise floor of a single training run.
All runs use identical loss/hyper-params (train_v3 defaults = the A_base recipe).
"""
import os, subprocess, time
os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/v6_val'
os.makedirs(OUT, exist_ok=True)
PCL = 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4'
D3 = 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4'
VARS = [
    ('pcl_384t2_seed2', PCL, 384, 2, 2),
    ('pcl_448t3_seed1', PCL, 448, 3, 1),
    ('d3_384t2_seed1', D3, 384, 2, 1),
    ('d3_448t3_seed1', D3, 448, 3, 1),
]
procs, gpu = [], 0
for sp in (16, 17):
    for tag, bbs, sz, tiles, seed in VARS:
        name = f'{tag}_s{sp}'
        if os.path.exists(f'{OUT}/{name}_epochs.npz') and os.path.exists(f'{OUT}/{name}.json'):
            continue
        cmd = ['python3', '-u', '-I', 'code/finetune/train_v3.py', '--backbones', bbs, '--seed', str(seed),
               '--epochs', '4', '--batch', '4', '--grad-accum', '2', '--workers', '6', '--size', str(sz),
               '--tiles', str(tiles), '--clean-repeat', '4', '--p0-clean-weight', '4.5',
               '--split', f'data/resplits/split_s{sp}.csv', '--dump-epochs', '--out', f'{OUT}/{name}.json']
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu % 8)
        procs.append((name, subprocess.Popen(cmd, env=env, stdout=open(f'{OUT}/{name}.log', 'w'), stderr=subprocess.STDOUT)))
        print('launched', name, 'GPU', gpu % 8, flush=True); gpu += 1
t0 = time.time()
for name, p in procs:
    p.wait(); print(f'[{time.time()-t0:.0f}s] done {name} rc={p.returncode}', flush=True)
