"""Launch Step 2: A/B validation of synthetic over-sharpening data on s16 & s17 splits.
Runs parallel on GPU 0 and GPU 1.
"""
import os, subprocess, sys, time

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/v3_ablate'
os.makedirs(OUT, exist_ok=True)

procs = []
jobs = [
    ('E_synth_s16', 0, 'data/resplits/split_s16_synth_sharp.csv', 'data/manifest_s16_synth_sharp.csv'),
    ('E_synth_s17', 1, 'data/resplits/split_s17_synth_sharp.csv', 'data/manifest_s17_synth_sharp.csv'),
]

for tag, gpu, split_csv, man_csv in jobs:
    cmd = [
        'python3', '-u', '-I', 'code/finetune/train_v3.py',
        '--backbones', 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',
        '--seed', '1',
        '--epochs', '4',
        '--batch', '4',
        '--grad-accum', '2',
        '--workers', '6',
        '--size', '384',
        '--tiles', '2',
        '--clean-repeat', '4',
        '--p0-clean-weight', '4.5',
        '--split', split_csv,
        '--manifest', man_csv,
        '--out', f'{OUT}/{tag}.json'
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    lf = open(f'{OUT}/{tag}.log', 'w')
    p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    procs.append((tag, p, lf))
    print(f"Launched {tag} on GPU {gpu}", flush=True)

t0 = time.time()
for tag, p, lf in procs:
    p.wait()
    lf.close()
    print(f"[{time.time()-t0:.0f}s] Finished {tag} rc={p.returncode}", flush=True)

print("\n--- Running Evaluation on Held-out Validation Test Sets ---", flush=True)
subprocess.run(['python3', 'code/finetune/eval_ablate_gen_p1.py'])
