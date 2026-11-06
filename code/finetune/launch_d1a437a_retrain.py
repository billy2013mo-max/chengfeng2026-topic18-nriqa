#!/usr/bin/env python3
"""
code/finetune/launch_d1a437a_retrain.py
Retrain the d1a437a V12 (ABMIL + 3-layer pyramid + asymmetric soft labels + SWA) and
V13 (Qwen3-VL-8B + Q-Align + saliency gate + soft CORN) families on the 5 official
blind splits (s16..s20) plus the full split, using an 8-GPU queue.

Each job is pinned to one GPU. Logs go to logs_retrain/<name>.log and a JSON status
file is written to logs_retrain/status.json.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
LOG_DIR = os.path.join(ROOT, 'logs_retrain')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs('results/v13_vlm_real', exist_ok=True)
os.makedirs('results/v12_abmil', exist_ok=True)

GPUS = [str(i) for i in range(8)]
SPLITS = ['s16', 's17', 's18', 's19', 's20']

V12_BACKBONES = 'internvit_abmil:lastk4,pecoreg_abmil:lastk4,cradio_abmil:lastk4,dinov3l_abmil:lastk4'

jobs = []
port = 29530
for s in SPLITS:
    jobs.append({
        'name': f'v13_{s}',
        'cmd': ['torchrun', '--nproc_per_node=1', f'--master_port={port}',
                'code/finetune/train_vlm_pure_real_corn.py',
                '--split', f'data/resplits/split_{s}.csv',
                '--manifest', 'data/manifest_combined.csv',
                '--model', 'models/qwen3vl-8b',
                '--epochs', '2', '--batch', '12', '--max-pixels', '802816',
                '--lr', '2e-4', '--lr-head', '1e-3', '--rank', '32',
                '--workers', '4', '--grad-ckpt',
                '--out', f'results/v13_vlm_real/vlm_corn_{s}.json'],
    })
    port += 1

jobs.append({
    'name': 'v13_full',
    'cmd': ['torchrun', '--nproc_per_node=1', f'--master_port={port}',
            'code/finetune/train_vlm_pure_real_corn.py',
            '--split', 'data/split_all_full.csv',
            '--manifest', 'data/manifest_combined.csv',
            '--model', 'models/qwen3vl-8b',
            '--epochs', '2', '--batch', '12', '--max-pixels', '802816',
            '--lr', '2e-4', '--lr-head', '1e-3', '--rank', '32',
            '--workers', '4', '--grad-ckpt',
            '--out', 'results/v13_vlm_real/vlm_corn_full.json'],
})
port += 1

for i, s in enumerate(SPLITS):
    jobs.append({
        'name': f'v12_{s}',
        'cmd': ['python3', '-u', 'code/finetune/train_v12_abmil_pure_real.py',
                '--split', f'data/resplits/split_{s}.csv',
                '--manifest', 'data/manifest_combined.csv',
                '--backbones', V12_BACKBONES,
                '--size', '448', '--tiles', '3',
                '--epochs', '4', '--batch', '2', '--grad-accum', '4',
                '--seed', str(16 + i), '--workers', '4',
                '--out', f'results/v12_abmil/v12_abmil_{s}.json'],
    })

jobs.append({
    'name': 'v12_full_m1',
    'cmd': ['python3', '-u', 'code/finetune/train_v12_abmil_pure_real.py',
            '--split', 'data/split_all_full.csv',
            '--manifest', 'data/manifest_combined.csv',
            '--backbones', V12_BACKBONES,
            '--size', '448', '--tiles', '3',
            '--epochs', '4', '--batch', '2', '--grad-accum', '4',
            '--seed', '101', '--workers', '4',
            '--out', 'results/v12_abmil/v12_abmil_full_m1.json'],
})

# ---------------------------------------------------------------------------
free_gpus = list(GPUS)
running = {}   # name -> dict(proc, gpu, log, start)
status = {j['name']: {'state': 'pending', 'gpu': None, 'rc': None, 'log': f"logs_retrain/{j['name']}.log"}
          for j in jobs}
pending = list(jobs)


def save_status():
    with open(os.path.join(LOG_DIR, 'status.json'), 'w') as f:
        json.dump(status, f, indent=2)


def launch_job(job, gpu):
    log_path = status[job['name']]['log']
    log_f = open(os.path.join(ROOT, log_path), 'w')
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = gpu
    env['PYTHONUNBUFFERED'] = '1'
    proc = subprocess.Popen(job['cmd'], cwd=ROOT, env=env,
                            stdout=log_f, stderr=subprocess.STDOUT)
    running[job['name']] = {'proc': proc, 'gpu': gpu, 'log_f': log_f, 'start': time.time()}
    status[job['name']].update({'state': 'running', 'gpu': gpu})
    print(f"[launch] {job['name']} on GPU {gpu} (pid={proc.pid})", flush=True)
    save_status()


def main():
    print(f"=== Launching {len(jobs)} jobs on {len(GPUS)} GPUs ===", flush=True)
    save_status()
    while pending or running:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            launch_job(pending.pop(0), gpu)

        time.sleep(5)
        finished = []
        for name, info in list(running.items()):
            rc = info['proc'].poll()
            if rc is None:
                continue
            finished.append((name, rc))
        for name, rc in finished:
            info = running.pop(name)
            info['log_f'].close()
            free_gpus.append(info['gpu'])
            el = time.time() - info['start']
            status[name].update({'state': 'done' if rc == 0 else 'failed', 'rc': rc,
                                 'elapsed_s': round(el, 1)})
            print(f"[{'OK' if rc == 0 else 'FAIL'}] {name} GPU {info['gpu']} "
                  f"rc={rc} {el/60:.1f}min", flush=True)
            save_status()

    failed = [n for n, s in status.items() if s['state'] != 'done']
    print(f"=== ALL DONE; failures={failed} ===", flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
