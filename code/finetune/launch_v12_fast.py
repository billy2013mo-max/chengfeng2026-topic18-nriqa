#!/usr/bin/env python3
"""
Launch the V12 ABMIL retrain with the faster 3-backbone / 5-view config on any GPU
that is currently idle (used memory < 5 GB). This lets the V12 queue coexist with the
already-running V13 Qwen3-VL jobs.

Config matches the previous best V12 m1 family (pecoreg + cradio + dinov3l, tiles=2),
but now runs the d1a437a code (saliency gate + 3-layer pyramid + asymmetric soft loss + SWA).
"""
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
os.makedirs('logs_retrain', exist_ok=True)
os.makedirs('results/v12_abmil', exist_ok=True)

SPLITS = ['s16', 's17', 's18', 's19', 's20']
BACKBONES = 'pecoreg_abmil:lastk4,cradio_abmil:lastk4,dinov3l_abmil:lastk4'

jobs = []
for i, s in enumerate(SPLITS):
    jobs.append({
        'name': f'v12fast_{s}',
        'cmd': ['python3', '-u', 'code/finetune/train_v12_abmil_pure_real.py',
                '--split', f'data/resplits/split_{s}.csv',
                '--manifest', 'data/manifest_combined.csv',
                '--backbones', BACKBONES,
                '--size', '448', '--tiles', '2',
                '--epochs', '4', '--batch', '4', '--grad-accum', '2',
                '--seed', str(16 + i), '--workers', '4',
                '--out', f'results/v12_abmil/v12_abmil_{s}.json'],
    })
jobs.append({
    'name': 'v12fast_full_m1',
    'cmd': ['python3', '-u', 'code/finetune/train_v12_abmil_pure_real.py',
            '--split', 'data/split_all_full.csv',
            '--manifest', 'data/manifest_combined.csv',
            '--backbones', BACKBONES,
            '--size', '448', '--tiles', '2',
            '--epochs', '4', '--batch', '4', '--grad-accum', '2',
            '--seed', '101', '--workers', '4',
            '--out', 'results/v12_abmil/v12_abmil_full_m1.json'],
})

FREE_MB = 5000
running = {}
pending = list(jobs)
status = {j['name']: {'state': 'pending', 'gpu': None, 'rc': None,
                      'log': f"logs_retrain/{j['name']}.log"} for j in jobs}


def free_gpus():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits']
    ).decode()
    free = []
    for line in out.strip().splitlines():
        idx, used = [x.strip() for x in line.split(',')]
        if int(used) < FREE_MB:
            free.append(idx)
    return free


def save():
    with open('logs_retrain/v12fast_status.json', 'w') as f:
        json.dump(status, f, indent=2)


def launch(job, gpu):
    log_f = open(f"logs_retrain/{job['name']}.log", 'w')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
    p = subprocess.Popen(job['cmd'], cwd=ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    running[job['name']] = {'p': p, 'gpu': gpu, 'f': log_f, 't': time.time()}
    status[job['name']].update({'state': 'running', 'gpu': gpu})
    print(f"[launch] {job['name']} on GPU {gpu} pid={p.pid}", flush=True)
    save()


def main():
    while pending or running:
        for gpu in free_gpus():
            if not pending:
                break
            if any(v['gpu'] == gpu for v in running.values()):
                continue
            launch(pending.pop(0), gpu)
        time.sleep(10)
        for name, info in list(running.items()):
            rc = info['p'].poll()
            if rc is None:
                continue
            info['f'].close()
            running.pop(name)
            status[name].update({'state': 'done' if rc == 0 else 'failed', 'rc': rc,
                                 'elapsed_s': round(time.time() - info['t'], 1)})
            print(f"[{'OK' if rc == 0 else 'FAIL'}] {name} GPU {info['gpu']} rc={rc}", flush=True)
            save()
    failed = [n for n, s in status.items() if s['state'] != 'done']
    print(f"=== V12 FAST DONE; failures={failed} ===", flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
