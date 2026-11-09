#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/launch_v17_cache_extractor_8gpus.py

Launches 8-backbone 8-view feature extraction in parallel across 8 GPUs (GPUs 0..7).
Saves all cached features into /dev/shm/v17_cache_{backbone}.npz.
"""
import os
import sys
import subprocess
import time

os.chdir('/wanqing-develop/luowenjing/18')

JOBS = [
    ('cradio',   0, 8),
    ('dinov3l',  1, 8),
    ('pecoreg',  2, 8),
    ('pecorel',  3, 8),
    ('siglip2g', 4, 6),
    ('siglip2',  5, 8),
    ('convnext', 6, 8),
    ('eupe',     7, 8),
]

def main():
    print('=== Launching Parallel 8-GPU 8-View Feature Extraction to /dev/shm ===', flush=True)
    procs = {}
    t0 = time.time()
    for backbone, gpu, batch in JOBS:
        out_file = f'/dev/shm/v17_cache_{backbone}.npz'
        if os.path.exists(out_file):
            print(f'[Skip] {out_file} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', 'code/finetune/extract_v17_8view_cache.py',
            '--backbone', backbone,
            '--batch', str(batch),
            '--workers', '8',
            '--out', out_file,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        log_f = open(f'/tmp/extract_v17_{backbone}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[backbone] = (p, gpu, out_file, log_f)
        print(f'[{time.time()-t0:4.0f}s] Launched extraction for {backbone:12s} on GPU {gpu}', flush=True)

    while procs:
        time.sleep(10)
        for bb in list(procs.keys()):
            p, gpu, out_file, log_f = procs[bb]
            ret = p.poll()
            if ret is not None:
                log_f.close()
                del procs[bb]
                status = 'OK' if (ret == 0 and os.path.exists(out_file)) else f'FAIL(code={ret})'
                sz = os.path.getsize(out_file)/(1024*1024) if os.path.exists(out_file) else 0
                print(f'[{time.time()-t0:4.0f}s] Completed {bb:12s} on GPU {gpu} -> {status} ({sz:.1f} MB)', flush=True)

    print(f'=== All Feature Caches Ready in {time.time()-t0:.1f}s ===', flush=True)

if __name__ == '__main__':
    main()
