"""Launch 4 additional Full-Data High-Res V7 OverGen-Pro Specialist Models on GPUs 2..5
while simultaneously running 2x Flip TTA on all 18 V6 Multi-Synth Checkpoints on GPUs 6..7!
"""
import json
import os
import subprocess
import sys
import time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from run_v3_super_ensemble import infer_v3_ckpt_tta
from run_v6_ablate_and_specialists import FULL_JOBS, OUT_V6

OUT_V7 = 'results/v7_overgen_pro'
os.makedirs(OUT_V7, exist_ok=True)

EXTRA_V7_JOBS = [
    ('v7_og_d3_448_t3_s2',    2, 2, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',  448, 3, 4, 2),
    ('v7_og_pcl_448_t3_s1',   3, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',  448, 3, 4, 2),
    ('v7_og_sig2g_384_t3_s1', 4, 1, 'siglip2g:lastk4,cradio:lastk4,dinov3l:lastk4', 384, 3, 4, 2),
    ('v7_og_pcg_384_t2_s1',   5, 1, 'siglip2:lastk4,cradio:lastk4,pecoreg:lastk4',  384, 2, 4, 2),
]


def main():
    t0 = time.time()
    procs = []
    for name, gpu, sd, bbs, sz, tiles, bs, accum in EXTRA_V7_JOBS:
        out_json = f'{OUT_V7}/train_{name}.json'
        ckpt_pt = f'{OUT_V7}/model_{name}.pt'
        if os.path.exists(out_json) and os.path.exists(ckpt_pt):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v5.py',
            '--backbones', bbs,
            '--seed', str(sd),
            '--epochs', '4',
            '--batch', str(bs),
            '--grad-accum', str(accum),
            '--workers', '6',
            '--size', str(sz),
            '--tiles', str(tiles),
            '--clean-repeat', '4',
            '--p0-clean-weight', '4.2',
            '--channel-neg-weight', '1.35',
            '--alpha-ex', '0.55',
            '--emd-weight', '0.50',
            '--split', 'data/split_all_full_overgen_pro.csv',
            '--manifest', 'data/manifest_combined_overgen_pro.csv',
            '--out', out_json,
            '--save-state', ckpt_pt,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        lf = open(f'{OUT_V7}/train_{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((name, p, lf))
        print(f'[launched] {name} on GPU {gpu}', flush=True)

    # Simultaneously infer all 18 V6 Multi-Synth checkpoints using GPU 6 & GPU 7
    print('\n=== Running 2x Flip TTA on All 18 V6 Multi-Synth Checkpoints (GPUs 6..7) ===', flush=True)
    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]
    v6_cache_path = f'{OUT_V6}/test_150_v6_models_probs.npz'
    v6_probs = {}
    if os.path.exists(v6_cache_path):
        d = np.load(v6_cache_path, allow_pickle=True)
        if 'data' in d:
            v6_probs.update(d['data'].item())

    idx = 0
    for name, gpu, sd, bbs, sz, tiles, bs, accum in FULL_JOBS:
        for suf, w in [('_ep3', 1.8), ('_swa', 2.0), ('', 1.8)]:
            kname = f'{name}{suf}'
            ckp = f'{OUT_V6}/model_{name}{suf}.pt'
            if not os.path.exists(ckp) or kname in v6_probs:
                continue
            dev = f'cuda:{6 + (idx % 2)}'
            idx += 1
            ti = time.time()
            p0_i, p1_i, p2_i, g_i = infer_v3_ckpt_tta(ckp, ordered_names, dev)
            v6_probs[kname] = (p0_i, p1_i, p2_i, g_i, w)
            np.savez_compressed(v6_cache_path, data=v6_probs)
            print(f'[{time.time()-ti:5.1f}s] Inferred V6 {kname:24s} on {dev} ({len(v6_probs)}/18)', flush=True)

    print(f'All {len(v6_probs)} V6 checkpoints cached in {v6_cache_path}!', flush=True)

    for name, p, lf in procs:
        rc = p.wait()
        lf.close()
        print(f'[{time.time()-t0:.0f}s] Finished extra V7 specialist {name} rc={rc}', flush=True)


if __name__ == '__main__':
    main()
