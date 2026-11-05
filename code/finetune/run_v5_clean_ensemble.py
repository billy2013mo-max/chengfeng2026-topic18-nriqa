"""V5 Clean Multi-Seed & High-Resolution Mega-Ensemble Trainer + Predictor:
Trains 8 clean (zero-synthetic-artifact) V5 models across GPUs 0..7 on split_all_full.csv:
  - GPU 0: v5_d3_448_t3_s1   (siglip2 + cradio + dinov3l @ 448x448, 3x3 tiles, seed 1)
  - GPU 1: v5_d3_448_t3_s2   (siglip2 + cradio + dinov3l @ 448x448, 3x3 tiles, seed 2)
  - GPU 2: v5_quad_448_t2_s1 (eupe + siglip2 + cradio + dinov3l @ 448x448, 2x2 tiles, seed 1)
  - GPU 3: v5_quad_448_t2_s2 (eupe + siglip2 + cradio + dinov3l @ 448x448, 2x2 tiles, seed 2)
  - GPU 4: v5_pcl_448_t3_s1  (siglip2 + cradio + pecorel @ 448x448, 3x3 tiles, seed 1)
  - GPU 5: v5_pcl_448_t3_s2  (siglip2 + cradio + pecorel @ 448x448, 3x3 tiles, seed 2)
  - GPU 6: v5_sig2g_384_t3_s1(siglip2g + cradio + dinov3l @ 384x384, 3x3 tiles, seed 1)
  - GPU 7: v5_pcg_384_t2_s1  (siglip2 + cradio + pecoreg @ 384x384, 2x2 tiles, seed 1)

Each model automatically saves:
  - `_ep3.pt` (pure Epoch-3 high-P0-recall checkpoint)
  - `.pt`     (Epoch-4 high-precision checkpoint)
  - `_swa.pt` (50/50 Checkpoint Soup of Ep3 + Ep4)

Then runs 2x Flip TTA on all 24 V5 checkpoints (8 x 3) and fuses them with the 52.43 baseline
models (V2 + V3) to produce the final upgraded pred_nriqa_dataset_test.json.
"""
import json, os, subprocess, sys, time
import numpy as np
import torch

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from run_v3_super_ensemble import infer_v3_ckpt_tta, decode_v3_shielded, GRADE

OUT_DIR = 'results/v5_clean'
os.makedirs(OUT_DIR, exist_ok=True)

JOBS = [
    # (name, gpu, seed, backbones, size, tiles, batch, accum)
    ('v5_d3_448_t3_s1',    0, 1, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 3, 4, 2),
    ('v5_d3_448_t3_s2',    1, 2, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 3, 4, 2),
    ('v5_quad_448_t2_s1',  2, 1, 'eupe:full,siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 448, 2, 4, 2),
    ('v5_quad_448_t2_s2',  3, 2, 'eupe:full,siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 448, 2, 4, 2),
    ('v5_pcl_448_t3_s1',   4, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',           448, 3, 4, 2),
    ('v5_pcl_448_t3_s2',   5, 2, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',           448, 3, 4, 2),
    ('v5_sig2g_384_t3_s1', 6, 1, 'siglip2g:lastk4,cradio:lastk4,dinov3l:lastk4',          384, 3, 4, 2),
    ('v5_pcg_384_t2_s1',   7, 1, 'siglip2:lastk4,cradio:lastk4,pecoreg:lastk4',           384, 2, 4, 2),
]


def step1_train_v5():
    print('=== Step 1: Training 8 Clean V5 Models on GPUs 0..7 ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, sd, bbs, sz, tiles, bs, accum in JOBS:
        out_json = f'{OUT_DIR}/train_{name}.json'
        ckpt_pt  = f'{OUT_DIR}/model_{name}.pt'
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
            '--lr-full', '8.875e-05',
            '--lr-last', '3e-05',
            '--lr-head', '1.5e-03',
            '--wd', '0.01',
            '--dropout', '0.1',
            '--split', 'data/split_all_full.csv',
            '--manifest', 'data/manifest_combined.csv',
            '--out', out_json,
            '--save-state', ckpt_pt,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['HF_HUB_OFFLINE'] = '1'
        env['TRANSFORMERS_OFFLINE'] = '1'
        log_f = open(f'{OUT_DIR}/train_{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_json, ckpt_pt, log_f)
        print(f'[{time.time()-t0:5.0f}s] Launched {name:20s} ({bbs[:35]} @ size={sz}, tiles={tiles}, seed={sd}) on GPU {gpu}', flush=True)

    while procs:
        time.sleep(15)
        for name in list(procs.keys()):
            p, gpu, out_json, ckpt_pt, log_f = procs[name]
            if p.poll() is not None:
                log_f.close()
                del procs[name]
                if os.path.exists(out_json):
                    d = json.load(open(out_json))
                    ts = d['test']
                    print(f'[{time.time()-t0:5.0f}s] DONE {name:20s} (GPU {gpu}) | val_scaled={ts["total"]:.2f} '
                          f'(S_pos={ts["S_pos"]:.3f}, S_p0={ts["S_p0"]:.3f}, S_clean={ts["S_clean"]:.3f})', flush=True)
                else:
                    print(f'[{time.time()-t0:5.0f}s] FAIL {name} (GPU {gpu}) rc={p.returncode}', flush=True)


def step2_infer_v5_tta():
    print('\n=== Step 2: Parallel 2x Flip TTA Inference on 24 V5 Checkpoints (Ep3 + SWA + Ep4) ===', flush=True)
    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]

    cache_v5_npz = f'{OUT_DIR}/test_150_v5_models_probs.npz'
    v5_probs = {}
    if os.path.exists(cache_v5_npz):
        v5_probs.update(np.load(cache_v5_npz, allow_pickle=True)['data'].item())

    v5_ckpts = []
    for name, gpu, sd, bbs, sz, tiles, bs, accum in JOBS:
        dev = f'cuda:{gpu}'
        pt_ep3  = f'{OUT_DIR}/model_{name}_ep3.pt'
        pt_swa  = f'{OUT_DIR}/model_{name}_swa.pt'
        pt_main = f'{OUT_DIR}/model_{name}.pt'
        if os.path.exists(pt_ep3):
            v5_ckpts.append((f'{name}_ep3', pt_ep3, 1.8, dev))
        if os.path.exists(pt_swa):
            v5_ckpts.append((f'{name}_swa', pt_swa, 2.0, dev))
        if os.path.exists(pt_main):
            v5_ckpts.append((name, pt_main, 1.8, dev))

    for name, ckp, w, dev in v5_ckpts:
        if name in v5_probs:
            print(f'[cached] {name:24s} loaded.', flush=True)
            continue
        t0 = time.time()
        p0_i, p1_i, p2_i, g_i = infer_v3_ckpt_tta(ckp, ordered_names, dev)
        v5_probs[name] = (p0_i, p1_i, p2_i, g_i, w)
        np.savez_compressed(cache_v5_npz, data=v5_probs)
        print(f'[{time.time()-t0:5.1f}s] Inferred {name:24s} on {dev}', flush=True)

    print(f'All {len(v5_probs)} V5 checkpoints inferred and cached in {cache_v5_npz}!', flush=True)


if __name__ == '__main__':
    step1_train_v5()
    step2_infer_v5_tta()
