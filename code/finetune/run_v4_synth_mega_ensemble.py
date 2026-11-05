"""V4 Mega-Ensemble with Full Augmented ISP Over-Sharpening Data & Multi-Seed Diversity.
Trains 7 models in parallel across GPUs 0, 1, 2, 3, 4, 5, 6:
- 5 Diverse Architectures (DINOv3-Large, EUPE, SigLIP2-Giant, PeCoRe-Giant, PeCoRe-Large)
- Multi-Seed Variance Reduction on the 2 best architectures (seed 1 & seed 2)
- High-Resolution (448x448) and Dense Multi-View (3x3 tiles, 10 views)
- Built-in SWA Checkpoint Soup (Ep3 + Ep4)
- 2x Flip TTA and Shielded Clean Decoding on test_150.
"""
import copy, json, os, subprocess, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import IQADataset, CLASSES, FULL
from models import build_backbone
from train_v2 import MultiNetV2, exact_probs_from_dual
from run_v3_super_ensemble import decode_v3_shielded, infer_v3_ckpt_tta, GRADE

OUT_DIR = 'results/v4_synth'
os.makedirs(OUT_DIR, exist_ok=True)
TEST_IMG_DIR = 'data/test_150'

JOBS = [
    # (name, gpu, seed, backbones, size, tiles, batch, accum)
    ('v4_d3_448_t3_s1',    0, 1, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 3, 4, 2),
    ('v4_quad_448_t2_s1',  1, 1, 'eupe:full,siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 448, 2, 4, 2),
    ('v4_sig2g_384_t3_s1', 2, 1, 'siglip2g:lastk4,cradio:lastk4,dinov3l:lastk4',          384, 3, 4, 2),
    ('v4_pcl_448_t3_s1',   3, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',           448, 3, 4, 2),
    ('v4_pcg_384_t2_s1',   4, 1, 'siglip2:lastk4,cradio:lastk4,pecoreg:lastk4',           384, 2, 4, 2),
    ('v4_d3_448_t3_s2',    5, 2, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 3, 4, 2),
    ('v4_quad_448_t2_s2',  6, 2, 'eupe:full,siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 448, 2, 4, 2),
]


def step1_train_all():
    print('=== Step 1: Training 7 Augmented V4 Models on GPUs 0-6 ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, sd, bbs, sz, tiles, bs, accum in JOBS:
        out_json = f'{OUT_DIR}/train_{name}.json'
        ckpt_pt  = f'{OUT_DIR}/model_{name}.pt'
        if os.path.exists(out_json) and os.path.exists(ckpt_pt):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v3.py',
            '--backbones', bbs,
            '--seed', str(sd),
            '--epochs', '4',
            '--batch', str(bs),
            '--grad-accum', str(accum),
            '--workers', '6',
            '--size', str(sz),
            '--tiles', str(tiles),
            '--clean-repeat', '4',
            '--p0-clean-weight', '4.5',
            '--lr-full', '8.875e-05',
            '--lr-last', '3e-05',
            '--lr-head', '1.5e-03',
            '--wd', '0.01',
            '--dropout', '0.1',
            '--split', 'data/split_all_full_synth.csv',
            '--manifest', 'data/manifest_combined_synth.csv',
            '--out', out_json,
            '--save-state', ckpt_pt,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        log_f = open(f'{OUT_DIR}/train_{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_json, ckpt_pt, log_f)
        print(f'[{time.time()-t0:5.0f}s] Launched {name:20s} ({bbs[:35]}... @ size={sz}, tiles={tiles}) on GPU {gpu}', flush=True)

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


def step2_infer_and_ensemble():
    print('\n=== Step 2: Mega-Ensemble TTA Inference on Test Set ===', flush=True)
    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]

    # 1. Existing V2 probabilities
    v2_probs = np.load('results/v2_super/test_150_all_models_probs.npz', allow_pickle=True)['data'].item()
    v2_fam_w = {'pcl': 1.0, 'd3': 1.0, 'eupe': 0.9, 'base': 0.7}

    # 2. Infer new V4 checkpoints
    v4_probs = {}
    cache_v4_npz = f'{OUT_DIR}/test_150_v4_models_probs.npz'
    if os.path.exists(cache_v4_npz):
        v4_probs.update(np.load(cache_v4_npz, allow_pickle=True)['data'].item())

    v4_ckpts = []
    for name, gpu, sd, bbs, sz, tiles, bs, accum in JOBS:
        # Check both the final checkpoint and the SWA checkpoint
        pt_main = f'{OUT_DIR}/model_{name}.pt'
        pt_swa  = f'{OUT_DIR}/model_{name}_swa.pt'
        dev = f'cuda:{gpu}'
        if os.path.exists(pt_main):
            v4_ckpts.append((name, pt_main, 2.2, dev))
        if os.path.exists(pt_swa):
            v4_ckpts.append((f'{name}_swa', pt_swa, 1.8, dev))

    for name, ckp, w, dev in v4_ckpts:
        if name in v4_probs:
            print(f'[cached] {name:22s} loaded.', flush=True)
            continue
        t0 = time.time()
        p0_i, p1_i, p2_i, g_i = infer_v3_ckpt_tta(ckp, ordered_names, dev)
        v4_probs[name] = (p0_i, p1_i, p2_i, g_i, w)
        np.savez_compressed(cache_v4_npz, data=v4_probs)
        print(f'[{time.time()-t0:5.1f}s] Inferred {name:22s} (2x flip TTA on 150 test images)', flush=True)

    # 3. Combine V2 (10 models) + V4 (augmented 7 models + 7 SWA = 14 models)
    p0_ens, p1_ens, p2_ens, tw_sum = 0.0, 0.0, 0.0, 0.0
    used = []
    for name, (p0_i, p1_i, p2_i, fam, is_iso) in v2_probs.items():
        wi = v2_fam_w.get(fam, 1.0) * (0.60 if is_iso else 0.85)
        p0_ens += wi * p0_i
        p1_ens += wi * p1_i
        p2_ens += wi * p2_i
        tw_sum += wi
        used.append((name, round(wi, 2)))

    for name, (p0_i, p1_i, p2_i, g_i, wi) in v4_probs.items():
        p0_ens += wi * p0_i
        p1_ens += wi * p1_i
        p2_ens += wi * p2_i
        tw_sum += wi
        used.append((name, round(wi, 2)))

    p0_ens /= tw_sum; p1_ens /= tw_sum; p2_ens /= tw_sum
    print(f'\nV4 Augmented Mega-Ensemble combined {len(used)} models: {used}', flush=True)

    # Backup previous best predictions (52.43)
    if os.path.exists('pred_nriqa_dataset_test.json'):
        subprocess.run(['cp', 'pred_nriqa_dataset_test.json', f'{OUT_DIR}/pred_nriqa_dataset_test_prev5243.json'])

    pred_150 = decode_v3_shielded(p0_ens, p1_ens, p2_ens)

    out_list = []
    for i, nm in enumerate(ordered_names):
        lab = {k: GRADE[int(pred_150[i, CLASSES.index(k)])] for k in ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']}
        out_list.append({'image': nm, 'label': lab})

    out_v4_json   = f'{OUT_DIR}/pred_nriqa_dataset_test.json'
    out_root_json = 'pred_nriqa_dataset_test.json'
    json.dump(out_list, open(out_v4_json, 'w'), ensure_ascii=False, indent=2)
    json.dump(out_list, open(out_root_json, 'w'), ensure_ascii=False, indent=2)

    print('\nPer-class grade distribution on nriqa_dataset_test.zip (150 images):', flush=True)
    for ci, c in enumerate(CLASSES):
        vals = [GRADE[int(v)] for v in pred_150[:, ci]]
        uniq = {v: vals.count(v) for v in ['无问题', 'P2', 'P1', 'P0'] if v in vals}
        print(f'  {c:6s}: {uniq}', flush=True)
    clean_cnt = int((pred_150 == 0).all(1).sum())
    print(f'All-clean images: {clean_cnt}/{len(pred_150)} ({100*clean_cnt/len(pred_150):.1f}%)', flush=True)

    prev = json.load(open(f'{OUT_DIR}/pred_nriqa_dataset_test_prev5243.json'))
    same_cells = sum(1 for a, b in zip(prev, out_list) for k in CLASSES if a['label'][k] == b['label'][k])
    diff_imgs = sum(1 for a, b in zip(prev, out_list) if any(a['label'][k] != b['label'][k] for k in CLASSES))
    print(f'Agreement vs 52.43 version: {same_cells}/900 cells ({100*same_cells/900:.1f}%), {diff_imgs}/150 images refined.', flush=True)
    print(f'Successfully updated: {out_root_json} and {out_v4_json}!', flush=True)


if __name__ == '__main__':
    step1_train_all()
    step2_infer_and_ensemble()
