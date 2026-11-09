#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch 8 End-to-End Two-Stage Sim2Real Microscope Specialists (`V17_S2R_SpecA/B/C/D`)
across 8 x NVIDIA H200 GPUs (GPUs 0..7) on Zero-Leakage Splits `s16` and `s17`
(`code/finetune/launch_v17_end2end_s16_s17_8gpus.py`).
"""
import csv
import glob
import json
import os
import subprocess
import sys
import time

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
from data import CLASSES, FULL
from extract_native_phys_feats import extract_for_names

OUT_DIR = 'results/v17_end2end_s16_s17'
SPLIT_DIR = 'data/v17_splits'
SYNTH_DIR = '/dev/shm/v17_synth_quads'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)


def prepare_v17_splits_and_manifests():
    man_real = list(csv.DictReader(open('data/manifest_combined.csv')))
    real_by_name = {r['name']: r for r in man_real}
    stem_to_real_name = {os.path.splitext(r['name'])[0]: r['name'] for r in man_real}

    # Discover all verified quadruplets in /dev/shm/v17_synth_quads/
    synth_records = []
    GRADE_MAP = {'P2': 1, 'P1': 2, 'P0': 3}
    for dim_c in range(6):
        c_name = CLASSES[dim_c]
        p0_files = sorted(glob.glob(f'{SYNTH_DIR}/*_d{dim_c}_P0.png'))
        for p0_p in p0_files:
            bn = os.path.basename(p0_p)
            stem = bn[:bn.index(f'_d{dim_c}_P0.png')]
            if stem not in stem_to_real_name:
                continue
            seed_name = stem_to_real_name[stem]
            if c_name in FULL:
                grades_to_use = ['P2', 'P1', 'P0']
            else:
                grades_to_use = ['P0']
            for g_str in grades_to_use:
                fname = f'{stem}_d{dim_c}_{g_str}.png'
                fpath = os.path.join(SYNTH_DIR, fname)
                if not os.path.exists(fpath):
                    continue
                row = {c: '0' for c in CLASSES}
                row[c_name] = str(GRADE_MAP[g_str])
                row['name'] = fname
                row['seed_name'] = seed_name
                row['dim_c'] = dim_c
                synth_records.append(row)

    # Pre-extract 60D native physical features ONCE before launching 8 GPU workers
    all_names_to_cache = list(real_by_name.keys()) + [r['name'] for r in synth_records]
    if os.path.exists('results/v2_full/template_150.json'):
        tpl = json.load(open('results/v2_full/template_150.json'))
        all_names_to_cache.extend([e['image'] for e in tpl])
    all_names_to_cache = sorted(set(all_names_to_cache))
    print(f'Ensuring 60D physical features are cached for {len(all_names_to_cache)} images...', flush=True)
    extract_for_names(all_names_to_cache, 'results/v8_experts/native_phys_feats.npz')

    for sp in (16, 17):
        real_split_rows = list(csv.DictReader(open(f'data/resplits/split_s{sp}.csv')))
        split_map = {r['name']: r['split'] for r in real_split_rows}

        out_split_rows = []
        out_man_rows = []
        for r in real_split_rows:
            nm = r['name']
            out_split_rows.append({'name': nm, 'split': r['split'], 'source': 'real'})
            mr = real_by_name[nm]
            out_man_rows.append({'name': nm, **{c: mr[c] for c in CLASSES}})

        n_synth_added = 0
        for sr in synth_records:
            # STRICT ZERO-LEAKAGE RULE: seed must belong strictly to `train` split of s{sp}!
            if split_map.get(sr['seed_name']) == 'train':
                out_split_rows.append({'name': sr['name'], 'split': 'train', 'source': f'v17_d{sr["dim_c"]}'})
                out_man_rows.append({'name': sr['name'], **{c: sr[c] for c in CLASSES}})
                n_synth_added += 1

        split_csv = f'{SPLIT_DIR}/split_s{sp}_v17.csv'
        man_csv = f'{SPLIT_DIR}/manifest_s{sp}_v17.csv'
        with open(split_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'split', 'source'])
            w.writeheader()
            w.writerows(out_split_rows)
        with open(man_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name'] + list(CLASSES))
            w.writeheader()
            w.writerows(out_man_rows)

        print(f'Prepared s{sp}: {len(real_split_rows)} real rows + {n_synth_added} zero-leakage verified synth rows = {len(out_split_rows)} total', flush=True)


JOBS = [
    # (name, gpu, sp, focus_classes, backbones, sz, tiles, ep_s1, ep_s2, w_neg, p0_cln, lr_hd, cln_rep)
    ('V17_S2R_SpecA_OverGen_TexLoss_s16', 0, 16, '过度生成,纹理损失',
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,convnext:full', 448, 2, 3, 2, 1.05, 4.2, 1.3e-3, 2),
    ('V17_S2R_SpecA_OverGen_TexLoss_s17', 1, 17, '过度生成,纹理损失',
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,convnext:full', 448, 2, 3, 2, 1.05, 4.2, 1.3e-3, 2),

    ('V17_S2R_SpecB_Sharp_TexLoss_s16', 2, 16, '过锐,纹理损失',
     'dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2, 3, 2, 1.05, 4.3, 1.3e-3, 2),
    ('V17_S2R_SpecB_Sharp_TexLoss_s17', 3, 17, '过锐,纹理损失',
     'dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2, 3, 2, 1.05, 4.3, 1.3e-3, 2),

    ('V17_S2R_SpecC_Blur_Bokeh_Screen_s16', 4, 16, '模糊,纹理损失,虚化突兀,花屏',
     'pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full', 448, 2, 3, 2, 1.02, 4.2, 1.3e-3, 2),
    ('V17_S2R_SpecC_Blur_Bokeh_Screen_s17', 5, 17, '模糊,纹理损失,虚化突兀,花屏',
     'pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full', 448, 2, 3, 2, 1.02, 4.2, 1.3e-3, 2),

    ('V17_S2R_SpecD_All6_Giant_s16', 6, 16, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6', 448, 2, 3, 2, 1.02, 4.2, 1.3e-3, 2),
    ('V17_S2R_SpecD_All6_Giant_s17', 7, 17, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6', 448, 2, 3, 2, 1.02, 4.2, 1.3e-3, 2),
]


def main():
    prepare_v17_splits_and_manifests()
    print('=== Launching 8 V17 Two-Stage Sim2Real End-to-End Specialists across GPUs 0..7 ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, sp, focus, bbs, sz, tiles, ep_s1, ep_s2, w_neg, p0_cln, lr_hd, cln_rep in JOBS:
        out_prefix = f'{OUT_DIR}/{name}'
        out_npz = f'{out_prefix}_probs.npz'
        if os.path.exists(out_npz):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v17_end2end_sim2real_specialist.py',
            '--backbones', bbs,
            '--focus-classes', focus,
            '--seed', str(500 + gpu),
            '--epochs-s1', str(ep_s1),
            '--epochs-s2', str(ep_s2),
            '--batch', '8',
            '--grad-accum', '1',
            '--workers', '8',
            '--size', str(sz),
            '--tiles', str(tiles),
            '--clean-repeat', str(cln_rep),
            '--p0-clean-weight', str(p0_cln),
            '--channel-neg-weight', str(w_neg),
            '--lr-full', '8.0e-05',
            '--lr-last', '2.8e-05',
            '--lr-head', str(lr_hd),
            '--wd', '0.01',
            '--dropout', '0.12',
            '--split', f'{SPLIT_DIR}/split_s{sp}_v17.csv',
            '--manifest', f'{SPLIT_DIR}/manifest_s{sp}_v17.csv',
            '--out', out_prefix,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        log_f = open(f'{OUT_DIR}/{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_npz, log_f)
        print(f'[{time.time()-t0:5.0f}s] Launched {name:38s} (s{sp} | Stage1={ep_s1}ep Stage2={ep_s2}ep) on GPU {gpu}', flush=True)

    while procs:
        time.sleep(20)
        for name in list(procs.keys()):
            p, gpu, out_npz, log_f = procs[name]
            ret = p.poll()
            if ret is not None:
                log_f.close()
                del procs[name]
                status = 'OK' if (ret == 0 and os.path.exists(out_npz)) else f'FAIL(code={ret})'
                print(f'[{time.time()-t0:5.0f}s] Finished {name:38s} on GPU {gpu} -> {status}', flush=True)

    print(f'=== All 8 V17 Two-Stage Sim2Real End-to-End Specialists Completed in {time.time()-t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
