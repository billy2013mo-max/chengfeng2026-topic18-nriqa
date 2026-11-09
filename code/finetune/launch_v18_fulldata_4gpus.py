#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/launch_v18_fulldata_4gpus.py
Launches the 4 V18.1 Specialists on 100% Full Training Data (3,574 real images + native corridor quads)
across 4 x H200 GPUs in parallel, and automatically saves V16-anchor-calibrated `*_probs.npz` files.
"""
import csv, glob, json, os, subprocess, sys, time

os.chdir('/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, 'code/finetune')
from data import CLASSES

OUT_DIR = 'results/v18_fulldata'
SPLIT_DIR = 'data/v18_full'
SYNTH_DIR = '/dev/shm/v18_synth_quads'
META_PATH = 'data/v17_splits/v18_synth_metadata.json'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)


def prepare_full_manifest_and_split():
    man_real = list(csv.DictReader(open('data/manifest_combined.csv')))
    split_ref = {r['name']: r['split'] for r in csv.DictReader(open('data/split_all_full.csv'))}
    synth_meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}

    GRADE_MAP = {'P2': 1, 'P1': 2, 'P0': 3, 'NatNeg': 0}
    synth_records = []
    for fname, info in sorted(synth_meta.items()):
        fpath = os.path.join(SYNTH_DIR, fname)
        if not os.path.exists(fpath):
            continue
        g_str = info['grade']
        if g_str not in GRADE_MAP:
            continue
        dim_c = int(info['dim_c'])
        c_name = CLASSES[dim_c]
        row = {c: '0' for c in CLASSES}
        row[c_name] = str(GRADE_MAP[g_str])
        row['name'] = fname
        row['dim_c'] = dim_c
        synth_records.append(row)

    out_split = []
    out_man = []
    for r in man_real:
        nm = r['name']
        sp = split_ref.get(nm, 'train')
        out_split.append({'name': nm, 'split': sp, 'source': 'real'})
        out_man.append({'name': nm, **{c: r[c] for c in CLASSES}})

    for sr in synth_records:
        out_split.append({'name': sr['name'], 'split': 'train', 'source': f'v18_d{sr["dim_c"]}'})
        out_man.append({'name': sr['name'], **{c: sr[c] for c in CLASSES}})

    split_csv = f'{SPLIT_DIR}/split_full_v18.csv'
    man_csv = f'{SPLIT_DIR}/manifest_full_v18.csv'
    with open(split_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'split', 'source'])
        w.writeheader()
        w.writerows(out_split)
    with open(man_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name'] + list(CLASSES))
        w.writeheader()
        w.writerows(out_man_rows if False else out_man)

    n_real = len(man_real)
    n_synth = len(synth_records)
    print(f'Prepared FullData V18.1: {n_real} real rows (all 3,574 available) + {n_synth} native corridor synth rows = {len(out_split)} total', flush=True)


JOBS = [
    ('V18_Full_SpecA_OverGen_TexLoss', 0, '过度生成,纹理损失',
     'pecoreg:lastk1,siglip2g:lastk1,cradio:lastk1,convnext:lastk2',
     'results/v18_paired_kd_s16_s17/V18_SpecA_OverGen_TexLoss_s16_swa.pt',
     'results/v16_specialists/V16_SpecA_OverGen_CleanMask_s16_probs.npz'),

    ('V18_Full_SpecB_Sharp_TexLoss', 1, '过锐,纹理损失',
     'dinov3l:lastk1,cradio:lastk1,eva02:lastk1,convnext:lastk2',
     'results/v18_paired_kd_s16_s17/V18_SpecB_Sharp_TexLoss_s16_swa.pt',
     'results/v16_specialists/V16_SpecB_SharpTex_Microscope_s16_probs.npz'),

    ('V18_Full_SpecC_Blur_Bokeh_Screen', 2, '模糊,纹理损失,虚化突兀,花屏',
     'pecorel:lastk1,dinov3l:lastk1,siglip2:lastk1,eupe:lastk2',
     'results/v18_paired_kd_s16_s17/V18_SpecC_Blur_Bokeh_Screen_s16_swa.pt',
     'results/v16_specialists/V16_SpecC_BlurBokeh_Microscope_s16_probs.npz'),

    ('V18_Full_SpecD_All6_Giant', 3, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecoreg:lastk1,dinov3l:lastk1,cradio:lastk1,eva02:lastk1',
     'results/v18_paired_kd_s16_s17/V18_SpecD_All6_Giant_s16_swa.pt',
     'results/v16_specialists/V16_SpecD_RealGiant_AllGrade_s16_probs.npz'),
]


def launch_all():
    prepare_full_manifest_and_split()
    split_csv = f'{SPLIT_DIR}/split_full_v18.csv'
    man_csv = f'{SPLIT_DIR}/manifest_full_v18.csv'

    procs = []
    print(f'\n=== Launching 4 FullData V18.1 Specialists across GPUs 0..3 ===', flush=True)
    for name, gpu, focus, backbones, init_ckpt, v16_npz in JOBS:
        out_prefix = f'{OUT_DIR}/{name}'
        log_file = f'{out_prefix}.log'

        cmd = [
            'python3', '-u', 'code/finetune/train_v18_paired_kd_specialist.py',
            '--split', split_csv,
            '--manifest', man_csv,
            '--focus-classes', focus,
            '--backbones', backbones,
            '--out', out_prefix,
            '--batch', '12',
            '--size', '448',
            '--tiles', '2',
            '--epochs-s1', '1',
            '--epochs-s2', '1',
            '--workers', '8',
            '--seed', '42',
        ]
        if os.path.exists(init_ckpt):
            cmd.extend(['--init-ckpt', init_ckpt])
        if os.path.exists(v16_npz):
            cmd.extend(['--v16-npz', v16_npz])

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

        lf = open(log_file, 'w')
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        procs.append((name, gpu, p, lf, log_file))
        print(f'  [GPU {gpu}] Launched {name} (PID {p.pid}) -> {log_file}', flush=True)

    print(f'\nAll 4 FullData V18.1 specialists running in parallel! Monitoring progression...', flush=True)
    start_t = time.time()
    while True:
        alive = sum(1 for _, _, p, _, _ in procs if p.poll() is None)
        elapsed = time.time() - start_t
        if alive == 0:
            break
        print(f'[{int(elapsed)}s] {alive}/4 FullData jobs still running...', flush=True)
        time.sleep(30)

    print(f'\nAll 4 FullData V18.1 Specialists completed in {time.time() - start_t:.1f}s!')
    for name, gpu, p, lf, log_file in procs:
        lf.close()
        print(f'  {name} exited with code {p.returncode}')


if __name__ == '__main__':
    launch_all()
