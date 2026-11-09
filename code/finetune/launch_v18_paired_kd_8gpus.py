#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/launch_v18_paired_kd_8gpus.py
Prepares Zero-Leakage V18 Splits/Manifests (including Native-Resolution Corridor-Verified Quadruplets
and Natural Optical Shallow-DOF Hard Negatives `_d5_NatNeg.png`) and launches all 8 V18 Specialists
(`V18_SpecA/B/C/D` on `s16` and `s17`) in parallel across 8 x H200 GPUs.
"""
import csv, glob, json, os, subprocess, sys, time

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
from data import CLASSES, FULL

OUT_DIR = 'results/v18_paired_kd_s16_s17'
SPLIT_DIR = 'data/v17_splits'
SYNTH_DIR = '/dev/shm/v18_synth_quads'
META_PATH = 'data/v17_splits/v18_synth_metadata.json'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)


def prepare_v18_splits_and_manifests():
    man_real = list(csv.DictReader(open('data/manifest_combined.csv')))
    real_by_name = {r['name']: r for r in man_real}

    synth_meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}
    synth_records = []
    GRADE_MAP = {'P2': 1, 'P1': 2, 'P0': 3, 'NatNeg': 0}

    for fname, info in sorted(synth_meta.items()):
        fpath = os.path.join(SYNTH_DIR, fname)
        if not os.path.exists(fpath):
            continue
        seed_name = info['seed_name']
        dim_c = int(info['dim_c'])
        g_str = info['grade']
        if g_str not in GRADE_MAP:
            continue
        c_name = CLASSES[dim_c]
        row = {c: '0' for c in CLASSES}
        row[c_name] = str(GRADE_MAP[g_str])
        row['name'] = fname
        row['seed_name'] = seed_name
        row['dim_c'] = dim_c
        synth_records.append(row)

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
        n_natneg_added = 0
        for sr in synth_records:
            # STRICT ZERO-LEAKAGE RULE: seed must belong strictly to `train` split of s{sp}!
            if split_map.get(sr['seed_name']) == 'train':
                out_split_rows.append({'name': sr['name'], 'split': 'train', 'source': f'v18_d{sr["dim_c"]}'})
                out_man_rows.append({'name': sr['name'], **{c: sr[c] for c in CLASSES}})
                n_synth_added += 1
                if '_NatNeg' in sr['name']:
                    n_natneg_added += 1

        split_csv = f'{SPLIT_DIR}/split_s{sp}_v18.csv'
        man_csv = f'{SPLIT_DIR}/manifest_s{sp}_v18.csv'
        with open(split_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'split', 'source'])
            w.writeheader()
            w.writerows(out_split_rows)
        with open(man_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name'] + list(CLASSES))
            w.writeheader()
            w.writerows(out_man_rows)

        print(f'Prepared s{sp} V18: {len(real_split_rows)} real rows + {n_synth_added} zero-leakage V18 native synth rows (including {n_natneg_added} NatNeg bokeh hard negatives) = {len(out_split_rows)} total', flush=True)


JOBS = [
    ('V18_SpecA_OverGen_TexLoss_s16', 0, 16, '过度生成,纹理损失',
     'pecoreg:lastk1,siglip2g:lastk1,cradio:lastk1,convnext:lastk2',
     'results/v17_end2end_s16_s17/V17_S2R_SpecA_OverGen_TexLoss_s16_ep5.pt',
     'results/v16_specialists/V16_SpecA_OverGen_CleanMask_s16_probs.npz'),
    ('V18_SpecA_OverGen_TexLoss_s17', 1, 17, '过度生成,纹理损失',
     'pecoreg:lastk1,siglip2g:lastk1,cradio:lastk1,convnext:lastk2',
     'results/v17_end2end_s16_s17/V17_S2R_SpecA_OverGen_TexLoss_s17_ep5.pt',
     'results/v16_specialists/V16_SpecA_OverGen_CleanMask_s17_probs.npz'),

    ('V18_SpecB_Sharp_TexLoss_s16', 2, 16, '过锐,纹理损失',
     'dinov3l:lastk1,cradio:lastk1,eva02:lastk1,convnext:lastk2',
     'results/v17_end2end_s16_s17/V17_S2R_SpecB_Sharp_TexLoss_s16_ep5.pt',
     'results/v16_specialists/V16_SpecB_SharpTex_Microscope_s16_probs.npz'),
    ('V18_SpecB_Sharp_TexLoss_s17', 3, 17, '过锐,纹理损失',
     'dinov3l:lastk1,cradio:lastk1,eva02:lastk1,convnext:lastk2',
     'results/v17_end2end_s16_s17/V17_S2R_SpecB_Sharp_TexLoss_s17_ep5.pt',
     'results/v16_specialists/V16_SpecB_SharpTex_Microscope_s17_probs.npz'),

    ('V18_SpecC_Blur_Bokeh_Screen_s16', 4, 16, '模糊,纹理损失,虚化突兀,花屏',
     'pecorel:lastk1,dinov3l:lastk1,siglip2:lastk1,eupe:lastk2',
     'results/v17_end2end_s16_s17/V17_S2R_SpecC_Blur_Bokeh_Screen_s16_ep5.pt',
     'results/v16_specialists/V16_SpecC_BlurBokeh_Microscope_s16_probs.npz'),
    ('V18_SpecC_Blur_Bokeh_Screen_s17', 5, 17, '模糊,纹理损失,虚化突兀,花屏',
     'pecorel:lastk1,dinov3l:lastk1,siglip2:lastk1,eupe:lastk2',
     'results/v17_end2end_s16_s17/V17_S2R_SpecC_Blur_Bokeh_Screen_s17_ep5.pt',
     'results/v16_specialists/V16_SpecC_BlurBokeh_Microscope_s17_probs.npz'),

    ('V18_SpecD_All6_Giant_s16', 6, 16, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecoreg:lastk1,dinov3l:lastk1,cradio:lastk1,eva02:lastk1',
     'results/v17_end2end_s16_s17/V17_S2R_SpecD_All6_Giant_s16_ep5.pt',
     'results/v16_specialists/V16_SpecD_RealGiant_AllGrade_s16_probs.npz'),
    ('V18_SpecD_All6_Giant_s17', 7, 17, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecoreg:lastk1,dinov3l:lastk1,cradio:lastk1,eva02:lastk1',
     'results/v17_end2end_s16_s17/V17_S2R_SpecD_All6_Giant_s17_ep5.pt',
     'results/v16_specialists/V16_SpecD_RealGiant_AllGrade_s17_probs.npz'),
]


def main():
    prepare_v18_splits_and_manifests()
    print('=== Launching 8 V18 Paired-Quadruplet + Teacher-KD Specialists across GPUs 0..7 ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, sp, focus, bbs, init_ckpt, v16_npz in JOBS:
        out_prefix = f'{OUT_DIR}/{name}'
        out_npz = f'{out_prefix}_probs.npz'
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v18_paired_kd_specialist.py',
            '--backbones', bbs,
            '--focus-classes', focus,
            '--init-ckpt', init_ckpt,
            '--v16-npz', v16_npz,
            '--seed', str(600 + gpu),
            '--epochs-s1', '1',
            '--epochs-s2', '1',
            '--batch', '12',
            '--grad-accum', '1',
            '--workers', '8',
            '--size', '448',
            '--tiles', '2',
            '--split', f'{SPLIT_DIR}/split_s{sp}_v18.csv',
            '--manifest', f'{SPLIT_DIR}/manifest_s{sp}_v18.csv',
            '--out', out_prefix,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        log_f = open(f'{OUT_DIR}/{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_npz, log_f)
        print(f'[{time.time()-t0:5.0f}s] Launched {name:35s} (s{sp}) on GPU {gpu}', flush=True)

    while procs:
        time.sleep(15)
        for name in list(procs.keys()):
            p, gpu, out_npz, log_f = procs[name]
            ret = p.poll()
            if ret is not None:
                log_f.close()
                del procs[name]
                status = 'OK' if (ret == 0 and os.path.exists(out_npz)) else f'FAIL(code={ret})'
                print(f'[{time.time()-t0:5.0f}s] Finished {name:35s} on GPU {gpu} -> {status}', flush=True)

    print(f'=== All 8 V18 Specialists Completed in {time.time()-t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
