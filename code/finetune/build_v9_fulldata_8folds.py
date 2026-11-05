"""Build 8-Fold Full-Coverage (`3,529 / 4,033 = 87.5%` Real Train per Fold, 100% Full 4,033 Coverage
across 8 Folds) Splits + Authentic Zero-Ripple Synthetic Ladders for V9 Full-Data Specialists
(`build_v9_fulldata_8folds.py`).
"""
import csv
import glob
import os
import time
from multiprocessing import Pool
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
import sys
sys.path.insert(0, 'code/finetune')

from build_v9_authentic_synth import (
    CLASSES,
    OUT_IMG_DIR,
    find_orig_path,
    worker_build_overgen_triplet,
    worker_build_sharp_triplet,
    worker_build_blur_tex_triplet,
    worker_build_bokeh_cutout,
)

FULL_DIR = 'data/v9_fulldata'
os.makedirs(FULL_DIR, exist_ok=True)
os.makedirs(OUT_IMG_DIR, exist_ok=True)


def build_8fold_full_coverage():
    t0 = time.time()
    with open('data/manifest_combined.csv') as f:
        man_rows = list(csv.DictReader(f))
    man_map = {r['name']: r for r in man_rows}

    # Stratified 8-fold partition of all 4,033 real human-annotated images
    rng = np.random.default_rng(20261105)
    # Group by primary rarest defect signature for exact stratification across 8 folds
    buckets = {}
    for r in man_rows:
        sig = (
            int(r['花屏']) > 0,
            int(r['虚化突兀']) > 0,
            int(r['过度生成']),
            int(r['过锐']),
            int(r['模糊']),
            int(r['纹理损失']),
        )
        buckets.setdefault(sig, []).append(r['name'])

    fold_val_names = [set() for _ in range(8)]
    idx_counter = 0
    for sig in sorted(buckets.keys()):
        names_b = list(buckets[sig])
        rng.shuffle(names_b)
        for n in names_b:
            fold_val_names[idx_counter % 8].add(n)
            idx_counter += 1

    for f_idx in range(8):
        val_set = fold_val_names[f_idx]
        tr_names = [r['name'] for r in man_rows if r['name'] not in val_set]
        print(f"Fold {f_idx}: real_train={len(tr_names)} (87.5%), real_val_oof={len(val_set)} (12.5%)", flush=True)

        # Save pure real split for F6 & F7 (RealHiRes_Giant)
        sp_real_out = f'{FULL_DIR}/split_f{f_idx}_real.csv'
        with open(sp_real_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'split'])
            w.writeheader()
            for r in man_rows:
                nm = r['name']
                if nm in val_set:
                    w.writerow({'name': nm, 'split': 'val'})
                    w.writerow({'name': nm, 'split': 'test'})
                else:
                    w.writerow({'name': nm, 'split': 'train'})

    # Build Full-Data Authentic Zero-Ripple Synthetic Ladders for Folds 0..5
    # Using all 4,033 images with strict fold-isolation (never using a fold's val_set as base or graft source!)
    gen_map = {}
    all_names = [r['name'] for r in man_rows]
    for p in sorted(glob.glob('/tmp/synth/flux3/*.png') + glob.glob('/tmp/synth/flux3/*.jpg') + glob.glob('/tmp/synth/out/*.png') + glob.glob('/tmp/synth/out/*.jpg')):
        bn = os.path.basename(p)
        for tn in all_names:
            stem = os.path.splitext(tn)[0]
            if stem in bn:
                gen_map[tn] = p
                break

    for f_idx in range(6):
        val_set = fold_val_names[f_idx]
        tr_names = [r['name'] for r in man_rows if r['name'] not in val_set]
        tr_clean = [n for n in tr_names if all(int(man_map[n][c]) == 0 for c in CLASSES)]
        tr_mild = [n for n in tr_names if int(man_map[n]['过度生成']) == 0 and int(man_map[n]['花屏']) == 0 and int(man_map[n]['虚化突兀']) == 0]
        tr_og_pos = [n for n in tr_names if int(man_map[n]['过度生成']) >= 2]

        sp_tag = 80 + f_idx  # unique prefix tag `s80..s85`
        og_candidates = [n for n in tr_names if n in gen_map] + [n for n in tr_clean + tr_mild if n not in gen_map]
        og_bases = og_candidates[:200]
        sh_bases = (tr_clean + tr_mild)[f_idx * 30: f_idx * 30 + 160]
        bt_bases = tr_clean[f_idx * 25: f_idx * 25 + 160]
        bk_bases = (tr_clean + tr_mild)[100 + f_idx * 30: 100 + f_idx * 30 + 260]

        og_args = [
            (sp_tag, i, bn, gen_map.get(bn, None), find_orig_path(tr_og_pos[i % len(tr_og_pos)]) if tr_og_pos else None)
            for i, bn in enumerate(og_bases)
        ]
        sh_args = [(sp_tag, i, bn) for i, bn in enumerate(sh_bases)]
        bt_args = [(sp_tag, i, bn) for i, bn in enumerate(bt_bases)]
        bk_args = [(sp_tag, i, bn) for i, bn in enumerate(bk_bases)]

        with Pool(24) as pool:
            if f_idx in (0, 1):
                og_syn = [r for sub in pool.map(worker_build_overgen_triplet, og_args) for r in sub]
                bk_syn = [r for sub in pool.map(worker_build_bokeh_cutout, bk_args) for r in sub]
                syn_rows = og_syn + bk_syn
                suite_tag = 'overgen_bk'
            elif f_idx in (2, 3):
                sh_syn = [r for sub in pool.map(worker_build_sharp_triplet, sh_args) for r in sub]
                bt_syn = [r for sub in pool.map(worker_build_blur_tex_triplet, bt_args) for r in sub]
                syn_rows = sh_syn + bt_syn
                suite_tag = 'sharptex'
            else:
                og_syn = [r for sub in pool.map(worker_build_overgen_triplet, og_args[:150]) for r in sub]
                sh_syn = [r for sub in pool.map(worker_build_sharp_triplet, sh_args[:120]) for r in sub]
                bt_syn = [r for sub in pool.map(worker_build_blur_tex_triplet, bt_args[:120]) for r in sub]
                bk_syn = [r for sub in pool.map(worker_build_bokeh_cutout, bk_args[:180]) for r in sub]
                syn_rows = og_syn + sh_syn + bt_syn + bk_syn
                suite_tag = 'all'

        man_out = f'{FULL_DIR}/manifest_f{f_idx}_v9.csv'
        with open(man_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name'] + CLASSES)
            w.writeheader()
            for r in man_rows:
                w.writerow({k: r[k] for k in ['name'] + CLASSES})
            for r in syn_rows:
                w.writerow(r)

        sp_out = f'{FULL_DIR}/split_f{f_idx}_{suite_tag}.csv'
        with open(sp_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'split'])
            w.writeheader()
            for r in man_rows:
                nm = r['name']
                if nm in val_set:
                    w.writerow({'name': nm, 'split': 'val'})
                    w.writerow({'name': nm, 'split': 'test'})
                else:
                    w.writerow({'name': nm, 'split': 'train'})
            for r in syn_rows:
                w.writerow({'name': r['name'], 'split': 'train'})

        print(f"[{time.time()-t0:.1f}s] Built Fold {f_idx} ({suite_tag}): +{len(syn_rows)} fold-isolated synth images", flush=True)


if __name__ == '__main__':
    build_8fold_full_coverage()
