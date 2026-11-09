"""Build 8-Fold Full-Data Clean-Masked (`v9_og + v9_bk` ONLY, ZERO `v9_sh / v9_bt`) Splits & Manifests
for `V16_Full_F0..F7` (`code/finetune/build_v16_fulldata_8folds.py`).

Strictly preserves the exact same 8-fold real train/val partitions (`split_f{0..7}_real.csv`) as
`V9_Full` and `V10_Full`, and ensures every fold `F0..F7` has strictly fold-isolated (`base not in val_set`
and `graft not in val_set`) `v9_og` (`过度生成` P2/P1/P0) + `v9_bk` (`虚化突兀` P0) counterfactuals,
with ZERO `v9_sh / v9_bt` bilateral-filter synthetic shift.
"""
import csv
import glob
import os
import time
from multiprocessing import Pool

os.chdir('/wanqing-develop/luowenjing/18')
import sys
sys.path.insert(0, 'code/finetune')

from build_v9_authentic_synth import (
    CLASSES,
    OUT_IMG_DIR,
    find_orig_path,
    worker_build_overgen_triplet,
    worker_build_bokeh_cutout,
)

FULL_DIR = 'data/v9_fulldata'
os.makedirs(FULL_DIR, exist_ok=True)
os.makedirs(OUT_IMG_DIR, exist_ok=True)


def main():
    t0 = time.time()
    with open('data/manifest_combined.csv') as f:
        man_rows = list(csv.DictReader(f))
    man_map = {r['name']: r for r in man_rows}
    all_names = [r['name'] for r in man_rows]

    gen_map = {}
    for p in sorted(
        glob.glob('/tmp/synth/flux3/*.png')
        + glob.glob('/tmp/synth/flux3/*.jpg')
        + glob.glob('/tmp/synth/out/*.png')
        + glob.glob('/tmp/synth/out/*.jpg')
    ):
        bn = os.path.basename(p)
        for tn in all_names:
            stem = os.path.splitext(tn)[0]
            if stem in bn:
                gen_map[tn] = p
                break

    for f_idx in range(8):
        # Load exact real train/val partition from split_f{f_idx}_real.csv
        with open(f'{FULL_DIR}/split_f{f_idx}_real.csv') as f:
            sp_rows = list(csv.DictReader(f))
        val_set = {r['name'] for r in sp_rows if r['split'] == 'val'}
        tr_names = [r['name'] for r in man_rows if r['name'] not in val_set]

        if f_idx in (0, 1):
            # Fold 0 & 1 already have pure og_syn + bk_syn in manifest_f{0,1}_v9.csv
            with open(f'{FULL_DIR}/manifest_f{f_idx}_v9.csv') as f:
                m_rows = list(csv.DictReader(f))
            syn_rows = [r for r in m_rows if r['name'].startswith(('v9_og_', 'v9_bk_'))]
        elif f_idx in (4, 5):
            # Fold 4 & 5 already have og_syn + bk_syn in manifest_f{4,5}_v9.csv; filter out v9_sh_ and v9_bt_!
            with open(f'{FULL_DIR}/manifest_f{f_idx}_v9.csv') as f:
                m_rows = list(csv.DictReader(f))
            syn_rows = [r for r in m_rows if r['name'].startswith(('v9_og_', 'v9_bk_'))]
        else:
            # Build fold-isolated og_syn + bk_syn for Fold 2, 3, 6, 7
            tr_clean = [n for n in tr_names if all(int(man_map[n][c]) == 0 for c in CLASSES)]
            tr_mild = [
                n for n in tr_names
                if int(man_map[n]['过度生成']) == 0
                and int(man_map[n]['花屏']) == 0
                and int(man_map[n]['虚化突兀']) == 0
            ]
            tr_og_pos = [n for n in tr_names if int(man_map[n]['过度生成']) >= 2]
            sp_tag = 90 + f_idx  # unique prefix tag s92, s93, s96, s97
            og_candidates = [n for n in tr_names if n in gen_map] + [
                n for n in tr_clean + tr_mild if n not in gen_map
            ]
            og_bases = og_candidates[:180]
            bk_bases = (tr_clean + tr_mild)[100 + f_idx * 25 : 100 + f_idx * 25 + 220]

            og_args = [
                (
                    sp_tag,
                    i,
                    bn,
                    gen_map.get(bn, None),
                    find_orig_path(tr_og_pos[i % len(tr_og_pos)]) if tr_og_pos else None,
                )
                for i, bn in enumerate(og_bases)
            ]
            bk_args = [(sp_tag, i, bn) for i, bn in enumerate(bk_bases)]
            with Pool(24) as pool:
                og_syn = [r for sub in pool.map(worker_build_overgen_triplet, og_args) for r in sub]
                bk_syn = [r for sub in pool.map(worker_build_bokeh_cutout, bk_args) for r in sub]
            syn_rows = og_syn + bk_syn

        man_out = f'{FULL_DIR}/manifest_f{f_idx}_v16_cleanog.csv'
        with open(man_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name'] + CLASSES)
            w.writeheader()
            for r in man_rows:
                w.writerow({k: r[k] for k in ['name'] + CLASSES})
            for r in syn_rows:
                w.writerow({k: r[k] for k in ['name'] + CLASSES})

        sp_out = f'{FULL_DIR}/split_f{f_idx}_v16_cleanog.csv'
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

        print(
            f'[{time.time()-t0:5.1f}s] Fold {f_idx}: real_train={len(tr_names)}, '
            f'real_val={len(val_set)}, clean_og_bk_synth={len(syn_rows)}',
            flush=True,
        )


if __name__ == '__main__':
    main()
