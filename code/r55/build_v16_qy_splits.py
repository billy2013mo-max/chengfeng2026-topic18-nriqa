"""V16 splits on the video-grouped qingyu folds (audit 2026-11-14, finding A1).

The shipped V16 OOF (`results/v16_fulldata`) was trained on `data/v9_fulldata/split_f*_v16_cleanog.csv`, whose folds are
frame-level random (`build_v9_fulldata_8folds.py`): 89-121 of ~447 val rows per fold have a sibling frame of the same
video in training. This rebuilds the same recipe on `data/qingyu_splits/fold_k.csv` (group-disjoint, the folds every other
member's OOF uses), including fresh fold-isolated v9_og / v9_bk synthetic rows whose base AND graft are training rows of
that qingyu fold.

  python3 -I code/r55/build_v16_qy_splits.py        # writes data/v16_qy/{split,manifest}_f{k}.csv, images to /tmp/v9_authentic_imgs
"""
import csv
import glob
import os
import sys
import time
from multiprocessing import Pool

REPO = '/wanqing-develop/luowenjing/18'
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, 'code/finetune'))
from build_v9_authentic_synth import CLASSES, find_orig_path, worker_build_overgen_triplet, worker_build_bokeh_cutout  # noqa: E402

OUT = os.path.join(REPO, 'data/v16_qy')
SP_BASE = 110  # name tags s110..s117, disjoint from s80..s97 used by the old folds


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    man_rows = list(csv.DictReader(open('data/manifest_combined.csv', encoding='utf-8')))
    man_map = {r['name']: r for r in man_rows}
    all_names = [r['name'] for r in man_rows]
    qy = {r['name']: int(r['fold']) for r in csv.DictReader(open('data/qingyu_splits/qingyu_8fold.csv', encoding='utf-8'))}
    assert set(qy) == set(all_names), 'qingyu folds and manifest_combined disagree'

    gen_map = {}
    for p in sorted(glob.glob('/tmp/synth/flux3/*.png') + glob.glob('/tmp/synth/flux3/*.jpg')
                    + glob.glob('/tmp/synth/out/*.png') + glob.glob('/tmp/synth/out/*.jpg')):
        bn = os.path.basename(p)
        for tn in all_names:
            if os.path.splitext(tn)[0] in bn:
                gen_map[tn] = p
                break
    print(f'gen_map {len(gen_map)}', flush=True)

    for k in range(8):
        val_set = {n for n, f in qy.items() if f == k}
        tr_names = [n for n in all_names if n not in val_set]
        tr_clean = [n for n in tr_names if all(int(man_map[n][c]) == 0 for c in CLASSES)]
        tr_mild = [n for n in tr_names if int(man_map[n]['过度生成']) == 0 and int(man_map[n]['花屏']) == 0
                   and int(man_map[n]['虚化突兀']) == 0]
        tr_og_pos = [n for n in tr_names if int(man_map[n]['过度生成']) >= 2]
        sp = SP_BASE + k
        og_bases = ([n for n in tr_names if n in gen_map] + [n for n in tr_clean + tr_mild if n not in gen_map])[:180]
        bk_bases = (tr_clean + tr_mild)[100 + k * 25: 100 + k * 25 + 220]
        og_args = [(sp, i, bn, gen_map.get(bn), find_orig_path(tr_og_pos[i % len(tr_og_pos)])) for i, bn in enumerate(og_bases)]
        bk_args = [(sp, i, bn) for i, bn in enumerate(bk_bases)]
        # isolation check: no base / graft from the validation fold (qingyu folds are video-grouped)
        assert not ({a[2] for a in og_args} | {tr_og_pos[i % len(tr_og_pos)] for i in range(len(og_args))} | set(bk_bases)) & val_set
        with Pool(24) as pool:
            syn = [r for sub in pool.map(worker_build_overgen_triplet, og_args) for r in sub]
            syn += [r for sub in pool.map(worker_build_bokeh_cutout, bk_args) for r in sub]
        with open(f'{OUT}/manifest_f{k}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['name'] + CLASSES)
            w.writeheader()
            for r in man_rows + syn:
                w.writerow({c: r[c] for c in ['name'] + CLASSES})
        with open(f'{OUT}/split_f{k}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'split'])
            w.writeheader()
            for n in all_names:
                if n in val_set:
                    w.writerow({'name': n, 'split': 'val'})
                    w.writerow({'name': n, 'split': 'test'})
                else:
                    w.writerow({'name': n, 'split': 'train'})
            for r in syn:
                w.writerow({'name': r['name'], 'split': 'train'})
        print(f'[{time.time() - t0:6.1f}s] fold {k}: train {len(tr_names)} val {len(val_set)} synth {len(syn)}', flush=True)


if __name__ == '__main__':
    main()
