"""Launch 8-Fold Full-Data (`87.5%` = `3,126` Real Train per Fold + Clean-Masked `v9_og + v9_bk`,
ZERO `v9_sh / v9_bt` Synthetic Shift) End-to-End 448px + 1:1 Native Contrast-Residual Microscope
Specialists across GPUs 0..7 (`code/finetune/launch_v16_fulldata_8gpus.py`).

Closes the critical 8-fold coverage holes in `V9_Full` and `V10_Full`:
  - Every fold `F0..F7` trains with `--focus-classes 模糊,纹理损失,过锐,花屏,过度生成,虚化突兀` (full 1.00 weight
    on all 6 classes across 100% of the dataset!).
  - Uses `train_v16_microscope_specialist.py` so `v9_og` ONLY supervises `过度生成`, `v9_bk` ONLY
    supervises `虚化突兀`, and `模糊, 纹理损失, 过锐, 花屏` are supervised by 100% pure real human annotations!
"""
import json
import os
import subprocess
import time

os.chdir('/wanqing-develop/luowenjing/18')
OUT_DIR = 'results/v16_fulldata'
os.makedirs(OUT_DIR, exist_ok=True)

ALL6 = '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀'

JOBS = [
    # (name, gpu, fold, backbones, size, tiles, split_csv, man_csv, w_neg, p0_cln, lr_hd, cln_rep)
    ('V16_Full_F0_AllGrade_Microscope', 0, 0,
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2,
     'data/v9_fulldata/split_f0_v16_cleanog.csv', 'data/v9_fulldata/manifest_f0_v16_cleanog.csv', 1.05, 4.2, 1.35e-3, 2),
    ('V16_Full_F1_AllGrade_Microscope', 1, 1,
     'pecoreg:lastk4,siglip2g:lastk4,dinov3l:lastk6,cradio:lastk4,convnext:full', 448, 2,
     'data/v9_fulldata/split_f1_v16_cleanog.csv', 'data/v9_fulldata/manifest_f1_v16_cleanog.csv', 1.05, 4.2, 1.35e-3, 2),
    ('V16_Full_F2_AllGrade_Microscope', 2, 2,
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2,
     'data/v9_fulldata/split_f2_v16_cleanog.csv', 'data/v9_fulldata/manifest_f2_v16_cleanog.csv', 1.05, 4.2, 1.35e-3, 2),
    ('V16_Full_F3_AllGrade_Microscope', 3, 3,
     'pecoreg:lastk4,siglip2g:lastk4,dinov3l:lastk6,cradio:lastk4,convnext:full', 448, 2,
     'data/v9_fulldata/split_f3_v16_cleanog.csv', 'data/v9_fulldata/manifest_f3_v16_cleanog.csv', 1.05, 4.2, 1.35e-3, 2),
    ('V16_Full_F4_AllGrade_Microscope', 4, 4,
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2,
     'data/v9_fulldata/split_f4_v16_cleanog.csv', 'data/v9_fulldata/manifest_f4_v16_cleanog.csv', 1.05, 4.2, 1.35e-3, 2),
    ('V16_Full_F5_AllGrade_Microscope', 5, 5,
     'pecoreg:lastk4,siglip2g:lastk4,dinov3l:lastk6,cradio:lastk4,convnext:full', 448, 2,
     'data/v9_fulldata/split_f5_v16_cleanog.csv', 'data/v9_fulldata/manifest_f5_v16_cleanog.csv', 1.05, 4.2, 1.35e-3, 2),
    ('V16_Full_F6_AllGrade_Microscope', 6, 6,
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2,
     'data/v9_fulldata/split_f6_v16_cleanog.csv', 'data/v9_fulldata/manifest_f6_v16_cleanog.csv', 1.05, 4.2, 1.35e-3, 2),
    ('V16_Full_F7_AllGrade_Microscope', 7, 7,
     'pecoreg:lastk4,siglip2g:lastk4,dinov3l:lastk6,cradio:lastk4,convnext:full', 448, 2,
     'data/v9_fulldata/split_f7_v16_cleanog.csv', 'data/v9_fulldata/manifest_f7_v16_cleanog.csv', 1.05, 4.2, 1.35e-3, 2),
]


def main():
    print('=== Launching 8-Fold Full-Data V16 AllGrade Microscope Specialists across GPUs 0..7 ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, fold, bbs, sz, tiles, split_csv, man_csv, w_neg, p0_cln, lr_hd, cln_rep in JOBS:
        out_base = f'{OUT_DIR}/{name}'
        out_npz = f'{out_base}_probs.npz'
        if os.path.exists(out_npz):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v16_microscope_specialist.py',
            '--backbones', bbs,
            '--focus-classes', ALL6,
            '--seed', str(600 + fold),
            '--epochs', '3',
            '--batch', '8',
            '--grad-accum', '1',
            '--workers', '8',
            '--size', str(sz),
            '--tiles', str(tiles),
            '--clean-repeat', str(cln_rep),
            '--p0-clean-weight', str(p0_cln),
            '--channel-neg-weight', str(w_neg),
            '--lr-full', '7.5e-05',
            '--lr-last', '2.6e-05',
            '--lr-head', str(lr_hd),
            '--wd', '0.012',
            '--dropout', '0.12',
            '--alpha-ex', '0.58',
            '--split', split_csv,
            '--manifest', man_csv,
            '--out', out_base,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        log_f = open(f'{OUT_DIR}/{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_npz, log_f)
        print(f'[{time.time()-t0:5.0f}s] Launched {name:34s} (Fold {fold} | sz={sz} t={tiles}) on GPU {gpu}', flush=True)

    while procs:
        time.sleep(15)
        for name in list(procs.keys()):
            p, gpu, out_npz, log_f = procs[name]
            ret = p.poll()
            if ret is not None:
                log_f.close()
                ok = os.path.exists(out_npz)
                if ok and os.path.exists(f'{OUT_DIR}/{name}.json'):
                    d = json.load(open(f'{OUT_DIR}/{name}.json'))
                    print(
                        f'[{time.time()-t0:5.0f}s] DONE {name:34s} (GPU {gpu}, exit={ret}) | '
                        f'Doc1={d["doc1"]:.2f} Doc2={d["doc2"]:.2f} (S_pos={d["S_pos"]:.3f} S_p0={d["S_p0"]:.3f} Clean2={d["Clean2"]:.3f})',
                        flush=True,
                    )
                else:
                    print(f'[{time.time()-t0:5.0f}s] FAIL {name:34s} (GPU {gpu}, exit={ret}, npz={ok})', flush=True)
                    lines = open(f'{OUT_DIR}/{name}.log').readlines()[-15:]
                    print(''.join('    ' + l for l in lines), flush=True)
                del procs[name]

    print(f'=== ALL 8 FULL-DATA V16 SPECIALISTS COMPLETED in {time.time()-t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
