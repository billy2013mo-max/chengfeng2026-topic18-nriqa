"""Launch 8 Full-Data V9 High-Resolution Spatial-Anomaly-Pooled Specialists across GPUs 0..7
(`launch_v9_fulldata_8gpus.py`).

Every model trains on 87.5% (`3,529 / 4,033`) of the real human-annotated dataset (+39.3% more real
positive defect samples than `s16/s17`!), and across the 8 folds (`Fold 0..7`), 100% of all 4,033
real human-annotated images are used for training AND 100% get Out-Of-Fold (`OOF`) zero-leakage validation!
"""
import json
import os
import subprocess
import time

os.chdir('/wanqing-develop/luowenjing/18')
OUT_DIR = 'results/v9_fulldata'
os.makedirs(OUT_DIR, exist_ok=True)

JOBS = [
    # (name, gpu, fold, focus_classes, backbones, size, tiles, split_csv, man_csv, w_neg, p0_cln, lr_hd, cln_rep)
    # 1. Fold 0 & Fold 1: Giant OverGen + Bokeh + Texture Specialists (PE-Core-G14 + SigLIP2-Giant + C-RADIOv4-H + ConvNeXt/EVA02)
    ('V9_Full_F0_OverGen_Giant', 0, 0, '过度生成,纹理损失,虚化突兀',
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,convnext:full', 448, 2,
     'data/v9_fulldata/split_f0_overgen_bk.csv', 'data/v9_fulldata/manifest_f0_v9.csv', 1.00, 4.2, 1.4e-3, 3),
    ('V9_Full_F1_OverGen_Giant', 1, 1, '过度生成,纹理损失,虚化突兀',
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,eva02:lastk6', 448, 2,
     'data/v9_fulldata/split_f1_overgen_bk.csv', 'data/v9_fulldata/manifest_f1_v9.csv', 1.00, 4.2, 1.4e-3, 3),

    # 2. Fold 2 & Fold 3: High-Res Sharpness + Texture + Blur Specialists (DINOv3-L + C-RADIOv4-H + EVA02-448 + ConvNeXtV2)
    ('V9_Full_F2_SharpTex_HiRes', 2, 2, '过锐,纹理损失,模糊',
     'dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2,
     'data/v9_fulldata/split_f2_sharptex.csv', 'data/v9_fulldata/manifest_f2_v9.csv', 1.00, 4.3, 1.4e-3, 3),
    ('V9_Full_F3_SharpTex_HiRes', 3, 3, '过锐,纹理损失,模糊',
     'pecorel:lastk6,dinov3l:lastk6,cradio:lastk4,convnext:full', 448, 2,
     'data/v9_fulldata/split_f3_sharptex.csv', 'data/v9_fulldata/manifest_f3_v9.csv', 1.00, 4.3, 1.4e-3, 3),

    # 3. Fold 4 & Fold 5: 3x3 Tile (12 Views!) All-Defect + P0 Recall Anchor Specialists (PE-Core-L14 + DINOv3-L + SigLIP2 + EUPE)
    ('V9_Full_F4_AllP0_Tile3', 4, 4, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full', 448, 3,
     'data/v9_fulldata/split_f4_all.csv', 'data/v9_fulldata/manifest_f4_v9.csv', 0.95, 4.2, 1.4e-3, 3),
    ('V9_Full_F5_AllP0_Tile3', 5, 5, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full', 448, 3,
     'data/v9_fulldata/split_f5_all.csv', 'data/v9_fulldata/manifest_f5_v9.csv', 0.95, 4.2, 1.4e-3, 3),

    # 4. Fold 6 & Fold 7: Pure Real Human-Annotated Giant High-Res Specialists (Zero Synthetic Shift!)
    ('V9_Full_F6_RealHiRes_Giant', 6, 6, '模糊,纹理损失,过锐,过度生成,虚化突兀',
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6', 448, 2,
     'data/v9_fulldata/split_f6_real.csv', 'data/manifest_combined.csv', 1.00, 4.2, 1.4e-3, 2),
    ('V9_Full_F7_RealHiRes_Giant', 7, 7, '模糊,纹理损失,过锐,过度生成,虚化突兀',
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6', 448, 2,
     'data/v9_fulldata/split_f7_real.csv', 'data/manifest_combined.csv', 1.00, 4.2, 1.4e-3, 2),
]


def main():
    subprocess.run(['pkill', '-f', 'train_v9_specialist.py'], check=False)
    time.sleep(3)
    print('=== Launching 8 Accelerated Full-Data (8-Fold 100% Coverage) V9 Specialists on GPUs 0..7 ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, fold, focus, bbs, sz, tiles, split_csv, man_csv, w_neg, p0_cln, lr_hd, cln_rep in JOBS:
        out_json = f'{OUT_DIR}/{name}.json'
        out_npz = f'{OUT_DIR}/{name}_probs.npz'
        if os.path.exists(out_json) and os.path.exists(out_npz):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        batch_sz = 9 if tiles == 3 else 8
        grad_acc = 1
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v9_specialist.py',
            '--backbones', bbs,
            '--focus-classes', focus,
            '--seed', str(100 + fold),
            '--epochs', '4',
            '--batch', str(batch_sz),
            '--grad-accum', str(grad_acc),
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
            '--dropout', '0.1',
            '--split', split_csv,
            '--manifest', man_csv,
            '--out', out_json,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        log_f = open(f'{OUT_DIR}/{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_json, log_f)
        print(f'[{time.time()-t0:5.0f}s] Launched {name:28s} (Fold {fold} | sz={sz} t={tiles}) on GPU {gpu}', flush=True)

    while procs:
        time.sleep(15)
        for name in list(procs.keys()):
            p, gpu, out_json, log_f = procs[name]
            if p.poll() is not None:
                log_f.close()
                del procs[name]
                if os.path.exists(out_json):
                    d = json.load(open(out_json))
                    ts = d['test']
                    print(f'[{time.time()-t0:5.0f}s] DONE {name:28s} (GPU {gpu}) | '
                          f'val_focus_F1={d["val_focus_f1"]:.3f} | oof_sca={ts["total"]:.2f} '
                          f'(S_pos={ts["S_pos"]:.3f}, S_p0={ts["S_p0"]:.3f}, S_cln={ts["S_clean"]:.3f})', flush=True)
                else:
                    print(f'[{time.time()-t0:5.0f}s] FAIL {name} (GPU {gpu}) rc={p.returncode}', flush=True)


if __name__ == '__main__':
    main()
