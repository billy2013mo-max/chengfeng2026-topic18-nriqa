"""Stage B: Launch 8 Full-Data (4,033 Real Images = 3,573 Initial + 460 Supplementary, 100% 8-Fold Coverage)
V10 Clean-Masked + 1:1 Native-Pixel Microscope Specialists across GPUs 0..7 (`launch_v10_stageB_fulldata_8gpus.py`).

Uses H200-optimized `batch=8, grad_accum=1, clean_repeat=2` and rare-only oversampling for ~2x faster training (~25-28 mins),
then automatically executes `build_v10_fulldata_test150_submission.py` to generate the new Full-Data `pred_nriqa_dataset_test.json`!
"""
import json, os, subprocess, sys, time

os.chdir('/wanqing-develop/luowenjing/18')
OUT_DIR = 'results/v10_fulldata'
os.makedirs(OUT_DIR, exist_ok=True)

# First ensure any previous s16/s17 V10 processes are cleanly stopped so GPUs 0..7 are 100% free
subprocess.run(['pkill', '-f', 'train_v10_clean_native_specialist.py'], check=False)
time.sleep(3)

JOBS = [
    # (name, gpu, fold, focus_classes, backbones, size, tiles, real_only, split_csv, man_csv, w_neg, p0_cln, lr_hd, cln_rep)
    # 1. Fold 0 & Fold 1: Giant OverGen Clean-Masked Specialists (PE-Core-G14 + SigLIP2-Giant + C-RADIOv4-H + ConvNeXt/EVA02)
    ('V10_Full_F0_OverGen_CleanMask', 0, 0, '过度生成',
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,convnext:full', 448, 2, False,
     'data/v9_fulldata/split_f0_overgen_bk.csv', 'data/v9_fulldata/manifest_f0_v9.csv', 1.05, 4.2, 1.4e-3, 2),
    ('V10_Full_F1_OverGen_CleanMask', 1, 1, '过度生成',
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,eva02:lastk6', 448, 2, False,
     'data/v9_fulldata/split_f1_overgen_bk.csv', 'data/v9_fulldata/manifest_f1_v9.csv', 1.05, 4.2, 1.4e-3, 2),

    # 2. Fold 2 & Fold 3: 1:1 Native-Pixel Edge-Microscope Sharpness + Texture Specialists
    ('V10_Full_F2_SharpTex_Microscope', 2, 2, '过锐,纹理损失',
     'dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2, False,
     'data/v9_fulldata/split_f2_sharptex.csv', 'data/v9_fulldata/manifest_f2_v9.csv', 1.05, 4.3, 1.4e-3, 2),
    ('V10_Full_F3_SharpTex_Microscope', 3, 3, '过锐,纹理损失',
     'pecorel:lastk6,dinov3l:lastk6,cradio:lastk4,convnext:full', 448, 2, False,
     'data/v9_fulldata/split_f3_sharptex.csv', 'data/v9_fulldata/manifest_f3_v9.csv', 1.05, 4.3, 1.4e-3, 2),

    # 3. Fold 4 & Fold 5: 3x3 Tile (13 Views!) Blur + Texture + Bokeh Clean-Masked Specialists
    ('V10_Full_F4_BlurBokeh_Tile3', 4, 4, '模糊,纹理损失,虚化突兀',
     'pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full', 448, 3, False,
     'data/v9_fulldata/split_f4_all.csv', 'data/v9_fulldata/manifest_f4_v9.csv', 1.05, 4.2, 1.4e-3, 2),
    ('V10_Full_F5_BlurBokeh_Tile3', 5, 5, '模糊,纹理损失,虚化突兀',
     'pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full', 448, 3, False,
     'data/v9_fulldata/split_f5_all.csv', 'data/v9_fulldata/manifest_f5_v9.csv', 1.05, 4.2, 1.4e-3, 2),

    # 4. Fold 6 & Fold 7: 100% Pure Real Human-Annotated Giant Specialists (`real_only=True`, Zero Synthetic Shift!)
    ('V10_Full_F6_RealGiant_CleanGate', 6, 6, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6', 448, 2, True,
     'data/v9_fulldata/split_f6_real.csv', 'data/manifest_combined.csv', 1.05, 4.2, 1.4e-3, 2),
    ('V10_Full_F7_RealGiant_CleanGate', 7, 7, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6', 448, 2, True,
     'data/v9_fulldata/split_f7_real.csv', 'data/manifest_combined.csv', 1.05, 4.2, 1.4e-3, 2),
]


def main():
    print('=== Stage B: Launching 8 Full-Data (4,033 Real Images, 8-Fold 100% Coverage) V10 Specialists across GPUs 0..7 ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, fold, focus, bbs, sz, tiles, real_only, split_csv, man_csv, w_neg, p0_cln, lr_hd, cln_rep in JOBS:
        out_base = f'{OUT_DIR}/{name}'
        out_npz = f'{out_base}_probs.npz'
        if os.path.exists(out_npz):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        batch_sz = 8
        grad_acc = 1
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v10_clean_native_specialist.py',
            '--backbones', bbs,
            '--focus-classes', focus,
            '--seed', str(200 + fold),
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
            '--dropout', '0.12',
            '--split', split_csv,
            '--manifest', man_csv,
            '--out', out_base,
        ]
        if real_only:
            cmd.append('--real-only')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        log_f = open(f'{OUT_DIR}/{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_npz, log_f)
        print(f'[{time.time()-t0:5.0f}s] Launched {name:34s} (Fold {fold} | 3529/4033 real | sz={sz} t={tiles} real_only={real_only}) on GPU {gpu}', flush=True)

    while procs:
        time.sleep(15)
        for name in list(procs.keys()):
            p, gpu, out_npz, log_f = procs[name]
            ret = p.poll()
            if ret is not None:
                log_f.close()
                ok = os.path.exists(out_npz)
                print(f'[{time.time()-t0:5.0f}s] DONE {name:34s} (GPU {gpu}, exit={ret}, npz={ok})', flush=True)
                if not ok:
                    lines = open(f'{OUT_DIR}/{name}.log').readlines()[-15:]
                    print(''.join('    ' + l for l in lines), flush=True)
                del procs[name]

    print(f'=== ALL 8 FULL-DATA V10 SPECIALISTS COMPLETED in {time.time()-t0:.0f}s ===', flush=True)
    print('\n=== Running build_v10_fulldata_test150_submission.py to evaluate 4,033 OOF & generate pred_nriqa_dataset_test.json ===', flush=True)
    subprocess.run(['python3', '-u', 'code/finetune/build_v10_fulldata_test150_submission.py'], check=True)


if __name__ == '__main__':
    main()
