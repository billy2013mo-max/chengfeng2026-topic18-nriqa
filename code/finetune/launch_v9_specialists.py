"""Launch 8 V9 High-Resolution Spatial-Anomaly-Pooled Multi-Backbone Specialists
across GPUs 0..7 in parallel.

Each V9 Specialist features:
1. Spatial Anomaly Patch Token Pooling (`[CLS, Patch_Mean, Patch_Std, Patch_TopK32_Mean]`)
2. Diverse Giant & High-Res Backbones (`pecoreg` 1.88B, `siglip2g` 1.9B, `dinov3l`, `cradio`, `eva02`, `convnext`, `eupe`, `pecorel`)
3. Zero-Ripple Authentic V9 Synthetic Ladders (`data/v9_ladders/*`)
4. P0-Protected Soft S_p0 + Soft S_pos + Pairwise Ordinal Severity Margin Loss
5. Full 2-way HFlip TTA output on `val` (500), `test` (1,000), and `test_150` (150)!
"""
import json
import os
import subprocess
import time

os.chdir('/wanqing-develop/luowenjing/18')
OUT_DIR = 'results/v9_specialists'
os.makedirs(OUT_DIR, exist_ok=True)

JOBS = [
    # (name, gpu, sp, focus_classes, backbones, size, tiles, split_csv, man_csv, w_neg, p0_cln, lr_hd)
    # 1. Giant OverGen + Texture Specialist (PE-Core-G14 + SigLIP2-Giant + C-RADIOv4-H + ConvNeXtV2) on Authentic Zero-Ripple OverGen
    ('V9_SpecA_OverGen_Giant_s16', 0, 16, '过度生成,纹理损失',
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,convnext:full', 448, 2,
     'data/v9_ladders/split_s16_overgen.csv', 'data/v9_ladders/manifest_s16_v9.csv', 1.00, 4.2, 1.4e-3),
    ('V9_SpecA_OverGen_Giant_s17', 1, 17, '过度生成,纹理损失',
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,convnext:full', 448, 2,
     'data/v9_ladders/split_s17_overgen.csv', 'data/v9_ladders/manifest_s17_v9.csv', 1.00, 4.2, 1.4e-3),

    # 2. High-Res Sharpness + Texture + Blur Specialist (DINOv3-L + C-RADIOv4-H + EVA02-448 + ConvNeXtV2) @ 448x448
    ('V9_SpecB_SharpTex_HiRes_s16', 2, 16, '过锐,纹理损失,模糊',
     'dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2,
     'data/v9_ladders/split_s16_sharptex.csv', 'data/v9_ladders/manifest_s16_v9.csv', 1.00, 4.3, 1.4e-3),
    ('V9_SpecB_SharpTex_HiRes_s17', 3, 17, '过锐,纹理损失,模糊',
     'dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2,
     'data/v9_ladders/split_s17_sharptex.csv', 'data/v9_ladders/manifest_s17_v9.csv', 1.00, 4.3, 1.4e-3),

    # 3. 3x3 Tile (12 Views!) High-Res All-Defect + P0 Recall Anchor Specialist (PE-Core-L14 + DINOv3-L + SigLIP2 + EUPE)
    ('V9_SpecC_AllP0_Tile3_s16', 4, 16, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full', 448, 3,
     'data/v9_ladders/split_s16_all.csv', 'data/v9_ladders/manifest_s16_v9.csv', 0.95, 4.2, 1.4e-3),
    ('V9_SpecC_AllP0_Tile3_s17', 5, 17, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full', 448, 3,
     'data/v9_ladders/split_s17_all.csv', 'data/v9_ladders/manifest_s17_v9.csv', 0.95, 4.2, 1.4e-3),

    # 4. Bokeh + OverGen Boundary & Local Artifact Specialist (PE-Core-G14 + DINOv3-L + C-RADIOv4-H + EVA02-448)
    ('V9_SpecD_BokehOG_HiRes_s16', 6, 16, '虚化突兀,过度生成,过锐',
     'pecoreg:lastk4,dinov3l:lastk4,cradio:lastk4,eva02:lastk6', 448, 2,
     'data/v9_ladders/split_s16_overgen_bk.csv', 'data/v9_ladders/manifest_s16_v9.csv', 1.00, 4.4, 1.4e-3),
    ('V9_SpecD_BokehOG_HiRes_s17', 7, 17, '虚化突兀,过度生成,过锐',
     'pecoreg:lastk4,dinov3l:lastk4,cradio:lastk4,eva02:lastk6', 448, 2,
     'data/v9_ladders/split_s17_overgen_bk.csv', 'data/v9_ladders/manifest_s17_v9.csv', 1.00, 4.4, 1.4e-3),
]


def main():
    print('=== Launching 8 V9 High-Resolution Spatial-Anomaly-Pooled Specialists on GPUs 0..7 ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, sp, focus, bbs, sz, tiles, split_csv, man_csv, w_neg, p0_cln, lr_hd in JOBS:
        out_json = f'{OUT_DIR}/{name}.json'
        out_npz = f'{OUT_DIR}/{name}_probs.npz'
        if os.path.exists(out_json) and os.path.exists(out_npz):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        batch_sz = 3 if tiles == 3 else 4
        grad_acc = 3 if tiles == 3 else 2
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v9_specialist.py',
            '--backbones', bbs,
            '--focus-classes', focus,
            '--seed', '9',
            '--epochs', '4',
            '--batch', str(batch_sz),
            '--grad-accum', str(grad_acc),
            '--workers', '6',
            '--size', str(sz),
            '--tiles', str(tiles),
            '--clean-repeat', '3',
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
        print(f'[{time.time()-t0:5.0f}s] Launched {name:28s} (focus={focus[:18]:18s} | sz={sz} t={tiles}) on GPU {gpu}', flush=True)

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
                          f'val_focus_F1={d["val_focus_f1"]:.3f} | test_sca={ts["total"]:.2f} '
                          f'(S_pos={ts["S_pos"]:.3f}, S_p0={ts["S_p0"]:.3f}, S_cln={ts["S_clean"]:.3f})', flush=True)
                else:
                    print(f'[{time.time()-t0:5.0f}s] FAIL {name} (GPU {gpu}) rc={p.returncode}', flush=True)


if __name__ == '__main__':
    main()
