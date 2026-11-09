"""Launch 8 V16 Clean-Masked + 1:1 Native-Pixel Edge/Texture Contrast-Residual Microscope + Sharp P1-Boosted
Specialists across 8 x H200 GPUs (GPUs 0..7) on Zero-Leakage Splits `s16` and `s17` (`launch_v16_8gpus.py`).
"""
import csv, json, os, subprocess, time

os.chdir('/wanqing-develop/luowenjing/18')
OUT_DIR = 'results/v16_specialists'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs('data/v10_splits', exist_ok=True)

for sp in (16, 17):
    real_rows = list(csv.DictReader(open(f'data/resplits/split_s{sp}.csv')))
    real_names = {r['name'] for r in real_rows}
    man_rows = list(csv.DictReader(open(f'data/v9_ladders/manifest_s{sp}_v9.csv')))
    out_rows = list(real_rows)
    for r in man_rows:
        if r['name'] not in real_names:
            out_rows.append({'name': r['name'], 'split': 'train', 'source': r.get('source', 'synth')})
    out_csv = f'data/v10_splits/split_s{sp}_v10.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'split', 'source'])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Prepared {out_csv}: {len(real_rows)} real rows + {len(out_rows)-len(real_rows)} synth rows = {len(out_rows)} total", flush=True)

JOBS = [
    # (name, gpu, sp, focus_classes, backbones, sz, tiles, real_only, w_neg, p0_cln, lr_hd, cln_rep)
    ('V16_SpecA_OverGen_CleanMask_s16', 0, 16, '过度生成,虚化突兀',
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,convnext:full', 448, 2, False, 1.05, 4.2, 1.3e-3, 2),
    ('V16_SpecA_OverGen_CleanMask_s17', 1, 17, '过度生成,虚化突兀',
     'pecoreg:lastk4,siglip2g:lastk4,cradio:lastk4,convnext:full', 448, 2, False, 1.05, 4.2, 1.3e-3, 2),

    ('V16_SpecB_SharpTex_Microscope_s16', 2, 16, '过锐,纹理损失',
     'dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2, False, 1.05, 4.3, 1.3e-3, 2),
    ('V16_SpecB_SharpTex_Microscope_s17', 3, 17, '过锐,纹理损失',
     'dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full', 448, 2, False, 1.05, 4.3, 1.3e-3, 2),

    ('V16_SpecC_BlurBokeh_Microscope_s16', 4, 16, '模糊,纹理损失,虚化突兀,花屏',
     'pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full', 448, 2, False, 1.02, 4.2, 1.3e-3, 2),
    ('V16_SpecC_BlurBokeh_Microscope_s17', 5, 17, '模糊,纹理损失,虚化突兀,花屏',
     'pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full', 448, 2, False, 1.02, 4.2, 1.3e-3, 2),

    ('V16_SpecD_RealGiant_AllGrade_s16', 6, 16, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6', 448, 2, True, 1.02, 4.2, 1.3e-3, 2),
    ('V16_SpecD_RealGiant_AllGrade_s17', 7, 17, '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀',
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6', 448, 2, True, 1.02, 4.2, 1.3e-3, 2),
]


def main():
    subprocess.run(['pkill', '-f', 'train_v16_microscope_specialist.py'], check=False)
    time.sleep(2)
    print('=== Launching 8 V16 Clean-Masked + 1:1 Native Microscope Contrast Specialists across GPUs 0..7 ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, sp, focus, bbs, sz, tiles, real_only, w_neg, p0_cln, lr_hd, cln_rep in JOBS:
        out_prefix = f'{OUT_DIR}/{name}'
        out_npz = f'{out_prefix}_probs.npz'
        if os.path.exists(out_npz):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v16_microscope_specialist.py',
            '--backbones', bbs,
            '--focus-classes', focus,
            '--seed', str(300 + gpu),
            '--epochs', '4',
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
            '--split', f'data/v10_splits/split_s{sp}_v10.csv',
            '--manifest', f'data/v9_ladders/manifest_s{sp}_v9.csv',
            '--out', out_prefix,
        ]
        if real_only:
            cmd.append('--real-only')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        log_f = open(f'{OUT_DIR}/{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_npz, log_f)
        print(f'[{time.time()-t0:5.0f}s] Launched {name:36s} (s{sp} | sz={sz} t={tiles} real_only={real_only}) on GPU {gpu}', flush=True)

    while procs:
        time.sleep(20)
        for name in list(procs.keys()):
            p, gpu, out_npz, log_f = procs[name]
            ret = p.poll()
            if ret is not None:
                log_f.close()
                del procs[name]
                print(f'[{time.time()-t0:5.0f}s] DONE {name:36s} (GPU {gpu}, exit={ret}, npz={os.path.exists(out_npz)})', flush=True)
                if os.path.exists(f'{OUT_DIR}/{name}.log'):
                    lines = open(f'{OUT_DIR}/{name}.log').read().strip().splitlines()
                    for ln in lines[-4:]:
                        print(f'    {ln}', flush=True)
    print(f'=== ALL 8 V16 SPECIALISTS COMPLETED in {time.time()-t0:.0f}s ===', flush=True)


if __name__ == '__main__':
    main()
