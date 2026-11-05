"""Wave-2 V9 High-Resolution Clean + Real-Oversampled Specialists on GPUs 2 & 3
(`V9_SpecE_CleanHiRes_Giant_s16` on GPU 2, `V9_SpecE_CleanHiRes_Giant_s17` on GPU 3).

Why `V9_SpecE_CleanHiRes_Giant`:
1. In `task-2175`, the highest-AP models on `模糊`, `纹理损失`, `过锐_P2`, and `过度生成_P0` were models
   trained on the REAL human-annotated training split (`data/resplits/split_s{sp}.csv`, `data/manifest_combined.csv`)
   at `448x448` (`v6_val/pcl_448t3_seed1` and `v6_val/d3_448t3_seed1`), because zero synthetic distribution shift
   occurs on real human-annotated labels!
2. However, NO `448x448` model on the real training split has ever been trained with:
   - Giant + High-Res 4-Backbone Fusion (`pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6`)
   - Spatial Anomaly Patch Token Pooling (`[CLS, Mean, Std, TopK32_Mean]`)
   - 2 Native 1:1 Zero-Resize Crops + 60-dim Physical Caliper Fusion
   - P0-Protected Competition Loss (`cw4=[1.0, 1.25, 1.35, 1.45]` + Soft `S_p0` + Soft `S_pos` + Pairwise Margin Loss)!
"""
import json
import os
import subprocess
import time

os.chdir('/wanqing-develop/luowenjing/18')
OUT_DIR = 'results/v9_specialists'
os.makedirs(OUT_DIR, exist_ok=True)

JOBS = [
    ('V9_SpecE_RealHiRes_Giant_s16', 2, 16, '模糊,纹理损失,过锐,过度生成,虚化突兀',
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6', 448, 2,
     'data/resplits/split_s16.csv', 'data/manifest_combined.csv', 1.00, 4.2, 1.4e-3),
    ('V9_SpecE_RealHiRes_Giant_s17', 3, 17, '模糊,纹理损失,过锐,过度生成,虚化突兀',
     'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6', 448, 2,
     'data/resplits/split_s17.csv', 'data/manifest_combined.csv', 1.00, 4.2, 1.4e-3),
]

def main():
    procs = {}
    t0 = time.time()
    for name, gpu, sp, focus, bbs, sz, tiles, split_csv, man_csv, w_neg, p0_cln, lr_hd in JOBS:
        out_json = f'{OUT_DIR}/{name}.json'
        out_npz = f'{OUT_DIR}/{name}_probs.npz'
        if os.path.exists(out_json) and os.path.exists(out_npz):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v9_specialist.py',
            '--backbones', bbs,
            '--focus-classes', focus,
            '--seed', '19',
            '--epochs', '4',
            '--batch', '4',
            '--grad-accum', '2',
            '--workers', '6',
            '--size', str(sz),
            '--tiles', str(tiles),
            '--clean-repeat', '2',
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
        print(f'[{time.time()-t0:5.0f}s] Launched {name} on GPU {gpu}', flush=True)

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
                    print(f'[{time.time()-t0:5.0f}s] DONE {name} (GPU {gpu}) | test_sca={ts["total"]:.2f} '
                          f'(S_pos={ts["S_pos"]:.3f}, S_p0={ts["S_p0"]:.3f}, S_cln={ts["S_clean"]:.3f})', flush=True)
                else:
                    print(f'[{time.time()-t0:5.0f}s] FAIL {name} (GPU {gpu}) rc={p.returncode}', flush=True)

if __name__ == '__main__':
    main()
