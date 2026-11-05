"""Launch Wave-2 High-Resolution / Multi-Backbone V8 Single-Feature Dedicated Experts
across GPUs 0..7 in parallel.
Each expert outputs:
1. 2-way HFlip TTA predictions on `s16` / `s17` (500 val + 1,000 isolated test images)
2. 2-way HFlip TTA predictions on `data/test_150` (150 blind evaluation images)!
"""
import json
import os
import subprocess
import sys
import time

os.chdir('/wanqing-develop/luowenjing/18')
OUT_DIR = 'results/v8_experts'
os.makedirs(OUT_DIR, exist_ok=True)

JOBS = [
    # (name, gpu, split_num, focus_classes, backbones, size, tiles, split_csv, manifest_csv, w_neg, p0_cln, lr_head)
    ('V8_W2_ExpA_overgen_s16', 0, 16, '过度生成',      'eupe:full,siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 384, 2,
     'data/v8_ladders/split_s16_overgen.csv', 'data/v8_ladders/manifest_s16_v8.csv', 1.00, 4.2, 1.5e-3),
    ('V8_W2_ExpA_overgen_s17', 1, 17, '过度生成',      'eupe:full,siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 384, 2,
     'data/v8_ladders/split_s17_overgen.csv', 'data/v8_ladders/manifest_s17_v8.csv', 1.00, 4.2, 1.5e-3),

    ('V8_W2_ExpB_sharp_s16',   2, 16, '过锐',          'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',           448, 2,
     'data/v8_ladders/split_s16_sharp.csv',   'data/v8_ladders/manifest_s16_v8.csv', 1.00, 4.4, 1.4e-3),
    ('V8_W2_ExpB_sharp_s17',   3, 17, '过锐',          'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',           448, 2,
     'data/v8_ladders/split_s17_sharp.csv',   'data/v8_ladders/manifest_s17_v8.csv', 1.00, 4.4, 1.4e-3),

    ('V8_W2_ExpC_blurtex_s16', 4, 16, '模糊,纹理损失', 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 2,
     'data/v8_ladders/split_s16_blurtex.csv', 'data/v8_ladders/manifest_s16_v8.csv', 1.00, 4.2, 1.4e-3),
    ('V8_W2_ExpC_blurtex_s17', 5, 17, '模糊,纹理损失', 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 2,
     'data/v8_ladders/split_s17_blurtex.csv', 'data/v8_ladders/manifest_s17_v8.csv', 1.00, 4.2, 1.4e-3),

    ('V8_W2_ExpD_bokeh_s16',   6, 16, '虚化突兀',      'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',           384, 2,
     'data/v8_ladders/split_s16_bokeh.csv',   'data/v8_ladders/manifest_s16_v8.csv', 1.00, 4.5, 1.5e-3),
    ('V8_W2_ExpD_bokeh_s17',   7, 17, '虚化突兀',      'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',           384, 2,
     'data/v8_ladders/split_s17_bokeh.csv',   'data/v8_ladders/manifest_s17_v8.csv', 1.00, 4.5, 1.5e-3),
]


def main():
    print('=== Launching 8 Wave-2 High-Res V8 Single-Feature Dedicated Experts on GPUs 0..7 ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, sp, focus, bbs, sz, tiles, split_csv, man_csv, w_neg, p0_cln, lr_hd in JOBS:
        out_json = f'{OUT_DIR}/{name}.json'
        out_npz = f'{OUT_DIR}/{name}_probs.npz'
        if os.path.exists(out_json) and os.path.exists(out_npz):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_single_expert_v8.py',
            '--backbones', bbs,
            '--focus-classes', focus,
            '--seed', '2',
            '--epochs', '4',
            '--batch', '4',
            '--grad-accum', '2',
            '--workers', '6',
            '--size', str(sz),
            '--tiles', str(tiles),
            '--clean-repeat', '3',
            '--p0-clean-weight', str(p0_cln),
            '--channel-neg-weight', str(w_neg),
            '--lr-full', '8.5e-05',
            '--lr-last', '3.0e-05',
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
        print(f'[{time.time()-t0:5.0f}s] Launched {name:24s} (focus={focus:9s} | {bbs} @ sz={sz}) on GPU {gpu}', flush=True)

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
                    print(f'[{time.time()-t0:5.0f}s] DONE {name:24s} (GPU {gpu}) | '
                          f'val_focus_F1={d["val_focus_f1"]:.3f} | test_sca={ts["total"]:.2f} '
                          f'(S_pos={ts["S_pos"]:.3f}, S_p0={ts["S_p0"]:.3f}, S_cln={ts["S_clean"]:.3f})', flush=True)
                else:
                    print(f'[{time.time()-t0:5.0f}s] FAIL {name} (GPU {gpu}) rc={p.returncode}', flush=True)


if __name__ == '__main__':
    main()
