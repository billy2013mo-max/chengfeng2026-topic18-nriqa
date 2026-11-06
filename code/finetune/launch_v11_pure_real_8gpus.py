"""Stage 1: Launch 8 Pure-Real H200 Training Jobs (`launch_v11_pure_real_8gpus.py`):
GPUs 0~4: 5 Zero-Leakage Independent Blind Evaluation Splits (s16 ~ s20)
GPUs 5~7: 3 Full-Data (3,573 Real Images 100% Coverage) Flagship Ensembles
"""
import os, subprocess, sys, time

os.chdir('/wanqing-develop/luowenjing/18')
OUT_DIR = 'results/v11_pure_real'
os.makedirs(OUT_DIR, exist_ok=True)

# Clean up any leftover train processes
subprocess.run(['pkill', '-f', 'train_v3_pure_real.py'], check=False)
time.sleep(2)

JOBS = [
    # (name, gpu, split, backbones, size, tiles, batch, accum, lr_last, lr_head, cln_rep, gen_rep)
    # --- GPUs 0~4: 5 Blind Test Splits (s16 ~ s20) for strict zero-leakage comparison vs backbone_fixed ---
    ('iv6b_s16', 0, 'data/resplits/split_s16.csv', 'internvit:lastk4,cradio:lastk4,siglip2:lastk4',
     448, 2, 4, 2, 1.5e-4, 1.5e-3, 4, 1),
    ('iv6b_s17', 1, 'data/resplits/split_s17.csv', 'internvit:lastk4,cradio:lastk4,siglip2:lastk4',
     448, 2, 4, 2, 1.5e-4, 1.5e-3, 4, 1),
    ('iv6b_s18', 2, 'data/resplits/split_s18.csv', 'internvit:lastk4,cradio:lastk4,siglip2:lastk4',
     448, 2, 4, 2, 1.5e-4, 1.5e-3, 4, 1),
    ('iv6b_s19', 3, 'data/resplits/split_s19.csv', 'internvit:lastk4,cradio:lastk4,siglip2:lastk4',
     448, 2, 4, 2, 1.5e-4, 1.5e-3, 4, 1),
    ('iv6b_s20', 4, 'data/resplits/split_s20.csv', 'internvit:lastk4,cradio:lastk4,siglip2:lastk4',
     448, 2, 4, 2, 1.5e-4, 1.5e-3, 4, 1),

    # --- GPUs 5~7: 3 Full-Data (100% 3,573 Real Images) Flagship Ensembles for test_150 submission ---
    ('iv6b_full_cradio_sig2_448_t2', 5, 'data/split_all_full.csv', 'internvit:lastk4,cradio:lastk4,siglip2:lastk4',
     448, 2, 4, 2, 1.5e-4, 1.5e-3, 4, 2),
    ('iv6b_full_d3_pcl_448_t2', 6, 'data/split_all_full.csv', 'internvit:lastk4,dinov3l:lastk4,pecorel:lastk4',
     448, 2, 4, 2, 1.5e-4, 1.5e-3, 4, 2),
    ('iv6b_full_eupe_conv_448_t3', 7, 'data/split_all_full.csv', 'internvit:lastk4,eupe:full,convnext:full',
     448, 3, 4, 2, 1.5e-4, 1.5e-3, 4, 2),
]


def main():
    print('=== Launching 8 Pure-Real H200 Training Jobs (Stage 1) ===', flush=True)
    procs = {}
    t0 = time.time()
    for name, gpu, split, bbs, sz, tiles, bs, accum, lr_last, lr_head, cln_rep, gen_rep in JOBS:
        out_json = f'{OUT_DIR}/{name}.json'
        out_npz = f'{OUT_DIR}/{name}_probs.npz'
        out_log = f'{OUT_DIR}/{name}.log'
        cmd = [
            'python3', '-u', 'code/finetune/train_v3_pure_real.py',
            '--backbones', bbs,
            '--split', split,
            '--manifest', 'data/manifest_combined.csv',
            '--size', str(sz),
            '--tiles', str(tiles),
            '--epochs', '4',
            '--batch', str(bs),
            '--grad-accum', str(accum),
            '--lr-last', str(lr_last),
            '--lr-head', str(lr_head),
            '--clean-repeat', str(cln_rep),
            '--gen-repeat', str(gen_rep),
            '--out', out_json,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['INTERNVIT_SIZE'] = str(sz)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        log_f = open(out_log, 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_json, out_npz, log_f)
        print(f'[{time.time()-t0:4.0f}s] Launched {name:28s} (GPU {gpu} | size={sz} t={tiles} split={os.path.basename(split)})', flush=True)

    print('\nAll 8 GPUs active. Monitoring training progress...', flush=True)
    while procs:
        time.sleep(30)
        for name in list(procs.keys()):
            p, gpu, out_json, out_npz, log_f = procs[name]
            if p.poll() is not None:
                log_f.close()
                del procs[name]
                npz_ok = os.path.exists(out_npz)
                print(f'[{time.time()-t0:4.0f}s] DONE {name:28s} (GPU {gpu}, exit={p.returncode}, npz={npz_ok})', flush=True)


if __name__ == '__main__':
    main()
