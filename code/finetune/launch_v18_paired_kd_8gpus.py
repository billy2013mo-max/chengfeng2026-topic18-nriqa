#!/usr/bin/env python3
"""
8-GPU Parallel Launcher for V18.1 Paired-Quadruplet + True 6B/V16 Teacher KD Specialists
========================================================================================
Trains 4 Specialists x 2 Splits (s16, s17) across 8x H200 GPUs simultaneously with
`--epochs-s1 2 --epochs-s2 2`.
"""
import os
import sys
import time
import subprocess

REPO_ROOT = '/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

SPECS = [
    ('V18_Spec1_OvlText', 'V16_Spec1_OvlText', '0,1', '1.8,1.8,0.6,0.5,0.7,0.6'),
    ('V18_Spec2_BlurArt', 'V16_Spec2_BlurArt', '2,3', '0.6,0.6,1.9,1.2,0.7,0.6'),
    ('V18_Spec3_OvergenBody', 'V16_Spec3_OvergenBody', '4,1', '0.7,1.2,0.6,0.5,2.0,0.7'),
    ('V18_Spec4_ColorComp', 'V16_Spec4_ColorComp', '3,5', '0.6,0.6,0.7,1.8,0.6,1.9'),
]


def main():
    os.makedirs('/tmp/v18_logs', exist_ok=True)
    os.makedirs(f'{REPO_ROOT}/checkpoints/v18_paired_kd', exist_ok=True)
    os.makedirs(f'{REPO_ROOT}/results/v18_paired_kd', exist_ok=True)

    procs = []
    gpu_id = 0
    t0 = time.time()

    for sp in ['16', '17']:
        for v18_name, v16_name, t_dims, d_weights in SPECS:
            run_name = f"{v18_name}_s{sp}"
            base_ckpt = f"{REPO_ROOT}/checkpoints/v16_specialists/{v16_name}_s{sp}_best.pt"
            v16_npz = f"{REPO_ROOT}/results/v16_specialists/{v16_name}_s{sp}_probs.npz"
            split_csv = f"{REPO_ROOT}/splits/split_s{sp}.csv"
            out_ckpt = f"{REPO_ROOT}/checkpoints/v18_paired_kd/{run_name}_best.pt"
            out_npz = f"{REPO_ROOT}/results/v18_paired_kd/{run_name}_probs.npz"
            log_file = f"/tmp/v18_logs/{run_name}.log"

            cmd = [
                sys.executable,
                f"{REPO_ROOT}/code/finetune/train_v18_paired_kd_specialist.py",
                "--name", run_name,
                "--base-ckpt", base_ckpt,
                "--v16-npz", v16_npz,
                "--split-csv", split_csv,
                "--target-dims", t_dims,
                "--dim-weights", d_weights,
                "--epochs-s1", "2",
                "--epochs-s2", "2",
                "--batch-size", "24",
                "--lr", "1.0e-4",
                "--kd-weight", "0.30",
                "--rank-weight", "0.25",
                "--anchor-weight", "0.55",
                "--out-ckpt", out_ckpt,
                "--out-npz", out_npz,
                "--seed", str(42 + gpu_id),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            f_log = open(log_file, "w")
            p = subprocess.Popen(cmd, env=env, stdout=f_log, stderr=subprocess.STDOUT)
            procs.append((gpu_id, run_name, p, f_log, log_file))
            print(f"[GPU {gpu_id}] Launched {run_name} -> {log_file}")
            gpu_id += 1

    print("\nWaiting for all 8 V18.1 specialist jobs to complete...")
    failed = 0
    for gid, rname, p, f_log, lfile in procs:
        rc = p.wait()
        f_log.close()
        print(f"\n=== [GPU {gid}] {rname} finished with code {rc} ({time.time()-t0:.1f}s) ===")
        if rc != 0:
            failed += 1
        with open(lfile, "r") as f:
            lines = [ln.rstrip() for ln in f.readlines() if ln.strip()]
            for ln in lines[-6:]:
                print("  ", ln)
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
