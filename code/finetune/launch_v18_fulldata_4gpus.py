#!/usr/bin/env python3
"""
4-GPU Full-Data Training Launcher for V18.1 Specialists (3,073 Train + 300 Val + 200 Test)
==========================================================================================
Trains 4 V18.1 Specialists on 86% of all labeled real data (`splits/split_full_v18.csv`)
plus native-resolution paired quadruplets and True 6B + V16_Full Teacher KD, and automatically
exports name-aligned V16-calibrated `results/v18_fulldata/V18_Full_Spec*_probs.npz`.
"""
import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd

REPO_ROOT = '/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

SPECS = [
    ('V18_Full_Spec1_OvlText', 'V16_Spec1_OvlText', '0,1', '1.8,1.8,0.6,0.5,0.7,0.6'),
    ('V18_Full_Spec2_BlurArt', 'V16_Spec2_BlurArt', '2,3', '0.6,0.6,1.9,1.2,0.7,0.6'),
    ('V18_Full_Spec3_OvergenBody', 'V16_Spec3_OvergenBody', '4,1', '0.7,1.2,0.6,0.5,2.0,0.7'),
    ('V18_Full_Spec4_ColorComp', 'V16_Spec4_ColorComp', '3,5', '0.6,0.6,0.7,1.8,0.6,1.9'),
]


def build_full_split():
    s16 = pd.read_csv(f'{REPO_ROOT}/splits/split_s16.csv')
    rng = np.random.RandomState(20260318)
    va_df = s16[s16['split'] == 'val'].sample(frac=1.0, random_state=rng).reset_index(drop=True)
    te_df = s16[s16['split'] == 'test'].sample(frac=1.0, random_state=rng).reset_index(drop=True)
    tr_df = pd.concat([
        s16[s16['split'] == 'train'],
        va_df.iloc[300:],
        te_df.iloc[200:],
    ], ignore_index=True)
    tr_df['split'] = 'train'
    va_keep = va_df.iloc[:300].copy()
    va_keep['split'] = 'val'
    te_keep = te_df.iloc[:200].copy()
    te_keep['split'] = 'test'

    full_df = pd.concat([tr_df, va_keep, te_keep], ignore_index=True)
    out_csv = f'{REPO_ROOT}/splits/split_full_v18.csv'
    full_df.to_csv(out_csv, index=False)
    print(f"Built {out_csv}: train={len(tr_df)}, val={len(va_keep)}, test={len(te_keep)}")
    return out_csv


def main():
    split_csv = build_full_split()
    os.makedirs('/tmp/v18_logs', exist_ok=True)
    os.makedirs(f'{REPO_ROOT}/checkpoints/v18_fulldata', exist_ok=True)
    os.makedirs(f'{REPO_ROOT}/results/v18_fulldata', exist_ok=True)

    procs = []
    t0 = time.time()
    for gpu_id, (v18_name, v16_name, t_dims, d_weights) in enumerate(SPECS):
        base_ckpt = f"{REPO_ROOT}/checkpoints/v16_specialists/{v16_name}_s16_best.pt"
        v16_npz = f"{REPO_ROOT}/results/v16_specialists/{v16_name}_s16_probs.npz"
        out_ckpt = f"{REPO_ROOT}/checkpoints/v18_fulldata/{v18_name}_best.pt"
        out_npz = f"{REPO_ROOT}/results/v18_fulldata/{v18_name}_probs.npz"
        log_file = f"/tmp/v18_logs/{v18_name}.log"

        cmd = [
            sys.executable,
            f"{REPO_ROOT}/code/finetune/train_v18_paired_kd_specialist.py",
            "--name", v18_name,
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
            "--seed", str(100 + gpu_id),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        f_log = open(log_file, "w")
        p = subprocess.Popen(cmd, env=env, stdout=f_log, stderr=subprocess.STDOUT)
        procs.append((gpu_id, v18_name, p, f_log, log_file))
        print(f"[GPU {gpu_id}] Launched {v18_name} -> {log_file}")

    failed = 0
    for gid, rname, p, f_log, lfile in procs:
        rc = p.wait()
        f_log.close()
        print(f"\n=== [GPU {gid}] {rname} finished with code {rc} ({time.time()-t0:.1f}s) ===")
        if rc != 0:
            failed += 1
        with open(lfile, "r") as f:
            for ln in [x.rstrip() for x in f.readlines() if x.strip()][-6:]:
                print("  ", ln)
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
