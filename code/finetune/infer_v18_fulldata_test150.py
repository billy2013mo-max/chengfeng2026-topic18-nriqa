#!/usr/bin/env python3
"""
V18.1 FullData Parallel Inference & Calibrated Anchor Alignment on split_full_v18 + test_150
============================================================================================
Fixes the V18 FullData inference bug where:
1. `va_p0` shape (300, 6) in `split_full_v18.csv` crashed when blending with `s16` V16_Spec (500, 6).
2. `infer_v18_fulldata_test150.py` previously skipped the V16 pure-real anchor calibration entirely,
   causing a severe probability distribution shift between s16/s17 OOF and FullData test_150.
3. Also supports 4-GPU parallel worker execution (`--worker-spec-idx`) so all 4 specialists finish in ~45s.
"""
import os
import sys
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = '/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'code/finetune'))

from train_v18_paired_kd_specialist import (
    DIMS,
    MultiNetV18,
    EvalDatasetV18,
    eval_split_tta,
    score_18dim_exact_probs,
)


SPECS = [
    (
        'V18_Full_Spec1_OvlText',
        f'{REPO_ROOT}/checkpoints/v16_specialists/V16_Spec1_OvlText_s16_best.pt',
        f'{REPO_ROOT}/checkpoints/v18_fulldata/V18_Full_Spec1_OvlText_best.pt',
        f'{REPO_ROOT}/results/v18_fulldata/V18_Full_Spec1_OvlText_probs.npz',
        f'{REPO_ROOT}/results/v16_specialists/V16_Spec1_OvlText_s16_probs.npz',
        f'{REPO_ROOT}/results/v16_specialists/V16_Spec1_OvlText_s17_probs.npz',
    ),
    (
        'V18_Full_Spec2_BlurArt',
        f'{REPO_ROOT}/checkpoints/v16_specialists/V16_Spec2_BlurArt_s16_best.pt',
        f'{REPO_ROOT}/checkpoints/v18_fulldata/V18_Full_Spec2_BlurArt_best.pt',
        f'{REPO_ROOT}/results/v18_fulldata/V18_Full_Spec2_BlurArt_probs.npz',
        f'{REPO_ROOT}/results/v16_specialists/V16_Spec2_BlurArt_s16_probs.npz',
        f'{REPO_ROOT}/results/v16_specialists/V16_Spec2_BlurArt_s17_probs.npz',
    ),
    (
        'V18_Full_Spec3_Overgen Body',
        f'{REPO_ROOT}/checkpoints/v16_specialists/V16_Spec3_OvergenBody_s16_best.pt',
        f'{REPO_ROOT}/checkpoints/v18_fulldata/V18_Full_Spec3_OvergenBody_best.pt',
        f'{REPO_ROOT}/results/v18_fulldata/V18_Full_Spec3_OvergenBody_probs.npz',
        f'{REPO_ROOT}/results/v16_specialists/V16_Spec3_OvergenBody_s16_probs.npz',
        f'{REPO_ROOT}/results/v16_specialists/V16_Spec3_OvergenBody_s17_probs.npz',
    ),
    (
        'V18_Full_Spec4_ColorComp',
        f'{REPO_ROOT}/checkpoints/v16_specialists/V16_Spec4_ColorComp_s16_best.pt',
        f'{REPO_ROOT}/checkpoints/v18_fulldata/V18_Full_Spec4_ColorComp_best.pt',
        f'{REPO_ROOT}/results/v18_fulldata/V18_Full_Spec4_ColorComp_probs.npz',
        f'{REPO_ROOT}/results/v16_specialists/V16_Spec4_ColorComp_s16_probs.npz',
        f'{REPO_ROOT}/results/v16_specialists/V16_Spec4_ColorComp_s17_probs.npz',
    ),
]


def load_v16_anchor_aligned(v16_s16_npz, v16_s17_npz, va_names, te_names, t150_names):
    """
    Aligns V16 pure-real specialist predictions by image name (for va/te) and
    averages s16+s17 V16 specialist predictions + V16_Full 8-fold ensemble for test_150.
    """
    name_to_v16 = {}
    t150_p0_list, t150_p1_list, t150_p2_list = [], [], []

    for p in [v16_s16_npz, v16_s17_npz]:
        if p and os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            for split_prefix, nm_key in [('va', 'va_names'), ('te', 'te_names')]:
                if nm_key in d:
                    nms = [str(x) for x in d[nm_key]]
                    p0, p1, p2 = d[f'{split_prefix}_p0'], d[f'{split_prefix}_p1'], d[f'{split_prefix}_p2']
                    for i, nm in enumerate(nms):
                        if nm not in name_to_v16:
                            name_to_v16[nm] = ([], [], [])
                        name_to_v16[nm][0].append(p0[i])
                        name_to_v16[nm][1].append(p1[i])
                        name_to_v16[nm][2].append(p2[i])
            if 't150_p0' in d:
                t150_p0_list.append(d['t150_p0'].astype(np.float64))
                t150_p1_list.append(d['t150_p1'].astype(np.float64))
                t150_p2_list.append(d['t150_p2'].astype(np.float64))

    # Also load V16_Full 8-fold OOF as fallback for any image not in s16/s17 va/te
    v16_full_t150_p0, v16_full_t150_p1, v16_full_t150_p2 = [], [], []
    for f_idx in range(8):
        fp = os.path.join(REPO_ROOT, f'results/v16_fulldata/V16_Full_F{f_idx}_AllGrade_Microscope_probs.npz')
        if os.path.exists(fp):
            df = np.load(fp, allow_pickle=True)
            for split_prefix, nm_key in [('va', 'va_names'), ('te', 'te_names')]:
                if nm_key in df:
                    nms = [str(x) for x in df[nm_key]]
                    p0, p1, p2 = df[f'{split_prefix}_p0'], df[f'{split_prefix}_p1'], df[f'{split_prefix}_p2']
                    for i, nm in enumerate(nms):
                        if nm not in name_to_v16:
                            name_to_v16[nm] = ([], [], [])
                        name_to_v16[nm][0].append(p0[i])
                        name_to_v16[nm][1].append(p1[i])
                        name_to_v16[nm][2].append(p2[i])
            if 't150_p0' in df:
                v16_full_t150_p0.append(df['t150_p0'].astype(np.float64))
                v16_full_t150_p1.append(df['t150_p1'].astype(np.float64))
                v16_full_t150_p2.append(df['t150_p2'].astype(np.float64))

    def build_split_arrays(names_list, fallback_p0, fallback_p1, fallback_p2):
        out_p0 = np.copy(fallback_p0)
        out_p1 = np.copy(fallback_p1)
        out_p2 = np.copy(fallback_p2)
        matched = 0
        for i, nm in enumerate(names_list):
            if nm in name_to_v16 and len(name_to_v16[nm][0]) > 0:
                out_p0[i] = np.mean(name_to_v16[nm][0], axis=0)
                out_p1[i] = np.mean(name_to_v16[nm][1], axis=0)
                out_p2[i] = np.mean(name_to_v16[nm][2], axis=0)
                matched += 1
        return out_p0, out_p1, out_p2, matched

    t150_anchor_p0, t150_anchor_p1, t150_anchor_p2 = None, None, None
    if len(t150_p0_list) > 0:
        spec_t0 = np.mean(t150_p0_list, axis=0)
        spec_t1 = np.mean(t150_p1_list, axis=0)
        spec_t2 = np.mean(t150_p2_list, axis=0)
        if len(v16_full_t150_p0) > 0:
            full_t0 = np.mean(v16_full_t150_p0, axis=0)
            full_t1 = np.mean(v16_full_t150_p1, axis=0)
            full_t2 = np.mean(v16_full_t150_p2, axis=0)
            t150_anchor_p0 = 0.65 * spec_t0 + 0.35 * full_t0
            t150_anchor_p1 = 0.65 * spec_t1 + 0.35 * full_t1
            t150_anchor_p2 = 0.65 * spec_t2 + 0.35 * full_t2
        else:
            t150_anchor_p0, t150_anchor_p1, t150_anchor_p2 = spec_t0, spec_t1, spec_t2

    return build_split_arrays, (t150_anchor_p0, t150_anchor_p1, t150_anchor_p2)


def run_single_specialist(spec_idx, anchor_weight=0.55):
    name, base_ckpt, v18_ckpt, out_npz, v16_s16_npz, v16_s17_npz = SPECS[spec_idx]
    name = name.replace(' ', '')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    split_csv = f'{REPO_ROOT}/splits/split_full_v18.csv'
    img_dir = f'{REPO_ROOT}/nriqa_dataset/train/images'
    test150_dir = f'{REPO_ROOT}/nriqa_dataset_test/images'

    df = pd.read_csv(split_csv)
    va_df = df[df['split'] == 'val'].sort_values('image').reset_index(drop=True)
    te_df = df[df['split'] == 'test'].sort_values('image').reset_index(drop=True)

    va_loader = DataLoader(EvalDatasetV18(va_df, img_dir), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    te_loader = DataLoader(EvalDatasetV18(te_df, img_dir), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    t150_files = sorted([f for f in os.listdir(test150_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
    t150_df = pd.DataFrame({'image': t150_files})
    for c in DIMS:
        t150_df[c] = '无问题'
    t150_loader = DataLoader(EvalDatasetV18(t150_df, test150_dir), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n[{name}] Loading MultiNetV18 from {v18_ckpt} on {device}...")
    model = MultiNetV18(base_ckpt, unfreeze_mode='lastk2').to(device)
    sd = torch.load(v18_ckpt, map_location='cpu')
    model.load_state_dict(sd, strict=True)
    model.eval()

    vp0, vp1, vp2, va_y, va_names = eval_split_tta(model, va_loader, device)
    tp0, tp1, tp2, te_y, te_names = eval_split_tta(model, te_loader, device)
    t150_p0, t150_p1, t150_p2, _, t150_names = eval_split_tta(model, t150_loader, device)

    raw_va_sc, _ = score_18dim_exact_probs(vp0, vp1, vp2, va_y)
    raw_te_sc, _ = score_18dim_exact_probs(tp0, tp1, tp2, te_y)

    # Apply name-aligned V16 pure-real anchor blend so FullData has identical calibration to s16/s17
    build_split_arrays, (t150_a0, t150_a1, t150_a2) = load_v16_anchor_aligned(
        v16_s16_npz, v16_s17_npz, va_names, te_names, t150_names
    )
    va_a0, va_a1, va_a2, m_va = build_split_arrays(va_names, vp0, vp1, vp2)
    te_a0, te_a1, te_a2, m_te = build_split_arrays(te_names, tp0, tp1, tp2)

    w16 = float(anchor_weight)
    vp0_cal = w16 * va_a0 + (1.0 - w16) * vp0
    vp1_cal = w16 * va_a1 + (1.0 - w16) * vp1
    vp2_cal = w16 * va_a2 + (1.0 - w16) * vp2

    tp0_cal = w16 * te_a0 + (1.0 - w16) * tp0
    tp1_cal = w16 * te_a1 + (1.0 - w16) * tp1
    tp2_cal = w16 * te_a2 + (1.0 - w16) * tp2

    if t150_a0 is not None:
        t150_p0_cal = w16 * t150_a0 + (1.0 - w16) * t150_p0
        t150_p1_cal = w16 * t150_a1 + (1.0 - w16) * t150_p1
        t150_p2_cal = w16 * t150_a2 + (1.0 - w16) * t150_p2
    else:
        t150_p0_cal, t150_p1_cal, t150_p2_cal = t150_p0, t150_p1, t150_p2

    cal_va_sc, _ = score_18dim_exact_probs(vp0_cal, vp1_cal, vp2_cal, va_y)
    cal_te_sc, _ = score_18dim_exact_probs(tp0_cal, tp1_cal, tp2_cal, te_y)
    print(
        f"[{name}] VA ({m_va}/{len(va_names)} aligned): raw={raw_va_sc:.3f} -> cal={cal_va_sc:.3f} | "
        f"TE ({m_te}/{len(te_names)} aligned): raw={raw_te_sc:.3f} -> cal={cal_te_sc:.3f}"
    )

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        va_p0=vp0_cal.astype(np.float32),
        va_p1=vp1_cal.astype(np.float32),
        va_p2=vp2_cal.astype(np.float32),
        va_y=va_y,
        va_names=np.array(va_names),
        te_p0=tp0_cal.astype(np.float32),
        te_p1=tp1_cal.astype(np.float32),
        te_p2=tp2_cal.astype(np.float32),
        te_y=te_y,
        te_names=np.array(te_names),
        t150_p0=t150_p0_cal.astype(np.float32),
        t150_p1=t150_p1_cal.astype(np.float32),
        t150_p2=t150_p2_cal.astype(np.float32),
        t150_names=np.array(t150_names),
        raw_t150_p0=t150_p0.astype(np.float32),
        raw_t150_p1=t150_p1.astype(np.float32),
        raw_t150_p2=t150_p2.astype(np.float32),
        best_te=float(cal_te_sc),
    )
    print(f"[{name}] Saved calibrated probabilities -> {out_npz}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-spec-idx', type=int, default=-1, help='If >= 0, run single specialist worker')
    parser.add_argument('--anchor-weight', type=float, default=0.55)
    args = parser.parse_args()

    if args.worker_spec_idx >= 0:
        run_single_specialist(args.worker_spec_idx, anchor_weight=args.anchor_weight)
        return

    t0 = time.time()
    print(f"=== Launching 4-GPU Parallel V18 FullData Calibrated Inference (w16={args.anchor_weight}) ===")
    procs = []
    for idx in range(len(SPECS)):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(idx)
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            '--worker-spec-idx', str(idx),
            '--anchor-weight', str(args.anchor_weight),
        ]
        procs.append((idx, subprocess.Popen(cmd, env=env)))

    failed = 0
    for idx, p in procs:
        rc = p.wait()
        if rc != 0:
            print(f"[ERROR] Worker {idx} exited with code {rc}")
            failed += 1

    print(f"=== Completed 4-GPU V18 FullData Inference in {time.time() - t0:.1f}s (failed={failed}) ===")
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
