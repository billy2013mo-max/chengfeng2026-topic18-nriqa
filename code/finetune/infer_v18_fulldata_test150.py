#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/infer_v18_fulldata_test150.py
Runs fast 4-GPU parallel 2x TTA inference on val_300 and test_150 using the 4 trained V18 FullData SWA checkpoints,
and applies name-aligned V16 Pure-Real Specialist Anchor Calibration (fixing the V18 FullData uncalibrated shift).
"""
import os, sys, json, time, argparse, subprocess, torch
import numpy as np
from torch.utils.data import DataLoader

os.chdir('/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, 'code/finetune')
from data import CLASSES, FULL
from train_v9_specialist import build_v9_backbone, set_seed
from train_v18_paired_kd_specialist import (
    MultiNetV18, V18FastMicroscopeDataset, load_roi_and_synth_meta,
    predict_v18, load_split, read_manifest, blend_with_v16_anchor_safe
)
from extract_native_phys_feats import extract_for_names
from unified_dual_metric_gate import evaluate_both_metrics
import comp_metrics as cm
from cv_decoder_calib import decode_pc, HAND

MODELS = [
    ('V18_Full_SpecA_OverGen_TexLoss', 0,
     'pecoreg:lastk1,siglip2g:lastk1,cradio:lastk1,convnext:lastk2',
     'results/v18_fulldata/V18_Full_SpecA_OverGen_TexLoss_swa.pt',
     'results/v16_specialists/V16_SpecA_OverGen_CleanMask_s16_probs.npz'),

    ('V18_Full_SpecB_Sharp_TexLoss', 1,
     'dinov3l:lastk1,cradio:lastk1,eva02:lastk1,convnext:lastk2',
     'results/v18_fulldata/V18_Full_SpecB_Sharp_TexLoss_swa.pt',
     'results/v16_specialists/V16_SpecB_SharpTex_Microscope_s16_probs.npz'),

    ('V18_Full_SpecC_Blur_Bokeh_Screen', 2,
     'pecorel:lastk1,dinov3l:lastk1,siglip2:lastk1,eupe:lastk2',
     'results/v18_fulldata/V18_Full_SpecC_Blur_Bokeh_Screen_swa.pt',
     'results/v16_specialists/V16_SpecC_BlurBokeh_Microscope_s16_probs.npz'),

    ('V18_Full_SpecD_All6_Giant', 3,
     'pecoreg:lastk1,dinov3l:lastk1,cradio:lastk1,eva02:lastk1',
     'results/v18_fulldata/V18_Full_SpecD_All6_Giant_swa.pt',
     'results/v16_specialists/V16_SpecD_RealGiant_AllGrade_s16_probs.npz'),
]


def ensure_full_split_exists():
    split_csv = 'data/v18_full/split_full_v18.csv'
    man_csv = 'data/v18_full/manifest_full_v18.csv'
    if not (os.path.exists(split_csv) and os.path.exists(man_csv)):
        from launch_v18_fulldata_4gpus import prepare_full_manifest_and_split
        prepare_full_manifest_and_split()
    return split_csv, man_csv


def run_inference_for_model(name, gpu_id, backbones_str, ckpt_path, v16_npz):
    print(f"\n=== Running FullData Calibrated Inference for {name} on GPU {gpu_id} ===", flush=True)
    dev = 'cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else f'cuda:{gpu_id}'
    roi_map, meta_map, synth_meta = load_roi_and_synth_meta()

    split_csv, man_csv = ensure_full_split_exists()
    rows = load_split(split_csv)
    man = read_manifest(man_csv)
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})

    va = [r for r in rows if r['split'] == 'val']
    tr_real = [r for r in rows if r['split'] == 'train' and not r['name'].startswith('v18_') and ('_NatNeg' not in r['name'])]
    all_names = [r['name'] for r in tr_real + va]

    phys_map = extract_for_names(all_names, 'results/v8_experts/native_phys_feats.npz')
    phys_tr_mat = np.stack([phys_map[r['name']] for r in tr_real], axis=0)
    phys_mu = np.mean(phys_tr_mat, axis=0)
    phys_sd = np.std(phys_tr_mat, axis=0) + 1e-5

    tpl = json.load(open('results/v2_full/template_150.json'))
    names_150 = [e['image'] for e in tpl]
    phys_150 = extract_for_names(names_150, 'results/v8_experts/native_phys_feats.npz')
    rows_150 = [{'name': n, **{c: 0 for c in CLASSES}} for n in names_150]

    dva = V18FastMicroscopeDataset(va, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, 448, False, hflip=False, tiles=2)
    dva_f = V18FastMicroscopeDataset(va, phys_map, phys_mu, phys_sd, roi_map, meta_map, synth_meta, 448, False, hflip=True, tiles=2)
    lva = DataLoader(dva, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)
    lva_f = DataLoader(dva_f, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)

    d150 = V18FastMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, roi_map, meta_map, synth_meta, 448, False, hflip=False, tiles=2)
    d150_f = V18FastMicroscopeDataset(rows_150, phys_150, phys_mu, phys_sd, roi_map, meta_map, synth_meta, 448, False, hflip=True, tiles=2)
    l150 = DataLoader(d150, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)
    l150_f = DataLoader(d150_f, batch_size=24, shuffle=False, num_workers=4, pin_memory=True)

    specs = []
    for item in backbones_str.split(','):
        nm, mode = item.split(':')
        m, fwd, mu, sd, bl, dim = build_v9_backbone(nm)
        specs.append((m, fwd, mu, sd, bl, dim))

    net = MultiNetV18(specs, phys_dim=60, dropout=0.12, n_quad=4).to(dev)
    print(f"  Loading SWA weights from {ckpt_path}...", flush=True)
    sd = torch.load(ckpt_path, map_location=dev)
    net.load_state_dict(sd, strict=False)
    net.eval()

    print("  Predicting on Val (N=300, 2x TTA)...", flush=True)
    vp0_a, vp1_a, vp2_a, vg_a, gva = predict_v18(net, lva, dev, 0.58)
    vp0_b, vp1_b, vp2_b, vg_b, _   = predict_v18(net, lva_f, dev, 0.58)
    vp0, vp1, vp2, vg = 0.5*(vp0_a+vp0_b), 0.5*(vp1_a+vp1_b), 0.5*(vp2_a+vp2_b), 0.5*(vg_a+vg_b)

    print("  Predicting on test_150 (N=150, 2x TTA)...", flush=True)
    p150_0a, p150_1a, p150_2a, p150_ga, _ = predict_v18(net, l150, dev, 0.58)
    p150_0b, p150_1b, p150_2b, p150_gb, _ = predict_v18(net, l150_f, dev, 0.58)
    p150_0, p150_1, p150_2, p150_g = 0.5*(p150_0a+p150_0b), 0.5*(p150_1a+p150_1b), 0.5*(p150_2a+p150_2b), 0.5*(p150_ga+p150_gb)

    va_names = [r['name'] for r in va]
    raw_pred_va = decode_pc(vp0, vp1, vp2, HAND)
    raw_st_val = cm.score(gva, raw_pred_va, 0.02)

    # Apply name-aligned V16 pure-real specialist anchor calibration (identical to s16/s17 specialists!)
    if v16_npz and os.path.exists(v16_npz):
        vp0, vp1, vp2, _, _, _, p150_0, p150_1, p150_2 = blend_with_v16_anchor_safe(
            v16_npz, vp0, vp1, vp2, va_names, None, None, None, [], p150_0, p150_1, p150_2
        )

    out_npz = f'results/v18_fulldata/{name}_probs.npz'
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(
        out_npz,
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vg, gva=gva, va_names=va_names,
        t150_p0=p150_0, t150_p1=p150_1, t150_p2=p150_2, t150_gate=p150_g, t150_names=names_150,
    )

    pred_va = decode_pc(vp0, vp1, vp2, HAND)
    st_val = cm.score(gva, pred_va, 0.02)
    clean_val = (gva == 0).all(1)
    s_clean_val = float((clean_val & (pred_va == 0).all(1)).sum() / max(1, clean_val.sum()))
    doc2_val = 25 * st_val['S_pos'] + 20 * st_val['S_p0'] + 20 * s_clean_val
    print(
        f"  [{name}] Val (N=300): Raw_Doc1={raw_st_val['total']:.2f} -> Calibrated_Doc1={st_val['total']:.2f} | "
        f"Doc2={doc2_val:.2f} (S_pos={st_val['S_pos']:.3f}, S_p0={st_val['S_p0']:.3f}, Clean={s_clean_val:.3f})"
    )
    print(f"  Saved calibrated predictions to {out_npz}!", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker-idx', type=int, default=-1)
    ap.add_argument('--anchor-weight', type=float, default=0.55)
    args = ap.parse_args()

    ensure_full_split_exists()

    if args.worker_idx >= 0:
        name, gpu, backbones_str, ckpt, v16_npz = MODELS[args.worker_idx]
        run_inference_for_model(name, gpu, backbones_str, ckpt, v16_npz)
        return

    t0 = time.time()
    print("=== Launching 4-GPU Parallel V18 FullData Calibrated Inference ===", flush=True)
    procs = []
    for idx, (name, gpu, _, _, _) in enumerate(MODELS):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        p = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__), '--worker-idx', str(idx)], env=env)
        procs.append((idx, name, p))

    failed = 0
    for idx, name, p in procs:
        rc = p.wait()
        if rc != 0:
            print(f"[ERROR] Worker {idx} ({name}) exited with code {rc}", flush=True)
            failed += 1

    print(f"\nAll 4 FullData models finished in {time.time()-t0:.1f}s (failed={failed})!", flush=True)
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
