"""V6 Multi-Defect Synthesis Validation (s16 & s17) + 6 High-Res Specialist Models (GPUs 0..7).

1. Runs `build_multi_synth.py --target all` once FLUX3 generation completes.
2. Launches in parallel across GPUs 0..7:
   - GPU 0: G_v6_s16 (ablation validation on unseen s16 test split)
   - GPU 1: G_v6_s17 (ablation validation on unseen s17 test split)
   - GPU 2: v6_d3_448_t3_s1   (siglip2 + cradio + dinov3l @ 448x448, 3x3 tiles, seed 1)
   - GPU 3: v6_d3_448_t3_s2   (siglip2 + cradio + dinov3l @ 448x448, 3x3 tiles, seed 2)
   - GPU 4: v6_quad_448_t2_s1 (eupe + siglip2 + cradio + dinov3l @ 448x448, 2x2 tiles, seed 1)
   - GPU 5: v6_pcl_448_t3_s1  (siglip2 + cradio + pecorel @ 448x448, 3x3 tiles, seed 1)
   - GPU 6: v6_sig2g_384_t3_s1(siglip2g + cradio + dinov3l @ 384x384, 3x3 tiles, seed 1)
   - GPU 7: v6_pcg_384_t2_s1  (siglip2 + cradio + pecoreg @ 384x384, 2x2 tiles, seed 1)
3. Evaluates s16 & s17 held-out test sets (standalone G_v6 + 70/30 Clean+Synth Specialist Ensemble).
4. Runs 2x Flip TTA on all 18 full-data V6 checkpoints (6 models x 3 ckpts) and produces the
   specialist-fused pred_nriqa_dataset_test.json.
"""
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from run_v3_super_ensemble import infer_v3_ckpt_tta, decode_v3_shielded

OUT_ABL = 'results/v3_ablate'
OUT_V6 = 'results/v6_synth'
os.makedirs(OUT_ABL, exist_ok=True)
os.makedirs(OUT_V6, exist_ok=True)

ABLATE_JOBS = [
    ('G_v6_s16', 0, 'data/resplits/split_s16_synth_v6.csv', 'data/manifest_s16_synth_v6.csv'),
    ('G_v6_s17', 1, 'data/resplits/split_s17_synth_v6.csv', 'data/manifest_s17_synth_v6.csv'),
]

FULL_JOBS = [
    # (name, gpu, seed, backbones, size, tiles, batch, accum)
    ('v6_d3_448_t3_s1',    2, 1, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 3, 4, 2),
    ('v6_d3_448_t3_s2',    3, 2, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 3, 4, 2),
    ('v6_quad_448_t2_s1',  4, 1, 'eupe:full,siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 448, 2, 4, 2),
    ('v6_pcl_448_t3_s1',   5, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',           448, 3, 4, 2),
    ('v6_sig2g_384_t3_s1', 6, 1, 'siglip2g:lastk4,cradio:lastk4,dinov3l:lastk4',          384, 3, 4, 2),
    ('v6_pcg_384_t2_s1',   7, 1, 'siglip2:lastk4,cradio:lastk4,pecoreg:lastk4',           384, 2, 4, 2),
]


def evaluate_ablation():
    print('\n=== Step 2: Evaluating Held-Out s16 & s17 Test Splits (Fixed HAND Decoder) ===', flush=True)
    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    hdr = f"{'variant':14s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)

    for n in ['A_base', 'E_synth', 'F_v5', 'G_v6']:
        rows = []
        for sp in (16, 17):
            p = f'{OUT_ABL}/{n}_s{sp}_probs.npz'
            if not os.path.exists(p):
                continue
            d = np.load(p)
            pred = decode_pc(d['te_p0'], d['te_p1'], d['te_p2'], HAND)
            s = cm.score(d['gte'], pred, TAU)
            f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
            rows.append((s, f1))
            print(f"{n+'_s'+str(sp):14s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want), flush=True)
        if rows:
            m = lambda k: np.mean([r[0][k] for r in rows])
            mf = [np.mean([r[1][w] for r in rows]) for w in want]
            print(f"{n+' MEAN':14s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf), flush=True)
            print(flush=True)

    # Evaluate Specialist Ensembles: (1 - w) * Clean(F_v5 + A_base) + w * G_v6
    for w_syn in [0.25, 0.35, 0.50]:
        tag = f'Ens_{int((1-w_syn)*100)}c_{int(w_syn*100)}s'
        rows = []
        for sp in (16, 17):
            da = np.load(f'{OUT_ABL}/A_base_s{sp}_probs.npz')
            df = np.load(f'{OUT_ABL}/F_v5_s{sp}_probs.npz')
            dg = np.load(f'{OUT_ABL}/G_v6_s{sp}_probs.npz')
            p0 = (1.0 - w_syn) * (0.45 * da['te_p0'] + 0.55 * df['te_p0']) + w_syn * dg['te_p0']
            p1 = (1.0 - w_syn) * (0.45 * da['te_p1'] + 0.55 * df['te_p1']) + w_syn * dg['te_p1']
            p2 = (1.0 - w_syn) * (0.45 * da['te_p2'] + 0.55 * df['te_p2']) + w_syn * dg['te_p2']
            pred = decode_pc(p0, p1, p2, HAND)
            s = cm.score(dg['gte'], pred, TAU)
            f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
            rows.append((s, f1))
            print(f"{tag+'_s'+str(sp):14s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want), flush=True)
        if rows:
            m = lambda k: np.mean([r[0][k] for r in rows])
            mf = [np.mean([r[1][w] for r in rows]) for w in want]
            print(f"{tag+' MEAN':14s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf), flush=True)
            print(flush=True)


def main():
    print('=== Step 0: Building Zero-Leakage V6 Multi-Defect Synthetic Datasets ===', flush=True)
    subprocess.run(['python3', 'code/finetune/build_multi_synth.py', '--target', 'all'], check=True)

    print('\n=== Step 1: Launching 2 Ablation Jobs (GPUs 0..1) + 6 Full Specialist Models (GPUs 2..7) ===', flush=True)
    abl_procs = []
    full_procs = []
    t0 = time.time()

    for tag, gpu, split_csv, man_csv in ABLATE_JOBS:
        out_json = f'{OUT_ABL}/{tag}.json'
        probs_npz = f'{OUT_ABL}/{tag}_probs.npz'
        if os.path.exists(out_json) and os.path.exists(probs_npz):
            print(f'[skip] {tag} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v5.py',
            '--backbones', 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',
            '--seed', '1',
            '--epochs', '4',
            '--batch', '4',
            '--grad-accum', '2',
            '--workers', '6',
            '--size', '384',
            '--tiles', '2',
            '--clean-repeat', '4',
            '--p0-clean-weight', '4.2',
            '--channel-neg-weight', '1.35',
            '--alpha-ex', '0.55',
            '--emd-weight', '0.50',
            '--split', split_csv,
            '--manifest', man_csv,
            '--out', out_json,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        lf = open(f'{OUT_ABL}/{tag}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        abl_procs.append((tag, p, lf))
        print(f'[launched] {tag} on GPU {gpu}', flush=True)

    for name, gpu, sd, bbs, sz, tiles, bs, accum in FULL_JOBS:
        out_json = f'{OUT_V6}/train_{name}.json'
        ckpt_pt = f'{OUT_V6}/model_{name}.pt'
        if os.path.exists(out_json) and os.path.exists(ckpt_pt):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v5.py',
            '--backbones', bbs,
            '--seed', str(sd),
            '--epochs', '4',
            '--batch', str(bs),
            '--grad-accum', str(accum),
            '--workers', '6',
            '--size', str(sz),
            '--tiles', str(tiles),
            '--clean-repeat', '4',
            '--p0-clean-weight', '4.2',
            '--channel-neg-weight', '1.35',
            '--alpha-ex', '0.55',
            '--emd-weight', '0.50',
            '--split', 'data/split_all_full_synth_v6.csv',
            '--manifest', 'data/manifest_combined_synth_v6.csv',
            '--out', out_json,
            '--save-state', ckpt_pt,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        lf = open(f'{OUT_V6}/train_{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        full_procs.append((name, p, lf))
        print(f'[launched] {name} on GPU {gpu}', flush=True)

    for tag, p, lf in abl_procs:
        rc = p.wait()
        lf.close()
        print(f'[{time.time()-t0:.0f}s] Finished ablation {tag} rc={rc}', flush=True)

    evaluate_ablation()

    for name, p, lf in full_procs:
        rc = p.wait()
        lf.close()
        print(f'[{time.time()-t0:.0f}s] Finished specialist {name} rc={rc}', flush=True)

    print('\n=== Step 3: Running 2x Flip TTA on All 18 V6 Checkpoints (Ep3 + Ep4 + SWA) ===', flush=True)
    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]
    v6_cache_path = f'{OUT_V6}/test_150_v6_models_probs.npz'
    v6_probs = {}
    if os.path.exists(v6_cache_path):
        v6_probs.update(np.load(v6_cache_path, allow_pickle=True)['data'].item())

    for name, gpu, sd, bbs, sz, tiles, bs, accum in FULL_JOBS:
        dev = f'cuda:{gpu}'
        for suf, w in [('_ep3', 1.8), ('_swa', 2.0), ('', 1.8)]:
            kname = f'{name}{suf}'
            ckp = f'{OUT_V6}/model_{name}{suf}.pt'
            if not os.path.exists(ckp) or kname in v6_probs:
                continue
            p0_i, p1_i, p2_i, g_i = infer_v3_ckpt_tta(ckp, ordered_names, dev)
            v6_probs[kname] = (p0_i, p1_i, p2_i, g_i, w)
            np.savez_compressed(v6_cache_path, data=v6_probs)
            print(f'Inferred {kname:24s} on {dev}', flush=True)
    print(f'Saved V6 18-ckpt TTA probability cache to {v6_cache_path}', flush=True)


if __name__ == '__main__':
    main()
