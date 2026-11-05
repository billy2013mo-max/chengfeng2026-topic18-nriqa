"""Build V7 OverGen-Pro Datasets and Launch OverGen-Pro Ablation (GPUs 0..1) + Full Specialists.

1. Builds `s16`, `s17`, and `full` OverGen-Pro datasets on GPU 0.
2. Trains `H_og_s16` (GPU 0) and `H_og_s17` (GPU 1) on the 1,000-image independent test splits.
3. Evaluates the 3-Tier Specialist Fusion (Clean + V6 Multi-Synth + V7 OverGen-Pro) on s16 & s17.
4. Trains 2 immediate full-data OverGen-Pro 448x448 Specialist Models on GPUs 0..1 (plus more once GPUs 2..7 free up).
"""
import json
import os
import subprocess
import sys
import time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm

OUT_ABL = 'results/v3_ablate'
OUT_V7 = 'results/v7_overgen_pro'
os.makedirs(OUT_ABL, exist_ok=True)
os.makedirs(OUT_V7, exist_ok=True)


def eval_3tier_specialist():
    print('\n=== Evaluating 3-Tier Specialist Ensemble on 1,000-Image Independent Test Sets (s16 & s17) ===', flush=True)
    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    hdr = f"{'variant':18s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} {'S_cln':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)

    for n in ['A_base', 'F_v5', 'G_v6', 'H_og']:
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
            print(f"{n+'_s'+str(sp):18s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want), flush=True)
        if rows:
            m = lambda k: np.mean([r[0][k] for r in rows])
            mf = [np.mean([r[1][w] for r in rows]) for w in want]
            print(f"{n+' MEAN':18s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf), flush=True)
            print(flush=True)

    # Compare Clean_Ens vs 2-Tier (Clean + V6) vs 3-Tier (Clean + V6 + E_sharp + H_og OverGen-Pro)
    # CLASSES = ['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']
    # FULL    = ['模糊','纹理损失','过锐','过度生成']
    for tag, use_3tier in [('Clean_Ens', False), ('3Tier_Spec_Ens', True)]:
        rows = []
        for sp in (16, 17):
            p_h = f'{OUT_ABL}/H_og_s{sp}_probs.npz'
            if not os.path.exists(p_h):
                continue
            da = np.load(f'{OUT_ABL}/A_base_s{sp}_probs.npz')
            df = np.load(f'{OUT_ABL}/F_v5_s{sp}_probs.npz')
            de = np.load(f'{OUT_ABL}/E_synth_s{sp}_probs.npz')
            dg = np.load(f'{OUT_ABL}/G_v6_s{sp}_probs.npz')
            dh = np.load(p_h)

            c_p0 = 0.45 * da['te_p0'] + 0.55 * df['te_p0']
            c_p1 = 0.45 * da['te_p1'] + 0.55 * df['te_p1']
            c_p2 = 0.45 * da['te_p2'] + 0.55 * df['te_p2']

            if not use_3tier:
                p0, p1, p2 = c_p0, c_p1, c_p2
            else:
                # Per-channel specialist routing:
                # 0:模糊, 1:纹理损失, 2:过锐, 3:花屏, 4:过度生成, 5:虚化突兀
                p0 = c_p0.copy()
                p1 = c_p1.copy()
                p2 = c_p2.copy()

                # 模糊 (0) & 纹理损失 (1): 70% Clean + 20% G_v6 + 10% H_og
                for ci in [0, 1]:
                    p0[:, ci] = 0.70 * c_p0[:, ci] + 0.20 * dg['te_p0'][:, ci] + 0.10 * dh['te_p0'][:, ci]
                    p1[:, ci] = 0.70 * c_p1[:, ci] + 0.20 * dg['te_p1'][:, ci] + 0.10 * dh['te_p1'][:, ci]
                    p2[:, ci] = 0.70 * c_p2[:, ci] + 0.20 * dg['te_p2'][:, ci] + 0.10 * dh['te_p2'][:, ci]

                # 过锐 (ci=2, fi=2): 65% Clean + 20% E_synth + 15% G_v6
                p0[:, 2] = 0.65 * c_p0[:, 2] + 0.20 * de['te_p0'][:, 2] + 0.15 * dg['te_p0'][:, 2]
                p1[:, 2] = 0.65 * c_p1[:, 2] + 0.20 * de['te_p1'][:, 2] + 0.15 * dg['te_p1'][:, 2]
                p2[:, 2] = 0.65 * c_p2[:, 2] + 0.20 * de['te_p2'][:, 2] + 0.15 * dg['te_p2'][:, 2]

                # 花屏 (ci=3): 100% Clean
                p0[:, 3] = c_p0[:, 3]

                # 过度生成 (ci=4, fi=3): 50% Clean + 25% G_v6 + 25% H_og (OverGen-Pro Specialist!)
                p0[:, 4] = 0.50 * c_p0[:, 4] + 0.25 * dg['te_p0'][:, 4] + 0.25 * dh['te_p0'][:, 4]
                p1[:, 3] = 0.50 * c_p1[:, 3] + 0.25 * dg['te_p1'][:, 3] + 0.25 * dh['te_p1'][:, 3]
                p2[:, 3] = 0.50 * c_p2[:, 3] + 0.25 * dg['te_p2'][:, 3] + 0.25 * dh['te_p2'][:, 3]

                # 虚化突兀 (ci=5): 65% Clean + 35% G_v6
                p0[:, 5] = 0.65 * c_p0[:, 5] + 0.35 * dg['te_p0'][:, 5]

            pred = decode_pc(p0, p1, p2, HAND)
            s = cm.score(dh['gte'], pred, TAU)
            f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
            rows.append((s, f1))
            print(f"{tag+'_s'+str(sp):18s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want), flush=True)
        if rows:
            m = lambda k: np.mean([r[0][k] for r in rows])
            mf = [np.mean([r[1][w] for r in rows]) for w in want]
            print(f"{tag+' MEAN':18s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf), flush=True)
            print(flush=True)


def main():
    print('=== Step 0: Building V7 OverGen-Pro Datasets ===', flush=True)
    subprocess.run(['python3', 'code/finetune/build_overgen_pro.py'], check=True)

    print('\n=== Step 1: Launching OverGen-Pro Ablation Models H_og_s16 (GPU 0) & H_og_s17 (GPU 1) ===', flush=True)
    abl_jobs = [
        ('H_og_s16', 0, 'data/resplits/split_s16_overgen_pro.csv', 'data/manifest_s16_overgen_pro.csv'),
        ('H_og_s17', 1, 'data/resplits/split_s17_overgen_pro.csv', 'data/manifest_s17_overgen_pro.csv'),
    ]
    procs = []
    t0 = time.time()
    for tag, gpu, split_csv, man_csv in abl_jobs:
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
        procs.append((tag, p, lf))
        print(f'[launched] {tag} on GPU {gpu}', flush=True)

    for tag, p, lf in procs:
        rc = p.wait()
        lf.close()
        print(f'[{time.time()-t0:.0f}s] Finished {tag} rc={rc}', flush=True)

    eval_3tier_specialist()

    print('\n=== Step 2: Training Full-Data High-Res OverGen-Pro Specialist Models on GPUs 0..1 ===', flush=True)
    full_og_jobs = [
        ('v7_og_d3_448_t3_s1',   0, 1, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 3, 4, 2),
        ('v7_og_quad_448_t2_s1', 1, 1, 'eupe:full,siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 448, 2, 4, 2),
    ]
    fprocs = []
    for name, gpu, sd, bbs, sz, tiles, bs, accum in full_og_jobs:
        out_json = f'{OUT_V7}/train_{name}.json'
        ckpt_pt = f'{OUT_V7}/model_{name}.pt'
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
            '--split', 'data/split_all_full_overgen_pro.csv',
            '--manifest', 'data/manifest_combined_overgen_pro.csv',
            '--out', out_json,
            '--save-state', ckpt_pt,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        lf = open(f'{OUT_V7}/train_{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        fprocs.append((name, p, lf))
        print(f'[launched] {name} on GPU {gpu}', flush=True)

    for name, p, lf in fprocs:
        rc = p.wait()
        lf.close()
        print(f'[{time.time()-t0:.0f}s] Finished full OverGen-Pro specialist {name} rc={rc}', flush=True)


if __name__ == '__main__':
    main()
