"""Evaluate Multi-Backbone High-Res + 4-Expert Routing on 1,000-Image Unseen Test Sets (s16 & s17),
and Fuse Cached V2/V3/V4/V5/V6/V7 TTA Probabilities into Final pred_nriqa_dataset_test.json.
"""
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from run_v3_super_ensemble import infer_v3_ckpt_tta, decode_v3_shielded, GRADE
from run_v6_ablate_and_specialists import FULL_JOBS, OUT_ABL, OUT_V6

OUT_V7 = 'results/v7_overgen_pro'
V7_JOBS = [
    ('v7_og_d3_448_t3_s1',    0, 1, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 3, 4, 2),
    ('v7_og_quad_448_t2_s1',  1, 1, 'eupe:full,siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 448, 2, 4, 2),
    ('v7_og_d3_448_t3_s2',    2, 2, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 3, 4, 2),
    ('v7_og_pcl_448_t3_s1',   3, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',           448, 3, 4, 2),
    ('v7_og_sig2g_384_t3_s1', 4, 1, 'siglip2g:lastk4,cradio:lastk4,dinov3l:lastk4',          384, 3, 4, 2),
    ('v7_og_pcg_384_t2_s1',   5, 1, 'siglip2:lastk4,cradio:lastk4,pecoreg:lastk4',           384, 2, 4, 2),
]


def eval_multi_backbone_4expert_blind_test():
    """Evaluate Multi-Backbone High-Res Clean Base + 4-Expert Routing on 1,000-image s16 & s17."""
    print('\n=== Evaluating Multi-Backbone High-Res + 4-Expert Routing on 1,000-Image Unseen Test Sets (s16 & s17) ===', flush=True)
    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    hdr = f"{'variant':22s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} {'S_cln':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)

    for mode in ['Single_384_Base', 'Single_384_4Exp', 'MultiBB_HiRes_Base', 'MultiBB_HiRes_4Exp']:
        rows = []
        for sp in (16, 17):
            da = np.load(f'{OUT_ABL}/A_base_s{sp}_probs.npz')
            df = np.load(f'{OUT_ABL}/F_v5_s{sp}_probs.npz')
            de = np.load(f'{OUT_ABL}/E_synth_s{sp}_probs.npz')
            dg = np.load(f'{OUT_ABL}/G_v6_s{sp}_probs.npz')
            dh = np.load(f'{OUT_ABL}/H_og_s{sp}_probs.npz')

            extra_clean = []
            for p in sorted(glob.glob(f'results/v6_val/*s{sp}*probs.npz') + glob.glob(f'{OUT_ABL}/B_*_s{sp}_probs.npz') + glob.glob(f'{OUT_ABL}/C_*_s{sp}_probs.npz') + glob.glob(f'{OUT_ABL}/D_*_s{sp}_probs.npz')):
                try:
                    d_ex = np.load(p)
                    if 'te_p0' in d_ex and d_ex['te_p0'].shape == da['te_p0'].shape:
                        extra_clean.append((os.path.basename(p), d_ex))
                except Exception:
                    pass

            if mode.startswith('Single_384'):
                c_p0 = 0.45 * da['te_p0'] + 0.55 * df['te_p0']
                c_p1 = 0.45 * da['te_p1'] + 0.55 * df['te_p1']
                c_p2 = 0.45 * da['te_p2'] + 0.55 * df['te_p2']
            else:
                w_tot = 0.45 + 0.55
                c_p0 = 0.45 * da['te_p0'] + 0.55 * df['te_p0']
                c_p1 = 0.45 * da['te_p1'] + 0.55 * df['te_p1']
                c_p2 = 0.45 * da['te_p2'] + 0.55 * df['te_p2']
                for _, d_ex in extra_clean:
                    c_p0 += 0.35 * d_ex['te_p0']
                    c_p1 += 0.35 * d_ex['te_p1']
                    c_p2 += 0.35 * d_ex['te_p2']
                    w_tot += 0.35
                c_p0 /= w_tot
                c_p1 /= w_tot
                c_p2 /= w_tot

            if mode.endswith('_Base'):
                p0, p1, p2 = c_p0, c_p1, c_p2
            else:
                p0, p1, p2 = c_p0.copy(), c_p1.copy(), c_p2.copy()
                # 模糊 (0) & 纹理损失 (1): 75% Clean + 15% V6 + 10% V7 (never demote P0 recall)
                for ci in [0, 1]:
                    p0[:, ci] = np.maximum(c_p0[:, ci], 0.75 * c_p0[:, ci] + 0.15 * dg['te_p0'][:, ci] + 0.10 * dh['te_p0'][:, ci])
                    p1[:, ci] = 0.75 * c_p1[:, ci] + 0.15 * dg['te_p1'][:, ci] + 0.10 * dh['te_p1'][:, ci]
                    p2[:, ci] = 0.75 * c_p2[:, ci] + 0.15 * dg['te_p2'][:, ci] + 0.10 * dh['te_p2'][:, ci]
                # 过锐 (2): 72% Clean + 16% E_sharp + 12% V6
                p0[:, 2] = 0.72 * c_p0[:, 2] + 0.16 * de['te_p0'][:, 2] + 0.12 * dg['te_p0'][:, 2]
                p1[:, 2] = 0.72 * c_p1[:, 2] + 0.16 * de['te_p1'][:, 2] + 0.12 * dg['te_p1'][:, 2]
                p2[:, 2] = 0.72 * c_p2[:, 2] + 0.16 * de['te_p2'][:, 2] + 0.12 * dg['te_p2'][:, 2]
                # 花屏 (3): 100% Clean
                p0[:, 3] = c_p0[:, 3]
                # 过度生成 (ci=4, fi=3): 75% Clean + 16% V7 + 9% V6
                p0[:, 4] = 0.75 * c_p0[:, 4] + 0.16 * dh['te_p0'][:, 4] + 0.09 * dg['te_p0'][:, 4]
                p1[:, 3] = 0.75 * c_p1[:, 3] + 0.16 * dh['te_p1'][:, 3] + 0.09 * dg['te_p1'][:, 3]
                p2[:, 3] = 0.75 * c_p2[:, 3] + 0.16 * dh['te_p2'][:, 3] + 0.09 * dg['te_p2'][:, 3]
                # 虚化突兀 (5): 75% Clean + 25% V6
                p0[:, 5] = 0.75 * c_p0[:, 5] + 0.25 * dg['te_p0'][:, 5]

            pred = decode_pc(p0, p1, p2, HAND)
            s = cm.score(da['gte'], pred, TAU)
            f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
            rows.append((s, f1))
            print(f"{mode+'_s'+str(sp):22s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want), flush=True)
        if rows:
            m = lambda k: np.mean([r[0][k] for r in rows])
            mf = [np.mean([r[1][w] for r in rows]) for w in want]
            print(f"{mode+' MEAN':22s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf), flush=True)
            print(flush=True)


def infer_and_fuse_all_specialists():
    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]

    # 1. Load V6 Multi-Synth Specialist checkpoints (18 ckpts)
    v6_cache_path = f'{OUT_V6}/test_150_v6_models_probs.npz'
    v6_probs = np.load(v6_cache_path, allow_pickle=True)['data'].item()

    # 2. Load V7 OverGen-Pro Specialist checkpoints (18 ckpts)
    v7_cache_path = f'{OUT_V7}/test_150_v7_models_probs.npz'
    v7_probs = np.load(v7_cache_path, allow_pickle=True)['data'].item()

    print(f'Loaded V6 ckpts: {len(v6_probs)}, V7 OverGen-Pro ckpts: {len(v7_probs)}', flush=True)

    eval_multi_backbone_4expert_blind_test()

    # 3. Load Expert 1 (Verified 52.43 Clean V2 + V3 Mega-Ensemble: 21 models)
    v2_probs = np.load('results/v2_super/test_150_all_models_probs.npz', allow_pickle=True)['data'].item()
    v2_fam_w = {'pcl': 1.3, 'd3': 1.3, 'eupe': 1.1, 'base': 0.8}
    v3_probs = np.load('results/v3_super/test_150_v3_models_probs.npz', allow_pickle=True)['data'].item()

    cln_p0, cln_p1, cln_p2, cln_w = 0.0, 0.0, 0.0, 0.0
    for name, (p0_i, p1_i, p2_i, fam, is_iso) in v2_probs.items():
        wi = v2_fam_w.get(fam, 1.0) * (0.70 if is_iso else 1.0)
        cln_p0 = cln_p0 + wi * p0_i
        cln_p1 = cln_p1 + wi * p1_i
        cln_p2 = cln_p2 + wi * p2_i
        cln_w += wi
    for name, val in v3_probs.items():
        p0_i, p1_i, p2_i = val[0], val[1], val[2]
        wi = float(val[-1])
        cln_p0 = cln_p0 + wi * p0_i
        cln_p1 = cln_p1 + wi * p1_i
        cln_p2 = cln_p2 + wi * p2_i
        cln_w += wi

    cln_p0 /= cln_w
    cln_p1 /= cln_w
    cln_p2 /= cln_w

    def avg_ckpt_dict(pdict, w_neg_corr=1.0):
        p0_acc, p1_acc, p2_acc, w_tot = 0.0, 0.0, 0.0, 0.0
        for k, val in pdict.items():
            p0_i, p1_i, p2_i = val[0], val[1], val[2]
            wi = float(val[-1]) if isinstance(val[-1], (int, float, np.floating)) else 1.0
            p0_acc = p0_acc + wi * p0_i
            p1_acc = p1_acc + wi * p1_i
            p2_acc = p2_acc + wi * p2_i
            w_tot += wi
        p0_m, p1_m, p2_m = p0_acc / w_tot, p1_acc / w_tot, p2_acc / w_tot
        if w_neg_corr != 1.0:
            # Exact Bayes odds correction for models trained with --channel-neg-weight=1.35
            # so their cumulative probabilities (p0, p0+p1, p0+p1+p2) align with w_neg=1.0 (52.43 base)
            def corr(q):
                q = np.clip(q, 1e-6, 1.0 - 1e-6)
                return (w_neg_corr * q) / (w_neg_corr * q + (1.0 - q))
            # Correct cumulative probabilities and recover marginal p0, p1, p2
            c0 = corr(p0_m)
            c01_full = corr(p0_m[:, [0, 1, 2, 4]] + p1_m)
            c012_full = corr(p0_m[:, [0, 1, 2, 4]] + p1_m + p2_m)
            p0_m = c0
            p1_m = np.maximum(0.0, c01_full - c0[:, [0, 1, 2, 4]])
            p2_m = np.maximum(0.0, c012_full - c01_full)
        return p0_m, p1_m, p2_m

    # Expert 4: V6 Multi-Synth Specialist (18 ckpts, trained with channel_neg_weight=1.35)
    s6_p0, s6_p1, s6_p2 = avg_ckpt_dict(v6_probs, w_neg_corr=1.35)
    # Expert 2: V7 OverGen-Pro Dedicated Specialist (18 ckpts, trained with channel_neg_weight=1.35)
    s7_p0, s7_p1, s7_p2 = avg_ckpt_dict(v7_probs, w_neg_corr=1.35)

    # Expert 3: V4 Sharp Specialist (trained with channel_neg_weight=1.0)
    v4_all = np.load('results/v4_synth/test_150_v4_models_probs.npz', allow_pickle=True)['data'].item()
    print(f'Loaded Expert 3 (Sharp) checkpoints: {len(v4_all)} models', flush=True)
    s4_p0, s4_p1, s4_p2 = avg_ckpt_dict(v4_all, w_neg_corr=1.0)

    # 4-Expert Per-Channel Specialist Routing:
    # CLASSES = ['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']
    # FULL    = ['模糊','纹理损失','过锐','过度生成']
    fin_p0 = cln_p0.copy()
    fin_p1 = cln_p1.copy()
    fin_p2 = cln_p2.copy()

    # 模糊 (0) & 纹理损失 (1): 75% Clean + 15% V6 + 10% V7 (never demote P0 recall)
    for ci in [0, 1]:
        fin_p0[:, ci] = np.maximum(cln_p0[:, ci], 0.75 * cln_p0[:, ci] + 0.15 * s6_p0[:, ci] + 0.10 * s7_p0[:, ci])
        fin_p1[:, ci] = 0.75 * cln_p1[:, ci] + 0.15 * s6_p1[:, ci] + 0.10 * s7_p1[:, ci]
        fin_p2[:, ci] = 0.75 * cln_p2[:, ci] + 0.15 * s6_p2[:, ci] + 0.10 * s7_p2[:, ci]

    # 过锐 (ci=2, fi=2): 72% Clean + 16% V4_Sharp + 12% V6
    fin_p0[:, 2] = 0.72 * cln_p0[:, 2] + 0.16 * s4_p0[:, 2] + 0.12 * s6_p0[:, 2]
    fin_p1[:, 2] = 0.72 * cln_p1[:, 2] + 0.16 * s4_p1[:, 2] + 0.12 * s6_p1[:, 2]
    fin_p2[:, 2] = 0.72 * cln_p2[:, 2] + 0.16 * s4_p2[:, 2] + 0.12 * s6_p2[:, 2]

    # 花屏 (ci=3): 100% Clean (F1 is already 0.970!)
    fin_p0[:, 3] = cln_p0[:, 3]

    # 过度生成 (ci=4, fi=3): 75% Clean + 16% V7 OverGen-Pro + 9% V6 Multi-Synth
    fin_p0[:, 4] = 0.75 * cln_p0[:, 4] + 0.16 * s7_p0[:, 4] + 0.09 * s6_p0[:, 4]
    fin_p1[:, 3] = 0.75 * cln_p1[:, 3] + 0.16 * s7_p1[:, 3] + 0.09 * s6_p1[:, 3]
    fin_p2[:, 3] = 0.75 * cln_p2[:, 3] + 0.16 * s7_p2[:, 3] + 0.09 * s6_p2[:, 3]

    # 虚化突兀 (ci=5): 75% Clean + 25% V6
    fin_p0[:, 5] = 0.75 * cln_p0[:, 5] + 0.25 * s6_p0[:, 5]

    b5243 = {e['image']: e['label'] for e in json.load(open('results/v3_super/pred_nriqa_dataset_test.json'))}
    pred_mat = decode_v3_shielded(fin_p0, fin_p1, fin_p2)

    out_entries = []
    for i, name in enumerate(ordered_names):
        old_lbl = b5243[name]
        if all(v == '无问题' for v in old_lbl.values()):
            # Strictly preserve verified 20.00/20.00 Clean Shield from 52.43 baseline
            lbl = old_lbl.copy()
        else:
            lbl = {k: GRADE[int(pred_mat[i, CLASSES.index(k)])] for k in ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']}
            if all(v == '无问题' for v in lbl.values()):
                lbl = old_lbl.copy()
        out_entries.append({'image': name, 'label': lbl})

    v7_json_path = f'{OUT_V7}/pred_nriqa_dataset_test.json'
    v6_json_path = f'{OUT_V6}/pred_nriqa_dataset_test.json'
    for p in [v7_json_path, v6_json_path, 'pred_nriqa_dataset_test.json']:
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(out_entries, f, ensure_ascii=False, indent=2)

    diffs = []
    trans = Counter()
    for e in out_entries:
        im = e['image']
        for c in CLASSES:
            old_v = b5243[im][c]
            new_v = e['label'][c]
            if old_v != new_v:
                diffs.append((im, c, old_v, new_v))
                trans[(c, f'{old_v}->{new_v}')] += 1

    print(f'\nSaved {v7_json_path} and updated root pred_nriqa_dataset_test.json!')
    print(f'Total cell diffs vs 52.43 baseline: {len(diffs)} / {len(ordered_names)*6}')
    for (c, tr), cnt in sorted(trans.items()):
        print(f'  {c:8s}: {tr:12s} x {cnt}')
    print('\nDetailed cell refinements vs 52.43 baseline:')
    for im, c, ov, nv in diffs:
        print(f'  {im:24s} | {c:6s}: {ov} -> {nv}')

    clean_old = sum(1 for e in json.load(open('results/v3_super/pred_nriqa_dataset_test.json')) if all(v == '无问题' for v in e['label'].values()))
    clean_new = sum(1 for e in out_entries if all(v == '无问题' for v in e['label'].values()))
    print(f'All-clean images: 52.43_baseline={clean_old}/{len(ordered_names)}, 4Expert_V7={clean_new}/{len(ordered_names)}')


if __name__ == '__main__':
    infer_and_fuse_all_specialists()
