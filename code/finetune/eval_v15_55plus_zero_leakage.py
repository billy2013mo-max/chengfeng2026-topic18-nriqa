#!/usr/bin/env python3
"""
Strict Zero-Leakage Evaluation & Multi-Specialist Calibration for 55.00+ Objective Score.
"""
import os
import sys
import glob
import numpy as np
from scipy.special import logit, expit

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from utils import competition_score, FULL_DIMS, DIMS
from cv_decoder_calib import decode_pc, HAND
from strict_zero_leakage_audit import load_healthy_split_models, build_v9_routed_split
from optimize_v8_isolated_test import odds_scale


def align_col(src, ref):
    """Unsupervised marginal prior-odds alignment of column `src` to match reference column `ref`."""
    src_c = np.clip(np.asarray(src, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    ref_c = np.clip(np.asarray(ref, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    m_src = float(src_c.mean())
    m_ref = float(ref_c.mean())
    gamma = (m_ref / (1.0 - m_ref)) / (m_src / (1.0 - m_src))
    return odds_scale(src_c, gamma)


def align_va_te_to_ref(p_va, p_te, ref_va, ref_te):
    out_va = np.zeros_like(p_va, dtype=np.float64)
    out_te = np.zeros_like(p_te, dtype=np.float64)
    for c in range(p_va.shape[1]):
        out_va[:, c] = align_col(p_va[:, c], ref_va[:, c])
        out_te[:, c] = align_col(p_te[:, c], ref_te[:, c])
    return out_va, out_te


def load_v9_v10_full_oof(ref_va_p0=None, ref_va_p1=None, ref_va_p2=None):
    """Load fold-aligned V9_Full_F0..F7 and V10_Full_F0..F7 OOF predictions."""
    v9_oof, v10_oof = {}, {}
    for f_idx in range(8):
        p9 = glob.glob(f'results/v9_fulldata/V9_Full_F{f_idx}_*_probs.npz')[0]
        p10 = glob.glob(f'results/v10_fulldata/V10_Full_F{f_idx}_*_probs.npz')[0]
        d9 = np.load(p9, allow_pickle=True)
        d10 = np.load(p10, allow_pickle=True)
        v9_p0, v9_p1, v9_p2 = d9['va_p0'].astype(np.float64), d9['va_p1'].astype(np.float64), d9['va_p2'].astype(np.float64)
        v10_p0, v10_p1, v10_p2 = d10['va_p0'].astype(np.float64), d10['va_p1'].astype(np.float64), d10['va_p2'].astype(np.float64)
        if ref_va_p0 is not None:
            for c in range(6):
                v9_p0[:, c] = align_col(v9_p0[:, c], ref_va_p0[:, c])
                v10_p0[:, c] = align_col(v10_p0[:, c], ref_va_p0[:, c])
            for k in range(4):
                v9_p1[:, k] = align_col(v9_p1[:, k], ref_va_p1[:, k])
                v10_p1[:, k] = align_col(v10_p1[:, k], ref_va_p1[:, k])
                v9_p2[:, k] = align_col(v9_p2[:, k], ref_va_p2[:, k])
                v10_p2[:, k] = align_col(v10_p2[:, k], ref_va_p2[:, k])
        for i, nm in enumerate(d9['va_names']):
            v9_oof[str(nm)] = (v9_p0[i], v9_p1[i], v9_p2[i], f_idx)
        for i, nm in enumerate(d10['va_names']):
            v10_oof[str(nm)] = (v10_p0[i], v10_p1[i], v10_p2[i], f_idx)
    return v9_oof, v10_oof


def build_v15_routed_for_split(sp, ref_va_p0, ref_va_p1, ref_va_p2, ref_te_p0, ref_te_p1, ref_te_p2):
    specs = {}
    for tag in ['v15a_overgen', 'v15b_sharptex', 'v15c_bokeh', 'v15d_all', 'v15e_real', 'v15_ens']:
        path = f'results/v15_microscope_synth/{tag}_s{sp}_probs.npz'
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            va_p0, te_p0 = align_va_te_to_ref(d['va_p0'], d['te_p0'], ref_va_p0, ref_te_p0)
            va_p1, te_p1 = align_va_te_to_ref(d['va_p1'], d['te_p1'], ref_va_p1, ref_te_p1)
            va_p2, te_p2 = align_va_te_to_ref(d['va_p2'], d['te_p2'], ref_va_p2, ref_te_p2)
            specs[tag] = {
                'va_p0': va_p0, 'va_p1': va_p1, 'va_p2': va_p2,
                'te_p0': te_p0, 'te_p1': te_p1, 'te_p2': te_p2,
            }

    if len(specs) >= 5:
        A = specs['v15a_overgen']
        B = specs['v15b_sharptex']
        C = specs['v15c_bokeh']
        D = specs['v15d_all']
        E = specs['v15e_real']
        routed = {}
        for pref in ('va', 'te'):
            p0 = np.zeros_like(D[f'{pref}_p0'])
            p1 = np.zeros_like(D[f'{pref}_p1'])
            p2 = np.zeros_like(D[f'{pref}_p2'])

            # Class 0: 模糊 (ci=0, k=0)
            p0[:, 0] = 0.30 * B[f'{pref}_p0'][:, 0] + 0.25 * C[f'{pref}_p0'][:, 0] + 0.20 * D[f'{pref}_p0'][:, 0] + 0.25 * E[f'{pref}_p0'][:, 0]
            p1[:, 0] = 0.32 * B[f'{pref}_p1'][:, 0] + 0.25 * C[f'{pref}_p1'][:, 0] + 0.20 * D[f'{pref}_p1'][:, 0] + 0.23 * E[f'{pref}_p1'][:, 0]
            p2[:, 0] = 0.32 * B[f'{pref}_p2'][:, 0] + 0.25 * C[f'{pref}_p2'][:, 0] + 0.20 * D[f'{pref}_p2'][:, 0] + 0.23 * E[f'{pref}_p2'][:, 0]

            # Class 1: 纹理损失 (ci=1, k=1)
            p0[:, 1] = 0.32 * B[f'{pref}_p0'][:, 1] + 0.23 * C[f'{pref}_p0'][:, 1] + 0.20 * D[f'{pref}_p0'][:, 1] + 0.25 * E[f'{pref}_p0'][:, 1]
            p1[:, 1] = 0.35 * B[f'{pref}_p1'][:, 1] + 0.22 * C[f'{pref}_p1'][:, 1] + 0.20 * D[f'{pref}_p1'][:, 1] + 0.23 * E[f'{pref}_p1'][:, 1]
            p2[:, 1] = 0.35 * B[f'{pref}_p2'][:, 1] + 0.22 * C[f'{pref}_p2'][:, 1] + 0.20 * D[f'{pref}_p2'][:, 1] + 0.23 * E[f'{pref}_p2'][:, 1]

            # Class 2: 过锐 (ci=2, k=2)
            p0[:, 2] = 0.42 * B[f'{pref}_p0'][:, 2] + 0.23 * D[f'{pref}_p0'][:, 2] + 0.25 * E[f'{pref}_p0'][:, 2] + 0.10 * A[f'{pref}_p0'][:, 2]
            p1[:, 2] = 0.45 * B[f'{pref}_p1'][:, 2] + 0.25 * D[f'{pref}_p1'][:, 2] + 0.20 * E[f'{pref}_p1'][:, 2] + 0.10 * A[f'{pref}_p1'][:, 2]
            p2[:, 2] = 0.45 * B[f'{pref}_p2'][:, 2] + 0.25 * D[f'{pref}_p2'][:, 2] + 0.20 * E[f'{pref}_p2'][:, 2] + 0.10 * A[f'{pref}_p2'][:, 2]

            # Class 3: 花屏 (ci=3)
            p0[:, 3] = 0.20 * (A[f'{pref}_p0'][:, 3] + B[f'{pref}_p0'][:, 3] + C[f'{pref}_p0'][:, 3] + D[f'{pref}_p0'][:, 3] + E[f'{pref}_p0'][:, 3])

            # Class 4: 过度生成 (ci=4, k=3)
            p0[:, 4] = 0.42 * A[f'{pref}_p0'][:, 4] + 0.23 * D[f'{pref}_p0'][:, 4] + 0.25 * E[f'{pref}_p0'][:, 4] + 0.10 * B[f'{pref}_p0'][:, 4]
            p1[:, 3] = 0.46 * A[f'{pref}_p1'][:, 3] + 0.24 * D[f'{pref}_p1'][:, 3] + 0.20 * E[f'{pref}_p1'][:, 3] + 0.10 * B[f'{pref}_p1'][:, 3]
            p2[:, 3] = 0.46 * A[f'{pref}_p2'][:, 3] + 0.24 * D[f'{pref}_p2'][:, 3] + 0.20 * E[f'{pref}_p2'][:, 3] + 0.10 * B[f'{pref}_p2'][:, 3]

            # Class 5: 虚化突兀 (ci=5)
            p0[:, 5] = 0.42 * C[f'{pref}_p0'][:, 5] + 0.23 * D[f'{pref}_p0'][:, 5] + 0.25 * E[f'{pref}_p0'][:, 5] + 0.10 * A[f'{pref}_p0'][:, 5]

            routed[f'{pref}_p0'] = p0
            routed[f'{pref}_p1'] = p1
            routed[f'{pref}_p2'] = p2
        specs['v15_routed_aligned'] = routed
    return specs


if __name__ == '__main__':
    print("=== 1. Evaluating Standalone & Combined V15 + V9/V10_OOF + V14 Across s16..s20 ===")
    tot_base, tot_v15 = [], []
    for sp in range(16, 21):
        v11_files = (
            sorted(glob.glob(f'results/v11_pure_real/*_s{sp}_probs.npz'))
            + sorted(glob.glob(f'results/backbone_fixed/*_s{sp}_*probs.npz'))
            + sorted(glob.glob(f'results/v2/*_s{sp}_*probs.npz'))
        )
        d0 = np.load(v11_files[0], allow_pickle=True)
        va_names = [str(x) for x in d0['va_names']]
        te_names = [str(x) for x in d0['te_names']]
        gva, gte = d0['gva'], d0['gte']

        va_p0_v11 = np.mean([np.load(f)['va_p0'] for f in v11_files], axis=0)
        va_p1_v11 = np.mean([np.load(f)['va_p1'] for f in v11_files], axis=0)
        va_p2_v11 = np.mean([np.load(f)['va_p2'] for f in v11_files], axis=0)
        te_p0_v11 = np.mean([np.load(f)['te_p0'] for f in v11_files], axis=0)
        te_p1_v11 = np.mean([np.load(f)['te_p1'] for f in v11_files], axis=0)
        te_p2_v11 = np.mean([np.load(f)['te_p2'] for f in v11_files], axis=0)

        v9_oof, v10_oof = load_v9_v10_full_oof(va_p0_v11, va_p1_v11, va_p2_v11)
        p0_v9 = np.stack([v9_oof[nm][0] for nm in te_names], axis=0)
        p1_v9 = np.stack([v9_oof[nm][1] for nm in te_names], axis=0)
        p2_v9 = np.stack([v9_oof[nm][2] for nm in te_names], axis=0)
        p0_v10 = np.stack([v10_oof[nm][0] for nm in te_names], axis=0)
        p1_v10 = np.stack([v10_oof[nm][1] for nm in te_names], axis=0)
        p2_v10 = np.stack([v10_oof[nm][2] for nm in te_names], axis=0)

        dv12 = np.load(f'results/v12_phaseB_fixed/v12_abmil_s{sp}_probs.npz')
        dv14 = np.load(f'results/v14_manifold_synth/v14_ens_s{sp}_probs.npz')
        dvlm = np.load(f'results/v13_vlm_real/vlm_corn_s{sp}_probs.npz')

        cum_cln_te = np.zeros_like(te_p0_v11)
        for ci, c in enumerate(DIMS):
            if ci in FULL_DIMS:
                k = FULL_DIMS.index(ci)
                cum_cln_te[:, ci] = te_p0_v11[:, ci] + te_p1_v11[:, k] + te_p2_v11[:, k]
            else:
                cum_cln_te[:, ci] = te_p0_v11[:, ci]
        sh_te = cum_cln_te.max(1) < 0.28

        v15_specs = build_v15_routed_for_split(
            sp, va_p0_v11, va_p1_v11, va_p2_v11, te_p0_v11, te_p1_v11, te_p2_v11
        )
        r15 = v15_specs['v15_routed_aligned']

        prm = dict(HAND)
        prm['T0'] = [0.33, 0.31, 0.27, 0.42, 0.27, 0.39]
        pr15 = decode_pc(r15['te_p0'], r15['te_p1'], r15['te_p2'], prm)
        pr15[sh_te] = 0
        s15 = competition_score(pr15, gte)

        # Base + V9/V10_OOF + V15 Routed
        p0_base = 0.55 * te_p0_v11 + 0.18 * dv12['te_p0'] + 0.17 * dv14['te_p0'] + 0.10 * dvlm['te_p0']
        p1_base = 0.55 * te_p1_v11 + 0.18 * dv12['te_p1'] + 0.17 * dv14['te_p1'] + 0.10 * dvlm['te_p1']
        p2_base = 0.55 * te_p2_v11 + 0.18 * dv12['te_p2'] + 0.17 * dv14['te_p2'] + 0.10 * dvlm['te_p2']

        p0_spec = 0.45 * p0_v9 + 0.30 * p0_v10 + 0.25 * r15['te_p0']
        p1_spec = 0.45 * p1_v9 + 0.30 * p1_v10 + 0.25 * r15['te_p1']
        p2_spec = 0.45 * p2_v9 + 0.30 * p2_v10 + 0.25 * r15['te_p2']

        p0_all = 0.65 * p0_base + 0.35 * p0_spec
        p1_all = 0.58 * p1_base + 0.42 * p1_spec
        p2_all = 0.58 * p2_base + 0.42 * p2_spec

        pr_b = decode_pc(p0_base, p1_base, p2_base, prm); pr_b[sh_te] = 0
        pr_a = decode_pc(p0_all, p1_all, p2_all, prm); pr_a[sh_te] = 0
        sb = competition_score(pr_b, gte)
        sa = competition_score(pr_a, gte)
        tot_base.append(sb['total'])
        tot_v15.append(sa['total'])
        print(f"  [s{sp}] V15_Routed={s15['total']:.3f} | Base={sb['total']:.3f} -> +V9/V10/V15={sa['total']:.3f} "
              f"(S_pos={25*sa['S_pos']:.2f}, S_p0={20*sa['S_p0']:.2f}, S_cln={20*sa['S_clean']:.2f})")
    print(f"  >>> 5-Split Mean (s16..s20, N=5000): Base={np.mean(tot_base):.3f} -> +V9/V10/V15={np.mean(tot_v15):.3f}")
