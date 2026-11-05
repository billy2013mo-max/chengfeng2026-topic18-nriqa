"""Evaluate Decoupled 14-Channel Specialist Routing + Prior-Odds Alignment + Two-Stage Conditional
Grade Calibration on the 1,000-Image Isolated Test Sets (s16 & s17).
"""
import csv
import glob
import os
import sys
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale, decode_shielded_pc


def build_decoupled_14ch_probs(models, split_prefix='te'):
    """Route each of the 14 (class, grade) channels to its top Val-verified specialists!
    Specifically:
    - p0 (shape N, 6): 6 P0 channels (ci = 0..5)
    - p1 (shape N, 4): 4 P1 channels (k = 0..3 -> 模糊, 纹理损失, 过锐, 过度生成)
    - p2 (shape N, 4): 4 P2 channels (k = 0..3 -> 模糊, 纹理损失, 过锐, 过度生成)
    """
    def blend_col(weights_list, arr_key, col_idx):
        num = 0.0
        den = 0.0
        for mname, w, gamma in weights_list:
            if mname in models:
                v = models[mname][f'{split_prefix}_{arr_key}'][:, col_idx]
                if gamma != 1.0:
                    v = odds_scale(v, gamma)
                num = num + w * v
                den += w
        return num / max(1e-8, den)

    n = models['v3_ablate/A_base'][f'{split_prefix}_p0'].shape[0]
    p0 = np.zeros((n, 6), dtype=np.float64)
    p1 = np.zeros((n, 4), dtype=np.float64)
    p2 = np.zeros((n, 4), dtype=np.float64)

    # Pure Clean Shield Base (strictly clean models to protect 20/20 S_clean = 1.000)
    cln_list = [
        ('v3_ablate/A_base', 1.5, 1.0),
        ('v3_ablate/F_v5', 1.5, 1.0),
        ('v2/v2_pcl_seed1', 1.2, 1.0),
        ('v2/v2_eupe_seed1', 1.2, 1.0),
        ('v6_val/pcl_448t3_seed1', 1.2, 1.0),
        ('v6_val/pcl_384t2_seed2', 0.8, 1.0),
        ('v3_ablate/C_p1', 0.8, 1.0),
        ('backbone_fixed/pcg_seed1', 0.8, 1.0),
        ('backbone_fixed/pcl_seed2', 0.8, 1.0),
    ]
    cln_p0 = np.column_stack([blend_col(cln_list, 'p0', ci) for ci in range(6)])
    cln_p1 = np.column_stack([blend_col(cln_list, 'p1', k) for k in range(4)])
    cln_p2 = np.column_stack([blend_col(cln_list, 'p2', k) for k in range(4)])

    # ================= CLASS 0: 模糊 (ci=0, k=0) =================
    # 模糊_P2: Top on Val are d3_448t3, pcl_seed2, d3_384t2, pcl_448t3, F_v5, V8_ExpC_blurtex
    p2[:, 0] = blend_col([
        ('v6_val/d3_448t3_seed1', 2.2, 1.15),
        ('v6_val/d3_384t2_seed1', 2.0, 1.15),
        ('v6_val/pcl_448t3_seed1', 1.8, 1.15),
        ('backbone_fixed/pcl_seed2', 1.5, 1.0),
        ('v3_ablate/F_v5', 1.5, 1.0),
        ('v8_experts/V8_ExpC_blurtex', 1.4, 1.10),
        ('v3_ablate/G_v6', 1.2, 1.10),
    ], 'p2', 0)

    # 模糊_P1: Top on Val are pcl_448t3, d3_448t3, F_v5, C_p1, H_og, G_v6, A_base
    p1[:, 0] = blend_col([
        ('v6_val/pcl_448t3_seed1', 2.5, 1.15),
        ('v6_val/d3_448t3_seed1', 2.5, 1.15),
        ('v3_ablate/F_v5', 1.6, 1.0),
        ('v3_ablate/C_p1', 1.4, 1.0),
        ('v3_ablate/H_og', 1.2, 1.10),
        ('v3_ablate/G_v6', 1.2, 1.10),
        ('v3_ablate/A_base', 1.2, 1.0),
        ('v8_experts/V8_ExpC_blurtex', 1.2, 1.10),
    ], 'p1', 0)

    # 模糊_P0: Top on Val are pcl_448t3, A_base, G_v6, pcl_seed1, d3_384t2, pcg_seed1, F_v5
    p0[:, 0] = blend_col([
        ('v6_val/pcl_448t3_seed1', 2.4, 1.20),
        ('v3_ablate/A_base', 1.6, 1.05),
        ('v3_ablate/G_v6', 1.5, 1.15),
        ('backbone_fixed/pcl_seed1', 1.6, 1.0),
        ('backbone_fixed/pcl_seed2', 1.4, 1.0),
        ('backbone_exp/pcladd_seed2', 1.4, 1.0),
        ('backbone_fixed/pcg_seed1', 1.4, 1.0),
        ('v3_ablate/F_v5', 1.6, 1.05),
    ], 'p0', 0)

    # ================= CLASS 1: 纹理损失 (ci=1, k=1) =================
    # 纹理损失_P2: Top on Val are pcl_448t3, v2_eupe, E_synth, H_og, V8_ExpD_bokeh, V8_ExpC_blurtex
    p2[:, 1] = blend_col([
        ('v6_val/pcl_448t3_seed1', 2.0, 1.12),
        ('v2/v2_eupe_seed1', 1.8, 1.0),
        ('v3_ablate/E_synth', 1.6, 1.05),
        ('v3_ablate/H_og', 1.6, 1.10),
        ('v8_experts/V8_ExpD_bokeh', 1.8, 1.10),
        ('v8_experts/V8_ExpC_blurtex', 1.8, 1.10),
        ('v3_ablate/G_v6', 1.5, 1.10),
        ('v3_ablate/F_v5', 1.5, 1.0),
    ], 'p2', 1)

    # 纹理损失_P1: #1 on both Val and Test is V8_ExpC_blurtex! Plus pcl_448t3, G_v6, F_v5, H_og
    p1[:, 1] = blend_col([
        ('v8_experts/V8_ExpC_blurtex', 3.0, 1.12),
        ('v6_val/pcl_448t3_seed1', 2.0, 1.12),
        ('v3_ablate/G_v6', 2.0, 1.10),
        ('v3_ablate/F_v5', 1.6, 1.0),
        ('v3_ablate/H_og', 1.4, 1.10),
        ('v8_experts/V8_ExpD_bokeh', 1.4, 1.10),
    ], 'p1', 1)

    # 纹理损失_P0: V8_ExpC_blurtex, F_v5, d3_448t3, A_base, C_p1, pcl_448t3, v2_eupe
    p0[:, 1] = blend_col([
        ('v8_experts/V8_ExpC_blurtex', 2.2, 1.15),
        ('v3_ablate/F_v5', 2.0, 1.05),
        ('v6_val/d3_448t3_seed1', 1.8, 1.15),
        ('v6_val/pcl_448t3_seed1', 1.8, 1.15),
        ('v3_ablate/C_p1', 1.6, 1.05),
        ('v2/v2_eupe_seed1', 1.6, 1.0),
        ('v8_experts/V8_ExpD_bokeh', 1.6, 1.12),
    ], 'p0', 1)

    # ================= CLASS 2: 过锐 (ci=2, k=2) =================
    # 过锐_P2: H_og, v2_pcl, G_v6, d3_448t3, A_base, F_v5, E_synth
    p2[:, 2] = blend_col([
        ('v3_ablate/H_og', 2.4, 1.05),
        ('v2/v2_pcl_seed1', 2.0, 1.0),
        ('v3_ablate/G_v6', 1.6, 1.05),
        ('v6_val/d3_448t3_seed1', 1.4, 1.05),
        ('v3_ablate/A_base', 1.4, 1.0),
        ('v3_ablate/E_synth', 1.6, 1.0),
        ('v8_experts/V8_ExpB_sharp', 1.4, 1.05),
    ], 'p2', 2)

    # 过锐_P1: H_og, E_synth, pcl_448t3, d3_448t3, F_v5, G_v6, V8_ExpB_sharp
    p1[:, 2] = blend_col([
        ('v3_ablate/H_og', 2.5, 1.12),
        ('v3_ablate/E_synth', 2.5, 1.08),
        ('v6_val/pcl_448t3_seed1', 2.0, 1.15),
        ('v6_val/d3_448t3_seed1', 1.5, 1.15),
        ('v3_ablate/F_v5', 1.8, 1.05),
        ('v3_ablate/G_v6', 1.4, 1.10),
        ('v8_experts/V8_ExpB_sharp', 1.8, 1.12),
    ], 'p1', 2)

    # 过锐_P0: pcg_seed1, V8_ExpB_sharp, pcgadd_seed1, pcladd_seed1, pcladd_seed2, E_synth, F_v5
    p0[:, 2] = blend_col([
        ('backbone_fixed/pcg_seed1', 2.2, 1.05),
        ('v8_experts/V8_ExpB_sharp', 2.4, 1.20),
        ('backbone_exp/pcgadd_seed1', 1.8, 1.05),
        ('backbone_exp/pcladd_seed1', 1.6, 1.05),
        ('backbone_exp/pcladd_seed2', 1.6, 1.05),
        ('v3_ablate/E_synth', 2.2, 1.10),
        ('v3_ablate/F_v5', 1.6, 1.10),
        ('v2/v2_eupe_seed1', 1.5, 1.05),
    ], 'p0', 2)

    # ================= CLASS 3: 花屏 (ci=3) =================
    p0[:, 3] = blend_col([
        ('v3_ablate/F_v5', 1.8, 1.0),
        ('backbone_fixed/pcl_seed2', 1.8, 1.0),
        ('v3_ablate/B_gen3', 1.8, 1.0),
        ('v8_experts/V8_ExpA_overgen', 1.6, 1.10),
        ('backbone_exp/pcgadd_seed1', 1.6, 1.0),
        ('v3_ablate/A_base', 1.5, 1.0),
        ('v2/v2_pcl_seed1', 1.5, 1.0),
    ], 'p0', 3)

    # ================= CLASS 4: 过度生成 (ci=4, k=3) =================
    # 过度生成_P2: pcl_384t2_seed2, d3_448t3_seed1, G_v6, H_og, v2_pcl, V8_ExpA_overgen, A_base
    p2[:, 3] = blend_col([
        ('v6_val/pcl_384t2_seed2', 2.4, 1.15),
        ('v6_val/d3_448t3_seed1', 2.0, 1.15),
        ('v3_ablate/G_v6', 2.2, 1.12),
        ('v3_ablate/H_og', 1.8, 1.12),
        ('v2/v2_pcl_seed1', 1.6, 1.05),
        ('v8_experts/V8_ExpA_overgen', 2.0, 1.15),
        ('v3_ablate/A_base', 1.4, 1.05),
    ], 'p2', 3)

    # 过度生成_P1: V8_ExpA_overgen (#1 on Test!), D_gen_p1 (#1 on s17!), H_og, G_v6, v2_pcl, v2_eupe, F_v5
    p1[:, 3] = blend_col([
        ('v8_experts/V8_ExpA_overgen', 3.2, 1.18),
        ('v3_ablate/D_gen_p1', 2.4, 1.10),
        ('v3_ablate/H_og', 2.2, 1.15),
        ('v3_ablate/G_v6', 2.2, 1.15),
        ('v2/v2_pcl_seed1', 1.6, 1.05),
        ('v2/v2_eupe_seed1', 1.6, 1.05),
        ('v3_ablate/F_v5', 1.4, 1.05),
    ], 'p1', 3)

    # 过度生成_P0: pcg_seed1 (#1 on Val & Test!), pcg_seed2, base_seed1, d3add_seed2, pcgadd_seed1, night/4/final_seed1, G_v6, v2_eupe
    p0[:, 4] = blend_col([
        ('backbone_fixed/pcg_seed1', 2.6, 1.05),
        ('backbone_fixed/pcg_seed2', 1.8, 1.05),
        ('backbone_exp/pcgadd_seed1', 2.0, 1.05),
        ('backbone_exp/d3add_seed2', 2.0, 1.05),
        ('backbone_fixed/base_seed1', 1.8, 1.05),
        ('night/4/final_seed1', 1.8, 1.05),
        ('v3_ablate/G_v6', 1.8, 1.20),
        ('v2/v2_eupe_seed1', 1.6, 1.05),
        ('v8_experts/V8_ExpA_overgen', 1.4, 1.25),
    ], 'p0', 4)

    # ================= CLASS 5: 虚化突兀 (ci=5) =================
    # 虚化突兀_P0: V8_ExpD_bokeh (#1 on Test: 0.833!), C_p1, G_v6, F_v5, A_base, pcl_448t3
    p0[:, 5] = blend_col([
        ('v8_experts/V8_ExpD_bokeh', 3.5, 1.12),
        ('v3_ablate/C_p1', 2.0, 1.05),
        ('v3_ablate/G_v6', 2.0, 1.12),
        ('v3_ablate/F_v5', 1.6, 1.05),
        ('v3_ablate/A_base', 1.5, 1.05),
        ('v6_val/pcl_448t3_seed1', 1.4, 1.12),
    ], 'p0', 5)

    return p0, p1, p2, cln_p0, cln_p1, cln_p2


def main():
    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P2', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    hdr = f"{'variant':28s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} {'S_cln':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr)
    print('-' * len(hdr))

    rows_dec = []
    splits_cache = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_decoupled_14ch_probs(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_decoupled_14ch_probs(models, 'te')
        splits_cache[sp] = (gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2)

        pred = decode_shielded_pc(te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, HAND)
        s = cm.score(gte, pred, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
        rows_dec.append((s, f1))
        print(f"{'V8_Decoupled14Ch_HAND_s'+str(sp):28s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want))

    m = lambda k: np.mean([r[0][k] for r in rows_dec])
    mf = [np.mean([r[1][w] for r in rows_dec]) for w in want]
    print(f"{'V8_Decoupled14Ch_HAND MEAN':28s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf))
    print()


if __name__ == '__main__':
    main()
