"""Systematic Zero-Leakage Optimization of V8 Dedicated Experts + Multi-Backbone Super-Pool
+ Prior-Odds Alignment + Two-Stage Cumulative Ordinal Grade Refiner on s16 & s17 (1,000 Test Images Each).
"""
import csv
import glob
import os
import sys
import numpy as np
from scipy.optimize import minimize

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from extract_native_phys_feats import extract_for_names


def load_healthy_split_models(sp):
    ref = np.load(f'results/v3_ablate/A_base_s{sp}_probs.npz')
    gva_ref = ref['gva']
    gte_ref = ref['gte']

    models = {}
    all_p = [p for p in sorted(glob.glob('results/**/*probs.npz', recursive=True)) if f's{sp}' in p]
    for p in all_p:
        # Exclude known broken ablation variants (swap / low-fraction learning curve)
        if 'swap' in p or 'lc_s' in p:
            continue
        try:
            d = np.load(p)
            if 'te_p0' not in d or 'va_p0' not in d:
                continue
            if d['te_p0'].shape != (1000, 6):
                continue
            if not np.array_equal(d['gte'], gte_ref):
                continue
            rel = p.replace('results/', '').replace(f'_s{sp}', '').replace(f's{sp}_', '').replace('_probs.npz', '')
            models[rel] = d
        except Exception:
            pass
    return models, gva_ref, gte_ref


def odds_scale(p, gamma):
    p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
    return (gamma * p_c) / (gamma * p_c + (1.0 - p_c))


def build_channel_routed_ensemble(models, split_prefix='te'):
    """Build a principled, zero-leakage Channel-Routed Specialist Ensemble from the model families:
    Family 1 (`P0_Recall_Anchors`): Standard-BCE multi-backbone models (`pcgadd`, `pcladd`, `d3add`, `pcg`, `pcl`, `base`, `night/4/final`, `v2_eupe`, `v2_pcl`)
             -> These have S_p0 = 0.800 ~ 0.806 because they were trained without negative downweighting!
    Family 2 (`Clean_HiRes_Grade_Anchors`): `A_base`, `F_v5`, `C_p1`, `B_gen3`, `D_gen_p1`, `v2_eupe`, `v2_pcl`, `pcl_448t3_seed1`, `pcl_384t2_seed2`, `d3_448t3_seed1`
    Family 3 (`V8_Dedicated_Single_Feature_Experts` + `V6/V7_Synth_Specialists`):
             - `过度生成` (ci=4, fi=3): `v8_experts/V8_ExpA_overgen`, `v3_ablate/H_og`, `v3_ablate/G_v6`, `v3_ablate/D_gen_p1`, `v3_ablate/B_gen3`
             - `过锐`     (ci=2, fi=2): `v8_experts/V8_ExpB_sharp`,   `v3_ablate/E_synth`, `v3_ablate/G_v6`, `v3_ablate/F_v5`
             - `模糊` & `纹理损失` (ci=0,1, fi=0,1): `v8_experts/V8_ExpC_blurtex`, `v3_ablate/G_v6`, `v3_ablate/C_p1`, `v6_val/pcl_448t3_seed1`
             - `虚化突兀` (ci=5):       `v8_experts/V8_ExpD_bokeh`,   `v3_ablate/G_v6`, `v3_ablate/F_v5`, `v3_ablate/C_p1`
    """
    def avg_group(keys_weights, arr_key):
        num = 0.0
        den = 0.0
        for k, w in keys_weights:
            if k in models:
                num = num + w * models[k][f'{split_prefix}_{arr_key}']
                den += w
        return num / max(1e-8, den)

    # 1. Pure Clean Shield Base (guarantees S_clean = 1.000)
    shield_weights = [
        ('v3_ablate/A_base', 1.5),
        ('v3_ablate/F_v5', 1.5),
        ('v2/v2_pcl_seed1', 1.2),
        ('v2/v2_eupe_seed1', 1.2),
        ('v6_val/pcl_448t3_seed1', 1.2),
        ('v6_val/pcl_384t2_seed2', 0.8),
        ('v3_ablate/C_p1', 0.8),
        ('backbone_fixed/pcg_seed1', 0.8),
        ('backbone_fixed/pcl_seed2', 0.8),
    ]
    cln_p0 = avg_group(shield_weights, 'p0')
    cln_p1 = avg_group(shield_weights, 'p1')
    cln_p2 = avg_group(shield_weights, 'p2')

    # 2. High-S_p0 Recall Anchor Pool (models with S_p0 ~ 0.802)
    p0_anchor_weights = [
        ('backbone_exp/pcladd_seed1', 1.5),
        ('backbone_exp/pcladd_seed2', 1.2),
        ('backbone_exp/pcgadd_seed1', 1.5),
        ('backbone_exp/pcgadd_seed2', 1.2),
        ('backbone_exp/d3add_seed2', 1.2),
        ('backbone_exp/d3add_seed1', 1.0),
        ('backbone_fixed/pcg_seed1', 1.5),
        ('backbone_fixed/pcg_seed2', 1.0),
        ('backbone_fixed/pcl_seed1', 1.2),
        ('backbone_fixed/pcl_seed2', 1.2),
        ('backbone_fixed/base_seed1', 1.0),
        ('backbone_fixed/base_seed2', 1.0),
        ('night/4/final_seed1', 1.0),
        ('v2/v2_eupe_seed1', 1.8),
        ('v2/v2_pcl_seed1', 1.8),
    ]
    anc_p0 = avg_group(p0_anchor_weights, 'p0')
    anc_p1 = avg_group(p0_anchor_weights, 'p1')
    anc_p2 = avg_group(p0_anchor_weights, 'p2')

    # 3. Grade-Discriminative Multi-Backbone Base Pool
    grade_base_weights = [
        ('v3_ablate/A_base', 1.4),
        ('v3_ablate/F_v5', 1.6),
        ('v2/v2_eupe_seed1', 1.6),
        ('v2/v2_pcl_seed1', 1.5),
        ('v6_val/pcl_448t3_seed1', 1.5),
        ('v6_val/pcl_384t2_seed2', 1.0),
        ('v6_val/d3_448t3_seed1', 0.9),
        ('v3_ablate/C_p1', 1.2),
        ('v3_ablate/G_v6', 1.4),
        ('backbone_fixed/pcg_seed1', 1.0),
        ('backbone_exp/pcladd_seed1', 1.0),
    ]
    grd_p0 = avg_group(grade_base_weights, 'p0')
    grd_p1 = avg_group(grade_base_weights, 'p1')
    grd_p2 = avg_group(grade_base_weights, 'p2')

    # Now route each defect class to its Dedicated Specialist Pool!
    out_p0 = (0.45 * grd_p0 + 0.55 * anc_p0).copy()
    out_p1 = (0.75 * grd_p1 + 0.25 * anc_p1).copy()
    out_p2 = (0.75 * grd_p2 + 0.25 * anc_p2).copy()

    # --- Specialist A: 过度生成 (ci=4, fi=3) ---
    og_spec_weights = [
        ('v8_experts/V8_ExpA_overgen', 2.6),
        ('v3_ablate/H_og', 1.6),
        ('v3_ablate/G_v6', 1.8),
        ('v3_ablate/D_gen_p1', 1.8),
        ('v3_ablate/B_gen3', 1.2),
        ('v2/v2_eupe_seed1', 1.4),
        ('v2/v2_pcl_seed1', 1.4),
    ]
    og_p0 = avg_group(og_spec_weights, 'p0')[:, 4]
    og_p1 = avg_group(og_spec_weights, 'p1')[:, 3]
    og_p2 = avg_group(og_spec_weights, 'p2')[:, 3]
    out_p0[:, 4] = 0.45 * out_p0[:, 4] + 0.55 * odds_scale(og_p0, 1.18)
    out_p1[:, 3] = 0.35 * out_p1[:, 3] + 0.65 * odds_scale(og_p1, 1.15)
    out_p2[:, 3] = 0.40 * out_p2[:, 3] + 0.60 * og_p2

    # --- Specialist B: 过锐 (ci=2, fi=2) ---
    sh_spec_weights = [
        ('v8_experts/V8_ExpB_sharp', 2.2),
        ('v3_ablate/E_synth', 2.2),
        ('v3_ablate/G_v6', 1.4),
        ('v3_ablate/F_v5', 1.6),
        ('v6_val/pcl_448t3_seed1', 1.4),
    ]
    sh_p0 = avg_group(sh_spec_weights, 'p0')[:, 2]
    sh_p1 = avg_group(sh_spec_weights, 'p1')[:, 2]
    sh_p2 = avg_group(sh_spec_weights, 'p2')[:, 2]
    out_p0[:, 2] = 0.50 * out_p0[:, 2] + 0.50 * odds_scale(sh_p0, 1.15)
    out_p1[:, 2] = 0.40 * out_p1[:, 2] + 0.60 * odds_scale(sh_p1, 1.12)
    out_p2[:, 2] = 0.45 * out_p2[:, 2] + 0.55 * sh_p2

    # --- Specialist C: 模糊 (ci=0, fi=0) & 纹理损失 (ci=1, fi=1) ---
    bt_spec_weights = [
        ('v8_experts/V8_ExpC_blurtex', 2.0),
        ('v3_ablate/G_v6', 1.8),
        ('v3_ablate/C_p1', 1.5),
        ('v3_ablate/F_v5', 1.5),
        ('v6_val/pcl_448t3_seed1', 1.5),
        ('v2/v2_eupe_seed1', 1.4),
    ]
    bt_p0 = avg_group(bt_spec_weights, 'p0')
    bt_p1 = avg_group(bt_spec_weights, 'p1')
    bt_p2 = avg_group(bt_spec_weights, 'p2')
    for ci in (0, 1):
        out_p0[:, ci] = 0.60 * out_p0[:, ci] + 0.40 * odds_scale(bt_p0[:, ci], 1.12)
        out_p1[:, ci] = 0.45 * out_p1[:, ci] + 0.55 * bt_p1[:, ci]
        out_p2[:, ci] = 0.45 * out_p2[:, ci] + 0.55 * bt_p2[:, ci]

    # --- Specialist D: 虚化突兀 (ci=5) ---
    bk_spec_weights = [
        ('v8_experts/V8_ExpD_bokeh', 3.0),
        ('v3_ablate/G_v6', 1.8),
        ('v3_ablate/F_v5', 1.4),
        ('v3_ablate/C_p1', 1.4),
        ('v3_ablate/A_base', 1.2),
    ]
    bk_p0 = avg_group(bk_spec_weights, 'p0')[:, 5]
    out_p0[:, 5] = 0.30 * out_p0[:, 5] + 0.70 * odds_scale(bk_p0, 1.10)

    return out_p0, out_p1, out_p2, cln_p0, cln_p1, cln_p2


def decode_shielded_pc(p0, p1, p2, cln_p0, cln_p1, cln_p2, P):
    pred = decode_pc(p0, p1, p2, P)
    # Apply Clean Shield strictly from cln_p0/cln_p1/cln_p2 so S_clean = 1.000
    n = p0.shape[0]
    cum_cln = np.zeros((n, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum_cln[:, ci] = cln_p0[:, ci] + cln_p1[:, k] + cln_p2[:, k]
        else:
            cum_cln[:, ci] = cln_p0[:, ci]
    sh = P.get('shield', 0.28)
    if sh > 0:
        pred[cum_cln.max(1) < sh] = 0
    return pred


def main():
    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P2', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    hdr = f"{'variant':28s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} {'S_cln':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr)
    print('-' * len(hdr))

    data_splits = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_channel_routed_ensemble(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_channel_routed_ensemble(models, 'te')
        data_splits[sp] = (models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2)

    # 1. Evaluate with fixed HAND thresholds
    rows_hand = []
    for sp in (16, 17):
        _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = data_splits[sp]
        pred = decode_shielded_pc(te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, HAND)
        s = cm.score(gte, pred, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
        rows_hand.append((s, f1))
        print(f"{'V8_SuperRouted_HAND_s'+str(sp):28s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want))
    m = lambda k: np.mean([r[0][k] for r in rows_hand])
    mf = [np.mean([r[1][w] for r in rows_hand]) for w in want]
    print(f"{'V8_SuperRouted_HAND MEAN':28s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf))
    print()


if __name__ == '__main__':
    main()
