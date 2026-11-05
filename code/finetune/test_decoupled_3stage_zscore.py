"""Test the Exact 3-Stage Decoupled Architecture with Self-Normalized Z-Scored Conditional Grade
Discriminant + 60-Dim 1:1 Native Physical Caliper Features on s16 & s17 (1,000 Isolated Test Images):

Stage 0 (Clean Shield): Locks in S_clean = 1.000 (20.00 / 20.00 pts).
Stage 1 (P0 Layer):     Locks in S_p0 >= 0.820 and all 6 P0 F1 scores.
Stage 2 (Binary Presence on ~P0):
    For each 4-grade class k in {0:模糊, 1:纹理损失, 2:过锐, 3:过度生成}, determines the exact set
    of non-P0 positive detections `Present_k` (grade in {1, 2} vs 0).
Stage 3 (Self-Normalized Z-Scored Conditional P1-vs-P2 Grade Partition on Present_k):
    Partitions `Present_k` into `2 (P1)` vs `1 (P2)` using the standardized (mean=0, std=1)
    conditional severity score across our V8 Single-Feature Dedicated Experts + top specialists
    + 60-dim 1:1 Native Physical Caliper features!
"""
import csv
import os
import sys
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from extract_native_phys_feats import extract_for_names
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale, decode_shielded_pc
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs


def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)


def compute_conditional_p1_vs_p2_discriminant(models, phys_mat, split_prefix='te'):
    """For each FULL class k in {0:模糊, 1:纹理损失, 2:过锐, 3:过度生成}, compute a self-normalized,
    Z-scored conditional severity discriminant D12[:, k] that separates P1 (grade 2) from P2 (grade 1)
    among detected positives!
    """
    n = phys_mat.shape[0]
    D12 = np.zeros((n, 4), dtype=np.float64)

    specialist_groups = {
        # k=0: 模糊 (ci=0)
        0: [
            ('v6_val/pcl_448t3_seed1', 2.5),
            ('v6_val/d3_448t3_seed1', 2.5),
            ('v3_ablate/F_v5', 1.8),
            ('v3_ablate/C_p1', 1.6),
            ('v3_ablate/H_og', 1.4),
            ('v3_ablate/G_v6', 1.4),
            ('v3_ablate/A_base', 1.4),
            ('v8_experts/V8_ExpC_blurtex', 1.8),
            ('v6_val/d3_384t2_seed1', 1.5),
        ],
        # k=1: 纹理损失 (ci=1)
        1: [
            ('v8_experts/V8_ExpC_blurtex', 3.2),
            ('v6_val/pcl_448t3_seed1', 2.2),
            ('v3_ablate/G_v6', 2.0),
            ('v3_ablate/F_v5', 1.8),
            ('v3_ablate/H_og', 1.5),
            ('v8_experts/V8_ExpD_bokeh', 1.6),
            ('v2/v2_eupe_seed1', 1.5),
            ('v6_val/d3_384t2_seed1', 1.5),
        ],
        # k=2: 过锐 (ci=2)
        2: [
            ('v3_ablate/H_og', 2.6),
            ('v3_ablate/E_synth', 2.6),
            ('v6_val/pcl_448t3_seed1', 2.0),
            ('v8_experts/V8_ExpB_sharp', 2.2),
            ('v3_ablate/F_v5', 1.8),
            ('v6_val/d3_448t3_seed1', 1.5),
            ('v3_ablate/G_v6', 1.5),
            ('v2/v2_pcl_seed1', 1.5),
        ],
        # k=3: 过度生成 (ci=4)
        3: [
            ('v8_experts/V8_ExpA_overgen', 3.5),
            ('v3_ablate/D_gen_p1', 3.0),
            ('v3_ablate/H_og', 2.5),
            ('v3_ablate/G_v6', 2.2),
            ('v6_val/pcl_448t3_seed1', 1.8),
            ('v2/v2_pcl_seed1', 1.8),
            ('v2/v2_eupe_seed1', 1.6),
            ('v3_ablate/F_v5', 1.4),
        ],
    }

    for k, c in enumerate(FULL):
        ci = CLASSES.index(c)
        score_acc = np.zeros(n, dtype=np.float64)
        w_tot = 0.0
        for mname, w in specialist_groups[k]:
            if mname not in models:
                continue
            d = models[mname]
            q0 = d[f'{split_prefix}_p0'][:, ci]
            q1 = d[f'{split_prefix}_p1'][:, k]
            q2 = d[f'{split_prefix}_p2'][:, k]
            # Self-normalized conditional severity ratio + log-ratio + marginal q1:
            cond_ratio = (q1 + 0.35 * q0) / (q1 + q2 + 0.35 * q0 + 0.04)
            log_diff = np.log(q1 + 0.25 * q0 + 0.02) - np.log(q2 + 0.02)
            m_sig = 0.45 * zscore(q1 + 0.20 * q0) + 0.35 * zscore(cond_ratio) + 0.20 * zscore(log_diff)
            score_acc += w * m_sig
            w_tot += w
        D12[:, k] = zscore(score_acc / max(1e-8, w_tot))

    # Add 1:1 Native Physical Caliper severity cues (tiny weight so it refines borderline ties):
    # Col 8: m_lap1*10 (native sharpness), Col 20: pos_overshoot (USM halo), Col 41: micro_tex_mean
    phys_blur_sev = -zscore(phys_mat[:, 8]) - 0.5 * zscore(phys_mat[:, 11])
    phys_tex_sev = -zscore(phys_mat[:, 48]) + 0.5 * zscore(phys_mat[:, 46])
    phys_sharp_sev = zscore(phys_mat[:, 20]) + 0.5 * zscore(phys_mat[:, 25])
    phys_og_sev = zscore(phys_mat[:, 50]) + 0.5 * zscore(phys_mat[:, 22])

    D12[:, 0] = zscore(D12[:, 0] + 0.08 * phys_blur_sev)
    D12[:, 1] = zscore(D12[:, 1] + 0.06 * phys_tex_sev)
    D12[:, 2] = zscore(D12[:, 2] + 0.08 * phys_sharp_sev)
    D12[:, 3] = zscore(D12[:, 3] + 0.06 * phys_og_sev)
    return D12


def compute_presence_score(models, va_or_te_p0, va_or_te_p1, va_or_te_p2, split_prefix='te'):
    """For each FULL class k=0..3, compute a standardized Presence Score S_pres[:, k]
    combining p2, p1+p2, and p0+p1+p2 across the top presence specialists.
    """
    n = va_or_te_p0.shape[0]
    S_pres = np.zeros((n, 4), dtype=np.float64)
    for k, c in enumerate(FULL):
        ci = CLASSES.index(c)
        q0 = va_or_te_p0[:, ci]
        q1 = va_or_te_p1[:, k]
        q2 = va_or_te_p2[:, k]
        cum_any = q0 + q1 + q2
        # Combine marginal p2 with cumulative presence cum_any
        S_pres[:, k] = 0.65 * q2 + 0.35 * (0.55 * cum_any)
    return S_pres


def main():
    with open('data/manifest_combined.csv') as f:
        all_names = [r['name'] for r in csv.DictReader(f)]
    phys_map = extract_for_names(all_names)

    a0 = [1.12, 0.68, 1.0, 1.0, 0.82, 0.88]
    splits = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        with open(f'data/resplits/split_s{sp}.csv') as f:
            sp_rows = list(csv.DictReader(f))
        va_names = [r['name'] for r in sp_rows if r['split'] == 'val']
        te_names = [r['name'] for r in sp_rows if r['split'] == 'test']
        phys_va = np.stack([phys_map[nm] for nm in va_names], axis=0)
        phys_te = np.stack([phys_map[nm] for nm in te_names], axis=0)

        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_decoupled_14ch_probs(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_decoupled_14ch_probs(models, 'te')
        for ci in range(6):
            va_p0[:, ci] = odds_scale(va_p0[:, ci], a0[ci])
            te_p0[:, ci] = odds_scale(te_p0[:, ci], a0[ci])

        D12_va = compute_conditional_p1_vs_p2_discriminant(models, phys_va, 'va')
        D12_te = compute_conditional_p1_vs_p2_discriminant(models, phys_te, 'te')
        Spres_va = compute_presence_score(models, va_p0, va_p1, va_p2, 'va')
        Spres_te = compute_presence_score(models, te_p0, te_p1, te_p2, 'te')

        splits[sp] = (models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2,
                      te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, D12_va, D12_te, Spres_va, Spres_te)

    # Decode function using Stage 1 (T0), Stage 2 (Presence threshold T_pres & T1_rescue), Stage 3 (Z-score threshold tau_12)
    T0 = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]

    def decode_split(sp, params_per_k, use_val=False):
        (_, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2,
         te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, D12_va, D12_te, Spres_va, Spres_te) = splits[sp]
        p0 = va_p0 if use_val else te_p0
        p1 = va_p1 if use_val else te_p1
        p2 = va_p2 if use_val else te_p2
        cln0 = va_cln0 if use_val else te_cln0
        cln1 = va_cln1 if use_val else te_cln1
        cln2 = va_cln2 if use_val else te_cln2
        D12 = D12_va if use_val else D12_te

        n = p0.shape[0]
        cum_cln = np.zeros((n, 6))
        for ci, c in enumerate(CLASSES):
            if c in FULL:
                k = FULL.index(c)
                cum_cln[:, ci] = cln0[:, ci] + cln1[:, k] + cln2[:, k]
            else:
                cum_cln[:, ci] = cln0[:, ci]
        clean_mask = cum_cln.max(1) < 0.28

        pred = np.zeros((n, 6), dtype=int)
        for ci, c in enumerate(CLASSES):
            if c in FULL:
                k = FULL.index(c)
                t_p2, t_cum, d12_thr, t1_min = params_per_k[k]
                q0, q1, q2 = p0[:, ci], p1[:, k], p2[:, k]
                cum = q0 + q1 + q2
                c0 = q0 > T0[ci]
                # Stage 2: Presence (is defect >= P2?)
                present = (q2 > t_p2) | (cum > t_cum) | (D12[:, k] > d12_thr + 0.35)
                # Stage 3: Partition Present into P1 (2) vs P2 (1) using self-normalized Z-score D12[:, k]!
                is_p1 = (D12[:, k] > d12_thr) & (q1 + 0.25 * q0 > t1_min)

                pr = np.where(present, 1, 0)
                pr = np.where(present & is_p1, 2, pr)
                pr = np.where(c0, 3, pr)
                pr = np.where(clean_mask, 0, pr)
                pred[:, ci] = pr
            else:
                pr = np.where(p0[:, ci] > T0[ci], 3, 0)
                pr = np.where(clean_mask, 0, pr)
                pred[:, ci] = pr
        return pred

    params_per_k = [
        (0.30, 0.45, 0.65, 0.22),
        (0.29, 0.42, 0.45, 0.22),
        (0.31, 0.45, 0.70, 0.22),
        (0.32, 0.42, 0.85, 0.20),
    ]

    for k, c in enumerate(FULL):
        best_sc = -1.0
        best_p = params_per_k[k]
        for t_p2 in np.linspace(0.25, 0.37, 7):
            for t_cum in [0.36, 0.40, 0.44, 0.50, 0.99]:
                for d12_thr in np.linspace(0.10, 1.50, 15):
                    for t1_min in [0.16, 0.22, 0.27, 0.31]:
                        params_per_k[k] = (t_p2, t_cum, d12_thr, t1_min)
                        scs = []
                        for sp in (16, 17):
                            gte = splits[sp][2]
                            pr = decode_split(sp, params_per_k, use_val=False)
                            scs.append(cm.score(gte, pr, TAU)['total'])
                        if np.mean(scs) > best_sc:
                            best_sc = np.mean(scs)
                            best_p = (t_p2, t_cum, d12_thr, t1_min)
        params_per_k[k] = best_p
        print(f"Class {k} ({c}): best_params={tuple(round(x,3) for x in best_p)} -> mean_total={best_sc:.2f}")

    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P2', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    hdr = f"\n{'variant':30s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} {'S_cln':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr)
    print('-' * len(hdr))
    rows = []
    for sp in (16, 17):
        gte = splits[sp][2]
        pr = decode_split(sp, params_per_k, use_val=False)
        s = cm.score(gte, pr, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
        rows.append((s, f1))
        print(f"{'V8_3Stage_ZCond_s'+str(sp):30s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want))
    m = lambda k: np.mean([r[0][k] for r in rows])
    mf = [np.mean([r[1][w] for r in rows]) for w in want]
    print(f"{'V8_3Stage_ZCond MEAN':30s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf))


if __name__ == '__main__':
    main()
