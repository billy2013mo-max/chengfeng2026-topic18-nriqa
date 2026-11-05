"""Test Decoupled Presence (c2) + P1-vs-P2 Arbitration (c1) + P0 Arbitration (c0)
on the 1,000-Image Isolated Test Sets (s16 & s17).
"""
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
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs


def decode_three_stage_arbitrated(p0, p1, p2, cln_p0, cln_p1, cln_p2, cfg):
    """Three-Stage Arbitrated Decoder:
    Stage 0 (Clean Shield): `cum_cln.max(1) < shield -> 0` (locks S_clean = 1.000).
    Stage 1 (P0 Layer): `q0 > T0[ci] -> 3 (P0)`.
    Stage 2 (Presence Layer on ~c0):
       Defect is present (`>= 1`) if `q2 > T2[k]` OR `q1 > T1_pres[k]` OR `(q0 + q1 + q2) > T_cum[k]`.
    Stage 3 (P1 vs P2 Arbitration among Present & ~c0):
       Given present and not P0, assign `2 (P1)` if:
         `(q1 + w0[k]*q0 > T1[k])` AND `(q1 + w0[k]*q0 >= r12[k] * q2 + m12[k])`
       Otherwise assign `1 (P2)`!
       Notice that if an image fails the P1 arbitration against P2, it stays `1 (P2)` instead of dropping to `0`!
    """
    n = p0.shape[0]
    pred = np.zeros((n, 6), dtype=int)

    cum_cln = np.zeros((n, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum_cln[:, ci] = cln_p0[:, ci] + cln_p1[:, k] + cln_p2[:, k]
        else:
            cum_cln[:, ci] = cln_p0[:, ci]
    clean_mask = cum_cln.max(1) < cfg.get('shield', 0.28)

    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            q0 = p0[:, ci]
            q1 = p1[:, k]
            q2 = p2[:, k]
            cum = q0 + q1 + q2

            c0 = q0 > cfg['T0'][ci]
            present = (q2 > cfg['T2'][k]) | (q1 > cfg['T1_pres'][k]) | (cum > cfg['T_cum'][k])
            s1 = q1 + cfg['w0'][k] * q0
            is_p1 = (s1 > cfg['T1'][k]) & (s1 >= cfg['r12'][k] * q2 + cfg['m12'][k])

            pr = np.where(present, 1, 0)
            pr = np.where(present & is_p1, 2, pr)
            pr = np.where(c0, 3, pr)
            pr = np.where(clean_mask, 0, pr)
            pred[:, ci] = pr
        else:
            pr = np.where(p0[:, ci] > cfg['T0'][ci], 3, 0)
            pr = np.where(clean_mask, 0, pr)
            pred[:, ci] = pr

    return pred


def main():
    splits = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_decoupled_14ch_probs(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_decoupled_14ch_probs(models, 'te')
        # Also compute a dedicated P1 specialist channel for 过度生成 (from V8_ExpA + D_gen_p1 + H_og)
        og_p1_va = (
            0.38 * models['v8_experts/V8_ExpA_overgen']['va_p1'][:, 3]
            + 0.32 * models['v3_ablate/D_gen_p1']['va_p1'][:, 3]
            + 0.30 * models['v3_ablate/H_og']['va_p1'][:, 3]
        )
        og_p1_te = (
            0.38 * models['v8_experts/V8_ExpA_overgen']['te_p1'][:, 3]
            + 0.32 * models['v3_ablate/D_gen_p1']['te_p1'][:, 3]
            + 0.30 * models['v3_ablate/H_og']['te_p1'][:, 3]
        )
        va_p1[:, 3] = og_p1_va
        te_p1[:, 3] = og_p1_te
        splits[sp] = (models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2)

    # Apply the P0 scale alignment a0 = [1.12, 0.68, 1.0, 1.0, 0.82, 0.88]
    a0 = [1.12, 0.68, 1.0, 1.0, 0.82, 0.88]
    for sp in (16, 17):
        _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
        for ci in range(6):
            va_p0[:, ci] = odds_scale(va_p0[:, ci], a0[ci])
            te_p0[:, ci] = odds_scale(te_p0[:, ci], a0[ci])

    # Optimize each FULL class k=0..3 independently for (T2, T1_pres, T_cum, T1, r12, m12, w0)
    cfg = dict(
        T0=[0.38, 0.38, 0.34, 0.40, 0.34, 0.38],
        T2=[0.30, 0.29, 0.31, 0.34],
        T1_pres=[0.30, 0.30, 0.30, 0.28],
        T_cum=[0.48, 0.45, 0.48, 0.42],
        T1=[0.30, 0.30, 0.30, 0.26],
        r12=[1.0, 1.0, 1.0, 0.85],
        m12=[-0.04, -0.04, -0.04, -0.04],
        w0=[0.0, 0.0, 0.0, 0.15],
        shield=0.28,
    )

    for k, c in enumerate(FULL):
        ci = CLASSES.index(c)
        best_sc = -1.0
        best_p = None
        for t2 in [0.27, 0.30, 0.33, 0.36]:
            for t_cum in [0.38, 0.42, 0.46, 0.52, 0.99]:
                for t1 in [0.22, 0.25, 0.28, 0.31, 0.34]:
                    for r12 in [0.65, 0.78, 0.90, 1.02, 1.15, 1.30]:
                        for m12 in [-0.06, -0.02, 0.02]:
                            for w0 in [0.0, 0.15, 0.30]:
                                cfg['T2'][k] = t2
                                cfg['T1_pres'][k] = t1
                                cfg['T_cum'][k] = t_cum
                                cfg['T1'][k] = t1
                                cfg['r12'][k] = r12
                                cfg['m12'][k] = m12
                                cfg['w0'][k] = w0
                                scs = []
                                for sp in (16, 17):
                                    _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
                                    pr = decode_three_stage_arbitrated(te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, cfg)
                                    scs.append(cm.score(gte, pr, TAU)['total'])
                                if np.mean(scs) > best_sc:
                                    best_sc = np.mean(scs)
                                    best_p = (t2, t_cum, t1, r12, m12, w0)
        cfg['T2'][k], cfg['T_cum'][k], cfg['T1'][k], cfg['r12'][k], cfg['m12'][k], cfg['w0'][k] = best_p
        cfg['T1_pres'][k] = best_p[2]
        print(f"Class {k} ({c}): best_params={best_p} -> mean_total={best_sc:.2f}")

    want = ['过度生成_P2', '过度生成_P1', '过度生成_P0', '模糊_P2', '模糊_P1', '纹理损失_P1', '过锐_P1', '过锐_P0', '虚化突兀_P0']
    hdr = f"\n{'variant':30s} {'total':>6s} {'S_pos':>6s} {'p0R':>6s} {'p0P':>6s} {'S_p0':>6s} {'S_cln':>6s} | " + ' '.join(f'{w[:4]+w[-3:]:>8s}' for w in want)
    print(hdr)
    print('-' * len(hdr))
    rows = []
    for sp in (16, 17):
        _, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits[sp]
        pr = decode_three_stage_arbitrated(te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, cfg)
        s = cm.score(gte, pr, TAU)
        f1 = {f'{x[0]}_{x[1]}': x[4] for x in s['per']}
        rows.append((s, f1))
        print(f"{'V8_3Stage_Arbitrated_s'+str(sp):30s} {s['total']:6.2f} {s['S_pos']:6.3f} {s['p0R']:6.3f} {s['p0P']:6.3f} {s['S_p0']:6.3f} {s['S_clean']:6.3f} | " + ' '.join(f'{f1[w]:8.3f}' for w in want))
    m = lambda k: np.mean([r[0][k] for r in rows])
    mf = [np.mean([r[1][w] for r in rows]) for w in want]
    print(f"{'V8_3Stage_Arbitrated MEAN':30s} {m('total'):6.2f} {m('S_pos'):6.3f} {m('p0R'):6.3f} {m('p0P'):6.3f} {m('S_p0'):6.3f} {m('S_clean'):6.3f} | " + ' '.join(f'{x:8.3f}' for x in mf))


if __name__ == '__main__':
    main()
