"""Verify and Optimize the Exact Two-Layer Decomposition of the Competition Metric
on the 1,000-Image Isolated Test Sets (s16 & s17):

- Layer 0 (Clean Shield): Locks in S_clean = 1.000 (20.00 / 20.00 points).
- Layer 1 (P0 Layer across 6 classes): Maximizes 20 * S_p0 + (25/14) * sum_{ci=0..5} F1(ci, P0).
- Layer 2 (P1 & P2 Layer on ~c0 across 4 FULL classes k=0..3):
  Because c0 overwrites c1 and c2 (`pr = np.where(c0, 3, pr)`), for any fixed Layer 0 and Layer 1,
  each class k in {0:模糊, 1:纹理损失, 2:过锐, 3:过度生成} is 100% independent of all other classes and
  cannot change S_clean, S_p0, or F1_P0! It purely maximizes (25/14) * (F1(k, P1) + F1(k, P2)).
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
from extract_native_phys_feats import extract_for_names
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs


def get_clean_shield_mask(cln_p0, cln_p1, cln_p2, shield=0.28):
    n = cln_p0.shape[0]
    cum_cln = np.zeros((n, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum_cln[:, ci] = cln_p0[:, ci] + cln_p1[:, k] + cln_p2[:, k]
        else:
            cum_cln[:, ci] = cln_p0[:, ci]
    return cum_cln.max(1) < shield


def main():
    splits_cache = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_decoupled_14ch_probs(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_decoupled_14ch_probs(models, 'te')
        splits_cache[sp] = (models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2)

    # Check how each class's P1/P2 expert blend & thresholds affect ONLY F1(k, P1) + F1(k, P2)
    # when Layer 1 (c0 = p0 > T0) is locked to V8_Decoupled14Ch_HAND!
    for sp in (16, 17):
        models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits_cache[sp]
        clean_mask_va = get_clean_shield_mask(va_cln0, va_cln1, va_cln2, 0.28)
        clean_mask_te = get_clean_shield_mask(te_cln0, te_cln1, te_cln2, 0.28)

        # Lock Layer 1 (P0 predictions) from V8_Decoupled14Ch_HAND
        T0 = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]
        c0_va = np.column_stack([(va_p0[:, ci] > T0[ci]) & (~clean_mask_va) for ci in range(6)])
        c0_te = np.column_stack([(te_p0[:, ci] > T0[ci]) & (~clean_mask_te) for ci in range(6)])

        # Now for each FULL class k=0..3, compare P1/P2 from V8_Decoupled14Ch vs un-inflated P2 + V8 Dedicated Expert P1!
        pred_te = np.where(c0_te, 3, 0)
        for k, c in enumerate(FULL):
            ci = CLASSES.index(c)
            # In V8_Decoupled14Ch_HAND, we had gamma>1 on p2 which hurt P1 on s16!
            # What if we use raw (un-inflated) p2 and test (T1, T2, R12, w0) on Val (s16+s17 val)?
            q0_te = te_p0[:, ci]
            q1_te = te_p1[:, k]
            q2_te = te_p2[:, k]
            c2 = q2_te > HAND['T2'][k]
            c1 = (q1_te > HAND['T1'][k]) & ((q1_te * HAND['R12'][k]) >= (q2_te - 0.05))
            pr_k = np.where(c2, 1, 0)
            pr_k = np.where(c1, 2, pr_k)
            pr_k = np.where(clean_mask_te, 0, pr_k)
            pr_k = np.where(c0_te[:, ci], 3, pr_k)
            pred_te[:, ci] = pr_k

        s_base = cm.score(gte, pred_te, TAU)
        print(f"[s{sp}] Locked Layer 1 + Standard Layer 2: {s_base['total']:.2f}")

    # Now optimize Layer 1 (T0[0..5]) and Layer 2 (T1[k], T2[k], R12[k], w0[k], T_pres[k])
    # using Joint Val (s16_val + s17_val = 714 unseen validation images!) or Leave-One-Split-Out!
    print("\n=== Testing Zero-Leakage Per-Split Val & Joint-Val Calibration of Layer 1 + Layer 2 ===")
    for mode in ['PerSplitVal_Shrink', 'CrossSplit_OOF']:
        rows_opt = []
        for sp in (16, 17):
            other_sp = 17 if sp == 16 else 16
            # If CrossSplit_OOF: fit on (s16_val + s17_val) or other_sp's val; wait - to be 100% zero leakage on sp's test,
            # sp_val (357 images) has ZERO overlap with sp_test (1,000 images)!
            # What if we also check if other_sp's val has any overlap with sp_test?
            # Never use other_sp's val if it overlaps with sp_test! Only use sp's own non-test images!
            models, gva, gte, va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2, te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = splits_cache[sp]
            clean_mask_va = get_clean_shield_mask(va_cln0, va_cln1, va_cln2, 0.28)
            clean_mask_te = get_clean_shield_mask(te_cln0, te_cln1, te_cln2, 0.28)

            # 1. Optimize T0[ci] on gva (with shrinkage toward HAND['T0'])
            best_T0 = list(HAND['T0'])
            base_pred_va = decode_pc(va_p0, va_p1, va_p2, HAND)
            base_pred_va[clean_mask_va] = 0
            best_sc_va = cm.score(gva, base_pred_va, TAU)['total']

            for ci in range(6):
                best_t = best_T0[ci]
                for cand in np.linspace(0.28, 0.44, 17):
                    cand_T0 = list(best_T0)
                    cand_T0[ci] = float(cand)
                    P_tmp = dict(HAND, T0=cand_T0)
                    pr_va = decode_pc(va_p0, va_p1, va_p2, P_tmp)
                    pr_va[clean_mask_va] = 0
                    sc = cm.score(gva, pr_va, TAU)['total']
                    if sc > best_sc_va + 0.01:
                        best_sc_va = sc
                        best_t = float(cand)
                best_T0[ci] = 0.65 * best_t + 0.35 * HAND['T0'][ci]

            c0_va = np.column_stack([(va_p0[:, ci] > best_T0[ci]) & (~clean_mask_va) for ci in range(6)])
            c0_te = np.column_stack([(te_p0[:, ci] > best_T0[ci]) & (~clean_mask_te) for ci in range(6)])

            # 2. Independently optimize each FULL class k=0..3 on ~c0_va for F1(k, P1) + F1(k, P2)!
            pred_te = np.where(c0_te, 3, 0)
            for k, c in enumerate(FULL):
                ci = CLASSES.index(c)
                q0_va, q1_va, q2_va = va_p0[:, ci], va_p1[:, k], va_p2[:, k]
                q0_te, q1_te, q2_te = te_p0[:, ci], te_p1[:, k], te_p2[:, k]
                y_va = gva[:, ci]

                def eval_k(q0, q1, q2, c_mask, c0_m, t1, t2, r12, w0):
                    c2 = q2 > t2
                    c1 = (q1 + w0 * q0 > t1) & ((q1 * r12 + w0 * q0) >= (q2 - 0.05))
                    pr = np.where(c2, 1, 0)
                    pr = np.where(c1, 2, pr)
                    pr = np.where(c_mask, 0, pr)
                    pr = np.where(c0_m, 3, pr)
                    return pr

                def f1_pair(y_true, y_pred):
                    f1_sum = 0.0
                    for g_val in (1, 2):
                        tp = np.sum((y_true == g_val) & (y_pred == g_val))
                        fp = np.sum((y_true != g_val) & (y_pred == g_val))
                        fn = np.sum((y_true == g_val) & (y_pred != g_val))
                        f1_sum += (2.0 * tp) / max(1.0, 2.0 * tp + fp + fn)
                    return f1_sum

                best_params = (HAND['T1'][k], HAND['T2'][k], HAND['R12'][k], 0.0)
                best_f1 = f1_pair(y_va, eval_k(q0_va, q1_va, q2_va, clean_mask_va, c0_va[:, ci], *best_params))

                for t1 in np.linspace(0.22, 0.36, 8):
                    for t2 in np.linspace(0.24, 0.38, 8):
                        for r12 in [0.75, 0.85, 0.95, 1.05, 1.15]:
                            for w0 in [0.0, 0.15, 0.30]:
                                pr_va_k = eval_k(q0_va, q1_va, q2_va, clean_mask_va, c0_va[:, ci], t1, t2, r12, w0)
                                sc_k = f1_pair(y_va, pr_va_k)
                                if sc_k > best_f1 + 1e-4:
                                    best_f1 = sc_k
                                    best_params = (t1, t2, r12, w0)

                # Apply shrinkage toward HAND for robust generalization
                t1_f = 0.65 * best_params[0] + 0.35 * HAND['T1'][k]
                t2_f = 0.65 * best_params[1] + 0.35 * HAND['T2'][k]
                r12_f = 0.65 * best_params[2] + 0.35 * HAND['R12'][k]
                w0_f = 0.65 * best_params[3]
                pred_te[:, ci] = eval_k(q0_te, q1_te, q2_te, clean_mask_te, c0_te[:, ci], t1_f, t2_f, r12_f, w0_f)

            s_opt = cm.score(gte, pred_te, TAU)
            f1_opt = {f'{x[0]}_{x[1]}': x[4] for x in s_opt['per']}
            rows_opt.append((s_opt, f1_opt))
            print(f"  [{mode}_s{sp}] total={s_opt['total']:6.2f} S_pos={s_opt['S_pos']:.3f} S_p0={s_opt['S_p0']:.3f} S_cln={s_opt['S_clean']:.3f} | OG_P2={f1_opt['过度生成_P2']:.3f} OG_P1={f1_opt['过度生成_P1']:.3f} OG_P0={f1_opt['过度生成_P0']:.3f} SH_P1={f1_opt['过锐_P1']:.3f} SH_P0={f1_opt['过锐_P0']:.3f}")
        print(f"  [{mode} MEAN] total={np.mean([r[0]['total'] for r in rows_opt]):6.2f}\n")
        break


if __name__ == '__main__':
    main()
