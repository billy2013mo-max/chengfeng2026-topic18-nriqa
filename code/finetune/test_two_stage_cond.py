"""Test Two-Stage Presence (C_any) + Conditional Severity Grade Decoding + V8 Specialist Routing
+ Low-Dimensional Physical Caliper Grade Refinement on the 1,000-Image Isolated Test Sets (s16 & s17).
"""
import csv
import glob
import os
import sys
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND, TAU
import comp_metrics as cm
from extract_native_phys_feats import extract_for_names
from optimize_v8_isolated_test import load_healthy_split_models, build_channel_routed_ensemble, decode_shielded_pc


def decode_two_stage_conditional(p0, p1, p2, cln_p0, cln_p1, cln_p2, cfg):
    """Two-Stage Presence + Conditional Grade Decoder:
    Stage 1 (Clean Shield): if max_c cln_cum[c] < shield -> all 0 (guarantees S_clean = 1.000).
    Stage 2 (Per-Class Presence): for 4-grade class k (ci), defect is present if:
       cum_w = w0*p0 + w1*p1 + p2 > T_pres[k]  OR  p0 > T0[ci]  OR  p1 > T1[k]  OR  p2 > T2[k]
    Stage 3 (Conditional Grade Assignment given defect is present):
       Once present, never drop back to 0! Choose Grade in {1 (P2), 2 (P1), 3 (P0)} using:
       - Grade 3 (P0) if p0 > T0[ci] or (p0 / cum >= R0_cond[k] and p0 + p1 > T_ge1[k])
       - Else Grade 2 (P1) if (p0 + p1) / cum >= R1_cond[k] or p1 > T1[k]
       - Else Grade 1 (P2)!
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

    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            q0 = p0[:, ci]
            q1 = p1[:, k]
            q2 = p2[:, k]
            cum = np.maximum(1e-6, q0 + q1 + q2)
            cum_ge1 = q0 + q1

            # Stage 2: Presence
            present = (cum > cfg['T_pres'][k]) | (q2 > cfg['T2'][k]) | (q1 > cfg['T1'][k]) | (q0 > cfg['T0'][ci])

            # Stage 3: Conditional Grade given presence
            is_p0 = (q0 > cfg['T0'][ci]) | ((cum_ge1 > cfg['T0_cum'][k]) & ((q0 / cum) >= cfg['R0_cond'][k]))
            is_p1 = (~is_p0) & (
                ((q1 > cfg['T1'][k]) & (q1 * cfg['R12'][k] >= q2 - 0.05))
                | (((cum_ge1 / cum) >= cfg['R1_cond'][k]) & (cum_ge1 > cfg['T1_cum'][k]))
            )

            pr = np.where(present, 1, 0)
            pr = np.where(present & is_p1, 2, pr)
            pr = np.where(present & is_p0, 3, pr)
            pred[:, ci] = pr
        else:
            pred[:, ci] = np.where(p0[:, ci] > cfg['T0'][ci], 3, 0)

    sh = cfg.get('shield', 0.28)
    if sh > 0:
        pred[cum_cln.max(1) < sh] = 0
    return pred


def main():
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        va_p0, va_p1, va_p2, va_cln0, va_cln1, va_cln2 = build_channel_routed_ensemble(models, 'va')
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_channel_routed_ensemble(models, 'te')

        pred_hand = decode_shielded_pc(te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2, HAND)
        sc_hand = cm.score(gte, pred_hand, TAU)
        print(f"\n================ Split s{sp} V8_SuperRouted_HAND: {sc_hand['total']:.2f} ================")
        print(f"  S_pos={sc_hand['S_pos']:.4f} ({25*sc_hand['S_pos']:.2f}/25), S_p0={sc_hand['S_p0']:.4f} ({20*sc_hand['S_p0']:.2f}/20, R={sc_hand['p0R']:.3f}, P={sc_hand['p0P']:.3f}), S_clean={sc_hand['S_clean']:.3f}")
        for row in sc_hand['per']:
            cls_name, grade_name, prec, rec, f1 = row[:5]
            print(f"  {cls_name+'_'+grade_name:14s}: P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

        # Count how many true defects in test have cum > 0.35 but pred_hand == 0!
        for ci, c in enumerate(CLASSES):
            if c in FULL:
                k = FULL.index(c)
                cum = te_p0[:, ci] + te_p1[:, k] + te_p2[:, k]
                missed_split_mass = np.sum((gte[:, ci] > 0) & (pred_hand[:, ci] == 0) & (cum > 0.36))
                print(f"  [{c}] True positives missed by HAND despite cum > 0.36 (split probability mass!): {missed_split_mass}")


if __name__ == '__main__':
    main()
