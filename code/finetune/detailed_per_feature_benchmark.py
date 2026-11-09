import sys, os, pickle
import numpy as np

sys.path.extend(['code', 'code/finetune', 'code/decision_eval'])
from utils import competition_score
from cv_decoder_calib import decode_pc
from data import CLASSES, FULL

from evaluate_v17_ensemble_benchmark import res_base, res_fused_base

print("=" * 110)
print("  OVERALL COMPETITION METRICS BREAKDOWN (s16 N=1,000 | s17 N=1,000 | Total N=2,000)")
print("=" * 110)

for tag, res in [('Base Baseline (Stack5457)', res_base), ('V17 Fused Ensemble (Base+V17)', res_fused_base)]:
    s16_tot = res[16]['sc']['total']
    s16_pos = res[16]['sc']['S_pos'] * 25
    s16_p0  = res[16]['sc']['S_p0'] * 20
    s16_cln = res[16]['sc']['S_clean'] * 20

    s17_tot = res[17]['sc']['total']
    s17_pos = res[17]['sc']['S_pos'] * 25
    s17_p0  = res[17]['sc']['S_p0'] * 20
    s17_cln = res[17]['sc']['S_clean'] * 20

    m_tot = 0.5 * (s16_tot + s17_tot)
    m_pos = 0.5 * (s16_pos + s17_pos)
    m_p0  = 0.5 * (s16_p0 + s17_p0)
    m_cln = 0.5 * (s16_cln + s17_cln)

    print(f"{tag:<32} | Total: {m_tot:6.3f} (s16={s16_tot:6.3f}, s17={s17_tot:6.3f}) | S_pos: {m_pos:5.2f}/25 | S_p0: {m_p0:5.2f}/20 | S_clean: {m_cln:5.2f}/20")

print("\n" + "=" * 110)
print("  PER-FEATURE DEFECT METRICS (COMBINED N=2,000 ISOLATED BLIND TEST SETS)")
print("=" * 110)

gt_all = np.concatenate([res_base[16]['gt'], res_base[17]['gt']], 0)
pr_base_all = np.concatenate([res_base[16]['pr'], res_base[17]['pr']], 0)
pr_fused_all = np.concatenate([res_fused_base[16]['pr'], res_fused_base[17]['pr']], 0)

print(f"{'Defect Feature':<10} | {'True Pos':<8} | {'Base Prec':<9} | {'Base Rec':<8} | {'Base F1':<8} | {'Fused Prec':<10} | {'Fused Rec':<9} | {'Fused F1':<8} | {'F1 Diff':<8}")
print("-" * 110)

for ci, c in enumerate(CLASSES):
    pos_gt = gt_all[:, ci] > 0
    n_pos = pos_gt.sum()

    # Base
    base_tp = ((pr_base_all[:, ci] > 0) & pos_gt).sum()
    base_fp = ((pr_base_all[:, ci] > 0) & (~pos_gt)).sum()
    base_fn = ((pr_base_all[:, ci] == 0) & pos_gt).sum()
    base_prec = base_tp / (base_tp + base_fp + 1e-9)
    base_rec = base_tp / (base_tp + base_fn + 1e-9)
    base_f1 = 2 * base_prec * base_rec / (base_prec + base_rec + 1e-9)

    # Fused
    fused_tp = ((pr_fused_all[:, ci] > 0) & pos_gt).sum()
    fused_fp = ((pr_fused_all[:, ci] > 0) & (~pos_gt)).sum()
    fused_fn = ((pr_fused_all[:, ci] == 0) & pos_gt).sum()
    fused_prec = fused_tp / (fused_tp + fused_fp + 1e-9)
    fused_rec = fused_tp / (fused_tp + fused_fn + 1e-9)
    fused_f1 = 2 * fused_prec * fused_rec / (fused_prec + fused_rec + 1e-9)

    diff = fused_f1 - base_f1
    print(f"{c:<10} | {n_pos:<8d} | {base_prec:8.2%}  | {base_rec:7.2%}  | {base_f1:7.4f}  | {fused_prec:9.2%}   | {fused_rec:8.2%}  | {fused_f1:7.4f}  | {diff:+7.4f}")

print("\n" + "=" * 110)
print("  PER-SPLIT INDIVIDUAL BREAKDOWN (s16 vs s17)")
print("=" * 110)
for sp in (16, 17):
    print(f"--- Split s{sp} (N=1,000) ---")
    gt_sp = res_base[sp]['gt']
    pr_b_sp = res_base[sp]['pr']
    pr_f_sp = res_fused_base[sp]['pr']
    for ci, c in enumerate(CLASSES):
        pos_gt = gt_sp[:, ci] > 0
        n_pos = pos_gt.sum()

        b_tp = ((pr_b_sp[:, ci] > 0) & pos_gt).sum()
        b_fp = ((pr_b_sp[:, ci] > 0) & (~pos_gt)).sum()
        b_fn = ((pr_b_sp[:, ci] == 0) & pos_gt).sum()
        b_f1 = 2 * b_tp / (2 * b_tp + b_fp + b_fn + 1e-9)

        f_tp = ((pr_f_sp[:, ci] > 0) & pos_gt).sum()
        f_fp = ((pr_f_sp[:, ci] > 0) & (~pos_gt)).sum()
        f_fn = ((pr_f_sp[:, ci] == 0) & pos_gt).sum()
        f_f1 = 2 * f_tp / (2 * f_tp + f_fp + f_fn + 1e-9)

        print(f"  {c:<8}: Pos={n_pos:3d} | Base F1={b_f1:.4f} -> Fused F1={f_f1:.4f} ({f_f1-b_f1:+.4f}) | TP: {b_tp}->{f_tp}, FP: {b_fp}->{f_fp}")

