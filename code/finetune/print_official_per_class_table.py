import sys, os, pickle
import numpy as np

sys.path.extend(['code', 'code/finetune', 'code/decision_eval'])
from utils import competition_score, DIMS, FULL_DIMS, GRADE_REV
from evaluate_v17_ensemble_benchmark import res_base, res_fused_base

print("=" * 125)
print(f"{'Defect Class':<12} | {'Base Class F1':<14} | {'Fused Class F1':<15} | {'Diff':<8} | {'Base P0 F1':<11} | {'Fused P0 F1':<12} | {'Base P1 F1':<11} | {'Fused P1 F1':<12} | {'Base P2 F1':<11} | {'Fused P2 F1':<12}")
print("-" * 125)

# Calculate on combined N=2,000
gt_all = np.concatenate([res_base[16]['gt'], res_base[17]['gt']], 0)
pr_base_all = np.concatenate([res_base[16]['pr'], res_base[17]['pr']], 0)
pr_fused_all = np.concatenate([res_fused_base[16]['pr'], res_fused_base[17]['pr']], 0)

sc_b = competition_score(pr_base_all, gt_all)
sc_f = competition_score(pr_fused_all, gt_all)

# Extract per-grade metrics
def parse_per(sc):
    res = {}
    for c, g, P, R, F, tp, fp, fn in sc['per']:
        if c not in res: res[c] = {}
        res[c][g] = F
    # Compute class macro F1
    class_macro = {}
    for c in DIMS:
        grades = ['P2', 'P1', 'P0'] if c in FULL_DIMS else ['P0']
        class_macro[c] = np.mean([res[c][g] for g in grades])
    return res, class_macro

per_b, cm_b = parse_per(sc_b)
per_f, cm_f = parse_per(sc_f)

for c in DIMS:
    b_cm = cm_b[c]
    f_cm = cm_f[c]
    diff = f_cm - b_cm
    
    b_p0 = per_b[c].get('P0', 0.0)
    f_p0 = per_f[c].get('P0', 0.0)
    
    b_p1 = per_b[c].get('P1', -1.0)
    f_p1 = per_f[c].get('P1', -1.0)
    
    b_p2 = per_b[c].get('P2', -1.0)
    f_p2 = per_f[c].get('P2', -1.0)
    
    p1_str_b = f"{b_p1:.4f}" if b_p1 >= 0 else "  N/A "
    p1_str_f = f"{f_p1:.4f}" if f_p1 >= 0 else "  N/A "
    p2_str_b = f"{b_p2:.4f}" if b_p2 >= 0 else "  N/A "
    p2_str_f = f"{f_p2:.4f}" if f_p2 >= 0 else "  N/A "
    
    print(f"{c:<12} | {b_cm:12.4f}   | {f_cm:13.4f}   | {diff:+7.4f} | {b_p0:9.4f}   | {f_p0:10.4f}   | {p1_str_b:<11} | {p1_str_f:<12} | {p2_str_b:<11} | {p2_str_f:<12}")

print("=" * 125)
