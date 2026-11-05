"""Test Decoupling Clean Shield:
1. `S_clean` in the competition metric (`comp_metrics.py` lines 71-74) ONLY penalizes `P0` (`pred == 3`)
   on all-clean images (`(gold == 0).all(1)`).
2. In previous versions, `sh = cum_cln.max(1) < 0.28` zeroed out BOTH `P0` (`3`) AND `P1/P2` (`2`, `1`)!
   Let's check what happens when:
   - `sh_p0` (`cum_cln.max(1) < 0.28`) shields `P0` (preventing any P0 false alarm on all-clean images -> `S_clean = 1.000`),
   - while `P1` and `P2` use a much lighter or zero shield `sh_p12` (`cum_cln.max(1) < th_p12`),
   - AND when `sh_p0` is triggered on an image that has strong specialist `q0`, instead of zeroing the cell to `0` (`无问题`), it demotes `3 (P0) -> 2 (P1)`!
"""
import os, sys
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from build_v8_final_submission_and_eval import run_isolated_1000_eval

data_sp, p0_rec, p1p2_rec = run_isolated_1000_eval()

for th_p12 in [0.0, 0.12, 0.16, 0.20, 0.24, 0.28]:
    tot_16, tot_17 = 0.0, 0.0
    for sp in (16, 17):
        te_cln0 = data_sp[sp]['te_cln0']
        te_cln1 = data_sp[sp]['te_cln1']
        te_cln2 = data_sp[sp]['te_cln2']
        cum_cln = np.zeros((1000, 6))
        for ci, c in enumerate(CLASSES):
            if c in FULL:
                k = FULL.index(c)
                cum_cln[:, ci] = te_cln0[:, ci] + te_cln1[:, k] + te_cln2[:, k]
            else:
                cum_cln[:, ci] = te_cln0[:, ci]
        sh_p0 = cum_cln.max(1) < 0.28
        sh_p12 = cum_cln.max(1) < th_p12

        pred = np.zeros((1000, 6), dtype=int)
        for ci, cname in enumerate(CLASSES):
            t0 = p0_rec[cname]['th']
            q0 = data_sp[sp]['te_p0'][:, ci]
            c0 = (q0 > t0) & (~sh_p0)
            if cname not in FULL:
                pred[c0, ci] = 3
            else:
                k = FULL.index(cname)
                rc = p1p2_rec[cname]
                q1 = data_sp[sp]['opt_q1_' + str(k)]
                q2p = data_sp[sp]['opt_q2p_' + str(k)]
                q2g = data_sp[sp]['opt_q2g_' + str(k)]
                cum = q0 + q1 + q2p
                present = ((q2p > rc['t2']) | (q1 > rc['t1p']) | (cum > rc['tcum'])) & (~sh_p12)
                s1 = q1 + rc['w0'] * q0
                is_p1 = (s1 > rc['t1']) & (s1 >= rc['r12'] * q2g)
                pr = np.where(present, 1, 0)
                pr = np.where(present & is_p1, 2, pr)
                # Also if q0 > t0 was blocked by sh_p0, demote to P1 (2) instead of 0 if not sh_p12!
                pr = np.where((q0 > t0) & sh_p0 & (~sh_p12), 2, pr)
                pr = np.where(c0, 3, pr)
                pred[:, ci] = pr
        sc = cm.score(data_sp[sp]['gte'], pred, 0.02)
        if sp == 16:
            s16 = sc
        else:
            s17 = sc
    print(f"th_p12={th_p12:.2f} -> s16={s16['total']:.2f}, s17={s17['total']:.2f}, mean={0.5*(s16['total']+s17['total']):.2f} (S_clean={s16['S_clean']:.3f}/{s17['S_clean']:.3f})")
