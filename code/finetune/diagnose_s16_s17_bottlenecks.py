"""Diagnose exact point bottlenecks on the 1,000-image isolated test sets (s16 & s17):
1. Binary Defect Detection F1 (Any Defect vs Clean) vs Per-Grade F1 (P2, P1, P0).
2. Confusion matrix between (无问题, P2, P1, P0) for each of the 6 classes.
3. Exact point contribution of improving each class (or handing 100% of a class to a dedicated expert) toward 53.0 ~ 55.0.
"""
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

OUT_ABL = 'results/v3_ablate'

def main():
    for sp in (16, 17):
        da = np.load(f'{OUT_ABL}/A_base_s{sp}_probs.npz')
        df = np.load(f'{OUT_ABL}/F_v5_s{sp}_probs.npz')
        de = np.load(f'{OUT_ABL}/E_synth_s{sp}_probs.npz')
        dg = np.load(f'{OUT_ABL}/G_v6_s{sp}_probs.npz')
        dh = np.load(f'{OUT_ABL}/H_og_s{sp}_probs.npz')

        extra_clean = []
        for p in sorted(glob.glob(f'results/v6_val/*s{sp}*probs.npz') + glob.glob(f'{OUT_ABL}/B_*_s{sp}_probs.npz') + glob.glob(f'{OUT_ABL}/C_*_s{sp}_probs.npz') + glob.glob(f'{OUT_ABL}/D_*_s{sp}_probs.npz')):
            try:
                d_ex = np.load(p)
                if 'te_p0' in d_ex and d_ex['te_p0'].shape == da['te_p0'].shape:
                    extra_clean.append(d_ex)
            except Exception:
                pass

        w_tot = 1.0
        c_p0 = 0.45 * da['te_p0'] + 0.55 * df['te_p0']
        c_p1 = 0.45 * da['te_p1'] + 0.55 * df['te_p1']
        c_p2 = 0.45 * da['te_p2'] + 0.55 * df['te_p2']
        for d_ex in extra_clean:
            c_p0 += 0.35 * d_ex['te_p0']
            c_p1 += 0.35 * d_ex['te_p1']
            c_p2 += 0.35 * d_ex['te_p2']
            w_tot += 0.35
        c_p0 /= w_tot; c_p1 /= w_tot; c_p2 /= w_tot

        pred = decode_pc(c_p0, c_p1, c_p2, HAND)
        gt = da['gte']
        s = cm.score(gt, pred, TAU)
        print(f"\n================ Split s{sp} (1,000 Isolated Test Images) Base Score: {s['total']:.2f} ================")
        print(f"S_pos={s['S_pos']:.3f} ({25*s['S_pos']:.2f}/25), S_p0={s['S_p0']:.3f} ({20*s['S_p0']:.2f}/20), S_clean={s['S_clean']:.3f} ({20*s['S_clean']:.2f}/20)")

        print("\nPer-Class Binary Presence F1 (Any Defect >0 vs 0) vs Grade-Exact F1 (P2 / P1 / P0):")
        f1_map = {f'{x[0]}_{x[1]}': (x[2], x[3], x[4]) for x in s['per']}
        for ci, c in enumerate(CLASSES):
            gt_any = (gt[:, ci] > 0)
            pr_any = (pred[:, ci] > 0)
            tp = np.sum(gt_any & pr_any)
            fp = np.sum((~gt_any) & pr_any)
            fn = np.sum(gt_any & (~pr_any))
            p_bin = tp / max(1, tp + fp)
            r_bin = tp / max(1, tp + fn)
            f1_bin = 2 * p_bin * r_bin / max(1e-8, p_bin + r_bin)

            # How many FP are on completely clean images vs images with other defects?
            # How many grade errors are adjacent grade (e.g. P2<->P1 or P1<->P0) where both gt>0 and pred>0?
            both_pos = gt_any & pr_any
            exact_grade = np.sum(gt[both_pos, ci] == pred[both_pos, ci])
            adj_conf = np.sum(gt[both_pos, ci] != pred[both_pos, ci])

            if c in FULL:
                g_str = f"P2 F1={f1_map[c+'_P2'][2]:.3f} | P1 F1={f1_map[c+'_P1'][2]:.3f} | P0 F1={f1_map[c+'_P0'][2]:.3f}"
            else:
                g_str = f"P0 F1={f1_map[c+'_P0'][2]:.3f}"
            print(f"  {c:6s} | Binary Any-Defect F1={f1_bin:.3f} (P={p_bin:.2f}, R={r_bin:.2f}) | Detected={tp}, ExactGrade={exact_grade}, WrongGrade={adj_conf} | {g_str}")

        # What if we fix ONLY adjacent-grade confusion (when both gt>0 and pred>0, grade is right)?
        pred_perfect_grade = pred.copy()
        both_mask = (gt > 0) & (pred > 0)
        pred_perfect_grade[both_mask] = gt[both_mask]
        s_pg = cm.score(gt, pred_perfect_grade, TAU)
        print(f"\n  -> If Grade Boundary Confusion (P2<->P1<->P0 among detected defects) is eliminated: Score = {s_pg['total']:.2f} (+{s_pg['total']-s['total']:.2f} pts!)")

        # What if each class achieves Oracle (or +20% error reduction)?
        for ci, c in enumerate(CLASSES):
            p_c = pred.copy()
            p_c[:, ci] = gt[:, ci]
            sc = cm.score(gt, p_c, TAU)
            print(f"  -> Max potential headroom in class [{c:4s}]: +{sc['total'] - s['total']:.2f} pts (to {sc['total']:.2f})")

if __name__ == '__main__':
    main()
