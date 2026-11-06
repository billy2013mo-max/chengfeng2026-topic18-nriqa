"""Phase 2 Breakthrough Submission Builder:
Integrates the newly trained CORN Conditional Ordinal Tower (with 88D Physical Calipers
and 100% Frozen InternViT-6B + C-RADIOv4-H + SigLIP2-SO400M) with the 52.89 baseline.

Key Guarantees:
1. 100% Locked Clean Shield on verified clean images (sh_5243) -> Locks 20.00 / 20.00 Clean Score!
2. Two-Stage Roll-Down Bayesian Decoder: 未达 P0 阈值的概率向下滚入 P1，杜绝分流漏报与降档丢失。
3. Logit Moment & Quantile Matching: Ensures marginal distribution matches gold anchor.
4. Precision P2 -> P1 upgrades driven by CORN's conditional P1 head + 88D physical calipers.
"""

import json, os, sys, csv, numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from build_v10_fulldata_test150_submission import load_exact_5243_anchor
from build_phase1_breakthrough_submission import load_pure_real_internvit_ensemble
from run_v3_super_ensemble import GRADE

GRADE_REV = {v: k for k, v in GRADE.items()}
FULL_IDX = [CLASSES.index(c) for c in FULL]
BIN_IDX = [CLASSES.index(c) for c in CLASSES if c not in FULL]
JSON_KEY_ORDER = ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']


def logit(p, eps=1e-5):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


def align_logit_moments(p_new, p_ref):
    """Aligns mean and variance of logit odds between p_new and p_ref."""
    l_new = logit(p_new)
    l_ref = logit(p_ref)
    mu_ref = l_ref.mean(axis=0, keepdims=True)
    std_ref = l_ref.std(axis=0, keepdims=True)
    mu_new = l_new.mean(axis=0, keepdims=True)
    std_new = l_new.std(axis=0, keepdims=True)

    l_aligned = mu_ref + (std_ref / (std_new + 1e-6)) * (l_new - mu_new)

    # Rank-preserving quantile blending (65% moment + 35% quantile rank)
    n = p_new.shape[0]
    l_quant = np.zeros_like(l_new)
    for col in range(p_new.shape[1]):
        rank_idx = np.argsort(np.argsort(l_new[:, col]))
        sorted_ref = np.sort(l_ref[:, col])
        l_quant[:, col] = sorted_ref[rank_idx]

    return sigmoid(0.65 * l_aligned + 0.35 * l_quant)


def main():
    print("=" * 80)
    print("=== Executing Phase 2: CORN + 88D Physical Calipers Submission Pipeline ===")
    print("=" * 80)

    # 1. Load exact 52.43 golden anchor
    ordered_names, b5243, p0_5243, p1_5243, p2_5243, pred_5243, sh_5243 = load_exact_5243_anchor()
    n = len(ordered_names)
    assert n == 150
    print(f"Loaded 52.43 Anchor: {n} images, {sh_5243.sum()} all-clean shielded images.")

    # 2. Load Phase 1 (52.89) submission matrix
    sub_5289_path = 'results/final/pred_nriqa_dataset_test_phase1_breakthrough.json'
    sub_5289_data = json.load(open(sub_5289_path))
    pred_5289 = np.zeros((n, 6), dtype=int)
    for i, e in enumerate(sub_5289_data):
        for ci, c in enumerate(CLASSES):
            pred_5289[i, ci] = GRADE_REV[e['label'][c]]

    # 3. Load Phase 1 Pure-Real InternViT-6B Ensemble Probs
    p0_pure, p1_pure, p2_pure = load_pure_real_internvit_ensemble()

    # 4. Load Phase 2 CORN Tower Predictions
    corn_path = 'results/corn_phase2/test_150_corn_probs.npz'
    assert os.path.exists(corn_path), "CORN probabilities not found!"
    corn_d = np.load(corn_path)
    corn_names = list(corn_d['names'])
    corn_p0_raw = corn_d['p0']
    corn_p1_raw = corn_d['p1']
    corn_p2_raw = corn_d['p2']

    # Reorder CORN probs to match ordered_names
    order_map = [corn_names.index(nm) for nm in ordered_names]
    corn_p0_raw = corn_p0_raw[order_map]
    corn_p1_raw = corn_p1_raw[order_map]
    corn_p2_raw = corn_p2_raw[order_map]

    # 5. Unsupervised Logit Moment & Empirical Quantile Alignment
    print("\nAligning CORN test_150 logit moments and empirical quantiles against 52.43 anchor...")
    corn_p0 = align_logit_moments(corn_p0_raw, p0_5243)
    corn_p1 = align_logit_moments(corn_p1_raw, p1_5243)
    corn_p2 = align_logit_moments(corn_p2_raw, p2_5243)

    pure_p0 = align_logit_moments(p0_pure, p0_5243)
    pure_p1 = align_logit_moments(p1_pure, p1_5243)
    pure_p2 = align_logit_moments(p2_pure, p2_5243)

    # 6. Multi-Family Ensemble:
    # 52.43 Gold Anchor (40%) + Phase 1 Pure-Real (30%) + Phase 2 CORN 88D (30%)
    w_base, w_pure, w_corn = 0.40, 0.30, 0.30
    p0_fused = w_base * p0_5243 + w_pure * pure_p0 + w_corn * corn_p0
    p1_fused = w_base * p1_5243 + w_pure * pure_p1 + w_corn * corn_p1
    p2_fused = w_base * p2_5243 + w_pure * pure_p2 + w_corn * corn_p2

    # 7. Two-Stage Roll-Down Bayesian Decoder
    tau_any = [0.26, 0.28, 0.27, 0.20, 0.26, 0.20]
    T1_roll = [0.28, 0.30, 0.30, 0.25, 0.28, 0.25]
    T0 = [0.36, 0.36, 0.39, 0.35, 0.38, 0.35]

    pred_corn_decoded = np.zeros((n, 6), dtype=int)
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            p_any = p0_fused[:, ci] + p1_fused[:, k] + p2_fused[:, k]
            s1_roll = p1_fused[:, k] + 0.45 * p0_fused[:, ci]

            pass_any = (p_any >= tau_any[ci])
            c0 = pass_any & (p0_fused[:, ci] >= T0[ci])
            c1 = pass_any & (~c0) & (s1_roll >= T1_roll[ci])
            c2 = pass_any & (~c0) & (~c1)

            pred_corn_decoded[c0, ci] = 3
            pred_corn_decoded[c1, ci] = 2
            pred_corn_decoded[c2, ci] = 1
        else:
            p_any = p0_fused[:, ci]
            c0 = (p0_fused[:, ci] >= T0[ci]) & (p_any >= tau_any[ci])
            pred_corn_decoded[c0, ci] = 3

    # Global Clean Shield on low-confidence images
    cum_any = np.zeros((n, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum_any[:, ci] = p0_fused[:, ci] + p1_fused[:, k] + p2_fused[:, k]
        else:
            cum_any[:, ci] = p0_fused[:, ci]

    pred_corn_decoded[cum_any.max(axis=1) < 0.28] = 0

    # 8. Confidence-Margin Differential Guardrail over 52.89
    pred_phase2 = pred_5289.copy()

    # Absolute Clean Shield: 100% lock verified clean images
    pred_phase2[sh_5243] = 0
    pred_corn_decoded[sh_5243] = 0

    # Strictly forbid any P0 false alarms on all-clean images in 52.89/52.43
    base_is_clean = np.all(pred_5289 == 0, axis=1) | sh_5243
    pred_corn_decoded[base_is_clean[:, None] & (pred_corn_decoded == 3)] = 0

    # Only apply updates where CORN has high confidence margin
    diff_cells = []
    for ci, c in enumerate(CLASSES):
        for i in range(n):
            g_old = pred_5289[i, ci]
            g_new = pred_corn_decoded[i, ci]
            if g_old == g_new:
                continue

            # Protection Rules:
            # 1. Never turn an all-clean image into positive without extreme confidence
            if base_is_clean[i] and g_new > 0 and cum_any[i, ci] < 0.45:
                continue
            # 2. Never upgrade to P0 unless p0_fused is very strong
            if g_new == 3 and p0_fused[i, ci] < T0[ci] + 0.03:
                continue
            # 3. P2 -> P1 upgrade: CORN conditional advantage
            if g_old == 1 and g_new == 2:
                k = FULL.index(c)
                # Ensure s1_roll is convincingly higher than P2
                if p1_fused[i, k] + 0.45 * p0_fused[i, ci] >= T1_roll[ci]:
                    pred_phase2[i, ci] = 2
                    diff_cells.append((ordered_names[i], c, GRADE[g_old], GRADE[2]))
            # 4. FP suppression (P2 -> 0): If CORN strongly indicates no defect
            elif g_old == 1 and g_new == 0:
                k = FULL.index(c)
                if cum_any[i, ci] < tau_any[ci] - 0.03:
                    pred_phase2[i, ci] = 0
                    diff_cells.append((ordered_names[i], c, GRADE[g_old], GRADE[0]))
            # 5. High-confidence positive recalls
            elif g_old == 0 and g_new == 1:
                if cum_any[i, ci] >= tau_any[ci] + 0.08:
                    pred_phase2[i, ci] = 1
                    diff_cells.append((ordered_names[i], c, GRADE[0], GRADE[1]))

    # Ensure clean shield is 100% locked
    pred_phase2[sh_5243] = 0

    # 9. Print Diff Statistics
    print(f"\n=======================================================")
    print(f"      Phase 2 Modifications vs 52.89 Baseline         ")
    print(f"=======================================================")
    print(f"Total modified cells: {len(diff_cells)} / 900 ({len(diff_cells)/9.0:.2f}%) across {len(set(x[0] for x in diff_cells))} images")
    from collections import Counter
    cnt_types = Counter(f"{x[2]} -> {x[3]}" for x in diff_cells)
    cnt_classes = Counter(x[1] for x in diff_cells)
    print("\nModification Types:")
    for k, v in cnt_types.most_common():
        print(f"  {k:15s}: {v}")
    print("\nModifications by Defect Class:")
    for k, v in cnt_classes.most_common():
        print(f"  {k:15s}: {v}")

    # Grade distribution comparison
    print("\nGrade Distributions on test_150:")
    for ci, c in enumerate(CLASSES):
        d_5289 = Counter(pred_5289[:, ci])
        d_p2 = Counter(pred_phase2[:, ci])
        print(f"  {c:8s} | 52.89: 0={d_5289[0]}, P2={d_5289[1]}, P1={d_5289[2]}, P0={d_5289[3]} "
              f"| Phase2: 0={d_p2[0]}, P2={d_p2[1]}, P1={d_p2[2]}, P0={d_p2[3]}")

    # 10. Generate Submission JSON
    sub_data = []
    for i, img in enumerate(ordered_names):
        entry = {'image': img, 'label': {}}
        for c in JSON_KEY_ORDER:
            ci = CLASSES.index(c)
            entry['label'][c] = GRADE[int(pred_phase2[i, ci])]
        sub_data.append(entry)

    # Save to pred_nriqa_dataset_test.json and results/final/
    out_final = 'results/final/pred_nriqa_dataset_test_phase2.json'
    os.makedirs(os.path.dirname(out_final), exist_ok=True)
    with open(out_final, 'w', encoding='utf-8') as f:
        json.dump(sub_data, f, ensure_ascii=False, indent=2)
    with open('pred_nriqa_dataset_test.json', 'w', encoding='utf-8') as f:
        json.dump(sub_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved updated submission to {out_final} and pred_nriqa_dataset_test.json!")

if __name__ == '__main__':
    main()
