#!/usr/bin/env python3
"""
code/utils.py
Shared constants, 14-channel probability/label conversions, competition metric evaluation,
and submission formatting utilities for the 55+ Ultimate NR-IQA Pipeline.
"""
import numpy as np

DIMS = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
FULL_DIMS = [0, 1, 2, 4]      # 模糊, 纹理损失, 过锐, 过度生成 (4 grades: 0, 1=P2, 2=P1, 3=P0)
P0_ONLY_DIMS = [3, 5]         # 花屏, 虚化突兀 (2 grades: 0, 3=P0)

JSON_KEY_ORDER = ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']
GRADE_MAP = {0: '无问题', 1: 'P2', 2: 'P1', 3: 'P0'}
GRADE_REV = {v: k for k, v in GRADE_MAP.items()}


def labels_to_14ch(y_6d: np.ndarray) -> np.ndarray:
    """
    Converts [N, 6] integer grades in {0, 1(P2), 2(P1), 3(P0)} into 14 cumulative binary channels:
      - ch 0..5:   P0 (Y >= 3) for all 6 dims
      - ch 6..9:   P1 cumulative (Y >= 2, i.e. P0 or P1) for FULL_DIMS [0, 1, 2, 4]
      - ch 10..13: P2 cumulative (Y >= 1, i.e. P0, P1, or P2) for FULL_DIMS [0, 1, 2, 4]
    """
    y_6d = np.asarray(y_6d, dtype=np.int64)
    N = y_6d.shape[0]
    y_14 = np.zeros((N, 14), dtype=np.float32)
    for d in range(6):
        y_14[:, d] = (y_6d[:, d] >= 3).astype(np.float32)
    for j, d in enumerate(FULL_DIMS):
        y_14[:, 6 + j] = (y_6d[:, d] >= 2).astype(np.float32)
        y_14[:, 10 + j] = (y_6d[:, d] >= 1).astype(np.float32)
    return y_14


def p012_to_14ch(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Converts marginal/exact grade probabilities:
      p0: [N, 6] P(Y == 3)
      p1: [N, 4] P(Y == 2) for FULL_DIMS
      p2: [N, 4] P(Y == 1) for FULL_DIMS
    into the 14 cumulative probability channels:
      ch 0..5:   P(Y >= 3) = p0
      ch 6..9:   P(Y >= 2) = p0[:, FULL_DIMS] + p1
      ch 10..13: P(Y >= 1) = p0[:, FULL_DIMS] + p1 + p2
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    N = p0.shape[0]
    out = np.zeros((N, 14), dtype=np.float64)
    out[:, :6] = np.clip(p0, 1e-6, 1.0 - 1e-6)
    for j, d in enumerate(FULL_DIMS):
        c1 = np.clip(p0[:, d] + p1[:, j], out[:, d], 1.0 - 1e-6)
        c2 = np.clip(p0[:, d] + p1[:, j] + p2[:, j], c1, 1.0 - 1e-6)
        out[:, 6 + j] = c1
        out[:, 10 + j] = c2
    return out


def ch14_to_p012(probs_14ch: np.ndarray):
    """
    Converts 14 cumulative probability channels back into marginal/exact (p0, p1, p2):
      p0: [N, 6], p1: [N, 4], p2: [N, 4]
    """
    probs_14ch = np.asarray(probs_14ch, dtype=np.float64)
    N = probs_14ch.shape[0]
    p0 = np.clip(probs_14ch[:, :6], 0.0, 1.0)
    p1 = np.zeros((N, 4), dtype=np.float64)
    p2 = np.zeros((N, 4), dtype=np.float64)
    for j, d in enumerate(FULL_DIMS):
        c0 = p0[:, d]
        c1 = np.maximum(c0, np.clip(probs_14ch[:, 6 + j], 0.0, 1.0))
        c2 = np.maximum(c1, np.clip(probs_14ch[:, 10 + j], 0.0, 1.0))
        p1[:, j] = np.clip(c1 - c0, 0.0, 1.0)
        p2[:, j] = np.clip(c2 - c1, 0.0, 1.0)
    return p0, p1, p2


def competition_score(preds: np.ndarray, gold: np.ndarray, tau: float = 0.02) -> dict:
    """
    Computes the official 65-point objective score (matching train_phase2_corn_tower.py):
      total = 25 * S_pos + 20 * S_p0 + 20 * S_clean
    where S_pos is the 6-class macro-averaged F1, and S_p0 = 0.6 * R_p0 + 0.4 * F1_p0.
    """
    preds = np.asarray(preds, dtype=np.int64)
    gold = np.asarray(gold, dtype=np.int64)
    per = []
    class_f1s = []
    p0_recs, p0_precs, p0_f1s = [], [], []

    for ci, c in enumerate(DIMS):
        grades = ['P2', 'P1', 'P0'] if ci in FULL_DIMS else ['P0']
        c_f1_list = []
        for g in grades:
            gi = GRADE_REV[g]
            yt = (gold[:, ci] == gi)
            yp = (preds[:, ci] == gi)
            tp = int((yt & yp).sum())
            fp = int((~yt & yp).sum())
            fn = int((yt & ~yp).sum())
            P = tp / max(1, tp + fp)
            R = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            F = 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            per.append((c, g, P, R, F, tp, fp, fn))
            c_f1_list.append(F)
            if g == 'P0':
                p0_recs.append(R)
                p0_precs.append(P)
                p0_f1s.append(F)
        class_f1s.append(float(np.mean(c_f1_list)))

    S_pos = float(np.mean(class_f1s))
    r_p0 = float(np.mean(p0_recs))
    p_p0 = float(np.mean(p0_precs))
    f_p0 = float(np.mean(p0_f1s))
    S_p0 = 0.6 * r_p0 + 0.4 * f_p0

    clean = (gold == 0).all(axis=1)
    flagged = ((preds == 3).any(axis=1) & clean)
    fpr = float(flagged.sum() / max(1, clean.sum())) if clean.sum() > 0 else 0.0
    S_clean = max(0.0, 1.0 - fpr / tau)

    clean_all_zero = float(((preds == 0).all(axis=1) & clean).sum() / max(1, clean.sum()))
    total = 25.0 * S_pos + 20.0 * S_p0 + 20.0 * S_clean
    return {
        'per': per,
        'S_pos': S_pos,
        'S_p0': S_p0,
        'S_clean': S_clean,
        'S_clean_strict': clean_all_zero,
        'p0R': r_p0,
        'p0P': p_p0,
        'p0F': f_p0,
        'fpr': fpr,
        'total': total,
    }


def eval_14ch(probs_14ch: np.ndarray, y_6d: np.ndarray, t0=None, t1=None, t2=None) -> dict:
    """
    Evaluates a [N, 14] cumulative probability matrix against [N, 6] ground truth grades.
    """
    if t0 is None:
        t0 = np.array([0.38, 0.38, 0.34, 0.40, 0.34, 0.38], dtype=np.float64)
    if t1 is None:
        t1 = np.array([0.30, 0.30, 0.30, 0.30], dtype=np.float64)
    if t2 is None:
        t2 = np.array([0.28, 0.28, 0.28, 0.30], dtype=np.float64)

    N = probs_14ch.shape[0]
    preds = np.zeros((N, 6), dtype=np.int64)
    for j, d in enumerate(FULL_DIMS):
        p0 = probs_14ch[:, d]
        p_ge2 = np.maximum(p0, probs_14ch[:, 6 + j])
        p_ge1 = np.maximum(p_ge2, probs_14ch[:, 10 + j])
        g = np.zeros(N, dtype=np.int64)
        g[p_ge1 >= t2[j]] = 1
        g[p_ge2 >= t1[j]] = 2
        g[p0 >= t0[d]] = 3
        preds[:, d] = g
    for d in P0_ONLY_DIMS:
        preds[:, d] = np.where(probs_14ch[:, d] >= t0[d], 3, 0)

    return competition_score(preds, y_6d)


def format_submission(ordered_names, pred_matrix):
    """
    Formats [N, 6] integer grade predictions into the official competition JSON list.
    """
    out_list = []
    for i, name in enumerate(ordered_names):
        row_dict = {}
        for k in JSON_KEY_ORDER:
            ci = DIMS.index(k)
            grade_int = int(pred_matrix[i, ci])
            row_dict[k] = GRADE_MAP[grade_int]
        out_list.append({'image': name, 'label': row_dict})
    return out_list
