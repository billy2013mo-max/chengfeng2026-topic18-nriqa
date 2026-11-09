#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/finetune/unified_dual_metric_gate.py
Evaluates any prediction matrix (N, 6) against ground truth (N, 6) under BOTH:
  - Doc1 (code/utils.py: 6-class grouped S_pos, 0.6R+0.4F1 S_p0, tau=0.02 P0 cliff S_clean)
  - Doc2 (Official Spec Appendix 2 / comp_metrics.py: 14-tag equal S_pos, 0.5R+0.3F1+0.2P S_p0, strict all-zero Clean2)
"""
import numpy as np
from utils import competition_score as score_doc1
import comp_metrics as cm


def evaluate_both_metrics(pred_6d: np.ndarray, gold_6d: np.ndarray, tau: float = 0.02) -> dict:
    s1 = score_doc1(pred_6d, gold_6d, tau=tau)
    s2 = cm.score(gold_6d, pred_6d, tau=tau)
    clean_mask = (gold_6d == 0).all(axis=1)
    clean2_strict = float(((pred_6d == 0).all(axis=1) & clean_mask).sum() / max(1, clean_mask.sum()))
    doc2_strict_total = 25.0 * s2['S_pos'] + 20.0 * s2['S_p0'] + 20.0 * clean2_strict
    return {
        'doc1_total': float(s1['total']),
        'doc1_S_pos': float(25.0 * s1['S_pos']),
        'doc1_S_p0': float(20.0 * s1['S_p0']),
        'doc1_S_clean': float(20.0 * s1['S_clean']),
        'doc2_cliff_total': float(s2['total']),
        'doc2_strict_total': float(doc2_strict_total),
        'doc2_S_pos': float(25.0 * s2['S_pos']),
        'doc2_S_p0': float(20.0 * s2['S_p0']),
        'doc2_Clean2': float(20.0 * clean2_strict),
        'per_14': s2['per'],
        'doc1': {
            'total': float(s1['total']),
            'S_pos': float(s1['S_pos']),
            'S_p0': float(s1['S_p0']),
            'S_clean': float(s1['S_clean']),
        },
        'doc2': {
            'total': float(doc2_strict_total),
            'cliff_total': float(s2['total']),
            'S_pos': float(s2['S_pos']),
            'S_p0': float(s2['S_p0']),
            'Clean2': float(clean2_strict),
        },
    }
