"""Ultra-fast (<3s) component-by-component diagnostic using `/tmp/qualirag_s16_s17_cache.pt`.
Tests each component of Scheme 1 (QualiRAG) and Scheme 2 (Two-Stage Clean Gate + Cross-Defect Conditional Decoder)
without the noisy 15-feature 400-row LogisticRegression step!
"""
import os, sys, time, warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
from cv_decoder_calib import decode_pc, HAND
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from run_scheme1_scheme2_breakthrough import FULL_IDX, BIN_IDX, score_both
from fast_test_qualirag_preserved_p0 import CACHE_PATH, align_prob_mean, decode_p0_protected_two_stage, tune_p0_protected_gate_on_val
from ablate_clean_gate_and_qualirag_lowdim import build_compact_clean_gate_features, fit_compact_clean_gate

t0 = time.time()
cache = torch.load(CACHE_PATH, weights_only=False)
print(f"Loaded cache in {time.time()-t0:.2f}s")

for sp in (16, 17):
    models, gva, gte = load_healthy_split_models(sp)
    c_sp = cache['splits'][sp]
    va_idx, te_idx, src_va = c_sp['va_idx'], c_sp['te_idx'], c_sp['src_va']
    q_feats, deqa_res = c_sp['q_feats'], c_sp['deqa_res']

    # Check standalone DeQAScore+QualiRAG 6-Tower
    r_dq = score_both(gte, decode_pc(deqa_res['te_p0'], deqa_res['te_p1'], deqa_res['te_p2'], HAND))
    print(f"\n[s{sp}] Standalone DeQAScore+QualiRAG 6-Tower (trained only on 2073 tr): Doc2={r_dq['total_doc2']:.2f} | Doc1={r_dq['total']:.2f} (S_pos={r_dq['S_pos']:.3f}, S_p0={r_dq['S_p0']:.3f}, Clean2={r_dq['S_clean_doc2']:.3f})")

    # Check pure k-NN posterior from X_knn (col 31..45 is K=35 tau=0.06: [clean(1), p0(6), p1(4), p2(4)])
    knn_te = q_feats['X_knn'][te_idx]
    knn_va = q_feats['X_knn'][va_idx]
    # Average k-NN posteriors across K=15 (0..15), K=35 (31..46), K=80 (62..77) and per-backbone k-NNs
    knn_p0_te = np.mean([knn_te[:, off+1:off+7] for off in [0, 31, 62, 93, 124, 155, 186, 217, 248]], axis=0)
    knn_p1_te = np.mean([knn_te[:, off+7:off+11] for off in [0, 31, 62, 93, 124, 155, 186, 217, 248]], axis=0)
    knn_p2_te = np.mean([knn_te[:, off+11:off+15] for off in [0, 31, 62, 93, 124, 155, 186, 217, 248]], axis=0)
    knn_cln_te = np.mean([knn_te[:, off] for off in [0, 31, 62, 93, 124, 155, 186, 217, 248]], axis=0)

    knn_p0_va = np.mean([knn_va[:, off+1:off+7] for off in [0, 31, 62, 93, 124, 155, 186, 217, 248]], axis=0)
    knn_p1_va = np.mean([knn_va[:, off+7:off+11] for off in [0, 31, 62, 93, 124, 155, 186, 217, 248]], axis=0)
    knn_p2_va = np.mean([knn_va[:, off+11:off+15] for off in [0, 31, 62, 93, 124, 155, 186, 217, 248]], axis=0)
    knn_cln_va = np.mean([knn_va[:, off] for off in [0, 31, 62, 93, 124, 155, 186, 217, 248]], axis=0)

    from fast_test_qualirag_preserved_p0 import build_p0_preserved_v9_14ch
    p0_v9_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va = build_p0_preserved_v9_14ch(models, 'va', None)
    p0_v9_te, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te = build_p0_preserved_v9_14ch(models, 'te', None)

    # Align knn + deqa to p0_v9, p1_v9, p2_v9 without ANY noisy 400-row regression!
    p0_q_va, p1_q_va, p2_q_va = p0_v9_va.copy(), p1_v9_va.copy(), p2_v9_va.copy()
    p0_q_te, p1_q_te, p2_q_te = p0_v9_te.copy(), p1_v9_te.copy(), p2_v9_te.copy()

    for ci in range(6):
        q0_va = 0.65 * deqa_res['va_p0'][:, ci] + 0.35 * knn_p0_va[:, ci]
        q0_te = 0.65 * deqa_res['te_p0'][:, ci] + 0.35 * knn_p0_te[:, ci]
        p0_q_va[:, ci] = 0.88 * p0_v9_va[:, ci] + 0.12 * align_prob_mean(q0_va, p0_v9_va[:, ci])
        p0_q_te[:, ci] = 0.88 * p0_v9_te[:, ci] + 0.12 * align_prob_mean(q0_te, p0_v9_te[:, ci])

    for k in range(4):
        q1_va = 0.65 * deqa_res['va_p1'][:, k] + 0.35 * knn_p1_va[:, k]
        q1_te = 0.65 * deqa_res['te_p1'][:, k] + 0.35 * knn_p1_te[:, k]
        p1_q_va[:, k] = 0.85 * p1_v9_va[:, k] + 0.15 * align_prob_mean(q1_va, p1_v9_va[:, k])
        p1_q_te[:, k] = 0.85 * p1_v9_te[:, k] + 0.15 * align_prob_mean(q1_te, p1_v9_te[:, k])

        q2_va = 0.65 * deqa_res['va_p2'][:, k] + 0.35 * knn_p2_va[:, k]
        q2_te = 0.65 * deqa_res['te_p2'][:, k] + 0.35 * knn_p2_te[:, k]
        p2_q_va[:, k] = 0.85 * p2_v9_va[:, k] + 0.15 * align_prob_mean(q2_va, p2_v9_va[:, k])
        p2_q_te[:, k] = 0.85 * p2_v9_te[:, k] + 0.15 * align_prob_mean(q2_te, p2_v9_te[:, k])

    r_q_hand = score_both(gte, decode_pc(p0_q_te, p1_q_te, p2_q_te, HAND))
    print(f"[s{sp}] V9 + Analytical QualiRAG (HAND decoder, before Stage-2 Gate): Doc2={r_q_hand['total_doc2']:.2f} | Doc1={r_q_hand['total']:.2f} (S_pos={r_q_hand['S_pos']:.3f}, S_p0={r_q_hand['S_p0']:.3f}, Clean2={r_q_hand['S_clean_doc2']:.3f})")

    # Now apply Stage-2 Clean Gate on V9 + Analytical QualiRAG (NO noisy 400-row channel regression!)
    cln_cum_va = cln0_va.copy()
    cln_cum_te = cln0_te.copy()
    for k, ci in enumerate(FULL_IDX):
        cln_cum_va[:, ci] += cln1_va[:, k] + cln2_va[:, k]
        cln_cum_te[:, ci] += cln1_te[:, k] + cln2_te[:, k]

    X_cln_va = build_compact_clean_gate_features(models, 'va', va_idx, p0_q_va, p1_q_va, p2_q_va, cln0_va, cln1_va, cln2_va, q_feats, deqa_res)
    X_cln_te = build_compact_clean_gate_features(models, 'te', te_idx, p0_q_te, p1_q_te, p2_q_te, cln0_te, cln1_te, cln2_te, q_feats, deqa_res)
    pcln_va, pcln_te = fit_compact_clean_gate(X_cln_va, cln_cum_va, gva, src_va, X_cln_te, cln_cum_te)
    # Blend analytical QualiRAG clean posterior directly into pcln!
    pcln_va = 0.80 * pcln_va + 0.10 * deqa_res['va_cln'] + 0.10 * knn_cln_va
    pcln_te = 0.80 * pcln_te + 0.10 * deqa_res['te_cln'] + 0.10 * knn_cln_te

    gprm = tune_p0_protected_gate_on_val(p0_q_va, p1_q_va, p2_q_va, cln0_va, cln1_va, cln2_va, pcln_va, gva)
    pred_te = decode_p0_protected_two_stage(p0_q_te, p1_q_te, p2_q_te, cln0_te, cln1_te, cln2_te, pcln_te, gprm)
    r_q_s2 = score_both(gte, pred_te)
    print(f"[s{sp}] V9 + Analytical QualiRAG + Stage-2 Gate (ZERO 400-row channel regression!): Doc2={r_q_s2['total_doc2']:.2f} | Doc1={r_q_s2['total']:.2f} (S_pos={r_q_s2['S_pos']:.3f}, S_p0={r_q_s2['S_p0']:.3f}, Clean2={r_q_s2['S_clean_doc2']:.3f})")
