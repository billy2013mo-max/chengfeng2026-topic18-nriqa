import os, sys, warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from cv_decoder_calib import decode_pc, HAND
from optimize_v8_isolated_test import load_healthy_split_models
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from run_scheme1_scheme2_breakthrough import FULL_IDX, score_both
from fast_test_qualirag_preserved_p0 import (
    CACHE_PATH, build_p0_preserved_v9_14ch,
    decode_p0_protected_two_stage, tune_p0_protected_gate_on_val
)
from ablate_clean_gate_and_qualirag_lowdim import build_compact_clean_gate_features, fit_compact_clean_gate

cache = torch.load(CACHE_PATH, weights_only=False)

rows = []
for sp in (16, 17):
    models, gva, gte = load_healthy_split_models(sp)
    c_sp = cache['splits'][sp]
    va_idx, te_idx, src_va = c_sp['va_idx'], c_sp['te_idx'], c_sp['src_va']
    q_feats, deqa_res = c_sp['q_feats'], c_sp['deqa_res']

    p0_v8_va, p1_v8_va, p2_v8_va, cln0_v8_va, cln1_v8_va, cln2_v8_va = build_decoupled_14ch_probs(models, 'va')
    p0_v8_te, p1_v8_te, p2_v8_te, cln0_v8_te, cln1_v8_te, cln2_v8_te = build_decoupled_14ch_probs(models, 'te')
    r_v8 = score_both(gte, decode_pc(p0_v8_te, p1_v8_te, p2_v8_te, HAND))

    _, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va = build_p0_preserved_v9_14ch(models, 'va', None)
    _, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te = build_p0_preserved_v9_14ch(models, 'te', None)

    # Combine V8's 0.818 P0 with V9's 0.628 P1/P2!
    p0_va, p0_te = p0_v8_va.copy(), p0_v8_te.copy()
    r_v89 = score_both(gte, decode_pc(p0_te, p1_v9_te, p2_v9_te, HAND))

    cln_cum_va = cln0_va.copy()
    cln_cum_te = cln0_te.copy()
    for k, ci in enumerate(FULL_IDX):
        cln_cum_va[:, ci] += cln1_va[:, k] + cln2_va[:, k]
        cln_cum_te[:, ci] += cln1_te[:, k] + cln2_te[:, k]

    # Scheme 2 Clean Gate without QualiRAG
    X_cln_va_noq = build_compact_clean_gate_features(models, 'va', va_idx, p0_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va)
    X_cln_te_noq = build_compact_clean_gate_features(models, 'te', te_idx, p0_te, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te)
    pcln_va_noq, pcln_te_noq = fit_compact_clean_gate(X_cln_va_noq, cln_cum_va, gva, src_va, X_cln_te_noq, cln_cum_te)
    gprm_noq = tune_p0_protected_gate_on_val(p0_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va, pcln_va_noq, gva)
    r_s2 = score_both(gte, decode_p0_protected_two_stage(p0_te, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te, pcln_te_noq, gprm_noq))

    # Scheme 1 + Scheme 2: QualiRAG Clean Gate (injecting Scene Physical Residual Z-score + Multi-Foundation k-NN Clean Posterior + DeQAScore Clean Head into Stage-2A Gate!)
    X_cln_va_q = build_compact_clean_gate_features(models, 'va', va_idx, p0_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va, q_feats, deqa_res)
    X_cln_te_q = build_compact_clean_gate_features(models, 'te', te_idx, p0_te, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te, q_feats, deqa_res)
    pcln_va_q, pcln_te_q = fit_compact_clean_gate(X_cln_va_q, cln_cum_va, gva, src_va, X_cln_te_q, cln_cum_te)
    gprm_q = tune_p0_protected_gate_on_val(p0_va, p1_v9_va, p2_v9_va, cln0_va, cln1_va, cln2_va, pcln_va_q, gva)
    r_s12 = score_both(gte, decode_p0_protected_two_stage(p0_te, p1_v9_te, p2_v9_te, cln0_te, cln1_te, cln2_te, pcln_te_q, gprm_q))

    print(f"\n=== Split s{sp} Blind Test (1,000 Images, Zero Leakage) ===")
    for tag, r in [
        ("0. V8 Decoupled 14ch (HAND)", r_v8),
        ("1. V8-P0 + V9-Grade 14ch (HAND)", r_v89),
        ("2. + Scheme 2 (Two-Stage All-Clean Gate)", r_s2),
        ("3. + Scheme 1 & 2 (QualiRAG Clean Gate + 2-Stage)", r_s12),
    ]:
        print(f"  {tag:50s} | Doc2={r['total_doc2']:5.2f} (S_pos={r['S_pos']:.3f}, S_p0={r['S_p0']:.3f}, Clean2={r['S_clean_doc2']:.3f}) | Doc1={r['total']:5.2f} (Clean1={r['S_clean']:.3f})")
    rows.append((r_v8, r_v89, r_s2, r_s12))

print("\n=== 2-SPLIT MEAN (s16_test + s17_test = 2,000 Blind Evaluations) ===")
for idx_m, tag in enumerate([
    "0. V8 Decoupled 14ch (HAND)",
    "1. V8-P0 + V9-Grade 14ch (HAND)",
    "2. + Scheme 2 (Two-Stage All-Clean Gate)",
    "3. + Scheme 1 & 2 (QualiRAG Clean Gate + 2-Stage)",
]):
    r16, r17 = rows[0][idx_m], rows[1][idx_m]
    print(f"  {tag:50s} | Mean Doc2 = {0.5*(r16['total_doc2']+r17['total_doc2']):5.2f} "
          f"(S_pos={0.5*(r16['S_pos']+r17['S_pos']):.3f}, S_p0={0.5*(r16['S_p0']+r17['S_p0']):.3f}, Clean2={0.5*(r16['S_clean_doc2']+r17['S_clean_doc2']):.3f}) | "
          f"Mean Doc1 = {0.5*(r16['total']+r17['total']):5.2f} (Clean1={0.5*(r16['S_clean']+r17['S_clean']):.3f})")
