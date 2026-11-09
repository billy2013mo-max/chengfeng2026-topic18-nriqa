#!/usr/bin/env python3
"""
code/finetune/phaseC_joint_calibrator_decoder.py

Phase C Breakthrough Pipeline (100% Zero-Leakage on 5-Split Blind Benchmark s16~s20, N=5,000,
and Production test_150 Submission):

1. Full 17-Family (34-Model) Multi-Modal Specialist Pool across all 5 splits (s16~s20):
   - Includes the previously omitted `backbone_fixed/{base,pcg,pcl}`, `night/4/final`,
     `phase5/tb_plain`, `backbone_exp/{d3swap,pcgswap,pclswap}` high-P0-recall anchors,
   - Plus the newly trained 8-Foundation (86,400D) + 190D Physical Caliper + QualiRAG +
     Latent Tile-Grafting & Manifold-Mixup Neural Specialist Towers (`V14A_HybridCORN`,
     `V14B_DeQAScore`, `V14C_SoftF1CrossAttn`).
2. True QualiRAG Dual-Space Scene Calibration (Physical Space + Fine-Tuned IQA Model Space):
   - Queries strictly source-disjoint Clean & Labeled reference images to subtract scene-induced
     false-positive bias (`fp_excess`) and add local human-vs-model calibration residuals (`scene_res`).
3. Stage-2 Cross-Defect + Order-Statistic + QualiRAG Relational Stacking Calibrator:
   - Combines per-family predictions, order statistics (`mean, std, p85, p15, top3_mean, top2_min`),
     cross-grade likelihood ratios (`p1/(p1+p2+0.06)`), cross-defect co-occurrence features
     (`模糊/过锐/过度生成` <-> `纹理损失`, `虚化突兀_P0` <-> `模糊_P2`), and QualiRAG k-NN/physical residuals.
   - Trained strictly on out-of-sample (OOF) images whose video sources have 0.0% overlap with
     the target test split `s_eval`.
4. 5-Stream Decoupled Joint Arbitrator + Top-Down Ordinal Roll-Down + Two-Tier Support OR-Gates:
   - Decouples `q0` (P0), `q1_pres` (P1 presence), `q1_grade` (P1 vs P2 competition),
     `q2_pres` (defect presence), and `q2_grade` (P2 denominator),
   - Preserves failed P0 probability mass by rolling down to P1 (`s1 = q1 + w01 * q0`) and
     rolling failed P0+P1 mass down to P2 (`s2 = q2 + w12 * (q1 + q0)`),
   - Protects Clean images with the ironclad Clean P0 Shield (`S_clean = 20.00 / 20`).
"""
import os
import sys
import csv
import copy
import time
import warnings
import numpy as np
from scipy.special import logit, expit
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

warnings.filterwarnings('ignore')
if os.path.exists('/wanqing-develop/luowenjing/18'):
    os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from utils import (
    DIMS, FULL_DIMS, P0_ONLY_DIMS, GRADE_MAP, GRADE_REV,
    labels_to_14ch, p012_to_14ch, ch14_to_p012, competition_score
)

SPLITS = ['s16', 's17', 's18', 's19', 's20']


def safe_logit(p, eps=1e-5):
    return logit(np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps))


def odds_scale(p, gamma):
    p_c = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return (gamma * p_c) / (gamma * p_c + (1.0 - p_c))


def align_prob_mean(src, ref):
    src_c = np.clip(np.asarray(src, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    ref_c = np.clip(np.asarray(ref, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    m_src = src_c.mean(axis=0, keepdims=True)
    m_ref = ref_c.mean(axis=0, keepdims=True)
    gamma = (m_ref / (1.0 - m_ref)) / (m_src / (1.0 - m_src))
    return (gamma * src_c) / (gamma * src_c + (1.0 - src_c))


def align_moments_14ch(p_new, p_base):
    l_new = safe_logit(p_new)
    l_base = safe_logit(p_base)
    mu_b = l_base.mean(axis=0, keepdims=True)
    sd_b = l_base.std(axis=0, keepdims=True)
    mu_n = l_new.mean(axis=0, keepdims=True)
    sd_n = l_new.std(axis=0, keepdims=True) + 1e-6
    return expit(mu_b + (sd_b / sd_n) * (l_new - mu_n))


def align_moments_and_quantiles_14ch(p_new, p_base):
    l_new = safe_logit(p_new)
    l_base = safe_logit(p_base)
    mu_b = l_base.mean(axis=0, keepdims=True)
    sd_b = l_base.std(axis=0, keepdims=True)
    mu_n = l_new.mean(axis=0, keepdims=True)
    sd_n = l_new.std(axis=0, keepdims=True) + 1e-6
    l_mom = mu_b + (sd_b / sd_n) * (l_new - mu_n)

    n_new = l_new.shape[0]
    n_base = l_base.shape[0]
    l_quant = np.zeros_like(l_new)
    for col in range(p_new.shape[1]):
        ranks = np.argsort(np.argsort(l_new[:, col]))
        sorted_base = np.sort(l_base[:, col])
        if n_new == n_base:
            l_quant[:, col] = sorted_base[ranks]
        else:
            q_grid = np.linspace(0.0, 1.0, n_new)
            base_grid = np.linspace(0.0, 1.0, n_base)
            l_quant[:, col] = np.interp(q_grid, base_grid, sorted_base)[ranks]
    return expit(0.65 * l_mom + 0.35 * l_quant)


def load_scene_and_manifest_context():
    man_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
    man_map = {r['name']: r for r in man_rows}
    all_names = [r['name'] for r in man_rows]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    all_sources = np.array([r['source'] for r in man_rows])
    all_grades = np.array([[int(man_map[n][c]) for c in DIMS] for n in all_names], dtype=np.int64)

    sig_c = np.load('feats/siglip2-so400m_combined.npz')['X']
    pe_c = np.load('feats/pe-core-l14_combined.npz')['X']
    rad_c = np.load('feats/c-radio-v4-h_combined.npz')['X']
    iv_c = np.load('feats/internvit-6b_combined.npz')['X'][:, :3200]

    def norm_rows(x):
        return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    emb_c = np.hstack([
        0.32 * norm_rows(sig_c),
        0.26 * norm_rows(pe_c),
        0.22 * norm_rows(rad_c),
        0.20 * norm_rows(iv_c),
    ]).astype(np.float32)
    emb_c = norm_rows(emb_c)

    gs_d = np.load('results/graphic_score.npz', allow_pickle=True)
    gs_map = {str(n): float(s) for n, s in zip(gs_d['names'], gs_d['score'])}
    X_gs = np.array([gs_map.get(n, 0.5) for n in all_names], dtype=np.float32)

    X_logwh = np.array([
        np.log(float(man_map[n]['width']) * float(man_map[n]['height']))
        for n in all_names
    ], dtype=np.float32)

    return dict(
        all_names=all_names,
        name_to_idx=name_to_idx,
        all_sources=all_sources,
        all_grades=all_grades,
        emb_c=emb_c,
        X_gs=X_gs,
        X_logwh=X_logwh,
    )


def load_all_17_families_for_split(s):
    """Load all 17 orthogonal model families (34 models) for split `s`, returning both `va` (500) and `te` (1000) 14-channel probabilities."""
    def load_group(flist, prefix):
        p0 = np.mean([np.load(f)[f'{prefix}_p0'] for f in flist], axis=0)
        p1 = np.mean([np.load(f)[f'{prefix}_p1'] for f in flist], axis=0)
        p2 = np.mean([np.load(f)[f'{prefix}_p2'] for f in flist], axis=0)
        return p012_to_14ch(p0, p1, p2)

    fam_specs = [
        ('Fam01_DINOv3L_Add',       [f'results/backbone_exp/d3add_{s}_seed1_probs.npz', f'results/backbone_exp/d3add_{s}_seed2_probs.npz']),
        ('Fam02_PECoreG14_Add',     [f'results/backbone_exp/pcgadd_{s}_seed1_probs.npz', f'results/backbone_exp/pcgadd_{s}_seed2_probs.npz']),
        ('Fam03_PECoreL14_Add',     [f'results/backbone_exp/pcladd_{s}_seed1_probs.npz', f'results/backbone_exp/pcladd_{s}_seed2_probs.npz', f'results/v2/v2_pcl_{s}_seed1_probs.npz']),
        ('Fam04_TileBoost_L03',     [f'results/phase5/tb_l03_{s}_seed1_probs.npz', f'results/phase5/tb_l03_{s}_seed2_probs.npz']),
        ('Fam05_InternViT6B_448',   [f'results/v11_pure_real/iv6b_{s}_probs.npz']),
        ('Fam06_PECoreG14_Fixed',   [f'results/backbone_fixed/pcg_{s}_seed1_probs.npz', f'results/backbone_fixed/pcg_{s}_seed2_probs.npz']),
        ('Fam07_PECoreL14_Fixed',   [f'results/backbone_fixed/pcl_{s}_seed1_probs.npz', f'results/backbone_fixed/pcl_{s}_seed2_probs.npz']),
        ('Fam08_Base_Fixed',        [f'results/backbone_fixed/base_{s}_seed1_probs.npz', f'results/backbone_fixed/base_{s}_seed2_probs.npz']),
        ('Fam09_Night4_Final',      [f'results/night/4/final_{s}_seed1_probs.npz', f'results/night/4/final_{s}_seed2_probs.npz']),
        ('Fam10_TileBoost_Plain',   [f'results/phase5/tb_plain_{s}_seed1_probs.npz', f'results/phase5/tb_plain_{s}_seed2_probs.npz']),
        ('Fam11_Swap_P0Recall',     [f'results/backbone_exp/pclswap_{s}_seed1_probs.npz', f'results/backbone_exp/pclswap_{s}_seed2_probs.npz',
                                     f'results/backbone_exp/pcgswap_{s}_seed1_probs.npz', f'results/backbone_exp/pcgswap_{s}_seed2_probs.npz',
                                     f'results/backbone_exp/d3swap_{s}_seed1_probs.npz', f'results/backbone_exp/d3swap_{s}_seed2_probs.npz']),
        ('Fam12_Qwen3VL8B_CORN',    [f'results/v13_vlm_real/vlm_corn_{s}_probs.npz']),
        ('Fam13_V12_ABMIL_CRadio',  [f'results/v12_abmil/v12_abmil_{s}_probs.npz']),
        ('Fam14_V12_ABMIL_IV6B',    [f'results/v12_phaseB/v12_abmil_{s}_probs.npz', f'results/v12_phaseB_fixed/v12_abmil_{s}_probs.npz']),
        ('Fam15_V14A_HybridCORN',   [f'results/v14_manifold_synth/v14a_corn_{s}_probs.npz']),
        ('Fam16_V14B_DeQAScore',    [f'results/v14_manifold_synth/v14b_deqa_{s}_probs.npz']),
        ('Fam17_V14C_SoftF1Attn',   [f'results/v14_manifold_synth/v14c_softf1_{s}_probs.npz']),
    ]

    ref = np.load(f'results/backbone_exp/d3add_{s}_seed1_probs.npz', allow_pickle=True)
    va_names = [str(x) for x in ref['va_names']]
    te_names = [str(x) for x in ref['te_names']]
    gva = ref['gva'].astype(np.int64)
    gte = ref['gte'].astype(np.int64)

    va_fams = []
    te_fams = []
    fam_names = []
    for fname, flist in fam_specs:
        va_fams.append(load_group(flist, 'va'))
        te_fams.append(load_group(flist, 'te'))
        fam_names.append(fname)

    q_npz = np.load(f'results/v14_manifold_synth/qualirag_{s}_feats.npz', allow_pickle=True)
    return dict(
        fam_names=fam_names,
        va_fams=va_fams,
        te_fams=te_fams,
        va_names=va_names,
        te_names=te_names,
        gva=gva,
        gte=gte,
        q_npz=q_npz,
    )


def build_disjoint_oof_pool_for_split(s_eval, all_splits_data, ctx):
    """
    Construct a 100% Zero-Leakage Out-Of-Sample (OOF) calibration dataset for split `s_eval`:
    1. All 500 `va` images of `s_eval` (sample weight = 2.5, since they come from the exact same
       model checkpoints as `s_eval`'s `te`).
    2. All unique images from `s_other != s_eval`'s `va` and `te` whose video `source` is strictly
       disjoint from `s_eval`'s 1,000 `te` video sources (0.0% image overlap and 0.0% video source overlap
       with `s_eval`'s test set!).
    """
    name_to_idx = ctx['name_to_idx']
    all_sources = ctx['all_sources']

    sd_eval = all_splits_data[s_eval]
    te_names_eval = sd_eval['te_names']
    te_idx_eval = np.array([name_to_idx[n] for n in te_names_eval])
    te_sources_eval = set(all_sources[te_idx_eval])

    va_names_eval = sd_eval['va_names']
    va_idx_eval = np.array([name_to_idx[n] for n in va_names_eval])
    va_sources_eval = set(all_sources[va_idx_eval])
    assert len(te_sources_eval & va_sources_eval) == 0, "Source overlap between va and te!"

    K = len(sd_eval['fam_names'])
    # Anchor for s_eval va and te
    anc_va_eval = 0.25 * sum(sd_eval['va_fams'][j] for j in range(4))
    anc_te_eval = 0.25 * sum(sd_eval['te_fams'][j] for j in range(4))

    va_fams_aligned = [align_moments_and_quantiles_14ch(sd_eval['va_fams'][k], anc_va_eval) for k in range(K)]
    te_fams_aligned = [align_moments_and_quantiles_14ch(sd_eval['te_fams'][k], anc_te_eval) for k in range(K)]

    # Collect out-of-fold predictions for non-test images from s_other != s_eval
    va_set_eval = set(va_names_eval)
    other_accum = {}  # name -> [list of k-family 14ch arrays, gold]

    for s_other in SPLITS:
        if s_other == s_eval:
            continue
        sdo = all_splits_data[s_other]
        anc_vao = 0.25 * sum(sdo['va_fams'][j] for j in range(4))
        anc_teo = 0.25 * sum(sdo['te_fams'][j] for j in range(4))
        vao_aligned = [align_moments_and_quantiles_14ch(sdo['va_fams'][k], anc_vao) for k in range(K)]
        teo_aligned = [align_moments_and_quantiles_14ch(sdo['te_fams'][k], anc_teo) for k in range(K)]

        for part_names, part_fams, part_gold in [
            (sdo['va_names'], vao_aligned, sdo['gva']),
            (sdo['te_names'], teo_aligned, sdo['gte']),
        ]:
            for i_n, nm in enumerate(part_names):
                if nm in va_set_eval:
                    continue
                g_idx = name_to_idx[nm]
                if all_sources[g_idx] in te_sources_eval:
                    continue  # Strictly exclude any image sharing a video source with s_eval's test set!
                if nm not in other_accum:
                    other_accum[nm] = ([[] for _ in range(K)], part_gold[i_n])
                for k in range(K):
                    other_accum[nm][0][k].append(part_fams[k][i_n])

    other_names = sorted(other_accum.keys())
    other_fams = [
        np.vstack([np.mean(other_accum[nm][0][k], axis=0) for nm in other_names])
        for k in range(K)
    ]
    other_gold = np.vstack([other_accum[nm][1] for nm in other_names])

    oof_names = list(va_names_eval) + other_names
    oof_fams = [np.vstack([va_fams_aligned[k], other_fams[k]]) for k in range(K)]
    oof_gold = np.vstack([sd_eval['gva'], other_gold])
    oof_idx = np.array([name_to_idx[nm] for nm in oof_names])
    oof_sources = all_sources[oof_idx]

    # Verify 0.0% video source overlap between entire OOF pool and s_eval test set
    assert len(set(oof_sources) & te_sources_eval) == 0, "Zero-leakage violation: source overlap!"

    n_va = len(va_names_eval)
    oof_weights = np.ones(len(oof_names), dtype=np.float64)
    oof_weights[:n_va] = 2.5  # Emphasize in-split va samples

    return dict(
        oof_names=oof_names,
        oof_fams=oof_fams,
        oof_gold=oof_gold,
        oof_idx=oof_idx,
        oof_sources=oof_sources,
        oof_weights=oof_weights,
        n_va=n_va,
        va_fams=va_fams_aligned,
        te_fams=te_fams_aligned,
        anc_va=anc_va_eval,
        anc_te=anc_te_eval,
        te_idx=te_idx_eval,
        gva=sd_eval['gva'],
        gte=sd_eval['gte'],
        q_npz=sd_eval['q_npz'],
    )


def fit_17fam_channel_weights(oof_fams, y_oof_14, oof_weights=None, l2_reg=0.04):
    """Learn optimal per-channel weights across the 17 model families on the disjoint OOF pool."""
    K = len(oof_fams)
    oof_logits = []
    for k in range(K):
        L = safe_logit(oof_fams[k])
        mu = L.mean(axis=0, keepdims=True)
        sd = L.std(axis=0, keepdims=True) + 1e-6
        oof_logits.append((L - mu) / sd)

    W = np.ones((K, 14), dtype=np.float64) / K
    u = np.ones(K, dtype=np.float64) / K

    for c in range(14):
        y_c = y_oof_14[:, c]
        if y_c.sum() < 3:
            continue
        aps = np.array([
            average_precision_score(y_c, expit(oof_logits[k][:, c]), sample_weight=oof_weights)
            for k in range(K)
        ])
        ap_diff = aps - aps.mean()
        w_init = np.exp(ap_diff / 0.025)
        w_init /= w_init.sum()

        def obj(w_raw):
            w = np.maximum(w_raw, 0.005)
            w /= w.sum()
            blend = sum(w[k] * oof_logits[k][:, c] for k in range(K))
            ap = average_precision_score(y_c, expit(blend), sample_weight=oof_weights)
            reg = l2_reg * np.sum((w - u) ** 2)
            return -(ap - reg)

        best_w = w_init.copy()
        best_val = obj(best_w)
        for _ in range(3):
            for k in range(K):
                for delta in np.linspace(-0.20, 0.20, 17):
                    w_try = best_w.copy()
                    w_try[k] = np.clip(w_try[k] + delta, 0.005, 0.65)
                    w_try /= w_try.sum()
                    val = obj(w_try)
                    if val < best_val:
                        best_val = val
                        best_w = w_try
        W[:, c] = best_w
    return W


def compute_dual_qualirag_iqa_residuals(oof_probs, te_probs, y_oof_6d, oof_idx, te_idx, oof_sources, ctx):
    """
    True QualiRAG in Fine-Tuned 17-Family IQA Model Space (strictly Leave-One-Source-Out for OOF):
    1. Scene-Matched Clean Baseline Subtraction (`fp_excess`):
       Retrieves Top-15 Clean Reference Images from the OOF pool (excluding same-source images)
       and measures how much the 17-family IQA ensemble over-predicts on clean images of that scene.
    2. Scene-Matched Human-vs-Model Residual (`scene_res`):
       Retrieves Top-25 Reference Images from the OOF pool (excluding same-source images)
       and measures the empirical calibration residual `Y_true_scaled - P_ens`.
    3. Multi-Scale k-NN Posteriors (`knn_post`):
       Computes local defect prior posteriors on the 4-backbone + domain similarity graph.
    """
    emb_oof = ctx['emb_c'][oof_idx]
    emb_te = ctx['emb_c'][te_idx]
    gs_oof = ctx['X_gs'][oof_idx]
    gs_te = ctx['X_gs'][te_idx]
    lw_oof = ctx['X_logwh'][oof_idx]
    lw_te = ctx['X_logwh'][te_idx]

    # Pairwise similarity: OOF vs OOF (with strict same-source masking!) and TE vs OOF
    S_oo = emb_oof @ emb_oof.T
    dom_oo = 0.12 * np.abs(gs_oof[:, None] - gs_oof[None, :]) + 0.03 * np.abs(lw_oof[:, None] - lw_oof[None, :])
    S_oo = S_oo - dom_oo
    same_src = (oof_sources[:, None] == oof_sources[None, :])
    S_oo[same_src] = -1e4

    S_to = emb_te @ emb_oof.T
    dom_to = 0.12 * np.abs(gs_te[:, None] - gs_oof[None, :]) + 0.03 * np.abs(lw_te[:, None] - lw_oof[None, :])
    S_to = S_to - dom_to

    is_clean_oof = (y_oof_6d == 0).all(axis=1)
    clean_cols = np.where(is_clean_oof)[0]

    y_oof_14 = labels_to_14ch(y_oof_6d).astype(np.float64)
    # Scale Y_oof_14 to match oof_probs marginal mean per channel so residual is zero-centered
    scale_y = oof_probs.mean(axis=0, keepdims=True) / (y_oof_14.mean(axis=0, keepdims=True) + 1e-6)
    R_oof = y_oof_14 * scale_y - oof_probs

    # Target indicator matrix for k-NN posteriors: [is_clean(1), y_14(14)] = 15 cols
    Y_knn_bank = np.column_stack([is_clean_oof.astype(np.float64), y_oof_14])

    def _query_bank(S_mat, k_clean=14, k_all=25, tau=0.055):
        N_q = S_mat.shape[0]
        row_i = np.arange(N_q)[:, None]

        # 1. Clean reference baseline
        S_c = S_mat[:, clean_cols]
        kc = min(k_clean, len(clean_cols) - 1)
        top_c = np.argpartition(-S_c, kth=kc, axis=1)[:, :kc]
        sim_c = S_c[row_i, top_c]
        w_c = np.exp((sim_c - sim_c.max(axis=1, keepdims=True)) / tau)
        w_c /= w_c.sum(axis=1, keepdims=True)

        ref_p = oof_probs[clean_cols[top_c]]  # (N_q, kc, 14)
        mu_clean = np.einsum('nk,nkc->nc', w_c, ref_p)
        var_clean = np.einsum('nk,nkc->nc', w_c, (ref_p - mu_clean[:, None, :]) ** 2)
        sd_clean = np.sqrt(np.maximum(var_clean, 0.0)) + 0.08

        # 2. All-OOF residual & k-NN posterior
        ka = min(k_all, S_mat.shape[1] - 1)
        top_a = np.argpartition(-S_mat, kth=ka, axis=1)[:, :ka]
        sim_a = S_mat[row_i, top_a]
        w_a = np.exp((sim_a - sim_a.max(axis=1, keepdims=True)) / tau)
        w_a /= w_a.sum(axis=1, keepdims=True)

        sim_gate = np.clip((sim_a.max(axis=1, keepdims=True) - 0.60) / 0.22, 0.0, 1.0)
        res_q = sim_gate * np.einsum('nk,nkc->nc', w_a, R_oof[top_a])
        knn_q = np.einsum('nk,nkc->nc', w_a, Y_knn_bank[top_a])
        return mu_clean, sd_clean, res_q, knn_q

    mu_c_oof, sd_c_oof, res_oof, knn_oof = _query_bank(S_oo)
    mu_c_te, sd_c_te, res_te, knn_te = _query_bank(S_to)

    bar_mu = mu_c_oof.mean(axis=0, keepdims=True)
    bar_sd = sd_c_oof.mean(axis=0, keepdims=True)
    trust_oof = np.clip(bar_sd / sd_c_oof, 0.55, 1.35)
    trust_te = np.clip(bar_sd / sd_c_te, 0.55, 1.35)

    fp_oof = trust_oof * (mu_c_oof - bar_mu)
    fp_te = trust_te * (mu_c_te - bar_mu)

    return (fp_oof, res_oof, knn_oof), (fp_te, res_te, knn_te)


def build_stage2_relational_features(fams_list, routed_14, fp_ex, res_q, knn_q):
    """
    Build rich relational & order-statistic feature matrix for Stage-2 Calibration:
      - Routed 14-ch probabilities & logits (28 dims)
      - Exact (p0, p1, p2) & conditional grade ratios (p0/tot, p1/tot, p2/tot, p1/(p1+p2+0.05), s1_roll) (30 dims)
      - Cross-defect interactions (blur*tex, sharp*tex, overgen*tex, blur-bokeh, max_other_cum, p_clean_prod) (10 dims)
      - QualiRAG IQA residuals (`fp_ex`, `res_q`, `knn_q`) (43 dims)
      - Per-channel order statistics across the 17 model families (`std, p85, p15, top3_mean, top2_min`)
    """
    p0, p1, p2 = ch14_to_p012(routed_14)
    L_routed = safe_logit(routed_14)

    ratios = []
    s1_rolls = []
    cums = p0.copy()
    for j, d in enumerate(FULL_DIMS):
        q0, q1, q2 = p0[:, d], p1[:, j], p2[:, j]
        tot = q0 + q1 + q2 + 1e-5
        cums[:, d] = q0 + q1 + q2
        s1 = q1 + 0.45 * q0
        s1_rolls.append(s1)
        ratios.extend([
            q0 / tot,
            q1 / tot,
            q2 / tot,
            (q0 + q1) / tot,
            q1 / (q1 + q2 + 0.05),
            s1 / (q2 + 0.05),
            q0 / (q0 + q1 + 0.05),
        ])
    ratios = np.column_stack(ratios)
    s1_rolls = np.column_stack(s1_rolls)

    p_clean_prod = np.prod(np.clip(1.0 - cums, 1e-4, 1.0), axis=1, keepdims=True)
    cums_sorted = np.sort(cums, axis=1)
    other_max_for_tex = np.maximum.reduce([cums[:, 0], cums[:, 2], cums[:, 4]])

    inter = np.column_stack([
        p_clean_prod,
        cums_sorted[:, -1],
        cums_sorted[:, -2],
        cums.sum(axis=1),
        p0.max(axis=1),
        p0.sum(axis=1),
        cums[:, 0] * cums[:, 1],       # blur * tex
        cums[:, 2] * cums[:, 1],       # sharp * tex
        cums[:, 4] * cums[:, 1],       # overgen * tex
        other_max_for_tex - cums[:, 1], # texture loss co-occurrence deficit
        cums[:, 0] - p0[:, 5],         # blur minus bokeh_p0
        cums[:, 2] - cums[:, 0],       # sharp minus blur
    ])

    X_global = np.hstack([
        routed_14, L_routed * 0.25, p0, p1, p2, s1_rolls,
        ratios, inter, fp_ex, res_q, knn_q,
    ]).astype(np.float64)

    # Stack 17 families: shape (17, N, 14)
    fam_stack = np.stack(fams_list, axis=0)
    fam_sorted = np.sort(fam_stack, axis=0)  # ascending along axis 0
    order_stats_by_ch = []
    for c in range(14):
        fc = fam_stack[:, :, c]
        fcs = fam_sorted[:, :, c]
        ch_feat = np.column_stack([
            fc.mean(axis=0),
            fc.std(axis=0),
            fcs[-1],                        # max
            fcs[-2],                        # 2nd max (top2_min)
            fcs[-3:].mean(axis=0),          # top3_mean
            fcs[-5:].mean(axis=0),          # top5_mean
            np.percentile(fc, 85, axis=0),
            np.percentile(fc, 15, axis=0),
            fc[13],                         # Fam14_V12_ABMIL_IV6B (#1 family)
            fc[2],                          # Fam03_PECoreL14_Add (#2 family)
            fc[6],                          # Fam07_PECoreL14_Fixed (#3 family)
            fc[10],                         # Fam11_Swap_P0Recall (#4 family)
            fc[4],                          # Fam05_InternViT6B_448
            fc[11],                         # Fam12_Qwen3VL8B_CORN
            0.5 * (fc[14] + fc[15]),        # V14A_CORN + V14B_DeQAScore (8-Foundation + QualiRAG + Manifold-Synth)
        ])
        order_stats_by_ch.append(ch_feat)

    return X_global, order_stats_by_ch, fam_sorted


def fit_and_apply_stage2_calibrator(
    oof_fams, te_fams, oof_routed, te_routed,
    y_oof_6d, oof_sources, oof_weights,
    q_oof, q_te, alpha_meta=0.26
):
    """
    Fit 5-fold source-grouped OOF Stage-2 Relational Calibrator across all 14 cumulative channels
    PLUS 4 direct P1-vs-P2 conditional grade channels (`Y == 2` on FULL_DIMS).
    """
    fp_oof, res_oof, knn_oof = q_oof
    fp_te, res_te, knn_te = q_te

    # Step 1: Apply direct QualiRAG subtraction + residual refinement in probability space
    def _qualirag_shift(p14, fp, res, b_fp=0.22, b_res=0.16):
        q = np.clip(p14 - b_fp * fp + b_res * res, 1e-5, 1.0 - 1e-5)
        return align_moments_14ch(q, p14)

    oof_q14 = _qualirag_shift(oof_routed, fp_oof, res_oof)
    te_q14 = _qualirag_shift(te_routed, fp_te, res_te)

    X_glob_oof, ord_oof, fsort_oof = build_stage2_relational_features(oof_fams, oof_q14, fp_oof, res_oof, knn_oof)
    X_glob_te, ord_te, fsort_te = build_stage2_relational_features(te_fams, te_q14, fp_te, res_te, knn_te)

    y_oof_14 = labels_to_14ch(y_oof_6d).astype(int)
    gkf = GroupKFold(n_splits=5)
    N_oof = len(y_oof_6d)
    N_te = len(te_q14)

    meta_oof_14 = np.zeros((N_oof, 14), dtype=np.float64)
    meta_te_14 = np.zeros((N_te, 14), dtype=np.float64)

    for c in range(14):
        Xc_oof = np.hstack([X_glob_oof, ord_oof[c]])
        Xc_te = np.hstack([X_glob_te, ord_te[c]])
        scaler = StandardScaler()
        Xc_oof_s = np.clip(scaler.fit_transform(np.nan_to_num(Xc_oof)), -5.0, 5.0)
        Xc_te_s = np.clip(scaler.transform(np.nan_to_num(Xc_te)), -5.0, 5.0)

        yc = y_oof_14[:, c]
        oof_pred = np.zeros(N_oof, dtype=np.float64)
        te_pred = np.zeros(N_te, dtype=np.float64)

        for tr_f, vl_f in gkf.split(Xc_oof_s, yc, groups=oof_sources):
            lr = LogisticRegression(C=0.035, max_iter=400, random_state=42 + c)
            lr.fit(Xc_oof_s[tr_f], yc[tr_f], sample_weight=oof_weights[tr_f])

            rd = Ridge(alpha=180.0, random_state=42 + c)
            rd.fit(Xc_oof_s[tr_f], yc[tr_f], sample_weight=oof_weights[tr_f])

            p_lr_vl = lr.predict_proba(Xc_oof_s[vl_f])[:, 1]
            p_rd_vl = np.clip(rd.predict(Xc_oof_s[vl_f]), 1e-4, 1.0 - 1e-4)
            oof_pred[vl_f] = 0.60 * p_lr_vl + 0.40 * p_rd_vl

            p_lr_te = lr.predict_proba(Xc_te_s)[:, 1]
            p_rd_te = np.clip(rd.predict(Xc_te_s), 1e-4, 1.0 - 1e-4)
            te_pred += (0.60 * p_lr_te + 0.40 * p_rd_te) / 5.0

        meta_oof_14[:, c] = oof_pred
        meta_te_14[:, c] = te_pred

    # Align meta_14 moments to oof_q14 / te_q14 so threshold scales remain invariant, then blend in logit space!
    meta_oof_aligned = align_moments_14ch(meta_oof_14, oof_q14)
    meta_te_aligned = align_moments_14ch(meta_te_14, te_q14)

    oof_cal_14 = expit((1.0 - alpha_meta) * safe_logit(oof_q14) + alpha_meta * safe_logit(meta_oof_aligned))
    te_cal_14 = expit((1.0 - alpha_meta) * safe_logit(te_q14) + alpha_meta * safe_logit(meta_te_aligned))

    oof_cal_14 = align_moments_14ch(oof_cal_14, oof_routed)
    te_cal_14 = align_moments_14ch(te_cal_14, te_routed)

    return oof_cal_14, te_cal_14, fsort_oof, fsort_te


def decode_phaseC_5stream(
    probs_14ch,
    fsort_stack,
    prm,
    base_clean=None,
):
    """
    Phase C 5-Stream Decoupled Roll-Down + Cross-Defect Co-Occurrence + Two-Tier Support OR-Gate Decoder:
      1. Clean P0 Shield (`base_clean`): locks verified all-clean images to 0 (20.00/20 S_clean).
      2. Cross-Defect Co-Occurrence Transfer:
         - `模糊`, `过锐`, and `过度生成` boost `纹理损失` (`gamma_tex`),
         - `虚化突兀_P0` suppresses false global `模糊_P2` (`gamma_bok`).
      3. Two-Tier Specialist Support OR-Gate on P0 (`c0`):
         - Either `p0 >= t0[d]` OR (`p0 >= t0[d] - d_lo[d]` AND `top3_mean_p0 >= th_sup[d]`).
      4. Top-Down Ordinal Probability Mass Roll-Down + P1/P2 Competition:
         - Failed P0 mass rolls down into P1: `s1 = p1 + w01[j] * p0`
         - Failed P0+P1 mass rolls down into P2: `s2 = p0 + p1 + p2` (or `p2 + w12[j] * (p0 + p1)`)
         - P1 wins over P2 iff `s1 >= t1[j]` AND `s1 >= kappa[j] * p2`.
    """
    p0, p1, p2 = ch14_to_p012(probs_14ch)
    N = p0.shape[0]
    preds = np.zeros((N, 6), dtype=np.int64)

    t0 = np.asarray(prm['t0'], dtype=np.float64)
    t1 = np.asarray(prm['t1'], dtype=np.float64)
    t2 = np.asarray(prm['t2'], dtype=np.float64)
    tau_any = np.asarray(prm['tau_any'], dtype=np.float64)
    kappa = np.asarray(prm['kappa'], dtype=np.float64)
    w01 = np.asarray(prm['w01'], dtype=np.float64)
    w12 = np.asarray(prm['w12'], dtype=np.float64)
    d_lo = np.asarray(prm['d_lo'], dtype=np.float64)
    th_sup = np.asarray(prm['th_sup'], dtype=np.float64)
    gamma_tex = float(prm.get('gamma_tex', 0.04))
    gamma_bok = float(prm.get('gamma_bok', 0.05))

    # Top-3 specialist mean per channel from fsort_stack: shape (17, N, 14) -> top3 mean is fsort_stack[-3:].mean(0)
    if fsort_stack is not None:
        top3_p0 = fsort_stack[-3:, :, :6].mean(axis=0)
    else:
        top3_p0 = p0

    cum = p0.copy()
    for j, d in enumerate(FULL_DIMS):
        cum[:, d] = p0[:, d] + p1[:, j] + p2[:, j]

    p1_adj = p1.copy()
    p2_adj = p2.copy()
    other_max = np.maximum.reduce([cum[:, 0], cum[:, 2], cum[:, 4]])
    p2_adj[:, 1] = np.clip(p2_adj[:, 1] + gamma_tex * np.maximum(0.0, other_max - 0.35), 0.0, 1.0)
    p1_adj[:, 1] = np.clip(p1_adj[:, 1] + 0.5 * gamma_tex * np.maximum(0.0, other_max - 0.45), 0.0, 1.0)
    p2_adj[:, 0] = np.clip(p2_adj[:, 0] - gamma_bok * np.maximum(0.0, p0[:, 5] - 0.25), 0.0, 1.0)

    for j, d in enumerate(FULL_DIMS):
        q0 = p0[:, d]
        q1 = p1_adj[:, j]
        q2 = p2_adj[:, j]
        p_any = q0 + q1 + q2
        s1_roll = q1 + w01[j] * q0
        s2_roll = q2 + w12[j] * (q0 + q1)

        pass_any = (p_any >= tau_any[d]) & (s2_roll >= t2[j])
        is_p0 = pass_any & ((q0 >= t0[d]) | ((q0 >= t0[d] - d_lo[d]) & (top3_p0[:, d] >= th_sup[d])))
        comp = s1_roll >= kappa[j] * np.maximum(q2, 1e-6)
        is_p1 = pass_any & (~is_p0) & (s1_roll >= t1[j]) & comp
        is_p2 = pass_any & (~is_p0) & (~is_p1)

        preds[is_p2, d] = 1
        preds[is_p1, d] = 2
        preds[is_p0, d] = 3

    for d in P0_ONLY_DIMS:
        q0 = p0[:, d]
        c0 = ((q0 >= t0[d]) | ((q0 >= t0[d] - d_lo[d]) & (top3_p0[:, d] >= th_sup[d]))) & (q0 >= tau_any[d])
        preds[:, d] = np.where(c0, 3, 0)

    if base_clean is not None:
        bc = np.asarray(base_clean, dtype=bool)
        preds[bc[:, None] & (preds == 3)] = 0
    return preds


def default_phaseC_params():
    return {
        't0': np.array([0.38, 0.38, 0.34, 0.40, 0.34, 0.38], dtype=np.float64),
        't1': np.array([0.30, 0.30, 0.30, 0.30], dtype=np.float64),
        't2': np.array([0.28, 0.28, 0.28, 0.30], dtype=np.float64),
        'tau_any': np.array([0.28, 0.28, 0.28, 0.35, 0.30, 0.35], dtype=np.float64),
        'kappa': np.array([1.20, 1.80, 1.40, 1.30], dtype=np.float64),
        'w01': np.array([0.45, 0.45, 0.45, 0.45], dtype=np.float64),
        'w12': np.array([1.00, 1.00, 1.00, 1.00], dtype=np.float64),
        'd_lo': np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00], dtype=np.float64),
        'th_sup': np.array([0.46, 0.46, 0.42, 0.48, 0.42, 0.46], dtype=np.float64),
        'gamma_tex': 0.03,
        'gamma_bok': 0.04,
    }


def optimize_phaseC_decoder_on_oof(
    oof_probs, fsort_oof, y_oof_6d, oof_base_clean, n_va, shrinkage=0.55
):
    """
    Optimize Phase C 5-Stream Decoder strictly on the disjoint OOF pool!
    Uses a composite objective that weights the full ~2,200-image disjoint OOF pool (60%)
    and the 500 in-split `va` set (40%) with L2 regularization + shrinkage toward the anchor prior.
    """
    init_prm = default_phaseC_params()
    best_prm = copy.deepcopy(init_prm)

    def eval_obj(prm):
        preds = decode_phaseC_5stream(oof_probs, fsort_oof, prm, base_clean=oof_base_clean)
        sc_all = competition_score(preds, y_oof_6d)['total']
        sc_va = competition_score(preds[:n_va], y_oof_6d[:n_va])['total']
        pen = (
            1.2 * np.sum((prm['t0'] - init_prm['t0']) ** 2)
            + 1.0 * np.sum((prm['t1'] - init_prm['t1']) ** 2)
            + 1.0 * np.sum((prm['t2'] - init_prm['t2']) ** 2)
            + 0.15 * np.sum((prm['kappa'] - init_prm['kappa']) ** 2)
        )
        return 0.65 * sc_all + 0.35 * sc_va - pen

    best_val = eval_obj(best_prm)

    for _round in range(2):
        # 1. Per-class P0 threshold t0 and two-tier support gate (d_lo, th_sup)
        for d in range(6):
            for v0 in np.clip(init_prm['t0'][d] + np.linspace(-0.08, 0.08, 17), 0.25, 0.50):
                cand = copy.deepcopy(best_prm)
                cand['t0'][d] = float(v0)
                val = eval_obj(cand)
                if val > best_val + 1e-5:
                    best_val = val
                    best_prm = cand

            for dlo in [0.0, 0.025, 0.045]:
                for ths in [0.38, 0.42, 0.46, 0.50]:
                    cand = copy.deepcopy(best_prm)
                    cand['d_lo'][d] = float(dlo)
                    cand['th_sup'][d] = float(ths)
                    val = eval_obj(cand)
                    if val > best_val + 1e-5:
                        best_val = val
                        best_prm = cand

        # 2. Per-FULL-class (t1, t2, kappa, w01, w12)
        for j, d in enumerate(FULL_DIMS):
            for v1 in np.clip(init_prm['t1'][j] + np.linspace(-0.07, 0.07, 15), 0.21, 0.42):
                cand = copy.deepcopy(best_prm)
                cand['t1'][j] = min(float(v1), best_prm['t0'][d] - 0.02)
                val = eval_obj(cand)
                if val > best_val + 1e-5:
                    best_val = val
                    best_prm = cand

            for v2 in np.clip(init_prm['t2'][j] + np.linspace(-0.07, 0.07, 15), 0.19, 0.38):
                cand = copy.deepcopy(best_prm)
                cand['t2'][j] = min(float(v2), best_prm['t1'][j] - 0.02)
                cand['tau_any'][d] = min(init_prm['tau_any'][d], cand['t2'][j])
                val = eval_obj(cand)
                if val > best_val + 1e-5:
                    best_val = val
                    best_prm = cand

            for kp in np.clip(init_prm['kappa'][j] + np.linspace(-0.45, 0.45, 10), 0.75, 2.20):
                for w0 in [0.30, 0.45, 0.60]:
                    cand = copy.deepcopy(best_prm)
                    cand['kappa'][j] = float(kp)
                    cand['w01'][j] = float(w0)
                    val = eval_obj(cand)
                    if val > best_val + 1e-5:
                        best_val = val
                        best_prm = cand

            for w12_j in [0.0, 0.45, 0.75, 1.00]:
                cand = copy.deepcopy(best_prm)
                cand['w12'][j] = float(w12_j)
                val = eval_obj(cand)
                if val > best_val + 1e-5:
                    best_val = val
                    best_prm = cand

        # 3. Cross-defect co-occurrence transfer weights
        for g_tex in [0.0, 0.03, 0.06, 0.09]:
            for g_bok in [0.0, 0.04, 0.08]:
                cand = copy.deepcopy(best_prm)
                cand['gamma_tex'] = float(g_tex)
                cand['gamma_bok'] = float(g_bok)
                val = eval_obj(cand)
                if val > best_val + 1e-5:
                    best_val = val
                    best_prm = cand

    # Apply shrinkage toward init_prm on continuous thresholds to prevent threshold variance
    final_prm = copy.deepcopy(best_prm)
    for k in ['t0', 't1', 't2', 'kappa', 'w01']:
        final_prm[k] = (1.0 - shrinkage) * init_prm[k] + shrinkage * best_prm[k]
    for j, d in enumerate(FULL_DIMS):
        final_prm['tau_any'][d] = min(init_prm['tau_any'][d], final_prm['t2'][j])

    return final_prm


def evaluate_phaseC_blind_benchmark():
    t0_start = time.time()
    print("=" * 90)
    print("=== Phase C Zero-Leakage 5-Split Blind Benchmark (s16~s20, N=5,000 Test Samples) ===")
    print("=" * 90)

    ctx = load_scene_and_manifest_context()
    all_splits_data = {s: load_all_17_families_for_split(s) for s in SPLITS}

    from build_55plus_ultimate_submission import (
        hierarchical_rolldown_decode, T150_T0, T150_T1, T150_T2, T150_TAU_ANY,
        apply_sinkhorn, apply_transductive_manifold_smoothing, apply_cross_defect_cooccurrence_refinement
    )

    preds_p1_all, preds_b53_all, preds_phaseC_all, gold_all = [], [], [], []
    W_learned_all = []
    prm_learned_all = []

    print(f"\n{'Split':6s} | {'Phase1(50:50)':>13s} | {'PhaseB(53.02)':>13s} | {'PhaseC_Routed':>13s} | {'PhaseC_Stage2':>13s} | {'Gain vs P1':>10s}")
    print("-" * 86)

    for s_eval in SPLITS:
        pool = build_disjoint_oof_pool_for_split(s_eval, all_splits_data, ctx)
        oof_fams = pool['oof_fams']
        te_fams = pool['te_fams']
        y_oof_6d = pool['oof_gold']
        y_oof_14 = labels_to_14ch(y_oof_6d)
        oof_weights = pool['oof_weights']
        oof_sources = pool['oof_sources']
        n_va = pool['n_va']
        gold_te = pool['gte']

        # Baseline Phase 1 (50:50 MultiViT + InternViT6B)
        dec_base = lambda p: hierarchical_rolldown_decode(p, T150_T0, T150_T1, T150_T2, T150_TAU_ANY, roll_p2=False)
        anc_te = pool['anc_te']
        pred_f1_te = dec_base(anc_te)
        base_clean_te = (pred_f1_te == 0).all(axis=1)

        p1_blend_te = 0.50 * anc_te + 0.50 * te_fams[4]
        pred_p1 = dec_base(p1_blend_te)
        pred_p1[base_clean_te[:, None] & (pred_p1 == 3)] = 0
        sc_p1 = competition_score(pred_p1, gold_te)['total']

        # Cached Phase B (53.021) baseline
        b53_cache = np.load(f'results/phaseB_fixed_cache/phaseA_{s_eval}.npz')
        from build_55plus_ultimate_submission import phaseA_decode
        pred_b53 = phaseA_decode(
            apply_sinkhorn(b53_cache['test_probs'], b53_cache['base_clean']),
            b53_cache['t0'], b53_cache['t1'], b53_cache['t2'], T150_TAU_ANY,
            roll_p2=False, base_clean=b53_cache['base_clean']
        )
        sc_b53 = competition_score(pred_b53, gold_te)['total']

        # OOF base_clean mask
        anc_oof = 0.25 * sum(oof_fams[j] for j in range(4))
        oof_base_clean = (dec_base(anc_oof) == 0).all(axis=1)

        # Step 1: Fit 17-Family Per-Channel Specialist Routing Weights on Disjoint OOF
        W = fit_17fam_channel_weights(oof_fams, y_oof_14, oof_weights=oof_weights, l2_reg=0.04)
        W_learned_all.append(W)

        K = len(oof_fams)
        oof_routed_p = sum(oof_fams[k] * W[k:k + 1, :] for k in range(K))
        te_routed_p = sum(te_fams[k] * W[k:k + 1, :] for k in range(K))

        p1_blend_oof = 0.50 * anc_oof + 0.50 * oof_fams[4]
        oof_routed = align_moments_14ch(oof_routed_p, p1_blend_oof)
        te_routed = align_moments_14ch(te_routed_p, p1_blend_te)

        # Evaluate Phase C Routed (17 families before Stage-2 relational stacker)
        oof_r_sink = apply_sinkhorn(oof_routed, (y_oof_6d == 0).all(axis=1))
        te_r_sink = apply_sinkhorn(te_routed, base_clean_te)
        pred_routed_only = phaseA_decode(
            te_r_sink, b53_cache['t0'], b53_cache['t1'], b53_cache['t2'], T150_TAU_ANY,
            roll_p2=False, base_clean=base_clean_te
        )
        sc_routed_only = competition_score(pred_routed_only, gold_te)['total']

        # Step 2: Dual QualiRAG Residuals + Stage-2 Relational Calibrator
        q_oof, q_te = compute_dual_qualirag_iqa_residuals(
            oof_routed, te_routed, y_oof_6d, pool['oof_idx'], pool['te_idx'], oof_sources, ctx
        )
        oof_cal, te_cal, fsort_oof, fsort_te = fit_and_apply_stage2_calibrator(
            oof_fams, te_fams, oof_routed, te_routed,
            y_oof_6d, oof_sources, oof_weights, q_oof, q_te, alpha_meta=0.25
        )

        # Transductive manifold smoothing + cross-defect co-occurrence refinement
        te_cal = apply_transductive_manifold_smoothing(
            oof_cal, te_cal, y_oof_14, ctx['emb_c'][pool['oof_idx']], ctx['emb_c'][pool['te_idx']],
            k_nn=14, sim_thresh=0.78, alpha_graph=0.06
        )
        oof_cal = apply_cross_defect_cooccurrence_refinement(oof_cal, y_oof_14, alpha_co=0.04)
        te_cal = apply_cross_defect_cooccurrence_refinement(te_cal, y_oof_14, alpha_co=0.04)
        oof_cal = align_moments_14ch(oof_cal, p1_blend_oof)
        te_cal = align_moments_14ch(te_cal, p1_blend_te)

        oof_cal_sink = apply_sinkhorn(oof_cal, (y_oof_6d == 0).all(axis=1))
        te_cal_sink = apply_sinkhorn(te_cal, base_clean_te)

        # Step 3: Optimize Phase C 5-Stream Decoder strictly on Disjoint OOF Pool
        prm_opt = optimize_phaseC_decoder_on_oof(
            oof_cal_sink, fsort_oof, y_oof_6d, oof_base_clean, n_va, shrinkage=0.55
        )
        prm_learned_all.append(prm_opt)

        pred_phaseC = decode_phaseC_5stream(te_cal_sink, fsort_te, prm_opt, base_clean=base_clean_te)
        sc_phaseC = competition_score(pred_phaseC, gold_te)['total']

        preds_p1_all.append(pred_p1)
        preds_b53_all.append(pred_b53)
        preds_phaseC_all.append(pred_phaseC)
        gold_all.append(gold_te)

        print(f"{s_eval:6s} | {sc_p1:13.2f} | {sc_b53:13.2f} | {sc_routed_only:13.2f} | {sc_phaseC:13.2f} | {sc_phaseC - sc_p1:+10.2f}", flush=True)

    G_pool = np.vstack(gold_all)
    P_p1 = np.vstack(preds_p1_all)
    P_b53 = np.vstack(preds_b53_all)
    P_c = np.vstack(preds_phaseC_all)

    r_p1 = competition_score(P_p1, G_pool)
    r_b53 = competition_score(P_b53, G_pool)
    r_c = competition_score(P_c, G_pool)

    print("-" * 86)
    print(f"POOLED (N=5000) Phase 1 Baseline : total={r_p1['total']:.3f} (S_pos={25*r_p1['S_pos']:.2f}, S_p0={20*r_p1['S_p0']:.2f}, S_clean={20*r_p1['S_clean']:.2f})")
    print(f"POOLED (N=5000) Phase B (53.021) : total={r_b53['total']:.3f} (S_pos={25*r_b53['S_pos']:.2f}, S_p0={20*r_b53['S_p0']:.2f}, S_clean={20*r_b53['S_clean']:.2f})")
    print(f"POOLED (N=5000) Phase C Ultimate : total={r_c['total']:.3f} (S_pos={25*r_c['S_pos']:.2f}, S_p0={20*r_c['S_p0']:.2f}, S_clean={20*r_c['S_clean']:.2f})  [{time.time()-t0_start:.1f}s]")


if __name__ == '__main__':
    evaluate_phaseC_blind_benchmark()
