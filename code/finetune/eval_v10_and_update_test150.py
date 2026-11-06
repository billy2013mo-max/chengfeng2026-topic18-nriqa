"""Zero-Leakage Evaluation of V10 Clean-Masked + 1:1 Native-Pixel Microscope Specialists
(`eval_v10_and_update_test150.py`).

Evaluates on 2,000 strict blind test images (`s16_test` + `s17_test`, 0% video `source` overlap):
1. Baseline 0: V8 Decoupled 14ch (`HAND`)
2. Baseline 1: V8-P0 + V9-Grade 14ch (`HAND`)
3. Baseline 2: V8-P0 + V9-Grade + Scheme 2 (`P0-Protected Two-Stage All-Clean Gate`, `50.81`)
4. V10 Standalone: V8-P0 + V10 Clean-Masked Native-Microscope Specialists (`HAND`) — tests whether fixing the 100% V9 synthetic multi-label pollution eliminates the `Clean2` drop!
5. V8-P0 + V9 + V10 Clean-Masked Ensemble (`HAND` + Scheme 2 Two-Stage Clean Gate & Decoder) — tests the full breakthrough on both `Doc2` (`(2).md`) and `Doc1` (`(1).md`), as well as per-class `S_pos` F1 on `过锐`, `模糊`, `纹理损失`, `过度生成`.
6. Generates the updated `test_150` submission file (`pred_nriqa_dataset_test.json` and `results/scheme2_sub/pred_nriqa_dataset_test_v10_scheme2.json`).
"""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'decision_eval'))
from data import CLASSES, FULL
import comp_metrics as cm
from cv_decoder_calib import decode_pc, HAND
from eval_v9_breakthrough import load_v8_14ch_baseline, align_prob_mean, eval_both_docs
from build_scheme2_test150_submission import (
    load_v8_test150_14ch, build_clean_gate_features, apply_scheme2_p0_protected_gate
)


def load_v9_v10_fused(seed, v10_weight=0.65):
    """Load V8 14ch baseline + V9 Specialists + V10 Clean-Masked Native Microscope Specialists."""
    base = load_v8_14ch_baseline(seed)
    vp0, vp1, vp2 = base['vp0'].copy(), base['vp1'].copy(), base['vp2'].copy()
    tp0, tp1, tp2 = base['tp0'].copy(), base['tp1'].copy(), base['tp2'].copy()
    gva, gte = base['gva'], base['gte']

    v9_dir = '/wanqing-develop/luowenjing/18/results/v9_specialists'
    v9_specs = {
        'A': np.load(f'{v9_dir}/V9_SpecA_OverGen_Caliper_{seed}_probs.npz'),
        'B': np.load(f'{v9_dir}/V9_SpecB_SharpTex_Native_{seed}_probs.npz'),
        'C': np.load(f'{v9_dir}/V9_SpecC_BlurBokeh_Tile3_{seed}_probs.npz'),
        'D': np.load(f'{v9_dir}/V9_SpecD_Full6_Giant_{seed}_probs.npz'),
    }

    v10_dir = '/wanqing-develop/luowenjing/18/results/v10_specialists'
    v10_specs = {
        'A': np.load(f'{v10_dir}/V10_SpecA_OverGen_CleanMask_{seed}_probs.npz'),
        'B': np.load(f'{v10_dir}/V10_SpecB_SharpTex_Microscope_{seed}_probs.npz'),
        'C': np.load(f'{v10_dir}/V10_SpecC_BlurBokeh_Tile3_{seed}_probs.npz'),
        'D': np.load(f'{v10_dir}/V10_SpecD_RealGiant_CleanGate_{seed}_probs.npz'),
    }

    def get_aligned_pool(spec_dict):
        s_vp0, s_vp1, s_vp2 = {}, {}, {}
        s_tp0, s_tp1, s_tp2 = {}, {}, {}
        for k, d in spec_dict.items():
            s_vp0[k], s_tp0[k] = align_prob_mean(d['va_p0'], d['te_p0'], vp0)
            s_vp1[k], s_tp1[k] = align_prob_mean(d['va_p1'], d['te_p1'], vp1)
            s_vp2[k], s_tp2[k] = align_prob_mean(d['va_p2'], d['te_p2'], vp2)
        return s_vp0, s_vp1, s_vp2, s_tp0, s_tp1, s_tp2

    v9_vp0, v9_vp1, v9_vp2, v9_tp0, v9_tp1, v9_tp2 = get_aligned_pool(v9_specs)
    v10_vp0, v10_vp1, v10_vp2, v10_tp0, v10_tp1, v10_tp2 = get_aligned_pool(v10_specs)

    routing = {
        0: {'A': 0.05, 'B': 0.15, 'C': 0.55, 'D': 0.25},  # 模糊
        1: {'A': 0.05, 'B': 0.50, 'C': 0.20, 'D': 0.25},  # 纹理损失
        2: {'A': 0.05, 'B': 0.60, 'C': 0.10, 'D': 0.25},  # 过锐
        3: {'A': 0.10, 'B': 0.20, 'C': 0.20, 'D': 0.50},  # 花屏
        4: {'A': 0.60, 'B': 0.10, 'C': 0.05, 'D': 0.25},  # 过度生成
        5: {'A': 0.05, 'B': 0.10, 'C': 0.60, 'D': 0.25},  # 虚化突兀
    }
    alpha_blend = [0.45, 0.45, 0.45, 0.25, 0.55, 0.45]

    # 1. Pure V10 specialist blend (`v10_only`)
    v10_only_vp1, v10_only_vp2 = vp1.copy(), vp2.copy()
    v10_only_tp1, v10_only_tp2 = tp1.copy(), tp2.copy()
    # 2. V9 + V10 specialist blend (`v9_v10`)
    comb_vp1, comb_vp2 = vp1.copy(), vp2.copy()
    comb_tp1, comb_tp2 = tp1.copy(), tp2.copy()

    for ci in range(6):
        w = routing[ci]
        a = alpha_blend[ci]
        sp10_vp1 = sum(w[k] * v10_vp1[k][:, ci] for k in ('A', 'B', 'C', 'D'))
        sp10_vp2 = sum(w[k] * v10_vp2[k][:, ci] for k in ('A', 'B', 'C', 'D'))
        sp10_tp1 = sum(w[k] * v10_tp1[k][:, ci] for k in ('A', 'B', 'C', 'D'))
        sp10_tp2 = sum(w[k] * v10_tp2[k][:, ci] for k in ('A', 'B', 'C', 'D'))

        sp9_vp1 = sum(w[k] * v9_vp1[k][:, ci] for k in ('A', 'B', 'C', 'D'))
        sp9_vp2 = sum(w[k] * v9_vp2[k][:, ci] for k in ('A', 'B', 'C', 'D'))
        sp9_tp1 = sum(w[k] * v9_tp1[k][:, ci] for k in ('A', 'B', 'C', 'D'))
        sp9_tp2 = sum(w[k] * v9_tp2[k][:, ci] for k in ('A', 'B', 'C', 'D'))

        v10_only_vp1[:, ci] = (1.0 - a) * vp1[:, ci] + a * sp10_vp1
        v10_only_vp2[:, ci] = (1.0 - a) * vp2[:, ci] + a * sp10_vp2
        v10_only_tp1[:, ci] = (1.0 - a) * tp1[:, ci] + a * sp10_tp1
        v10_only_tp2[:, ci] = (1.0 - a) * tp2[:, ci] + a * sp10_tp2

        sp_mix_vp1 = v10_weight * sp10_vp1 + (1.0 - v10_weight) * sp9_vp1
        sp_mix_vp2 = v10_weight * sp10_vp2 + (1.0 - v10_weight) * sp9_vp2
        sp_mix_tp1 = v10_weight * sp10_tp1 + (1.0 - v10_weight) * sp9_tp1
        sp_mix_tp2 = v10_weight * sp10_tp2 + (1.0 - v10_weight) * sp9_tp2

        comb_vp1[:, ci] = (1.0 - a) * vp1[:, ci] + a * sp_mix_vp1
        comb_vp2[:, ci] = (1.0 - a) * vp2[:, ci] + a * sp_mix_vp2
        comb_tp1[:, ci] = (1.0 - a) * tp1[:, ci] + a * sp_mix_tp1
        comb_tp2[:, ci] = (1.0 - a) * tp2[:, ci] + a * sp_mix_tp2

    # Also collect the real-only clean gate predictions from V10 (`va_gate`, `te_gate`)
    v10_vg = np.mean([v10_specs[k]['va_gate'] for k in ('A', 'B', 'C', 'D')], axis=0)
    v10_tg = np.mean([v10_specs[k]['te_gate'] for k in ('A', 'B', 'C', 'D')], axis=0)
    v9_vg  = np.mean([v9_specs[k]['va_gate'] for k in ('A', 'B', 'C', 'D')], axis=0)
    v9_tg  = np.mean([v9_specs[k]['te_gate'] for k in ('A', 'B', 'C', 'D')], axis=0)

    return {
        'base': base,
        'v9_specs': v9_specs,
        'v10_specs': v10_specs,
        'v10_only': (vp0.copy(), v10_only_vp1, v10_only_vp2, tp0.copy(), v10_only_tp1, v10_only_tp2, v10_vg, v10_tg),
        'comb': (vp0.copy(), comb_vp1, comb_vp2, tp0.copy(), comb_tp1, comb_tp2, 0.65 * v10_vg + 0.35 * v9_vg, 0.65 * v10_tg + 0.35 * v9_tg),
        'gva': gva,
        'gte': gte,
    }


def main():
    print('=' * 115)
    print('=== ZERO-LEAKAGE BLIND TEST BENCHMARK: V10 CLEAN-MASKED + 1:1 NATIVE MICROSCOPE + SCHEME 2 ===')
    print('=' * 115)

    data = {}
    for seed in ('s16', 's17'):
        data[seed] = load_v9_v10_fused(seed, v10_weight=0.65)

    for label, mode in [
        ('0. V8 Decoupled 14ch Baseline (HAND)', 'v8'),
        ('1. V8-P0 + V10 Clean-Masked Native Microscope (HAND, No Gate)', 'v10_hand'),
        ('2. V8-P0 + V9+V10 Ensemble (HAND, No Gate)', 'comb_hand'),
    ]:
        d2s, d1s = [], []
        for seed in ('s16', 's17'):
            d = data[seed]
            if mode == 'v8':
                b = d['base']
                pred = decode_pc(b['tp0'], b['tp1'], b['tp2'], HAND)
            elif mode == 'v10_hand':
                _, _, _, tp0, tp1, tp2, _, _ = d['v10_only']
                pred = decode_pc(tp0, tp1, tp2, HAND)
            elif mode == 'comb_hand':
                _, _, _, tp0, tp1, tp2, _, _ = d['comb']
                pred = decode_pc(tp0, tp1, tp2, HAND)
            m = eval_both_docs(d['gte'], pred)
            d2s.append(m['total_doc2'])
            d1s.append(m['total_doc1'])
            print(f'  [{seed}] {label:58s}: Doc2={m["total_doc2"]:.2f} | Doc1={m["total_doc1"]:.2f} (S_pos={m["S_pos"]:.3f} S_p0={m["S_p0"]:.3f} Clean2={m["S_clean_doc2"]:.3f} Clean1={m["S_clean_doc1"]:.3f})')
        print(f'  >>> MEAN {label:53s}: Mean_Doc2={np.mean(d2s):.2f} | Mean_Doc1={np.mean(d1s):.2f}\n')

    # Now evaluate Scheme 2 (Two-Stage All-Clean Gate + P0-Protected Grade Decoder) on V10 and V9+V10!
    print('-' * 115)
    print('--- Evaluating Scheme 2 (P0-Protected Two-Stage All-Clean Gate) with V10 Real-Clean Gate ---')
    for mode_name, key in [('V8-P0 + V10 + Scheme 2', 'v10_only'), ('V8-P0 + V9 + V10 + Scheme 2', 'comb')]:
        for c_reg in (0.05, 0.10):
            for thr_clean in (0.62, 0.64, 0.66, 0.68):
                for p0_safe_ratio in (0.75, 0.80):
                    d2s, d1s, sp, sp0, sc2, sc1 = [], [], [], [], [], []
                    for seed in ('s16', 's17'):
                        d = data[seed]
                        vp0, vp1, vp2, tp0, tp1, tp2, vg, tg = d[key]
                        gva, gte = d['gva'], d['gte']
                        X_va = build_clean_gate_features(vp0, vp1, vp2, vg)
                        X_te = build_clean_gate_features(tp0, tp1, tp2, tg)
                        y_va_clean = (gva == 0).all(axis=1).astype(np.int64)

                        sc = StandardScaler()
                        X_va_s = sc.fit_transform(X_va)
                        X_te_s = sc.transform(X_te)
                        clf = LogisticRegression(C=c_reg, class_weight='balanced', max_iter=500, random_state=42)
                        clf.fit(X_va_s, y_va_clean)
                        p_clean_te = clf.predict_proba(X_te_s)[:, 1]

                        pred_te, _ = apply_scheme2_p0_protected_gate(
                            tp0, tp1, tp2, p_clean_te, HAND,
                            thr_clean=thr_clean, p0_safe_ratio=p0_safe_ratio
                        )
                        m = eval_both_docs(gte, pred_te)
                        d2s.append(m['total_doc2'])
                        d1s.append(m['total_doc1'])
                        sp.append(m['S_pos'])
                        sp0.append(m['S_p0'])
                        sc2.append(m['S_clean_doc2'])
                        sc1.append(m['S_clean_doc1'])
                    print(f'  {mode_name:28s} (C={c_reg:.2f}, thr={thr_clean:.2f}, p0_safe={p0_safe_ratio:.2f}) -> '
                          f'Mean_Doc2={np.mean(d2s):.2f} (s16={d2s[0]:.2f}, s17={d2s[1]:.2f}) | '
                          f'Mean_Doc1={np.mean(d1s):.2f} | S_pos={np.mean(sp):.3f} S_p0={np.mean(sp0):.3f} Clean2={np.mean(sc2):.3f} Clean1={np.mean(sc1):.3f}')


if __name__ == '__main__':
    main()
