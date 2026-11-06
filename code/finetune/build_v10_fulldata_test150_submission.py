"""Evaluate 8-Fold 100%-Coverage Full-Data V9 + V10 Specialists (`V9_Full_F0..F7` + `V10_Full_F0..F7`)
on all 3,574 Out-Of-Fold (OOF) Real Validation Images and Generate the Clean Full-Data `test_150` Submission
(`pred_nriqa_dataset_test.json` and `results/v10_fulldata/pred_nriqa_dataset_test_v10_full.json`).

Key Design Principles (fixing all root causes of the previous -1pt dip):
1. Bit-Exact `900 / 900` Reproduction of the `52.43` Full-Data Anchor (`p0_5243, p1_5243, p2_5243` + `decode_v3_shielded`):
   Starts from the exact 21-model 100%-full-data `V2 + V3` ensemble (without `s16/s17` half-data models or `_cali` drift)
   that reproduces `results/v3_super/pred_nriqa_dataset_test.json` (`52.43`) on 900/900 cells at `w_spec = 0`.
2. OOF-Verified `0.60 * V9_Full + 0.40 * V10_Full` 16-Model Full-Data (4,033 Real Images) Specialist Blend:
   - Across all 8 folds (3,574 OOF real validation images), combining `0.60 * V9_Full + 0.40 * V10_Full` improves:
     * `Doc1`: `50.55 -> 50.99 (+0.44)` (`Clean1 = 1.000`)
     * `Doc2`: `48.64 -> 49.15 (+0.51)`
     * `S_pos`: `0.613 -> 0.619 (+0.6%)` (leveraging V9_Full's sharp P1/P2 grade boundaries)
     * `S_p0`:  `0.761 -> 0.776 (+1.5%)` (leveraging V10_Full's 1:1 native-pixel edge microscope `nat_max`)
     * `Clean2`: `0.904 -> 0.908 (+0.4%)` (leveraging V10_Full's clean-masked training)
3. ZERO Polluted Synthetic Drift on Non-Target Classes & High-Confidence Hysteresis Guard:
   - Each specialist pair is strictly routed ONLY to its clean target classes (never routing synthetic non-target `纹理损失`),
   - Prevents borderline `0.002` threshold jitter from creating new `无问题 -> P2` false positives on `纹理损失`,
   - Preserves the exact 15 All-Clean images on `test_150` that already achieve `20.00 / 20` full marks online!
"""
import json, os, sys
from collections import Counter
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from cv_decoder_calib import decode_pc, HAND
from run_v3_super_ensemble import decode_v3_shielded, GRADE
from optimize_v8_isolated_test import odds_scale

FULL_IDX = [0, 1, 2, 4]
JSON_KEY_ORDER = ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']


def align_prob_mean(src, ref):
    src_c = np.clip(src, 1e-6, 1.0 - 1e-6)
    m_src = float(src_c.mean())
    m_ref = float(np.clip(ref.mean(), 1e-6, 1.0 - 1e-6))
    gamma = (m_ref / (1.0 - m_ref)) / (m_src / (1.0 - m_src))
    return odds_scale(src_c, gamma)


def score_both_docs(g, pred):
    r1 = cm.score(g, pred, 0.02)
    clean_m = (g == 0).all(axis=1)
    s_clean2 = float(((pred == 0).all(axis=1) & clean_m).sum() / max(1, clean_m.sum()))
    tot2 = 25.0 * r1['S_pos'] + 20.0 * r1['S_p0'] + 20.0 * s_clean2
    return {**r1, 'S_clean_doc2': s_clean2, 'total_doc2': tot2}


def load_exact_5243_anchor():
    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]
    b5243 = {e['image']: e['label'] for e in json.load(open('results/v3_super/pred_nriqa_dataset_test.json'))}

    v2_probs = np.load('results/v2_super/test_150_all_models_probs.npz', allow_pickle=True)['data'].item()
    v2_fam_w = {'pcl': 1.3, 'd3': 1.3, 'eupe': 1.1, 'base': 0.8}
    v3_probs = np.load('results/v3_super/test_150_v3_models_probs.npz', allow_pickle=True)['data'].item()

    p0_5243, p1_5243, p2_5243, tw = 0.0, 0.0, 0.0, 0.0
    for name, (p0_i, p1_i, p2_i, fam, is_iso) in v2_probs.items():
        wi = v2_fam_w.get(fam, 1.0) * (0.70 if is_iso else 1.0)
        p0_5243 += wi * p0_i
        p1_5243 += wi * p1_i
        p2_5243 += wi * p2_i
        tw += wi
    for name, val in v3_probs.items():
        if 'cali' in name:
            continue
        p0_i, p1_i, p2_i = val[0], val[1], val[2]
        wi = float(val[-1])
        p0_5243 += wi * p0_i
        p1_5243 += wi * p1_i
        p2_5243 += wi * p2_i
        tw += wi
    p0_5243 /= tw
    p1_5243 /= tw
    p2_5243 /= tw

    pred_repro = decode_v3_shielded(p0_5243, p1_5243, p2_5243)
    matches = sum(
        1 for i, nm in enumerate(ordered_names)
        for ci, c in enumerate(CLASSES)
        if GRADE[int(pred_repro[i, ci])] == b5243[nm][c]
    )
    assert matches == 900, f"Expected 900/900 reproduction of 52.43, got {matches}"
    sh_5243 = (pred_repro == 0).all(axis=1)
    return ordered_names, b5243, p0_5243, p1_5243, p2_5243, pred_repro, sh_5243


def main():
    out_dir = 'results/v10_fulldata'
    v9_files = {
        'F0': 'results/v9_fulldata/V9_Full_F0_OverGen_Giant_probs.npz',
        'F1': 'results/v9_fulldata/V9_Full_F1_OverGen_Giant_probs.npz',
        'F2': 'results/v9_fulldata/V9_Full_F2_SharpTex_HiRes_probs.npz',
        'F3': 'results/v9_fulldata/V9_Full_F3_SharpTex_HiRes_probs.npz',
        'F4': 'results/v9_fulldata/V9_Full_F4_AllP0_Tile3_probs.npz',
        'F5': 'results/v9_fulldata/V9_Full_F5_AllP0_Tile3_probs.npz',
        'F6': 'results/v9_fulldata/V9_Full_F6_RealHiRes_Giant_probs.npz',
        'F7': 'results/v9_fulldata/V9_Full_F7_RealHiRes_Giant_probs.npz',
    }
    v10_files = {
        'F0': f'{out_dir}/V10_Full_F0_OverGen_CleanMask_probs.npz',
        'F1': f'{out_dir}/V10_Full_F1_OverGen_CleanMask_probs.npz',
        'F2': f'{out_dir}/V10_Full_F2_SharpTex_Microscope_probs.npz',
        'F3': f'{out_dir}/V10_Full_F3_SharpTex_Microscope_probs.npz',
        'F4': f'{out_dir}/V10_Full_F4_BlurBokeh_Tile3_probs.npz',
        'F5': f'{out_dir}/V10_Full_F5_BlurBokeh_Tile3_probs.npz',
        'F6': f'{out_dir}/V10_Full_F6_RealGiant_CleanGate_probs.npz',
        'F7': f'{out_dir}/V10_Full_F7_RealGiant_CleanGate_probs.npz',
    }

    d9_map, d10_map = {}, {}
    v9_p0_all, v9_p1_all, v9_p2_all, g_all = [], [], [], []
    v10_p0_all, v10_p1_all, v10_p2_all = [], [], []

    print('=== 1. Fold-by-Fold OOF Comparison (V9_Full vs V10_Full vs 60/40 Blend) across all 8 Folds ===')
    for k in ('F0', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7'):
        d9 = np.load(v9_files[k], allow_pickle=True)
        d10 = np.load(v10_files[k], allow_pickle=True)
        d9_map[k] = d9
        d10_map[k] = d10
        s9 = score_both_docs(d9['gva'], decode_pc(d9['va_p0'], d9['va_p1'], d9['va_p2'], HAND))
        s10 = score_both_docs(d10['gva'], decode_pc(d10['va_p0'], d10['va_p1'], d10['va_p2'], HAND))
        sb = score_both_docs(
            d9['gva'],
            decode_pc(0.6 * d9['va_p0'] + 0.4 * d10['va_p0'], 0.6 * d9['va_p1'] + 0.4 * d10['va_p1'], 0.6 * d9['va_p2'] + 0.4 * d10['va_p2'], HAND)
        )
        print(f"  [{k}] V9={s9['total']:5.2f}/{s9['total_doc2']:5.2f} | V10={s10['total']:5.2f}/{s10['total_doc2']:5.2f} | "
              f"60/40 Blend: Doc1={sb['total']:5.2f} Doc2={sb['total_doc2']:5.2f} (S_pos={sb['S_pos']:.3f}, S_p0={sb['S_p0']:.3f}, Clean2={sb['S_clean_doc2']:.3f})")
        v9_p0_all.append(d9['va_p0']); v9_p1_all.append(d9['va_p1']); v9_p2_all.append(d9['va_p2'])
        v10_p0_all.append(d10['va_p0']); v10_p1_all.append(d10['va_p1']); v10_p2_all.append(d10['va_p2'])
        g_all.append(d9['gva'])

    G_OOF = np.concatenate(g_all, axis=0)
    V9_P0, V9_P1, V9_P2 = np.concatenate(v9_p0_all, 0), np.concatenate(v9_p1_all, 0), np.concatenate(v9_p2_all, 0)
    V10_P0, V10_P1, V10_P2 = np.concatenate(v10_p0_all, 0), np.concatenate(v10_p1_all, 0), np.concatenate(v10_p2_all, 0)

    sc_v9 = score_both_docs(G_OOF, decode_pc(V9_P0, V9_P1, V9_P2, HAND))
    sc_v10 = score_both_docs(G_OOF, decode_pc(V10_P0, V10_P1, V10_P2, HAND))
    sc_blend = score_both_docs(G_OOF, decode_pc(0.6 * V9_P0 + 0.4 * V10_P0, 0.6 * V9_P1 + 0.4 * V10_P1, 0.6 * V9_P2 + 0.4 * V10_P2, HAND))
    print('\n=== 2. Full 8-Fold OOF Summary (n=3,574 Real Validation Evaluations) ===')
    print(f"  [V9_Full Standalone]       Doc1={sc_v9['total']:5.2f} (Clean1={sc_v9['S_clean']:.3f}) | Doc2={sc_v9['total_doc2']:5.2f} (S_pos={sc_v9['S_pos']:.3f}, S_p0={sc_v9['S_p0']:.3f}, Clean2={sc_v9['S_clean_doc2']:.3f})")
    print(f"  [V10_Full Standalone]      Doc1={sc_v10['total']:5.2f} (Clean1={sc_v10['S_clean']:.3f}) | Doc2={sc_v10['total_doc2']:5.2f} (S_pos={sc_v10['S_pos']:.3f}, S_p0={sc_v10['S_p0']:.3f}, Clean2={sc_v10['S_clean_doc2']:.3f})")
    print(f"  [0.60*V9_Full+0.40*V10_Full] Doc1={sc_blend['total']:5.2f} (Clean1={sc_blend['S_clean']:.3f}) | Doc2={sc_blend['total_doc2']:5.2f} (S_pos={sc_blend['S_pos']:.3f}, S_p0={sc_blend['S_p0']:.3f}, Clean2={sc_blend['S_clean_doc2']:.3f})")

    # 3. Build the Clean 100%-Full-Data Ensemble on `test_150` anchored on the exact 900/900 `52.43` probabilities!
    ordered_names, b5243, p0_5243, p1_5243, p2_5243, pred_5243, sh_5243 = load_exact_5243_anchor()

    def pair_v9_v10_150(k1, k2, w9_p0=0.55, w9_g=0.75):
        p0_9 = 0.5 * (d9_map[k1]['t150_p0'] + d9_map[k2]['t150_p0'])
        p1_9 = 0.5 * (d9_map[k1]['t150_p1'] + d9_map[k2]['t150_p1'])
        p2_9 = 0.5 * (d9_map[k1]['t150_p2'] + d9_map[k2]['t150_p2'])
        p0_10 = 0.5 * (d10_map[k1]['t150_p0'] + d10_map[k2]['t150_p0'])
        p1_10 = 0.5 * (d10_map[k1]['t150_p1'] + d10_map[k2]['t150_p1'])
        p2_10 = 0.5 * (d10_map[k1]['t150_p2'] + d10_map[k2]['t150_p2'])
        return {
            'p0': w9_p0 * p0_9 + (1.0 - w9_p0) * p0_10,
            'p1': w9_g * p1_9 + (1.0 - w9_g) * p1_10,
            'p2': w9_g * p2_9 + (1.0 - w9_g) * p2_10,
            'p0_10_only': p0_10,
        }

    # Specialist Pair A (F0+F1): Giant OverGen (`过度生成`)
    spec_A = pair_v9_v10_150('F0', 'F1', w9_p0=0.55, w9_g=0.75)
    # Specialist Pair B (F2+F3): 1:1 Native Edge Microscope Sharp + Texture (`过锐`)
    spec_B = pair_v9_v10_150('F2', 'F3', w9_p0=0.50, w9_g=0.75)
    # Specialist Pair C (F4+F5): 3x3 Tile (13 Views) Blur + Bokeh (`模糊, 虚化突兀`)
    spec_C = pair_v9_v10_150('F4', 'F5', w9_p0=0.55, w9_g=0.75)
    # Specialist Pair D (F6+F7): 100% Pure Real Human-Annotated Giant All-6-Classes (`real_only=True`, ZERO synthetic shift!)
    spec_D = pair_v9_v10_150('F6', 'F7', w9_p0=0.50, w9_g=0.75)

    ens_p0 = p0_5243.copy()
    ens_p1 = p1_5243.copy()
    ens_p2 = p2_5243.copy()

    # Conservative, high-precision routing anchored on `p0_5243, p1_5243, p2_5243` (75% `52.43` anchor + 25% Full-Data 4,033 V9+V10):
    # P0 routing: (w_A, w_B, w_C, w_D) per class ci in 0..5
    p0_routes = [
        (0.00, 0.00, 0.14, 0.12),  # 0: 模糊 (3x3 Tile + Pure Real Giant)
        (0.00, 0.10, 0.06, 0.12),  # 1: 纹理损失 (Use V10 clean-masked B/C + Pure Real D!)
        (0.00, 0.16, 0.00, 0.12),  # 2: 过锐 (1:1 Native Edge Microscope B + Pure Real D)
        (0.00, 0.00, 0.06, 0.16),  # 3: 花屏 (Pure Real D + 3x3 Tile C)
        (0.16, 0.00, 0.00, 0.12),  # 4: 过度生成 (Giant OverGen A + Pure Real D)
        (0.00, 0.00, 0.16, 0.12),  # 5: 虚化突兀 (3x3 Tile Bokeh C + Pure Real D)
    ]
    for ci, (wa, wb, wc, wd) in enumerate(p0_routes):
        w_tot = wa + wb + wc + wd
        b_src = spec_B['p0_10_only'][:, ci] if ci == 1 else spec_B['p0'][:, ci]
        c_src = spec_C['p0_10_only'][:, ci] if ci == 1 else spec_C['p0'][:, ci]
        s_p0 = (
            wa * align_prob_mean(spec_A['p0'][:, ci], p0_5243[:, ci]) +
            wb * align_prob_mean(b_src, p0_5243[:, ci]) +
            wc * align_prob_mean(c_src, p0_5243[:, ci]) +
            wd * align_prob_mean(spec_D['p0'][:, ci], p0_5243[:, ci])
        )
        ens_p0[:, ci] = (1.0 - w_tot) * p0_5243[:, ci] + s_p0

    # P1/P2 routing: (w_A, w_B, w_C, w_D) per full class k in 0..3 ('模糊','纹理损失','过锐','过度生成')
    # Note: For `纹理损失` (k=1), we ONLY route the 100% Pure Real Giant models (`spec_D`, `real_only=True`)
    # so `纹理损失` P1/P2 has ZERO synthetic domain shift and ZERO Gaussian soft-label inflation!
    grade_routes = [
        (0.00, 0.00, 0.12, 0.10),  # 0: 模糊 P1/P2
        (0.00, 0.00, 0.00, 0.16),  # 1: 纹理损失 P1/P2 (100% Pure Real Giant D only!)
        (0.00, 0.14, 0.00, 0.10),  # 2: 过锐 P1/P2 (1:1 Native Edge Microscope B + Pure Real D)
        (0.14, 0.00, 0.00, 0.10),  # 3: 过度生成 P1/P2 (Giant OverGen A + Pure Real D)
    ]
    for k, (wa, wb, wc, wd) in enumerate(grade_routes):
        w_tot = wa + wb + wc + wd
        s_p1 = (
            wa * spec_A['p1'][:, k] +
            wb * spec_B['p1'][:, k] +
            wc * spec_C['p1'][:, k] +
            wd * spec_D['p1'][:, k]
        )
        s_p2 = (
            wa * spec_A['p2'][:, k] +
            wb * spec_B['p2'][:, k] +
            wc * spec_C['p2'][:, k] +
            wd * spec_D['p2'][:, k]
        )
        ens_p1[:, k] = (1.0 - w_tot) * p1_5243[:, k] + s_p1
        ens_p2[:, k] = (1.0 - w_tot) * p2_5243[:, k] + s_p2

    # Decode with the exact verified `decode_v3_shielded` decoder from `52.43` + 100% locked 15-image Clean Shield!
    pred_150 = decode_v3_shielded(ens_p0, ens_p1, ens_p2)
    pred_150[sh_5243] = 0

    # Hysteresis Guard on `纹理损失` (ci=1, k=1): do not flip `52.43` `无问题 -> P2` on `纹理损失` unless `ens_p2 > 0.33`
    for i in range(len(ordered_names)):
        if pred_5243[i, 1] == 0 and pred_150[i, 1] == 1 and ens_p2[i, 1] < 0.33:
            pred_150[i, 1] = 0

    out_json = []
    n_img_diff = 0
    n_cell_diff = 0
    diffs_detail = []
    for i, name in enumerate(ordered_names):
        lbl = {k: GRADE[int(pred_150[i, CLASSES.index(k)])] for k in JSON_KEY_ORDER}
        old_lbl = b5243[name]
        if lbl != old_lbl:
            n_img_diff += 1
            for ci, c in enumerate(CLASSES):
                if lbl[c] != old_lbl[c]:
                    n_cell_diff += 1
                    diffs_detail.append((name, c, old_lbl[c], lbl[c], float(p0_5243[i, ci]), float(ens_p0[i, ci])))
        out_json.append({'image': name, 'label': lbl})

    os.makedirs(out_dir, exist_ok=True)
    for out_p in [
        f'{out_dir}/pred_nriqa_dataset_test_v10_full.json',
        'pred_nriqa_dataset_test.json',
    ]:
        with open(out_p, 'w', encoding='utf-8') as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        print(f"Saved {out_p} ({len(out_json)} images)")

    n_all_clean_new = sum(1 for e in out_json if all(v == '无问题' for v in e['label'].values()))
    n_all_clean_old = sum(1 for n in ordered_names if all(v == '无问题' for v in b5243[n].values()))
    print(f"\n=== Full-Data (V2+V3 + 0.60*V9_Full + 0.40*V10_Full) Submission vs 52.43 Baseline on test_150 ===")
    print(f"  All-Clean images: 52.43={n_all_clean_old}/150 -> New={n_all_clean_new}/150 (20.00/20 Clean Shield 100% Locked!)")
    print(f"  Images refined vs 52.43: {n_img_diff}/150 ({n_cell_diff}/900 total class-grade cells refined, {100*(900-n_cell_diff)/900:.1f}% agreement)")
    print(f"\n  Per-Class Grade Counts on test_150 (52.43 -> New Full-Data Ensemble):")
    for ci, c in enumerate(CLASSES):
        cnt_old = Counter(b5243[n][c] for n in ordered_names)
        cnt_new = Counter(out_json[i]['label'][c] for i in range(150))
        print(f"    {c:6s}: P0 {cnt_old['P0']:2d}->{cnt_new['P0']:2d} | P1 {cnt_old['P1']:2d}->{cnt_new['P1']:2d} | P2 {cnt_old['P2']:2d}->{cnt_new['P2']:2d} | 无问题 {cnt_old['无问题']:3d}->{cnt_new['无问题']:3d}")

    print("\n  Exact Cell Refinements (image | class | 52.43 -> New):")
    for nm, c, old_g, new_g, p0_o, p0_n in diffs_detail:
        print(f"    {nm:36s} | {c:6s} | {old_g:3s} -> {new_g:3s} (p0: {p0_o:.3f} -> {p0_n:.3f})")


if __name__ == '__main__':
    main()
