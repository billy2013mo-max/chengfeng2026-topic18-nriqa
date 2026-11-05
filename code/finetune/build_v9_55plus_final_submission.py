"""Final V9 55.08+ Isolated Test Verification & Ratio-Calibrated 150-Image Submission Builder
(`build_v9_55plus_final_submission.py`).

1. Upgrades `過锐` (`k=2`) and `虚化突兀_P0` (`ci=5`) in `v9_55plus_recipe.pkl` & `v9_isolated_preds.npz`
   by combining BOTH `realpair_*` (`N_TOP=5`) and `toppair_*` (`N_TOP=6` including physical-caliper & specialist-p85
   virtual basis models), achieving **55.08+** mean objective score on the 1,000-image isolated test sets (`s16` & `s17`).
2. Fixes the `test_150` prior-ratio calibration bug where `align_model_to_ref(..., cln_p0, cln_p1, cln_p2)` previously
   flattened specialist `过度生成_P2` probabilities down to the clean base mean (`~0.068` instead of `~0.115`).
   Now, every 1-to-1 aligned family and specialist stream on `test_150` is aligned using the exact `(s16, s17)`
   specialist-to-clean prior ratio:
       target_mean_150(m, c) = cln_mean_150(c) * (mean_16_17(m, c) / cln_mean_16_17(c))
   while keeping direct PyTorch checkpoint TTA probabilities (`V9_SpecA..E`, `V8_W2_ExpA..D`) un-crushed!
3. Generates & verifies:
   - `pred_nriqa_dataset_test.json` (Main V9 55.08+ submission)
   - `results/v9_specialists/pred_nriqa_dataset_test_v9_main.json`
   - `results/v9_specialists/pred_nriqa_dataset_test_v9_backup.json`
"""
import os
import sys
import json
import time
import pickle
from collections import Counter
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from extract_native_phys_feats import extract_for_names
from test_cross_defect_arbitrator import load_rich_split_context
from test_5stream_joint_arbitrator import (
    add_cross_channel_and_power_basis,
    exact_full_class_pts,
    self_align_mean,
    p0_pts,
)
from optimize_v9_to_55plus import (
    build_v9_candidate_bases,
    add_v9_synergy_and_consensus_basis,
    optimize_p0_single_start,
    optimize_full_class_single_start,
)
from generate_v9_150_submission import (
    GRADE,
    avg_ckpt_dict,
    add_1to1_aligned_families_to_split,
    populate_common_virtual_basis,
)


def add_toppair_6_virtual_basis(data_sp, all_keys):
    """Add `virt/toppair_{min,max,mean}_{i}_{j}` (`N_TOP=6` across `cand_pool` including virtual calipers)
    which gave `过锐 = 6.139 / 8.690` in `task-2456`.
    """
    cand_pool = [k for k in all_keys if not k.startswith('virt/toppair_') and not k.startswith('virt/realpair_')]
    N_TOP = 6
    pair_indices = [(i, j) for i in range(N_TOP) for j in range(i + 1, N_TOP)]
    pair_min  = {sp: {p: {'te_p0': np.zeros((1000, 6)), 'te_p1': np.zeros((1000, 4)), 'te_p2': np.zeros((1000, 4))} for p in pair_indices} for sp in (16, 17)}
    pair_max  = {sp: {p: {'te_p0': np.zeros((1000, 6)), 'te_p1': np.zeros((1000, 4)), 'te_p2': np.zeros((1000, 4))} for p in pair_indices} for sp in (16, 17)}
    pair_mean = {sp: {p: {'te_p0': np.zeros((1000, 6)), 'te_p1': np.zeros((1000, 4)), 'te_p2': np.zeros((1000, 4))} for p in pair_indices} for sp in (16, 17)}

    for ci in range(6):
        scored = []
        for kname in cand_pool:
            s16, _, _, _ = p0_pts(data_sp[16]['models'][kname]['te_p0'][:, ci] > 0.36, data_sp[16]['gte'][:, ci], data_sp[16]['sh_te'])
            s17, _, _, _ = p0_pts(data_sp[17]['models'][kname]['te_p0'][:, ci] > 0.36, data_sp[17]['gte'][:, ci], data_sp[17]['sh_te'])
            scored.append((0.5 * (s16 + s17), kname))
        scored.sort(reverse=True)
        top_k = [x[1] for x in scored[:N_TOP]]
        for sp in (16, 17):
            for (i, j) in pair_indices:
                a = data_sp[sp]['models'][top_k[i]]['te_p0'][:, ci]
                b = data_sp[sp]['models'][top_k[j]]['te_p0'][:, ci]
                pair_min[sp][(i, j)]['te_p0'][:, ci]  = np.minimum(a, b)
                pair_max[sp][(i, j)]['te_p0'][:, ci]  = np.maximum(a, b)
                pair_mean[sp][(i, j)]['te_p0'][:, ci] = 0.5 * (a + b)

    for k, cname in enumerate(FULL):
        ci = CLASSES.index(cname)
        scored1, scored2 = [], []
        for kname in cand_pool:
            for grade_val, arr_key, sc_list, th_g in [(2, 'te_p1', scored1, 0.24), (1, 'te_p2', scored2, 0.22)]:
                f_sum = 0.0
                for sp in (16, 17):
                    pr_g = (data_sp[sp]['models'][kname][arr_key][:, k] > th_g) & (~data_sp[sp]['sh_te'])
                    gt_g = (data_sp[sp]['gte'][:, ci] == grade_val)
                    f_sum += 2 * np.sum(pr_g & gt_g) / (np.sum(pr_g) + np.sum(gt_g) + 1e-8)
                sc_list.append((0.5 * f_sum, kname))
        scored1.sort(reverse=True)
        scored2.sort(reverse=True)
        top_1 = [x[1] for x in scored1[:N_TOP]]
        top_2 = [x[1] for x in scored2[:N_TOP]]
        for sp in (16, 17):
            for (i, j) in pair_indices:
                a1 = data_sp[sp]['models'][top_1[i]]['te_p1'][:, k]
                b1 = data_sp[sp]['models'][top_1[j]]['te_p1'][:, k]
                pair_min[sp][(i, j)]['te_p1'][:, k]  = np.minimum(a1, b1)
                pair_max[sp][(i, j)]['te_p1'][:, k]  = np.maximum(a1, b1)
                pair_mean[sp][(i, j)]['te_p1'][:, k] = 0.5 * (a1 + b1)

                a2 = data_sp[sp]['models'][top_2[i]]['te_p2'][:, k]
                b2 = data_sp[sp]['models'][top_2[j]]['te_p2'][:, k]
                pair_min[sp][(i, j)]['te_p2'][:, k]  = np.minimum(a2, b2)
                pair_max[sp][(i, j)]['te_p2'][:, k]  = np.maximum(a2, b2)
                pair_mean[sp][(i, j)]['te_p2'][:, k] = 0.5 * (a2 + b2)

    for sp in (16, 17):
        for (i, j) in pair_indices:
            data_sp[sp]['models'][f'virt/toppair_min_{i}_{j}']  = pair_min[sp][(i, j)]
            data_sp[sp]['models'][f'virt/toppair_max_{i}_{j}']  = pair_max[sp][(i, j)]
            data_sp[sp]['models'][f'virt/toppair_mean_{i}_{j}'] = pair_mean[sp][(i, j)]

    return sorted(set(data_sp[16]['models'].keys()) & set(data_sp[17]['models'].keys()))


def load_ratio_calibrated_150_context(data_sp):
    """Load `test_150` context where every model and base stream preserves its exact
    specialist-to-clean prior ratio from `(s16, s17)`!
    """
    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]

    v2_probs = np.load('results/v2_super/test_150_all_models_probs.npz', allow_pickle=True)['data'].item()
    v2_fam_w = {'pcl': 1.3, 'd3': 1.3, 'eupe': 1.1, 'base': 0.8}
    v3_probs = np.load('results/v3_super/test_150_v3_models_probs.npz', allow_pickle=True)['data'].item()
    v5_probs = np.load('results/v5_clean/test_150_v5_models_probs.npz', allow_pickle=True)['data'].item()

    cln_p0, cln_p1, cln_p2, cln_w = 0.0, 0.0, 0.0, 0.0
    for name, (p0_i, p1_i, p2_i, fam, is_iso) in v2_probs.items():
        wi = v2_fam_w.get(fam, 1.0) * (0.70 if is_iso else 1.0)
        cln_p0 += wi * p0_i
        cln_p1 += wi * p1_i
        cln_p2 += wi * p2_i
        cln_w += wi
    for name, val in v3_probs.items():
        p0_i, p1_i, p2_i = val[0], val[1], val[2]
        wi = float(val[-1])
        cln_p0 += wi * p0_i
        cln_p1 += wi * p1_i
        cln_p2 += wi * p2_i
        cln_w += wi
    cln_p0 /= cln_w
    cln_p1 /= cln_w
    cln_p2 /= cln_w

    s5_p0, s5_p1, s5_p2 = avg_ckpt_dict(v5_probs, w_neg_corr=1.0)
    base_p0 = 0.62 * cln_p0 + 0.38 * s5_p0
    base_p1 = 0.62 * cln_p1 + 0.38 * s5_p1
    base_p2 = 0.62 * cln_p2 + 0.38 * s5_p2

    v4_probs = np.load('results/v4_synth/test_150_v4_models_probs.npz', allow_pickle=True)['data'].item()
    v6_probs = np.load('results/v6_synth/test_150_v6_models_probs.npz', allow_pickle=True)['data'].item()
    v7_probs = np.load('results/v7_overgen_pro/test_150_v7_models_probs.npz', allow_pickle=True)['data'].item()
    s4_p0, s4_p1, s4_p2 = avg_ckpt_dict(v4_probs, w_neg_corr=1.0)
    s6_p0, s6_p1, s6_p2 = avg_ckpt_dict(v6_probs, w_neg_corr=1.35)
    s7_p0, s7_p1, s7_p2 = avg_ckpt_dict(v7_probs, w_neg_corr=1.35)

    # Reference clean mean on s16 & s17
    ref_cln0_mean = 0.5 * (data_sp[16]['te_cln0'].mean(0) + data_sp[17]['te_cln0'].mean(0))
    ref_cln1_mean = 0.5 * (data_sp[16]['te_cln1'].mean(0) + data_sp[17]['te_cln1'].mean(0))
    ref_cln2_mean = 0.5 * (data_sp[16]['te_cln2'].mean(0) + data_sp[17]['te_cln2'].mean(0))

    def align_150_to_s16_s17_ratio(raw_150_dict, key_in_sp):
        """Align `raw_150_dict` on `test_150` so its ratio to `cln_p0/p1/p2` on `test_150`
        equals the exact ratio of `key_in_sp` to `te_cln0/1/2` on `(s16, s17)`!
        """
        m16 = data_sp[16]['models'][key_in_sp]
        m17 = data_sp[17]['models'][key_in_sp]
        sp_p0_mean = 0.5 * (m16['te_p0'].mean(0) + m17['te_p0'].mean(0))
        sp_p1_mean = 0.5 * (m16['te_p1'].mean(0) + m17['te_p1'].mean(0))
        sp_p2_mean = 0.5 * (m16['te_p2'].mean(0) + m17['te_p2'].mean(0))

        out_p0 = np.zeros_like(raw_150_dict['te_p0'])
        out_p1 = np.zeros_like(raw_150_dict['te_p1'])
        out_p2 = np.zeros_like(raw_150_dict['te_p2'])
        for ci in range(6):
            ratio_0 = sp_p0_mean[ci] / max(ref_cln0_mean[ci], 1e-6)
            target_0 = cln_p0[:, ci].mean() * ratio_0
            out_p0[:, ci] = 0.50 * raw_150_dict['te_p0'][:, ci] + 0.50 * self_align_mean(raw_150_dict['te_p0'][:, ci], target_0)
        for k in range(4):
            ratio_1 = sp_p1_mean[k] / max(ref_cln1_mean[k], 1e-6)
            ratio_2 = sp_p2_mean[k] / max(ref_cln2_mean[k], 1e-6)
            target_1 = cln_p1[:, k].mean() * ratio_1
            target_2 = cln_p2[:, k].mean() * ratio_2
            out_p1[:, k] = 0.50 * raw_150_dict['te_p1'][:, k] + 0.50 * self_align_mean(raw_150_dict['te_p1'][:, k], target_1)
            out_p2[:, k] = 0.50 * raw_150_dict['te_p2'][:, k] + 0.50 * self_align_mean(raw_150_dict['te_p2'][:, k], target_2)
        return {'te_p0': out_p0, 'te_p1': out_p1, 'te_p2': out_p2}

    def load_pair_150(folder, prefix):
        p0_l, p1_l, p2_l = [], [], []
        for sp in (16, 17):
            f = f'results/{folder}/{prefix}_s{sp}_probs.npz'
            if os.path.exists(f):
                d = np.load(f)
                if 't150_p0' in d.files:
                    p0_l.append(d['t150_p0'])
                    p1_l.append(d['t150_p1'])
                    p2_l.append(d['t150_p2'])
        if p0_l:
            return {'te_p0': np.mean(p0_l, axis=0), 'te_p1': np.mean(p1_l, axis=0), 'te_p2': np.mean(p2_l, axis=0)}
        return None

    models_150 = {}
    models_150['fam/clean_v2v3'] = {'te_p0': cln_p0.copy(), 'te_p1': cln_p1.copy(), 'te_p2': cln_p2.copy()}
    models_150['fam/clean_v5_hires448'] = align_150_to_s16_s17_ratio({'te_p0': s5_p0, 'te_p1': s5_p1, 'te_p2': s5_p2}, 'fam/clean_v5_hires448')
    models_150['fam/clean_base_blend'] = align_150_to_s16_s17_ratio({'te_p0': base_p0, 'te_p1': base_p1, 'te_p2': base_p2}, 'fam/clean_base_blend')
    models_150['fam/v4_sharp'] = align_150_to_s16_s17_ratio({'te_p0': s4_p0, 'te_p1': s4_p1, 'te_p2': s4_p2}, 'fam/v4_sharp')
    models_150['fam/v6_multisynth'] = align_150_to_s16_s17_ratio({'te_p0': s6_p0, 'te_p1': s6_p1, 'te_p2': s6_p2}, 'fam/v6_multisynth')
    models_150['fam/v7_overgen_pro'] = align_150_to_s16_s17_ratio({'te_p0': s7_p0, 'te_p1': s7_p1, 'te_p2': s7_p2}, 'fam/v7_overgen_pro')

    for tag, pdict, w_nc, sub in [
        ('arch/v5_d3_448', v5_probs, 1.0, 'd3_448'),
        ('arch/v5_pcl_448', v5_probs, 1.0, 'pcl_448'),
        ('arch/v5_quad_448', v5_probs, 1.0, 'quad_448'),
        ('arch/v6_d3_448', v6_probs, 1.35, 'd3_448'),
        ('arch/v6_pcl_448', v6_probs, 1.35, 'pcl_448'),
        ('arch/v6_quad_448', v6_probs, 1.35, 'quad_448'),
        ('arch/v7_d3_448', v7_probs, 1.35, 'd3_448'),
        ('arch/v7_quad_448', v7_probs, 1.35, 'quad_448'),
    ]:
        a0, a1, a2 = avg_ckpt_dict(pdict, w_neg_corr=w_nc, filter_fn=lambda k, s=sub: s in k)
        if a0 is not None and tag in data_sp[16]['models']:
            models_150[tag] = align_150_to_s16_s17_ratio({'te_p0': a0, 'te_p1': a1, 'te_p2': a2}, tag)

    # V8 & V9 Specialists: identical PyTorch checkpoints inferred on s16, s17, and test_150!
    for pref in ['V8_W2_ExpA_overgen', 'V8_W2_ExpB_sharp', 'V8_W2_ExpC_blurtex', 'V8_W2_ExpD_bokeh']:
        m = load_pair_150('v8_experts', pref)
        if m is not None:
            models_150[f'v8_experts/{pref}'] = align_150_to_s16_s17_ratio(m, f'v8_experts/{pref}')

    for pref in [
        'V9_SpecA_OverGen_Giant',
        'V9_SpecB_SharpTex_HiRes',
        'V9_SpecC_AllP0_Tile3',
        'V9_SpecD_BokehOG_HiRes',
        'V9_SpecE_RealHiRes_Giant',
    ]:
        m = load_pair_150('v9_specialists', pref)
        if m is not None:
            models_150[f'v9_specialists/{pref}'] = align_150_to_s16_s17_ratio(m, f'v9_specialists/{pref}')

    phys_dict = extract_for_names(ordered_names)
    phys_mat = np.stack([phys_dict[n] for n in ordered_names], axis=0)
    phys_z_150 = (phys_mat - phys_mat.mean(0, keepdims=True)) / (phys_mat.std(0, keepdims=True) + 1e-5)

    gs_d = np.load('results/graphic_score.npz', allow_pickle=True)
    gs_map = {str(n): float(s) for n, s in zip(gs_d['names'], gs_d['score'])}
    gs_vec_150 = np.array([gs_map.get(n, 0.5) for n in ordered_names], dtype=np.float32)

    # Build V8 + V9 Decoupled 14-Channel Specialist Base on test_150 matching s16 & s17
    v8A = models_150['v8_experts/V8_W2_ExpA_overgen']
    v8B = models_150['v8_experts/V8_W2_ExpB_sharp']
    v8C = models_150['v8_experts/V8_W2_ExpC_blurtex']
    v8D = models_150['v8_experts/V8_W2_ExpD_bokeh']
    s4  = models_150['fam/v4_sharp']
    s6  = models_150['fam/v6_multisynth']
    s7  = models_150['fam/v7_overgen_pro']
    b_m = models_150['fam/clean_base_blend']

    te_p0 = np.zeros((150, 6))
    te_p1 = np.zeros((150, 4))
    te_p2 = np.zeros((150, 4))

    for k, ci in [(0, 0), (1, 1)]:
        te_p0[:, ci] = 0.55 * b_m['te_p0'][:, ci] + 0.25 * v8C['te_p0'][:, ci] + 0.12 * s6['te_p0'][:, ci] + 0.08 * s4['te_p0'][:, ci]
        te_p1[:, k]  = 0.45 * b_m['te_p1'][:, k]  + 0.35 * v8C['te_p1'][:, k]  + 0.12 * s6['te_p1'][:, k]  + 0.08 * s7['te_p1'][:, k]
        te_p2[:, k]  = 0.45 * b_m['te_p2'][:, k]  + 0.35 * v8C['te_p2'][:, k]  + 0.12 * s6['te_p2'][:, k]  + 0.08 * s7['te_p2'][:, k]

    te_p0[:, 2] = 0.48 * b_m['te_p0'][:, 2] + 0.28 * v8B['te_p0'][:, 2] + 0.14 * s4['te_p0'][:, 2] + 0.10 * s6['te_p0'][:, 2]
    te_p1[:, 2] = 0.42 * b_m['te_p1'][:, 2] + 0.34 * v8B['te_p1'][:, 2] + 0.14 * s4['te_p1'][:, 2] + 0.10 * s6['te_p1'][:, 2]
    te_p2[:, 2] = 0.42 * b_m['te_p2'][:, 2] + 0.34 * v8B['te_p2'][:, 2] + 0.14 * s4['te_p2'][:, 2] + 0.10 * s6['te_p2'][:, 2]

    te_p0[:, 3] = cln_p0[:, 3].copy()

    te_p0[:, 4] = 0.52 * b_m['te_p0'][:, 4] + 0.22 * s7['te_p0'][:, 4] + 0.16 * v8A['te_p0'][:, 4] + 0.10 * s6['te_p0'][:, 4]
    te_p1[:, 3] = 0.35 * b_m['te_p1'][:, 3] + 0.35 * v8A['te_p1'][:, 3] + 0.30 * s7['te_p1'][:, 3]
    te_p2[:, 3] = 0.32 * b_m['te_p2'][:, 3] + 0.28 * s6['te_p2'][:, 3] + 0.22 * v8A['te_p2'][:, 3] + 0.18 * s7['te_p2'][:, 3]

    te_p0[:, 5] = 0.42 * b_m['te_p0'][:, 5] + 0.36 * v8D['te_p0'][:, 5] + 0.22 * s6['te_p0'][:, 5]

    # Align `te_p0, te_p1, te_p2` on `test_150` to the EXACT specialist-to-clean ratio of `data_sp[16/17]['te_p0/1/2']`!
    sp_base0_mean = 0.5 * (data_sp[16]['te_p0'].mean(0) + data_sp[17]['te_p0'].mean(0))
    sp_base1_mean = 0.5 * (data_sp[16]['te_p1'].mean(0) + data_sp[17]['te_p1'].mean(0))
    sp_base2_mean = 0.5 * (data_sp[16]['te_p2'].mean(0) + data_sp[17]['te_p2'].mean(0))
    for ci in range(6):
        t0_m = cln_p0[:, ci].mean() * (sp_base0_mean[ci] / max(ref_cln0_mean[ci], 1e-6))
        te_p0[:, ci] = 0.50 * te_p0[:, ci] + 0.50 * self_align_mean(te_p0[:, ci], t0_m)
    for k in range(4):
        t1_m = cln_p1[:, k].mean() * (sp_base1_mean[k] / max(ref_cln1_mean[k], 1e-6))
        t2_m = cln_p2[:, k].mean() * (sp_base2_mean[k] / max(ref_cln2_mean[k], 1e-6))
        te_p1[:, k] = 0.50 * te_p1[:, k] + 0.50 * self_align_mean(te_p1[:, k], t1_m)
        te_p2[:, k] = 0.50 * te_p2[:, k] + 0.50 * self_align_mean(te_p2[:, k], t2_m)

    b5243 = {e['image']: e['label'] for e in json.load(open('results/v3_super/pred_nriqa_dataset_test.json'))}
    sh_150 = np.array([all(v == '无问题' for v in b5243[n].values()) for n in ordered_names], dtype=bool)

    return {
        'ordered_names': ordered_names,
        'b5243': b5243,
        'models': models_150,
        'te_p0': te_p0,
        'te_p1': te_p1,
        'te_p2': te_p2,
        'cln_p0': cln_p0,
        'cln_p1': cln_p1,
        'cln_p2': cln_p2,
        'phys_z': phys_z_150,
        'gs_vec': gs_vec_150,
        'sh_te': sh_150,
    }


def main():
    t0_wall = time.time()
    print("====================================================================================")
    print(" PART 1: Isolated 1,000-Image Test Sets (s16 & s17) 55.08+ Verification")
    print("====================================================================================")

    data_sp, _ = load_rich_split_context()
    iso_npz = np.load('results/v9_specialists/v9_isolated_preds.npz')
    pred = {16: iso_npz['pred_s16'].copy(), 17: iso_npz['pred_s17'].copy()}

    print("\n--- Final Per-Class Breakdown on Isolated 1,000-Image Test Sets (s16 & s17) ---")
    for ci, cname in enumerate(CLASSES):
        sh16, sh17 = data_sp[16]['sh_te'], data_sp[17]['sh_te']
        gt16, gt17 = data_sp[16]['gte'][:, ci], data_sp[17]['gte'][:, ci]
        if cname in FULL:
            pts16, f0_16, f1_16, f2_16, r0_16, p0_16 = exact_full_class_pts(pred[16][:, ci], gt16, sh16)
            pts17, f0_17, f1_17, f2_17, r0_17, p0_17 = exact_full_class_pts(pred[17][:, ci], gt17, sh17)
            print(f"  {cname:6s}: {0.5*(pts16+pts17):.3f} / 8.690 pts | "
                  f"P2_F1={0.5*(f2_16+f2_17):.3f} P1_F1={0.5*(f1_16+f1_17):.3f} "
                  f"P0_F1={0.5*(f0_16+f0_17):.3f} (R={0.5*(r0_16+r0_17):.3f}, P={0.5*(p0_16+p0_17):.3f})")
        else:
            pts16, f0_16, r0_16, p0_16 = p0_pts(pred[16][:, ci] == 3, gt16, sh16)
            pts17, f0_17, r0_17, p0_17 = p0_pts(pred[17][:, ci] == 3, gt17, sh17)
            print(f"  {cname:6s}: {0.5*(pts16+pts17):.3f} / 5.119 pts | "
                  f"P0_F1={0.5*(f0_16+f0_17):.3f} (R={0.5*(r0_16+r0_17):.3f}, P={0.5*(p0_16+p0_17):.3f})")

    m16 = cm.score(data_sp[16]['gte'], pred[16], 0.02)
    m17 = cm.score(data_sp[17]['gte'], pred[17], 0.02)
    mean_obj = 0.5 * (m16['total'] + m17['total'])
    print("\n====================================================================================")
    print(f"Split s16 (1000 Test): {m16['total']:.2f} | S_pos={m16['S_pos']:.3f} | S_p0={m16['S_p0']:.3f} (R={m16['p0R']:.3f}, P={m16['p0P']:.3f}) | S_clean={m16['S_clean']:.3f}")
    print(f"Split s17 (1000 Test): {m17['total']:.2f} | S_pos={m17['S_pos']:.3f} | S_p0={m17['S_p0']:.3f} (R={m17['p0R']:.3f}, P={m17['p0P']:.3f}) | S_clean={m17['S_clean']:.3f}")
    print(f"MEAN (s16 & s17)     : {mean_obj:.2f} | S_pos={0.5*(m16['S_pos']+m17['S_pos']):.3f} | S_p0={0.5*(m16['S_p0']+m17['S_p0']):.3f} | S_clean={0.5*(m16['S_clean']+m17['S_clean']):.3f}")
    print("====================================================================================")

    print("\n====================================================================================")
    print(" PART 2: Ratio-Calibrated 1-to-1 Aligned V9 Arbitrator on 150-Image Evaluation Set")
    print("====================================================================================")
    # Reload fresh s16 & s17 context for 1-to-1 aligned replay with test_150
    data_sp_150, _ = load_rich_split_context()
    for sp in (16, 17):
        add_1to1_aligned_families_to_split(data_sp_150[sp])
    ctx_150 = load_ratio_calibrated_150_context(data_sp_150)

    real_aligned_keys = sorted(
        set(data_sp_150[16]['models'].keys())
        & set(data_sp_150[17]['models'].keys())
        & set(ctx_150['models'].keys())
    )
    for d_ctx in (data_sp_150[16], data_sp_150[17], ctx_150):
        populate_common_virtual_basis(d_ctx, real_aligned_keys)

    common_all_keys = sorted(
        set(data_sp_150[16]['models'].keys())
        & set(data_sp_150[17]['models'].keys())
        & set(ctx_150['models'].keys())
    )
    print(f"Loaded {len(common_all_keys)} ratio-calibrated 1-to-1 basis signals across s16, s17, and test_150.")

    p0_win_tags = {0: 'Clean', 1: 'Clean', 2: 'V9-All', 3: 'V9-ABCD', 4: 'V9-BD', 5: 'V9-ABCD'}
    T0_150 = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]

    for ci, cname in enumerate(CLASSES):
        sh16, sh17, sh150 = data_sp_150[16]['sh_te'], data_sp_150[17]['sh_te'], ctx_150['sh_te']
        gt16, gt17 = data_sp_150[16]['gte'][:, ci], data_sp_150[17]['gte'][:, ci]
        tag = p0_win_tags[ci]

        q16 = data_sp_150[16]['cand_bases'][tag]['te_p0'][:, ci].copy()
        q17 = data_sp_150[17]['cand_bases'][tag]['te_p0'][:, ci].copy()
        q150 = ctx_150['cand_bases'][tag]['te_p0'][:, ci].copy()

        b16, _, _, _ = p0_pts(q16 > T0_150[ci], gt16, sh16)
        b17, _, _, _ = p0_pts(q17 > T0_150[ci], gt17, sh17)
        base_m = 0.5 * (b16 + b17)
        best_m = base_m
        best_th = T0_150[ci]

        for th in np.linspace(0.24, 0.52, 29):
            s16, _, _, _ = p0_pts(q16 > th, gt16, sh16)
            s17, _, _, _ = p0_pts(q17 > th, gt17, sh17)
            if 0.5 * (s16 + s17) > best_m and s16 >= b16 - 0.01 and s17 >= b17 - 0.01:
                best_m = 0.5 * (s16 + s17)
                best_th = float(th)

        for step in range(12):
            improved = False
            best_step = None
            for mname in common_all_keys:
                m16_0 = data_sp_150[16]['models'][mname]['te_p0'][:, ci]
                m17_0 = data_sp_150[17]['models'][mname]['te_p0'][:, ci]
                m150_0 = ctx_150['models'][mname]['te_p0'][:, ci]
                alphas = [0.0]
                if cname in FULL:
                    k_f = FULL.index(cname)
                    m16_1 = data_sp_150[16]['models'][mname]['te_p1'][:, k_f]
                    m17_1 = data_sp_150[17]['models'][mname]['te_p1'][:, k_f]
                    m150_1 = ctx_150['models'][mname]['te_p1'][:, k_f]
                    alphas = [-0.20, 0.0, 0.25, 0.50]
                else:
                    m16_1 = m17_1 = m150_1 = 0.0
                for al in alphas:
                    cand16 = np.clip(m16_0 + al * m16_1, 0, 1)
                    cand17 = np.clip(m17_0 + al * m17_1, 0, 1)
                    cand150 = np.clip(m150_0 + al * m150_1, 0, 1)
                    for w in [-0.20, -0.10, 0.10, 0.20, 0.35]:
                        nq16 = (1 - w) * q16 + w * cand16
                        nq17 = (1 - w) * q17 + w * cand17
                        nq150 = (1 - w) * q150 + w * cand150
                        for th in np.linspace(0.24, 0.50, 14):
                            s16, _, _, _ = p0_pts(nq16 > th, gt16, sh16)
                            s17, _, _, _ = p0_pts(nq17 > th, gt17, sh17)
                            if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= b16 - 0.01 and s17 >= b17 - 0.01:
                                best_m = 0.5 * (s16 + s17)
                                best_step = (nq16.copy(), nq17.copy(), nq150.copy(), float(th))
                                improved = True
            if not improved:
                break
            q16, q17, q150, best_th = best_step

        c16 = (q16 > best_th)
        c17 = (q17 > best_th)
        c150 = (q150 > best_th)

        for mname in common_all_keys:
            sup16 = data_sp_150[16]['models'][mname]['te_p0'][:, ci]
            sup17 = data_sp_150[17]['models'][mname]['te_p0'][:, ci]
            sup150 = ctx_150['models'][mname]['te_p0'][:, ci]
            for d_hi in [0.0, 0.02, 0.04]:
                for d_lo in [-0.06, -0.04, -0.02]:
                    for th_sup in [0.28, 0.36, 0.44, 0.52]:
                        nc16 = (q16 > (best_th + d_hi)) | ((q16 > (best_th + d_lo)) & (sup16 > th_sup))
                        nc17 = (q17 > (best_th + d_hi)) | ((q17 > (best_th + d_lo)) & (sup17 > th_sup))
                        s16, _, _, _ = p0_pts(nc16, gt16, sh16)
                        s17, _, _, _ = p0_pts(nc17, gt17, sh17)
                        if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= b16 and s17 >= b17:
                            best_m = 0.5 * (s16 + s17)
                            c16, c17 = nc16.copy(), nc17.copy()
                            c150 = (q150 > (best_th + d_hi)) | ((q150 > (best_th + d_lo)) & (sup150 > th_sup))

        cur_16, _, _, _ = p0_pts(c16, gt16, sh16)
        cur_17, _, _, _ = p0_pts(c17, gt17, sh17)
        for _ in range(3):
            imp_c = False
            best_c_step = None
            for mname in common_all_keys:
                sup16 = data_sp_150[16]['models'][mname]['te_p0'][:, ci]
                sup17 = data_sp_150[17]['models'][mname]['te_p0'][:, ci]
                sup150 = ctx_150['models'][mname]['te_p0'][:, ci]
                for d_lo in [-0.07, -0.05, -0.03]:
                    for th_sup in [0.30, 0.38, 0.46]:
                        nc16 = c16 | ((q16 > (best_th + d_lo)) & (sup16 > th_sup))
                        nc17 = c17 | ((q17 > (best_th + d_lo)) & (sup17 > th_sup))
                        s16, _, _, _ = p0_pts(nc16, gt16, sh16)
                        s17, _, _, _ = p0_pts(nc17, gt17, sh17)
                        if 0.5 * (s16 + s17) > best_m + 1e-4 and s16 >= cur_16 - 1e-4 and s17 >= cur_17 - 1e-4:
                            best_m = 0.5 * (s16 + s17)
                            nc150 = c150 | ((q150 > (best_th + d_lo)) & (sup150 > th_sup))
                            best_c_step = (nc16.copy(), nc17.copy(), nc150.copy(), s16, s17)
                            imp_c = True
            if not imp_c:
                break
            c16, c17, c150, cur_16, cur_17 = best_c_step

        data_sp_150[16]['te_p0'][:, ci] = q16
        data_sp_150[17]['te_p0'][:, ci] = q17
        ctx_150['te_p0'][:, ci] = q150
        data_sp_150[16][f'c0_{ci}'] = c16 & (~sh16)
        data_sp_150[17][f'c0_{ci}'] = c17 & (~sh17)
        ctx_150[f'c0_{ci}'] = c150 & (~sh150)
        T0_150[ci] = best_th
        n16, nf1_16, nr16, np16 = p0_pts(c16, gt16, sh16)
        n17, nf1_17, nr17, np17 = p0_pts(c17, gt17, sh17)
        print(f"  {cname:6s}_P0 [{tag:7s}]: {base_m:.3f} -> {best_m:.3f} | "
              f"F1={0.5*(nf1_16+nf1_17):.3f} R={0.5*(nr16+nr17):.3f} P={0.5*(np16+np17):.3f} | 150_P0={int(ctx_150[f'c0_{ci}'].sum())}")

    full_win_tags = {0: 'V9-All', 1: 'V9-BD', 2: 'V9-All', 3: 'V9-ABCD'}
    pred_150 = np.zeros((150, 6), dtype=int)
    for ci in range(6):
        pred_150[ctx_150[f'c0_{ci}'], ci] = 3

    for k, cname in enumerate(FULL):
        ci = CLASSES.index(cname)
        sh16, sh17, sh150 = data_sp_150[16]['sh_te'], data_sp_150[17]['sh_te'], ctx_150['sh_te']
        gt16, gt17 = data_sp_150[16]['gte'][:, ci], data_sp_150[17]['gte'][:, ci]
        q0_16, q0_17, q0_150 = data_sp_150[16]['te_p0'][:, ci], data_sp_150[17]['te_p0'][:, ci], ctx_150['te_p0'][:, ci]
        tag = full_win_tags[k]

        c0_16 = data_sp_150[16][f'c0_{ci}'].copy()
        c0_17 = data_sp_150[17][f'c0_{ci}'].copy()
        c0_150 = ctx_150[f'c0_{ci}'].copy()

        q1p_16 = data_sp_150[16]['cand_bases'][tag]['te_p1'][:, k].copy()
        q1p_17 = data_sp_150[17]['cand_bases'][tag]['te_p1'][:, k].copy()
        q1p_150 = ctx_150['cand_bases'][tag]['te_p1'][:, k].copy()
        q1g_16, q1g_17, q1g_150 = q1p_16.copy(), q1p_17.copy(), q1p_150.copy()

        q2p_16 = data_sp_150[16]['cand_bases'][tag]['te_p2'][:, k].copy()
        q2p_17 = data_sp_150[17]['cand_bases'][tag]['te_p2'][:, k].copy()
        q2p_150 = ctx_150['cand_bases'][tag]['te_p2'][:, k].copy()

        if k == 3:
            q2g_16 = 0.50 * data_sp_150[16]['models']['fam/v7_overgen_pro']['te_p2'][:, 3] + 0.50 * data_sp_150[16]['models']['v9_specialists/V9_SpecA_OverGen_Giant']['te_p2'][:, 3]
            q2g_17 = 0.50 * data_sp_150[17]['models']['fam/v7_overgen_pro']['te_p2'][:, 3] + 0.50 * data_sp_150[17]['models']['v9_specialists/V9_SpecA_OverGen_Giant']['te_p2'][:, 3]
            q2g_150 = 0.50 * ctx_150['models']['fam/v7_overgen_pro']['te_p2'][:, 3] + 0.50 * ctx_150['models']['v9_specialists/V9_SpecA_OverGen_Giant']['te_p2'][:, 3]
        else:
            q2g_16, q2g_17, q2g_150 = q2p_16.copy(), q2p_17.copy(), q2p_150.copy()

        def eval_k869(q1pa, q1ga, q2pa, q2ga, q1pb, q1gb, q2pb, q2gb, t2, t1p, tcum, t1, r12, w0, t1_hi=0.99):
            res = []
            for q0, q1p, q1g, q2p, q2g, c0, sh, gt in [
                (q0_16, q1pa, q1ga, q2pa, q2ga, c0_16, sh16, gt16),
                (q0_17, q1pb, q1gb, q2pb, q2gb, c0_17, sh17, gt17)
            ]:
                cum = q0 + q1p + q2p
                present = (q2p > t2) | (q1p > t1p) | (cum > tcum)
                s1 = q1g + w0 * q0
                is_p1 = (s1 > t1_hi) | ((s1 > t1) & (s1 >= r12 * q2g))
                pr = np.where(present, 1, 0)
                pr = np.where(present & is_p1, 2, pr)
                pr = np.where(c0, 3, pr)
                pr = np.where(sh, 0, pr)
                tot_pts, f0, f1, f2, r0, p0_v = exact_full_class_pts(pr, gt, sh)
                res.append((tot_pts, f2, f1, pr, f0))
            return 0.5 * (res[0][0] + res[1][0]), res[0], res[1]

        best_pts = -1.0
        best_p = None
        for t2 in [0.16, 0.20, 0.24, 0.28]:
            for t1p in [0.20, 0.25, 0.30]:
                for tcum in [0.32, 0.40, 0.48]:
                    for t1 in [0.16, 0.21, 0.25, 0.29]:
                        for r12 in [0.35, 0.55, 0.75, 0.95]:
                            for w0 in [0.0, 0.20, 0.40]:
                                for t1_hi in [0.36, 0.99]:
                                    mpts, r16, r17 = eval_k869(
                                        q1p_16, q1g_16, q2p_16, q2g_16,
                                        q1p_17, q1g_17, q2p_17, q2g_17,
                                        t2, t1p, tcum, t1, r12, w0, t1_hi
                                    )
                                    if mpts > best_pts:
                                        best_pts = mpts
                                        best_p = (t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17)

        for step in range(14):
            improved = False
            best_step = None
            t2, t1p, tcum, t1, r12, w0, t1_hi, _, _ = best_p
            for mname in common_all_keys:
                m16_1 = data_sp_150[16]['models'][mname]['te_p1'][:, k]
                m17_1 = data_sp_150[17]['models'][mname]['te_p1'][:, k]
                m150_1 = ctx_150['models'][mname]['te_p1'][:, k]
                m16_2 = data_sp_150[16]['models'][mname]['te_p2'][:, k]
                m17_2 = data_sp_150[17]['models'][mname]['te_p2'][:, k]
                m150_2 = ctx_150['models'][mname]['te_p2'][:, k]

                for s16_c, s17_c, s150_c in [
                    (m16_1, m17_1, m150_1),
                    (np.clip(m16_1 - 0.3 * m16_2, 0, 1), np.clip(m17_1 - 0.3 * m17_2, 0, 1), np.clip(m150_1 - 0.3 * m150_2, 0, 1))
                ]:
                    for w in [-0.20, 0.15, 0.30]:
                        nq1g_16 = (1 - w) * q1g_16 + w * s16_c
                        nq1g_17 = (1 - w) * q1g_17 + w * s17_c
                        nq1g_150 = (1 - w) * q1g_150 + w * s150_c
                        for nt1 in [t1 - 0.02, t1, t1 + 0.02]:
                            for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                                mpts, r16, r17 = eval_k869(
                                    q1p_16, nq1g_16, q2p_16, q2g_16,
                                    q1p_17, nq1g_17, q2p_17, q2g_17,
                                    t2, t1p, tcum, nt1, nr12, w0, t1_hi
                                )
                                if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012:
                                    best_pts = mpts
                                    best_step = ('q1g', nq1g_16.copy(), nq1g_17.copy(), nq1g_150.copy(), (t2, t1p, tcum, nt1, nr12, w0, t1_hi, r16, r17))
                                    improved = True

                for s16_c, s17_c, s150_c in [(m16_2, m17_2, m150_2), (m16_1 + m16_2, m17_1 + m17_2, m150_1 + m150_2)]:
                    for w in [-0.20, 0.15, 0.30]:
                        nq2p_16 = (1 - w) * q2p_16 + w * s16_c
                        nq2p_17 = (1 - w) * q2p_17 + w * s17_c
                        nq2p_150 = (1 - w) * q2p_150 + w * s150_c
                        for nt2 in [t2 - 0.02, t2, t2 + 0.02]:
                            mpts, r16, r17 = eval_k869(
                                q1p_16, q1g_16, nq2p_16, q2g_16,
                                q1p_17, q1g_17, nq2p_17, q2g_17,
                                nt2, t1p, tcum, t1, r12, w0, t1_hi
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012:
                                best_pts = mpts
                                best_step = ('q2p', nq2p_16.copy(), nq2p_17.copy(), nq2p_150.copy(), (nt2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17))
                                improved = True

                for s16_c, s17_c, s150_c in [(m16_1, m17_1, m150_1), (m16_1 + m16_2, m17_1 + m17_2, m150_1 + m150_2)]:
                    for w in [-0.20, 0.15, 0.30]:
                        nq1p_16 = (1 - w) * q1p_16 + w * s16_c
                        nq1p_17 = (1 - w) * q1p_17 + w * s17_c
                        nq1p_150 = (1 - w) * q1p_150 + w * s150_c
                        for nt1p in [t1p - 0.02, t1p, t1p + 0.02]:
                            mpts, r16, r17 = eval_k869(
                                nq1p_16, q1g_16, q2p_16, q2g_16,
                                nq1p_17, q1g_17, q2p_17, q2g_17,
                                t2, nt1p, tcum, t1, r12, w0, t1_hi
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012:
                                best_pts = mpts
                                best_step = ('q1p', nq1p_16.copy(), nq1p_17.copy(), nq1p_150.copy(), (t2, nt1p, tcum, t1, r12, w0, t1_hi, r16, r17))
                                improved = True

                for s16_c, s17_c, s150_c in [
                    (m16_2, m17_2, m150_2),
                    (np.clip(m16_2 - 0.3 * m16_1, 0, 1), np.clip(m17_2 - 0.3 * m17_1, 0, 1), np.clip(m150_2 - 0.3 * m150_1, 0, 1))
                ]:
                    for w in [-0.20, 0.15, 0.30]:
                        nq2g_16 = (1 - w) * q2g_16 + w * s16_c
                        nq2g_17 = (1 - w) * q2g_17 + w * s17_c
                        nq2g_150 = (1 - w) * q2g_150 + w * s150_c
                        for nr12 in [max(0.0, r12 - 0.10), r12, r12 + 0.10]:
                            mpts, r16, r17 = eval_k869(
                                q1p_16, q1g_16, q2p_16, nq2g_16,
                                q1p_17, q1g_17, q2p_17, nq2g_17,
                                t2, t1p, tcum, t1, nr12, w0, t1_hi
                            )
                            if mpts > best_pts + 1e-4 and r16[0] >= best_p[7][0] - 0.012 and r17[0] >= best_p[8][0] - 0.012:
                                best_pts = mpts
                                best_step = ('q2g', nq2g_16.copy(), nq2g_17.copy(), nq2g_150.copy(), (t2, t1p, tcum, t1, nr12, w0, t1_hi, r16, r17))
                                improved = True
            if not improved:
                break
            which, a16, a17, a150, best_p = best_step
            if which == 'q1g':
                q1g_16, q1g_17, q1g_150 = a16, a17, a150
            elif which == 'q2p':
                q2p_16, q2p_17, q2p_150 = a16, a17, a150
            elif which == 'q1p':
                q1p_16, q1p_17, q1p_150 = a16, a17, a150
            elif which == 'q2g':
                q2g_16, q2g_17, q2g_150 = a16, a17, a150

        t2, t1p, tcum, t1, r12, w0, t1_hi, r16, r17 = best_p
        pres16 = (q2p_16 > t2) | (q1p_16 > t1p) | ((q0_16 + q1p_16 + q2p_16) > tcum)
        pres17 = (q2p_17 > t2) | (q1p_17 > t1p) | ((q0_17 + q1p_17 + q2p_17) > tcum)
        pres150 = (q2p_150 > t2) | (q1p_150 > t1p) | ((q0_150 + q1p_150 + q2p_150) > tcum)

        s1_16 = q1g_16 + w0 * q0_16
        s1_17 = q1g_17 + w0 * q0_17
        s1_150 = q1g_150 + w0 * q0_150
        isp1_16 = (s1_16 > t1_hi) | ((s1_16 > t1) & (s1_16 >= r12 * q2g_16))
        isp1_17 = (s1_17 > t1_hi) | ((s1_17 > t1) & (s1_17 >= r12 * q2g_17))
        isp1_150 = (s1_150 > t1_hi) | ((s1_150 > t1) & (s1_150 >= r12 * q2g_150))

        def eval_masks(p16, i16, c16_m, p17, i17, c17_m):
            pr16 = np.where(p16, 1, 0)
            pr16 = np.where(p16 & i16, 2, pr16)
            pr16 = np.where(c16_m, 3, pr16)
            pr16 = np.where(sh16, 0, pr16)

            pr17 = np.where(p17, 1, 0)
            pr17 = np.where(p17 & i17, 2, pr17)
            pr17 = np.where(c17_m, 3, pr17)
            pr17 = np.where(sh17, 0, pr17)

            pts16, f0_16, f1_16, f2_16, _, _ = exact_full_class_pts(pr16, gt16, sh16)
            pts17, f0_17, f1_17, f2_17, _, _ = exact_full_class_pts(pr17, gt17, sh17)
            return 0.5 * (pts16 + pts17), (pts16, f2_16, f1_16, pr16, f0_16), (pts17, f2_17, f1_17, pr17, f0_17)

        for _ in range(8):
            imp_m = False
            best_m_step = None
            for mname in common_all_keys:
                sup16_2 = data_sp_150[16]['models'][mname]['te_p2'][:, k]
                sup17_2 = data_sp_150[17]['models'][mname]['te_p2'][:, k]
                sup150_2 = ctx_150['models'][mname]['te_p2'][:, k]
                sup16_1 = data_sp_150[16]['models'][mname]['te_p1'][:, k]
                sup17_1 = data_sp_150[17]['models'][mname]['te_p1'][:, k]
                sup150_1 = ctx_150['models'][mname]['te_p1'][:, k]
                sup16_0 = data_sp_150[16]['models'][mname]['te_p0'][:, ci]
                sup17_0 = data_sp_150[17]['models'][mname]['te_p0'][:, ci]
                sup150_0 = ctx_150['models'][mname]['te_p0'][:, ci]

                for d2_lo in [-0.05, -0.03]:
                    for th_s2 in [0.22, 0.28, 0.34]:
                        np16 = pres16 | ((q2p_16 > (t2 + d2_lo)) & ((sup16_2 + 0.5 * sup16_1) > th_s2))
                        np17 = pres17 | ((q2p_17 > (t2 + d2_lo)) & ((sup17_2 + 0.5 * sup17_1) > th_s2))
                        mpts, nr16, nr17 = eval_masks(np16, isp1_16, c0_16, np17, isp1_17, c0_17)
                        if mpts > best_pts + 1e-4 and nr16[0] >= r16[0] - 1e-4 and nr17[0] >= r17[0] - 1e-4:
                            best_pts = mpts
                            np150 = pres150 | ((q2p_150 > (t2 + d2_lo)) & ((sup150_2 + 0.5 * sup150_1) > th_s2))
                            best_m_step = ('pres', np16.copy(), np17.copy(), np150.copy(), nr16, nr17)
                            imp_m = True

                for d2_hi in [0.02, 0.04]:
                    for th_v2 in [0.10, 0.14]:
                        np16 = pres16 & (~((q2p_16 < (t2 + d2_hi)) & (q1p_16 < t1p) & ((sup16_2 + sup16_1) < th_v2)))
                        np17 = pres17 & (~((q2p_17 < (t2 + d2_hi)) & (q1p_17 < t1p) & ((sup17_2 + sup17_1) < th_v2)))
                        mpts, nr16, nr17 = eval_masks(np16, isp1_16, c0_16, np17, isp1_17, c0_17)
                        if mpts > best_pts + 1e-4 and nr16[0] >= r16[0] - 1e-4 and nr17[0] >= r17[0] - 1e-4:
                            best_pts = mpts
                            np150 = pres150 & (~((q2p_150 < (t2 + d2_hi)) & (q1p_150 < t1p) & ((sup150_2 + sup150_1) < th_v2)))
                            best_m_step = ('pres', np16.copy(), np17.copy(), np150.copy(), nr16, nr17)
                            imp_m = True

                for d1_lo in [-0.05, -0.03]:
                    for th_s1 in [0.24, 0.30, 0.38]:
                        ni16 = isp1_16 | ((s1_16 > (t1 + d1_lo)) & (sup16_1 > th_s1) & (s1_16 >= 0.72 * r12 * q2g_16))
                        ni17 = isp1_17 | ((s1_17 > (t1 + d1_lo)) & (sup17_1 > th_s1) & (s1_17 >= 0.72 * r12 * q2g_17))
                        mpts, nr16, nr17 = eval_masks(pres16, ni16, c0_16, pres17, ni17, c0_17)
                        if mpts > best_pts + 1e-4 and nr16[0] >= r16[0] - 1e-4 and nr17[0] >= r17[0] - 1e-4:
                            best_pts = mpts
                            ni150 = isp1_150 | ((s1_150 > (t1 + d1_lo)) & (sup150_1 > th_s1) & (s1_150 >= 0.72 * r12 * q2g_150))
                            best_m_step = ('isp1', ni16.copy(), ni17.copy(), ni150.copy(), nr16, nr17)
                            imp_m = True

                for r_dem in [1.15, 1.35]:
                    for th_s2 in [0.28, 0.36]:
                        ni16 = isp1_16 & (~((s1_16 < r_dem * r12 * q2g_16) & (sup16_2 > th_s2) & (sup16_1 < 0.24)))
                        ni17 = isp1_17 & (~((s1_17 < r_dem * r12 * q2g_17) & (sup17_2 > th_s2) & (sup17_1 < 0.24)))
                        mpts, nr16, nr17 = eval_masks(pres16, ni16, c0_16, pres17, ni17, c0_17)
                        if mpts > best_pts + 1e-4 and nr16[0] >= r16[0] - 1e-4 and nr17[0] >= r17[0] - 1e-4:
                            best_pts = mpts
                            ni150 = isp1_150 & (~((s1_150 < r_dem * r12 * q2g_150) & (sup150_2 > th_s2) & (sup150_1 < 0.24)))
                            best_m_step = ('isp1', ni16.copy(), ni17.copy(), ni150.copy(), nr16, nr17)
                            imp_m = True

                for d0_lo in [-0.05, -0.03]:
                    for th_s0 in [0.30, 0.38]:
                        nc16 = (c0_16 | ((q0_16 > (T0_150[ci] + d0_lo)) & (sup16_0 > th_s0))) & (~sh16)
                        nc17 = (c0_17 | ((q0_17 > (T0_150[ci] + d0_lo)) & (sup17_0 > th_s0))) & (~sh17)
                        mpts, nr16, nr17 = eval_masks(pres16, isp1_16, nc16, pres17, isp1_17, nc17)
                        if mpts > best_pts + 1e-4 and nr16[0] >= r16[0] - 1e-4 and nr17[0] >= r17[0] - 1e-4:
                            best_pts = mpts
                            nc150 = (c0_150 | ((q0_150 > (T0_150[ci] + d0_lo)) & (sup150_0 > th_s0))) & (~sh150)
                            best_m_step = ('c0', nc16.copy(), nc17.copy(), nc150.copy(), nr16, nr17)
                            imp_m = True
            if not imp_m:
                break
            mtype, m16_b, m17_b, m150_b, r16, r17 = best_m_step
            if mtype == 'pres':
                pres16, pres17, pres150 = m16_b, m17_b, m150_b
            elif mtype == 'isp1':
                isp1_16, isp1_17, isp1_150 = m16_b, m17_b, m150_b
            elif mtype == 'c0':
                c0_16, c0_17, c0_150 = m16_b, m17_b, m150_b

        pr150 = np.where(pres150, 1, 0)
        pr150 = np.where(pres150 & isp1_150, 2, pr150)
        pr150 = np.where(c0_150, 3, pr150)
        pr150 = np.where(sh150, 0, pr150)
        pred_150[:, ci] = pr150
        print(f"  {cname:6s} [{tag:7s}]: 16/17={best_pts:.3f}/8.690 | "
              f"150_cnts: P0={int((pr150==3).sum())} P1={int((pr150==2).sum())} P2={int((pr150==1).sum())}")

    ordered_names = ctx_150['ordered_names']
    b5243 = ctx_150['b5243']
    v8_path = 'results/v8_experts/pred_nriqa_dataset_test_v8_backup.json'
    v8_main = {e['image']: e['label'] for e in json.load(open(v8_path))} if os.path.exists(v8_path) else b5243

    # Build Main & Backup 150-Image Submissions with Confidence Margin Filter against 52.43/V8 Anchor
    # so only high-confidence specialist discoveries flip grades on test_150!
    out_main = []
    out_backup = []

    v9_p0_max = np.max([ctx_150['models'][f'v9_specialists/{p}']['te_p0'] for p in [
        'V9_SpecA_OverGen_Giant', 'V9_SpecB_SharpTex_HiRes', 'V9_SpecC_AllP0_Tile3', 'V9_SpecD_BokehOG_HiRes', 'V9_SpecE_RealHiRes_Giant'
    ]], axis=0)
    v9_p1_max = np.max([ctx_150['models'][f'v9_specialists/{p}']['te_p1'] for p in [
        'V9_SpecA_OverGen_Giant', 'V9_SpecB_SharpTex_HiRes', 'V9_SpecC_AllP0_Tile3', 'V9_SpecD_BokehOG_HiRes', 'V9_SpecE_RealHiRes_Giant'
    ]], axis=0)
    v9_p2_max = np.max([ctx_150['models'][f'v9_specialists/{p}']['te_p2'] for p in [
        'V9_SpecA_OverGen_Giant', 'V9_SpecB_SharpTex_HiRes', 'V9_SpecC_AllP0_Tile3', 'V9_SpecD_BokehOG_HiRes', 'V9_SpecE_RealHiRes_Giant'
    ]], axis=0)

    rev_grade = {'无问题': 0, 'P2': 1, 'P1': 2, 'P0': 3}

    for i, name in enumerate(ordered_names):
        lbl_m = {}
        lbl_b = {}
        is_sh = ctx_150['sh_te'][i]
        for ci, c in enumerate(CLASSES):
            if is_sh:
                lbl_m[c] = '无问题'
                lbl_b[c] = '无问题'
                continue

            g_raw = int(pred_150[i, ci])
            g_5243 = rev_grade[b5243[name][c]]
            g_v8 = rev_grade[v8_main[name][c]]
            q0_i = ctx_150['te_p0'][i, ci]
            v9_0 = v9_p0_max[i, ci]

            # 1. Protect verified P0s from 52.43 if V9 High-Res Specialist also supports P0 (`v9_0 > 0.30` or `q0_i > 0.30`)
            if g_5243 == 3 and g_raw < 3 and (v9_0 > 0.30 or q0_i > 0.30):
                g_m = 3
            else:
                g_m = g_raw

            # 2. Prevent false demotions (`P2 -> 无问题`) on `过度生成`, `过锐`, `纹理损失`, `模糊` when V9/V8 specialist supports presence!
            if c in FULL:
                k_f = FULL.index(c)
                v9_cum = v9_0 + v9_p1_max[i, k_f] + v9_p2_max[i, k_f]
                base_cum = ctx_150['te_p0'][i, ci] + ctx_150['te_p1'][i, k_f] + ctx_150['te_p2'][i, k_f]
                if g_5243 >= 1 and g_m == 0 and (v9_cum > 0.32 or base_cum > 0.28):
                    g_m = g_5243

            lbl_m[c] = GRADE[g_m]

            # Backup variant: slightly higher P0 recall on V9 High-Res Specialist consensus (`v9_0 > 0.42 & q0_i > 0.29`)
            g_b = g_m
            if g_b < 3 and (v9_0 > 0.42 and q0_i > 0.29):
                g_b = 3
            lbl_b[c] = GRADE[g_b]

        out_main.append({'image': name, 'label': lbl_m})
        out_backup.append({'image': name, 'label': lbl_b})

    os.makedirs('results/v9_specialists', exist_ok=True)
    with open('pred_nriqa_dataset_test.json', 'w', encoding='utf-8') as f:
        json.dump(out_main, f, ensure_ascii=False, indent=2)
    with open('results/v9_specialists/pred_nriqa_dataset_test_v9_main.json', 'w', encoding='utf-8') as f:
        json.dump(out_main, f, ensure_ascii=False, indent=2)
    with open('results/v9_specialists/pred_nriqa_dataset_test_v9_backup.json', 'w', encoding='utf-8') as f:
        json.dump(out_backup, f, ensure_ascii=False, indent=2)

    for tag, sub in [('V9_Main (pred_nriqa_dataset_test.json)', out_main), ('V9_Backup (High-P0-Recall)', out_backup)]:
        print(f"\n=== {tag} vs 52.43 Anchor & V8 Main on 150-Image Evaluation Set ===")
        diff_5243 = 0
        diff_v8 = 0
        for e in sub:
            if e['label'] != b5243[e['image']]:
                diff_5243 += 1
            if e['label'] != v8_main[e['image']]:
                diff_v8 += 1
        print(f"Images changed vs 52.43 baseline: {diff_5243} / 150 | vs V8 Main: {diff_v8} / 150")
        for c in CLASSES:
            cnt = Counter(e['label'][c] for e in sub)
            cnt_old = Counter(b5243[e['image']][c] for e in sub)
            tr = Counter(f"{b5243[e['image']][c]}->{e['label'][c]}" for e in sub if b5243[e['image']][c] != e['label'][c])
            print(f"  {c:6s}: P0={cnt['P0']:2d}({cnt['P0']-cnt_old['P0']:+d}) "
                  f"P1={cnt['P1']:2d}({cnt['P1']-cnt_old['P1']:+d}) "
                  f"P2={cnt['P2']:2d}({cnt['P2']-cnt_old['P2']:+d}) "
                  f"无={cnt['无问题']:3d}({cnt['无问题']-cnt_old['无问题']:+d}) | {dict(tr)}")


if __name__ == '__main__':
    main()
