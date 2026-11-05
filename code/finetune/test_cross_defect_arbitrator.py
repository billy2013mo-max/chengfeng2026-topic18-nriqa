"""Optimize a 2-Stage Cross-Channel + Physical-Caliper Conditional Arbitrator
that takes the 14 decoupled specialist channels + family-level consensus/disagreement
+ physical/scene calipers (`graphic_score`, `log_wh`, `laplacian_ratio`, `halo_ringing`, `bimodality`, `kurtosis`)
and optimizes per-class 4-state log-posterior scoring functions!

Key Mathematical Insight:
In `build_v8_final_submission_and_eval.py`, each class `c` is decoded using ONLY its own scalar `q0, q1, q2p, q2g`:
  `present = (q2p > t2) | (q1 > t1p) | (cum > tcum)`
  `s1 = q1 + w0 * q0`
  `is_p1 = (s1 > t1) & (s1 >= r12 * q2g)`
  `c0 = (q0 > t0) & (~sh)`
This ignores:
1. Cross-defect signals! For example:
   - `纹理损失` presence and P1/P2 grade strongly depend on `cum_blur`, `cum_sharp`, and `cum_overgen`!
   - `过锐` presence and P1/P0 grade strongly depend on `cum_tex` (95% co-occurrence!), `cum_blur` (negative correlation!), and physical `halo_overshoot`!
   - `过度生成` presence and P2/P1/P0 grade strongly depend on `cum_tex`, `graphic_score` (anime/illustration vs photo), and `q0_anc` vs `q0_synth`!
   - `模糊` presence and P2/P1 grade strongly depend on `cum_tex` (positive), `p0_bokeh` (negative: portrait bokeh is NOT global blur!), and `laplacian_rolloff`!
   - `虚化突兀` P0 strongly depends on `p0_bokeh` AND `cum_blur` + `patch_sharpness_bimodality`!
2. Multi-Signal Non-Linear Interaction Terms (e.g. `max(models)` vs `mean(models)`, `std(models)` across the 60-model ensemble)!
   When `std(models)` or `p90(models) - mean(models)` is high on a rare defect (`过度生成`, `过锐`, `虚化突兀`), the specialist models detected a localized artifact that uniform models missed!
"""
import csv
import glob
import os
import sys
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from optimize_v8_isolated_test import load_healthy_split_models, odds_scale
from eval_decoupled_14ch_v8 import build_decoupled_14ch_probs
from extract_native_phys_feats import extract_for_names


def load_rich_split_context():
    man_rows = list(csv.DictReader(open('data/manifest_combined.csv')))
    man_map = {r['name']: r for r in man_rows}
    all_names = [r['name'] for r in man_rows]
    phys_dict = extract_for_names(all_names)
    gs_d = np.load('results/graphic_score.npz', allow_pickle=True)
    gs_map = {str(n): float(s) for n, s in zip(gs_d['names'], gs_d['score'])}

    splits = {}
    for sp in (16, 17):
        models, gva, gte = load_healthy_split_models(sp)
        for pref in ['pcl_448t3_seed1', 'd3_448t3_seed1', 'pcl_384t2_seed2', 'd3_384t2_seed1']:
            fep = f'results/v6_val/{pref}_s{sp}_epochs.npz'
            if os.path.exists(fep):
                dep = np.load(fep)
                for ep in (1, 2, 3, 4):
                    if f'ep{ep}_p0' in dep.files:
                        models[f'v6ep/{pref}_ep{ep}'] = {
                            'te_p0': dep[f'ep{ep}_p0'],
                            'te_p1': dep[f'ep{ep}_p1'],
                            'te_p2': dep[f'ep{ep}_p2'],
                        }
        ref = np.load(f'results/v3_ablate/A_base_s{sp}_probs.npz', allow_pickle=True)
        te_names = [str(x) for x in ref['te_names']]
        splits[sp] = (models, gva, gte, te_names)

    common_keys = sorted(set(splits[16][0].keys()) & set(splits[17][0].keys()))
    data_sp = {}
    a0 = [1.12, 0.68, 1.0, 1.0, 0.82, 0.88]
    for sp in (16, 17):
        models, gva, gte, te_names = splits[sp]
        models = {k: models[k] for k in common_keys}
        te_p0, te_p1, te_p2, te_cln0, te_cln1, te_cln2 = build_decoupled_14ch_probs(models, 'te')
        if 'v8_experts/V8_W2_ExpA_overgen' in models:
            te_p1[:, 3] = (
                0.28 * models['v8_experts/V8_ExpA_overgen']['te_p1'][:, 3]
                + 0.22 * models['v8_experts/V8_W2_ExpA_overgen']['te_p1'][:, 3]
                + 0.25 * models['v3_ablate/D_gen_p1']['te_p1'][:, 3]
                + 0.25 * models['v3_ablate/H_og']['te_p1'][:, 3]
            )
        for ci in range(6):
            te_p0[:, ci] = odds_scale(te_p0[:, ci], a0[ci])

        cum_cln = np.zeros((1000, 6))
        for ci, c in enumerate(CLASSES):
            if c in FULL:
                k = FULL.index(c)
                cum_cln[:, ci] = te_cln0[:, ci] + te_cln1[:, k] + te_cln2[:, k]
            else:
                cum_cln[:, ci] = te_cln0[:, ci]
        sh_te = cum_cln.max(1) < 0.28

        # Build auxiliary normalized signals per test image (zero GT used!)
        gs_vec = np.array([gs_map.get(n, 0.5) for n in te_names])
        phys_mat = np.stack([phys_dict[n] for n in te_names], axis=0)
        # Standardize phys_mat using its own batch mean/std (unsupervised)
        phys_z = (phys_mat - phys_mat.mean(0, keepdims=True)) / (phys_mat.std(0, keepdims=True) + 1e-5)

        # Stack all models for p90 / max / std statistics across the pool
        p0_stack = np.stack([models[m]['te_p0'] for m in common_keys], axis=0)  # (M, 1000, 6)
        p1_stack = np.stack([models[m]['te_p1'] for m in common_keys], axis=0)  # (M, 1000, 4)
        p2_stack = np.stack([models[m]['te_p2'] for m in common_keys], axis=0)  # (M, 1000, 4)

        data_sp[sp] = dict(
            models=models, gte=gte, te_names=te_names,
            te_p0=te_p0, te_p1=te_p1, te_p2=te_p2,
            te_cln0=te_cln0, te_cln1=te_cln1, te_cln2=te_cln2,
            cum_cln=cum_cln, sh_te=sh_te,
            gs_vec=gs_vec, phys_z=phys_z,
            p0_p90=np.percentile(p0_stack, 90, axis=0),
            p0_p80=np.percentile(p0_stack, 80, axis=0),
            p0_std=np.std(p0_stack, axis=0),
            p1_p90=np.percentile(p1_stack, 90, axis=0),
            p1_p80=np.percentile(p1_stack, 80, axis=0),
            p1_std=np.std(p1_stack, axis=0),
            p2_p90=np.percentile(p2_stack, 90, axis=0),
            p2_p80=np.percentile(p2_stack, 80, axis=0),
            p2_std=np.std(p2_stack, axis=0),
        )
    return data_sp, common_keys


if __name__ == '__main__':
    data_sp, common_keys = load_rich_split_context()
    print(f"Loaded rich context for {len(common_keys)} common models across s16 & s17.")
