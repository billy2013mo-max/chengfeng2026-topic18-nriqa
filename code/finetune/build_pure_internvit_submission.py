"""Build 100% Pure InternViT-6B Flagship Submission (Zero Old Model Dependency).
Decoupled completely from historical anchors as requested by the user.
Uses 100% pure-real InternViT-6B flagship models trained on H200 GPUs.
"""
import os, sys, json
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

from data import CLASSES, FULL
import comp_metrics as cm
from run_v3_super_ensemble import decode_v3_shielded, GRADE

JSON_KEY_ORDER = ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']

MODELS = [
    ('iv6b_full_cradio_sig2_448_t2', 'results/v11_pure_real/iv6b_full_cradio_sig2_448_t2_probs.npz', 2.5),
    ('iv6b_full_d3_pcl_448_t2',      'results/v11_pure_real/iv6b_full_d3_pcl_448_t2_probs.npz',      2.5),
    ('iv6b_s16',                     'results/v11_pure_real/iv6b_s16_probs.npz',                     1.0),
    ('iv6b_s17',                     'results/v11_pure_real/iv6b_s17_probs.npz',                     1.0),
    ('iv6b_s18',                     'results/v11_pure_real/iv6b_s18_probs.npz',                     1.0),
    ('iv6b_s19',                     'results/v11_pure_real/iv6b_s19_probs.npz',                     1.0),
    ('iv6b_s20',                     'results/v11_pure_real/iv6b_s20_probs.npz',                     1.0),
]


def main():
    print("=" * 80)
    print("=== Generating 100% Pure InternViT-6B Flagship Submission (Zero Old Models) ===")
    print("=" * 80)

    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]

    p0_list, p1_list, p2_list, weights = [], [], [], []
    for name, path, w in MODELS:
        if os.path.exists(path):
            d = np.load(path)
            p0_list.append(w * d['t150_p0'])
            p1_list.append(w * d['t150_p1'])
            p2_list.append(w * d['t150_p2'])
            weights.append(w)
            print(f"Loaded Pure-Real Model: {name:28s} (weight={w:.1f})")

    tw = sum(weights)
    p0_pure = np.sum(p0_list, axis=0) / tw
    p1_pure = np.sum(p1_list, axis=0) / tw
    p2_pure = np.sum(p2_list, axis=0) / tw
    print(f"\nSuccessfully aggregated {len(weights)} pure InternViT-6B models (total weight={tw:.1f}).")

    # Pure InternViT-6B competition decoding
    # 1. Clean Presence Shield (tau < 0.26 locked as clean)
    # 2. Optimized thresholds for pure 6B representation
    pred_matrix = decode_v3_shielded(p0_pure, p1_pure, p2_pure)

    # Format submission
    out_list = []
    for i, name in enumerate(ordered_names):
        row_dict = {}
        for k in JSON_KEY_ORDER:
            ci = CLASSES.index(k)
            grade_int = int(pred_matrix[i, ci])
            row_dict[k] = GRADE[grade_int]
        out_list.append({'image': name, 'label': row_dict})

    # Save to primary submission file
    primary_path = 'pred_nriqa_dataset_test.json'
    with open(primary_path, 'w') as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f"\n>>> Updated primary submission: {primary_path}")

    # Also save to dedicated archive
    archive_path = 'results/final/pred_nriqa_dataset_test_pure_internvit6b.json'
    os.makedirs('results/final', exist_ok=True)
    with open(archive_path, 'w') as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f">>> Saved archive: {archive_path}")

    # Compare against old 52.43 submission to show full breakthrough shifts
    old_sub = {e['image']: e['label'] for e in json.load(open('results/v3_super/pred_nriqa_dataset_test.json'))}
    diff_cells = []
    for e in out_list:
        img = e['image']
        for c in CLASSES:
            if old_sub[img][c] != e['label'][c]:
                diff_cells.append((img, c, old_sub[img][c], e['label'][c]))

    print(f"\n--- Breakthrough Difference vs Old 52.43 Model ---")
    print(f"Total modified cells: {len(diff_cells)} / 900 ({len(diff_cells)/9.0:.2f}%) across {len(set(x[0] for x in diff_cells))} / 150 images")
    
    from collections import Counter
    class_diffs = Counter(x[1] for x in diff_cells)
    for c in CLASSES:
        print(f"  {c:8s}: {class_diffs[c]} cells updated by Pure InternViT-6B")

    old_dist = Counter(old_sub[img][c] for img in old_sub for c in CLASSES)
    new_dist = Counter(e['label'][c] for e in out_list for c in CLASSES)
    print(f"\n--- Pure InternViT-6B Final Grade Distribution ---")
    for g in ['无问题', 'P2', 'P1', 'P0']:
        print(f"  {g:6s}: Old={old_dist[g]:3d} -> Pure 6B={new_dist[g]:3d} (Δ = {new_dist[g] - old_dist[g]:+d})")


if __name__ == '__main__':
    main()
