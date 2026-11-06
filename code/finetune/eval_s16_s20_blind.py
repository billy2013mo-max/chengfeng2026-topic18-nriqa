"""Evaluation and Statistical Significance Analysis on s16 ~ s20 Blind Test Splits.
Compares InternViT-6B pure-real models against baseline fixed backbones (base, pcl, pcg).
"""
import os, json, glob
import numpy as np

SPLITS = ['s16', 's17', 's18', 's19', 's20']
DEFECTS = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']


def load_split_results(prefix_pattern):
    results = {}
    for sp in SPLITS:
        files = glob.glob(prefix_pattern.format(sp=sp))
        if files:
            with open(files[0]) as f:
                d = json.load(f)
                results[sp] = d
    return results


def summarize_models(name, res_dict):
    if len(res_dict) < len(SPLITS):
        print(f"[{name}] Incomplete: found {len(res_dict)}/{len(SPLITS)} splits.")
        return None
    totals = [res_dict[sp]['test']['total'] for sp in SPLITS]
    spos = [res_dict[sp]['test']['S_pos'] for sp in SPLITS]
    sp0 = [res_dict[sp]['test']['S_p0'] for sp in SPLITS]
    sclean = [res_dict[sp]['test']['S_clean'] for sp in SPLITS]
    fpr = [res_dict[sp]['test']['fpr'] for sp in SPLITS]
    
    summary = {
        'name': name,
        'total_mean': float(np.mean(totals)),
        'total_std': float(np.std(totals)),
        'spos_mean': float(np.mean(spos)),
        'sp0_mean': float(np.mean(sp0)),
        'sclean_mean': float(np.mean(sclean)),
        'fpr_mean': float(np.mean(fpr)),
        'totals': totals,
        'spos': spos,
        'sp0': sp0,
    }
    return summary


def main():
    print("=" * 80)
    print("=== Independent Blind Evaluation on s16 ~ s20 Zero-Leakage Splits ===")
    print("=" * 80)
    
    baselines = {
        'C-RADIO+SigLIP2 (base)': 'results/backbone_fixed/base_{sp}_seed1.json',
        'PE-Core-L (pcl)': 'results/backbone_fixed/pcl_{sp}_seed1.json',
        'PE-Core-G (pcg)': 'results/backbone_fixed/pcg_{sp}_seed1.json',
        'InternViT-6B Pure-Real': 'results/v11_pure_real/iv6b_{sp}.json',
    }
    
    summaries = {}
    for label, pat in baselines.items():
        res = load_split_results(pat)
        s = summarize_models(label, res)
        if s:
            summaries[label] = s
            print(f"{label:26s} | Total: {s['total_mean']:.3f} ± {s['total_std']:.3f} | S_pos: {s['spos_mean']:.4f} | S_p0: {s['sp0_mean']:.4f} | S_clean: {s['sclean_mean']:.4f} | FPR: {s['fpr_mean']:.4f}")

    if 'InternViT-6B Pure-Real' in summaries and 'C-RADIO+SigLIP2 (base)' in summaries:
        iv = summaries['InternViT-6B Pure-Real']
        base = summaries['C-RADIO+SigLIP2 (base)']
        deltas = np.array(iv['totals']) - np.array(base['totals'])
        spos_deltas = np.array(iv['spos']) - np.array(base['spos'])
        sp0_deltas = np.array(iv['sp0']) - np.array(base['sp0'])
        print("-" * 80)
        print("Paired Deltas vs Base (InternViT-6B Pure-Real - Base):")
        for i, sp in enumerate(SPLITS):
            print(f"  {sp}: Total Δ={deltas[i]:+.3f} | S_pos Δ={spos_deltas[i]:+.4f} | S_p0 Δ={sp0_deltas[i]:+.4f}")
        print(f"Mean Total Delta: {deltas.mean():+.3f} pts (Positive in {np.sum(deltas > 0)}/5 splits)")
        print(f"Mean S_pos Delta: {spos_deltas.mean():+.4f} ({spos_deltas.mean()*25:+.3f} pts)")
        print(f"Mean S_p0 Delta:  {sp0_deltas.mean():+.4f} ({sp0_deltas.mean()*20:+.3f} pts)")


if __name__ == '__main__':
    main()
