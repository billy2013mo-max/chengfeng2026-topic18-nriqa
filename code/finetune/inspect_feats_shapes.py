import numpy as np

for p in [
    'feats/internvit-6b_combined.npz',
    'feats/c-radio-v4-h_combined.npz',
    'feats/siglip2-so400m_combined.npz',
    'feats/pe-core-l14_combined.npz',
    'feats/eupe-vit-b_combined.npz',
    'feats/convnextv2-base_combined.npz',
    'results/v8_experts/native_phys_feats.npz',
    'results/v9_specialists/V9_SpecA_OverGen_Giant_s16_probs.npz',
    'results/v3_ablate/A_base_s16_probs.npz',
]:
    d = np.load(p, allow_pickle=True)
    info = {k: (d[k].shape, d[k].dtype) for k in d.files}
    print(p, "->", info)
