import glob, numpy as np
for p in [
    'feats/internvit-6b_combined_oof.npz',
    'feats/c-radio-v4-h_combined_oof.npz',
    'feats/fuse_siglip_radio_combined_oof.npz',
    'results/hand_features.npz',
    'results/graphic_score.npz',
]:
    if glob.glob(p):
        d = np.load(p, allow_pickle=True)
        print(p, "->", {k: (d[k].shape, d[k].dtype) for k in d.files})

phys = np.load('results/v8_experts/native_phys_feats.npz', allow_pickle=True)['data'].item()
k0 = next(iter(phys.keys()))
print("native_phys_feats count:", len(phys), "sample key:", k0, "shape:", phys[k0].shape)
