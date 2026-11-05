import glob, os
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
ref16 = np.load('results/v3_ablate/A_base_s16_probs.npz')['gte']
ref17 = np.load('results/v3_ablate/A_base_s17_probs.npz')['gte']

m16, m17 = [], []
for p in sorted(glob.glob('results/**/*.npz', recursive=True)):
    try:
        d = np.load(p)
        if 'gte' in d and d['gte'].shape == (1000, 6):
            if np.array_equal(d['gte'], ref16):
                m16.append(p)
            elif np.array_equal(d['gte'], ref17):
                m17.append(p)
    except Exception:
        pass

print(f"Total exact gte match for s16: {len(m16)}, for s17: {len(m17)}")
for p in m16:
    print("  s16:", p)
