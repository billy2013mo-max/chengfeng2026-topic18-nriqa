import os, sys, csv
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
for d in ['documents', 'code/synth', 'code/hpo']:
    if os.path.exists(d):
        print(f"=== {d} ===")
        for f in sorted(os.listdir(d)):
            print(" ", f)

for f in ['results/swift_qwen3vl2b_lora_probs.npz', 'results/swift_qwen3vl8b_lora_probs.npz', 'results/hand_features.npz', 'results/graphic_score.npz']:
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        print(f"\n{f}:", {k: d[k].shape for k in d.files})
