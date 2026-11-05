import os, sys, glob
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')

print("=== 1. Checking Available Pretrained Backbones & Weights on Disk ===")
search_dirs = [
    '/wanqing-develop/luowenjing/18/pretrained',
    '/root/.cache/huggingface/hub',
    '/root/.cache/torch/hub/checkpoints',
    '/wanqing-develop/luowenjing',
]
for d in search_dirs:
    if os.path.exists(d):
        print(f"\nDirectory: {d}")
        for root, dirs, files in os.walk(d):
            if any(x in root for x in ['results', 'work_dirs', 'synth_', 'data/images']):
                continue
            for f in files:
                if f.endswith(('.pth', '.pt', '.safetensors', '.bin', '.ckpt', '.npz')) and not root.startswith('/wanqing-develop/luowenjing/18/results'):
                    fp = os.path.join(root, f)
                    sz = os.path.getsize(fp) / (1024*1024)
                    if sz > 5:
                        print(f"  {fp} ({sz:.1f} MB)")

import timm
print(f"\ntimm version: {timm.__version__}")
