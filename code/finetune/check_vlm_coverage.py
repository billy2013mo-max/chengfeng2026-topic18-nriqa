import os, sys, json, collections
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')
from common import CLASSES, FULL

LET = 'ABCD'
FULL_IDX = [i for i, c in enumerate(CLASSES) if c in FULL]

def load_vlm_dict(path):
    by = collections.defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        by[r['image']][r['defect']] = r
    out = {}
    for im, d in by.items():
        if len(d) != 6:
            continue
        p0 = np.zeros(6, dtype=np.float32)
        p1 = np.zeros(4, dtype=np.float32)
        p2 = np.zeros(4, dtype=np.float32)
        for ci, c in enumerate(CLASSES):
            r = d[c]
            o = r['options']
            def gp(g):
                return float(r['probs'].get(LET[o.index(g)], 0.0)) if g in o else 0.0
            p0[ci] = gp('P0')
            if c in FULL:
                j = FULL_IDX.index(ci)
                p1[j] = gp('P1')
                p2[j] = gp('P2')
        out[im] = (p0, p1, p2)
    return out

for name, fp in [
    ('qwen38-27b', 'results/qwen38-27b.jsonl'),
    ('jev-native', 'results/jev-native.jsonl'),
    ('clef',       'results/clef.jsonl'),
    ('clef-flash', 'results/clef-flash.jsonl'),
    ('nemotron',   'results/nemotron-omni.jsonl'),
    ('qwen3vl-2b', 'results/qwen3vl-2b.jsonl'),
]:
    vd = load_vlm_dict(fp)
    ref16 = np.load('results/v3_ablate/A_base_s16_probs.npz', allow_pickle=True)
    ref17 = np.load('results/v3_ablate/A_base_s17_probs.npz', allow_pickle=True)
    te16 = [str(x) for x in ref16['te_names']]
    te17 = [str(x) for x in ref17['te_names']]
    cov16 = sum(1 for x in te16 if x in vd)
    cov17 = sum(1 for x in te17 if x in vd)
    print(f"{name:12s}: total={len(vd)} | s16_te cov={cov16}/1000 | s17_te cov={cov17}/1000")
