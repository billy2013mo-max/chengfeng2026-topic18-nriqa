import json, glob
import numpy as np
from collections import Counter

for p in sorted(glob.glob('results/**/*.json', recursive=True)) + ['pred_nriqa_dataset_test.json']:
    try:
        d = json.load(open(p))
        if isinstance(d, list) and len(d) == 150 and 'label' in d[0]:
            n_cln = sum(1 for e in d if all(v == '无问题' for v in e['label'].values()))
            print(f"{p:65s}: All-Clean = {n_cln:3d}/150 ({n_cln/150:.1%})")
    except Exception:
        pass
