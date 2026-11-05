import csv
import glob
import os
from collections import Counter

man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']

print("Total manifest_combined:", len(man))
for c in CLASSES:
    for g in [1, 2, 3]:
        sub = [r for r in man.values() if int(r[c]) == g]
        if not sub:
            continue
        co = {oc: Counter(int(r[oc]) for r in sub) for oc in CLASSES if oc != c}
        # Most common full 6-tuple pattern for this (c, g)
        tuples = Counter(tuple(int(r[x]) for x in CLASSES) for r in sub).most_common(5)
        print(f"\n=== {c} == P{3-g} (grade {g}, n={len(sub)}) ===")
        print("  Top 5 full label tuples (模糊,纹理,过锐,花屏,过度生成,虚化):", tuples)
        for oc in CLASSES:
            if oc != c:
                pos_rate = sum(1 for r in sub if int(r[oc]) > 0) / len(sub)
                if pos_rate > 0.05:
                    print(f"  Co-occurs with {oc}: {pos_rate*100:.1f}% -> {dict(co[oc])}")

for sp_name, sp_path in [('s16', 'data/splits/s16.csv'), ('s17', 'data/splits/s17.csv'), ('final', 'data/split_final.csv')]:
    sp = {r['name']: r['split'] for r in csv.DictReader(open(sp_path))}
    flx_files = [os.path.basename(p) for p in glob.glob('/tmp/synth/flux3/*_*.png')]
    flx_src = [f.split('_', 1)[1] for f in flx_files]
    in_splits = Counter(sp.get(n, 'missing') for n in flx_src)
    print(f"\nSplit {sp_name}: flux3 src distribution = {dict(in_splits)}")
    sr_files = [os.path.basename(p) for p in glob.glob('/tmp/synth/out/p1_*')]
    sr_src = [f[3:] for f in sr_files]
    print(f"Split {sp_name}: swin2sr src distribution = {dict(Counter(sp.get(n, 'missing') for n in sr_src))}")
