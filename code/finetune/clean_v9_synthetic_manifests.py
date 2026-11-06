import csv, glob
from collections import Counter

for p in sorted(glob.glob('data/v9_ladders/manifest_*.csv')):
    rows = list(csv.DictReader(open(p)))
    syn_rows = [r for r in rows if r.get('split') == 'synth' or r['name'].startswith('v9_') or r['name'].startswith('v8_') or r['name'].startswith('syn_')]
    prefixes = Counter(r['name'].split('_')[0] + '_' + r['name'].split('_')[1] for r in syn_rows)
    print(p, "total:", len(rows), "synth:", len(syn_rows), "prefixes:", dict(prefixes))
    for pref in sorted(prefixes.keys()):
        sub = [r for r in syn_rows if r['name'].startswith(pref)]
        c_pos = {c: sum(int(r[c]) > 0 for r in sub) for c in ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']}
        c_grades = {c: dict(Counter(int(r[c]) for r in sub if int(r[c]) > 0)) for c in ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀'] if sum(int(r[c]) > 0 for r in sub) > 0}
        print(f"  {pref:12s} (n={len(sub):3d}): pos={c_pos} | grades={c_grades}")
