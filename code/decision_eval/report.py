import json, glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import metrics
from common import CLASSES

files = sorted(glob.glob('results/*.jsonl'))
files = [f for f in files if 'smoke' not in f]
rows = []
for f in files:
    try:
        if sum(1 for _ in open(f)) < 600: continue
        m = metrics(f)
        if 'error' in m: continue
        rows.append((os.path.basename(f)[:-6], m))
    except Exception as e:
        print('# skip', f, e)
hdr = ['model','valid','mAP14','mAP_P0'] + [c for c in CLASSES] + ['P0R@1%','S_pos@1%','S_pos@2%']
print('| ' + ' | '.join(hdr) + ' |')
print('|' + '|'.join(['---']*len(hdr)) + '|')
for name, m in sorted(rows, key=lambda x: -x[1]['mAP14']):
    per = [m['per_p0_ap'].get(f'{c}_P0') for c in CLASSES]
    line = [name, f"{m['valid']:.2f}", f"{m['mAP14']:.3f}", f"{m['mAP_P0']:.3f}"] + \
           [('-' if v is None else f'{v:.3f}') for v in per] + \
           [f"{m['FPR<=1%']['P0R']:.3f}", f"{m['FPR<=1%']['S_pos']:.3f}", f"{m['FPR<=2%']['S_pos']:.3f}"]
    print('| ' + ' | '.join(line) + ' |')
