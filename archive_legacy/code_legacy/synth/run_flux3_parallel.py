import csv
import glob
import os
import subprocess
import time

os.chdir('/wanqing-develop/luowenjing/18')
os.makedirs('/tmp/synth/flux3', exist_ok=True)
os.makedirs('/tmp/synth/shards', exist_ok=True)

man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
s16 = {r['name']: r['split'] for r in csv.DictReader(open('data/resplits/split_s16.csv'))}
s17 = {r['name']: r['split'] for r in csv.DictReader(open('data/resplits/split_s17.csv'))}
fin = {r['name']: r['split'] for r in csv.DictReader(open('data/split_final.csv'))}
CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']

# Candidates: strictly train in s16, s17, and split_final
common_clean = [
    n for n, r in man.items()
    if s16.get(n) == 'train' and s17.get(n) == 'train' and fin.get(n) == 'train'
    and all(int(r[c]) == 0 for c in CLASSES)
]

existing = set()
for p in glob.glob('/tmp/synth/flux3/strong_*'):
    bn = os.path.basename(p)
    if not bn.endswith('.txt'):
        existing.add(bn[len('strong_'):])

todo = [n for n in common_clean if n not in existing]
print(f"Total common_clean: {len(common_clean)}, already in flux3: {len(common_clean) - len(todo)}, todo: {len(todo)}")

# Take up to 80 (10 per GPU across 8 GPUs)
todo = todo[:80]
num_gpus = 8
shards = [todo[i::num_gpus] for i in range(num_gpus)]

procs = []
for g, sh in enumerate(shards):
    if not sh:
        continue
    shard_csv = f'/tmp/synth/shards/shard_{g}.csv'
    with open(shard_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name'])
        for n in sh:
            w.writerow([n])
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(g)
    log_f = open(f'/tmp/synth/shards/shard_{g}.log', 'w')
    cmd = [
        '/wanqing-develop/luowenjing/18/venv_flux/bin/python',
        'code/synth/flux3_gen.py',
        '--list', shard_csv,
        '--out', '/tmp/synth/flux3',
        '--steps', '12',
        '--levels', 'strong'
    ]
    procs.append((g, subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT), log_f))
    print(f"Launched GPU {g} with {len(sh)} images")

t0 = time.time()
for g, p, lf in procs:
    rc = p.wait()
    lf.close()
    print(f"GPU {g} finished rc={rc} in {time.time()-t0:.1f}s")
print("Total strong_* in /tmp/synth/flux3:", len([p for p in glob.glob('/tmp/synth/flux3/strong_*') if not p.endswith('.txt')]))
