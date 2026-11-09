import csv
import glob
import os
import subprocess
import time

os.chdir('/wanqing-develop/luowenjing/18')
os.makedirs('/tmp/synth/flux3', exist_ok=True)
os.makedirs('/tmp/synth/shards_513', exist_ok=True)

man = {r['name']: r for r in csv.DictReader(open('data/manifest_combined.csv'))}
CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']

# All 513 globally clean images
all_clean = [
    n for n, r in man.items()
    if all(int(r[c]) == 0 for c in CLASSES)
]

existing_strong = set()
for p in glob.glob('/tmp/synth/flux3/strong_*'):
    bn = os.path.basename(p)
    if not bn.endswith('.txt'):
        existing_strong.add(bn[len('strong_'):])

existing_mild = set()
for p in glob.glob('/tmp/synth/flux3/mild_*'):
    bn = os.path.basename(p)
    if not bn.endswith('.txt'):
        existing_mild.add(bn[len('mild_'):])

todo = [n for n in all_clean if (n not in existing_strong or n not in existing_mild)]
print(f"Total all_clean seeds: {len(all_clean)}")
print(f"Already have strong: {len(existing_strong)}, mild: {len(existing_mild)}")
print(f"To generate: {len(todo)} images across both mild & strong")

num_gpus = 8
shards = [todo[i::num_gpus] for i in range(num_gpus)]

procs = []
for g, sh in enumerate(shards):
    if not sh:
        continue
    shard_csv = f'/tmp/synth/shards_513/shard_{g}.csv'
    with open(shard_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name'])
        for n in sh:
            w.writerow([n])
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(g)
    log_f = open(f'/tmp/synth/shards_513/shard_{g}.log', 'w')
    cmd = [
        '/wanqing-develop/luowenjing/18/venv_flux/bin/python',
        'code/synth/flux3_gen.py',
        '--list', shard_csv,
        '--out', '/tmp/synth/flux3',
        '--steps', '12',
        '--levels', 'mild,strong'
    ]
    procs.append((g, subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT), log_f))
    print(f"Launched GPU {g} with {len(sh)} images")

t0 = time.time()
for g, p, lf in procs:
    rc = p.wait()
    lf.close()
    print(f"GPU {g} finished rc={rc} in {time.time()-t0:.1f}s")

total_s = len([p for p in glob.glob('/tmp/synth/flux3/strong_*') if not p.endswith('.txt')])
total_m = len([p for p in glob.glob('/tmp/synth/flux3/mild_*') if not p.endswith('.txt')])
print(f"Done! /tmp/synth/flux3 now contains: strong={total_s}, mild={total_m}")
