"""Overnight experiment queue (see results/plan_overnight.md).

Phases
  2a  class-specific clean penalty on the 虚化突兀 P0 head   lambda x 5 splits x 2 seeds
  2b  hard-clean mining: stage1 trains + dumps train-clean scores, stage2 retrains
      with the top-scoring clean images repeated
  2c  bokeh/beautify augmentation on clean training images   2 strengths
  3a  epochs 6  3b two backbones  3c input size 448
  4   final recipe on 5 never-used splits (s16-s20)

Every job runs the standard recipe unless overridden:
  eupe:full,siglip2:lastk4,cradio:lastk4, 4 ep, batch 6, lr_full 8.875e-5,
  lr_last 3e-5, lr_head 2.2754e-3, wd 0.01, dropout 0, clean_repeat 7, size 384
"""
import csv, json, os, subprocess, time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
ROOT = 'results/night'
GPUS = list(range(8))
TRIPLE = 'eupe:full,siglip2:lastk4,cradio:lastk4'
MAX_HOURS = 9.0

BASE = dict(backbones=TRIPLE, epochs='4', batch='6', workers='8', clean_penalty='0',
            clean_repeat='7', lr_full='8.875e-05', lr_last='3e-05', lr_head='2.2754e-03',
            wd='0.01', dropout='0', size='384')

SPLITS = [11, 12, 13, 14, 15]
SEEDS = [1, 2]

JOBS = []


def add(phase, group, sp, sd, over=None, extra=None, need=None):
    JOBS.append(dict(phase=phase, group=group, sp=sp, sd=sd,
                     over=over or {}, extra=extra or [], need=need))


# ---- Phase 2a: class-5 clean penalty ----
for lam in ['0.3', '0.6', '1.0']:
    for sp in SPLITS:
        for sd in SEEDS:
            add('2a', f'cls5_lam{lam}', sp, sd,
                extra=['--clean-penalty', lam, '--clean-penalty-class', '5'])

# ---- Phase 2b: hard-clean mining (two stage) ----
for sp in SPLITS:
    for sd in SEEDS:
        add('2b', 'stage2_hard', sp, sd,
            extra=['--hard-clean-file', f'{ROOT}/2b/hard_s{sp}_{sd}.txt', '--hard-repeat', '4'],
            need=f'{ROOT}/2b/hard_s{sp}_{sd}.txt')
        add('2b', 'stage1_dump', sp, sd,
            extra=['--dump-train-clean', f'{ROOT}/2b/dump_s{sp}_{sd}.csv'])

# ---- Phase 2c: bokeh augmentation ----
for tag, rad in [('bokeh_r6', '6'), ('bokeh_r12', '12')]:
    for sp in SPLITS:
        for sd in SEEDS:
            add('2c', tag, sp, sd, extra=['--clean-bokeh', '0.5', '--bokeh-radius', rad])

# ---- Phase 3a / 3b / 3c ----
for sp in SPLITS:
    for sd in SEEDS:
        add('3a', 'epochs6', sp, sd, over={'epochs': '6'})
        add('3b', 'two_backbone', sp, sd, over={'backbones': 'siglip2:lastk4,cradio:lastk4'})
        add('3c', 'size448', sp, sd, over={'size': '448'})

# ---- Phase 4: final recipe on never-used splits ----
for sp in [16, 17, 18, 19, 20]:
    for sd in SEEDS:
        add('4', 'final', sp, sd)


def gpu_free():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                          '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    free = []
    for line in out.strip().splitlines():
        i, m = line.split(',')
        if int(m) < 2000:
            free.append(int(i))
    return [g for g in GPUS if g in free]


def cmd(j, out):
    c = dict(BASE); c.update(j['over'])
    sp = j['sp']
    split = 'data/split_trainval.csv' if sp == 0 else f'data/resplits/split_s{sp}.csv'
    args = ['python3', '-u', '-I', 'code/finetune/train_multi.py',
            '--backbones', c['backbones'], '--seed', str(j['sd']), '--epochs', c['epochs'],
            '--batch', c['batch'], '--workers', c['workers'], '--size', c['size'],
            '--clean-penalty', c['clean_penalty'], '--clean-repeat', c['clean_repeat'],
            '--lr-full', c['lr_full'], '--lr-last', c['lr_last'], '--lr-head', c['lr_head'],
            '--wd', c['wd'], '--dropout', c['dropout'],
            '--split', split, '--out', out]
    return args + j['extra']


def make_hard_file(dump, hard, top_frac=0.05, min_k=10):
    rows = list(csv.DictReader(open(dump)))
    scores = sorted((float(r['maxp0']) for r in rows), reverse=True)
    if not scores:
        return
    k = max(min_k, int(len(scores) * top_frac))
    thr = scores[min(k, len(scores)) - 1]
    keep = [r['name'] for r in rows if float(r['maxp0']) >= thr]
    with open(hard, 'w') as f:
        f.write('\n'.join(keep) + '\n')
    print(f'  hard file {hard}: {len(keep)} images (thr={thr:.3f}, clean train n={len(rows)})', flush=True)


def main():
    os.makedirs(ROOT, exist_ok=True)
    pending = [j for j in JOBS]
    running = {}
    t0 = time.time()
    done = 0
    while pending or running:
        if time.time() - t0 > MAX_HOURS * 3600:
            print('MAX_HOURS reached, stopping', flush=True); break
        for g in [x for x in gpu_free() if x not in running]:
            nxt = None
            for j in pending:
                if j['need'] and not os.path.exists(j['need']):
                    continue
                nxt = j; break
            if nxt is None:
                break
            pending.remove(nxt)
            pdir = f'{ROOT}/{nxt["phase"]}'
            os.makedirs(pdir, exist_ok=True)
            out = f'{pdir}/{nxt["group"]}_s{nxt["sp"]}_seed{nxt["sd"]}.json'
            if os.path.exists(out):
                print(f'skip existing {out}', flush=True); done += 1; continue
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
            f = open(out.replace('.json', '.log'), 'w')
            p = subprocess.Popen(cmd(nxt, out), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, nxt, out, f)
            print(f'[{time.time()-t0:6.0f}s] gpu{g} start {nxt["phase"]}/{nxt["group"]} '
                  f's{nxt["sp"]} seed{nxt["sd"]}', flush=True)
        time.sleep(15)
        for g in list(running):
            p, j, out, f = running[g]
            if p.poll() is not None:
                f.close(); del running[g]; done += 1
                ok = os.path.exists(out)
                msg = ''
                if ok:
                    d = json.load(open(out)); t = d['test']
                    msg = f'test={t["total"]:.2f} S_pos={t["S_pos"]:.3f} S_clean={t["S_clean"]:.3f}'
                print(f'[{time.time()-t0:6.0f}s] gpu{g} done {j["group"]} s{j["sp"]} seed{j["sd"]} {msg}'
                      f'{" FAILED" if not ok else ""}', flush=True)
                # stage1 -> build hard file for the matching stage2
                if j['group'] == 'stage1_dump' and ok:
                    dump = f'{ROOT}/2b/dump_s{j["sp"]}_{j["sd"]}.csv'
                    hard = f'{ROOT}/2b/hard_s{j["sp"]}_{j["sd"]}.txt'
                    if os.path.exists(dump):
                        make_hard_file(dump, hard)
    summarize()


def summarize():
    print('\n==== night summary ====', flush=True)
    for phase in ['2a', '2b', '2c', '3a', '3b', '3c', '4']:
        pdir = f'{ROOT}/{phase}'
        if not os.path.isdir(pdir):
            continue
        groups = {}
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith('.json'):
                continue
            d = json.load(open(f'{pdir}/{fn}'))
            key = fn.rsplit('_s', 1)[0]
            groups.setdefault(key, []).append(d)
        print(f'\n-- phase {phase} --')
        for key, ds in groups.items():
            v = np.array([d['test']['total'] for d in ds])
            sp = np.array([d['test']['S_pos'] for d in ds])
            cl = np.array([d['test']['S_clean'] for d in ds])
            print(f'  {key:24s} n={len(v):2d} mean={v.mean():6.2f} median={np.median(v):6.2f} '
                  f'min={v.min():6.2f} fails<45={int((v<45).sum())} '
                  f'S_pos={sp.mean():.3f} S_clean={cl.mean():.3f} (0-clean={int((cl==0).sum())})')


if __name__ == '__main__':
    main()
