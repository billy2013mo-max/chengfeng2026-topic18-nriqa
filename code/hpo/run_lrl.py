"""Small sweep over --lr-last for the fixed best multi-backbone config.

The two lastk4 backbones (siglip2, cradio) share --lr-last, default 1e-5, which
was never touched by the BoTorch HPO (it only searched --lr-full/--lr-head).
Here we scan {3e-6, 1e-5, 3e-5, 1e-4, 3e-4} x seeds {1,2} on the original
split, 5 GPUs (2,3,4,6,7) in parallel, then summarise.

Optionally the winning value can be re-verified on split_s11/split_s13 with
LRl_CONFIRM=<value> python3 -I code/hpo/run_lrl.py  (writes results/lrl_confirm/).
"""
import json, os, subprocess, sys, time, csv
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
GPUS = [2, 3, 4, 6, 7]
VALUES = ['3e-06', '1e-05', '3e-05', '1e-04', '3e-04']
SEEDS = [1, 2]
EPOCHS = 4
BASE = dict(backbones='eupe:full,siglip2:lastk4,cradio:lastk4', clean_penalty=0.0,
            clean_repeat=7, lr_full=8.875358285474408e-05, lr_head=0.0022753949862994462,
            wd=0.01, dropout=0.0, batch=6)

CONFIRM = os.environ.get('LRL_CONFIRM')  # e.g. '3e-05'
if CONFIRM:
    OUT = 'results/lrl_confirm'
    # (run name, lr_last, split path, split label, seed)
    JOBS = [('confirm_%s_s%d_s%d' % (CONFIRM, sp, sd), CONFIRM,
             'data/resplits/split_s%d.csv' % sp, 's%d' % sp, sd)
            for sp in (11, 13) for sd in SEEDS]
else:
    OUT = 'results/lrl'
    JOBS = [('lrl_%s_s%d' % (v, sd), v, 'data/split_trainval.csv', 'orig', sd)
            for v in VALUES for sd in SEEDS]
os.makedirs(OUT, exist_ok=True)


def cmd(lr_last, split, seed, out):
    return ['python3', '-u', '-I', 'code/finetune/train_multi.py',
            '--backbones', BASE['backbones'], '--seed', str(seed), '--epochs', str(EPOCHS),
            '--batch', str(BASE['batch']), '--workers', '8',
            '--clean-penalty', '%.4f' % BASE['clean_penalty'], '--clean-repeat', str(BASE['clean_repeat']),
            '--lr-full', '%.3e' % BASE['lr_full'], '--lr-last', lr_last,
            '--lr-head', '%.3e' % BASE['lr_head'],
            '--wd', '%.4f' % BASE['wd'], '--dropout', '%.3f' % BASE['dropout'],
            '--split', split, '--out', out]


def gpu_is_free(g):
    """Check a GPU has no processes (be a good neighbour, don't disturb others)."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
            text=True, stderr=subprocess.DEVNULL)
        idx = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
            text=True, stderr=subprocess.DEVNULL)
        uuid = {int(l.split(',')[0]): l.split(',')[1].strip() for l in idx.strip().splitlines()}
        busy = set()
        for l in out.strip().splitlines():
            if l.strip():
                busy.add(l.split(',')[0].strip())
        return uuid.get(g) not in busy
    except Exception:
        return True


def main():
    pending = list(JOBS)
    running = {}  # gpu -> (proc, name, out, logf)
    t0 = time.time()
    while pending or running:
        for g in GPUS:
            if not pending:
                break
            if g in running or (running and not gpu_is_free(g)):
                continue
            name, lr_last, split, _split_label, sd = pending.pop(0)
            out = '%s/%s.json' % (OUT, name)
            log = '%s/%s.log' % (OUT, name)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(g)
            f = open(log, 'w')
            p = subprocess.Popen(cmd(lr_last, split, sd, out),
                                 env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, name, out, f)
            print('[%6.0fs] gpu%d start %s' % (time.time() - t0, g, name), flush=True)
        time.sleep(10)
        for g in list(running):
            p, name, out, f = running[g]
            if p.poll() is not None:
                f.close()
                del running[g]
                try:
                    d = json.load(open(out))
                    print('[%6.0fs] gpu%d done  %s test=%.2f val=%.2f'
                          % (time.time() - t0, g, name, d['test']['total'], d['val_total']), flush=True)
                except Exception as e:
                    print('[%6.0fs] gpu%d FAIL  %s (%s)'
                          % (time.time() - t0, g, name, e), flush=True)
    summarize()


def _val_p99(name):
    try:
        pr = np.load('%s/%s_probs.npz' % (OUT, name), allow_pickle=True)
        clean = (pr['gva'] == 0).all(1)
        if clean.any():
            return float(np.percentile(pr['va_p0'][clean].max(1), 99))
    except Exception:
        pass
    return None


def summarize():
    rows = []
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith('.json'):
            continue
        name = fn[:-5]
        d = json.load(open('%s/%s' % (OUT, fn)))
        s = d['test']
        if CONFIRM:
            parts = name.split('_')
            lr_label, split = parts[1], parts[2]
        else:
            parts = name.split('_')
            lr_label, split = parts[1], 'orig'
        seed = int(name.rsplit('_s', 1)[1])
        rows.append(dict(lr_last=lr_label, split=split, seed=seed, val=d['val_total'],
                         total=s['total'], S_pos=s['S_pos'], p0R=s['p0R'], p0P=s['p0P'],
                         p0F=s['p0F'], S_clean=s['S_clean'], fpr=s['fpr'],
                         val_p99=_val_p99(name)))
    if not rows:
        print('no results')
        return
    cols = ['lr_last', 'split', 'seed', 'val', 'total', 'S_pos', 'p0R', 'p0P', 'p0F',
            'S_clean', 'fpr', 'val_p99']
    with open('%s/summary.csv' % OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r['lr_last'], r['split'], r['seed'])):
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in cols})

    # aggregate per lr_last (and per split when confirming)
    groups = {}
    for r in rows:
        groups.setdefault((r['lr_last'], r['split']), []).append(r)

    order = ['3e-06', '1e-05', '3e-05', '1e-04', '3e-04']
    lines = []
    lines.append('| lr_last | split | n | test total mean±std | S_pos | p0R | p0P | S_clean | fpr | val | val p99 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for key in sorted(groups, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1])):
        ss = groups[key]
        t = np.array([r['total'] for r in ss])
        std = t.std(ddof=1) if len(t) > 1 else 0.0
        lines.append('| %s | %s | %d | %.2f ± %.2f | %.3f | %.3f | %.3f | %.3f | %.4f | %.2f | %.4f |' % (
            key[0], key[1], len(ss), t.mean(), std,
            np.mean([r['S_pos'] for r in ss]), np.mean([r['p0R'] for r in ss]),
            np.mean([r['p0P'] for r in ss]), np.mean([r['S_clean'] for r in ss]),
            np.mean([r['fpr'] for r in ss]), np.mean([r['val'] for r in ss]),
            np.mean([r['val_p99'] for r in ss if r['val_p99'] is not None]) if any(
                r['val_p99'] is not None for r in ss) else float('nan')))
    lines.append('')
    lines.append('Per-seed test totals:')
    for key in sorted(groups, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1])):
        ss = sorted(groups[key], key=lambda r: r['seed'])
        lines.append('- %s/%s: %s' % (key[0], key[1],
                     ', '.join('s%d=%.2f' % (r['seed'], r['total']) for r in ss)))
    txt = '\n'.join(lines) + '\n'
    with open('%s/summary.md' % OUT, 'w') as f:
        f.write(txt)
    print('\n==== summary ====\n' + txt, flush=True)


if __name__ == '__main__':
    main()
