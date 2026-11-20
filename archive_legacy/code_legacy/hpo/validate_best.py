"""Validate the best HPO config on 5 splits x 2 seeds, plus the incumbent."""
import argparse, json, os, subprocess, time
import numpy as np

os.chdir('/wanqing-develop/luowenjing/18')
OUT = 'results/hpo2_validate'
os.makedirs(OUT, exist_ok=True)
SPLITS = [11, 12, 13, 14, 15]
SEEDS = [1, 2]
INCUMBENT = dict(clean_repeat=7, lr_full=8.875e-5, lr_last=3e-5, lr_head=2.2754e-3,
                 dropout=0.0, wd=0.01, batch=6)


def cmd(cfg, sp, sd, out):
    return ['python3', '-u', '-I', 'code/finetune/train_multi.py',
            '--backbones', 'siglip2:lastk4,cradio:lastk4', '--seed', str(sd), '--epochs', '4',
            '--batch', str(cfg['batch']), '--workers', '8', '--clean-penalty', '0',
            '--clean-repeat', str(cfg['clean_repeat']),
            '--lr-full', f"{cfg['lr_full']:.3e}", '--lr-last', f"{cfg['lr_last']:.3e}",
            '--lr-head', f"{cfg['lr_head']:.3e}", '--wd', f"{cfg['wd']:.4f}",
            '--dropout', f"{cfg['dropout']:.3f}",
            '--split', f'data/resplits/split_s{sp}.csv', '--out', out]


def gpu_free():
    o = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True).stdout
    fr = []
    for line in o.strip().splitlines():
        i, m = line.split(',')
        if int(m) < 2000:
            fr.append(int(i))
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--best', default='results/hpo2/best.json')
    ap.add_argument('--n', type=int, default=1)
    a = ap.parse_args()
    cands = json.load(open(a.best))[:a.n]
    groups = {f'hpo{i}': c['cfg'] for i, c in enumerate(cands)}
    groups['incumbent'] = INCUMBENT
    jobs = [(g, sp, sd) for g in groups for sp in SPLITS for sd in SEEDS]
    pending = list(jobs); running = {}; t0 = time.time()
    while pending or running:
        for g in [x for x in gpu_free() if x not in running]:
            if not pending:
                break
            grp, sp, sd = pending.pop(0)
            out = f'{OUT}/{grp}_s{sp}_seed{sd}.json'
            if os.path.exists(out):
                continue
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(g)
            f = open(out.replace('.json', '.log'), 'w')
            p = subprocess.Popen(cmd(groups[grp], sp, sd, out), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[g] = (p, grp, sp, sd, out, f)
            print(f'[{time.time()-t0:6.0f}s] gpu{g} start {grp} s{sp} seed{sd}', flush=True)
        time.sleep(15)
        for g in list(running):
            p, grp, sp, sd, out, f = running[g]
            if p.poll() is not None:
                f.close(); del running[g]
                try:
                    d = json.load(open(out))
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} done {grp} s{sp} seed{sd} test={d["test"]["total"]:.2f}', flush=True)
                except Exception as e:
                    print(f'[{time.time()-t0:6.0f}s] gpu{g} FAIL {grp} s{sp} seed{sd} {e}', flush=True)
    print('\n==== validation ====', flush=True)
    for grp in groups:
        v = []
        for sp in SPLITS:
            for sd in SEEDS:
                f = f'{OUT}/{grp}_s{sp}_seed{sd}.json'
                if os.path.exists(f):
                    d = json.load(open(f)); v.append((d['test']['total'], d['test']['S_pos'], d['test']['S_clean']))
        if v:
            a_ = np.array(v)
            print(f'{grp:10s} {json.dumps(groups[grp])}')
            print(f'   n={len(v)} total={a_[:,0].mean():.2f} min={a_[:,0].min():.2f} '
                  f'S_pos={a_[:,1].mean():.3f} S_clean={a_[:,2].mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
