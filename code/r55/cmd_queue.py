"""Minimal GPU queue for arbitrary commands (audit 2026-11-14): each line of --jobs-file is
    <need_gb> <log_path> <shell command ...>
and is started with CUDA_VISIBLE_DEVICES set to a GPU that has >= need_gb free and fewer than --per-gpu jobs from this
queue.  A job whose log ends with a line "CMDQ_DONE rc=0" from an earlier run is skipped (resumable).

  python3 code/r55/cmd_queue.py --gpus 0 1 2 3 4 5 6 7 --jobs-file logs/r55/audit14/p3_jobs.txt
"""
import argparse
import os
import subprocess
import time

REPO = '/wanqing-develop/luowenjing/18'


def free_gb():
    o = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits']).decode()
    return {int(i): (float(t) - float(u)) / 1024 for i, t, u in (l.split(',') for l in o.strip().splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', type=int, nargs='+', required=True)
    ap.add_argument('--per-gpu', type=int, default=2)
    ap.add_argument('--jobs-file', required=True)
    ap.add_argument('--settle', type=int, default=180, help='seconds between two launches on one GPU')
    a = ap.parse_args()
    todo = []
    for l in open(a.jobs_file):
        l = l.strip()
        if not l or l.startswith('#'):
            continue
        need, log, cmd = l.split(None, 2)
        if os.path.exists(log) and open(log, errors='ignore').read().rstrip().endswith('CMDQ_DONE rc=0'):
            print('skip', log, flush=True)
            continue
        todo.append((float(need), log, cmd))
    running, last = {}, {}
    while todo or running:
        for p in list(running):
            if p.poll() is not None:
                g, log = running.pop(p)
                print(time.strftime('%H:%M:%S'), 'done rc', p.returncode, 'gpu', g, log, flush=True)
        fg = free_gb()
        for g in a.gpus:
            if not todo:
                break
            n = sum(1 for gg, _ in running.values() if gg == g)
            if n >= a.per_gpu or time.time() - last.get(g, 0) < a.settle or fg.get(g, 0) < todo[0][0]:
                continue
            need, log, cmd = todo.pop(0)
            os.makedirs(os.path.dirname(log), exist_ok=True)
            full = f'({cmd}) > {log} 2>&1; echo "CMDQ_DONE rc=$?" >> {log}'
            p = subprocess.Popen(['bash', '-c', full], cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)))
            running[p] = (g, log)
            last[g] = time.time()
            print(time.strftime('%H:%M:%S'), 'start gpu', g, log, flush=True)
        time.sleep(20)


if __name__ == '__main__':
    main()
