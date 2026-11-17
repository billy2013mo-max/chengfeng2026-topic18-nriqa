"""Retrain the shipped V16 recipe on the video-grouped qingyu folds (audit 2026-11-14, A1).

Same hyper-parameters as code/finetune/launch_v16_fulldata_8gpus.py, but every fold uses the backbone set of the shipped
full-fit model (F0 set) so the OOF matches the deployed member, and splits come from data/v16_qy (build_v16_qy_splits.py).
Output: results/r55/v16_qy/V16_Full_F{k}_AllGrade_Microscope_{probs.npz,json,log} — the v16 OOF format of common55.

  python3 -I code/r55/launch_v16_qy.py [--folds 0-7] [--gpus 0,1,...]
"""
import argparse
import os
import subprocess
import time

REPO = '/wanqing-develop/luowenjing/18'
OUT = os.path.join(REPO, 'results/r55/v16_qy')
BB = 'pecoreg:lastk4,dinov3l:lastk6,cradio:lastk4,eva02:lastk6,convnext:full'
ALL6 = '模糊,纹理损失,过锐,花屏,过度生成,虚化突兀'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', default='0-7')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a = ap.parse_args()
    lo, hi = map(int, a.folds.split('-'))
    gpus = a.gpus.split(',')
    os.makedirs(OUT, exist_ok=True)
    procs = []
    for i, k in enumerate(range(lo, hi + 1)):
        name = f'V16_Full_F{k}_AllGrade_Microscope'
        if os.path.exists(f'{OUT}/{name}_probs.npz'):
            print('skip', name, flush=True)
            continue
        cmd = ['python3', '-u', '-I', 'code/finetune/train_v16_microscope_specialist.py', '--backbones', BB,
               '--focus-classes', ALL6, '--seed', str(600 + k), '--epochs', '3', '--batch', '8', '--grad-accum', '1',
               '--workers', '8', '--size', '448', '--tiles', '2', '--clean-repeat', '2', '--p0-clean-weight', '4.2',
               '--channel-neg-weight', '1.05', '--lr-full', '7.5e-05', '--lr-last', '2.6e-05', '--lr-head', '1.35e-3',
               '--wd', '0.012', '--dropout', '0.12', '--alpha-ex', '0.58',
               '--split', f'data/v16_qy/split_f{k}.csv', '--manifest', f'data/v16_qy/manifest_f{k}.csv',
               '--out', f'{OUT}/{name}']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)], PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        log = open(f'{OUT}/{name}.log', 'w')
        procs.append((name, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(time.strftime('%H:%M:%S'), 'launched', name, 'gpu', env['CUDA_VISIBLE_DEVICES'], flush=True)
    for name, p, log in procs:
        rc = p.wait()
        log.close()
        print(time.strftime('%H:%M:%S'), 'done', name, 'rc', rc, 'npz', os.path.exists(f'{OUT}/{name}_probs.npz'), flush=True)


if __name__ == '__main__':
    main()
