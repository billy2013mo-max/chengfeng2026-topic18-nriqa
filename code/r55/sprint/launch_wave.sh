#!/usr/bin/env bash
# usage: launch_wave.sh NAME BACKBONE "EXTRA ARGS"  -> 8 folds on GPUs 0-7 (one per GPU)
set -u
cd /wanqing-develop/luowenjing/18
NAME=$1; BB=$2; EXTRA=$3
LOG=results/r55/sprint_os_og/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$f nohup python3 -u code/r55/train_testbed.py --backbone $BB --fold $f --gpu 0 --seed 2026 \
    --out-dir results/next55/towers/$NAME --select fixed --fast-fixed --skip-t150 --no-save-ckpt --workers 4 $EXTRA \
    > $LOG/${NAME}_f$f.log 2>&1 &
done
echo launched $NAME
