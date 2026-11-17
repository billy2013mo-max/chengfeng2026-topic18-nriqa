#!/usr/bin/env bash
# per-GPU worker: repeatedly claims a job file from queue/pending (atomic mv), runs it, moves to done/failed
GPU=$1; Q=/wanqing-develop/luowenjing/18/results/r55/sprint_os_og/queue
cd /wanqing-develop/luowenjing/18
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  [ -f $Q/STOP ] && exit 0
  J=$(ls $Q/pending 2>/dev/null | sort | head -1)
  if [ -z "$J" ]; then sleep 30; continue; fi
  mv $Q/pending/$J $Q/running/$J.gpu$GPU 2>/dev/null || continue
  CMD=$(cat $Q/running/$J.gpu$GPU)
  echo "[$(date +%H:%M:%S)] gpu$GPU start $J" >> $Q/worker.log
  if CUDA_VISIBLE_DEVICES=$GPU bash -c "$CMD"; then mv $Q/running/$J.gpu$GPU $Q/done/$J; st=ok; else mv $Q/running/$J.gpu$GPU $Q/failed/$J; st=FAIL; fi
  echo "[$(date +%H:%M:%S)] gpu$GPU $st $J" >> $Q/worker.log
done
