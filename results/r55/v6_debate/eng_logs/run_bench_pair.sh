#!/usr/bin/env bash
# chunk=2 then chunk=1 bench on GPU 0, each with an external nvidia-smi poller (100 ms)
cd /wanqing-develop/luowenjing/18
L=results/r55/v6_debate/eng_logs
for CH in 2 1; do
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; sleep 0.1; done ) > $L/nvsmi_chunk${CH}.txt &
  POLL=$!
  python3 -u code/r55/final_engine.py --set resident_v6_core --half --gpu 0 --bench 20 --v16-chunk $CH --bench-tag _chunk${CH} > $L/bench_chunk${CH}.log 2>&1
  kill $POLL
  echo "chunk $CH done: nvidia-smi max $(sort -n $L/nvsmi_chunk${CH}.txt | tail -1) MiB"
done
