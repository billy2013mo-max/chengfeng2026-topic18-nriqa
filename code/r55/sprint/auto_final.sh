#!/usr/bin/env bash
# re-run final_eval whenever a new sprint tower reaches 8 folds; stop when queue drained
cd /wanqing-develop/luowenjing/18
Q=results/r55/sprint_os_og/queue; seen=""
while true; do
  extras=""; tagset=""
  for d in results/next55/towers/r55_sp_*/*/; do
    n=$(ls $d/fold_*_probs.npz 2>/dev/null | wc -l)
    if [ "$n" -ge 8 ]; then nm=$(basename $(dirname $d)); extras="$extras ${nm}=${d%/}"; tagset="$tagset $nm"; fi
  done
  if [ "$tagset" != "$seen" ] && [ -n "$tagset" ]; then
    seen="$tagset"; k=$(echo $tagset | wc -w)
    (cd code/r55/sprint && python3 final_eval.py "v7+${k}sprint" $extras >> ../../../results/r55/sprint_os_og/logs/auto_final.log 2>&1)
  fi
  if [ -z "$(ls $Q/pending)" ] && [ -z "$(ls $Q/running)" ]; then break; fi
  sleep 120
done
echo DONE >> results/r55/sprint_os_og/logs/auto_final.log
