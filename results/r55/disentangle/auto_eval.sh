#!/usr/bin/env bash
# waits for each disentangle tower (8 folds) and evaluates it against the same-seed control tower
cd /wanqing-develop/luowenjing/18
T=results/next55/towers; L=results/r55/disentangle
for spec in og_l5_s2026:r55_sp_ntgpe_focus_s2026 osog_l5_s2026:r55_sp_ntgpe_focus_s2026 og_l20_s2026:r55_sp_ntgpe_focus_s2026 og_l5_s2027:r55_sp_ntgpe_focus_s2027 osog_l5_s2027:r55_sp_ntgpe_focus_s2027; do
  tag=${spec%%:*}; ctl=${spec#*:}
  until [ "$(ls $T/r55_dis_$tag/pe_core_large/*probs.npz 2>/dev/null | wc -l)" -ge 8 ]; do sleep 60; done
  echo "TOWER DONE $tag $(date +%H:%M)"
  python -u code/r55/eval_disentangle.py $tag=r55_dis_$tag:$ctl > $L/eval_$tag.log 2>&1 && echo "EVAL DONE $tag -> $L/eval_$tag.log" || echo "EVAL FAILED $tag (see $L/eval_$tag.log)"
done
echo "ALL DONE"
