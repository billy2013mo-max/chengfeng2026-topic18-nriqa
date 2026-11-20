#!/usr/bin/env bash
set -euo pipefail

# 一键评测集推理与格式校验脚本 (命题18)
# 默认配置: resident_v8_boundary + deconf_v6_boundary + --half --share-weights --embed-on-host
# (总激活参数 13.26B <= 30B，单卡 RTX 4090 常驻显存 15.78 GB，nvidia-smi 推理峰值 18.49 GiB / 加载峰值 19.98 GiB < 24 GiB，全量 test_150 实测均值 3.03s / 最大 6.93s <= 15s)
#
# 用法:
#   bash run_final_infer.sh <模板JSON> <图片目录> <输出JSON> [GPU_ID=0] [SET_NAME=resident_v8_boundary] [DECODER=deconf_v6_boundary]
# 示例:
#   bash run_final_infer.sh results/final/template_final_644.json /opt/final_eval_644 pred_eval_644.json 0 resident_v8_boundary deconf_v6_boundary

if [ "$#" -lt 3 ]; then
  echo "Usage: bash run_final_infer.sh <TEMPLATE_JSON> <IMG_DIR> <OUT_JSON> [GPU_ID=0] [SET_NAME=resident_v8_boundary] [DECODER=deconf_v6_boundary]"
  exit 1
fi

TEMPLATE_JSON="$1"
IMG_DIR="$2"
OUT_JSON="$3"
GPU_ID="${4:-0}"
SET_NAME="${5:-resident_v8_boundary}"
DECODER="${6:-deconf_v6_boundary}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# expandable segments: closes torch-peak vs nvidia-smi-peak fragmentation gap, labels 100% unchanged
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# optional Stage-4 sidecars (never change OUT_JSON), e.g.
#   R55_EXTRA_ARGS="--emit-attribution attr.json --tat-out tat.json --video-mode" bash run_final_infer.sh ...
R55_EXTRA_ARGS="${R55_EXTRA_ARGS:-}"

echo "=== [1/2] Running FinalEngine (set=${SET_NAME}, --half --share-weights --embed-on-host, decoder=${DECODER}, gpu=${GPU_ID}) ==="
python3 -u code/r55/final_engine.py \
  --set "${SET_NAME}" \
  --gpu "${GPU_ID}" \
  --half \
  --share-weights \
  --embed-on-host \
  --decoder "${DECODER}" \
  --template "${TEMPLATE_JSON}" \
  --img-dir "${IMG_DIR}" \
  --out "${OUT_JSON}" ${R55_EXTRA_ARGS}

echo "=== [2/2] Validating submission format & ID alignment ==="
python3 -u code/r55/validate_submission.py "${OUT_JSON}" --template "${TEMPLATE_JSON}"

echo "=== DONE: ${OUT_JSON} is ready for submission ==="
