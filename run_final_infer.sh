#!/usr/bin/env bash
set -euo pipefail

# 一键评测集推理与格式校验脚本 (命题18)
# 用法:
#   bash run_final_infer.sh <模板JSON> <图片目录> <输出JSON> [GPU_ID] [SET_NAME] [DECODER]
# 示例:
#   bash run_final_infer.sh documents/eval_654_template.json data/eval_654 pred_eval_654.json 0 resident_v6_core routed_gate_eu

if [ "$#" -lt 3 ]; then
  echo "Usage: bash run_final_infer.sh <TEMPLATE_JSON> <IMG_DIR> <OUT_JSON> [GPU_ID=0] [SET_NAME=resident_v6_core] [DECODER=routed_gate_eu]"
  exit 1
fi

TEMPLATE_JSON="$1"
IMG_DIR="$2"
OUT_JSON="$3"
GPU_ID="${4:-0}"
SET_NAME="${5:-resident_v6_core}"
DECODER="${6:-routed_gate_eu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# expandable segments: closes most of the torch-peak vs nvidia-smi-peak gap (fragmentation), labels unchanged
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# optional Stage-4 sidecars (never change OUT_JSON), e.g.
#   R55_EXTRA_ARGS="--emit-attribution attr.json --tat-out tat.json --video-mode" bash run_final_infer.sh ...
R55_EXTRA_ARGS="${R55_EXTRA_ARGS:-}"

echo "=== [1/2] Running FinalEngine (set=${SET_NAME}, --half, decoder=${DECODER}, gpu=${GPU_ID}) ==="
python3 -u code/r55/final_engine.py \
  --set "${SET_NAME}" \
  --gpu "${GPU_ID}" \
  --half \
  --decoder "${DECODER}" \
  --template "${TEMPLATE_JSON}" \
  --img-dir "${IMG_DIR}" \
  --out "${OUT_JSON}" ${R55_EXTRA_ARGS}

echo "=== [2/2] Validating submission format & ID alignment ==="
python3 -u code/r55/validate_submission.py "${OUT_JSON}" --template "${TEMPLATE_JSON}"

echo "=== DONE: ${OUT_JSON} is ready for submission ==="
