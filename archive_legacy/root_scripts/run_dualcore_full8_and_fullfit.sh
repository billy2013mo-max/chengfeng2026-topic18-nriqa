#!/usr/bin/env bash
# ==============================================================================
# run_dualcore_full8_and_fullfit.sh
# 8xH200 Balanced 4-Lane Runner:
#   - Completes remaining Folds 2..7 + 100% Full-Fit (`--fold -1`) for BOTH:
#       1) `InternViT-6B-NativeTiles` (`code/r55/train_iv6b_native_tiles.py`)
#       2) `Qwen3-VL-8B-HiRes-ROI` (`code/r55/train_qwen3vl_hires_roi.py`)
#   - Fits 100% Package-Free `NaturalPriorRoutedDecoder` across all 3,573 OOF rows
#   - Runs fresh 8-GPU Full-Fit inference on `test_150` and `final_644` + N=1 & schema checks.
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p results/r55/iv6b_native_pilot results/r55/qwen3vl_hires_pilot results/r55/dualcore_fullfit_final/logs

run_iv6b() {
  local f="$1"
  local gpus="$2"
  local port="$3"
  local tag="fold${f}"
  if [[ "${f}" == "-1" ]]; then
    tag="fullfit"
    if [[ -f "results/r55/iv6b_native_pilot/fullfit_best.pt" ]]; then
      echo "[SKIP] iv6b ${tag} already exists."
      return 0
    fi
  else
    if [[ -f "results/r55/iv6b_native_pilot/fold_${f}_probs.npz" && -f "results/r55/iv6b_native_pilot/fold_${f}_best.pt" ]]; then
      echo "[SKIP] iv6b ${tag} already exists."
      return 0
    fi
  fi
  echo "[START $(date +%H:%M:%S)] iv6b ${tag} on GPUs ${gpus} (port ${port})"
  python3 -u code/r55/train_iv6b_native_tiles.py \
    --fold "${f}" \
    --gpus "${gpus}" \
    --master-port "${port}" \
    --tune-mode lora_top30 \
    --lora-rank 128 \
    --lora-alpha 256 \
    --ckpt-blocks 20 \
    --epochs 5 \
    --out-dir results/r55/iv6b_native_pilot \
    > "results/r55/iv6b_native_pilot/train_${tag}.log" 2>&1
  echo "[DONE  $(date +%H:%M:%S)] iv6b ${tag} on GPUs ${gpus}"
}

run_qwen() {
  local f="$1"
  local gpus="$2"
  local port="$3"
  local tag="fold_${f}"
  if [[ "${f}" == "-1" ]]; then
    tag="fullfit"
    if [[ -f "results/r55/qwen3vl_hires_pilot/ckpts/qwen3vl_hires_s2026_fullfit_swa.pt" ]]; then
      echo "[SKIP] qwen ${tag} already exists."
      return 0
    fi
  else
    if [[ -f "results/r55/qwen3vl_hires_pilot/fold_${f}_probs.npz" && -f "results/r55/qwen3vl_hires_pilot/ckpts/qwen3vl_hires_s2026_fold_${f}_swa.pt" ]]; then
      echo "[SKIP] qwen ${tag} already exists."
      return 0
    fi
  fi
  echo "[START $(date +%H:%M:%S)] qwen ${tag} on GPUs ${gpus} (port ${port})"
  CUDA_VISIBLE_DEVICES="${gpus}" torchrun --nproc_per_node=2 --master_port="${port}" \
    code/r55/train_qwen3vl_hires_roi.py \
    --fold "${f}" \
    --epochs 5 \
    --batch-size 4 \
    --grad-accum 1 \
    --gc-mode llm_24 \
    --lora-rank 64 \
    --lambda-kd 0.25 \
    --workers 4 \
    > "results/r55/qwen3vl_hires_pilot/${tag}.log" 2>&1
  echo "[DONE  $(date +%H:%M:%S)] qwen ${tag} on GPUs ${gpus}"
}

echo "=== Launching 4 Balanced 2-GPU Lanes across GPUs 0..7 ($(date)) ==="

# Lane 0 (GPUs 0,1): iv6b F2 -> iv6b F3 -> iv6b FullFit (-1)
(
  run_iv6b 2 "0,1" 29561
  run_iv6b 3 "0,1" 29563
  run_iv6b -1 "0,1" 29565
) > results/r55/dualcore_fullfit_final/logs/lane0_gpu01.log 2>&1 &
PID_L0=$!

# Lane 1 (GPUs 2,3): iv6b F4 -> iv6b F5 -> qwen F2
(
  run_iv6b 4 "2,3" 29571
  run_iv6b 5 "2,3" 29573
  run_qwen 2 "2,3" 29575
) > results/r55/dualcore_fullfit_final/logs/lane1_gpu23.log 2>&1 &
PID_L1=$!

# Lane 2 (GPUs 4,5): iv6b F6 -> qwen F3 -> qwen F4 -> qwen F5
(
  run_iv6b 6 "4,5" 29581
  run_qwen 3 "4,5" 29583
  run_qwen 4 "4,5" 29585
  run_qwen 5 "4,5" 29587
) > results/r55/dualcore_fullfit_final/logs/lane2_gpu45.log 2>&1 &
PID_L2=$!

# Lane 3 (GPUs 6,7): iv6b F7 -> qwen F6 -> qwen F7 -> qwen FullFit (-1)
(
  run_iv6b 7 "6,7" 29591
  run_qwen 6 "6,7" 29593
  run_qwen 7 "6,7" 29595
  run_qwen -1 "6,7" 29597
) > results/r55/dualcore_fullfit_final/logs/lane3_gpu67.log 2>&1 &
PID_L3=$!

echo "  Lane PIDs: L0=${PID_L0} L1=${PID_L1} L2=${PID_L2} L3=${PID_L3}"
FAIL=0
for pid in "${PID_L0}" "${PID_L1}" "${PID_L2}" "${PID_L3}"; do
  if ! wait "${pid}"; then
    echo "[ERROR] A training lane (${pid}) exited with non-zero status!" >&2
    FAIL=1
  fi
done
if [[ "${FAIL}" -ne 0 ]]; then
  exit 1
fi

echo "=== All 8 Folds + 100% FullFit Checkpoints Complete! Running Final OOF Fit & Fresh Inference ($(date)) ==="
python3 -u code/r55/clean_natural_prior_decoder.py --finalize-full8 --gpus 0,1,2,3,4,5,6,7 \
  | tee results/r55/dualcore_fullfit_final/finalize_full8.log
