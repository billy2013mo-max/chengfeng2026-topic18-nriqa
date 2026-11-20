#!/usr/bin/env bash
# ==============================================================================
# run_gpu0to6_finish.sh
# Strictly uses GPUs 0..6 ONLY — leaving GPU 7 100% FREE for the user!
# Coordinates seamlessly with existing active lanes:
#   - L0 (PID 2812086 on GPUs 0,1): iv6b F2 -> iv6b F3 -> iv6b FullFit
#   - L1 (PID 2812087 on GPUs 2,3): iv6b F4 -> iv6b F5 -> qwen F2  (then picks up qwen FullFit!)
#   - L2 (PID 2812088 on GPUs 4,5): iv6b F6 -> qwen F3 -> qwen F4 -> qwen F5 (then picks up iv6b F7!)
#   - Single GPU 6: qwen F6 -> qwen F7 (--batch-size 4 --grad-accum 2 -> exact eff_bsz=8!)
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

wait_pid_gone() {
  local pid="$1"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 15
  done
}

run_qwen_1gpu() {
  local f="$1"
  local gpu="$2"
  local tag="fold_${f}"
  if [[ -f "results/r55/qwen3vl_hires_pilot/fold_${f}_probs.npz" && -f "results/r55/qwen3vl_hires_pilot/ckpts/qwen3vl_hires_s2026_fold_${f}_swa.pt" ]]; then
    echo "[SKIP] qwen ${tag} already exists."
    return 0
  fi
  echo "[START $(date +%H:%M:%S)] qwen ${tag} on single GPU ${gpu} (eff_bsz=8)"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 -u code/r55/train_qwen3vl_hires_roi.py \
    --gpu 0 \
    --fold "${f}" \
    --epochs 5 \
    --batch-size 4 \
    --grad-accum 2 \
    --gc-mode llm_24 \
    --lora-rank 64 \
    --lambda-kd 0.25 \
    --workers 6 \
    > "results/r55/qwen3vl_hires_pilot/${tag}.log" 2>&1
  echo "[DONE  $(date +%H:%M:%S)] qwen ${tag} on single GPU ${gpu}"
}

run_qwen_2gpu() {
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

run_iv6b_2gpu() {
  local f="$1"
  local gpus="$2"
  local port="$3"
  local tag="fold${f}"
  if [[ -f "results/r55/iv6b_native_pilot/fold_${f}_probs.npz" && -f "results/r55/iv6b_native_pilot/fold_${f}_best.pt" ]]; then
    echo "[SKIP] iv6b ${tag} already exists."
    return 0
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

echo "=== [7-GPU Mode: GPUs 0..6 ONLY, GPU 7 100% RESERVED FOR USER] ($(date)) ==="

# 1) Single GPU 6 lane: qwen F6 -> qwen F7
(
  run_qwen_1gpu 6 6
  run_qwen_1gpu 7 6
) > results/r55/dualcore_fullfit_final/logs/lane_single_gpu6.log 2>&1 &
PID_G6=$!

# 2) Follow-up on GPUs 2,3 right after L1 (2812087) finishes: qwen FullFit (-1) + iv6b F7!
#    Wait: L1 finishes at ~265m from now -> qwen FullFit (~75m -> 340m); L0/L2 finish at ~315m -> immediately run iv6b F7!
(
  wait_pid_gone 2812087
  run_qwen_2gpu -1 "2,3" 29577
) > results/r55/dualcore_fullfit_final/logs/lane1_followup_gpu23.log 2>&1 &
PID_F23=$!

# 3) Follow-up on GPUs 4,5 right after L2 (2812088) finishes: iv6b F7!
(
  wait_pid_gone 2812088
  rm -f results/r55/iv6b_native_pilot/fold_7_probs.npz results/r55/iv6b_native_pilot/fold_7_best.pt
  run_iv6b_2gpu 7 "4,5" 29589
) > results/r55/dualcore_fullfit_final/logs/lane2_followup_gpu45.log 2>&1 &
PID_F45=$!

# Wait for L0 (2812086 on GPUs 0,1) + all three continuation tracks
wait_pid_gone 2812086
wait "${PID_G6}"
wait "${PID_F23}"
wait "${PID_F45}"

echo "=== All 8 Folds + 100% FullFit Complete on GPUs 0..6! Finalizing OOF & Fresh Inference on GPUs 0..6 ($(date)) ==="
python3 -u code/r55/clean_natural_prior_decoder.py --finalize-full8 --gpus 0,1,2,3,4,5,6 \
  | tee results/r55/dualcore_fullfit_final/finalize_full8.log
