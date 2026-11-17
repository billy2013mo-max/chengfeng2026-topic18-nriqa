#!/usr/bin/env bash
# ==============================================================================
# run_os_og_breakthrough.sh
# Turnkey End-to-End Launcher:
#   1) Method 1: 8-GPU Parallel Native 1:1 Spatial Salience x Artifact Extraction
#      (8-fold leak-free OOF on 3,573 images + sharded full-fit on test_150)
#   2) Method 2 + Joint (1+2): Strict 8-Fold Video-Grouped Nested CV Calibration,
#      Ablation Scoreboard & Clean-Locked Column-Isolated Ordinal Decoder Fit
#   3) Multi-GPU Data-Parallel Fresh Ensemble Inference on test_150 + N=1 Audit +
#      Official Submission Schema Validation
#
# Usage (8xH200 default):
#   bash run_os_og_breakthrough.sh
#
# Optional overrides:
#   SET_NAME=resident_v7_core GPUS=0,1,2,3,4,5,6,7 bash run_os_og_breakthrough.sh
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SET_NAME="${SET_NAME:-resident_v7}"
TEMPLATE="${TEMPLATE:-results/v2_full/template_150.json}"
IMG_DIR="${IMG_DIR:-data/test_150}"
OUT_DIR="${OUT_DIR:-results/r55/os_og_breakthrough}"
SPATIAL_DIR="${OUT_DIR}/spatial_cache"
FINAL_JSON="${FINAL_JSON:-${OUT_DIR}/pred_nriqa_dataset_test_os_og_breakthrough.json}"
FORCE_SPATIAL="${FORCE_SPATIAL:-0}"

if [[ -z "${GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
  else
    GPUS="0"
  fi
fi
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_GPUS="${#GPU_ARR[@]}"

mkdir -p "${OUT_DIR}" "${SPATIAL_DIR}" "${OUT_DIR}/logs"

echo "=============================================================================="
echo "[OS/OG Breakthrough] Repo       : ${REPO_DIR}"
echo "[OS/OG Breakthrough] Ensemble   : ${SET_NAME}"
echo "[OS/OG Breakthrough] Active GPUs: ${GPUS} (${NUM_GPUS} GPUs)"
echo "[OS/OG Breakthrough] Template   : ${TEMPLATE} (img_dir=${IMG_DIR})"
echo "[OS/OG Breakthrough] Output JSON: ${FINAL_JSON}"
echo "=============================================================================="
df -h "${REPO_DIR}" | head -n 2 || true

# ------------------------------------------------------------------------------
# Stage 0: Fast synthetic self-tests + ensure baseline OOF decoder exists
# ------------------------------------------------------------------------------
echo "[Stage 0/4] Running deterministic unit & N=1 self-tests..."
"${PYTHON_BIN}" code/r55/spatial_salience_probe.py --mode self-test
"${PYTHON_BIN}" code/r55/deconfound_calibrator.py --self-test

BASE_DEC_PKL="results/r55/final/${SET_NAME}_decoder_routed_gate_eu.pkl"
if [[ ! -f "${BASE_DEC_PKL}" ]]; then
  echo "[Stage 0/4] Fitting baseline RoutedGateEUDecoder for '${SET_NAME}' on OOF..."
  "${PYTHON_BIN}" -u code/r55/final_engine.py --set "${SET_NAME}" --fit-decoder
fi

# ------------------------------------------------------------------------------
# Stage 1A: 8-Fold Leak-Free OOF Spatial Salience Extraction (Parallel across GPUs)
# ------------------------------------------------------------------------------
SPATIAL_OOF_NPZ="${SPATIAL_DIR}/spatial_oof_3573.npz"
if [[ -f "${SPATIAL_OOF_NPZ}" && "${FORCE_SPATIAL}" != "1" ]]; then
  echo "[Stage 1A/4] Found existing leak-free OOF spatial cache: ${SPATIAL_OOF_NPZ} (skip; set FORCE_SPATIAL=1 to recompute)"
else
  echo "[Stage 1A/4] Extracting 8-fold leak-free native 1:1 spatial salience features across ${NUM_GPUS} GPUs..."
  fold_idx=0
  while [[ "${fold_idx}" -lt 8 ]]; do
    pids=()
    logs=()
    for (( slot=0; slot<NUM_GPUS && fold_idx<8; slot++, fold_idx++ )); do
      gpu_id="${GPU_ARR[$slot]}"
      fold_npz="${SPATIAL_DIR}/oof_fold_${fold_idx}.npz"
      if [[ -f "${fold_npz}" && "${FORCE_SPATIAL}" != "1" ]]; then
        echo "  - Fold ${fold_idx} cache exists (${fold_npz}), skipping."
        continue
      fi
      log_f="${OUT_DIR}/logs/spatial_oof_fold_${fold_idx}_gpu${gpu_id}.log"
      echo "  - Launching OOF Fold ${fold_idx} on GPU ${gpu_id} -> ${log_f}"
      CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u code/r55/spatial_salience_probe.py \
        --mode oof \
        --fold "${fold_idx}" \
        --gpu "${gpu_id}" \
        --out-dir "${SPATIAL_DIR}" \
        --out-npz "${fold_npz}" > "${log_f}" 2>&1 &
      pids+=("$!")
      logs+=("${log_f}")
    done
    if [[ "${#pids[@]}" -gt 0 ]]; then
      for idx_p in "${!pids[@]}"; do
        if ! wait "${pids[$idx_p]}"; then
          echo "[ERROR] OOF Spatial Salience worker failed (PID=${pids[$idx_p]}, log=${logs[$idx_p]}):" >&2
          tail -n 45 "${logs[$idx_p]}" >&2 || true
          exit 1
        fi
      done
    fi
  done

  "${PYTHON_BIN}" -u code/r55/spatial_salience_probe.py \
    --mode merge-oof \
    --out-dir "${SPATIAL_DIR}" \
    --out-npz "${SPATIAL_OOF_NPZ}"
fi

# ------------------------------------------------------------------------------
# Stage 1B: Sharded Multi-GPU Spatial Salience Extraction on test_150
# ------------------------------------------------------------------------------
SPATIAL_TGT_NPZ="${SPATIAL_DIR}/spatial_target.npz"
rm -f "${SPATIAL_DIR}"/target_shard_*_of_*.npz
echo "[Stage 1B/4] Extracting native 1:1 spatial salience features on '${TEMPLATE}' across ${NUM_GPUS} GPUs..."
pids=()
logs=()
for (( s=0; s<NUM_GPUS; s++ )); do
  gpu_id="${GPU_ARR[$s]}"
  log_f="${OUT_DIR}/logs/spatial_target_s${s}_gpu${gpu_id}.log"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u code/r55/spatial_salience_probe.py \
    --mode target \
    --template "${TEMPLATE}" \
    --img-dir "${IMG_DIR}" \
    --shard "${s}" \
    --num-shards "${NUM_GPUS}" \
    --gpu "${gpu_id}" \
    --out-dir "${SPATIAL_DIR}" > "${log_f}" 2>&1 &
  pids+=("$!")
  logs+=("${log_f}")
done
for idx_p in "${!pids[@]}"; do
  if ! wait "${pids[$idx_p]}"; then
    echo "[ERROR] Target Spatial Salience shard ${idx_p} failed (log=${logs[$idx_p]}):" >&2
    tail -n 45 "${logs[$idx_p]}" >&2 || true
    exit 1
  fi
done

"${PYTHON_BIN}" -u code/r55/spatial_salience_probe.py \
  --mode merge-target \
  --template "${TEMPLATE}" \
  --num-shards "${NUM_GPUS}" \
  --out-dir "${SPATIAL_DIR}" \
  --out-npz "${SPATIAL_TGT_NPZ}"

# ------------------------------------------------------------------------------
# Stage 2: Strict 8-Fold Nested OOF Calibration & Ablation (Method 2 vs 1 vs 1+2)
# ------------------------------------------------------------------------------
echo "[Stage 2/4] Running 8-fold video-grouped nested OOF ablation & fitting bundle..."
"${PYTHON_BIN}" -u code/r55/deconfound_calibrator.py \
  --set "${SET_NAME}" \
  --spatial-oof "${SPATIAL_OOF_NPZ}" \
  --out-dir "${OUT_DIR}"

# ------------------------------------------------------------------------------
# Stage 3: Multi-GPU Parallel Fresh Ensemble Inference + N=1 Audit + Validation
# ------------------------------------------------------------------------------
echo "[Stage 3/4] Running ${NUM_GPUS}-GPU parallel fresh FinalEngine inference & generating test_150 JSONs..."
"${PYTHON_BIN}" -u code/r55/multi_gpu_infer.py \
  --set "${SET_NAME}" \
  --gpus "${GPUS}" \
  --template "${TEMPLATE}" \
  --img-dir "${IMG_DIR}" \
  --bundle "${OUT_DIR}/${SET_NAME}_os_og_bundle.pkl" \
  --spatial-target "${SPATIAL_TGT_NPZ}" \
  --out "${FINAL_JSON}"

STEM="${FINAL_JSON%.json}"
echo "=============================================================================="
echo "[DONE] All stages passed strict nested-OOF, N=1 invariance, and schema gates!"
echo "  1) Primary Total-Score Optimal JSON : ${FINAL_JSON}"
echo "  2) Companion P1/P2 Balanced JSON    : ${STEM}_balanced.json"
echo "  3) Fresh Baseline Reference JSON    : ${STEM}_v7_baseline_fresh.json"
echo "  4) Nested OOF Ablation Report       : ${OUT_DIR}/${SET_NAME}_os_og_report.json"
echo "=============================================================================="
