#!/usr/bin/env bash
# ==============================================================================
# PRISM (Pairwise Reference-anchored Integrated Severity Microscope) Pipeline
# ==============================================================================
# Modes:
#   1) Fast Reference-Anchored Hazard Calibration & Evaluation (~30s):
#        bash run_prism_pipeline.sh --eval-only
#
#   2) Full End-to-End PRISM Retraining & Evaluation Pipeline (8xH200 / 4x4090):
#      Stage 1: Split-isolated Real-Defect Multi-Band Poisson Quadruplet Synthesis
#      Stage 2: Parallel 8-GPU Training of PRISM Three Full-Spectrum Towers:
#               - Tower 3 (`Prism-Quad` All-6 Deep Microscope on s16 & s17)
#               - Tower 1 (`Prism-ViT6B` 6B + 1:1 Microscope on s16 & s17)
#               - Tower 2 (`Prism-VLM8B` 960x640 Composite Native Microscope on s16 & s17)
#      Stage 3: Unified Reference-Anchored Hazard Calibration & Final Submission Export
#        bash run_prism_pipeline.sh --full
# ==============================================================================
set -euo pipefail

if [ -d "/wanqing-develop/luowenjing/18" ]; then
    REPO_ROOT="/wanqing-develop/luowenjing/18"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/code:${REPO_ROOT}/code/finetune:${REPO_ROOT}/code/decision_eval:${REPO_ROOT}/code/prism:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODE="${1:---eval-only}"
mkdir -p "${REPO_ROOT}/results/prism" "${REPO_ROOT}/results/final" "${REPO_ROOT}/data/v17_splits"

if [ "${MODE}" = "--eval-only" ]; then
    echo "======================================================================"
    echo " [PRISM Fast Mode] Reference-Anchored Conditional Hazard Calibration"
    echo "======================================================================"
    python3 -u code/prism/evaluate_prism.py --out-tag prism_ra_iqa
    echo "======================================================================"
    echo " [PRISM Complete] Exported submission to:"
    echo "   - ${REPO_ROOT}/pred_nriqa_dataset_test.json"
    echo "   - ${REPO_ROOT}/results/final/pred_nriqa_dataset_test_prism_ra_iqa.json"
    echo "======================================================================"
    exit 0
fi

echo "======================================================================"
echo " [Stage 1/3] Generating Split-Isolated Real-Defect Poisson Quadruplets"
echo "======================================================================"
python3 -u code/prism/real_defect_synthesizer.py --split data/resplits/split_s16.csv --max-quads 60
python3 -u code/prism/real_defect_synthesizer.py --split data/resplits/split_s17.csv --max-quads 60

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)
echo "======================================================================"
echo " [Stage 2/3] Launching PRISM Three Full-Spectrum Towers (Detected ${NUM_GPUS} GPUs)"
echo "======================================================================"

if [ "${NUM_GPUS}" -ge 8 ]; then
    echo " -> Launching all 6 PRISM Tower jobs concurrently across GPUs 0..7..."
    CUDA_VISIBLE_DEVICES=0 python3 -u code/prism/train_prism_quad.py \
        --split data/v17_splits/split_s16_prism.csv \
        --manifest data/v17_splits/manifest_s16_prism.csv \
        --init-ckpt results/v17_end2end_s16_s17/V17_S2R_SpecD_All6_Giant_s16_ep5.pt \
        --epochs-s1 3 --epochs-s2 2 \
        --out results/prism/prism_quad_s16.json > results/prism/prism_quad_s16.log 2>&1 &
    PID_Q16=$!

    CUDA_VISIBLE_DEVICES=1 python3 -u code/prism/train_prism_quad.py \
        --split data/v17_splits/split_s17_prism.csv \
        --manifest data/v17_splits/manifest_s17_prism.csv \
        --init-ckpt results/v17_end2end_s16_s17/V17_S2R_SpecD_All6_Giant_s17_ep5.pt \
        --epochs-s1 3 --epochs-s2 2 \
        --out results/prism/prism_quad_s17.json > results/prism/prism_quad_s17.log 2>&1 &
    PID_Q17=$!

    CUDA_VISIBLE_DEVICES=2 python3 -u code/prism/train_prism_vit6b.py \
        --split data/resplits/split_s16.csv \
        --epochs 4 --batch 2 --grad-accum 4 \
        --out results/prism/prism_vit6b_s16.json > results/prism/prism_vit6b_s16.log 2>&1 &
    PID_V16=$!

    CUDA_VISIBLE_DEVICES=3 python3 -u code/prism/train_prism_vit6b.py \
        --split data/resplits/split_s17.csv \
        --epochs 4 --batch 2 --grad-accum 4 \
        --out results/prism/prism_vit6b_s17.json > results/prism/prism_vit6b_s17.log 2>&1 &
    PID_V17=$!

    CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29518 code/prism/train_prism_vlm8b.py \
        --split data/resplits/split_s16.csv \
        --epochs 3 --batch 8 \
        --out results/prism/prism_vlm8b_s16.json > results/prism/prism_vlm8b_s16.log 2>&1 &
    PID_M16=$!

    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29519 code/prism/train_prism_vlm8b.py \
        --split data/resplits/split_s17.csv \
        --epochs 3 --batch 8 \
        --out results/prism/prism_vlm8b_s17.json > results/prism/prism_vlm8b_s17.log 2>&1 &
    PID_M17=$!

    echo " -> Monitoring parallel jobs (logs in results/prism/*.log)..."
    wait "${PID_Q16}" "${PID_Q17}" "${PID_V16}" "${PID_V17}" "${PID_M16}" "${PID_M17}"
else
    echo " -> Wave 1: Training Prism-Quad & Prism-ViT6B on GPUs 0..3..."
    CUDA_VISIBLE_DEVICES=0 python3 -u code/prism/train_prism_quad.py \
        --split data/v17_splits/split_s16_prism.csv \
        --manifest data/v17_splits/manifest_s16_prism.csv \
        --epochs-s1 3 --epochs-s2 2 \
        --out results/prism/prism_quad_s16.json &
    PID_Q16=$!

    CUDA_VISIBLE_DEVICES=1 python3 -u code/prism/train_prism_quad.py \
        --split data/v17_splits/split_s17_prism.csv \
        --manifest data/v17_splits/manifest_s17_prism.csv \
        --epochs-s1 3 --epochs-s2 2 \
        --out results/prism/prism_quad_s17.json &
    PID_Q17=$!

    CUDA_VISIBLE_DEVICES=2 python3 -u code/prism/train_prism_vit6b.py \
        --split data/resplits/split_s16.csv \
        --epochs 4 --batch 2 --grad-accum 4 \
        --out results/prism/prism_vit6b_s16.json &
    PID_V16=$!

    CUDA_VISIBLE_DEVICES=3 python3 -u code/prism/train_prism_vit6b.py \
        --split data/resplits/split_s17.csv \
        --epochs 4 --batch 2 --grad-accum 4 \
        --out results/prism/prism_vit6b_s17.json &
    PID_V17=$!

    wait "${PID_Q16}" "${PID_Q17}" "${PID_V16}" "${PID_V17}"

    echo " -> Wave 2: Training Prism-VLM8B on GPUs 0..3..."
    torchrun --nproc_per_node=4 --master_port=29518 code/prism/train_prism_vlm8b.py \
        --split data/resplits/split_s16.csv \
        --epochs 3 --batch 6 \
        --out results/prism/prism_vlm8b_s16.json

    torchrun --nproc_per_node=4 --master_port=29519 code/prism/train_prism_vlm8b.py \
        --split data/resplits/split_s17.csv \
        --epochs 3 --batch 6 \
        --out results/prism/prism_vlm8b_s17.json
fi

echo "======================================================================"
echo " [Stage 3/3] Running PRISM Reference-Anchored Multi-Tower Evaluation"
echo "======================================================================"
python3 -u code/prism/evaluate_prism.py --out-tag prism_full_55plus
echo "======================================================================"
echo " [PRISM Full Pipeline Complete] Final submission exported to:"
echo "   - ${REPO_ROOT}/pred_nriqa_dataset_test.json"
echo "   - ${REPO_ROOT}/results/final/pred_nriqa_dataset_test_prism_full_55plus.json"
echo "======================================================================"
