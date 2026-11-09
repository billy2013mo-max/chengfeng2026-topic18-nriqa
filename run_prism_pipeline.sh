#!/usr/bin/env bash
# ==============================================================================
# PRISM (Pairwise Reference-anchored Integrated Severity Microscope) Pipeline
# ==============================================================================
# Modes:
#   1) Fast Reference-Anchored Hazard Calibration & Evaluation (~30s):
#      Builds split-isolated Scene-Matched Real Anchor Banks (`split == 'train'`),
#      trains the Reference-Anchored Cross-Attention Comparator on validation splits,
#      evaluates on blind test splits `s16` & `s17` (N=2,000), and exports `pred_nriqa_dataset_test.json`:
#        bash run_prism_pipeline.sh --eval-only
#
#   2) Full End-to-End PRISM Retraining & Evaluation Pipeline (for 4x4090 / 8xH200):
#      Stage 1: Split-isolated Real-Defect Multi-Band Poisson Quadruplet Synthesis
#      Stage 2: Tower 3 (`Prism-Quad` All-6-Class Deep Microscope, lastk6 + Layer Decay 0.85)
#      Stage 3: Tower 1 (`Prism-ViT6B` 1:1 Microscope + ABMIL + Conditional Hazard Head)
#      Stage 4: Tower 2 (`Prism-VLM8B` Multi-Crop Global + 1:1 Native Edge/Body Qwen3-VL-8B)
#      Stage 5: Unified Reference-Anchored Hazard Calibration & Final Submission Export
#        bash run_prism_pipeline.sh --full
# ==============================================================================
set -euo pipefail

if [ -d "/wanqing-develop/luowenjing/18" ]; then
    REPO_ROOT="/wanqing-develop/luowenjing/18"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/code:${REPO_ROOT}/code/finetune:${REPO_ROOT}/code/decision_eval:${PYTHONPATH:-}"

MODE="${1:---eval-only}"
mkdir -p "${REPO_ROOT}/results/prism" "${REPO_ROOT}/results/final"

if [ "${MODE}" = "--eval-only" ]; then
    echo "======================================================================"
    echo " [PRISM Fast Mode] Reference-Anchored Conditional Hazard Calibration"
    echo "======================================================================"
    python3 code/prism/evaluate_prism.py --out-tag prism_ra_iqa
    echo "======================================================================"
    echo " [PRISM Complete] Exported submission to:"
    echo "   - ${REPO_ROOT}/pred_nriqa_dataset_test.json"
    echo "   - ${REPO_ROOT}/results/final/pred_nriqa_dataset_test_prism_ra_iqa.json"
    echo "======================================================================"
    exit 0
fi

echo "======================================================================"
echo " [Stage 1/5] Generating Split-Isolated Real-Defect Poisson Quadruplets"
echo "======================================================================"
python3 code/prism/real_defect_synthesizer.py --split data/resplits/split_s16.csv --max-quads 80
python3 code/prism/real_defect_synthesizer.py --split data/resplits/split_s17.csv --max-quads 80

echo "======================================================================"
echo " [Stage 2/5] Training PRISM Tower 3: Prism-Quad (All-6 Deep Microscope)"
echo "======================================================================"
CUDA_VISIBLE_DEVICES=0 python3 code/prism/train_prism_quad.py \
    --split data/resplits/split_s16.csv \
    --backbones "dinov3l:lastk6,pecoreg:lastk6,siglip2g:lastk6,convnext:full" \
    --epochs-s1 4 --epochs-s2 3 \
    --out results/prism/prism_quad_s16.json &
PID_Q16=$!

CUDA_VISIBLE_DEVICES=1 python3 code/prism/train_prism_quad.py \
    --split data/resplits/split_s17.csv \
    --backbones "dinov3l:lastk6,pecoreg:lastk6,siglip2g:lastk6,convnext:full" \
    --epochs-s1 4 --epochs-s2 3 \
    --out results/prism/prism_quad_s17.json &
PID_Q17=$!

echo "======================================================================"
echo " [Stage 3/5] Training PRISM Tower 1: Prism-ViT6B (6B + 1:1 Microscope)"
echo "======================================================================"
CUDA_VISIBLE_DEVICES=2 python3 code/prism/train_prism_vit6b.py \
    --split data/resplits/split_s16.csv \
    --epochs 6 \
    --out results/prism/prism_vit6b_s16.json &
PID_V16=$!

CUDA_VISIBLE_DEVICES=3 python3 code/prism/train_prism_vit6b.py \
    --split data/resplits/split_s17.csv \
    --epochs 6 \
    --out results/prism/prism_vit6b_s17.json &
PID_V17=$!

wait "${PID_Q16}" "${PID_Q17}" "${PID_V16}" "${PID_V17}"

echo "======================================================================"
echo " [Stage 4/5] Training PRISM Tower 2: Prism-VLM8B (Multi-Crop Qwen3-VL-8B)"
echo "======================================================================"
torchrun --nproc_per_node=4 --master_port=29518 code/prism/train_prism_vlm8b.py \
    --split data/resplits/split_s16.csv \
    --epochs 4 \
    --out results/prism/prism_vlm8b_s16.json

torchrun --nproc_per_node=4 --master_port=29519 code/prism/train_prism_vlm8b.py \
    --split data/resplits/split_s17.csv \
    --epochs 4 \
    --out results/prism/prism_vlm8b_s17.json

echo "======================================================================"
echo " [Stage 5/5] Running PRISM Reference-Anchored Multi-Tower Evaluation"
echo "======================================================================"
python3 code/prism/evaluate_prism.py --out-tag prism_full_55plus
echo "======================================================================"
echo " [PRISM Full Pipeline Complete] Final submission exported to:"
echo "   - ${REPO_ROOT}/pred_nriqa_dataset_test.json"
echo "   - ${REPO_ROOT}/results/final/pred_nriqa_dataset_test_prism_full_55plus.json"
echo "======================================================================"
