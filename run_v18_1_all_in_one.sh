#!/usr/bin/env bash
# ==============================================================================
# V18.1 One-Click Execution Pipeline for Remote 8xH200 Agent
# ==============================================================================
# Usage:
#   1) Instant Hotfix Only (~2 mins, NO retraining needed, fixes the V18 test_150
#      recall collapse & calibration mismatch immediately using existing checkpoints):
#        bash run_v18_1_all_in_one.sh --stage1-only
#
#   2) Full Pipeline (~15 mins on 8xH200: Stage 1 Instant Hotfix + Stage 2 Native
#      Overgen Quadruplet Expansion + True 6B/V16 KD Retraining + Final Export):
#        bash run_v18_1_all_in_one.sh --full
# ==============================================================================
set -euo pipefail

if [ -d "/wanqing-develop/luowenjing/18" ]; then
    REPO_ROOT="/wanqing-develop/luowenjing/18"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}"

MODE="${1:---full}"

echo "======================================================================"
echo " [Stage 1/2] V18.1 Instant Calibration & Synergy Router Hotfix (~2 min)"
echo "======================================================================"
echo " -> Step 1.1: Running 4-GPU Parallel Calibrated V18_Full Inference (with V16 Anchor Alignment)..."
python3 code/finetune/infer_v18_fulldata_test150.py --anchor-weight 0.55

echo " -> Step 1.2: Running Upgraded 6B + 4-Spec + FullData Synergy Router & Split-Mass Decoder..."
python3 code/finetune/upgrade_6b_4spec_synergy_router.py

echo " -> Step 1.3: Verifying Unified Single-Card 4090 DAG Engine & Exporting Submission JSON..."
python3 code/deploy/unified_4090_dag_engine.py --out-json "${REPO_ROOT}/pred_nriqa_dataset_test.json"

cp -f "${REPO_ROOT}/pred_nriqa_dataset_test.json" "${REPO_ROOT}/results/final/pred_nriqa_dataset_test_v18_1_stage1_hotfix.json"
echo " [Stage 1 Complete] Saved hotfix submission to:"
echo "   - ${REPO_ROOT}/pred_nriqa_dataset_test.json"
echo "   - ${REPO_ROOT}/results/final/pred_nriqa_dataset_test_v18_1_stage1_hotfix.json"

if [ "${MODE}" = "--stage1-only" ]; then
    echo "======================================================================"
    echo " Finished --stage1-only mode. You can immediately test pred_nriqa_dataset_test.json!"
    echo "======================================================================"
    exit 0
fi

echo ""
echo "======================================================================"
echo " [Stage 2/2] V18.1 Native Quadruplet Expansion + True 6B/V16 KD Retrain"
echo "======================================================================"
echo " -> Step 2.1: Building Native-Resolution Corridor-Verified Quadruplets (200+ Overgen Quads)..."
python3 code/synth/build_v18_native_corridor_quads.py

echo " -> Step 2.2: Launching 8-GPU Parallel V18.1 Specialists (s16 & s17, True 6B+V16 KD, 2+2 Epochs)..."
python3 code/finetune/launch_v18_paired_kd_8gpus.py

echo " -> Step 2.3: Launching 4-GPU Parallel V18.1 FullData Specialists (3,073 Real + Quads)..."
python3 code/finetune/launch_v18_fulldata_4gpus.py

echo " -> Step 2.4: Running Final 6B + 4-Spec + FullData Synergy Router & 4090 DAG Export..."
python3 code/finetune/upgrade_6b_4spec_synergy_router.py
python3 code/deploy/unified_4090_dag_engine.py --out-json "${REPO_ROOT}/pred_nriqa_dataset_test.json"

cp -f "${REPO_ROOT}/pred_nriqa_dataset_test.json" "${REPO_ROOT}/results/final/pred_nriqa_dataset_test_v18_1_final.json"
echo "======================================================================"
echo " [All Stages Complete] Final submission ready at:"
echo "   - ${REPO_ROOT}/pred_nriqa_dataset_test.json"
echo "   - ${REPO_ROOT}/results/final/pred_nriqa_dataset_test_v18_1_final.json"
echo "   - ${REPO_ROOT}/results/final/pred_nriqa_dataset_test_v18_fulldata.json"
echo "======================================================================"
