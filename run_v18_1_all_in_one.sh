#!/usr/bin/env bash
# ==============================================================================
# V18.2 One-Click Execution Pipeline for Remote 8xH200 Agent
# ==============================================================================
# Usage:
#   1) Instant V18.2 Router & Decoder Hotfix Only (~30s - 2min, NO retraining needed;
#      fixes the 4 exact root causes of the V18.1 51.80 negative optimization):
#        bash run_v18_1_all_in_one.sh --stage1-only
#
#   2) Full Pipeline (~15 mins on 8xH200: Stage 1 Instant Hotfix + Stage 2 Native
#      Overgen Quadruplet Expansion + Prior-Preserving KD Retraining + Final Export):
#        bash run_v18_1_all_in_one.sh --full
# ==============================================================================
set -euo pipefail

if [ -d "/wanqing-develop/luowenjing/18" ]; then
    REPO_ROOT="/wanqing-develop/luowenjing/18"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${REPO_ROOT}"

MODE="${1:---stage1-only}"

echo "======================================================================"
echo " [Stage 1/2] V18.2 Foundation-Anchored Synergy Router & Ratio-Gated Hotfix"
echo "======================================================================"
if [ ! -f "${REPO_ROOT}/results/v18_fulldata/V18_Full_SpecA_OverGen_TexLoss_probs.npz" ] && [ -f "${REPO_ROOT}/results/v18_fulldata/V18_Full_SpecA_OverGen_TexLoss_swa.pt" ]; then
    echo " -> Step 1.1: Running 4-GPU Parallel Calibrated V18_Full Inference..."
    python3 code/finetune/infer_v18_fulldata_test150.py --anchor-weight 0.55
else
    echo " -> Step 1.1: Calibrated V18_Full probability archives already exist (or skipped); proceeding directly to V18.2 Router..."
fi

echo " -> Step 1.2: Running V18.2 Pure-Real Foundation-Anchored Synergy Router & Strict Ratio-Gated Decoder on s16 + s17..."
python3 code/finetune/upgrade_6b_4spec_synergy_router.py

echo " -> Step 1.3: Running Unified Single-Card 4090 DAG Engine & Exporting V18.2 Submission JSON..."
python3 code/deploy/unified_4090_dag_engine.py --out-json "${REPO_ROOT}/pred_nriqa_dataset_test.json"

cp -f "${REPO_ROOT}/pred_nriqa_dataset_test.json" "${REPO_ROOT}/results/final/pred_nriqa_dataset_test_v18_2_hotfix.json"
echo " [Stage 1 Complete] Saved V18.2 submission to:"
echo "   - ${REPO_ROOT}/pred_nriqa_dataset_test.json"
echo "   - ${REPO_ROOT}/results/final/pred_nriqa_dataset_test_v18_2_hotfix.json"

if [ "${MODE}" = "--stage1-only" ]; then
    echo "======================================================================"
    echo " Finished --stage1-only mode. Ready to evaluate pred_nriqa_dataset_test.json!"
    echo "======================================================================"
    exit 0
fi

echo ""
echo "======================================================================"
echo " [Stage 2/2] V18.2 Native Quadruplet Expansion + Prior-Preserving KD Retrain"
echo "======================================================================"
echo " -> Step 2.1: Building Native-Resolution Corridor-Verified Quadruplets..."
python3 code/synth/build_v18_native_corridor_quads.py

echo " -> Step 2.2: Launching 8-GPU Parallel V18.2 Specialists (s16 & s17, Zero P1-Inflation Stage 2)..."
python3 code/finetune/launch_v18_paired_kd_8gpus.py

echo " -> Step 2.3: Launching 4-GPU Parallel V18.2 FullData Specialists..."
python3 code/finetune/launch_v18_fulldata_4gpus.py

echo " -> Step 2.4: Running Final V18.2 Foundation-Anchored Synergy Router & 4090 DAG Export..."
python3 code/finetune/upgrade_6b_4spec_synergy_router.py
python3 code/deploy/unified_4090_dag_engine.py --out-json "${REPO_ROOT}/pred_nriqa_dataset_test.json"

cp -f "${REPO_ROOT}/pred_nriqa_dataset_test.json" "${REPO_ROOT}/results/final/pred_nriqa_dataset_test_v18_2_final.json"
echo "======================================================================"
echo " [All Stages Complete] Final submission ready at:"
echo "   - ${REPO_ROOT}/pred_nriqa_dataset_test.json"
echo "   - ${REPO_ROOT}/results/final/pred_nriqa_dataset_test_v18_2_final.json"
echo "======================================================================"
