#!/usr/bin/env bash
# ==============================================================================
# PRISM (Pairwise Reference-anchored Integrated Severity Microscope) Pipeline
# ==============================================================================
# Modes:
#   1) Fast Reference-Anchored Hazard Calibration & Evaluation (~15s):
#        bash run_prism_pipeline.sh --eval-only
#
#   2) Full End-to-End PRISM Pipeline with Smart Resume (skips already-completed
#      towers in results/prism/*_probs.npz and trains any missing/failed towers):
#        bash run_prism_pipeline.sh --full
#
#   3) Force Clean Retraining of All 6 Towers from Scratch:
#        bash run_prism_pipeline.sh --force-retrain
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
FORCE_RETRAIN=0
if [ "${MODE}" = "--force-retrain" ]; then
    FORCE_RETRAIN=1
    MODE="--full"
fi

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
if [ "${FORCE_RETRAIN}" -eq 1 ] || [ ! -s "data/v17_splits/split_s16_prism.csv" ] || [ ! -s "data/v17_splits/manifest_s16_prism.csv" ]; then
    python3 -u code/prism/real_defect_synthesizer.py --split data/resplits/split_s16.csv --max-quads 60
else
    echo " [Skip] data/v17_splits/split_s16_prism.csv already prepared."
fi

if [ "${FORCE_RETRAIN}" -eq 1 ] || [ ! -s "data/v17_splits/split_s17_prism.csv" ] || [ ! -s "data/v17_splits/manifest_s17_prism.csv" ]; then
    python3 -u code/prism/real_defect_synthesizer.py --split data/resplits/split_s17.csv --max-quads 60
else
    echo " [Skip] data/v17_splits/split_s17_prism.csv already prepared."
fi

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)
echo "======================================================================"
echo " [Stage 2/3] Launching PRISM Three Full-Spectrum Towers (Detected ${NUM_GPUS} GPUs)"
echo "======================================================================"

declare -a JOB_NAMES=()
declare -a JOB_PIDS=()
declare -a JOB_NPZS=()
declare -a JOB_LOGS=()

launch_job_if_needed() {
    local jname="$1"
    local npz_path="results/prism/${jname}_probs.npz"
    local log_path="results/prism/${jname}.log"
    shift
    if [ "${FORCE_RETRAIN}" -eq 0 ] && [ -s "${npz_path}" ]; then
        echo " [Skip] ${jname} already completed (${npz_path} exists)."
        return 0
    fi
    rm -f "${npz_path}"
    echo " [Launch] ${jname} -> ${log_path}"
    "$@" > "${log_path}" 2>&1 &
    local pid=$!
    JOB_NAMES+=("${jname}")
    JOB_PIDS+=("${pid}")
    JOB_NPZS+=("${npz_path}")
    JOB_LOGS+=("${log_path}")
}

wait_and_verify_jobs() {
    local n_jobs="${#JOB_PIDS[@]}"
    if [ "${n_jobs}" -eq 0 ]; then
        echo " -> All PRISM Tower probability files are already ready!"
        return 0
    fi
    echo " -> Monitoring ${n_jobs} active PRISM training job(s)..."
    while true; do
        local alive=0
        local status_line=""
        for idx in "${!JOB_PIDS[@]}"; do
            local pid="${JOB_PIDS[$idx]}"
            local jname="${JOB_NAMES[$idx]}"
            local lfile="${JOB_LOGS[$idx]}"
            if kill -0 "${pid}" 2>/dev/null; then
                alive=$((alive + 1))
                local last_ep
                last_ep=$(grep -E "ep=|ep[0-9]|FINAL" "${lfile}" 2>/dev/null | tail -n 1 || true)
                status_line="${status_line} [${jname}: ${last_ep:-running}]"
            fi
        done
        if [ "${alive}" -eq 0 ]; then
            break
        fi
        echo " [$(date +%H:%M:%S)] Active=${alive}/${n_jobs}${status_line}"
        sleep 25
    done

    local any_failed=0
    for idx in "${!JOB_PIDS[@]}"; do
        local pid="${JOB_PIDS[$idx]}"
        local jname="${JOB_NAMES[$idx]}"
        local npz_f="${JOB_NPZS[$idx]}"
        local lfile="${JOB_LOGS[$idx]}"
        local rc=0
        wait "${pid}" || rc=$?
        if [ "${rc}" -ne 0 ] || [ ! -s "${npz_f}" ]; then
            echo "======================================================================"
            echo " [ERROR] Job ${jname} FAILED (exit_code=${rc}, npz=${npz_f}). Last 45 log lines:"
            echo "----------------------------------------------------------------------"
            tail -n 45 "${lfile}" || true
            echo "======================================================================"
            any_failed=1
        else
            echo " [OK] ${jname} verified -> ${npz_f}"
        fi
    done

    if [ "${any_failed}" -ne 0 ]; then
        echo "[FATAL] One or more PRISM Tower training jobs failed. Fix or inspect logs above."
        exit 1
    fi
}

if [ "${NUM_GPUS}" -ge 8 ]; then
    launch_job_if_needed "prism_quad_s16" env CUDA_VISIBLE_DEVICES=0 python3 -u code/prism/train_prism_quad.py \
        --split data/v17_splits/split_s16_prism.csv \
        --manifest data/v17_splits/manifest_s16_prism.csv \
        --init-ckpt results/v17_end2end_s16_s17/V17_S2R_SpecD_All6_Giant_s16_ep5.pt \
        --epochs-s1 3 --epochs-s2 2 --batch 4 --grad-accum 4 \
        --out results/prism/prism_quad_s16.json

    launch_job_if_needed "prism_quad_s17" env CUDA_VISIBLE_DEVICES=1 python3 -u code/prism/train_prism_quad.py \
        --split data/v17_splits/split_s17_prism.csv \
        --manifest data/v17_splits/manifest_s17_prism.csv \
        --init-ckpt results/v17_end2end_s16_s17/V17_S2R_SpecD_All6_Giant_s17_ep5.pt \
        --epochs-s1 3 --epochs-s2 2 --batch 4 --grad-accum 4 \
        --out results/prism/prism_quad_s17.json

    launch_job_if_needed "prism_vit6b_s16" env CUDA_VISIBLE_DEVICES=2 python3 -u code/prism/train_prism_vit6b.py \
        --split data/resplits/split_s16.csv \
        --epochs 4 --batch 2 --grad-accum 4 \
        --out results/prism/prism_vit6b_s16.json

    launch_job_if_needed "prism_vit6b_s17" env CUDA_VISIBLE_DEVICES=3 python3 -u code/prism/train_prism_vit6b.py \
        --split data/resplits/split_s17.csv \
        --epochs 4 --batch 2 --grad-accum 4 \
        --out results/prism/prism_vit6b_s17.json

    launch_job_if_needed "prism_vlm8b_s16" env CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29518 code/prism/train_prism_vlm8b.py \
        --split data/resplits/split_s16.csv \
        --epochs 3 --batch 8 \
        --out results/prism/prism_vlm8b_s16.json

    launch_job_if_needed "prism_vlm8b_s17" env CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=29519 code/prism/train_prism_vlm8b.py \
        --split data/resplits/split_s17.csv \
        --epochs 3 --batch 8 \
        --out results/prism/prism_vlm8b_s17.json

    wait_and_verify_jobs
else
    launch_job_if_needed "prism_quad_s16" env CUDA_VISIBLE_DEVICES=0 python3 -u code/prism/train_prism_quad.py \
        --split data/v17_splits/split_s16_prism.csv \
        --manifest data/v17_splits/manifest_s16_prism.csv \
        --epochs-s1 3 --epochs-s2 2 \
        --out results/prism/prism_quad_s16.json

    launch_job_if_needed "prism_quad_s17" env CUDA_VISIBLE_DEVICES=1 python3 -u code/prism/train_prism_quad.py \
        --split data/v17_splits/split_s17_prism.csv \
        --manifest data/v17_splits/manifest_s17_prism.csv \
        --epochs-s1 3 --epochs-s2 2 \
        --out results/prism/prism_quad_s17.json

    launch_job_if_needed "prism_vit6b_s16" env CUDA_VISIBLE_DEVICES=2 python3 -u code/prism/train_prism_vit6b.py \
        --split data/resplits/split_s16.csv \
        --epochs 4 --batch 2 --grad-accum 4 \
        --out results/prism/prism_vit6b_s16.json

    launch_job_if_needed "prism_vit6b_s17" env CUDA_VISIBLE_DEVICES=3 python3 -u code/prism/train_prism_vit6b.py \
        --split data/resplits/split_s17.csv \
        --epochs 4 --batch 2 --grad-accum 4 \
        --out results/prism/prism_vit6b_s17.json

    wait_and_verify_jobs

    JOB_NAMES=()
    JOB_PIDS=()
    JOB_NPZS=()
    JOB_LOGS=()
    if [ "${FORCE_RETRAIN}" -eq 1 ] || [ ! -s "results/prism/prism_vlm8b_s16_probs.npz" ]; then
        torchrun --nproc_per_node=4 --master_port=29518 code/prism/train_prism_vlm8b.py \
            --split data/resplits/split_s16.csv --epochs 3 --batch 6 --out results/prism/prism_vlm8b_s16.json
    fi
    if [ "${FORCE_RETRAIN}" -eq 1 ] || [ ! -s "results/prism/prism_vlm8b_s17_probs.npz" ]; then
        torchrun --nproc_per_node=4 --master_port=29519 code/prism/train_prism_vlm8b.py \
            --split data/resplits/split_s17.csv --epochs 3 --batch 6 --out results/prism/prism_vlm8b_s17.json
    fi
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
