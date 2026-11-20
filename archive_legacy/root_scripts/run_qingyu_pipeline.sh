#!/usr/bin/env bash
# ==============================================================================
# 晴羽观感体系 (命题18) 全链路工程一键执行主控脚本 (支持断点智能续跑!)
# 支持模式:
#   bash run_qingyu_pipeline.sh --stage1-probe           # 第一阶段: 零重训纯真实资产止血探针 + 8折无泄漏划分
#   bash run_qingyu_pipeline.sh --stage2-tier12          # 第二阶段: 双擎物理预计算 + Tier-1(5主干) & Tier-2(6B) 8卡8折并行训练
#   bash run_qingyu_pipeline.sh --stage3-tier3-distill   # 第三阶段: Tier-3(Clef-9B & Qwen3VL-8B) 8折 + 双向跨模态OOF互蒸馏 + DFW融合
#   bash run_qingyu_pipeline.sh --stage4-soup-deploy     # 第四阶段: 8折 Model Soup + 单卡4090推理/归因 + 视频聚合验证 + 终评双射硬门禁
#   bash run_qingyu_pipeline.sh --all                    # 端到端顺序执行全部四个阶段 (自动跳过已完成模块与单折!)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${SCRIPT_DIR}/code:${PYTHONPATH:-}"

MODE="${1:---all}"
EXTRA_TEST_TEMPLATE="${EXTRA_TEST_TEMPLATE:-}"
EXTRA_TEST_DIR="${EXTRA_TEST_DIR:-}"
NUM_GPUS="${NUM_GPUS:-8}"

mkdir -p logs/qingyu results/qingyu results/final checkpoints/qingyu/soup_4090

log_banner() {
    echo "===================================================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "===================================================================================================="
}

# 并行在最多 NUM_GPUS 张卡上执行尚未完成的折，严格检查每个子进程退出码
# 参数: $1=task_title, $2=subdir_in_results, $3=force_rerun("0"或"1"), 其余为命令行
run_8folds_parallel() {
    local task_title="$1"
    local sub_dir="$2"
    local force_rerun="$3"
    shift 3
    local base_cmd=("$@")

    local pending_folds=()
    for ((f = 0; f < 8; f++)); do
        if [ "${force_rerun}" = "0" ] && [ -f "results/qingyu/${sub_dir}/fold_${f}_probs.npz" ]; then
            echo "  [Resume Skip] ${task_title} Fold ${f} 已存在 results/qingyu/${sub_dir}/fold_${f}_probs.npz，自动跳过。"
        else
            pending_folds+=("${f}")
        fi
    done

    if [ "${#pending_folds[@]}" -eq 0 ]; then
        echo "  [Resume OK] ${task_title} 全部 8 折产物已就绪!"
        return 0
    fi

    log_banner "启动 8 折并行训练: ${task_title} (待跑折数=${#pending_folds[@]}, GPUs=${NUM_GPUS})"
    local idx_ptr=0
    local total_pending="${#pending_folds[@]}"
    while [ "${idx_ptr}" -lt "${total_pending}" ]; do
        local pids=()
        local batch_folds=()
        for ((g = 0; g < NUM_GPUS && idx_ptr < total_pending; g++, idx_ptr++)); do
            local fold="${pending_folds[$idx_ptr]}"
            local log_file="logs/qingyu/${task_title}_fold${fold}.log"
            echo "  -> [Fold ${fold}] GPU ${g} | log: ${log_file}"
            "${base_cmd[@]}" --fold "${fold}" --gpu "${g}" > "${log_file}" 2>&1 &
            pids+=("$!")
            batch_folds+=("${fold}")
        done

        for idx in "${!pids[@]}"; do
            local pid="${pids[$idx]}"
            local f_id="${batch_folds[$idx]}"
            if ! wait "${pid}"; then
                echo "[FATAL] ${task_title} Fold ${f_id} 执行失败! 请查看日志: logs/qingyu/${task_title}_fold${f_id}.log" >&2
                tail -n 45 "logs/qingyu/${task_title}_fold${f_id}.log" >&2 || true
                exit 1
            fi
        done
    done
    echo "  [OK] ${task_title} 全部 8 折训练与验证推理完成!"
}

run_stage1_probe() {
    log_banner "Stage 1: 参数池预算审计 + DINOv3 组约束 8 折分层划分 + 零重训纯真实止血探针"
    python3 -u code/qingyu/config_and_paths.py
    if [ ! -f "data/qingyu_splits/qingyu_8fold.csv" ]; then
        python3 -u code/qingyu/split_8fold_stratified.py
    else
        echo "[Stage 1] 已存在 data/qingyu_splits/qingyu_8fold.csv，跳过重复划分。"
    fi
    python3 -u code/qingyu/stage1_pure_real_probe.py --out-tag qingyu_stage1_stopbleed
    if [ -f "results/final/pred_nriqa_dataset_test_qingyu_stage1_stopbleed.json" ]; then
        python3 -u code/qingyu/validate_submission.py \
            --pred "results/final/pred_nriqa_dataset_test_qingyu_stage1_stopbleed.json" \
            --template "results/v2_full/template_150.json"
    fi
}

run_stage2_tier12() {
    log_banner "Stage 2A: 双擎 1:1 物理算子 (MEOR / M_UI / Depth-Anything S_fg & A_cutout) 多卡并行预计算"
    if [ ! -f "data/qingyu_splits/qingyu_8fold.csv" ]; then
        python3 -u code/qingyu/split_8fold_stratified.py
    fi

    if [ -f "results/qingyu/dual_engine_meta.npz" ]; then
        echo "[Resume Skip] 已存在 results/qingyu/dual_engine_meta.npz，直接复用。"
    else
        local pre_pids=()
        for ((r = 0; r < NUM_GPUS; r++)); do
            python3 -u code/qingyu/dual_engine_preprocess.py \
                --rank "${r}" \
                --world-size "${NUM_GPUS}" \
                --extra-template "${EXTRA_TEST_TEMPLATE}" \
                --extra-test-dir "${EXTRA_TEST_DIR}" \
                --out "results/qingyu/dual_engine_meta.npz" \
                > "logs/qingyu/dual_engine_shard${r}.log" 2>&1 &
            pre_pids+=("$!")
        done
        for pid in "${pre_pids[@]}"; do
            wait "${pid}"
        done
        python3 -u code/qingyu/dual_engine_preprocess.py \
            --merge-shards "${NUM_GPUS}" \
            --out "results/qingyu/dual_engine_meta.npz"
    fi

    log_banner "Stage 2B: 第一阶 5 个轻量与原生比例主干 (1.96B) 8 卡 8 折并行纯真实训练"
    for bname in convnextv2_base pe_core_large dinov3_vitl16 siglip2_so400m_naflex cradio_v4_h; do
        if [ -f "results/qingyu/tier1_${bname}_8fold_oof.npz" ]; then
            echo "[Resume Skip] 已存在 results/qingyu/tier1_${bname}_8fold_oof.npz，跳过 ${bname}。"
        else
            run_8folds_parallel "tier1_${bname}" "${bname}" "0" \
                python3 -u code/qingyu/train_tier1_5backbones_8fold.py \
                --backbone "${bname}" \
                --extra-test-template "${EXTRA_TEST_TEMPLATE}" \
                --extra-test-dir "${EXTRA_TEST_DIR}"
            python3 -u code/qingyu/train_tier1_5backbones_8fold.py \
                --backbone "${bname}" \
                --aggregate-8fold
        fi
    done

    # 先对已完成的 5 座第一阶新塔执行一次快速 OOF 阶段快照审计
    python3 -u code/qingyu/evaluate_and_fuse_qingyu.py \
        --out-tag "qingyu_stage2b_tier1_5towers" \
        --export-oof-teacher ""

    log_banner "Stage 2C: 第二阶 InternViT-6B-448px-V2.5 (金字塔 + 门控 ABMIL + 梯度检查点) 8 卡 8 折并行训练"
    if [ -f "results/qingyu/tier2_internvit_6b_8fold_oof.npz" ]; then
        echo "[Resume Skip] 已存在 results/qingyu/tier2_internvit_6b_8fold_oof.npz，跳过 InternViT-6B。"
    else
        run_8folds_parallel "tier2_internvit_6b" "internvit_6b" "0" \
            python3 -u code/qingyu/train_tier2_internvit6b_8fold.py \
            --batch-size 4 --grad-accum 3 \
            --extra-test-template "${EXTRA_TEST_TEMPLATE}" \
            --extra-test-dir "${EXTRA_TEST_DIR}"
        python3 -u code/qingyu/train_tier2_internvit6b_8fold.py --aggregate-8fold
    fi
}

run_stage3_tier3_distill() {
    log_banner "Stage 3A: 第三阶 Clef-Flash-9B (System-1 联合模式头) & Qwen3-VL-8B (三图 MRoPE) 8 折并行微调"
    if [ -f "results/qingyu/tier3_clef_flash_9b_8fold_oof.npz" ]; then
        echo "[Resume Skip] 已存在 results/qingyu/tier3_clef_flash_9b_8fold_oof.npz，跳过首轮 Clef-Flash-9B。"
    else
        run_8folds_parallel "tier3_clef_flash_9b" "clef_flash_9b" "0" \
            python3 -u code/qingyu/train_tier3_clef_flash9b_8fold.py \
            --batch-size 3 --grad-accum 3 \
            --extra-test-template "${EXTRA_TEST_TEMPLATE}" \
            --extra-test-dir "${EXTRA_TEST_DIR}"
        python3 -u code/qingyu/train_tier3_clef_flash9b_8fold.py --aggregate-8fold
    fi

    if [ -f "results/qingyu/tier3_qwen3_vl_8b_8fold_oof.npz" ]; then
        echo "[Resume Skip] 已存在 results/qingyu/tier3_qwen3_vl_8b_8fold_oof.npz，跳过 Qwen3-VL-8B。"
    else
        run_8folds_parallel "tier3_qwen3_vl_8b" "qwen3_vl_8b" "0" \
            python3 -u code/qingyu/train_tier3_qwen3vl8b_8fold.py \
            --batch-size 3 --grad-accum 3 \
            --extra-test-template "${EXTRA_TEST_TEMPLATE}" \
            --extra-test-dir "${EXTRA_TEST_DIR}"
        python3 -u code/qingyu/train_tier3_qwen3vl8b_8fold.py --aggregate-8fold
    fi

    log_banner "Stage 3B: 第一轮三阶 8 塔确定性门控融合 -> 导出严格折外双向跨模态软教师"
    python3 -u code/qingyu/evaluate_and_fuse_qingyu.py \
        --export-oof-teacher "results/qingyu/oof_ensemble_teacher.npz" \
        --out-tag "qingyu_round1_pre_kd"

    log_banner "Stage 3C: 25.28B 预算池内跨模态正交 OOF 互蒸馏精调 (VLM语义 -> Tier1 & 1:1物理 -> Clef-9B)"
    for bname in pe_core_large dinov3_vitl16 cradio_v4_h; do
        if [ -f "results/qingyu/.done_kd_${bname}" ]; then
            echo "[Resume Skip] 已完成跨模态互蒸馏 ${bname}。"
        else
            run_8folds_parallel "kd_tier1_${bname}" "${bname}" "1" \
                python3 -u code/qingyu/train_tier1_5backbones_8fold.py \
                --backbone "${bname}" \
                --oof-teacher-npz "results/qingyu/oof_tier3_semantic_teacher.npz" \
                --lambda-kd 0.25 \
                --extra-test-template "${EXTRA_TEST_TEMPLATE}" \
                --extra-test-dir "${EXTRA_TEST_DIR}"
            python3 -u code/qingyu/train_tier1_5backbones_8fold.py --backbone "${bname}" --aggregate-8fold
            touch "results/qingyu/.done_kd_${bname}"
        fi
    done

    if [ -f "results/qingyu/.done_kd_clef_flash_9b" ]; then
        echo "[Resume Skip] 已完成跨模态互蒸馏 clef_flash_9b。"
    else
        run_8folds_parallel "kd_tier3_clef_flash_9b" "clef_flash_9b" "1" \
            python3 -u code/qingyu/train_tier3_clef_flash9b_8fold.py \
            --batch-size 3 --grad-accum 3 \
            --oof-teacher-npz "results/qingyu/oof_tier12_physical_teacher.npz" \
            --lambda-kd 0.25 \
            --extra-test-template "${EXTRA_TEST_TEMPLATE}" \
            --extra-test-dir "${EXTRA_TEST_DIR}"
        python3 -u code/qingyu/train_tier3_clef_flash9b_8fold.py --aggregate-8fold
        touch "results/qingyu/.done_kd_clef_flash_9b"
    fi

    log_banner "Stage 3D: 终版三阶异构确定性门控 Stacking (C=0.10) + 30步阻尼 Frank-Wolfe 贝叶斯解码"
    python3 -u code/qingyu/evaluate_and_fuse_qingyu.py \
        --out-tag "qingyu_full_56plus"

    python3 -u code/qingyu/validate_submission.py \
        --pred "results/final/pred_nriqa_dataset_test_qingyu_full_56plus.json" \
        --template "results/v2_full/template_150.json"
}

run_stage4_soup_deploy() {
    log_banner "Stage 4: 8 折参数空间 Model Soup 融合 + 单卡 4090 全常驻推理/归因 + 视频迟滞聚合验证 + 双射硬门禁"
    python3 -u code/qingyu/model_soup_and_quant.py \
        --ckpt-dir "checkpoints/qingyu" \
        --out-dir "checkpoints/qingyu/soup_4090"

    local target_tpl="${EXTRA_TEST_TEMPLATE:-results/v2_full/template_150.json}"
    local out_run1="results/final/pred_nriqa_dataset_4090_soup.json"
    local out_run2="results/final/pred_nriqa_dataset_4090_soup_run2_verify.json"

    python3 -u code/qingyu/deploy_4090_qingyu_engine.py \
        --test-template "${target_tpl}" \
        --test-dir "${EXTRA_TEST_DIR}" \
        --soup-dir "checkpoints/qingyu/soup_4090" \
        --out-json "${out_run1}" \
        --out-attrib-json "results/final/pred_nriqa_dataset_4090_grounded_attrib.json"

    python3 -u code/qingyu/deploy_4090_qingyu_engine.py \
        --test-template "${target_tpl}" \
        --test-dir "${EXTRA_TEST_DIR}" \
        --soup-dir "checkpoints/qingyu/soup_4090" \
        --out-json "${out_run2}" \
        --out-attrib-json ""

    python3 -u code/qingyu/validate_submission.py \
        --pred "${out_run1}" \
        --template "${target_tpl}" \
        --compare-run2 "${out_run2}"
    rm -f "${out_run2}"

    python3 -u code/video_aggregator.py --run-mc-benchmark
}

case "${MODE}" in
    --stage1-probe)
        run_stage1_probe
        ;;
    --stage2-tier12)
        run_stage2_tier12
        ;;
    --stage3-tier3-distill)
        run_stage3_tier3_distill
        ;;
    --stage4-soup-deploy)
        run_stage4_soup_deploy
        ;;
    --all)
        run_stage1_probe
        run_stage2_tier12
        run_stage3_tier3_distill
        run_stage4_soup_deploy
        ;;
    *)
        echo "用法: bash run_qingyu_pipeline.sh [--stage1-probe | --stage2-tier12 | --stage3-tier3-distill | --stage4-soup-deploy | --all]" >&2
        exit 2
        ;;
esac
