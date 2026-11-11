#!/usr/bin/env bash
# ==============================================================================
# next55 — step-by-step runner.  See NEXT55_HANDOFF.md for gates / expected numbers.
#   bash run_next55.sh <step> [args]
#
#   diag                         CPU   OOF diagnostics + baseline stack           (~5 min)
#   smoke_iv6b                   1 GPU InternViT-6B OOM-fix check                  (~3 min)
#   tier1_seedbag <seed> [bb..]  8 GPU 8-fold Tier-1 re-runs with a new seed       (~35-45 min / backbone)
#   v16_seedbag <seed>           8 GPU 8-fold V16 re-run with a new seed           (~1.6 h)
#   iv6b_8fold                   8 GPU 8-fold InternViT-6B (Tier-2)                (long; see handoff)
#   vlm_smoke                    1 GPU Qwen3-VL-8B Tier-3 fold-0, time-boxed       (<= 3 h)
#   vlm_8fold                    8 GPU Qwen3-VL-8B Tier-3 all folds
#   stack <tag> [towers..]       CPU   nested fusion/decoder + bootstrap gate      (~2-10 min)
#   t150 <tag>                   CPU   test150 submission from 8-fold models (research A/B only)
#   fullfit <tag>                8 GPU 100%-data refit of the chosen towers (shippable model)
#   final <tag>                  GPU   final-set inference with the full-fit models + submission
# ==============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NUM_GPUS="${NUM_GPUS:-8}"
FINAL_TEMPLATE="${FINAL_TEMPLATE:-}"
FINAL_IMG_DIR="${FINAL_IMG_DIR:-}"
DUAL_NPZ="${NEXT55_DUAL_NPZ:-results/qingyu/dual_engine_meta.npz}"
LOG=logs/next55
OBJECTIVE="${NEXT55_OBJECTIVE:-doc2_strict_total}"
mkdir -p "${LOG}" results/next55

banner() { echo "=================================================================="; echo "[$(date '+%F %T')] $*"; echo "=================================================================="; }

# 8 folds on up to NUM_GPUS GPUs; aborts on the first failing fold (never swallows errors)
run_8folds() {
  local title="$1"; shift
  local fold=0
  while [ "${fold}" -lt 8 ]; do
    local pids=() fl=()
    for ((g = 0; g < NUM_GPUS && fold < 8; g++, fold++)); do
      "$@" --fold "${fold}" --gpu "${g}" > "${LOG}/${title}_fold${fold}.log" 2>&1 &
      pids+=("$!"); fl+=("${fold}")
    done
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        echo "[FATAL] ${title} fold ${fl[$i]} failed -> ${LOG}/${title}_fold${fl[$i]}.log" >&2
        tail -n 30 "${LOG}/${title}_fold${fl[$i]}.log" >&2 || true
        exit 1
      fi
    done
  done
  echo "[OK] ${title}: 8 folds done"
}

step="${1:-}"; shift || true
case "${step}" in
  diag)
    banner "diag: per-tower + ensemble OOF (nested decoder), objective ${OBJECTIVE}"
    python3 -u code/next55/diag_oof.py --no-combos --objective "${OBJECTIVE}" | tee "${LOG}/diag_oof.log"
    python3 -u code/next55/stack55.py --tag s0 --fusions mean --objective "${OBJECTIVE}" | tee "${LOG}/stack_s0.log"
    ;;
  smoke_iv6b)
    CUDA_VISIBLE_DEVICES=0 python3 -u code/next55/smoke_iv6b_mem.py | tee "${LOG}/smoke_iv6b.log"
    ;;
  tier1_seedbag)
    seed="${1:?seed}"; shift
    bbs=("$@"); [ "${#bbs[@]}" -eq 0 ] && bbs=(pe_core_large cradio_v4_h siglip2_so400m_naflex)
    for bb in "${bbs[@]}"; do
      banner "tier1_seedbag seed=${seed} backbone=${bb}"
      run_8folds "qy_s${seed}_${bb}" python3 -u code/qingyu/train_tier1_5backbones_8fold.py \
        --backbone "${bb}" --seed "${seed}" --out-dir "results/next55/towers/qy_s${seed}" \
        --dual-engine-npz "${DUAL_NPZ}" --extra-test-template "${FINAL_TEMPLATE}" --extra-test-dir "${FINAL_IMG_DIR}"
    done
    ;;
  v16_seedbag)
    seed="${1:?seed}"
    banner "v16_seedbag seed=${seed}"
    python3 -u code/next55/launch_v16_seedbag.py --seed "${seed}" | tee "${LOG}/v16_seedbag_s${seed}.log"
    ;;
  iv6b_8fold)
    banner "InternViT-6B 8-fold (grad-ckpt fix, QY_IV6B_GRAD_CKPT=1)"
    QY_IV6B_GRAD_CKPT=1 run_8folds "iv6b" python3 -u code/qingyu/train_tier2_internvit6b_8fold.py \
      --out-dir results/next55/towers/iv6b --dual-engine-npz "${DUAL_NPZ}" \
      --extra-test-template "${FINAL_TEMPLATE}" --extra-test-dir "${FINAL_IMG_DIR}"
    ;;
  vlm_smoke)
    banner "Qwen3-VL-8B fold 0 (time-boxed 3h)"
    timeout 3h python3 -u code/qingyu/train_tier3_qwen3vl8b_8fold.py --fold 0 --gpu 0 \
      --out-dir results/next55/towers/vlm --dual-engine-npz "${DUAL_NPZ}" 2>&1 | tee "${LOG}/vlm_smoke_fold0.log"
    ;;
  vlm_8fold)
    run_8folds "vlm" python3 -u code/qingyu/train_tier3_qwen3vl8b_8fold.py \
      --out-dir results/next55/towers/vlm --dual-engine-npz "${DUAL_NPZ}" \
      --extra-test-template "${FINAL_TEMPLATE}" --extra-test-dir "${FINAL_IMG_DIR}"
    ;;
  stack)
    tag="${1:?tag}"; shift
    if [ "$#" -gt 0 ]; then
      python3 -u code/next55/stack55.py --tag "${tag}" --objective "${OBJECTIVE}" --towers "$@" | tee "${LOG}/stack_${tag}.log"
    else
      python3 -u code/next55/stack55.py --tag "${tag}" --objective "${OBJECTIVE}" | tee "${LOG}/stack_${tag}.log"
    fi
    ;;
  t150)
    tag="${1:?tag}"
    python3 -u code/next55/build_submission55.py --stack-tag "${tag}" --target t150 --deploy 8fold \
      --template results/v2_full/template_150.json --out "results/final/pred_nriqa_dataset_test_next55_${tag}.json"
    ;;
  fullfit)
    tag="${1:?tag}"
    python3 -u code/next55/make_fullfit_splits.py
    python3 -u code/next55/launch_fullfit.py --stack-tag "${tag}" | tee "${LOG}/fullfit_${tag}.log"
    ;;
  final)
    tag="${1:?tag}"
    : "${FINAL_TEMPLATE:?export FINAL_TEMPLATE (run prep_final_set.py first)}"
    : "${FINAL_IMG_DIR:?export FINAL_IMG_DIR}"
    final_tag="${FINAL_TAG:-final654}"
    banner "final-set inference for full-fit towers of stack ${tag}"
    python3 - "${tag}" <<'EOF' > "${LOG}/final_jobs_${tag}.txt"
import os, pickle, sys
sys.path.insert(0, "code/next55")
from common55 import OUT_DIR, fullfit_dir
for t in pickle.load(open(os.path.join(OUT_DIR, "stack", sys.argv[1], "final.pkl"), "rb"))["towers"]:
    fmt, d = fullfit_dir(t)
    print(fmt, os.path.basename(d.rstrip("/")), d)
EOF
    g=0
    while read -r fmt bb d; do
      if [ "${fmt}" = "v16" ]; then
        CUDA_VISIBLE_DEVICES="${g}" python3 -u code/next55/infer_v16_ckpt.py --tower-dir "${d}" --fold 0 \
          --template "${FINAL_TEMPLATE}" --tag "${final_tag}" --verify-va > "${LOG}/final_${bb}.log" 2>&1 &
      else
        python3 -u code/next55/infer_tier1_ckpt.py --backbone "${bb}" --tower-dir "${d}" --folds 0 \
          --template "${FINAL_TEMPLATE}" --img-dir "${FINAL_IMG_DIR}" --tag "${final_tag}" --gpu "${g}" \
          --dual-engine-npz "${DUAL_NPZ}" --verify-va > "${LOG}/final_${bb}.log" 2>&1 &
      fi
      g=$(( (g + 1) % NUM_GPUS ))
    done < "${LOG}/final_jobs_${tag}.txt"
    wait
    grep -l "Traceback\|AssertionError" "${LOG}"/final_*.log && { echo "[FATAL] final inference failed (see logs above)"; exit 1; } || true
    python3 -u code/next55/build_submission55.py --stack-tag "${tag}" --target "${final_tag}" --deploy fullfit \
      --template "${FINAL_TEMPLATE}" --out "results/final/pred_${final_tag}_next55_${tag}.json"
    ;;
  *)
    sed -n '2,20p' "$0"; exit 2 ;;
esac
