#!/usr/bin/env bash
# run_r55.sh — step runner for the R55 redesign (see R55_HANDOFF.md).  Run from the repo root.
#
#   bash run_r55.sh base <seed>                 # baseline pe_core_large, 8 folds, no auxiliary data
#   bash run_r55.sh noise                       # seed-noise floor from the 3 baseline seeds -> MIN_DELTA
#   bash run_r55.sh lc                          # learning curve: 25/50/75 % of the training groups
#   bash run_r55.sh gen_rank [per_dim]          # classical ranking ladders (CPU)
#   bash run_r55.sh gen_overgen [n_seeds]       # FLUX 过度生成 ladder (8 GPUs, venv_flux)
#   bash run_r55.sh ingest <dir|hf:repo> <tag> [max]   # external unlabelled frames
#   bash run_r55.sh prep                        # template + physics meta + fold-teacher soft labels
#   bash run_r55.sh certify [reject_list]       # certified-clean external frames + contact sheets
#   bash run_r55.sh exp <name> <seed> [folds]   # one experiment preset (see presets below), default 8 folds
#   bash run_r55.sh gate <name> [seeds...]      # paired gate vs the baseline seeds (default seeds 2026 2027)
#
# Env: NUM_GPUS (8)  BACKBONE (pe_core_large)  MIN_DELTA (0.30)  JOBS_PER_GPU (1)
set -euo pipefail
cd "$(dirname "$0")"
NUM_GPUS="${NUM_GPUS:-8}"
BB="${BACKBONE:-pe_core_large}"
MIN_DELTA="${MIN_DELTA:-0.30}"
LOG=logs/r55
TOW=results/next55/towers
mkdir -p "${LOG}"

preset() {  # experiment name -> trainer flags
  case "$1" in
    rank)     echo "--aux-kinds rank" ;;                         # E1 ordering only (模糊/纹理损失/过锐)
    rankkd)   echo "--aux-kinds rank --kd-synth" ;;              # E2 + fold-teacher soft labels on the ladders
    og)       echo "--aux-kinds overgen --kd-synth" ;;           # E3 generative 过度生成 ladder
    ext)      echo "--aux-kinds ext" ;;                          # E4 self-training on external frames
    clean)    echo "--clean-neg" ;;                              # E5 certified-clean negatives
    tag)      echo "--lambda-tag 0.2" ;;                         # E6 train-only pipeline-tag head
    combo)    echo "${R55_COMBO_FLAGS:?export R55_COMBO_FLAGS with the union of the flags that PASSED}" ;;
    *) echo "unknown preset $1" >&2; exit 2 ;;
  esac
}

run_folds() {  # run_folds <logprefix> <folds...> -- <cmd...>   (one job per GPU, round robin)
  local prefix="$1"; shift
  local folds=(); while [ "$1" != "--" ]; do folds+=("$1"); shift; done; shift
  local pids=() i=0
  for f in "${folds[@]}"; do
    "$@" --fold "${f}" --gpu "$(( i % NUM_GPUS ))" > "${LOG}/${prefix}_f${f}.log" 2>&1 &
    pids+=($!); i=$(( i + 1 ))
  done
  local rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "${rc}" -eq 0 ] || { echo "[run_r55] ${prefix}: at least one fold failed, see ${LOG}/${prefix}_f*.log"; exit 1; }
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
}

base_dir() {  # baseline tower dir for a seed
  case "$1" in
    2026) echo "results/qingyu/${BB}" ;;
    *)    echo "${TOW}/qy_s$1/${BB}" ;;
  esac
}

step="${1:?step}"; shift || true
df -h . | tail -1
case "${step}" in
  base)
    seed="${1:?seed}"
    run_folds "base_s${seed}" 0 1 2 3 4 5 6 7 -- python3 -u code/r55/train_testbed.py --backbone "${BB}" --seed "${seed}" \
      --out-dir "${TOW}/qy_s${seed}"
    ;;
  noise)
    python3 code/r55/gate_r55.py --name noise --noise --base "$(base_dir 2026)" "$(base_dir 2027)" "$(base_dir 2028)"
    ;;
  lc)
    for frac in 0.25 0.50 0.75; do
      run_folds "lc_${frac}" 0 1 2 3 4 5 6 7 -- python3 -u code/r55/train_testbed.py --backbone "${BB}" --seed 2026 \
        --train-frac "${frac}" --out-dir "${TOW}/r55_lc${frac}_s2026"
    done
    for frac in 0.25 0.50 0.75; do
      python3 code/r55/gate_r55.py --name "lc_${frac}" --base "$(base_dir 2026)" --exp "${TOW}/r55_lc${frac}_s2026/${BB}" | grep -E "Doc2S|gate"
    done
    ;;
  gen_rank)
    python3 code/r55/build_rank_tuples.py --per-dim "${1:-500}" --workers 64
    ;;
  gen_overgen)
    n="${1:-400}"; pids=()
    for r in $(seq 0 $(( NUM_GPUS - 1 ))); do
      CUDA_VISIBLE_DEVICES="${r}" venv_flux/bin/python code/r55/build_overgen_ladder.py --rank "${r}" --world "${NUM_GPUS}" \
        --n-seeds "${n}" > "${LOG}/overgen_r${r}.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    python3 code/r55/build_overgen_ladder.py --merge
    ;;
  ingest)
    src="${1:?dir or hf:repo}"; tag="${2:?tag}"; max="${3:-4000}"
    if [[ "${src}" == hf:* ]]; then
      python3 code/r55/ingest_external_pool.py --hf-repo "${src#hf:}" --tag "${tag}" --max "${max}"
    else
      python3 code/r55/ingest_external_pool.py --src "${src}" --tag "${tag}" --max "${max}"
    fi
    ;;
  prep)
    python3 code/r55/prep_aux.py --gpus "${NUM_GPUS}"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
    ;;
  certify)
    python3 code/r55/certify_clean.py ${1:+--reject-list "$1"}
    ;;
  exp)
    name="${1:?name}"; seed="${2:?seed}"; shift 2
    folds=("$@"); [ "${#folds[@]}" -eq 0 ] && folds=(0 1 2 3 4 5 6 7)
    # shellcheck disable=SC2046
    run_folds "exp_${name}_s${seed}" "${folds[@]}" -- python3 -u code/r55/train_testbed.py --backbone "${BB}" --seed "${seed}" \
      --out-dir "${TOW}/r55_${name}_s${seed}" $(preset "${name}")
    ;;
  gate)
    name="${1:?name}"; shift
    seeds=("$@"); [ "${#seeds[@]}" -eq 0 ] && seeds=(2026 2027)
    base=(); exp=()
    for s in "${seeds[@]}"; do base+=("$(base_dir "${s}")"); exp+=("${TOW}/r55_${name}_s${s}/${BB}"); done
    python3 code/r55/gate_r55.py --name "${name}" --min-delta "${MIN_DELTA}" --base "${base[@]}" --exp "${exp[@]}"
    ;;
  *) echo "unknown step ${step}"; exit 2 ;;
esac
