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
# Env: NUM_GPUS (8)  GPU_OFFSET (0)  BACKBONE (pe_core_large)  MIN_DELTA (0.30)
set -euo pipefail
cd "$(dirname "$0")"
NUM_GPUS="${NUM_GPUS:-8}"
BB="${BACKBONE:-pe_core_large}"
MIN_DELTA="${MIN_DELTA:-0.30}"
GPU_OFFSET="${GPU_OFFSET:-0}"   # first GPU index to use (share the box: e.g. NUM_GPUS=4 GPU_OFFSET=3 -> GPUs 3-6)
LOG=logs/r55
TOW=results/next55/towers
mkdir -p "${LOG}"

preset() {  # experiment name -> trainer flags
  case "$1" in
    rank)     echo "--aux-kinds rank" ;;                         # E1 ordering only (模糊/纹理损失/过锐)
    rank25)   echo "--aux-kinds rank --lambda-tuple 0.25" ;;     # E1 single allowed retry at half weight
    rankkd)   echo "--aux-kinds rank --kd-synth" ;;              # E2 + fold-teacher soft labels on the ladders
    og)       echo "--aux-kinds overgen --kd-synth" ;;           # E3 generative 过度生成 ladder
    ext)      echo "--aux-kinds ext" ;;                          # E4 self-training on external frames
    clean)    echo "--clean-neg" ;;                              # E5 certified-clean negatives
    tag)      echo "--lambda-tag 0.2" ;;                         # E6 train-only pipeline-tag head
    ctrl)     echo "" ;;                                         # control: no flags, same trainer and GPU sharing as the screens
    fixed)    echo "--select fixed" ;;
    nT)       echo "--select fixed --noise-alpha 0.5" ;;              # audit 2026-11-14 P4a: sibling grade-transition soft targets
    radj)     echo "--select fixed --rank-adj-w 1.0" ;;               # P4b: no extra weight on adjacent-grade ranking pairs
    rsame)    echo "--select fixed --rank-same-group" ;;              # P4b: rank frames of the same video too
    etaps)    echo "--select fixed --pe-taps 7,13,19" ;;              # P6: earlier PE-Core taps
    lbox)     echo "--select fixed --global-letterbox" ;;             # P6: aspect-preserving global view                           # protocol control: no validation-based checkpoint choice
    rcrop)    echo "--select fixed --crop-rand 0.5 --crop-jitter 96" ;;   # random native-crop augmentation (compare with `fixed`)
    halfA)    echo "--select fixed --half A --save-pool" ;;        # teacher / learning-curve point on half A of each pool
    halfB)    echo "--select fixed --half B --save-pool" ;;        # teacher on the other half
    ssl)      echo "--select fixed --half A --ssl --aux-singles 12 --kd-a-dirs ${TOW}/r55_halfA_s2026/${BB} ${TOW}/r55_halfA_s2027/${BB}" ;;   # labelled A + unlabelled B (teacher = halfA models)
    xd)       echo "--select fixed --lambda-kd-lab 1.0 --kd-a-dirs ${TOW}/r55_halfA_s2026/${BB} ${TOW}/r55_halfA_s2027/${BB} --kd-b-dirs ${TOW}/r55_halfB_s2026/${BB} ${TOW}/r55_halfB_s2027/${BB}" ;;   # all labels + cross-fitted soft labels
    boost)    echo "--select fixed --boost-proc-clean 8" ;;        # repeat the 31 clean+processed rows (train-only sampling)
    kvq)      echo "--select fixed --aux-kinds kvq --aux-singles 8 --aux-manifest data/r55_aux/manifest_kvq.csv --aux-meta-npz results/r55/dual_engine_meta_aux_kvq.npz --aux-teacher-npz results/r55/aux_teacher_kvq.npz" ;;   # in-domain self-training
    kvqclean) echo "--select fixed --aux-kinds kvq --aux-singles 8 --clean-neg --clean-list data/r55_aux/kvq_clean.csv --aux-manifest data/r55_aux/manifest_kvq.csv --aux-meta-npz results/r55/dual_engine_meta_aux_kvq.npz --aux-teacher-npz results/r55/aux_teacher_kvq.npz" ;;
    shift)    echo "--select fixed --split-dir data/r55_tagsplit" ;;   # shift test: each fold holds out a whole pipeline / source family (folds 0-6, reporting only)
    shiftrc)  echo "--select fixed --split-dir data/r55_tagsplit --crop-rand 0.5 --crop-jitter 96" ;;   # same folds, random-crop recipe
    kvqcert)  echo "--select fixed --aux-kinds kvq --aux-singles 8 --clean-neg --clean-list data/r55_aux/kvq_clean.csv --clean-cert 0.85 --aux-manifest data/r55_aux/manifest_kvq.csv --aux-meta-npz results/r55/dual_engine_meta_aux_kvq.npz --aux-teacher-npz results/r55/aux_teacher_kvq.npz" ;;   # kvq self-training + clean negatives = high-MOS frames that this fold's teacher also calls clean (use this, not kvqclean)
    shiftkvq) echo "--select fixed --split-dir data/r55_tagsplit --aux-kinds kvq --aux-singles 8 --aux-manifest data/r55_aux/manifest_kvq.csv --aux-meta-npz results/r55/dual_engine_meta_aux_kvq.npz --aux-teacher-npz results/r55/aux_teacher_kvq_shift.npz" ;;   # shift folds + KVQ self-training with shift-fold teachers (prep_shift_teacher.py)
    glitch)   echo "--select fixed --aux-kinds glitch --aux-singles 4 --aux-manifest data/r55_aux/manifest_glitch.csv --aux-meta-npz results/r55/dual_engine_meta_aux_glitch.npz --aux-teacher-npz results/r55/aux_teacher_glitch.npz" ;;   # E3: simulated decoder corruption as 花屏 examples (standard folds: no-harm check)
    shiftglitch) echo "--select fixed --split-dir data/r55_tagsplit --aux-kinds glitch --aux-singles 4 --aux-manifest data/r55_aux/manifest_glitch.csv --aux-meta-npz results/r55/dual_engine_meta_aux_glitch.npz --aux-teacher-npz results/r55/aux_teacher_glitch_shift.npz" ;;   # E3 on the shift folds (fold 4 holds out the family with the real 花屏)
    glitch2)  echo "--select fixed --aux-kinds glitch --aux-singles 4 --aux-manifest data/r55_aux/manifest_glitch.csv --aux-meta-npz results/r55/dual_engine_meta_aux_glitch.npz --aux-teacher-npz results/r55/aux_teacher_glitch.npz" ;;   # E3 v2: simulated 花屏 + matched uncorrupted frames (same flags as glitch; the data changed)
    shiftglitch2) echo "--select fixed --split-dir data/r55_tagsplit --aux-kinds glitch --aux-singles 4 --aux-manifest data/r55_aux/manifest_glitch.csv --aux-meta-npz results/r55/dual_engine_meta_aux_glitch.npz --aux-teacher-npz results/r55/aux_teacher_glitch_shift.npz" ;;
    fullglitch) echo "--select fixed --split-dir data/next55_fullfit/qingyu --aux-kinds glitch --aux-singles 4 --aux-manifest data/r55_aux/manifest_glitch.csv --aux-meta-npz results/r55/dual_engine_meta_aux_glitch.npz --aux-teacher-npz results/r55/aux_teacher_glitch_full.npz" ;;   # deliverable: 100 %-data tower with simulated 花屏 + matched negatives (fold 0 only)
    synth-*)  r="${1#synth-}"; echo "--select fixed --aux-kinds synth --aux-singles ${R55_SYNTH_N:-4} --aux-manifest data/r55_aux/manifest_${r}.csv --aux-meta-npz results/r55/dual_engine_meta_aux_${r}.npz --aux-teacher-npz results/r55/aux_teacher_${r}.npz" ;;   # SynthLab recipe <r> on the standard folds
    shiftsynth-*) r="${1#shiftsynth-}"; echo "--select fixed --split-dir data/r55_tagsplit --aux-kinds synth --aux-singles ${R55_SYNTH_N:-4} --aux-manifest data/r55_aux/manifest_${r}_shift.csv --aux-meta-npz results/r55/dual_engine_meta_aux_${r}.npz --aux-teacher-npz results/r55/aux_teacher_${r}_shift.npz" ;;   # the same on the source-held-out folds
    baug)     echo "--select fixed --aug-benign 0.5" ;;        # in-place mild JPEG / tone augmentation (invariance without extra labelled copies)
    shiftbaug) echo "--select fixed --split-dir data/r55_tagsplit --aug-benign 0.5" ;;
    tw-*)     echo "--select fixed --aug-twins ${1#tw-} --aug-twin-p ${R55_TWIN_P:-0.5}" ;;        # label-preserving SynthLab recipe used IN PLACE (no extra rows)
    shifttw-*) echo "--select fixed --split-dir data/r55_tagsplit --aug-twins ${1#shifttw-} --aug-twin-p ${R55_TWIN_P:-0.5}" ;;
    nt)       echo "--select fixed --native-tiles tiles --batch-size 6" ;;          # every pixel at native resolution (owner, 2026-11-13): tiles only
    ntg)      echo "--select fixed --native-tiles tiles+global --batch-size 6" ;;   # ... plus the squashed whole frame
    hp)       echo "--select fixed --input-hp 2.0 --hp-gain 2.0" ;;            # forensic tower: high-pass residual input (screen of 2026-11-13)
    origonly) echo "--select fixed --train-batch orig" ;;        # release-batch study: train on the first batch only
    supponly) echo "--select fixed --train-batch supp" ;;        # ... on the supplement only
    sibxA)    echo "--select fixed --split-dir data/r55_sibsplit_A" ;;   # sibling-exposed folds: one frame of a multi-frame video is validated, its other frames are trained on
    sibxB)    echo "--select fixed --split-dir data/r55_sibsplit_B" ;;   # same, another frame held out
    combo)    echo "${R55_COMBO_FLAGS:?export R55_COMBO_FLAGS with the union of the flags that PASSED}" ;;
    *) echo "unknown preset $1" >&2; exit 2 ;;
  esac
}

run_folds() {  # run_folds <logprefix> <folds...> -- <cmd...>   (one job per GPU, round robin)
  local prefix="$1"; shift
  local folds=(); while [ "$1" != "--" ]; do folds+=("$1"); shift; done; shift
  local pids=() i=0
  for f in "${folds[@]}"; do
    "$@" --fold "${f}" --gpu "$(( GPU_OFFSET + i % NUM_GPUS ))" > "${LOG}/${prefix}_f${f}.log" 2>&1 &
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
[ "${step}" = "preset" ] || df -h . | tail -1
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
  preset)
    preset "${1:?name}"          # prints the trainer flags of a preset (used by code/r55/queue_run.py)
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
