#!/usr/bin/env bash
# code/r55/synth_iter.sh — one SynthLab iteration for one recipe: build -> fidelity -> screen -> verdict in the ledger.
#
#   bash code/r55/synth_iter.sh <recipe> rows  [target defect indices...]   # construction-label recipe, extra rows   (synth-<R>)
#   bash code/r55/synth_iter.sh <recipe> twins [target defect indices...]   # label-preserving recipe, used in place  (tw-<R>)
#   STAGE=confirm bash code/r55/synth_iter.sh <recipe> rows 2              # remaining six folds + the pool gate
#
# Env: GPUS="0 1 2 3" (default all eight), GPU=0 (fidelity), SEED=2026.  Recipes live in code/r55/synth_recipes.py.
# Every verdict is appended to results/r55/synthlab/LEDGER.md; nothing here reads a test image or an online score.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=code OMP_NUM_THREADS=8
R="${1:?recipe}"; MODE="${2:-rows}"; shift 2 || true
SEED="${SEED:-2026}"; STAGE="${STAGE:-screen}"
case "$MODE" in rows) P="synth-" ;; twins) P="tw-" ;; *) echo "mode must be rows or twins" >&2; exit 2 ;; esac
mkdir -p logs/r55/synthlab
if [ ! -f "data/r55_aux/manifest_${R}.csv" ]; then
  python3 code/r55/synth_build.py --recipe "$R" 2>&1 | tee "logs/r55/synthlab/build_${R}.log" | tail -3
fi
if [ "$STAGE" = screen ]; then
  python3 -u code/r55/synth_fidelity.py --recipe "$R" --gpu "${GPU:-0}" 2>&1 | tee "logs/r55/synthlab/fidelity_${R}.log" | grep -a "===\|  " | tail -12
  python3 -u code/r55/queue_run.py --gpus ${GPUS:-0 1 2 3 4 5 6 7} --per-gpu 1 --jobs "${P}${R}:${SEED}:0-1" "shift${P}${R}:${SEED}:0-2" | tail -2
  python3 code/r55/synth_eval.py --recipe "$R" --prefix "$P" --stage screen --seed "$SEED" --targets "$@" 2>&1 | grep -v "^\[bank\]" | tee "logs/r55/synthlab/eval_${P}${R}_screen.log"
else
  python3 -u code/r55/queue_run.py --gpus ${GPUS:-0 1 2 3 4 5 6 7} --per-gpu 1 --jobs "${P}${R}:${SEED}:2-7" "shift${P}${R}:${SEED}:3-6" | tail -2
  python3 code/r55/synth_eval.py --recipe "$R" --prefix "$P" --stage confirm --seed "$SEED" --targets "$@" 2>&1 | grep -v "^\[bank\]" | tee "logs/r55/synthlab/eval_${P}${R}_confirm.log"
  python3 code/r55/synth_eval.py --recipe "$R" --prefix "$P" --stage pool --seed "$SEED" 2>&1 | grep -v "^\[bank\]" | tee "logs/r55/synthlab/eval_${P}${R}_pool.log"
fi
