#!/usr/bin/env bash
set -euo pipefail

REPO="/wanqing-develop/luowenjing/18"
cd "$REPO"
mkdir -p logs/r55/wave2b

check_disk() {
  local avail_gb
  avail_gb=$(df -BG /wanqing-develop | awk 'NR==2 {gsub("G","",$4); print $4}')
  echo "[Disk Guard] /wanqing-develop available: ${avail_gb}G"
  if [ "$avail_gb" -lt 80 ]; then
    echo "[ABORT] Free disk < 80G (${avail_gb}G)!" >&2
    exit 1
  fi
}

run_8fold() {
  local tag="$1"
  local bb="$2"
  local ep="$3"
  local bs="$4"
  local ga="$5"
  local lrb="$6"
  local lrh="$7"
  local out_dir="results/next55/towers/${tag}"

  check_disk
  echo "=== [Wave 2B] Launching 8-fold parallel training: ${tag} (${bb}, E=${ep}, bs=${bs}x${ga}) ==="
  pids=()
  for f in 0 1 2 3 4 5 6 7; do
    if [ -f "${out_dir}/${bb}/fold_${f}_probs.npz" ]; then
      echo "  [skip] ${out_dir}/${bb}/fold_${f}_probs.npz already exists"
      continue
    fi
    CUDA_VISIBLE_DEVICES=$f python3 -u code/r55/train_p12_specialist.py \
      --backbone "$bb" \
      --fold "$f" \
      --gpu 0 \
      --seed 2026 \
      --out-dir "$out_dir" \
      --dual-engine-npz results/r55/semantic_roi_meta.npz \
      --epochs "$ep" \
      --batch-size "$bs" \
      --grad-accum "$ga" \
      --lr-backbone "$lrb" \
      --lr-head "$lrh" \
      --select fixed \
      --no-save-ckpt \
      > "logs/r55/wave2b/${tag}_${bb}_f${f}.log" 2>&1 &
    pids+=($!)
  done

  for pid in "${pids[@]:-}"; do
    if [ -n "$pid" ]; then
      wait "$pid"
    fi
  done
  echo "=== [Wave 2B] Finished 8-fold: ${tag} ==="
}

# 1) P12 Semantic-ROI + Routed Head + BoundaryMarginXBM on PE-Core-L14-336 (0.6 GB bf16)
run_8fold "r55_p12_e5_s2026" "pe_core_large" 5 12 1 4e-5 5e-4
python3 -u code/r55/eval_specialist_gate.py \
  --cand "p12_pe=results/next55/towers/r55_p12_e5_s2026/pe_core_large" \
  | tee logs/r55/wave2b/gate_after_pe.log

# 2) P12 Semantic-ROI + Routed Head + BoundaryMarginXBM on C-RADIOv4-H (1.2 GB bf16, optimal E=4)
run_8fold "r55_p12_e4_s2026" "cradio_v4_h" 4 12 1 4e-5 5e-4
python3 -u code/r55/eval_specialist_gate.py \
  --cand "p12_pe=results/next55/towers/r55_p12_e5_s2026/pe_core_large" \
  --cand "p12_cradio=results/next55/towers/r55_p12_e4_s2026/cradio_v4_h" \
  | tee logs/r55/wave2b/gate_after_cradio.log

# 3) P12 Semantic-ROI + Routed Head + BoundaryMarginXBM on Brand-New SigLIP2-Giant 1.1B (2.2 GB bf16, 40L 1536d)
run_8fold "r55_p12_e4_s2026" "siglip2_giant" 4 12 1 3e-5 4e-4
python3 -u code/r55/eval_specialist_gate.py \
  --cand "p12_pe=results/next55/towers/r55_p12_e5_s2026/pe_core_large" \
  --cand "p12_cradio=results/next55/towers/r55_p12_e4_s2026/cradio_v4_h" \
  --cand "p12_siggiant=results/next55/towers/r55_p12_e4_s2026/siglip2_giant" \
  | tee logs/r55/wave2b/gate_after_siggiant.log
