#!/usr/bin/env bash
set -euo pipefail

REPO="/wanqing-develop/luowenjing/18"
cd "$REPO"
mkdir -p logs/r55/wave3

check_disk() {
  local avail_gb
  avail_gb=$(df -BG /wanqing-develop | awk 'NR==2 {gsub("G","",$4); print $4}')
  echo "[Disk Guard] /wanqing-develop available: ${avail_gb}G"
  if [ "$avail_gb" -lt 80 ]; then
    echo "[ABORT] Free disk < 80G (${avail_gb}G)!" >&2
    exit 1
  fi
}

run_8fold_kd() {
  local tag="$1"
  local bb="$2"
  local ep="$3"
  local bs="$4"
  local ga="$5"
  local lrb="$6"
  local lrh="$7"
  local unfreeze="$8"
  local lkd="$9"
  local out_dir="results/next55/towers/${tag}"

  check_disk
  echo "=== [Wave 3] Launching 8-fold parallel training: ${tag} (${bb}, E=${ep}, bs=${bs}x${ga}, unfreeze=${unfreeze}, kd=${lkd}) ==="
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
      --unfreeze-blocks "$unfreeze" \
      --nested-kd-npz results/r55/nested_kd_teacher.npz \
      --lambda-kd "$lkd" \
      --select fixed \
      --no-save-ckpt \
      > "logs/r55/wave3/${tag}_${bb}_f${f}.log" 2>&1 &
    pids+=($!)
  done

  for pid in "${pids[@]:-}"; do
    if [ -n "$pid" ]; then
      wait "$pid"
    fi
  done
  echo "=== [Wave 3] Finished 8-fold: ${tag} (${bb}) ==="
}

# 3A) PE-Core-L14 (0.35B) + non-letterbox fix + 5 Semantic ROIs + Routed Head + BoundaryMarginXBM + Leak-Free Nested KD
run_8fold_kd "r55_p12_kd_e5_s2026" "pe_core_large" 5 12 1 4e-5 5e-4 9 0.35
python3 -u code/r55/eval_specialist_gate.py \
  --cand "p12_pe=results/next55/towers/r55_p12_e5_s2026/pe_core_large" \
  --cand "p12_pe_kd=results/next55/towers/r55_p12_kd_e5_s2026/pe_core_large" \
  | tee logs/r55/wave3/gate_after_pe_kd.log

# 3B) PE-Core-G14-448 (1.88B Native-448 Giant Vision Specialist, 3.55 GB bf16) + Pretrained MAP + ABMIL + 5 Semantic ROIs + Leak-Free Nested KD
run_8fold_kd "r55_p12_kd_e4_s2026" "pe_core_g14" 4 12 1 2.5e-5 4e-4 12 0.35
python3 -u code/r55/eval_specialist_gate.py \
  --cand "p12_pe_kd=results/next55/towers/r55_p12_kd_e5_s2026/pe_core_large" \
  --cand "p12_pe_g14_kd=results/next55/towers/r55_p12_kd_e4_s2026/pe_core_g14" \
  | tee logs/r55/wave3/gate_after_pe_g14_kd.log
