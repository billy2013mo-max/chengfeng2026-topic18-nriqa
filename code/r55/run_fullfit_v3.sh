#!/usr/bin/env bash
set -euo pipefail

REPO="/wanqing-develop/luowenjing/18"
cd "$REPO"
mkdir -p logs/r55/fullfit_v3

avail_gb=$(df -BG /wanqing-develop | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "[Disk Guard] /wanqing-develop available: ${avail_gb}G"
if [ "$avail_gb" -lt 80 ]; then
  echo "[ABORT] Free disk < 80G (${avail_gb}G)!" >&2
  exit 1
fi

pids=()

# 1) C-RADIOv4-H fixed-E4 SWA (100% fullfit)
CUDA_VISIBLE_DEVICES=0 python3 -u code/r55/train_testbed.py \
  --backbone cradio_v4_h --fold 0 --gpu 0 --seed 2026 \
  --split-dir data/next55_fullfit/qingyu \
  --out-dir results/r55/fullfit_v3_tier1 \
  --dual-engine-npz results/qingyu/dual_engine_meta.npz \
  --epochs 4 --select fixed \
  > logs/r55/fullfit_v3/tier1_cradio_e4.log 2>&1 &
pids+=($!)

# 2) PE-Core-L14 Tier-1 fixed-E8 SWA (100% fullfit)
CUDA_VISIBLE_DEVICES=1 python3 -u code/r55/train_testbed.py \
  --backbone pe_core_large --fold 0 --gpu 0 --seed 2026 \
  --split-dir data/next55_fullfit/qingyu \
  --out-dir results/r55/fullfit_v3_tier1 \
  --dual-engine-npz results/qingyu/dual_engine_meta.npz \
  --epochs 8 --select fixed \
  > logs/r55/fullfit_v3/tier1_pe_e8.log 2>&1 &
pids+=($!)

# 3) SigLIP2-SO400M-NaFlex Tier-1 fixed-E8 SWA (100% fullfit)
CUDA_VISIBLE_DEVICES=2 python3 -u code/r55/train_testbed.py \
  --backbone siglip2_so400m_naflex --fold 0 --gpu 0 --seed 2026 \
  --split-dir data/next55_fullfit/qingyu \
  --out-dir results/r55/fullfit_v3_tier1 \
  --dual-engine-npz results/qingyu/dual_engine_meta.npz \
  --epochs 8 --select fixed \
  > logs/r55/fullfit_v3/tier1_sig_e8.log 2>&1 &
pids+=($!)

# 4) P12 Specialist: PE-Core-L14 + 5 Semantic ROIs + RoutedOrdinalHead + BoundaryMarginXBM + Out-of-Sample KD (fixed-E5 SWA, 100% fullfit)
CUDA_VISIBLE_DEVICES=3 python3 -u code/r55/train_p12_specialist.py \
  --backbone pe_core_large --fold 0 --gpu 0 --seed 2026 \
  --split-dir data/next55_fullfit/qingyu \
  --out-dir results/r55/fullfit_v3_p12 \
  --dual-engine-npz results/r55/semantic_roi_meta.npz \
  --epochs 5 --batch-size 12 --lr-backbone 4e-5 --lr-head 5e-4 --unfreeze-blocks 9 \
  --nested-kd-npz results/r55/fullfit_halfAB_kd_teacher.npz --lambda-kd 0.35 \
  --select fixed \
  > logs/r55/fullfit_v3/p12_pe_kd_e5.log 2>&1 &
pids+=($!)

# 5) P12 Giant Specialist: PE-Core-G14-448 (1.88B Native-448) + MAP + ABMIL + 5 Semantic ROIs + Out-of-Sample KD (fixed-E4 SWA, 100% fullfit)
CUDA_VISIBLE_DEVICES=4 python3 -u code/r55/train_p12_specialist.py \
  --backbone pe_core_g14 --fold 0 --gpu 0 --seed 2026 \
  --split-dir data/next55_fullfit/qingyu \
  --out-dir results/r55/fullfit_v3_p12 \
  --dual-engine-npz results/r55/semantic_roi_meta.npz \
  --epochs 4 --batch-size 12 --lr-backbone 2.5e-5 --lr-head 4e-4 --unfreeze-blocks 12 \
  --nested-kd-npz results/r55/fullfit_halfAB_kd_teacher.npz --lambda-kd 0.35 \
  --select fixed \
  > logs/r55/fullfit_v3/p12_pe_g14_kd_e4.log 2>&1 &
pids+=($!)

echo "[FullFit-v3] Launched 5 parallel 100%-fullfit jobs on GPUs 0..4: ${pids[*]}"
for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "[FullFit-v3] ALL 5 DELIVERABLE CHECKPOINTS COMPLETED SUCCESSFULLY!"
ls -lh results/r55/fullfit_v3_tier1/*/*.pt results/r55/fullfit_v3_p12/*/*.pt
