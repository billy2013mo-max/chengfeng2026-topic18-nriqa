#!/usr/bin/env bash
# ==============================================================================
# run_final_644_breakthrough.sh
# Turnkey 8-GPU Final 644-Image Evaluation Set Inference & Diff Audit vs 53.99:
#   Stage 0: Verify / auto-extract 644 images from zip into /opt/final_eval_644
#            (or /dev/shm/final_eval_644 — zero extra space used on CephFS)
#   Stage 1: Verify OOF-fitted resident_v7_os_og_bundle.pkl (extract 644 spatial
#            features only if bundle.mode requires them; skipped for 'deconf')
#   Stage 2: 8-GPU Data-Parallel FinalEngine Inference on 644 images (~5-7 min),
#            strict N=1 == N=644 check (0/3864 diff), and schema validation
#   Stage 3: Detailed cell/image/Clean/dimension diff vs 53.99 (resident_v6)
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SET_NAME="${SET_NAME:-resident_v7}"
TEMPLATE_644="${TEMPLATE_644:-results/final/template_final_644.json}"
REF_5399_JSON="${REF_5399_JSON:-results/final/pred_nriqa_dataset_final_644_r55_resident_v6_routed_gate_eu.json}"
ZIP_644="${ZIP_644:-/wanqing-develop/luowenjing/总测试集_剔除每个维度10组数据.zip}"
IMG_DIR_644="${IMG_DIR_644:-/opt/final_eval_644}"

BUNDLE_DIR="${BUNDLE_DIR:-results/r55/os_og_breakthrough}"
BUNDLE_PKL="${BUNDLE_PKL:-${BUNDLE_DIR}/${SET_NAME}_os_og_bundle.pkl}"

OUT_DIR="${OUT_DIR:-results/r55/os_og_breakthrough_644}"
SPATIAL_DIR_644="${OUT_DIR}/spatial_cache"
FINAL_644_JSON="${FINAL_644_JSON:-${OUT_DIR}/pred_nriqa_dataset_final_644_os_og_breakthrough.json}"
REUSE_PROBS="${REUSE_PROBS:-0}"

if [[ -z "${GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
  else
    GPUS="0"
  fi
fi
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
NUM_GPUS="${#GPU_ARR[@]}"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

echo "=============================================================================="
echo "[Final 644 Eval] Repo          : ${REPO_DIR}"
echo "[Final 644 Eval] Ensemble      : ${SET_NAME}"
echo "[Final 644 Eval] Active GPUs   : ${GPUS} (${NUM_GPUS} GPUs)"
echo "[Final 644 Eval] Template      : ${TEMPLATE_644}"
echo "[Final 644 Eval] 53.99 Ref JSON: ${REF_5399_JSON}"
echo "[Final 644 Eval] Output JSON   : ${FINAL_644_JSON}"
echo "=============================================================================="
df -h "${REPO_DIR}" | head -n 2 || true

# ------------------------------------------------------------------------------
# Stage 0: Verify or Auto-Extract All 644 Images (Zero CephFS Disk Footprint)
# ------------------------------------------------------------------------------
echo "[Stage 0/3] Verifying 644 evaluation images & template alignment..."
RESOLVED_IMG_DIR="$("${PYTHON_BIN}" - "${TEMPLATE_644}" "${IMG_DIR_644}" "${ZIP_644}" <<'PY'
import json, os, shutil, sys, zipfile

tmpl_p, pref_dir, zip_p = sys.argv[1], sys.argv[2], sys.argv[3]
tmpl = json.load(open(tmpl_p, encoding="utf-8"))
names = [str(x["image"]) for x in tmpl]

def check_dir(d: str) -> bool:
    if not d or not os.path.isdir(d):
        return False
    return all(os.path.isfile(os.path.join(d, nm)) and os.path.getsize(os.path.join(d, nm)) > 0 for nm in names)

cands = [pref_dir, "/opt/final_eval_644", "/dev/shm/final_eval_644", "data/final_eval_644", "data/eval_644"]
for base_d in list(cands):
    if os.path.isdir(base_d):
        for sub in os.listdir(base_d):
            sp = os.path.join(base_d, sub)
            if os.path.isdir(sp):
                cands.append(sp)

for d in cands:
    if check_dir(d):
        print(os.path.abspath(d))
        sys.exit(0)

cand_zips = [zip_p, os.path.join(os.getcwd(), os.path.basename(zip_p))]
src_zip = next((zp for zp in cand_zips if zp and os.path.isfile(zp)), "")
if not src_zip:
    sys.stderr.write(f"[ERROR] 644 images not found in {cands[:4]} and zip not found at {zip_p}\n")
    sys.exit(1)

target_dir = pref_dir
try:
    os.makedirs(target_dir, exist_ok=True)
    test_f = os.path.join(target_dir, ".write_probe")
    open(test_f, "w").close()
    os.remove(test_f)
except Exception:
    target_dir = "/dev/shm/final_eval_644"
    os.makedirs(target_dir, exist_ok=True)

need = set(names)
sys.stderr.write(f"  [extract] Extracting {len(need)} images from {src_zip} -> {target_dir} ...\n")
with zipfile.ZipFile(src_zip, "r") as zf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        raw_name = info.filename
        try:
            dec_name = raw_name.encode("cp437").decode("utf-8")
        except Exception:
            try:
                dec_name = raw_name.encode("cp437").decode("gbk")
            except Exception:
                dec_name = raw_name
        bn = os.path.basename(dec_name)
        if bn not in need:
            bn_raw = os.path.basename(raw_name)
            bn = bn_raw if bn_raw in need else bn
        if bn in need:
            dst = os.path.join(target_dir, bn)
            if not (os.path.isfile(dst) and os.path.getsize(dst) > 0):
                with zf.open(info) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)

missing = [nm for nm in names if not (os.path.isfile(os.path.join(target_dir, nm)) and os.path.getsize(os.path.join(target_dir, nm)) > 0)]
if missing:
    sys.stderr.write(f"[ERROR] Extracted {src_zip} to {target_dir}, still missing {len(missing)}/{len(names)} images: {missing[:5]}\n")
    sys.exit(1)

print(os.path.abspath(target_dir))
PY
)"
echo "  [PASS] Verified all 644 images ready in: ${RESOLVED_IMG_DIR}"

# ------------------------------------------------------------------------------
# Stage 1: Verify Bundle & Conditional Spatial Salience Check
# ------------------------------------------------------------------------------
if [[ ! -f "${BUNDLE_PKL}" ]]; then
  echo "[Stage 1/3] Bundle ${BUNDLE_PKL} not found; fitting on OOF cache..."
  "${PYTHON_BIN}" -u code/r55/deconfound_calibrator.py \
    --set "${SET_NAME}" \
    --spatial-oof "${BUNDLE_DIR}/spatial_cache/spatial_oof_3573.npz" \
    --out-dir "${BUNDLE_DIR}"
fi

BUNDLE_MODE="$("${PYTHON_BIN}" -c "
import sys, pickle
sys.path[:0] = ['code', 'code/next55', 'code/r55', 'code/finetune', 'code/decision_eval']
import deconfound_calibrator, routed_eu_decoder
class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            for mod in (deconfound_calibrator, routed_eu_decoder):
                if hasattr(mod, name):
                    return getattr(mod, name)
        return super().find_class(module, name)
with open('${BUNDLE_PKL}', 'rb') as f:
    d = _U(f).load()
print(d['bundle'].mode)
")"
echo "[Stage 1/3] Loaded ${BUNDLE_PKL} (selected_mode=${BUNDLE_MODE})"

SPATIAL_644_ARG=()
if [[ "${BUNDLE_MODE}" == "spatial" || "${BUNDLE_MODE}" == "joint" ]]; then
  mkdir -p "${SPATIAL_DIR_644}"
  SPATIAL_644_NPZ="${SPATIAL_DIR_644}/spatial_target_644.npz"
  if [[ ! -f "${SPATIAL_644_NPZ}" ]]; then
    echo "[Stage 1/3] Extracting 644 spatial features across ${NUM_GPUS} GPUs..."
    rm -f "${SPATIAL_DIR_644}"/target_shard_*_of_*.npz
    pids=()
    logs=()
    for (( s=0; s<NUM_GPUS; s++ )); do
      gpu_id="${GPU_ARR[$s]}"
      log_f="${OUT_DIR}/logs/spatial_644_s${s}_gpu${gpu_id}.log"
      CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u code/r55/spatial_salience_probe.py \
        --mode target \
        --template "${TEMPLATE_644}" \
        --img-dir "${RESOLVED_IMG_DIR}" \
        --shard "${s}" \
        --num-shards "${NUM_GPUS}" \
        --gpu "${gpu_id}" \
        --out-dir "${SPATIAL_DIR_644}" > "${log_f}" 2>&1 &
      pids+=("$!")
      logs+=("${log_f}")
    done
    for idx_p in "${!pids[@]}"; do
      if ! wait "${pids[$idx_p]}"; then
        echo "[ERROR] Spatial 644 shard ${idx_p} failed (${logs[$idx_p]}):" >&2
        tail -n 45 "${logs[$idx_p]}" >&2 || true
        exit 1
      fi
    done
    "${PYTHON_BIN}" -u code/r55/spatial_salience_probe.py \
      --mode merge-target \
      --template "${TEMPLATE_644}" \
      --num-shards "${NUM_GPUS}" \
      --out-dir "${SPATIAL_DIR_644}" \
      --out-npz "${SPATIAL_644_NPZ}"
  fi
  SPATIAL_644_ARG=(--spatial-target "${SPATIAL_644_NPZ}")
else
  echo "[Stage 1/3] Mode '${BUNDLE_MODE}' uses pure ensemble cross-defect covariates -> skipping Stage 1B tile probe on 644."
fi

# ------------------------------------------------------------------------------
# Stage 2: 8-GPU Sharded FinalEngine Inference on 644 Set + N=1 & Schema Gates
# ------------------------------------------------------------------------------
STEM_644="${FINAL_644_JSON%.json}"
PROBS_644_NPZ="${STEM_644}_member_probs.npz"
REUSE_ARG=()
if [[ "${REUSE_PROBS}" == "1" && -f "${PROBS_644_NPZ}" ]]; then
  echo "[Stage 2/3] REUSE_PROBS=1: reusing cached member probabilities ${PROBS_644_NPZ}"
  REUSE_ARG=(--reuse-probs-npz "${PROBS_644_NPZ}")
else
  echo "[Stage 2/3] Launching ${NUM_GPUS}-GPU parallel FinalEngine inference on 644 images..."
fi

"${PYTHON_BIN}" -u code/r55/multi_gpu_infer.py \
  --set "${SET_NAME}" \
  --gpus "${GPUS}" \
  --template "${TEMPLATE_644}" \
  --img-dir "${RESOLVED_IMG_DIR}" \
  --bundle "${BUNDLE_PKL}" \
  --out "${FINAL_644_JSON}" \
  ${SPATIAL_644_ARG[@]+"${SPATIAL_644_ARG[@]}"} \
  ${REUSE_ARG[@]+"${REUSE_ARG[@]}"}

# ------------------------------------------------------------------------------
# Stage 3: Detailed Diff Audit vs 53.99 (resident_v6) & Fresh v7 Baseline
# ------------------------------------------------------------------------------
echo "[Stage 3/3] Auditing differences vs 53.99 process-review submission (${REF_5399_JSON})..."
"${PYTHON_BIN}" -u code/r55/compare_final_644.py \
  --ref "${REF_5399_JSON}" \
  --v7-base "${STEM_644}_v7_baseline_fresh.json" \
  --new "${FINAL_644_JSON}" \
  --probs-npz "${STEM_644}_calibrated_probs.npz" \
  --out-report "${OUT_DIR}/diff_vs_5399_v6_report.json"

echo "=============================================================================="
echo "[DONE] 644 Final Evaluation Set Inference & Diff vs 53.99 Completed!"
echo "  1) Primary Recommended 644 JSON : ${FINAL_644_JSON}"
echo "  2) Pure resident_v7 Base JSON   : ${STEM_644}_v7_baseline_fresh.json"
echo "  3) Full Diff Report vs 53.99    : ${OUT_DIR}/diff_vs_5399_v6_report.json"
echo "=============================================================================="
