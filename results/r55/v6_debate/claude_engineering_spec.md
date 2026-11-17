# Autonomous Engineering Task: Maximize Subjective 35 pts (`S_deploy` 15 + `S_cost` 20) + Semantic Bonus 5 pts for `resident_v6_core`

You have full read/write/command execution permissions in `/wanqing-develop/luowenjing/18`. Complete ALL 5 engineering upgrades below end-to-end, run real GPU verification tests (8xH200s are currently idle; use `--gpu 0` for tests), and write a concise final verification summary to `/wanqing-develop/luowenjing/18/results/r55/v6_debate/claude_engineering_report.md`.

## Strict Rules & Invariants (DO NOT VIOLATE)
1. Follow `/wanqing-develop/luowenjing/18/AGENTS.md` Rules 1–3 strictly: zero legacy JSON anchoring, zero test-set tuning, keep `resident_v6_core` 6-member weights & `RoutedGateEUDecoder` untouched.
2. **ZERO label drift**: `results/final/pred_nriqa_dataset_test_r55_resident_v6_core_routed_gate_eu.json` (on `data/test_150`) MUST remain **0 / 900 cell diff** (`N=1 == N=150` batch-invariant) and pass `validate_submission.py`.
3. Do NOT touch `/wanqing-develop` disk recklessly (`133 GB` free); do not retrain any models.

## Engineering Upgrades to Implement & Verify

### 1. `S_cost` Peak VRAM Headroom Optimization (`code/r55/final_engine.py`, `run_final_infer.sh`)
- Close the gap between torch peak (`17.74 GB`) and `nvidia-smi` peak (`21.16 GiB`):
  - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (in `run_final_infer.sh` and `os.environ.setdefault` before torch import if helpful).
  - Call `gc.collect()` + `torch.cuda.empty_cache()` after loading each large member / after startup.
  - Benchmark `V16_CHUNK_BY_SET["resident_v6_core"] = 1` vs `2`: verify exact-match `max|dp| == 0.0` (or `< 1e-4`), measure `nvidia-smi` peak MiB and `N=1` latency (`--bench 20 --half`), and keep `chunk=1` if it lowers `nvidia-smi` peak without materially hurting latency (`<= 15s/img` target).

### 2. Decouple Inference from Training Data (`final_engine.py`)
- Currently `final_engine.py` (around line 737) reads `data/manifest_combined.csv` and 24 training images at inference runtime to calibrate the clean footprint.
- Precompute the clean-footprint baseline into a small `.npz` file under `results/r55/final/resident_v6_core_clean_footprint.npz` (and support loading it automatically at inference time so inference has **zero runtime dependency on `data/manifest_combined.csv` or training images**).

### 3. Online + Batch HTTP Service (`code/r55/serve_engine.py`) & TAT/极目 Adapter Docs
- Create `code/r55/serve_engine.py` (use `fastapi`+`uvicorn` if installed, otherwise stdlib `http.server.ThreadingHTTPServer` so there are zero missing dependencies) that loads `resident_v6_core` once at startup and provides:
  - `GET /healthz`: returns status, `--set`, `load_s`, `torch_peak_gb`, `nvidia_smi_mib`.
  - `POST /infer`: single-image online endpoint (accepts JSON `{image_path}` or `{image_base64}`), runs the full `N=1` prediction + per-image measured-evidence semantic attribution + `tat_jimu_payload`, returns JSON + `latency_ms`.
  - `POST /infer_batch`: batch endpoint (accepts list of image paths) returning per-image predictions + video-level `EffectiveFrameORVideoAggregator` summary.
- Document the explicit field-mapping table from `tat_jimu_payload` to 极目 / TAT / VQA schemas in `docs/` / `README.md` (clearly stating it is an adapter mapping, not claiming an undocumented internal schema).

### 4. Upgrade `Semantic Score` (额外 5 分) to Measured-Evidence Attribution in `N=1` Path
- Replace fixed `_EVIDENCE_CUES` templates and hard `[:40]` truncation (`reason_40ch`) with **measured box-vs-global image physics evidence** (reusing `defect_attribution_map.py` / `semantic_roi_meta.py` physics: e.g., edge overshoot amplitude for `过锐`, Laplacian variance for `模糊`, blockiness/chroma-jump for `花屏`, matte-boundary sharpness jump for `虚化突兀`, local HF/coherence ratio for `过度生成`, compared against clean-set percentiles, explicitly marking uncertain causes rather than fabricating causal assertions).
- Ensure per-image attribution runs inside the `N=1` path (skipping expensive localization on clean images where `pred == 0` on all 6 dims, or running fast `<0.3s`!) and verify `N=1` end-to-end latency (including attribution) and **byte-identical `label` output** vs attribution-off.

### 5. Dependency Lock, `sm_89` 4090 Compatibility Check & Packaging Update
- Create `requirements_infer.txt` and `code/r55/check_deploy_env.py` (verifies Python deps including `third_party/pydeps/bitsandbytes`, CUDA/SM capability, precomputed decoder & clean-footprint `.npz`, and checkpoints).
- Run end-to-end verification tests:
  1. `python3 code/r55/check_deploy_env.py`
  2. `final_engine.py --set resident_v6_core --half --gpu 0 --bench 20` (report exact `nvidia-smi` peak MiB, `load_s`, mean & P95 `N=1` latency)
  3. Fresh `test_150` inference + `validate_submission.py` + verify `0/900` label diff vs `results/final/pred_nriqa_dataset_test_r55_resident_v6_core_routed_gate_eu.json`
  4. Live HTTP test of `serve_engine.py` (`/healthz`, `/infer`, `/infer_batch`) on GPU 0 and shut down the server cleanly after testing.
  5. Update `docs/SUBMISSION_CHECKLIST.md`, `docs/SELF_EVALUATION_REPORT.md`, `README.md`, and run `code/r55/package_submission.py` if appropriate.

Write all observed test results, exact commands, file paths, and metrics to `/wanqing-develop/luowenjing/18/results/r55/v6_debate/claude_engineering_report.md`.
