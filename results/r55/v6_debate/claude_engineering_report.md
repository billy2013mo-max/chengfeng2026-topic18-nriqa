# `resident_v6_core` 部署工程升级：实测验证报告（2026-11-16）

All runs are on GPU 0 (H200, sm_90) unless stated otherwise. Logs are in `results/r55/v6_debate/eng_logs/` and outputs in `results/r55/v6_debate/eng_runs/`.
Invariants: no model was retrained. Member set, weights and `RoutedGateEUDecoder` pickles are untouched. No historical JSON is read as input; the reference JSON is used **only** as the comparison target for label drift.

## 0. Headline

| Check | Result |
| :--- | :--- |
| test_150 fresh inference via `run_final_infer.sh` (attribution off) | **byte-identical** to `results/final/pred_nriqa_dataset_test_r55_resident_v6_core_routed_gate_eu.json` (sha256 `4e23cad19d48…`), **0/900** cells, `validate_submission.py` PASS |
| Same run with attribution + TAT + video sidecars on | output JSON again **byte-identical**, 0/900, PASS |
| N=1 vs N=150 decode (checked in every template run) | **0/900** |
| Training-data file opens during inference (audit hook) | **0** |
| `check_deploy_env.py` | 50 checks, **0 FAIL / 0 WARN** |
| `nvidia-smi` peak, inference (bench 20) | **20,290 MiB (19.81 GiB)**; same-machine control without expandable segments: 22,278 MiB (**−1,988 MiB**) |
| `nvidia-smi` peak incl. loading | 20,688 MiB (20.20 GiB); historical figure was 21.16 GiB |
| N=1 latency (bench 20, `--half`) | mean **2.07 s**, median 1.96 s, P95 **3.81 s**, max 3.99 s |
| test_150 end-to-end per image | 2.07 s (attribution off); **2.35 s** mean / P95 3.31 s (attribution on) |
| Attribution cost | mean 0.166 s, P95 0.245 s, max 0.332 s (131/150 defect images; clean images 0.06 ms) |
| HTTP service (FastAPI and stdlib) | all endpoints 200, 15/15 labels match the submission, errors return 404/400, SIGTERM gives exit code 0 and GPU memory goes back to 0 MiB |

## 1. `S_cost`: VRAM headroom

Changes:
- `code/r55/final_engine.py`: `os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")` before torch is imported.
- `gc.collect()` + `torch.cuda.empty_cache()` after every member load (`note()`) and once more at the end of start-up.
- A new `NvmlPeakSampler` runs a 20 ms NVML thread on the GPU torch is using (matched by UUID), so the bench now reports device-level (`nvidia-smi`) peaks.
- `run_final_infer.sh`: `export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}`, plus an optional `R55_EXTRA_ARGS` sidecar pass-through.

Commands:
```bash
bash results/r55/v6_debate/eng_logs/run_bench_pair.sh   # = for CH in 2 1: final_engine.py --set resident_v6_core --half --gpu 0 --bench 20 --v16-chunk $CH --bench-tag _chunk$CH  (+ external nvidia-smi poll)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False python3 code/r55/final_engine.py --set resident_v6_core --half --gpu 1 --bench 20 --bench-tag _noexpand_ablation
```

| config (20 imgs, same image list) | mean / P95 s | torch resident / peak infer GB | nvidia-smi peak load MiB | after warm-up MiB | **peak infer MiB** | torch reserved GiB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| expandable **off** (control, gc on, chunk 2, GPU 1) | 2.08 / 3.66 | 15.77 / 17.74 | 20,826 | 18,864 | **22,278** | 20.47 |
| expandable on, **v16_chunk=2** (kept) | **2.07 / 3.81** | 15.68 / 17.64 | 20,688 | 18,068 | **20,290** | 18.53 |
| expandable on, v16_chunk=1 | 2.48 / 4.07 | 15.68 / 17.64 | 20,668 | 18,068 | **20,290** | 18.53 |

- **chunk decision: keep 2.** chunk=1 gives exactly the same `nvidia-smi` peak, because V16 is not the peak member (ntg_pe peaks at 17.64 GB, v16 at 16.25 GB). It is 0.41 s/img slower (v16 0.99 s → 1.41 s) and it changes V16 probabilities by up to max|dp| = 1.8e-4, which is above the 1e-4 bar; every other member is 0.0 and no argmax changes. chunk=2 is also the setting the reference submission was produced with.
- The external `nvidia-smi` poller (≈0.3 s cadence) recorded a maximum of 20,053 MiB. The 20 ms NVML sampler catches higher short spikes, so the larger NVML figures are the ones reported.
- Files: `results/r55/final/resident_v6_core_bench_half_chunk{1,2}.json`, `…_noexpand_ablation.json`, and `eng_logs/bench_chunk{1,2}.log` / `bench_noexpand_gpu1.log`.
- The load time varies with network-disk cache state: 320 s cold, then 142–150 s warm.

## 2. Inference decoupled from training data

- New offline step: `python3 code/r55/final_engine.py --set resident_v6_core --build-clean-footprint --half --gpu 0` (35 s), which writes `results/r55/final/resident_v6_core_clean_footprint.npz` (48.5 KiB). Its contents:
  - `fp__peg__midblock_4x4` and `fp__peg__sub8_blk15`: the same 24 clean training frames and the same `PEAttributionOperator.fit_clean_footprint` call as the old runtime code.
  - `phys_tile__*` / `phys_glob__*`: sorted clean-reference physics values from 160 clean training frames.
  - `meta_json`: tower, checkpoint, frame names, source.
- `AttributionRuntime` loads the npz automatically. If it is missing, inference stops with the build command. The old runtime read of `data/manifest_combined.csv` plus 24 training images has been removed from `_run_stage4_sidecars`.
- Evidence: test_150 run with `R55_AUDIT_TRAIN_READS=1`, using a `sys.addaudithook` that records Python-level opens of `manifest_combined.csv`, `data/images*`, split CSVs and `*_oof.npz`. Result: `training-data file opens during this inference run: 0`. The hook does not see C-level reads (cv2), but none of those paths go through cv2 in the inference code.
- Packaged layout: `clean_footprint_path()` resolves to `03_模型产物/decoders/…npz`. This was checked by extracting the lite zip and importing `final_engine` from it.
- Not separately verified: that the standalone-built footprint is numerically identical to one computed inside the full engine. It is the same tower, the same weights and the same function, and it only affects attribution, never labels.

## 3. HTTP service `code/r55/serve_engine.py` + adapter docs

- `GET /healthz` returns: status, set, members, decoder, `load_s`, `warmup_s`, `torch_peak_gb`, `torch_allocated/reserved_gb`, `nvidia_smi_mib`, `nvidia_smi_peak_mib`, `params_b`, `requests`, `uptime_s`, `clean_footprint`, `pytorch_cuda_alloc_conf`.
- `POST /infer` takes `{image_path}` or `{image_base64, image_name}` and runs the full N=1 path: `FinalEngine.infer` → row-wise decoder → measured-evidence attribution → `tat_jimu_payload`. It returns `latency_ms{infer,decode,attr,total,request}`.
- `POST /infer_batch` takes `{image_paths[] | items[], video_id?}` and returns per-image records plus a per-clip `EffectiveFrameORVideoAggregator` summary, using the fitted `resident_v6_core_video_aggregator.pkl`.
- Transport: FastAPI/uvicorn when installed, otherwise the stdlib `ThreadingHTTPServer` (`--backend` forces one). GPU work is serialised by a lock.
- Uploads are staged under unique `srv_<uuid>_` names (a symlink for path requests) and deleted after each request, so they cannot collide with a cached training frame of the same name. 0 leftovers were found after the tests.
- The server binds `127.0.0.1` by default; `--path-root` restricts readable paths.
- Start-up includes a warm-up on a synthetic image (4.9 s).
- On SIGTERM, uvicorn shuts down gracefully and then re-raises the signal; a Python handler turns that into exit code 0.
- Live test: `python3 results/r55/v6_debate/eng_logs/http_live_test.py 8055 fastapi` and `… 8056 stdlib`. Results are in `eng_runs/http_live_test_{fastapi,stdlib}.json`.

| request (FastAPI run) | status | label == submission | latency_ms (infer / attr / total) |
| :--- | :--- | :--- | :--- |
| `/infer` path, clean image 196160772653_10.03s.png | 200 | yes | 3562 / 0.06 / 3607 |
| `/infer` path, 137212400938_frame14.png (花屏 P0) | 200 | yes | 1836 / 268 / 2122 |
| `/infer` path ×2 more | 200 | yes | ≈2370 / 222–232 / ≈2630 |
| `/infer` base64 152861394819_53.png | 200 | yes | 2306 / 196 / 2527 |
| `/infer_batch` 4 images, auto grouping | 200 | 4/4 | 4 clips of M=1 (test_150 frames come from distinct clips) |
| `/infer_batch` 6 images, `video_id=demo_clip_M6` | 200 | 6/6 | one M=6 video summary, request 15.0 s |
| missing path / bad body | 404 / 400 | — | — |
| SIGTERM | exit 0, log "shut down cleanly", GPU 0: 19,255 → 0 MiB | | |

- The stdlib backend gave the same results: all labels match, 404/400 on bad requests, exit 0.
- Ready-after times were 153 s (FastAPI) and 146 s (stdlib), counting from process start and including model load and warm-up.
- Adapter docs: `docs/SERVING_AND_TAT_JIMU_ADAPTER.md` holds the endpoint table and the `tat_jimu_payload` → 极目 / TAT / VQA field-mapping table. It states explicitly that the target names are adapter placeholders, not a documented internal schema. It is linked from `README.md` §3.

## 4. Semantic score: measured-evidence attribution in the N=1 path

- `code/r55/measured_evidence.py` (new) computes physics on 448² native-pixel tiles using OpenCV, and only for the metrics the flagged dimensions need:

| dimension | metric | direction |
| :--- | :--- | :--- |
| 模糊 | `lapvar` | low |
| 纹理损失 | `tex_hf` | low |
| 过锐 | `overshoot` | high |
| 花屏 | `blocky`, `chroma` | high |
| 过度生成 | `coh`, `hf_ratio` (box vs global) | high |
| 虚化突兀 | `cliff` | high |

- Scope: the value inside the attribution box is ranked against the clean-tile distribution. When the map is diffuse, the median of a 3×3 tile grid is ranked against the clean-image distribution. The P10/P90 thresholds are fixed a priori, and the clean reference comes from training data only.
- `subjective_engineering.generate_evidence_attribution` no longer uses the `_EVIDENCE_CUES` templates or the `[:40]` slice. Each defect now has `evidence{scope, measurements[{metric, name, value, clean_pct, supports, verdict, global_value}], measured_support, verdict}`.
  - `reason_40ch` is composed from the measured value and stays ≤40 characters by construction; the longest observed was 40.
  - When the physics does not support the call, the reason says so, e.g. `全图P2级纹理损失：纹理高频能量洁净P91反向，物理证据不支持，待复核`, and `uncertainty_note` explains why.
  - I also fixed an existing bug where the alternative-grade note could print "P2与P2".
- The N=1 path is `N1Service.predict`, used by both the template loop and the server. Images with all six dimensions predicted clean skip GPU localisation and physics entirely.
- Label invariance: attribution reads only the decoded prediction. With attribution on, the test_150 output JSON is byte-identical to the attribution-off run and to the reference.
- test_150 attribution statistics over the 332 defect calls (`eng_runs/test150_attribution.json`, `test150_tat_jimu.json`, `test150_attribution_sheet.jpg`):

| dimension | supports | inconclusive | contradicts |
| :--- | :--- | :--- | :--- |
| 模糊 | 16 | 52 | 7 |
| 纹理损失 | 14 | 106 | 7 |
| 过锐 | 6 | 55 | 1 |
| 花屏 | 4 | 6 | 0 |
| 过度生成 | 5 | 39 | 0 |
| 虚化突兀 | 0 | 12 | 2 |
| **total** | **45 (13.6%)** | **270** | **17** |

  - 328 calls have a box; 4 are "全图".
  - 135 of 150 images carry `causal_uncertainty = true`.
  - **Caveat:** classical pixel statistics independently confirm only a minority of the model's calls. The output now says this honestly instead of asserting a cause. Whether graders reward this more than the old fixed cue text is a judgement call and was not measured.

## 5. Dependency lock, sm_89 check, packaging

- `requirements_infer.txt` pins exact versions of this environment: Python 3.11.11, torch 2.13.0+cu130, transformers 5.6.0, timm 1.0.30, and so on. FastAPI/uvicorn are marked optional, and bitsandbytes 0.50.2 is noted as vendored.
- `code/r55/check_deploy_env.py` checks:
  - every pinned package
  - vendored bitsandbytes: import, `libbitsandbytes_cuda130.so`, and a live NF4 4-bit matmul on the GPU
  - CUDA, bf16 and VRAM
  - sm_89 compatibility: this torch build has no native sm_89, but it ships sm_80/sm_86 SASS, which is binary-compatible on Ada (same major version)
  - decoder and video aggregator, plus clean-footprint npz keys and tower match
  - all 6 member checkpoints and the local base weights (Depth-Anything-V2-Small comes from the HF cache)
- Command: `python3 code/r55/check_deploy_env.py` → `50 checks: 0 FAIL, 0 WARN -> READY` (JSON in `eng_runs/check_deploy_env.json`).
- Not covered: the 5 V16 sub-backbone base weights are not checked individually (only the V16 checkpoint is), and there is no real RTX 4090 run. sm_89 is a static compatibility argument; no 4090 hardware was available.
- `code/r55/package_submission.py` changes:
  - adds `serve_engine.py`, `check_deploy_env.py`, `measured_evidence.py`, `subjective_engineering.py` and `defect_attribution_map.py` as seeds
  - adds `requirements_infer.txt`, the clean-footprint npz, the video aggregator, the chunk-2 bench JSON and the adapter doc
  - adds `--tag` and `--lite-only` options, and updates the model-manifest numbers
- Packaging output: `topic18_submission_lite_20261116.zip` (0.41 MB, 74 files) and `topic18_submission_full_20261116.zip` (5.94 GB, 151 files); 53 Python files pass `py_compile`. The older 20261114/20261115 zips were left in place (≈13.6 GB); delete them if they are no longer needed.
- Docs updated: `README.md` (headline numbers + §3 部署工程), `docs/SUBMISSION_CHECKLIST.md` (new code/decoder files + §2 复核), `docs/SELF_EVALUATION_REPORT.md` (§四), and the new `docs/SERVING_AND_TAT_JIMU_ADAPTER.md`.

## Exact verification commands

```bash
python3 -u code/r55/final_engine.py --set resident_v6_core --build-clean-footprint --half --gpu 0
bash results/r55/v6_debate/eng_logs/run_bench_pair.sh
bash run_final_infer.sh results/v2_full/template_150.json data/test_150 results/r55/v6_debate/eng_runs/pred_test150_official_script.json 0 resident_v6_core routed_gate_eu
R55_AUDIT_TRAIN_READS=1 R55_EXTRA_ARGS="--emit-attribution results/r55/v6_debate/eng_runs/test150_attribution.json --tat-out results/r55/v6_debate/eng_runs/test150_tat_jimu.json --video-mode" \
  bash run_final_infer.sh results/v2_full/template_150.json data/test_150 results/r55/v6_debate/eng_runs/pred_test150_attr_on.json 0 resident_v6_core routed_gate_eu
cmp results/r55/v6_debate/eng_runs/pred_test150_attr_on.json results/final/pred_nriqa_dataset_test_r55_resident_v6_core_routed_gate_eu.json   # identical
python3 results/r55/v6_debate/eng_logs/http_live_test.py 8055 fastapi ; python3 results/r55/v6_debate/eng_logs/http_live_test.py 8056 stdlib
python3 code/r55/check_deploy_env.py --json results/r55/v6_debate/eng_runs/check_deploy_env.json
python3 code/r55/package_submission.py --tag 20261116
```

Files changed or added: `code/r55/final_engine.py`, `code/r55/subjective_engineering.py`, `code/r55/measured_evidence.py` (new), `code/r55/serve_engine.py` (new), `code/r55/check_deploy_env.py` (new), `code/r55/package_submission.py`, `run_final_infer.sh`, `requirements_infer.txt` (new), `results/r55/final/resident_v6_core_clean_footprint.npz` (new), the docs listed above, plus the logs and test scripts under `results/r55/v6_debate/eng_logs/`. Every server process was shut down and GPU 0 is idle (0 MiB).
