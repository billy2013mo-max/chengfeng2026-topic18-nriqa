# 【2026乘风挑战赛·命题18】最终提交产物核对清单

根据命题文档第 5 节《提交要求》，提交压缩包严格按以下 6 类官方产物分目录整理：

## 1. 官方产物清单核对

- [x] **01_预测结果文件_核心交付物/**
  - 公开打榜集独立从头推理结果：`pred_nriqa_dataset_test_r55_resident_v6_core_routed_gate_eu.json`（150 条样本，ID 唯一，6 类标签全部合法，`Clean=19/150`，`PosDims=332/900`，`N=1 == N=150` 差异 `0/900`）。

- [x] **02_可执行代码/**
  - 一键推理与校验脚本：[`run_final_infer.sh`](file:///wanqing-develop/luowenjing/18/run_final_infer.sh)（默认 `resident_v6_core` + `routed_gate_eu`）
  - 核心常驻推理引擎：[`code/r55/final_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/final_engine.py)
  - 提交格式校验工具：[`code/r55/validate_submission.py`](file:///wanqing-develop/luowenjing/18/code/r55/validate_submission.py)
  - 在线/批量 HTTP 服务：`code/r55/serve_engine.py`（`/healthz` `/infer` `/infer_batch`，2026-11-16 实测通过）
  - 部署环境自检：`code/r55/check_deploy_env.py` + 依赖锁 `requirements_infer.txt`（50 项 0 FAIL）
  - 实测证据归因：`code/r55/measured_evidence.py`（经 `code/r55/subjective_engineering.py` 调用）
  - 完整代码闭包：`code/r55/`、`code/next55/`、`code/qingyu/`、`code/phoenix/` 等实际被引用模块，以及 `third_party/pydeps`（`bitsandbytes` 4-bit NF4 依赖，full 包内置）。

- [x] **03_模型产物/**
  - `README_模型与基座版本说明.md`（基座版本、6 成员结构、9.656B 参数量与显存说明）
  - `decoders/`：`resident_v6_core_decoder_routed_gate_eu.pkl` / `.json`（含 `eu_st` 等完整校准参数）、`resident_v6_core_decoder_routed_eu.pkl` / `.json`、`resident_v6_core_decoder.pkl`、`resident_v6_core_bench_half.json`、`resident_v6_core_bench_half_chunk2.json`（含 `nvidia-smi` 实测峰值）、`resident_v6_core_clean_footprint.npz`（归因离线洁净基线，推理不读训练数据）、`resident_v6_core_video_aggregator.pkl`
  - `checkpoints/`（`full` 版包含）：
    - `cradio_v3/fold_0_best.pt`（`c-radio-v4-h`）
    - `peg/fold_0_best.pt`（`pe_core_large` + 花屏专项增强）
    - `ntg_pe/fold_0_best.pt`（`pe_core_large` 原生分辨率 tile+global）
    - `p12_pe/fold_0_best.pt`（`pe_core_large` 语义 ROI + KD）
    - `v16/V16_Full_F0_AllGrade_Microscope_swa.pt`（`InternViT-6B` 局部显微镜）
    - `p1_qy/p1_c233397_s2026_fullfit_swa.pt`（`Qwen3-VL-8B` DoRA `r=24`）

- [x] **04_技术方案文档/**
  - [`docs/TECHNICAL_REPORT.md`](file:///wanqing-develop/luowenjing/18/docs/TECHNICAL_REPORT.md)
  - [`docs/RESIDENT_V6_PIPELINE_ARCHITECTURE.md`](file:///wanqing-develop/luowenjing/18/docs/RESIDENT_V6_PIPELINE_ARCHITECTURE.md)
  - [`docs/SUBMISSION_CHECKLIST.md`](file:///wanqing-develop/luowenjing/18/docs/SUBMISSION_CHECKLIST.md)
  - [`docs/SERVING_AND_TAT_JIMU_ADAPTER.md`](file:///wanqing-develop/luowenjing/18/docs/SERVING_AND_TAT_JIMU_ADAPTER.md)（服务接口与 极目/TAT/VQA 适配映射建议）

- [x] **05_自评测报告/**
  - [`docs/SELF_EVALUATION_REPORT.md`](file:///wanqing-develop/luowenjing/18/docs/SELF_EVALUATION_REPORT.md)
  - [`results/r55/vlm_qy8/prune_v6_audit.json`](file:///wanqing-develop/luowenjing/18/results/r55/vlm_qy8/prune_v6_audit.json)（9 成员→6 成员收敛审计记录）

- [x] **06_AI使用说明/**
  - [`docs/AI_USAGE_STATEMENT.md`](file:///wanqing-develop/luowenjing/18/docs/AI_USAGE_STATEMENT.md)

## 2. 2026-11-16 部署复核（全部在 GPU 0 实测）

- [x] `python3 code/r55/check_deploy_env.py` → 50 项，0 FAIL / 0 WARN（`sm_89` 经 `sm_80/sm_86` SASS 二进制兼容）
- [x] `bash run_final_infer.sh results/v2_full/template_150.json data/test_150 <out> 0` → `validate_submission.py` PASS，与提交 JSON **字节一致**（sha256 `4e23cad1…`，`0/900`），`N=1 == N=150` 0/900
- [x] 同命令 + `R55_EXTRA_ARGS="--emit-attribution … --tat-out … --video-mode"` → 输出 JSON 仍字节一致；训练数据文件读取 0 次
- [x] `--bench 20 --half`：均值 2.07s / P95 3.81s；`nvidia-smi` 峰值 20,290 MiB（推理）/ 20,688 MiB（加载）
- [x] `serve_engine.py`（FastAPI 与 stdlib 两种后端）：`/healthz` `/infer`（路径与 base64）`/infer_batch`（自动分组与 `video_id`）全部 200 且标签一致；错误请求 404/400；SIGTERM 退出码 0，显存归零
