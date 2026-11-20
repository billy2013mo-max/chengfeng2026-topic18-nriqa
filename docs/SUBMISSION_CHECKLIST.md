# 【2026乘风挑战赛·命题18】最终提交产物核对清单 (`SUBMISSION_CHECKLIST.md`)

根据命题文档第 5 节《提交要求》，本提交压缩包（`topic18_submission_full_20261120_v8.zip` / `topic18_submission_lite_20261120_v8.zip`，主推架构 **`resident_v8_boundary` + `deconf_v6_boundary`**，公开打榜实测 **`54.77 / 65`**）严格按官方规定的 **六大产物目录** 结构化整理，全项已通过单卡断网复现与校验：

---

## 1. 官方六类交付物逐项核对

- [x] **`01_预测结果文件_核心交付物/`**
  - **终评集 644 图主交付文件**：`pred_nriqa_dataset_final_644_r55_resident_v8_boundary_deconf_v6_boundary.json`（644 条样本，ID 100% 唯一对齐模板，`Clean = 110/644`，阳性缺陷格 `1,346/3,864`，`N=1 == N=644` 差异 `0/3,864`，`max_q_diff = 3.30e-13`，SHA256 `2f6d188ea648a662f8a828d30c038802d9685e60d3e19893083e993db585528f`）。
  - **公开打榜集 150 图主交付文件**：`pred_nriqa_dataset_test_r55_resident_v8_boundary_deconf_v6_boundary.json`（150 条样本，线上打榜实测再破新高 **`54.77 / 65`**：正向缺陷 `17.46/25`、P0 识别 `17.31/20`、`Clean = 20.00/20` 满分，阳性缺陷格 `327/900`，`N=1 == N=150` 差异 `0/900`，`max_q_diff = 3.72e-13`，SHA256 `93a39abea3d4c812c517e3e6c576c45f6191ee7affcba51444431b91ef9d029f`）。
  - **主观语义归因与极目/视频旁路目录** `sidecars_语义归因与极目视频载荷/`（全部基于 `resident_v8_boundary + deconf_v6_boundary` 重新生成并 100% 格级对齐）：
    - `final644_attribution_v8.json`、`final644_attribution_v8_sheet.jpg`、`final644_tat_jimu_v8.json`、`final644_video_summary_v8.json`
    - `test150_attribution_v8.json`、`test150_attribution_v8_sheet.jpg`、`test150_tat_jimu_v8.json`、`test150_video_summary_v8.json`

- [x] **`02_可执行代码/`**
  - 一键推理与校验脚本：[`run_final_infer.sh`](file:///wanqing-develop/luowenjing/18/run_final_infer.sh)（默认 `resident_v8_boundary` + `deconf_v6_boundary` + `--half --share-weights --embed-on-host`）
  - 离线生产容器构建定义：[`Dockerfile`](file:///wanqing-develop/luowenjing/18/Dockerfile)（CUDA 13.0 + Python 3.11 + `HEALTHCHECK /healthz` + 断网环境变量锁定）
  - 推理环境依赖锁：[`requirements_infer.txt`](file:///wanqing-develop/luowenjing/18/requirements_infer.txt)
  - 核心常驻引擎、边界专精塔、双适配器共享 VLM 与去混淆校准：[`code/r55/final_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/final_engine.py)、[`code/r55/dual_vlm_shared_4090.py`](file:///wanqing-develop/luowenjing/18/code/r55/dual_vlm_shared_4090.py)、[`code/r55/train_boundary_specialist.py`](file:///wanqing-develop/luowenjing/18/code/r55/train_boundary_specialist.py)、[`code/r55/upgrade_deconf_v6_boundary.py`](file:///wanqing-develop/luowenjing/18/code/r55/upgrade_deconf_v6_boundary.py)、[`code/r55/upgrade_deconf_v5_boundary.py`](file:///wanqing-develop/luowenjing/18/code/r55/upgrade_deconf_v5_boundary.py)、[`code/r55/upgrade_deconf_v4_tile.py`](file:///wanqing-develop/luowenjing/18/code/r55/upgrade_deconf_v4_tile.py)、[`code/r55/ntg_tile_peak_probe.py`](file:///wanqing-develop/luowenjing/18/code/r55/ntg_tile_peak_probe.py)、[`code/r55/spatial_salience_probe.py`](file:///wanqing-develop/luowenjing/18/code/r55/spatial_salience_probe.py)、[`code/r55/deconfound_calibrator.py`](file:///wanqing-develop/luowenjing/18/code/r55/deconfound_calibrator.py)、[`code/r55/routed_eu_decoder.py`](file:///wanqing-develop/luowenjing/18/code/r55/routed_eu_decoder.py)
  - 主观归因与视频聚合算子：[`code/r55/measured_evidence.py`](file:///wanqing-develop/luowenjing/18/code/r55/measured_evidence.py)、[`code/r55/subjective_engineering.py`](file:///wanqing-develop/luowenjing/18/code/r55/subjective_engineering.py)、[`code/r55/defect_attribution_map.py`](file:///wanqing-develop/luowenjing/18/code/r55/defect_attribution_map.py)
  - 在线/批量双后端 HTTP 微服务：[`code/r55/serve_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/serve_engine.py)
  - 断网部署自检与提交校验：[`code/r55/check_deploy_env.py`](file:///wanqing-develop/luowenjing/18/code/r55/check_deploy_env.py)、[`code/r55/validate_submission.py`](file:///wanqing-develop/luowenjing/18/code/r55/validate_submission.py)
  - 完整 Python 依赖闭包（全部通过 `py_compile` 语法检查与解压导入核验）+ `third_party/pydeps`（内置 `bitsandbytes` CUDA NF4 原生动态库，`full` 包内置）。

- [x] **`03_模型产物/`**
  - `README_模型与基座版本说明.md`（详列 14 个专精成员的开源基座版本、激活去重参数 `13.262B <= 30B` 与双重无损共享显存机制）
  - `decoders/`：`resident_v8_boundary_decoder_deconf_v6_boundary.pkl` / `.json`（主推校准解码器，附 `deconf_v5_boundary / deconf_v4_tile / deconf_v3 / deconf_v2 / routed_gate_eu` 演进对照）、`resident_v8_boundary_clean_footprint.npz`（离线洁净归因基线，`75.2 KiB`）、`resident_v8_boundary_video_aggregator.pkl`（多帧视频聚合器）、`resident_v8_boundary_bench_half.json`，以及纯视觉轻量备选档 `resident_v7_novlm_*` 全套解码器与基准文件
  - `checkpoints/`（`full` 版完整包含全部 **14 个** Full-Fit 权重）：
    - `cradio_v3/fold_0_best.pt`、`pe_v3/fold_0_best.pt`、`peg/fold_0_best.pt`、`sig_v3/fold_0_best.pt`、`pe_bkseg1/fold_0_best.pt`
    - `ntg_pe/fold_0_best.pt`、`ntg_convnext/fold_0_best.pt`、`p12_pe/fold_0_best.pt`
    - `bndry_pe/fullfit_best.pt`、`bndry_dual/fullfit_best.pt`、`qwen_hr/qwen3vl_hires_s2026_fullfit_swa.pt`
    - `v16/V16_Full_F0_AllGrade_Microscope_swa.pt`、`p3_qy/pillar3_fulldata_swa.pt`、`p1_qy/p1_c233397_s2026_fullfit_swa.pt`

- [x] **`04_技术方案文档/`**
  - [`docs/TECHNICAL_REPORT.md`](file:///wanqing-develop/luowenjing/18/docs/TECHNICAL_REPORT.md)（完整技术方案报告）
  - [`docs/RESIDENT_V7_PIPELINE_ARCHITECTURE.md`](file:///wanqing-develop/luowenjing/18/docs/RESIDENT_V7_PIPELINE_ARCHITECTURE.md)（单图端到端数据流全拆解）
  - [`docs/FINAL_DEFENSE_HIGHLIGHTS.md`](file:///wanqing-develop/luowenjing/18/docs/FINAL_DEFENSE_HIGHLIGHTS.md)（主观评审 40 分得分点速查与答辩 Q&A）
  - [`docs/SERVING_AND_TAT_JIMU_ADAPTER.md`](file:///wanqing-develop/luowenjing/18/docs/SERVING_AND_TAT_JIMU_ADAPTER.md)（HTTP 服务与极目/TAT/VQA 适配层规范）
  - [`docs/SUBMISSION_CHECKLIST.md`](file:///wanqing-develop/luowenjing/18/docs/SUBMISSION_CHECKLIST.md)（本清单）

- [x] **`05_自评测报告/`**
  - [`docs/SELF_EVALUATION_REPORT.md`](file:///wanqing-develop/luowenjing/18/docs/SELF_EVALUATION_REPORT.md)
  - 配套定量审计证据：`resident_v8_boundary_oof_deconf_v6_boundary_summary.json`、`resident_v8_boundary_video_oof_audit.json`、`resident_v8_boundary_cost_pareto_oof.json`、`resident_v8_boundary_deploy_env_check.json`、`resident_v8_boundary_attribution_audit.json`、`resident_v8_boundary_http_service_audit.json`

- [x] **`06_AI使用说明/`**
  - [`docs/AI_USAGE_STATEMENT.md`](file:///wanqing-develop/luowenjing/18/docs/AI_USAGE_STATEMENT.md)（定量人机分工、四大真实协同攻坚案例与持久化防过拟合工程护栏）

---

## 2. 2026-11-20 单卡全项验收复核账本（`resident_v8_boundary` + `deconf_v6_boundary`，线上 `54.77 / 65`）

- [x] **断网部署环境自检与容器化交付**：`python3 code/r55/check_deploy_env.py --set resident_v8_boundary --decoder deconf_v6_boundary` $\rightarrow$ **65 项检查：`0 FAIL, 0 WARN -> READY`**；提供离线生产 [`Dockerfile`](file:///wanqing-develop/luowenjing/18/Dockerfile)。
- [x] **100% 从头独立推理与零训练读取**：`test_150`（单卡全量 150 图）与 `final_644`（644 图）全部从最新 FullFit 权重独立生成，`R55_AUDIT_TRAIN_READS=1` 全程训练文件打开次数 **`0`**，`N=1 == N=Batch` 差异 **`0/900`** 与 **`0/3,864`**（`max_q_diff <= 3.72e-13`），且启用 `v16[pecoreg]` 纯 PyTorch `INT8-g128` 量化与比特精确分块后全量 `test_150` 复测严格 **`0 / 900` 格差异**。
- [x] **单卡 24GB 显存余量与耗时预算**：`resident_v8_boundary` 去重参数 **`13.262B <= 30B`**，常驻 `15.78 GB`、PyTorch 推理峰值 `16.95 GB`、`nvidia-smi` 加载峰值 `19.98 GiB` / **推理峰值 `18,930 MiB` (`18.49 GiB < 24 GiB`, 单张 RTX 4090 余量 `~5.50 GiB`)**，全量 `test_150` 实测均值 **`3.03s`**、P95 **`4.23s`**、最大 **`6.93s <= 15s`**（12 图带同步压测中位 `3.41s` / 均值 `3.66s`）；备选纯视觉档 `resident_v8_boundary_novlm`（线上 `54.52 / 65`）`nvidia-smi` 峰值 **`12.75 GiB`**、中位 **`2.39s`**。
- [x] **主观语义归因全量质检**：`test_150`（327/327）与 `final_644`（1,346/1,346）标签描述 **100.0% 一致**，Clean `20/20` & `110/110`，字数最大 `40 <= 40`（`0` 超长、`0` 截断、`0` 自相矛盾、`0` 零全局值 Bug），物理实测支持率 **`36.4%` / `40.2%`**，唯一理由占比 **`94.5%` / `87.4%`**，真图局部定位物理对比命中率 **`61.5%`**（`final_644`），可视化拼接图 `_sheet.jpg` 已刷新，单帧因果不确定性说明 100% 覆盖。
- [x] **在线/批量双后端 HTTP 微服务实测**：`FastAPI` 与标准库 `stdlib` 均以 `--set resident_v8_boundary --decoder deconf_v6_boundary` 通过 `/healthz`、`/infer`（路径 & Base64）、`/infer_batch`（视频多帧聚合）与异常码测试（404/400），返回标签与提交 JSON 100% 一致，含归因实时显存 **`18.56 GiB`**（含冷启动全链路峰 **`19.98 GiB`**），`SIGTERM` 退出码 `0`。
