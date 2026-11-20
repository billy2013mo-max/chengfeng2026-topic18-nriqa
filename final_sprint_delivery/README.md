# 最后冲刺与最终交付一站式总目录 (`final_sprint_delivery/`)

本目录已按 **最终定稿旗舰主推版 `resident_v8_boundary` + `deconf_v6_boundary`（`test_150` 线上实测再破历史新高 `54.77 / 65`：正向缺陷 `17.46/25`、P0 识别 `17.31/20`、Clean Score `20.00/20` 满分；8 折零泄漏 Nested OOF `orig = 52.665 / all = 51.958` 全库最高，`Perm654(Pos<15) = 0.00%` / `ClBoot654 = 0.08%`）** 完成全链路代码、无损显存与加速优化（`18.49 GiB` NVML 推理峰、`3.03s/图` 且 `0/900` 格差异）、离线 `Dockerfile`、升级版实测语义归因旁路、审计账本、答辩文档与终评 ZIP 压缩包的 100% 同步定稿（`2026-11-20`）。

---

## 目录结构与文件速查

### 1. [`00_最终提交压缩包/`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/00_最终提交压缩包)（2026-11-20 `resident_v8_boundary + deconf_v6_boundary` `54.77` 终极满分定稿包）
- [`topic18_submission_full_20261120_v8.zip`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/00_最终提交压缩包/topic18_submission_full_20261120_v8.zip)（含全部 **14 个 Full-Fit 微调权重** + 完整已编译校验代码闭包 + 离线 [`Dockerfile`](file:///wanqing-develop/luowenjing/18/Dockerfile) + `bitsandbytes` 离线库 + `v8` 预测文件与升级版归因旁路 + `resident_v8_boundary_novlm` `54.52` 轻量档解码器 + 全套文档/ADR 与审计账本，**直接用于终评完整提交**）
- [`topic18_submission_lite_20261120_v8.zip`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/00_最终提交压缩包/topic18_submission_lite_20261120_v8.zip)（不含大权重的轻量归档包）
  *(注：前序 `20261119_v8` 与历史压缩包均已安全归档至 `archive_legacy/` 留底)*

### 2. [`01_核心预测结果JSON/`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/01_核心预测结果JSON)
- **⭐ 最终定稿旗舰交付版（`resident_v8_boundary + deconf_v6_boundary`，线上 `54.77 / 65`：正向 `17.46 / P0 17.31 / Clean 20.00`）**：
  - **终评 644 图主交付**：[`pred_nriqa_dataset_final_644_r55_resident_v8_boundary_deconf_v6_boundary.json`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/01_核心预测结果JSON/pred_nriqa_dataset_final_644_r55_resident_v8_boundary_deconf_v6_boundary.json)（Clean `110/644`，阳性 `1,346/3,864`，`N=1 == N=644` 差异 `0/3,864`，SHA256 `2f6d188ea648a662...`）
  - **公开 150 图主交付**：[`pred_nriqa_dataset_test_r55_resident_v8_boundary_deconf_v6_boundary.json`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/01_核心预测结果JSON/pred_nriqa_dataset_test_r55_resident_v8_boundary_deconf_v6_boundary.json)（Clean `20/150`，阳性 `327/900`，`N=1 == N=150` 差异 `0/900`，SHA256 `93a39abea3d4c812...`）
- **历史演进对照版**：`deconf_v5_boundary`（Phase 1）、`deconf_v4_tile`（前序线上 `54.68`）、`deconf_v3`、`deconf_v2`（线上 `54.44`）、`ref_543`（线上 `54.30`）

### 3. [`02_主观分加分旁路_语义归因与极目视频载荷/`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/02_主观分加分旁路_语义归因与极目视频载荷)（全部已按 `resident_v8_boundary + deconf_v6_boundary` 重新生成，100.0% 格级对齐）
- **可视化缺陷定位核验拼图**：[`final644_attribution_v8_sheet.jpg`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/02_主观分加分旁路_语义归因与极目视频载荷/final644_attribution_v8_sheet.jpg)、[`test150_attribution_v8_sheet.jpg`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/02_主观分加分旁路_语义归因与极目视频载荷/test150_attribution_v8_sheet.jpg)
- **全量实测物理归因与因果不确定性说明（`+5` 语义分，100.0% 锁定 `v8_boundary` 标签）**：[`final644_attribution_v8.json`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/02_主观分加分旁路_语义归因与极目视频载荷/final644_attribution_v8.json)（`1,346/1,346` 一致）、[`test150_attribution_v8.json`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/02_主观分加分旁路_语义归因与极目视频载荷/test150_attribution_v8.json)（`327/327` 一致）
- **极目 / TAT 标准质检载荷与多帧视频聚合摘要（`S_deploy` 15 分）**：`final644_tat_jimu_v8.json`、`final644_video_summary_v8.json`、`test150_tat_jimu_v8.json`、`test150_video_summary_v8.json`

### 4. [`03_最后冲刺核心代码与脚本/`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/03_最后冲刺核心代码与脚本)
- 一键推理入口：`run_final_infer.sh`（默认已切至 `resident_v8_boundary` + `deconf_v6_boundary`，真身位于 [`/wanqing-develop/luowenjing/18/run_final_infer.sh`](file:///wanqing-develop/luowenjing/18/run_final_infer.sh)）
- 全维度边界攻坚核心源码：`train_boundary_specialist.py`（Stage 2B 边界专精双塔）、`dual_vlm_shared_4090.py`（单 4-bit 底座双适配器共享 VLM）、`upgrade_deconf_v6_boundary.py`（Pre-LRO 零稀释融合校准）、`upgrade_deconf_v5_boundary.py`、`upgrade_deconf_v4_tile.py`、`run_v8_boundary_fresh_infer.py`、`build_v8_delivery_sidecars_and_audits.py`、`audit_v8_http_service.py`
- 常驻推理引擎、服务与打包：`final_engine.py`、`serve_engine.py`、`check_deploy_env.py`、`package_submission.py`、`measured_evidence.py`、`subjective_engineering.py`

### 5. [`04_最终校准解码器与实测审计账本/`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/04_最终校准解码器与实测审计账本)
- **主推解码器与离线资产**：`resident_v8_boundary_decoder_deconf_v6_boundary.pkl` / `.json`、`resident_v8_boundary_clean_footprint.npz`、`resident_v8_boundary_video_aggregator.pkl`
- **2026-11-20 `resident_v8_boundary` 全量实测账本**：
  - `resident_v8_boundary_oof_deconf_v6_boundary_summary.json`（8 折 `ORIG 52.665 / ALL 51.958`，`Perm654<15 = 0.00%`）
  - `resident_v8_boundary_bench_half.json`（`13.262B`，常驻 `15.78 GB`，`nvidia-smi` 推理峰值 `18.49 GiB` / 加载峰值 `19.98 GiB < 24 GiB`，12 图压测均耗 `3.66s` / `test_150` 全量均耗 `3.03s`、最大 `6.93s`、`0/900` 格差异）
  - `resident_v8_boundary_novlm_bench_half.json`（纯视觉 12 成员轻量档，线上 `54.52 / 65`，`nvidia-smi` 峰值 `12.75 GiB`，中位 `2.39s`）
  - `resident_v8_boundary_deploy_env_check.json`（**65/65 READY**，`0 FAIL, 0 WARN`）
  - `resident_v8_boundary_attribution_audit.json`（`327/327` & `1,346/1,346` 100.0% 一致，实测支持率 `36.4% / 40.2%`，唯一理由率 `94.5% / 87.4%`，局部框对比命中 `61.5%`，零全局值 Bug `0`）
  - `resident_v8_boundary_http_service_audit.json`（FastAPI & stdlib 双后端全通过，实时显存 `18.56 GiB` / 全链路峰值 `19.98 GiB`，`SIGTERM` 退出码 `0`）
  - `resident_v8_boundary_video_oof_audit.json`、`resident_v8_boundary_cost_pareto_oof.json`

### 6. [`05_全套答辩与主观评审文档/`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/05_全套答辩与主观评审文档)（全部已刷新至 `resident_v8_boundary + deconf_v6_boundary` `54.77` 定稿版）
- **答辩首选速查手册**：[`FINAL_DEFENSE_HIGHLIGHTS.md`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/05_全套答辩与主观评审文档/FINAL_DEFENSE_HIGHLIGHTS.md)
- 完整技术方案 [`TECHNICAL_REPORT.md`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/05_全套答辩与主观评审文档/TECHNICAL_REPORT.md)、单图架构拆解 [`RESIDENT_V7_PIPELINE_ARCHITECTURE.md`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/05_全套答辩与主观评审文档/RESIDENT_V7_PIPELINE_ARCHITECTURE.md)、自评测报告 [`SELF_EVALUATION_REPORT.md`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/05_全套答辩与主观评审文档/SELF_EVALUATION_REPORT.md)、极目/TAT 适配规范 [`SERVING_AND_TAT_JIMU_ADAPTER.md`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/05_全套答辩与主观评审文档/SERVING_AND_TAT_JIMU_ADAPTER.md)、AI 使用说明 [`AI_USAGE_STATEMENT.md`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/05_全套答辩与主观评审文档/AI_USAGE_STATEMENT.md) 与核对清单 [`SUBMISSION_CHECKLIST.md`](file:///wanqing-develop/luowenjing/18/final_sprint_delivery/05_全套答辩与主观评审文档/SUBMISSION_CHECKLIST.md)
