# 【2026乘风挑战赛·命题18】晴羽观感体系·画质BadCase的无参检测模型

本仓库为 **【2026乘风挑战赛】命题18：晴羽观感体系·画质BadCase的无参检测模型** 的完整代码库、收敛训练产物、零泄漏交叉验证审计、主观语义归因与单卡 RTX 4090 一键推理与微服务交付方案。

> **最终定稿主推交付架构（2026-11-20）**：**`resident_v8_boundary` 共享主干 + 双适配器 VLM 14 专精成员集成 + `deconf_v6_boundary` 全维度 `P0/P1/P2` 边界 Pre-LRO 零稀释校准器**（[`code/r55/final_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/final_engine.py)、[`code/r55/train_boundary_specialist.py`](file:///wanqing-develop/luowenjing/18/code/r55/train_boundary_specialist.py)、[`code/r55/dual_vlm_shared_4090.py`](file:///wanqing-develop/luowenjing/18/code/r55/dual_vlm_shared_4090.py)、[`code/r55/upgrade_deconf_v6_boundary.py`](file:///wanqing-develop/luowenjing/18/code/r55/upgrade_deconf_v6_boundary.py)）。
> - **客观指标**：在全部 **3,573 张人工标注真图** 严格按视频/场景分组的 **8-Fold 嵌套盲测（Nested OOF, `Doc2_Strict` / 65 分）** 上得分 **`orig = 52.665 / all = 51.958`（全库新高）**（`25·S_pos = 16.22`, `20·S_p0 = 16.60`, `20·S_clean = 19.84`，`mAP14 = 0.6695`，`Perm654(Pos<15) = 0.00%` / `ClBoot654 = 0.08%`），四项官方准入基线超高余量达标，公开打榜集实测再破新高 **`54.77 / 65`**（`正向缺陷 17.46/25`、`P0识别 17.31/20`、`Clean Score 20.00/20` 满分）。
> - **单张 RTX 4090（24 GB）成本与延迟**：去重激活参数 **`13.262B <= 30B`**；通过 **冻结层 `torch.equal` 无损显存去重（省 `5.81 GB`）+ `DualAdapterQwen3VL4090` 单 4-bit 底座双适配器共享（双 8B VLM 合计仅 `5.62 GB`，省 `3.40 GB`）+ VLM 词表 Host 驻留（省 `1.16 GB`）+ 纯 PyTorch 对称 `INT8-g128` 量化 `v16` 独有 `PE-Core-G14`（省 `1.72 GiB`，`test_150` 严格 `0/900` 格差异）+ 比特精确激活分块**，全 14 成员常驻显存仅 **`15.78 GB`**、PyTorch 推理峰值 **`16.95 GB`**、`nvidia-smi` 加载峰值 **`19.98 GiB`**、**推理峰值仅 `18,930 MiB`（`18.49 GiB < 24 GiB`，单张 RTX 4090 余量 `~5.50 GiB`）**；全量 `test_150` 单图端到端均值 **`3.03s`**、P95 **`4.23s`**、最大 **`6.93s <= 15s`**（12 图带同步压测中位 `3.41s` / 均值 `3.66s`）。
> - **一键切换纯视觉轻量档**：**`resident_v8_boundary_novlm`（12 成员纯视觉，线上实测 `54.52 / 65`）** 仅 `8.910B` 参数、常驻 **`10.19 GB`**、`nvidia-smi` 推理峰值仅 **`13,060 MiB`（`12.75 GiB`）**、中位耗时 **`2.39s`**（Nested OOF `52.327 / 51.864`）；另备 10 成员 `resident_v7_novlm`（`7.569B`，`14.36 GiB`，`2.59s`）。

---

## 1. 交付产物清单（严格对齐命题文档第 5 节《提交要求》六大目录）

| 官方产物类别 | 归档目录（压缩包内） | 仓库原路径 / 核心文件 |
| :--- | :--- | :--- |
| **1. 预测结果文件【核心交付物】** | `01_预测结果文件_核心交付物/` | • 终评 644 图主交付：[`results/final/pred_nriqa_dataset_final_644_r55_resident_v8_boundary_deconf_v6_boundary.json`](file:///wanqing-develop/luowenjing/18/results/final/pred_nriqa_dataset_final_644_r55_resident_v8_boundary_deconf_v6_boundary.json)<br>• 公开 150 图主交付：[`results/final/pred_nriqa_dataset_test_r55_resident_v8_boundary_deconf_v6_boundary.json`](file:///wanqing-develop/luowenjing/18/results/final/pred_nriqa_dataset_test_r55_resident_v8_boundary_deconf_v6_boundary.json)<br>• `sidecars_语义归因与极目视频载荷/`（含 `644` 与 `150` 全量 `v8` 归因 JSON、定位拼接图 `_sheet.jpg`、极目/TAT 载荷与视频级聚合摘要） |
| **2. 可执行代码 & 一键脚本** | `02_可执行代码/` | [`run_final_infer.sh`](file:///wanqing-develop/luowenjing/18/run_final_infer.sh)、[`requirements_infer.txt`](file:///wanqing-develop/luowenjing/18/requirements_infer.txt)、[`code/r55/final_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/final_engine.py)、[`code/r55/dual_vlm_shared_4090.py`](file:///wanqing-develop/luowenjing/18/code/r55/dual_vlm_shared_4090.py)、[`code/r55/train_boundary_specialist.py`](file:///wanqing-develop/luowenjing/18/code/r55/train_boundary_specialist.py)、[`code/r55/upgrade_deconf_v6_boundary.py`](file:///wanqing-develop/luowenjing/18/code/r55/upgrade_deconf_v6_boundary.py)、[`code/r55/serve_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/serve_engine.py)、[`code/r55/check_deploy_env.py`](file:///wanqing-develop/luowenjing/18/code/r55/check_deploy_env.py)、[`code/r55/validate_submission.py`](file:///wanqing-develop/luowenjing/18/code/r55/validate_submission.py) 及完整 Python 源码闭包 + `third_party/pydeps` |
| **3. 模型产物（权重 + 解码器 + 基座说明）** | `03_模型产物/` | `checkpoints/`（14 个 full-fit 权重）、`decoders/`（[`resident_v8_boundary_decoder_deconf_v6_boundary.pkl`](file:///wanqing-develop/luowenjing/18/results/r55/final/resident_v8_boundary_decoder_deconf_v6_boundary.pkl)、[`resident_v8_boundary_clean_footprint.npz`](file:///wanqing-develop/luowenjing/18/results/r55/final/resident_v8_boundary_clean_footprint.npz)、[`resident_v8_boundary_video_aggregator.pkl`](file:///wanqing-develop/luowenjing/18/results/r55/final/resident_v8_boundary_video_aggregator.pkl) 及 `resident_v7 / novlm` 备选档）与 `README_模型与基座版本说明.md` |
| **4. 技术方案文档** | `04_技术方案文档/` | [`docs/TECHNICAL_REPORT.md`](file:///wanqing-develop/luowenjing/18/docs/TECHNICAL_REPORT.md)、[`docs/RESIDENT_V7_PIPELINE_ARCHITECTURE.md`](file:///wanqing-develop/luowenjing/18/docs/RESIDENT_V7_PIPELINE_ARCHITECTURE.md)、[`docs/FINAL_DEFENSE_HIGHLIGHTS.md`](file:///wanqing-develop/luowenjing/18/docs/FINAL_DEFENSE_HIGHLIGHTS.md)、[`docs/SERVING_AND_TAT_JIMU_ADAPTER.md`](file:///wanqing-develop/luowenjing/18/docs/SERVING_AND_TAT_JIMU_ADAPTER.md)、[`docs/SUBMISSION_CHECKLIST.md`](file:///wanqing-develop/luowenjing/18/docs/SUBMISSION_CHECKLIST.md) |
| **5. 自评测报告** | `05_自评测报告/` | [`docs/SELF_EVALUATION_REPORT.md`](file:///wanqing-develop/luowenjing/18/docs/SELF_EVALUATION_REPORT.md) 及嵌套 OOF、多帧视频误报抑制、断网自检、归因一致性与 HTTP 服务实测 JSON |
| **6. AI 使用说明** | `06_AI使用说明/` | [`docs/AI_USAGE_STATEMENT.md`](file:///wanqing-develop/luowenjing/18/docs/AI_USAGE_STATEMENT.md) |

---

## 2. 一键推理与评测集复现（仓库根目录或归档解压后均可完全离线运行）

```bash
# 0) 推理前一键环境与权重完整性自检（65 项检查，0 FAIL / 0 WARN）：
python3 code/r55/check_deploy_env.py --set resident_v8_boundary --decoder deconf_v6_boundary --gpu 0

# 1) 在仓库根目录一键运行评测集推理 + 官方格式校验（默认开启 --half --share-weights --embed-on-host）：
bash run_final_infer.sh <评测集模板JSON> <评测集图片目录> <输出预测JSON> [GPU_ID=0] [SET_NAME=resident_v8_boundary] [DECODER=deconf_v6_boundary]

# 2) 若从提交压缩包解压后运行：
bash 02_可执行代码/run_final_infer.sh <评测集模板JSON> <评测集图片目录> <输出预测JSON> 0 resident_v8_boundary deconf_v6_boundary
```

示例（同时可选导出主观语义归因、极目/TAT 载荷与视频级聚合摘要，**主预测 JSON 保证 100% 逐字节不变**）：
```bash
R55_EXTRA_ARGS="--emit-attribution attr_out.json --tat-out tat_out.json --video-mode" \
bash run_final_infer.sh \
  results/v2_full/template_150.json \
  data/test_150 \
  pred_test_150.json \
  0 \
  resident_v8_boundary \
  deconf_v6_boundary
```

脚本自动完成：
1. 调用 [`code/r55/final_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/final_engine.py) 逐图独立执行 `2×` 水平翻转 TTA + 在线原生物理特征提取 + `deconf_v6_boundary` 校准解码（严格 `N=1 == N=Batch`，不使用任何历史推理 JSON 或特征缓存，训练数据文件读取 **0 次**）；
2. 调用 [`code/r55/validate_submission.py`](file:///wanqing-develop/luowenjing/18/code/r55/validate_submission.py) 严格校验样本数量、ID 集合对齐与六维标签合法性。

---

## 3. 主观评分项核心工程能力速览（`S_cost` 20 分 + `S_deploy` 15 分 + 语义归因 `+5` 分）

| 工程能力模块 | 入口命令 / 核心源码 | 2026-11-20 单卡实测指标 |
| :--- | :--- | :--- |
| **断网部署预检 & 容器镜像** | `python3 code/r55/check_deploy_env.py` & [`Dockerfile`](file:///wanqing-develop/luowenjing/18/Dockerfile) | **65 项检查：`0 FAIL, 0 WARN -> READY`**（含 14 个权重、本地基座、RTX 4090 `sm_89` 兼容与离线生产 `Dockerfile`） |
| **四重无损显存压缩与分块** | `--half --share-weights --embed-on-host` + `DualAdapterQwen3VL4090` + 纯 PyTorch `INT8-g128` (`v16[pecoreg]`) | 冻结层去重省 **`5.81 GB`** + 双 VLM 共享底座省 **`3.40 GB`** + Host 词表省 **`1.16 GB`** + `INT8-g128` 省 **`1.72 GiB`**，`nvidia-smi` 推理峰值 **`18,930 MiB` (`18.49 GiB < 24 GiB`，余量 `~5.50 GiB`)**，加载峰值 **`19.98 GiB`**；全量 `test_150` 严格 **`0 / 900` 格差异**，均耗 **`3.03s` / 最大 `6.93s <= 15s`** |
| **纯视觉高吞吐轻量档** | `--set resident_v8_boundary_novlm --decoder deconf_v6_boundary` | `8.910B` 参数，线上实测 **`54.52 / 65`**，`nvidia-smi` 峰值 **`13,060 MiB` (`12.75 GiB`)**，中位 **`2.39s/图`**，Nested OOF **`52.327`** |
| **推理与训练数据零耦合** | `R55_AUDIT_TRAIN_READS=1` + `resident_v8_boundary_clean_footprint.npz`（75.2 KiB） | 全量推理期间训练标注/图像/OOF 文件打开次数严格为 **`0`**；`N=1 == N=Batch` 逐格零差异（`0/900` & `0/3,864`） |
| **双后端在线/批量 HTTP 服务** | `python3 code/r55/serve_engine.py --gpu 0 --port 8055` | FastAPI 与标准库 `stdlib` 双后端实测通过，`/infer` & `/infer_batch` 与离线提交标签 **100% 一致**，含归因实时显存 **`18.56 GiB`**（全链路峰 **`19.98 GiB`**），`SIGTERM` 退出码 `0` |
| **多帧视频误报抑制聚合器** | `EffectiveFrameORVideoAggregator`（[`code/r55/subjective_engineering.py`](file:///wanqing-develop/luowenjing/18/code/r55/subjective_engineering.py)） | 真实 320 段多帧视频 **`84.45%` P0 召回 + `100.0%` Clean 特异度**（`Doc2=51.922`），高密度抽样 Clean 误报率降低 `2.6×~3.4×` |
| **实测证据归因与不确定性表达** | [`code/r55/measured_evidence.py`](file:///wanqing-develop/luowenjing/18/code/r55/measured_evidence.py) + `--emit-attribution` | 标签一致性 **`100.0%`**（`327/327` & `1,346/1,346`），`≤ 40` 字（`0` 超长、`0` 截断、`0` 自相矛盾、`0` 零全局值 Bug），物理实测支持率 **`36.4% / 40.2%`**，去模板唯一理由占比 **`94.5% / 87.4%`**，局部框物理对比命中率 **`61.5%`**，`100%` 覆盖因果不确定性 |
