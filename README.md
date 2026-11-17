# 【2026乘风挑战赛·命题18】晴羽观感体系·画质BadCase的无参检测模型

本仓库为 **【2026乘风挑战赛】命题18：晴羽观感体系·画质BadCase的无参检测模型** 的完整代码库、收敛训练产物、交叉验证审计与单卡 RTX 4090 一键推理方案。

> **当前收敛主推交付基线（2026-11-15）**：`resident_v6_core`（6 成员多家族集成：`cradio_v3` + `peg` + `v16` + `ntg_pe` + `p12_pe` + `p1_qy`，配合 `RoutedGateEUDecoder` 与 `--half` 常驻推理引擎 [`code/r55/final_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/final_engine.py)），总激活参数 **9.656B**（远低于 30B 上限），单卡常驻显存 **15.68 GB**、推理峰值 **17.64 GB**（`expandable_segments` 后 `nvidia-smi` 实测峰值：推理 **20,290 MiB = 19.81 GiB**、含加载 **20,688 MiB = 20.20 GiB**，原 21.16 GiB；均 <= 24 GB）、单图端到端推理均值 **2.07s**（P95 **3.81s**，含实测证据归因 **2.35s**，远低于 15s 上限）；在全部 3,573 张真图上的严格 8-Fold 嵌套盲测（Nested OOF, `Doc2_Strict`）得分 **`all=51.597 / orig=51.873`**（`Pos_orig=15.81`, `P0_orig=16.30`, `Cln_orig=19.77`），四项准入基线全部高余量达标。

---

## 1. 交付产物清单（按官方第 5 节《提交要求》分类）

| 官方产物类别 | 归档目录（压缩包内） | 仓库原路径 / 核心文件 |
| :--- | :--- | :--- |
| **1. 预测结果文件【核心交付物】** | `01_预测结果文件_核心交付物/` | [`results/final/pred_nriqa_dataset_test_r55_resident_v6_core_routed_gate_eu.json`](file:///wanqing-develop/luowenjing/18/results/final/pred_nriqa_dataset_test_r55_resident_v6_core_routed_gate_eu.json) |
| **2. 可执行代码 & 一键脚本** | `02_可执行代码/` | [`run_final_infer.sh`](file:///wanqing-develop/luowenjing/18/run_final_infer.sh)、[`code/r55/final_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/final_engine.py)、[`code/r55/validate_submission.py`](file:///wanqing-develop/luowenjing/18/code/r55/validate_submission.py)、`code/` 闭环 |
| **3. 模型产物（权重 + 解码器 + 基座说明）** | `03_模型产物/` | `checkpoints/`（6 个 full-fit 权重）、`decoders/`（[`resident_v6_core_decoder_routed_gate_eu.pkl`](file:///wanqing-develop/luowenjing/18/results/r55/final/resident_v6_core_decoder_routed_gate_eu.pkl) 等）及 `README_模型与基座版本说明.md` |
| **4. 技术方案文档** | `04_技术方案文档/` | [`docs/TECHNICAL_REPORT.md`](file:///wanqing-develop/luowenjing/18/docs/TECHNICAL_REPORT.md)、[`docs/RESIDENT_V6_PIPELINE_ARCHITECTURE.md`](file:///wanqing-develop/luowenjing/18/docs/RESIDENT_V6_PIPELINE_ARCHITECTURE.md)、[`docs/SUBMISSION_CHECKLIST.md`](file:///wanqing-develop/luowenjing/18/docs/SUBMISSION_CHECKLIST.md) |
| **5. 自评测报告** | `05_自评测报告/` | [`docs/SELF_EVALUATION_REPORT.md`](file:///wanqing-develop/luowenjing/18/docs/SELF_EVALUATION_REPORT.md)、[`results/r55/vlm_qy8/prune_v6_audit.json`](file:///wanqing-develop/luowenjing/18/results/r55/vlm_qy8/prune_v6_audit.json) |
| **6. AI 使用说明** | `06_AI使用说明/` | [`docs/AI_USAGE_STATEMENT.md`](file:///wanqing-develop/luowenjing/18/docs/AI_USAGE_STATEMENT.md) |

---

## 2. 一键推理与评测集复现（解压后或仓库内均可直接运行）

```bash
# 在仓库根目录：
bash run_final_infer.sh <评测集模板JSON> <评测集图片目录> <输出预测JSON> [GPU_ID=0] [SET_NAME=resident_v6_core] [DECODER=routed_gate_eu]

# 若从提交压缩包解压后运行：
bash 02_可执行代码/run_final_infer.sh <评测集模板JSON> <评测集图片目录> <输出预测JSON> 0 resident_v6_core routed_gate_eu
```

示例：
```bash
bash run_final_infer.sh \
  documents/eval_654_template.json \
  data/eval_654 \
  pred_eval_654.json \
  0 \
  resident_v6_core \
  routed_gate_eu
```

脚本将自动执行：
1. 调用 `code/r55/final_engine.py --set resident_v6_core --half --decoder routed_gate_eu` 对模板中的每张图片从头独立推理（2x 水平翻转 TTA + 在线原生分辨率物理特征计算，严格 `N=1 == N=150`，不使用任何历史缓存或历史推理 JSON）；
2. 调用 `code/r55/validate_submission.py` 严格校验输出 JSON 的 ID 集合与模板 100% 一致、无重复/缺失 ID、每张图完整包含六类合法标签。

---

## 3. 部署工程（2026-11-16 更新，不改变任何预测标签）

| 能力 | 入口 | 实测 |
| :--- | :--- | :--- |
| 部署环境自检（依赖锁、`bitsandbytes`、CUDA/`sm_89` 兼容、解码器/footprint/权重） | `python3 code/r55/check_deploy_env.py`、[`requirements_infer.txt`](requirements_infer.txt) | 50 项 0 FAIL 0 WARN |
| 显存余量 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（脚本与引擎默认）+ 每成员加载后 `gc`/`empty_cache` | `nvidia-smi` 推理峰值 22,278 → **20,290 MiB**（同机对照，−1,988 MiB）；`v16_chunk=1` 无显存收益且慢 0.41s/图，保留 `2` |
| 推理与训练数据解耦 | 归因 clean footprint + 洁净物理分位离线预计算 `results/r55/final/resident_v6_core_clean_footprint.npz`（48.5 KiB） | test_150 推理全程训练数据文件读取 **0** 次（审计钩子 `R55_AUDIT_TRAIN_READS=1`） |
| 在线 / 批量 HTTP 服务 | `python3 code/r55/serve_engine.py --set resident_v6_core --gpu 0`（FastAPI，缺依赖时自动回退标准库） | `/healthz` `/infer` `/infer_batch` 实测通过，标签与提交 JSON 全部一致，SIGTERM 干净退出 |
| 实测证据语义归因（+5 分项） | `--emit-attribution/--tat-out`（或 `R55_EXTRA_ARGS=...`），`code/r55/measured_evidence.py` | 每图归因 0.166s 均值 / 0.332s 最大；理由 ≤40 字（按构造，无截断）；开/关归因输出 JSON 字节一致 |
| 极目 / TAT / VQA 适配映射 | [`docs/SERVING_AND_TAT_JIMU_ADAPTER.md`](docs/SERVING_AND_TAT_JIMU_ADAPTER.md)（适配层建议，非平台内部规范） | — |

完整命令与数字：[`results/r55/v6_debate/claude_engineering_report.md`](results/r55/v6_debate/claude_engineering_report.md)。
