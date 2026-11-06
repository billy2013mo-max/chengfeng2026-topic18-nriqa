# d1a437a 最终版本执行报告（2026-11-06）

## 1. 执行摘要

本轮按 `POST_5289_ITERATION_AND_STATUS_REPORT.md` 的待办顺序执行：同步 `d1a437a` → 修复数据/提交口径 → 新代码自检 → 重训 V12/V13 → 盲测门禁 → 构建最终提交 → 单卡部署实测。

**最终提交候选**：

| 项目 | 值 |
| :--- | :--- |
| 文件 | `pred_nriqa_dataset_test.json`（副本 `results/final/pred_nriqa_dataset_test_FINAL_d1a437a.json`） |
| SHA-256 | `c72849e8e91b652f9941ad9d41fc6c704d8cbe183e524405be6735ba501ab5cc` |
| 基线 | `b983d87` 的 52.89 + 10 处 QualiRAG 改动（15 张 Clean） |
| 本轮新增 | 8 处经≥3/6 专家共识的微创修订 |
| 不变量 | Clean 图片集合、P0 图片集合、花屏 P0 数量与基线完全一致 |
| s16–s20 盲测门禁 | Pooled **+0.524**，95% CI **[+0.264, +0.785]**，p<0.0001 |

> 重要口径：线上已实测的 52.89 来自 Phase-1 文件（16 张 Clean）；`b983d87` 的 10 处改动与本轮 8 处改动都尚未线上验证。建议用剩余提前评测机会验证，并保留回滚文件。

---

## 2. 代码与数据完整性修复

1. **同步远程**：本地从 `2969b0f` 快进到 `d1a437a`。
2. **确认 V12/V13 的 full-split 泄漏护栏已存在**：
   - V12 `train_v12_abmil_pure_real.py` 与 V13 `train_vlm_pure_real_corn.py` 都会把 val 中与 train 重叠的图片从 train 剔除；
   - 旧版 full 产物（V12 `val_total=63.09`、V13 `55.03`）属于训练集内成绩，已全部归档到 `results/_archive_2969b0f/`，不再参与最终融合。
3. **提交基线显式化**：`build_55plus_ultimate_submission.py` 新增 `BASELINE_5289_PATH`（默认 Phase-1）与 `FREEZE_DIFF_FROM`，避免把 Phase-1、b983、旧 55+ 三个文件混用。
4. **P0/修订预算**：新增 `--max-p0-upgrades 4`、`--max-p0-downgrades 0`，默认不降级任何已验证 P0；修订按置信度排序并受预算约束，从构造上保证 Gate 3（P0 漂移 ≤4）。
5. **专家共识阈值**：P1↔P2 与 无问题→P2 的专家票数门槛统一收紧到 **≥3/6**。
6. **SWA 记录修正**：SWA 被采用时记录 SWA 自己的分数/阈值，不再用旧的 `max(best, swa)` 分数配新阈值。
7. **新增自检脚本** `code/finetune/test_d1a437a_sanity.py`：27 项 CPU 检查全部通过（零初始化恒等、τ 边界、金字塔 67/83 hook、缓存清理、Clean 软标签零泄漏、head 列序 `[clean,P2,P1,P0]`）。

---

## 3. 新模型重训结果

### V13 Qwen3-VL-8B + Q-Align + 显著性门控 + 软 CORN（每折 2 epoch）

| Split | 新 test total | S_pos | S_p0 | S_clean | 旧 test total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| s16 | **51.82** | 0.640 | 0.792 | 1.000 | 43.12 |
| s17 | 37.83 | 0.621 | 0.782 | 0.333 | 51.25 |
| s18 | **51.28** | 0.611 | 0.800 | 1.000 | 35.72 |
| s19 | 44.42 | 0.621 | 0.769 | 0.675 | 50.94 |
| s20 | 50.61 | 0.624 | 0.751 | 1.000 | 52.32 |
| full（300 张不重叠 holdout） | **50.96** | 0.628 | 0.764 | 1.000 | 55.03（泄漏） |

### V12 ABMIL + 三层金字塔 + 非对称软标签 + SWA（每折 4 epoch）

| Split | 新 test total | S_pos | S_clean | 旧 test total |
| :--- | ---: | ---: | ---: | ---: |
| s16 | 48.92 | 0.576 | 1.000 | 47.70 |
| s17 | **49.67** | 0.562 | 1.000 | 41.39 |
| s18 | 42.70 | 0.567 | 0.632 | 48.00 |
| s19 | **50.14** | 0.605 | 1.000 | 29.07 |
| s20 | 48.97 | 0.570 | 1.000 | 40.70 |
| full（300 张不重叠 holdout） | 49.40 | 0.589 | 1.000 | 63.09（泄漏） |

**结论**：V12 的 s16–s20 均值从 41.37 提升到 48.08；V13 从 46.64 提升到 47.19，且 s16/s18 的 Clean 崩溃被修复。V13 在 s17 仍有 Clean 假阳性（S_clean=0.333），因此质量过滤会排除该折。

---

## 4. 五折盲测门禁（最终判定）

```
Split   | Phase1(50:50) | Ultimate(55+) | Gain
s16     |     52.13     |     52.74     | +0.62
s17     |     51.55     |     52.68     | +1.13
s18     |     52.19     |     52.32     | +0.13
s19     |     51.77     |     52.00     | +0.23
s20     |     52.11     |     52.65     | +0.54

POOLED (N=5000): 51.958 -> 52.482
  S_pos  15.41 -> 15.78
  S_p0   16.55 -> 16.71
  S_clean 20.00 -> 20.00
Paired Bootstrap: +0.524, 95% CI [+0.264, +0.785], p < 0.0001
```

家族平均路由权重：Fam6 Qwen3-VL **0.199**、Fam7 V12 ABMIL **0.180**、Fam3 PE-Core-L14 0.168、Fam5 InternViT-6B 0.167。所有单折增益均为正。

---

## 5. 最终提交构建与审计

- 基线：`results/final/pred_nriqa_dataset_test_qualirag_consensus.json`（b983d87，15 Clean / 102 P0 cells / 347 positive cells）。
- 冻结：b983 相对 Phase-1 的 10 处 QualiRAG 黄金改动全部冻结，不再修订。
- 新增 8 处修订（全部专家票数 ≥3/6）：

| 图片 | 类别 | 修订 | 共识 |
| :--- | :--- | :--- | :--- |
| 200596734934_v8L3UtV9_6.70s.png | 过度生成 | 无问题 → P2 | 5/6 |
| 206111225892_...frame_0018.png | 纹理损失 | P1 → P2 | 6/6 |
| 181204643103_v7L3HiV2_18101.png | 过锐 | P2 → P1 | 3/6 |
| 181422473010_v7L3HiV2_3881.png | 过锐 | P1 → P2 | 6/6 |
| 204474242952_...frame_0027.png | 模糊 | P2 → 无问题 | 6/6 |
| 181258970129_v7L3HiV2_7164.png | 过锐 | P2 → 无问题 | 5/6 |
| 200646809044_v8L3HiV9_25.11s.png | 过锐 | 无问题 → P2 | 3/6 |
| 200648357808_frame1317.png | 纹理损失 | P1 → P2 | 3/6 |

7 道 Gate 全部通过：

```
[Gate 1] All-Clean Images Locked : 15 / 150 (identical)
[Gate 2] 花屏_P0 Locked          : 10 / 10 (identical)
[Gate 3] Total P0 Count          : 102 (delta +0)
[Gate 4] Total Positive Cells    : 347 (delta +0)
[Gate 5] Ranked Upgrades         : 8 cells
```

格式校验：150 个唯一 ID、与模板顺序完全一致、六类字段齐全、虚化突兀/花屏仅含 P0/无问题。

---

## 6. 单卡部署实测（真实端到端，非演示）

`code/finetune/deploy_single_gpu_student_pipeline.py` 已改为真实推理链路：

```
输入图片 → SigLIP2-SO400M + PE-Core-L14 + C-RADIOv4-H (28416 维)
        → 蒸馏学生 CORN 塔 → 14 通道 → 分层阈值解码
        → 3×3 空间归因 + 成因解释
```

实测（H200 单卡，20 张 test_150）：

| 指标 | 实测 | 赛题要求 |
| :--- | ---: | ---: |
| 平均端到端时延 | **332.92 ms/图** | < 15,000 ms |
| 吞吐 | 3.00 图/秒 | — |
| 峰值显存 | **4,817.6 MB** | < 24,576 MB（RTX 4090） |
| 学生权重 | `results/final/single_gpu_student_corn_weights.pt` | — |

---

## 7. 产物清单

| 路径 | 说明 |
| :--- | :--- |
| `pred_nriqa_dataset_test.json` | **最终提交候选** |
| `results/final/pred_nriqa_dataset_test_FINAL_d1a437a.json` | 最终候选副本（同 SHA-256） |
| `results/final/pred_nriqa_dataset_test_55plus_ultimate.json` | 同一最终候选的归档名 |
| `results/final/pred_nriqa_dataset_test_55plus_unconstrained.json` | 无约束候选（29 处路由改动，**不建议提交**） |
| `results/final/pred_candidate_base_phase1.json` | 备选：以 Phase-1 为基线的 11 处修订（16 Clean） |
| `results/final/pred_candidate_b983_unfrozen.json` | 备选：b983 基线、未冻结黄金改动的 16 处修订 |
| `results/final/pred_backup_*.json` | 本轮开始前的备份 |
| `results/_archive_2969b0f/` | 旧版泄漏 full 产物与旧 per-split 产物 |
| `logs_retrain/` | 12 个训练任务日志、最终 builder 日志、状态文件 |
| `code/finetune/test_d1a437a_sanity.py` | 27 项新代码自检 |
| `code/audit_final_submission.py` | 提交前后差异审计 |

---

## 8. 复现命令

```bash
# 1) 新代码自检
python3 -u code/finetune/test_d1a437a_sanity.py

# 2) V13（每个 split 单卡；full 用 data/split_all_full.csv）
torchrun --nproc_per_node=1 code/finetune/train_vlm_pure_real_corn.py \
  --split data/resplits/split_s16.csv --epochs 2 --batch 12 --max-pixels 802816 \
  --lr 2e-4 --lr-head 1e-3 --rank 32 --grad-ckpt \
  --out results/v13_vlm_real/vlm_corn_s16.json

# 3) V12（3 backbone / 5 views 快配置）
python3 -u code/finetune/train_v12_abmil_pure_real.py \
  --split data/resplits/split_s16.csv --tiles 2 --epochs 4 --batch 4 --grad-accum 2 \
  --backbones pecoreg_abmil:lastk4,cradio_abmil:lastk4,dinov3l_abmil:lastk4 \
  --out results/v12_abmil/v12_abmil_s16.json

# 4) 最终构建（基线 b983 + 冻结黄金改动）
BASELINE_5289_PATH=results/final/pred_nriqa_dataset_test_qualirag_consensus.json \
FREEZE_DIFF_FROM=results/final/pred_nriqa_dataset_test_phase1_breakthrough.json \
python3 -u code/build_55plus_ultimate_submission.py

# 5) 审计
python3 code/audit_final_submission.py \
  --candidate pred_nriqa_dataset_test.json \
  --baseline results/final/pred_nriqa_dataset_test_qualirag_consensus.json

# 6) 单卡部署实测
python3 -u code/finetune/deploy_single_gpu_student_pipeline.py --benchmark --benchmark-images 20
```

---

## 9. 风险与建议

1. **线上未验证**：本轮 8 处修订与 b983 的 10 处改动都只有离线证据。建议用剩余提前评测机会提交 `pred_nriqa_dataset_test_FINAL_d1a437a.json`，并记录三项子分；若 `S_clean` 下跌，立即回滚到 `pred_candidate_base_phase1.json` 或 b983 原件。
2. **V13 的 Clean 方差仍是最大隐患**：s17 的 `S_clean=0.333`、s18 的 `S_clean=0.632` 说明单折 Clean 假阳性会显著拖分。当前依赖质量过滤排除差折；后续建议增加按类 Clean 门控。
3. **无约束候选不要提交**：`55plus_unconstrained` 会新增 P0 并改动 P0 图片集合，违反已验证护栏。
4. **654 张最终评测集**：仓库目前只有 `data/test_150`，官方最终评测是 654 张。部署复评前需把推理脚本参数化为任意目录，并用同一模型生成。
5. **full 模型分数已诚实**：新 full 产物为 300 张不重叠 holdout，V12=49.40、V13=50.96；不要再引用旧的 63.09/55.03。
