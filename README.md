# 【2026乘风挑战赛·命题18】晴羽观感体系·画质BadCase的无参检测模型 — 完整复现指南 (README)

> **最新状态（2026-11-06，commit `d1a437a`）**：当前最终提交候选为
> `pred_nriqa_dataset_test_FINAL_d1a437a.json`（b983d87 + 8 处专家共识修订，
> Clean/P0 集合不变）。五折盲测门禁 +0.524（p<0.0001）。
> 详细执行记录见 [`docs/D1A437A_FINAL_RESULT_REPORT.md`](docs/D1A437A_FINAL_RESULT_REPORT.md)，
> 提交与回滚记录见 [`docs/SUBMISSION_LEDGER.md`](docs/SUBMISSION_LEDGER.md)。
> 下文第 1–5 节为 V9 阶段的历史复现说明。

本仓库为 **【2026乘风挑战赛】命题18：晴羽观感体系·画质BadCase的无参检测模型** 的完整训练、合成数据构造、零泄漏交叉验证与一键推理代码库。

---

## 1. 方案核心亮点概览

针对快手消费链路中六类画质 BadCase（**模糊、纹理损失、过度生成、虚化突兀、过锐、花屏**）及四档严重程度（**P0 / P1 / P2 / 无问题**）的无参画质评估（NR-IQA）任务，本方案构建了 **V9 高分辨率空间异常池化与物理表征协同专精体系（HiRes Spatial-Anomaly & Physical-Prior Specialist Ensemble）**：

1. **多视觉基座互补表征（远低于 30B 训练规模上限）**：
   - 联合视觉感知基座 `PE-Core-G14/L14`、自监督几何结构基座 `DINOv3-ViT-L16`、多教师蒸馏全能基座 `C-RADIOv4-H`、细粒度语义基座 `SigLIP2-Giant/SO400M`、纯卷积局部高频基座 `ConvNeXtV2-L` 与 `EVA02-L-448`，只解冻最后 $k \in \{4,6\}$ 层 Transformer Block 并将浅层冻结为 `bfloat16`，兼顾跨场景泛化与抗过拟合。
2. **448px 多视角切块与空间异常池化（Spatial Anomaly Pooling）**：
   - 突破传统全局平均池化稀释局部伪影（如局部多余手指/局部虚化割裂/局部边缘过锐白边）的瓶颈，对每张图提取 `1×Global + 2×2 Tiles + 3×3 Tiles`（7~12 个 `448×448` 高清视角），融合 **Global Token + Tile Mean + Top-2 Tile Anomaly Max-Mean + Tile Spatial Std** 四重空间聚合表征。
3. **60 维原生分辨率物理信号分支（Native Physical Prior Branch）**：
   - 在图片缩放前直接于原生分辨率提取 60 维物理特征（多尺度 Laplacian 高频能量比、Unsharp 边缘白边过冲比、块效应/条纹色块异常度、前景-背景梯度过渡陡峭度、局部熵与频谱斜率），经 `LayerNorm + MLP` 注入分类头。
4. **按折源图隔离的零振铃等级阶梯合成管线（Zero-Ripple Severity Ladder Synthesis）**：
   - 在全部 **3,573 张公开开发真图（首批 1,902 张 + 补充 1,671 张）** 上采用 **8-Fold 场景隔离交叉验证**；每折仅使用该折训练集内的无问题底图构造 `Clean → P2 → P1 → P0` 严格单调退化阶梯，并采用无正弦波纹的 **Zero-Ripple Unsharp Halo** 消除 `过锐 ↔ 花屏` 混淆。
5. **14 通道解耦校准 + 全局无问题门控（Clean Gate）+ 视频级多帧聚合**：
   - 严格对齐新版评审标准（`25·S_pos + 20·S_p0 + 20·S_clean`），通过全局 `Clean Gate` 保护全无问题样本不被微弱 `P2/P1` 误伤，同时释放 `P0` 高风险样本召回率；支持单张 RTX 4090（24GB）端到端 ~1.2 秒/图高速推理（远低于 15 秒上限）及结构化异常区域/原因归因输出。

---

## 2. 目录结构说明

```text
.
├── README.md                                     # 一键复现说明与工程部署指南
├── docs/
│   ├── TECHNICAL_REPORT.md                       # ① 技术方案文档（问题分析/数据构造/训练超参/消融实验/部署成本）
│   ├── SELF_EVALUATION_REPORT.md                 # ② 自评测与8折交叉验证报告（含逐档P/R/F1详表）
│   ├── AI_USAGE_STATEMENT.md                     # ③ AI 使用说明（组委会统一要求）
│   ├── reports/                                  # 历史调研、阶段报告与 Word/PDF
│   └── archive/                                  # 聊天记录存档
├── documents/                                    # 赛题原文与空白提交模板
├── NEXT55_HANDOFF.md                             # 已停用的 next55 交接（指标结论已过时）
├── R55_HANDOFF.md                                # 当前实验交接，按此执行
├── run_next55.sh
├── run_r55.sh
├── code/
│   ├── decision_eval/
│   │   ├── common.py                             # 六类缺陷与等级常量定义
│   │   └── comp_metrics.py                       # 官方竞赛评分计算模块 (S_pos / S_p0 / S_clean)
│   └── finetune/
│       ├── extract_native_phys_feats.py          # 60维原生分辨率频域/边缘/梯度物理特征提取器
│       ├── build_v9_specialists.py               # 零振铃等级阶梯合成管线 (s16/s17 验证划分)
│       ├── build_v9_fulldata_8folds.py           # 3,573张全量真图 8-Fold 零泄漏划分与按折隔离合成数据生成
│       ├── train_v9_specialist.py                # V9 高分辨率空间异常池化专精模型训练主脚本 (Fused-BF16加速)
│       ├── launch_v9_fulldata_8gpus.py           # 8卡并行训练 8-Fold 全量 V9 专精模型启动器
│       ├── strict_zero_leakage_audit.py          # 1,000张独立盲测集严格零泄漏只读审计脚本
│       └── build_v9_55plus_final_submission.py   # 多专家概率校准融合与评测集 JSON 导出脚本
├── results/                                      # 各阶段消融实验、交叉验证与盲测结果 JSON
└── pred_nriqa_dataset_test.json                  # 最新评测集标准格式提交文件
```

---

## 3. 环境依赖与基座模型准备

### 3.1 Python 环境
- Python `>= 3.10`, PyTorch `>= 2.2` (支持 `bfloat16` 与 `AdamW(fused=True)`), `timm`, `transformers`, `opencv-python`, `pillow`, `scikit-learn`, `scipy`, `numpy`.

### 3.2 预训练基座权重目录 (`models/`)
将以下开源预训练权重放置于项目根目录 `models/` 下（离线加载，无需外网 API）：
- `models/pe-core-g14` (`PE-Core-G14-448`, 1.88B)
- `models/pe-core-l14` (`PE-Core-L14-336`, 0.32B)
- `models/dinov3-vitl16` (`DINOv3-ViT-L16`, 0.30B)
- `models/c-radio-v4-h` (`C-RADIOv4-H`, 0.63B)
- `models/siglip2-giant` (`SigLIP2-Giant-opt-patch16-384`, 1.14B)
- `models/siglip2-so400m` (`SigLIP2-SO400M-patch14-384`, 0.40B)
- `models/convnextv2-large` (`ConvNeXtV2-Large-22k-384`, 0.20B)
- `models/eva02-large-448` (`EVA02-Large-Patch14-448`, 0.30B)
- `models/eupe-vitb` (`EUPE-ViT-B`, 0.09B)

> **参数量合规声明**：单个专精模型总参数量为 `1.21B ~ 3.89B`（其中可训练参数仅 `0.38B ~ 0.54B`），全套互不重叠基座模型参数量总和 `< 6.5B`，远低于赛题规定的 **30B 参数量上限**。

---

## 4. 一键复现流程（数据构造 → 训练 → 零泄漏验证 → 评测集推理）

### Step 1：构造 8-Fold 全量防泄漏划分与按折隔离零振铃合成数据
```bash
python3 -u code/finetune/build_v9_fulldata_8folds.py
```
- 自动读取首批（1,902 张）与补充数据集（1,671 张）共 **3,573 张独立真图**，按多标签分层划分为 `Fold 0 ~ Fold 7`；
- 在 `Fold 0 ~ Fold 5` 内部分别仅用该折训练集的 Clean 图生成零振铃等级阶梯合成图（存于 `data/v9_fulldata/synth_f0..f5/`），保证验证集 100% 零源图泄漏。

### Step 2：提取 60 维原生分辨率物理特征缓存
```bash
python3 -u code/finetune/extract_native_phys_feats.py
```
- 对所有真图、合成图及评测集图片在原生分辨率下提取 60 维频域/边缘/梯度物理特征，缓存至 `feats/native_phys_60d.npz`。

### Step 3：启动 V9 高分空间异常专精模型训练
```bash
# 8卡并行训练 8-Fold 全量专精模型 (Fused-BF16 加速)
python3 -u code/finetune/launch_v9_fulldata_8gpus.py
```
- 若在单卡（如单张 RTX 4090 24GB）上顺序训练或微调单折模型，可直接调用：
```bash
python3 -u code/finetune/train_v9_specialist.py \
  --backbones "pecorel:lastk6,dinov3l:lastk6,siglip2:lastk6,eupe:full" \
  --focus-classes "模糊,纹理损失,过锐,花屏,过度生成,虚化突兀" \
  --epochs 4 --batch 4 --grad-accum 2 --size 448 --tiles 3 \
  --split data/v9_fulldata/split_f4_all.csv \
  --manifest data/v9_fulldata/manifest_f4_v9.csv \
  --out results/v9_fulldata/V9_Full_F4_AllP0_Tile3.json
```

### Step 4：运行独立测试集严格零泄漏只读审计
```bash
python3 -u code/finetune/strict_zero_leakage_audit.py
```

### Step 5：生成最终评测集预测结果 (`pred_nriqa_dataset_test.json`)
```bash
python3 -u code/finetune/build_v9_55plus_final_submission.py
```
- 自动对输出文件执行完整性校验（样本 ID 100% 对齐、无重复/缺失 ID、每张图完整包含六类合法标签 `P0 / P1 / P2 / 无问题`，其中 `虚化突兀` 与 `花屏` 仅含 `P0 / 无问题`）。

---

## 5. 单张 RTX 4090 部署、视频多帧聚合与可解释归因方案

- **单张 RTX 4090（24GB）低成本部署（满足 `S_deploy` 与 `S_cost` 0.7~1.0 最高档要求）**：
  - 在 `bfloat16` 精度下，我们的统一 12 视角全缺陷模型（`V9_AllP0_Tile3`，1.21B 参数）显存占用仅 **~4.8 GB**，单图端到端推理耗时（含 60 维物理特征提取与前后处理）仅 **~0.42 秒**；即使串行运行 3 个互补专精模型，单张 4090 峰值显存 `< 9.5 GB`、单图总耗时 **`~1.35 秒`**，不及官方 15 秒上限的 **1/10**！
- **视频级多帧抽帧聚合与误判控制方案** 与 **局部 Tile 空间定位 + 语义归因输出** 详见 [docs/TECHNICAL_REPORT.md](file:///wanqing-develop/luowenjing/18/docs/TECHNICAL_REPORT.md)。
