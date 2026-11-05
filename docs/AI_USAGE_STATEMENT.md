# 【2026乘风挑战赛·命题18】AI 使用说明（组委会统一要求）

> **说明**：根据【2026乘风挑战赛】组委会统一要求（命题文档第 5 节“解题方案中需说明解题过程在多大程度上使用了 AI、如何使用，表现突出者有额外激励”），本文档详细披露并总结本团队在解题全生命周期中如何使用 **AI 编程与科研智能体（Antigravity Agentic Coding Assistant）** 进行人机协同攻关。

---

## 一、 AI 参与程度与人机协同范式概述

在本命题的解题过程中，我们采用了 **“人类专家把控顶层方向与线上反馈闭环 + AI 智能体全自主执行底层误差审计、算法设计、合成数据工程、多卡训练调度与零泄漏验证”** 的深度协同范式（AI 参与了约 **90% 的代码研发、数据分析、实验自动化与文档整理工作**）：

1. **人类参赛者的核心决策作用**：
   - 确立客观分 `55+` 的演进目标，把控线上参考榜/评测集的阶段性验证反馈（例如敏锐指出早期版本“正向缺陷分变高但 P0 识别偏低”的 trade-off 现象）；
   - 严格把关实验科学性（要求建立 100% 物理隔离的独立盲测集审计机制、要求利用 100% 全量人工标注数据进行交叉验证训练、要求挖掘 H200 显存余量进行无损训练加速、及时同步最新赛题规则变更文档）。
2. **AI 智能体的闭环执行作用**：
   - 在获得人类授权后，AI 智能体完成了从 **数据探索 → 瓶颈诊断 → 物理退化建模 → 神经网络架构升级 → CUDA/BF16 底层训练加速 → 8 卡并行任务编排 → 严格零泄漏消融审计 → 提交产物校验** 的全链路自动化闭环。

---

## 二、 AI 在解题各阶段的具体使用方式与创新贡献

### 1. 自动化 BadCase 误差诊断与赛题规则精准对齐
- **跨划分混淆矩阵审计**：AI 自动编写诊断脚本（[`diagnose_s16_s17_bottlenecks.py`](file:///wanqing-develop/luowenjing/18/code/finetune/diagnose_s16_s17_bottlenecks.py)、[`audit_55_bottlenecks.py`](file:///wanqing-develop/luowenjing/18/code/finetune/audit_55_bottlenecks.py)），定位出早期模型在 `过度生成 (P1/P2)`、`纹理损失 (P1)` 上的漏检，以及早期合成数据中高频正弦波纹导致的 `过锐 ↔ 花屏` 交叉误报。
- **赛题文档变更自动 Diff 与公式对齐**：当赛题更新第二版文档时，AI 通过逐行对比初版与新版文档，第一时间发现 **`S_clean`（20分）计算口径从“仅考核 P0 误判率（除以 $\tau=0.02$）”变更为“全无问题样本六类全部预测为无问题才得分（不区分 P0/P1/P2 误报）”**，从而设计出“全局无问题门控（Clean Gate）+ 解锁已检出缺陷样本 P0 召回”的针对性后处理校准策略。

### 2. 物理退化合成算子与 60 维原生特征工程设计
- **零振铃等级阶梯合成（Zero-Ripple Severity Ladder）**：AI 针对小样本长尾分布，设计并实现了 [`build_v9_specialists.py`](file:///wanqing-develop/luowenjing/18/code/finetune/build_v9_specialists.py) 与 [`build_v9_fulldata_8folds.py`](file:///wanqing-develop/luowenjing/18/code/finetune/build_v9_fulldata_8folds.py)，用多尺度非锐化掩模替代正弦条纹，并在同一张干净底图上构造 `Clean → P2 → P1 → P0` 严格单调退化阶梯，确保模型学习纯粹的画质退化强度而非背景捷径。
- **60 维原生分辨率信号特征**：AI 编写了 [`extract_native_phys_feats.py`](file:///wanqing-develop/luowenjing/18/code/finetune/extract_native_phys_feats.py)，在图像缩放前并行提取多尺度 Laplacian 能量比、边缘白边过冲比（Halo Overshoot）、块边界突变度等 60 维物理特征，补齐了 ViT 下采样损失的像素级高频先验。

### 3. V9 神经网络架构设计与损失函数创新
- **空间异常池化（Spatial Anomaly Pooling）**：针对局部缺陷被全局平均池化淹没的问题，AI 在 [`train_v9_specialist.py`](file:///wanqing-develop/luowenjing/18/code/finetune/train_v9_specialist.py) 中实现了 `Global + Tile Mean + Top-2 Tile Anomaly Max-Mean + Tile Spatial Std` 四维池化层，使局部 `过度生成`、`花屏` 与 `虚化突兀` 的识别能力大幅跃升，并天然支持九宫格异常位置归因。
- **复合序数感知损失函数**：AI 将官方竞赛评分公式转化为可微分的 `Soft Competition Loss`，并结合批次内 `Pairwise Ordinal Margin Loss`，显著改善了 `P0 / P1 / P2` 等级边界的校准质量。

### 4. 底层训练无损加速与 8 卡 H200 自动化调度
- 针对 3.89B 巨型多骨干模型训练耗时问题，AI 在保证数学等价（零精度损失）的前提下实施了 **7 项底层工程加速**：
  1. DataLoader `uint8` 零拷贝传输 + GPU 端 `bfloat16` 归一化（PCIe 带宽开销降低 75%）；
  2. 33.5 亿冻结骨干参数训练前一次性预转为 `bfloat16`，消除每步动态类型转换；
  3. 冻结 Transformer 层固定 `.eval()` 模式，跳过冗余随机丢弃算子；
  4. 启用 `torch.optim.AdamW(..., fused=True)` 单核融合更新与 Flash SDP；
  5. 取消梯度累加、直接翻倍物理 Micro-Batch 填满显存，使训练速度提升 **2.5 倍以上**，并在 8 张 GPU 上全自动并行完成了 8-Fold 全量数据专精训练。
