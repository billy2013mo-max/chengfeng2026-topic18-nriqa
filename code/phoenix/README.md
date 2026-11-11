# Phoenix 55+ 合规彻底重构套件 (`code/phoenix/`)

本目录实现了针对 **2026 快手乘风计划赛题 18（无参考画质评估与质检 NR-IQA）** 的零历史包袱、100% 符合官方硬性约束的彻底重构方案（目标客观分 **55+ / 65**，同时覆盖 **35 分主观技术答辩** 的归因与视频级落地模块）。

## 1. 官方赛制硬红线审计（启动即自动校验）

| 约束维度 | 官方硬性上限 | Phoenix 实测预算 | 状态 |
| :--- | :--- | :--- | :--- |
| 训练与推理使用模型总参数量 | `<= 30.0B` | **`18.655B`**（正交三支柱各自单实例 Basin-SWA 合并权重，预留 `11.345B` 缓冲） | 严格合规 |
| 单张 RTX 4090 (24GB) 推理显存 | `<= 24.0 GB` | **`~15.48 GB`**（全常驻零切模：ViT/物理分支 BF16 + LLM W4A16 + Head FP32） | 严格合规 |
| 单图端到端推理耗时 | `<= 15.0 s/张` | **`~3.05 s/张`**（Non-Thinking 单次前向直出 + 向量化 `1,024` 态闭式解码） | 严格合规 |
| 单样本严格独立性 | 禁止测试批统计偷看 | 恒满足 **`f([x_i]) == f(X)[i]`**（零分位数对齐、零 Sinkhorn、零旧 JSON 锚定） | 严格合规 |
| 真实数据纯度 | 禁止低质合成污染主干 | 主干仅训练 **`3,573` 张纯真图**（自动拦截一切 `_synth` / `_natneg` 样本） | 严格合规 |

## 2. 目录模块与六大核心突破映射

1. **`config.py`**：统一赛题 Schema、`1,024` 合法六维联合状态表、参数量/显存硬预算审计、纯真图解析器（兼容 `/dev/shm/race18_native` 零拷贝缓存）、双口径评分器（同时输出 **官方附录二 14 通道严格口径** 与 **练习榜 2% 悬崖口径**）及提交 JSON 双向校验器。
2. **`data_split_and_audit.py`**：
   - 修复 `manifest_combined.csv` 中 `0016__211661657305_...`、`002_212299294787_...` 等序号前缀导致的同视频跨折泄漏，结合可选视觉余弦图连通分量执行零泄漏 5 折多标签分层切分（输出 `data/phoenix_splits/phoenix_5fold.csv`）。
   - 量化首批发版（`orig=1,902`）与增补批次（`supp=1,671`）的系统性先验偏移，自动检测同视频 `<0.25s` 邻近帧标注冲突并输出异方差软标签平滑标记（`batch_shift_and_audit.json`）。
3. **`native_physics_and_views.py`**：
   - 废除一切 `448x448` 双线性压扁缩放与拼图网格，实现 **原生宽高比动态缩放** + **内生双焦点不重叠 `1:1` 原尺寸 `384x384` 切片器**（自动抑制高对比度短视频花字/字幕描边）。
   - 计算原生 `1:1` 2D Haar 小波高频子带、USM **双极性过冲振铃包络**（精准区分 `过锐` 与清晰字幕）、`8x8` 宏块色度溢出图（`花屏`）以及 **光学弥散圆（CoC）深度-散焦雅可比场**（区分 `虚化突兀` 抠图断崖与自然浅景深）。
4. **`losses_and_heads.py`**：
   - **`PhoenixUnimodalDeQAHead`**：解除旧架构累积 Sigmoid 对 `P1` 峰值概率的压制，采用 Query2Defect 空间交叉注意力 + 直接四分类单峰软分布头 + 仅训练期激活的 `BatchShiftCalibration` + 独立全无问题门控头。
   - **`DeQAOrdinalDistributionLoss` + `CrossBatchThurstoneRL2RLoss`**：单样本可加异方差高斯软标签 KL + 平方累积分布 EMD + `1,024` 跨批记忆队列（XBM）同场景难例成对 Thurstone 保真度排序与 Non-Thinking 离散 GRPO 强化信号。
   - **`BasinAlignedSWACollector`**：同一热启动盆地内多低学习率分支参数平均，以 **1 倍参数量** 获得多随机种子集成增益。
5. **`stage0_real_restoration_tapt.py`**：可选的纯真实神经增强/超分偏好对（`FGResQ`）与真实局部伪影热力图（`PAL4VST`）任务自适应预训练（无外部目录时自动安全跳过）。
6. **`train_pillar12_native_vlm.py` & `train_pillar3_hires_physics.py`**：
   - Pillar 1（`Qwen3-VL-8B` 原生 2D-RoPE + 全层 ViT DoRA）、Pillar 2（`InternVL3.5-8B` / `Qwen2.5-VL-7B` 异构排序 VLM）、Pillar 3（`DINOv3-L + C-RADIOv4-H + ConvNeXtV2-L` 四层中间超列 + `1:1` Haar 小波物理分支 + `FGResQ/CoC/PAL4VST`）统一支持 `cv5`、`stage2-rl2r`、`fulldata-swa` 三种训练模式。
7. **`inductive_anchor_and_joint_decoder.py`**：
   - **`InductiveDualAnchorBank`**：纯训练集聚类生成的 `128 Clean Shield + 128 Defect Anchor` 双原型库，单图独立修正硬负样本误报边界。
   - **`JointCompetitionUtilityDecoder`**：在全部 `1,024` 个合法 6D 标签态上闭式求解竞赛期望效用最大化（结合官方指标雅可比边际收益与训练集共现势函数），所有 `<=22` 个结构参数均通过 **300 次按视频组 Bootstrap 20% 下置信界（LCB）** 拟合。
8. **`deploy_single_4090_engine.py`**：
   - 单卡 4090 全常驻推理、提交文件双向强校验、**后验标签锁定的 `<=40` 字结构化归因**（`--emit-attribution`）以及 **CUSUM 对数似然比多帧视频迟滞聚合器**（`PhoenixVideoHysteresisAggregator`）。

## 3. 快速运行指南

```bash
# 1. 本地/服务器一键执行预算审计、3573真图零泄漏5折切分及全模块单图独立性测试
bash run_phoenix_pipeline.sh --verify-local

# 2. 在多卡 H800 / GPU 服务器端到端执行完整训练与单卡 4090 合规推理
bash run_phoenix_pipeline.sh --all \
  --test-template pred_nriqa_dataset_test.json \
  --test-dir data/test_150 \
  --out-json results/phoenix_submissions/pred_nriqa_dataset_test.json

# 3. 终审盲测集（如 654 张）单卡 RTX 4090 一键推理（可选开启 <=40字归因附加输出）
bash run_phoenix_pipeline.sh --step5-deploy-4090 \
  --test-template /path/to/blind_test_template.json \
  --test-dir /path/to/blind_test_images \
  --out-json results/phoenix_submissions/pred_blind_final.json \
  --emit-attribution
```
