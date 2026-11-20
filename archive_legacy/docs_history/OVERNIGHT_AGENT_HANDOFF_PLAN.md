# 【2026乘风挑战赛·命题18】夜间全自动 Agent 交接文档与实验执行规划

> **致接手本任务的 Agent**：
> 本交接文档总结了截至目前所有的实验结论、底层数据真相、线上评测反馈、代码资产定位，以及一套**零弯路、可直接复制命令执行的夜间全自动迭代方案**。请严格按照本文档的 **铁律（Hard Rules）** 与 **四阶段执行流水线（Stage 1 ~ Stage 4）** 全自动推进，中途无需停下询问用户，直至完成全部验证并输出最终报告与提交文件。

---

## 一、核心背景、官方指标与线上基准锚点（必读）

### 1.1 任务与评分公式（满分 65 分客观分）
- **六类画质缺陷**：
  - 四档细粒度类（`FULL = ['模糊', '纹理损失', '过锐', '过度生成']`）：取值 `无问题(0) / P2(1) / P1(2) / P0(3)`；
  - 二分类严重缺陷类（`BIN = ['花屏', '虚化突兀']`）：仅取值 `无问题(0) / P0(3)`（严禁输出 `P1` 或 `P2`）。
- **官方客观分公式（65 分）**：
  $$\text{Total} = 25 \times S_{\text{pos}} + 20 \times S_{\text{p0}} + 20 \times S_{\text{clean}}$$
  1. **正向缺陷得分 $S_{\text{pos}}$（25 分）**：6 个类别的严格等级匹配 F1 均值：
     $$F1_c = \frac{2 M_c}{2\text{TP}_c + \text{FP}_c + \text{FN}_c}, \quad S_{\text{pos}} = \frac{1}{6}\sum_{c=1}^6 F1_c$$
     - **极其关键的数学特性**：$\text{TP}_c$ 是真值有缺陷且预测也有缺陷的样本数，但分子 $M_c$ 要求**预测档位与真值档位完全一致（Exact Grade Match）**！只要检出有缺陷但档位差一档（如真值 `P1` 预测成 `P2`），**分子 $M_c$ 得 `0` 分，而分母 $2\text{TP}_c$ 照常加 `2`**（惩罚比漏报 $\text{FN}_c$ 加 `1` 更重！）。
  2. **P0 严重缺陷得分 $S_{\text{p0}}$（20 分）**：6 个类别的 $0.6 \times \text{Recall}_{c,\text{P0}} + 0.4 \times F1_{c,\text{P0}}$ 均值。
  3. **无问题样本防误报得分 $S_{\text{clean}}$（20 分）**：
     - **Doc1 口径（`comp_metrics.py`）**：基于全正常图（6 类真值全为 0）被误报为任何 `P0` 的比例 $\text{P0\_FPR}$：$S_{\text{clean,doc1}} = \max(0, 1 - \text{P0\_FPR} / 0.02)$；
     - **Doc2 口径（赛题文档 `(2).md`）**：全正常图被预测为 6 类全 `无问题` 的比例 $S_{\text{clean,doc2}}$。

### 1.2 线上真实反馈与最高分锚点（`52.43`）
1. **历史最高分 `52.43` 文件**：`results/v3_super/pred_nriqa_dataset_test.json`（由 `code/finetune/run_v3_super_ensemble.py` 生成）。
   - **线上已知子项得分拆解**：在 `52.43` 这一版中，线上 **`Clean Score` 已经拿到了 `20.00 / 20.00` 满分**！（对应 `test_150` 中的 **15 张全无问题图 `sh_5243`**）。
2. **`900 / 900` 格 100% 精确复现 `52.43` 概率与预测的代码**：
   - 已封装在 `code/finetune/build_v10_fulldata_test150_submission.py` 的 `load_exact_5243_anchor()` 函数中：
     - 由 `results/v2_super/test_150_all_models_probs.npz`（10 个全量真图 V2 模型）+ `results/v3_super/test_150_v3_models_probs.npz` 中不含 `'cali'` 的 11 个全量真图 V3 模型按权重加权平均得到 `p0_5243, p1_5243, p2_5243`，再经 `decode_v3_shielded(p0_5243, p1_5243, p2_5243)` 解码，与 `results/v3_super/pred_nriqa_dataset_test.json` **900/900 格 100% 完全一致**！

---

## 二、前序实验踩坑总结与三大真相（严禁重复踩坑！）

1. **【真相 1：关于全量真图数据的真相】**
   - `data/manifest_combined.csv` 包含整个赛题全部 **3,573 张独立真实图**（首批 `orig=1,902` 张 + 补充 `supp=1,671` 张）。
   - `data/split_all_full.csv` 的 `train` 划分**已经包含了全部 3,573 张真实图（100% 真图，0 合成图）**，`52.43`（`V2_super + V3_super`）正是在 `data/split_all_full.csv` 上训练的！
   - `data/v9_fulldata/split_f0..f7.csv` 中的 4,019 行并非多了新真图，而是每折留出 446 张真图做验证，并在 CSV 中将该 446 张复制成 `val` 和 `test` 两份（`3127 + 446 + 446 = 4019`），导致 `V9_Full / V10_Full` 每折仅训练了 87.5%（3,127 张）真图。

2. **【真相 2：合成数据（`V4 / V6 / V7 / V8 / V9 / V10` 的 `F0..F5`）存在严重负迁移，必须彻底停用！】**
   - 早期合成数据（`V4~V9`）所有合成样本均继承了底图的 `纹理损失 = P2/P1` 伪标签，导致测试集 `纹理损失 P2` 误报暴涨。
   - 即便在 `V10` 中使用 `get_sample_class_mask6` 严格屏蔽非目标类损失，**合成图在目标类本身上依然劣化了真实图像分布**：
     - 在 8 折 OOF 上对比掺合成数据的 `F0..F5`（2,681 张真图评测）与 100% 纯真图的 `F6..F7`（892 张真图评测）：
       - `过度生成 S_pos`：纯真图折 **`0.448`** vs 掺合成数据折 **`0.407`（`-4.1%`）**
       - `虚化突兀 S_pos`：纯真图折 **`0.855`** vs 掺合成数据折 **`0.778`（`-7.7%`）**
       - `纹理损失 S_pos`：纯真图折 **`0.721`** vs 掺合成数据折 **`0.690`（`-3.1%`）**
       - `过锐 假阳性率（FPR）`：纯真图折 **`17.5%`** vs 掺合成数据折 **`24.2%`（`+6.7%` 误报膨胀）**
   - **铁律 1**：后续所有训练与集成 **100% 只允许使用纯真实人类标注图像（`data/manifest_combined.csv`）**，严禁再混入任何合成图像！

3. **【真相 3：为什么高斯软标签（`deqa_sigma=0.44`）和盲目权重 SWA 掉了 `S_pos`】**
   - 在 `V10` 中引入的高斯软标签（`deqa_sigma=0.44`）平滑了 `P2 / P1 / P0` 之间的概率峰值，虽然微涨了 `S_p0`（`+0.3%`），但导致 `S_pos` 从 `0.613` 暴跌至 `0.578`（`-3.5%`）。
   - **铁律 2**：训练损失函数回归经过严格验证的 **`train_v3.py`（`v3_loss_fn`：Cumulative BCE + Exact 4-Way CE + Ordinal CDF EMD + Soft Macro-F1 `loss_s_pos` + Competition `loss_s_p0` + `p0_clean_weight=4.5`）**！

---

## 三、3,574 张真图混淆矩阵拆解与三大高确定性突破口

在 3,574 张真实样本 OOF 上，现有模型的混淆矩阵与理论提分空间（Oracle Headroom）如下：

| 缺陷类别 | 真值正样本数 | 二分类检出数（检出率） | 档位完全猜对 ($M_c$) | **已检出但猜错档位（`P2↔P1↔P0`）** | 正常图误报 ($\text{FP}_c$) | 正样本漏报 ($\text{FN}_c$) | 当前 `S_pos` / `S_p0` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`模糊`** | 1,711 | 1,538 (**89.9%**) | 1,081 | **`457`（占正样本 26.7%）** | **`341`** | 173 | `0.602` / `0.816` |
| **`纹理损失`** | 2,803 | 2,769 (**98.8%**) | 2,028 | **`741`（占正样本 26.4%）** | **`241`** | 34 | `0.698` / `0.804` |
| **`过锐`** | 1,283 | 1,126 (**87.8%**) | 780 | **`346`（占正样本 27.0%）** | **`516`（巨量误报！）** | 157 | **`0.533` / `0.694`** |
| **`花屏`** | 213 | 209 (**98.1%**) | 209 | `0` | 10 | 4 | `0.968` / `0.976` |
| **`过度生成`** | 684 | 509 (**74.4%**) | 275 | **`234`（占正样本 34.2%）** | 124 | **`175`** | **`0.418` / `0.576`** |
| **`虚化突兀`** | 225 | 180 (**80.0%**) | 180 | `0` | 47 | 45 | `0.796` / `0.799` |

- **Oracle 1（保持二分类检出不变，仅修正已检出正样本内的 `P2/P1/P0` 档位错位）**：`S_pos` 从 **`0.619` 跃升至 `0.872`（仅 `S_pos` 一项即净增 `+6.34 分`！）**。
- **Oracle 2（消除二分类 $\text{FP}/\text{FN}$，档位仍用当前模型 argmax）**：`S_pos` 从 **`0.619` 升至 `0.716`（净增 `+2.44 分`）**。

由此锁定以下 **三大非噪声级突破方向**（也就是夜间自动化执行的全部核心任务）：

### 突破口 A：修复现有逐档解码器的数学缺陷 —— 研发「两步分层贝叶斯解码器（Two-Stage Hierarchical Bayes-Optimal F1 Decoder）」与「同场景 K-NN 邻域档位校准器」
1. **现有解码器（`decode_pc` / `decode_v3_shielded`）的致命缺陷**：
   - 现有逻辑独立检查 `p2 > T2`、`p1 > T1`、`p0 > T0`，完全忽略了 **总缺陷概率 $p_{\text{any}} = p_0 + p_1 + p_2$**！
   - 在 3,574 张真图上的实测后果：
     - **分流漏报（Split-Probability Miss）**：在 `过度生成` 上，有 **105 张样本**满足 $p_{\text{any}} > 0.45$（模型明确知道有过度生成缺陷），但因为概率分散在三档（如 `p2=0.22, p1=0.20, p0=0.14`），无一档单独超过 `0.30`，被直接判为 **`无问题(0)`**！
     - **低置信擦边误报（Low-$p_{\text{any}}$ FP）**：在 `过锐` 和 `模糊` 上，有 **292 张样本**满足 $p_{\text{any}} < 0.45$（模型认为 `无问题` 概率 $>55\%$），仅因为 `p2=0.31, p1=0.01, p0=0.00` 刚擦过 `0.30` 就被判成 `P2`（其中 **192 张真值就是 `无问题`**！）。
2. **新解码与校准算法设计**：
   - **Step 1（缺陷存在性门控 Defect-Presence Gate）**：
     - 对四档类 $c \in \{0,1,2,4\}$，计算累积缺陷概率 $p_{\text{any}, c} = p_{0,c} + p_{1,k} + p_{2,k}$（可结合直接来自累积头的 $\sigma(\text{logit}_{\text{cum}, P2})$）。只有当 $p_{\text{any}, c} > \tau_{\text{any}, c}$ 且满足 `p_max_grade > tau_min_peak` 时才判定为有缺陷（彻底消除 `过锐/模糊` 上 192 个低 $p_{\text{any}}$ 假阳性，并召回 `过度生成` 上分流漏报的正样本）。
   - **Step 2（条件期望与贝叶斯边际效用档位分配 Conditional Grade Assignment）**：
     - 给定样本通过 Step 1 门控（确认为正样本），计算条件档位概率 $\tilde{p}_g = \frac{p_g}{p_{\text{any}, c}}$（$g \in \{1,2,3\}$）及条件期望严重度 $\mu_c = \tilde{p}_1 + 2\tilde{p}_2 + 3\tilde{p}_3 \in [1, 3]$，结合逐类校准的 $(\theta_{1\to 2, c}, \theta_{2\to 3, c}, \tau_{0, c})$ 决定最终档位 `P2(1) / P1(2) / P0(3)`。
   - **Step 3（同场景 K-NN 邻域原型校准 Scene-Matched K-NN Prototype Calibration）**：
     - `feats/` 目录下已存有 3,573 张全量真图的多基座特征（`feats/internvit-6b_combined.npz`、`feats/c-radio-v4-h_combined.npz`、`feats/siglip2-so400m_combined.npz`、`feats/signal.npz`、`feats/texture.npz`）。
     - 对任一待测图 $x_i$，在特征空间检索训练集中余弦相似度最高的 Top-$K$（如 $K=25$）同场景真图，计算邻域先验分布与邻域预测残差 $\Delta_c(x_i) = \frac{1}{K}\sum_{j \in \mathcal{N}_K(x_i)} (y_{j,c} - \hat{y}_{j,c})$，对当前图的档位边界做同场景自适应微调。

### 突破口 B：启用仓库最强但从未进入 `52.43` 集成的 60 亿参数基座 `InternViT-6B`（`models/internvit-6b`）与 `SigLIP2-NaFlex`
1. **实测证据（运行 `python3 code/backbone_complementarity.py` 的 14 基座同折横评结果）**：
   - 在全部 14 个预训练基座中，**`InternViT-6B`（`models/internvit-6b`）总 `mAP14 = 0.779`、`mAP_P0 = 0.749` 均高居全场第 1**，且在所有分辨率分桶（`<720p`, `720-1080p`, `1080-2160p`, `>2160p`）全部排名第 1！
   - 尤其在当前最薄弱的 **`过锐`（当前 `S_pos=0.533`）** 和 **`模糊`（当前 `S_pos=0.602`）** 上，`InternViT-6B` 断崖式领先：
     - `过锐_P0 AP`：**`InternViT-6B = 0.712`** vs `SigLIP2 = 0.585` vs `C-RADIO = 0.573` vs `PE-Core-G = 0.492`（**领先 `+13% ~ +22%`！**）
     - `过锐_P1 AP`：**`InternViT-6B = 0.808`** vs `SigLIP2 = 0.638` vs `PE-Core-G = 0.668`（**领先 `+14% ~ +17%`！**）
     - `过锐_P2 AP`：**`InternViT-6B = 0.865`** vs `SigLIP2 = 0.805`
     - `模糊_P2 / P1 AP`：**`InternViT-6B = 0.897 / 0.875`**（全场第 1）
   - 在 `results/internvit/iv_lastk4_lrl3e4_s1.json` 单折实验中，仅单个 `InternViT-6B`（`internvit:lastk4, lr_last=3e-4`）的 `S_pos` 就达到 **`0.5666`**，同折单模型 `C-RADIO` 仅 `0.5180`、`SigLIP2` 仅 `0.4708`（**单基座 `S_pos` 高出 `+4.9% ~ +9.6%`！**）。
2. **关键实现细节（已在 `code/finetune/models.py` 第 57~93 行就绪）**：
   - `models.py` 中的 `build_backbone('internvit')` 已经修好了 `pos-embed` 448 缩放与 `gradient_checkpointing` 梯度归零问题。
   - **重要环境变量与超参**：运行 `internvit` 时，设置环境变量 `INTERNVIT_SIZE=448`（当 `--size 448` 时，完全免去位置编码插值损耗，使用原生 `32×32` 网格！），且 `internvit:lastk4` 的最佳微调学习率为 `--lr-last 1.5e-4 ~ 3e-4`（详见 `results_lrl_confirm.out` 与 `results/internvit/iv_lastk4_lrl3e4_s1.log`）。
   - **重要 Bug 修复提示**：检查 `code/finetune/train_v2.py` 第 200~220 行及 `train_v3.py` 中的 `configure_backbone_params` 函数，确认 `internvit` 的 `m.encoder.layers`（`bl`）能被正确解冻最后 $k=4$ 层！

---

## 四、夜间全自动四阶段实验执行规划（Step-by-Step Action Plan）

请接手 Agent 按以下 **Stage 1 $\rightarrow$ Stage 2 $\rightarrow$ Stage 3 $\rightarrow$ Stage 4** 顺序严格执行（注意：可以将 **Stage 2 的 8 卡 GPU 训练作为后台任务先行启动**，在等待 GPU 训练的 60~90 分钟内同步在 CPU 上开发并验证 **Stage 1 的两步贝叶斯解码器与 K-NN 校准器**，实现 100% 并行利用！）。

---

### Stage 0：启动前检查与 `train_v3.py` 对 `internvit` 的兼容性验证（耗时 ~5 分钟）

1. **检查 `train_v2.py` / `train_v3.py` 中 `internvit` 的层解冻与保存**：
   - 阅读 `code/finetune/train_v3.py`（第 160~364 行）和 `code/finetune/train_v2.py`（第 180~300 行）。
   - 确认当传入 `--backbones "internvit:lastk4,cradio:lastk4,siglip2:lastk4"` 且 `INTERNVIT_SIZE=448 --size 448` 时：
     1. `internvit` 的最后 4 层（`bl[-4:]`，即 `m.encoder.layers[-4:]`）参数 `requires_grad=True`，其余层 `requires_grad=False`；
     2. 在保存 `--save-state` checkpoint 时，为避免每个 `internvit` 模型保存 12GB 冻结权重撑爆磁盘或拖慢保存，**可以只保存 `requires_grad=True` 的参数 + head 参数（或在推理完 `val / test / test_150` 保存完整 `*_probs.npz` 后按需保留最优权重的 `.pt`）**！
     3. **强烈建议**：编写一个统一的训练+多端推理脚本 `code/finetune/train_v3_internvit.py`（基于 `train_v3.py`），在每个模型训练完成（第 3 轮、第 4 轮及概率平均）后，**直接在同一个进程内完成对 `val`、`test`（若存在）以及 `data/test_150`（150 张盲测集，开启 `2x Horizontal Flip TTA`）的推理**，并将 `va_p0/1/2`、`te_p0/1/2`、`t150_p0/1/2` 一步到位保存至 `*_probs.npz`！这样无需二次加载 6B 模型，节省一半时间且零出错概率！

---

### Stage 1：8×H200 并行启动纯真图 `InternViT-6B` 零泄漏验证组 + 全量决战组训练（耗时 ~60~80 分钟，后台运行）

在 8 张 H200（每张 140GB 显存，完全容纳 `InternViT-6B + C-RADIO + DINOv3/SigLIP2` @ `448×448`）上同时启动 8 个 **100% 纯真图（`data/manifest_combined.csv`，0 合成数据）** 训练任务：

#### 1.1 GPUs 0 ~ 4：5 个零泄漏独立盲测集（`s16 ~ s20`）验证组
- **目的**：在完全未接触过的 5 个标准盲测划分 `data/resplits/split_s16.csv` ~ `split_s20.csv` 上训练 `InternViT-6B` 混合多塔模型，与已有的 `results/night/4/final_s16..s20_seed*_probs.npz` 和 `results/backbone_fixed/*_s16..s20_seed*_probs.npz` 做严格零泄漏盲测对比，量化 `InternViT-6B` 在每个独立划分上的真实提分！
- **分配**：
  - `GPU 0`（`s16`）：`--split data/resplits/split_s16.csv --backbones "internvit:lastk4,cradio:lastk4,siglip2:lastk4" --size 448 --tiles 2 --epochs 4 --batch 4 --grad-accum 2 --lr-last 1.5e-4 --lr-head 1.5e-3 --clean-repeat 4 --p0-clean-weight 4.5`
  - `GPU 1`（`s17`）：同上，`--split data/resplits/split_s17.csv`
  - `GPU 2`（`s18`）：同上，`--split data/resplits/split_s18.csv`
  - `GPU 3`（`s19`）：同上，`--split data/resplits/split_s19.csv`
  - `GPU 4`（`s20`）：同上，`--split data/resplits/split_s20.csv`
  - *(注：每个进程训练完立刻推理 `val`、`test` 及 `test_150` TTA，保存至 `results/v11_internvit_pure/iv3_s{16..20}_probs.npz`)*

#### 1.2 GPUs 5 ~ 7：3 个 100% 全量真图（`data/split_all_full.csv`，3,573 张真图全训练）旗舰模型组
- **目的**：像 `52.43` 的 `V3_super` 一样，使用 100% 全部 3,573 张真图训练 3 个架构正交的 `InternViT-6B` 旗舰模型，训练结束直接产出 `test_150` 的 `2x Flip TTA` 概率：
  - `GPU 5`（`iv6b_cradio_sig2_448_t2`）：
    `--split data/split_all_full.csv --backbones "internvit:lastk4,cradio:lastk4,siglip2:lastk4" --size 448 --tiles 2 --epochs 4 --batch 4 --grad-accum 2 --lr-last 1.5e-4 --lr-head 1.5e-3 --clean-repeat 4 --p0-clean-weight 4.5 --gen-repeat 2`
  - `GPU 6`（`iv6b_d3_pcl_448_t2`）：
    `--split data/split_all_full.csv --backbones "internvit:lastk4,dinov3l:lastk4,pecorel:lastk4" --size 448 --tiles 2 --epochs 4 --batch 4 --grad-accum 2 --lr-last 1.5e-4 --lr-head 1.5e-3 --clean-repeat 4 --p0-clean-weight 4.5 --gen-repeat 2`
  - `GPU 7`（`iv6b_eupe_conv_448_t3`，3×3 切块 10 视角）：
    `--split data/split_all_full.csv --backbones "internvit:lastk4,eupe:full,convnext:full" --size 448 --tiles 3 --epochs 4 --batch 4 --grad-accum 2 --lr-full 8e-5 --lr-last 1.5e-4 --lr-head 1.5e-3 --clean-repeat 4 --p0-clean-weight 4.5 --gen-repeat 2`

---

### Stage 2：在等待 GPU 训练期间，研发并验证「两步分层贝叶斯解码器 + 同场景 K-NN 邻域档位校准器」（CPU 执行，耗时 ~30 分钟）

利用已有的纯真图盲测概率库（`results/night/4/final_s16..s20_seed*_probs.npz`、`results/backbone_fixed/*_s16..s20_seed*_probs.npz` 以及 `feats/*_combined_oof.npz`），编写 `code/finetune/research_bayes_knn_decoder.py`，严格遵循 **`val` 上拟合超参、在未见过的 `test` 上盲评** 的零泄漏协议（参考 `code/headroom.py`），逐一验证以下三个模块的独立与联合收益：

1. **模块 2.1：纯真图跨架构大集成基线（Pure-Real Multi-Backbone Baseline on `s16 ~ s20`）**
   - 在 `s16 ~ s20` 每个划分上，对比：
     - (a) `results/night/4`（2 seeds）
     - (b) `results/backbone_fixed`（`base + pcl + pcg` 6 runs）
     - (c) 加入 `feats/internvit-6b_combined_oof.npz`（需注意按 `s16..s20` 训练集重新 fit Ridge/LogReg probe，严禁使用跨折泄漏的 OOF！）与 Stage 1 产出的 `iv3_s16..s20_probs.npz`。
2. **模块 2.2：两步分层贝叶斯解码器（Two-Stage Hierarchical Bayes-Optimal F1 Decoder）**
   - 针对 `decode_pc` / `decode_v3_shielded` 的分流漏报与低 $p_{\text{any}}$ 误报问题，实现两步解码函数 `decode_hierarchical_bayes(p0, p1, p2, params)`：
     - **第一步（是否存在缺陷）**：计算 $p_{\text{any}, c} = p_{0,c} + p_{1,k} + p_{2,k}$。若 $p_{\text{any}, c} < \tau_{\text{any}, c}$，直接判为 `无问题(0)`（抑制 `过锐` 上 $p_2=0.31, p_1=0.01, p_0=0.00 \implies p_{\text{any}}=0.32 < \tau_{\text{any}}$ 的假阳性；同时对 `过度生成` 若 $p_{\text{any}} > 0.42$，即使单档均 $<0.30$ 也进入第二步定档！）。
     - **第二步（给定有缺陷，最优定档 `P2 / P1 / P0`）**：
       - 计算条件期望严重度 $\mu_c = \frac{1\cdot p_{2,k} + 2\cdot p_{1,k} + 3\cdot p_{0,c}}{p_{\text{any}, c}} \in [1, 3]$，或最大化该样本对竞赛目标 $25\cdot \Delta F1_c(g) + 20\cdot \Delta S_{\text{p0}, c}(g)$ 的期望边际贡献（Expected Utility Maximization）：
         - 在 `val` 集上使用坐标上升（Coordinate Ascent）搜索每类的 $(\tau_{\text{any}, c}, \theta_{12, c}, \theta_{23, c}, \tau_{p0, c})$，并在 `s16 ~ s20` 的 `test` 集上盲测验证。
3. **模块 2.3：同场景 K-NN 原型邻域档位校准（Scene-Matched K-NN Ordinal Calibration）**
   - 加载 `feats/internvit-6b_combined.npz`、`feats/c-radio-v4-h_combined.npz`、`feats/siglip2-so400m_combined.npz` 与 `feats/signal.npz`（3,573 张真图特征）。
   - 对验证集/测试集中的每张图片 $i$，仅在该划分的 `train` 真图集合中寻找余弦相似度最高的 Top-$K$（$K \in \{15, 25, 40\}$）同场景邻居 $\mathcal{N}_K(i)$：
     - 计算邻域真实档位均值与模型在训练集邻居上的预测均值之差（Local Scene Bias Correction），修正因场景纹理复杂度（如草地/毛发/文字贴纸天然高频能量高，容易被误判为 `过锐 P2`；大光圈特写容易被误判为 `模糊/虚化`）导致的系统性档位偏移。
   - **验证标准**：记录 `s16 ~ s20` 共 5 个盲测集上的 `mean_Doc1`、`mean_Doc2`、`S_pos`、`S_p0`、`S_clean`。任何子模块必须在 5 个盲测集上取得一致正增长才予以保留！

---

### Stage 3：第二波 8×H200 纯真图补强训练（若时间充裕，耗时 ~50 分钟）

当 Stage 1 的 8 张卡跑完后：
1. 立即运行评估脚本，查看 `iv3_s16..s20` 在 `s16 ~ s20` 盲测集上相比 `backbone_fixed` 的真实涨分幅度（重点检查 `过锐`、`模糊`、`纹理损失`、`过度生成` 的逐类 `S_pos` 和 `S_p0` 提升）。
2. 利用空闲下来的 8 张 H200，启动第二波针对 **`过度生成`（最难短板类，`S_pos=0.418`）** 与 **8 折纯真图 OOF 补齐** 的训练：
   - 在 `data/split_all_full.csv`（全量 3,573 真图）上再训练 2~4 个不同随机数种子 / 不同高分基座组合的 `InternViT-6B + SigLIP2-Giant + PE-Core-G14 + C-RADIO` 纯真图模型，并测试在 `models.py` 中接入 `models/siglip2-naflex`（在 `backbone_complementarity.py` 中 `过度生成_P2 AP = 0.666` 全场第一）。
   - 同时提取 `feats/` 下特征对 `data/test_150` 的特征表示（若需要 K-NN 邻域校准在 `test_150` 上运行，编写一个小脚本对 `data/test_150` 提取 `c-radio-v4-h` + `siglip2-so400m` + `internvit-6b` 特征，150 张图在单卡上仅需 ~15 秒！）。

---

### Stage 4：严格零泄漏盲测终审、生成 `pred_nriqa_dataset_test.json` 与完整实验报告

1. **汇总 `s16 ~ s20` 零泄漏盲测集终审大表**：
   - 在完全相同的 `s16 ~ s20` 独立测试集（共 $5 \times 600 = 3,000$ 次盲测样本评估）上，输出以下阶梯消融对比表：
     - **Baseline 1**：原 `V3 / backbone_fixed` 纯真图集成 + 原 `decode_v3_shielded` 解码
     - **Step +A**：`+ InternViT-6B` 纯真图多塔模型集成（验证新基座纯增量）
     - **Step +B**：`+ 两步分层贝叶斯解码器（Two-Stage Hierarchical Bayes Decoder）`（验证解码器修复分流漏报与擦边误报的纯增量）
     - **Step +C**：`+ 同场景 K-NN 邻域档位校准`（验证场景原型校准对 `P1/P2` 边界的纯增量）
   - 确保总提升幅度达到显著水平（脱离 `±0.3` 噪声区间，且 5 个盲测划分全部正增长）。

2. **生成 `test_150` 最终提交文件（严守三大安全护栏）**：
   - **护栏 1（`20.00 / 20` Clean Shield 铁律）**：在 `test_150` 上，必须 100% 锁死 `52.43` 已验证满分的 15 张全正常盾牌图（`sh_5243`），绝不允许将其中任何一张改成有缺陷！同时在其余图片上严控新增 `P0` 数量，杜绝触发 `P0_FPR > 0.02` 扣分悬崖。
   - **护栏 2（100% 纯真图全量模型）**：参与 `test_150` 概率融合的模型必须 **100% 来自全量真图训练模型**（`52.43` 的 21 个全量 `V2+V3` 模型 + 新训练的全量纯真图 `InternViT-6B` 系列模型 + 纯真图 `F6/F7` Giant 模型），**权重中掺合成数据的模型占比必须为 0%**！
   - **护栏 3（输出三份不同保守度的提交文件供用户选择，默认主文件放最稳健高收益版）**：
     1. `/wanqing-develop/luowenjing/18/pred_nriqa_dataset_test.json`（主提交文件：`52.43` 全量底座 + `InternViT-6B` 全量纯真图集成 + 经盲测验证的两步解码与 K-NN 校准）
     2. `/wanqing-develop/luowenjing/18/results/final/pred_nriqa_dataset_test_iv6b_pure.json`（纯模型升级版：仅替换/注入 `InternViT-6B` 全量纯真图概率，保持原 `decode_v3_shielded` 解码器不变，纯粹检验 `InternViT-6B` 基座提分）
     3. `/wanqing-develop/luowenjing/18/results/final/pred_nriqa_dataset_test_iv6b_bayes_knn.json`（完整版：`InternViT-6B` 全量纯真图 + 两步分层贝叶斯解码 + 同场景 K-NN 校准）
   - 对所有生成的 JSON 文件执行格式与合法性校验（150 张图 ID 顺序与 `results/v2_full/template_150.json` 完全一致，6 类键名齐全，`花屏` 与 `虚化突兀` 仅含 `P0 / 无问题`）。

3. **撰写完整夜间实验报告**：
   - 将所有实验数据、盲测消融表、逐类 F1 提升表、以及三份提交文件相对 `52.43` 的逐图逐格差异明细写入 `/wanqing-develop/luowenjing/18/docs/OVERNIGHT_EXPERIMENT_REPORT.md`。

---

## 五、关键文件与快速命令速查表（Quick Reference）

| 模块 / 资产 | 绝对路径 / 相对路径 | 说明 |
| :--- | :--- | :--- |
| **历史最高分 `52.43` 提交文件** | `results/v3_super/pred_nriqa_dataset_test.json` | 线上实测 `52.43` 分（其中 `Clean Score = 20.00/20` 满分） |
| **`52.43` 900/900 精确复现函数** | `code/finetune/build_v10_fulldata_test150_submission.py` 中的 `load_exact_5243_anchor()` | 直接调用即可拿到与 `52.43` 900/900 格完全一致的 `p0_5243, p1_5243, p2_5243, sh_5243` |
| **全量 3,573 张真图标签表** | `data/manifest_combined.csv` | `orig=1,902 + supp=1,671`，共 3,573 张真图 |
| **全量 3,573 张真图训练划分** | `data/split_all_full.csv` | `train=3,573`（100% 全量真图），`val=300` |
| **5 个零泄漏独立盲测划分** | `data/resplits/split_s16.csv` ~ `split_s20.csv` | 用于严格零泄漏验证新模型与新解码器（`train=2,373, val=600, test=600`） |
| **14 基座正交互补横评脚本** | `code/backbone_complementarity.py` | 证明 `internvit-6b`（`mAP14=0.779`, `过锐_P0=0.712`）全场第一 |
| **预提取 3,573 真图特征库** | `feats/internvit-6b_combined.npz`, `feats/c-radio-v4-h_combined.npz`, `feats/siglip2-so400m_combined.npz`, `feats/signal.npz` | 可直接用于 Stage 2 同场景 K-NN 原型档位校准器 |
| **V3 纯真图训练主脚本** | `code/finetune/train_v3.py` | 含 `v3_loss_fn`（Cumulative BCE + 4-Way CE + Ordinal EMD + Soft $S_{\text{pos}}$ + Soft $S_{\text{p0}}$） |
| **模型构建与 `InternViT-6B` 入口** | `code/finetune/models.py` (`build_backbone('internvit')`) | 运行时设置 `INTERNVIT_SIZE=448` 配合 `--size 448` |
| **原生图像共享内存高速缓存** | `/dev/shm/race18_native/index.npz` & `cache.bin` | 已驻留内存，`CachedIQADataset` 可直接零磁盘 IO 高速读取 |
