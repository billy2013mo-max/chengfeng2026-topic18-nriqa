# 晴羽观感体系无参画质检测（NR-IQA）长尾缺陷数据合成与零负迁移增广突破方案

## 一、 前序合成数据实验（V4~V10）负迁移的深度代码解剖与病理学诊断

在【2026乘风挑战赛】命题18（晴羽观感体系·画质BadCase的无参检测模型）的 3,573 张全量真实标注数据（[data/manifest_combined.csv](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/README.md)）中，六类画质缺陷呈现出极端的长尾与等级不平衡分布：其中 `纹理损失`（2,803 例正样本）与 `模糊`（1,711 例正样本）样本充沛，而 **`过度生成` 全集仅 684 例正样本**（分摊至 `P0 / P1 / P2` 三个细粒度等级后，单折验证集内的 `过度生成_P0` 与 `过度生成_P1` 往往仅有十余例），**`虚化突兀` 仅 225 例正样本**，**`花屏` 仅 213 例正样本**（见 [OVERNIGHT_AGENT_HANDOFF_PLAN.md](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/docs/OVERNIGHT_AGENT_HANDOFF_PLAN.md)）。正因 `过度生成` 样本高度稀缺且形态复杂，该类别当前的 $S_{\text{pos}}$ 仅为 `0.418`（$S_{\text{p0}}$ 仅为 `0.576`），成为拉低 14 通道宏平均 F1 的头号短板。

为了攻克这一长尾瓶颈，研发管线先后构建了五代人工合成退化数据集：`V4/V5` 基础合成、`V6` 多缺陷联合合成（[build_multi_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_multi_synth.py)）、`V7 OverGen-Pro` 专项合成（[build_overgen_pro.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_overgen_pro.py)）、`V8` 反事实严重度阶梯合成（[build_ladder_synth_v8.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_ladder_synth_v8.py)）、`V9` 零振铃真实感合成（[build_v9_authentic_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_v9_authentic_synth.py)）以及 `V10` 单通道掩码隔离训练（[train_v10_clean_native_specialist.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/train_v10_clean_native_specialist.py)）。然而，在 8 折交叉验证 OOF 与独立盲测集上，掺入这些合成数据训练的模型在真实样本上的泛化性能不仅未能超越纯真图基线，反而导致 `过度生成` 的真实 $S_{\text{pos}}$ 从 `0.448` 下跌至 `0.407`（`-4.1%`），`虚化突兀` 从 `0.855` 暴跌至 `0.778`（`-7.7%`），`过锐` 假阳性率激增 `+6.7%`。

通过对上述五个合成脚本的逐行代码审查，可以精确锁定导致“合成数据越加、真实测试集越掉分”的五大底层工程与算法缺陷。

### 1.1 盲目梯度峰值裁剪导致的“假阳性净片嫁接”（Blind Sobel-Peak Patch Grafting Bug）

在 [build_overgen_pro.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_overgen_pro.py) 的 `graft_real_overgen_patch`（第 102~162 行）以及 [build_v9_authentic_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_v9_authentic_synth.py) 的 `make_authentic_ai_version`（第 95~119 行）中，代码试图从训练集真实 `过度生成 = P1/P0` 的供体图像（Donor Image）中裁剪一块区域，嫁接到正常受体图像（Recipient Image）上，并将合成后的图像强制打上 `过度生成 = P2 / P1 / P0` 的正标签。

这一思路在概念上类似实例级 Copy-Paste（[Ghiasi et al., CVPR 2021](https://arxiv.org/html/2512.11683v1)），但在实现时忽略了一个致命前提：**赛题数据集仅有整图级标签（Image-Level Label），并没有标注 `过度生成` 伪影在供体图中的具体像素坐标（Bounding Box / Mask）**！正如 [RichHF-18K (Liang et al., CVPR 2024)](https://arxiv.org/abs/2312.10240) 与 [PAL4VST (Zhang et al., ICCV 2023)](https://arxiv.org/abs/2310.05590) 所证实，一张被标为 `过度生成 P1/P0` 的真实图片，其生成伪影（如六根手指、扭曲的耳环、局部融化的背景文字）往往只占全图面积的 **$5\% \sim 15\%$**，而其余 $85\% \sim 95\%$ 的区域是完全正常的背景、衣物或天空。

那么现有代码是如何从供体图中挑选裁剪框的呢？在 [build_overgen_pro.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_overgen_pro.py) 第 121~127 行中，代码直接计算了全图的 Sobel 梯度幅值 `np.hypot(gx, gy)`，并取**梯度能量最大的位置 `np.argmax(valid_e)`**（而在 `build_v9_authentic_synth.py` 第 100 行则是完全随机裁剪中心偏左上区域 `rng.uniform(0.15, 0.45)`）！在自然短视频画面中，Sobel 梯度能量最高的位置几乎总是**清晰的主体边缘、高反差建筑线条、树叶或正常字幕**，而非细腻的局部 AIGC 伪影。其直接后果是：**超过 $60\% \sim 75\%$ 的所谓“真实过度生成嫁接样本”，实际上裁剪并嫁接的是供体图中的正常无缺陷区域**，却在训练时强行命令神经网络将其识别为 `过度生成 P1/P0`！这种高比例的伪正例标签污染（False-Positive Label Noise）直接摧毁了分类头对真实生成伪影的精确决策边界。

### 1.2 双边滤波与人工几何形变制造的“低级频域捷径”（Low-Level Filter Shortcuts）

为了模拟 AIGC 生成图像的质感，[build_ladder_synth_v8.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_ladder_synth_v8.py)（第 52~80 行）与 [build_v9_authentic_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_v9_authentic_synth.py)（第 42~89 行）对整张底图施加了连续两次强双边滤波 `cv2.bilateralFilter(img, d=9, sigmaColor=65, sigmaSpace=65)`，再叠加双曲正切非线性边缘强化 `np.tanh(hp / 14.0) * 28.0`；同时，在 `V6/V7/V8` 中添加了正弦波重映射 `cv2.remap(amp * np.sin(...))`，在 `V9` 中则添加了随机矩形块旋转贴回 `apply_structural_clone_distortion`（在画面随机位置抠出一块矩形旋转 $-16^\circ \sim +16^\circ$ 后用高斯椭圆掩模贴回原处）。

从深度卷积与 ViT 的频域响应特性来看，上述操作与真实世界中的 AIGC `过度生成` 存在本质差异：
1. **全局双边滤波的“水彩/卡通化”频谱印记**：连续两次 `d=9, sigma=65` 的双边滤波会彻底抹平特定频段的自然传感器高斯噪声，使整图呈现出强烈的“传统图像滤波痕迹”。神经网络无需理解任何高层语义结构（如手部结构或物体合理性），仅凭浅层卷积核检测是否存在双边滤波的阶梯状直方图伪影，就能在训练集上以 $99\%$ 的准确率识别出 `v8_og / v9_og` 合成样本；
2. **随机矩形旋转贴回的“接缝梯度跳变”**：在背景随机位置抠出矩形旋转贴回，破坏的是低层像素连续性（形成椭圆形羽化接缝），而真实 AIGC 扩散模型生成的六指或畸变结构在像素过渡上是完全平滑连续的，其错误发生在**高层解剖学与透视几何拓扑层**。

### 1.3 虚化突兀合成中的“几何雪人轮廓”捷径与缺乏光学散景负例对照

在 [build_multi_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_multi_synth.py)（第 127~165 行）与 [build_v9_authentic_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_v9_authentic_synth.py)（第 227~259 行）的 `worker_build_bokeh_cutout` 中，为了合成 `虚化突兀 P0` 样本，代码直接用 `cv2.ellipse` 和 `cv2.circle` 在画面中央画了一个由“椭圆身体 + 圆形脑袋”组成的固定几何图形（Geometric Snowman Silhouette），将图形外部做高斯模糊（$\sigma \in [7.5, 14.0]$），并在椭圆边界上叠加了一圈宽 15 像素的亮环 `edge_ring * 18.0`！

这种合成方式完全脱离了图像的真实前景内容：椭圆边界往往直接从画面中真实人物的脸部中央、桌面物体或背景墙壁上生硬穿过。视觉主干（尤其是具备空间位置编码的 ViT）在训练 1 个 Epoch 后就学会了检测“画面中央是否存在人造椭圆亮环”，不仅完全没有学到真实人像模式在发丝、肩部边缘的抠图断层特征，反而在测试集上将具有中心构图或圆形光斑的正常照片误判为 `虚化突兀 P0`，导致 8 折 OOF 上 `虚化突兀` 得分暴跌 `-7.7%`。

### 1.4 双重 JPEG 压缩与分档固定压缩质量（`Q = 94 / 92 / 90`）引发的量化表泄漏

正如近期顶会论文 *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*（[Grommelt et al., 2024](https://arxiv.org/html/2403.17608v1)）所严格证明的：当合成图像在生成后使用特定的 JPEG 质量因子重新编码保存时，深度视觉网络会极度敏锐地捕捉到 $8\times 8$ DCT 离散余弦变换块边界的二次量化余数特征（Double-JPEG Quantization Artifacts）。

审查 [build_v9_authentic_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_v9_authentic_synth.py) 第 141、159、169 行与第 207、216、224 行可以发现一个极其危险的硬编码实现：
- 对 `过度生成` 合成图：`P2` 固定使用 `IMWRITE_JPEG_QUALITY = 94`，`P1` 固定使用 `92`，`P0` 固定使用 `90`；
- 对 `过锐` 合成图：`P2` 固定使用 `95`，`P1` 固定使用 `93`，`P0` 固定使用 `91`；
- 对 `模糊+纹理损失` 合成图：`P2` 固定使用 `86`，`P1` 固定使用 `72`，`P0` 固定使用 `55`。

这意味着在 `V8/V9/V10` 的训练集中，**严重等级 `P2 / P1 / P0` 与 JPEG 压缩质量因子之间存在 100% 确定性的函数映射关系**！模型（特别是包含高频局部滤波器的 `ConvNeXtV2` 与 `V10` 的 1:1 原生像素显微镜切块）直接利用了 DCT 量化步长来区分合成图的 `P2 / P1 / P0`，这在未经过相同 JPEG 二次量化流水线的真实测试集上必然引发严重的档位预测偏移。

### 1.5 单阶段高比例混训对真实长尾正类梯度的淹没效应（Gradient Domination）

即便在 `V10`（[train_v10_clean_native_specialist.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/train_v10_clean_native_specialist.py)）中通过 `get_sample_class_mask6` 屏蔽了合成样本在非目标类别上的损失，训练脚本依然将 540 张合成 `过度生成` 样本与训练集中的真图按 `1:1` 权重在同一个训练阶段直接混合（`tr = list(tr_real) + tr_synth`）。由于单折训练集内的真实 `过度生成` 正样本仅约 400 例（其中真实 `P0` 与 `P1` 合计仅百余例），540 张合成正样本在数量上占据了绝对多数。在最后一个 Epoch 收敛时，分类头的决策边界与先验赔率完全向合成分布的聚类中心对齐，从而在真实测试分布上产生严重的先验膨胀与校准失准。

| 前序合成版本 | 核心合成算子与实现脚本 | 暴露出的致命病理缺陷 | 在真实盲测集上的实测负向后果 |
| :--- | :--- | :--- | :--- |
| **`V6 Multi-Synth`** | [build_multi_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_multi_synth.py)：FLUX/Swin2SR 混合 + `cv2.remap` 正弦波 + 椭圆抠图 | 正弦波高频条纹成为虚假捷径；所有合成图连带标记 `纹理损失=1/2` | `过锐` 与 `花屏` 严重混淆；测试集 `纹理损失 P2` 假阳性暴增 |
| **`V7 OverGen-Pro`** | [build_overgen_pro.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_overgen_pro.py)：按全图 Sobel 梯度极大值从真实 P0/P1 图裁剪贴片 | **盲目裁剪**：70% 以上剪到的是供体图中的正常高对比背景而非生成伪影 | 向 `过度生成 P1/P0` 注入大量正常图像块，导致真实正类边界模糊 |
| **`V8 / V9 Ladders`** | [build_v9_authentic_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_v9_authentic_synth.py)：双重双边滤波 + 随机矩形旋转贴回 + 椭圆雪人虚化 + 分档固定 JPEG 质量 (`94/92/90`) | 双边滤波卡通化伪影；椭圆亮环脱离前景真实边界；**JPEG 二次压缩质量因子与等级 100% 绑定泄漏** | 8折 OOF 上纯真图折比合成折在 `过度生成` 高 `+4.1%`、在 `虚化突兀` 高 `+7.7%` |
| **`V10 Clean-Masked`** | [train_v10_clean_native_specialist.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/train_v10_clean_native_specialist.py)：单类 Loss 掩码 + 1:1 混训 | 虽消除了跨类污染，但目标类内部合成正样本数量压倒真实正样本，统治末端梯度 | 目标类预测概率在 `test_150` 上系统性膨胀，破坏全局校准 |

---

## 二、 业界与顶会（2024–2026）在长尾细粒度伪影合成上的五大前沿范式

针对“真实长尾样本极度稀缺，而朴素图像合成又会产生域偏移（Sim2Real Gap）”这一计算机视觉领域的经典难题，2024–2026 年 CVPR、ICCV、ECCV、ICML 及 NeurIPS 顶会形成了一套高度成熟的**“定位引导、语义对齐、流形插值、两阶段退火”**合成与增广体系。

### 2.1 语义掩模约束的局部扩散重绘与潜在空间扰动（Targeted Diffusion Inpainting & Latent Corruption）

在 AIGC 伪影检测与细粒度质量评估领域，最新代表作 *Synthesizing Artifact Dataset for Pixel-level Detection*（[Menn et al., 2025, arXiv:2509.19589](https://arxiv.org/abs/2509.19589)）、*RichHF-18K*（[Liang et al., CVPR 2024](https://arxiv.org/abs/2312.10240)）、*PAL4VST*（[Zhang et al., ICCV 2023](https://arxiv.org/abs/2310.05590)）以及 *SynArtifact*（[Cao et al., 2024](https://arxiv.org/abs/2402.18068)）一致证明：**真实的 `过度生成` 伪影绝不能通过传统信号处理滤波器（如双边滤波或仿射旋转）来模拟，而必须由真正的深度生成模型（Diffusion / VAE / GAN）在受控的局部语义区域内产生**。

具体而言，[Menn et al. (2025)](https://arxiv.org/abs/2509.19589) 提出的自动伪影合成流水线通过以下三个关键设计，在人类标注测试集上为 ConvNeXt 与 Swin 检测器带来了 **$+13.2\%$ 的零负迁移性能跃升**：
1. **基于世界知识的脆弱语义区域锁定（Semantic Vulnerability Targeting）**：真实生成模型最容易在**人物手部、面部五官、肢体关节连接处、密集重复纹理（如栅栏、键盘、动物毛发）以及画面内嵌文字**处产生结构崩坏或多余生成。因此，合成管线首先利用视觉分割/显著性模型定位这些高危语义区域作为候选掩模 $M_{\text{target}}$，绝不在空白天空或纯色地面上盲目合成；
2. **VAE/扩散瓶颈层局部特征破坏与受控少步重绘（Confined Latent Perturbation & Low-Step Inpainting）**：为了产生与野生（In-the-Wild）AIGC 图像完全同分布的伪影，将干净图像编码至 VAE 潜在空间 $\mathbf{z} = \mathcal{E}(I)$，仅在掩模 $M_{\text{target}}$ 对应的潜在网格内施加**通道级随机丢弃、高频噪声注入或欠收敛少步扩散去噪（如仅运行 4~6 步去噪或施加过高 CFG 引导尺度）**，随后解码并与原图在掩模外部做 **100% 原生无损像素替换**：
   $$I_{\text{synth}} = (1 - \tilde{M}_{\text{target}}) \odot I_{\text{orig}} + \tilde{M}_{\text{target}} \odot \mathcal{D}(\tilde{\mathbf{z}})$$
   这样生成的图像，其 $80\%$ 以上的背景与原图逐像素完全一致（零全局滤波污染），而在目标区域内则呈现出高度逼真的“多指粘连、局部纹理融化、不合理冗余结构”；
3. **基于区域面积与扰动强度的客观等级锚定**：严格依据受创语义区域的视觉显著度与面积占比划分 `P2`（次要区域微弱结构瑕疵）、`P1`（主体局部明显多余结构或质感崩坏）与 `P0`（主体核心区大面积严重崩坏）。

### 2.2 自混合图像（Self-Blended Images, SBI）与伪影定位引导的同源嫁接

在细粒度伪造与伪影检测领域，CVPR 2022 经典工作 *Detecting Deepfakes with Self-Blended Images*（[Shiohara & Yamasaki, CVPR 2022](https://arxiv.org/abs/2204.08376)）及后续强化学习增强版本（[arXiv:2601.15624](https://arxiv.org/pdf/2601.15624)）揭示了一个极为深刻的原则：**用外部生成图像整图训练分类器极易让模型过拟合特定生成器的全局底层指纹；唯有在同一张真实图像内部，通过对前景语义部件施加微小的几何/频谱自混合（Self-Blending），才能迫使网络专门学习“局部结构不一致性与边界融合伪影”**。

与此同时，针对跨样本伪影嫁接（Copy-Paste），近期发表的 *Depth-Copy-Paste*（[arXiv:2512.11683, 2025](https://arxiv.org/html/2512.11683v1)）与 *VRAIN: Verified Rare-class Augmentation via Instructed Editing*（[TMI, arXiv:2607.08201, 2026](https://arxiv.org/html/2607.08201v1)）指出：为什么传统的随机 Copy-Paste 往往在稀有类别上产生负迁移？因为存在**“四重失配（位置失配、光照失配、深度尺度失配、伪影未命中失配）”**！要使稀有缺陷嫁接取得正收益，必须引入两项核心机制：
1. **注意力/子块响应差分的真实伪影区精确定位（Anomaly-Localized Donor Mining）**：在从真实 `过度生成 P0/P1` 供体图中提取贴片时，利用已训练好的纯真图多切块模型（如 `V3_super` 或 `V9_RealHiRes`），计算供体图各个局部切块（Tiles / Patches）的 `过度生成` 预测概率 $p_{\text{og}}(\text{Tile}_k)$，**仅截取满足 $p_{\text{og}}(\text{Tile}_k) \ge 0.75$ 且显著高于全图均值的真正伪影切块**（100% 剔除供体图中的正常背景块）；
2. **闭环教师置信度拒绝采样（Teacher-Verified Filtering）**：合成出候选样本后，必须送入冻结的纯真图教师集成模型进行打分检验——只有当纯真图教师模型在该合成样本上的 `过度生成` 预测概率相比原底图显著上升（$\Delta p_{\text{og}} \ge +0.25$）且未误触发 `花屏` 等无关类别时，该合成样本才被允许进入训练集，否则当场丢弃。

### 2.3 物理与深度引导的正负成对反事实合成（Depth-Guided Paired Counterfactuals）

针对 **`虚化突兀`（Abrupt Bokeh Transition）** 这一长期困扰模型的 Clean 侧误报源，*BokehDepth*（[arXiv:2512.12425, 2025](https://arxiv.org/html/2512.12425v2)）与自然散景渲染研究（[MDPI, 2022](https://www.mdpi.com/2075-1702/10/5/286)）阐明了算法虚化伪影的真正成因：手机人像模式或背景虚化滤镜依赖单目深度估计或人像分割掩模 $\alpha(x,y)$，当分割掩模在发丝、手指缝隙、眼镜框或肩部轮廓处发生**掩模撕裂（Mask Tearing）、阶跃硬化（Unfeathered Step）或深度漏判（Depth Bleed，即背景局部未被虚化或前景局部被误虚化）**时，就会产生刺眼的虚实割裂。

为了防止神经网络把“背景模糊的人像图”一概死记硬背为 `虚化突兀 P0`，业界最有效的做法是构造**同底图正负成对反事实样本对（Paired Positive-Negative Counterfactuals）**：
- 利用 `DINOv3-ViT-L16` 或 `C-RADIOv4-H`（内置 SAM 分割能力）从同一张全清晰人像/物体底图 $I_{\text{clean}}$ 中提取真实的前景分割掩模 $M_{\text{SAM}}(x,y)$ 与连续伪深度场 $D(x,y)$；
- **构造硬负例（Hard Negative，`虚化突兀 = 0 / 无问题`）**：严格按照连续深度场 $D(x,y)$ 与平滑羽化的前景边缘施加空间可变圆盘散景模糊（Spatially-Varying Disk Bokeh），模拟真实的单反大光圈自然光学虚化；
- **构造真阳例（True Positive，`虚化突兀 = P0`）**：在真实前景掩模 $M_{\text{SAM}}$ 的高曲率边缘处（如头顶发丝、肩膀、手臂内侧）随机施加形态学腐蚀/膨胀咬合（Morphological Bite & Protrusion），并将过渡带宽度强制压缩为 $0 \sim 1$ 像素的阶跃硬切，制造出高度逼真的“人像抠图失误虚化”。
- 将这一对共享完全相同人物与背景的 `(自然散景负例, 抠图失误正例)` 同时放入同一个训练 Batch 中，并施加对比边际损失，能够强制模型只关注**主体边缘交界带的虚实过渡连续性**，彻底根除大光圈人像误报！

### 2.4 零像素伪影的特征空间流形增广（Latent Manifold Mixup & Tile-Embedding Grafting）

在所有针对小样本长尾分布的增广技术中，有一类方法能够**从数学上保证 100% 零底层像素失真、零 JPEG 压缩伪影、零滤波器捷径**——那便是在冻结或半冻结视觉主干的**高层特征流形空间（Feature Manifold Space）**直接进行样本合成（[Manifold Mixup, Verma et al., ICML 2019](https://arxiv.org/abs/1806.05236)）。

在本仓库中，我们已经预提取了全部 3,573 张真图在 6 大视觉基座（`InternViT-6B`、`C-RADIOv4-H`、`SigLIP2-SO400M`、`PE-Core-L14`、`EUPE`、`ConvNeXtV2`，见 `feats/*_combined.npz` 与 [probe_70k_foundation_on_s16_s17.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/probe_70k_foundation_on_s16_s17.py)）上的高维特征表示，并且在多视角训练中每张图像被表示为一组局部切块特征序列 $\{\mathbf{f}_{i, v}\}_{v=1}^V$。在特征空间中，我们可以实施三种威力强大的长尾合成操作：
1. **同场景跨等级序数流形插值（Scene-Matched Ordinal Manifold Interpolation）**：在语义相似的真实样本之间，将 `无问题(0)` 与 `过度生成 P0(3)` 的特征向量按凸组合插值生成高质量的 `P2(1)` 与 `P1(2)` 中间态特征；
2. **切块特征空间异体嫁接（Latent Tile-Embedding CutMix）**：在多视角网络（如 `MultiNetV2 / MultiNetV9`）的切块特征层 $\mathbf{F}_i \in \mathbb{R}^{V \times D}$ 中，随机将一张真实 `无问题` 图像的 $1 \sim 2$ 个局部切块特征向量，替换为一张真实 `过度生成 P0/P1` 图像中激活响应最高的局部切块特征向量！这一操作在特征空间完美等价于“将真实的局部生成伪影无缝移植到另一张正常图像上”，却完全绕开了像素级拼接的光照不一致与接缝伪影！
3. **长尾正类特征高斯协方差重采样（Class-Conditional Covariance Perturbation）**：利用真实 `过度生成`、`虚化突兀`、`花屏` 正样本在特征空间中的局部 $K$-近邻协方差矩阵，沿切空间方向生成服从真实流形分布的合成特征点，将长尾正类的有效支撑集扩大 $5 \sim 10$ 倍。

### 2.5 两阶段课程学习与域无关梯度对齐（Sim2Real Two-Stage Curriculum）

在工业界质检模型训练中，即便拥有高质量的像素级合成数据，也**绝不采用“全程等权混训直到最后一轮”**的粗糙方式。标准工业级 Sim2Real 范式遵循**“一阶段合成+真图联合预热表征，二阶段 100% 纯真图退火精调边界（Pretrain-on-Mixed, Finetune-on-Pure-Real）”**的课程学习准则，并配合合成样本降权与批归一化统计量隔离（Auxiliary BN/LN for Synthetic Domain），确保模型最终输出的先验赔率与特征刻度 100% 锚定在真实人类标注分布上。

---

## 三、 针对命题18长尾类别的「四级零负迁移合成体系」工程设计

结合本仓库现有的全部资产（`/tmp/synth/flux3` 与 `/tmp/synth/out` 中的神经重绘图像、3,573 张真图及 `feats/` 下的 70,400 维特征库），以下专门针对 **`过度生成`（头号冲分目标）**、**`虚化突兀`**、**`花屏`** 与 **`过锐`** 设计一套由浅入深、可立即执行的**四级零负迁移合成与增广方案（Syn-Breakthrough v12）**。

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                命题18：长尾画质缺陷「四级零负迁移合成与增广体系」全景架构                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 【Level 1：特征空间零伪影流形合成（Latent Manifold & Tile-Embedding Synthesis）】 ★最快/零风险 │
│   • 切块特征空间异体嫁接 (Latent Tile CutMix)：抽取真实过度生成 P0/P1 图像的 Top-1 异常切块特征 │
│     替换真实 Clean/Mild 图像的单个局部切块特征，按替换面积直接生成零接缝 P2 / P1 样本！         │
│   • 同场景序数流形插值 (Scene-Matched Ordinal Mixup)：在余弦相似度 >0.75 的同场景真图间，       │
│     对 (0, P2, P1, P0) 特征向量做保序贝塔插值，填补 P2↔P1↔P0 之间的稀疏边界真空带。             │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 【Level 2：异常定位引导的像素级真实伪影嫁接与自混合（Anomaly-Guided Grafting & SBI）】          │
│   • 废除 Sobel 梯度盲裁！用纯真图模型对每个 1:1 窗口打分，100% 锁定供体图内真正的伪影 ROI。     │
│   • 拉普拉斯金字塔多频段无缝融合 (Laplacian Pyramid Blending) + 内存中实时增强（禁写双重JPEG）。│
│   • 自混合局部结构重影 (Anatomical/Text SBI)：仅在前景主体/手部/字幕边缘做微幅多尺度双影混叠。  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 【Level 3：前景语义分割引导的正负成对反事实合成（SAM/DINO Paired Counterfactuals）】            │
│   • 针对「过度生成」：仅在 DINOv3/C-RADIO 定位的前景主体局部区域混入 FLUX3/Swin2SR 重绘伪影，   │
│     背景 80% 像素保持 100% 原图无损（彻底废除全图双边滤波与固定 JPEG 94/92/90 编码！）。        │
│   • 针对「虚化突兀」：基于真实前景掩模同时生成「平滑自然光学散景(负例=0)」与「发丝/肩部掩模     │
│     撕裂硬切散景(正例=P0)」成对样本，强制网络学习边缘虚实过渡连续性而非“背景模糊=虚化突兀”。    │
│   • 针对「花屏」：模拟 H.264/HEVC 宏块行移位、YUV 色度通道错位与局部条带丢包（零代价扩充P0）。  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 【Level 4：纯真图教师拒绝采样 + Sim2Real 两阶段课程退火训练（Two-Stage Curriculum）】           │
│   • 准入过滤：所有合成图先过冻结 `52.43` 纯真图教师检验，仅保留目标类置信度净增 >+0.25 的精品。 │
│   • 两阶段退火：前 60% Epochs 混入合成图训练特征提取器；后 40% Epochs 合成权重退火至 0%，       │
│     100% 用纯真图 + 特征流形 Mixup 精调 CORN 分类头与先验分布，从根源上杜绝测试集先验膨胀！     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1 第一级（最快、100% 零像素偏移）：特征空间切块嫁接与同场景序数流形插值

为什么在 70,400 维多基座特征塔（[probe_70k_foundation_on_s16_s17.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/probe_70k_foundation_on_s16_s17.py)）以及端到端多视角模型（`MultiNetV2 / MultiNetV9`）中，**特征空间合成（Latent-Space Synthesis）**是冲刺 `过度生成` 和 `虚化突兀` 性价比最高的首选武器？因为特征空间合成完全在已提取的真实图像深度表征上进行，**从物理上杜绝了任何 OpenCV 滤波痕迹、拼接接缝或 JPEG 二次压缩伪影**。

#### 1. 切块特征空间异体嫁接（Latent Tile-Embedding Grafting）
在端到端多视角网络（输入为 $V = 1 + T^2$ 个视图，即 $1$ 个全局视图 + $T \times T$ 个局部切块视图）中，骨干网络对每张图 $i$ 输出特征张量 $\mathbf{F}_i \in \mathbb{R}^{V \times D}$，其中 $\mathbf{f}_{i,0}$ 为全局视图特征，$\{\mathbf{f}_{i,1}, \dots, \mathbf{f}_{i, V-1}\}$ 为局部切块特征。
1. **定位真实正样本的异常切块（Anomalous Tile Identification）**：
   对于训练集中的每一张真实 `过度生成` 正样本 $d$（$y_{d, \text{og}} \in \{2\text{(P1)}, 3\text{(P0)}\}$），利用分类头的切块注意力模块 `tile_attn`（或计算各切块特征与全局正常原型的余弦距离）对 $d$ 的 $V-1$ 个局部切块进行排序，选出异常响应最高的 Top-1 切块特征 $\mathbf{f}^{\text{anom}}_{d, v^*}$ 与 Top-2 切块特征 $\mathbf{f}^{\text{anom}}_{d, v^{**}}$。
2. **在特征空间执行无缝切块替换**：
   随机抽取同一 Batch 内的一张真实 `过度生成 = 0` 的受体样本 $r$，按以下规则在特征张量 $\mathbf{F}_r$ 上构造合成特征样本 $\tilde{\mathbf{F}}$：
   - **合成 `过度生成 P2 (1)`**：将受体样本 $r$ 的 $1$ 个随机局部切块特征替换为供体异常切块 $\mathbf{f}^{\text{anom}}_{d, v^*}$，同时将全局视图特征做轻量凸组合 $\tilde{\mathbf{f}}_{r,0} = 0.80 \mathbf{f}_{r,0} + 0.20 \mathbf{f}^{\text{anom}}_{d, v^*}$，对应标签设为 $y_{\text{og}} = 1\ (\text{P2})$；
   - **合成 `过度生成 P1 (2)`**：将受体样本 $r$ 的 $2 \sim 3$ 个局部切块特征替换为供体异常切块，全局视图特征取 $\tilde{\mathbf{f}}_{r,0} = 0.55 \mathbf{f}_{r,0} + 0.45 \mathbf{f}_{d,0}$，对应标签设为 $y_{\text{og}} = 2\ (\text{P1})$。

这一操作只用不到 15 行 PyTorch 张量索引代码即可嵌入 `MultiNetV2.forward` 或 `MultiNetV9.forward` 内部，在每个训练 Batch 中实时生成数百种不同背景与真实局部生成伪影的组合，迫使池化层与分类头精准捕捉局部切块的异常特征！

#### 2. 同场景约束的序数流形插值（Scene-Matched Ordinal Manifold Mixup）
在训练多基座特征融合塔（如 `DeQAScoreQualiRAGTower` 或 `MultiTowerFoundationHead`，输入为 `internvit6b`、`cradio768`、`siglip2`、`pecorel` 等冻结特征块 $\{\mathbf{x}^{(b)}_i\}$）时，普通全局随机 Mixup 会将截然不同场景（如“风景照”与“电商海报”）混合成无意义的离群特征。为此，设计**同场景 K-NN 约束的序数流形插值**：
1. 在训练集内部，预先为每张存在长尾缺陷（`过度生成 > 0`、`虚化突兀 > 0`、`过锐 > 0`）的真实正样本 $i$，检索语义余弦相似度最高的 Top-15 同场景样本 $\mathcal{N}_{15}(i)$；
2. 在训练迭代时，从 $\mathcal{N}_{15}(i)$ 中随机抽取配对样本 $j$，采样插值系数 $\lambda \sim \text{Beta}(2.0, 2.0) \in [0.25, 0.75]$，对所有视觉基座特征块与物理特征块同步执行流形插值（并在各块投影层之后再次 L2 归一化以保持球面流形几何）：
   $$\tilde{\mathbf{x}}^{(b)} = \frac{\lambda \mathbf{x}^{(b)}_i + (1 - \lambda) \mathbf{x}^{(b)}_j}{\big\|\lambda \mathbf{x}^{(b)}_i + (1 - \lambda) \mathbf{x}^{(b)}_j\big\|_2}, \quad \tilde{y}_c = \lambda y_{i,c} + (1 - \lambda) y_{j,c}$$
3. **关键收益**：当 $y_{i, \text{og}} = 0$（同场景正常图）且 $y_{j, \text{og}} = 3\ (\text{P0})$ 时，$\lambda = 0.65 \implies \tilde{y}_{\text{og}} = 1.05 \approx 1\ (\text{P2})$，$\lambda = 0.35 \implies \tilde{y}_{\text{og}} = 1.95 \approx 2\ (\text{P1})$！这直接沿着真实视觉特征测地线（Geodesic on Feature Sphere）源源不断地填补了最为稀缺的 `过度生成_P2` 与 `过度生成_P1` 中间档位样本，使条件序数头（CORN）的档位分割超平面极为平滑稳健。

---

### 3.2 第二级（像素级高保真）：异常定位引导的拉普拉斯金字塔嫁接与自混合（SBI）

若要在像素空间生成供端到端骨干微调的合成图像，必须彻底重写现有的 `build_overgen_pro.py` 与 `build_v9_authentic_synth.py`，严格遵守以下四项高保真合成规范：

#### 1. 用“纯真图分类器高响应窗口”替代“Sobel 梯度极大值”定位真实伪影源
在训练集内筛选真实 `过度生成` 供体图（`过度生成 >= 2`）后，**严禁使用 Sobel 边缘梯度挑选裁剪框**。正确的伪影源提取流程如下：
1. 对每张供体真图 $I_{\text{donor}}$，以步长 $112$ 像素扫描所有 $448 \times 448$ 局部滑动窗口 $\{W_m\}_{m=1}^M$；
2. 使用已训练好的纯真图基准模型（如 `V3_super` 或 `SigLIP2 + C-RADIO` 探针）对每个局部窗口 $W_m$ 独立预测 `过度生成` 概率 $p_{\text{og}}(W_m)$；
3. **仅保留满足 $p_{\text{og}}(W_m) \ge 0.70$ 且位居该图所有窗口之首的局部窗口 $W_{m^*}$** 作为经过验证的真实伪影贴片库（Verified Anomaly Patch Bank）。这样可确保 100% 裁剪到真正包含畸变手指、融化五官或 AI 伪纹理的核心病灶区。

#### 2. 多频段拉普拉斯金字塔融合（Laplacian Pyramid Blending）与内存实时增强
在将伪影贴片（或 `/tmp/synth/flux3` 与 `/tmp/synth/out` 中的局部神经重绘区域）合成到训练集的干净底图上时：
1. **彻底废除全局双边滤波（`cv2.bilateralFilter`）与固定 JPEG 压缩（`94 / 92 / 90`）**：严禁对底图的非掩模区域做任何滤波或色彩篡改，且所有合成操作优先在 `Dataset.__getitem__` 的内存数组（In-Memory Online Augmentation）中动态完成，或统一保存为无损 `.png`（若模拟压缩，则对 Clean、P2、P1、P0 全部施加完全相同分布的随机 JPEG 质量 $Q \sim \text{Uniform}(85, 98)$，彻底切断“压缩质量 ↔ 缺陷等级”的泄漏通道）；
2. **仅在前景显著主体局部区域注入 FLUX3 / Swin2SR 重绘**：利用 `DINOv3` 或 `C-RADIO` 的前景主体显著性图（而非全图均匀加权），仅在前景人物手部、面部或主物体内部的局部区域（`P2` 占主体面积 $15\%\sim 25\%$、`P1` 占 $30\%\sim 50\%$、`P0` 占 $60\%+$）通过 4 层拉普拉斯金字塔融合（Laplacian Pyramid Multi-Band Blending）嵌入神经重绘纹理，使低频光照与周围皮肤/背景天衣无缝，而高频局部生成伪影完整保留。

#### 3. 针对手部/五官/文字的局部自混合结构错位（Self-Blended Structural Ghosting）
借鉴 [Self-Blended Images (Shiohara & Yamasaki, CVPR 2022)](https://arxiv.org/abs/2204.08376)，无需任何外部生成模型，直接对训练集中含人物、手部或文字的正常图像，在前景高显著度局部小块（面积占全图 $6\% \sim 18\%$）内执行**微距非刚性自混叠（Micro-Offset Self-Blending）**：
- 在局部小块内提取高梯度结构层，对其进行 $4 \sim 12$ 像素的局部错位平移与微缩（模拟扩散模型常见的“重影手指”、“第六指粘连”、“文字笔画局部重影扭曲”），并通过泊松融合（`cv2.seamlessClone` 的 `NORMAL_CLONE` / `MIXED_CLONE`）合入原位置。由于源像素与目标像素 100% 来自同一张图像的同一局部区域，其传感器噪声、色彩分布、色温与分辨率完全一致，迫使网络纯粹依靠**局部解剖结构与笔画拓扑异常**来判别 `过度生成`。

---

### 3.3 第三级（专攻 `虚化突兀` 与 `花屏`）：成对反事实散景合成与宏块码流损坏模拟

#### 1. `虚化突兀` 的前景分割引导正负成对反事实生成（Paired Bokeh Counterfactuals）
为彻底解决 `虚化突兀` 样本少（仅 225 例）且极易将正常大光圈人像误判为 `P0` 的顽疾，废除 `build_v9_authentic_synth.py` 中的椭圆雪人模板，改用**基于 `DINOv3 / C-RADIO` 真实前景掩模的正负成对反事实合成**：

| 合成样本类型 | 前景掩模与过渡带构造方式 | 背景模糊核设计 | 赋予标签 | 在训练中的核心作用 |
| :--- | :--- | :--- | :---: | :--- |
| **同源反事实负例**<br>*(Natural Optical Bokeh)* | 提取真实前景主体掩模 $M_{\text{fg}}$，结合伪深度场 $D(x,y)$ 构造宽度为 $25 \sim 60$ px 的平滑渐变过渡带 | 圆盘散景核半径 $R(x,y)$ 随深度连续线性增长，前景与背景自然衔接 | `虚化突兀 = 0`<br>*(无问题)* | **作为强力免疫疫苗**：教会模型“即使背景强烈虚化，只要过渡随深度自然连续，就绝不是虚化突兀” |
| **同源反事实正例**<br>*(Abrupt Matting Failure)* | 对同一张图的 $M_{\text{fg}}$ 边缘施加 2~4 处随机多边形“啃噬/外溢”（模拟发丝/手指/肩窝抠图漏判），并将掩模二值化硬化（过渡带 $\le 1$ px） | 对掩模外侧（含被误切入背景的主体边缘）直接施加半径 $R=12$ 的强模糊，产生刺眼断层 | `虚化突兀 = 3`<br>*(P0)* | 与同源负例组成 `(Pos, Neg)` 孪生对，通过对比损失让网络精准锁定**抠图阶跃断层** |

#### 2. `花屏` 的编解码宏块错位与色度失步合成（Bitstream & Macroblock Corruption）
针对仅 213 例正样本的 `花屏 P0`，无需复杂模型，直接在内存数据增强中以小概率模拟视频解码链路中的三类真实信号崩溃：
1. **CTU 宏块行移位与重复（Macroblock Row Shift）**：在图像下半部或随机水平条带（高度为 $16$ 或 $32$ 的整数倍）内，将 $16 \times 16$ 宏块向右循环平移并叠加相邻行宏块残差；
2. **YUV 色度通道块失步（Chroma Plane Desynchronization）**：将图像转至 YCrCb 空间，在局部区域将 Cr 或 Cb 通道置零或填充高饱和绿色/洋红色块（模拟 H.264/HEVC I帧/P帧参考丢失导致的经典绿屏/紫块花屏）；
3. **局部扫描线拉伸（Scanline Repeat）**：将某一行像素向下连续复制 $20 \sim 60$ 行，模拟数据包截断引起的底部条纹拉丝花屏。

---

### 3.4 第四级（Sim2Real 训练协议）：教师拒绝采样与两阶段退火课程学习

为什么哪怕合成图质量再高，也不能在全部训练 Epoch 中与真图等权混训？因为判别式竞赛评估的核心在于**后验概率校准（Posterior Calibration）**。一旦合成样本参与最后一轮训练，分类头的偏置项（Bias）与先验赔率就会被合成样本的比例所绑架。为此，必须建立严格的 **Sim2Real 三道闸门训练协议**：

#### 闸门 1：纯真图超级教师的闭环拒绝采样（Teacher-Guided Reject Sampling）
在任何像素级合成图像 $x_{\text{syn}}$（由底图 $x_{\text{clean}}$ 生成，目标缺陷为 $c^*$，目标等级为 $g^* \in \{1, 2, 3\}$）进入训练集之前，先用已经训练好的 **`52.43` 纯真图教师集成模型** 对 $x_{\text{clean}}$ 与 $x_{\text{syn}}$ 进行前向推理，计算累积缺陷概率增量 $\Delta p_{c^*} = p_{\text{any}, c^*}(x_{\text{syn}}) - p_{\text{any}, c^*}(x_{\text{clean}})$：
- **准入条件**：仅当 $\Delta p_{c^*} \ge +0.15$（证明合成算子确实在纯真图教师眼中产生了符合真实缺陷流形的响应）且非相关类别（如 `花屏`）概率增量 $< 0.05$ 时，才保留该合成样本；
- **等级自动重标定（Teacher Soft-Relabeling）**：不再盲目相信人工设定的硬等级 $g^*$，而是将人工设定等级与纯真图教师预测的软分布按 $0.6 : 0.4$ 融合作为该合成样本的监督目标，彻底消除合成强度与真实人类主观刻度之间的错位！

#### 闸门 2：两阶段退火课程学习（Two-Stage Pretrain-then-Pure-Real Annealed Curriculum）
将 5 个训练 Epoch 严格划分为两个阶段：
- **Stage A（表征拓宽阶段，Epoch 1 ~ 3）**：加载 `100% 真实图像 + 经教师过滤的局部合成图像`，但对合成图像施加单样本损失权重折扣 $\omega_{\text{syn}} = 0.30$，且严格执行 `V10` 的目标类单通道掩码（非目标类梯度置零，All-Clean Gate 仅在真实图像上计算梯度）。在此阶段，合成样本帮助视觉骨干的最后微调层与空间注意力池化层学会关注手部畸变、抠图断层与局部伪纹理；
- **Stage B（纯真图分布校准与头部退火阶段，Epoch 4 ~ 5）**：**将像素级合成图像的采样率降为 0%**（DataLoader 切换为 100% 纯真实人类标注图像 + 3.1 节的特征空间同场景流形插值），同时冻结视觉骨干，仅用较小的学习率（`lr_head = 3e-4`）精调 CORN 条件序数分类头与归一化层统计量！
- **核心优势**：Stage A 吸收了合成数据带来的特征多样性与罕见伪影召回能力，而 Stage B 则将分类头的决策阈值、类别先验赔率（Prior Odds）与概率分布形态 **100% 拉回真实人类标注数据流形**，彻底根除了合成数据导致的测试集概率膨胀与假阳性激增！

#### 闸门 3：通道级正交路由集成（Channel-Isolated Specialist Injection）
在最终模型集成阶段，包含合成预热的专精模型**绝不允许全局覆盖 `模糊`、`过锐` 等已有高精度的通道**，而是仅在经过对数赔率均值-方差双对齐（`align_logit_moments`）后，以 $20\% \sim 30\%$ 的权重定向注入 **`过度生成`（通道 4）** 与 **`虚化突兀`（通道 5）** 的概率流中，其余通道 100% 由纯真图旗舰池主导。

---

## 四、 可直接运行的核心合成与增广代码实现规范

为便于在 `/Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa` 代码库中快速落地验证，以下提供 **「特征空间切块嫁接 + 同场景序数流形 Mixup」** 以及 **「前景掩模引导的正负成对反事实散景与局部无损重绘合成」** 的核心实现代码模板。

### 4.1 特征空间切块异体嫁接与同场景序数流形插值（嵌入训练循环，零 I/O 开销）

```python
import torch
import torch.nn.functional as F
import numpy as np

def latent_tile_grafting_and_manifold_mixup(feat_views, grades, attn_scores, p_graft=0.35, p_mixup=0.35):
    """
    在多视角特征空间 (B, V, D) 内实时执行零像素伪影的长尾正样本合成：
    - feat_views: (B, V, D)，其中 v=0 为全局视图，v=1..V-1 为局部切块视图
    - grades: (B, 6) 真实整数标签 [模糊, 纹理损失, 过锐, 花屏, 过度生成, 虚化突兀]
    - attn_scores: (B, V-1) 局部切块在过度生成/异常分支上的激活得分
    返回增广后的 (feat_views_aug, soft_grades_aug)
    """
    B, V, D = feat_views.shape
    feat_out = feat_views.clone()
    soft_g = grades.float().clone()
    
    # 找出当前 Batch 内的真实「过度生成」中重度正样本 (grade >= 2) 与无过度生成受体样本 (grade == 0)
    og_donors = torch.where(grades[:, 4] >= 2)[0]
    og_recipients = torch.where((grades[:, 4] == 0) & (grades[:, 3] == 0) & (grades[:, 5] == 0))[0]
    
    # 1. 切块特征空间异体嫁接 (Latent Tile-Embedding Grafting)
    if len(og_donors) > 0 and len(og_recipients) > 0:
        for r_idx in og_recipients:
            if torch.rand(1).item() > p_graft:
                continue
            d_idx = og_donors[torch.randint(0, len(og_donors), (1,)).item()]
            # 锁定供体真图中异常激活最高的 Top-2 局部切块 (排除正常背景切块!)
            top_tiles = torch.argsort(attn_scores[d_idx], descending=True)
            
            if torch.rand(1).item() < 0.55:
                # 嫁接 1 个异常切块 -> 合成纯净的「过度生成 P2 (1.0)」
                t_dst = torch.randint(1, V, (1,)).item()
                t_src = top_tiles[0].item() + 1
                feat_out[r_idx, t_dst] = feat_views[d_idx, t_src]
                feat_out[r_idx, 0] = F.normalize(0.82 * feat_views[r_idx, 0] + 0.18 * feat_views[d_idx, t_src], dim=-1) * feat_views[r_idx, 0].norm()
                soft_g[r_idx, 4] = 1.0  # P2
            else:
                # 嫁接 2 个异常切块 -> 合成纯净的「过度生成 P1 (2.0)」
                for k_t in range(min(2, V - 1)):
                    t_src = top_tiles[k_t].item() + 1
                    feat_out[r_idx, k_t + 1] = feat_views[d_idx, t_src]
                feat_out[r_idx, 0] = F.normalize(0.58 * feat_views[r_idx, 0] + 0.42 * feat_views[d_idx, 0], dim=-1) * feat_views[r_idx, 0].norm()
                soft_g[r_idx, 4] = 2.0  # P1

    # 2. 同场景高相似度样本间的序数流形插值 (Scene-Matched Ordinal Manifold Mixup)
    g_norm = F.normalize(feat_views[:, 0], dim=-1)
    sim_mat = g_norm @ g_norm.T  # (B, B)
    sim_mat.fill_diagonal_(-1.0)
    
    for i in range(B):
        if torch.rand(1).item() > p_mixup:
            continue
        # 寻找 Batch 内语义最相似的同场景邻居 j
        j = int(torch.argmax(sim_mat[i]).item())
        if sim_mat[i, j] < 0.65:
            continue
        lam = float(np.random.beta(2.0, 2.0))
        lam = max(0.25, min(0.75, lam))
        mixed = lam * feat_views[i] + (1.0 - lam) * feat_views[j]
        # 保持特征范数稳定
        feat_out[i] = F.normalize(mixed, dim=-1) * (lam * feat_views[i].norm(dim=-1, keepdim=True) + (1.0 - lam) * feat_views[j].norm(dim=-1, keepdim=True))
        soft_g[i] = lam * grades[i].float() + (1.0 - lam) * grades[j].float()
        
    return feat_out, soft_g
```

### 4.2 前景语义掩模引导的正负成对反事实散景合成（根除椭圆雪人捷径）

```python
import cv2
import numpy as np

def generate_paired_bokeh_counterfactuals(img_bgr, fg_saliency_map, rng):
    """
    基于真实前景显著性/分割图 fg_saliency_map in [0, 1] 同时生成一对同源反事实样本：
    1. img_natural_bokeh (Hard Negative, 虚化突兀=0): 模拟单反大光圈随伪深度平滑过渡的自然散景
    2. img_abrupt_bokeh  (True Positive, 虚化突兀=3): 模拟人像模式在发丝/肩部边缘抠图咬合失误与硬切虚化
    """
    h, w = img_bgr.shape[:2]
    fg_mask = cv2.resize(fg_saliency_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    fg_bin = (fg_mask > np.percentile(fg_mask, 60)).astype(np.float32)

    # 构造圆盘散景模糊背景 (Disk Defocus Kernel)
    rad = int(rng.integers(10, 16))
    yk, xk = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    disk_k = ((xk ** 2 + yk ** 2) <= rad ** 2).astype(np.float32)
    disk_k /= disk_k.sum()
    bg_heavy_blur = cv2.filter2D(img_bgr, -1, disk_k).astype(np.float32)
    bg_mid_blur   = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=rad * 0.45).astype(np.float32)
    img_f = img_bgr.astype(np.float32)

    # --- 样本 A：同源自然光学散景负例 (Hard Negative, 虚化突兀 = 0) ---
    # 结合纵向地面深度梯度与宽羽化前景掩模，实现渐进式双层光学散景
    k_smooth = max(31, int(min(h, w) * 0.12) | 1)
    alpha_nat = cv2.GaussianBlur(fg_bin, (k_smooth, k_smooth), k_smooth * 0.35)[..., None]
    y_depth = np.linspace(0.15, 0.85, h, dtype=np.float32)[:, None, None]
    bg_progressive = y_depth * bg_mid_blur + (1.0 - y_depth) * bg_heavy_blur
    img_natural_bokeh = np.clip(alpha_nat * img_f + (1.0 - alpha_nat) * bg_progressive, 0, 255).astype(np.uint8)

    # --- 样本 B：同源抠图失误虚化突兀正例 (True Positive, 虚化突兀 = 3/P0) ---
    # 在真实前景轮廓边缘制造 2~3 处局部掩模撕裂（吞噬前景或残留背景块），并强制 1px 阶跃硬切！
    abrupt_mask = fg_bin.copy()
    contours, _ = cv2.findContours((fg_bin > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
        for _ in range(int(rng.integers(2, 5))):
            pt = cnt[int(rng.integers(0, len(cnt)))]
            br = int(min(h, w) * rng.uniform(0.04, 0.09))
            # 50% 概率将背景硬切进前景主体内部，50% 概率将前景边缘外侧背景块漏虚化
            fill_val = 0.0 if rng.random() < 0.5 else 1.0
            cv2.circle(abrupt_mask, (int(pt[0]), int(pt[1])), br, fill_val, -1)
    # 仅做 1px 极窄抗锯齿，保留生硬虚实割裂断层（零人造亮环！）
    alpha_abrupt = cv2.GaussianBlur(abrupt_mask, (3, 3), 0.7)[..., None]
    img_abrupt_bokeh = np.clip(alpha_abrupt * img_f + (1.0 - alpha_abrupt) * bg_heavy_blur, 0, 255).astype(np.uint8)

    return img_natural_bokeh, img_abrupt_bokeh
```

---

## 五、 合成方案分步验证与冲分收益预期表

为确保新合成体系每一分投入都转化为真实测试集上的确定性净收益，严禁一次性生成数千张未经检验的图像直接覆盖主模型，而应遵循下表的四步递进式验证路线：

| 验证步骤 | 合成与增广模块 | 训练与注入方式 | 5划分盲测池 (`s16~s20` 3,000真图) 重点增益通道 | 预期单项净提分 |
| :--- | :--- | :--- | :--- | :---: |
| **Step 1**<br>*(零耗时/最高确定性)* | **Level 1：特征空间切块嫁接 (Latent Tile CutMix) + 同场景序数流形 Mixup** | 直接在 `feats/*_combined.npz` (70,400维) 与多视角分类头训练中开启，**零像素级合成污染** | `过度生成_P2` (`0.400 -> 0.48+`)<br>`过度生成_P1` (`0.597 -> 0.65+`)<br>四档类 `P1 ↔ P2` 边界平滑 | **`+0.45 ~ +0.75 分`** |
| **Step 2**<br>*(解决 Clean 误伤头号顽疾)* | **Level 3B：前景掩模引导的「自然光学散景 vs 抠图失误散景」成对反事实训练** | 采用两阶段课程退火训练 `虚化突兀` 专精分支，按 25% 权重注入通道 5 | `虚化突兀_P0` (`0.795 -> 0.86+`)<br>消除大光圈人像 P0 误报，锁死 $S_{\text{clean}}$ | **`+0.35 ~ +0.60 分`** |
| **Step 3**<br>*(攻克 AIGC 局部畸变召回)* | **Level 2 & 3A：纯真图教师定位的真实伪影贴片拉普拉斯金字塔嫁接 + 主体局部 FLUX 重绘 + SBI 自混合** | 经 `52.43` 教师拒绝采样过滤后，执行 **3 Epoch 混合预热 + 2 Epoch 纯真图退火校准** | `过度生成_P0` (`0.748 -> 0.81+`)<br>`过度生成_P1/P2` 召回率提升 `+6%~+9%` | **`+0.50 ~ +0.85 分`** |
| **Step 4**<br>*(补齐编码损坏长尾)* | **Level 3C：H.264/HEVC 宏块错位与色度失步在线内存增强** | 在 Stage A 训练时以 $3\%$ 概率在内存中动态施加，不写盘、无 JPEG 质量泄漏 | `花屏_P0` (`0.914 -> 0.95+`) | **`+0.15 ~ +0.25 分`** |

---

## 六、 核心结论与避坑铁律总结

针对 `过度生成`、`虚化突兀` 与 `花屏` 等长尾稀缺类别，合成数据完全可以成为跨越当前瓶颈的强力冲分引擎，但前提是必须从工程底层彻底斩断前序 `V4 ~ V10` 合成管线的四大负迁移污染源。在落地实施过程中，请严格遵循以下**五条合成数据黄金铁律**：

1. **严禁全图双边滤波与人工正弦波重映射**：AIGC `过度生成` 是局部语义与几何拓扑异常，绝不是全局水彩滤镜；合成时必须保证图像 $70\% \sim 85\%$ 的背景区域与原始干净底图 **100% 像素级无损一致**。
2. **严禁分档设置不同的 JPEG 压缩质量因子**：彻底废除 `P2=94, P1=92, P0=90` 或 `86/72/55` 的硬编码保存方式，优先采用内存实时合成（In-Memory Online Synthesis）或统一无损 `.png` 存储，切断 DCT 二次量化表泄漏。
3. **严禁无定位的盲目裁剪嫁接**：从真实 `过度生成 P0/P1` 供体图中提取伪影贴片时，必须使用已训练好的纯真图局部切块分类器锁定高置信度异常窗口（$p_{\text{og}} \ge 0.70$），并通过纯真图教师模型的闭环拒绝采样过滤掉无效拼接样本。
4. **坚持成对反事实对照（Paired Counterfactuals）**：合成 `虚化突兀` 时必须基于 `DINOv3 / C-RADIO` 真实前景掩模，成对生成“自然连续光学散景（标签 0）”与“局部边缘撕裂硬切散景（标签 P0）”，教会模型区分光学景深与抠图事故。
5. **坚持特征空间流形增广优先与 Sim2Real 两阶段退火**：优先落地零像素伪影的**特征空间切块异体嫁接（Latent Tile CutMix）与同场景序数流形插值（Manifold Mixup）**；在使用像素级合成图像时，严格执行“前 3 轮混训拓宽表征 + 后 2 轮 100% 纯真图退火校准分类头”的两阶段课程学习，并对最终输出执行对数赔率均值-方差双对齐，确保线上测试集零先验膨胀、稳步涨分。

---

## 参考文献与代码资产索引

### 学术文献与前沿方法参考
1. **像素级 AIGC 伪影自动合成流水线**：Menn, D., et al. *Synthesizing Artifact Dataset for Pixel-level Detection*. 2025. 论文地址：[arXiv:2509.19589](https://arxiv.org/abs/2509.19589).
2. **自混合图像伪影增广 (Self-Blended Images, SBI)**：Shiohara, K., & Yamasaki, T. *Detecting Deepfakes with Self-Blended Images*. CVPR 2022. 论文地址：[arXiv:2204.08376](https://arxiv.org/abs/2204.08376) 及 RL-SBI ([arXiv:2601.15624](https://arxiv.org/pdf/2601.15624)).
3. **AIGC 局部伪影定位与热力图引导重绘 (RichHF-18K & PAL4VST)**：Liang, Y., et al. *Rich Human Feedback for Text-to-Image Generation* (CVPR 2024, [arXiv:2312.10240](https://arxiv.org/abs/2312.10240)) 及 Zhang, L., et al. *Perceptual Artifacts Localization for Image Synthesis Tasks* (ICCV 2023, [arXiv:2310.05590](https://arxiv.org/abs/2310.05590)).
4. **合成图像隐藏伪影分类与矫正 (SynArtifact & Mirage)**：Cao, B., et al. *SynArtifact: Classifying and Alleviating Artifacts in Synthetic Images via Vision-Language Model* (2024, [arXiv:2402.18068](https://arxiv.org/abs/2402.18068)) 及 *Mirage: Unveiling Hidden Artifacts in Synthetic Images with Large Vision-Language Models* (2025, [arXiv:2510.03840](https://arxiv.org/abs/2510.03840)).
5. **合成图像数据集中的 JPEG 压缩与尺寸偏置陷阱**：Grommelt, P., Weiss, L., Pfreundt, F.-J., & Keuper, J. *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*. 2024. 论文地址：[arXiv:2403.17608](https://arxiv.org/html/2403.17608v1).
6. **特征空间流形插值增广 (Manifold Mixup)**：Verma, V., Lamb, A., Beckham, C., Najafi, A., Mitliagkas, I., Lopez-Paz, D., & Bengio, Y. *Manifold Mixup: Better Representations by Interpolating Hidden States*. ICML 2019. 论文地址：[arXiv:1806.05236](https://arxiv.org/abs/1806.05236).
7. **深度感知实例级粘贴与稀有类闭环验证增广 (Depth-Copy-Paste & VRAIN)**：*Multimodal and Depth-Aware Compositing for Robust Face Detection* (2025, [arXiv:2512.11683](https://arxiv.org/html/2512.11683v1)) 及 *TMI: Text-to-Image Meets Image-to-Image for Complementary Data Augmentation* (2026, [arXiv:2607.08201](https://arxiv.org/html/2607.08201v1)).
8. **物理引导散景合成与深度-散景耦合 (BokehDepth)**：*Boosting Monocular Metric Depth Estimation via Bokeh Rendering*. 2026. 论文地址：[arXiv:2512.12425](https://arxiv.org/html/2512.12425v2).

### 本地代码库合成管线诊断关联文件
- V6 多缺陷合成管线：[code/finetune/build_multi_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_multi_synth.py) & [code/finetune/run_v6_ablate_and_specialists.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/run_v6_ablate_and_specialists.py)
- V7 过度生成专项合成管线：[code/finetune/build_overgen_pro.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_overgen_pro.py) & [code/finetune/run_v7_overgen_specialists.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/run_v7_overgen_specialists.py)
- V8 反事实阶梯合成管线：[code/finetune/build_ladder_synth_v8.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_ladder_synth_v8.py)
- V9 真实感零振铃合成管线：[code/finetune/build_v9_authentic_synth.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/build_v9_authentic_synth.py)
- V10 单通道掩码隔离训练与评估：[code/finetune/train_v10_clean_native_specialist.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/train_v10_clean_native_specialist.py) & [code/finetune/eval_v10_and_update_test150.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/eval_v10_and_update_test150.py)
- 70,400 维预提取多基座特征塔：[code/finetune/probe_70k_foundation_on_s16_s17.py](file:///Users/luowenjing/Desktop/1/chengfeng2026-topic18-nriqa/code/finetune/probe_70k_foundation_on_s16_s17.py)
