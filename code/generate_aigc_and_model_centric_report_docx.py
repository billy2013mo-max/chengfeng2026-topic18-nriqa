#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Word Document (.docx) and Markdown (.md) summarizing:
1. Current Status & Post-Mortem of the Latest Negative Optimization (Why test_150 post-processing & frozen-feature head tuning hit a dead end)
2. Strategic Pivot: Shifting 100% to Model-Centric Improvement & AIGC Synthetic Data Expansion (Zero test_150 targeting)
3. Complete AIGC, VLM, Vision Backbone, and Compute Environment Inventory on Our Machine
4. Post-Mortem of Previous Synthetic Data Attempts & Concrete Exploration Directions for Local Gemini
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_task_and_dilemma_report_docx import write_docx, write_markdown

REPORT_MD_PATH = "/wanqing-develop/luowenjing/18/NR_IQA_Status_Update_and_AIGC_Environment_Report.md"
REPORT_DOCX_PATH = "/wanqing-develop/luowenjing/18/NR_IQA_Status_Update_and_AIGC_Environment_Report.docx"


def build_report_blocks():
    blocks = []

    def h1(text):
        blocks.append(("h1", text))

    def h2(text):
        blocks.append(("h2", text))

    def h3(text):
        blocks.append(("h3", text))

    def p(text):
        blocks.append(("p", text))

    def callout(text):
        blocks.append(("callout", text))

    def ul(items):
        blocks.append(("ul", items))

    def table(headers, rows):
        blocks.append(("table", (headers, rows)))

    blocks.append((
        "title",
        "NR-IQA 竞赛现状复盘、AIGC 合成环境清单与“回归模型本体+合成数据扩充”破局请教文档"
    ))
    blocks.append((
        "subtitle",
        "核心结论：围绕现有 3,573 张真实图像的冻结特征训练侧挖掘与针对 test_150 的解码/后处理修补已彻底触顶，继续在决策边界上挪动个别单元格只会引发线上负优化。下一阶段我们将彻底放弃针对 test_150 的定制后处理，转向“基于本机 AIGC 与多模态大模型环境的高保真合成数据扩充 + 模型本体表征能力升级”。本文档详述最新复盘、我方完整 AIGC 软硬件与模型环境，以及准备投入的探索方向。"
    ))

    # =========================================================================
    # 第一部分：最新现状与刚才一轮尝试的复盘
    # =========================================================================
    h1("一、 最新现状说明与刚才一轮尝试的复盘（为什么依然是负优化？）")

    h2("1.1 刚才一轮尝试所做的完整工作")
    p("在刚刚完成的一轮实验中，我们严格按照破局方案执行了以下两阶段工作：")
    ul([
        "1. 彻底剔除旧版离线合成特征流（V14/V15 权重归零），仅使用 3,573 张真实标注图像。",
        "2. 提取了 264D 前背景解耦物理统计特征 + 原生无缩放 1:1 局部块特征，并训练了 8 折四流同图簇 Siamese 单调 CORN 模型（V17_PureReal_FourStream，结合软化 F1 损失、同场景排序损失、OOF 分歧边界软标签与直推伪标签）。",
        "3. 在集成与解码侧，下线了旧版 PhaseA 解码器的 KAPPA_P1P2 温度缩放，改用 8 折 Valid-Mask 有效掩码动态重归一化 + Step 23/24 跨等级概率质量归并 + 65 分竞赛期望增益坐标上升解码器。",
        "4. 最终生成的提交文件将再上一版（19 处修改，负优化）中的 11 处 P1->P2 降级、3 处 P2->无问题 抹杀、2 处高危误报全部撤回，仅保留了相对 52.89 基线的 9 处改动（4 处 P2->P1 升级、4 处 无问题->P2 唤醒、1 处 P1->P0 升级）。"
    ])

    h2("1.2 为什么即使只改了 9 个高置信单元格，线上依然是负优化？")
    callout(
        "核心诊断：在仅有 3,573 张真实图像、且冻结视觉 Backbone 只训练顶层分类头（MLP / ABMIL）的前提下，模型本体的真实特征判别力已经彻底饱和！此时在固定特征上改损失函数、改集成权重、改解码器，本质上只是在同一个模糊不清的 P1/P2 和 P2/无问题 决策边界上“平移阈值”。"
    )
    p("具体而言，连续两轮负优化暴露了三个根本性事实：")
    ul([
        "第一，【围绕 test_150 的后处理与边界搬移已经走到死胡同】：52.89 分基线（Phase 1 Breakthrough：50% 多主干锚点 + 50% 纯真实 InternViT-6B）已经把现有 3,573 张真实数据在冻结特征下的后验概率红利吃尽了。刚才的 9 处改动全部位于模型概率处于 0.45 ~ 0.55 之间的“骑墙边界样本”（例如 P=[0.024, 0.432, 0.543, 0.0] 或 P=[0.426, 0.544, 0.029, 0.0]）。由于模型本体对这些边界样本并没有真正的区分把握，强行通过 Step 23/24 或期望 F1 坐标上升把它们翻过去，在线上 N=150 的小样本宏平均 F1 下极易因为 FP 增量大于 TP 增量而导致掉分。",
        "第二，【在已有 3,573 张真实数据上，纯训练侧（不加新数据、不训视觉主干）能做的手段已经全部穷尽】：我们已经尝试过 8 种视觉主干、264D 物理卡尺、原图 1:1 切块 ABMIL、CORN 序数回归、同图簇原型残差、软标签清洗、4 项联合损失函数。但在冻结预训练特征上叠加再复杂的分类头，也无法凭空创造新的视觉判别信息。",
        "第三，【真实训练集的正样本绝对数量太少，尤其是中间等级 P1/P2 与长尾类别】：在 3,573 张真实图中，诸如『过度生成 P1』（全量仅约 200 张）、『过锐 P1』、『虚化突兀 P0』（仅约 90 张）、『花屏 P0』（仅约 90 张）的正样本极度匮乏。没有高质量合成数据去填充特征空间中的稀疏区域，模型面对真实短视频中千变万化的 AI 伪影、ISP 锐化光晕和抠图虚化边界时，泛化能力天然受限。"
    ])

    h2("1.3 下一阶段的明确战略转向")
    ul([
        "1. 【彻底停止针对 test_150 的定向优化与后处理魔改】：不再盯着 test_150 的哪几个单元格去调阈值、不再使用 test_150 直推伪标签、不再搞复杂的后处理概率搬移（如 Step 23/24、PhaseA KAPPA 等），推理端保持最干净、最标准的阈值解码。",
        "2. 【完全从模型本体出发，通过高质量 AIGC 合成数据扩充训练集】：利用我们机器上完备的 AIGC 生成/重绘/超分环境与多模态大模型（VLM）环境，围绕 6 大画质缺陷的真实物理/生成退化机理，合成大规模、高保真、经过 VLM 裁判严格质检的训练样本加入训练集。",
        "3. 【让合成数据真正转化为模型本体的能力】：不仅在分类头层面训练，还可以利用本机 8×H200 的充足算力，在“真实数据 + 严选 AIGC 合成数据”上进行视觉主干（如 InternViT-6B / Qwen3-VL-8B / DINOv3-L / SigLIP2）的端到端 LoRA 微调或两阶段 Sim2Real 课程微调，从根源上提升模型对细粒度画质缺陷的表征能力。"
    ])

    # =========================================================================
    # 第二部分：我方完整算力、AIGC 模型与软件环境清单
    # =========================================================================
    h1("二、 我方机器的算力硬件与 AIGC / VLM / 视觉模型环境详单")
    p("为了让本地 Gemini 能够设计出完全适配我方环境、可直接落地执行的 AIGC 数据合成与模型训练方案，以下完整列出我方服务器当前的硬件规格、虚拟环境、本地已下载的 AIGC 生成模型、多模态评判大模型及视觉主干权重。")

    h2("2.1 硬件算力与存储规格")
    table(
        ["硬件维度", "具体配置规格", "对 AIGC 合成与模型训练的支撑能力"],
        [
            ["GPU 算力", "8 × NVIDIA H200 (单卡 140 GB HBM3e，总显存 1,120 GB)", "可 8 卡并行全速运行 FLUX.3 扩散重绘、30B 级多模态 VLM 裁判过滤，以及 6B~8B 视觉大模型全量/LoRA 微调"],
            ["CPU 与内存", "192 核 Intel Xeon Platinum 8558 + 2.0 TB 系统内存", "支持百进程并发进行原生分辨率图像信号处理（ISP 退化、光学散焦模拟、频域滤波）与内存级特征缓存"],
            ["存储空间", "共享盘 /wanqing-develop (剩余 1.2 TB) + 根分区 /tmp (剩余 2.0 TB)", "充足容纳数万张高分辨率 AIGC 合成图像及其多尺度特征缓存"],
            ["网络环境", "已配置高速 HTTP/HTTPS 代理，HuggingFace、HF-Mirror、PyPI 全部直连互通 (Status 200)", "除本地已有的 18 个大模型外，如需额外开源模型（如 ControlNet、SAM2、Depth-Anything、Real-ESRGAN 等）或 Python 库均可随时秒级拉取安装"]
        ]
    )

    h2("2.2 本地已就绪的 AIGC 图像生成 / 编辑 / 超分模型清单")
    p("我方在 /wanqing-develop/luowenjing/18/models/ 及 HuggingFace 缓存中已部署好以下 AIGC 生成与图像重构模型，并配有专属虚拟环境 venv_flux（Python 3.11 + PyTorch 2.13.0+cu130 + diffusers 0.42.1 + transformers 5.6.0 + peft 0.18.1）：")
    table(
        ["模型名称", "本地路径 / 标识", "参数规模与架构", "核心能力与可合成的目标缺陷类型"],
        [
            [
                "FLUX.3-Image-Klein (官方无门禁完整版)",
                "models/flux3-klein (含 bf16 主权重及 fp8r / nvfp4 变体)",
                "5.5B 参数 DiT (Flux3DistilledModularPipeline) + Qwen3-VL 文本编码器 + VAE",
                "支持 Image-to-Image（图生图条件重绘）与局部指令编辑。非常适合在保持原图构图、主体、色彩不变的前提下，通过控制推理步数（如 8~16 步）、噪声强度和提示词，真实制造『过度生成 (P2/P1/P0)』（AI 皮肤塑料化、手指/五官/文字笔画畸变、伪造高频纹理）以及『纹理损失 (P2/P1/P0)』（AI 涂抹感）"
            ],
            [
                "FLUX.3-Image-Klein Prompt Upsampler (PUP)",
                "models/flux3-pup",
                "9B 参数多模态模型 (基于 Qwen3.5-9B)",
                "官方配套图生图指令扩写器。支持输入原图 + 简短退化指令 + 可选局部 Bounding Box 坐标，自动生成精准控制局部画质退化而保持背景不变的长 Prompt"
            ],
            [
                "Swin2SR-Classical-SR-x2-64",
                "caidas/swin2SR-classical-sr-x2-64 (HF 本地缓存)",
                "Swin Transformer 2x 图像超分重构网络",
                "通过『原图 -> 2x AI 超分 -> Lanczos 缩回原尺寸（单次或两次迭代）』或『先下采样再 AI 超分放大』，真实复现短视频经过 AI 画质修复/超分滤镜后特有的『轻度/中度过度生成 (P2/P1)』与『伪锐化/涂抹纹理损失』"
            ],
            [
                "DiffSynth-Studio 框架",
                "/wanqing-develop/caijiehang/diffsynth-studio",
                "完整开源扩散合成与编辑工具箱",
                "支持灵活定制扩散采样步、Latent 空间混合（Latent Blending）、局部重绘（Inpainting）及多模型管线串联"
            ]
        ]
    )

    h2("2.3 本地已就绪的多模态大模型（VLM 裁判 / 合成数据质检 / 端到端微调）")
    p("合成数据最大的风险是“合成出来的图不像真实缺陷，或者标签给错了（例如把 P2 标成 P0）”。我们本地拥有极其强大的多模态视觉语言大模型矩阵（配合 venv_swift 中的 ms-swift 4.6.0 与 vllm 0.29.0），不仅可以用于微调，更可以作为 **AIGC 合成数据的自动质检裁判（VLM Judge）**：")
    table(
        ["多模态模型名称", "本地路径", "架构与规模", "在 AIGC 合成管线与模型训练中的作用"],
        [
            ["Qwen3-VL-8B-Instruct", "models/qwen3vl-8b", "8B 参数最新一代视觉语言模型", "既可作为主力 VLM 基座进行端到端 LoRA 微调，也可与原图成对输入（Pairwise Comparison）评估合成图的真实感与退化等级（P2/P1/P0）"],
            ["Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-2B-Instruct (HF 缓存)", "2B 参数轻量高速 VLM", "适合高速批量初筛合成样本、生成图像局部主体/背景/文字区域描述"],
            ["Clef-27B & Clef-Flash-9B", "models/clef & models/clef-flash", "27B (基于 Qwen3.8-27B) / 9B (基于 Qwen3.5-9B) 多模态决策模型", "专为多模态结构化分类与校准概率输出设计的决策大模型，非常适合作为高精度的合成样本质量裁判（过滤伪影失真或打标不准的合成图）"],
            ["JEV-27B-VL & JEV-Gemma4-26B-A4B", "models/JEV-27B-VL & models/JEV-Gemma4-26B-A4B", "27B / 26B MoE 多模态校准打分模型", "擅长零样本/少样本图像成对偏好打分与细粒度严重度排序（可用于给合成的退化阶梯 P2 < P1 < P0 进行打分校验）"],
            ["Nemotron-3-Nano-Omni-30B-A3B", "models/nemotron-omni", "31B 总参数 (Mamba2-Transformer MoE) 全模态推理模型", "具备强大的视觉异常推理能力，可用于复杂缺陷（如过度生成、虚化突兀）的合成质量把关"]
        ]
    )

    h2("2.4 本地已就绪的视觉表征骨干模型（Vision Backbones）")
    p("在视觉编码器侧，我们本地已部署好 11 个主流视觉主干（均在 models/ 或 HF 缓存中）：")
    ul([
        "1. 超大参数量主力视觉主干：InternViT-6B-448px-V2.5 (6B 参数，当前单模型最强基座)、C-RADIOv4-H (NVIDIA 多教师蒸馏高分辨率主干)、PE-Core-Gigantic-patch14-448 (Meta 感知编码器 Gigantic 版)、SigLIP2-Giant-opt-patch16-384。",
        "2. 原生/多尺度与自监督几何主干：DINOv3-ViT-L/16 (timm/vit_large_patch16_dinov3.lvd1689m)、PE-Core-Large-patch14-336、SigLIP2-SO400M-patch16-NaFlex（原生任意长宽比不拉伸主干）、SigLIP2-SO400M-patch14-384、EUPE-ViT-B、ConvNeXtV2-Base、EVA02-Base-448。"
    ])

    # =========================================================================
    # 第三部分：前期合成数据尝试的复盘（我们已经做过什么、为什么之前没奏效）
    # =========================================================================
    h1("三、 前期合成数据尝试的深度复盘（为什么之前的合成数据没带来线上正收益？）")
    p("实际上，我们在前期迭代（V6、V7 OverGen-Pro、V8/V9 Ladders、V14 流形合成、V15 像素合成）中已经做过几轮合成数据尝试，并在磁盘上留下了部分合成脚本与图片（如 code/synth/flux3_gen.py、code/synth/pilot_gen.py、code/finetune/build_v9_authentic_synth.py）。但回顾为什么之前的合成尝试没有在线上产生正向突破，我们总结出以下 4 个致命缺陷：")

    table(
        ["前期合成尝试的问题维度", "具体做法复盘", "导致的致命后果（为什么之前没奏效）"],
        [
            [
                "问题 1：手工 CV 算子留下了明显的『人工捷径（Shortcut）』",
                "在 V6/V7 中，为了模拟『过度生成 P0』的扭曲感，使用了 cv2.remap 正弦波纹扭曲；为了做局部融合，使用了简单的随机椭圆高斯掩码（soft_feathered_mask）和固定规则的仿射克隆贴块。",
                "深度视觉骨干（如 InternViT / DINOv3）极其敏锐地记住了『正弦波纹』和『椭圆羽化接缝』这些非自然的合成痕迹。模型在含合成数据的验证集上 F1 虚高，但一到真实图上（没有正弦波纹和椭圆接缝）反而把真实缺陷漏掉，或者对特定纹理产生幻觉误报。"
            ],
            [
                "问题 2：FLUX.3 / Swin2SR 生成规模太小，且缺乏语义分割掩码与 VLM 质检闭环",
                "前期仅对约 80~180 张无问题图跑了 FLUX.3-Klein（/tmp/synth/flux3 下仅 180 张图）和 100 张 Swin2SR，然后靠传统 OpenCV 反复拼贴扩充出上千张图；且生成后没有用 VLM 裁判检查『到底有没有产生预期等级的缺陷』。",
                "一方面 180 张基底图反复拼贴导致源图同质化严重；另一方面 FLUX.3 在不同原图上的重绘幅度差异极大（有的图重绘后几乎没变却被硬标为 P1/P0，有的图已经面目全非却被标为 P2），未经质检直接塞进训练集引入了大量标签噪声。"
            ],
            [
                "问题 3：只拿合成数据训练顶层 MLP 分类头，视觉主干（Backbone）完全冻结",
                "前期所有加入合成数据的实验（V6~V9, V14, V15），都是先把合成图过一遍冻结的预训练 Backbone 抽成固定特征向量，再和真实图特征混在一起训练顶层线性/MLP 分类头。",
                "冻结的预训练 Backbone 会把『合成图』和『真实短视频截图』投影到特征空间的两个不同簇（Domain Gap）。顶层 MLP 分类头为了同时拟合这两个簇，被迫扭曲了真实数据流形上的 P1/P2 决策边界！"
            ],
            [
                "问题 4：标签共现规则（Co-occurrence）靠硬编码指定，缺乏真实多维联合校准",
                "在脚本里硬编码『只要合成过度生成 P1，就强行把纹理损失标为 2、模糊标为 1』。",
                "硬编码的辅助标签往往与合成图像的实际视觉观感不符，污染了『模糊』、『纹理损失』和『过锐』通道的梯度。"
            ]
        ]
    )

    # =========================================================================
    # 第四部分：下一阶段围绕“模型本体 + 高质量 AIGC 合成训练集”的探索方向与请教清单
    # =========================================================================
    h1("四、 下一阶段准备投入的探索方向（请本地 Gemini 指导具体实施蓝图）")
    p("痛定思痛，下一阶段我们准备**完全抛弃针对 test_150 的后处理修补**，充分发挥 8×H200 的算力与本地 AIGC + VLM 模型库优势，从**“高保真 AIGC 合成数据构建”**与**“模型本体表征及训练范式升级”**两个核心层面入手。我们拟定了以下 5 个具体方向，恳请本地 Gemini 给出详尽的指导方案：")

    h2("方向 1：基于 FLUX.3-Klein + Swin2SR + 真实物理退化的“零人工捷径”6 类缺陷合成配方")
    p("针对竞赛的 6 个维度，我们希望彻底摒弃简单的椭圆贴块和正弦波扭曲，建立贴合真实工业退化机理的合成管线：")
    ul([
        "1. 『过度生成 (P2 / P1 / P0)』：如何利用本地 8 卡 H200 并行跑 FLUX.3-Klein + FLUX.3-PUP 及 Swin2SR，在大规模真实无问题图（甚至真实轻度模糊/低清图）上，通过精细控制 Denoising Strength / 步数 / 超分回缩次数，直接生成全图或基于语义主体掩码（而非随机椭圆）的 P2（轻微塑料感/微细节伪造）、P1（明显五官/手指/文字笔画 AI 畸变或重度油画感）、P0（严重结构崩坏）？",
        "2. 『过锐 (P2 / P1 / P0)』与『纹理损失 (P2 / P1 / P0)』：真实短视频的过锐与纹理损失几乎全部来自『强降噪涂抹 + 编码压缩 + ISP 多尺度 USM 锐化白边』。如何设计参数连续可控、无人工痕迹的 ISP 退化算子链（或者结合 Swin2SR/FLUX 轻度重绘 + ISP 锐化）来精准合成 P2 与 P1 阶梯？",
        "3. 『虚化突兀 (P0)』与『模糊 (P2 / P1 / P0)』：如何避免椭圆抠图假痕迹，利用前景显著性/深度分割掩码制造真正符合手机人像模式抠图翻车（发丝/手指边缘阶跃断裂 + 散景光斑）的『虚化突兀 P0』，以及前景主体失焦/运动模糊的『模糊 P2/P1/P0』？",
        "4. 『花屏 (P0)』：如何模拟真实的视频码流解码宏块错位、局部色彩通道溢出、绿屏/紫边条纹断裂（而非简单的随机彩色方块）？"
    ])

    h2("方向 2：引入本地多模态大模型（Qwen3-VL-8B / Clef-27B / JEV-27B）构建“AIGC 合成质量门控与重打标（VLM Judge）”")
    p("为了彻底解决合成数据“标签与实际视觉观感不符”以及“分布偏移”的问题，我们希望利用本地现成的 8B~30B 视觉大模型建立一套**合成数据准入与软标签校准机制**：")
    ul([
        "1. 合成图生成后，如何用真实训练集中的典型 P2 / P1 / P0 锚点图作为参考（In-Context Few-Shot 或特征余弦相似度 + VLM Pairwise 对比），自动剔除“合成失败/产生非自然伪影”的废片？",
        "2. 如何对通过质检的合成图赋予多维度联合软标签（Soft Labels），而不是在脚本里生硬地写死 `纹理损失=2`？"
    ])

    h2("方向 3：让视觉主干（Backbone）真正消化合成数据——端到端 LoRA 微调 vs Sim2Real 两阶段训练")
    p("此前我们只训冻结特征上的 MLP 头，导致合成数据与真实数据在特征空间割裂。在 8×H200 (1.12TB 显存) 下，我们完全有能力微调视觉主干本身：")
    ul([
        "1. 我们是否应该在『3,573 真实图 + 高质量 AIGC 合成集』上，直接对主力视觉大模型（如 InternViT-6B-448px-V2.5、Qwen3-VL-8B-Instruct、DINOv3-ViT-L/16、SigLIP2-NaFlex）进行端到端 LoRA 微调？",
        "2. 如何设计 **Two-Stage Sim2Real（第一阶段：真实+合成混合预微调拓宽边界；第二阶段：100% 纯真实数据退火精调校准分布）** 或 **Domain-Adversarial / 梯度投影（让合成数据的梯度只更新与真实数据同向的分量）**，确保模型吸收了合成数据的缺陷特征，又绝对不会把合成域的偏差带到最终预测中？"
    ])

    h2("方向 4：输入分辨率与多尺度架构在端到端微调下的最佳形态")
    ul([
        "1. 当我们把合成数据加入并对视觉 Backbone 做端到端微调时，如何处理原图非正方形长宽比（保持长宽比 Resize + Pad，还是原生分辨率多块 Crop + 全局缩略图双流输入）？",
        "2. 分类头应当采用何种最简洁、最稳健的序数输出结构（例如干净的单调 CORN 头或条件序数头），使得推理端无需任何复杂的后处理魔改、仅靠最朴素的固定阈值就能直接输出高质量的等级预测？"
    ])

    h2("方向 5：纯模型层面的离线验证准则（彻底杜绝面向 test_150 调参）")
    ul([
        "1. 在完全不触碰 test_150 的前提下，如何仅在 3,573 张真实图像的严格分组交叉验证（GroupKFold OOF，且验证集 100% 为纯真实图像、绝不混入任何合成图）上评估 AIGC 合成数据与模型本体升级的真实净增益？",
        "2. 合成数据与真实数据的最佳采样配比（例如每个 Batch 中 Real : Synth = 4 : 1 还是仅针对稀缺等级按需过采样）应当如何设定？"
    ])

    return blocks


def main():
    blocks = build_report_blocks()
    write_markdown(blocks, REPORT_MD_PATH)
    write_docx(blocks, REPORT_DOCX_PATH)
    print(f"Successfully generated:\n  - DOCX: {REPORT_DOCX_PATH}\n  - MD  : {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
