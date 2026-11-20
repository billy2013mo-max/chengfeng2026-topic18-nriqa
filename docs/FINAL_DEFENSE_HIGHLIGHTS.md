# 【2026乘风挑战赛·命题18】主观评审核心亮点与答辩速查手册 (`FINAL_DEFENSE_HIGHLIGHTS.md`)

> 本文档面向主观评审专家与答辩环节，按官方评分细则逐项汇总本方案在 **执行成本（`S_cost` 20 分）**、**工程可落地性（`S_deploy` 15 分）**、**主观语义判断能力（`+5` 奖励分）**、**客观核心算法创新（`resident_v8_boundary + deconf_v6_boundary`，线上实测再破新高 `54.77 / 65`）** 与 **人机协同研发（AI 使用说明额外激励）** 五个维度的核心证据链与快速索引。

---

## 一、 主观分 40 分逐项得分点与核心证据对照表

| 评分维度与分值 | 官方评审关注点 | 本方案核心实现与实测硬指标 | 对应代码与审计证据 |
| :--- | :--- | :--- | :--- |
| **`S_cost` 执行成本**<br>**（20 分，第一档 `0.7~1.0`）** | • 单张 RTX 4090 (24GB) 稳定运行全 6 维<br>• 零外部/付费 API<br>• 鼓励量化、蒸馏、剪枝、**共享骨干**等轻量化设计 | 1. **无损共享冻结主干（`--share-weights`）**：7 个 `PE-Core-Large` 塔（含 `bndry_pe/bndry_dual`）及 `cradio_v3/bndry_dual/p3_qy/v16` 冻结层经 `torch.equal` 校验共享显存，**零误差省 `5.81 GB`**；<br>2. **单 4-bit 底座双适配器共享 VLM（`DualAdapterQwen3VL4090`）+ Host 词表（`--embed-on-host`）**：单一 `Qwen3-VL-8B` NF4 底座同时承载 `p1_qy`（DoRA r=24）与 `qwen_hr`（LoRA r=64），双 8B VLM 合计仅 **`5.62 GB`**（省 `3.40 GB` + 词表 `1.16 GB`），`argmax` 100.0% 一致；<br>3. **纯 PyTorch `INT8-g128` 量化未共享大骨干 + 语义 ROI 蒸馏 + 激活分块**：对 `v16` 独有 `PE-Core-G14` (1.88B) 启用零依赖对称 `INT8-g128` 权重量化（省 **`1.72 GiB`**，全量 `test_150` 严格 **`0 / 900` 格差异**），叠加 `v16_chunk=4`、`ntg_convnext tile_chunk=4`、`cradio_v3` 比特精确激活分块；<br>4. **单卡实测**：去重激活参数 **`13.262B <= 30B`**，常驻 **`15.78 GB`**，PyTorch 推理峰值 **`16.95 GB`**，`nvidia-smi` 加载峰值 **`19.98 GiB`**、**推理峰值 `18,930 MiB` (`18.49 GiB < 24 GiB`, 单张 RTX 4090 余量 `~5.50 GiB`)**；<br>5. **纯视觉轻量档 `resident_v8_boundary_novlm`**：`8.910B`，线上实测 **`54.52 / 65`**，`nvidia-smi` 峰值 **`12.75 GiB`**（16GB 卡充裕运行），中位 **`2.39s`**。 | [`final_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/final_engine.py)<br>[`dual_vlm_shared_4090.py`](file:///wanqing-develop/luowenjing/18/code/r55/dual_vlm_shared_4090.py)<br>[`resident_v8_boundary_bench_half.json`](file:///wanqing-develop/luowenjing/18/results/r55/final/resident_v8_boundary_bench_half.json)<br>[`resident_v8_boundary_novlm_bench_half.json`](file:///wanqing-develop/luowenjing/18/results/r55/final/resident_v8_boundary_novlm_bench_half.json) |
| **`S_deploy` 可落地性**<br>**（15 分）** | • 单图端到端延迟 `<= 15s`<br>• 极目 / TAT / VQA 链路兼容<br>• 集成改造工作量小、离线可复现<br>• **多帧视频抽样误报抑制验证（§5 进阶）** | 1. **超高余量低延迟**：全量 `test_150` 单卡实测均值 **`3.03s`**、P95 **`4.23s`**、最大 **`6.93s <= 15s`**（冷启动加载提速 `2.1×` 至 `~170s`；12 图压测中位 `3.41s` / 均值 `3.66s`；纯视觉档 `2.39s`）；<br>2. **100% 断网开箱即用 + 离线容器镜像**：[`Dockerfile`](file:///wanqing-develop/luowenjing/18/Dockerfile) + `check_deploy_env.py` **65 项全绿（`0 FAIL, 0 WARN`）**，推理期训练数据读取 **`0` 次**（`R55_AUDIT_TRAIN_READS=1`），严格 `N=1 == N=Batch`（`0 / 900` & `0 / 3,864` 差异）；<br>3. **一键脚本 + 双后端 HTTP 微服务**：`serve_engine.py`（FastAPI / 标准库自动回退）提供 `/healthz`、`/infer`、`/infer_batch` 并直出 **极目 / TAT** 载荷，含归因实时显存 **`18.56 GiB`**；<br>4. **有效独立帧视频聚合器 `EffectiveFrameOR`**：真实 320 段多帧视频保持 **`84.45%` 视频 P0 召回** + **`100.0%` Clean 特异度**（`Doc2=51.922`），高密度抽帧误报压低 `2.6×~3.4×`。 | [`Dockerfile`](file:///wanqing-develop/luowenjing/18/Dockerfile)<br>[`run_final_infer.sh`](file:///wanqing-develop/luowenjing/18/run_final_infer.sh)<br>[`serve_engine.py`](file:///wanqing-develop/luowenjing/18/code/r55/serve_engine.py)<br>[`check_deploy_env.py`](file:///wanqing-develop/luowenjing/18/code/r55/check_deploy_env.py)<br>[`resident_v8_boundary_video_oof_audit.json`](file:///wanqing-develop/luowenjing/18/results/r55/final/resident_v8_boundary_video_oof_audit.json) |
| **`Semantic Score`**<br>**语义判断能力（`+5` 分）** | ① 描述与预测一致性（`0~1`）<br>② 位置描述准确性（`0~2`）<br>③ 解释与图像证据一致性及不确定性表达（`0~2`） | 1. **100% 标签锁定一致**：严锁 `pred_6d`（`test_150` `327/327`、`final_644` `1,346/1,346` 全部一致），所有阳性维度逐一输出 `≤ 40` 字紧凑图像特异解释（零超长、零截断、零自相矛盾、零全局值 Bug，唯一理由占比 **`94.5%` / `87.4%`**）；<br>2. **解决“全图退化乱画框”+ 局部定位定量实证**：未膨胀 `4×4` 粗网格集中度门控区分 `"全图"` 弥散退化（`final_644` **301 次**）与 `8×8` `[x1, y1, x2, y2]` 局部伪影框（**1,045 次**，配零幻觉视觉内容提示如 `人像肤色区/高反差边缘`），真图局部框物理对比命中率达 **`61.5%`**（`花屏 70.0%`、`模糊 69.2%`）；<br>3. **分层洁净分位 + 局部/全图相对对比 + 诚实不确定性**：12 项实测物理比值支持率跃升至 **`36.4%` (`test_150`) / `40.2%` (`final_644`)**，100% 覆盖单帧因果不确定性说明。 | [`measured_evidence.py`](file:///wanqing-develop/luowenjing/18/code/r55/measured_evidence.py)<br>[`subjective_engineering.py`](file:///wanqing-develop/luowenjing/18/code/r55/subjective_engineering.py)<br>[`resident_v8_boundary_attribution_audit.json`](file:///wanqing-develop/luowenjing/18/results/r55/final/resident_v8_boundary_attribution_audit.json) |
| **`AI 使用说明`**<br>**（表现突出额外激励）** | 客观说明使用程度、协同方式、工程护栏与真实攻坚案例 | 详实披露 88% 工程自动化与 100% 人类科学把关边界，复盘“证伪重型双 VLM”、“`deconf_v2 → v4_tile → v6_boundary` 全维度边界攻坚（`54.77`）”、“共享主干 + 双适配器 VLM 省 9.2GB”与“AI 自检修复归因过画框”四大真实闭环案例及 `AGENTS.md` 防过拟合铁律。 | [`AI_USAGE_STATEMENT.md`](file:///wanqing-develop/luowenjing/18/docs/AI_USAGE_STATEMENT.md) |

---

## 二、 评审专家高频提问与标准答辩要点（Q&A Cheat Sheet）

### Q1：你们最终打到 `54.77 / 65` 的核心方法 `resident_v8_boundary + deconf_v6_boundary` 到底做了什么？会不会过拟合？
- **两阶段全维度 `P0/P1/P2` 边界攻坚**：
  - **第一阶段（`deconf_v4_tile → deconf_v5_boundary`，`52.464 → 52.519`）**：在 `deconf_v4_tile`（`过锐` 11D 显微镜去混淆 + `过度生成` 24D `ntg_pe` 1:1 切片局部尖峰 + `BokehSynergyV3` + 双向 Clean 锁死，线上 `54.68`）基础上，严格锁死 Clean 与非目标维，在 `模糊`/`纹理损失` 正样本内部做无泄漏边界阶梯去混淆；
  - **第二阶段（模型层 + 架构层 + Pre-LRO 零稀释融合 `deconf_v6_boundary`，`52.519 → 52.665`，线上 `54.77`）**：
    1. 全新训练 2 个 100% Tag-Free **Stage 2B 边界专精塔（`bndry_pe`, `bndry_dual`）**：引入 1:1 边缘峰值 vs 全局/纹理残差 Token + 7-Query 跨缺陷自注意力 + 边界加权 RPS + 有界 Hinge 间隔 + 条件 XBM + 退火零泄漏 KD，单模 `PosRPS = 0.1219`（全库第 2），`过度生成` 单模全库第 1；
    2. 通过 **`DualAdapterQwen3VL4090`** 在单一 4-bit 底座上挂载双适配器，零爆显存合入高分语义 ROI 大模型 `qwen_hr`（`1344² + 3×448`）；
    3. 提出 **Pre-LRO 零稀释融合**：在 `过度生成` 24 维去混淆校准**之前**先融合 `(bndry_pe, bndry_dual, qwen_hr)`，使 `过度生成` `P0/P1/P2` 三档 F1 同步大涨（`0.604→0.620 / 0.377→0.406 / 0.448→0.455`），`模糊 P1` 升至 `0.577`，其余维度与 Clean 零回退。
- **为什么零过拟合风险？**
  1. **100% 在训练 OOF 内层视频分组 CV 上拟合**，未读取测试集任何统计量，100% 从最新 FullFit 权重从头独立推理（`R55_AUDIT_TRAIN_READS=1` 零训练读取，`N=1 == N=Batch` 差异 `0/900` & `0/3,864`）；
  2. **极强正则收缩（$L_2 = 1200$）+ 双向 Clean 锁死 + 零自由截距秩-1 约束**；
  3. 8 折零泄漏 Nested OOF 跃升至 **`orig = 52.665 / all = 51.958`（全库最高）**，10,000 次 654 图视频簇置换检验 `Perm654(Pos<15)` 风险降至 **`0.00%`**（`ClBoot654 = 0.08%`），线上实测同方向再破新高 **`54.77 / 65`**（`正向缺陷 17.46/25`、`P0 17.31/20`、`Clean 20.00/20` 满分）。

### Q2：14 个专精成员（含两套 8B Qwen3-VL 适配器）同时常驻单张 RTX 4090（24 GB），为什么显存依然只有 `15.78 GB` 常驻 / `18.49 GiB` 推理峰值？
- 因为我们采用了 **“四重无损压缩”极简显存架构**：
  1. **视觉冻结层共享（`--share-weights`）**：7 个含 `PE-Core-Large` 的专精塔与 `C-RADIOv4-H` 多塔的冻结底层块经 `torch.equal` 逐位校验合并至同一物理显存块（**无损省 `5.81 GB`**，新增 `bndry_pe` 仅增 `+0.28 GB`、`bndry_dual` 仅增 `+0.63 GB`）；
  2. **VLM 4-bit 底座双适配器共享（`DualAdapterQwen3VL4090` + `--embed-on-host`）**：`p1_qy`（DoRA r=24）与 `qwen_hr`（LoRA r=64）共享同一个 4-bit NF4 `Qwen3-VL-8B` 底座并将 15.2 万词表驻留主机内存（省 `1.16 GB`），两套 8B VLM 合计常驻仅 **`5.62 GB`**（较双独立实例省 **`3.40 GB`**）；
  3. **纯 PyTorch 对称 `INT8 group-128` 量化未共享巨型骨干**：仅对 `v16` 独有未共享的 `PE-Core-G14`（1.88B）在 CPU 加载时做纯 PyTorch `INT8-g128` 权重量化（100% BF16 激活，零外部依赖，**再省 `1.72 GiB`**，全量 `test_150` 实测严格 **`0 / 900` 格差异**）；
  4. **比特精确激活削峰与推理基础设施加速**：`v16_chunk=4`、`ntg_convnext tile_chunk=4`、`cradio_v3` 分块与 `expandable_segments:True` 将全 14 成员常驻压至 **`15.78 GB`**、`nvidia-smi` 加载峰值 **`19.98 GiB`**、**推理峰值 `18.49 GiB < 24 GiB`（留足 `~5.50 GiB` 充裕余量）**；全量 `test_150` 均值仅 **`3.03s`（最大 `6.93s <= 15s`）**；若需部署 16 GB 卡，一键切 `--set resident_v8_boundary_novlm` 线上实测 **`54.52 / 65`**、峰值仅 **`12.75 GiB`**、中位 **`2.39s`**。

### Q3：如果将单帧模型直接用于视频多帧抽样质检，如何防止 Clean 视频误报随抽帧数爆炸？
- 若对抽取的 $M$ 帧直接取逻辑或（Naive OR），单帧微小假阳性在 $M=16/30$ 帧下会膨胀至 **`11.23% / 19.63%`**；但若简单对多帧概率取平均，又会把仅出现 1~2 帧的瞬态解码花屏或局部崩坏稀释漏检（真实 320 段多帧视频实测视频 `P0` 召回从 `84.45%` 暴跌至 `74.37%`）。
- 本方案设计的 **`EffectiveFrameORVideoAggregator`** 从训练 OOF 中估计出各缺陷的同视频帧间相关系数 $\rho_d \in [0.15, 0.48]$，将抽样帧数折算为有效独立帧数 $M_{\text{eff}}(M) = 1 + (M-1)(1-\rho_d)$，仅对长视频上的“孤立单帧擦边阳性”按 $M_{\text{eff}}$ 适度收紧后验阈值，而对 $\ge 2$ 帧共现或高置信单帧 `P0` 100% 放行。实测在 320 段真实多帧视频上保持 **`84.45%` 视频 P0 召回率** + **`100.0%` Clean 特异度**（`Doc2=51.922`），并将 $M=8/16/30$ 帧 Clean 视频误报率较朴素 OR 分别压低 **`3.4× / 2.8× / 2.6×`**（$M=16$：`11.23% → 4.07%`）。

### Q4：语义归因模块如何避免大模型“幻觉编造上游成因”和“全图模糊乱画局部框”？
1. **不调生成式 LLM 编故事**：归因完全由确定性空间算子与 12 项实测像素/无量纲物理指标生成，`pred_6d` 100% 锁定不可篡改，耗时仅 `~0.16s/图`；
2. **`4×4` 粗网格集中度门控**：修复了子块锐化导致的虚高集中度，使全局均匀失焦/降噪准确输出 `"全图"`，仅在局部高集中度伪影上输出 `[x1, y1, x2, y2]` 像素框；
3. **显式单帧因果不确定性**：严格遵循赛题附录二精神，对单帧无法唯一判定的上游成因（如失焦 vs 运动模糊、前处理降噪 vs 编码低码率、实拍大光圈浅景深 vs 算法抠图虚化）明确给出不确定性说明，绝不强下武断因果结论。
