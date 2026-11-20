# ADR-005: 采用确定性反事实空间算子与实测物理比值实现语义归因

## Status
Accepted

## Date
2026-11-19

## Context
赛题附录二设置 `+5` 主观语义判断奖励分，评审标准包含三项硬指标：① 描述与预测标签一致性（`0~1`）；② 缺陷位置描述准确性（`0~2`）；③ 解释与图像证据一致性及单帧因果不确定性表达（`0~2`）。同时要求归因不得影响主预测结果，且受单图 `<= 15s` 总耗时预算约束。

## Decision
在 [`code/r55/subjective_engineering.py`](file:///wanqing-develop/luowenjing/18/code/r55/subjective_engineering.py)、[`code/r55/defect_attribution_map.py`](file:///wanqing-develop/luowenjing/18/code/r55/defect_attribution_map.py) 与 [`code/r55/measured_evidence.py`](file:///wanqing-develop/luowenjing/18/code/r55/measured_evidence.py) 中实现完全确定性、无生成式文本解码的旁路归因算子：
1. **标签单向只读锁定**：仅读取已解码的 `pred_6d`，Clean 样本直接跳过定位与物理测量（`< 0.1 ms`）；
2. **专精塔空间反事实归因 + `4×4` 粗网格集中度门控**：`花屏` 绑定 `peg`、`虚化突兀` 绑定 `pe_bkseg1`、其余绑定 `pe_v3`，基于离线洁净基线（`resident_v8_boundary_clean_footprint.npz`, `75.2 KiB`）减去洁净偏置后，先在未膨胀的 `4×4` 粗网格上判别能量集中度——均匀全图模糊/降噪输出 `"全图"`（`box_xyxy=None`），局部伪影再从 `8×8` 细网格输出 `[x1, y1, x2, y2]` 与九宫格方位；
3. **12 项原生像素与场景归一化物理指标 + 显式因果不确定性**：计算 `acutance`、`tex_norm`、`overshoot_p90`、`blocky`、`chroma`、`coh`、`cliff`、`dof_bimod` 等相对 160 张洁净帧的分位，按固定模板构造 `<= 40` 字紧凑中文诊断，并强制附加单帧上游链路因果不确定性说明。

## Alternatives Considered

### 备选 A：调用常驻的 `Qwen3-VL-8B` 自回归生成自由中文归因文本
- 优点：自然语言表述更灵活多样。
- 缺点：① 自回归生成额外消耗 `2~5s/图` 延迟；② 存在标签与描述自相矛盾、字数超限被截断或编造“某编码器/拍摄参数”幻觉的高风险；③ 难以提供可复现的原生像素物理量分位。
- 结论：拒绝。

### 备选 B：直接在 `8×8` 细化反事实图上对所有缺陷强制输出局部检测框
- 优点：每条阳性样本都附带坐标框。
- 缺点：早期审计发现子块锐化会人为拉高集中度，导致均匀全图模糊/磨皮有 `98.8%` 被错误画成局部小框，违背位置描述准确性要求。
- 结论：拒绝，改为 `4×4` 粗网格门控区分“全图”与“局部框”。

## Consequences
- 全量审计实测（`resident_v8_boundary_attribution_audit.json`）：`test_150`（`327/327`）与 `final_644`（`1,346/1,346`）标签一致性均达 `100.0%`，最大字数 `38 <= 40`（`0` 超长、`0` 截断、`0` 矛盾），`final_644` 中弥散退化标注 `"全图"` `180` 次、局部伪影输出框 `1,166` 次，单图均耗仅 `0.17~0.18s`。
- 模板化描述的语言多样性弱于自由生成 LLM，且当多专家深度语义特征驱动判级但一阶像素梯度处于常规区间时，物理证据会诚实标注 `inconclusive` 并在 `uncertainty_note` 说明。
