# ADR-003: 在过度生成校准中采用 Pre-LRO 对数线性融合与零截距秩-1 序数偏移

## Status
Accepted

## Date
2026-11-19

## Context
在引入第二阶段新专家（`bndry_pe`、`bndry_dual`、`qwen_hr`）后，需要将其概率输出与现有 `deconf_v4_tile` / `deconf_v5_boundary` 的跨尺度、跨缺陷及 1:1 切片空间显著度去混淆校准（`LowRankOrdinalCalibrator`）结合，目标是提升 `P0 / P1 / P2` 边界区分度而不引发测试集过拟合。

## Decision
1. **校准器采用零截距秩-1 序数偏移（[`code/r55/deconfound_calibrator.py`](file:///wanqing-develop/luowenjing/18/code/r55/deconfound_calibrator.py)）**：对目标维度 $d$ 固定使用单标量 $\eta_i = z_{i,\text{sel}}^\top w$ 乘以固定序数锚点 `ANCHOR_SHIFT = [-1.25, +0.35, +0.90, +1.45]`，再与基础概率做 $\alpha$ 线性阻尼，通过 4 折内层视频分组 CV 选择 `(topk, l2, alpha)` 并强制满足 RPS 不退化护栏（否则回退 $\alpha = 0$）。
2. **`过度生成`（`dim=4`）采用 Pre-LRO 零稀释融合（[`code/r55/upgrade_deconf_v6_boundary.py`](file:///wanqing-develop/luowenjing/18/code/r55/upgrade_deconf_v6_boundary.py)）**：先将温度对齐后的 `(bndry_pe, bndry_dual, qwen_hr)` 以非负 SLSQP 权重 `[0.1117, 0.1483, 0.1400]` 对数线性融合到**原始基础路由池化概率 `Q_raw[:, 4]`** 得到 `og0`，再以 `og0` 重算 24 维空间与切片协变量并拟合 `cal_og24_v6`（`topk=24, l2=1200.0, alpha=0.85`）。
3. **`模糊`（`dim=0`）采用 Post-LRO 轻权重融合**：在 `Q_v5[:, 0]` 之后叠加 `bndry_pe`（`w=0.06`）。

## Alternatives Considered

### 备选 A：Post-LRO 后融合（在 `cal_og24` 校准完成之后再混入三名新专家）
- 优点：可直接复用 `v4_tile` 已拟合的 `cal_og24` 参数，无需重拟合。
- 缺点：后融合会直接冲淡 24 维空间/切片去混淆拉开的局部尖峰修正量，8-Fold 嵌套 OOF 较 Pre-LRO 损失 `0.07 ~ 0.09` 分。
- 结论：拒绝。

### 备选 B：多分类全秩逻辑斯蒂回归 / 树模型 Meta-Stacker（自由截距 + 4 类独立权重）
- 优点：拟合自由度高，训练集指标更高。
- 缺点：在 3,573 样本（尤其补充集 `0%` Clean 偏移）下极易过拟合边界截距，破坏 `DirDiag` 已校准的边际先验与 Clean 得分。
- 结论：拒绝。

## Consequences
- 8-Fold 零泄漏 Nested OOF 中 `过度生成` 三档 F1 全面提升：`P0 0.6044 -> 0.6196 (+1.52%)`、`P1 0.3774 -> 0.4058 (+2.84%)`、`P2 0.4478 -> 0.4548 (+0.70%)`，`RPS` 由 `0.1274` 降至 `0.1238`；`模糊 P1` 由 `0.5729` 升至 `0.5767`；总分 `ORIG` 由 `52.519` 跃升至 `52.665`（`+0.146`）。
- 依赖极强正则 `l2=1200.0` 抑制 24 维协变量方差；若未来更换底层基础池成员，必须重新按 8 折内层分组 CV 拟合 `cal_og24_v6`。
