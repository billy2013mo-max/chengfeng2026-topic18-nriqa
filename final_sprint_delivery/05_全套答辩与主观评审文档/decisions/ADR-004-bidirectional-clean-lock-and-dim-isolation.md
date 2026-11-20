# ADR-004: 在多版本级联解码器中实施双向 Clean 整图锁与非目标维隔离

## Status
Accepted

## Date
2026-11-19

## Context
官方客观指标 `Doc2_Strict` 中 `20·S_clean` 存在强非对称一票否决特性：全无问题图只要任意一个维度被擦边判为 `P2`，整图即判为假阳性并扣除 Clean 分。在 `deconf_v2` 已于 `test_150` 达到 `Clean Score = 20.00 / 20` 满分、8-Fold OOF `S_clean = 19.8441` 后，后续每次针对单个缺陷（如 `过度生成`、`模糊`）升级校准器时，期望效用解码器 `TrueGateConstrainedEU` 的耦合全局项 `pos_gain_sum + clean_diff` 会把该维度的微小概率变动传导到整图 Clean 判定和其他无关维度，造成跨维度回归。

## Decision
在级联升级链 `deconf_v2 -> deconf_v4_tile -> deconf_v5_boundary -> deconf_v6_boundary` 中，每层升级解码器显式强制执行两项结构化不变量：
1. **非目标维隔离**：每层仅释放本层经 8-Fold 嵌套 OOF 验证显著提分的维度（`v4_tile` 仅释放 `{过锐, 过度生成, 虚化突兀}`；`v5_boundary` 仅释放 `{模糊, 纹理损失}`；`v6_boundary` 仅释放 `{模糊, 过度生成}`），其余维度直接继承上一稳定层的离散输出；
2. **双向 Clean 整图锁**：记上一层全无问题掩码为 `cl_prev = (pr_prev == 0).all(axis=1)`，强制 `pr_new[cl_prev] = 0`（防止旧 Clean 误杀），同时对 `newly_clean = (~cl_prev) & (pr_new == 0).all(axis=1)` 回写 `pr_new[newly_clean] = pr_prev[newly_clean]`（防止真缺陷图因单维微调整体漏检）。

## Alternatives Considered

### 备选 A：每代升级后全 6 维端到端联合重新解码（不加 Clean 锁与维隔离）
- 优点：解码逻辑单层扁平，代码更少。
- 缺点：单维校准器调整 `p0_og` 或协变量时，通过 `clean_diff` 耦合项改变 `纹理损失/过锐` 边界并引入 Clean 误杀波动，导致“修一维、退两维”。
- 结论：拒绝。

## Consequences
- 同一组模型概率输入下，升级链路在 `test_150`（`150/150`）与 `final_644`（`644/644`）上保持 `100%` 相同的 Clean 掩码，且在 8-Fold OOF 上实现非目标维严格零回退。
- 推理期解码器对象为 4 层嵌套组合（`BoundaryDeconfV6Bundle -> BoundaryDeconfV5Bundle -> TileSalienceDeconfV4Bundle -> UpgradedDeconfBundle`），序列化与反序列化需要依次导入对应模块；单图 CPU 耗时仍极低（实测 `~4~5 ms`），且严格保持逐行 `N=1 == N=Batch` 不变性（`max_q_diff <= 3.72e-13`）。
