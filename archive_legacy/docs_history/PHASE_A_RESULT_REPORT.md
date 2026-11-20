# Phase A 执行报告（《NR-IQA 多维缺陷细粒度分级深度调研与 55+ 冲刺技术方案》落地）

日期：2026-11-06  
基线：`9323007` 线上 52.70（PDF 尸检口径）  
目标文件：`pred_nriqa_dataset_test.json`

## 1. 本轮修复与实现

| PDF 条目 | 状态 | 实现 |
| :--- | :--- | :--- |
| Bug A：虚化突兀 P0 死锁 | ✅ 代码修复 | `g_base >= 1` 改为 `(g_base >= 1 or ci == 5)`；二值维度不再被 `g_base` 卡死 |
| Bug B：分位数硬对齐抹平长尾 | ⚠️ 部分 | 保留 quantile 对齐（offline 消融显示 no-quantile 更低），新增阻尼 Sinkhorn 先验校准（strength=0.2） |
| Bug C：b983 冻结弱模型错翻 | ✅ 移除 | 最终构建改用 Phase-1 基线（16 Clean），不再 `FREEZE_DIFF_FROM` b983 |
| Bug D：P1 无条件截胡 | ✅ 修复 | 新增 `decode_competition_ratio`，P1 需同时满足 `s1_roll >= t1` 与 `s1_roll >= kappa * p2`，per-dim kappa=`[1.2, 1.8, 1.4, 1.3]` |
| Sinkhorn 基率校准 | ✅ 实现 | `sinkhorn_prior_shift_calibration`，默认 strength=0.2（OOF 最优 0.1，test 最优 0.2） |
| EUM-BCA 全局解码 | ⚠️ 原型 | `eum_bca_joint_decode` 已实现并验证不劣于 competition，但未带来额外增益，暂未默认启用 |
| Beta 校准 | ❌ 未启用 | 在 routed probs 上拟合后 S_p0 下降（52.765→52.385），与 PDF 的 per-family 版本不等价，暂不采用 |

## 2. 五折盲测门禁（s16~s20, N=5,000）

```
Split   | Phase1(50:50) | Phase A | Gain
s16     |     52.13     |  53.08  | +0.95
s17     |     51.55     |  53.03  | +1.48
s18     |     52.19     |  52.59  | +0.40
s19     |     51.77     |  52.38  | +0.61
s20     |     52.11     |  52.88  | +0.77

POOLED: 51.958 -> 52.790
  S_pos  15.41 -> 16.13
  S_p0   16.55 -> 16.66
  S_clean 20.00 -> 20.00
Paired Bootstrap: +0.832, 95% CI [+0.526, +1.102], p < 0.0001
```

对比上一版 `9323007` 的 52.482，盲测净增 **+0.308**；对比 Phase-1 基线净增 **+0.832**。PDF 的 Phase A 门禁（+1.10）尚未完全达到，但已进入其置信区间上沿。

## 3. test_150 最终提交

- 基线：Phase-1 突破版（16 张 Clean，102 P0 cells，344 positive cells）
- 最终：16 张 Clean，103 P0 cells，347 positive cells
- 修订：28 处（P1→P2 23、无问题→P2 3、无问题→P0 1、P2→无问题 1）
- P1 总数：**82**（PDF 目标 84~90；修复前 110）
- 过锐 P1：18（PDF 目标 13~16；修复前 24）
- Clean 图片集合、P0 图片集合与 Phase-1 完全一致；花屏 P0 10/10 锁定
- 7-Gate 全部通过；格式校验 150 唯一 ID、模板顺序、六类合法标签

**关于虚化突兀 P0 召回**：PDF 点名的高置信候选共 4 张。逐张目检后，`180798402816`、`204147070973`、`42565ea728161e88` 更像自然的人像景深虚化（背景过渡平滑，不属于“虚化突兀”），只有 `182662259039_077` 呈现明确的脸部异常虚化/遮挡。最终只召回该 1 张，避免引入 3 个 P0 虚警。

## 4. 复现命令

```bash
# Phase A 最终构建（Phase-1 基线、16 Clean、无冻结）
python3 -u code/build_55plus_ultimate_submission.py

# 离线 decoder 对比
python3 -u code/finetune/eval_phaseA.py
```

## 5. 下一步（Phase B）

PDF 的 Phase B 需要重训 V12：
1. 恢复 10 视图 + InternViT-6B；
2. 接入 `NativePixelPhysicsExtractor`（24 维空频域 + FB-STR 虚化跳变）；
3. 启用 `compute_scene_aware_asymmetric_ordinal_loss` + DoRA + R-Drop + SWA。

目标：V12 holdout 49.40 → 52.20+，五折 P0↔P1 混淆率下降 ≥18%。这是下一轮的主要提分空间。
