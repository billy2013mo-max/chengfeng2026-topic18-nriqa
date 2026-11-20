# Phase A + Phase B 修复与重训完整执行报告

日期：2026-11-07  
最终提交：[`pred_nriqa_dataset_test.json`](file:///wanqing-develop/luowenjing/18/pred_nriqa_dataset_test.json)  
归档备份：[`results/final/pred_nriqa_dataset_test_PHASEB_FIXED.json`](file:///wanqing-develop/luowenjing/18/results/final/pred_nriqa_dataset_test_PHASEB_FIXED.json)  
SHA-256：`b0e884f0b220f97307bd3deb71ddef85c5420d53b7fd194db0bbf9c89503803a`

---

## 1. 修复的核心 Bug 与工程遗漏汇总

| 模块 | 原版问题（Claude Code 实现） | 修复方案与代码落点 |
| :--- | :--- | :--- |
| **Phase B 物理旁路零梯度断路** | `train_v12_abmil_pure_real.py` 中 `phys_norm/phys_proj/phys_gate` 未加入 `AdamW`，且 `phys_gate=0` 与 `phys_proj[-1]=0` 同时为零导致梯度恒为 0，24 维物理特征在训练与推理中输出恒为 0 | [`train_v12_abmil_pure_real.py`](file:///wanqing-develop/luowenjing/18/code/finetune/train_v12_abmil_pure_real.py#L160-L186)：`phys_gate` 初始化为 `0.25`（保留 `phys_proj[-1]=0` 实现 step-0 零残差同时打通 step-1 梯度），并将 `phys_params` 纳入 `AdamW` 的 `head_params` |
| **Phase B 24 维物理算子缺失比值与显著性掩码** | `NativePixelPhysicsExtractor` 对 3 个切块直接 `mean(1)` 抹平了峰值与中位切块对比，且 `FB-STR` 未使用全局图与 ViT Saliency Map | [`abmil_layers.py`](file:///wanqing-develop/luowenjing/18/code/finetune/abmil_layers.py#L245-L380)：显式计算 `med_to_peak_hi/lap/gvar` 峰值-中位能量比，并从 `TokenGatedABMILPool.last_sal_weights` 提取全局显著性图计算前景-背景锐度跳变比 `FB-STR` |
| **Phase B `SA-AOR` 损失权重自相抵消** | `(-diff * margin).sigmoid()` 将 `margin` 当作乘子而非加性裕度，且 `batch=2` 下单对正负样本 `(loss * w).sum() / w.sum()` 将场景权重 `w_scene` 100% 约分抵消 | [`train_v12_abmil_pure_real.py`](file:///wanqing-develop/luowenjing/18/code/finetune/train_v12_abmil_pure_real.py#L102-L148)：改为加性裕度 `F.softplus(4.0 * (margin - diff)) / 4.0` 并以 `w.numel()` 归一化，同时使用 `--batch 4 --grad-accum 2` 构造 $4\times 4$ 样本对矩阵 |
| **Phase A 期望效用坐标上升 `EUM-BCA` 恒等空操作** | `phaseA_decoder.py` 中 `eum_bca_joint_decode` 将 `gold = pseudo.copy()` 设为自身初始预测，导致效用恒为满分、零更新 | [`phaseA_decoder.py`](file:///wanqing-develop/luowenjing/18/code/finetune/phaseA_decoder.py#L262-L385)：重构为基于校准后验边缘概率的 $O(1)$ 增量期望效用坐标上升解码器，并补齐 `fit_extremal_beta_calibration_14ch` 与 `apply_true_oof_quali_rag_v2` |
| **`test_150` 修订预算被零裕度 `P1->P2` 占满并按字母序截断** | `KAPPA_P1P2=[1.2,1.8,1.4,1.3]` 强加给全部 6 个专家投票，产生 16 个 `margin=0.0, score=0.160` 并列项并按文件名字母序截断，误将 8 处 `p1` 概率实际上升或 `p1>p2` 的格降级为 `P2`，同时锁死全部 `P2->P1` 升级（0 处） | [`build_55plus_ultimate_submission.py`](file:///wanqing-develop/luowenjing/18/code/build_55plus_ultimate_submission.py#L978-L1095)：解耦专家投票与全局 `kappa`，拦截概率反向变化的伪降级，要求 `P1->P2` 具备真实 `s1` 下降与 `p2 > s1 + 0.07` 连续裕度，同时打通高置信 `P2->P1` 升级与 `虚化突兀_P0` 召回 |

---

## 2. Phase B V12 单模重训结果（`s16~s20` + `full`）

| Split | Phase A V12 | 原版 Phase B V12 | **修复后 Phase B V12** | 相比原版 Phase B | 测试集 `S_clean` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `s16` | 48.92 | 49.69 | **49.89** | **+0.20** | 1.000 |
| `s17` | 49.67 | 50.03 | **50.17** | **+0.14** | 1.000 |
| `s18` | 42.70 | 50.54 | **50.62** | **+0.08** | 1.000 |
| `s19` | 50.14 | 49.57 | **50.46** | **+0.89** | 1.000 |
| `s20` | 48.97 | 50.48 | 49.39 | -1.09 | 1.000 |
| `full` (holdout) | 49.40 | 50.96 | **51.20** | **+0.24** | 1.000 |

融合三大正交 V12 目录（`results/v12_abmil` C-RADIOv4 + `results/v12_phaseB` InternViT-6B + `results/v12_phaseB_fixed` InternViT-6B + 24D Physics + SA-AOR，共 18 个 V12 模型）后，Family 7 在 OOF 14 通道解耦路由中获得显著更高权重（`过锐_P0: 0.22`, `过锐_P1cum: 0.21`, `虚化突兀_P0: 0.18`, `过度生成_P1cum: 0.17`）。

---

## 3. 五折盲测门禁结果（`s16~s20`, $N=5,000$）

```text
Split    | MultiViT_Base | InternViT6B | Phase1(50:50) | Ultimate(55+) | Gain vs P1
----------------------------------------------------------------------------------
s16      |         52.25 |       50.24 |         52.13 |         53.27 |      +1.15
s17      |         51.98 |       44.13 |         51.55 |         53.16 |      +1.61
s18      |         51.85 |       51.93 |         52.19 |         52.77 |      +0.58
s19      |         51.50 |       44.63 |         51.77 |         52.64 |      +0.87
s20      |         51.56 |       38.36 |         52.11 |         53.23 |      +1.12
--------------------------------------------------------------------------------
POOLED (N=5000) Phase 1 Baseline : total=51.958 (S_pos=15.41, S_p0=16.55, S_clean=20.00)
POOLED (N=5000) Ultimate 55+     : total=53.021 (S_pos=16.25, S_p0=16.77, S_clean=20.00)
Paired Bootstrap Gain: +1.063 pts | 95% CI: [+0.780, +1.325] | p-value=0.0000
```

- 相比 Phase-1 基线（`51.958`）净增 **`+1.063 pts`**（突破 **53.00+** 大关，$p < 0.0001$）。
- 相比 Claude Code 原版 Phase A（`52.790`）净增 **`+0.231 pts`**，相比原版 Phase B（`52.938`）净增 **`+0.083 pts`**。

---

## 4. 最终 `test_150` 提交审计（7-Gate 全通过）

- **All-Clean Images**：**16 / 150**（与 52.89 基线 100% 一致，`S_clean = 20.00/20` 锁定）
- **`花屏_P0`**：**10 / 10**（100% 锁定）
- **P0 总格数**：**103**（较 52.89 基线 `+1`，新增 `182662259039_077.png` `虚化突兀 无问题 -> P0`）
- **P1 总格数**：**88**（精准落在调研方案 `84~90` 目标区间中心）
- **Positive 总格数**：**345 / 900**（较 52.89 基线 `+1`）
- **总修订数**：**21 处**（跨越 19 张图：`P1 -> P2` 12 处、`P2 -> 无问题` 3 处、`无问题 -> P2` 3 处、`P2 -> P1` 2 处、`无问题 -> P0` 1 处），且 12 模型（Phase A+）与 18 模型（Phase A+B Fixed）在 `test_150` 上完全收敛至同一份 21 处离散修订（SHA-256 一致）。
