# Phase B 执行报告（原生像素物理旁路 + SA-AOR 重训）

日期：2026-11-06  
代码基线：`a0d5672`（Phase A）  
最终提交：`pred_nriqa_dataset_test.json`  
SHA-256：`946919af3ccae0b21e0248f2e5793d6c112c7683c54cf7b3a58035865a0607e2`

## 1. 实现内容

| 组件 | 实现 |
| :--- | :--- |
| 3 张 1:1 原生像素显微镜切块 | `data._native_patches`：峰值 Sobel、峰值 Laplacian、中位纹理窗口；零缩放 448×448 |
| 24 维空频域物理算子 | `abmil_layers.NativePixelPhysicsExtractor`：8 维 Signed-Laplacian bipole ringing + 8 维 log-PSD/edge-texture + 8 维 FB-STR 中心-边界跳变；无额外可学习参数 |
| 物理特征注入 | V12 模型零初始化残差 `fused + tanh(gate) * MLP(LN(phys))`；step-0 与基线完全一致 |
| 场景感知非对称序数损失 | `compute_scene_aware_asymmetric_ordinal_loss`：P0↔P1 margin 0.65、P1↔P2 margin 0.45、Pos↔0 margin 0.80；场景相似度加权 1+1.5·exp((cos−1)/0.15)；Clean 2.5× focal 惩罚 |
| 主干 | InternViT-6B + PE-Core-G14 + DINOv3-L，last-k=4；8 视图（1 global + 2×2 tiles + 3 native） |
| 训练 | 4 epochs、batch 2 × grad-accum 4、SWA；s16~s20 + full 共 6 个模型 |
| 未实现 | DoRA（当前是 last-k 全块微调，不是 LoRA）；R-Drop（双前向成本过高，暂缓） |

## 2. V12 单模提升（s16~s20 test）

| Split | Phase A V12 | Phase B V12 | Δ |
| :--- | ---: | ---: | ---: |
| s16 | 48.92 | **49.69** | +0.77 |
| s17 | 49.67 | **50.03** | +0.36 |
| s18 | 42.70 | **50.54** | **+7.84** |
| s19 | 50.14 | 49.57 | −0.58 |
| s20 | 48.97 | **50.48** | +1.51 |
| **均值** | **48.08** | **50.06** | **+1.98** |

s18 的 Clean 崩溃被彻底修复（S_clean 0.632 → 1.000）。full holdout：val 50.96 / test 50.96（Phase A 49.40）。

## 3. 五折盲测门禁（s16~s20, N=5,000）

```
Split   | Phase1(50:50) | Phase B | Gain
s16     |     52.13     |  53.33  | +1.21
s17     |     51.55     |  53.10  | +1.55
s18     |     52.19     |  52.73  | +0.54
s19     |     51.77     |  52.40  | +0.62
s20     |     52.11     |  53.10  | +0.99

POOLED: 51.958 -> 52.938
  S_pos  15.41 -> 16.22
  S_p0   16.55 -> 16.72
  S_clean 20.00 -> 20.00
Paired Bootstrap: +0.980, 95% CI [+0.700, +1.259], p < 0.0001
```

对比 Phase A 的 52.790，Phase B 再增 **+0.148**；对比 `9323007` 的 52.482，净增 **+0.456**；对比 Phase-1 基线净增 **+0.980**。PDF 的 Phase A 门禁 +1.10 已接近，且所有单折为正。

## 4. 最终 test_150 提交

- Clean：**16 / 150**（与 Phase-1 完全一致）
- P0 cells：**102**（与 Phase-1 一致；P0 图片集合不变）
- Positive cells：**345**（+1）
- 修订：28 处（P1→P2 23、无→P2 3、P2→无 2）
- 7-Gate 全部通过；格式校验 150 唯一 ID、模板顺序、六类合法标签

**注意**：Phase B 的这一版没有选中 Phase A 的 `182662259039_077` 虚化 P0 召回（routed P0 概率下降），因此该版本 P0 集合与 Phase-1 完全一致、更保守。如果线上更看重 P0 召回，可在 Phase A 版本上单独加回该格。

## 5. 复现命令

```bash
# 5 个 split + full
for spec in "s16 data/resplits/split_s16.csv 16" "s17 data/resplits/split_s17.csv 17" \
            "s18 data/resplits/split_s18.csv 18" "s19 data/resplits/split_s19.csv 19" \
            "s20 data/resplits/split_s20.csv 20" "full data/split_all_full.csv 101"; do
  set -- $spec
  python3 -u code/finetune/train_v12_abmil_pure_real.py \
    --split $2 --backbones "internvit_abmil:lastk4,pecoreg_abmil:lastk4,dinov3l_abmil:lastk4" \
    --size 448 --tiles 2 --native 3 --epochs 4 --batch 2 --grad-accum 4 --seed $3 \
    --out results/v12_phaseB/v12_abmil_$1.json
done

# 最终构建（Phase B V12 目录）
V12_DIR=results/v12_phaseB python3 -u code/build_55plus_ultimate_submission.py
```

## 6. 结论与下一步

- Phase B 的核心收益来自 **InternViT-6B + 原生物理旁路 + SA-AOR** 的联合作用，V12 单模均值 +1.98，集成盲测 +0.980。
- 仍未达到 PDF 的 Phase C 目标（线上 55.5~56.4），但已经把 `9323007` 的 52.70 落后（−0.19）扭转为一个盲测 +0.98 的正向版本。
- 建议：先提交 Phase B 候选（剩余提前评测机会），记录 `S_pos/S_p0/S_clean`；若低于 Phase A 或 52.89，回滚到 `pred_nriqa_dataset_test_PHASEA.json`。
