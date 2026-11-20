# Project Core Engineering & Inference Rules (Persistent Memory)

## 1. 严禁基于历史推理 JSON 做对齐或锚定覆写 (No Alignment or Anchoring to Legacy Inference JSONs)
- **绝对禁止**读取任何历史版本的测试集推理结果（如 `pred_nriqa_dataset_test_phaseB_overgen8.json`、`pred_nriqa_dataset_test_phase1_breakthrough.json`、`52.43` / `52.89` / `53.03` 等旧 JSON）来作为基底（`pred_base`）、做差分门控覆写（Margin Gate Overriding）或借用其 Clean 图像名单。
- **每次训练出新版本模型后，必须 100% 使用最新训练出的模型权重从头独立重新推理生成全部预测结果**，绝不掺杂或继承任何老模型 JSON 的输出标签。

## 2. 严禁过拟合与针对测试集的硬编码调参 (Zero Overfitting & Zero Test-Set Tuning)
- **禁止任何形式的测试集过拟合**：不得针对 `test_150` 的特定图片名称硬编码规则、不得为凑特定目标格数（如强行凑 P0/P1/P2 数量）去反向微调阈值、不得使用测试集分布信息做后验校准。
- 所有概率尺度对齐与决策阈值必须 **100% 严格基于训练/验证集（`va`）确定**，并在独立盲测集（`s16`、`s17` 等 `te`）上完成零泄漏验证后，原封不动应用于测试集推理。

## 3. 坚持简洁、干净、易交付与易复用的上线架构 (Lean, Clean & Deliverable Architecture)
- **严禁使用 69 模型历史缓存（如 `fast_stack5553_cache.pkl`）或二十几个异构模型的庞大堆叠**作为中间评估基座或推理模型。
- **不强制绑定固定成员模板**（不必强行凑 `InternViT-6B + 4 专家`），但无论是中间验证还是最终推理，均须围绕**结构简洁、无批次脏标记（100% Tag-Free）、工程上易交付与易复用的精简模型组合**（严守总参 `≤ 30B`、单张 RTX 4090 `24GB` 显存安全余量与耗时预算）开展，以真实泛化精度与工程简洁度共同择优。
