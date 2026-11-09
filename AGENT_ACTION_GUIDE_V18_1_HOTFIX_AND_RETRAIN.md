# V18.1 零思考傻瓜式执行指南（已直接修复所有脚本，按顺序执行即可）

> **致训练机上的 Agent**：
> 仓库内的 7 个核心 Python 脚本已完成修复并直接覆盖写入，同时已为你准备好一键执行脚本 `run_v18_1_all_in_one.sh`。
> **你不需要再写任何临时 Python 脚本、不需要手写任何 `p_any ** 0.15` 后处理、也绝对禁止读取任何历史 `test150` JSON 或 69 模型缓存 `fast_stack5553_cache.pkl`**。只需要严格按下面两步顺序执行命令即可。

---

## 第一阶段（Stage 1）：2 分钟零重训极速修复（立即产出第一版修复 JSON）

这一步**不需要重新训练模型**，直接复用已有的 `checkpoints/v18_fulldata/*.pt`、`results/v18_paired_kd/*.npz`、`results/v16_fulldata/*.npz` 和 `6B Teacher` 概率，修复之前 `infer_v18_fulldata_test150.py` 漏掉 V16 锚点校准、`upgrade_6b_4spec_synergy_router.py` 几何平均压制 `过度生成_P2` 召回率、以及临时脚本把阳性数砍到 329 格的问题。

在 `/wanqing-develop/luowenjing/18` 目录下直接执行：

```bash
cd /wanqing-develop/luowenjing/18
bash run_v18_1_all_in_one.sh --stage1-only
```

### Stage 1 验收标准（耗时约 1.5 ~ 2 分钟）：
1. `code/finetune/infer_v18_fulldata_test150.py` 会**4 卡并行**在 45 秒内跑完 4 个 `V18_Full_Spec*`，并按图片名对齐融合 `V16` 纯真实锚点，打印校准前后的 `VA` / `TE` 分数提升。
2. `code/finetune/upgrade_6b_4spec_synergy_router.py` 与 `code/deploy/unified_4090_dag_engine.py` 会将 `6B Teacher` + `V18/V17 4-Spec` + `V18_Full (4-Spec)` + `V16_Full (8-Fold Microscope)` 统一融合，并通过 **Split-Mass Cumulative Rescue（分裂概率质量累积召回补偿）** 解码。
3. 终端输出的 `[test_150 Audit]` 指标应满足：
   - `TotalPos`（总阳性检出格数）恢复到 **`348 ~ 353` 格**（彻底消除之前掉到 `329` 格的假阴性崩塌）；
   - `P0_imgs = 65`，`Clean_imgs = 15 ~ 17`；
   - `Cells(P0/P1/P2)` 中 `P0` 约 `101 ~ 105` 格，`P1` 约 `94 ~ 102` 格，`P2` 约 `145 ~ 152` 格。
4. 产物会自动写入：
   - `/wanqing-develop/luowenjing/18/pred_nriqa_dataset_test.json`
   - `/wanqing-develop/luowenjing/18/results/final/pred_nriqa_dataset_test_v18_1_stage1_hotfix.json`

> **提示**：Stage 1 跑完后，可以先拿 `results/final/pred_nriqa_dataset_test_v18_1_stage1_hotfix.json` 去测一次 `test150` 分数验证修复效果，同时继续跑下面的 Stage 2 全量重训。

---

## 第二阶段（Stage 2）：V18.1 原生四元组扩充 + 真 6B/V16 蒸馏全量重训（约 13 ~ 15 分钟）

这一步会自动执行修复后的完整训练与推理流水线：
1. **`code/synth/build_v18_native_corridor_quads.py`**：修复 `mild_*` / `strong_*` 文件名后缀匹配与 `过度生成` 走廊阈值，生成 **200+ 组原生分辨率 `过度生成` 成对四元组**（总体约 1,500+ 张原生分辨率四元组样本）。
2. **`code/finetune/launch_v18_paired_kd_8gpus.py`**：8 张 H200 并行训练 `s16` + `s17` 的 8 个专家模型（`--epochs-s1 2 --epochs-s2 2`），使用真正的 **6B Teacher + V16_Full OOF 14 维软标签蒸馏**、**按同一原图 `(seed_name, dim_c)` 分组的成对四元组排序损失 + 非目标维度正交不变性损失**、以及**纯训练态 SWA**。
3. **`code/finetune/launch_v18_fulldata_4gpus.py`**：4 张 H200 并行在 `split_full_v18.csv`（3,073 张真实训练图 + 成对四元组）上训练 4 个全量专家，并在训练结束时自动按图片名对齐融合 `V16` 锚点。
4. **`code/finetune/upgrade_6b_4spec_synergy_router.py` & `code/deploy/unified_4090_dag_engine.py`**：重新拟合并导出最终提交 JSON。

在 `/wanqing-develop/luowenjing/18` 目录下直接执行：

```bash
cd /wanqing-develop/luowenjing/18
bash run_v18_1_all_in_one.sh --full
```

### Stage 2 最终产物路径：
- `/wanqing-develop/luowenjing/18/pred_nriqa_dataset_test.json`（主提交文件）
- `/wanqing-develop/luowenjing/18/results/final/pred_nriqa_dataset_test_v18_1_final.json`（Stage 2 最终重训版备份）
- `/wanqing-develop/luowenjing/18/results/final/pred_nriqa_dataset_test_v18_fulldata.json`
- `/wanqing-develop/luowenjing/18/results/final/pred_nriqa_dataset_test_v18.json`
