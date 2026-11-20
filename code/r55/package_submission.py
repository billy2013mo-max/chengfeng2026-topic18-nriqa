#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build clean, official-checklist-organized lite and full submission ZIP archives for 命题18 (`resident_v8_boundary` + `deconf_v6_boundary`, Online 54.77 / 65).

Archive directory structure matches the 6 official deliverable categories in 命题文档第5节《提交要求》:
  README.md
  01_预测结果文件_核心交付物/
  02_可执行代码/
  03_模型产物/
  04_技术方案文档/
  05_自评测报告/
  06_AI使用说明/
"""
import argparse
import ast
import os
import py_compile
import shutil
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_ROOT = os.path.join(REPO, "code")

# (repo_relative_path, archive_relative_path)
DELIVERABLE_MAP = [
    ("README.md", "README.md"),
    # 01. 预测结果文件【核心交付物】（终评 644 图与公开打榜 150 图主推 v8_boundary + deconf_v6_boundary 预测 JSON + 主观归因/极目视频旁路产物）
    (
        "results/final/pred_nriqa_dataset_final_644_r55_resident_v8_boundary_deconf_v6_boundary.json",
        "01_预测结果文件_核心交付物/pred_nriqa_dataset_final_644_r55_resident_v8_boundary_deconf_v6_boundary.json",
    ),
    (
        "results/final/pred_nriqa_dataset_test_r55_resident_v8_boundary_deconf_v6_boundary.json",
        "01_预测结果文件_核心交付物/pred_nriqa_dataset_test_r55_resident_v8_boundary_deconf_v6_boundary.json",
    ),
    (
        "results/r55/final/final644_attribution_v8.json",
        "01_预测结果文件_核心交付物/sidecars_语义归因与极目视频载荷/final644_attribution_v8.json",
    ),
    (
        "results/r55/final/final644_attribution_v8_sheet.jpg",
        "01_预测结果文件_核心交付物/sidecars_语义归因与极目视频载荷/final644_attribution_v8_sheet.jpg",
    ),
    (
        "results/r55/final/final644_tat_jimu_v8.json",
        "01_预测结果文件_核心交付物/sidecars_语义归因与极目视频载荷/final644_tat_jimu_v8.json",
    ),
    (
        "results/r55/final/final644_video_summary_v8.json",
        "01_预测结果文件_核心交付物/sidecars_语义归因与极目视频载荷/final644_video_summary_v8.json",
    ),
    (
        "results/r55/final/test150_attribution_v8.json",
        "01_预测结果文件_核心交付物/sidecars_语义归因与极目视频载荷/test150_attribution_v8.json",
    ),
    (
        "results/r55/final/test150_attribution_v8_sheet.jpg",
        "01_预测结果文件_核心交付物/sidecars_语义归因与极目视频载荷/test150_attribution_v8_sheet.jpg",
    ),
    (
        "results/r55/final/test150_tat_jimu_v8.json",
        "01_预测结果文件_核心交付物/sidecars_语义归因与极目视频载荷/test150_tat_jimu_v8.json",
    ),
    (
        "results/r55/final/test150_video_summary_v8.json",
        "01_预测结果文件_核心交付物/sidecars_语义归因与极目视频载荷/test150_video_summary_v8.json",
    ),
    # 02. 可执行代码（入口脚本、容器镜像与依赖锁）
    ("run_final_infer.sh", "02_可执行代码/run_final_infer.sh"),
    ("requirements_infer.txt", "02_可执行代码/requirements_infer.txt"),
    ("Dockerfile", "02_可执行代码/Dockerfile"),
    # 03. 模型产物（主推 resident_v8_boundary + deconf_v6_boundary 解码器、演进对照解码器、备选纯视觉轻量档 novlm 解码器、离线洁净基线、视频聚合器与单卡实测报告）
    (
        "results/r55/final/resident_v8_boundary_decoder_deconf_v6_boundary.pkl",
        "03_模型产物/decoders/resident_v8_boundary_decoder_deconf_v6_boundary.pkl",
    ),
    (
        "results/r55/final/resident_v8_boundary_decoder_deconf_v6_boundary.json",
        "03_模型产物/decoders/resident_v8_boundary_decoder_deconf_v6_boundary.json",
    ),
    (
        "results/r55/final/resident_v8_boundary_clean_footprint.npz",
        "03_模型产物/decoders/resident_v8_boundary_clean_footprint.npz",
    ),
    (
        "results/r55/final/resident_v8_boundary_video_aggregator.pkl",
        "03_模型产物/decoders/resident_v8_boundary_video_aggregator.pkl",
    ),
    (
        "results/r55/final/resident_v8_boundary_bench_half.json",
        "03_模型产物/decoders/resident_v8_boundary_bench_half.json",
    ),
    (
        "results/r55/final/resident_v8_boundary_novlm_decoder_deconf_v6_boundary.pkl",
        "03_模型产物/decoders/resident_v8_boundary_novlm_decoder_deconf_v6_boundary.pkl",
    ),
    (
        "results/r55/final/resident_v8_boundary_novlm_decoder_deconf_v6_boundary.json",
        "03_模型产物/decoders/resident_v8_boundary_novlm_decoder_deconf_v6_boundary.json",
    ),
    (
        "results/r55/final/resident_v8_boundary_novlm_bench_half.json",
        "03_模型产物/decoders/resident_v8_boundary_novlm_bench_half.json",
    ),
    (
        "results/r55/final/resident_v7_decoder_deconf_v5_boundary.pkl",
        "03_模型产物/decoders/resident_v7_decoder_deconf_v5_boundary.pkl",
    ),
    (
        "results/r55/final/resident_v7_decoder_deconf_v4_tile.pkl",
        "03_模型产物/decoders/resident_v7_decoder_deconf_v4_tile.pkl",
    ),
    (
        "results/r55/final/resident_v7_decoder_deconf_v4_tile.json",
        "03_模型产物/decoders/resident_v7_decoder_deconf_v4_tile.json",
    ),
    (
        "results/r55/final/resident_v7_decoder_deconf_v3.pkl",
        "03_模型产物/decoders/resident_v7_decoder_deconf_v3.pkl",
    ),
    (
        "results/r55/final/resident_v7_decoder_deconf_v2.pkl",
        "03_模型产物/decoders/resident_v7_decoder_deconf_v2.pkl",
    ),
    (
        "results/r55/final/resident_v7_decoder_deconf_v2.json",
        "03_模型产物/decoders/resident_v7_decoder_deconf_v2.json",
    ),
    (
        "results/r55/final/resident_v7_decoder_routed_gate_eu.pkl",
        "03_模型产物/decoders/resident_v7_decoder_routed_gate_eu.pkl",
    ),
    (
        "results/r55/final/resident_v7_decoder_routed_gate_eu.json",
        "03_模型产物/decoders/resident_v7_decoder_routed_gate_eu.json",
    ),
    (
        "results/r55/final/resident_v7_decoder_routed_eu.pkl",
        "03_模型产物/decoders/resident_v7_decoder_routed_eu.pkl",
    ),
    (
        "results/r55/final/resident_v7_decoder.pkl",
        "03_模型产物/decoders/resident_v7_decoder.pkl",
    ),
    (
        "results/r55/final/resident_v7_clean_footprint.npz",
        "03_模型产物/decoders/resident_v7_clean_footprint.npz",
    ),
    (
        "results/r55/final/resident_v7_video_aggregator.pkl",
        "03_模型产物/decoders/resident_v7_video_aggregator.pkl",
    ),
    (
        "results/r55/final/resident_v7_bench_half_deconf_v4_tile.json",
        "03_模型产物/decoders/resident_v7_bench_half_deconf_v4_tile.json",
    ),
    (
        "results/r55/final/resident_v7_bench_half_deconf_v2.json",
        "03_模型产物/decoders/resident_v7_bench_half_empty_gpu_baseline.json",
    ),
    (
        "results/r55/final/resident_v7_novlm_decoder_routed_gate_eu.pkl",
        "03_模型产物/decoders/resident_v7_novlm_decoder_routed_gate_eu.pkl",
    ),
    (
        "results/r55/final/resident_v7_novlm_decoder_routed_gate_eu.json",
        "03_模型产物/decoders/resident_v7_novlm_decoder_routed_gate_eu.json",
    ),
    (
        "results/r55/final/resident_v7_novlm_clean_footprint.npz",
        "03_模型产物/decoders/resident_v7_novlm_clean_footprint.npz",
    ),
    (
        "results/r55/final/resident_v7_novlm_video_aggregator.pkl",
        "03_模型产物/decoders/resident_v7_novlm_video_aggregator.pkl",
    ),
    (
        "results/r55/final/resident_v7_novlm_bench_half_share.json",
        "03_模型产物/decoders/resident_v7_novlm_bench_half_share.json",
    ),
    # 04. 技术方案文档
    ("docs/TECHNICAL_REPORT.md", "04_技术方案文档/TECHNICAL_REPORT.md"),
    ("docs/END_TO_END_TECHNICAL_REPORT.md", "04_技术方案文档/END_TO_END_TECHNICAL_REPORT.md"),
    ("docs/RESIDENT_V7_PIPELINE_ARCHITECTURE.md", "04_技术方案文档/RESIDENT_V7_PIPELINE_ARCHITECTURE.md"),
    ("docs/FINAL_DEFENSE_HIGHLIGHTS.md", "04_技术方案文档/FINAL_DEFENSE_HIGHLIGHTS.md"),
    ("docs/SERVING_AND_TAT_JIMU_ADAPTER.md", "04_技术方案文档/SERVING_AND_TAT_JIMU_ADAPTER.md"),
    ("docs/SUBMISSION_CHECKLIST.md", "04_技术方案文档/SUBMISSION_CHECKLIST.md"),
    ("docs/decisions/ADR-001-shared-trunk-specialists-over-heavy-vlms.md", "04_技术方案文档/decisions/ADR-001-shared-trunk-specialists-over-heavy-vlms.md"),
    ("docs/decisions/ADR-002-dual-adapter-vlm-and-weight-sharing.md", "04_技术方案文档/decisions/ADR-002-dual-adapter-vlm-and-weight-sharing.md"),
    ("docs/decisions/ADR-003-pre-lro-fusion-and-rank1-calibration.md", "04_技术方案文档/decisions/ADR-003-pre-lro-fusion-and-rank1-calibration.md"),
    ("docs/decisions/ADR-004-bidirectional-clean-lock-and-dim-isolation.md", "04_技术方案文档/decisions/ADR-004-bidirectional-clean-lock-and-dim-isolation.md"),
    ("docs/decisions/ADR-005-deterministic-counterfactual-attribution.md", "04_技术方案文档/decisions/ADR-005-deterministic-counterfactual-attribution.md"),
    # 05. 自评测报告与各项定量审计 JSON
    ("docs/SELF_EVALUATION_REPORT.md", "05_自评测报告/SELF_EVALUATION_REPORT.md"),
    ("results/r55/final/resident_v8_boundary_oof_deconf_v6_boundary_summary.json", "05_自评测报告/resident_v8_boundary_oof_deconf_v6_boundary_summary.json"),
    ("results/r55/final/resident_v8_boundary_video_oof_audit.json", "05_自评测报告/resident_v8_boundary_video_oof_audit.json"),
    ("results/r55/final/resident_v8_boundary_cost_pareto_oof.json", "05_自评测报告/resident_v8_boundary_cost_pareto_oof.json"),
    ("results/r55/final/resident_v8_boundary_deploy_env_check.json", "05_自评测报告/resident_v8_boundary_deploy_env_check.json"),
    ("results/r55/final/resident_v8_boundary_attribution_audit.json", "05_自评测报告/resident_v8_boundary_attribution_audit.json"),
    ("results/r55/final/resident_v8_boundary_http_service_audit.json", "05_自评测报告/resident_v8_boundary_http_service_audit.json"),
    ("results/r55/final/resident_v7_oof_deconf_v4_tile_summary.json", "05_自评测报告/resident_v7_oof_deconf_v4_tile_summary.json"),
    # 06. AI 使用说明
    ("docs/AI_USAGE_STATEMENT.md", "06_AI使用说明/AI_USAGE_STATEMENT.md"),
]

PYTHON_SEEDS = [
    "code/r55/final_engine.py",
    "code/r55/validate_submission.py",
    "code/r55/serve_engine.py",
    "code/r55/check_deploy_env.py",
    "code/r55/package_submission.py",
    "code/r55/measured_evidence.py",
    "code/r55/subjective_engineering.py",
    "code/r55/defect_attribution_map.py",
    "code/r55/routed_eu_decoder.py",
    "code/r55/deconfound_calibrator.py",
    "code/r55/upgrade_os_og_bokeh_deconf.py",
    "code/r55/upgrade_deconf_v3_cleanlocked.py",
    "code/r55/upgrade_deconf_v4_tile.py",
    "code/r55/upgrade_deconf_v5_boundary.py",
    "code/r55/upgrade_deconf_v6_boundary.py",
    "code/r55/build_v8_boundary_novlm_decoder.py",
    "code/r55/dual_vlm_shared_4090.py",
    "code/r55/train_boundary_specialist.py",
    "code/r55/train_qwen3vl_hires_roi.py",
    "code/r55/run_v8_boundary_fresh_infer.py",
    "code/r55/build_v8_delivery_sidecars_and_audits.py",
    "code/r55/audit_v8_http_service.py",
    "code/r55/ntg_tile_peak_probe.py",
    "code/r55/spatial_salience_probe.py",
    "code/r55/native_tiles.py",
    "code/r55/vlm_c233397_qy8.py",
    "code/r55/train_p12_specialist.py",
    "code/r55/train_testbed.py",
    "code/r55/build_glitch.py",
    "code/r55/build_rank_tuples.py",
    "code/r55/build_v16_qy_splits.py",
    "code/qingyu/train_tier1_5backbones_8fold.py",
    "code/finetune/train_v16_microscope_specialist.py",
    "code/phoenix/train_pillar3_hires_physics.py",
]

CKPT_MAP = [
    (
        "results/r55/fullfit_v3_tier1/cradio_v4_h/fold_0_best.pt",
        "03_模型产物/checkpoints/cradio_v3/fold_0_best.pt",
    ),
    (
        "results/r55/fullfit_v3_tier1/pe_core_large/fold_0_best.pt",
        "03_模型产物/checkpoints/pe_v3/fold_0_best.pt",
    ),
    (
        "results/next55/towers/r55_fullglitch_s2026/pe_core_large/fold_0_best.pt",
        "03_模型产物/checkpoints/peg/fold_0_best.pt",
    ),
    (
        "results/r55/fullfit_v3_tier1/siglip2_so400m_naflex/fold_0_best.pt",
        "03_模型产物/checkpoints/sig_v3/fold_0_best.pt",
    ),
    (
        "results/r55/fullfit_v7_bkseg1/pe_core_large/fold_0_best.pt",
        "03_模型产物/checkpoints/pe_bkseg1/fold_0_best.pt",
    ),
    (
        "results/r55/fullfit_v5_ntg/pe_core_large/fold_0_best.pt",
        "03_模型产物/checkpoints/ntg_pe/fold_0_best.pt",
    ),
    (
        "results/r55/fullfit_v7_ntg_convnext/convnextv2_base/fold_0_best.pt",
        "03_模型产物/checkpoints/ntg_convnext/fold_0_best.pt",
    ),
    (
        "results/r55/fullfit_v3_p12/pe_core_large/fold_0_best.pt",
        "03_模型产物/checkpoints/p12_pe/fold_0_best.pt",
    ),
    (
        "results/next55/fullfit/v16_full8/V16_Full_F0_AllGrade_Microscope_swa.pt",
        "03_模型产物/checkpoints/v16/V16_Full_F0_AllGrade_Microscope_swa.pt",
    ),
    (
        "results/r55/p3_qy_honest_fullfit/pillar3_fulldata_swa.pt",
        "03_模型产物/checkpoints/p3_qy/pillar3_fulldata_swa.pt",
    ),
    (
        "results/r55/vlm_qy8/ckpts/p1_c233397_s2026_fullfit_swa.pt",
        "03_模型产物/checkpoints/p1_qy/p1_c233397_s2026_fullfit_swa.pt",
    ),
    (
        "results/r55/boundary_v2/v2_pe_ntg/fullfit_best.pt",
        "03_模型产物/checkpoints/bndry_pe/fullfit_best.pt",
    ),
    (
        "results/r55/boundary_v2/v2_dual_pe_cradio/fullfit_best.pt",
        "03_模型产物/checkpoints/bndry_dual/fullfit_best.pt",
    ),
    (
        "results/r55/qwen3vl_hires_pilot/ckpts/qwen3vl_hires_s2026_fullfit_swa.pt",
        "03_模型产物/checkpoints/qwen_hr/qwen3vl_hires_s2026_fullfit_swa.pt",
    ),
]

MODEL_MANIFEST_TEXT = """# 模型产物与开源基座版本说明 (`resident_v8_boundary` + `deconf_v6_boundary`)

本交付集为最终定稿主推架构 **共享主干 + 双适配器 VLM 14 专精成员集成 `resident_v8_boundary` + Pre-LRO 边界零稀释校准器 `deconf_v6_boundary`**（公开打榜集实测 **`54.77 / 65`**：正向缺陷 `17.46/25`、P0 识别 `17.31/20`、Clean `20.00/20` 满分；8 折零泄漏 Nested OOF **`orig = 52.665 / all = 51.958`**）：
- **激活去重总参数量**：**`13.262B <= 30B`**（纯视觉轻量档 `resident_v8_boundary_novlm` 为 **`8.910B`**、线上实测 **`54.52 / 65`**；`resident_v7_novlm` 为 **`7.569B`**）；
- **单卡 24GB 显存实测（开启 `--half --share-weights --embed-on-host`，内置纯 PyTorch `INT8-g128` 量化未共享 `v16[pecoreg]` 与比特精确激活分块）**：
  - 冻结底层块 `torch.equal` 无损显存去重节省 **`5.81 GB`**（新增 `bndry_pe` 仅增 `+0.28 GB`、`bndry_dual` 仅增 `+0.63 GB`）；
  - **`DualAdapterQwen3VL4090` 共享 4-bit NF4 底座双适配器**：单一 `Qwen3-VL-8B` 4-bit 底座 + 主机内存词表（省 `1.16 GB`）同时挂载 `p1_qy`（DoRA r=24）与 `qwen_hr`（LoRA r=64），双 8B VLM 合计常驻仅 **`5.62 GB`**（较双独立实例节省 `3.40 GB`），`argmax` 100.0% 一致；
  - **纯 PyTorch 对称 `INT8 group-128` 权重量化**（仅作用于 `v16` 独有未共享 1.88B `PE-Core-G14`，零外部依赖、100% BF16 激活）：再省 **`1.72 GiB`**，配合 `ntg_convnext (tile_chunk=4)` / `cradio_v3` 比特精确激活分块，全量 `test_150` 严格 **`0 / 900` 格差异**；
  - PyTorch 常驻显存 **`15.78 GB`**、推理峰值 **`16.95 GB`**、预留 **`17.19 GiB`**；
  - 设备级 `nvidia-smi` 加载峰值 **`20,460 MiB`（`19.98 GiB`）**、推理峰值 **`18,930 MiB`（`18.49 GiB < 24 GiB`，单张 RTX 4090 推理余量达 `~5.50 GiB`）**；
- **单图端到端耗时**：全量 `test_150` 单卡实测均值 **`3.03s`**、P95 **`4.23s`**、最大 **`6.93s <= 15s`**（12 图带成员同步压测均值 **`3.66s`** / 中位数 **`3.41s`** / 最大 **`7.23s`**；纯视觉轻量档 `resident_v8_boundary_novlm` 线上 **`54.52 / 65`**、`nvidia-smi` 峰值 **`12.75 GiB`**、中位 **`2.39s`**）。

---

## 1. 14 个常驻专精成员权重与开源基座版本对照表

| 成员代号 | 家族 / 专精职责 | 开源预训练基座及本地目录 (`models/`) | 归档微调权重路径 (`03_模型产物/checkpoints/`) |
| :--- | :--- | :--- | :--- |
| `cradio_v3` | Tier-1 原生比例多教师蒸馏塔 | `nvidia/C-RADIOv4-H` (`models/c-radio-v4-h`) | `checkpoints/cradio_v3/fold_0_best.pt` |
| `pe_v3` | Tier-1 通用多级感知与归因主塔 | `PE-Core-L14-336` (`models/pe-core-l14`) | `checkpoints/pe_v3/fold_0_best.pt` |
| `peg` | 未知源码流/宏块花屏特种雷达 (`peg>=0.90` 救援) | `PE-Core-L14-336` (`models/pe-core-l14`, 共享冻结层) | `checkpoints/peg/fold_0_best.pt` |
| `sig_v3` | 原生可变长宽比 NaFlex 视觉塔 | `google/siglip2-so400m-patch16-naflex` (`models/siglip2-naflex`) | `checkpoints/sig_v3/fold_0_best.pt` |
| `pe_bkseg1` | 抠图散景断层专精雷达 + 虚化空间定位算子 | `PE-Core-L14-336` (`models/pe-core-l14`, 共享冻结层) | `checkpoints/pe_bkseg1/fold_0_best.pt` |
| `ntg_pe` | 1:1 原生全覆盖平铺网格 ViT 塔 | `PE-Core-L14-336` (`models/pe-core-l14`, 共享冻结层) | `checkpoints/ntg_pe/fold_0_best.pt` |
| `ntg_convnext` | 1:1 原生全覆盖平铺网格纯卷积塔 | `timm/convnextv2_base.fcmae_ft_in22k_in1k` (`models/convnextv2-base`) | `checkpoints/ntg_convnext/fold_0_best.pt` |
| `p12_pe` | 语义 ROI（人像/文字/散景）跨家族知识蒸馏塔 | `PE-Core-L14-336` (`models/pe-core-l14`, 共享冻结层) | `checkpoints/p12_pe/fold_0_best.pt` |
| `bndry_pe` | **Stage 2B 边界专精塔（单干 1:1 残差 + 跨缺陷自注意力）** | `PE-Core-L14-336` (`models/pe-core-l14`, 共享冻结层 +0.28 GB) | `checkpoints/bndry_pe/fullfit_best.pt` |
| `bndry_dual` | **Stage 2B 双干联合边界专精塔（全库 `过度生成` 单模第 1）** | `PE-Core-L14` + `C-RADIOv4-H` (双干共享冻结层 +0.63 GB) | `checkpoints/bndry_dual/fullfit_best.pt` |
| `v16` | 8 视角 1:1 边缘-纹理对比残差显微镜 + 60D 物理卡尺 | 5 子骨干：`PE-Core-G14` + `DINOv3-ViT-L/16` + `C-RADIOv4-H` + `EVA02-Base-448` + `ConvNeXtV2-Base` | `checkpoints/v16/V16_Full_F0_AllGrade_Microscope_swa.pt` |
| `p3_qy` | 四阶段超列 + 9ch 双极小波 + 8ch CoC 散焦场专家 | `DINOv3-ViT-L/16` + `C-RADIOv4-H` + `ConvNeXtV2-Base` + `PE-Core-L14` + `Depth-Anything-V2-Small` | `checkpoints/p3_qy/pillar3_fulldata_swa.pt` |
| `p1_qy` | 共享底座 VLM 适配器 1（DoRA `r=24`, 784² + 2×384 ROI） | `Qwen/Qwen3-VL-8B-Instruct` (`models/qwen3vl-8b`, 4-bit NF4 + Host 词表) | `checkpoints/p1_qy/p1_c233397_s2026_fullfit_swa.pt` |
| `qwen_hr` | **共享底座 VLM 适配器 2（LoRA `r=64`, 1344² + 3×448 语义 ROI）** | 与 `p1_qy` **共享同一 4-bit NF4 `Qwen3-VL-8B` 底座**（双 VLM 共 5.62 GB） | `checkpoints/qwen_hr/qwen3vl_hires_s2026_fullfit_swa.pt` |

> **基座目录解析说明**：所有开源基座默认从 `<仓库根>/models/`（或解压目录下的 `03_模型产物/base_models/`、环境变量 `R55_BASE_MODELS_DIR`）离线读取，无需访问公网或预热 HuggingFace 缓存。

---

## 2. 校准解码器、归因洁净基线与视频聚合器（`03_模型产物/decoders/`）

- **主推解码器**：`decoders/resident_v8_boundary_decoder_deconf_v6_boundary.pkl`（配套完整校准参数与 8-Fold OOF 审计说明 `resident_v8_boundary_decoder_deconf_v6_boundary.json`，同时附带 `deconf_v5_boundary / deconf_v4_tile / deconf_v3 / deconf_v2 / routed_gate_eu` 完整演进对照包）；
- **纯视觉轻量备选档解码器**：`decoders/resident_v7_novlm_decoder_routed_gate_eu.pkl`（通过 `--set resident_v7_novlm --decoder routed_gate_eu` 启用）；
- **离线实测归因基线**：`decoders/resident_v8_boundary_clean_footprint.npz`（75.2 KiB，由训练集洁净帧一次性离线构建，推理期零训练数据读取）；
- **多帧视频聚合器**：`decoders/resident_v8_boundary_video_aggregator.pkl`（`EffectiveFrameORVideoAggregator`）；
- **单卡断网延迟与显存实测报告**：`decoders/resident_v8_boundary_bench_half.json`、`decoders/resident_v7_bench_half_deconf_v4_tile.json`、`decoders/resident_v7_novlm_bench_half_share.json`。
"""


def trace_python_closure(seeds):
    mod_map = {}
    for root, _, files in os.walk(CODE_ROOT):
        if "__pycache__" in root:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, REPO)
            pkg_rel = os.path.relpath(full, CODE_ROOT)[:-3].replace(os.sep, ".")
            mod_map[pkg_rel] = rel
            mod_map[fn[:-3]] = rel
            if fn == "__init__.py":
                mod_map[os.path.dirname(pkg_rel)] = rel

    visited = set()
    queue = list(seeds)
    while queue:
        rel = queue.pop(0)
        full = os.path.join(REPO, rel)
        if rel in visited or not os.path.exists(full):
            continue
        visited.add(rel)
        d = os.path.dirname(rel)
        while d and d != "code":
            init_rel = os.path.join(d, "__init__.py")
            if os.path.exists(os.path.join(REPO, init_rel)) and init_rel not in visited:
                queue.append(init_rel)
            d = os.path.dirname(d)

        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read(), filename=rel)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for nm in names:
                parts = nm.split(".")
                for k in range(len(parts), 0, -1):
                    cand = ".".join(parts[:k])
                    if cand in mod_map and mod_map[cand] not in visited:
                        queue.append(mod_map[cand])
    return sorted(visited)


def collect_pydeps_pairs():
    pydeps_dir = os.path.join(REPO, "third_party/pydeps")
    pairs = []
    if not os.path.isdir(pydeps_dir):
        return pairs
    for root, _, files in os.walk(pydeps_dir):
        if "__pycache__" in root:
            continue
        for fn in files:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, REPO)
            pairs.append((rel, os.path.join("02_可执行代码", rel)))
    return sorted(pairs)


def write_zip(out_path, deflated_pairs, stored_pairs=()):
    seen_arc = set()
    with zipfile.ZipFile(out_path, "w", allowZip64=True) as zf:
        zf.writestr(
            "03_模型产物/README_模型与基座版本说明.md",
            MODEL_MANIFEST_TEXT.encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        seen_arc.add("03_模型产物/README_模型与基座版本说明.md")
        for src_rel, arc_name in deflated_pairs:
            if arc_name in seen_arc:
                continue
            seen_arc.add(arc_name)
            full = os.path.join(REPO, src_rel)
            if not os.path.exists(full):
                raise FileNotFoundError(f"Missing file: {full}")
            zf.write(full, arcname=arc_name, compress_type=zipfile.ZIP_DEFLATED)
        for src_rel, arc_name in stored_pairs:
            if arc_name in seen_arc:
                continue
            seen_arc.add(arc_name)
            full = os.path.join(REPO, src_rel)
            if not os.path.exists(full):
                raise FileNotFoundError(f"Missing checkpoint: {full}")
            zf.write(full, arcname=arc_name, compress_type=zipfile.ZIP_STORED)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    total_files = len(seen_arc)
    print(f"[OK] {out_path} ({size_mb:.2f} MB, {total_files} files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20261119")
    ap.add_argument("--lite-only", action="store_true", help="skip the multi-GB full archive (checkpoints)")
    a = ap.parse_args()
    free_gb = shutil.disk_usage(REPO).free / (1024 ** 3)
    print(f"Free disk space: {free_gb:.1f} GB")
    if free_gb < 15.0:
        raise RuntimeError(f"Insufficient disk space ({free_gb:.1f} GB < 15 GB)")

    for src_rel, _ in DELIVERABLE_MAP:
        if not os.path.exists(os.path.join(REPO, src_rel)):
            raise FileNotFoundError(f"Missing required file: {src_rel}")

    py_files = trace_python_closure(PYTHON_SEEDS)
    for rel in py_files:
        py_compile.compile(os.path.join(REPO, rel), doraise=True)
    print(f"[OK] Verified {len(py_files)} Python files (py_compile passed)")

    code_pairs = [(rel, os.path.join("02_可执行代码", rel)) for rel in py_files]
    lite_pairs = DELIVERABLE_MAP + code_pairs

    lite_zip = os.path.join(REPO, f"topic18_submission_lite_{a.tag}.zip")
    full_zip = os.path.join(REPO, f"topic18_submission_full_{a.tag}.zip")

    write_zip(lite_zip, lite_pairs)
    if a.lite_only:
        return
    pydeps_pairs = collect_pydeps_pairs()
    write_zip(full_zip, lite_pairs + pydeps_pairs, CKPT_MAP)


if __name__ == "__main__":
    main()
