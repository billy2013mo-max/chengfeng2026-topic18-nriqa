#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build clean, official-checklist-organized lite and full submission ZIP archives for 命题18 (`resident_v6_core`).

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
    # 01. 预测结果文件【核心交付物】
    (
        "results/final/pred_nriqa_dataset_test_r55_resident_v6_core_routed_gate_eu.json",
        "01_预测结果文件_核心交付物/pred_nriqa_dataset_test_r55_resident_v6_core_routed_gate_eu.json",
    ),
    # 02. 可执行代码（入口脚本）
    ("run_final_infer.sh", "02_可执行代码/run_final_infer.sh"),
    ("requirements_infer.txt", "02_可执行代码/requirements_infer.txt"),
    # 03. 模型产物（解码器与基准报告）
    (
        "results/r55/final/resident_v6_core_decoder_routed_gate_eu.pkl",
        "03_模型产物/decoders/resident_v6_core_decoder_routed_gate_eu.pkl",
    ),
    (
        "results/r55/final/resident_v6_core_decoder_routed_gate_eu.json",
        "03_模型产物/decoders/resident_v6_core_decoder_routed_gate_eu.json",
    ),
    (
        "results/r55/final/resident_v6_core_decoder_routed_eu.pkl",
        "03_模型产物/decoders/resident_v6_core_decoder_routed_eu.pkl",
    ),
    (
        "results/r55/final/resident_v6_core_decoder_routed_eu.json",
        "03_模型产物/decoders/resident_v6_core_decoder_routed_eu.json",
    ),
    (
        "results/r55/final/resident_v6_core_decoder.pkl",
        "03_模型产物/decoders/resident_v6_core_decoder.pkl",
    ),
    (
        "results/r55/final/resident_v6_core_bench_half.json",
        "03_模型产物/decoders/resident_v6_core_bench_half.json",
    ),
    (
        "results/r55/final/resident_v6_core_bench_half_chunk2.json",
        "03_模型产物/decoders/resident_v6_core_bench_half_chunk2.json",
    ),
    # precomputed attribution clean footprint + clean physics percentiles (inference never reads training data)
    (
        "results/r55/final/resident_v6_core_clean_footprint.npz",
        "03_模型产物/decoders/resident_v6_core_clean_footprint.npz",
    ),
    (
        "results/r55/final/resident_v6_core_video_aggregator.pkl",
        "03_模型产物/decoders/resident_v6_core_video_aggregator.pkl",
    ),
    # 04. 技术方案文档
    ("docs/TECHNICAL_REPORT.md", "04_技术方案文档/TECHNICAL_REPORT.md"),
    ("docs/RESIDENT_V6_PIPELINE_ARCHITECTURE.md", "04_技术方案文档/RESIDENT_V6_PIPELINE_ARCHITECTURE.md"),
    ("docs/SUBMISSION_CHECKLIST.md", "04_技术方案文档/SUBMISSION_CHECKLIST.md"),
    ("docs/SERVING_AND_TAT_JIMU_ADAPTER.md", "04_技术方案文档/SERVING_AND_TAT_JIMU_ADAPTER.md"),
    # 05. 自评测报告
    ("docs/SELF_EVALUATION_REPORT.md", "05_自评测报告/SELF_EVALUATION_REPORT.md"),
    ("results/r55/vlm_qy8/prune_v6_audit.json", "05_自评测报告/prune_v6_audit.json"),
    # 06. AI 使用说明
    ("docs/AI_USAGE_STATEMENT.md", "06_AI使用说明/AI_USAGE_STATEMENT.md"),
]

PYTHON_SEEDS = [
    "code/r55/final_engine.py",
    "code/r55/validate_submission.py",
    "code/r55/serve_engine.py",
    "code/r55/check_deploy_env.py",
    "code/r55/measured_evidence.py",
    "code/r55/subjective_engineering.py",
    "code/r55/defect_attribution_map.py",
    "code/r55/routed_eu_decoder.py",
    "code/r55/eval_v6_gates.py",
    "code/r55/prune_v6_audit.py",
    "code/r55/vlm_c233397_qy8.py",
    "code/r55/train_p12_specialist.py",
    "code/r55/build_glitch.py",
    "code/r55/build_v16_qy_splits.py",
    "code/qingyu/train_tier1_5backbones_8fold.py",
    "code/next55/train_v16_microscope_specialist.py",
]

CKPT_MAP = [
    (
        "results/r55/fullfit_v3_tier1/cradio_v4_h/fold_0_best.pt",
        "03_模型产物/checkpoints/cradio_v3/fold_0_best.pt",
    ),
    (
        "results/next55/towers/r55_fullglitch_s2026/pe_core_large/fold_0_best.pt",
        "03_模型产物/checkpoints/peg/fold_0_best.pt",
    ),
    (
        "results/r55/fullfit_v5_ntg/pe_core_large/fold_0_best.pt",
        "03_模型产物/checkpoints/ntg_pe/fold_0_best.pt",
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
        "results/r55/vlm_qy8/ckpts/p1_c233397_s2026_fullfit_swa.pt",
        "03_模型产物/checkpoints/p1_qy/p1_c233397_s2026_fullfit_swa.pt",
    ),
]

MODEL_MANIFEST_TEXT = """# 模型产物与基座版本说明 (`resident_v6_core`)

本交付集为 **6 成员收敛架构 `resident_v6_core`**（激活总参数量 **9.656B <= 30B**，单卡 4090 / H200 `--half` 常驻显存 **15.68 GB**，推理峰值 **17.64 GB**，`expandable_segments` 下 `nvidia-smi` 实测峰值 **20,290 MiB（推理）/ 20,688 MiB（含加载）<= 24 GiB**，单图均值 **2.07s（含归因 2.35s）<= 15s**）。

## 1. 6 个模型成员权重与基座说明

| 成员名 | 家族 / 角色 | 基座模型及版本 | 权重路径（`checkpoints/`） |
| :--- | :--- | :--- | :--- |
| `cradio_v3` | Tier-1 全局视觉塔 | `c-radio-v4-h` | `checkpoints/cradio_v3/fold_0_best.pt` |
| `peg` | Tier-1 + 花屏专项增强（`peg>=0.90` 救援） | `PE-Core-L14-336` (`pe_core_large`) | `checkpoints/peg/fold_0_best.pt` |
| `ntg_pe` | 原生分辨率 Tile+Global 塔 | `PE-Core-L14-336` (`pe_core_large`) | `checkpoints/ntg_pe/fold_0_best.pt` |
| `p12_pe` | 语义 ROI（人像/文字/虚化）+ 知识蒸馏塔 | `PE-Core-L14-336` (`pe_core_large`) | `checkpoints/p12_pe/fold_0_best.pt` |
| `v16` | 局部显微镜多视角专家 | `InternViT-6B-448px-V2_5`（多骨干拼接） | `checkpoints/v16/V16_Full_F0_AllGrade_Microscope_swa.pt` |
| `p1_qy` | 多模态 VLM 语义缺陷专家（DoRA `r=24`, 4-bit NF4） | `Qwen3-VL-8B-Instruct` | `checkpoints/p1_qy/p1_c233397_s2026_fullfit_swa.pt` |

## 2. 路由与期望效用解码器（`decoders/`）

- 主推解码器：`decoders/resident_v6_core_decoder_routed_gate_eu.pkl`（配套参数说明 `resident_v6_core_decoder_routed_gate_eu.json`，含 `temps`、`W`、`dirdiag_params`、`eu_lam_cl`、`eu_p0_scale`、`eu_st` 等）。
- 备选解码器：`resident_v6_core_decoder_routed_eu.pkl`、`resident_v6_core_decoder.pkl`。
- 40 图实测延迟与显存报告：`decoders/resident_v6_core_bench_half.json`。
- 20 图复测（含 `nvidia-smi` 峰值）：`decoders/resident_v6_core_bench_half_chunk2.json`。
- 归因离线洁净基线：`decoders/resident_v6_core_clean_footprint.npz`（训练集无问题帧，推理时不读取任何训练数据）；视频聚合器：`decoders/resident_v6_core_video_aggregator.pkl`。
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
    with zipfile.ZipFile(out_path, "w", allowZip64=True) as zf:
        zf.writestr(
            "03_模型产物/README_模型与基座版本说明.md",
            MODEL_MANIFEST_TEXT.encode("utf-8"),
            compress_type=zipfile.ZIP_DEFLATED,
        )
        for src_rel, arc_name in deflated_pairs:
            full = os.path.join(REPO, src_rel)
            if not os.path.exists(full):
                raise FileNotFoundError(f"Missing file: {full}")
            zf.write(full, arcname=arc_name, compress_type=zipfile.ZIP_DEFLATED)
        for src_rel, arc_name in stored_pairs:
            full = os.path.join(REPO, src_rel)
            if not os.path.exists(full):
                raise FileNotFoundError(f"Missing checkpoint: {full}")
            zf.write(full, arcname=arc_name, compress_type=zipfile.ZIP_STORED)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    total_files = 1 + len(deflated_pairs) + len(stored_pairs)
    print(f"[OK] {out_path} ({size_mb:.2f} MB, {total_files} files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20261116")
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
