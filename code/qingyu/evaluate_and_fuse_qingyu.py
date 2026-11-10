#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/evaluate_and_fuse_qingyu.py
=======================================
晴羽观感体系 — 支柱三 (4.4 & 4.5 节) + 支柱四 (5.3 节) 全链路三阶异构融合与闭式贝叶斯解码器:
1. 自动扫描并严格按图像名对齐全部可用纯真实 8 折 OOF 预测资产 (`N = 3,573`):
   - 第一阶高频/原生比例轻量主干 (`1.958B`):
     `convnextv2_base`, `pe_core_large`, `dinov3_vitl16`, `siglip2_so400m_naflex`, `cradio_v4_h`
   - 第二阶深层局部感知中坚 (`6.080B`):
     `internvit_6b`
   - 第三阶多图原生语义与联合决策双塔 (`17.220B`):
     `clef_flash_9b`, `qwen3_vl_8b`
2. 逐塔输出 8 折全量 OOF `Doc1` / `Doc2_Strict` / `ECE` / `Clean_P0_FPR` 审计报表。
3. 嵌套 8 折训练 `CenteredLogOddsCertaintyGatedStacker(C=0.10)` (向量温度缩放 + 中心化对数几率
   + 归一化负熵确定性门控 + 最大对数几率包络 + 36D 单图物理标量 + 全局无损头 `Q_clean`)。
4. 导出严格折外集成软教师 `results/qingyu/oof_ensemble_teacher.npz`，供 4.4 节池内跨模态互蒸馏直接调用。
5. 运行 30 步阻尼 Frank-Wolfe 闭式混淆矩阵雅可比迭代 + 三候选全局无损贝叶斯插件解码器
   (`DampedFrankWolfeBayesianDecoder(dfw_steps=30)`)，输出 14 通道诊断明细与 8 折均值±标准差。
6. 强制执行 `batch=1` 逐图独立推理与全批次矩阵推理等价性断言 (`0` 离散标签差异)，生成官方格式提交 JSON。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.config_and_paths import (
    DIMS,
    QINGYU_CKPT_DIR,
    QINGYU_FINAL_DIR,
    QINGYU_RESULTS_DIR,
    QINGYU_SPLITS_DIR,
    REPO_ROOT,
    enforce_deterministic_runtime,
    evaluate_qingyu_dual_metrics,
    format_official_submission,
    print_pool_budget_summary,
    setup_sys_path_and_cwd,
)
from qingyu.certainty_gated_stacker import (
    CenteredLogOddsCertaintyGatedStacker,
    compute_ece_6x4,
    p012_to_6x4_simplex,
)
from qingyu.dfw_bayesian_decoder import DampedFrankWolfeBayesianDecoder


CANONICAL_TOWER_SPECS: Tuple[Tuple[str, str], ...] = (
    ("convnextv2_base", "tier1_convnextv2_base_8fold_oof.npz"),
    ("pe_core_large", "tier1_pe_core_large_8fold_oof.npz"),
    ("dinov3_vitl16", "tier1_dinov3_vitl16_8fold_oof.npz"),
    ("siglip2_so400m_naflex", "tier1_siglip2_so400m_naflex_8fold_oof.npz"),
    ("cradio_v4_h", "tier1_cradio_v4_h_8fold_oof.npz"),
    ("internvit_6b", "tier2_internvit_6b_8fold_oof.npz"),
    ("clef_flash_9b", "tier3_clef_flash_9b_8fold_oof.npz"),
    ("qwen3_vl_8b", "tier3_qwen3_vl_8b_8fold_oof.npz"),
)


def load_canonical_8fold_table(split_csv: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """加载 `data/qingyu_splits/qingyu_8fold.csv` 作为唯一权威样本顺序、真实标签与折编号基准。"""
    full_p = split_csv if os.path.isabs(split_csv) else os.path.join(REPO_ROOT, split_csv)
    names: List[str] = []
    gold_list: List[List[int]] = []
    fold_list: List[int] = []
    with open(full_p, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            names.append(r["name"].strip())
            gold_list.append([int(r[c]) for c in DIMS])
            fold_list.append(int(r["fold"]))
    return names, np.asarray(gold_list, dtype=np.int64), np.asarray(fold_list, dtype=np.int64)


def _align_array_by_names(
    target_names: Sequence[str],
    src_names: Sequence[str],
    arr: np.ndarray,
    context: str,
) -> np.ndarray:
    pos = {str(nm): idx for idx, nm in enumerate(src_names)}
    missing = [nm for nm in target_names if str(nm) not in pos]
    if missing:
        raise KeyError(f"[{context}] 缺失 {len(missing)} 张样本 (示例: {missing[:3]})")
    indices = [pos[str(nm)] for nm in target_names]
    return arr[indices]


def load_phys_matrix_aligned(
    ordered_names: Sequence[str],
    dual_engine_npz: str,
) -> Optional[np.ndarray]:
    full_p = dual_engine_npz if os.path.isabs(dual_engine_npz) else os.path.join(REPO_ROOT, dual_engine_npz)
    if not os.path.isfile(full_p):
        return None
    d = np.load(full_p, allow_pickle=True)
    src_names = [str(x) for x in d["names"]]
    pos = {nm: i for i, nm in enumerate(src_names)}
    if not all(str(nm) in pos for nm in ordered_names):
        return None
    idx = [pos[str(nm)] for nm in ordered_names]
    return np.asarray(d["phys_vecs"][idx], dtype=np.float32)


def load_available_qingyu_towers(
    results_dir: Path,
    canonical_oof_names: Sequence[str],
    only_towers: Optional[Sequence[str]] = None,
) -> Tuple[
    List[str],
    List[np.ndarray],
    List[np.ndarray],
    Optional[List[str]],
    List[np.ndarray],
    Optional[List[str]],
    List[np.ndarray],
]:
    """加载并严格对齐所有已生成的 8 折 OOF 与测试集 `.npz` 概率资产 (绝不读取任何历史 legacy JSON 或 69 模型缓存)。"""
    allow = set(only_towers) if only_towers else None
    tower_names: List[str] = []
    oof_6x4_list: List[np.ndarray] = []
    oof_qc_list: List[np.ndarray] = []
    t150_names_ref: Optional[List[str]] = None
    t150_6x4_list: List[np.ndarray] = []
    extra_names_ref: Optional[List[str]] = None
    extra_6x4_list: List[np.ndarray] = []

    for key, fname in CANONICAL_TOWER_SPECS:
        if allow is not None and key not in allow:
            continue
        fpath = results_dir / fname
        if not fpath.is_file():
            continue
        d = np.load(str(fpath), allow_pickle=True)
        src_oof_names = [str(x) for x in d["oof_names"]]
        if "oof_probs_6x4" in d.files:
            raw_oof = np.asarray(d["oof_probs_6x4"], dtype=np.float32)
        else:
            raw_oof = p012_to_6x4_simplex(d["oof_p0"], d["oof_p1"], d["oof_p2"])
        oof_aligned = _align_array_by_names(canonical_oof_names, src_oof_names, raw_oof, key)
        qc_raw = (
            np.asarray(d["oof_qclean"], dtype=np.float32)
            if "oof_qclean" in d.files
            else np.prod(oof_aligned[:, :, 0], axis=1).astype(np.float32)
        )
        qc_aligned = (
            _align_array_by_names(canonical_oof_names, src_oof_names, qc_raw, f"{key}.qclean")
            if "oof_qclean" in d.files
            else qc_raw
        )

        # 对齐 test_150
        src_t150_names = [str(x) for x in d["t150_names"]]
        if t150_names_ref is None:
            t150_names_ref = src_t150_names
        raw_t150 = (
            np.asarray(d["t150_probs_6x4"], dtype=np.float32)
            if "t150_probs_6x4" in d.files
            else p012_to_6x4_simplex(d["t150_p0"], d["t150_p1"], d["t150_p2"])
        )
        t150_aligned = _align_array_by_names(t150_names_ref, src_t150_names, raw_t150, f"{key}.t150")

        tower_names.append(key)
        oof_6x4_list.append(oof_aligned)
        oof_qc_list.append(qc_aligned)
        t150_6x4_list.append(t150_aligned)

        if "extra_probs_6x4" in d.files and "extra_names" in d.files:
            src_ext_names = [str(x) for x in d["extra_names"]]
            if extra_names_ref is None:
                extra_names_ref = src_ext_names
            ext_aligned = _align_array_by_names(
                extra_names_ref,
                src_ext_names,
                np.asarray(d["extra_probs_6x4"], dtype=np.float32),
                f"{key}.extra",
            )
            extra_6x4_list.append(ext_aligned)

    if len(extra_6x4_list) != len(tower_names):
        extra_names_ref = None
        extra_6x4_list = []

    return (
        tower_names,
        oof_6x4_list,
        oof_qc_list,
        t150_names_ref,
        t150_6x4_list,
        extra_names_ref,
        extra_6x4_list,
    )


def verify_single_image_invariance(
    stacker: CenteredLogOddsCertaintyGatedStacker,
    decoder: DampedFrankWolfeBayesianDecoder,
    test_models_6x4: Sequence[np.ndarray],
    test_phys: Optional[np.ndarray],
    max_check: int = 150,
) -> Dict[str, Any]:
    """强制逐张单样本 (`batch=1`) 推理并比对全批次推理输出，验证零批次统计耦合 (`0` 离散标签差异)。"""
    batch_probs, batch_qc = stacker.predict_proba(test_models_6x4, phys_mat=test_phys)
    batch_preds = decoder.decode(batch_probs, batch_qc)

    n_check = min(len(batch_probs), max_check)
    single_probs_list = []
    single_preds_list = []
    for i in range(n_check):
        m_i = [pm[i : i + 1] for pm in test_models_6x4]
        ph_i = test_phys[i : i + 1] if test_phys is not None else None
        sp, sq = stacker.predict_proba(m_i, phys_mat=ph_i)
        sy = decoder.decode(sp, sq)
        single_probs_list.append(sp[0])
        single_preds_list.append(sy[0])

    single_probs = np.stack(single_probs_list, axis=0)
    single_preds = np.stack(single_preds_list, axis=0)
    max_prob_diff = float(np.max(np.abs(single_probs - batch_probs[:n_check])))
    label_mismatches = int(np.sum(single_preds != batch_preds[:n_check]))

    if max_prob_diff > 1e-5 or label_mismatches != 0:
        raise AssertionError(
            f"单图独立性硬校验失败! max_prob_diff={max_prob_diff:.3e}, label_mismatches={label_mismatches}"
        )
    return {
        "checked_samples": n_check,
        "max_abs_prob_diff": max_prob_diff,
        "label_mismatches": label_mismatches,
        "batch_probs": batch_probs,
        "batch_qclean": batch_qc,
        "batch_preds": batch_preds,
    }


def print_diagnostic_report(
    title: str,
    ev: Dict[str, Any],
    ece: float,
    fold_preds: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    print("=" * 108)
    print(f"=== {title} ===")
    print("=" * 108)
    print(
        f"  OOF Doc1        : {ev['doc1_total']:.3f} / 65.00 "
        f"(正向缺陷={ev['doc1_S_pos']:.2f}/25, P0识别={ev['doc1_S_p0']:.2f}/20, Clean={ev['doc1_S_clean']:.2f}/20)"
    )
    print(
        f"  OOF Doc2_Strict : {ev['doc2_strict_total']:.3f} / 65.00 "
        f"(正向缺陷={ev['doc2_S_pos']:.2f}/25, P0识别={ev['doc2_S_p0']:.2f}/20, Clean2={ev['doc2_Clean2']:.2f}/20)"
    )
    print(
        f"  Calibration     : ECE={ece:.4f} | Clean_P0_FPR={ev['clean_p0_fpr']*100:.3f}% (安全悬崖阈值 < 2.000%)"
    )

    if fold_preds is not None:
        preds, gold, folds = fold_preds
        f_doc1, f_doc2 = [], []
        for fk in sorted(set(int(x) for x in folds)):
            m = folds == fk
            ef = evaluate_qingyu_dual_metrics(preds[m], gold[m])
            f_doc1.append(ef["doc1_total"])
            f_doc2.append(ef["doc2_strict_total"])
        print(
            f"  8-Fold Stability: Doc1={np.mean(f_doc1):.3f} ± {np.std(f_doc1):.3f} | "
            f"Doc2_Strict={np.mean(f_doc2):.3f} ± {np.std(f_doc2):.3f}"
        )

    print("-" * 108)
    print(f"  {'Channel':14s} | {'Prec':>7s} | {'Rec':>7s} | {'F1':>7s} | {'TP':>5s} | {'FP':>5s} | {'FN':>5s}")
    for row in ev["per_14"]:
        print(
            f"  {row['channel']:14s} | {row['P']:7.3f} | {row['R']:7.3f} | {row['F1']:7.3f} | "
            f"{row['tp']:5d} | {row['fp']:5d} | {row['fn']:5d}"
        )
    print("=" * 108)


def summarize_submission_counts(pred_6d: np.ndarray) -> Dict[str, Any]:
    n_total = int(len(pred_6d))
    n_clean = int(np.sum((pred_6d == 0).all(axis=1)))
    n_p0_imgs = int(np.sum((pred_6d == 3).any(axis=1)))
    n_p0_cells = int(np.sum(pred_6d == 3))
    n_p1_cells = int(np.sum(pred_6d == 2))
    n_p2_cells = int(np.sum(pred_6d == 1))
    return {
        "total_images": n_total,
        "all_clean_images": n_clean,
        "p0_images": n_p0_imgs,
        "p0_cells": n_p0_cells,
        "p1_cells": n_p1_cells,
        "p2_cells": n_p2_cells,
        "total_positive_cells": n_p0_cells + n_p1_cells + n_p2_cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Qingyu 3-Tier Certainty-Gated Stacking & 30-Step DFW Decoder")
    ap.add_argument("--split-csv", default=str(QINGYU_SPLITS_DIR / "qingyu_8fold.csv"))
    ap.add_argument("--results-dir", default=str(QINGYU_RESULTS_DIR))
    ap.add_argument("--dual-engine-npz", default=str(QINGYU_RESULTS_DIR / "dual_engine_meta.npz"))
    ap.add_argument("--soup-dir", default=str(QINGYU_CKPT_DIR / "soup_4090"))
    ap.add_argument("--towers", nargs="*", default=None, help="可选指定参与融合的子集模型名")
    ap.add_argument("--stacker-c", type=float, default=0.10)
    ap.add_argument("--clean-c", type=float, default=0.15)
    ap.add_argument("--dfw-steps", type=int, default=30)
    ap.add_argument("--eta-doc1", type=float, default=0.50, help="0.50 同时兼顾 Doc1 与 Doc2_Strict；0.0 纯优 Doc2_Strict")
    ap.add_argument("--max-clean-p0-fpr", type=float, default=0.005)
    ap.add_argument("--export-oof-teacher", default=str(QINGYU_RESULTS_DIR / "oof_ensemble_teacher.npz"))
    ap.add_argument("--out-tag", default="qingyu_full_56plus")
    ap.add_argument("--write-default-submission", action="store_true", default=True)
    args = ap.parse_args()

    setup_sys_path_and_cwd(chdir_repo=True)
    enforce_deterministic_runtime(2026)
    print_pool_budget_summary()

    oof_names, gold_6d, fold_ids = load_canonical_8fold_table(args.split_csv)
    results_dir = Path(args.results_dir)
    (
        tower_names,
        oof_6x4_list,
        oof_qc_list,
        t150_names,
        t150_6x4_list,
        extra_names,
        extra_6x4_list,
    ) = load_available_qingyu_towers(results_dir, oof_names, only_towers=args.towers)

    if not tower_names:
        raise FileNotFoundError(
            f"在 {results_dir} 下尚未找到已汇总的 8 折 `tier*_8fold_oof.npz` 文件。"
            f"若需在零重训前提下对现有纯真实资产执行止血验证，请先运行 `python3 code/qingyu/stage1_pure_real_probe.py`；"
            f"若需运行全量晴羽训练管线，请执行 `bash run_qingyu_pipeline.sh --all`。"
        )

    print(f"\n[Qingyu Fusion Inventory] 已加载 {len(tower_names)} 座纯真实 8 折质量塔 (N={len(oof_names)}):")
    for idx, tname in enumerate(tower_names):
        solo_dec = DampedFrankWolfeBayesianDecoder(
            dfw_steps=20, eta_doc1=args.eta_doc1, max_clean_p0_fpr=args.max_clean_p0_fpr
        ).fit(oof_6x4_list[idx], oof_qc_list[idx], gold_6d)
        solo_ev = evaluate_qingyu_dual_metrics(solo_dec.decode(oof_6x4_list[idx], oof_qc_list[idx]), gold_6d)
        solo_ece = compute_ece_6x4(oof_6x4_list[idx], gold_6d)
        print(
            f"  [{idx+1}/{len(tower_names)}] {tname:24s} | OOF Doc1={solo_ev['doc1_total']:6.3f} | "
            f"Doc2_Strict={solo_ev['doc2_strict_total']:6.3f} | ECE={solo_ece:.4f} | CleanP0_FPR={solo_ev['clean_p0_fpr']*100:.2f}%"
        )

    oof_phys = load_phys_matrix_aligned(oof_names, args.dual_engine_npz)
    t150_phys = load_phys_matrix_aligned(t150_names, args.dual_engine_npz) if t150_names is not None else None
    if oof_phys is None or t150_phys is None:
        oof_phys, t150_phys = None, None
        print("[Qingyu Fusion] 物理特征缓存未完整覆盖测试集，使用纯视觉确定性门控对数几率模式。")
    else:
        print(f"[Qingyu Fusion] 已对齐 36D 单图确定性物理特征 (`dual_engine_meta.npz`)。")

    stacker = CenteredLogOddsCertaintyGatedStacker(C=args.stacker_c, clean_C=args.clean_c)
    oof_fused_6x4, oof_fused_qclean = stacker.fit_oof_and_full(
        models_6x4=oof_6x4_list,
        gold_6d=gold_6d,
        fold_ids=fold_ids,
        phys_mat=oof_phys,
        model_names=tower_names,
    )

    if args.export_oof_teacher:
        os.makedirs(os.path.dirname(os.path.abspath(args.export_oof_teacher)), exist_ok=True)
        np.savez_compressed(
            args.export_oof_teacher,
            oof_names=np.array(oof_names),
            oof_probs_6x4=oof_fused_6x4,
            oof_qclean=oof_fused_qclean,
            oof_folds=fold_ids,
            towers=np.array(tower_names),
        )
        print(f"[Qingyu Mutual-KD] 已导出严格折外全池集成软教师 -> {args.export_oof_teacher}")

        # 同步导出方案 4.4 节双向跨模态正交互蒸馏软教师:
        #   (1) oof_tier3_semantic_teacher.npz: 由双 VLM 蒸馏语义伪影知识给 Tier-1 / Tier-2
        #   (2) oof_tier12_physical_teacher.npz: 由 1:1 物理与大图塔反哺高频光晕/微纹理校准给双 VLM
        t_dir = os.path.dirname(os.path.abspath(args.export_oof_teacher))
        tier3_idx = [i for i, n in enumerate(tower_names) if n in ("clef_flash_9b", "qwen3_vl_8b")]
        tier12_idx = [i for i, n in enumerate(tower_names) if n not in ("clef_flash_9b", "qwen3_vl_8b")]
        if tier3_idx:
            s_t3 = CenteredLogOddsCertaintyGatedStacker(C=args.stacker_c, clean_C=args.clean_c)
            p_t3, q_t3 = s_t3.fit_oof_and_full(
                [oof_6x4_list[i] for i in tier3_idx], gold_6d, fold_ids, phys_mat=oof_phys,
                model_names=[tower_names[i] for i in tier3_idx],
            )
            np.savez_compressed(
                os.path.join(t_dir, "oof_tier3_semantic_teacher.npz"),
                oof_names=np.array(oof_names), oof_probs_6x4=p_t3, oof_qclean=q_t3, oof_folds=fold_ids,
            )
        if tier12_idx:
            s_t12 = CenteredLogOddsCertaintyGatedStacker(C=args.stacker_c, clean_C=args.clean_c)
            p_t12, q_t12 = s_t12.fit_oof_and_full(
                [oof_6x4_list[i] for i in tier12_idx], gold_6d, fold_ids, phys_mat=oof_phys,
                model_names=[tower_names[i] for i in tier12_idx],
            )
            np.savez_compressed(
                os.path.join(t_dir, "oof_tier12_physical_teacher.npz"),
                oof_names=np.array(oof_names), oof_probs_6x4=p_t12, oof_qclean=q_t12, oof_folds=fold_ids,
            )

    decoder = DampedFrankWolfeBayesianDecoder(
        dfw_steps=args.dfw_steps,
        eta_doc1=args.eta_doc1,
        max_clean_p0_fpr=args.max_clean_p0_fpr,
    ).fit(oof_fused_6x4, oof_fused_qclean, gold_6d)

    oof_preds = decoder.decode(oof_fused_6x4, oof_fused_qclean)
    oof_ev = evaluate_qingyu_dual_metrics(oof_preds, gold_6d)
    oof_ece = compute_ece_6x4(oof_fused_6x4, gold_6d)
    print_diagnostic_report(
        f"晴羽三阶异构确定性门控融合 + {args.dfw_steps}步 DFW 贝叶斯解码 (N={len(gold_6d)}, Towers={len(tower_names)})",
        oof_ev,
        oof_ece,
        fold_preds=(oof_preds, gold_6d, fold_ids),
    )
    print(
        f"[DFW Stationary Multipliers] lambda_clean={decoder.lambda_clean:.4f}, "
        f"lambda_cliff={decoder.lambda_cliff:.4f}, DFW_obj={decoder.dfw_history[0]:.3f}->{decoder.dfw_history[-1]:.3f}"
    )

    # 持久化 Stacker 与 DFW 解码器供单卡 4090 部署引擎直接加载
    soup_dir = Path(args.soup_dir)
    soup_dir.mkdir(parents=True, exist_ok=True)
    stacker.save(str(soup_dir / "stacker.pkl"))
    decoder.save(str(soup_dir / "dfw_decoder.pkl"))
    stacker.save(str(results_dir / "stacker.pkl"))
    decoder.save(str(results_dir / "dfw_decoder.pkl"))

    # test_150 单图独立性硬校验 + 生成官方提交 JSON
    if t150_names is not None and t150_6x4_list:
        inv_audit = verify_single_image_invariance(stacker, decoder, t150_6x4_list, t150_phys)
        t150_preds = inv_audit["batch_preds"]
        sub_150, _ = format_official_submission(t150_names, t150_preds)
        QINGYU_FINAL_DIR.mkdir(parents=True, exist_ok=True)
        tagged_json = QINGYU_FINAL_DIR / f"pred_nriqa_dataset_test_{args.out_tag}.json"
        with open(tagged_json, "w", encoding="utf-8") as f:
            json.dump(sub_150, f, ensure_ascii=False, indent=2)
        if args.write_default_submission:
            default_json = QINGYU_FINAL_DIR / "pred_nriqa_dataset_test.json"
            with open(default_json, "w", encoding="utf-8") as f:
                json.dump(sub_150, f, ensure_ascii=False, indent=2)

        cnt_150 = summarize_submission_counts(t150_preds)
        print(
            f"[Qingyu test_150 Audit] Single==Batch Invariance PASS (max_diff={inv_audit['max_abs_prob_diff']:.2e}, "
            f"mismatches=0) | Clean={cnt_150['all_clean_images']} | P0_Imgs={cnt_150['p0_images']} "
            f"(P0={cnt_150['p0_cells']}, P1={cnt_150['p1_cells']}, P2={cnt_150['p2_cells']}) -> {tagged_json}"
        )

    # 若存在终评扩展测试集 (如 654 图)，同步做单图独立性校验与导出
    if extra_names is not None and extra_6x4_list:
        ext_phys = load_phys_matrix_aligned(extra_names, args.dual_engine_npz) if oof_phys is not None else None
        ext_inv = verify_single_image_invariance(stacker, decoder, extra_6x4_list, ext_phys, max_check=200)
        ext_preds = ext_inv["batch_preds"]
        sub_ext, _ = format_official_submission(extra_names, ext_preds)
        ext_json = QINGYU_FINAL_DIR / f"pred_nriqa_dataset_final_{len(extra_names)}_{args.out_tag}.json"
        with open(ext_json, "w", encoding="utf-8") as f:
            json.dump(sub_ext, f, ensure_ascii=False, indent=2)
        cnt_ext = summarize_submission_counts(ext_preds)
        print(
            f"[Qingyu Final-{len(extra_names)} Audit] Single==Batch Invariance PASS | "
            f"Clean={cnt_ext['all_clean_images']} | P0_Imgs={cnt_ext['p0_images']} -> {ext_json}"
        )

    summary_json = results_dir / f"fusion_summary_{args.out_tag}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "towers": tower_names,
                "oof_metrics": oof_ev,
                "oof_ece": oof_ece,
                "decoder": decoder.to_json_dict(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
