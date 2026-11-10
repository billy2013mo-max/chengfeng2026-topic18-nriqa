#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/validate_submission.py
==================================
晴羽观感体系 — 支柱五 (7.3 节):
赛题官方终评 JSON 格式、ID 一对一双射、二值维度合法枚举与双次运行 100% 确定性硬门禁。

检查项 (任一不满足立即抛出详细错误并以非零状态码退出):
1. JSON 顶层必须是非空 `list`，长度与模板文件 (`150` 练习集或 `654` 终评集) 完全一致；
2. `image` 字段与模板构成严格一一双射 (无遗漏、无重复、顺序一致)；
3. `label` 字典必须且仅包含 `{"模糊", "纹理损失", "过度生成", "虚化突兀", "过锐", "花屏"}` 6 个键；
4. 四档维度 (`模糊`, `纹理损失`, `过锐`, `过度生成`) 取值严格属于 `{"无问题", "P2", "P1", "P0"}`；
5. 二值维度 (`花屏`, `虚化突兀`) 取值严格属于 `{"无问题", "P0"}` (绝不允许出现 `P1` 或 `P2`)；
6. 无任何 `None`/`NaN`/非法字符串；
7. 可选 `--compare-run2`: 校验两次独立推理生成的 JSON 100% 标签完全一致 (`0` 差异)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set

REQUIRED_KEYS: Set[str] = {"模糊", "纹理损失", "过度生成", "虚化突兀", "过锐", "花屏"}
FOUR_GRADE_DIMS: Set[str] = {"模糊", "纹理损失", "过锐", "过度生成"}
BINARY_P0_DIMS: Set[str] = {"花屏", "虚化突兀"}
VALID_4_GRADES: Set[str] = {"无问题", "P2", "P1", "P0"}
VALID_2_GRADES: Set[str] = {"无问题", "P0"}


def validate_submission_file(
    pred_path: str,
    template_path: str = "",
    compare_run2_path: str = "",
    require_exact_order: bool = True,
) -> Dict[str, Any]:
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"未找到待校验预测文件: {pred_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    if not isinstance(preds, list) or len(preds) == 0:
        raise ValueError(f"预测文件顶层必须是非空 list，实际类型: {type(preds)}")

    seen_images: Set[str] = set()
    ordered_pred_imgs: List[str] = []
    all_clean_cnt = 0
    any_p0_cnt = 0
    dim_grade_hist: Dict[str, Dict[str, int]] = {
        k: {"无问题": 0, "P2": 0, "P1": 0, "P0": 0} for k in REQUIRED_KEYS
    }

    for idx, entry in enumerate(preds):
        if not isinstance(entry, dict):
            raise ValueError(f"第 {idx} 条记录不是 dict: {entry!r}")
        if "image" not in entry or "label" not in entry:
            raise ValueError(f"第 {idx} 条记录缺失 'image' 或 'label' 字段: {list(entry.keys())}")
        extra_top = set(entry.keys()) - {"image", "label"}
        if extra_top:
            raise ValueError(
                f"第 {idx} 条记录包含非官方顶层字段 {extra_top} (归因字段请写入独立 sidecar JSON，保持主提交纯净)"
            )

        img_name = entry["image"]
        if not isinstance(img_name, str) or not img_name.strip():
            raise ValueError(f"第 {idx} 条记录 'image' 非法: {img_name!r}")
        if img_name in seen_images:
            raise ValueError(f"发现重复 'image' ID: {img_name}")
        seen_images.add(img_name)
        ordered_pred_imgs.append(img_name)

        lbl = entry["label"]
        if not isinstance(lbl, dict):
            raise ValueError(f"图像 {img_name} 的 'label' 不是 dict")
        keys = set(lbl.keys())
        if keys != REQUIRED_KEYS:
            missing = REQUIRED_KEYS - keys
            extra = keys - REQUIRED_KEYS
            raise ValueError(f"图像 {img_name} 'label' 键集合不匹配: 缺失={missing}, 多余={extra}")

        is_all_clean = True
        has_p0 = False
        for k, v in lbl.items():
            if not isinstance(v, str):
                raise ValueError(f"图像 {img_name} 维度 {k} 值类型非法 ({type(v)}): {v!r}")
            if k in BINARY_P0_DIMS:
                if v not in VALID_2_GRADES:
                    raise ValueError(
                        f"二值 P0 维度非法取值! 图像 {img_name} 的 [{k}]={v!r}，仅允许 {VALID_2_GRADES}"
                    )
            else:
                if v not in VALID_4_GRADES:
                    raise ValueError(
                        f"四档维度非法取值! 图像 {img_name} 的 [{k}]={v!r}，仅允许 {VALID_4_GRADES}"
                    )
            dim_grade_hist[k][v] += 1
            if v != "无问题":
                is_all_clean = False
            if v == "P0":
                has_p0 = True

        if is_all_clean:
            all_clean_cnt += 1
        if has_p0:
            any_p0_cnt += 1

    if template_path:
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"未找到模板文件: {template_path}")
        with open(template_path, "r", encoding="utf-8") as tf:
            tpl = json.load(tf)
        tpl_imgs = [str(x["image"] if isinstance(x, dict) else x) for x in tpl]
        if len(ordered_pred_imgs) != len(tpl_imgs):
            raise ValueError(
                f"样本总数与模板不一致: pred={len(ordered_pred_imgs)} vs template={len(tpl_imgs)}"
            )
        if set(ordered_pred_imgs) != set(tpl_imgs):
            missing_ids = list(set(tpl_imgs) - set(ordered_pred_imgs))[:5]
            extra_ids = list(set(ordered_pred_imgs) - set(tpl_imgs))[:5]
            raise ValueError(f"ID 集合非双射! 缺失示例={missing_ids}, 多余示例={extra_ids}")
        if require_exact_order and ordered_pred_imgs != tpl_imgs:
            first_diff = next(
                i for i, (a, b) in enumerate(zip(ordered_pred_imgs, tpl_imgs)) if a != b
            )
            raise ValueError(
                f"图像顺序与模板在索引 {first_diff} 处不一致: pred={ordered_pred_imgs[first_diff]} vs tpl={tpl_imgs[first_diff]}"
            )

    determinism_checked = False
    if compare_run2_path:
        with open(compare_run2_path, "r", encoding="utf-8") as f2:
            preds2 = json.load(f2)
        if len(preds) != len(preds2):
            raise ValueError(f"双次确定性对比长度不一致: {len(preds)} vs {len(preds2)}")
        diffs = [
            i for i, (a, b) in enumerate(zip(preds, preds2))
            if a.get("image") != b.get("image") or a.get("label") != b.get("label")
        ]
        if diffs:
            i0 = diffs[0]
            raise AssertionError(
                f"双次运行确定性校验失败! 共 {len(diffs)} 张图像输出不一致 (首例 idx={i0}: {preds[i0]} vs {preds2[i0]})"
            )
        determinism_checked = True

    omp_threads = os.environ.get("OMP_NUM_THREADS", "")
    return {
        "status": "PASS",
        "pred_path": pred_path,
        "total_images": len(preds),
        "all_clean_images": all_clean_cnt,
        "any_p0_images": any_p0_cnt,
        "omp_num_threads": omp_threads or "unset",
        "determinism_verified": determinism_checked,
        "per_dim_histogram": dim_grade_hist,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="晴羽终评提交 JSON 双射、合法枚举与确定性硬门禁")
    ap.add_argument("--pred", required=True, help="待校验的官方格式 JSON 路径")
    ap.add_argument("--template", default="results/v2_full/template_150.json", help="标准模板 JSON 路径")
    ap.add_argument("--compare-run2", default="", help="可选第二次独立运行 JSON 路径以校验 100%% 确定性")
    ap.add_argument("--allow-reordered", action="store_true")
    args = ap.parse_args()

    rep = validate_submission_file(
        pred_path=args.pred,
        template_path=args.template,
        compare_run2_path=args.compare_run2,
        require_exact_order=(not args.allow_reordered),
    )
    print("=" * 92)
    print(
        f"[Submission Hard Gate: PASS] N={rep['total_images']} | Clean={rep['all_clean_images']} "
        f"| Any_P0={rep['any_p0_images']} | Determinism={rep['determinism_verified']} | OMP={rep['omp_num_threads']}"
    )
    for d in ("模糊", "纹理损失", "过锐", "过度生成", "虚化突兀", "花屏"):
        h = rep["per_dim_histogram"][d]
        print(f"  - {d:6s}: 无问题={h['无问题']:4d} | P2={h['P2']:4d} | P1={h['P1']:4d} | P0={h['P0']:4d}")
    print("=" * 92)


if __name__ == "__main__":
    main()
