#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate a prediction JSON against the competition submission specification (命题18).
Checks:
  1. Valid JSON list of dicts with 'image' and 'label' keys.
  2. Exact ID set match with template (if --template is provided), no missing/extra/duplicate IDs.
  3. All 6 defect keys present with legal severity values.
"""
import argparse
import json
import sys

ALLOWED = {
    "模糊": {"P0", "P1", "P2", "无问题"},
    "纹理损失": {"P0", "P1", "P2", "无问题"},
    "过度生成": {"P0", "P1", "P2", "无问题"},
    "过锐": {"P0", "P1", "P2", "无问题"},
    "虚化突兀": {"P0", "无问题"},
    "花屏": {"P0", "无问题"},
}


def validate(pred_path: str, template_path: str = "") -> None:
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{pred_path}: expected non-empty JSON list")

    ids = []
    counts = {k: {} for k in ALLOWED}
    clean_n = 0
    for idx, row in enumerate(data):
        if not isinstance(row, dict) or "image" not in row or "label" not in row:
            raise ValueError(f"Row {idx}: missing 'image' or 'label'")
        img_id = row["image"]
        ids.append(img_id)
        lbl = row["label"]
        if not isinstance(lbl, dict) or set(lbl.keys()) != set(ALLOWED.keys()):
            raise ValueError(f"Row {idx} ({img_id}): invalid defect keys {set(lbl.keys()) if isinstance(lbl, dict) else type(lbl)}")
        for k, v in lbl.items():
            if v not in ALLOWED[k]:
                raise ValueError(f"Row {idx} ({img_id}): illegal label {v!r} for {k!r}")
            counts[k][v] = counts[k].get(v, 0) + 1
        if all(v == "无问题" for v in lbl.values()):
            clean_n += 1

    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image IDs found: {len(ids)} total vs {len(set(ids))} unique")

    if template_path:
        with open(template_path, "r", encoding="utf-8") as f:
            tmpl = json.load(f)
        tmpl_ids = [r["image"] for r in tmpl]
        if len(tmpl_ids) != len(set(tmpl_ids)):
            raise ValueError(f"Template {template_path} has duplicate IDs")
        if set(ids) != set(tmpl_ids):
            missing = set(tmpl_ids) - set(ids)
            extra = set(ids) - set(tmpl_ids)
            raise ValueError(f"ID mismatch with template: missing={len(missing)}, extra={len(extra)}")

    print(f"[PASS] {pred_path}: {len(data)} samples, unique IDs, 6/6 valid defect labels, clean={clean_n}")
    for k, dist in counts.items():
        print(f"  - {k}: {dist}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="Path to prediction JSON")
    ap.add_argument("--template", default="", help="Optional template JSON to verify exact ID alignment")
    args = ap.parse_args()
    try:
        validate(args.pred, args.template)
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
