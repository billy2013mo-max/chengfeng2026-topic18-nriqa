#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/r55_common.py — shared paths + the auxiliary-data manifest used by the R55 redesign.

The auxiliary pool lives OUTSIDE the labelled data and is never scored:
  data/r55_aux/images/        flat directory of every auxiliary image (PNG)
  data/r55_aux/manifest.csv   one row per auxiliary image:
      name       file name inside images/
      kind       rank    = classical same-scene degradation ladder (ordering only, never a grade label)
                 overgen = generative re-render ladder for 过度生成 (ordering only)
                 ext     = external unlabelled real frame (teacher soft label; optionally certified clean)
      seed       labelled image the row was derived from ('' for ext)
      seed_fold  fold of the seed in data/qingyu_splits (-1 for ext).  A student of fold k NEVER sees rows
                 with seed_fold == k, so no pixel of a validation image leaks into its training.
      tuple_id   rows that share a tuple_id are the same scene at increasing strength
      dim        index into DIMS of the manipulated dimension (-1 for ext)
      level      1..3 = strength rank inside the tuple (the seed itself is level 0)
      op         free-text operator tag

Nothing here reads test labels, legacy submission JSONs or file-name tags as model input (AGENTS.md rules 1-2).
"""
import csv
import os
import sys
from typing import Dict, List

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE = os.path.join(REPO, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

AUX_DIR = os.path.join(REPO, "data/r55_aux")
AUX_IMG = os.path.join(AUX_DIR, "images")
AUX_MANIFEST = os.path.join(AUX_DIR, "manifest.csv")
AUX_TEMPLATE = os.path.join(AUX_DIR, "template.json")
AUX_CLEAN = os.path.join(AUX_DIR, "certified_clean.csv")
R55_OUT = os.path.join(REPO, "results/r55")
AUX_META = os.path.join(R55_OUT, "dual_engine_meta_aux.npz")
AUX_TEACHER = os.path.join(R55_OUT, "aux_teacher.npz")
SPLIT_CSV = os.path.join(REPO, "data/qingyu_splits/qingyu_8fold.csv")
FIELDS = ["name", "kind", "seed", "seed_fold", "tuple_id", "dim", "level", "op"]

# Tier-1 towers of the accepted stack `s1` = the teachers (each has fold_{0..7}_best.pt).
TEACHER_TOWERS = [
    ("pe_core_large", "results/qingyu/pe_core_large"),
    ("cradio_v4_h", "results/qingyu/cradio_v4_h"),
    ("siglip2_so400m_naflex", "results/qingyu/siglip2_so400m_naflex"),
    ("pe_core_large", "results/next55/towers/qy_s2027/pe_core_large"),
    ("cradio_v4_h", "results/next55/towers/qy_s2027/cradio_v4_h"),
    ("siglip2_so400m_naflex", "results/next55/towers/qy_s2027/siglip2_so400m_naflex"),
]


def read_manifest(path: str = AUX_MANIFEST) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_manifest(rows: List[Dict[str, object]], path: str = AUX_MANIFEST) -> int:
    """Appends rows, skipping names that are already present. Returns the number added."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    have = {r["name"] for r in read_manifest(path)}
    new = [r for r in rows if r["name"] not in have]
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for r in new:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    return len(new)


def load_split_rows() -> List[Dict[str, str]]:
    with open(SPLIT_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))
