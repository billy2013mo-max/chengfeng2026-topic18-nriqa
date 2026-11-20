#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/phoenix/config.py
======================
Centralized competition schema, strict <=30B / <=24GB single-4090 budget audit,
pure-real image & manifest resolution, dual-caliber 65-point metric evaluation
(Official Appendix-2 Strict vs Practice-Board Cliff), and bijective JSON submission
validator for the Phoenix 55+ Clean-Slate Refactoring Pipeline.

Hard Engineering Guarantees:
  - Standalone package (`import phoenix.*` or direct script execution without shadowing stdlib `code`).
  - Zero access to legacy `test150` JSON anchors or historical `.npz` caches.
  - Zero test-batch dynamic statistics.
  - Works both on full PyTorch/NumPy GPU servers AND bare stdlib Python laptops for manifest/schema audits.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - bare stdlib laptop fallback
    np = None  # type: ignore[assignment]

PHOENIX_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(PHOENIX_DIR)
REPO_ROOT = os.path.abspath(os.path.join(PHOENIX_DIR, "../.."))

PHOENIX_SPLITS_DIR = Path(REPO_ROOT) / "data" / "phoenix_splits"
PHOENIX_CKPT_DIR = Path(REPO_ROOT) / "checkpoints" / "phoenix"
PHOENIX_OOF_DIR = Path(REPO_ROOT) / "results" / "phoenix"
PHOENIX_SUB_DIR = Path(REPO_ROOT) / "results" / "phoenix_submissions"

# Official 6 defect dimensions & severity encoding
# Internal canonical order matches manifest_combined.csv columns:
#   0: 模糊 (0,1,2,3), 1: 纹理损失 (0,1,2,3), 2: 过锐 (0,1,2,3),
#   3: 花屏 (0,3),     4: 过度生成 (0,1,2,3), 5: 虚化突兀 (0,3)
DIMS: List[str] = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
FULL_DIMS: List[int] = [0, 1, 2, 4]
P0_ONLY_DIMS: List[int] = [3, 5]

# Exact key order required in submission JSON files
JSON_KEY_ORDER: List[str] = ["模糊", "纹理损失", "过度生成", "虚化突兀", "过锐", "花屏"]
GRADE_INT_TO_STR: Dict[int, str] = {0: "无问题", 1: "P2", 2: "P1", 3: "P0"}
GRADE_STR_TO_INT: Dict[str, int] = {v: k for k, v in GRADE_INT_TO_STR.items()}

# 14 positive binary channels in Official Appendix-2 Positive Macro-F1:
#   0..5:   P0 (grade == 3) for all 6 dimensions
#   6..9:   P1 (grade == 2) for FULL_DIMS [模糊, 纹理损失, 过锐, 过度生成]
#   10..13: P2 (grade == 1) for FULL_DIMS [模糊, 纹理损失, 过锐, 过度生成]
CHANNEL_14_SPECS: List[Tuple[str, int, int]] = (
    [(f"{DIMS[d]}_P0", d, 3) for d in range(6)]
    + [(f"{DIMS[d]}_P1", d, 2) for d in FULL_DIMS]
    + [(f"{DIMS[d]}_P2", d, 1) for d in FULL_DIMS]
)
CHANNEL_14_NAMES: List[str] = [s[0] for s in CHANNEL_14_SPECS]


def build_joint_1024_state_table() -> Any:
    """
    Enumerates all |Y| = 4^4 * 2^2 = 1,024 legal 6-dimensional label configurations.
    State 0 is guaranteed to be the all-clean vector [0, 0, 0, 0, 0, 0].
    Returns `np.ndarray` of shape `(1024, 6)` when NumPy is installed (or nested list otherwise).
    """
    per_dim_choices = [
        [0, 1, 2, 3] if d in FULL_DIMS else [0, 3]
        for d in range(6)
    ]
    raw_states = [list(s) for s in itertools.product(*per_dim_choices)]
    assert len(raw_states) == 1024 and raw_states[0] == [0, 0, 0, 0, 0, 0]
    if np is not None:
        return np.array(raw_states, dtype=np.int64)
    return raw_states


JOINT_1024_STATES: Any = build_joint_1024_state_table()

# Strictly Compliant Orthogonal 3-Pillar Budget (18.655B <= 30.000B, 15.48 GB <= 24.00 GB)
PHOENIX_MODEL_BUDGET: Dict[str, Dict[str, object]] = {
    "pillar1_qwen3vl_8b": {
        "pillar": "Pillar-1 (Semantic-Perceptual Native VLM)",
        "display": "Qwen3-VL-8B-Instruct (Native 2D-RoPE + ViT Full-DoRA + DeQA Head)",
        "params_b": 8.750,
        "vram_gb_4090": 5.75,
        "precision_4090": "ViT-BF16 + LLM-W4A16 + Head-FP32",
        "candidates": [
            "models/qwen3vl-8b",
            "models/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
        ],
    },
    "pillar2_hetero_vlm_8b": {
        "pillar": "Pillar-2 (Heterogeneous RL2R Ranking VLM)",
        "display": "Clef-Flash-9B / InternVL3.5-8B (Pairwise Thurstone GRPO + DeQA Head)",
        "params_b": 9.120,
        "vram_gb_4090": 5.65,
        "precision_4090": "ViT-BF16 + LLM-W4A16 + Head-FP32",
        "candidates": [
            "models/clef-flash",
            "models/clef-flash-9b",
            "models/internvl3.5-8b",
            "models/internvl3-8b",
            "models/qwen2.5-vl-7b",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ],
    },
    "pillar3_dinov3_l": {
        "pillar": "Pillar-3 (1:1 High-Freq & Physics Specialist)",
        "display": "DINOv3-ViT-L/16 (4-Layer Hypercolumn + Full-DoRA)",
        "params_b": 0.304,
        "vram_gb_4090": 0.64,
        "precision_4090": "BF16",
        "candidates": [
            "models/dinov3-vitl16",
            "facebook/dinov2-large",
        ],
    },
    "pillar3_cradio_v4_h": {
        "pillar": "Pillar-3 (1:1 High-Freq & Physics Specialist)",
        "display": "C-RADIOv4-H (Multi-Teacher Perceptual Hypercolumn)",
        "params_b": 0.630,
        "vram_gb_4090": 1.28,
        "precision_4090": "BF16",
        "candidates": [
            "models/c-radio-v4-h",
        ],
    },
    "pillar3_convnextv2_l": {
        "pillar": "Pillar-3 (1:1 High-Freq & Physics Specialist)",
        "display": "ConvNeXtV2-Large + 1:1 Haar Wavelet Bipolar Stem",
        "params_b": 0.200,
        "vram_gb_4090": 0.44,
        "precision_4090": "BF16",
        "candidates": [
            "models/convnextv2-large",
            "models/convnextv2-base",
            "convnextv2_large.fcmae_ft_in22k_in1k_384",
        ],
    },
    "pillar3_fgresq_adapter": {
        "pillar": "Pillar-3 (1:1 High-Freq & Physics Specialist)",
        "display": "FGResQ Restoration-Degradation Branch (or PE-Core-L Perceptual Tower)",
        "params_b": 0.350,
        "vram_gb_4090": 0.72,
        "precision_4090": "BF16",
        "candidates": [
            "models/fgresq/Degradation.pth",
            "models/fgresq",
            "models/pe-core-l14",
        ],
    },
    "pillar3_depth_anything_s": {
        "pillar": "Pillar-3 (CoC Depth-Defocus Jacobian Field)",
        "display": "Depth-Anything-V2-Small (Frozen Optical CoC Consistency Probe)",
        "params_b": 0.025,
        "vram_gb_4090": 0.06,
        "precision_4090": "FP16",
        "candidates": [
            "models/depth-anything-v2-small",
            "models/Depth-Anything-V2-Small-hf",
            "depth-anything/Depth-Anything-V2-Small-hf",
        ],
    },
    "pillar3_pal4vst_probe": {
        "pillar": "Pillar-3 (Localized AIGC Artifact Saliency)",
        "display": "PAL4VST / Query2Defect Spatial Artifact & Anchor Calibrator",
        "params_b": 0.046,
        "vram_gb_4090": 0.14,
        "precision_4090": "FP32",
        "candidates": [
            "models/pal4vst",
        ],
    },
}

TOTAL_PARAMS_B: float = round(sum(float(v["params_b"]) for v in PHOENIX_MODEL_BUDGET.values()), 3)
WEIGHT_VRAM_GB: float = round(sum(float(v["vram_gb_4090"]) for v in PHOENIX_MODEL_BUDGET.values()), 2)
RUNTIME_ACTIVATION_GB: float = 1.00
PEAK_4090_VRAM_GB: float = round(WEIGHT_VRAM_GB + RUNTIME_ACTIVATION_GB, 2)


def set_deterministic_seed(seed: int = 2026) -> None:
    """Enforces deterministic RNG state across Python, NumPy, and PyTorch."""
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def resolve_local_or_hf_model(model_key: str, root: str = REPO_ROOT) -> str:
    """Prefers offline local weights under `models/`, falling back to HuggingFace/timm ID."""
    spec = PHOENIX_MODEL_BUDGET[model_key]
    extra_base = os.environ.get("R55_BASE_MODELS_DIR", "").strip()
    roots = [root, os.path.dirname(root), os.path.join(os.path.dirname(root), "03_模型产物")]
    for cand in spec["candidates"]:  # type: ignore[index]
        if os.path.isabs(cand) and os.path.exists(cand):
            return cand
        if extra_base and str(cand).startswith("models/"):
            p_env = os.path.join(extra_base, str(cand)[len("models/"):])
            if os.path.exists(p_env):
                return p_env
        for rt in roots:
            for rel_c in (str(cand), str(cand).replace("models/", "base_models/", 1)):
                p = os.path.join(rt, rel_c)
                if os.path.exists(p):
                    return p
    return str(spec["candidates"][-1])  # type: ignore[index]


class PureRealImageResolver:
    """
    Loads strictly pure-real images from `/dev/shm/race18_native` memmap (if available)
    or standard real image directories. Blocks any synthetic paths by design.
    """

    DEFAULT_DIRS = (
        "data/images",
        "data/images_supp",
        "data/test_150",
        "data/test_images",
        "dataset/images",
        "dataset/test_images",
    )

    def __init__(
        self,
        root: str = REPO_ROOT,
        extra_dirs: Optional[Sequence[str]] = None,
        shm_bin: str = "/dev/shm/race18_native/cache.bin",
        shm_index: str = "/dev/shm/race18_native/index.npz",
    ) -> None:
        self.root = root
        self.search_dirs: List[str] = []
        if extra_dirs:
            for d in extra_dirs:
                if d:
                    full = d if os.path.isabs(d) else os.path.join(root, d)
                    if full not in self.search_dirs:
                        self.search_dirs.append(full)
        for d in self.DEFAULT_DIRS:
            full = os.path.join(root, d)
            if full not in self.search_dirs:
                self.search_dirs.append(full)

        self._memmap = None
        self._shm_pos: Dict[str, int] = {}
        self._shm_h = None
        self._shm_w = None
        self._shm_off = None
        if np is not None and os.path.exists(shm_bin) and os.path.exists(shm_index):
            try:
                idx = np.load(shm_index, allow_pickle=True)
                self._shm_pos = {str(nm): i for i, nm in enumerate(idx["names"])}
                self._shm_h = idx["h"]
                self._shm_w = idx["w"]
                self._shm_off = idx["offsets"]
                self._memmap = np.memmap(shm_bin, dtype=np.uint8, mode="r")
            except Exception:
                self._memmap = None

    @staticmethod
    def assert_not_synthetic(name: str) -> None:
        lowered = name.lower()
        for banned in ("v9_synth", "v8_synth", "v7_synth", "v6_synth", "_natneg", "_prism_", "synth_f"):
            if banned in lowered:
                raise ValueError(f"Synthetic sample '{name}' is strictly forbidden in Phoenix pure-real pipeline.")

    def find_path(self, name: str) -> Optional[str]:
        self.assert_not_synthetic(name)
        if os.path.isabs(name) and os.path.exists(name):
            return name
        rel = os.path.join(self.root, name)
        if os.path.exists(rel):
            return rel
        base = os.path.basename(name)
        for d in self.search_dirs:
            for cand in (os.path.join(d, name), os.path.join(d, base)):
                if os.path.exists(cand):
                    return cand
        return None

    def load_rgb_pil(self, name: str):
        from PIL import Image

        self.assert_not_synthetic(name)
        if np is not None and self._memmap is not None and name in self._shm_pos:
            j = self._shm_pos[name]
            h = int(self._shm_h[j])
            w = int(self._shm_w[j])
            off = int(self._shm_off[j])
            arr = np.asarray(self._memmap[off : off + h * w * 3]).reshape(h, w, 3)
            return Image.fromarray(arr, mode="RGB")
        path = self.find_path(name)
        if path is None:
            raise FileNotFoundError(
                f"Real image '{name}' not found under {self.search_dirs}. "
                f"Supply --test-dir pointing to the image folder."
            )
        return Image.open(path).convert("RGB")


def load_real_manifest(manifest_rel_or_abs: str = "data/manifest_combined.csv") -> List[Dict[str, object]]:
    """Loads all 3,573 pure-real annotated samples from `data/manifest_combined.csv`."""
    path = (
        manifest_rel_or_abs
        if os.path.isabs(manifest_rel_or_abs)
        else os.path.join(REPO_ROOT, manifest_rel_or_abs)
    )
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nm = r["name"].strip()
            lowered = nm.lower()
            if any(b in lowered for b in ("v9_synth", "v8_synth", "v7_synth", "v6_synth", "_natneg", "_prism_")):
                continue
            grades = [int(r[c]) for c in DIMS]
            split_tag = r.get("split", "orig").strip()
            batch_id = 0 if split_tag == "orig" else 1
            records.append(
                {
                    "name": nm,
                    "raw_source": r.get("source", "").strip(),
                    "width": int(r.get("width", 0) or 0),
                    "height": int(r.get("height", 0) or 0),
                    "split": split_tag,
                    "batch_id": batch_id,
                    "grades": grades,
                    **{c: int(r[c]) for c in DIMS},
                }
            )
    return records


def load_ordered_test_names(template_or_csv: str) -> List[str]:
    """
    Supports loading ordered test image names (ONLY image IDs, NEVER labels!) from:
      - Official JSON template (`[{"image": "...", "label": {...}}, ...]`)
      - Single-column CSV (`data/manifest_test_150.csv` with header `name`)
    Automatically checks `documents/`, `docs/reports/`, and `data/manifest_test_150.csv`
    fallbacks if root template files were moved during repo cleanup.
    """
    path = template_or_csv if os.path.isabs(template_or_csv) else os.path.join(REPO_ROOT, template_or_csv)
    if not os.path.exists(path):
        base = os.path.basename(template_or_csv)
        fallbacks = [
            os.path.join(REPO_ROOT, "documents", base),
            os.path.join(REPO_ROOT, "docs", "reports", base),
        ]
        if base == "pred_nriqa_dataset_test.json":
            fallbacks.extend(
                [
                    os.path.join(REPO_ROOT, "data", "manifest_test_150.csv"),
                    os.path.join(REPO_ROOT, "results", "v2_full", "template_150.json"),
                ]
            )
        for alt in fallbacks:
            if os.path.exists(alt):
                path = alt
                break
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as f:
            return [row["name"].strip() for row in csv.DictReader(f) if row.get("name", "").strip()]
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in template {path}")
    return [str(item["image"] if isinstance(item, dict) else item).strip() for item in payload]


def _to_2d_int_rows(arr_or_seq: Any) -> List[List[int]]:
    if hasattr(arr_or_seq, "tolist"):
        raw = arr_or_seq.tolist()
    else:
        raw = list(arr_or_seq)
    return [[int(v) for v in row] for row in raw]


def evaluate_official_metrics(
    pred_6d: Any,
    gold_6d: Any,
    cliff_tau: float = 0.02,
) -> Dict[str, object]:
    """
    Evaluates `[N, 6]` integer predictions in `{0, 1(P2), 2(P1), 3(P0)}` under BOTH:
      1. `strict` (Official Spec Appendix-2):
         - S_pos (25 pts): Equal-weight Macro-F1 across all 14 positive labels.
           When TP+FP == 0 -> P = 0; when TP+FN == 0 -> R = 0; when 2TP+FP+FN == 0 -> F1 = 0.
         - S_p0  (20 pts): Equal-weight average over 6 P0 labels of (0.5*R + 0.3*F1 + 0.2*P).
         - S_clean (20 pts): Exact 6-dimension all-zero match rate on true Clean samples.
      2. `practice_cliff` (Legacy Online Practice Board):
         - S_pos (25 pts): 6-dimension hierarchical Macro-F1.
         - S_p0  (20 pts): 0.6*R_p0 + 0.4*F1_p0.
         - S_clean (20 pts): max(0, 1 - FPR_clean_P0 / 0.02).
    """
    preds_rows = _to_2d_int_rows(pred_6d)
    gold_rows = _to_2d_int_rows(gold_6d)
    n_samples = len(gold_rows)
    if len(preds_rows) != n_samples or (n_samples > 0 and len(preds_rows[0]) != 6):
        raise ValueError("Shape mismatch in evaluate_official_metrics")

    # Enforce binary domain {0, 3} on P0_ONLY_DIMS
    for r in preds_rows:
        for d in P0_ONLY_DIMS:
            r[d] = 3 if r[d] >= 2 else 0

    per_channel = []
    f1_by_dim_practice: List[float] = []
    p0_r_strict, p0_r_prac, p0_p_list, p0_f1_list = [], [], [], []

    for d_idx, d_name in enumerate(DIMS):
        grades = [(1, "P2"), (2, "P1"), (3, "P0")] if d_idx in FULL_DIMS else [(3, "P0")]
        dim_f1s = []
        for g_int, g_str in grades:
            tp = fp = fn = 0
            for i in range(n_samples):
                yt = gold_rows[i][d_idx] == g_int
                yp = preds_rows[i][d_idx] == g_int
                if yt and yp:
                    tp += 1
                elif (not yt) and yp:
                    fp += 1
                elif yt and (not yp):
                    fn += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_strict = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            rec_prac = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

            per_channel.append(
                {
                    "channel": f"{d_name}_{g_str}",
                    "dim": d_idx,
                    "grade": g_int,
                    "P": float(prec),
                    "R": float(rec_strict),
                    "F1": float(f1),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
            dim_f1s.append(f1)
            if g_int == 3:
                p0_r_strict.append(rec_strict)
                p0_r_prac.append(rec_prac)
                p0_p_list.append(prec)
                p0_f1_list.append(f1)
        f1_by_dim_practice.append(sum(dim_f1s) / len(dim_f1s))

    # 1. Official Appendix-2 Strict Metrics
    s_pos_strict = sum(float(c["F1"]) for c in per_channel) / len(per_channel)
    p0_rec = sum(p0_r_strict) / len(p0_r_strict)
    p0_prec = sum(p0_p_list) / len(p0_p_list)
    p0_f1 = sum(p0_f1_list) / len(p0_f1_list)
    s_p0_strict = 0.5 * p0_rec + 0.3 * p0_f1 + 0.2 * p0_prec

    n_clean = 0
    n_clean_exact_hit = 0
    n_clean_p0_fp = 0
    n_clean_pred = 0
    n_pos_cells_pred = 0
    for i in range(n_samples):
        gt_clean = all(v == 0 for v in gold_rows[i])
        pr_clean = all(v == 0 for v in preds_rows[i])
        if pr_clean:
            n_clean_pred += 1
        n_pos_cells_pred += sum(1 for v in preds_rows[i] if v > 0)
        if gt_clean:
            n_clean += 1
            if pr_clean:
                n_clean_exact_hit += 1
            if any(v == 3 for v in preds_rows[i]):
                n_clean_p0_fp += 1

    if n_clean > 0:
        s_clean_strict = n_clean_exact_hit / float(n_clean)
        clean_p0_fpr = n_clean_p0_fp / float(n_clean)
    else:
        s_clean_strict = 0.0
        clean_p0_fpr = 0.0
    s_clean_cliff = max(0.0, 1.0 - clean_p0_fpr / cliff_tau)

    pts_pos_strict = 25.0 * s_pos_strict
    pts_p0_strict = 20.0 * s_p0_strict
    pts_clean_strict = 20.0 * s_clean_strict
    total_strict = pts_pos_strict + pts_p0_strict + pts_clean_strict
    total_strict_with_cliff = pts_pos_strict + pts_p0_strict + 20.0 * s_clean_cliff

    # 2. Practice-Board Legacy Metrics
    s_pos_prac = sum(f1_by_dim_practice) / len(f1_by_dim_practice)
    s_p0_prac = 0.6 * (sum(p0_r_prac) / len(p0_r_prac)) + 0.4 * p0_f1
    pts_pos_prac = 25.0 * s_pos_prac
    pts_p0_prac = 20.0 * s_p0_prac
    pts_clean_prac = 20.0 * s_clean_cliff
    total_practice = pts_pos_prac + pts_p0_prac + pts_clean_prac

    passes_4_baselines_strict = bool(
        total_strict > 40.0
        and pts_pos_strict >= 15.0
        and pts_p0_strict >= 13.0
        and pts_clean_strict >= 12.0
    )

    return {
        "strict_total": float(total_strict),
        "strict_S_pos_pts": float(pts_pos_strict),
        "strict_S_p0_pts": float(pts_p0_strict),
        "strict_S_clean_pts": float(pts_clean_strict),
        "strict_with_cliff_total": float(total_strict_with_cliff),
        "practice_total": float(total_practice),
        "practice_S_pos_pts": float(pts_pos_prac),
        "practice_S_p0_pts": float(pts_p0_prac),
        "practice_S_clean_pts": float(pts_clean_prac),
        "S_pos_raw": float(s_pos_strict),
        "S_p0_raw": float(s_p0_strict),
        "S_clean_strict_raw": float(s_clean_strict),
        "S_clean_cliff_raw": float(s_clean_cliff),
        "p0_recall": float(p0_rec),
        "p0_precision": float(p0_prec),
        "p0_f1": float(p0_f1),
        "clean_p0_fpr": float(clean_p0_fpr),
        "n_samples": n_samples,
        "n_clean_gt": n_clean,
        "n_clean_pred": n_clean_pred,
        "n_pos_cells_pred": n_pos_cells_pred,
        "passes_4_baselines_strict": passes_4_baselines_strict,
        "per_channel": per_channel,
    }


def export_and_verify_submission(
    ordered_names: Sequence[str],
    pred_6d: Any,
    out_json_path: str,
    attributions: Optional[Sequence[Dict[str, object]]] = None,
    out_attrib_json_path: Optional[str] = None,
) -> List[Dict[str, object]]:
    """
    Exports predictions in exact competition JSON format and immediately validates:
      - Exact sample count & unique ID bijection with `ordered_names`
      - Exact 6 keys in `JSON_KEY_ORDER`
      - Legal label domain (`P0/P1/P2/无问题` on 4-grade dims, `P0/无问题` on binary dims)
    """
    preds_rows = _to_2d_int_rows(pred_6d)
    if len(preds_rows) != len(ordered_names) or (preds_rows and len(preds_rows[0]) != 6):
        raise ValueError(f"Expected ({len(ordered_names)}, 6) predictions, got len={len(preds_rows)}")
    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("Duplicate image IDs detected in ordered_names!")

    payload: List[Dict[str, object]] = []
    attrib_payload: List[Dict[str, object]] = []

    for i, img_name in enumerate(ordered_names):
        label_obj: Dict[str, str] = {}
        for k in JSON_KEY_ORDER:
            d_idx = DIMS.index(k)
            g_int = int(preds_rows[i][d_idx])
            if d_idx in P0_ONLY_DIMS:
                g_int = 3 if g_int >= 2 else 0
            else:
                g_int = max(0, min(3, g_int))
            g_str = GRADE_INT_TO_STR[g_int]
            if d_idx in P0_ONLY_DIMS and g_str not in ("无问题", "P0"):
                raise AssertionError(f"Illegal binary label {g_str} on {k} for {img_name}")
            label_obj[k] = g_str
        payload.append({"image": str(img_name), "label": label_obj})
        if attributions is not None:
            attrib_payload.append(
                {
                    "image": str(img_name),
                    "label": label_obj,
                    "attribution": attributions[i],
                }
            )

    os.makedirs(os.path.dirname(os.path.abspath(out_json_path)), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if out_attrib_json_path and attributions is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_attrib_json_path)), exist_ok=True)
        with open(out_attrib_json_path, "w", encoding="utf-8") as f:
            json.dump(attrib_payload, f, ensure_ascii=False, indent=2)

    # Read-back integrity verification
    with open(out_json_path, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert len(reloaded) == len(ordered_names), "Reloaded JSON length mismatch!"
    assert [x["image"] for x in reloaded] == list(ordered_names), "Image ID order mismatch!"
    for item in reloaded:
        lbl = item["label"]
        assert list(lbl.keys()) == JSON_KEY_ORDER, "Label key order mismatch!"
        for k, v in lbl.items():
            if k in ("虚化突兀", "花屏"):
                assert v in ("无问题", "P0"), f"Illegal value {v} for binary class {k}"
            else:
                assert v in ("无问题", "P2", "P1", "P0"), f"Illegal value {v} for {k}"
    return payload


def print_budget_audit() -> None:
    print("=" * 106)
    print("=== [Phoenix 55+ Refactor] Official <=30B Parameter & Single-RTX-4090 (<=24GB) Compliance Audit ===")
    print("=" * 106)
    for _, spec in PHOENIX_MODEL_BUDGET.items():
        print(
            f"  [{str(spec['pillar']):42s}] {str(spec['display']):62s} | "
            f"{float(spec['params_b']):6.3f}B | {float(spec['vram_gb_4090']):5.2f} GB"
        )
    print("-" * 106)
    assert TOTAL_PARAMS_B <= 30.0, f"Parameter budget exceeded: {TOTAL_PARAMS_B}B > 30B!"
    assert PEAK_4090_VRAM_GB <= 24.0, f"VRAM budget exceeded: {PEAK_4090_VRAM_GB}GB > 24GB!"
    print(
        f"  TOTAL ACTIVE/TRAINED PARAMS : {TOTAL_PARAMS_B:6.3f}B / 30.000B "
        f"(Safety Margin: {30.0 - TOTAL_PARAMS_B:6.3f}B) -> [STRICT PASS]"
    )
    print(
        f"  SINGLE RTX 4090 PEAK VRAM   : {PEAK_4090_VRAM_GB:6.2f} GB / 24.00 GB "
        f"(Weights {WEIGHT_VRAM_GB:.2f} GB + Runtime {RUNTIME_ACTIVATION_GB:.2f} GB, Zero Swapping) -> [STRICT PASS]"
    )
    print("=" * 106)


if __name__ == "__main__":
    print_budget_audit()
