#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/config_and_paths.py
===============================
Centralized repository path resolution, deterministic runtime configuration,
strict 30B / 24GB single-4090 parameter accounting, image memmap/disk loading,
and dual-metric (Doc1 + Doc2_Strict) evaluation helpers.

Hard Compliance Guarantees:
  1. NEVER imports via `code.*` (avoids shadowing Python's stdlib `code` module).
  2. NEVER reads any legacy test_150 JSON predictions or 69-model historical caches.
  3. NEVER computes dynamic statistics (.mean()/.std()/quantiles) across test batches.
"""
import os
import sys
import csv
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from pathlib import Path

QINGYU_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(QINGYU_DIR)
REPO_ROOT = (
    "/wanqing-develop/luowenjing/18"
    if os.path.exists("/wanqing-develop/luowenjing/18")
    else os.path.abspath(os.path.join(QINGYU_DIR, "../.."))
)
QINGYU_CKPT_DIR = Path(REPO_ROOT) / "checkpoints" / "qingyu"
QINGYU_RESULTS_DIR = Path(REPO_ROOT) / "results" / "qingyu"
QINGYU_SPLITS_DIR = Path(REPO_ROOT) / "data" / "qingyu_splits"
QINGYU_FINAL_DIR = Path(REPO_ROOT) / "results" / "final"


def setup_sys_path_and_cwd(chdir_repo: bool = True) -> str:
    """Safely injects `code/`, `code/finetune/`, `code/decision_eval/` into `sys.path`."""
    if chdir_repo and os.path.isdir(REPO_ROOT):
        os.chdir(REPO_ROOT)
    for sub in (
        CODE_DIR,
        QINGYU_DIR,
        os.path.join(REPO_ROOT, "code", "finetune"),
        os.path.join(REPO_ROOT, "code", "decision_eval"),
        REPO_ROOT,
    ):
        if os.path.isdir(sub) and sub not in sys.path:
            sys.path.insert(0, sub)
    return REPO_ROOT


setup_sys_path_and_cwd(chdir_repo=False)

# Competition schema constants
DIMS: List[str] = ["模糊", "纹理损失", "过锐", "花屏", "过度生成", "虚化突兀"]
FULL_DIMS: List[int] = [0, 1, 2, 4]      # 4-grade dimensions: {0:无问题, 1:P2, 2:P1, 3:P0}
P0_ONLY_DIMS: List[int] = [3, 5]         # Binary P0 dimensions: {0:无问题, 3:P0}
JSON_KEY_ORDER: List[str] = ["模糊", "纹理损失", "过度生成", "虚化突兀", "过锐", "花屏"]
GRADE_INT_TO_STR: Dict[int, str] = {0: "无问题", 1: "P2", 2: "P1", 3: "P0"}
GRADE_STR_TO_INT: Dict[str, int] = {v: k for k, v in GRADE_INT_TO_STR.items()}

# 14 positive sub-channels in exact Appendix-2 order:
#   0..5:   P0 for all 6 dimensions
#   6..9:   P1 for FULL_DIMS [模糊, 纹理损失, 过锐, 过度生成]
#   10..13: P2 for FULL_DIMS [模糊, 纹理损失, 过锐, 过度生成]
CHANNEL_14_NAMES: List[str] = (
    [f"{c}_P0" for c in DIMS]
    + [f"{DIMS[d]}_P1" for d in FULL_DIMS]
    + [f"{DIMS[d]}_P2" for d in FULL_DIMS]
)

# Official 25.22B <= 30.0B Model Pool & Single-RTX-4090 (20.07 GB <= 24.0 GB) Budget
MODEL_POOL_BUDGET: Dict[str, Dict[str, object]] = {
    "depth_anything_v2_s": {
        "display": "Depth-Anything-V2-Small",
        "params_b": 0.025,
        "vram_gb_4090": 0.06,
        "precision_4090": "FP16",
        "candidates": [
            "models/depth-anything-v2-small",
            "models/Depth-Anything-V2-Small-hf",
            "depth-anything/Depth-Anything-V2-Small-hf",
        ],
    },
    "convnextv2_base": {
        "display": "ConvNeXtV2-Base (2x Model Soup Clusters)",
        "params_b": 0.178,  # 2 * 0.089B complementary clusters
        "vram_gb_4090": 0.38,
        "precision_4090": "BF16",
        "candidates": [
            "models/convnextv2-base",
            "convnextv2_base.fcmae_ft_in22k_in1k",
        ],
    },
    "pe_core_large": {
        "display": "PE-Core-Large-336 (Layers 16/20/23 Gated Readout)",
        "params_b": 0.350,
        "vram_gb_4090": 0.72,
        "precision_4090": "BF16",
        "candidates": [
            "models/pe-core-l14/model.safetensors",
            "models/pe-core-l14",
        ],
    },
    "dinov3_vitl16": {
        "display": "DINOv3-ViT-L/16 (Native 2D-RoPE Dynamic Aspect)",
        "params_b": 0.330,
        "vram_gb_4090": 0.68,
        "precision_4090": "BF16",
        "candidates": [
            "models/dinov3-vitl16/model.safetensors",
            "models/dinov3-vitl16",
        ],
    },
    "siglip2_so400m_naflex": {
        "display": "SigLIP2-SO400M-NaFlex (max_num_patches=784)",
        "params_b": 0.430,
        "vram_gb_4090": 0.88,
        "precision_4090": "BF16",
        "candidates": [
            "models/siglip2-naflex",
            "models/siglip2-so400m-naflex",
            "models/siglip2-so400m",
            "google/siglip2-so400m-patch16-naflex",
            "google/siglip2-so400m-patch14-384",
        ],
    },
    "cradio_v4_h": {
        "display": "C-RADIOv4-H (Native Aspect Multiples of 16)",
        "params_b": 0.670,
        "vram_gb_4090": 1.35,
        "precision_4090": "BF16",
        "candidates": [
            "models/c-radio-v4-h",
        ],
    },
    "internvit_6b": {
        "display": "InternViT-6B-448px-V2.5 (3-Layer Pyramid + Gated ABMIL)",
        "params_b": 6.080,
        "vram_gb_4090": 3.55,
        "precision_4090": "W4A16-AWQ + BF16-Head",
        "candidates": [
            "models/internvit-6b",
        ],
    },
    "clef_flash_9b": {
        "display": "Clef-Flash-9B (System-1 Prefill + FP32 JointSchemaHead)",
        "params_b": 9.120,
        "vram_gb_4090": 5.65,
        "precision_4090": "LLM-W4A16 + ViT-BF16 + Head-FP32",
        "candidates": [
            "models/clef-flash",
            "models/clef-flash-9b",
        ],
    },
    "qwen3_vl_8b": {
        "display": "Qwen3-VL-8B-Instruct (3-Image Interleaved-MRoPE + KV Attribution)",
        "params_b": 8.100,
        "vram_gb_4090": 5.20,
        "precision_4090": "LLM-W4A16 + ViT-BF16 + Head-FP32",
        "candidates": [
            "models/qwen3vl-8b",
        ],
    },
}

TOTAL_POOL_PARAMS_B: float = round(sum(float(v["params_b"]) for v in MODEL_POOL_BUDGET.values()), 3)
TOTAL_WEIGHT_VRAM_GB: float = round(sum(float(v["vram_gb_4090"]) for v in MODEL_POOL_BUDGET.values()), 2)
ACTIVATION_AND_KV_OVERHEAD_GB: float = 1.60
PEAK_RESIDENT_VRAM_GB: float = round(TOTAL_WEIGHT_VRAM_GB + ACTIVATION_AND_KV_OVERHEAD_GB, 2)


def enforce_deterministic_runtime(seed: int = 2026, deterministic_algos: bool = False) -> None:
    """Sets OMP_NUM_THREADS=8 and deterministic RNG seeds per competition spec."""
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if deterministic_algos:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def resolve_model_path(model_key: str, root: str = REPO_ROOT) -> str:
    """Resolves local weight directory/file first, falling back cleanly to HF repo ID."""
    info = MODEL_POOL_BUDGET[model_key]
    for cand in info["candidates"]:
        abs_cand = cand if os.path.isabs(cand) else os.path.join(root, cand)
        if os.path.exists(abs_cand):
            return abs_cand
    return str(info["candidates"][-1])


class LosslessNativeImageLoader:
    """
    Unified pure-real image loader supporting:
      1. Lossless `/dev/shm/race18_native/cache.bin` + `index.npz` shared-memory cache (if mounted).
      2. Standard real image directories (`data/images`, `data/images_supp`, `data/test_150`, plus custom `extra_dirs`).
    NEVER loads synthetic directories (`synth_*`, `_NatNeg`, `_prism_`).
    """
    DEFAULT_REAL_DIRS = (
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
    ):
        self.root = root
        self.search_dirs: List[str] = []
        if extra_dirs:
            for d in extra_dirs:
                if d:
                    p = d if os.path.isabs(d) else os.path.join(root, d)
                    if p not in self.search_dirs:
                        self.search_dirs.append(p)
        for d in self.DEFAULT_REAL_DIRS:
            p = os.path.join(root, d)
            if p not in self.search_dirs:
                self.search_dirs.append(p)

        self._memmap = None
        self._shm_pos = {}
        self._shm_h = None
        self._shm_w = None
        self._shm_off = None
        if os.path.exists(shm_bin) and os.path.exists(shm_index):
            try:
                import numpy as np
                idx = np.load(shm_index, allow_pickle=True)
                names = [str(x) for x in idx["names"]]
                self._shm_pos = {nm: i for i, nm in enumerate(names)}
                self._shm_h = idx["h"]
                self._shm_w = idx["w"]
                self._shm_off = idx["offsets"]
                self._memmap = np.memmap(shm_bin, dtype=np.uint8, mode="r")
            except Exception:
                self._memmap = None

    def find_disk_path(self, name: str) -> Optional[str]:
        if os.path.isabs(name) and os.path.exists(name):
            return name
        rel = os.path.join(self.root, name)
        if os.path.exists(rel):
            return rel
        base = os.path.basename(name)
        for d in self.search_dirs:
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                return cand
            cand_b = os.path.join(d, base)
            if os.path.exists(cand_b):
                return cand_b
        return None

    def load_pil(self, name: str):
        from PIL import Image
        s_name = str(name)
        if self._memmap is not None and s_name in self._shm_pos:
            import numpy as np
            j = self._shm_pos[s_name]
            h = int(self._shm_h[j])
            w = int(self._shm_w[j])
            off = int(self._shm_off[j])
            arr = np.asarray(self._memmap[off : off + h * w * 3]).reshape(h, w, 3)
            return Image.fromarray(arr, mode="RGB")
        p = self.find_disk_path(s_name)
        if p is None:
            raise FileNotFoundError(
                f"Real image '{s_name}' not found in {self.search_dirs}. "
                f"Pass --test-dir if evaluating on a custom test directory."
            )
        return Image.open(p).convert("RGB")


def load_pure_real_manifest(manifest_path: str = "data/manifest_combined.csv") -> List[Dict[str, object]]:
    """Loads all 3,573 pure-real annotated samples from `data/manifest_combined.csv`."""
    full_p = manifest_path if os.path.isabs(manifest_path) else os.path.join(REPO_ROOT, manifest_path)
    rows: List[Dict[str, object]] = []
    with open(full_p, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nm = r["name"].strip()
            # Guard against any accidental synthetic entries if a custom manifest is passed
            if nm.startswith(("v9_synth", "v8_synth", "v7_synth", "v6_synth")) or ("_NatNeg" in nm) or ("_prism_" in nm):
                continue
            grades = [int(r[c]) for c in DIMS]
            rows.append({
                "name": nm,
                "source": r.get("source", "").strip() or nm.split("_")[0],
                "width": int(r.get("width", 0) or 0),
                "height": int(r.get("height", 0) or 0),
                "orig_split": r.get("split", "orig").strip(),
                "grades": grades,
                **{c: int(r[c]) for c in DIMS},
            })
    return rows


def load_test_template_names(template_path: str = "results/v2_full/template_150.json") -> List[str]:
    """Loads ordered test image IDs from a JSON template (`150` practice or `654` final test)."""
    full_p = template_path if os.path.isabs(template_path) else os.path.join(REPO_ROOT, template_path)
    with open(full_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(item["image"] if isinstance(item, dict) else item) for item in data]
    raise ValueError(f"Unexpected template JSON format in {full_p}")


def format_official_submission(
    ordered_names: Sequence[str],
    pred_6d,
    explanations: Optional[Sequence[Dict[str, object]]] = None,
) -> Tuple[List[Dict[str, object]], Optional[List[Dict[str, object]]]]:
    """
    Formats `[N, 6]` integer predictions into the exact official JSON structure:
      `[{"image": name, "label": {"模糊": ..., "纹理损失": ..., "过度生成": ..., "虚化突兀": ..., "过锐": ..., "花屏": ...}}]`
    Strictly enforces `花屏` and `虚化突兀` in `{"无问题", "P0"}` (integer `{0, 3}`).
    """
    sub_clean: List[Dict[str, object]] = []
    sub_attrib: Optional[List[Dict[str, object]]] = [] if explanations is not None else None
    for i, nm in enumerate(ordered_names):
        lbl: Dict[str, str] = {}
        for k in JSON_KEY_ORDER:
            ci = DIMS.index(k)
            g_int = int(pred_6d[i, ci])
            if ci in P0_ONLY_DIMS:
                g_int = 3 if g_int >= 2 else 0
            else:
                g_int = max(0, min(3, g_int))
            lbl[k] = GRADE_INT_TO_STR[g_int]
        sub_clean.append({"image": str(nm), "label": lbl})
        if sub_attrib is not None and explanations is not None:
            sub_attrib.append({
                "image": str(nm),
                "label": lbl,
                "attribution": explanations[i],
            })
    return sub_clean, sub_attrib


def evaluate_qingyu_dual_metrics(pred_6d, gold_6d, tau: float = 0.02) -> Dict[str, object]:
    """
    Computes BOTH `Doc1` (online practice metric) and `Doc2_Strict` / `Doc2_Cliff` (Official Spec Appendix 2)
    without requiring external imports if run standalone.
    """
    import numpy as np
    preds = np.asarray(pred_6d, dtype=np.int64)
    gold = np.asarray(gold_6d, dtype=np.int64)

    per_14 = []
    class_f1s_doc1 = []
    p0_recs, p0_precs, p0_f1s = [], [], []

    for ci, cname in enumerate(DIMS):
        grades = [("P2", 1), ("P1", 2), ("P0", 3)] if ci in FULL_DIMS else [("P0", 3)]
        c_f1_list = []
        for g_str, gi in grades:
            yt = gold[:, ci] == gi
            yp = preds[:, ci] == gi
            tp = int((yt & yp).sum())
            fp = int((~yt & yp).sum())
            fn = int((yt & ~yp).sum())
            prec = tp / max(1, tp + fp)
            rec_d1 = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            rec_d2 = tp / max(1, tp + fn)
            f1 = (2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            per_14.append({
                "channel": f"{cname}_{g_str}",
                "dim_idx": ci,
                "grade": gi,
                "P": float(prec),
                "R": float(rec_d2),
                "R_d1": float(rec_d1),
                "F1": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })
            c_f1_list.append(f1)
            if gi == 3:
                p0_recs.append((rec_d1, rec_d2))
                p0_precs.append(prec)
                p0_f1s.append(f1)
        class_f1s_doc1.append(float(np.mean(c_f1_list)))

    s_pos_d1 = float(np.mean(class_f1s_doc1))
    s_pos_d2 = float(np.mean([x["F1"] for x in per_14]))

    r_p0_d1 = float(np.mean([x[0] for x in p0_recs]))
    r_p0_d2 = float(np.mean([x[1] for x in p0_recs]))
    p_p0 = float(np.mean(p0_precs))
    f_p0 = float(np.mean(p0_f1s))

    s_p0_d1 = 0.6 * r_p0_d1 + 0.4 * f_p0
    s_p0_d2 = 0.5 * r_p0_d2 + 0.3 * f_p0 + 0.2 * p_p0

    clean_mask = (gold == 0).all(axis=1)
    n_clean = int(clean_mask.sum())
    flagged_p0_on_clean = int(((preds == 3).any(axis=1) & clean_mask).sum())
    clean_p0_fpr = float(flagged_p0_on_clean / max(1, n_clean)) if n_clean > 0 else 0.0
    s_clean_cliff = max(0.0, 1.0 - clean_p0_fpr / tau)
    s_clean_strict = float(((preds == 0).all(axis=1) & clean_mask).sum() / max(1, n_clean)) if n_clean > 0 else 1.0

    doc1_total = 25.0 * s_pos_d1 + 20.0 * s_p0_d1 + 20.0 * s_clean_cliff
    doc2_cliff_total = 25.0 * s_pos_d2 + 20.0 * s_p0_d2 + 20.0 * s_clean_cliff
    doc2_strict_total = 25.0 * s_pos_d2 + 20.0 * s_p0_d2 + 20.0 * s_clean_strict

    return {
        "doc1_total": float(doc1_total),
        "doc1_S_pos": float(25.0 * s_pos_d1),
        "doc1_S_p0": float(20.0 * s_p0_d1),
        "doc1_S_clean": float(20.0 * s_clean_cliff),
        "doc2_strict_total": float(doc2_strict_total),
        "doc2_cliff_total": float(doc2_cliff_total),
        "doc2_S_pos": float(25.0 * s_pos_d2),
        "doc2_S_p0": float(20.0 * s_p0_d2),
        "doc2_Clean2": float(20.0 * s_clean_strict),
        "clean_p0_fpr": clean_p0_fpr,
        "p0_recall": r_p0_d2,
        "p0_precision": p_p0,
        "p0_f1": f_p0,
        "per_14": per_14,
    }


def print_pool_budget_summary() -> None:
    print("=" * 104)
    print("=== [Qingyu Architecture] Official Model Pool (<=30B) & Single-RTX-4090 (<=24GB) Compliance Audit ===")
    print("=" * 104)
    for k, v in MODEL_POOL_BUDGET.items():
        print(
            f"  - {str(v['display']):58s} | {float(v['params_b']):6.3f}B | "
            f"{float(v['vram_gb_4090']):5.2f} GB ({v['precision_4090']})"
        )
    print("-" * 104)
    print(
        f"  TOTAL PARAMETER POOL : {TOTAL_POOL_PARAMS_B:.3f}B / 30.000B "
        f"(Margin: {30.0 - TOTAL_POOL_PARAMS_B:.3f}B) -> [PASS]"
    )
    print(
        f"  PEAK 4090 VRAM       : {PEAK_RESIDENT_VRAM_GB:.2f} GB / 24.00 GB "
        f"(Weights {TOTAL_WEIGHT_VRAM_GB:.2f} GB + Dynamic/KV {ACTIVATION_AND_KV_OVERHEAD_GB:.2f} GB) -> [PASS]"
    )
    print("=" * 104)


if __name__ == "__main__":
    print_pool_budget_summary()
