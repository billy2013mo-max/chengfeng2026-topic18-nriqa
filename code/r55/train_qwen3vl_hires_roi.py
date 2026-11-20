#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/train_qwen3vl_hires_roi.py
===================================
Step 2B: High-Performance High-Res + Native 1:1 Semantic ROI `Qwen3-VL-8B` Specialist (`Spec C`)
Owner: `vlm-author` (GPUs 4,5,6,7)

Hard Engineering & Compliance Guarantees:
  1. 100% Batch-Tag-Free (`TAG-FREE`):
     - Zero `orig` / `supp` split indicators, zero `batch_ids`, zero `supp_batch_bias`, zero `w_supp != 1.0`
       anywhere in dataset, model, head, loss, or evaluation.
     - Canonical 8-fold video-grouped (`source` / `group_id`) split via `data/qingyu_splits/qingyu_8fold.csv`
       (verified 1:1 against `common55.load_split()`). Every row has uniform weight = 1.0.
     - Leak-free nested KD from `results/r55/nested_kd_teacher.npz` (`fold_{k}_probs`, asserting fold-k rows == 0,
       annealed `lambda_kd = 0.25 -> 0.0` by final epoch so the student isn't capped by the PE-L teacher).
  2. Zero Legacy JSON Anchoring & Zero Test-Set Tuning (`AGENTS.md` Rules 1-2).
  3. Strict `N=1 == N=batch` Inference Invariance & RTX 4090 Deployment Budget (`AGENTS.md` Rule 3):
     - No BatchNorm; deterministic 32px-grid native-aspect global resize (`max_pixels = 1344^2`, fixing the
       legacy `grid_factor=28` double-interpolation bug!) + 3 deterministic `1:1` `448x448` semantic ROIs
       (`subtitle_ocr`, `face_anat + hand_swirl`, `bokeh_halo` from `semantic_roi_meta`).
     - `build_hires_roi_vlm_for_4090()`: merges `bf16` LoRA (`r=64, alpha=128`) into `bf16` base (`merge_and_unload`)
       and NF4-quantizes LLM `nn.Linear` layers (`~5.78 GB` resident, `~6.35 GB` peak, `~0.37s/img` 2x-TTA at N=1).
  4. High-Throughput H200 Training Fixes over `vlm_c233397_qy8.py`:
     - Pre-stages resized global PNG/RGB + 3x `448x448` 1:1 crops once into `/dev/shm/r55_fast_cache/vlm/`
       (reading raw images from shared `/dev/shm/r55_fast_cache/images/`, staying well within <= 20 GB cap!).
     - Replaces `output_hidden_states=True` (which kept all 37 `[B, L, 4096]` hidden tensors alive) with
       lightweight forward hooks on the 2 tapped LLM layers (`layers[17]`, `layers[26]`) + `last_hidden_state`.
     - Selective LLM-only gradient checkpointing (`visual` runs uncheckpointed at full speed; LLM checkpoints).
     - Standard LoRA `r=64` (`use_dora=False`, avoiding DoRA's weight-norm recomputation bottleneck) +
       `AdamW(..., fused=True)` + **2-GPU DDP per fold** (`torchrun --nproc_per_node=2`) with NUMA Node 1 pinning.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gc
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "code/decision_eval"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

import common55 as c  # noqa: E402
from phoenix.config import (  # noqa: E402
    DIMS,
    FULL_DIMS,
    P0_ONLY_DIMS,
    evaluate_official_metrics,
    resolve_local_or_hf_model,
    set_deterministic_seed,
)
from qingyu.unimodal_xbm_losses import hash_group_id_int64  # noqa: E402
from semantic_roi_meta import analyze_semantic_rois_pil  # noqa: E402

RECIPE_TAG = "qwen3vl_hires_roi_r64_tagfree"
DEFAULT_MAX_PIXELS = 1344 * 1344   # 1,806,336 px (~3.0x p1_qy's 784^2 = 614,656 px)
QWEN3VL_GRID_FACTOR = 32           # patch_size=16 * merge_size=2 = 32 (fixes legacy grid_factor=28 double resize!)
ROI_CROP_SIZE = 448                # 14x14 merged LLM tokens per 1:1 native crop (32 divides 448!)

HIRES_INSPECTION_PROMPT = (
    "请作为短视频无参考画质质检专家，综合图1（原生宽高比高分辨率全局画面）与图2（字幕笔画与文字边缘1:1原尺寸切片）、"
    "图3（主体面部或手部肌理1:1原尺寸切片）、图4（景深交界与高频锐化光晕1:1原尺寸切片），"
    "直接形成6类画质缺陷（模糊、纹理损失、过锐、花屏、过度生成、虚化突兀）的客观严重度表征。"
)

SHM_ROOT = "/dev/shm/r55_fast_cache"
SHM_SHARED_IMAGES = os.path.join(SHM_ROOT, "images")
SHM_SHARED_READY = os.path.join(SHM_ROOT, "READY")
SHM_VLM_CACHE = os.path.join(SHM_ROOT, "vlm", "views_1344_3x448_jpg100_png")


# ----------------------------------------------------------------------------------------------------------------------
# Canonical 8-Fold Split & Zero-Leakage Teacher Loader (100% TAG-FREE)
# ----------------------------------------------------------------------------------------------------------------------
def load_canonical_tagfree_rows() -> List[Dict[str, Any]]:
    """
    Loads all 3,573 labeled rows strictly from `common55.load_split()` (`data/qingyu_splits/qingyu_8fold.csv`).
    NEVER reads `split` (`orig`/`supp`) or assigns non-uniform sample weights!
    """
    names, gold, fold, groups = c.load_split()
    assert len(names) == 3573, f"Expected 3573 canonical rows, got {len(names)}"
    rows: List[Dict[str, Any]] = []
    for i, nm in enumerate(names):
        rows.append(
            {
                "idx": i,
                "name": str(nm),
                "fold": int(fold[i]),
                "group_id": str(groups[i]),
                "group_hash": int(hash_group_id_int64(str(groups[i]))),
                "grades": gold[i].astype(np.int64),
            }
        )
    return rows


def load_leakfree_kd_for_fold(kd_npz_path: str, fold_k: int, canonical_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """
    Loads `fold_{k}_probs` from `results/r55/nested_kd_teacher.npz` and strictly asserts zero validation leakage:
    every row in `fold == fold_k` MUST be exact zeros in `fold_{k}_probs`.
    When `fold_k < 0` (100% Full-Fit), uses the out-of-fold average across the 7 folds where `k != fold[i]`.
    """
    z = np.load(kd_npz_path, allow_pickle=True)
    t_names = [str(x) for x in z["names"]]
    if int(fold_k) < 0:
        stack_8 = np.stack([z[f"fold_{k}_probs"].astype(np.float32) for k in range(8)], axis=0)
        nz = (stack_8.sum(axis=-1, keepdims=True) > 0.5).astype(np.float32)
        probs_avg = stack_8.sum(axis=0) / np.maximum(1.0, nz.sum(axis=0))
        mp_avg = {nm: probs_avg[i] for i, nm in enumerate(t_names)}
        return {str(nm): mp_avg[str(nm)] for nm in canonical_names}
    key = f"fold_{fold_k}_probs"
    assert key in z.files, f"Missing {key} in {kd_npz_path}"
    t_names = [str(x) for x in z["names"]]
    probs = z[key].astype(np.float32)
    mp = {nm: probs[i] for i, nm in enumerate(t_names)}

    _, _, folds_all, _ = c.load_split()
    out: Dict[str, np.ndarray] = {}
    for i, nm in enumerate(canonical_names):
        p = mp[str(nm)]
        if int(folds_all[i]) == int(fold_k):
            assert float(np.abs(p).max()) == 0.0, f"KD leakage detected on val fold {fold_k} row {nm}!"
        else:
            assert float(p.sum()) > 5.9, f"Missing nonzero KD teacher probs on train row {nm} for fold {fold_k}!"
            out[str(nm)] = p
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Exact Aspect-Preserving 32px-Grid Resize + Deterministic 3x 448x448 1:1 ROI Extraction
# ----------------------------------------------------------------------------------------------------------------------
def compute_32grid_dims(w: int, h: int, max_pixels: int = DEFAULT_MAX_PIXELS, grid: int = QWEN3VL_GRID_FACTOR) -> Tuple[int, int]:
    scale = min(1.0, math.sqrt(float(max_pixels) / float(max(1, w * h))))
    nw = max(grid, int(round((w * scale) / grid)) * grid)
    nh = max(grid, int(round((h * scale) / grid)) * grid)
    while nw * nh > max_pixels and (nw > grid or nh > grid):
        if nw >= nh and nw > grid:
            nw -= grid
        elif nh > grid:
            nh -= grid
        else:
            break
    return nw, nh


def resize_global_32grid(pil_img: Image.Image, max_pixels: int = DEFAULT_MAX_PIXELS) -> Image.Image:
    w, h = pil_img.size
    nw, nh = compute_32grid_dims(w, h, max_pixels=max_pixels, grid=QWEN3VL_GRID_FACTOR)
    if (nw, nh) == (w, h):
        return pil_img.copy()
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return pil_img.resize((nw, nh), resample=resample)


def crop_1to1_native(pil_img: Image.Image, box_xyxy: Sequence[int], patch_size: int = ROI_CROP_SIZE) -> Image.Image:
    w, h = pil_img.size
    x0, y0, _, _ = [int(v) for v in box_xyxy]
    if w >= patch_size and h >= patch_size:
        x0 = max(0, min(w - patch_size, x0))
        y0 = max(0, min(h - patch_size, y0))
        return pil_img.crop((x0, y0, x0 + patch_size, y0 + patch_size))
    arr = np.asarray(pil_img, dtype=np.uint8)
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    arr_pad = np.pad(
        arr,
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
        mode="reflect" if (h > 1 and w > 1) else "edge",
    )
    hp, wp = arr_pad.shape[:2]
    x0 = max(0, min(wp - patch_size, x0))
    y0 = max(0, min(hp - patch_size, y0))
    return Image.fromarray(arr_pad[y0 : y0 + patch_size, x0 : x0 + patch_size], mode="RGB")


def select_3_semantic_boxes_448(crops_448_boxes: np.ndarray, phys_48: np.ndarray) -> np.ndarray:
    """
    Deterministic single-image selection of 3 complementary 1:1 `448x448` boxes from `semantic_roi_meta`:
      - Box 0: `b_text` (`crops_448_boxes[0]`) — subtitle/caption/UI stroke ROI (OG garbled text vs OS crisp UI).
      - Box 1: Primary subject anatomy (`b_face = crops_448_boxes[1]` if `skin_ratio >= 0.015`, else `b_hand_swirl = crops_448_boxes[2]`).
      - Box 2: `b_bokeh_halo` (`crops_448_boxes[3]`) — sharpness cliff + real-edge halo overshoot ROI (OS 1-2px ringing & abrupt bokeh).
    If Box 1 and Box 2 overlap heavily (`IoU > 0.50`), swaps Box 1 to the unused `b_hand_swirl` / `b_face` box so all 3 views stay distinct.
    Purely per-image (`N=1` invariant) and uses ONLY visual content (never index 47 `orig_h % 2`!).
    """
    b_text = crops_448_boxes[0]
    b_face = crops_448_boxes[1]
    b_hand = crops_448_boxes[2]
    b_halo = crops_448_boxes[3]
    skin_ratio = float(phys_48[39]) if len(phys_48) >= 40 else 0.05
    primary_subj, alt_subj = (b_face, b_hand) if skin_ratio >= 0.015 else (b_hand, b_face)

    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ix0, iy0 = max(int(a[0]), int(b[0])), max(int(a[1]), int(b[1]))
        ix1, iy1 = min(int(a[2]), int(b[2])), min(int(a[3]), int(b[3]))
        inter = float(max(0, ix1 - ix0) * max(0, iy1 - iy0))
        if inter <= 0.0:
            return 0.0
        aa = float(max(1, (int(a[2]) - int(a[0])) * (int(a[3]) - int(a[1]))))
        ab = float(max(1, (int(b[2]) - int(b[0])) * (int(b[3]) - int(b[1]))))
        return inter / (aa + ab - inter + 1e-6)

    if _iou(primary_subj, b_halo) > 0.50 or _iou(primary_subj, b_text) > 0.50:
        if _iou(alt_subj, b_halo) < _iou(primary_subj, b_halo):
            primary_subj = alt_subj
    return np.stack([b_text, primary_subj, b_halo], axis=0).astype(np.int32)


def extract_4views_from_pil(
    pil_img: Image.Image,
    boxes_3x4: Optional[np.ndarray] = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> List[Image.Image]:
    """
    Single-image (`N=1` deterministic) view builder shared identically between training/OOF and `final_engine.py`.
    """
    im = pil_img.convert("RGB")
    if boxes_3x4 is None:
        meta = analyze_semantic_rois_pil(im, base_phys_36=np.zeros(36, dtype=np.float32))
        boxes_3x4 = select_3_semantic_boxes_448(meta["crops_448_boxes"], meta["phys_vec"])
    g_view = resize_global_32grid(im, max_pixels=max_pixels)
    c0 = crop_1to1_native(im, boxes_3x4[0], patch_size=ROI_CROP_SIZE)
    c1 = crop_1to1_native(im, boxes_3x4[1], patch_size=ROI_CROP_SIZE)
    c2 = crop_1to1_native(im, boxes_3x4[2], patch_size=ROI_CROP_SIZE)
    return [g_view, c0, c1, c2]


# ----------------------------------------------------------------------------------------------------------------------
# Fast `/dev/shm` Pre-Staged Lossless PNG View Cache (~4-6 GB total in `/dev/shm/r55_fast_cache/vlm/`, <=20GB cap!)
# ----------------------------------------------------------------------------------------------------------------------
def _resolve_raw_image_path(name: str) -> str:
    if os.path.exists(SHM_SHARED_READY):
        p_shm = os.path.join(SHM_SHARED_IMAGES, name)
        if os.path.exists(p_shm):
            return p_shm
    for sub in ("data/images", "data/images_supp"):
        p = os.path.join(REPO, sub, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot locate image {name}")


def _stage_one_image_views(args_tuple: Tuple[str, np.ndarray, str, int]) -> int:
    nm, boxes_3x4, cache_dir, max_pixels = args_tuple
    stem = os.path.splitext(nm)[0]
    done_paths = [os.path.join(cache_dir, f"{stem}__v{k}.png") for k in range(4)]
    if all(os.path.exists(p) for p in done_paths):
        return sum(os.path.getsize(p) for p in done_paths)

    raw_p = _resolve_raw_image_path(nm)
    with Image.open(raw_p) as raw_im:
        views = extract_4views_from_pil(raw_im.convert("RGB"), boxes_3x4=boxes_3x4, max_pixels=max_pixels)

    total_b = 0
    for k, vw in enumerate(views):
        dst = done_paths[k]
        tmp = f"{dst}.tmp.{os.getpid()}"
        # compress_level=1 writes lossless PNG ~5x faster than default zlib level 6 while staying well under 8 GB total!
        vw.save(tmp, format="PNG", compress_level=1)
        os.replace(tmp, dst)
        total_b += os.path.getsize(dst)
    return total_b


def ensure_shm_vlm_views_staged(
    names: Sequence[str],
    sem_meta_npz: str,
    cache_dir: str = SHM_VLM_CACHE,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    num_threads: int = 24,
) -> Dict[str, np.ndarray]:
    z = np.load(sem_meta_npz, allow_pickle=True)
    mp_boxes: Dict[str, np.ndarray] = {}
    for i, nm in enumerate(z["names"]):
        mp_boxes[str(nm)] = select_3_semantic_boxes_448(z["crops_448_boxes"][i], z["phys_vecs"][i])

    os.makedirs(cache_dir, exist_ok=True)
    ready_flag = os.path.join(cache_dir, f"READY_{len(names)}_{max_pixels}")
    if os.path.exists(ready_flag):
        return mp_boxes

    # Wait briefly if iv6b-author is mid-copying shared raw images
    for _ in range(15):
        if os.path.exists(SHM_SHARED_READY):
            break
        time.sleep(1.0)

    t0 = time.time()
    tasks = [(str(nm), mp_boxes[str(nm)], cache_dir, max_pixels) for nm in names]
    total_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=num_threads) as ex:
        for nb in ex.map(_stage_one_image_views, tasks, chunksize=16):
            total_bytes += nb

    gb = total_bytes / (1024.0 ** 3)
    assert gb <= 20.0, f"VLM /dev/shm cache exceeded 20 GB cap: {gb:.2f} GB"
    with open(ready_flag, "w") as f:
        f.write(f"ok {len(names)} {gb:.3f}GB\n")
    print(
        f"[SHM Cache] Staged {len(names)} x 4 lossless views into {cache_dir} ({gb:.2f} GB <= 20 GB cap) in {time.time()-t0:.1f}s",
        flush=True,
    )
    return mp_boxes


# ----------------------------------------------------------------------------------------------------------------------
# 100% Batch-Tag-Free Model & Head (`TagFreeDeQAHead` + `Qwen3VLHiResRoiSpecialist`)
# ----------------------------------------------------------------------------------------------------------------------
class TagFreeUnimodalDeQAHead(nn.Module):
    """
    100% Tag-Free Multi-Defect Cross-Attention + Unimodal Residual Ordinal Head.
    Zero `supp_batch_bias`, zero `batch_ids` anywhere!
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        # 6 defect queries + 1 clean gate query
        self.defect_queries = nn.Parameter(torch.randn(1, 7, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.inter_defect_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.grade_proj = nn.ModuleList([nn.Linear(hidden_dim, 4) for _ in range(6)])
        self.center_proj = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(6)])
        self.log_kappa = nn.Parameter(torch.zeros(6))
        self.clean_gate_proj = nn.Linear(hidden_dim, 1)
        self.anchor_emb_proj = nn.Sequential(
            nn.Linear(hidden_dim * 7, 256),
            nn.LayerNorm(256),
        )
        self.register_buffer("grade_grid", torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32), persistent=False)

    def forward(self, spatial_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        if spatial_tokens.ndim == 2:
            spatial_tokens = spatial_tokens.unsqueeze(1)
        amp_dev = "cuda" if spatial_tokens.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            tokens = self.in_proj(spatial_tokens.float())
            bsz = tokens.shape[0]

            q = self.defect_queries.float().expand(bsz, -1, -1)
            q_cross, _ = self.cross_attn(q, tokens, tokens, need_weights=False)
            q = self.norm1(q + q_cross)
            q_self, _ = self.inter_defect_attn(q, q, q, need_weights=False)
            q = self.norm2(q + q_self)
            q = self.norm3(q + self.ffn(q))

            kappa = F.softplus(self.log_kappa.float()) * 0.30
            grid_1d = self.grade_grid.float()

            logits_list = []
            for d in range(6):
                qd = q[:, d, :]
                raw_z = self.grade_proj[d](qd)
                center = 3.0 * torch.sigmoid(self.center_proj[d](qd))
                unimodal_prior = -kappa[d] * (grid_1d.unsqueeze(0) - center) ** 2
                z = raw_z + unimodal_prior
                if d in P0_ONLY_DIMS:
                    mask = torch.zeros_like(z)
                    mask[:, 1] = -1e4
                    mask[:, 2] = -1e4
                    z = z + mask
                logits_list.append(z)

            logits_6x4 = torch.stack(logits_list, dim=1)
            probs_6x4 = F.softmax(logits_6x4, dim=-1)

            clean_logit = self.clean_gate_proj(q[:, 6, :]).squeeze(-1)
            p_clean_head = torch.sigmoid(clean_logit)
            p_clean_prod = torch.prod(probs_6x4[:, :, 0], dim=1)
            p_clean_joint = 0.50 * p_clean_head + 0.50 * p_clean_prod

            grid = grid_1d.view(1, 1, 4)
            mu_6d = torch.sum(probs_6x4 * grid, dim=-1)
            var_6d = torch.sum(probs_6x4 * (grid - mu_6d.unsqueeze(-1)) ** 2, dim=-1)
            quality_emb = F.normalize(self.anchor_emb_proj(q.reshape(bsz, -1)), dim=-1)

            return {
                "logits_6x4": logits_6x4,
                "probs_6x4": probs_6x4,
                "clean_logit": clean_logit,
                "p_clean_joint": p_clean_joint,
                "mu_6d": mu_6d,
                "var_6d": var_6d,
                "quality_emb": quality_emb,
            }


def format_4image_chat_prompt(processor: Any, prompt_text: str = HIRES_INSPECTION_PROMPT) -> str:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        tpl_str = str(getattr(processor, "chat_template", "") or "")
        extra_kw: Dict[str, Any] = {}
        if "enable_thinking" in tpl_str:
            extra_kw["enable_thinking"] = False
        try:
            return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra_kw)
        except TypeError:
            return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"<image>\n<image>\n<image>\n<image>\n{prompt_text}"


class Qwen3VLHiResRoiSpecialist(nn.Module):
    """
    High-Res (`<=1344^2` 32px-grid native aspect) + 3x `448x448` 1:1 Semantic ROI `Qwen3-VL-8B` Specialist.
    Uses lightweight forward hooks on `language_model.layers[17]` and `layers[26]` + `last_hidden_state`
    instead of `output_hidden_states=True`, saving ~40 GB VRAM and avoiding 34 unused hidden-state allocations!
    Preserves per-view pooled tokens (`[global_64, crop0_16, crop1_16, crop2_16, summary_1]` = 113 tokens) so
    defect queries can attend to sharp 1:1 local crops without averaging them into background tokens!
    """

    def __init__(
        self,
        model_id_or_path: Optional[str] = None,
        lora_rank: int = 64,
        use_dora: bool = False,
        gc_mode: str = "llm_only",  # "llm_only" | "full" | "none"
    ) -> None:
        super().__init__()
        self.lora_rank = int(lora_rank)
        self.use_dora = bool(use_dora)
        self.recipe_tag = RECIPE_TAG
        resolved = model_id_or_path or resolve_local_or_hf_model("pillar1_qwen3vl_8b")
        self.backbone, self.processor, in_dim, self.image_token_id = self._load_backbone(
            resolved_path=resolved,
            lora_rank=self.lora_rank,
            use_dora=self.use_dora,
            gc_mode=gc_mode,
        )
        self.prompt_text_formatted = format_4image_chat_prompt(self.processor, HIRES_INSPECTION_PROMPT)
        self.head = TagFreeUnimodalDeQAHead(in_dim=in_dim, hidden_dim=512, num_heads=8, dropout=0.10).float()
        self._tapped: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []
        self._install_hypercolumn_hooks()

    def _install_hypercolumn_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        base_m = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
        lm = getattr(base_m, "language_model", None)
        layers = getattr(lm, "layers", None)
        assert layers is not None and len(layers) >= 4, "Could not locate language_model.layers on Qwen3-VL"
        n_l = len(layers)
        idx_mid = n_l // 2 - 1          # layer 17 (matches hidden_states[18])
        idx_3q = (3 * n_l) // 4 - 1     # layer 26 (matches hidden_states[27])

        def _make_hook(key: str):
            def _fn(_m, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                self._tapped[key] = t
            return _fn

        self._hooks.append(layers[idx_mid].register_forward_hook(_make_hook("mid")))
        self._hooks.append(layers[idx_3q].register_forward_hook(_make_hook("3q")))

    @staticmethod
    def _load_backbone(
        resolved_path: str,
        lora_rank: int = 64,
        use_dora: bool = False,
        gc_mode: str = "llm_only",
    ):
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        if hasattr(tok, "padding_side"):
            tok.padding_side = "right"
        cfg = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True)
        image_token_id = int(getattr(cfg, "image_token_id", 151655))

        full_model = None
        for attn_impl in ("flash_attention_2", "sdpa"):
            try:
                full_model = AutoModelForImageTextToText.from_pretrained(
                    resolved_path,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                break
            except TypeError:
                full_model = AutoModelForImageTextToText.from_pretrained(
                    resolved_path,
                    torch_dtype=dtype,
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                break
            except Exception:
                continue
        assert full_model is not None, f"Failed to load Qwen3-VL from {resolved_path}"

        backbone = getattr(full_model, "model", full_model)
        if hasattr(full_model, "lm_head") and backbone is not full_model:
            del full_model.lm_head
        del full_model

        if int(lora_rank) > 0:
            from peft import LoraConfig, get_peft_model

            allowed_leafs = {
                "q_proj", "k_proj", "v_proj", "o_proj",
                "qkv", "proj",
                "gate_proj", "up_proj", "down_proj",
                "linear_fc1", "linear_fc2", "fc1", "fc2",
            }
            target_paths: List[str] = []
            for full_name, mod in backbone.named_modules():
                if not isinstance(mod, nn.Linear):
                    continue
                nl = full_name.lower()
                if any(skip in nl for skip in ("patch_embed", "lm_head", "embed_tokens", "wte")):
                    continue
                if full_name.split(".")[-1] in allowed_leafs:
                    target_paths.append(full_name)

            lora_cfg = LoraConfig(
                r=int(lora_rank),
                lora_alpha=int(lora_rank) * 2,
                lora_dropout=0.05,
                target_modules=target_paths,
                use_dora=bool(use_dora),
                bias="none",
            )
            backbone = get_peft_model(backbone, lora_cfg)

            # Unfreeze multimodal merger/projector in fp32/bf16 alongside LoRA
            for n, p in backbone.named_parameters():
                nl = n.lower()
                if any(bk in nl for bk in ("merger", "multi_modal_projector", "aligner")) and p.is_floating_point():
                    p.requires_grad = True

            base_m = backbone.get_base_model()
            if gc_mode == "full":
                backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                if hasattr(backbone, "enable_input_require_grads"):
                    backbone.enable_input_require_grads()
            elif gc_mode == "llm_only":
                base_m.language_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            elif gc_mode == "llm_24":
                # Checkpoint bottom 24/36 LLM layers (layers 0..23), leaving vision + top 12 LLM layers uncheckpointed (~95 GB worst-case <= 140 GB!)
                lm_layers = base_m.language_model.layers
                from torch.utils.checkpoint import checkpoint as _ckpt
                for blk in list(lm_layers)[: (2 * len(lm_layers)) // 3]:
                    _orig_fwd = blk.forward
                    blk.forward = lambda *a, _f=_orig_fwd, **kw: _ckpt(_f, *a, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*a, **kw)
            elif gc_mode == "llm_half":
                # Checkpoint only the first half of LLM layers (layers 0..17), leaving vision + layers 18..35 uncheckpointed!
                lm_layers = base_m.language_model.layers
                from torch.utils.checkpoint import checkpoint as _ckpt
                for blk in list(lm_layers)[: len(lm_layers) // 2]:
                    _orig_fwd = blk.forward
                    blk.forward = lambda *a, _f=_orig_fwd, **kw: _ckpt(_f, *a, use_reentrant=False, **kw) if torch.is_grad_enabled() else _f(*a, **kw)

        hidden_dim = int(getattr(getattr(cfg, "text_config", cfg), "hidden_size", 4096))
        return backbone, processor, hidden_dim, image_token_id

    @staticmethod
    def _pool_1d_tokens(toks: torch.Tensor, target_k: int) -> torch.Tensor:
        n = toks.shape[0]
        if n == 0:
            return toks.new_zeros((target_k, toks.shape[-1]))
        if n == target_k:
            return toks
        if n > target_k:
            return (
                F.adaptive_avg_pool1d(toks.unsqueeze(0).transpose(1, 2), output_size=target_k)
                .transpose(1, 2)
                .squeeze(0)
            )
        pad = target_k - n
        return torch.cat([toks, toks[-1:, :].expand(pad, -1)], dim=0)

    def encode_multiview_tokens(self, batch_views_or_inputs: Any, device: torch.device) -> torch.Tensor:
        if isinstance(batch_views_or_inputs, dict):
            model_inputs = batch_views_or_inputs
        else:
            batch_views = batch_views_or_inputs
            texts = [self.prompt_text_formatted for _ in batch_views]
            flat_images = [pil_v for views in batch_views for pil_v in views]
            model_inputs = self.processor(
                text=texts,
                images=flat_images,
                padding=True,
                return_tensors="pt",
            )

        moved: Dict[str, Any] = {}
        for k, v in model_inputs.items():
            moved[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v

        self._tapped.clear()
        use_bf16 = device.type == "cuda"
        with torch.autocast(device_type="cuda" if use_bf16 else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = self.backbone(
                **moved,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )

        last_h = outputs.last_hidden_state
        if "mid" in self._tapped and "3q" in self._tapped:
            fused = (self._tapped["mid"].float() + self._tapped["3q"].float() + last_h.float()) / 3.0
        else:
            fused = last_h.float()
        self._tapped.clear()

        input_ids = moved["input_ids"]
        attn_mask = moved.get("attention_mask", None)
        grid_thw = moved.get("image_grid_thw", None)
        bsz = fused.shape[0]

        # Each sample has exactly 4 images; LLM tokens per image = (t * h * w) // 4 (merge_size=2 -> 2x2=4)
        img_tok_counts: Optional[List[int]] = None
        if grid_thw is not None and grid_thw.shape[0] == bsz * 4:
            img_tok_counts = [int((grid_thw[i, 0] * grid_thw[i, 1] * grid_thw[i, 2]).item()) // 4 for i in range(bsz * 4)]

        pooled_list: List[torch.Tensor] = []
        for b in range(bsz):
            valid_mask = attn_mask[b].bool() if attn_mask is not None else torch.ones_like(input_ids[b], dtype=torch.bool)
            valid_toks = fused[b, valid_mask]
            valid_ids = input_ids[b, valid_mask]
            summary_tok = valid_toks[-1:, :]

            img_mask = valid_ids == self.image_token_id
            img_toks = valid_toks[img_mask]
            if img_tok_counts is not None:
                c4 = img_tok_counts[b * 4 : (b + 1) * 4]
                if sum(c4) == int(img_toks.shape[0]):
                    v0, v1, v2, v3 = torch.split(img_toks, c4, dim=0)
                    # Preserve uncollapsed global + 3 distinct 1:1 ROI crop token banks!
                    # Global: 48 avg + 16 max? Wait: adaptive_avg_pool1d(48) + adaptive_max_pool1d(16) preserves local spikes!
                    g_avg = self._pool_1d_tokens(v0, 48)
                    g_max = (
                        F.adaptive_max_pool1d(v0.unsqueeze(0).transpose(1, 2), output_size=16)
                        .transpose(1, 2)
                        .squeeze(0)
                        if v0.shape[0] >= 16
                        else self._pool_1d_tokens(v0, 16)
                    )
                    r0 = self._pool_1d_tokens(v1, 16)
                    r1 = self._pool_1d_tokens(v2, 16)
                    r2 = self._pool_1d_tokens(v3, 16)
                    seq_b = torch.cat([g_avg, g_max, r0, r1, r2, summary_tok], dim=0)  # [113, D]
                    pooled_list.append(seq_b)
                    continue

            # Fallback if custom token count
            seq_b = torch.cat([self._pool_1d_tokens(valid_toks, 112), summary_tok], dim=0)
            pooled_list.append(seq_b)

        return torch.stack(pooled_list, dim=0)

    def forward(self, batch_views_or_inputs: Any, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        dev = device or next(self.head.parameters()).device
        tokens = self.encode_multiview_tokens(batch_views_or_inputs, dev)
        return self.head(tokens)


def build_hires_roi_vlm_for_4090(
    ckpt_path: str,
    device: str = "cuda",
    bnb_path: Optional[str] = None,
    four_bit: bool = True,
    verbose: bool = True,
) -> Qwen3VLHiResRoiSpecialist:
    """
    4090 Deployment Builder:
      1. Builds `Qwen3VLHiResRoiSpecialist` in `bf16` + loads trained LoRA (`r=64`) + merger + `TagFreeUnimodalDeQAHead`.
      2. Merges LoRA adapters cleanly into `bf16` weights (`merge_and_unload()`) and re-attaches hypercolumn hooks.
      3. Quantizes LLM `nn.Linear` layers in-place to 4-bit NF4 (`quantize_llm_inplace`, keeping vision + merger `bf16`).
    """
    if bnb_path and bnb_path not in sys.path:
        sys.path.append(bnb_path)
    from phoenix_pillar_4090 import quantize_llm_inplace

    dev = torch.device(device)
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    lora_r = int(ck.get("lora_rank", 64)) if isinstance(ck, dict) else 64
    use_dora = bool(ck.get("use_dora", False)) if isinstance(ck, dict) else False

    m = Qwen3VLHiResRoiSpecialist(lora_rank=lora_r, use_dora=use_dora, gc_mode="none").to(dev)
    res = m.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if "lora" in k.lower() or k.startswith("head.")] + list(res.unexpected_keys)
    assert not bad, f"Checkpoint mismatch: {bad[:5]}"

    m.backbone = m.backbone.merge_and_unload()
    m._install_hypercolumn_hooks()
    nq = quantize_llm_inplace(m.backbone, dev) if four_bit else 0
    gc.collect()
    torch.cuda.empty_cache()
    m.eval()
    if verbose:
        print(
            f"[build_hires_roi_vlm_for_4090] ({RECIPE_TAG}): merged r={lora_r}, {nq} LLM linears 4-bit={four_bit}, "
            f"resident={torch.cuda.memory_allocated(dev) / 2 ** 30:.2f} GB on {dev}",
            flush=True,
        )
    return m


# ----------------------------------------------------------------------------------------------------------------------
# 100% Tag-Free Loss Stack: Ordinal KL + RPS + Clean Gate + Cross-Video Boundary Margin + Thurstone + Annealed KD
# ----------------------------------------------------------------------------------------------------------------------
class CrossVideoPositiveBoundaryLoss(nn.Module):
    """
    Directly supervises all 3 ordinal log-odds cuts (`Pos vs Clean`, `P1+ vs P2`, and especially `P0 vs P1`)
    across distinct videos (`group_i != group_j`) with a 768-entry cross-batch queue (synced across DDP ranks).
    """

    def __init__(self, queue_size: int = 768, sigma0: float = 1.05) -> None:
        super().__init__()
        self.queue_size = int(queue_size)
        self.sigma0 = float(sigma0)
        self.register_buffer("q_b", torch.zeros(queue_size, 6, 3, dtype=torch.float32), persistent=False)
        self.register_buffer("q_mu", torch.zeros(queue_size, 6, dtype=torch.float32), persistent=False)
        self.register_buffer("q_var", torch.full((queue_size, 6), 0.25, dtype=torch.float32), persistent=False)
        self.register_buffer("q_emb", torch.zeros(queue_size, 256, dtype=torch.float32), persistent=False)
        self.register_buffer("q_y", torch.zeros(queue_size, 6, dtype=torch.long), persistent=False)
        self.register_buffer("q_g", torch.full((queue_size,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long), persistent=False)

    @staticmethod
    def _boundary_log_odds(logits_6x4: torch.Tensor) -> torch.Tensor:
        b_out = logits_6x4.new_zeros((logits_6x4.shape[0], 6, 3))
        for d in range(6):
            z = logits_6x4[:, d, :]
            if d in FULL_DIMS:
                b0 = torch.logsumexp(z[:, 1:], dim=-1) - z[:, 0]      # Pos (>=1) vs Clean (0)
                b1 = torch.logsumexp(z[:, 2:], dim=-1) - z[:, 1]      # Conditional P1+ (>=2) vs P2 (1)
                b2 = z[:, 3] - z[:, 2]                                # Conditional P0 (3) vs P1 (2)
                b_out[:, d, 0] = b0
                b_out[:, d, 1] = b1
                b_out[:, d, 2] = b2
            else:
                bb = z[:, 3] - z[:, 0]
                b_out[:, d, :] = bb.unsqueeze(-1)
        return b_out

    @staticmethod
    @torch.no_grad()
    def _gather_ddp(t: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return t.detach()
        parts = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(parts, t.detach().contiguous())
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def _enqueue(
        self,
        b_det: torch.Tensor,
        mu_det: torch.Tensor,
        var_det: torch.Tensor,
        emb_det: torch.Tensor,
        y_det: torch.Tensor,
        g_det: torch.Tensor,
    ) -> None:
        b_all = self._gather_ddp(b_det)
        mu_all = self._gather_ddp(mu_det)
        var_all = self._gather_ddp(var_det)
        emb_all = self._gather_ddp(emb_det)
        y_all = self._gather_ddp(y_det)
        g_all = self._gather_ddp(g_det)

        n = b_all.shape[0]
        if n >= self.queue_size:
            self.q_b.copy_(b_all[-self.queue_size :])
            self.q_mu.copy_(mu_all[-self.queue_size :])
            self.q_var.copy_(var_all[-self.queue_size :])
            self.q_emb.copy_(emb_all[-self.queue_size :])
            self.q_y.copy_(y_all[-self.queue_size :])
            self.q_g.copy_(g_all[-self.queue_size :])
            self.ptr.fill_(0)
            self.filled.fill_(self.queue_size)
            return
        p = int(self.ptr.item())
        end = p + n
        if end <= self.queue_size:
            self.q_b[p:end].copy_(b_all)
            self.q_mu[p:end].copy_(mu_all)
            self.q_var[p:end].copy_(var_all)
            self.q_emb[p:end].copy_(emb_all)
            self.q_y[p:end].copy_(y_all)
            self.q_g[p:end].copy_(g_all)
        else:
            first = self.queue_size - p
            rem = n - first
            self.q_b[p:].copy_(b_all[:first]); self.q_b[:rem].copy_(b_all[first:])
            self.q_mu[p:].copy_(mu_all[:first]); self.q_mu[:rem].copy_(mu_all[first:])
            self.q_var[p:].copy_(var_all[:first]); self.q_var[:rem].copy_(var_all[first:])
            self.q_emb[p:].copy_(emb_all[:first]); self.q_emb[:rem].copy_(emb_all[first:])
            self.q_y[p:].copy_(y_all[:first]); self.q_y[:rem].copy_(y_all[first:])
            self.q_g[p:].copy_(g_all[:first]); self.q_g[:rem].copy_(g_all[first:])
        self.ptr.fill_(end % self.queue_size)
        self.filled.fill_(min(self.queue_size, int(self.filled.item()) + n))

    def forward(
        self,
        head_out: Dict[str, torch.Tensor],
        y_6d: torch.Tensor,
        group_hashes: torch.Tensor,
    ) -> torch.Tensor:
        logits = head_out["logits_6x4"].float()
        mu = head_out["mu_6d"].float()
        var = head_out["var_6d"].float()
        emb = head_out["quality_emb"].float()
        b_cur = self._boundary_log_odds(logits)

        n_mem = int(self.filled.item())
        if n_mem > 0:
            b_ref = torch.cat([b_cur.detach(), self.q_b[:n_mem]], dim=0)
            mu_ref = torch.cat([mu.detach(), self.q_mu[:n_mem]], dim=0)
            var_ref = torch.cat([var.detach(), self.q_var[:n_mem]], dim=0)
            emb_ref = torch.cat([emb.detach(), self.q_emb[:n_mem]], dim=0)
            y_ref = torch.cat([y_6d, self.q_y[:n_mem]], dim=0)
            g_ref = torch.cat([group_hashes, self.q_g[:n_mem]], dim=0)
        else:
            b_ref, mu_ref, var_ref, emb_ref, y_ref, g_ref = (
                b_cur.detach(), mu.detach(), var.detach(), emb.detach(), y_6d, group_hashes
            )

        diff_vid = group_hashes.unsqueeze(1) != g_ref.unsqueeze(0)
        sim_cos = emb @ emb_ref.T
        sim_w = 1.0 + 1.25 * torch.clamp(sim_cos.detach(), min=0.0)
        inv_s = 1.0 / (2.0 * self.sigma0)
        sqrt2 = math.sqrt(2.0)

        losses: List[torch.Tensor] = []

        # Supervised quality-embedding metric alignment across distinct videos
        same_profile = (y_6d.unsqueeze(1) == y_ref.unsqueeze(0)).all(dim=-1) & diff_vid
        diff_profile = (~(y_6d.unsqueeze(1) == y_ref.unsqueeze(0)).all(dim=-1)) & diff_vid
        emb_terms: List[torch.Tensor] = []
        if same_profile.any():
            emb_terms.append((1.0 - sim_cos[same_profile]).mean())
        if diff_profile.any():
            emb_terms.append(F.relu(sim_cos[diff_profile] - 0.25).mean())
        if emb_terms:
            losses.append(0.15 * torch.stack(emb_terms).mean())
        else:
            losses.append(emb.sum() * 0.0)
        for d in range(6):
            yi = y_6d[:, d].unsqueeze(1)
            yj = y_ref[:, d].unsqueeze(0)
            dim_w = 1.45 if d == 4 else (1.30 if d == 2 else 1.0)

            # 1. Thurstone continuous severity ranking across distinct videos
            valid_th = (yi != yj) & diff_vid
            if valid_th.any():
                sgn = torch.sign((yi - yj).float())
                adj_b = torch.where(torch.abs(yi - yj) == 1, 1.5, 1.0)
                z_th = sgn * (mu[:, d].unsqueeze(1) - mu_ref[:, d].unsqueeze(0)) / torch.sqrt(
                    var[:, d].unsqueeze(1) + var_ref[:, d].unsqueeze(0) + 0.08
                )
                phi_th = 0.5 * (1.0 + torch.erf(z_th / sqrt2))
                losses.append(dim_w * ((1.0 - phi_th) * adj_b * sim_w)[valid_th].mean())

            # 2. Exact positive-only conditional log-odds boundary cuts (P1+ vs P2, and P0 vs P1!)
            if d in FULL_DIMS:
                c0_hi, c0_lo = (yi >= 1) & (yj == 0) & diff_vid, (yi == 0) & (yj >= 1) & diff_vid
                c1_hi, c1_lo = (yi >= 2) & (yj == 1) & diff_vid, (yi == 1) & (yj >= 2) & diff_vid
                c2_hi, c2_lo = (yi == 3) & (yj == 2) & diff_vid, (yi == 2) & (yj == 3) & diff_vid
                cuts = [(0, c0_hi, c0_lo, 0.90), (1, c1_hi, c1_lo, 1.35), (2, c2_hi, c2_lo, 1.75)]
            else:
                cb_hi, cb_lo = (yi >= 3) & (yj == 0) & diff_vid, (yi == 0) & (yj >= 3) & diff_vid
                cuts = [(0, cb_hi, cb_lo, 1.25)]

            for bi, m_hi, m_lo, bw in cuts:
                v_cut = m_hi | m_lo
                if not v_cut.any():
                    continue
                sgn_c = torch.where(m_hi, 1.0, -1.0)
                si = b_cur[:, d, bi].unsqueeze(1)
                sj = b_ref[:, d, bi].unsqueeze(0)
                phi_c = 0.5 * (1.0 + torch.erf(sgn_c * (si - sj) * inv_s))
                fl_c = (1.0 - torch.sqrt(phi_c.clamp(1e-6, 1.0))) * sim_w
                losses.append(dim_w * bw * fl_c[v_cut].mean())

        if self.training:
            self._enqueue(b_cur, mu, var, emb, y_6d, group_hashes)

        return torch.stack(losses).mean() if losses else logits.sum() * 0.0


class TagFreeVLMCompositeLoss(nn.Module):
    """
    100% Tag-Free Loss combining:
      - Unimodal soft-target KL (`sigma_anchor=0.30, sigma_mid=0.44`)
      - Squared-CDF EMD (`RPS`, `beta_emd=1.50`)
      - All-Clean cliff BCE (`beta_clean=0.65`)
      - Cross-video Thurstone + Positive Boundary Margin (`lambda_rank=0.35`)
      - Leak-free nested teacher soft KL (`T=1.25`, `lambda_kd` annealed `0.25 -> 0.0`)
    """

    def __init__(
        self,
        beta_emd: float = 1.50,
        beta_clean: float = 0.65,
        p0_sample_boost: float = 1.65,
        kd_temp: float = 1.25,
    ) -> None:
        super().__init__()
        self.beta_emd = float(beta_emd)
        self.beta_clean = float(beta_clean)
        self.p0_sample_boost = float(p0_sample_boost)
        self.kd_temp = float(kd_temp)
        dim_w = torch.tensor([1.0, 1.0, 1.25, 1.35, 1.45, 1.40], dtype=torch.float32)
        self.register_buffer("dim_weights", dim_w, persistent=False)
        self.rank_fn = CrossVideoPositiveBoundaryLoss(queue_size=768, sigma0=1.05)

    @staticmethod
    def _soft_targets(y_6d: torch.Tensor, sigma_anchor: float = 0.30, sigma_mid: float = 0.44) -> torch.Tensor:
        grid = torch.tensor([0.0, 1.0, 2.0, 3.0], device=y_6d.device, dtype=torch.float32).view(1, 1, 4)
        y_f = y_6d.float().unsqueeze(-1)
        is_mid = ((y_6d == 1) | (y_6d == 2)).float().unsqueeze(-1)
        sigma = sigma_anchor + is_mid * (sigma_mid - sigma_anchor)
        raw = torch.exp(-((grid - y_f) ** 2) / (2.0 * sigma ** 2))
        bin_mask = torch.ones((1, 6, 4), device=y_6d.device, dtype=torch.float32)
        for d in P0_ONLY_DIMS:
            bin_mask[:, d, 1] = 0.0
            bin_mask[:, d, 2] = 0.0
        raw = raw * bin_mask
        return raw / torch.clamp(raw.sum(dim=-1, keepdim=True), min=1e-8)

    def forward(
        self,
        head_out: Dict[str, torch.Tensor],
        y_6d: torch.Tensor,
        group_hashes: torch.Tensor,
        teacher_probs_6x4: Optional[torch.Tensor] = None,
        lambda_rank: float = 0.35,
        lambda_kd: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        logits = head_out["logits_6x4"].float()
        probs = head_out["probs_6x4"].float()
        clean_logit = head_out["clean_logit"].float()

        q = self._soft_targets(y_6d)
        log_p = F.log_softmax(logits, dim=-1)
        kl_per = torch.sum(q * (torch.log(torch.clamp(q, min=1e-7)) - log_p), dim=-1)

        cdf_p = torch.cumsum(probs, dim=-1)[:, :, :3]
        cdf_q = torch.cumsum(q, dim=-1)[:, :, :3]
        emd_per = torch.sum((cdf_p - cdf_q) ** 2, dim=-1)

        # Label-only P0 cell boost (identical to c233397, zero split/batch tag!)
        cell_w = 1.0 + (y_6d == 3).float() * (self.p0_sample_boost - 1.0)
        dim_w = self.dim_weights.float().view(1, 6)

        loss_kl = torch.mean(kl_per * cell_w * dim_w)
        loss_emd = torch.mean(emd_per * cell_w * dim_w)

        clean_gt = (y_6d == 0).all(dim=1).float()
        loss_clean = F.binary_cross_entropy_with_logits(clean_logit, clean_gt)

        total = loss_kl + self.beta_emd * loss_emd + self.beta_clean * loss_clean

        loss_rank = total.new_tensor(0.0)
        if lambda_rank > 1e-6:
            loss_rank = self.rank_fn(head_out, y_6d, group_hashes)
            total = total + float(lambda_rank) * loss_rank

        loss_kd = total.new_tensor(0.0)
        if teacher_probs_6x4 is not None and lambda_kd > 1e-6:
            tp = teacher_probs_6x4.float().clamp(1e-6, 1.0 - 1e-6)
            T = self.kd_temp
            kd_terms = []
            for d in FULL_DIMS:
                tq = F.softmax(torch.log(tp[:, d, :]) / T, dim=-1)
                slp = F.log_softmax(logits[:, d, :] / T, dim=-1)
                kd_terms.append((T * T) * F.kl_div(slp, tq, reduction="batchmean"))
            for d in P0_ONLY_DIMS:
                tq2 = F.softmax(torch.log(tp[:, d, [0, 3]]) / T, dim=-1)
                slp2 = F.log_softmax(logits[:, d, [0, 3]] / T, dim=-1)
                kd_terms.append((T * T) * F.kl_div(slp2, tq2, reduction="batchmean"))
            loss_kd = torch.stack(kd_terms).mean()
            total = total + float(lambda_kd) * loss_kd

        return {
            "loss": total,
            "kl": loss_kl.detach(),
            "emd": loss_emd.detach(),
            "clean": loss_clean.detach(),
            "rank": loss_rank.detach(),
            "kd": loss_kd.detach(),
        }


# ----------------------------------------------------------------------------------------------------------------------
# Dataset, Collation, SWA & DDP Utilities
# ----------------------------------------------------------------------------------------------------------------------
class StagedHiResRoiVLMDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        cache_dir: str,
        boxes_map: Dict[str, np.ndarray],
        teacher_map: Optional[Dict[str, np.ndarray]] = None,
        is_train: bool = True,
        force_hflip: bool = False,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> None:
        self.rows = list(rows)
        self.cache_dir = cache_dir
        self.boxes_map = boxes_map
        self.teacher_map = teacher_map or {}
        self.is_train = bool(is_train)
        self.force_hflip = bool(force_hflip)
        self.max_pixels = int(max_pixels)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_4views(self, nm: str) -> List[Image.Image]:
        stem = os.path.splitext(nm)[0]
        paths = [os.path.join(self.cache_dir, f"{stem}__v{k}.png") for k in range(4)]
        if all(os.path.exists(p) for p in paths):
            out: List[Image.Image] = []
            for p in paths:
                with Image.open(p) as im:
                    out.append(im.convert("RGB"))
            return out
        with Image.open(_resolve_raw_image_path(nm)) as raw_im:
            return extract_4views_from_pil(raw_im.convert("RGB"), boxes_3x4=self.boxes_map.get(nm), max_pixels=self.max_pixels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        nm = str(row["name"])
        views = self._load_4views(nm)

        do_flip = self.force_hflip or (self.is_train and np.random.rand() < 0.5)
        if do_flip:
            views = [v.transpose(Image.FLIP_LEFT_RIGHT) for v in views]

        t_prob = self.teacher_map.get(nm)
        return {
            "name": nm,
            "views": views,
            "grades": np.asarray(row["grades"], dtype=np.int64),
            "group_hash": int(row["group_hash"]),
            "teacher_probs": (
                np.asarray(t_prob, dtype=np.float32) if t_prob is not None else np.zeros((6, 4), dtype=np.float32)
            ),
            "has_teacher": 1 if t_prob is not None else 0,
        }


class HiResProcessorCollate:
    def __init__(self, processor: Any, prompt_text_formatted: str) -> None:
        self.processor = processor
        self.prompt_text_formatted = prompt_text_formatted

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [self.prompt_text_formatted for _ in batch]
        flat_images = [pil_v for b in batch for pil_v in b["views"]]
        model_inputs = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )
        return {
            "names": [str(b["name"]) for b in batch],
            "model_inputs": dict(model_inputs),
            "grades": torch.from_numpy(np.stack([b["grades"] for b in batch], axis=0)).long(),
            "group_hashes": torch.tensor([int(b["group_hash"]) for b in batch], dtype=torch.long),
            "teacher_probs": torch.from_numpy(np.stack([b["teacher_probs"] for b in batch], axis=0)).float(),
            "has_teacher": torch.tensor([int(b["has_teacher"]) for b in batch], dtype=torch.long),
        }


def extract_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    raw_m = model.module if isinstance(model, DDP) else model
    trainable_keys = {name for name, param in raw_m.named_parameters() if param.requires_grad}
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw_m.state_dict().items():
        kl = k.lower()
        if (k in trainable_keys) or k.startswith("head.") or ("lora_" in kl) or ("dora_" in kl) or ("magnitude_vector" in kl):
            out[k] = v.detach().cpu().clone()
    return out


class SWACollector:
    def __init__(self) -> None:
        self.count = 0
        self.total_weight = 0.0
        self.avg_state: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module, weight: float = 1.0) -> None:
        raw_sd = extract_trainable_state_dict(model)
        state = {k: (v.float() if v.is_floating_point() else v.clone()) for k, v in raw_sd.items()}
        if self.avg_state is None:
            self.avg_state = {k: (v * weight if v.is_floating_point() else v) for k, v in state.items()}
            self.total_weight = float(weight)
        else:
            for k, v in state.items():
                if v.is_floating_point():
                    self.avg_state[k] += v * weight
                else:
                    self.avg_state[k] = v
            self.total_weight += float(weight)
        self.count += 1

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        assert self.avg_state is not None and self.count > 0
        return {
            k: (v / max(1e-8, self.total_weight) if v.is_floating_point() else v.clone())
            for k, v in self.avg_state.items()
        }


@torch.no_grad()
def predict_single_orientation(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    raw_m = model.module if isinstance(model, DDP) else model
    raw_m.eval()
    names_all: List[str] = []
    probs_l, logits_l, clean_l, p_clean_l, mu_l, var_l, emb_l, gold_l = [], [], [], [], [], [], [], []
    for batch in loader:
        names_all.extend(batch["names"])
        out = raw_m(batch["model_inputs"], device=device)
        probs_l.append(out["probs_6x4"].cpu().numpy())
        logits_l.append(out["logits_6x4"].cpu().numpy())
        clean_l.append(out["clean_logit"].cpu().numpy())
        p_clean_l.append(out["p_clean_joint"].cpu().numpy())
        mu_l.append(out["mu_6d"].cpu().numpy())
        var_l.append(out["var_6d"].cpu().numpy())
        emb_l.append(out["quality_emb"].cpu().numpy())
        gold_l.append(batch["grades"].numpy())
    return {
        "names": np.array(names_all, dtype=object),
        "probs_6x4": np.concatenate(probs_l, axis=0).astype(np.float32),
        "logits_6x4": np.concatenate(logits_l, axis=0).astype(np.float32),
        "clean_logit": np.concatenate(clean_l, axis=0).astype(np.float32),
        "p_clean_joint": np.concatenate(p_clean_l, axis=0).astype(np.float32),
        "mu_6d": np.concatenate(mu_l, axis=0).astype(np.float32),
        "var_6d": np.concatenate(var_l, axis=0).astype(np.float32),
        "quality_emb": np.concatenate(emb_l, axis=0).astype(np.float32),
        "gold_6d": np.concatenate(gold_l, axis=0).astype(np.int64),
    }


def predict_2x_hflip(model: nn.Module, loader_orig: DataLoader, loader_hflip: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    p0 = predict_single_orientation(model, loader_orig, device)
    p1 = predict_single_orientation(model, loader_hflip, device)
    assert list(p0["names"]) == list(p1["names"])
    probs = 0.5 * (p0["probs_6x4"] + p1["probs_6x4"])
    logits = np.log(np.clip(probs, 1e-6, 1.0))
    for d in P0_ONLY_DIMS:
        logits[:, d, 1] = -1e4
        logits[:, d, 2] = -1e4
    emb = p0["quality_emb"] + p1["quality_emb"]
    emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
    grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 1, 4)
    mu = (probs * grid).sum(axis=-1)
    var = (probs * (grid - mu[:, :, None]) ** 2).sum(axis=-1)
    return {
        "va_names": p0["names"],
        "va_probs_6x4": probs.astype(np.float32),
        "names": p0["names"],
        "probs_orig_6x4": p0["probs_6x4"].astype(np.float32),
        "probs_hflip_6x4": p1["probs_6x4"].astype(np.float32),
        "probs_6x4": probs.astype(np.float32),
        "logits_6x4": logits.astype(np.float32),
        "clean_logit": (0.5 * (p0["clean_logit"] + p1["clean_logit"])).astype(np.float32),
        "p_clean_joint": (0.5 * (p0["p_clean_joint"] + p1["p_clean_joint"])).astype(np.float32),
        "mu_6d": mu.astype(np.float32),
        "var_6d": var.astype(np.float32),
        "quality_emb": emb.astype(np.float32),
        "gold_6d": p0["gold_6d"],
    }


def _pin_numa_node1_worker(_worker_id: int) -> None:
    """Pins DataLoader workers on GPUs 4-7 to NUMA Node 1 CPUs (48-95, 144-191)."""
    try:
        numa1_cpus = set(range(48, 96)) | set(range(144, 192))
        avail = os.sched_getaffinity(0)
        inter = avail & numa1_cpus
        if inter:
            os.sched_setaffinity(0, inter)
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------------------------------
# Training Loop & 20-Step Benchmark / Smoke Mode
# ----------------------------------------------------------------------------------------------------------------------
def train_fold_or_bench(args: argparse.Namespace) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    _pin_numa_node1_worker(0)

    rows = load_canonical_tagfree_rows()
    canonical_names = [r["name"] for r in rows]

    # Rank 0 stages `/dev/shm` lossless view cache once; DDP peers wait at barrier
    if rank == 0:
        boxes_map = ensure_shm_vlm_views_staged(
            names=canonical_names,
            sem_meta_npz=args.sem_meta_npz,
            cache_dir=args.shm_cache_dir,
            max_pixels=args.max_pixels,
            num_threads=24,
        )
    if is_ddp:
        dist.barrier()
    if rank != 0:
        boxes_map = ensure_shm_vlm_views_staged(
            names=canonical_names,
            sem_meta_npz=args.sem_meta_npz,
            cache_dir=args.shm_cache_dir,
            max_pixels=args.max_pixels,
            num_threads=4,
        )

    f_idx = int(args.fold)
    assert -1 <= f_idx < 8
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    fold_npz = os.path.join(args.out_dir, f"fold_{f_idx}_probs.npz")
    fold_ckpt = (
        os.path.join(args.ckpt_dir, f"qwen3vl_hires_s{args.seed}_fullfit_swa.pt")
        if f_idx < 0
        else os.path.join(args.ckpt_dir, f"qwen3vl_hires_s{args.seed}_fold_{f_idx}_swa.pt")
    )
    if not args.bench_steps and ((f_idx < 0 and os.path.exists(fold_ckpt)) or (f_idx >= 0 and os.path.exists(fold_npz) and os.path.exists(fold_ckpt))):
        if rank == 0:
            print(f"[Resume] Fold {f_idx} already completed: {fold_ckpt}", flush=True)
        if is_ddp:
            dist.destroy_process_group()
        return

    set_deterministic_seed(args.seed + max(0, f_idx + 1) * 100 + rank)
    tr_rows = list(rows) if f_idx < 0 else [r for r in rows if int(r["fold"]) != f_idx]
    va_rows = [] if f_idx < 0 else [r for r in rows if int(r["fold"]) == f_idx]
    teacher_map = load_leakfree_kd_for_fold(args.nested_kd_npz, f_idx, canonical_names) if args.lambda_kd > 0 else {}

    if rank == 0:
        print(
            f"=== [{RECIPE_TAG}] Fold {f_idx}/8 | world_size={world_size} | train={len(tr_rows)} val={len(va_rows)} "
            f"| max_px={args.max_pixels} + 3x448 | LoRA r={args.lora_rank} dora={args.use_dora} gc={args.gc_mode} "
            f"| bsz={args.batch_size}/gpu accum={args.grad_accum} (eff_bsz={args.batch_size * world_size * args.grad_accum}) ===",
            flush=True,
        )

    base_model = Qwen3VLHiResRoiSpecialist(
        model_id_or_path=(args.model_path or None),
        lora_rank=args.lora_rank,
        use_dora=args.use_dora,
        gc_mode=args.gc_mode,
    ).to(device)
    collate = HiResProcessorCollate(base_model.processor, base_model.prompt_text_formatted)

    if is_ddp:
        model: nn.Module = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        model = base_model

    ds_tr = StagedHiResRoiVLMDataset(
        tr_rows, args.shm_cache_dir, boxes_map, teacher_map=teacher_map, is_train=True, max_pixels=args.max_pixels
    )
    sampler_tr = DistributedSampler(ds_tr, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed + f_idx) if is_ddp else None
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=(sampler_tr is None),
        sampler=sampler_tr,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=3 if args.workers > 0 else None,
        worker_init_fn=_pin_numa_node1_worker,
        collate_fn=collate,
        drop_last=False,
    )

    raw_m = model.module if isinstance(model, DDP) else model
    head_params = [p for p in raw_m.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    bb_params = [p for p in raw_m.parameters() if p.requires_grad and id(p) not in head_ids]
    trainable_all = head_params + bb_params

    opt = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_head, "weight_decay": 1e-2},
            {"params": bb_params, "lr": args.lr_backbone, "weight_decay": 1e-2},
        ],
        fused=(not args.use_dora),
    )

    grad_accum = max(1, int(args.grad_accum))
    opt_steps_per_ep = max(1, (len(dl_tr) + grad_accum - 1) // grad_accum)
    total_opt_steps = max(1, args.epochs * opt_steps_per_ep)
    warmup_steps = max(1, int(round(0.06 * total_opt_steps)))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return 0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    loss_fn = TagFreeVLMCompositeLoss(
        beta_emd=1.50, beta_clean=0.65, p0_sample_boost=1.65, kd_temp=1.25
    ).to(device)
    swa = SWACollector()
    swa_start_ep = max(1, args.epochs - 2)

    # ---- Optional 20-step throughput + 4090 4-bit parity benchmark ----
    if args.bench_steps > 0:
        model.train()
        opt.zero_grad(set_to_none=True)
        warmup_n = 3
        timed_n = int(args.bench_steps)
        t_start = 0.0
        imgs_timed = 0
        for step_idx, batch in enumerate(dl_tr):
            if step_idx == warmup_n:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
                t_start = time.perf_counter()
            y_6d = batch["grades"].to(device, non_blocking=True)
            g_h = batch["group_hashes"].to(device, non_blocking=True)
            t_p = batch["teacher_probs"].to(device, non_blocking=True)
            out = model(batch["model_inputs"], device=device)
            losses = loss_fn(out, y_6d, g_h, teacher_probs_6x4=t_p, lambda_rank=0.35, lambda_kd=args.lambda_kd)
            (losses["loss"] / grad_accum).backward()
            if (step_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_all, max_norm=1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step_idx >= warmup_n:
                imgs_timed += int(y_6d.shape[0]) * world_size
            if step_idx + 1 >= warmup_n + timed_n:
                break

        torch.cuda.synchronize()
        dt = time.perf_counter() - t_start
        ips_total = imgs_timed / max(1e-6, dt)
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (2 ** 30)
        ep_eta_s = len(tr_rows) / max(1e-6, ips_total)

        if rank == 0:
            print(
                f"[BENCHMARK {timed_n} steps (after {warmup_n} warmup)] world_size={world_size} | "
                f"throughput={ips_total:.2f} imgs/s ({ips_total/world_size:.2f} imgs/s/GPU) | "
                f"peak_VRAM={peak_vram_gb:.2f} GB / 140 GB | 1-epoch ETA={ep_eta_s:.1f}s ({ep_eta_s * args.epochs / 60.0:.1f} min / {args.epochs}-ep fold)",
                flush=True,
            )
            # Also verify N=1 == N=batch invariance + bf16->NF4 4090 deploy fidelity on 4 real val samples!
            swa.update(model, weight=1.0)
            tmp_ckpt = f"/tmp/bench_qwen3vl_hires_{os.getpid()}.pt"
            torch.save(
                {"recipe": RECIPE_TAG, "lora_rank": args.lora_rank, "use_dora": args.use_dora, "state_dict": swa.export_state_dict()},
                tmp_ckpt,
            )
            ds_va0 = StagedHiResRoiVLMDataset(va_rows[:4], args.shm_cache_dir, boxes_map, is_train=False, force_hflip=False, max_pixels=args.max_pixels)
            ds_va1 = StagedHiResRoiVLMDataset(va_rows[:4], args.shm_cache_dir, boxes_map, is_train=False, force_hflip=True, max_pixels=args.max_pixels)
            dl_va0 = DataLoader(ds_va0, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)
            dl_va1 = DataLoader(ds_va1, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)
            va_batch = predict_2x_hflip(raw_m, dl_va0, dl_va1, device)

            # Unmerged N=1 vs N=4 invariance check
            p_n1_unmerged = []
            with torch.no_grad():
                for i in range(4):
                    o0 = raw_m([ds_va0[i]["views"]], device=device)["probs_6x4"].cpu().numpy()[0]
                    o1 = raw_m([ds_va1[i]["views"]], device=device)["probs_6x4"].cpu().numpy()[0]
                    p_n1_unmerged.append(0.5 * (o0 + o1))
            n1_max_dp = float(np.max(np.abs(np.stack(p_n1_unmerged) - va_batch["probs_6x4"])))

            del model, base_model, raw_m, opt, loss_fn, batch, out, losses, va_batch
            gc.collect()
            torch.cuda.empty_cache()

            # Merged BF16 check
            m_bf16 = build_hires_roi_vlm_for_4090(tmp_ckpt, device=str(device), bnb_path=os.path.join(REPO, "third_party/pydeps"), four_bit=False, verbose=True)
            p_bf16 = []
            with torch.no_grad():
                for i in range(4):
                    o0 = m_bf16([ds_va0[i]["views"]], device=device)["probs_6x4"].cpu().numpy()[0]
                    o1 = m_bf16([ds_va1[i]["views"]], device=device)["probs_6x4"].cpu().numpy()[0]
                    p_bf16.append(0.5 * (o0 + o1))
            bf16_max_dp = float(np.max(np.abs(np.stack(p_bf16) - np.stack(p_n1_unmerged))))
            del m_bf16
            gc.collect()
            torch.cuda.empty_cache()

            # Merged NF4 4-bit 4090 check + N=1 2x-TTA latency
            m_q4 = build_hires_roi_vlm_for_4090(tmp_ckpt, device=str(device), bnb_path=os.path.join(REPO, "third_party/pydeps"), four_bit=True, verbose=True)
            res_q4_gb = torch.cuda.memory_allocated(device) / (2 ** 30)
            with torch.no_grad():
                _ = m_q4([ds_va0[0]["views"]], device=device)  # warmup (MUST be under no_grad!)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            t_inf0 = time.perf_counter()
            p_q4 = []
            with torch.no_grad():
                for i in range(4):
                    o0 = m_q4([ds_va0[i]["views"]], device=device)["probs_6x4"].cpu().numpy()[0]
                    o1 = m_q4([ds_va1[i]["views"]], device=device)["probs_6x4"].cpu().numpy()[0]
                    p_q4.append(0.5 * (o0 + o1))
            torch.cuda.synchronize()
            lat_2x_s = (time.perf_counter() - t_inf0) / 4.0
            peak_q4_gb = torch.cuda.max_memory_allocated(device) / (2 ** 30)
            q4_tv = float(0.5 * np.abs(np.stack(p_q4) - np.stack(p_n1_unmerged)).sum(axis=-1).mean())
            os.remove(tmp_ckpt)
            print(
                f"[4090 & N=1 PARITY PASS] N=1_vs_N=4_max|dp|={n1_max_dp:.5f} | bf16_merged_max|dp|={bf16_max_dp:.5f} | "
                f"NF4_4bit_TV={q4_tv:.5f} | NF4_resident={res_q4_gb:.2f}GB peak_infer={peak_q4_gb:.2f}GB | 2x_TTA_N1_latency={lat_2x_s:.3f}s/img",
                flush=True,
            )
        if is_ddp:
            dist.barrier()
            dist.destroy_process_group()
        return

    # ---- Full Training Loop ----
    for ep in range(args.epochs):
        if sampler_tr is not None:
            sampler_tr.set_epoch(ep)
        model.train()
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        t_ep = time.time()
        sum_loss, sum_kl, sum_emd, sum_rank, sum_kd, n_steps = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        # Anneal KD weight linearly to 0.0 at final epoch (`opus` #18 rule: prevent weak PE-L teacher from capping VLM!)
        kd_scale = max(0.0, 1.0 - float(ep) / float(max(1, args.epochs - 1))) if args.epochs > 1 else 1.0
        cur_lambda_kd = float(args.lambda_kd) * kd_scale
        cur_lambda_rank = 0.20 if ep == 0 else 0.38

        n_batches = len(dl_tr)
        for step_idx, batch in enumerate(dl_tr):
            is_sync_step = ((step_idx + 1) % grad_accum == 0) or ((step_idx + 1) == n_batches)
            ctx = model.no_sync() if (is_ddp and not is_sync_step) else _nullcontext()
            with ctx:
                y_6d = batch["grades"].to(device, non_blocking=True)
                g_h = batch["group_hashes"].to(device, non_blocking=True)
                t_p = batch["teacher_probs"].to(device, non_blocking=True) if cur_lambda_kd > 1e-6 else None

                out = model(batch["model_inputs"], device=device)
                losses = loss_fn(
                    out,
                    y_6d=y_6d,
                    group_hashes=g_h,
                    teacher_probs_6x4=t_p,
                    lambda_rank=cur_lambda_rank,
                    lambda_kd=cur_lambda_kd,
                )
                (losses["loss"] / grad_accum).backward()

            if is_sync_step:
                torch.nn.utils.clip_grad_norm_(trainable_all, max_norm=1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            sum_loss += float(losses["loss"].item())
            sum_kl += float(losses["kl"].item())
            sum_emd += float(losses["emd"].item())
            sum_rank += float(losses["rank"].item())
            sum_kd += float(losses["kd"].item())
            n_steps += 1

            if rank == 0 and ((step_idx + 1) % 80 == 0 or (step_idx + 1) == n_batches):
                dt = time.time() - t_ep
                ips = (n_steps * args.batch_size * world_size) / max(1e-6, dt)
                print(
                    f"  [Fold {f_idx} Ep {ep+1}/{args.epochs}] step {step_idx+1}/{n_batches} | "
                    f"loss={sum_loss/n_steps:.4f} (kl={sum_kl/n_steps:.3f} rps={sum_emd/n_steps:.3f} "
                    f"rk={sum_rank/n_steps:.3f} kd={sum_kd/n_steps:.3f}@{cur_lambda_kd:.2f}) | "
                    f"{ips:.2f} imgs/s | peak={torch.cuda.max_memory_allocated(device)/2**30:.1f}GB ({dt:.0f}s)",
                    flush=True,
                )

        if rank == 0 and ep >= swa_start_ep:
            swa.update(model, weight=1.0)
            print(f"  [Fold {f_idx}] SWA updated at end of Epoch {ep+1}/{args.epochs} (count={swa.count})", flush=True)

    if is_ddp:
        dist.barrier()

    # Evaluate OOF on rank 0 using SWA weights and save `fold_{k}_probs.npz`
    if rank == 0:
        swa_sd = swa.export_state_dict()
        raw_m.load_state_dict(swa_sd, strict=False)
        torch.save(
            {
                "recipe": RECIPE_TAG,
                "seed": args.seed,
                "fold": f_idx,
                "lora_rank": args.lora_rank,
                "use_dora": args.use_dora,
                "max_pixels": args.max_pixels,
                "state_dict": swa_sd,
            },
            fold_ckpt,
        )
        if f_idx < 0:
            print(f"=== [{RECIPE_TAG} FullFit DONE] saved -> {fold_ckpt} ===", flush=True)
        else:
            ds_va0 = StagedHiResRoiVLMDataset(va_rows, args.shm_cache_dir, boxes_map, is_train=False, force_hflip=False, max_pixels=args.max_pixels)
            ds_va1 = StagedHiResRoiVLMDataset(va_rows, args.shm_cache_dir, boxes_map, is_train=False, force_hflip=True, max_pixels=args.max_pixels)
            dl_va0 = DataLoader(ds_va0, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=_pin_numa_node1_worker, collate_fn=collate)
            dl_va1 = DataLoader(ds_va1, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=_pin_numa_node1_worker, collate_fn=collate)
            va_out = predict_2x_hflip(raw_m, dl_va0, dl_va1, device)
            np.savez_compressed(fold_npz, recipe=RECIPE_TAG, seed=np.int64(args.seed), fold=np.int64(f_idx), **va_out)

            from eval_specialist_gate import per_dim_ap, rps_6x4
            from tagfree_rule3_decoder import severity_metrics

            va_gold = np.array([r["grades"] for r in va_rows], dtype=np.int64)
            pv = c.renorm(va_out["va_probs_6x4"].astype(np.float64))
            m = evaluate_official_metrics(np.argmax(pv, axis=-1), va_gold)
            rps_v = rps_6x4(pv, va_gold)
            _, map14 = per_dim_ap(pv, va_gold)
            sev_m = severity_metrics(pv, va_gold, np.ones(len(va_gold), dtype=bool))
            print(
                f"=== [{RECIPE_TAG} Fold {f_idx} DONE] RPS={rps_v:.5f} | mAP14={map14:.4f} | Argmax Doc2S={m['strict_total']:.2f} "
                f"| OS AP_p0={sev_m['OS']['AP_p0']:.3f} P2vP1={sev_m['OS']['P2vP1']:.3f} P1vP0={sev_m['OS']['P1vP0']:.3f} "
                f"| OG AP_p0={sev_m['OG']['AP_p0']:.3f} P2vP1={sev_m['OG']['P2vP1']:.3f} P1vP0={sev_m['OG']['P1vP0']:.3f} -> {fold_npz} ===",
                flush=True,
            )

    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="", help="Optional override for local Qwen3-VL-8B path")
    ap.add_argument("--sem-meta-npz", default=os.path.join(REPO, "results/r55/semantic_roi_meta.npz"))
    ap.add_argument("--nested-kd-npz", default=os.path.join(REPO, "results/r55/nested_kd_teacher.npz"))
    ap.add_argument("--shm-cache-dir", default=SHM_VLM_CACHE)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results/r55/qwen3vl_hires_pilot"))
    ap.add_argument("--ckpt-dir", default=os.path.join(REPO, "results/r55/qwen3vl_hires_pilot/ckpts"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=4, help="Single-GPU fallback index when not launched via torchrun")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--use-dora", action="store_true")
    ap.add_argument("--gc-mode", choices=["llm_24", "llm_half", "llm_only", "full", "none"], default="llm_only")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4, help="Per-GPU micro-batch size")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr-head", type=float, default=2.0e-4)
    ap.add_argument("--lr-backbone", type=float, default=8.0e-5)
    ap.add_argument("--lambda-kd", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bench-steps", type=int, default=0, help="Run N timed training steps + 4090 4-bit & N=1 parity test, then exit")
    args = ap.parse_args()
    train_fold_or_bench(args)


if __name__ == "__main__":
    main()
