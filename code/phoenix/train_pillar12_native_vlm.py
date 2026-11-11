#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/phoenix/train_pillar12_native_vlm.py
=========================================
Breakthrough 3 & 4 (Upgraded with View-Separated Peak-Anomaly Token Readout, DoRA r=32,
Pure-Real Rare P1/P0 Oversampling, Stage-1->2 Warm-Start, and 2x HFlip Val TTA):
  - Pillar 1 (`--pillar pillar1`): `Qwen3-VL-8B-Instruct` (`models/qwen3vl-8b`)
  - Pillar 2 (`--pillar pillar2`): `Clef-Flash-9B` / `InternVL3.5-8B` (`models/clef-flash`)

Key Representation & Convergence Safeguards:
  1. View-Separated + Unaveraged Peak-Anomaly Token Readout (`[B, 64, D]`):
     Splits visual tokens into the 3 independent views (`Global_Native`, `Crop0_1to1`,
     `Crop1_1to1`) via `image_grid_thw` and feeds `20` coarse structure tokens + `42`
     UN-AVERAGED top-anomaly peak patch tokens + `1` spatial dispersion token + `1`
     instruction summary token directly into `PhoenixUnimodalDeQAHead`!
  2. Proven VLM Convergence Regime (`r=32, alpha=64`, `lr_backbone=1.6e-4`, `lr_head=6.0e-4`)
     + Stage-1 Adapter/Head Per-Fold Checkpoint Saving (`~200 MB`) so `stage2-rl2r` warm-starts
     directly from Stage-1 weights instead of restarting from cold zero!
  3. Pure-Real Rare `P1/P0` Oversampling (repeating authentic real frames with `grade >= 2` on
     `过锐, 花屏, 过度生成, 虚化突兀` — zero synthetic samples, zero `P2` inflation!).
  4. 2x Horizontal-Flip Val TTA + Multi-GPU Per-Fold Sharding & Zero-Loss Resume.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(THIS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(THIS_DIR))

from phoenix.config import (
    DIMS,
    PHOENIX_CKPT_DIR,
    PHOENIX_OOF_DIR,
    PHOENIX_SPLITS_DIR,
    PureRealImageResolver,
    evaluate_official_metrics,
    resolve_local_or_hf_model,
    set_deterministic_seed,
)
from phoenix.losses_and_heads import (
    BasinAlignedSWACollector,
    CrossBatchThurstoneRL2RLoss,
    DeQAOrdinalDistributionLoss,
    PhoenixUnimodalDeQAHead,
    extract_trainable_state_dict,
)
from phoenix.native_physics_and_views import (
    CoCDepthDefocusProbe,
    compute_1to1_wavelet_and_artifact_maps,
    resize_native_aspect_pil,
    select_scrutiny_1to1_patches,
)

CONCISE_INSPECTION_PROMPT = (
    "请作为短视频无参考画质质检专家，综合图1（原生宽高比全局画面：判别景深关系、整体模糊与花屏）、"
    "图2（主体解剖与景深边缘1:1原尺寸切片：判别五官手指畸变过度生成与抠图虚化突兀）、"
    "图3（高频纹理与边缘1:1原尺寸切片：判别1至2像素过锐白边光晕与涂抹塑料感纹理损失），"
    "直接形成6类画质缺陷的客观严重度状态。"
)


def detect_lora_rank_from_state_dict(sd: Dict[str, Any], default_rank: int = 32) -> int:
    """Auto-detects PEFT LoRA/DoRA rank `r` from saved adapter state_dict (`lora_A` shape[0])."""
    inner = sd.get("state_dict", sd) if isinstance(sd, dict) else {}
    for k, v in inner.items():
        if "lora_a" in str(k).lower() and hasattr(v, "ndim") and v.ndim == 2:
            return int(v.shape[0])
    return int(default_rank)


def _has_safe_disk_headroom(path_dir: str, min_free_gb: float = 25.0) -> bool:
    try:
        os.makedirs(path_dir, exist_ok=True)
        free_gb = shutil.disk_usage(path_dir).free / (1024.0 ** 3)
        return free_gb >= min_free_gb
    except Exception:
        return True


def oversample_rare_real_p1_p0(
    rows: Sequence[Dict[str, object]],
    teacher_probs: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, object]], Optional[np.ndarray]]:
    """Repeats ONLY authentic real training frames with `grade >= 2` (`P1/P0`) on `(过锐, 花屏, 过度生成, 虚化突兀)`."""
    base_rows = list(rows)
    extra_idx = [
        i
        for i, r in enumerate(base_rows)
        if any(int(r["grades"][d]) >= 2 for d in (2, 3, 4, 5))  # type: ignore[index]
    ]
    if not extra_idx:
        return base_rows, teacher_probs
    aug_rows = base_rows + [base_rows[i] for i in extra_idx]
    aug_teacher = (
        np.concatenate([teacher_probs, teacher_probs[extra_idx]], axis=0)
        if teacher_probs is not None
        else None
    )
    return aug_rows, aug_teacher


def load_split_rows(split_csv_path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(split_csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "name": r["name"].strip(),
                    "canon_group": r["canon_group"].strip(),
                    "fold": int(r["fold"]),
                    "batch_id": int(r["batch_id"]),
                    "noisy_flag": int(r.get("noisy_boundary_flag", 0)),
                    "grades": [int(r[c]) for c in DIMS],
                }
            )
    return rows


def load_batch_shift_init(audit_json_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(audit_json_path):
        return None
    with open(audit_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "log_odds_shift_supp_vs_orig" in data:
        return np.asarray(data["log_odds_shift_supp_vs_orig"], dtype=np.float32)
    return None


def get_or_build_scrutiny_box_cache(
    names: Sequence[str],
    resolver: PureRealImageResolver,
    device_str: str = "cuda",
    cache_npz_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    cache_map: Dict[str, np.ndarray] = {}
    if cache_npz_path and os.path.exists(cache_npz_path):
        try:
            d = np.load(cache_npz_path, allow_pickle=True)
            c_names = [str(x) for x in d["names"]]
            c_boxes = np.asarray(d["boxes_2x4"], dtype=np.int32)
            for nm, bx in zip(c_names, c_boxes):
                cache_map[nm] = bx
        except Exception:
            cache_map = {}

    missing = [nm for nm in names if nm not in cache_map]
    if not missing:
        return cache_map

    print(f"[Phoenix Scrutiny Precompute] Computing 1:1 Wavelet+CoC crop boxes for {len(missing)} images...", flush=True)
    coc_probe = CoCDepthDefocusProbe(
        model_path=resolve_local_or_hf_model("pillar3_depth_anything_s"),
        device=device_str,
    )
    t0 = time.time()
    for idx, nm in enumerate(missing):
        pil_img = resolver.load_rgb_pil(nm)
        rgb_np = np.asarray(pil_img, dtype=np.uint8)
        phys_maps = compute_1to1_wavelet_and_artifact_maps(rgb_np)
        coc_field = coc_probe.compute_coc_jacobian_field(pil_img, phys_maps)
        _, meta = select_scrutiny_1to1_patches(pil_img, phys_maps, coc_field, patch_size=384, num_patches=2)
        boxes = np.array([m["box_xyxy"] for m in meta], dtype=np.int32)
        cache_map[nm] = boxes
        if (idx + 1) % 500 == 0 or (idx + 1) == len(missing):
            print(f"  [{idx+1}/{len(missing)}] ({(idx+1)/max(1e-3, time.time()-t0):.1f} img/s)", flush=True)

    del coc_probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cache_npz_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_npz_path)), exist_ok=True)
            all_k = sorted(cache_map.keys())
            tmp_p = f"{cache_npz_path}.tmp.{os.getpid()}.npz"
            np.savez_compressed(
                tmp_p,
                names=np.array(all_k, dtype=object),
                boxes_2x4=np.stack([cache_map[k] for k in all_k], axis=0),
            )
            os.replace(tmp_p, cache_npz_path)
        except Exception:
            pass
    return cache_map


def crop_1to1_from_box(pil_img, box_xyxy: Sequence[int], patch_size: int = 384):
    from PIL import Image

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


class NativeMultiViewVLMDataset(Dataset):
    """Fork-safe pure-PIL dataset yielding `[Global Native-Aspect, Patch-0 1:1, Patch-1 1:1]`."""

    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        resolver: PureRealImageResolver,
        box_cache: Dict[str, np.ndarray],
        is_train: bool = True,
        force_hflip: bool = False,
        max_global_pixels: int = 784 * 784,
        oof_teacher_probs: Optional[np.ndarray] = None,
    ) -> None:
        self.rows = list(rows)
        self.resolver = resolver
        self.box_cache = box_cache
        self.is_train = is_train
        self.force_hflip = force_hflip
        self.max_global_pixels = max_global_pixels
        self.oof_teacher_probs = oof_teacher_probs

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        from PIL import Image

        row = self.rows[idx]
        nm = str(row["name"])
        pil_img = self.resolver.load_rgb_pil(nm)
        orig_w, _ = pil_img.size
        boxes = self.box_cache[nm].copy()

        do_flip = self.force_hflip or (self.is_train and np.random.rand() < 0.5)
        if do_flip:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            old_x0 = boxes[:, 0].copy()
            old_x1 = boxes[:, 2].copy()
            boxes[:, 0] = np.maximum(0, orig_w - old_x1)
            boxes[:, 2] = boxes[:, 0] + 384

        crop0 = crop_1to1_from_box(pil_img, boxes[0], patch_size=384)
        crop1 = crop_1to1_from_box(pil_img, boxes[1], patch_size=384)
        global_view = resize_native_aspect_pil(
            pil_img,
            max_pixels=self.max_global_pixels,
            grid_factor=28,
        )

        item: Dict[str, object] = {
            "name": nm,
            "views": [global_view, crop0, crop1],
            "batch_id": int(row.get("batch_id", 0)),
            "noisy_flag": int(row.get("noisy_flag", 0)),
            "grades": np.asarray(row.get("grades", [0] * 6), dtype=np.int64),
        }
        if self.oof_teacher_probs is not None:
            item["teacher_probs"] = self.oof_teacher_probs[idx]
        return item


def _format_3image_chat_prompt(processor: Any, prompt_text: str) -> str:
    msgs = [
        {
            "role": "user",
            "content": [
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
    return f"<image>\n<image>\n<image>\n{prompt_text}"


def _infer_image_token_id(cfg: Any, processor: Any) -> int:
    for attr in ("image_token_id", "image_token_index"):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val >= 0:
            return int(val)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    for cand in ("<|image_pad|>", "<image>", "<|vision_pad|>", "<IMG_CONTEXT>"):
        try:
            tid = tok.convert_tokens_to_ids(cand)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(tok, "unk_token_id", -1):
                return int(tid)
        except Exception:
            pass
    return 151655


def _extract_view_structure_and_peak_tokens(
    v_toks: torch.Tensor,
    n_struct: int,
    n_peak: int,
    fallback_tok: torch.Tensor,
) -> torch.Tensor:
    """
    Preserves BOTH global spatial context (`n_struct` pooled landmarks) AND `n_peak`
    UN-AVERAGED peak-anomaly patch tokens (`||t - mean(t)||_2` top-k) so localized
    3-patch defects (`过度生成` fingers/text, `1~2px` `过锐` halos) are NEVER diluted!
    """
    m = v_toks.shape[0]
    if m == 0:
        return fallback_tok.expand(n_struct + n_peak, -1)

    struct = F.adaptive_avg_pool1d(
        v_toks.unsqueeze(0).transpose(1, 2), output_size=n_struct
    ).transpose(1, 2).squeeze(0)

    mu = v_toks.mean(dim=0, keepdim=True)
    anom_mag = torch.norm(v_toks - mu, dim=-1)
    k_eff = min(m, n_peak)
    _, top_idx = torch.topk(anom_mag, k=k_eff, dim=0)
    top_idx_sorted, _ = torch.sort(top_idx)
    peaks = v_toks[top_idx_sorted]
    if k_eff < n_peak:
        peaks = torch.cat([peaks, peaks[-1:].expand(n_peak - k_eff, -1)], dim=0)

    return torch.cat([struct, peaks], dim=0)


class NativeVLMPillarModel(nn.Module):
    def __init__(
        self,
        pillar_name: str = "pillar1",
        model_id_or_path: Optional[str] = None,
        init_batch_shift: Optional[np.ndarray] = None,
        quantize_4bit_infer: bool = False,
        apply_dora: bool = True,
        lora_rank: int = 32,
        mock_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.pillar_name = pillar_name
        self.mock_backbone = mock_backbone
        self.lora_rank = int(lora_rank)
        self.processor = None
        self.image_token_id = 151655

        if mock_backbone:
            hidden_size = 384
            self.mock_encoder = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=7, stride=4, padding=3),
                nn.GELU(),
                nn.Conv2d(128, hidden_size, kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d((6, 6)),
            )
            self.backbone = None
            in_dim = hidden_size
        else:
            key = "pillar1_qwen3vl_8b" if pillar_name == "pillar1" else "pillar2_hetero_vlm_8b"
            resolved = model_id_or_path or resolve_local_or_hf_model(key)
            self.backbone, self.processor, in_dim, self.image_token_id = self._load_vlm_backbone(
                resolved_path=resolved,
                quantize_4bit_infer=quantize_4bit_infer,
                apply_dora=apply_dora,
                lora_rank=self.lora_rank,
            )

        self.head = PhoenixUnimodalDeQAHead(
            in_dim=in_dim,
            hidden_dim=512,
            num_heads=8,
            dropout=0.10,
            init_batch_shift=init_batch_shift,
        ).float()

    @staticmethod
    def _load_vlm_backbone(
        resolved_path: str,
        quantize_4bit_infer: bool,
        apply_dora: bool,
        lora_rank: int = 32,
    ):
        import transformers
        from transformers import AutoConfig, AutoProcessor

        quant_cfg = None
        if quantize_4bit_infer and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=["visual", "vision_tower", "vision_model", "mlp1", "merger"],
                )
            except Exception:
                quant_cfg = None

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        processor = AutoProcessor.from_pretrained(resolved_path, trust_remote_code=True)
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        if hasattr(tok, "padding_side"):
            tok.padding_side = "right"
        cfg = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True)
        img_tok_id = _infer_image_token_id(cfg, processor)

        loader_classes = []
        for cls_name in (
            "AutoModelForImageTextToText",
            "Qwen3VLForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
            "AutoModel",
        ):
            cls_obj = getattr(transformers, cls_name, None)
            if cls_obj is not None and cls_obj not in loader_classes:
                loader_classes.append(cls_obj)

        full_model = None
        last_err: Optional[Exception] = None
        attn_choices = (None,) if quant_cfg is not None else ("flash_attention_2", "sdpa", None)
        for attn_impl in attn_choices:
            for loader_cls in loader_classes:
                try:
                    kw: Dict[str, Any] = {
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                        "dtype": dtype,
                    }
                    if attn_impl is not None:
                        kw["attn_implementation"] = attn_impl
                    if quant_cfg is not None:
                        kw["quantization_config"] = quant_cfg
                    try:
                        full_model = loader_cls.from_pretrained(resolved_path, **kw)
                    except TypeError:
                        kw.pop("dtype", None)
                        kw["torch_dtype"] = dtype
                        full_model = loader_cls.from_pretrained(resolved_path, **kw)
                    break
                except Exception as exc:
                    last_err = exc
                    continue
            if full_model is not None:
                break

        if full_model is None:
            raise RuntimeError(f"Failed to load VLM from {resolved_path}: {last_err}")

        backbone = getattr(full_model, "model", full_model)
        if hasattr(full_model, "lm_head") and backbone is not full_model:
            try:
                del full_model.lm_head
            except Exception:
                pass

        if apply_dora:
            try:
                from peft import LoraConfig, get_peft_model

                allowed_leafs = {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "qkv",
                    "proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "linear_fc1",
                    "linear_fc2",
                    "fc1",
                    "fc2",
                    "in_proj_qkvz",
                    "in_proj_ba",
                    "out_proj",
                }
                exact_linear_paths: List[str] = []
                for full_name, mod in backbone.named_modules():
                    if not isinstance(mod, nn.Linear):
                        continue
                    nl = full_name.lower()
                    if any(skip in nl for skip in ("patch_embed", "lm_head", "embed_tokens", "wte")):
                        continue
                    leaf = full_name.split(".")[-1]
                    if leaf in allowed_leafs:
                        exact_linear_paths.append(full_name)

                use_dora_flag = not quantize_4bit_infer
                lora_cfg = LoraConfig(
                    r=int(lora_rank),
                    lora_alpha=int(lora_rank) * 2,
                    lora_dropout=0.05,
                    target_modules=exact_linear_paths if exact_linear_paths else ["q_proj", "v_proj"],
                    use_dora=use_dora_flag,
                    bias="none",
                )
                backbone = get_peft_model(backbone, lora_cfg)
            except Exception as exc:
                print(f"[Warn] PEFT attachment fallback ({exc}); unfreezing multimodal merger.", flush=True)

            for n, p in backbone.named_parameters():
                nl = n.lower()
                if any(bk in nl for bk in ("merger", "multi_modal_projector", "aligner")) and p.is_floating_point():
                    if not quantize_4bit_infer:
                        p.requires_grad = True

            if not quantize_4bit_infer and hasattr(backbone, "gradient_checkpointing_enable"):
                try:
                    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    backbone.gradient_checkpointing_enable()
                if hasattr(backbone, "enable_input_require_grads"):
                    backbone.enable_input_require_grads()

        hidden_dim = None
        if hasattr(cfg, "text_config") and getattr(cfg.text_config, "hidden_size", None):
            hidden_dim = int(cfg.text_config.hidden_size)
        elif hasattr(cfg, "llm_config") and getattr(cfg.llm_config, "hidden_size", None):
            hidden_dim = int(cfg.llm_config.hidden_size)
        elif getattr(cfg, "hidden_size", None):
            hidden_dim = int(cfg.hidden_size)
        return backbone, processor, int(hidden_dim or 3584), img_tok_id

    def encode_multiview_tokens(self, batch_views: List[List[object]], device: torch.device) -> torch.Tensor:
        if self.mock_backbone:
            seq_list = []
            for views in batch_views:
                view_tokens = []
                for pil_v in views:
                    arr = np.asarray(pil_v.resize((192, 192)), dtype=np.float32) / 255.0
                    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
                    fmap = self.mock_encoder(t)
                    view_tokens.append(fmap.flatten(2).transpose(1, 2).squeeze(0))
                cat_v = torch.cat(view_tokens, dim=0)
                seq_list.append(_extract_view_structure_and_peak_tokens(cat_v, 24, 40, cat_v[:1]))
            return torch.stack(seq_list, dim=0)

        texts = [_format_3image_chat_prompt(self.processor, CONCISE_INSPECTION_PROMPT) for _ in batch_views]
        flat_images = [pil_v for views in batch_views for pil_v in views]

        model_inputs = self.processor(
            text=texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )
        moved_inputs: Dict[str, Any] = {}
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                moved_inputs[k] = v.to(device, non_blocking=True)
            else:
                moved_inputs[k] = v

        use_bf16 = device.type == "cuda"
        with torch.autocast(device_type="cuda" if use_bf16 else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = self.backbone(
                **moved_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        h_states = getattr(outputs, "hidden_states", None)
        if h_states is not None and len(h_states) >= 4:
            n_layers = len(h_states)
            idx_mid = max(0, n_layers // 2)
            idx_3q = max(0, (3 * n_layers) // 4)
            fused = (
                0.25 * h_states[idx_mid].float()
                + 0.35 * h_states[idx_3q].float()
                + 0.40 * h_states[-1].float()
            )
        else:
            last_h = getattr(outputs, "last_hidden_state", h_states[-1])
            fused = last_h.float()

        input_ids = moved_inputs.get("input_ids", None)
        attn_mask = moved_inputs.get("attention_mask", None)
        grid_thw = moved_inputs.get("image_grid_thw", None)
        bsz = fused.shape[0]
        pooled_list = []

        for b in range(bsz):
            valid_mask = attn_mask[b].bool() if attn_mask is not None else torch.ones(fused.shape[1], dtype=torch.bool, device=device)
            valid_toks = fused[b, valid_mask]
            if valid_toks.shape[0] == 0:
                valid_toks = fused[b]
            last_summary = valid_toks[-1:, :]

            img_mask = (input_ids[b] == self.image_token_id) & valid_mask if input_ids is not None else None
            if img_mask is not None and int(img_mask.sum().item()) >= 12:
                vis_all = fused[b, img_mask]
                n_vis = int(vis_all.shape[0])

                if grid_thw is not None and grid_thw.shape[0] >= (b + 1) * 3:
                    raw_c = [
                        int(grid_thw[b * 3 + k, 0].item() * grid_thw[b * 3 + k, 1].item() * grid_thw[b * 3 + k, 2].item())
                        for k in range(3)
                    ]
                    tot_c = max(1, sum(raw_c))
                    c0 = max(1, int(round(n_vis * raw_c[0] / tot_c)))
                    c1 = max(1, int(round(n_vis * raw_c[1] / tot_c)))
                else:
                    c0 = max(1, n_vis // 2)
                    c1 = max(1, (n_vis - c0) // 2)

                v0 = vis_all[:c0]
                v1 = vis_all[c0 : min(n_vis, c0 + c1)]
                v2 = vis_all[min(n_vis, c0 + c1) :]

                t_v0 = _extract_view_structure_and_peak_tokens(v0, n_struct=8, n_peak=16, fallback_tok=last_summary)
                t_v1 = _extract_view_structure_and_peak_tokens(v1, n_struct=6, n_peak=13, fallback_tok=last_summary)
                t_v2 = _extract_view_structure_and_peak_tokens(v2, n_struct=6, n_peak=13, fallback_tok=last_summary)
                vis_std = torch.sqrt(vis_all.var(dim=0, keepdim=True, unbiased=False) + 1e-5)
                seq_b = torch.cat([t_v0, t_v1, t_v2, vis_std, last_summary], dim=0)
            else:
                seq_b = _extract_view_structure_and_peak_tokens(
                    valid_toks, n_struct=23, n_peak=40, fallback_tok=last_summary
                )
                seq_b = torch.cat([seq_b, last_summary], dim=0)

            pooled_list.append(seq_b)

        return torch.stack(pooled_list, dim=0)

    def forward(
        self,
        batch_views: List[List[object]],
        batch_ids: Optional[torch.Tensor] = None,
        apply_batch_bias: Optional[bool] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        dev = device or next(self.head.parameters()).device
        tokens = self.encode_multiview_tokens(batch_views, dev)
        return self.head(tokens, batch_ids=batch_ids, apply_batch_bias=apply_batch_bias)


def _collate_multiview(batch: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "names": [b["name"] for b in batch],
        "views": [b["views"] for b in batch],
        "batch_ids": torch.tensor([int(b["batch_id"]) for b in batch], dtype=torch.long),
        "noisy_flags": torch.tensor([int(b["noisy_flag"]) for b in batch], dtype=torch.long),
        "grades": torch.from_numpy(np.stack([b["grades"] for b in batch], axis=0)).long(),
    }
    if "teacher_probs" in batch[0]:
        out["teacher_probs"] = torch.from_numpy(np.stack([b["teacher_probs"] for b in batch], axis=0)).float()
    return out


@torch.no_grad()
def predict_vlm_dataset(
    model: NativeVLMPillarModel,
    loader: DataLoader,
    device: torch.device,
    loader_hflip: Optional[DataLoader] = None,
) -> Dict[str, np.ndarray]:
    model.eval()

    def _single_pass(dl: DataLoader) -> Dict[str, np.ndarray]:
        names_all: List[str] = []
        probs_list, logits_list, clean_l_list, p_clean_list = [], [], [], []
        mu_list, var_list, emb_list, gold_list = [], [], [], []
        for batch in dl:
            names_all.extend(batch["names"])
            out = model(
                batch_views=batch["views"],
                batch_ids=None,
                apply_batch_bias=False,
                device=device,
            )
            probs_list.append(out["probs_6x4"].cpu().numpy())
            logits_list.append(out["logits_6x4"].cpu().numpy())
            clean_l_list.append(out["clean_logit"].cpu().numpy())
            p_clean_list.append(out["p_clean_joint"].cpu().numpy())
            mu_list.append(out["mu_6d"].cpu().numpy())
            var_list.append(out["var_6d"].cpu().numpy())
            emb_list.append(out["quality_emb"].cpu().numpy())
            gold_list.append(batch["grades"].numpy())
        return {
            "names": np.array(names_all, dtype=object),
            "probs_6x4": np.concatenate(probs_list, axis=0).astype(np.float32),
            "logits_6x4": np.concatenate(logits_list, axis=0).astype(np.float32),
            "clean_logit": np.concatenate(clean_l_list, axis=0).astype(np.float32),
            "p_clean_joint": np.concatenate(p_clean_list, axis=0).astype(np.float32),
            "mu_6d": np.concatenate(mu_list, axis=0).astype(np.float32),
            "var_6d": np.concatenate(var_list, axis=0).astype(np.float32),
            "quality_emb": np.concatenate(emb_list, axis=0).astype(np.float32),
            "gold_6d": np.concatenate(gold_list, axis=0).astype(np.int64),
        }

    p0 = _single_pass(loader)
    if loader_hflip is None:
        return p0

    p1 = _single_pass(loader_hflip)
    probs = 0.5 * (p0["probs_6x4"] + p1["probs_6x4"])
    logits = np.log(np.clip(probs, 1e-6, 1.0))
    from phoenix.config import P0_ONLY_DIMS

    for d in P0_ONLY_DIMS:
        logits[:, d, 1] = -1e4
        logits[:, d, 2] = -1e4
    emb = p0["quality_emb"] + p1["quality_emb"]
    emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
    grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 1, 4)
    mu = (probs * grid).sum(axis=-1)
    var = (probs * (grid - mu[:, :, None]) ** 2).sum(axis=-1)

    return {
        "names": p0["names"],
        "probs_6x4": probs.astype(np.float32),
        "logits_6x4": logits.astype(np.float32),
        "clean_logit": (0.5 * (p0["clean_logit"] + p1["clean_logit"])).astype(np.float32),
        "p_clean_joint": (0.5 * (p0["p_clean_joint"] + p1["p_clean_joint"])).astype(np.float32),
        "mu_6d": mu.astype(np.float32),
        "var_6d": var.astype(np.float32),
        "quality_emb": emb.astype(np.float32),
        "gold_6d": p0["gold_6d"],
    }


def train_one_run(
    model: NativeVLMPillarModel,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 4,
    lr_head: float = 6.0e-4,
    lr_backbone: float = 1.6e-4,
    grad_accum: int = 2,
    enable_rl2r: bool = True,
    swa_start_epoch: Optional[int] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    bb_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]

    opt = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head, "weight_decay": 1e-2},
            {"params": bb_params, "lr": lr_backbone, "weight_decay": 1e-2},
        ]
    )
    grad_accum = max(1, int(grad_accum))
    opt_steps_per_ep = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_steps = max(1, epochs * opt_steps_per_ep)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr_backbone * 0.1)

    deqa_loss_fn = DeQAOrdinalDistributionLoss().to(device)
    rl2r_loss_fn = CrossBatchThurstoneRL2RLoss(queue_size=1024, emb_dim=256).to(device)
    swa = BasinAlignedSWACollector() if swa_start_epoch is not None else None
    trainable_all = [p for p in model.parameters() if p.requires_grad]

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(train_loader):
            y_6d = batch["grades"].to(device, non_blocking=True)
            b_ids = batch["batch_ids"].to(device, non_blocking=True)
            noisy = batch["noisy_flags"].to(device, non_blocking=True)
            t_probs = batch.get("teacher_probs")
            if t_probs is not None:
                t_probs = t_probs.to(device, non_blocking=True)

            out = model(
                batch_views=batch["views"],
                batch_ids=b_ids,
                apply_batch_bias=True,
                device=device,
            )
            losses = deqa_loss_fn(
                head_out=out,
                y_6d=y_6d,
                noisy_flags=noisy,
                oof_teacher_probs=t_probs,
                supp_bias_param=model.head.supp_batch_bias,
            )
            loss = losses["loss"]
            if enable_rl2r:
                loss = loss + 0.45 * rl2r_loss_fn(out, y_6d)

            (loss / grad_accum).backward()
            if (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_all, max_norm=1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            ep_loss += float(loss.item())
            if (step_idx + 1) % 100 == 0:
                print(
                    f"    [{model.pillar_name}] Ep {ep+1}/{epochs} step {step_idx+1}/{len(train_loader)} "
                    f"| avg_loss={ep_loss/(step_idx+1):.4f} ({time.time()-t0:.0f}s)",
                    flush=True,
                )

        if swa is not None and ep >= swa_start_epoch:
            swa.update(model, weight=1.0)
        print(
            f"  [{model.pillar_name}] Epoch {ep+1}/{epochs} done in {time.time()-t0:.0f}s "
            f"| loss={ep_loss / max(1, len(train_loader)):.4f}",
            flush=True,
        )

    return swa.export_state_dict() if (swa is not None and swa.count > 0) else None


def aggregate_5fold_oof(
    rows: Sequence[Dict[str, object]],
    pillar: str,
    mode: str,
    out_dir: str,
) -> str:
    n = len(rows)
    oof_probs = np.zeros((n, 6, 4), dtype=np.float32)
    oof_logits = np.zeros((n, 6, 4), dtype=np.float32)
    oof_clean_l = np.zeros(n, dtype=np.float32)
    oof_p_clean = np.zeros(n, dtype=np.float32)
    oof_mu = np.zeros((n, 6), dtype=np.float32)
    oof_var = np.zeros((n, 6), dtype=np.float32)
    oof_emb = np.zeros((n, 256), dtype=np.float32)
    gold_all = np.array([r["grades"] for r in rows], dtype=np.int64)
    folds_all = np.array([int(r["fold"]) for r in rows], dtype=np.int64)
    name_to_pos = {str(r["name"]): i for i, r in enumerate(rows)}

    for f_idx in range(5):
        f_path = os.path.join(out_dir, f"{pillar}_{mode}_fold{f_idx}.npz")
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Missing fold artifact for aggregation: {f_path}")
        part = np.load(f_path, allow_pickle=True)
        p_names = [str(x) for x in part["names"]]
        dest_idx = [name_to_pos[nm] for nm in p_names]
        oof_probs[dest_idx] = part["probs_6x4"]
        oof_logits[dest_idx] = part["logits_6x4"]
        oof_clean_l[dest_idx] = part["clean_logit"]
        oof_p_clean[dest_idx] = part["p_clean_joint"]
        oof_mu[dest_idx] = part["mu_6d"]
        oof_var[dest_idx] = part["var_6d"]
        oof_emb[dest_idx] = part["quality_emb"]

    out_npz = os.path.join(out_dir, f"{pillar}_{mode}_oof.npz")
    np.savez_compressed(
        out_npz,
        names=np.array([r["name"] for r in rows], dtype=object),
        folds=folds_all,
        gold_6d=gold_all,
        probs_6x4=oof_probs,
        logits_6x4=oof_logits,
        clean_logit=oof_clean_l,
        p_clean_joint=oof_p_clean,
        mu_6d=oof_mu,
        var_6d=oof_var,
        quality_emb=oof_emb,
    )
    m = evaluate_official_metrics(np.argmax(oof_probs, axis=-1), gold_all)
    print(
        f"=== [{pillar} {mode} 5-Fold OOF Summary] Naive Argmax Strict={m['strict_total']:.2f} "
        f"| Practice={m['practice_total']:.2f} -> {out_npz} ===",
        flush=True,
    )
    return out_npz


def run_smoke_test(pillar: str) -> None:
    from PIL import Image

    set_deterministic_seed(2026)
    device = torch.device("cpu")
    dummy_rows = []
    for i in range(6):
        dummy_rows.append(
            {
                "name": f"smoke_{i:03d}.png",
                "batch_id": i % 2,
                "noisy_flag": int(i == 2),
                "grades": [i % 4, (i + 1) % 4, 0, 3 if i == 5 else 0, i % 3, 0],
                "views": [
                    Image.fromarray(np.random.randint(0, 255, (560, 336, 3), dtype=np.uint8)),
                    Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)),
                    Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)),
                ],
            }
        )
    loader = DataLoader(dummy_rows, batch_size=3, shuffle=False, collate_fn=_collate_multiview)
    model = NativeVLMPillarModel(
        pillar_name=pillar,
        init_batch_shift=np.zeros((6, 4), dtype=np.float32),
        mock_backbone=True,
    ).to(device)
    swa_state = train_one_run(model, loader, device, epochs=2, grad_accum=1, enable_rl2r=True, swa_start_epoch=0)
    assert swa_state is not None
    model.load_state_dict(swa_state, strict=False)
    preds = predict_vlm_dataset(model, loader, device)
    assert preds["probs_6x4"].shape == (6, 6, 4)
    assert np.allclose(preds["probs_6x4"].sum(axis=-1), 1.0, atol=1e-5)
    single_loader = DataLoader(dummy_rows[:1], batch_size=1, shuffle=False, collate_fn=_collate_multiview)
    single_pred = predict_vlm_dataset(model, single_loader, device)
    max_diff = float(np.max(np.abs(single_pred["probs_6x4"][0] - preds["probs_6x4"][0])))
    assert max_diff < 1e-6, f"Single-image independence violated: max_diff={max_diff}"
    print(f"[{pillar} Peak-Anomaly DoRA-r32 VLM Smoke Test PASS] probs={preds['probs_6x4'].shape} | single_img_diff={max_diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix Pillar-1 & Pillar-2 Native Multi-View VLM Trainer")
    parser.add_argument("--pillar", choices=["pillar1", "pillar2"], default="pillar1")
    parser.add_argument("--mode", choices=["cv5", "stage2-rl2r", "fulldata-swa"], default="cv5")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--split-csv", default=str(PHOENIX_SPLITS_DIR / "phoenix_5fold.csv"))
    parser.add_argument("--audit-json", default=str(PHOENIX_SPLITS_DIR / "batch_shift_and_audit.json"))
    parser.add_argument("--teacher-oof-npz", default="")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr-head", type=float, default=6.0e-4)
    parser.add_argument("--lr-backbone", type=float, default=1.6e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--aggregate-cv5", action="store_true")
    parser.add_argument("--precompute-boxes-only", action="store_true")
    parser.add_argument("--out-dir", default=str(PHOENIX_OOF_DIR))
    parser.add_argument("--ckpt-dir", default=str(PHOENIX_CKPT_DIR))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(args.pillar)
        return

    rows = load_split_rows(args.split_csv)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.aggregate_cv5:
        aggregate_5fold_oof(rows, args.pillar, args.mode, args.out_dir)
        return

    set_deterministic_seed(2026 + max(0, args.fold))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    resolver = PureRealImageResolver()
    box_cache_path = os.path.join(args.out_dir, "scrutiny_boxes_384_cache.npz")
    box_cache = get_or_build_scrutiny_box_cache(
        names=[str(r["name"]) for r in rows],
        resolver=resolver,
        device_str=str(device),
        cache_npz_path=box_cache_path,
    )
    if args.precompute_boxes_only:
        return

    batch_shift = load_batch_shift_init(args.audit_json)
    teacher_probs_all = None
    if args.teacher_oof_npz and os.path.exists(args.teacher_oof_npz):
        t_npz = np.load(args.teacher_oof_npz, allow_pickle=True)
        t_map = {str(nm): i for i, nm in enumerate(t_npz["names"])}
        teacher_probs_all = np.stack([t_npz["probs_6x4"][t_map[str(r["name"])]] for r in rows], axis=0)

    if args.mode in ("cv5", "stage2-rl2r"):
        full_oof_path = os.path.join(args.out_dir, f"{args.pillar}_{args.mode}_oof.npz")
        if args.fold < 0 and os.path.exists(full_oof_path):
            print(f"[{args.pillar} Resume] {full_oof_path} already exists; skipping.", flush=True)
            return

        target_folds = list(range(5)) if args.fold < 0 else [args.fold]
        gold_all = np.array([r["grades"] for r in rows], dtype=np.int64)

        for f_idx in target_folds:
            fold_npz = os.path.join(args.out_dir, f"{args.pillar}_{args.mode}_fold{f_idx}.npz")
            if os.path.exists(fold_npz):
                print(f"[{args.pillar} Resume] Fold {f_idx} already completed ({fold_npz}); skipping.", flush=True)
                continue

            tr_idx = [i for i, r in enumerate(rows) if int(r["fold"]) != f_idx]
            va_idx = [i for i, r in enumerate(rows) if int(r["fold"]) == f_idx]
            tr_teacher = teacher_probs_all[tr_idx] if teacher_probs_all is not None else None
            tr_rows_aug, tr_teacher_aug = oversample_rare_real_p1_p0([rows[i] for i in tr_idx], tr_teacher)

            ds_tr = NativeMultiViewVLMDataset(
                tr_rows_aug,
                resolver=resolver,
                box_cache=box_cache,
                is_train=True,
                oof_teacher_probs=tr_teacher_aug,
            )
            ds_va = NativeMultiViewVLMDataset(
                [rows[i] for i in va_idx],
                resolver=resolver,
                box_cache=box_cache,
                is_train=False,
                force_hflip=False,
            )
            ds_va_flip = NativeMultiViewVLMDataset(
                [rows[i] for i in va_idx],
                resolver=resolver,
                box_cache=box_cache,
                is_train=False,
                force_hflip=True,
            )
            dl_tr = DataLoader(
                ds_tr,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=_collate_multiview,
            )
            dl_va = DataLoader(
                ds_va,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=_collate_multiview,
            )
            dl_va_flip = DataLoader(
                ds_va_flip,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=_collate_multiview,
            )

            model = NativeVLMPillarModel(
                pillar_name=args.pillar,
                model_id_or_path=args.model_path or None,
                init_batch_shift=batch_shift,
                apply_dora=True,
            ).to(device)

            s1_fold_ckpt = os.path.join(args.ckpt_dir, f"{args.pillar}_cv5_fold{f_idx}.pt")
            eff_epochs = max(4, args.epochs) if args.mode == "stage2-rl2r" else args.epochs
            eff_lr_h, eff_lr_b = args.lr_head, args.lr_backbone
            if args.mode == "stage2-rl2r" and os.path.exists(s1_fold_ckpt):
                sd_prev = torch.load(s1_fold_ckpt, map_location="cpu")
                model.load_state_dict(sd_prev.get("state_dict", sd_prev), strict=False)
                eff_epochs = max(2, min(3, args.epochs))
                eff_lr_h, eff_lr_b = args.lr_head * 0.45, args.lr_backbone * 0.45
                print(f"  [{args.pillar} Fold {f_idx}] Warm-started Stage-2 from {s1_fold_ckpt}", flush=True)

            print(
                f"=== [{args.pillar} {args.mode}] Fold {f_idx}/5 | train={len(ds_tr)} (rare-P1/P0 boosted) val={len(va_idx)} (2x HFlip TTA) | device={device} ===",
                flush=True,
            )
            swa_sd = train_one_run(
                model=model,
                train_loader=dl_tr,
                device=device,
                epochs=eff_epochs,
                lr_head=eff_lr_h,
                lr_backbone=eff_lr_b,
                grad_accum=args.grad_accum,
                enable_rl2r=True,
                swa_start_epoch=max(1, eff_epochs - 2),
            )
            if swa_sd is not None:
                model.load_state_dict(swa_sd, strict=False)
                if args.mode == "cv5" and _has_safe_disk_headroom(args.ckpt_dir):
                    torch.save({"pillar": args.pillar, "state_dict": swa_sd}, s1_fold_ckpt)

            va_out = predict_vlm_dataset(model, dl_va, device, loader_hflip=dl_va_flip)
            np.savez_compressed(fold_npz, fold=np.int64(f_idx), **va_out)

            naive_pred = np.argmax(va_out["probs_6x4"], axis=-1)
            m = evaluate_official_metrics(naive_pred, gold_all[va_idx])
            print(
                f"[{args.pillar} Fold {f_idx}] Naive Argmax Strict={m['strict_total']:.2f} | Practice={m['practice_total']:.2f} "
                f"(Pos={m['strict_S_pos_pts']:.2f}, P0={m['strict_S_p0_pts']:.2f}, Clean={m['strict_S_clean_pts']:.2f})",
                flush=True,
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.fold < 0:
            aggregate_5fold_oof(rows, args.pillar, args.mode, args.out_dir)

    elif args.mode == "fulldata-swa":
        ckpt_path = os.path.join(args.ckpt_dir, f"{args.pillar}_fulldata_swa.pt")
        if os.path.exists(ckpt_path):
            print(f"[{args.pillar} Resume] Full-data SWA checkpoint exists ({ckpt_path}); skipping.", flush=True)
            return

        rows_aug, teacher_aug = oversample_rare_real_p1_p0(rows, teacher_probs_all)
        ds_full = NativeMultiViewVLMDataset(
            rows_aug,
            resolver=resolver,
            box_cache=box_cache,
            is_train=True,
            oof_teacher_probs=teacher_aug,
        )
        model = NativeVLMPillarModel(
            pillar_name=args.pillar,
            model_id_or_path=args.model_path or None,
            init_batch_shift=batch_shift,
            apply_dora=True,
        ).to(device)

        dl_warm = DataLoader(
            ds_full,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate_multiview,
        )
        warm_epochs = max(2, args.epochs - 1)
        train_one_run(
            model=model,
            train_loader=dl_warm,
            device=device,
            epochs=warm_epochs,
            lr_head=args.lr_head,
            lr_backbone=args.lr_backbone,
            grad_accum=args.grad_accum,
            enable_rl2r=True,
            swa_start_epoch=None,
        )
        warm_state = extract_trainable_state_dict(model)
        basin_swa = BasinAlignedSWACollector()

        for branch_idx, branch_seed in enumerate((2026, 2027)):
            set_deterministic_seed(branch_seed)
            model.load_state_dict(warm_state, strict=False)
            dl_branch = DataLoader(
                ds_full,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=_collate_multiview,
            )
            train_one_run(
                model=model,
                train_loader=dl_branch,
                device=device,
                epochs=2,
                lr_head=args.lr_head * 0.35,
                lr_backbone=args.lr_backbone * 0.35,
                grad_accum=args.grad_accum,
                enable_rl2r=True,
                swa_start_epoch=None,
            )
            basin_swa.update(model, weight=1.0)
            print(f"[{args.pillar} FullData Basin-SWA] Collected branch {branch_idx+1}/2 (seed={branch_seed})", flush=True)

        merged_state = basin_swa.export_state_dict()
        torch.save({"pillar": args.pillar, "state_dict": merged_state}, ckpt_path)
        print(f"[{args.pillar}] Exported SINGLE Basin-Aligned SWA adapter+head checkpoint -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
