#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/train_tier3_clef_flash9b_8fold.py
=============================================
晴羽观感体系 — 支柱三 (4.2 & 4.3 & 4.4 节):
Clef-Flash-9B 系统1单次 Prefill 隐状态 + 显式 FP32 联合模式头 (`ClefJointSchemaHead`)
+ 三图独立 Interleaved-MRoPE 原生无接缝 8 折并行微调器。

核心创新与硬性约束:
1. 彻底废除 `960x640` 硬拼贴图 (零人造阶跃接缝!):
   直接通过原生多图消息模板传入 `[img_global_native_768, roi_anatomy_384, roi_text_tex_384]` 3 张独立图像，
   由 Interleaved-MRoPE 赋予独立 image_idx 与连续 2D 空间位置编码。
2. 零自回归解码开销 (System-1 Single-Pass Prefill):
   仅执行 1 次前向 Prefill (`output_hidden_states=True`, 禁用 CoT `enable_thinking=False`)，
   直接截取中高层视觉隐藏态与末尾决策摘要隐向量送入显式 FP32 `ClefJointSchemaHead`。
3. 显式 FP32 `ClefJointSchemaHead` (4.2 节):
   - 证据路由多头交叉注意力 (Evidence-Routing Cross-Attention): 6 个可学习缺陷查询向量跨 3 路视图视觉 Token 检索局部证据；
   - 2 层跨字段自注意力 (Cross-Field Self-Attention): 建模 6 个缺陷间的耦合互斥与共生关系 (如 `模糊` vs `虚化突兀`)；
   - 单峰二次残差序数头 (`UnimodalQuadraticResidualHead`) + 128 纯真实视觉原型库 (`TipAdapterPrototypeBank`)；
   - 软序数 KL + Brier Score 概率校准损失 + 1D 平方 EMD (RPS) + XBM(512) 跨批排序损失 + 跨模态 OOF 互蒸馏。
4. 训练期启用 6 维判定准则顺序随机置换增强 (Question-Order Permutation)，消除多准则位置偏置；推理期固定标准顺序。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.config_and_paths import (
    DIMS,
    FULL_DIMS,
    P0_ONLY_DIMS,
    QINGYU_CKPT_DIR,
    REPO_ROOT,
    LosslessNativeImageLoader,
    enforce_deterministic_runtime,
    evaluate_qingyu_dual_metrics,
    load_test_template_names,
    resolve_model_path,
    setup_sys_path_and_cwd,
)
from qingyu.dual_engine_preprocess import (
    PHYS_FEAT_DIM,
    DualEnginePhysicsExtractor,
    PrecomputedDualEngineStore,
    extract_1to1_crop_pil,
    resize_native_aspect_pil,
)
from qingyu.unimodal_xbm_losses import (
    QingyuUnimodalCompositeLoss,
    UnimodalQuadraticResidualHead,
    hash_group_id_int64,
)
from qingyu.certainty_gated_stacker import compute_ece_6x4, simplex_6x4_to_p012
from qingyu.dfw_bayesian_decoder import DampedFrankWolfeBayesianDecoder


DEFECT_RUBRIC_ITEMS: Tuple[Tuple[str, str], ...] = (
    ("模糊", "【模糊】(无问题/P2/P1/P0): 考察前景主体真实对焦清晰度，严格排除自然浅景深光学背景虚化。"),
    ("纹理损失", "【纹理损失】(无问题/P2/P1/P0): 考察皮肤、织物、发丝、地面等前景高频微纹理是否出现涂抹感、塑料感或降噪过度。"),
    ("过锐", "【过锐】(无问题/P2/P1/P0): 考察真实物体轮廓是否存在 1-3px 白边/黑边光晕振铃，严格排除锐利矢量花字/UI描边。"),
    ("花屏", "【花屏】(无问题/P0): 考察解码宏块错位、局部彩色条纹撕裂或严重绿屏/紫块损坏。"),
    ("过度生成", "【过度生成】(无问题/P2/P1/P0): 考察 AIGC 解剖结构畸变(多指/断肢/五官错乱)、伪文字乱码或违反物理透视规律。"),
    ("虚化突兀", "【虚化突兀】(无问题/P0): 考察人像抠图算法在发丝、肩颈、手指边缘造成的断崖式假虚化阶跃伪影，区分自然镜头焦外渐变。"),
)


def build_clef_3image_prompt(permute_rubric: bool = False, rng: Optional[random.Random] = None) -> str:
    items = list(DEFECT_RUBRIC_ITEMS)
    if permute_rubric:
        if rng is not None:
            rng.shuffle(items)
        else:
            random.shuffle(items)
    rubric_block = "\n".join(f"{idx + 1}. {desc}" for idx, (_, desc) in enumerate(items))
    return (
        "你是晴羽观感体系首席无参画质决策专家。已输入同一帧的 3 个独立原生视图：\n"
        "- 图1：保持原始构图宽高比的全图视图（用于全局景深关系、构图物理合理性与宏块花屏判别）；\n"
        "- 图2：前景解剖/主体边缘 1:1 原生像素局部切片（用于五官手指畸变、抠图虚化突兀与主体对焦判别）；\n"
        "- 图3：高频纹理/文字细节 1:1 原生像素局部切片（用于微纹理涂抹塑料感、过锐光晕与伪文字乱码判别）。\n"
        "请严格依据以下六维客观质量准则完成系统1联合隐状态表征：\n"
        f"{rubric_block}\n"
        "请综合三图证据直接形成六维严重度决策状态。"
    )


def apply_3image_chat_template(processor: Any, prompt_text: str) -> str:
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
    tpl_str = str(getattr(processor, "chat_template", "") or "")
    extra_kw: Dict[str, Any] = {}
    if "enable_thinking" in tpl_str:
        extra_kw["enable_thinking"] = False
    try:
        return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra_kw)
    except TypeError:
        return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _infer_image_token_id(config: Any, tokenizer: Any) -> int:
    for attr in ("image_token_id", "image_token_index", "vision_start_token_id"):
        val = getattr(config, attr, None)
        if isinstance(val, int) and val >= 0:
            if attr != "vision_start_token_id":
                return val
    for cand_tok in ("<|image_pad|>", "<image>", "<|vision_pad|>", "<IMG_CONTEXT>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(cand_tok)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
                return tid
        except Exception:
            pass
    return 151655


def _infer_hidden_size(config: Any) -> int:
    if hasattr(config, "text_config") and getattr(config.text_config, "hidden_size", None):
        return int(config.text_config.hidden_size)
    if getattr(config, "hidden_size", None):
        return int(config.hidden_size)
    return 4096


def load_clef_flash_backbone_and_lora(
    model_path: str,
    lora_rank: int = 32,
    freeze_vision_tower: bool = True,
    grad_checkpointing: bool = True,
) -> Tuple[nn.Module, Any, int, int]:
    """
    自适应加载本地 `models/clef-flash` (兼容 Qwen3.5 / Clef-Flash / Qwen3-VL 多模态架构)，
    冻结视觉编码器，仅对多模态桥接器与 LLM 骨干注入 LoRA (`r=32`)，并安全剥离 `lm_head` 节省显存。
    """
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from peft import LoraConfig, get_peft_model

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    if hasattr(tok, "padding_side"):
        tok.padding_side = "right"

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    img_token_id = _infer_image_token_id(cfg, tok)
    hidden_size = _infer_hidden_size(cfg)

    loader_classes = []
    for cls_name in (
        "AutoModelForImageTextToText",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
        "AutoModel",
    ):
        cls_obj = getattr(transformers, cls_name, None)
        if cls_obj is not None and cls_obj not in loader_classes:
            loader_classes.append(cls_obj)

    full_model = None
    last_err: Optional[Exception] = None
    for attn_impl in ("flash_attention_2", "sdpa", None):
        for loader_cls in loader_classes:
            try:
                kw: Dict[str, Any] = {"trust_remote_code": True, "dtype": torch.bfloat16}
                if attn_impl is not None:
                    kw["attn_implementation"] = attn_impl
                try:
                    full_model = loader_cls.from_pretrained(model_path, **kw)
                except TypeError:
                    kw.pop("dtype", None)
                    kw["torch_dtype"] = torch.bfloat16
                    full_model = loader_cls.from_pretrained(model_path, **kw)
                break
            except Exception as e:
                last_err = e
                continue
        if full_model is not None:
            break

    if full_model is None:
        raise RuntimeError(f"无法加载 Clef-Flash-9B 模型 ({model_path}): {last_err}")

    # 剥离外层 CausalLM 的大词表 lm_head，直接使用内部多模态骨干执行单次 Prefill
    base_backbone = getattr(full_model, "model", full_model)
    if hasattr(full_model, "lm_head") and base_backbone is not full_model:
        try:
            del full_model.lm_head
        except Exception:
            pass

    # 冻结视觉编码器 (方案 4.2 节明确要求冻结视觉塔以防 3,573 样本过拟合，同时降低 55% 激活显存)
    if freeze_vision_tower:
        for n, p in base_backbone.named_parameters():
            nl = n.lower()
            if any(vk in nl for vk in ("visual.", "vision_tower.", "vision_model.")):
                if "merger" not in nl and "projector" not in nl and "aligner" not in nl:
                    p.requires_grad = False

    # 动态探测 LLM 内部可注入 LoRA 的 Linear 模块名 (兼容标准 Attention + Qwen3.5 Gated DeltaNet 线性层)
    candidate_suffixes = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj_qkvz", "in_proj_ba", "out_proj",
    }
    matched_names: List[str] = []
    for mod_name, mod in base_backbone.named_modules():
        if isinstance(mod, nn.Linear):
            nl = mod_name.lower()
            if any(vk in nl for vk in ("visual.", "vision_tower.", "vision_model.", "lm_head")):
                continue
            leaf = mod_name.split(".")[-1]
            if leaf in candidate_suffixes:
                matched_names.append(leaf)

    target_modules = sorted(set(matched_names)) if matched_names else ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    peft_backbone = get_peft_model(base_backbone, lora_cfg)

    # 保持多模态桥接器 (merger / multi_modal_projector) 可微调，对齐 1:1 ROI Token 表征
    for n, p in peft_backbone.named_parameters():
        nl = n.lower()
        if any(bk in nl for bk in ("merger", "multi_modal_projector", "aligner")):
            p.requires_grad = True

    if grad_checkpointing and hasattr(peft_backbone, "gradient_checkpointing_enable"):
        peft_backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(peft_backbone, "enable_input_require_grads"):
            peft_backbone.enable_input_require_grads()

    return peft_backbone, proc, img_token_id, hidden_size


class ClefJointSchemaHead(nn.Module):
    """
    方案 4.2 节显式 FP32 联合模式头 (`JointSchemaHead`):
      1. 多深度视觉 Token 金字塔融合 (50% + 75% + 100% 层隐状态) + 3 视图位置嵌入 (`Global / AnatomyROI / TextTexROI`)；
      2. 证据路由多头交叉注意力 (`Evidence-Routing Cross-Attention`): 6 个缺陷专属查询向量主动检索三图视觉 Token；
      3. 2 层跨字段自注意力 Transformer (`Cross-Field Self-Attention`): 显式建模 6 类缺陷耦合与互斥结构；
      4. 汇总 6 个缺陷上下文 + 全局指令摘要隐状态 + 36D 单图物理向量，接入 `UnimodalQuadraticResidualHead`
         与 128 纯真实原型库 `TipAdapterPrototypeBank`。
    全程强制运行于 FP32，彻底免疫 BF16 舍入误差与多步文本解码截断。
    """

    def __init__(
        self,
        llm_hidden_size: int,
        d_model: int = 768,
        num_heads: int = 8,
        phys_dim: int = PHYS_FEAT_DIM,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.llm_hidden_size = int(llm_hidden_size)
        self.d_model = int(d_model)

        # 中高层视觉 Token 投影 + 零初始化金字塔残差适配器
        self.vis_proj_last = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, d_model),
        )
        self.vis_proj_mid = nn.Sequential(
            nn.LayerNorm(llm_hidden_size * 2),
            nn.Linear(llm_hidden_size * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.vis_proj_mid[-1].weight)
        nn.init.zeros_(self.vis_proj_mid[-1].bias)

        # 3 路独立图像视图嵌入: 0=GlobalNative, 1=ROI_Anatomy_1to1, 2=ROI_TextTex_1to1
        self.view_embed = nn.Embedding(3, d_model)
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, d_model),
            nn.GELU(),
        )

        # 6 个可学习缺陷查询向量 Q_def in R^{6 x d_model}
        self.defect_queries = nn.Parameter(torch.randn(1, 6, d_model) * 0.02)
        self.evidence_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # 2 层跨字段自注意力编码器 (Cross-Field Reasoning across the 6 defects)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_field_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # 融合 6 个字段专属状态 (6 * 256) + 全局决策摘要 (d_model) -> UnimodalQuadraticResidualHead
        self.per_defect_compress = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
        )
        fused_in_dim = 6 * 256 + d_model
        self.unimodal_head = UnimodalQuadraticResidualHead(
            in_dim=fused_in_dim,
            phys_dim=phys_dim,
            hidden_dim=768,
            proj_dim=512,
            dropout=0.20,
            max_prototypes=128,
        )

    def forward(
        self,
        vis_tokens_list: List[torch.Tensor],
        vis_mid_list: List[torch.Tensor],
        vis_view_ids_list: List[torch.Tensor],
        summary_vecs: torch.Tensor,
        phys_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 全程强制 FP32 运算 (兼容 PyTorch 2.x 无弃用警告)
        amp_dev = "cuda" if summary_vecs.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            B = summary_vecs.shape[0]
            dev = summary_vecs.device
            sum_proj = self.summary_proj(summary_vecs.float())  # [B, d_model]

            proj_tokens_per_b: List[torch.Tensor] = []
            max_len = 1
            for b in range(B):
                vt = vis_tokens_list[b].float()
                vm = vis_mid_list[b].float()
                vids = vis_view_ids_list[b].long().clamp(0, 2)
                if vt.shape[0] > 0:
                    tok = self.vis_proj_last(vt) + 0.35 * self.vis_proj_mid(vm) + self.view_embed(vids)
                else:
                    tok = sum_proj[b : b + 1]
                proj_tokens_per_b.append(tok)
                if tok.shape[0] > max_len:
                    max_len = tok.shape[0]

            kv_pad = summary_vecs.new_zeros((B, max_len, self.d_model), dtype=torch.float32)
            key_padding_mask = torch.ones((B, max_len), dtype=torch.bool, device=dev)
            for b, tok in enumerate(proj_tokens_per_b):
                L = tok.shape[0]
                kv_pad[b, :L] = tok
                key_padding_mask[b, :L] = False

            # 6 个缺陷查询注入全局指令上下文后检索三图局部证据
            q_init = self.defect_queries.expand(B, -1, -1) + 0.25 * sum_proj.unsqueeze(1)  # [B, 6, d_model]
            attn_out, _ = self.evidence_cross_attn(
                query=q_init,
                key=kv_pad,
                value=kv_pad,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            field_states = self.cross_norm(q_init + attn_out)
            field_states = self.cross_field_transformer(field_states)  # [B, 6, d_model]

            comp_fields = self.per_defect_compress(field_states).reshape(B, 6 * 256)
            joint_feat = torch.cat([comp_fields, sum_proj], dim=-1)
            return self.unimodal_head(joint_feat, phys_feat=phys_feat.float() if phys_feat is not None else None)


class ClefFlash9BSystem1Tower(nn.Module):
    """封装 Clef-Flash-9B 多图 Prefill 骨干 + FP32 `ClefJointSchemaHead`。"""

    def __init__(self, peft_backbone: nn.Module, image_token_id: int, hidden_size: int, phys_dim: int = PHYS_FEAT_DIM):
        super().__init__()
        self.base = peft_backbone
        self.image_token_id = int(image_token_id)
        self.schema_head = ClefJointSchemaHead(
            llm_hidden_size=hidden_size,
            d_model=768,
            num_heads=8,
            phys_dim=phys_dim,
        ).float()

    @property
    def head(self) -> UnimodalQuadraticResidualHead:
        return self.schema_head.unimodal_head

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in self.state_dict().items():
            kl = k.lower()
            if (
                k.startswith("schema_head.")
                or "lora_" in kl
                or "merger" in kl
                or "multi_modal_projector" in kl
                or "aligner" in kl
            ):
                out[k] = v.detach().cpu().clone()
        return out

    @staticmethod
    def _assign_3view_ids(num_vis_tokens: int, image_grid_thw: Optional[torch.Tensor], sample_offset: int) -> torch.Tensor:
        """根据每张图的合并后空间 Token 计数为 `[img_global, roi_anatomy, roi_text_tex]` 分配视图 ID `{0,1,2}`。"""
        if image_grid_thw is None or image_grid_thw.shape[0] < sample_offset + 3:
            # 均匀近似分段回退
            c0 = max(1, num_vis_tokens // 2)
            c1 = max(1, (num_vis_tokens - c0) // 2)
            c2 = max(0, num_vis_tokens - c0 - c1)
            return torch.cat([
                torch.zeros(c0, dtype=torch.long),
                torch.ones(c1, dtype=torch.long),
                torch.full((c2,), 2, dtype=torch.long),
            ])[:num_vis_tokens]

        raw_counts = []
        for k in range(3):
            thw = image_grid_thw[sample_offset + k]
            raw_counts.append(int(thw[0].item() * thw[1].item() * thw[2].item()))
        raw_sum = max(1, sum(raw_counts))
        # Qwen 视觉 Merger 通常做 2x2=4 空间聚合，按比例精确映射至实际序列视觉 Token 数
        c0 = int(round(num_vis_tokens * raw_counts[0] / raw_sum))
        c1 = int(round(num_vis_tokens * raw_counts[1] / raw_sum))
        c2 = max(0, num_vis_tokens - c0 - c1)
        ids = torch.cat([
            torch.zeros(max(0, c0), dtype=torch.long),
            torch.ones(max(0, c1), dtype=torch.long),
            torch.full((max(0, c2),), 2, dtype=torch.long),
        ])
        if ids.numel() < num_vis_tokens:
            ids = F.pad(ids, (0, num_vis_tokens - ids.numel()), value=2)
        return ids[:num_vis_tokens]

    def forward(self, model_inputs: Dict[str, torch.Tensor], phys_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        image_grid_thw = model_inputs.get("image_grid_thw", None)

        fwd_kwargs = dict(model_inputs)
        fwd_kwargs["output_hidden_states"] = True
        fwd_kwargs["return_dict"] = True
        fwd_kwargs["use_cache"] = False

        out = self.base(**fwd_kwargs)
        hs = getattr(out, "hidden_states", None)
        h_last = out.last_hidden_state if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None else hs[-1]
        if hs is not None and len(hs) >= 6:
            n_l = len(hs) - 1
            h_50 = hs[max(1, n_l // 2)]
            h_75 = hs[max(2, (3 * n_l) // 4)]
        else:
            h_50, h_75 = h_last, h_last

        B = h_last.shape[0]
        dev = h_last.device
        seq_lens = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        summary_vecs = h_last[torch.arange(B, device=dev), seq_lens].float()

        img_mask = input_ids == self.image_token_id
        vis_last_list: List[torch.Tensor] = []
        vis_mid_list: List[torch.Tensor] = []
        vis_view_list: List[torch.Tensor] = []

        for b in range(B):
            m = img_mask[b]
            if m.any():
                vt_100 = h_last[b, m].float()
                vt_mid = torch.cat([h_50[b, m].float(), h_75[b, m].float()], dim=-1)
                vids = self._assign_3view_ids(int(vt_100.shape[0]), image_grid_thw, b * 3).to(dev)
            else:
                vt_100 = summary_vecs[b : b + 1]
                vt_mid = torch.cat([h_50[b, seq_lens[b] : seq_lens[b] + 1].float(), h_75[b, seq_lens[b] : seq_lens[b] + 1].float()], dim=-1)
                vids = torch.zeros(1, dtype=torch.long, device=dev)
            vis_last_list.append(vt_100)
            vis_mid_list.append(vt_mid)
            vis_view_list.append(vids)

        return self.schema_head(
            vis_tokens_list=vis_last_list,
            vis_mid_list=vis_mid_list,
            vis_view_ids_list=vis_view_list,
            summary_vecs=summary_vecs,
            phys_feat=phys_feat,
        )


def compute_brier_calibration_loss(probs_6x4: torch.Tensor, grades: torch.Tensor) -> torch.Tensor:
    """方案 4.2 节显式多分类 Brier Score 严格正常评分规则正则项，惩罚过度自信对数几率尖峰。"""
    y = grades.long()
    terms = []
    for d in FULL_DIMS:
        one_hot = F.one_hot(y[:, d].clamp(0, 3), num_classes=4).float()
        terms.append(((probs_6x4[:, d, :].float() - one_hot) ** 2).sum(dim=-1).mean())
    for d in P0_ONLY_DIMS:
        yb = (y[:, d] >= 3).float()
        terms.append(2.0 * ((probs_6x4[:, d, 3].float() - yb) ** 2).mean())
    return torch.stack(terms).mean()


def load_fold_rows(split_csv: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    full_p = split_csv if os.path.isabs(split_csv) else os.path.join(REPO_ROOT, split_csv)
    tr, va = [], []
    with open(full_p, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = {
                "name": r["name"].strip(),
                "group_id": r.get("group_id", r["name"].split("_")[0]).strip(),
                "grades": [int(r[c]) for c in DIMS],
            }
            if r["split"].strip() == "val":
                va.append(item)
            else:
                tr.append(item)
    return tr, va


class Tier3MultiImageMRoPEDataset(Dataset):
    """生成 `[img_global_native, roi_anatomy_384, roi_text_tex_384]` 三图无接缝输入与 36D 单图物理向量。"""

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        img_loader: LosslessNativeImageLoader,
        meta_store: PrecomputedDualEngineStore,
        global_max_pixels: int = 768 * 768,
        train: bool = False,
        hflip_tta: bool = False,
        permute_prompt: bool = False,
        teacher_oof_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.rows = list(rows)
        self.loader = img_loader
        self.meta_store = meta_store
        self.global_max_pixels = int(global_max_pixels)
        self.train = bool(train)
        self.hflip_tta = bool(hflip_tta)
        self.permute_prompt = bool(permute_prompt)
        self.teacher_oof_map = teacher_oof_map or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        nm = str(r["name"])
        im = self.loader.load_pil(nm)
        meta = self.meta_store.get(nm, pil_im=im)
        orig_w, _ = im.size

        do_flip = self.hflip_tta or (self.train and random.random() < 0.5)
        r_boxes = meta["rois_384_boxes"].copy()
        if do_flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            old_x0 = r_boxes[:, 0].copy()
            old_x1 = r_boxes[:, 2].copy()
            r_boxes[:, 0] = np.maximum(0, orig_w - old_x1)
            r_boxes[:, 2] = r_boxes[:, 0] + 384

        if self.train:
            if random.random() < 0.30:
                im = ImageEnhance.Brightness(im).enhance(1.0 + random.uniform(-0.04, 0.04))
            if random.random() < 0.30:
                im = ImageEnhance.Contrast(im).enhance(1.0 + random.uniform(-0.04, 0.04))

        img_global = resize_native_aspect_pil(
            im, max_pixels=self.global_max_pixels, min_pixels=256 * 256, patch_mult=32
        )
        roi_anat = extract_1to1_crop_pil(im, r_boxes[0], crop_size=384)
        roi_text = extract_1to1_crop_pil(im, r_boxes[1], crop_size=384)

        prompt = build_clef_3image_prompt(permute_rubric=(self.train and self.permute_prompt))
        phys = torch.from_numpy(np.asarray(meta["phys_vec"], dtype=np.float32).copy())
        grades = torch.tensor(r["grades"], dtype=torch.long)
        ghash = hash_group_id_int64(str(r.get("group_id", nm)))
        t_prob = self.teacher_oof_map.get(nm, None)
        t_tensor = torch.from_numpy(t_prob.copy()) if t_prob is not None else torch.zeros((6, 4), dtype=torch.float32)

        return {
            "images_3": [img_global, roi_anat, roi_text],
            "prompt": prompt,
            "phys": phys,
            "grades": grades,
            "ghash": ghash,
            "name": nm,
            "teacher_6x4": t_tensor,
            "has_teacher": 1 if t_prob is not None else 0,
        }


def make_clef_3image_collate(processor: Any):
    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [apply_3image_chat_template(processor, b["prompt"]) for b in batch]
        flat_images: List[Image.Image] = []
        for b in batch:
            flat_images.extend(b["images_3"])
        enc = processor(text=texts, images=flat_images, padding=True, return_tensors="pt")
        return {
            "model_inputs": dict(enc),
            "phys": torch.stack([b["phys"] for b in batch], dim=0),
            "grades": torch.stack([b["grades"] for b in batch], dim=0),
            "ghash": torch.tensor([b["ghash"] for b in batch], dtype=torch.long),
            "names": [b["name"] for b in batch],
            "teacher_6x4": torch.stack([b["teacher_6x4"] for b in batch], dim=0),
            "has_teacher": all(b["has_teacher"] == 1 for b in batch),
        }

    return _collate


def _move_inputs_to_device(model_inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def populate_clef_prototypes(
    model: ClefFlash9BSystem1Tower,
    tr_rows: Sequence[Dict[str, Any]],
    img_loader: LosslessNativeImageLoader,
    meta_store: PrecomputedDualEngineStore,
    collate_fn: Any,
    device: torch.device,
    global_max_pixels: int,
    batch_size: int,
    workers: int,
) -> int:
    model.eval()
    model.head.proto_bank.num_active.fill_(0)
    ds = Tier3MultiImageMRoPEDataset(
        tr_rows, img_loader, meta_store, global_max_pixels=global_max_pixels, train=False, hflip_tta=False
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    zs, ys, ps = [], [], []
    for b in dl:
        m_in = _move_inputs_to_device(b["model_inputs"], device)
        phys = b["phys"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(m_in, phys)
        zs.append(out["z_norm"].float().cpu().numpy())
        ys.append(b["grades"].numpy())
        ps.append(b["phys"].numpy())
    return model.head.proto_bank.fit_from_train_embeddings(
        np.concatenate(zs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(ps, axis=0),
    )


@torch.no_grad()
def predict_clef_2x_tta(
    model: ClefFlash9BSystem1Tower,
    rows: Sequence[Dict[str, Any]],
    img_loader: LosslessNativeImageLoader,
    meta_store: PrecomputedDualEngineStore,
    collate_fn: Any,
    device: torch.device,
    global_max_pixels: int,
    batch_size: int,
    workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    probs_passes = []
    qclean_passes = []
    gold_out = None
    names_out: List[str] = []

    for flip_flag in (False, True):
        ds = Tier3MultiImageMRoPEDataset(
            rows,
            img_loader,
            meta_store,
            global_max_pixels=global_max_pixels,
            train=False,
            hflip_tta=flip_flag,
            permute_prompt=False,
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
        p_list, qc_list, g_list, n_list = [], [], [], []
        for b in dl:
            m_in = _move_inputs_to_device(b["model_inputs"], device)
            phys = b["phys"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(m_in, phys)
            p_list.append(out["probs_6x4"].float().cpu().numpy())
            qc_list.append(torch.sigmoid(out["clean_logit"].float()).cpu().numpy())
            if not flip_flag:
                g_list.append(b["grades"].numpy())
                n_list.extend(b["names"])
        probs_passes.append(np.concatenate(p_list, axis=0))
        qclean_passes.append(np.concatenate(qc_list, axis=0))
        if not flip_flag:
            gold_out = np.concatenate(g_list, axis=0)
            names_out = n_list

    return (
        (0.5 * (probs_passes[0] + probs_passes[1])).astype(np.float32),
        (0.5 * (qclean_passes[0] + qclean_passes[1])).astype(np.float32),
        gold_out,
        names_out,
    )


def train_single_fold(args: argparse.Namespace) -> str:
    setup_sys_path_and_cwd(chdir_repo=True)
    enforce_deterministic_runtime(args.seed + args.fold)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    split_csv = os.path.join(args.split_dir, f"fold_{args.fold}.csv")
    tr_rows, va_rows = load_fold_rows(split_csv)

    # 严格折外跨模态互蒸馏教师过滤 (验证折样本绝对禁止进入 teacher_map)
    teacher_map: Dict[str, np.ndarray] = {}
    if args.oof_teacher_npz and os.path.exists(args.oof_teacher_npz):
        td = np.load(args.oof_teacher_npz, allow_pickle=True)
        t_names = [str(x) for x in td["oof_names"]]
        t_probs = np.asarray(td["oof_probs_6x4"], dtype=np.float32)
        va_set = set(r["name"] for r in va_rows)
        for i, nm in enumerate(t_names):
            if nm not in va_set:
                teacher_map[nm] = t_probs[i]

    extra_dirs = [args.extra_test_dir] if args.extra_test_dir else None
    img_loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=extra_dirs)
    lazy_ext = DualEnginePhysicsExtractor(device=str(device), allow_depth_download=True)
    meta_store = PrecomputedDualEngineStore(args.dual_engine_npz, lazy_extractor=lazy_ext)

    model_path = args.model_path or resolve_model_path("clef_flash_9b")
    peft_backbone, processor, img_token_id, hidden_size = load_clef_flash_backbone_and_lora(
        model_path=model_path,
        lora_rank=args.lora_rank,
        freeze_vision_tower=True,
        grad_checkpointing=True,
    )
    model = ClefFlash9BSystem1Tower(
        peft_backbone=peft_backbone,
        image_token_id=img_token_id,
        hidden_size=hidden_size,
        phys_dim=PHYS_FEAT_DIM,
    ).to(device)
    model.schema_head.float()

    # 仅在当前训练折上拟合单图物理特征均值与标准差
    tr_phys_mat = np.stack(
        [meta_store.get(str(r["name"]), pil_im=img_loader.load_pil(str(r["name"])) if str(r["name"]) not in meta_store.map else None)["phys_vec"] for r in tr_rows],
        axis=0,
    )
    model.head.set_physics_scaler(tr_phys_mat.mean(axis=0), tr_phys_mat.std(axis=0) + 1e-5)

    collate_fn = make_clef_3image_collate(processor)
    ds_tr = Tier3MultiImageMRoPEDataset(
        tr_rows,
        img_loader,
        meta_store,
        global_max_pixels=args.global_max_pixels,
        train=True,
        permute_prompt=True,
        teacher_oof_map=teacher_map,
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("base.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("base.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.01},
            {"params": head_params, "lr": args.lr_head, "weight_decay": 0.01},
        ]
    )
    grad_accum = max(1, int(args.grad_accum))
    opt_steps_per_epoch = max(1, math.ceil(len(dl_tr) / grad_accum))
    total_opt_steps = max(1, opt_steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_lora, args.lr_head],
        total_steps=total_opt_steps,
        pct_start=0.08,
    )
    criterion = QingyuUnimodalCompositeLoss(
        soft_sigma=0.35, lambda_rps=0.45, lambda_rank=0.25, lambda_delta=1e-2, xbm_capacity=512
    ).to(device)

    res_dir = os.path.join(REPO_ROOT, args.out_dir, "clef_flash_9b")
    ckpt_dir = QINGYU_CKPT_DIR / "clef_flash_9b"
    os.makedirs(res_dir, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = os.path.join(res_dir, f"fold_{args.fold}_best.pt")

    print(
        f"=== [Qingyu Tier-3 Clef-Flash-9B System-1] Fold {args.fold}/8 | train={len(tr_rows)} val={len(va_rows)} "
        f"| 3-Image Interleaved-MRoPE | img_tok={img_token_id} hid={hidden_size} | device={device} ===",
        flush=True,
    )

    best_obj = -1e9
    best_sd: Optional[Dict[str, torch.Tensor]] = None
    prev_sd: Optional[Dict[str, torch.Tensor]] = None

    for ep in range(args.epochs):
        model.train()
        criterion.train()
        t0 = time.time()
        loss_acc = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step_idx, b in enumerate(dl_tr):
            m_in = _move_inputs_to_device(b["model_inputs"], device)
            phys = b["phys"].to(device, non_blocking=True)
            grades_d = b["grades"].to(device, non_blocking=True)
            ghash_d = b["ghash"].to(device, non_blocking=True)
            t_kd = b["teacher_6x4"].to(device, non_blocking=True) if b["has_teacher"] else None
            lam_kd = args.lambda_kd if b["has_teacher"] else 0.0

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(m_in, phys)
            losses = criterion(out, grades_d, ghash_d, teacher_probs_6x4=t_kd, lambda_kd=lam_kd)
            brier = compute_brier_calibration_loss(out["probs_6x4"], grades_d)
            raw_loss = losses["total"] + args.lambda_brier * brier

            (raw_loss / grad_accum).backward()
            if (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == len(dl_tr):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                if scheduler.last_epoch < total_opt_steps - 1:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_acc += float(raw_loss.item())
            n_batches += 1

        n_proto = (
            populate_clef_prototypes(
                model,
                tr_rows,
                img_loader,
                meta_store,
                collate_fn,
                device,
                args.global_max_pixels,
                batch_size=args.batch_size * 2,
                workers=args.workers,
            )
            if ep >= max(0, args.epochs - 2)
            else 0
        )

        vp, vqc, gv, _ = predict_clef_2x_tta(
            model,
            va_rows,
            img_loader,
            meta_store,
            collate_fn,
            device,
            args.global_max_pixels,
            batch_size=args.batch_size * 2,
            workers=args.workers,
        )
        qdec = DampedFrankWolfeBayesianDecoder(dfw_steps=20, eta_doc1=0.50, max_clean_p0_fpr=0.008).fit(vp, vqc, gv)
        ev = evaluate_qingyu_dual_metrics(qdec.decode(vp, vqc), gv)
        ece = compute_ece_6x4(vp, gv)
        val_obj = 0.5 * ev["doc1_total"] + 0.5 * ev["doc2_strict_total"] - 2.0 * ece

        print(
            f"  [Clef-Flash-9B F{args.fold}] Ep {ep+1}/{args.epochs} | loss={loss_acc/max(1,n_batches):.4f} | "
            f"Doc1={ev['doc1_total']:.2f} | Doc2={ev['doc2_strict_total']:.2f} | ECE={ece:.4f} | protos={n_proto} ({time.time()-t0:.0f}s)",
            flush=True,
        )

        cur_sd = model.trainable_state_dict()
        if ep == max(0, args.epochs - 2):
            prev_sd = cur_sd
        if val_obj > best_obj:
            best_obj = val_obj
            best_sd = cur_sd
            torch.save({"state_dict": best_sd, "best_val_score": float(val_obj)}, ckpt_best)

    # SWA 校验
    if best_sd is not None and prev_sd is not None:
        swa_sd = {
            k: (0.5 * prev_sd[k].float() + 0.5 * best_sd[k].float()).to(best_sd[k].dtype)
            if best_sd[k].is_floating_point()
            else best_sd[k]
            for k in best_sd.keys()
        }
        model.load_state_dict(swa_sd, strict=False)
        populate_clef_prototypes(
            model, tr_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size * 2, args.workers
        )
        vp_s, vqc_s, gv_s, _ = predict_clef_2x_tta(
            model, va_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size * 2, args.workers
        )
        qdec_s = DampedFrankWolfeBayesianDecoder(dfw_steps=20, eta_doc1=0.50, max_clean_p0_fpr=0.008).fit(vp_s, vqc_s, gv_s)
        ev_s = evaluate_qingyu_dual_metrics(qdec_s.decode(vp_s, vqc_s), gv_s)
        ece_s = compute_ece_6x4(vp_s, gv_s)
        score_s = 0.5 * ev_s["doc1_total"] + 0.5 * ev_s["doc2_strict_total"] - 2.0 * ece_s
        if score_s >= best_obj - 0.08:
            best_obj = max(best_obj, score_s)
            best_sd = model.trainable_state_dict()

    if best_sd is not None:
        model.load_state_dict(best_sd, strict=False)
        populate_clef_prototypes(
            model, tr_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size * 2, args.workers
        )
        final_payload = {"state_dict": model.trainable_state_dict(), "best_val_score": float(best_obj)}
        torch.save(final_payload, ckpt_best)
        shutil.copy2(ckpt_best, str(ckpt_dir / f"fold_{args.fold}_best.pt"))

    va_probs, va_qclean, gva, va_names = predict_clef_2x_tta(
        model, va_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size * 2, args.workers
    )
    va_p0, va_p1, va_p2 = simplex_6x4_to_p012(va_probs)

    ordered_150 = load_test_template_names("results/v2_full/template_150.json")
    t150_rows = [{"name": nm, "group_id": nm, "grades": [0] * 6} for nm in ordered_150]
    t150_probs, t150_qclean, _, _ = predict_clef_2x_tta(
        model, t150_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size * 2, args.workers
    )
    t150_p0, t150_p1, t150_p2 = simplex_6x4_to_p012(t150_probs)

    save_dict: Dict[str, Any] = dict(
        fold=np.int64(args.fold),
        va_probs_6x4=va_probs,
        va_qclean=va_qclean,
        va_p0=va_p0,
        va_p1=va_p1,
        va_p2=va_p2,
        gva=gva,
        va_names=np.array(va_names),
        t150_probs_6x4=t150_probs,
        t150_qclean=t150_qclean,
        t150_p0=t150_p0,
        t150_p1=t150_p1,
        t150_p2=t150_p2,
        t150_names=np.array(ordered_150),
    )

    if args.extra_test_template and os.path.exists(args.extra_test_template):
        ext_names = load_test_template_names(args.extra_test_template)
        ext_rows = [{"name": nm, "group_id": nm, "grades": [0] * 6} for nm in ext_names]
        ext_probs, ext_qclean, _, _ = predict_clef_2x_tta(
            model, ext_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size * 2, args.workers
        )
        save_dict["extra_probs_6x4"] = ext_probs
        save_dict["extra_qclean"] = ext_qclean
        save_dict["extra_names"] = np.array(ext_names)

    out_npz = os.path.join(res_dir, f"fold_{args.fold}_probs.npz")
    np.savez_compressed(out_npz, **save_dict)
    print(f"[Qingyu Tier-3 Clef-Flash-9B] Saved Fold {args.fold} OOF & test -> {out_npz}", flush=True)
    return out_npz


def aggregate_8fold_oof(out_dir: str, n_folds: int = 8) -> str:
    setup_sys_path_and_cwd(chdir_repo=True)
    b_dir = os.path.join(REPO_ROOT, out_dir, "clef_flash_9b")
    parts = [np.load(os.path.join(b_dir, f"fold_{f}_probs.npz"), allow_pickle=True) for f in range(n_folds)]

    oof_names = np.concatenate([p["va_names"] for p in parts], axis=0)
    oof_probs = np.concatenate([p["va_probs_6x4"] for p in parts], axis=0)
    oof_qclean = np.concatenate([p["va_qclean"] for p in parts], axis=0)
    oof_gold = np.concatenate([p["gva"] for p in parts], axis=0)
    oof_folds = np.concatenate([np.full(len(p["va_names"]), f, dtype=np.int64) for f, p in enumerate(parts)], axis=0)

    t150_probs = np.mean([p["t150_probs_6x4"] for p in parts], axis=0).astype(np.float32)
    t150_qclean = np.mean([p["t150_qclean"] for p in parts], axis=0).astype(np.float32)
    t150_names = parts[0]["t150_names"]

    dec = DampedFrankWolfeBayesianDecoder(dfw_steps=30, eta_doc1=0.50, max_clean_p0_fpr=0.005).fit(
        oof_probs, oof_qclean, oof_gold
    )
    ev = evaluate_qingyu_dual_metrics(dec.decode(oof_probs, oof_qclean), oof_gold)
    ece = compute_ece_6x4(oof_probs, oof_gold)
    print(
        f"=== [Qingyu Tier-3 8-Fold OOF Summary] clef_flash_9b (N={len(oof_gold)}) | "
        f"Doc1={ev['doc1_total']:.3f} | Doc2_Strict={ev['doc2_strict_total']:.3f} | ECE={ece:.4f} ===",
        flush=True,
    )

    agg_dict: Dict[str, Any] = dict(
        backbone="clef_flash_9b",
        oof_names=oof_names,
        oof_probs_6x4=oof_probs,
        oof_qclean=oof_qclean,
        oof_gold=oof_gold,
        oof_folds=oof_folds,
        t150_names=t150_names,
        t150_probs_6x4=t150_probs,
        t150_qclean=t150_qclean,
    )
    if all("extra_probs_6x4" in p.files for p in parts):
        agg_dict["extra_names"] = parts[0]["extra_names"]
        agg_dict["extra_probs_6x4"] = np.mean([p["extra_probs_6x4"] for p in parts], axis=0).astype(np.float32)
        agg_dict["extra_qclean"] = np.mean([p["extra_qclean"] for p in parts], axis=0).astype(np.float32)

    agg_path = os.path.join(REPO_ROOT, out_dir, "tier3_clef_flash_9b_8fold_oof.npz")
    np.savez_compressed(agg_path, **agg_dict)
    with open(agg_path.replace(".npz", ".json"), "w", encoding="utf-8") as jf:
        json.dump({"backbone": "clef_flash_9b", "oof_metrics": ev, "ece": ece}, jf, ensure_ascii=False, indent=2)
    return agg_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Qingyu Tier-3 Clef-Flash-9B System-1 8-Fold Trainer")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--split-dir", default="data/qingyu_splits")
    ap.add_argument("--dual-engine-npz", default="results/qingyu/dual_engine_meta.npz")
    ap.add_argument("--out-dir", default="results/qingyu")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--global-max-pixels", type=int, default=768 * 768)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr-lora", type=float, default=1.5e-4)
    ap.add_argument("--lr-head", type=float, default=6.0e-4)
    ap.add_argument("--lambda-brier", type=float, default=0.20)
    ap.add_argument("--oof-teacher-npz", default="")
    ap.add_argument("--lambda-kd", type=float, default=0.30)
    ap.add_argument("--extra-test-template", default="")
    ap.add_argument("--extra-test-dir", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--aggregate-8fold", action="store_true")
    args = ap.parse_args()

    if args.aggregate_8fold:
        aggregate_8fold_oof(args.out_dir)
    else:
        train_single_fold(args)


if __name__ == "__main__":
    main()
