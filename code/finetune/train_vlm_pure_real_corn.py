#!/usr/bin/env python3
"""
code/finetune/train_vlm_pure_real_corn.py
Pure-Real 5-Fold Qwen3-VL-8B Specialist with Visual-Token Gated ABMIL + CORN + Ordinal EMD.
Directly targets Over-Generation (extra fingers, melted limbs, garbled text) and Abrupt Bokeh
at native dynamic resolution up to 1024x1024.
"""
import os
import json
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import (
    DIMS, FULL_DIMS, P0_ONLY_DIMS,
    labels_to_14ch, eval_14ch,
)

MODEL_PATH = "/wanqing-develop/luowenjing/18/models/Qwen3-VL-8B"
DIAGNOSTIC_PROMPT = (
    "请作为专业画质与AIGC缺陷检测专家，仔细审视图像细节："
    "1.是否存在过度生成与AI结构崩坏（如多余或残缺手指、肢体融化粘连、五官扭曲、文字乱码）；"
    "2.是否存在背景虚化突兀（抠图边缘断层、景深过渡不自然）；"
    "3.是否存在模糊、纹理涂抹损失、边缘过锐伪影或花屏色块。"
    "请综合给出画质缺陷表征。"
)


def labels_to_corn_targets(y_6d: np.ndarray):
    N = y_6d.shape[0]
    t_corn = np.zeros((N, 14), dtype=np.float32)
    m_corn = np.ones((N, 14), dtype=np.float32)

    for j, d in enumerate(FULL_DIMS):
        y = y_6d[:, d]
        t_corn[:, 10 + j] = (y >= 1).astype(np.float32)
        m_corn[:, 10 + j] = 1.0
        t_corn[:, 6 + j] = (y >= 2).astype(np.float32)
        m_corn[:, 6 + j] = (y >= 1).astype(np.float32)
        t_corn[:, d] = (y >= 3).astype(np.float32)
        m_corn[:, d] = (y >= 2).astype(np.float32)

    for d in P0_ONLY_DIMS:
        y = y_6d[:, d]
        t_corn[:, d] = (y >= 3).astype(np.float32)
        m_corn[:, d] = 1.0

    return t_corn, m_corn


def corn_logits_to_unconditional_14ch(logits: torch.Tensor) -> torch.Tensor:
    q = torch.sigmoid(logits)
    probs = torch.zeros_like(q)
    for j, d in enumerate(FULL_DIMS):
        p_ge1 = q[:, 10 + j]
        p_ge2 = p_ge1 * q[:, 6 + j]
        p_ge3 = p_ge2 * q[:, d]
        probs[:, 10 + j] = p_ge1
        probs[:, 6 + j] = p_ge2
        probs[:, d] = p_ge3
    for d in P0_ONLY_DIMS:
        probs[:, d] = q[:, d]
    return probs


class Qwen3VL_ABMIL_CORN(nn.Module):
    def __init__(self, base_model, hidden_size: int = 4096):
        super().__init__()
        self.base = base_model
        self.image_token_id = 151655

        self.vis_anom_score = nn.Linear(hidden_size, 1)
        self.vis_attn_V = nn.Linear(hidden_size, 256)
        self.vis_attn_U = nn.Linear(hidden_size, 256)
        self.vis_attn_w = nn.Linear(256, 1)

        in_dim = hidden_size * 4
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(0.25),
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.15),
            nn.Linear(1024, 14),
        )

    def _pool_visual_tokens(self, v_toks: torch.Tensor):
        M, D = v_toks.shape
        v_mean = v_toks.mean(dim=0)
        if M <= 4:
            return v_mean, v_mean, v_mean

        k = max(1, int(math.ceil(M * 0.05)))
        s = self.vis_anom_score(v_toks).squeeze(-1) / math.sqrt(D)
        top_vals, top_idx = torch.topk(s, k=k, dim=0)
        top_w = F.softmax(top_vals, dim=0).unsqueeze(-1)
        v_top5 = (top_w * v_toks[top_idx]).sum(dim=0)

        mil_a = self.vis_attn_w(
            torch.tanh(self.vis_attn_V(v_toks)) * torch.sigmoid(self.vis_attn_U(v_toks))
        ).squeeze(-1)
        mil_w = F.softmax(mil_a, dim=0).unsqueeze(-1)
        v_abmil = (mil_w * v_toks).sum(dim=0)
        return v_mean, v_abmil, v_top5

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.hidden_states[-1]
        B = h.shape[0]

        seq_lens = attention_mask.sum(dim=1) - 1
        last_vecs = h[torch.arange(B, device=h.device), seq_lens]

        img_mask = (input_ids == self.image_token_id)
        vis_means, vis_abmils, vis_top5s = [], [], []
        for b in range(B):
            m = img_mask[b]
            if m.any():
                vm, va, vt = self._pool_visual_tokens(h[b, m])
            else:
                vm = va = vt = last_vecs[b]
            vis_means.append(vm)
            vis_abmils.append(va)
            vis_top5s.append(vt)

        feat = torch.cat([
            torch.stack(vis_means, dim=0),
            torch.stack(vis_abmils, dim=0),
            torch.stack(vis_top5s, dim=0),
            last_vecs,
        ], dim=-1)
        return self.head(feat.float())


def compute_corn_emd_loss(logits: torch.Tensor, t_corn: torch.Tensor, m_corn: torch.Tensor, y_14ch: torch.Tensor):
    bce_raw = F.binary_cross_entropy_with_logits(logits, t_corn, reduction="none")
    corn_loss = (bce_raw * m_corn).sum() / m_corn.sum().clamp(min=1.0)

    probs_14 = corn_logits_to_unconditional_14ch(logits)
    emd_loss = F.mse_loss(probs_14, y_14ch)

    ch_weights = torch.ones(14, device=logits.device)
    ch_weights[[2, 8, 12]] = 2.5
    ch_weights[3] = 2.0

    tp = (probs_14 * y_14ch).sum(dim=0)
    fp = (probs_14 * (1.0 - y_14ch)).sum(dim=0)
    fn = ((1.0 - probs_14) * y_14ch).sum(dim=0)
    soft_f1 = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
    f1_loss = ((1.0 - soft_f1) * ch_weights).sum() / ch_weights.sum()

    return corn_loss + 0.35 * emd_loss + 0.25 * f1_loss
