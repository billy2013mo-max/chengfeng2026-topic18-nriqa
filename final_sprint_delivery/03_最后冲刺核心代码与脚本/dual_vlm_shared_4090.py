#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/dual_vlm_shared_4090.py
================================
Single-RTX-4090 Shared-Base Dual-Adapter Qwen3-VL-8B (`p1_qy` DoRA r=24 + `qwen3vl_hires` LoRA r=64).

Loads ONE 4-bit NF4 `Qwen3-VL-8B` LLM + `bf16` ViT base (`~4.44 GB` with `--embed-on-host`) and attaches
both trained adapters (`p1_qy` DoRA r=24 and `qwen_hr` LoRA r=64) plus their exact 24 non-LoRA merger
tensors (~20 MB) and ordinal heads (~4 MB), keeping BOTH 8B VLMs simultaneously resident in `~5.6 GB`
total (~3.4 GB VRAM saved over two standalone 4-bit VLMs!) with exact hypercolumn + per-ROI pooling parity.
"""
from __future__ import annotations

import gc
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("code", "code/next55", "code/r55", "code/finetune", "third_party/pydeps"):
    _ap = os.path.join(REPO, _p)
    if os.path.isdir(_ap) and _ap not in sys.path:
        sys.path.insert(0, _ap)


class DualAdapterQwen3VL4090(nn.Module):
    def __init__(
        self,
        p1_ckpt: str,
        qh_ckpt: str,
        device: str = "cuda",
        embed_on_host: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from phoenix.config import resolve_local_or_hf_model
        from phoenix_pillar_4090 import quantize_llm_inplace
        from vlm_c233397_qy8 import (
            CONCISE_INSPECTION_PROMPT_C233397,
            PhoenixUnimodalDeQAHeadC233397,
            format_3image_chat_prompt_c233397,
        )
        from train_qwen3vl_hires_roi import (
            HIRES_INSPECTION_PROMPT,
            TagFreeUnimodalDeQAHead,
            format_4image_chat_prompt,
        )

        self.dev = torch.device(device)
        model_dir = resolve_local_or_hf_model("pillar1_qwen3vl_8b")
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        if hasattr(tok, "padding_side"):
            tok.padding_side = "right"

        full = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
        )
        raw = full.model.to(self.dev)
        self.image_token_id = int(getattr(full.config, "image_token_id", 151655))
        del full
        gc.collect()

        nq = quantize_llm_inplace(raw, self.dev)
        self.embed_host_bytes = 0
        if embed_on_host:
            from final_engine import _move_token_embedding_to_host

            class _Tmp:
                def __init__(self, b):
                    self.backbone = b

            self.embed_host_bytes = _move_token_embedding_to_host(_Tmp(raw))
        gc.collect()
        torch.cuda.empty_cache()

        ck_p1 = torch.load(p1_ckpt, map_location="cpu", weights_only=False)
        sd_p1 = ck_p1.get("state_dict", ck_p1) if isinstance(ck_p1, dict) else ck_p1
        ck_qh = torch.load(qh_ckpt, map_location="cpu", weights_only=False)
        sd_qh = ck_qh.get("state_dict", ck_qh) if isinstance(ck_qh, dict) else ck_qh

        targets = [
            "down_proj", "gate_proj", "k_proj", "linear_fc1", "linear_fc2",
            "o_proj", "proj", "q_proj", "qkv", "up_proj", "v_proj",
        ]
        cfg_p1 = LoraConfig(r=24, lora_alpha=48, lora_dropout=0.0, target_modules=targets, use_dora=True, bias="none")
        peft_m = get_peft_model(raw, cfg_p1, adapter_name="p1_qy")
        qh_r = int(ck_qh.get("lora_rank", 64)) if isinstance(ck_qh, dict) else 64
        qh_dora = bool(ck_qh.get("use_dora", False)) if isinstance(ck_qh, dict) else False
        cfg_qh = LoraConfig(
            r=qh_r, lora_alpha=qh_r * 2, lora_dropout=0.0, target_modules=targets, use_dora=qh_dora, bias="none"
        )
        peft_m.add_adapter("qwen_hr", cfg_qh)

        def _split_sd(sd: Dict[str, torch.Tensor], adp: str):
            lora_d, nonlora_d, head_d = {}, {}, {}
            for k, v in sd.items():
                if k.startswith("head."):
                    head_d[k[len("head.") :]] = v
                elif k.startswith("backbone."):
                    nk = k[len("backbone.") :].replace(".default.", f".{adp}.")
                    if "lora" in nk.lower():
                        lora_d[nk] = v
                    else:
                        nonlora_d[nk] = v.to(self.dev, dtype=torch.bfloat16)
            return lora_d, nonlora_d, head_d

        lora_p1, self._nonlora_p1, hd_p1 = _split_sd(sd_p1, "p1_qy")
        lora_qh, self._nonlora_qh, hd_qh = _split_sd(sd_qh, "qwen_hr")

        r1 = peft_m.load_state_dict(lora_p1, strict=False)
        r2 = peft_m.load_state_dict(lora_qh, strict=False)
        assert not r1.unexpected_keys, f"p1_qy unexpected LoRA keys: {r1.unexpected_keys[:5]}"
        assert not r2.unexpected_keys, f"qwen_hr unexpected LoRA keys: {r2.unexpected_keys[:5]}"

        for n, p in peft_m.named_parameters():
            p.requires_grad = False
            if "lora" in n.lower() and p.is_floating_point():
                p.data = p.data.to(torch.bfloat16)
        self.backbone = peft_m.to(self.dev).eval()

        hid = int(getattr(getattr(raw.config, "text_config", raw.config), "hidden_size", 4096))
        self.head_p1 = PhoenixUnimodalDeQAHeadC233397(hid).to(self.dev).eval()
        self.head_p1.load_state_dict(hd_p1, strict=True)
        self.head_qh = TagFreeUnimodalDeQAHead(hid).to(self.dev).eval()
        self.head_qh.load_state_dict(hd_qh, strict=True)

        # Exact hypercolumn hooks matching BOTH `vlm_c233397_qy8` and `train_qwen3vl_hires_roi`:
        # layers[17] (hidden_states[18]), layers[26] (hidden_states[27]), and outputs.last_hidden_state (after RMSNorm!)
        layers = self.backbone.base_model.model.language_model.layers
        n_l = len(layers)
        idx_mid = n_l // 2 - 1
        idx_3q = (3 * n_l) // 4 - 1
        self._tapped: Dict[str, torch.Tensor] = {}
        layers[idx_mid].register_forward_hook(self._make_hook("mid"))
        layers[idx_3q].register_forward_hook(self._make_hook("3q"))

        self.p1_prompt = format_3image_chat_prompt_c233397(self.processor, CONCISE_INSPECTION_PROMPT_C233397)
        self.qh_prompt = format_4image_chat_prompt(self.processor, HIRES_INSPECTION_PROMPT)
        self._active_nonlora: Optional[str] = None
        del ck_p1, sd_p1, ck_qh, sd_qh, lora_p1, lora_qh
        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print(
                f"[DualAdapterQwen3VL4090] Shared 4-bit LLM ({nq} linears, emb_host={self.embed_host_bytes/2**30:.2f} GB) "
                f"+ p1_qy(DoRA r=24) + qwen_hr(LoRA r={qh_r}) | resident={torch.cuda.memory_allocated(self.dev)/2**30:.2f} GB",
                flush=True,
            )

    def _make_hook(self, key: str):
        def _hk(_mod, _inp, out):
            self._tapped[key] = out[0] if isinstance(out, tuple) else out
        return _hk

    def _activate(self, adapter: str) -> None:
        self.backbone.set_adapter(adapter)
        if self._active_nonlora != adapter:
            src = self._nonlora_p1 if adapter == "p1_qy" else self._nonlora_qh
            sd_cur = self.backbone.state_dict()
            for k, v in src.items():
                if k in sd_cur:
                    sd_cur[k].copy_(v)
            self._active_nonlora = adapter

    @staticmethod
    def _pool_1d_tokens(seq: torch.Tensor, target_len: int) -> torch.Tensor:
        L = int(seq.shape[0])
        if L == target_len:
            return seq
        if L < target_len:
            pad = seq[-1:].expand(target_len - L, -1)
            return torch.cat([seq, pad], dim=0)
        return F.adaptive_avg_pool1d(seq.unsqueeze(0).transpose(1, 2), output_size=target_len).transpose(1, 2).squeeze(0)

    @torch.no_grad()
    def _forward_p1_one(self, views: List[Image.Image]) -> np.ndarray:
        self._activate("p1_qy")
        inputs = self.processor(text=[self.p1_prompt], images=views, padding=True, return_tensors="pt")
        moved = {k: (v.to(self.dev) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        self._tapped.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.backbone(**moved, output_hidden_states=False, return_dict=True, use_cache=False)
        fused = (self._tapped["mid"].float() + self._tapped["3q"].float() + outputs.last_hidden_state.float()) / 3.0
        self._tapped.clear()

        mask = moved["attention_mask"][0].bool()
        valid = fused[0, mask]
        summary = valid[-1:, :]
        body = valid[:-1, :] if valid.shape[0] > 1 else valid
        b63 = self._pool_1d_tokens(body, 63)
        tok = torch.cat([b63, summary], dim=0).unsqueeze(0)
        out = self.head_p1(tok, batch_ids=None, apply_batch_bias=False)
        return out["probs_6x4"].detach().float().cpu().numpy()[0]

    @torch.no_grad()
    def _forward_qh_one(self, views: List[Image.Image]) -> np.ndarray:
        self._activate("qwen_hr")
        inputs = self.processor(text=[self.qh_prompt], images=views, padding=True, return_tensors="pt")
        moved = {k: (v.to(self.dev) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        self._tapped.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.backbone(**moved, output_hidden_states=False, return_dict=True, use_cache=False)
        fused = (self._tapped["mid"].float() + self._tapped["3q"].float() + outputs.last_hidden_state.float()) / 3.0
        self._tapped.clear()

        mask = moved["attention_mask"][0].bool()
        valid_toks = fused[0, mask]
        valid_ids = moved["input_ids"][0, mask]
        summary_tok = valid_toks[-1:, :]
        img_toks = valid_toks[valid_ids == self.image_token_id]
        grid_thw = moved.get("image_grid_thw", None)
        if grid_thw is not None and grid_thw.shape[0] == 4:
            c4 = [int((grid_thw[i, 0] * grid_thw[i, 1] * grid_thw[i, 2]).item()) // 4 for i in range(4)]
            if sum(c4) == int(img_toks.shape[0]):
                v0, v1, v2, v3 = torch.split(img_toks, c4, dim=0)
                g_avg = self._pool_1d_tokens(v0, 48)
                g_max = (
                    F.adaptive_max_pool1d(v0.unsqueeze(0).transpose(1, 2), output_size=16).transpose(1, 2).squeeze(0)
                    if v0.shape[0] >= 16
                    else self._pool_1d_tokens(v0, 16)
                )
                seq = torch.cat(
                    [g_avg, g_max, self._pool_1d_tokens(v1, 16), self._pool_1d_tokens(v2, 16), self._pool_1d_tokens(v3, 16), summary_tok],
                    dim=0,
                )
                out = self.head_qh(seq.unsqueeze(0))
                return out["probs_6x4"].detach().float().cpu().numpy()[0]

        seq = torch.cat([self._pool_1d_tokens(valid_toks, 112), summary_tok], dim=0)
        out = self.head_qh(seq.unsqueeze(0))
        return out["probs_6x4"].detach().float().cpu().numpy()[0]

    @torch.no_grad()
    def infer_p1_qy(self, views_orig: List[Image.Image]) -> np.ndarray:
        views_flip = [v.transpose(Image.FLIP_LEFT_RIGHT) for v in views_orig]
        return 0.5 * (self._forward_p1_one(views_orig) + self._forward_p1_one(views_flip))

    @torch.no_grad()
    def infer_qwen_hr(self, views_orig: List[Image.Image]) -> np.ndarray:
        views_flip = [v.transpose(Image.FLIP_LEFT_RIGHT) for v in views_orig]
        return 0.5 * (self._forward_qh_one(views_orig) + self._forward_qh_one(views_flip))
