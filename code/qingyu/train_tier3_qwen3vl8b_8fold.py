#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/train_tier3_qwen3vl8b_8fold.py
==========================================
晴羽观感体系 — 支柱三 (4.3 & 4.4 节):
Qwen3-VL-8B-Instruct 三图独立 Interleaved-MRoPE 原生无接缝输入 + 3 层视觉-语言金字塔读出
+ Q-Align 离散锚点残差先验 + 单峰二次残差序数头 (`UnimodalQuadraticResidualHead`) 8 折微调器。

核心设计:
1. 彻底废除 `960x640` 合成硬拼贴图:
   将 `[img_global_native_768, roi_anatomy_384, roi_text_tex_384]` 作为 3 个独立 `{"type": "image"}` 对象输入
   Qwen3-VL 原生 Interleaved-MRoPE，零跨图接缝伪影、完整保留解剖局部与文字纹理 1:1 像素频率。
2. 3 层视觉-语言金字塔读出 (50% / 75% / 100% 深度) + 视图分离门控 ABMIL 与 Top-5% 局部异常池化:
   不仅汇聚全图视觉 Token，而且单独对 `roi_anatomy` 与 `roi_text_tex` 提取高敏局部特征，防止微瑕被背景平均稀释。
3. 接入 `UnimodalQuadraticResidualHead` + 36D 单图物理特征 + 128 纯真实视觉原型库 `TipAdapterPrototypeBank`
   + `QingyuUnimodalCompositeLoss` (含 XBM-512 跨批排序、RPS、Brier 校准与 25.28B 池内跨模态 OOF 互蒸馏)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.config_and_paths import (
    DIMS,
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
)
from qingyu.unimodal_xbm_losses import (
    QingyuUnimodalCompositeLoss,
    UnimodalQuadraticResidualHead,
)
from qingyu.certainty_gated_stacker import compute_ece_6x4, simplex_6x4_to_p012
from qingyu.dfw_bayesian_decoder import DampedFrankWolfeBayesianDecoder
from qingyu.train_tier3_clef_flash9b_8fold import (
    ClefFlash9BSystem1Tower,
    Tier3MultiImageMRoPEDataset,
    _move_inputs_to_device,
    compute_brier_calibration_loss,
    load_fold_rows,
    make_clef_3image_collate,
)


class Qwen3VL3ImageInterleavedTower(nn.Module):
    """
    Qwen3-VL-8B 三图独立 Interleaved-MRoPE 质量决策塔:
      - 提取 LLM 50% / 75% / 100% 三层视觉隐状态金字塔；
      - 分别对 Global 图与两个 1:1 语义 ROI (`roi_anatomy`, `roi_text_tex`) 执行门控 ABMIL + Top-5% 异常池化；
      - 叠加 Q-Align 预训练程度词嵌入残差先验；
      - 送入 FP32 `UnimodalQuadraticResidualHead` + 128 纯真实原型库 `TipAdapterPrototypeBank`。
    """

    def __init__(
        self,
        peft_base: nn.Module,
        qalign_weight: Optional[torch.Tensor],
        image_token_id: int = 151655,
        hidden_size: int = 4096,
        phys_dim: int = PHYS_FEAT_DIM,
    ):
        super().__init__()
        self.base = peft_base
        self.image_token_id = int(image_token_id)
        self.hidden_size = int(hidden_size)

        self.pyr_norm_50 = nn.LayerNorm(hidden_size)
        self.pyr_norm_75 = nn.LayerNorm(hidden_size)
        self.pyr_adapter = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        nn.init.zeros_(self.pyr_adapter[-1].weight)
        nn.init.zeros_(self.pyr_adapter[-1].bias)

        self.vis_attn_V = nn.Linear(hidden_size, 256)
        self.vis_attn_U = nn.Linear(hidden_size, 256)
        self.vis_attn_w = nn.Linear(256, 1)
        self.vis_anom_score = nn.Linear(hidden_size, 1)

        # 5 组池化特征: [global_abmil, global_top5, roi_anat_abmil, roi_text_abmil, last_summary_vec]
        in_dim = hidden_size * 5
        self.unimodal_head = UnimodalQuadraticResidualHead(
            in_dim=in_dim,
            phys_dim=phys_dim,
            hidden_dim=768,
            proj_dim=512,
            dropout=0.20,
            max_prototypes=128,
        )

        if qalign_weight is not None:
            self.register_buffer("qalign_w", qalign_weight.float())
            self.qalign_proj = nn.Sequential(
                nn.LayerNorm(qalign_weight.shape[0]),
                nn.Linear(qalign_weight.shape[0], 6 * 4),
            )
            nn.init.zeros_(self.qalign_proj[-1].weight)
            nn.init.zeros_(self.qalign_proj[-1].bias)
        else:
            self.qalign_w = None
            self.qalign_proj = None

    @property
    def head(self) -> UnimodalQuadraticResidualHead:
        return self.unimodal_head

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in self.state_dict().items():
            kl = k.lower()
            if (not k.startswith("base.")) or ("lora_" in kl) or ("merger" in kl):
                out[k] = v.detach().cpu().clone()
        return out

    def _abmil_and_top5(self, toks: torch.Tensor, fallback: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if toks.shape[0] == 0:
            return fallback, fallback
        tf = toks.float()
        M, D = tf.shape
        mil_a = self.vis_attn_w(torch.tanh(self.vis_attn_V(tf)) * torch.sigmoid(self.vis_attn_U(tf))).squeeze(-1)
        mil_w = F.softmax(mil_a, dim=0).unsqueeze(-1)
        v_abmil = (mil_w * tf).sum(dim=0)

        k = max(1, int(math.ceil(M * 0.05)))
        sc = self.vis_anom_score(tf).squeeze(-1) / math.sqrt(float(D))
        top_v, top_i = torch.topk(sc, k=k, dim=0)
        top_w = F.softmax(top_v, dim=0).unsqueeze(-1)
        v_top5 = (top_w * tf[top_i]).sum(dim=0)
        return v_abmil, v_top5

    def forward(
        self,
        model_inputs: Dict[str, torch.Tensor],
        phys_feat: torch.Tensor,
        return_kv_cache: bool = False,
    ) -> Dict[str, Any]:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        image_grid_thw = model_inputs.get("image_grid_thw", None)

        fwd_kw = dict(model_inputs)
        fwd_kw["output_hidden_states"] = True
        fwd_kw["return_dict"] = True
        fwd_kw["use_cache"] = bool(return_kv_cache)

        out = self.base(**fwd_kw)
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
        last_vecs = h_last[torch.arange(B, device=dev), seq_lens].float()
        img_mask = input_ids == self.image_token_id

        g_abmils, g_top5s, r1_abmils, r2_abmils = [], [], [], []
        for b in range(B):
            m = img_mask[b]
            if m.any():
                vt_100 = h_last[b, m].float()
                vt_50 = self.pyr_norm_50(h_50[b, m].float())
                vt_75 = self.pyr_norm_75(h_75[b, m].float())
                vt_fused = vt_100 + 0.35 * self.pyr_adapter(torch.cat([vt_50, vt_75], dim=-1))
                vids = ClefFlash9BSystem1Tower._assign_3view_ids(
                    int(vt_fused.shape[0]), image_grid_thw, b * 3
                ).to(dev)
                ga, gt = self._abmil_and_top5(vt_fused[vids == 0], last_vecs[b])
                r1a, _ = self._abmil_and_top5(vt_fused[vids == 1], ga)
                r2a, _ = self._abmil_and_top5(vt_fused[vids == 2], ga)
            else:
                ga = gt = r1a = r2a = last_vecs[b]
            g_abmils.append(ga)
            g_top5s.append(gt)
            r1_abmils.append(r1a)
            r2_abmils.append(r2a)

        feat = torch.cat(
            [
                torch.stack(g_abmils, dim=0),
                torch.stack(g_top5s, dim=0),
                torch.stack(r1_abmils, dim=0),
                torch.stack(r2_abmils, dim=0),
                last_vecs,
            ],
            dim=-1,
        )
        head_out = self.unimodal_head(feat.float(), phys_feat=phys_feat.float())

        if self.qalign_w is not None and self.qalign_proj is not None:
            q_raw = F.linear(last_vecs, self.qalign_w)
            q_res = self.qalign_proj(q_raw).view(B, 6, 4)
            logits_6x4 = head_out["logits_6x4"]
            from qingyu.config_and_paths import FULL_DIMS, P0_ONLY_DIMS
            new_logits = logits_6x4.clone()
            for d in FULL_DIMS:
                new_logits[:, d, :] = logits_6x4[:, d, :] + (q_res[:, d, :] - q_res[:, d, :].mean(dim=-1, keepdim=True))
            for d in P0_ONLY_DIMS:
                q_b = q_res[:, d, [0, 3]] - q_res[:, d, [0, 3]].mean(dim=-1, keepdim=True)
                new_logits[:, d, 0] = logits_6x4[:, d, 0] + q_b[:, 0]
                new_logits[:, d, 3] = logits_6x4[:, d, 3] + q_b[:, 1]
            probs_6x4 = torch.zeros_like(new_logits)
            for d in FULL_DIMS:
                probs_6x4[:, d, :] = F.softmax(new_logits[:, d, :], dim=-1)
            for d in P0_ONLY_DIMS:
                p2 = F.softmax(new_logits[:, d, [0, 3]], dim=-1)
                probs_6x4[:, d, 0] = p2[:, 0]
                probs_6x4[:, d, 3] = p2[:, 1]
            head_out["logits_6x4"] = new_logits
            head_out["probs_6x4"] = probs_6x4

        if return_kv_cache and getattr(out, "past_key_values", None) is not None:
            head_out["past_key_values"] = out.past_key_values
            head_out["prefill_seq_len"] = int(input_ids.shape[1])
        return head_out


def load_qwen3vl8b_backbone_and_lora(
    model_path: str,
    lora_rank: int = 32,
    grad_checkpointing: bool = True,
    keep_lm_head: bool = False,
) -> Tuple[nn.Module, Any, Optional[torch.Tensor], int, int]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(proc, "tokenizer"):
        proc.tokenizer.padding_side = "right"

    full = None
    for attn_impl in ("flash_attention_2", "sdpa"):
        try:
            full = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, dtype=torch.bfloat16, attn_implementation=attn_impl
            )
            break
        except Exception:
            continue
    if full is None:
        full = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    qalign_words = ["无", "轻", "中", "重", "正常", "轻微", "中度", "严重", "0", "1", "2", "3"]
    q_ids = [tok.encode(w, add_special_tokens=False)[0] for w in qalign_words]
    qalign_w = full.lm_head.weight.data[q_ids].detach().float().clone() if hasattr(full, "lm_head") else None

    saved_lm_head = None
    if keep_lm_head and hasattr(full, "lm_head"):
        saved_lm_head = full.lm_head.eval().requires_grad_(False)
    base = full.model
    if hasattr(full, "lm_head") and not keep_lm_head:
        del full.lm_head
    img_tok_id = int(full.config.image_token_id)
    hid = int(full.config.text_config.hidden_size)

    # 冻结底层 ViT 块，仅对 LLM 层及 ViT 顶层块与 Merger 微调，兼顾感知灵敏度与低方差泛化
    target = r".*(language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))|visual\.blocks\.(2[4-9]|3\d)\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2)))$"
    cfg = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05, target_modules=target, bias="none")
    peft_base = get_peft_model(base, cfg)
    if saved_lm_head is not None:
        peft_base._saved_lm_head = saved_lm_head
    for n, p in peft_base.named_parameters():
        if "merger" in n:
            p.requires_grad = True
    if grad_checkpointing:
        peft_base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        peft_base.enable_input_require_grads()

    return peft_base, proc, qalign_w, img_tok_id, hid


@torch.no_grad()
def populate_qwen_prototypes(
    model: Qwen3VL3ImageInterleavedTower,
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
def predict_qwen_2x_tta(
    model: Qwen3VL3ImageInterleavedTower,
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
    probs_passes, qclean_passes = [], []
    gold_out = None
    names_out: List[str] = []

    for flip_flag in (False, True):
        ds = Tier3MultiImageMRoPEDataset(
            rows, img_loader, meta_store, global_max_pixels=global_max_pixels, train=False, hflip_tta=flip_flag
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

    model_path = args.model_path or resolve_model_path("qwen3_vl_8b")
    peft_base, processor, qalign_w, img_tok_id, hid = load_qwen3vl8b_backbone_and_lora(
        model_path=model_path,
        lora_rank=args.lora_rank,
        grad_checkpointing=True,
    )
    model = Qwen3VL3ImageInterleavedTower(
        peft_base=peft_base,
        qalign_weight=qalign_w,
        image_token_id=img_tok_id,
        hidden_size=hid,
        phys_dim=PHYS_FEAT_DIM,
    ).to(device)
    for sub_m in (
        model.pyr_norm_50,
        model.pyr_norm_75,
        model.pyr_adapter,
        model.vis_attn_V,
        model.vis_attn_U,
        model.vis_attn_w,
        model.vis_anom_score,
        model.unimodal_head,
    ):
        sub_m.float()
    if model.qalign_proj is not None:
        model.qalign_proj.float()

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
    total_opt_steps = max(1, math.ceil(len(dl_tr) / grad_accum) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_lora, args.lr_head],
        total_steps=total_opt_steps,
        pct_start=0.08,
    )
    criterion = QingyuUnimodalCompositeLoss(
        soft_sigma=0.35, lambda_rps=0.45, lambda_rank=0.25, lambda_delta=1e-2, xbm_capacity=512
    ).to(device)

    res_dir = os.path.join(REPO_ROOT, args.out_dir, "qwen3_vl_8b")
    ckpt_dir = QINGYU_CKPT_DIR / "qwen3_vl_8b"
    os.makedirs(res_dir, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = os.path.join(res_dir, f"fold_{args.fold}_best.pt")

    print(
        f"=== [Qingyu Tier-3 Qwen3-VL-8B 3-Image MRoPE] Fold {args.fold}/8 | train={len(tr_rows)} val={len(va_rows)} | device={device} ===",
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
            populate_qwen_prototypes(
                model, tr_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size, args.workers
            )
            if ep >= max(0, args.epochs - 2)
            else 0
        )

        vp, vqc, gv, _ = predict_qwen_2x_tta(
            model, va_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size, args.workers
        )
        qdec = DampedFrankWolfeBayesianDecoder(dfw_steps=20, eta_doc1=0.50, max_clean_p0_fpr=0.008).fit(vp, vqc, gv)
        ev = evaluate_qingyu_dual_metrics(qdec.decode(vp, vqc), gv)
        ece = compute_ece_6x4(vp, gv)
        val_obj = 0.5 * ev["doc1_total"] + 0.5 * ev["doc2_strict_total"] - 2.0 * ece

        print(
            f"  [Qwen3-VL-8B F{args.fold}] Ep {ep+1}/{args.epochs} | loss={loss_acc/max(1,n_batches):.4f} | "
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

    if best_sd is not None and prev_sd is not None:
        swa_sd = {
            k: (0.5 * prev_sd[k].float() + 0.5 * best_sd[k].float()).to(best_sd[k].dtype)
            if best_sd[k].is_floating_point()
            else best_sd[k]
            for k in best_sd.keys()
        }
        model.load_state_dict(swa_sd, strict=False)
        populate_qwen_prototypes(
            model, tr_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size, args.workers
        )
        vp_s, vqc_s, gv_s, _ = predict_qwen_2x_tta(
            model, va_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size, args.workers
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
        populate_qwen_prototypes(
            model, tr_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size, args.workers
        )
        final_payload = {"state_dict": model.trainable_state_dict(), "best_val_score": float(best_obj)}
        torch.save(final_payload, ckpt_best)
        shutil.copy2(ckpt_best, str(ckpt_dir / f"fold_{args.fold}_best.pt"))

    va_probs, va_qclean, gva, va_names = predict_qwen_2x_tta(
        model, va_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size, args.workers
    )
    va_p0, va_p1, va_p2 = simplex_6x4_to_p012(va_probs)

    ordered_150 = load_test_template_names("results/v2_full/template_150.json")
    t150_rows = [{"name": nm, "group_id": nm, "grades": [0] * 6} for nm in ordered_150]
    t150_probs, t150_qclean, _, _ = predict_qwen_2x_tta(
        model, t150_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size, args.workers
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
        ext_probs, ext_qclean, _, _ = predict_qwen_2x_tta(
            model, ext_rows, img_loader, meta_store, collate_fn, device, args.global_max_pixels, args.batch_size, args.workers
        )
        save_dict["extra_probs_6x4"] = ext_probs
        save_dict["extra_qclean"] = ext_qclean
        save_dict["extra_names"] = np.array(ext_names)

    out_npz = os.path.join(res_dir, f"fold_{args.fold}_probs.npz")
    np.savez_compressed(out_npz, **save_dict)
    print(f"[Qingyu Tier-3 Qwen3-VL-8B] Saved Fold {args.fold} OOF & test -> {out_npz}", flush=True)
    return out_npz


def aggregate_8fold_oof(out_dir: str, n_folds: int = 8) -> str:
    setup_sys_path_and_cwd(chdir_repo=True)
    b_dir = os.path.join(REPO_ROOT, out_dir, "qwen3_vl_8b")
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
        f"=== [Qingyu Tier-3 8-Fold OOF Summary] qwen3_vl_8b (N={len(oof_gold)}) | "
        f"Doc1={ev['doc1_total']:.3f} | Doc2_Strict={ev['doc2_strict_total']:.3f} | ECE={ece:.4f} ===",
        flush=True,
    )

    agg_dict: Dict[str, Any] = dict(
        backbone="qwen3_vl_8b",
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

    agg_path = os.path.join(REPO_ROOT, out_dir, "tier3_qwen3_vl_8b_8fold_oof.npz")
    np.savez_compressed(agg_path, **agg_dict)
    with open(agg_path.replace(".npz", ".json"), "w", encoding="utf-8") as jf:
        json.dump({"backbone": "qwen3_vl_8b", "oof_metrics": ev, "ece": ece}, jf, ensure_ascii=False, indent=2)
    return agg_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Qingyu Tier-3 Qwen3-VL-8B 3-Image Interleaved-MRoPE 8-Fold Trainer")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--split-dir", default="data/qingyu_splits")
    ap.add_argument("--dual-engine-npz", default="results/qingyu/dual_engine_meta.npz")
    ap.add_argument("--out-dir", default="results/qingyu")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--global-max-pixels", type=int, default=768 * 768)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr-lora", type=float, default=1.8e-4)
    ap.add_argument("--lr-head", type=float, default=8.0e-4)
    ap.add_argument("--lambda-brier", type=float, default=0.15)
    ap.add_argument("--oof-teacher-npz", default="")
    ap.add_argument("--lambda-kd", type=float, default=0.25)
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
