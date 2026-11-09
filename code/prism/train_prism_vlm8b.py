#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/prism/train_prism_vlm8b.py
===============================
PRISM Tower 2: `Prism-VLM8B` Multi-Crop (Global + 1:1 Peak Edge + 1:1 Subject Interior)
Qwen3-VL-8B Conditional Hazard Trainer.

Why Previous VLM (`train_vlm_pure_real_corn.py`) Missed Fine-Grained Severity Boundaries:
  1. It only fed a single downscaled image (`max_pixels=896*896`), which destroyed 1-2px oversharpening
     halos (`过锐`), micro-texture smoothing (`纹理损失`), and small hand/text glitches (`过度生成`).
  2. It used `soft_sigma=0.32` Gaussian label smearing (`labels_to_corn_targets`), blurring `P1` and `P2`.

How `Prism-VLM8B` Upgrades the 8B Multimodal Tower:
  1. Multi-Image Interleaved Input (`[Global View, 1:1 Peak Edge Patch (448x448), 1:1 Subject Mid Patch (448x448)]`):
     Provides `Qwen3-VL-8B` with both holistic scene semantics and zero-resize 1:1 native pixel crops!
  2. Hard Conditional Hazard Supervision (`soft_sigma = 0.0`):
     Trains `P(Y >= 1)`, `P(Y >= 2 | Y >= 1)`, and `P(Y == 3 | Y >= 2)` with zero label smearing,
     plus Direct Exact-Grade Soft-F1 (`P2, P1, P0`) and Q-Align Discrete Grade Token Priors.
"""
import os
import sys
import csv
import json
import math
import time
import random
import datetime
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

REPO_ROOT = '/wanqing-develop/luowenjing/18' if os.path.exists('/wanqing-develop/luowenjing/18') else os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
os.chdir(REPO_ROOT)
sys.path.extend([REPO_ROOT, os.path.join(REPO_ROOT, 'code'), os.path.join(REPO_ROOT, 'code/finetune'), os.path.join(REPO_ROOT, 'code/decision_eval')])

from utils import DIMS, FULL_DIMS, P0_ONLY_DIMS, labels_to_14ch, ch14_to_p012, eval_14ch
from train_vlm_pure_real_corn import (
    MODEL_PATH,
    find_real_img,
    resize_cap,
    labels_to_corn_targets,
    corn_logits_to_unconditional_14ch,
    Qwen3VL_ABMIL_CORN,
    to_dev,
    predict_corn,
)

PRISM_MULTI_CROP_PROMPT = (
    "请作为专业画质与AIGC缺陷检测专家，结合【图1：全局场景】、【图2：1:1高频边缘显微镜切片】与【图3：1:1主体内部纹理切片】，"
    "精确区分轻度(P2)、中度(P1)与重度(P0)画质缺陷："
    "1.过度生成与AI结构崩坏（手指肢体畸变、文字乱码、塑料假纹理）；"
    "2.虚化突兀（抠图边缘断层）；"
    "3.模糊、纹理涂抹损失、边缘过锐白边(Halo)或花屏："
)


def extract_1to1_peak_and_mid_patches(im: Image.Image, crop_size: int = 384):
    """Extracts 1:1 native-pixel Peak Edge Patch and Median Subject Texture Patch."""
    w, h = im.size
    S = crop_size
    if w < S or h < S:
        im = im.resize((max(w, S), max(h, S)), Image.BICUBIC)
        w, h = im.size

    xs = [0, max(0, (w - S) // 2), max(0, w - S)] if w > S else [0]
    ys = [0, max(0, (h - S) // 2), max(0, h - S)] if h > S else [0]
    cands, energies = [], []
    for y0 in ys:
        for x0 in xs:
            patch = im.crop((x0, y0, x0 + S, y0 + S))
            arr = np.asarray(patch, dtype=np.uint8)[::4, ::4, 1].astype(np.float32)
            gx = np.abs(arr[:, 1:] - arr[:, :-1]).mean()
            gy = np.abs(arr[1:, :] - arr[:-1, :]).mean()
            cands.append(patch)
            energies.append(float(gx + gy))
    order = np.argsort(energies)
    return cands[int(order[-1])], cands[int(order[len(order) // 2])]


class PrismMultiCropVLMDataset(Dataset):
    def __init__(self, rows, max_pixels: int = 768 * 768, crop_size: int = 384, train: bool = True, flip: bool = False):
        self.rows = rows
        self.mp = max_pixels
        self.crop_size = crop_size
        self.train = train
        self.flip = flip

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(find_real_img(r["name"])).convert("RGB")
        if self.flip or (self.train and random.random() < 0.5):
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        p_peak, p_mid = extract_1to1_peak_and_mid_patches(im, crop_size=self.crop_size)
        im_glob = resize_cap(im, self.mp)
        g = np.array([int(r[c]) for c in DIMS], dtype=np.int64)
        return [im_glob, p_peak, p_mid], g, r["name"]


class PrismMultiCropCollate:
    def __init__(self, processor):
        self.p = processor
        msg = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": PRISM_MULTI_CROP_PROMPT},
            ],
        }]
        self.text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def __call__(self, batch):
        nested_ims = [b[0] for b in batch]
        enc = self.p(text=[self.text] * len(nested_ims), images=nested_ims, padding=True, return_tensors="pt")
        g_np = np.stack([b[1] for b in batch], axis=0)
        # Hard Conditional Hazard targets (soft_sigma = 0.0 -> ZERO P1/P2 Gaussian smearing!)
        t_corn, m_corn = labels_to_corn_targets(g_np, soft_sigma=0.0)
        y_14 = labels_to_14ch(g_np)
        enc["grades"] = torch.from_numpy(g_np)
        enc["t_corn"] = torch.from_numpy(t_corn)
        enc["m_corn"] = torch.from_numpy(m_corn)
        enc["y_14ch"] = torch.from_numpy(y_14)
        enc["names"] = [b[2] for b in batch]
        return enc


def compute_prism_vlm_exact_hazard_loss(logits: torch.Tensor, t_corn: torch.Tensor, m_corn: torch.Tensor, y_14ch: torch.Tensor, grades: torch.Tensor):
    """
    PRISM Conditional Hazard + Exact-Grade F1 Loss for Qwen3-VL-8B:
      1. Hard Conditional Hazard BCE on valid subsets (zero P1/P2 label smearing).
      2. Direct Exact-Grade Soft-F1 on P2, P1, and P0 channels.
      3. In-batch Margin Separation on conditional hazard logits (`P1 vs P2` on `q_ge2`, `P0 vs P1` on `q_ge3`).
    """
    is_clean = (grades == 0).all(dim=1)
    w_corn = m_corn.clone()
    if is_clean.any():
        w_corn[is_clean, :6] = 4.0

    # Boost P1 conditional hazard channel (6..9) by 1.45x
    w_corn[:, 6:10] = w_corn[:, 6:10] * 1.45
    bce_raw = F.binary_cross_entropy_with_logits(logits, t_corn, reduction="none")
    corn_loss = (bce_raw * w_corn).sum() / w_corn.sum().clamp(min=1.0)

    probs_14 = corn_logits_to_unconditional_14ch(logits)
    emd_loss = F.mse_loss(probs_14, y_14ch)

    # Exact-grade probabilities for P2, P1, P0
    q = torch.sigmoid(logits)
    exact_f1_loss = logits.new_tensor(0.0)
    margin_loss = logits.new_tensor(0.0)
    for j, d in enumerate(FULL_DIMS):
        yc = grades[:, d]
        h_det = q[:, 10 + j]
        h_s1  = q[:, 6 + j]
        h_s0  = q[:, d]

        p_p2 = h_det * (1.0 - h_s1)
        p_p1 = h_det * h_s1 * (1.0 - h_s0)
        p_p0 = h_det * h_s1 * h_s0

        for g_val, p_g, w_g in [(1, p_p2, 1.3), (2, p_p1, 1.8), (3, p_p0, 1.6)]:
            t_g = (yc == g_val).float()
            tp = (p_g * t_g).sum()
            fp = (p_g * (1.0 - t_g)).sum()
            fn = ((1.0 - p_g) * t_g).sum()
            f1_g = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
            exact_f1_loss = exact_f1_loss + w_g * (1.0 - f1_g)

        # Pairwise separation margin on P1 vs P2 (`logits[:, 6+j]`) and P0 vs P1 (`logits[:, d]`)
        pos_p1, neg_p2 = (yc == 2), (yc == 1)
        if pos_p1.any() and neg_p2.any():
            diff_12 = logits[pos_p1, 6 + j].unsqueeze(1) - logits[neg_p2, 6 + j].unsqueeze(0)
            margin_loss = margin_loss + (F.relu(0.65 - diff_12) ** 2).mean()
        pos_p0, neg_p1 = (yc == 3), (yc == 2)
        if pos_p0.any() and neg_p1.any():
            diff_01 = logits[pos_p0, d].unsqueeze(1) - logits[neg_p1, d].unsqueeze(0)
            margin_loss = margin_loss + (F.relu(0.65 - diff_01) ** 2).mean()

    return corn_loss + 0.35 * emd_loss + 0.30 * (exact_f1_loss / 12.0) + 0.25 * (margin_loss / 4.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/resplits/split_s16.csv")
    ap.add_argument("--manifest", default="data/manifest_combined.csv")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-pixels", type=int, default=768 * 768)
    ap.add_argument("--crop-size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1.8e-4)
    ap.add_argument("--lr-head", type=float, default=9.0e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=90))
    rank = dist.get_rank()
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr_)
    dev = f"cuda:{lr_}"
    random.seed(29 + rank)
    np.random.seed(29 + rank)
    torch.manual_seed(29)
    is0 = (rank == 0)
    log = (lambda *s: print(*s, flush=True)) if is0 else (lambda *s: None)

    man = {r["name"].strip(): r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))}
    rows = []
    for r in csv.DictReader(open(a.split, encoding="utf-8")):
        n = r["name"].strip()
        if n in man and not n.startswith(("v9_", "v8_", "v7_", "v6_")) and ("_d" not in n):
            rows.append({"name": n, "split": r["split"].strip(), **{c: int(man[n][c]) for c in DIMS}})

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"] or va

    va_names_set = set(r["name"] for r in va)
    tr = [r for r in tr if r["name"] not in va_names_set]
    rare_p1_p0 = [r for r in tr if r["过度生成"] >= 2 or r["过锐"] >= 2 or r["虚化突兀"] >= 2]
    clean_tr = [r for r in tr if all(r[c] == 0 for c in DIMS)]
    tr = tr + rare_p1_p0 + clean_tr

    log(f"=== [PRISM Tower 2: Prism-VLM8B Multi-Crop] train={len(tr)} val={len(va)} test={len(te)} ===")

    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "right"
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    qalign_words = ["无", "轻", "中", "重", "正常", "轻微", "中度", "严重", "0", "1", "2", "3"]
    q_ids = [proc.tokenizer.encode(w, add_special_tokens=False)[0] for w in qalign_words]
    qalign_w = full.lm_head.weight.data[q_ids].detach().float().clone()

    base = full.model
    del full.lm_head
    img_tok_id = full.config.image_token_id
    hid = full.config.text_config.hidden_size

    target = r".*(language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))|visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2)))$"
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=target, bias="none")
    base = get_peft_model(base, cfg)
    for n, p in base.named_parameters():
        if "merger" in n:
            p.requires_grad = True
    if a.grad_ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()

    model = Qwen3VL_ABMIL_CORN(base, qalign_weight=qalign_w, image_token_id=img_tok_id, hidden_size=hid).to(dev)
    for m in [model.vis_saliency_proj, model.vis_anom_score, model.vis_attn_V, model.vis_attn_U, model.vis_attn_w, model.head, model.qalign_adapter]:
        m.float()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_], find_unused_parameters=False)
    collate_fn = PrismMultiCropCollate(proc)
    sampler = DistributedSampler(PrismMultiCropVLMDataset(tr, a.max_pixels, a.crop_size, True), shuffle=True, seed=29)
    ltr = DataLoader(
        PrismMultiCropVLMDataset(tr, a.max_pixels, a.crop_size, True),
        batch_size=a.batch, sampler=sampler, num_workers=a.workers,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=(a.workers > 0), drop_last=True,
    )
    mk_eval = lambda rs, flip=False: DataLoader(
        PrismMultiCropVLMDataset(rs, a.max_pixels, a.crop_size, False, flip=flip),
        batch_size=a.batch * 2, shuffle=False, num_workers=a.workers, collate_fn=collate_fn, pin_memory=True,
    )
    lva = mk_eval(va, False)
    lte = mk_eval(te, False)

    head_p = [p for n, p in model.module.named_parameters() if p.requires_grad and not n.startswith("base.")]
    lora_p = [p for n, p in model.module.named_parameters() if p.requires_grad and n.startswith("base.")]
    opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr}, {"params": head_p, "lr": a.lr_head}], weight_decay=0.01)
    total_steps = max(1, len(ltr) * a.epochs)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)

    best_score = -1.0
    ckpt_path = a.out.replace(".json", "_best.pt")

    for ep in range(a.epochs):
        sampler.set_epoch(ep)
        model.train()
        t0 = time.time()
        tl, nb = 0.0, 0
        for b in ltr:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**to_dev(b, dev))
            loss = compute_prism_vlm_exact_hazard_loss(
                lg.float(),
                b["t_corn"].to(dev, non_blocking=True),
                b["m_corn"].to(dev, non_blocking=True),
                b["y_14ch"].to(dev, non_blocking=True),
                b["grades"].to(dev, non_blocking=True),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            tl += loss.item()
            nb += 1

        dist.barrier()
        if is0:
            vp0, vp1, vp2, vp14, gv, _ = predict_corn(model.module, lva, dev)
            sc = eval_14ch(vp14, gv)
            log(f"== ep{ep+1}/{a.epochs} loss={tl/max(1,nb):.4f} val total={sc['total']:.2f} S_pos={sc['S_pos']:.3f} S_p0={sc['S_p0']:.3f} S_clean={sc['S_clean']:.3f} ({time.time()-t0:.0f}s)")
            if sc["total"] > best_score:
                best_score = sc["total"]
                torch.save(
                    {k: v.cpu() for k, v in model.module.state_dict().items() if "lora" in k or not k.startswith("base.") or "merger" in k},
                    ckpt_path,
                )
        dist.barrier()

    if is0:
        if os.path.exists(ckpt_path):
            model.module.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        vp0, vp1, vp2, vp14, gv, vn = predict_corn(model.module, lva, dev)
        tp0, tp1, tp2, tp14, gt, tn = predict_corn(model.module, lte, dev)

        tpl = json.load(open("results/v2_full/template_150.json"))
        ordered_150 = [e["image"] for e in tpl]
        rows_150 = [{"name": nm, **{c: 0 for c in DIMS}} for nm in ordered_150]
        p0_1, p1_1, p2_1, _, _, _ = predict_corn(model.module, mk_eval(rows_150, False), dev)
        p0_2, p1_2, p2_2, _, _, _ = predict_corn(model.module, mk_eval(rows_150, True), dev)

        st = eval_14ch(tp14, gt)
        json.dump({"test": st, "args": vars(a)}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        np.savez_compressed(
            a.out.replace(".json", "_probs.npz"),
            va_p0=vp0, va_p1=vp1, va_p2=vp2, gva=gv, va_names=np.array(vn),
            te_p0=tp0, te_p1=tp1, te_p2=tp2, gte=gt, te_names=np.array(tn),
            t150_p0=0.5 * (p0_1 + p0_2), t150_p1=0.5 * (p1_1 + p1_2), t150_p2=0.5 * (p2_1 + p2_2), names_150=np.array(ordered_150),
        )
        log("Saved PRISM-VLM8B predictions to:", a.out.replace(".json", "_probs.npz"))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
