#!/usr/bin/env python3
"""
code/finetune/train_vlm_pure_real_corn.py
Pure-Real Qwen3-VL-8B Specialist with Visual-Token Gated ABMIL + CORN + Ordinal EMD.
Directly targets Over-Generation (extra fingers, melted limbs, garbled text) and Abrupt Bokeh
at native dynamic resolution up to 1024x1024, with automatic 2x Flip TTA on test_150.
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

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "decision_eval")))

from utils import (
    DIMS, FULL_DIMS, P0_ONLY_DIMS,
    labels_to_14ch, ch14_to_p012, eval_14ch, competition_score,
)

MODEL_PATH = "/wanqing-develop/luowenjing/18/models/qwen3vl-8b"
IMG_DIRS = ["data/images", "data/images_supp", "data/test_150"]
DIAGNOSTIC_PROMPT = (
    "请作为专业画质与AIGC缺陷检测专家，仔细审视图像细节："
    "1.是否存在过度生成与AI结构崩坏（如多余或残缺手指、肢体融化粘连、五官扭曲、文字乱码）；"
    "2.是否存在背景虚化突兀（抠图边缘断层、景深过渡不自然）；"
    "3.是否存在模糊、纹理涂抹损失、边缘过锐伪影或花屏色块。"
    "请综合给出画质缺陷表征。"
)


def find_real_img(name: str) -> str:
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Real image not found: {name}")


def resize_cap(im: Image.Image, max_pixels: int, min_pixels: int = 256 * 256, mult: int = 32) -> Image.Image:
    w, h = im.size
    s = 1.0
    if w * h > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
    elif w * h < min_pixels:
        s = math.sqrt(min_pixels / (w * h))
    nw = max(mult, int(round(w * s / mult)) * mult)
    nh = max(mult, int(round(h * s / mult)) * mult)
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.BICUBIC)
    return im


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
    def __init__(self, base_model, image_token_id: int = 151655, hidden_size: int = 4096):
        super().__init__()
        self.base = base_model
        self.image_token_id = image_token_id

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
        v_toks_f = v_toks.float()
        M, D = v_toks_f.shape
        v_mean = v_toks_f.mean(dim=0)
        if M <= 4:
            return v_mean, v_mean, v_mean

        k = max(1, int(math.ceil(M * 0.05)))
        s = self.vis_anom_score(v_toks_f).squeeze(-1) / math.sqrt(D)
        top_vals, top_idx = torch.topk(s, k=k, dim=0)
        top_w = F.softmax(top_vals, dim=0).unsqueeze(-1)
        v_top5 = (top_w * v_toks_f[top_idx]).sum(dim=0)

        mil_a = self.vis_attn_w(
            torch.tanh(self.vis_attn_V(v_toks_f)) * torch.sigmoid(self.vis_attn_U(v_toks_f))
        ).squeeze(-1)
        mil_w = F.softmax(mil_a, dim=0).unsqueeze(-1)
        v_abmil = (mil_w * v_toks_f).sum(dim=0)
        return v_mean, v_abmil, v_top5

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids=None, **extra):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            return_dict=True,
        )
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
        B = h.shape[0]

        seq_lens = attention_mask.sum(dim=1) - 1
        last_vecs = h[torch.arange(B, device=h.device), seq_lens].float()

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
    # Penalize Clean false P0 activations strongly to protect S_clean = 20/20
    is_clean = (y_14ch[:, 10:].sum(dim=1) == 0) & (y_14ch[:, :6].sum(dim=1) == 0)
    w_corn = m_corn.clone()
    if is_clean.any():
        w_corn[is_clean, :6] = 3.5

    bce_raw = F.binary_cross_entropy_with_logits(logits, t_corn, reduction="none")
    corn_loss = (bce_raw * w_corn).sum() / w_corn.sum().clamp(min=1.0)

    probs_14 = corn_logits_to_unconditional_14ch(logits)
    emd_loss = F.mse_loss(probs_14, y_14ch)

    ch_weights = torch.ones(14, device=logits.device)
    ch_weights[[4, 9, 13]] = 2.8   # 过度生成 P0, P1, P2
    ch_weights[[2, 8, 12]] = 2.0   # 过锐 P0, P1, P2
    ch_weights[5] = 2.2            # 虚化突兀 P0

    tp = (probs_14 * y_14ch).sum(dim=0)
    fp = (probs_14 * (1.0 - y_14ch)).sum(dim=0)
    fn = ((1.0 - probs_14) * y_14ch).sum(dim=0)
    soft_f1 = (2.0 * tp + 1e-4) / (2.0 * tp + fp + fn + 1e-4)
    f1_loss = ((1.0 - soft_f1) * ch_weights).sum() / ch_weights.sum()

    return corn_loss + 0.35 * emd_loss + 0.25 * f1_loss


class RealVLMDaDataset(Dataset):
    def __init__(self, rows, max_pixels: int, train: bool, flip: bool = False):
        self.rows = rows
        self.mp = max_pixels
        self.train = train
        self.flip = flip

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(find_real_img(r["name"])).convert("RGB")
        if self.flip or (self.train and random.random() < 0.5):
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = resize_cap(im, self.mp)
        g = np.array([int(r[c]) for c in DIMS], dtype=np.int64)
        return im, g, r["name"]


class VLMCollate:
    def __init__(self, processor):
        self.p = processor
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DIAGNOSTIC_PROMPT}]}]
        self.text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def __call__(self, batch):
        ims = [b[0] for b in batch]
        enc = self.p(text=[self.text] * len(ims), images=ims, padding=True, return_tensors="pt")
        g_np = np.stack([b[1] for b in batch], axis=0)
        t_corn, m_corn = labels_to_corn_targets(g_np)
        y_14 = labels_to_14ch(g_np)
        enc["grades"] = torch.from_numpy(g_np)
        enc["t_corn"] = torch.from_numpy(t_corn)
        enc["m_corn"] = torch.from_numpy(m_corn)
        enc["y_14ch"] = torch.from_numpy(y_14)
        enc["names"] = [b[2] for b in batch]
        return enc


def to_dev(b, dev):
    keys = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
    return {k: b[k].to(dev, non_blocking=True) for k in keys if k in b}


@torch.no_grad()
def predict_corn(model, loader, dev):
    model.eval()
    P14, G, N = [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(**to_dev(b, dev))
        probs_14 = corn_logits_to_unconditional_14ch(lg.float()).cpu().numpy()
        P14.append(probs_14)
        G.append(b["grades"].numpy())
        N.extend(b["names"])
    P14 = np.concatenate(P14, axis=0)
    G = np.concatenate(G, axis=0)
    p0, p1, p2 = ch14_to_p012(P14)
    return p0, p1, p2, P14, G, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/resplits/split_s16.csv")
    ap.add_argument("--manifest", default="data/manifest_combined.csv")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--max-pixels", type=int, default=896 * 896)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=90))
    rank = dist.get_rank()
    world = dist.get_world_size()
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr_)
    dev = f"cuda:{lr_}"
    random.seed(17 + rank)
    np.random.seed(17 + rank)
    torch.manual_seed(17)
    is0 = (rank == 0)
    log = (lambda *s: print(*s, flush=True)) if is0 else (lambda *s: None)

    man = {r["name"].strip(): r for r in csv.DictReader(open(a.manifest, encoding="utf-8"))}
    rows = []
    for r in csv.DictReader(open(a.split, encoding="utf-8")):
        n = r["name"].strip()
        # Strictly 100% pure real human-annotated images only!
        if n in man and not n.startswith(("v9_", "v8_", "v7_", "v6_")):
            rows.append({"name": n, "split": r["split"].strip(), **{c: int(man[n][c]) for c in DIMS}})

    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]
    te = [r for r in rows if r["split"] == "test"]
    if not te:
        te = va

    # Oversample pure-real Over-Generation & Abrupt Bokeh + Clean to sharpen decision boundaries
    og_bokeh = [r for r in tr if r["过度生成"] > 0 or r["虚化突兀"] > 0]
    clean_tr = [r for r in tr if all(r[c] == 0 for c in DIMS)]
    tr = tr + og_bokeh + clean_tr
    log(f"[Stage B Qwen3-VL-8B ABMIL+CORN] Pure-Real train={len(tr)} val={len(va)} test={len(te)} world={world}")

    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = "right"
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
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

    model = Qwen3VL_ABMIL_CORN(base, image_token_id=img_tok_id, hidden_size=hid).to(dev)
    for m in [model.vis_anom_score, model.vis_attn_V, model.vis_attn_U, model.vis_attn_w, model.head]:
        m.float()

    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable params: {ntr / 1e6:.1f}M / {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_], find_unused_parameters=False)

    col = VLMCollate(proc)
    sampler = DistributedSampler(RealVLMDaDataset(tr, a.max_pixels, True), shuffle=True, seed=17)
    ltr = DataLoader(
        RealVLMDaDataset(tr, a.max_pixels, True),
        batch_size=a.batch, sampler=sampler, num_workers=a.workers,
        collate_fn=col, pin_memory=True, persistent_workers=(a.workers > 0), drop_last=True,
    )
    mk_eval = lambda rs, flip=False: DataLoader(
        RealVLMDaDataset(rs, a.max_pixels, False, flip=flip),
        batch_size=a.batch * 2, shuffle=False, num_workers=a.workers, collate_fn=col, pin_memory=True,
    )
    lva = mk_eval(va, False)
    lte = mk_eval(te, False)

    head_p = [p for n, p in model.module.named_parameters() if p.requires_grad and not n.startswith("base.")]
    lora_p = [p for n, p in model.module.named_parameters() if p.requires_grad and n.startswith("base.")]
    opt = torch.optim.AdamW(
        [{"params": lora_p, "lr": a.lr}, {"params": head_p, "lr": a.lr_head}],
        weight_decay=0.01, fused=True,
    )
    total_steps = max(1, len(ltr) * a.epochs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr_head], total_steps=total_steps, pct_start=0.08)

    best_score = -1.0
    best_ep = 1
    ckpt_path = a.out.replace(".json", "_best.pt")

    step = 0
    for ep in range(a.epochs):
        sampler.set_epoch(ep)
        model.train()
        t0 = time.time()
        tl, nb = 0.0, 0
        for b in ltr:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**to_dev(b, dev))
            loss = compute_corn_emd_loss(
                lg.float(),
                b["t_corn"].to(dev, non_blocking=True),
                b["m_corn"].to(dev, non_blocking=True),
                b["y_14ch"].to(dev, non_blocking=True),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            tl += loss.item()
            nb += 1
            step += 1
            if is0 and (step % 20 == 0 or step == 3):
                mem = torch.cuda.max_memory_allocated() / 2**30
                log(f"ep{ep+1} step {step}/{total_steps} loss={tl/nb:.4f} mem={mem:.1f}GiB {(time.time()-t0)/nb:.2f}s/it")

        dist.barrier()
        if is0:
            vp0, vp1, vp2, vp14, gv, _ = predict_corn(model.module, lva, dev)
            sc = eval_14ch(vp14, gv)
            pos_p0 = 25.0 * sc["S_pos"] + 20.0 * sc["S_p0"]
            log(f"== ep{ep+1} val total={sc['total']:.2f} S_pos={sc['S_pos']:.3f} S_p0={sc['S_p0']:.3f} pos+p0={pos_p0:.2f} t={time.time()-t0:.0f}s")
            if pos_p0 > best_score:
                best_score = pos_p0
                best_ep = ep + 1
                torch.save(
                    {k: v.cpu() for k, v in model.module.state_dict().items() if "lora" in k or not k.startswith("base.") or "merger" in k},
                    ckpt_path,
                )
        dist.barrier()

    if is0:
        sd = torch.load(ckpt_path, map_location="cpu")
        model.module.load_state_dict(sd, strict=False)
        vp0, vp1, vp2, vp14, gv, vn = predict_corn(model.module, lva, dev)
        tp0, tp1, tp2, tp14, gt, tn = predict_corn(model.module, lte, dev)

        # Run 2x Flip TTA on test_150
        tpl = json.load(open("results/v2_full/template_150.json"))
        ordered_150 = [e["image"] for e in tpl]
        rows_150 = [{"name": nm, **{c: 0 for c in DIMS}} for nm in ordered_150]
        l150_orig = mk_eval(rows_150, flip=False)
        l150_flip = mk_eval(rows_150, flip=True)
        p0_1, p1_1, p2_1, _, _, _ = predict_corn(model.module, l150_orig, dev)
        p0_2, p1_2, p2_2, _, _, _ = predict_corn(model.module, l150_flip, dev)
        t150_p0 = 0.5 * (p0_1 + p0_2)
        t150_p1 = 0.5 * (p1_1 + p1_2)
        t150_p2 = 0.5 * (p2_1 + p2_2)

        st = eval_14ch(tp14, gt)
        res = {
            "best_epoch": best_ep,
            "test": {k: st[k] for k in ["total", "S_pos", "S_p0", "p0R", "p0P", "S_clean"]},
            "args": vars(a),
        }
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        npz_path = a.out.replace(".json", "_probs.npz")
        np.savez_compressed(
            npz_path,
            va_p0=vp0, va_p1=vp1, va_p2=vp2, gva=gv, va_names=np.array(vn),
            te_p0=tp0, te_p1=tp1, te_p2=tp2, gte=gt, te_names=np.array(tn),
            t150_p0=t150_p0, t150_p1=t150_p1, t150_p2=t150_p2, names_150=np.array(ordered_150),
        )
        log("Saved Stage B Qwen3-VL-8B ABMIL+CORN predictions to:", npz_path)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
