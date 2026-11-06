"""Qwen3-VL-8B + LoRA (vision tower + LLM) + discriminative severity heads (`train_vlm_cls.py`).

- No text generation: hidden states -> [mean(image tokens) || last token] -> 6 heads
  (4-way for 模糊/纹理损失/过锐/过度生成, 2-way for 花屏/虚化突兀).
- Native resolution input (capped at --max-pixels), so pixel-level evidence is kept.
- Same leak-free split + synthetic ladders as V9_SpecA_OverGen_Giant (data/v9_ladders/split_s{16,17}_overgen.csv),
  evaluated with the exact same decode/threshold code for a fair comparison.
- DDP via torchrun.
"""
import argparse, csv, json, math, os, random, sys, time, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
sys.path.insert(0, 'code/decision_eval')
import comp_metrics as cm

CLASSES = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
FULL = ['模糊', '纹理损失', '过锐', '过度生成']
NCLS = [4 if c in FULL else 2 for c in CLASSES]
IMG_DIRS = ['/tmp/v9_authentic_imgs', '/tmp/v8_ladder_imgs', '/tmp/v7_overgen_pro_imgs', '/tmp/v6_synth_imgs',
            'data/images', 'data/images_supp', 'data/test_150', 'data/train', 'data/test']
PROMPT = ('请评估这张图片的画质缺陷：模糊、纹理损失、过锐、花屏、过度生成（多余或不合理的物体/肢体/纹理）、'
          '虚化突兀，并判断每类的严重程度。')


def find_img(name):
    for d in IMG_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def resize_cap(im, max_pixels, min_pixels=256 * 256, mult=32):
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


class DS(Dataset):
    def __init__(self, rows, max_pixels, train):
        self.rows, self.mp, self.train = rows, max_pixels, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(find_img(r['name'])).convert('RGB')
        if self.train and random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = resize_cap(im, self.mp)
        g = torch.tensor([int(r[c]) for c in CLASSES], dtype=torch.long)
        return im, g, r['name']


class Collate:
    def __init__(self, processor):
        self.p = processor
        msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
        self.text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def __call__(self, batch):
        ims = [b[0] for b in batch]
        enc = self.p(text=[self.text] * len(ims), images=ims, padding=True, return_tensors='pt')
        enc['grades'] = torch.stack([b[1] for b in batch])
        enc['names'] = [b[2] for b in batch]
        return enc


class VLMCls(nn.Module):
    def __init__(self, base, img_tok_id, hid):
        super().__init__()
        self.base = base  # Qwen3VLModel (possibly peft-wrapped)
        self.img_tok_id = img_tok_id
        self.norm = nn.LayerNorm(hid * 2)
        self.drop = nn.Dropout(0.1)
        self.proj = nn.Sequential(nn.Linear(hid * 2, 1024), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(1024, n) for n in NCLS])

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, **extra):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                        pixel_values=pixel_values, image_grid_thw=image_grid_thw, **extra)
        h = out.last_hidden_state  # (B, T, H)
        im_mask = (input_ids == self.img_tok_id).unsqueeze(-1).to(h.dtype)
        im_mean = (h * im_mask).sum(1) / im_mask.sum(1).clamp(min=1)
        last_idx = attention_mask.sum(1) - 1
        last = h[torch.arange(h.size(0), device=h.device), last_idx]
        z = self.proj(self.drop(self.norm(torch.cat([im_mean, last], -1).float())))
        return [hd(z) for hd in self.heads]


def loss_fn(logits, g, focus_w):
    tot = 0.0
    for ci, lg in enumerate(logits):
        w = focus_w[ci]
        if NCLS[ci] == 4:
            tgt = g[:, ci]
            cw = torch.tensor([1.0, 1.3, 1.4, 1.5], device=lg.device)
            ce = F.cross_entropy(lg, tgt, weight=cw, label_smoothing=0.02)
            pr = lg.softmax(-1)
            cdf_p = pr.cumsum(-1)[:, :3]
            cdf_t = F.one_hot(tgt, 4).float().cumsum(-1)[:, :3]
            emd = ((cdf_p - cdf_t) ** 2).sum(-1).mean()
            tot = tot + w * (ce + 0.8 * emd)
        else:
            tgt = (g[:, ci] == 3).long()
            tot = tot + w * F.cross_entropy(lg, tgt, weight=torch.tensor([1.0, 1.5], device=lg.device), label_smoothing=0.02)
    return tot / sum(focus_w)


def to_exact(logits_list):
    """-> p0 (n,6), p1 (n,4), p2 (n,4) exact-grade probabilities."""
    n = logits_list[0].shape[0]
    p0 = np.zeros((n, 6)); p1 = np.zeros((n, 4)); p2 = np.zeros((n, 4))
    for ci, c in enumerate(CLASSES):
        pr = torch.tensor(logits_list[ci]).float().softmax(-1).numpy()
        if c in FULL:
            k = FULL.index(c)
            p2[:, k], p1[:, k], p0[:, ci] = pr[:, 1], pr[:, 2], pr[:, 3]
        else:
            p0[:, ci] = pr[:, 1]
    return p0, p1, p2


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items() if torch.is_tensor(v) and k != 'grades'}


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    L = [[] for _ in CLASSES]; G = []; N = []
    for b in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(**to_dev(b, dev))
        for ci in range(6):
            L[ci].append(lg[ci].float().cpu().numpy())
        G.append(b['grades'].numpy()); N += b['names']
    L = [np.concatenate(x) for x in L]
    p0, p1, p2 = to_exact(L)
    return p0, p1, p2, np.concatenate(G), N


def ap(y, s):
    o = np.argsort(-s); y = y[o]
    if y.sum() == 0:
        return float('nan')
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--split', required=True)
    ap_.add_argument('--manifest', required=True)
    ap_.add_argument('--model', default='models/qwen3vl-8b')
    ap_.add_argument('--focus', default='过度生成')
    ap_.add_argument('--epochs', type=int, default=3)
    ap_.add_argument('--batch', type=int, default=8)
    ap_.add_argument('--max-pixels', type=int, default=1024 * 1024)
    ap_.add_argument('--lr', type=float, default=2e-4)
    ap_.add_argument('--lr-head', type=float, default=1e-3)
    ap_.add_argument('--rank', type=int, default=64)
    ap_.add_argument('--workers', type=int, default=8)
    ap_.add_argument('--max-steps', type=int, default=-1)
    ap_.add_argument('--grad-ckpt', action='store_true')
    ap_.add_argument('--out', required=True)
    a = ap_.parse_args()

    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=90))
    rank = dist.get_rank(); world = dist.get_world_size()
    lr_ = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(lr_); dev = f'cuda:{lr_}'
    random.seed(7 + rank); np.random.seed(7 + rank); torch.manual_seed(7)
    is0 = rank == 0
    log = (lambda *s: print(*s, flush=True)) if is0 else (lambda *s: None)

    man = {r['name'].strip(): r for r in csv.DictReader(open(a.manifest, encoding='utf-8'))}
    rows = []
    for r in csv.DictReader(open(a.split, encoding='utf-8')):
        n = r['name'].strip()
        if n in man:
            rows.append({'name': n, 'split': r['split'].strip(), **{c: int(man[n][c]) for c in CLASSES}})
    tr = [r for r in rows if r['split'] == 'train']
    va = [r for r in rows if r['split'] == 'val']
    te = [r for r in rows if r['split'] == 'test']
    focus_cis = [CLASSES.index(c) for c in a.focus.split(',')]
    real_tr = [r for r in tr if not r['name'].startswith(('v9_', 'v8_', 'v7_', 'v6_'))]
    tr = tr + [r for r in real_tr if any(r[CLASSES[ci]] > 0 for ci in focus_cis)]
    tr = tr + [r for r in real_tr if all(r[c] == 0 for c in CLASSES)]  # clean x2
    log(f'train={len(tr)} (synth={sum(r["name"].startswith("v9_") for r in tr)}) val={len(va)} test={len(te)} world={world}')

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    proc = AutoProcessor.from_pretrained(a.model)
    proc.tokenizer.padding_side = 'right'
    full = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation='flash_attention_2')
    base = full.model
    del full.lm_head
    img_tok_id = full.config.image_token_id
    hid = full.config.text_config.hidden_size
    target = r'.*(language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))|visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2)))$'
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, target_modules=target, bias='none')
    base = get_peft_model(base, cfg)
    # also train the vision merger (patch -> LLM projector) fully
    for n, p in base.named_parameters():
        if 'merger' in n:
            p.requires_grad = True
    if a.grad_ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.enable_input_require_grads()
    model = VLMCls(base, img_tok_id, hid).to(dev)
    for m in [model.norm, model.proj, model.heads]:
        m.float()
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'trainable params: {ntr/1e6:.1f}M / {sum(p.numel() for p in model.parameters())/1e9:.2f}B')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_], find_unused_parameters=False)

    col = Collate(proc)
    sampler = DistributedSampler(DS(tr, a.max_pixels, True), shuffle=True, seed=7)
    ltr = DataLoader(DS(tr, a.max_pixels, True), batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                     collate_fn=col, pin_memory=True, persistent_workers=True, prefetch_factor=4, drop_last=True)
    mk = lambda rs: DataLoader(DS(rs, a.max_pixels, False), batch_size=a.batch * 2, shuffle=False,
                               num_workers=a.workers, collate_fn=col, pin_memory=True)
    lva, lte = mk(va), mk(te)

    head_p = [p for n, p in model.module.named_parameters() if p.requires_grad and not n.startswith('base.')]
    lora_p = [p for n, p in model.module.named_parameters() if p.requires_grad and n.startswith('base.')]
    opt = torch.optim.AdamW([{'params': lora_p, 'lr': a.lr}, {'params': head_p, 'lr': a.lr_head}], weight_decay=0.01, fused=True)
    steps = len(ltr) * a.epochs if a.max_steps < 0 else a.max_steps
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr_head], total_steps=steps, pct_start=0.06)
    focus_w = [2.0 if ci in focus_cis else 1.0 for ci in range(6)]

    from train_v9_specialist import optimise_scaled, decode_scaled_argmax
    step = 0; best = None
    for ep in range(a.epochs):
        sampler.set_epoch(ep); model.train(); t0 = time.time(); tl = 0.0; nb = 0
        for b in ltr:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(**to_dev(b, dev))
            loss = loss_fn([x.float() for x in lg], b['grades'].to(dev), focus_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            tl += loss.item(); nb += 1; step += 1
            if is0 and (step % 20 == 0 or step == 3):
                mem = torch.cuda.max_memory_allocated() / 2**30
                log(f'ep{ep+1} step {step}/{steps} loss={tl/nb:.4f} mem={mem:.1f}GiB {(time.time()-t0)/nb:.2f}s/it')
            if a.max_steps > 0 and step >= a.max_steps:
                break
        dist.barrier()
        if is0:
            vp0, vp1, vp2, gv, _ = predict(model.module, lva, dev)
            b_sca, th = optimise_scaled(gv, vp0, vp1, vp2, 0.02)
            ogv = {f'{x[0]}_{x[1]}': round(x[4], 3) for x in b_sca['per'] if x[0] in [CLASSES[c] for c in focus_cis]}
            log(f'== ep{ep+1} val_sca={b_sca["total"]:.2f} S_pos={b_sca["S_pos"]:.3f} S_p0={b_sca["S_p0"]:.3f} focus={ogv} t={time.time()-t0:.0f}s')
            fs = float(np.mean(list(ogv.values())))
            if best is None or fs > best[0]:
                best = (fs, ep + 1)
                torch.save({k: v.cpu() for k, v in model.module.state_dict().items() if 'lora' in k or not k.startswith('base.') or 'merger' in k},
                           a.out.replace('.json', '_best.pt'))
        dist.barrier()
        if a.max_steps > 0 and step >= a.max_steps:
            break

    if is0:
        sd = torch.load(a.out.replace('.json', '_best.pt'))
        model.module.load_state_dict(sd, strict=False)
        vp0, vp1, vp2, gv, vn = predict(model.module, lva, dev)
        tp0, tp1, tp2, gt, tn = predict(model.module, lte, dev)
        _, th = optimise_scaled(gv, vp0, vp1, vp2, 0.02)
        st = cm.score(gt, decode_scaled_argmax(tp0, tp1, tp2, *th), 0.02)
        per = {f'{x[0]}_{x[1]}': {'P': x[2], 'R': x[3], 'F1': x[4]} for x in st['per']}
        k = FULL.index('过度生成')
        aps = {'过度生成_P2': ap(gt[:, 4] == 1, tp2[:, k]), '过度生成_P1': ap(gt[:, 4] == 2, tp1[:, k]), '过度生成_P0': ap(gt[:, 4] == 3, tp0[:, 4])}
        res = {'best_epoch': best[1], 'test': {k2: st[k2] for k2 in ['total', 'S_pos', 'S_p0', 'p0R', 'p0P', 'S_clean']},
               'per_label': per, 'overgen_AP': aps, 'args': vars(a)}
        json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
        np.savez_compressed(a.out.replace('.json', '_probs.npz'), va_p0=vp0, va_p1=vp1, va_p2=vp2, gva=gv, va_names=np.array(vn),
                            te_p0=tp0, te_p1=tp1, te_p2=tp2, gte=gt, te_names=np.array(tn))
        log('TEST', json.dumps(res['test'], ensure_ascii=False))
        log('OVERGEN F1', {kk: round(per[kk]['F1'], 3) for kk in per if '过度生成' in kk}, 'AP', {kk: round(v, 3) for kk, v in aps.items()})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
