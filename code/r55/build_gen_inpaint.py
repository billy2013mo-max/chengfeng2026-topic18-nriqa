"""Audit 2026-11-14, P5: local, hard-masked generative edits for 过度生成 with matched negatives (run with venv_flux).

Why this differs from the closed FLUX ladder (`build_overgen_ladder.py`, whole-frame instruction re-render, not recognised
by the real-data teachers): the edit is confined to a mask (pixels outside it are the seed's), the negative goes through
the SAME pipeline, mask, seed and strength with a "keep unchanged" instruction (so generator traces are not a shortcut),
and the work is done at native scale on a crop around the mask (`padding_mask_crop`).

Recipes (`--recipe`):
  ogk     FLUX.1-Kontext inpaint; positive instruction = one of several "invent content" edits (extra object / extra
          fingers / hallucinated texture / garbled fine detail), negative = faithful reconstruction.
  ogkup   "generative upscaling": the masked region is first degraded (downscale x2-4, back up), then the positive asks
          the model to restore sharp detail (it invents), the negative restores at low strength.

Seeds: all-clean labelled frames (513, first batch).  Labels by construction: positive = 过度生成 present (level 9,
grade unknown), negative = level 0; every other defect is left unsupervised by synth_build conventions (`mask`).
Output: data/r55_aux/images/r55syn_<R>_{p,n}__<seed>, data/r55_aux/manifest_<R>.csv (SynthLab format) plus the edit
masks in data/r55_aux/masks_<R>/ for mask-supervised use.

  CUDA_VISIBLE_DEVICES=0 venv_flux/bin/python -I code/r55/build_gen_inpaint.py --recipe ogk --n 64 --shard 0 --nshards 1
  python3 code/r55/build_gen_inpaint.py --recipe ogk --merge --nshards 8
"""
import argparse
import csv
import os
import random
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = '/wanqing-develop/luowenjing/18'
DIMS = ['模糊', '纹理损失', '过锐', '花屏', '过度生成', '虚化突兀']
IMG_OUT = os.path.join(REPO, 'data/r55_aux/images')
POS_PROMPTS = [
    'add a small strange extra object here that does not belong in the scene, rendered with an artificial AI-generated look',
    'redraw this area with hallucinated, overly detailed, unnatural AI-generated texture that was not in the original',
    'add distorted extra fingers or duplicated small body parts here, like a failed AI image generation',
    'fill this region with garbled, invented fine details and fake repeating patterns typical of AI over-enhancement',
    'turn the surface here into a waxy, plastic, over-sharpened AI-generated texture with invented detail',
]
NEG_PROMPT_K = 'Keep the image exactly the same, a faithful photographic reproduction. Change nothing.'


def seeds_clean():
    q = list(csv.DictReader(open(os.path.join(REPO, 'data/qingyu_splits/qingyu_8fold.csv'), encoding='utf-8')))
    rb = {r['name']: r['split'] for r in csv.DictReader(open(os.path.join(REPO, 'data/manifest_combined.csv'), encoding='utf-8'))}
    out = [(r['name'], int(r['fold'])) for r in q if all(int(r[d]) == 0 for d in DIMS) and rb[r['name']] == 'orig']
    return sorted(out)


def find_img(name):
    for d in ('data/images', 'data/images_supp'):
        p = os.path.join(REPO, d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def textured_mask(im, rng, area=(0.03, 0.10)):
    """Ellipse mask over a window with high gradient energy (where invented detail is visible)."""
    g = np.asarray(im.convert('L'), dtype=np.float32)
    gy, gx = np.gradient(g)
    e = np.hypot(gx, gy)
    h, w = g.shape
    a = rng.uniform(*area) * h * w
    ar = rng.uniform(0.6, 1.6)
    mh, mw = int(np.sqrt(a / ar)), int(np.sqrt(a * ar))
    mh, mw = max(32, min(mh, h - 2)), max(32, min(mw, w - 2))
    best = None
    for _ in range(24):
        y0, x0 = rng.integers(0, h - mh), rng.integers(0, w - mw)
        s = e[y0:y0 + mh, x0:x0 + mw].mean()
        if best is None or s > best[0]:
            best = (s, y0, x0)
    _, y0, x0 = best
    m = Image.new('L', (w, h), 0)
    ImageDraw.Draw(m).ellipse([x0, y0, x0 + mw, y0 + mh], fill=255)
    return m


def degrade_region(im, mask, rng):
    f = rng.uniform(2.0, 4.0)
    w, h = im.size
    lo = im.resize((max(8, int(w / f)), max(8, int(h / f))), Image.BICUBIC).resize((w, h), Image.BICUBIC)
    return Image.composite(lo, im, mask)


def kontext_edit(pipe, im, mask, prompt, seed, steps, gs=2.5):
    """Edit a native-pixel crop around the mask with FLUX.1-Kontext and paste the result back inside the mask only."""
    import torch
    x0, y0, x1, y1 = mask.getbbox()
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)            # crop = 2x the mask's longer side
    W, H = im.size
    bx0, by0 = int(max(0, cx - half)), int(max(0, cy - half))
    bx1, by1 = int(min(W, cx + half)), int(min(H, cy + half))
    crop = im.crop((bx0, by0, bx1, by1))
    cw, ch = crop.size
    sc = 1024 / max(cw, ch)
    tw, th = max(256, int(round(cw * sc / 16)) * 16), max(256, int(round(ch * sc / 16)) * 16)
    g = torch.Generator('cuda').manual_seed(seed)
    out = pipe(image=crop.resize((tw, th), Image.LANCZOS), prompt=prompt, guidance_scale=gs, num_inference_steps=steps,
               height=th, width=tw, generator=g, max_area=tw * th, _auto_resize=False).images[0]
    out = out.resize((cw, ch), Image.LANCZOS)
    full = im.copy()
    full.paste(out, (bx0, by0))
    return Image.composite(full, im, mask.filter(ImageFilter.GaussianBlur(2)))


def build(a):
    import torch
    from diffusers import FluxKontextPipeline
    pipe = FluxKontextPipeline.from_pretrained('/wanqing-models/FLUX.1-Kontext-dev', torch_dtype=torch.bfloat16).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    seeds = seeds_clean()
    rs = random.Random(20261114)
    rs.shuffle(seeds)
    seeds = seeds[:a.n][a.shard::a.nshards]
    os.makedirs(IMG_OUT, exist_ok=True)
    mdir = os.path.join(REPO, f'data/r55_aux/masks_{a.recipe}')
    os.makedirs(mdir, exist_ok=True)
    rows = []
    for i, (name, fold) in enumerate(seeds):
        rng = np.random.default_rng(zlib.crc32(f"{a.recipe}|{name}".encode()))
        im = Image.open(find_img(name)).convert('RGB')
        mask = textured_mask(im, rng)
        base = degrade_region(im, mask, rng) if a.recipe == 'ogkup' else im
        if a.recipe == 'ogk':
            prompt = 'In the center of the image, ' + POS_PROMPTS[int(rng.integers(len(POS_PROMPTS)))] + '. Make the change clearly visible.'
        else:
            prompt = 'Upscale and enhance this image: restore it with sharp, crisp, highly detailed high-resolution texture.'
        seed = int(rng.integers(2 ** 31))
        outs = {'p': kontext_edit(pipe, base, mask, prompt, seed, a.steps, gs=a.gs),
                'n': kontext_edit(pipe, base, mask, NEG_PROMPT_K, seed, a.steps) if a.recipe == 'ogk' else
                     Image.composite(base, im, mask)}                 # ogkup negative: the same degraded region, not regenerated
        stem = os.path.splitext(name)[0]
        for tag, lv in (('p', 9), ('n', 0)):
            nm = f'r55syn_{a.recipe}_{tag}__{stem}.png'
            outs[tag].save(os.path.join(IMG_OUT, nm))
            rows.append(dict(name=nm, kind='synth', seed=name, seed_fold=fold, tuple_id=f'{a.recipe}_{stem}', dim=4, level=lv,
                             op=f'kontext:{a.recipe}:' + (prompt[23:70] if tag == 'p' else 'neg')))
        mask.save(os.path.join(mdir, f'{stem}.png'))
        print(f'[{a.recipe} shard {a.shard}] {i + 1}/{len(seeds)} {name}', flush=True)
    with open(os.path.join(REPO, f'data/r55_aux/manifest_{a.recipe}.part{a.shard}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'kind', 'seed', 'seed_fold', 'tuple_id', 'dim', 'level', 'op'])
        w.writeheader()
        w.writerows(rows)


def merge(a):
    rows = []
    for s in range(a.nshards):
        p = os.path.join(REPO, f'data/r55_aux/manifest_{a.recipe}.part{s}.csv')
        rows += list(csv.DictReader(open(p, encoding='utf-8')))
    with open(os.path.join(REPO, f'data/r55_aux/manifest_{a.recipe}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('merged', len(rows))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipe', choices=['ogk', 'ogkup'], required=True)
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--steps', type=int, default=24)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--gs', type=float, default=4.0, help='Kontext guidance for the positive edit (negative keeps 2.5)')
    a = ap.parse_args()
    merge(a) if a.merge else build(a)
