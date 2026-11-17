#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/build_overgen_ladder.py — generative re-render ladder for 过度生成 (run with venv_flux/bin/python, 1 GPU per rank).

Each labelled seed is re-rendered by FLUX.3-Image-Klein (image-conditioned) with three instructions of
increasing strength.  This is the same mechanism as the real defect (AI enhancement / AI re-rendering), and the
re-render is global, so any crop shows it.  The three outputs only carry the ORDERING
    overgen(seed) <= overgen(mild) <= overgen(medium) <= overgen(strong)
plus, later, the fold-k teacher's soft label.  No grade is assigned by recipe.

Seeds: labelled images whose 过度生成 grade is 无问题 or P2 (room above them), max side <= --max-side.

Usage (8 GPUs):
  for r in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$r venv_flux/bin/python code/r55/build_overgen_ladder.py \
       --rank $r --world 8 --n-seeds 400 & done; wait
"""
import argparse
import json
import os
import random
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import AUX_IMG, REPO, append_manifest, load_split_rows, read_manifest  # noqa: E402

KEEP = "Keep the content, layout, composition, text, colours and identity identical."
INSTR = [
    ("L1", "Apply a restrained AI photo enhancement to this image: slightly smooth skin and fine texture, "
           "clean up micro detail, a faint plastic quality. " + KEEP),
    ("L2", "Apply a clearly visible AI enhancement to this image: skin and surfaces become smooth and waxy, "
           "natural grain is replaced by regenerated texture, some small details are redrawn. " + KEEP),
    ("L3", "Apply a heavy AI enhancement and generation pipeline to this image: unnaturally smooth plastic skin, "
           "fine texture and natural grain gone, small details smeared or invented, hair and fabric redrawn with "
           "artificial repetitive strands. " + KEEP),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/flux3-klein")
    ap.add_argument("--pup", default="models/flux3-pup")
    ap.add_argument("--n-seeds", type=int, default=400)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--seed", type=int, default=55)
    a = ap.parse_args()
    os.chdir(REPO)
    os.makedirs(AUX_IMG, exist_ok=True)

    rng = random.Random(a.seed)
    cand = [r for r in load_split_rows() if int(r["过度生成"]) <= 1]
    rng.shuffle(cand)
    seeds = cand[: a.n_seeds][a.rank :: a.world]
    done = {r["name"] for r in read_manifest()}

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from diffusers import ModularPipeline

    t0 = time.time()
    pup_model = Qwen3_5ForConditionalGeneration.from_pretrained(a.pup, dtype=torch.bfloat16, device_map="cuda").eval()
    pup_proc = AutoProcessor.from_pretrained(a.pup)
    sysp = json.load(open(os.path.join(a.pup, "system_prompts.json")))["i2i"]
    blocks = ModularPipeline.from_pretrained(a.model).blocks
    blocks.sub_blocks.pop("prompt_upsampler", None)
    pipe = blocks.init_pipeline(a.model)
    pipe.load_components(dtype={"vae": torch.float32, "default": torch.bfloat16})
    pipe.to("cuda")
    print(f"[overgen r{a.rank}] models loaded {time.time() - t0:.0f}s, {len(seeds)} seeds", flush=True)

    @torch.no_grad()
    def caption(im, instr):
        msgs = [{"role": "system", "content": [{"type": "text", "text": sysp}]},
                {"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": instr}]}]
        text = pup_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = pup_proc(text=[text], images=[im], return_tensors="pt").to("cuda")
        out = pup_model.generate(**inp, max_new_tokens=700, do_sample=False)
        return pup_proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    for i, r in enumerate(seeds):
        name = r["name"]
        p = next((os.path.join(d, name) for d in ("data/images", "data/images_supp") if os.path.exists(os.path.join(d, name))), None)
        if p is None:
            continue
        im = Image.open(p).convert("RGB")
        if max(im.size) > a.max_side or min(im.size) < 448:
            continue
        stem = os.path.splitext(name)[0]
        rows = []
        try:
            for lv, (tag, instr) in enumerate(INSTR, start=1):
                out_name = f"r55og_{tag}__{stem}.png"
                out_p = os.path.join(AUX_IMG, out_name)
                if out_name not in done and not os.path.exists(out_p):
                    res = pipe(prompt=caption(im, instr), image=im, num_inference_steps=a.steps,
                               generator=torch.Generator(device="cpu").manual_seed(a.seed + i), output=["images", "prompt"])
                    gen = res["images"][0]
                    if gen.size != im.size:  # keep native geometry so the tuple is pixel-aligned in scale
                        gen = gen.resize(im.size, Image.BICUBIC)
                    gen.save(out_p + ".tmp.png")
                    os.replace(out_p + ".tmp.png", out_p)
                rows.append(dict(name=out_name, kind="overgen", seed=name, seed_fold=int(r["fold"]),
                                 tuple_id=f"og_{stem}", dim=4, level=lv, op="flux3_" + tag))
        except Exception as e:  # noqa: BLE001
            print("FAIL", name, type(e).__name__, str(e)[:200], flush=True)
            continue
        if len(rows) == 3:
            # one writer per rank; append is name-deduplicated and ranks touch disjoint seeds
            with open(os.path.join(AUX_IMG, f".og_rows_r{a.rank}.jsonl"), "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (i + 1) % 10 == 0:
            print(f"[overgen r{a.rank}] {i + 1}/{len(seeds)} {time.time() - t0:.0f}s", flush=True)
    print(f"[overgen r{a.rank}] done", flush=True)


def _deviation(job):
    """Mean absolute difference to the seed at 512 px on the long side (how far the re-render moved)."""
    import cv2
    import numpy as np
    seed, name = job
    p = next((os.path.join(REPO, d, seed) for d in ("data/images", "data/images_supp")
              if os.path.exists(os.path.join(REPO, d, seed))), None)
    a = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) if p else None
    b = cv2.imdecode(np.fromfile(os.path.join(AUX_IMG, name), np.uint8), cv2.IMREAD_COLOR)
    if a is None or b is None:
        return name, -1.0
    s = 512.0 / max(a.shape[:2])
    size = (max(1, int(a.shape[1] * s)), max(1, int(a.shape[0] * s)))
    a, b = (cv2.resize(x, size, interpolation=cv2.INTER_AREA).astype(np.float32) for x in (a, b))
    return name, float(np.abs(a - b).mean())


def relevel(rows, min_ratio=1.15, min_dev=1.0):
    """The instruction strength does not reliably order the outputs (the smoke test had L2 fully re-drawn and
    L3 almost untouched), so the level is re-assigned from the measured departure from the seed: levels are
    sorted by deviation, a level closer than `min_ratio` to the previous kept one is dropped as ambiguous,
    and a tuple needs >= 2 surviving levels.  The original instruction stays in `op`."""
    from multiprocessing import Pool
    with Pool(32) as pool:
        dev = dict(pool.map(_deviation, [(r["seed"], r["name"]) for r in rows], chunksize=8))
    by = {}
    for r in rows:
        by.setdefault(r["tuple_id"], []).append(r)
    out, n_reordered = [], 0
    for tid, rs in by.items():
        rs = sorted((r for r in rs if dev[r["name"]] >= min_dev), key=lambda r: dev[r["name"]])
        kept = []
        for r in rs:
            if not kept or dev[r["name"]] >= min_ratio * dev[kept[-1]["name"]]:
                kept.append(r)
        if len(kept) < 2:
            continue
        n_reordered += [r["op"] for r in kept] != sorted(r["op"] for r in kept)
        for lv, r in enumerate(kept, start=1):
            out.append(dict(r, level=lv))
    print(f"[overgen] relevel: {len(by)} tuples -> {len(out)} images in {len({r['tuple_id'] for r in out})} tuples "
          f"({n_reordered} re-ordered by measured deviation)")
    return out


def merge():
    """Run once after all ranks finish: folds the per-rank row files into the manifest."""
    rows = []
    for fn in sorted(os.listdir(AUX_IMG)):
        if fn.startswith(".og_rows_r") and fn.endswith(".jsonl"):
            with open(os.path.join(AUX_IMG, fn), encoding="utf-8") as f:
                rows += [json.loads(x) for x in f if x.strip()]
    rows = [r for r in rows if os.path.exists(os.path.join(AUX_IMG, r["name"]))]
    rows = list({r["name"]: r for r in rows}.values())
    rows = relevel(rows)
    print(f"[overgen] merged {append_manifest(rows)} new rows into the manifest")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        merge()
    else:
        main()
