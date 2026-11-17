#!/usr/bin/env python3
"""Extract frozen vision features (orig + hflip averaged) for any HF vision encoder."""
import argparse
import os
import sys
import time
import numpy as np
import torch
from PIL import ImageOps
from transformers import AutoImageProcessor, AutoModel

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

import common55 as c
from vlm_c233397_qy8 import PureRealImageResolver


def pool_features(out):
    # out: BaseModelOutput or similar
    hs = out.last_hidden_state.float()  # (B, L, D)
    mean_p = hs.mean(dim=1)
    std_p = hs.std(dim=1)
    parts = [mean_p, std_p]
    if getattr(out, "pooler_output", None) is not None:
        parts.append(out.pooler_output.float())
    return torch.cat(parts, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--interp-size", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    t0 = time.time()

    try:
        proc = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=False)
        model = AutoModel.from_pretrained(args.model, trust_remote_code=False, dtype=torch.bfloat16).to(device).eval()
    except Exception:
        proc = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
    print(f"[Loaded {args.model}] in {time.time()-t0:.1f}s | VRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    names, gold, fold, grp = c.load_split()
    resolver = PureRealImageResolver()

    feats = []
    bs = args.batch_size
    for i in range(0, len(names), bs):
        batch_nm = names[i:i + bs]
        imgs_orig = [resolver.load_rgb_pil(str(nm)) for nm in batch_nm]
        imgs_flip = [ImageOps.mirror(im) for im in imgs_orig]

        kw = {"return_tensors": "pt"}
        if args.interp_size > 0:
            kw["size"] = {"height": args.interp_size, "width": args.interp_size}

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            inp0 = proc(images=imgs_orig, **kw).to(device)
            inp1 = proc(images=imgs_flip, **kw).to(device)
            fwd_kw = {"interpolate_pos_encoding": True} if args.interp_size > 0 else {}
            try:
                f0 = pool_features(model(**inp0, **fwd_kw))
                f1 = pool_features(model(**inp1, **fwd_kw))
            except TypeError:
                f0 = pool_features(model(**inp0))
                f1 = pool_features(model(**inp1))
            f_avg = (0.5 * (f0 + f1)).cpu().numpy().astype(np.float32)
        feats.append(f_avg)
        if (i // bs) % 40 == 0:
            print(f"  [{i+len(batch_nm)}/{len(names)}] {time.time()-t0:.1f}s", flush=True)

    X = np.concatenate(feats, axis=0)
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz, names=np.array(names, dtype=object), X=X)
    print(f"[Saved] {args.out_npz} shape={X.shape} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
