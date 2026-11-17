#!/usr/bin/env python3
"""Extract Top-3 FLUX.2-klein-4B DiT internal denoiser features + velocity residual at small noise t=0.05, 0.15."""
import os
import sys
import time
import numpy as np
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

REPO = "/wanqing-develop/luowenjing/18"
for _p in ("code", "code/next55", "code/r55"):
    p_abs = os.path.join(REPO, _p)
    if p_abs not in sys.path:
        sys.path.insert(0, p_abs)

import common55 as c
from vlm_c233397_qy8 import PureRealImageResolver


def main():
    t0 = time.time()
    device = torch.device("cuda:0")
    pipe = Flux2KleinPipeline.from_pretrained("/wanqing-models/FLUX.2-klein-4B", torch_dtype=torch.bfloat16).to(device)
    pipe.vae.eval()
    pipe.transformer.eval()

    with torch.no_grad():
        prompt_embeds, txt_ids = pipe.encode_prompt(prompt="", device=device, num_images_per_prompt=1, max_sequence_length=64)
    # Free text encoder to save VRAM
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.to("cpu")
        del pipe.text_encoder
        torch.cuda.empty_cache()

    hook_buf = {}
    def _hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        hook_buf["mid"] = t.detach()

    # Register hook on middle single_transformer_block (or transformer_blocks[-1])
    if hasattr(pipe.transformer, "single_transformer_blocks") and len(pipe.transformer.single_transformer_blocks) > 0:
        mid_idx = len(pipe.transformer.single_transformer_blocks) // 2
        h = pipe.transformer.single_transformer_blocks[mid_idx].register_forward_hook(_hook)
    else:
        mid_idx = len(pipe.transformer.transformer_blocks) - 1
        h = pipe.transformer.transformer_blocks[mid_idx].register_forward_hook(_hook)
    print(f"[FLUX.2-klein-4B loaded] mid_idx={mid_idx} in {time.time()-t0:.1f}s", flush=True)

    names, gold, fold, grp = c.load_split()
    resolver = PureRealImageResolver()

    # Prepare fixed 512x512 latent ids
    H, W = 512, 512
    feats = []
    rng = torch.Generator(device=device).manual_seed(2026)

    for i, nm in enumerate(names):
        im = resolver.load_rgb_pil(str(nm)).resize((W, H), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            z0 = pipe.vae.encode(x).latent_dist.mean  # (1, 32, H/8, W/8)
            z0 = pipe._patchify_latents(z0)  # (1, 128, H/16, W/16)
            # Normalize latent if bn exists in Flux2 VAE
            if hasattr(pipe.vae, "bn"):
                mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(z0)
                std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(z0)
                z0_n = (z0 - mean) / std
            else:
                z0_n = z0

            z_packed = pipe._pack_latents(z0_n)  # (1, L, C_p)
            img_ids = pipe._prepare_latent_ids(z0_n).to(device)

            eps = torch.randn(z_packed.shape, generator=rng, device=device, dtype=z_packed.dtype)
            row_parts = []
            # Latent stats
            zf = z0_n.float()
            row_parts.append(zf.mean(dim=(2, 3)).squeeze(0))
            row_parts.append(zf.std(dim=(2, 3)).squeeze(0))

            for t_val in (0.05, 0.15):
                zt = (1.0 - t_val) * z_packed + t_val * eps
                ts = torch.tensor([t_val], device=device, dtype=z_packed.dtype)
                v_pred = pipe.transformer(
                    hidden_states=zt,
                    timestep=ts,
                    encoder_hidden_states=prompt_embeds,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    return_dict=False,
                )[0]
                v_tgt = eps - z_packed
                err = (v_pred - v_tgt).float()
                vp = v_pred.float()
                # Summary stats + mid-block pooled features
                stats = torch.stack([
                    err.pow(2).mean(), err.abs().mean(), err.std(),
                    vp.pow(2).mean(), vp.abs().mean(),
                ])
                mid = hook_buf["mid"].float().squeeze(0)  # (L_tot, D)
                # Take image tokens (last z_packed.shape[1] or first? concat mean/std over all)
                row_parts.extend([stats, mid.mean(dim=0), mid.std(dim=0)])

            feat_vec = torch.cat(row_parts, dim=0).cpu().numpy().astype(np.float32)
        feats.append(feat_vec)
        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{len(names)}] {time.time()-t0:.1f}s", flush=True)

    h.remove()
    X = np.stack(feats, axis=0)
    out_npz = os.path.join(REPO, "results/r55/probes/flux2_klein_dit_feats.npz")
    np.savez_compressed(out_npz, names=np.array(names, dtype=object), X=X)
    print(f"[Saved] {out_npz} shape={X.shape} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
