"""Extract frozen multi-scale features for a given backbone.

Views per image:
  G : whole image resized to S x S
  L : 2x2 tiles of the whole image, each resized to Sl x Sl
Features = [global(avg,max), local-mean(avg,max), local-max(avg,max)]
"""
import argparse, csv, os, numpy as np, torch
from PIL import Image

BF16 = True
CONFIGS = {
    'dinov2-base':      dict(kind='hf',   repo='facebook/dinov2-base',                    size=448),
    'dinov2-large':     dict(kind='hf',   repo='facebook/dinov2-large',                   size=448),
    'siglip2-so400m':   dict(kind='hf',   repo='google/siglip2-so400m-patch14-384',       size=384),
    'clip-vit-large':   dict(kind='hf',   repo='openai/clip-vit-large-patch14',           size=224),
    'convnextv2-base':  dict(kind='timm', repo='convnextv2_base.fcmae_ft_in22k_in1k',    size=384),
    'eva02-base':       dict(kind='timm', repo='eva02_base_patch14_448.mim_in22k_ft_in1k', size=448),
    'vit-base-sup':     dict(kind='timm', repo='vit_base_patch16_224.augreg_in21k_ft_in1k', size=336),
    'pe-core-l14':      dict(kind='timm_local', arch='vit_pe_core_large_patch14_336', repo='models/pe-core-l14/model.safetensors', size=336),
    'pe-core-g14':      dict(kind='timm', repo='vit_pe_core_gigantic_patch14_448', size=448),
    'metaclip2-h14':    dict(kind='metaclip', repo='facebook/metaclip-2-worldwide-huge-quickgelu', size=224),
    'siglip2-giant':    dict(kind='hf', repo='models/siglip2-giant', size=384),
    'pe-core-g14':      dict(kind='timm_local', arch='vit_pe_core_gigantic_patch14_448', repo='models/pe-core-g14/model.safetensors', size=448),
    'internvit-6b':     dict(kind='intern', repo='models/internvit-6b', size=448),
}

def load_model(name, device):
    cfg = CONFIGS[name]
    if cfg['kind'] in ('hf', 'metaclip', 'intern'):
        from transformers import AutoModel, AutoImageProcessor, CLIPVisionModel, SiglipVisionModel
        repo = cfg['repo']
        try:
            proc = AutoImageProcessor.from_pretrained(repo)
            mean = np.array(proc.image_mean); std = np.array(proc.image_std)
        except Exception:
            print('no image processor, fallback to ImageNet stats')
            mean = np.array([0.485,0.456,0.406]); std = np.array([0.229,0.224,0.225])
        if cfg['kind'] == 'intern':
            from transformers import AutoModel
            model = AutoModel.from_pretrained(repo, trust_remote_code=True, attn_implementation='eager')
        elif cfg['kind'] == 'metaclip':
            from transformers import MetaClip2VisionModel
            model = MetaClip2VisionModel.from_pretrained(repo, attn_implementation='sdpa')
        elif 'clip' in repo:
            model = CLIPVisionModel.from_pretrained(repo, attn_implementation='sdpa')
        elif 'siglip' in repo:
            model = SiglipVisionModel.from_pretrained(repo, attn_implementation='sdpa')
        else:
            model = AutoModel.from_pretrained(repo, attn_implementation='sdpa')
        if BF16:
            model = model.to(torch.bfloat16)
    elif cfg['kind'] == 'timm_local':
        import timm
        from safetensors.torch import load_file
        model = timm.create_model(cfg['arch'], pretrained=False, num_classes=0)
        sd = load_file(cfg['repo'])
        r = model.load_state_dict(sd, strict=False)
        print('loaded local', cfg['arch'], 'missing', len(r.missing_keys), 'unexpected', len(r.unexpected_keys))
        dc = timm.data.resolve_data_config({}, model=model)
        mean, std = np.array(dc['mean']), np.array(dc['std'])
        if BF16: model = model.to(torch.bfloat16)
    else:
        import timm
        model = timm.create_model(cfg['repo'], pretrained=True, num_classes=0)
        dc = timm.data.resolve_data_config({}, model=model)
        mean, std = np.array(dc['mean']), np.array(dc['std'])
    model.eval().to(device)
    return model, mean, std, cfg

@torch.no_grad()
def forward_pool(model, batch, kind):
    if next(model.parameters()).dtype == torch.bfloat16:
        batch = batch.to(torch.bfloat16)
    if kind in ('hf', 'metaclip', 'intern'):
        out = model(pixel_values=batch)
        tok = out.last_hidden_state           # B,N,D
    else:
        tok = model.forward_features(batch)   # B,N,D for ViT
        if tok.ndim == 4:                     # conv: B,C,H,W
            tok = tok.flatten(2).transpose(1, 2)
    return torch.cat([tok.mean(1), tok.amax(1)], dim=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--imgdir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    device = 'cuda'
    model, mean, std, cfg = load_model(args.model, device)
    S, Sl = cfg['size'], cfg['size']
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1,3,1,1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1,3,1,1)

    rows = list(csv.DictReader(open(args.manifest)))
    feats = []; names = []
    def prep(im, size):
        im = im.resize((size, size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2,0,1) / 255.
        return x
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i+args.batch]
        g = []; lt = []
        for r in chunk:
            im = Image.open(os.path.join(args.imgdir, r['name'])).convert('RGB')
            w, h = im.size
            g.append(prep(im, S))
            for dy in (0, 1):
                for dx in (0, 1):
                    box = (dx*w//2, dy*h//2, (dx+1)*w//2, (dy+1)*h//2)
                    lt.append(prep(im.crop(box), Sl))
        gb = torch.stack(g).to(device)
        lb = torch.stack(lt).to(device)
        fg = forward_pool(model, (gb-mean_t)/std_t, cfg['kind'])
        fl = forward_pool(model, (lb-mean_t)/std_t, cfg['kind'])
        n = len(chunk)
        fl = fl.view(n, 4, -1)
        f = torch.cat([fg, fl.mean(1), fl.amax(1)], dim=1).float().cpu().numpy()
        feats.append(f); names += [r['name'] for r in chunk]
        if (i//args.batch) % 20 == 0:
            print(f'{i}/{len(rows)}', flush=True)
    X = np.concatenate(feats, 0)
    np.savez_compressed(args.out, X=X.astype(np.float32), names=np.array(names))
    print('saved', args.out, X.shape)

if __name__ == '__main__':
    main()
