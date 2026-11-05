import os, sys, torch, torch.nn as nn
sys.path.insert(0, '/tmp/race18')
# 2026-10-28: fixed InternViT-6B finetuning: (1) pretrained pos-embed was 448-only so 384 inputs
# crashed (730 vs 1025 tokens) -> bicubic-resize pos-embed to the pipeline size; (2) encoder hard-codes
# gradient_checkpointing=True, which silently zeroes grads for last-k finetuning (frozen embeddings) ->
# disable checkpointing unless embeddings are trainable; all params/buffers kept in bf16.


def _enable_pos_interp(m):
    """Allow SigLIP vision tower to take resolutions other than its native grid."""
    emb = m.embeddings
    orig = emb.forward
    def fwd(pixel_values, **kw):
        return orig(pixel_values, interpolate_pos_encoding=True)
    emb.forward = fwd
    return m

def build_backbone(name, device='cuda'):
    """Returns (module, dim, mean, std, block_list_accessor, dtype)."""
    if name == 'siglip2':
        from transformers import SiglipVisionModel, AutoImageProcessor
        repo = 'google/siglip2-so400m-patch14-384'
        try:
            m = SiglipVisionModel.from_pretrained(repo, attn_implementation='sdpa', local_files_only=True)
            proc = AutoImageProcessor.from_pretrained(repo, local_files_only=True)
        except Exception:
            m = SiglipVisionModel.from_pretrained(repo, attn_implementation='sdpa')
            proc = AutoImageProcessor.from_pretrained(repo)
        mean, std = proc.image_mean, proc.image_std
        _enable_pos_interp(m)
        def fwd(x): return m(pixel_values=x).last_hidden_state.mean(1)
        return m, fwd, mean, std, m.encoder.layers, 1152
    if name == 'siglip2g':
        from transformers import SiglipVisionModel, AutoImageProcessor
        repo = 'models/siglip2-giant'
        m = SiglipVisionModel.from_pretrained(repo, attn_implementation='sdpa')
        proc = AutoImageProcessor.from_pretrained(repo)
        mean, std = proc.image_mean, proc.image_std
        _enable_pos_interp(m)
        def fwd(x): return m(pixel_values=x).last_hidden_state.mean(1)
        return m, fwd, mean, std, m.encoder.layers, 1536
    if name == 'eupe':
        m = torch.hub.load('/tmp/race18/eupe_repo', 'eupe_vitb16', source='local',
                           weights='models/eupe-vit-b/EUPE-ViT-B.pt')
        def fwd(x):
            o = m.forward_features(x)
            return torch.cat([o['x_norm_clstoken'], o['x_norm_patchtokens'].mean(1)], 1)
        return m, fwd, (0.485,0.456,0.406), (0.229,0.224,0.225), m.blocks, 1536
    if name == 'convnext':
        import timm
        m = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0)
        dc = timm.data.resolve_data_config({}, model=m)
        def fwd(x):
            y = m.forward_features(x)
            return y.mean((2,3)) if y.ndim == 4 else y
        return m, fwd, dc['mean'], dc['std'], m.stages, 1024
    if name == 'internvit':
        import glob
        from safetensors.torch import load_file
        from internvit_pkg.configuration_intern_vit import InternVisionConfig
        from internvit_pkg.modeling_intern_vit import InternVisionModel
        cfg = InternVisionConfig.from_pretrained('models/internvit-6b')
        m = InternVisionModel(cfg)
        sd = {}
        for f in sorted(glob.glob('models/internvit-6b/model-*.safetensors')): sd.update(load_file(f))
        m.load_state_dict(sd, strict=False)
        del sd
        # pos-embed is frozen at 448 (32x32+1); resize once at build time, before the optimizer
        # is created, so the optimizer keeps a reference to the same Parameter object.
        _iv_size = int(os.environ.get('INTERNVIT_SIZE', '384'))
        if _iv_size != cfg.image_size:
            m.resize_pos_embeddings(cfg.image_size, _iv_size, cfg.patch_size)
        _iv_grid = [_iv_size]
        m = m.to(torch.bfloat16)
        def _fit_pos(s):
            # in-place pos-embed interpolation for a non-default input size (keeps param identity)
            if s == _iv_grid[0]: return
            pe = m.embeddings.position_embedding
            p = cfg.patch_size
            with torch.no_grad():
                cls = pe[:, :1, :]
                rest = pe[:, 1:, :].reshape(1, _iv_grid[0]//p, _iv_grid[0]//p, -1).permute(0,3,1,2)
                rest = torch.nn.functional.interpolate(rest.float(), size=(s//p, s//p), mode='bicubic', align_corners=False)
                rest = rest.to(pe.dtype).reshape(1, pe.shape[-1], -1).permute(0,2,1)
                pe.data = torch.cat([cls, rest], 1).contiguous()
            _iv_grid[0] = s
        def fwd(x):
            # frozen embeddings => gradient checkpointing severs grad flow (last-k grads all None)
            m.encoder.gradient_checkpointing = bool(m.embeddings.patch_embedding.weight.requires_grad)
            _fit_pos(x.shape[-1])
            t = m(pixel_values=x.to(torch.bfloat16)).last_hidden_state
            return torch.cat([t[:,0], t[:,1:].mean(1)], 1)
        return m, fwd, (0.485,0.456,0.406), (0.229,0.224,0.225), m.encoder.layers, 6400
    if name == 'cradio':
        from transformers import AutoModel
        m = AutoModel.from_pretrained('models/c-radio-v4-h', trust_remote_code=True)
        def fwd(x):
            summ, feat = m(x)
            return torch.cat([summ, feat.mean(1)], 1)
        return m, fwd, (0,0,0), (1,1,1), m.radio_model.model.blocks, 2560+1280
    if name in ('pecorel','pecoreg'):
        # PE-Core (Meta Perception Encoder). patch14 needs a divisible input; the shared
        # pipeline runs at 384, so resize to the native 448 here (linear op, commutes with norm).
        import timm
        from safetensors.torch import load_file
        arch,path,size={'pecorel':('vit_pe_core_large_patch14_336','models/pe-core-l14/model.safetensors',448),
                        'pecoreg':('vit_pe_core_gigantic_patch14_448','models/pe-core-g14/model.safetensors',448)}[name]
        m=timm.create_model(arch,pretrained=False,num_classes=0,dynamic_img_size=True)
        m.load_state_dict(load_file(path),strict=False)   # only the classification head differs
        dc=timm.data.resolve_data_config({},model=m)
        def fwd(x):
            if x.shape[-1]!=size:
                x=torch.nn.functional.interpolate(x,size=(size,size),mode='bicubic',align_corners=False)
            t=m.forward_features(x)
            if m.num_prefix_tokens:
                return torch.cat([t[:,0],t[:,m.num_prefix_tokens:].mean(1)],1)
            return t.mean(1)
        d=2*m.embed_dim if m.num_prefix_tokens else m.embed_dim
        return m, fwd, dc['mean'], dc['std'], m.blocks, d
    if name == 'dinov3l':
        import timm
        from safetensors.torch import load_file
        m=timm.create_model('vit_large_patch16_dinov3',pretrained=False,num_classes=0,dynamic_img_size=True)
        m.load_state_dict(load_file('models/dinov3-vitl16/model.safetensors'),strict=True)
        dc=timm.data.resolve_data_config({},model=m)
        def fwd(x):
            t=m.forward_features(x)      # CLS + 4 register tokens + patch tokens
            return torch.cat([t[:,0],t[:,m.num_prefix_tokens:].mean(1)],1)
        return m, fwd, dc['mean'], dc['std'], m.blocks, 2*m.embed_dim
    raise ValueError(name)

class Net(nn.Module):
    def __init__(self, m, fwd, dim, head_hidden=0, dropout=0.2):
        super().__init__()
        self.m = m; self.fwd = fwd
        d = dim * 3
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, head_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, 14)) \
            if head_hidden else nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 14))
    def forward(self, x):
        B, V, C, H, W = x.shape
        f = self.fwd(x.view(B*V, C, H, W)).view(B, V, -1)
        f = torch.cat([f[:,0], f[:,1:].mean(1), f[:,1:].amax(1)], 1)
        return self.head(f)

def set_freeze(m, mode, k=4):
    for p in m.parameters(): p.requires_grad = (mode == 'full')
    if mode == 'lastk':
        # unfreeze last k blocks if accessible
        for name, mod in m.named_modules():
            pass
    return

def lora_wrap(m, r=32):
    from peft import LoraConfig, get_peft_model
    tgt = ['q_proj','k_proj','v_proj','out_proj','fc1','fc2']
    cfg = LoraConfig(r=r, lora_alpha=r*2, lora_dropout=0.05, bias='none', target_modules=tgt)
    return get_peft_model(m, cfg)
