"""8-GPU Parallel High-Resolution 448px + 1:1 Native-Pixel Edge/Texture Microscope + Spatial Anomaly Token
Feature Extractor (`extract_v15_microscope_cache.py`).

Extracts for all 13,643 images (3,573 real + 150 test_150 + 8,100 s16..s20 authentic synth + 1,820 full synth):
- Backbones (448x448 with Spatial Anomaly Patch Token Pooling `[CLS, patch_mean, patch_std, patch_top32]`):
  1. `dinov3l`  (4096D)
  2. `eva02`    (3072D)
  3. `convnext` (3072D)
  4. `siglip2`  (3456D)
  5. `pecoreg`  (4608D)
  Total per view = 18,304D.
- Views per image (4 views at 448x448):
  * View 0: Global 448x448 view
  * View 1: Top-1 Peak Laplacian/Sobel High-Frequency Edge 1:1 Native-Pixel 448x448 Crop (zero resize!)
  * View 2: Top-2 Second High-Frequency Edge 1:1 Native-Pixel 448x448 Crop (zero resize!)
  * View 3: Median-Gradient Subject Texture 1:1 Native-Pixel 448x448 Crop (zero resize!)
- Saved views per image (stored in `/dev/shm/v15_microscope_cache/` in float16):
  * `glob` (View 0)
  * `edge_max` (`elementwise_max(View 1, View 2)`)
  * `edge_mean` (`0.5 * (View 1 + View 2)`)
  * `tex` (View 3)
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code/finetune')
from train_v9_specialist import build_v9_backbone, CACHE_OK

CACHE_DIR = '/dev/shm/v15_microscope_cache'
os.makedirs(CACHE_DIR, exist_ok=True)
BACKBONES = ['dinov3l', 'eva02', 'convnext', 'siglip2', 'pecoreg']


def collect_all_image_names():
    names = []
    with open('data/manifest_combined.csv') as f:
        names.extend([r['name'] for r in csv.DictReader(f)])
    if os.path.exists('results/v2_full/template_150.json'):
        tpl = json.load(open('results/v2_full/template_150.json'))
        names.extend([e['image'] for e in tpl])
    for m_csv in sorted(glob.glob('data/v9_ladders/manifest_s*_v9.csv') + glob.glob('data/v9_fulldata/manifest_f*_v9.csv')):
        with open(m_csv) as f:
            names.extend([r['name'] for r in csv.DictReader(f)])
    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


class MicroscopeExtractDataset(Dataset):
    def __init__(self, names, size=448):
        self.names = names
        self.size = size
        if CACHE_OK:
            d = np.load('/dev/shm/race18_native/index.npz', allow_pickle=True)
            c_names = list(d['names'])
            self.offsets = d['offsets']
            self.h = d['h']
            self.w = d['w']
            self.pos = {n: i for i, n in enumerate(c_names)}
            self.data = np.memmap('/dev/shm/race18_native/cache.bin', dtype=np.uint8, mode='r')
        else:
            self.pos = {}

    def __len__(self):
        return len(self.names)

    def _load(self, name):
        if name in self.pos:
            j = self.pos[name]
            h = int(self.h[j])
            w = int(self.w[j])
            off = int(self.offsets[j])
            arr = np.asarray(self.data[off:off + h * w * 3]).reshape(h, w, 3)
            return Image.fromarray(arr)
        for d in ['/tmp/v9_authentic_imgs', 'data/images', 'data/images_supp', 'data/train', 'data/test_150', 'data/test']:
            p = os.path.join('/wanqing-develop/luowenjing/18', d, name) if not d.startswith('/') else os.path.join(d, name)
            if os.path.exists(p):
                return Image.open(p).convert('RGB')
        raise FileNotFoundError(f'Cannot find {name}')

    def _extract_native_microscope_crops(self, im):
        S = self.size
        w, h = im.size
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
                lx = np.abs(2.0 * arr[:, 1:-1] - arr[:, :-2] - arr[:, 2:]).mean()
                ly = np.abs(2.0 * arr[1:-1, :] - arr[:-2, :] - arr[2:, :]).mean()
                cands.append(patch)
                energies.append(float(gx + gy + 0.8 * (lx + ly)))
        if len(cands) == 1:
            return [cands[0], cands[0], cands[0]]
        order = np.argsort(energies)
        top1 = cands[int(order[-1])]
        top2 = cands[int(order[-2])]
        med = cands[int(order[len(order) // 2])]
        return [top1, top2, med]

    def __getitem__(self, idx):
        name = self.names[idx]
        im = self._load(name)
        S = self.size
        v_glob = im.resize((S, S), Image.BICUBIC)
        v_top1, v_top2, v_med = self._extract_native_microscope_crops(im)
        views = []
        for v in (v_glob, v_top1, v_top2, v_med):
            a = np.asarray(v, dtype=np.uint8).transpose(2, 0, 1).copy()
            views.append(torch.from_numpy(a))
        return idx, torch.stack(views, dim=0)  # (4, 3, S, S) uint8


def normalize_subblocks(feat, d_emb):
    """L2-normalize each [CLS/mean/std/top32] sub-block of width `d_emb`."""
    n_blocks = feat.shape[1] // d_emb
    parts = []
    for b in range(n_blocks):
        sub = feat[:, b * d_emb:(b + 1) * d_emb]
        parts.append(F.normalize(sub.float(), dim=-1))
    return torch.cat(parts, dim=-1)


def run_worker(rank, world_size):
    dev = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(dev)
    all_names = collect_all_image_names()
    my_names = all_names[rank::world_size]
    N = len(my_names)
    print(f'[GPU {rank}] Starting extraction for {N} / {len(all_names)} images...', flush=True)

    ds = MicroscopeExtractDataset(my_names, size=448)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    # Pre-extract all batches into CPU/GPU or iterate per backbone
    for bname in BACKBONES:
        out_path = os.path.join(CACHE_DIR, f'{bname}_part{rank}.npz')
        if os.path.exists(out_path):
            print(f'[GPU {rank}] {bname} already exists at {out_path}, skipping.', flush=True)
            continue
        t0 = time.time()
        m, fwd, mean, std, _, out_dim = build_v9_backbone(bname)
        m = m.to(dev).eval().to(torch.bfloat16)
        d_emb = 1536 if bname == 'pecoreg' else (1152 if bname == 'siglip2' else (1024 if bname in ('dinov3l', 'convnext') else 768))
        mean_t = torch.tensor(mean, device=dev, dtype=torch.bfloat16).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=dev, dtype=torch.bfloat16).view(1, 3, 1, 1)

        f_glob = np.zeros((N, out_dim), dtype=np.float16)
        f_emax = np.zeros((N, out_dim), dtype=np.float16)
        f_emean = np.zeros((N, out_dim), dtype=np.float16)
        f_tex = np.zeros((N, out_dim), dtype=np.float16)

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for idxs, x_u8 in dl:
                B = x_u8.shape[0]
                x = x_u8.to(dev, non_blocking=True).view(B * 4, 3, 448, 448).to(torch.bfloat16) / 255.0
                if bname == 'siglip2':
                    x = F.interpolate(x, size=(384, 384), mode='bicubic', align_corners=False)
                x = (x - mean_t) / std_t
                out = fwd(x)  # (B*4, out_dim)
                out = normalize_subblocks(out, d_emb).view(B, 4, out_dim)
                idx_np = idxs.numpy()
                f_glob[idx_np] = out[:, 0].cpu().numpy().astype(np.float16)
                f_emax[idx_np] = torch.maximum(out[:, 1], out[:, 2]).cpu().numpy().astype(np.float16)
                f_emean[idx_np] = (0.5 * (out[:, 1] + out[:, 2])).cpu().numpy().astype(np.float16)
                f_tex[idx_np] = out[:, 3].cpu().numpy().astype(np.float16)

        np.savez(out_path, names=np.array(my_names), glob=f_glob, emax=f_emax, emean=f_emean, tex=f_tex)
        print(f'[GPU {rank}] Finished {bname} (dim={out_dim}) for {N} imgs in {time.time()-t0:.1f}s', flush=True)
        del m
        torch.cuda.empty_cache()


def merge_parts(world_size=8):
    all_names = collect_all_image_names()
    N = len(all_names)
    np.save(os.path.join(CACHE_DIR, 'names.npy'), np.array(all_names))
    for bname in BACKBONES:
        merged_path = os.path.join(CACHE_DIR, f'{bname}_merged.npz')
        if os.path.exists(merged_path):
            continue
        t0 = time.time()
        p0 = np.load(os.path.join(CACHE_DIR, f'{bname}_part0.npz'), allow_pickle=True)
        out_dim = p0['glob'].shape[1]
        glob_all = np.zeros((N, out_dim), dtype=np.float16)
        emax_all = np.zeros((N, out_dim), dtype=np.float16)
        emean_all = np.zeros((N, out_dim), dtype=np.float16)
        tex_all = np.zeros((N, out_dim), dtype=np.float16)
        for r in range(world_size):
            pr = np.load(os.path.join(CACHE_DIR, f'{bname}_part{r}.npz'), allow_pickle=True)
            glob_all[r::world_size] = pr['glob']
            emax_all[r::world_size] = pr['emax']
            emean_all[r::world_size] = pr['emean']
            tex_all[r::world_size] = pr['tex']
        np.savez(merged_path, glob=glob_all, emax=emax_all, emean=emean_all, tex=tex_all)
        for r in range(world_size):
            try:
                os.remove(os.path.join(CACHE_DIR, f'{bname}_part{r}.npz'))
            except OSError:
                pass
        print(f'[MERGE] Merged {bname} shape=({N}, {out_dim}) in {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, default=-1)
    ap.add_argument('--world-size', type=int, default=8)
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    if args.merge:
        merge_parts(args.world_size)
    elif args.rank >= 0:
        run_worker(args.rank, args.world_size)
    else:
        import subprocess
        t_start = time.time()
        # First update 60D native physical features in parallel on CPU while GPUs run!
        p_phys = subprocess.Popen([sys.executable, 'code/finetune/extract_native_phys_feats.py'])
        procs = [
            subprocess.Popen([sys.executable, __file__, '--rank', str(r), '--world-size', str(args.world_size)])
            for r in range(args.world_size)
        ]
        for p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f'Worker exited with {rc}')
        p_phys.wait()
        merge_parts(args.world_size)
        print(f'=== ALL 8 GPUS + 60D PHYS EXTRACTION COMPLETE IN {time.time()-t_start:.1f}s ===', flush=True)
