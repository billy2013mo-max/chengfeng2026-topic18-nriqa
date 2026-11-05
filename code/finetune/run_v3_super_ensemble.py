"""Launch 5 High-Resolution / Giant-Backbone V3 Models across GPUs 2, 4, 5, 6, 7
and fuse them with the 10 V2 models into a 15-Model Multi-Resolution Mega-Ensemble
with Clean-Gate Shield and P0-Recall Boosted Decoding for nriqa_dataset_test.zip.
"""
import json, os, subprocess, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/finetune')
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/decision_eval')

import comp_metrics as cm
from data import IQADataset, CLASSES, FULL
from models import build_backbone
from train_v2 import MultiNetV2, exact_probs_from_dual, decode_scaled_argmax

GRADE = {0: '无问题', 1: 'P2', 2: 'P1', 3: 'P0'}
FULL_IDX = [i for i, c in enumerate(CLASSES) if c in FULL]
BIN_IDX  = [i for i, c in enumerate(CLASSES) if c not in FULL]
OUT_DIR = 'results/v3_super'
TEST_IMG_DIR = 'data/test_150'
os.makedirs(OUT_DIR, exist_ok=True)


def step1_train_v3_models():
    print('=== Step 1: Training 5 High-Res / Giant-Backbone V3 Models on GPUs 2, 4, 5, 6, 7 ===', flush=True)
    jobs = [
        # (name, gpu, seed, backbones, size, tiles, batch, accum)
        ('v3_d3_448_t3',    2, 1, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',           448, 3, 4, 2),
        ('v3_pcl_448_t3',   4, 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4',           448, 3, 4, 2),
        ('v3_sig2g_384_t3', 5, 1, 'siglip2g:lastk4,cradio:lastk4,dinov3l:lastk4',          384, 3, 4, 2),
        ('v3_quad_448_t2',  6, 1, 'eupe:full,siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 448, 2, 4, 2),
        ('v3_pcg_384_t2',   7, 1, 'siglip2:lastk4,cradio:lastk4,pecoreg:lastk4',           384, 2, 4, 2),
    ]

    procs = {}
    t0 = time.time()
    for name, gpu, sd, bbs, sz, tiles, bs, accum in jobs:
        out_json = f'{OUT_DIR}/train_{name}.json'
        ckpt_pt  = f'{OUT_DIR}/model_{name}.pt'
        if os.path.exists(out_json) and os.path.exists(ckpt_pt):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v3.py',
            '--backbones', bbs,
            '--seed', str(sd),
            '--epochs', '4',
            '--batch', str(bs),
            '--grad-accum', str(accum),
            '--workers', '6',
            '--size', str(sz),
            '--tiles', str(tiles),
            '--clean-repeat', '4',
            '--p0-clean-weight', '4.5',
            '--lr-full', '8.875e-05',
            '--lr-last', '3e-05',
            '--lr-head', '1.5e-03',
            '--wd', '0.01',
            '--dropout', '0.1',
            '--split', 'data/split_all_full.csv',
            '--out', out_json,
            '--save-state', ckpt_pt,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        log_f = open(f'{OUT_DIR}/train_{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_json, ckpt_pt, log_f)
        print(f'[{time.time()-t0:5.0f}s] Launched {name:16s} ({bbs} @ size={sz}, tiles={tiles}) on GPU {gpu}', flush=True)

    while procs:
        time.sleep(15)
        for name in list(procs.keys()):
            p, gpu, out_json, ckpt_pt, log_f = procs[name]
            if p.poll() is not None:
                log_f.close()
                del procs[name]
                if os.path.exists(out_json):
                    d = json.load(open(out_json))
                    ts = d['test']
                    tc = d['test_cascade']
                    print(f'[{time.time()-t0:5.0f}s] DONE {name:16s} (GPU {gpu}) | '
                          f'val_scaled={ts["total"]:.2f} (S_pos={ts["S_pos"]:.3f}, S_p0={ts["S_p0"]:.3f}, '
                          f'S_clean={ts["S_clean"]:.3f}) | val_cascade={tc["total"]:.2f}', flush=True)
                else:
                    print(f'[{time.time()-t0:5.0f}s] FAIL {name} (GPU {gpu}) rc={p.returncode} - check {OUT_DIR}/train_{name}.log', flush=True)


@torch.no_grad()
def infer_v3_ckpt_tta(ckpt_path, ordered_names, dev):
    """Run 2x Horizontal Flip TTA on the 150 test images using the checkpoint's native size and tiles."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    bbs = ck['backbones']
    sz = int(ck.get('size', 384))
    tiles = int(ck.get('tiles', 2))
    alpha_ex = float(ck.get('alpha_ex', 0.45))

    specs = []
    for s in bbs.split(','):
        nm, mode = s.split(':')
        specs.append((build_backbone(nm), nm, mode))
    net = MultiNetV2([x[0] for x in specs], dropout=0.0).to(dev)
    net.load_state_dict(ck['state'])
    net.eval()

    test_rows = [{'name': nm, **{c: 0 for c in CLASSES}} for nm in ordered_names]
    ds_150 = IQADataset(test_rows, TEST_IMG_DIR, TEST_IMG_DIR, set(ordered_names), size=sz, train=False, tiles=tiles)
    dl_150 = DataLoader(ds_150, batch_size=6, shuffle=False, num_workers=6, pin_memory=True)

    P0, P1, P2, GATE = [], [], [], []
    for x, _, _ in dl_150:
        x = x.to(dev)
        x_flip = torch.flip(x, dims=[4])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lc1, le1, g1 = net(x)
            lc2, le2, g2 = net(x_flip)
        p0_1, p1_1, p2_1, _ = exact_probs_from_dual(lc1.float(), le1.float(), alpha_ex=alpha_ex)
        p0_2, p1_2, p2_2, _ = exact_probs_from_dual(lc2.float(), le2.float(), alpha_ex=alpha_ex)
        P0.append((0.5 * (p0_1 + p0_2)).cpu().numpy())
        P1.append((0.5 * (p1_1 + p1_2)).cpu().numpy())
        P2.append((0.5 * (p2_1 + p2_2)).cpu().numpy())
        GATE.append((0.5 * (torch.sigmoid(g1.float()) + torch.sigmoid(g2.float()))).cpu().numpy())

    del net, specs, ck
    torch.cuda.empty_cache()
    return np.concatenate(P0, 0), np.concatenate(P1, 0), np.concatenate(P2, 0), np.concatenate(GATE, 0)


def decode_v3_shielded(p0, p1, p2):
    """Competition-aligned decoder:
    1. Clean Shield: images where max defect cumulative probability (p0 + p1 + p2) across all 6 classes
       is < 0.28 are locked as Clean (preserving S_clean = 20.00 / 20).
    2. Class-prior-aware P2/P1/P0 thresholds:
       - Lower t0 (0.34) on hard P0 classes (过锐_P0, 过度生成_P0) to boost S_p0 = 0.6*R_p0 + 0.4*F1_p0.
       - Slightly higher t2 (0.34) on low-prior 过度生成_P2 to suppress false positives.
    """
    n = p0.shape[0]
    pred = np.zeros((n, 6), int)

    # CLASSES = ['模糊','纹理损失','过锐','花屏','过度生成','虚化突兀']
    # FULL    = ['模糊','纹理损失','过锐','过度生成']
    T0  = [0.38, 0.38, 0.34, 0.40, 0.34, 0.38]
    T1  = [0.30, 0.30, 0.30, 0.30]
    T2  = [0.30, 0.29, 0.31, 0.34]
    R12 = [0.85, 0.85, 0.85, 0.85]

    cum_any = np.zeros((n, 6))
    for ci, c in enumerate(CLASSES):
        if c in FULL:
            k = FULL.index(c)
            cum_any[:, ci] = p0[:, ci] + p1[:, k] + p2[:, k]
            c_p2 = p2[:, k] > T2[k]
            c_p1 = (p1[:, k] > T1[k]) & ((p1[:, k] * R12[k]) >= (p2[:, k] - 0.05))
            c_p0 = p0[:, ci] > T0[ci]
            pr = np.where(c_p2, 1, 0)
            pr = np.where(c_p1, 2, pr)
            pr = np.where(c_p0, 3, pr)
            pred[:, ci] = pr
        else:
            cum_any[:, ci] = p0[:, ci]
            pred[:, ci] = np.where(p0[:, ci] > T0[ci], 3, 0)

    # Clean Shield: if max cumulative defect probability across all 6 classes < 0.28, lock as Clean
    clean_shield = cum_any.max(axis=1) < 0.28
    pred[clean_shield] = 0
    return pred


def step2_mega_ensemble_predict():
    print('\n=== Step 2: Running 15-Model Multi-Resolution Mega-Ensemble TTA on Test Set ===', flush=True)
    dev = 'cuda:2' if torch.cuda.device_count() > 2 else 'cuda:0'

    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]

    # 1. Load existing 10 V2 models' TTA probabilities
    v2_probs = np.load('results/v2_super/test_150_all_models_probs.npz', allow_pickle=True)['data'].item()
    v2_fam_w = {'pcl': 1.3, 'd3': 1.3, 'eupe': 1.1, 'base': 0.8}

    # 2. Infer new V3 models with 2x flip TTA at their native resolutions (448/384) and tile grids (3x3/2x2)
    v3_ckpts = [
        ('v3_d3_448_t3',    f'{OUT_DIR}/model_v3_d3_448_t3.pt',    2.2),
        ('v3_pcl_448_t3',   f'{OUT_DIR}/model_v3_pcl_448_t3.pt',   2.2),
        ('v3_sig2g_384_t3', f'{OUT_DIR}/model_v3_sig2g_384_t3.pt', 2.0),
        ('v3_quad_448_t2',  f'{OUT_DIR}/model_v3_quad_448_t2.pt',  2.2),
        ('v3_pcg_384_t2',   f'{OUT_DIR}/model_v3_pcg_384_t2.pt',   2.0),
    ]

    cache_v3_npz = f'{OUT_DIR}/test_150_v3_models_probs.npz'
    v3_probs = {}
    if os.path.exists(cache_v3_npz):
        v3_probs.update(np.load(cache_v3_npz, allow_pickle=True)['data'].item())

    for name, ckp, w in v3_ckpts:
        if not os.path.exists(ckp):
            print(f'[warn] Missing V3 checkpoint {ckp}, skipping.', flush=True)
            continue
        if name in v3_probs:
            print(f'[cached] {name:16s} V3 TTA probabilities loaded.', flush=True)
            continue
        t0 = time.time()
        p0_i, p1_i, p2_i, g_i = infer_v3_ckpt_tta(ckp, ordered_names, dev)
        v3_probs[name] = (p0_i, p1_i, p2_i, g_i, w)
        np.savez_compressed(cache_v3_npz, data=v3_probs)
        print(f'[{time.time()-t0:5.1f}s] Inferred {name:16s} (2x flip TTA on 150 test images)', flush=True)

    # 3. Combine all 10 V2 models + 5 V3 models
    p0_ens, p1_ens, p2_ens, tw_sum = 0.0, 0.0, 0.0, 0.0
    used = []
    for name, (p0_i, p1_i, p2_i, fam, is_iso) in v2_probs.items():
        wi = v2_fam_w.get(fam, 1.0) * (0.70 if is_iso else 1.0)
        p0_ens = p0_ens + wi * p0_i
        p1_ens = p1_ens + wi * p1_i
        p2_ens = p2_ens + wi * p2_i
        tw_sum += wi
        used.append((name, round(wi, 2)))

    for name, (p0_i, p1_i, p2_i, g_i, wi) in v3_probs.items():
        p0_ens = p0_ens + wi * p0_i
        p1_ens = p1_ens + wi * p1_i
        p2_ens = p2_ens + wi * p2_i
        tw_sum += wi
        used.append((name, round(wi, 2)))

    p0_ens /= tw_sum; p1_ens /= tw_sum; p2_ens /= tw_sum
    print(f'\nMega-Ensemble combined {len(used)} models: {used}', flush=True)

    pred_150 = decode_v3_shielded(p0_ens, p1_ens, p2_ens)

    out_list = []
    for i, nm in enumerate(ordered_names):
        lab = {k: GRADE[int(pred_150[i, CLASSES.index(k)])] for k in ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']}
        out_list.append({'image': nm, 'label': lab})

    out_v3_json   = f'{OUT_DIR}/pred_nriqa_dataset_test.json'
    out_root_json = 'pred_nriqa_dataset_test.json'
    json.dump(out_list, open(out_v3_json, 'w'), ensure_ascii=False, indent=2)
    json.dump(out_list, open(out_root_json, 'w'), ensure_ascii=False, indent=2)

    print('\nPer-class grade distribution on nriqa_dataset_test.zip (150 images):', flush=True)
    for ci, c in enumerate(CLASSES):
        vals = [GRADE[int(v)] for v in pred_150[:, ci]]
        uniq = {v: vals.count(v) for v in ['无问题', 'P2', 'P1', 'P0'] if v in vals}
        print(f'  {c:6s}: {uniq}', flush=True)
    clean_cnt = int((pred_150 == 0).all(1).sum())
    print(f'All-clean images: {clean_cnt}/{len(pred_150)} ({100*clean_cnt/len(pred_150):.1f}%)', flush=True)

    if os.path.exists('results/v2_super/pred_nriqa_dataset_test.json'):
        prev = json.load(open('results/v2_super/pred_nriqa_dataset_test.json'))
        same_cells = sum(1 for a, b in zip(prev, out_list) for k in CLASSES if a['label'][k] == b['label'][k])
        diff_imgs = sum(1 for a, b in zip(prev, out_list) if any(a['label'][k] != b['label'][k] for k in CLASSES))
        print(f'Agreement vs V2 Super-Ensemble (51.46): {same_cells}/900 cells '
              f'({100*same_cells/900:.1f}%), {diff_imgs}/150 images refined.', flush=True)

    print(f'Saved V3 Mega-Ensemble predictions to: {out_root_json} and {out_v3_json}', flush=True)


if __name__ == '__main__':
    step1_train_v3_models()
    step2_mega_ensemble_predict()
