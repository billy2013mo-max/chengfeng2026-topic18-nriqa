"""Companion booster for V3 Mega-Ensemble:
1. Watches the 5 V3 training jobs in results/v3_super/.
2. As soon as early-finishing GPUs (GPU 6, GPU 5, GPU 7) complete their 4-epoch run
   (10-17 minutes ahead of GPU 4), utilizes their idle window to:
   - Train a dedicated 3-epoch (--epochs 3) high-P0-recall V3 model on GPU 6 (saving exact ep3 weights).
   - Run a fast 100-step P0-recall un-suppression pass (clean_repeat=1, p0_clean_weight=1.8, lr=1.2e-5)
     from each finished V3 checkpoint to recover the Epoch-3 P0 sharpness that clean_repeat=4
     suppressed in Epoch 4, saving both `model_<name>_ep3_cali.pt` and the SWA Checkpoint Soup
     `model_<name>_swa.pt` (0.5 * ep4 + 0.5 * ep3_cali).
3. Once task-410 finishes, infers 2x Flip TTA on all Epoch-3 / SWA checkpoints, fuses all 21 models
   (10 V2 + 5 V3_ep4 + 5 V3_swa/ep3 + 1 V3_iso_ep3), and updates pred_nriqa_dataset_test.json.
"""
import copy, csv, json, os, subprocess, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/finetune')
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/decision_eval')

from data import IQADataset, CachedIQADataset, load_split, read_manifest, targets_from_grades, CLASSES, FULL
from models import build_backbone
from train_v2 import MultiNetV2, exact_probs_from_dual, CACHE_OK
from train_v3 import v3_loss_fn
from run_v3_super_ensemble import infer_v3_ckpt_tta, decode_v3_shielded, GRADE, OUT_DIR, TEST_IMG_DIR


def make_unsuppressed_ep3_and_swa(ckpt_path, dev='cuda:0', steps=100):
    """Run a 100-step un-suppressed P0-recall recovery pass (clean_repeat=1, p0_clean_weight=1.8)
    starting from the converged Epoch-4 checkpoint to restore Epoch-3 P0 sensitivity, and save
    both `_ep3_cali.pt` and `_swa.pt` (50/50 weight average with Epoch-4).
    """
    cali_path = ckpt_path.replace('.pt', '_ep3_cali.pt')
    swa_path  = ckpt_path.replace('.pt', '_swa.pt')
    if os.path.exists(cali_path) and os.path.exists(swa_path):
        return cali_path, swa_path

    t0 = time.time()
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ep4_state = copy.deepcopy(ck['state'])
    bbs = ck['backbones']
    sz = int(ck.get('size', 384))
    tiles = int(ck.get('tiles', 2))

    specs = []
    for s in bbs.split(','):
        nm, mode = s.split(':')
        specs.append((build_backbone(nm), nm, mode))
    net = MultiNetV2([x[0] for x in specs], dropout=0.1).to(dev)
    net.load_state_dict(ep4_state)

    # Freeze backbones, only gently adapt tile_attn + norm + ordinal/exact heads with un-suppressed P0 loss
    for p in net.parameters():
        p.requires_grad = False
    tr_p = []
    for mod in [net.tile_attn, net.norm, net.head_cum, net.head_ex, net.gate]:
        for p in mod.parameters():
            p.requires_grad = True
            tr_p.append(p)

    rows = load_split('data/split_all_full.csv')
    man = read_manifest('data/manifest_combined.csv')
    for r in rows:
        r.update({k: man[r['name']][k] for k in CLASSES})
    tr = [r for r in rows if r['split'] == 'train']
    orig = set(r['name'] for r in csv.DictReader(open('data/manifest.csv')))
    dtr = (CachedIQADataset(tr, sz, True, tiles=tiles)
           if CACHE_OK else IQADataset(tr, 'data/images', 'data/images_supp', orig, sz, True, tiles=tiles))
    ltr = DataLoader(dtr, batch_size=6, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    Y = targets_from_grades(torch.tensor([[int(r[c]) for c in CLASSES] for r in tr]))
    pw = ((len(Y) - Y.sum(0)) / Y.sum(0).clamp(min=1)).clamp(1.0, 8.0).to(dev)
    bce_loss_none = torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')

    opt = torch.optim.AdamW(tr_p, lr=1.5e-4, weight_decay=1e-2)
    net.train()
    step = 0
    for x, g, _ in ltr:
        x = x.to(dev)
        g = g.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lc, le, gl = net(x)
        # Un-suppressed P0 clean weight (1.8 instead of 4.5) to restore Epoch-3 P0 recall
        loss = v3_loss_fn(lc.float(), le.float(), gl.float(), g, bce_loss_none, p0_clean_weight=1.8)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tr_p, 1.0)
        opt.step()
        step += 1
        if step >= steps:
            break

    cali_state = {k: v.cpu() for k, v in net.state_dict().items()}
    swa_state = {}
    for k in ep4_state.keys():
        if ep4_state[k].is_floating_point():
            swa_state[k] = 0.55 * ep4_state[k].cpu() + 0.45 * cali_state[k]
        else:
            swa_state[k] = ep4_state[k].cpu()

    ck_cali = dict(ck, state=cali_state)
    ck_swa  = dict(ck, state=swa_state)
    torch.save(ck_cali, cali_path)
    torch.save(ck_swa, swa_path)
    del net, specs, ck, ck_cali, ck_swa
    torch.cuda.empty_cache()
    print(f'[{time.time()-t0:5.1f}s] Saved Epoch-3-style un-suppressed & SWA checkpoints: {cali_path}, {swa_path}', flush=True)
    return cali_path, swa_path


def main():
    print('=== Monitoring V3 Training Jobs for Epoch-3 & SWA Checkpoint Generation ===', flush=True)
    v3_names = [
        ('v3_quad_448_t2',  6),
        ('v3_sig2g_384_t3', 5),
        ('v3_pcg_384_t2',   7),
        ('v3_d3_448_t3',    2),
        ('v3_pcl_448_t3',   4),
    ]

    launched_ep3_iso = False
    ep3_iso_proc = None
    ep3_iso_out  = f'{OUT_DIR}/train_v3_d3_448_t2_ep3.json'
    ep3_iso_ckpt = f'{OUT_DIR}/model_v3_d3_448_t2_ep3.pt'

    processed_swa = set()
    t_start = time.time()

    while len(processed_swa) < len(v3_names):
        for name, gpu in v3_names:
            if name in processed_swa:
                continue
            out_json = f'{OUT_DIR}/train_{name}.json'
            ckpt_pt  = f'{OUT_DIR}/model_{name}.pt'
            if os.path.exists(out_json) and os.path.exists(ckpt_pt):
                print(f'[{time.time()-t_start:5.0f}s] Detected finished {name} on GPU {gpu}! Building SWA + Ep3-Unsuppressed...', flush=True)
                make_unsuppressed_ep3_and_swa(ckpt_pt, dev=f'cuda:{gpu}', steps=100)
                processed_swa.add(name)

                # If GPU 6 finishes early (v3_quad_448_t2 finishes ~15 min before GPU 4),
                # launch a dedicated 3-epoch (--epochs 3) high-res V3 model on GPU 6!
                if name == 'v3_quad_448_t2' and not launched_ep3_iso and not os.path.exists(ep3_iso_ckpt):
                    launched_ep3_iso = True
                    cmd = [
                        'python3', '-u', '-I', 'code/finetune/train_v3.py',
                        '--backbones', 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4',
                        '--seed', '2',
                        '--epochs', '3',
                        '--batch', '4',
                        '--grad-accum', '2',
                        '--workers', '6',
                        '--size', '448',
                        '--tiles', '2',
                        '--clean-repeat', '3',
                        '--p0-clean-weight', '3.5',
                        '--lr-full', '8.875e-05',
                        '--lr-last', '3e-05',
                        '--lr-head', '1.5e-03',
                        '--wd', '0.01',
                        '--dropout', '0.1',
                        '--split', 'data/split_iso_full.csv',
                        '--out', ep3_iso_out,
                        '--save-state', ep3_iso_ckpt,
                    ]
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = '6'
                    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                    log_f = open(f'{OUT_DIR}/train_v3_d3_448_t2_ep3.log', 'w')
                    ep3_iso_proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
                    print(f'[{time.time()-t_start:5.0f}s] Launched dedicated 3-Epoch V3 model (v3_d3_448_t2_ep3) on early-freed GPU 6!', flush=True)

        time.sleep(15)

    if ep3_iso_proc is not None:
        print('Waiting for dedicated 3-Epoch V3 model on GPU 6 to finish...', flush=True)
        ep3_iso_proc.wait()
        print('Dedicated 3-Epoch V3 model finished!', flush=True)

    # Wait until task-410 completely finishes step2_mega_ensemble_predict()
    cache_v3_npz = f'{OUT_DIR}/test_150_v3_models_probs.npz'
    v3_pred_json = f'{OUT_DIR}/pred_nriqa_dataset_test.json'
    print('Waiting for task-410 step2 (initial 15-model Ep4 ensemble JSON) to finish...', flush=True)
    for _ in range(180):
        if os.path.exists(v3_pred_json) and os.path.exists(cache_v3_npz):
            if (time.time() - os.path.getmtime(v3_pred_json)) > 3.0:
                break
        time.sleep(5)

    #Backup the 15-model Ep4-only prediction before upgrading to 21-model Ep3+SWA+Ep4 ensemble
    if os.path.exists(f'{OUT_DIR}/pred_nriqa_dataset_test.json'):
        subprocess.run(['cp', f'{OUT_DIR}/pred_nriqa_dataset_test.json', f'{OUT_DIR}/pred_nriqa_dataset_test_15m_ep4.json'])

    print('\n=== Running TTA Inference on SWA & Dedicated Epoch-3 V3 Checkpoints ===', flush=True)
    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]

    v2_probs = np.load('results/v2_super/test_150_all_models_probs.npz', allow_pickle=True)['data'].item()
    v2_fam_w = {'pcl': 1.3, 'd3': 1.3, 'eupe': 1.1, 'base': 0.8}

    v3_probs = np.load(cache_v3_npz, allow_pickle=True)['data'].item()

    extra_ckpts = [
        ('v3_d3_448_t3_swa',    f'{OUT_DIR}/model_v3_d3_448_t3_swa.pt',    1.6, 'cuda:2'),
        ('v3_pcl_448_t3_swa',   f'{OUT_DIR}/model_v3_pcl_448_t3_swa.pt',   1.6, 'cuda:4'),
        ('v3_sig2g_384_t3_swa', f'{OUT_DIR}/model_v3_sig2g_384_t3_swa.pt', 1.5, 'cuda:5'),
        ('v3_quad_448_t2_swa',  f'{OUT_DIR}/model_v3_quad_448_t2_swa.pt',  1.6, 'cuda:6'),
        ('v3_pcg_384_t2_swa',   f'{OUT_DIR}/model_v3_pcg_384_t2_swa.pt',   1.5, 'cuda:7'),
        ('v3_d3_448_t2_ep3',    ep3_iso_ckpt,                              2.0, 'cuda:6'),
    ]

    for name, ckp, w, dev in extra_ckpts:
        if not os.path.exists(ckp):
            continue
        if name in v3_probs:
            continue
        t0 = time.time()
        p0_i, p1_i, p2_i, g_i = infer_v3_ckpt_tta(ckp, ordered_names, dev)
        v3_probs[name] = (p0_i, p1_i, p2_i, g_i, w)
        np.savez_compressed(cache_v3_npz, data=v3_probs)
        print(f'[{time.time()-t0:5.1f}s] Inferred {name:20s} (2x flip TTA)', flush=True)

    # Fuse all models (10 V2 + 5 V3_ep4 + 5 V3_swa + 1 V3_ep3 = 21 models)
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
    print(f'\nFinal Ep3+SWA+Ep4 Mega-Ensemble combined {len(used)} models: {used}', flush=True)

    pred_150 = decode_v3_shielded(p0_ens, p1_ens, p2_ens)

    out_list = []
    for i, nm in enumerate(ordered_names):
        lab = {k: GRADE[int(pred_150[i, CLASSES.index(k)])] for k in ['模糊', '纹理损失', '过度生成', '虚化突兀', '过锐', '花屏']}
        out_list.append({'image': nm, 'label': lab})

    out_v3_json   = f'{OUT_DIR}/pred_nriqa_dataset_test.json'
    out_root_json = 'pred_nriqa_dataset_test.json'
    json.dump(out_list, open(out_v3_json, 'w'), ensure_ascii=False, indent=2)
    json.dump(out_list, open(out_root_json, 'w'), ensure_ascii=False, indent=2)

    print('\nFinal Ep3+SWA+Ep4 Per-class grade distribution on nriqa_dataset_test.zip (150 images):', flush=True)
    for ci, c in enumerate(CLASSES):
        vals = [GRADE[int(v)] for v in pred_150[:, ci]]
        uniq = {v: vals.count(v) for v in ['无问题', 'P2', 'P1', 'P0'] if v in vals}
        print(f'  {c:6s}: {uniq}', flush=True)
    clean_cnt = int((pred_150 == 0).all(1).sum())
    print(f'All-clean images: {clean_cnt}/{len(pred_150)} ({100*clean_cnt/len(pred_150):.1f}%)', flush=True)
    print(f'Successfully updated {out_root_json} and {out_v3_json}!', flush=True)


if __name__ == '__main__':
    main()
