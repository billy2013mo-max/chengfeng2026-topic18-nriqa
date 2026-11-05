"""Train an 8-GPU heterogeneous multi-backbone V2 Super-Ensemble on the full dataset
and generate the strongest test-set predictions for nriqa_dataset_test.zip.

Models combined (11 V2 models across 5 heterogeneous backbone families):
  Already trained in results/v2_full/:
    1. base_all_s1 : siglip2:lastk4,cradio:lastk4 (split_all_full.csv, seed=1)
    2. base_all_s2 : siglip2:lastk4,cradio:lastk4 (split_all_full.csv, seed=2)
    3. base_iso_s1 : siglip2:lastk4,cradio:lastk4 (split_iso_full.csv, seed=1)
  Newly trained in parallel across GPUs 0..7 in results/v2_super/:
    4. pcl_all_s1  (GPU 0): siglip2:lastk4,cradio:lastk4,pecorel:lastk4 (split_all_full.csv, seed=1)
    5. pcl_all_s2  (GPU 1): siglip2:lastk4,cradio:lastk4,pecorel:lastk4 (split_all_full.csv, seed=2)
    6. pcl_iso_s1  (GPU 2): siglip2:lastk4,cradio:lastk4,pecorel:lastk4 (split_iso_full.csv, seed=1)
    7. eupe_all_s1 (GPU 3): eupe:full,siglip2:lastk4,cradio:lastk4      (split_all_full.csv, seed=1)
    8. eupe_iso_s1 (GPU 4): eupe:full,siglip2:lastk4,cradio:lastk4      (split_iso_full.csv, seed=1)
    9. d3_all_s1   (GPU 5): siglip2:lastk4,cradio:lastk4,dinov3l:lastk4 (split_all_full.csv, seed=1)
   10. d3_iso_s1   (GPU 6): siglip2:lastk4,cradio:lastk4,dinov3l:lastk4 (split_iso_full.csv, seed=1)
   11. pcg_all_s1  (GPU 7): siglip2:lastk4,cradio:lastk4,pecoreg:lastk4 (split_all_full.csv, seed=1, batch=4)
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
from train_v2 import MultiNetV2, exact_probs_from_dual, decode_scaled_argmax, optimise_scaled

GRADE = {0: '无问题', 1: 'P2', 2: 'P1', 3: 'P0'}
FULL_IDX = [i for i, c in enumerate(CLASSES) if c in FULL]
BIN_IDX  = [i for i, c in enumerate(CLASSES) if c not in FULL]
OUT_DIR = 'results/v2_super'
TEST_IMG_DIR = 'data/test_150'
os.makedirs(OUT_DIR, exist_ok=True)


def decode_super(p0, p1, p2, t0_full, t0_bin, t1, t2, r12=1.0):
    """Scaled argmax decoder with decoupled P0 thresholds for 4-grade vs binary classes."""
    n = p0.shape[0]
    pred = np.zeros((n, 6), int)
    pred[:, FULL_IDX] = np.where(p2 > t2, 1, 0)
    cond_p1 = (p1 > t1) & ((p1 * r12) >= (p2 - 0.05))
    pred[:, FULL_IDX] = np.where(cond_p1, 2, pred[:, FULL_IDX])
    pred[:, FULL_IDX] = np.where(p0[:, FULL_IDX] > t0_full, 3, pred[:, FULL_IDX])
    pred[:, BIN_IDX]  = np.where(p0[:, BIN_IDX] > t0_bin, 3, 0)
    return pred


def optimise_super(gt, p0, p1, p2, base_th, tau=0.02):
    """Fine-grained (0.025 step) search over (t0_full, t0_bin, t1, t2, r12) around base_th=(t0, t1, t2, r12)."""
    t0_0, t1_0, t2_0, r12_0 = base_th
    best = None
    best_th = (float(t0_0), float(t0_0), float(t1_0), float(t2_0), float(r12_0))
    t0f_grid = np.clip(np.arange(t0_0 - 0.125, t0_0 + 0.13, 0.025), 0.25, 0.85)
    t0b_grid = np.clip(np.arange(t0_0 - 0.125, t0_0 + 0.18, 0.025), 0.30, 0.90)
    t1_grid  = np.clip(np.arange(t1_0 - 0.10, t1_0 + 0.105, 0.025), 0.10, 0.80)
    t2_grid  = np.clip(np.arange(t2_0 - 0.10, t2_0 + 0.105, 0.025), 0.15, 0.85)
    for r12 in (0.75, 0.85, 1.0, 1.15):
        for t0_f in t0f_grid:
            for t0_b in t0b_grid:
                for t1 in t1_grid:
                    for t2 in t2_grid:
                        s = cm.score(gt, decode_super(p0, p1, p2, t0_f, t0_b, t1, t2, r12), tau)
                        key = (round(s['total'], 4), -s['fpr'], s['S_pos'], s['S_p0'])
                        if best is None or key > best_key:
                            best, best_key = s, key
                            best_th = (float(t0_f), float(t0_b), float(t1), float(t2), float(r12))
    return best, best_th


def step1_train_super():
    print('=== Step 1: Training 8 Heterogeneous V2 Multi-Backbone Models on GPUs 0..7 ===', flush=True)
    jobs = [
        # (name, gpu, split_csv, seed, backbones, batch)
        ('pcl_all_s1',  0, 'data/split_all_full.csv', 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4', 6),
        ('pcl_all_s2',  1, 'data/split_all_full.csv', 2, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4', 6),
        ('pcl_iso_s1',  2, 'data/split_iso_full.csv', 1, 'siglip2:lastk4,cradio:lastk4,pecorel:lastk4', 6),
        ('eupe_all_s1', 3, 'data/split_all_full.csv', 1, 'eupe:full,siglip2:lastk4,cradio:lastk4',      6),
        ('eupe_iso_s1', 4, 'data/split_iso_full.csv', 1, 'eupe:full,siglip2:lastk4,cradio:lastk4',      6),
        ('d3_all_s1',   5, 'data/split_all_full.csv', 1, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 6),
        ('d3_iso_s1',   6, 'data/split_iso_full.csv', 1, 'siglip2:lastk4,cradio:lastk4,dinov3l:lastk4', 6),
        ('pcg_all_s1',  7, 'data/split_all_full.csv', 1, 'siglip2:lastk4,cradio:lastk4,pecoreg:lastk4', 4),
    ]

    procs = {}
    t0 = time.time()
    for name, gpu, sp_csv, sd, bbs, bs in jobs:
        out_json = f'{OUT_DIR}/train_{name}.json'
        ckpt_pt  = f'{OUT_DIR}/model_{name}.pt'
        if os.path.exists(out_json) and os.path.exists(ckpt_pt):
            print(f'[skip] {name} already exists.', flush=True)
            continue
        cmd = [
            'python3', '-u', '-I', 'code/finetune/train_v2.py',
            '--backbones', bbs,
            '--seed', str(sd),
            '--epochs', '4',
            '--batch', str(bs),
            '--workers', '6',
            '--size', '384',
            '--clean-repeat', '4',
            '--p0-clean-weight', '4.0',
            '--lr-full', '8.875e-05',
            '--lr-last', '3e-05',
            '--lr-head', '1.5e-03',
            '--wd', '0.01',
            '--dropout', '0.1',
            '--split', sp_csv,
            '--out', out_json,
            '--save-state', ckpt_pt,
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        log_f = open(f'{OUT_DIR}/train_{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs[name] = (p, gpu, out_json, ckpt_pt, log_f)
        print(f'[{time.time()-t0:5.0f}s] Launched {name:12s} ({bbs}) on GPU {gpu}', flush=True)

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
                    print(f'[{time.time()-t0:5.0f}s] DONE {name:12s} (GPU {gpu}) | '
                          f'val_scaled={ts["total"]:.2f} (S_pos={ts["S_pos"]:.3f}, S_p0={ts["S_p0"]:.3f}, '
                          f'S_clean={ts["S_clean"]:.3f}) | val_cascade={tc["total"]:.2f}', flush=True)
                else:
                    print(f'[{time.time()-t0:5.0f}s] FAIL {name} (GPU {gpu}) rc={p.returncode} - check {OUT_DIR}/train_{name}.log', flush=True)


@torch.no_grad()
def infer_ckpt_test_tta(ckpt_path, dl_150, dev):
    """Run 2x Horizontal Flip TTA inference on the 150 test images for a single checkpoint."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    bbs = ck['backbones']
    specs = []
    for s in bbs.split(','):
        nm, mode = s.split(':')
        specs.append((build_backbone(nm), nm, mode))
    net = MultiNetV2([x[0] for x in specs], dropout=0.0).to(dev)
    net.load_state_dict(ck['state'])
    net.eval()
    alpha_ex = ck.get('alpha_ex', 0.45)

    P0, P1, P2 = [], [], []
    for x, _, _ in dl_150:
        x = x.to(dev)
        x_flip = torch.flip(x, dims=[4])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lc1, le1, _ = net(x)
            lc2, le2, _ = net(x_flip)
        p0_1, p1_1, p2_1, _ = exact_probs_from_dual(lc1.float(), le1.float(), alpha_ex=alpha_ex)
        p0_2, p1_2, p2_2, _ = exact_probs_from_dual(lc2.float(), le2.float(), alpha_ex=alpha_ex)
        P0.append((0.5 * (p0_1 + p0_2)).cpu().numpy())
        P1.append((0.5 * (p1_1 + p1_2)).cpu().numpy())
        P2.append((0.5 * (p2_1 + p2_2)).cpu().numpy())

    del net, specs, ck
    torch.cuda.empty_cache()
    return np.concatenate(P0, 0), np.concatenate(P1, 0), np.concatenate(P2, 0)


def step2_calibrate_and_predict():
    print('\n=== Step 2: Calibrating on Held-Out Validation & Running Super-Ensemble TTA on Test Set ===', flush=True)
    dev = 'cuda:0'

    tpl = json.load(open('results/v2_full/template_150.json'))
    ordered_names = [e['image'] for e in tpl]
    test_rows = [{'name': nm, **{c: 0 for c in CLASSES}} for nm in ordered_names]
    ds_150 = IQADataset(test_rows, TEST_IMG_DIR, TEST_IMG_DIR, set(ordered_names), size=384, train=False, tiles=2)
    dl_150 = DataLoader(ds_150, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)

    # 1. Load held-out validation predictions (309 images from split_iso_full.csv) across the 4 iso models
    iso_models = [
        ('base_iso_s1', 'results/v2_full/train_iso_s1_probs.npz',   'base'),
        ('pcl_iso_s1',  f'{OUT_DIR}/train_pcl_iso_s1_probs.npz',    'pcl'),
        ('eupe_iso_s1', f'{OUT_DIR}/train_eupe_iso_s1_probs.npz',   'eupe'),
        ('d3_iso_s1',   f'{OUT_DIR}/train_d3_iso_s1_probs.npz',     'd3'),
    ]
    val_preds = {}
    gva = None
    for name, npz_path, fam in iso_models:
        if os.path.exists(npz_path):
            d = np.load(npz_path, allow_pickle=True)
            val_preds[name] = (d['va_p0'], d['va_p1'], d['va_p2'], fam)
            if gva is None:
                gva = d['gva'].astype(int)
            s_i, th_i = optimise_scaled(gva, d['va_p0'], d['va_p1'], d['va_p2'], 0.02)
            print(f'  Held-out 309-val [{name:12s}]: scaled={s_i["total"]:.2f} '
                  f'(S_pos={s_i["S_pos"]:.3f}, S_p0={s_i["S_p0"]:.3f}, S_clean={s_i["S_clean"]:.3f}) th={th_i}', flush=True)

    # Optimize family weights on held-out 309-image validation set
    best_ens_score = -1.0
    best_fam_w = {'base': 1.0, 'pcl': 1.5, 'eupe': 1.25, 'd3': 1.25, 'pcg': 1.5}
    for w_base in (0.8, 1.0, 1.2):
        for w_pcl in (1.2, 1.5, 1.8):
            for w_eupe in (1.0, 1.3, 1.6):
                for w_d3 in (0.8, 1.1, 1.4):
                    fw = {'base': w_base, 'pcl': w_pcl, 'eupe': w_eupe, 'd3': w_d3, 'pcg': w_pcl}
                    p0_v, p1_v, p2_v, w_sum = 0.0, 0.0, 0.0, 0.0
                    for nm, (vp0, vp1, vp2, fam) in val_preds.items():
                        wi = fw.get(fam, 1.0)
                        p0_v = p0_v + wi * vp0
                        p1_v = p1_v + wi * vp1
                        p2_v = p2_v + wi * vp2
                        w_sum += wi
                    p0_v /= w_sum; p1_v /= w_sum; p2_v /= w_sum
                    s_sc, _ = optimise_scaled(gva, p0_v, p1_v, p2_v, 0.02)
                    if s_sc['total'] > best_ens_score:
                        best_ens_score = s_sc['total']
                        best_fam_w = fw

    p0_v, p1_v, p2_v, w_sum = 0.0, 0.0, 0.0, 0.0
    for nm, (vp0, vp1, vp2, fam) in val_preds.items():
        wi = best_fam_w.get(fam, 1.0)
        p0_v = p0_v + wi * vp0
        p1_v = p1_v + wi * vp1
        p2_v = p2_v + wi * vp2
        w_sum += wi
    p0_v /= w_sum; p1_v /= w_sum; p2_v /= w_sum

    s_sc, th_sc = optimise_scaled(gva, p0_v, p1_v, p2_v, 0.02)
    s_su, th_su = optimise_super(gva, p0_v, p1_v, p2_v, th_sc, 0.02)
    print(f'\nHeld-out 309-val Ensemble (4-param scaled): total={s_sc["total"]:.2f} '
          f'(S_pos={s_sc["S_pos"]:.3f}, S_p0={s_sc["S_p0"]:.3f}, S_clean={s_sc["S_clean"]:.3f}) th={th_sc}', flush=True)
    print(f'Held-out 309-val Ensemble (5-param super) : total={s_su["total"]:.2f} '
          f'(S_pos={s_su["S_pos"]:.3f}, S_p0={s_su["S_p0"]:.3f}, S_clean={s_su["S_clean"]:.3f}) th={th_su}', flush=True)
    print(f'Optimal family weights: {best_fam_w}\n', flush=True)

    # 2. Run 2x Horizontal Flip TTA on all 11 checkpoints for the 150 test images
    ckpts = [
        ('base_all_s1', 'results/v2_full/model_all_s1.pt',  'base', False),
        ('base_all_s2', 'results/v2_full/model_all_s2.pt',  'base', False),
        ('base_iso_s1', 'results/v2_full/model_iso_s1.pt',  'base', True),
        ('pcl_all_s1',  f'{OUT_DIR}/model_pcl_all_s1.pt',   'pcl',  False),
        ('pcl_all_s2',  f'{OUT_DIR}/model_pcl_all_s2.pt',   'pcl',  False),
        ('pcl_iso_s1',  f'{OUT_DIR}/model_pcl_iso_s1.pt',   'pcl',  True),
        ('eupe_all_s1', f'{OUT_DIR}/model_eupe_all_s1.pt',  'eupe', False),
        ('eupe_iso_s1', f'{OUT_DIR}/model_eupe_iso_s1.pt',  'eupe', True),
        ('d3_all_s1',   f'{OUT_DIR}/model_d3_all_s1.pt',    'd3',   False),
        ('d3_iso_s1',   f'{OUT_DIR}/model_d3_iso_s1.pt',    'd3',   True),
        ('pcg_all_s1',  f'{OUT_DIR}/model_pcg_all_s1.pt',   'pcg',  False),
    ]

    cache_npz = f'{OUT_DIR}/test_150_all_models_probs.npz'
    test_model_probs = {}
    if os.path.exists(cache_npz):
        loaded = np.load(cache_npz, allow_pickle=True)['data'].item()
        test_model_probs.update(loaded)

    for name, ckp, fam, is_iso in ckpts:
        if not os.path.exists(ckp):
            print(f'[warn] Missing checkpoint {ckp}, skipping.', flush=True)
            continue
        if name in test_model_probs:
            print(f'[cached] {name:12s} test TTA probabilities loaded.', flush=True)
            continue
        t0 = time.time()
        p0_i, p1_i, p2_i = infer_ckpt_test_tta(ckp, dl_150, dev)
        test_model_probs[name] = (p0_i, p1_i, p2_i, fam, is_iso)
        np.savez_compressed(cache_npz, data=test_model_probs)
        print(f'[{time.time()-t0:5.1f}s] Inferred {name:12s} (2x flip TTA on 150 test images)', flush=True)

    # Combine all available models weighted by family weight (full models: 1.0x, iso models: 0.65x)
    p0_ens, p1_ens, p2_ens, tw_sum = 0.0, 0.0, 0.0, 0.0
    used_models = []
    for name, (p0_i, p1_i, p2_i, fam, is_iso) in test_model_probs.items():
        wi = best_fam_w.get(fam, 1.0) * (0.65 if is_iso else 1.0)
        p0_ens = p0_ens + wi * p0_i
        p1_ens = p1_ens + wi * p1_i
        p2_ens = p2_ens + wi * p2_i
        tw_sum += wi
        used_models.append((name, round(wi, 3)))
    p0_ens /= tw_sum; p1_ens /= tw_sum; p2_ens /= tw_sum
    print(f'\nCombined {len(used_models)} models for test inference: {used_models}', flush=True)

    # Choose 5-param super decoder if it beats 4-param scaled by >= 0.05 with non-worse FPR
    if s_su['total'] >= s_sc['total'] + 0.05 and s_su['fpr'] <= s_sc['fpr']:
        pred_150 = decode_super(p0_ens, p1_ens, p2_ens, *th_su)
        chosen_mode = f'super_5param {th_su}'
    else:
        pred_150 = decode_scaled_argmax(p0_ens, p1_ens, p2_ens, *th_sc)
        chosen_mode = f'scaled_4param {th_sc}'
    print(f'Selected decoding mode: {chosen_mode}', flush=True)

    out_list = []
    for i, nm in enumerate(ordered_names):
        lab = {}
        for k in ["模糊", "纹理损失", "过度生成", "虚化突兀", "过锐", "花屏"]:
            ci = CLASSES.index(k)
            lab[k] = GRADE[int(pred_150[i, ci])]
        out_list.append({"image": nm, "label": lab})

    out_super_json = f'{OUT_DIR}/pred_nriqa_dataset_test.json'
    out_root_json  = 'pred_nriqa_dataset_test.json'
    json.dump(out_list, open(out_super_json, 'w'), ensure_ascii=False, indent=2)
    json.dump(out_list, open(out_root_json, 'w'), ensure_ascii=False, indent=2)

    print('\nPer-class grade distribution on nriqa_dataset_test.zip (150 images):', flush=True)
    for ci, c in enumerate(CLASSES):
        vals = [GRADE[int(v)] for v in pred_150[:, ci]]
        uniq = {v: vals.count(v) for v in ['无问题', 'P2', 'P1', 'P0'] if v in vals}
        print(f'  {c:6s}: {uniq}', flush=True)
    clean_cnt = int((pred_150 == 0).all(1).sum())
    print(f'All-clean images: {clean_cnt}/{len(pred_150)} ({100*clean_cnt/len(pred_150):.1f}%)', flush=True)

    if os.path.exists('results/v2_full/pred_nriqa_dataset_test.json'):
        prev = json.load(open('results/v2_full/pred_nriqa_dataset_test.json'))
        same_cells, total_cells, diff_imgs = 0, 0, 0
        for a, b in zip(prev, out_list):
            img_diff = False
            for k in CLASSES:
                total_cells += 1
                if a['label'][k] == b['label'][k]:
                    same_cells += 1
                else:
                    img_diff = True
            if img_diff:
                diff_imgs += 1
        print(f'Agreement vs previous V2 dual-backbone (50.41): {same_cells}/{total_cells} cells '
              f'({100*same_cells/total_cells:.1f}%), {diff_imgs}/150 images refined.', flush=True)

    print(f'Saved strongest test predictions to: {out_root_json} and {out_super_json}', flush=True)


if __name__ == '__main__':
    step1_train_super()
    step2_calibrate_and_predict()
