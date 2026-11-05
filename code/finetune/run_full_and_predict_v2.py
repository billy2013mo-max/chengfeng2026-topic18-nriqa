"""Train V2 dual-backbone (siglip2:lastk4,cradio:lastk4) on the full 3573+ dataset
and run inference on /wanqing-develop/luowenjing/18/nriqa_dataset_test.zip."""
import csv, json, os, shutil, subprocess, sys, time, zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/finetune')
sys.path.insert(0, '/wanqing-develop/luowenjing/18/code/decision_eval')
from data import IQADataset, CLASSES, FULL
from models import build_backbone
from train_v2 import MultiNetV2, exact_probs_from_dual, decode_scaled_argmax, optimise_scaled
import comp_metrics as cm

OUT_DIR = 'results/v2_full'
TEST_IMG_DIR = 'data/test_150'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEST_IMG_DIR, exist_ok=True)

# 1. Extract nriqa_dataset_test.zip cleanly
print("=== Step 1: Extracting nriqa_dataset_test.zip ===")
zf = zipfile.ZipFile('nriqa_dataset_test.zip')
extracted_names = []
for info in zf.infolist():
    if info.is_dir():
        continue
    bn = os.path.basename(info.filename)
    if not bn or bn.startswith('._') or '__MACOSX' in info.filename:
        continue
    if bn.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
        target_path = os.path.join(TEST_IMG_DIR, bn)
        if not os.path.exists(target_path):
            with zf.open(info) as src, open(target_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        extracted_names.append(bn)

extracted_names = sorted(set(extracted_names))
print(f"Extracted {len(extracted_names)} test images into {TEST_IMG_DIR}")

# Build template matching results/iso/pred_150_all.json order if available
if os.path.exists('results/iso/pred_150_all.json'):
    ref_tpl = json.load(open('results/iso/pred_150_all.json'))
    ref_names = [x['image'] for x in ref_tpl]
    assert set(ref_names) == set(extracted_names), f"Mismatch: {len(ref_names)} vs {len(extracted_names)}"
    ordered_names = ref_names
    print("Order verified against results/iso/pred_150_all.json (150 images).")
else:
    ordered_names = extracted_names

tpl_150 = [
    {"image": nm, "label": {"模糊": "", "纹理损失": "", "过度生成": "", "虚化突兀": "", "过锐": "", "花屏": ""}}
    for nm in ordered_names
]
tpl_150_path = os.path.join(OUT_DIR, 'template_150.json')
json.dump(tpl_150, open(tpl_150_path, 'w'), ensure_ascii=False, indent=2)

# 2. Launch 3 parallel V2 full-dataset training jobs on GPUs 0, 1, 2
print("\n=== Step 2: Training V2 Dual-Backbone (siglip2:lastk4,cradio:lastk4) on Full Dataset ===")
JOBS = [
    # (gpu, name, split_csv, seed)
    (0, 'all_s1', 'data/split_all_full.csv', 1),
    (1, 'all_s2', 'data/split_all_full.csv', 2),
    (2, 'iso_s1', 'data/split_iso_full.csv', 1),
]

procs = {}
t0 = time.time()
for gpu, name, split_csv, seed in JOBS:
    out_json = f'{OUT_DIR}/train_{name}.json'
    ckpt_pt = f'{OUT_DIR}/model_{name}.pt'
    if os.path.exists(out_json) and os.path.exists(ckpt_pt):
        print(f"  {name} already exists, skipping training.")
        continue
    cmd = [
        'python3', '-u', '-I', 'code/finetune/train_v2.py',
        '--backbones', 'siglip2:lastk4,cradio:lastk4',
        '--seed', str(seed),
        '--epochs', '4',
        '--batch', '6',
        '--workers', '6',
        '--size', '384',
        '--clean-repeat', '4',
        '--p0-clean-weight', '4.0',
        '--lr-full', '8.875e-05',
        '--lr-last', '3e-05',
        '--lr-head', '1.5e-03',
        '--wd', '0.01',
        '--dropout', '0.1',
        '--split', split_csv,
        '--out', out_json,
        '--save-state', ckpt_pt,
    ]
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    lf = open(f'{OUT_DIR}/train_{name}.log', 'w')
    p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    procs[name] = (p, gpu, out_json, ckpt_pt, lf)
    print(f"[{time.time()-t0:5.0f}s] Launched {name} ({split_csv}, seed={seed}) on GPU {gpu}", flush=True)

while procs:
    time.sleep(15)
    for name in list(procs):
        p, gpu, out_json, ckpt_pt, lf = procs[name]
        if p.poll() is not None:
            lf.close()
            del procs[name]
            if os.path.exists(out_json):
                d = json.load(open(out_json))
                ts = d['test']
                tc = d['test_cascade']
                print(f"[{time.time()-t0:5.0f}s] DONE {name} (GPU {gpu}) | "
                      f"val_scaled={ts['total']:.2f} (S_pos={ts['S_pos']:.3f}, S_p0={ts['S_p0']:.3f}, S_clean={ts['S_clean']:.3f}) | "
                      f"val_cascade={tc['total']:.2f}", flush=True)
            else:
                print(f"[{time.time()-t0:5.0f}s] FAIL {name} (GPU {gpu}) rc={p.returncode}", flush=True)

# 3. Calibrate thresholds using iso_s1 (held-out val) + s16..s20 v2 runs, and run 2-seed + TTA inference on the 150 test images
print("\n=== Step 3: Running Ensemble + TTA Inference on nriqa_dataset_test.zip (150 images) ===")
dev = 'cuda:0'
specs = []
for s in 'siglip2:lastk4,cradio:lastk4'.split(','):
    nm, mode = s.split(':')
    specs.append((build_backbone(nm), nm, mode))

net = MultiNetV2([x[0] for x in specs], dropout=0.0).to(dev)
net.eval()

test_rows = [{'name': nm, **{c: 0 for c in CLASSES}} for nm in ordered_names]
ds_150 = IQADataset(test_rows, TEST_IMG_DIR, TEST_IMG_DIR, set(ordered_names), size=384, train=False, tiles=2)
dl_150 = DataLoader(ds_150, batch_size=10, shuffle=False, num_workers=6, pin_memory=True)


@torch.no_grad()
def infer_with_tta(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu')
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
    return np.concatenate(P0, 0), np.concatenate(P1, 0), np.concatenate(P2, 0), ck


ckpts = [f'{OUT_DIR}/model_all_s1.pt', f'{OUT_DIR}/model_all_s2.pt']
p0_ens, p1_ens, p2_ens = 0.0, 0.0, 0.0
for ckp in ckpts:
    p0_i, p1_i, p2_i, _ = infer_with_tta(ckp)
    p0_ens = p0_ens + p0_i / len(ckpts)
    p1_ens = p1_ens + p1_i / len(ckpts)
    p2_ens = p2_ens + p2_i / len(ckpts)

# Determine honest thresholds from iso_s1 (held-out 309 val images) and s16..s20 v2 runs
iso_npz = np.load(f'{OUT_DIR}/train_iso_s1_probs.npz', allow_pickle=True)
_, th_iso_sca = optimise_scaled(iso_npz['gva'].astype(int), iso_npz['va_p0'], iso_npz['va_p1'], iso_npz['va_p2'], 0.02)
_, th_iso_cas = cm.optimise(iso_npz['gva'].astype(int), iso_npz['va_p0'], iso_npz['va_p1'], iso_npz['va_p2'], 0.02)
print(f"Held-out iso_s1 optimal scaled thresholds : {th_iso_sca}")
print(f"Held-out iso_s1 optimal cascade thresholds: {th_iso_cas}")

# Save probabilities for the 150 test images
np.savez_compressed(f'{OUT_DIR}/pred_150_probs.npz', names=np.array(ordered_names), p0=p0_ens, p1=p1_ens, p2=p2_ens)

GRADE = {0: '无问题', 1: 'P2', 2: 'P1', 3: 'P0'}
pred_150 = decode_scaled_argmax(p0_ens, p1_ens, p2_ens, *th_iso_sca)

out_list = []
for i, nm in enumerate(ordered_names):
    lab = {}
    for k in ["模糊", "纹理损失", "过度生成", "虚化突兀", "过锐", "花屏"]:
        ci = CLASSES.index(k)
        lab[k] = GRADE[int(pred_150[i, ci])]
    out_list.append({"image": nm, "label": lab})

out_path_1 = f'{OUT_DIR}/pred_nriqa_dataset_test.json'
out_path_2 = 'pred_nriqa_dataset_test.json'
json.dump(out_list, open(out_path_1, 'w'), ensure_ascii=False, indent=2)
json.dump(out_list, open(out_path_2, 'w'), ensure_ascii=False, indent=2)

print("\nPer-class grade distribution on nriqa_dataset_test.zip (150 images):")
for ci, c in enumerate(CLASSES):
    vals = [GRADE[int(v)] for v in pred_150[:, ci]]
    uniq = {v: vals.count(v) for v in ['无问题', 'P2', 'P1', 'P0'] if v in vals}
    print(f"  {c:6s}: {uniq}")
clean_cnt = int((pred_150 == 0).all(1).sum())
print(f"All-clean images: {clean_cnt}/{len(pred_150)} ({100*clean_cnt/len(pred_150):.1f}%)")
print(f"Saved test predictions to: {out_path_1} and {out_path_2}")
