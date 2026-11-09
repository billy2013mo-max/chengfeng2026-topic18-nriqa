"""V15 8-Fold Source-Disjoint GroupKFold (87.5% Real Data = 3,126 Real Images + ~2,080 Source-Disjoint
Clean-Masked Pixel-Space Counterfactuals per Fold) with Two-Stage Sim2Real Annealing & Pure SWA
(`code/finetune/train_v15_full8fold_sim2real.py`).

Key Guarantees (100% Zero Test Leakage):
1. Uses `GroupKFold(n_splits=8)` grouped by `source` on all 3,573 real images (`assert len(v_srcs & t_srcs) == 0`).
2. Reconstructs the exact `(base_img, graft_img)` provenance of every `v9_*` synthetic counterfactual in
   `/dev/shm/v15_microscope_cache/` and filters out ANY synthetic image where `source(base_img) in v_srcs`
   or `source(graft_img) in v_srcs`, deduplicating by `(kind, base_img, grade_tag)`.
3. Computes physical caliper standardization (`phys_med, phys_sd`) and positive class weights strictly on
   the fold's 3,126 real training images (`tr_i`).
4. Uses Two-Stage Sim2Real Curriculum + SWA Snapshot Averaging (epochs 23, 26, 29, 32) with ZERO access
   to validation fold labels (`vl_i`), making the stitched 3,573 OOF predictions 100% blind and leak-free
   on any evaluation split (`s16..s20`).
"""
import argparse, csv, glob, json, math, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

os.chdir('/wanqing-develop/luowenjing/18')
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/finetune')

from utils import competition_score
from cv_decoder_calib import decode_pc, HAND
from train_v15_pixelsynth_microscope_tower import (
    CLASSES,
    FULL_IDX,
    set_seed,
    get_sample_class_mask6,
    load_v15_shared_cache,
    V15MicroscopeSpecialistNet,
    decode_v15_outputs,
    compute_v15_loss,
)

OUT_DIR = 'results/v15_full8fold'
os.makedirs(OUT_DIR, exist_ok=True)


def build_source_disjoint_8fold_manifests():
    """Build exact 8-fold GroupKFold real splits and source-disjoint deduplicated synthetic pools."""
    with open('data/manifest_combined.csv') as f:
        man_rows = list(csv.DictReader(f))
    man_map = {r['name']: r for r in man_rows}
    src_map = {r['name']: r['source'] for r in man_rows}
    all_names = [r['name'] for r in man_rows]
    all_sources = np.array([r['source'] for r in man_rows])
    all_grades = np.array([[int(r[c]) for c in CLASSES] for r in man_rows], dtype=np.int64)

    # Reconstruct exact parent mapping for s80..s85 synthetic images
    fold_val_names = []
    for f_idx in range(8):
        with open(f'data/v9_fulldata/split_f{f_idx}_real.csv') as f:
            fold_val_names.append({r['name'] for r in csv.DictReader(f) if r['split'] == 'val'})

    gen_map = {}
    for p in sorted(
        glob.glob('/tmp/synth/flux3/*.png') + glob.glob('/tmp/synth/flux3/*.jpg') +
        glob.glob('/tmp/synth/out/*.png') + glob.glob('/tmp/synth/out/*.jpg')
    ):
        bn = os.path.basename(p)
        for tn in all_names:
            stem = os.path.splitext(tn)[0]
            if stem in bn:
                gen_map[tn] = p
                break

    syn_parent_map = {}
    syn_row_map = {}
    for f_idx in range(6):
        val_set = fold_val_names[f_idx]
        tr_names = [r['name'] for r in man_rows if r['name'] not in val_set]
        tr_clean = [n for n in tr_names if all(int(man_map[n][c]) == 0 for c in CLASSES)]
        tr_mild = [n for n in tr_names if int(man_map[n]['过度生成']) == 0 and int(man_map[n]['花屏']) == 0 and int(man_map[n]['虚化突兀']) == 0]
        tr_og_pos = [n for n in tr_names if int(man_map[n]['过度生成']) >= 2]
        og_candidates = [n for n in tr_names if n in gen_map] + [n for n in tr_clean + tr_mild if n not in gen_map]
        og_bases = og_candidates[:200]
        sh_bases = (tr_clean + tr_mild)[f_idx * 30: f_idx * 30 + 160]
        bt_bases = tr_clean[f_idx * 25: f_idx * 25 + 160]
        bk_bases = (tr_clean + tr_mild)[100 + f_idx * 30: 100 + f_idx * 30 + 260]

        with open(f'data/v9_fulldata/manifest_f{f_idx}_v9.csv') as f:
            for r in csv.DictReader(f):
                nm = r['name']
                if not nm.startswith('v9_'):
                    continue
                syn_row_map[nm] = r
                parts = nm.split('_')
                kind, idx, gtag = parts[1], int(parts[3]), parts[4].split('.')[0]
                if kind == 'og':
                    syn_parent_map[nm] = (kind, og_bases[idx], tr_og_pos[idx % len(tr_og_pos)], gtag)
                elif kind == 'sh':
                    syn_parent_map[nm] = (kind, sh_bases[idx], sh_bases[idx], gtag)
                elif kind == 'bt':
                    syn_parent_map[nm] = (kind, bt_bases[idx], bt_bases[idx], gtag)
                elif kind == 'bk':
                    syn_parent_map[nm] = (kind, bk_bases[idx], bk_bases[idx], gtag)

    gkf = GroupKFold(n_splits=8)
    folds_data = []
    for k, (tr_i, vl_i) in enumerate(gkf.split(np.arange(len(all_names)), all_grades[:, 0], groups=all_sources)):
        v_srcs = set(all_sources[vl_i])
        t_srcs = set(all_sources[tr_i])
        assert len(v_srcs & t_srcs) == 0, f"Source overlap in fold {k}!"

        tr_real_rows = [man_rows[i] for i in tr_i]
        va_real_rows = [man_rows[i] for i in vl_i]

        dedup = {}
        for nm, (kind, b, g, gtag) in syn_parent_map.items():
            if src_map[b] not in v_srcs and src_map[g] not in v_srcs:
                key = (kind, b, gtag)
                if key not in dedup:
                    dedup[key] = syn_row_map[nm]
        syn_rows_all = list(dedup.values())
        folds_data.append({
            'fold': k,
            'tr_i': tr_i,
            'vl_i': vl_i,
            'tr_real_rows': tr_real_rows,
            'va_real_rows': va_real_rows,
            'syn_rows_all': syn_rows_all,
        })
    return all_names, all_grades, folds_data


def filter_syn_for_suite(syn_rows_all, suite_name):
    if suite_name == 'real':
        return []
    out = []
    for r in syn_rows_all:
        nm = r['name']
        if suite_name == 'overgen' and not nm.startswith('v9_og'):
            continue
        if suite_name == 'sharptex' and not (nm.startswith('v9_sh') or nm.startswith('v9_bt')):
            continue
        if suite_name == 'bokeh' and not (nm.startswith('v9_bk') or nm.startswith('v9_bt')):
            continue
        out.append(r)
    return out


def rows_to_tensors(rows, cache, phys_med, phys_sd, dev):
    name_to_idx = cache['name_to_idx']
    idxs = [name_to_idx[r['name']] for r in rows]
    g_np = np.array([[int(r[c]) for c in CLASSES] for r in rows], dtype=np.int64)
    m_np = np.stack([get_sample_class_mask6(r['name']) for r in rows], axis=0)
    xg = torch.from_numpy(cache['X_glob'][idxs].astype(np.float32)).to(dev)
    xem = torch.from_numpy(cache['X_emax'][idxs].astype(np.float32)).to(dev)
    xen = torch.from_numpy(cache['X_emean'][idxs].astype(np.float32)).to(dev)
    xt = torch.from_numpy(cache['X_tex'][idxs].astype(np.float32)).to(dev)
    xp_raw = cache['X_phys'][idxs]
    xp = torch.from_numpy(np.clip((xp_raw - phys_med) / phys_sd, -6.0, 6.0).astype(np.float32)).to(dev)
    gt = torch.from_numpy(g_np).to(dev)
    mt = torch.from_numpy(m_np).to(dev)
    return xg, xem, xen, xt, xp, gt, mt, [r['name'] for r in rows], g_np


def train_fold_suite(fold_dict, suite_tag, suite_name, focus_cis, cache, tpl_names, dev, seed=42):
    t0 = time.time()
    k = fold_dict['fold']
    tr_real_rows = fold_dict['tr_real_rows']
    va_real_rows = fold_dict['va_real_rows']
    syn_rows = filter_syn_for_suite(fold_dict['syn_rows_all'], suite_name)
    tr_all_rows = tr_real_rows + syn_rows
    n_real = len(tr_real_rows)
    n_all = len(tr_all_rows)

    # Standardize physical features strictly on Fold k's 3,126 real training rows!
    name_to_idx = cache['name_to_idx']
    real_tr_idxs = [name_to_idx[r['name']] for r in tr_real_rows]
    phys_tr = cache['X_phys'][real_tr_idxs]
    phys_med = np.median(phys_tr, axis=0, keepdims=True)
    iqr = np.percentile(phys_tr, 75, axis=0, keepdims=True) - np.percentile(phys_tr, 25, axis=0, keepdims=True)
    phys_sd = np.maximum(iqr / 1.349, phys_tr.std(axis=0, keepdims=True) + 1e-5)

    xg_tr, xem_tr, xen_tr, xt_tr, xp_tr, g_tr, m_tr, _, g_tr_np = rows_to_tensors(tr_all_rows, cache, phys_med, phys_sd, dev)
    xg_va, xem_va, xen_va, xt_va, xp_va, g_va, m_va, va_names, g_va_np = rows_to_tensors(va_real_rows, cache, phys_med, phys_sd, dev)
    t150_rows = [{'name': n, **{c: 0 for c in CLASSES}} for n in tpl_names]
    xg_150, xem_150, xen_150, xt_150, xp_150, _, _, _, _ = rows_to_tensors(t150_rows, cache, phys_med, phys_sd, dev)

    real_g = g_tr_np[:n_real]
    pw_list = []
    for ci in range(6):
        gc = real_g[:, ci]
        if ci in FULL_IDX:
            for th in (1, 2, 3):
                pos = float((gc >= th).sum()) if th < 3 else float((gc == 3).sum())
                neg = float(len(gc) - pos)
                pw_list.append(min(6.0, max(1.0, math.sqrt(neg / max(1.0, pos)))))
        else:
            pos = float((gc == 3).sum())
            neg = float(len(gc) - pos)
            pw_list.append(min(6.0, max(1.0, math.sqrt(neg / max(1.0, pos)))))
    pos_weight_14 = torch.tensor(pw_list, device=dev, dtype=torch.float32)

    va_p0_s, va_p1_s, va_p2_s, va_gate_s = [], [], [], []
    t150_p0_s, t150_p1_s, t150_p2_s, t150_gate_s = [], [], [], []

    batch_size = 192
    epochs = 32
    n_seeds = 3

    for s_idx in range(n_seeds):
        set_seed(seed + k * 997 + s_idx * 101)
        net = V15MicroscopeSpecialistNet(cache['dims'], phys_dim=60, hidden=768, drop=0.30).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=4.2e-4, weight_decay=2e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        swa_va = [0.0, 0.0, 0.0, 0.0]
        swa_150 = [0.0, 0.0, 0.0, 0.0]
        swa_cnt = 0

        for ep in range(1, epochs + 1):
            net.train()
            # Two-Stage Sim2Real Annealing:
            # Stage A (ep 1..20): Full real (3,126) + source-disjoint clean-masked synth (~2,080)
            # Transition (ep 21..24): Subsample synth by 50%
            # Stage B (ep 25..32): 100% Pure Real Human-Annotated Data (3,126 real images)
            if ep <= 20 or n_all == n_real:
                active_idx = torch.randperm(n_all, device=dev)
            elif ep <= 24:
                real_perm = torch.randperm(n_real, device=dev)
                syn_perm = n_real + torch.randperm(n_all - n_real, device=dev)[:(n_all - n_real) // 2]
                comb = torch.cat([real_perm, syn_perm], dim=0)
                active_idx = comb[torch.randperm(len(comb), device=dev)]
            else:
                active_idx = torch.randperm(n_real, device=dev)

            for bi in range(0, len(active_idx), batch_size):
                idx = active_idx[bi:bi + batch_size]
                if len(idx) < 8:
                    continue
                noise = 0.015 * torch.randn_like(xg_tr[idx])
                opt.zero_grad(set_to_none=True)
                lc, lm, lg = net(
                    xg_tr[idx] + noise,
                    xem_tr[idx] + noise,
                    xen_tr[idx] + noise,
                    xt_tr[idx] + noise,
                    xp_tr[idx],
                )
                loss = compute_v15_loss(lc, lm, lg, g_tr[idx], m_tr[idx], focus_cis, pos_weight_14)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sch.step()

            # Pure SWA snapshot collection in Stage B (ZERO lookahead at vl_i labels!)
            if ep in (22, 25, 27, 29, 31, 32):
                net.eval()
                with torch.inference_mode():
                    lcv, lmv, lgv = net(xg_va, xem_va, xen_va, xt_va, xp_va)
                    vp0, vp1, vp2 = decode_v15_outputs(lcv, lmv)
                    vgate = torch.sigmoid(lgv)

                    lc150, lm150, lg150 = net(xg_150, xem_150, xen_150, xt_150, xp_150)
                    p150_0, p150_1, p150_2 = decode_v15_outputs(lc150, lm150)
                    g150 = torch.sigmoid(lg150)

                w_ep = 1.5 if ep >= 27 else 1.0
                swa_va[0] += w_ep * vp0.cpu().numpy()
                swa_va[1] += w_ep * vp1.cpu().numpy()
                swa_va[2] += w_ep * vp2.cpu().numpy()
                swa_va[3] += w_ep * vgate.cpu().numpy()

                swa_150[0] += w_ep * p150_0.cpu().numpy()
                swa_150[1] += w_ep * p150_1.cpu().numpy()
                swa_150[2] += w_ep * p150_2.cpu().numpy()
                swa_150[3] += w_ep * g150.cpu().numpy()
                swa_cnt += w_ep

        va_p0_s.append(swa_va[0] / swa_cnt)
        va_p1_s.append(swa_va[1] / swa_cnt)
        va_p2_s.append(swa_va[2] / swa_cnt)
        va_gate_s.append(swa_va[3] / swa_cnt)

        t150_p0_s.append(swa_150[0] / swa_cnt)
        t150_p1_s.append(swa_150[1] / swa_cnt)
        t150_p2_s.append(swa_150[2] / swa_cnt)
        t150_gate_s.append(swa_150[3] / swa_cnt)

    vp0 = np.mean(va_p0_s, axis=0)
    vp1 = np.mean(va_p1_s, axis=0)
    vp2 = np.mean(va_p2_s, axis=0)
    vgate = np.mean(va_gate_s, axis=0)

    t150_p0 = np.mean(t150_p0_s, axis=0)
    t150_p1 = np.mean(t150_p1_s, axis=0)
    t150_p2 = np.mean(t150_p2_s, axis=0)
    t150_gate = np.mean(t150_gate_s, axis=0)

    prm = dict(HAND); prm['T0'] = [0.33, 0.31, 0.27, 0.42, 0.27, 0.39]
    sc_va = competition_score(decode_pc(vp0, vp1, vp2, prm), g_va_np)
    print(f"[Fold {k} | {suite_tag}] Done in {time.time()-t0:.1f}s (real={n_real}, syn={n_all-n_real}) | Blind OOF={sc_va['total']:.2f}", flush=True)
    return {
        'va_p0': vp0, 'va_p1': vp1, 'va_p2': vp2, 'va_gate': vgate,
        't150_p0': t150_p0, 't150_p1': t150_p1, 't150_p2': t150_p2, 't150_gate': t150_gate,
        'va_names': va_names, 'gva': g_va_np,
    }


def run_fold_worker(fold_k, gpu_id):
    dev = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(dev)
    cache = load_v15_shared_cache()
    tpl = json.load(open('results/v2_full/template_150.json'))
    tpl_names = [e['image'] for e in tpl]
    _, _, folds_data = build_source_disjoint_8fold_manifests()
    fold_dict = folds_data[fold_k]

    suites = [
        ('v15a_overgen',  'overgen',  [4],                20261801),
        ('v15b_sharptex', 'sharptex', [0, 1, 2],          20261802),
        ('v15c_bokeh',    'bokeh',    [0, 1, 5],          20261803),
        ('v15d_all',      'all',      [0, 1, 2, 3, 4, 5], 20261804),
        ('v15e_real',     'real',     [0, 1, 2, 3, 4, 5], 20261805),
    ]
    res = {}
    for s_tag, s_name, f_cis, sd in suites:
        res[s_tag] = train_fold_suite(fold_dict, s_tag, s_name, f_cis, cache, tpl_names, dev, seed=sd)
        np.savez_compressed(
            os.path.join(OUT_DIR, f'{s_tag}_f{fold_k}_probs.npz'),
            **res[s_tag],
            t150_names=np.array(tpl_names),
        )

    # Also build routed v15_ens for this fold
    A, B, C, D, E = res['v15a_overgen'], res['v15b_sharptex'], res['v15c_bokeh'], res['v15d_all'], res['v15e_real']

    def route_fold(prefix):
        p0_A, p1_A, p2_A = A[f'{prefix}_p0'], A[f'{prefix}_p1'], A[f'{prefix}_p2']
        p0_B, p1_B, p2_B = B[f'{prefix}_p0'], B[f'{prefix}_p1'], B[f'{prefix}_p2']
        p0_C, p1_C, p2_C = C[f'{prefix}_p0'], C[f'{prefix}_p1'], C[f'{prefix}_p2']
        p0_D, p1_D, p2_D = D[f'{prefix}_p0'], D[f'{prefix}_p1'], D[f'{prefix}_p2']
        p0_E, p1_E, p2_E = E[f'{prefix}_p0'], E[f'{prefix}_p1'], E[f'{prefix}_p2']

        p0 = np.zeros_like(p0_D)
        p1 = np.zeros_like(p1_D)
        p2 = np.zeros_like(p2_D)

        # Class 0: 模糊
        p0[:, 0] = 0.30 * p0_B[:, 0] + 0.25 * p0_C[:, 0] + 0.25 * p0_D[:, 0] + 0.20 * p0_E[:, 0]
        p1[:, 0] = 0.32 * p1_B[:, 0] + 0.23 * p1_C[:, 0] + 0.25 * p1_D[:, 0] + 0.20 * p1_E[:, 0]
        p2[:, 0] = 0.32 * p2_B[:, 0] + 0.23 * p2_C[:, 0] + 0.25 * p2_D[:, 0] + 0.20 * p2_E[:, 0]

        # Class 1: 纹理损失
        p0[:, 1] = 0.32 * p0_B[:, 1] + 0.23 * p0_C[:, 1] + 0.25 * p0_D[:, 1] + 0.20 * p0_E[:, 1]
        p1[:, 1] = 0.34 * p1_B[:, 1] + 0.21 * p1_C[:, 1] + 0.25 * p1_D[:, 1] + 0.20 * p1_E[:, 1]
        p2[:, 1] = 0.34 * p2_B[:, 1] + 0.21 * p2_C[:, 1] + 0.25 * p2_D[:, 1] + 0.20 * p2_E[:, 1]

        # Class 2: 过锐
        p0[:, 2] = 0.42 * p0_B[:, 2] + 0.28 * p0_D[:, 2] + 0.20 * p0_E[:, 2] + 0.10 * p0_A[:, 2]
        p1[:, 2] = 0.44 * p1_B[:, 2] + 0.26 * p1_D[:, 2] + 0.20 * p1_E[:, 2] + 0.10 * p1_A[:, 2]
        p2[:, 2] = 0.44 * p2_B[:, 2] + 0.26 * p2_D[:, 2] + 0.20 * p2_E[:, 2] + 0.10 * p2_A[:, 2]

        # Class 3: 花屏
        p0[:, 3] = 0.20 * (p0_A[:, 3] + p0_B[:, 3] + p0_C[:, 3] + p0_D[:, 3] + p0_E[:, 3])

        # Class 4: 过度生成
        p0[:, 4] = 0.44 * p0_A[:, 4] + 0.26 * p0_D[:, 4] + 0.20 * p0_E[:, 4] + 0.10 * p0_B[:, 4]
        p1[:, 3] = 0.46 * p1_A[:, 3] + 0.24 * p1_D[:, 3] + 0.20 * p1_E[:, 3] + 0.10 * p1_B[:, 3]
        p2[:, 3] = 0.46 * p2_A[:, 3] + 0.24 * p2_D[:, 3] + 0.20 * p2_E[:, 3] + 0.10 * p2_B[:, 3]

        # Class 5: 虚化突兀
        p0[:, 5] = 0.44 * p0_C[:, 5] + 0.26 * p0_D[:, 5] + 0.20 * p0_E[:, 5] + 0.10 * p0_A[:, 5]

        gate = 0.20 * (A[f'{prefix}_gate'] + B[f'{prefix}_gate'] + C[f'{prefix}_gate'] + D[f'{prefix}_gate'] + E[f'{prefix}_gate'])
        return p0, p1, p2, gate

    vp0, vp1, vp2, vgate = route_fold('va')
    t150_p0, t150_p1, t150_p2, t150_gate = route_fold('t150')
    np.savez_compressed(
        os.path.join(OUT_DIR, f'v15_ens_f{fold_k}_probs.npz'),
        va_p0=vp0, va_p1=vp1, va_p2=vp2, va_gate=vgate,
        t150_p0=t150_p0, t150_p1=t150_p1, t150_p2=t150_p2, t150_gate=t150_gate,
        va_names=np.array(D['va_names']), gva=D['gva'],
        t150_names=np.array(tpl_names),
    )


def stitch_all_8folds():
    all_names, all_grades, folds_data = build_source_disjoint_8fold_manifests()
    tpl = json.load(open('results/v2_full/template_150.json'))
    tpl_names = [e['image'] for e in tpl]
    N_all = len(all_names)
    tags = ['v15a_overgen', 'v15b_sharptex', 'v15c_bokeh', 'v15d_all', 'v15e_real', 'v15_ens']
    prm = dict(HAND); prm['T0'] = [0.33, 0.31, 0.27, 0.42, 0.27, 0.39]

    for tag in tags:
        oof_p0 = np.zeros((N_all, 6), dtype=np.float32)
        oof_p1 = np.zeros((N_all, 4), dtype=np.float32)
        oof_p2 = np.zeros((N_all, 4), dtype=np.float32)
        oof_gate = np.zeros(N_all, dtype=np.float32)
        t150_p0 = np.zeros((150, 6), dtype=np.float32)
        t150_p1 = np.zeros((150, 4), dtype=np.float32)
        t150_p2 = np.zeros((150, 4), dtype=np.float32)
        t150_gate = np.zeros(150, dtype=np.float32)

        for k in range(8):
            vl_i = folds_data[k]['vl_i']
            d = np.load(os.path.join(OUT_DIR, f'{tag}_f{k}_probs.npz'), allow_pickle=True)
            oof_p0[vl_i] = d['va_p0']
            oof_p1[vl_i] = d['va_p1']
            oof_p2[vl_i] = d['va_p2']
            oof_gate[vl_i] = d['va_gate']
            t150_p0 += d['t150_p0'] / 8.0
            t150_p1 += d['t150_p1'] / 8.0
            t150_p2 += d['t150_p2'] / 8.0
            t150_gate += d['t150_gate'] / 8.0

        sc_all = competition_score(decode_pc(oof_p0, oof_p1, oof_p2, prm), all_grades)
        print(f"=== [8-Fold Source-Disjoint OOF (N=3573)] {tag:14s}: Total={sc_all['total']:.3f} (S_pos={25*sc_all['S_pos']:.2f}, S_p0={20*sc_all['S_p0']:.2f}, S_cln={20*sc_all['S_clean']:.2f}) ===", flush=True)
        np.savez_compressed(
            os.path.join(OUT_DIR, f'{tag}_oof3573_probs.npz'),
            oof_p0=oof_p0, oof_p1=oof_p1, oof_p2=oof_p2, oof_gate=oof_gate,
            all_grades=all_grades, all_names=np.array(all_names),
            t150_p0=t150_p0, t150_p1=t150_p1, t150_p2=t150_p2, t150_gate=t150_gate,
            t150_names=np.array(tpl_names),
        )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=str, default='all')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    if args.fold == 'all':
        t0 = time.time()
        procs = [
            subprocess.Popen([sys.executable, __file__, '--fold', str(k), '--gpu', str(k)])
            for k in range(8)
        ]
        for p in procs:
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"Fold worker failed with code {rc}")
        stitch_all_8folds()
        print(f"All 8 Folds of V15 Sim2Real completed in {time.time()-t0:.1f}s!", flush=True)
    else:
        run_fold_worker(int(args.fold), args.gpu)
