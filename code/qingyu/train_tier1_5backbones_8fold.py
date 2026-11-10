#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/train_tier1_5backbones_8fold.py
===========================================
Stage 2A (Sections 3.5, 5.1, 5.2, 7.2 of Blueprint):
Pure-Real 8-Fold Parallel Trainer for All 5 Tier-1 Lightweight & Native-Aspect Backbones (`1.96B` total):
  - `convnextv2_base`       (0.089B | 5 views: global 448 + 4 deterministic 1:1 448x448 physics crops)
  - `pe_core_large`         (0.350B | 5 views: global 448 + 4 deterministic 1:1 448x448 crops, Layers 16/20/23 readout)
  - `dinov3_vitl16`         (0.330B | Native-aspect 2D-RoPE global ~784 patches + 2 key 1:1 448x448 crops)
  - `siglip2_so400m_naflex` (0.430B | Native variable-aspect NaFlex max_num_patches=784 + 2 key 1:1 crops)
  - `cradio_v4_h`           (0.670B | Native-aspect multi-teacher global + 2 key 1:1 448x448 halo/tex crops)

Execution Modes:
  1. Single-Fold on Dedicated GPU (`--backbone pe_core_large --fold 0 --gpu 0`):
     Allows launching all 8 folds simultaneously across 8xH200 GPUs with zero DDP communication overhead!
  2. Full 8-Fold OOF Aggregator (`--backbone pe_core_large --aggregate-8fold`):
     Merges `fold_0..7` OOF predictions (`N = 3,573`), `test_150` predictions, and any custom `--extra-test-template`
     (`654` final images) into `results/qingyu/tier1_<backbone>_8fold_oof.npz`.

Strict Pure-Real Discipline:
  - ZERO synthetic images (`_NatNeg`, `_prism_`, `synth_*` are never loaded).
  - Conservative label-preserving augmentations ONLY (random horizontal flip + tiny +-5% brightness/contrast).
  - Fits `TipAdapterPrototypeBank` (`128` pure-real prototypes) on the training fold before final validation/test TTA!
"""
import os
import sys
import csv
import json
import time
import random
import argparse
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.config_and_paths import (
    REPO_ROOT,
    DIMS,
    LosslessNativeImageLoader,
    enforce_deterministic_runtime,
    evaluate_qingyu_dual_metrics,
    load_test_template_names,
    setup_sys_path_and_cwd,
)
from qingyu.dual_engine_preprocess import (
    PHYS_FEAT_DIM,
    DualEnginePhysicsExtractor,
    PrecomputedDualEngineStore,
    extract_1to1_crop_pil,
    resize_native_aspect_pil,
)
from qingyu.unimodal_xbm_losses import (
    QingyuUnimodalCompositeLoss,
    hash_group_id_int64,
)
from qingyu.backbones_qingyu import QingyuVisionQualityTower
from qingyu.certainty_gated_stacker import compute_ece_6x4, simplex_6x4_to_p012
from qingyu.dfw_bayesian_decoder import DampedFrankWolfeBayesianDecoder


def load_fold_rows(split_csv: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    full_p = split_csv if os.path.isabs(split_csv) else os.path.join(REPO_ROOT, split_csv)
    tr, va = [], []
    with open(full_p, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = {
                "name": r["name"].strip(),
                "group_id": r.get("group_id", r["name"].split("_")[0]).strip(),
                "grades": [int(r[c]) for c in DIMS],
            }
            if r["split"].strip() == "val":
                va.append(item)
            else:
                tr.append(item)
    return tr, va


class Tier1DualEngineDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        img_loader: LosslessNativeImageLoader,
        meta_store: PrecomputedDualEngineStore,
        mode_tag: str,
        train: bool = False,
        hflip_tta: bool = False,
        teacher_oof_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.rows = list(rows)
        self.loader = img_loader
        self.meta_store = meta_store
        self.mode_tag = mode_tag
        self.train = train
        self.hflip_tta = hflip_tta
        self.teacher_oof_map = teacher_oof_map or {}

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _pil_to_chw01(im: Image.Image) -> torch.Tensor:
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy())

    def __getitem__(self, idx: int) -> Dict[str, object]:
        r = self.rows[idx]
        nm = str(r["name"])
        im = self.loader.load_pil(nm)
        meta = self.meta_store.get(nm, pil_im=im)
        orig_w, _ = im.size

        do_flip = self.hflip_tta or (self.train and random.random() < 0.5)
        c_boxes = meta["crops_448_boxes"].copy()
        if do_flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            # Mirror crop boxes horizontally so guided crops track exact flipped physical peaks!
            old_x0 = c_boxes[:, 0].copy()
            old_x1 = c_boxes[:, 2].copy()
            c_boxes[:, 0] = np.maximum(0, orig_w - old_x1)
            c_boxes[:, 2] = c_boxes[:, 0] + 448

        if self.train:
            # Conservative photometric jitter ONLY (never alters sharpness, ringing, or compression!)
            if random.random() < 0.35:
                im = ImageEnhance.Brightness(im).enhance(1.0 + random.uniform(-0.05, 0.05))
            if random.random() < 0.35:
                im = ImageEnhance.Contrast(im).enhance(1.0 + random.uniform(-0.05, 0.05))

        phys = torch.from_numpy(np.asarray(meta["phys_vec"], dtype=np.float32).copy())
        grades = torch.tensor(r["grades"], dtype=torch.long)
        ghash = hash_group_id_int64(str(r.get("group_id", nm)))
        t_prob = self.teacher_oof_map.get(nm, None)
        t_tensor = torch.from_numpy(t_prob.copy()) if t_prob is not None else torch.zeros((6, 4), dtype=torch.float32)
        has_t = 1 if t_prob is not None else 0

        if self.mode_tag == "naflex_dict":
            # Return PIL views `[global_native, crop_focus_448, crop_meor_448]` for NaFlex processor in collate_fn
            im_nat = resize_native_aspect_pil(im, max_pixels=784 * 256, min_pixels=256 * 256, patch_mult=16)
            c0 = extract_1to1_crop_pil(im, c_boxes[0], 448)
            c1 = extract_1to1_crop_pil(im, c_boxes[1], 448)
            return {
                "pil_views": [im_nat, c0, c1],
                "phys": phys,
                "grades": grades,
                "ghash": ghash,
                "name": nm,
                "teacher_6x4": t_tensor,
                "has_teacher": has_t,
            }

        if self.mode_tag == "native_aspect_plus_crops":
            # Native aspect global (~784 patches of 16x16 = 200,704 pixels) + 2 key 1:1 448x448 crops
            im_nat = resize_native_aspect_pil(im, max_pixels=784 * 256, min_pixels=256 * 256, patch_mult=16)
            nat_t = self._pil_to_chw01(im_nat)
            # Pick [focus, cutout] for DINOv3, [meor, tex] or [focus, meor] -> include 3 guided crops for full coverage
            crops_t = torch.stack([self._pil_to_chw01(extract_1to1_crop_pil(im, c_boxes[k], 448)) for k in (0, 1, 2)], dim=0)
            return {
                "nat_global": nat_t,
                "views_01": crops_t,
                "phys": phys,
                "grades": grades,
                "ghash": ghash,
                "name": nm,
                "teacher_6x4": t_tensor,
                "has_teacher": has_t,
            }

        # Default `multi_crop_448` (ConvNeXtV2-Base & PE-Core-Large & InternViT-6B): global 448 + all 4 1:1 448 crops
        g_448 = im.resize((448, 448), Image.BICUBIC)
        views = [g_448] + [extract_1to1_crop_pil(im, c_boxes[k], 448) for k in range(4)]
        views_t = torch.stack([self._pil_to_chw01(v) for v in views], dim=0)
        return {
            "views_01": views_t,
            "phys": phys,
            "grades": grades,
            "ghash": ghash,
            "name": nm,
            "teacher_6x4": t_tensor,
            "has_teacher": has_t,
        }


def make_tier1_collate_fn(mode_tag: str, naflex_processor=None, max_num_patches: int = 784):
    def _collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        out: Dict[str, object] = {
            "phys": torch.stack([b["phys"] for b in batch], dim=0),
            "grades": torch.stack([b["grades"] for b in batch], dim=0),
            "ghash": torch.tensor([b["ghash"] for b in batch], dtype=torch.long),
            "names": [b["name"] for b in batch],
            "teacher_6x4": torch.stack([b["teacher_6x4"] for b in batch], dim=0),
            "has_teacher": all(b["has_teacher"] == 1 for b in batch),
        }
        if mode_tag == "naflex_dict":
            n_views = len(batch[0]["pil_views"])
            naflex_views = []
            for vi in range(n_views):
                ims_v = [b["pil_views"][vi] for b in batch]
                enc = naflex_processor(images=ims_v, return_tensors="pt", max_num_patches=max_num_patches)
                naflex_views.append(dict(enc))
            out["naflex_views"] = naflex_views
        elif mode_tag == "native_aspect_plus_crops":
            out["nat_global"] = [b["nat_global"].unsqueeze(0) for b in batch]
            out["views_01"] = torch.stack([b["views_01"] for b in batch], dim=0)
        else:
            out["views_01"] = torch.stack([b["views_01"] for b in batch], dim=0)
        return out

    return _collate


def forward_batch_on_device(model: QingyuVisionQualityTower, batch: Dict[str, object], device: torch.device):
    phys = batch["phys"].to(device, non_blocking=True)
    if model.mode_tag == "naflex_dict":
        nv = [{k: v.to(device, non_blocking=True) for k, v in d.items()} for d in batch["naflex_views"]]
        return model(phys_feat=phys, naflex_views_list=nv)
    if model.mode_tag == "native_aspect_plus_crops":
        ng = [t.to(device, non_blocking=True) for t in batch["nat_global"]]
        v01 = batch["views_01"].to(device, non_blocking=True)
        return model(views_01=v01, phys_feat=phys, native_global_list=ng)
    v01 = batch["views_01"].to(device, non_blocking=True)
    return model(views_01=v01, phys_feat=phys)


@torch.no_grad()
def populate_prototype_bank(
    model: QingyuVisionQualityTower,
    tr_rows: Sequence[Dict[str, object]],
    img_loader: LosslessNativeImageLoader,
    meta_store: PrecomputedDualEngineStore,
    collate_fn,
    device: torch.device,
    batch_size: int = 16,
    workers: int = 4,
) -> int:
    model.eval()
    model.head.proto_bank.num_active.fill_(0)
    ds = Tier1DualEngineDataset(tr_rows, img_loader, meta_store, model.mode_tag, train=False, hflip_tta=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    zs, ys, ps = [], [], []
    for b in dl:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = forward_batch_on_device(model, b, device)
        zs.append(out["z_norm"].float().cpu().numpy())
        ys.append(b["grades"].numpy())
        ps.append(b["phys"].numpy())
    return model.head.proto_bank.fit_from_train_embeddings(
        np.concatenate(zs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(ps, axis=0),
    )


@torch.no_grad()
def predict_2x_tta(
    model: QingyuVisionQualityTower,
    rows: Sequence[Dict[str, object]],
    img_loader: LosslessNativeImageLoader,
    meta_store: PrecomputedDualEngineStore,
    collate_fn,
    device: torch.device,
    batch_size: int = 16,
    workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    probs_passes = []
    qclean_passes = []
    gold_out = None
    names_out: List[str] = []

    for flip_flag in (False, True):
        ds = Tier1DualEngineDataset(rows, img_loader, meta_store, model.mode_tag, train=False, hflip_tta=flip_flag)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
        p_list, qc_list, g_list, n_list = [], [], [], []
        for b in dl:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = forward_batch_on_device(model, b, device)
            p_list.append(out["probs_6x4"].float().cpu().numpy())
            qc_list.append(torch.sigmoid(out["clean_logit"].float()).cpu().numpy())
            if not flip_flag:
                g_list.append(b["grades"].numpy())
                n_list.extend(b["names"])
        probs_passes.append(np.concatenate(p_list, axis=0))
        qclean_passes.append(np.concatenate(qc_list, axis=0))
        if not flip_flag:
            gold_out = np.concatenate(g_list, axis=0)
            names_out = n_list

    probs_avg = 0.5 * (probs_passes[0] + probs_passes[1])
    qclean_avg = 0.5 * (qclean_passes[0] + qclean_passes[1])
    return probs_avg.astype(np.float32), qclean_avg.astype(np.float32), gold_out, names_out


def train_single_fold(args) -> str:
    setup_sys_path_and_cwd(chdir_repo=True)
    enforce_deterministic_runtime(args.seed + args.fold)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    split_csv = os.path.join(args.split_dir, f"fold_{args.fold}.csv")
    tr_rows, va_rows = load_fold_rows(split_csv)

    # Optional Cross-Modal OOF Mutual Distillation Teacher (strictly out-of-fold probabilities on `tr_rows` ONLY!)
    teacher_map: Dict[str, np.ndarray] = {}
    if args.oof_teacher_npz and os.path.exists(args.oof_teacher_npz):
        td = np.load(args.oof_teacher_npz, allow_pickle=True)
        t_names = [str(x) for x in td["oof_names"]]
        t_probs = np.asarray(td["oof_probs_6x4"], dtype=np.float32)
        va_set = set(r["name"] for r in va_rows)
        for i, nm in enumerate(t_names):
            if nm not in va_set:
                teacher_map[nm] = t_probs[i]

    extra_dirs = [args.extra_test_dir] if args.extra_test_dir else None
    img_loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=extra_dirs)
    lazy_ext = DualEnginePhysicsExtractor(device=str(device), allow_depth_download=True)
    meta_store = PrecomputedDualEngineStore(args.dual_engine_npz, lazy_extractor=lazy_ext)

    model = QingyuVisionQualityTower(
        backbone_key=args.backbone,
        phys_dim=PHYS_FEAT_DIM,
        unfreeze_last_k=args.unfreeze_blocks,
    ).to(device)

    # Fit physics standardizer strictly on `tr_rows`!
    tr_phys_list = []
    for r in tr_rows:
        nm = str(r["name"])
        meta = meta_store.get(nm, pil_im=img_loader.load_pil(nm) if nm not in meta_store.map else None)
        tr_phys_list.append(meta["phys_vec"])
    tr_phys_mat = np.stack(tr_phys_list, axis=0)
    model.head.set_physics_scaler(tr_phys_mat.mean(axis=0), tr_phys_mat.std(axis=0) + 1e-5)

    naflex_proc = getattr(model, "naflex_processor", None)
    collate_fn = make_tier1_collate_fn(model.mode_tag, naflex_processor=naflex_proc, max_num_patches=784)

    ds_tr = Tier1DualEngineDataset(
        tr_rows, img_loader, meta_store, model.mode_tag, train=True, teacher_oof_map=teacher_map
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": 0.02},
            {"params": head_params, "lr": args.lr_head, "weight_decay": 0.01},
        ]
    )
    total_steps = max(1, len(dl_tr) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone, args.lr_head],
        total_steps=total_steps,
        pct_start=0.10,
    )
    criterion = QingyuUnimodalCompositeLoss(
        soft_sigma=0.35, lambda_rps=0.45, lambda_rank=0.25, lambda_delta=1e-2, xbm_capacity=512
    ).to(device)

    out_dir = os.path.join(REPO_ROOT, args.out_dir, args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_best = os.path.join(out_dir, f"fold_{args.fold}_best.pt")

    print(
        f"=== [Qingyu Tier-1] {args.backbone} | Fold {args.fold}/8 | train={len(tr_rows)} val={len(va_rows)} "
        f"| mode={model.mode_tag} | device={device} ===",
        flush=True,
    )

    best_obj = -1e9
    best_sd = None
    swa_sds: List[Dict[str, torch.Tensor]] = []

    for ep in range(args.epochs):
        model.train()
        criterion.train()
        t0 = time.time()
        loss_acc = 0.0
        n_steps = 0

        for b in dl_tr:
            t_kd = b["teacher_6x4"].to(device, non_blocking=True) if b["has_teacher"] else None
            lam_kd = args.lambda_kd if b["has_teacher"] else 0.0
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = forward_batch_on_device(model, b, device)
                losses = criterion(
                    out,
                    b["grades"].to(device, non_blocking=True),
                    b["ghash"].to(device, non_blocking=True),
                    teacher_probs_6x4=t_kd,
                    lambda_kd=lam_kd,
                )
                loss = losses["total"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            if scheduler.last_epoch < total_steps - 1:
                scheduler.step()

            loss_acc += float(loss.item())
            n_steps += 1

        # Refresh pure-real Tip-Adapter prototypes from last 2 epochs onwards
        if ep >= max(1, args.epochs - 3):
            n_proto = populate_prototype_bank(
                model, tr_rows, img_loader, meta_store, collate_fn, device, batch_size=args.batch_size * 2, workers=args.workers
            )
        else:
            n_proto = 0

        vp, vqc, gv, _ = predict_2x_tta(
            model, va_rows, img_loader, meta_store, collate_fn, device, batch_size=args.batch_size * 2, workers=args.workers
        )
        quick_dec = DampedFrankWolfeBayesianDecoder(dfw_steps=20, eta_doc1=0.50, max_clean_p0_fpr=0.008).fit(vp, vqc, gv)
        vpred = quick_dec.decode(vp, vqc)
        ev = evaluate_qingyu_dual_metrics(vpred, gv)
        ece = compute_ece_6x4(vp, gv)
        val_obj = 0.5 * ev["doc1_total"] + 0.5 * ev["doc2_strict_total"] - 2.0 * ece

        print(
            f"  [{args.backbone} F{args.fold}] Ep {ep+1}/{args.epochs} | loss={loss_acc/max(1,n_steps):.4f} | "
            f"Doc1={ev['doc1_total']:.2f} | Doc2={ev['doc2_strict_total']:.2f} | ECE={ece:.4f} | protos={n_proto} ({time.time()-t0:.0f}s)",
            flush=True,
        )

        cur_sd = model.trainable_state_dict()
        if ep >= max(0, args.epochs - 2):
            swa_sds.append(cur_sd)
        if val_obj > best_obj:
            best_obj = val_obj
            best_sd = cur_sd
            torch.save(best_sd, ckpt_best)

    # SWA check over final epochs + best checkpoint
    if best_sd is not None and len(swa_sds) >= 2:
        avg_sd = {}
        cands = swa_sds + [best_sd]
        for k in best_sd.keys():
            if best_sd[k].is_floating_point():
                avg_sd[k] = sum(sd[k].float() for sd in cands) / float(len(cands))
                avg_sd[k] = avg_sd[k].to(best_sd[k].dtype)
            else:
                avg_sd[k] = best_sd[k]
        model.load_state_dict(avg_sd, strict=False)
        populate_prototype_bank(model, tr_rows, img_loader, meta_store, collate_fn, device, batch_size=args.batch_size * 2, workers=args.workers)
        vp_swa, vqc_swa, gv_swa, _ = predict_2x_tta(model, va_rows, img_loader, meta_store, collate_fn, device, batch_size=args.batch_size * 2, workers=args.workers)
        qdec_s = DampedFrankWolfeBayesianDecoder(dfw_steps=20, eta_doc1=0.50, max_clean_p0_fpr=0.008).fit(vp_swa, vqc_swa, gv_swa)
        ev_s = evaluate_qingyu_dual_metrics(qdec_s.decode(vp_swa, vqc_swa), gv_swa)
        ece_s = compute_ece_6x4(vp_swa, gv_swa)
        if 0.5 * ev_s["doc1_total"] + 0.5 * ev_s["doc2_strict_total"] - 2.0 * ece_s >= best_obj - 0.08:
            best_sd = model.trainable_state_dict()
            torch.save(best_sd, ckpt_best)

    if best_sd is not None:
        model.load_state_dict(best_sd, strict=False)
        populate_prototype_bank(model, tr_rows, img_loader, meta_store, collate_fn, device, batch_size=args.batch_size * 2, workers=args.workers)
        torch.save(model.trainable_state_dict(), ckpt_best)

    # Final OOF & Test Inference (2x Flip TTA)
    va_probs, va_qclean, gva, va_names = predict_2x_tta(
        model, va_rows, img_loader, meta_store, collate_fn, device, batch_size=args.batch_size * 2, workers=args.workers
    )
    va_p0, va_p1, va_p2 = simplex_6x4_to_p012(va_probs)

    ordered_150 = load_test_template_names("results/v2_full/template_150.json")
    t150_rows = [{"name": nm, "group_id": nm, "grades": [0] * 6} for nm in ordered_150]
    t150_probs, t150_qclean, _, _ = predict_2x_tta(
        model, t150_rows, img_loader, meta_store, collate_fn, device, batch_size=args.batch_size * 2, workers=args.workers
    )
    t150_p0, t150_p1, t150_p2 = simplex_6x4_to_p012(t150_probs)

    save_dict = dict(
        fold=np.int64(args.fold),
        va_probs_6x4=va_probs,
        va_qclean=va_qclean,
        va_p0=va_p0,
        va_p1=va_p1,
        va_p2=va_p2,
        gva=gva,
        va_names=np.array(va_names),
        t150_probs_6x4=t150_probs,
        t150_qclean=t150_qclean,
        t150_p0=t150_p0,
        t150_p1=t150_p1,
        t150_p2=t150_p2,
        t150_names=np.array(ordered_150),
    )

    if args.extra_test_template and os.path.exists(args.extra_test_template):
        ext_names = load_test_template_names(args.extra_test_template)
        ext_rows = [{"name": nm, "group_id": nm, "grades": [0] * 6} for nm in ext_names]
        ext_probs, ext_qclean, _, _ = predict_2x_tta(
            model, ext_rows, img_loader, meta_store, collate_fn, device, batch_size=args.batch_size * 2, workers=args.workers
        )
        save_dict["extra_probs_6x4"] = ext_probs
        save_dict["extra_qclean"] = ext_qclean
        save_dict["extra_names"] = np.array(ext_names)

    out_npz = os.path.join(out_dir, f"fold_{args.fold}_probs.npz")
    np.savez_compressed(out_npz, **save_dict)
    print(f"[Qingyu Tier-1] Saved Fold {args.fold} OOF & test predictions -> {out_npz}", flush=True)
    return out_npz


def aggregate_8fold_oof(backbone: str, out_dir: str, n_folds: int = 8) -> str:
    setup_sys_path_and_cwd(chdir_repo=True)
    b_dir = os.path.join(REPO_ROOT, out_dir, backbone)
    parts = [np.load(os.path.join(b_dir, f"fold_{f}_probs.npz"), allow_pickle=True) for f in range(n_folds)]

    oof_names = np.concatenate([p["va_names"] for p in parts], axis=0)
    oof_probs = np.concatenate([p["va_probs_6x4"] for p in parts], axis=0)
    oof_qclean = np.concatenate([p["va_qclean"] for p in parts], axis=0)
    oof_gold = np.concatenate([p["gva"] for p in parts], axis=0)
    oof_folds = np.concatenate([np.full(len(p["va_names"]), f, dtype=np.int64) for f, p in enumerate(parts)], axis=0)

    t150_probs = np.mean([p["t150_probs_6x4"] for p in parts], axis=0).astype(np.float32)
    t150_qclean = np.mean([p["t150_qclean"] for p in parts], axis=0).astype(np.float32)
    t150_names = parts[0]["t150_names"]

    dec = DampedFrankWolfeBayesianDecoder(dfw_steps=30, eta_doc1=0.50, max_clean_p0_fpr=0.005).fit(
        oof_probs, oof_qclean, oof_gold
    )
    ev = evaluate_qingyu_dual_metrics(dec.decode(oof_probs, oof_qclean), oof_gold)
    ece = compute_ece_6x4(oof_probs, oof_gold)
    print(
        f"=== [Qingyu Tier-1 8-Fold OOF Summary] {backbone} (N={len(oof_gold)}) | "
        f"Doc1={ev['doc1_total']:.3f} | Doc2_Strict={ev['doc2_strict_total']:.3f} | ECE={ece:.4f} ===",
        flush=True,
    )

    agg_dict = dict(
        backbone=backbone,
        oof_names=oof_names,
        oof_probs_6x4=oof_probs,
        oof_qclean=oof_qclean,
        oof_gold=oof_gold,
        oof_folds=oof_folds,
        t150_names=t150_names,
        t150_probs_6x4=t150_probs,
        t150_qclean=t150_qclean,
    )
    if all("extra_probs_6x4" in p.files for p in parts):
        agg_dict["extra_names"] = parts[0]["extra_names"]
        agg_dict["extra_probs_6x4"] = np.mean([p["extra_probs_6x4"] for p in parts], axis=0).astype(np.float32)
        agg_dict["extra_qclean"] = np.mean([p["extra_qclean"] for p in parts], axis=0).astype(np.float32)

    agg_path = os.path.join(REPO_ROOT, out_dir, f"tier1_{backbone}_8fold_oof.npz")
    np.savez_compressed(agg_path, **agg_dict)
    with open(agg_path.replace(".npz", ".json"), "w", encoding="utf-8") as jf:
        json.dump({"backbone": backbone, "oof_metrics": ev, "ece": ece}, jf, ensure_ascii=False, indent=2)
    return agg_path


def main():
    ap = argparse.ArgumentParser(description="Qingyu Stage-2A Tier-1 5-Backbone 8-Fold Pure-Real Trainer")
    ap.add_argument(
        "--backbone",
        required=True,
        choices=["convnextv2_base", "pe_core_large", "dinov3_vitl16", "siglip2_so400m_naflex", "cradio_v4_h"],
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split-dir", default="data/qingyu_splits")
    ap.add_argument("--dual-engine-npz", default="results/qingyu/dual_engine_meta.npz")
    ap.add_argument("--out-dir", default="results/qingyu")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr-backbone", type=float, default=4e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--unfreeze-blocks", type=int, default=8)
    ap.add_argument("--oof-teacher-npz", default="")
    ap.add_argument("--lambda-kd", type=float, default=0.25)
    ap.add_argument("--extra-test-template", default="")
    ap.add_argument("--extra-test-dir", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--aggregate-8fold", action="store_true")
    args = ap.parse_args()

    if args.aggregate_8fold:
        aggregate_8fold_oof(args.backbone, args.out_dir)
    else:
        train_single_fold(args)


if __name__ == "__main__":
    main()
