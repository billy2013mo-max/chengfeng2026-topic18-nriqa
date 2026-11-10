#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/qingyu/train_tier2_internvit6b_8fold.py
============================================
Stage 2B (Sections 3.5, 5.1, 5.2, 7.2 of Blueprint):
Pure-Real 8-Fold Parallel Trainer for Tier-2 `InternViT-6B-448px-V2.5` (`6.08B`).
"""
import os
import sys
import json
import time
import shutil
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from qingyu.config_and_paths import (
    QINGYU_CKPT_DIR,
    REPO_ROOT,
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
)
from qingyu.unimodal_xbm_losses import QingyuUnimodalCompositeLoss
from qingyu.backbones_qingyu import QingyuVisionQualityTower
from qingyu.certainty_gated_stacker import compute_ece_6x4, simplex_6x4_to_p012
from qingyu.dfw_bayesian_decoder import DampedFrankWolfeBayesianDecoder
from qingyu.train_tier1_5backbones_8fold import (
    Tier1DualEngineDataset,
    load_fold_rows,
    make_tier1_collate_fn,
    forward_batch_on_device,
    populate_prototype_bank,
    predict_2x_tta,
)


def train_internvit6b_fold(args) -> str:
    setup_sys_path_and_cwd(chdir_repo=True)
    enforce_deterministic_runtime(args.seed + args.fold)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    split_csv = os.path.join(args.split_dir, f"fold_{args.fold}.csv")
    tr_rows, va_rows = load_fold_rows(split_csv)

    teacher_map = {}
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
        backbone_key="internvit_6b",
        phys_dim=PHYS_FEAT_DIM,
        use_lora_6b=True,
        lora_rank=args.lora_rank,
    ).to(device)

    tr_phys_mat = np.stack(
        [meta_store.get(str(r["name"]), pil_im=img_loader.load_pil(str(r["name"])) if str(r["name"]) not in meta_store.map else None)["phys_vec"] for r in tr_rows],
        axis=0,
    )
    model.head.set_physics_scaler(tr_phys_mat.mean(axis=0), tr_phys_mat.std(axis=0) + 1e-5)

    collate_fn = make_tier1_collate_fn("multi_crop_448")
    ds_tr = Tier1DualEngineDataset(
        tr_rows, img_loader, meta_store, "multi_crop_448", train=True, teacher_oof_map=teacher_map
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

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr_lora, "weight_decay": 0.01},
            {"params": head_params, "lr": args.lr_head, "weight_decay": 0.01},
        ]
    )
    eff_steps_per_ep = max(1, (len(dl_tr) + max(1, args.grad_accum) - 1) // max(1, args.grad_accum))
    total_steps = max(1, eff_steps_per_ep * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr_lora, args.lr_head], total_steps=total_steps, pct_start=0.10
    )
    criterion = QingyuUnimodalCompositeLoss(
        soft_sigma=0.35, lambda_rps=0.45, lambda_rank=0.25, lambda_delta=1e-2, xbm_capacity=512
    ).to(device)

    out_dir = os.path.join(REPO_ROOT, args.out_dir, "internvit_6b")
    ckpt_dir = QINGYU_CKPT_DIR / "internvit_6b"
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = os.path.join(out_dir, f"fold_{args.fold}_best.pt")

    print(
        f"=== [Qingyu Tier-2 InternViT-6B] Fold {args.fold}/8 | train={len(tr_rows)} val={len(va_rows)} "
        f"| bs={args.batch_size}x{args.grad_accum} | LoRA_r={args.lora_rank} | device={device} ===",
        flush=True,
    )

    best_obj = -1e9
    best_sd = None
    swa_sds = []
    eval_bs = max(1, args.batch_size)

    for ep in range(args.epochs):
        model.train()
        criterion.train()
        t0 = time.time()
        loss_acc = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for step_i, b in enumerate(dl_tr):
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
                loss = losses["total"] / float(args.grad_accum)

            loss.backward()
            loss_acc += float(losses["total"].item())
            n_batches += 1

            if (step_i + 1) % args.grad_accum == 0 or (step_i + 1) == len(dl_tr):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        n_proto = populate_prototype_bank(
            model, tr_rows, img_loader, meta_store, collate_fn, device, batch_size=eval_bs, workers=args.workers
        )
        vp, vqc, gv, _ = predict_2x_tta(
            model, va_rows, img_loader, meta_store, collate_fn, device, batch_size=eval_bs, workers=args.workers
        )
        qdec = DampedFrankWolfeBayesianDecoder(dfw_steps=20, eta_doc1=0.50, max_clean_p0_fpr=0.008).fit(vp, vqc, gv)
        ev = evaluate_qingyu_dual_metrics(qdec.decode(vp, vqc), gv)
        ece = compute_ece_6x4(vp, gv)
        val_obj = 0.5 * ev["doc1_total"] + 0.5 * ev["doc2_strict_total"] - 2.0 * ece

        print(
            f"  [InternViT-6B F{args.fold}] Ep {ep+1}/{args.epochs} | loss={loss_acc/max(1,n_batches):.4f} | "
            f"Doc1={ev['doc1_total']:.2f} | Doc2={ev['doc2_strict_total']:.2f} | ECE={ece:.4f} | protos={n_proto} ({time.time()-t0:.0f}s)",
            flush=True,
        )

        cur_sd = model.trainable_state_dict()
        if ep >= max(0, args.epochs - 2):
            swa_sds.append(cur_sd)
        if val_obj > best_obj:
            best_obj = val_obj
            best_sd = cur_sd
            torch.save({"state_dict": best_sd, "best_val_score": float(val_obj)}, ckpt_best)

    if best_sd is not None and len(swa_sds) >= 2:
        cands = swa_sds + [best_sd]
        avg_sd = {
            k: ((sum(sd[k].float() for sd in cands) / float(len(cands))).to(best_sd[k].dtype) if best_sd[k].is_floating_point() else best_sd[k])
            for k in best_sd.keys()
        }
        model.load_state_dict(avg_sd, strict=False)
        populate_prototype_bank(model, tr_rows, img_loader, meta_store, collate_fn, device, batch_size=eval_bs, workers=args.workers)
        vp_s, vqc_s, gv_s, _ = predict_2x_tta(model, va_rows, img_loader, meta_store, collate_fn, device, batch_size=eval_bs, workers=args.workers)
        qdec_s = DampedFrankWolfeBayesianDecoder(dfw_steps=20, eta_doc1=0.50, max_clean_p0_fpr=0.008).fit(vp_s, vqc_s, gv_s)
        ev_s = evaluate_qingyu_dual_metrics(qdec_s.decode(vp_s, vqc_s), gv_s)
        ece_s = compute_ece_6x4(vp_s, gv_s)
        sc_s = 0.5 * ev_s["doc1_total"] + 0.5 * ev_s["doc2_strict_total"] - 2.0 * ece_s
        if sc_s >= best_obj - 0.10:
            best_obj = max(best_obj, sc_s)
            best_sd = model.trainable_state_dict()

    if best_sd is not None:
        model.load_state_dict(best_sd, strict=False)
        populate_prototype_bank(model, tr_rows, img_loader, meta_store, collate_fn, device, batch_size=eval_bs, workers=args.workers)
        final_payload = {"state_dict": model.trainable_state_dict(), "best_val_score": float(best_obj)}
        torch.save(final_payload, ckpt_best)
        shutil.copy2(ckpt_best, str(ckpt_dir / f"fold_{args.fold}_best.pt"))

    va_probs, va_qclean, gva, va_names = predict_2x_tta(
        model, va_rows, img_loader, meta_store, collate_fn, device, batch_size=eval_bs, workers=args.workers
    )
    va_p0, va_p1, va_p2 = simplex_6x4_to_p012(va_probs)

    ordered_150 = load_test_template_names("results/v2_full/template_150.json")
    t150_rows = [{"name": nm, "group_id": nm, "grades": [0] * 6} for nm in ordered_150]
    t150_probs, t150_qclean, _, _ = predict_2x_tta(
        model, t150_rows, img_loader, meta_store, collate_fn, device, batch_size=eval_bs, workers=args.workers
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
            model, ext_rows, img_loader, meta_store, collate_fn, device, batch_size=eval_bs, workers=args.workers
        )
        save_dict["extra_probs_6x4"] = ext_probs
        save_dict["extra_qclean"] = ext_qclean
        save_dict["extra_names"] = np.array(ext_names)

    out_npz = os.path.join(out_dir, f"fold_{args.fold}_probs.npz")
    np.savez_compressed(out_npz, **save_dict)
    print(f"[Qingyu Tier-2] Saved InternViT-6B Fold {args.fold} predictions -> {out_npz}", flush=True)
    return out_npz


def aggregate_internvit6b_8fold(out_dir: str, n_folds: int = 8) -> str:
    setup_sys_path_and_cwd(chdir_repo=True)
    b_dir = os.path.join(REPO_ROOT, out_dir, "internvit_6b")
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
        f"=== [Qingyu Tier-2 InternViT-6B 8-Fold OOF Summary] (N={len(oof_gold)}) | "
        f"Doc1={ev['doc1_total']:.3f} | Doc2_Strict={ev['doc2_strict_total']:.3f} | ECE={ece:.4f} ===",
        flush=True,
    )

    agg_dict = dict(
        backbone="internvit_6b",
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

    agg_path = os.path.join(REPO_ROOT, out_dir, "tier2_internvit_6b_8fold_oof.npz")
    np.savez_compressed(agg_path, **agg_dict)
    with open(agg_path.replace(".npz", ".json"), "w", encoding="utf-8") as jf:
        json.dump({"backbone": "internvit_6b", "oof_metrics": ev, "ece": ece}, jf, ensure_ascii=False, indent=2)
    return agg_path


def main():
    ap = argparse.ArgumentParser(description="Qingyu Stage-2B Tier-2 InternViT-6B 8-Fold Pure-Real Trainer")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split-dir", default="data/qingyu_splits")
    ap.add_argument("--dual-engine-npz", default="results/qingyu/dual_engine_meta.npz")
    ap.add_argument("--out-dir", default="results/qingyu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=3)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr-lora", type=float, default=1.2e-4)
    ap.add_argument("--lr-head", type=float, default=6e-4)
    ap.add_argument("--oof-teacher-npz", default="")
    ap.add_argument("--lambda-kd", type=float, default=0.25)
    ap.add_argument("--extra-test-template", default="")
    ap.add_argument("--extra-test-dir", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--aggregate-8fold", action="store_true")
    args = ap.parse_args()

    if args.aggregate_8fold:
        aggregate_internvit6b_8fold(args.out_dir)
    else:
        train_internvit6b_fold(args)


if __name__ == "__main__":
    main()
