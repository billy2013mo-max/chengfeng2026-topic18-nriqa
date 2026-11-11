#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/train_testbed.py — the R55 testbed trainer (1 GPU per fold).

It is the Tier-1 trainer (code/qingyu/train_tier1_5backbones_8fold.py: same model, loss, schedule, SWA,
prototype bank, 2x flip TTA, output files) plus four optional, independently switchable additions:

  --aux-kinds rank overgen   same-scene ladders (build_rank_tuples.py / build_overgen_ladder.py).  A whole tuple
                             (labelled seed + its 3 stronger versions) sits in one batch and gets ONLY a ranking
                             loss on the manipulated dimension.  No grade is assigned by recipe.
  --kd-synth                 additionally distil the fold-k teachers' soft label on those synthetic images.
  --aux-kinds ext            external unlabelled frames with the fold-k teachers' soft label (self-training).
  --clean-neg                external frames certified clean (certify_clean.py) as low-weight clean negatives.
  --lambda-tag X             train-only auxiliary head that predicts the processing-pipeline tag parsed from the
                             TRAINING file names.  The tag is a target, never an input; inference is unchanged.
  --train-frac F             learning curve: keep a fraction F of the training GROUPS.

With none of these flags it reproduces the Tier-1 baseline (same seed -> same numbers up to GPU nondeterminism).

Leak rules enforced here:
  * auxiliary rows derived from a seed of the validation fold are dropped (seed_fold == fold);
  * soft labels come from probs_folds[fold], i.e. models that never saw this fold's validation data.

Output: <out-dir>/<backbone>/fold_<k>_{best.pt,probs.npz}, the format common55.discover_towers() reads.
"""
import argparse
import csv
import os
import random
import re
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r55_common import AUX_CLEAN, AUX_IMG, AUX_MANIFEST, AUX_META, AUX_TEACHER, read_manifest  # noqa: E402
from qingyu.config_and_paths import (  # noqa: E402
    REPO_ROOT, LosslessNativeImageLoader, enforce_deterministic_runtime, evaluate_qingyu_dual_metrics,
    load_test_template_names, setup_sys_path_and_cwd)
from qingyu.dual_engine_preprocess import PHYS_FEAT_DIM, DualEnginePhysicsExtractor, PrecomputedDualEngineStore  # noqa: E402
from qingyu.unimodal_xbm_losses import QingyuUnimodalCompositeLoss  # noqa: E402
from qingyu.backbones_qingyu import QingyuVisionQualityTower  # noqa: E402
from qingyu.certainty_gated_stacker import compute_ece_6x4, simplex_6x4_to_p012  # noqa: E402
from qingyu.dfw_bayesian_decoder import DampedFrankWolfeBayesianDecoder  # noqa: E402
from qingyu.train_tier1_5backbones_8fold import (  # noqa: E402
    Tier1DualEngineDataset, forward_batch_on_device, load_fold_rows, make_tier1_collate_fn,
    populate_prototype_bank, predict_2x_tta)

FULL_DIMS, P0_ONLY_DIMS = (0, 1, 2, 4), (3, 5)
TAG_RE = re.compile(r"_(v\d+[A-Za-z0-9]*?)(?:\.mp4|_|\.)")


def pipeline_tag(name: str) -> str:
    m = TAG_RE.search(name)
    return m.group(1) if m else "NA"


class R55Dataset(Tier1DualEngineDataset):
    """Tier-1 item + per-row auxiliary fields (all optional; defaults are neutral)."""
    def __getitem__(self, idx):
        it = super().__getitem__(idx)
        r = self.rows[idx]
        t = r.get("teacher")
        it["aux_teacher"] = torch.from_numpy(np.asarray(t, dtype=np.float32)) if t is not None else torch.zeros(6, 4)
        it["aux_has_t"] = int(t is not None)
        it["aux_tuple"] = int(r.get("tuple_idx", -1))
        it["aux_level"] = int(r.get("level", 0))
        it["aux_dim"] = int(r.get("dim", 0))
        it["aux_clean"] = int(r.get("is_clean", 0))
        it["tag_idx"] = int(r.get("tag_idx", 0))
        return it


def make_collate(base):
    def _c(batch):
        out = base(batch)
        out["aux_teacher"] = torch.stack([b["aux_teacher"] for b in batch])
        for k in ("aux_has_t", "aux_tuple", "aux_level", "aux_dim", "aux_clean", "tag_idx"):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        return out
    return _c


class AuxBatchSampler(Sampler):
    """Endless batches: `n_tuples` whole tuples (seed + levels) followed by `n_single` single images."""
    def __init__(self, tuples: List[List[int]], singles: List[int], n_tuples: int, n_single: int, seed: int):
        self.tuples, self.singles = tuples, singles
        self.n_tuples = n_tuples if tuples else 0
        self.n_single = n_single if singles else 0
        self.rng = random.Random(seed)

    def __iter__(self):
        while True:
            b: List[int] = []
            for t in self.rng.sample(range(len(self.tuples)), min(self.n_tuples, len(self.tuples))):
                b += self.tuples[t]
            if self.n_single:
                b += self.rng.sample(self.singles, min(self.n_single, len(self.singles)))
            yield b

    def __len__(self):
        return 1 << 30


def build_aux_catalog(args, tr_rows, meta_store):
    """Returns (rows, tuples, singles) for this fold, after the leak filters."""
    man = [r for r in read_manifest(args.aux_manifest) if r["kind"] in args.aux_kinds or
           (args.clean_neg and r["kind"] == "ext")]
    man = [r for r in man if int(r["seed_fold"]) != args.fold and os.path.exists(os.path.join(args.aux_img_dir, r["name"]))]
    man = [r for r in man if r["name"] in meta_store.map]
    teacher = {}
    if args.kd_synth or "ext" in args.aux_kinds:
        z = np.load(args.aux_teacher_npz, allow_pickle=True)
        pos = {str(n): i for i, n in enumerate(z["names"])}
        pf = z["probs_folds"][args.fold]
        teacher = {n: pf[i] for n, i in pos.items()}
    clean = set()
    if args.clean_neg and os.path.exists(args.clean_list):
        clean = {r["name"] for r in csv.DictReader(open(args.clean_list, encoding="utf-8"))}
    train_by_name = {str(r["name"]): r for r in tr_rows}

    rows: List[Dict[str, object]] = []
    tuples: List[List[int]] = []
    singles: List[int] = []
    by_tuple: Dict[str, List[Dict[str, str]]] = {}
    for r in man:
        if r["kind"] in ("rank", "overgen") and r["kind"] in args.aux_kinds:
            by_tuple.setdefault(r["tuple_id"], []).append(r)
    for ti, (tid, members) in enumerate(sorted(by_tuple.items())):
        seed = members[0]["seed"]
        if seed not in train_by_name or len(members) < 2:
            continue                                   # seed must be a training image of this fold
        d = int(members[0]["dim"])
        idxs = [len(rows)]
        rows.append(dict(name=seed, group_id=seed, grades=train_by_name[seed]["grades"], tuple_idx=ti, level=0, dim=d))
        for m in sorted(members, key=lambda x: int(x["level"])):
            idxs.append(len(rows))
            rows.append(dict(name=m["name"], group_id=seed, grades=[0] * 6, tuple_idx=ti, level=int(m["level"]), dim=d,
                             teacher=teacher.get(m["name"]) if args.kd_synth else None))
        tuples.append(idxs)
    for r in man:
        if r["kind"] != "ext":
            continue
        is_c = int(r["name"] in clean)
        use_t = "ext" in args.aux_kinds and r["name"] in teacher
        if not (is_c or use_t):
            continue
        singles.append(len(rows))
        rows.append(dict(name=r["name"], group_id=r["name"], grades=[0] * 6, is_clean=is_c,
                         teacher=teacher[r["name"]] if use_t else None))
    return rows, tuples, singles


def aux_losses(out, b, device, rank_tau: float):
    """Ranking inside tuples, KD to the fold teacher, and certified-clean negatives. Returns 3 scalars."""
    logits = out["logits_6x4"].float()
    probs = out["probs_6x4"].float()
    s = out["severity_s"].float()
    zero = logits.sum() * 0.0
    tid, lv, dm = b["aux_tuple"].to(device), b["aux_level"].to(device), b["aux_dim"].to(device)

    l_rank = zero
    pair = (tid[:, None] == tid[None, :]) & (tid[:, None] >= 0) & (lv[:, None] < lv[None, :])
    if pair.any():
        i, j = pair.nonzero(as_tuple=True)            # image j is the stronger version of image i
        d = dm[j]
        exp_grade = (probs * torch.arange(4, device=device, dtype=probs.dtype)).sum(-1)   # (B, 6) in [0, 3]
        l_rank = F.softplus(-(s[j, d] - s[i, d]) / rank_tau).mean() + F.relu(exp_grade[i, d] - exp_grade[j, d]).mean()

    l_kd = zero
    has_t = b["aux_has_t"].to(device).bool()
    if has_t.any():
        tp = b["aux_teacher"].to(device).float()[has_t].clamp(1e-6, 1.0)
        lg = logits[has_t]
        terms = []
        for d in FULL_DIMS:
            q = tp[:, d] / tp[:, d].sum(-1, keepdim=True)
            terms.append(F.kl_div(F.log_softmax(lg[:, d], dim=-1), q, reduction="batchmean"))
        for d in P0_ONLY_DIMS:
            q = tp[:, d][:, [0, 3]]
            q = q / q.sum(-1, keepdim=True)
            terms.append(F.kl_div(F.log_softmax(lg[:, d][:, [0, 3]], dim=-1), q, reduction="batchmean"))
        l_kd = torch.stack(terms).mean()

    l_clean = zero
    is_c = b["aux_clean"].to(device).bool()
    if is_c.any():
        l_clean = -torch.log(probs[is_c][:, :, 0].clamp(min=1e-6)).mean() + \
            F.binary_cross_entropy_with_logits(out["clean_logit"].float()[is_c], torch.ones(int(is_c.sum()), device=device))
    return l_rank, l_kd, l_clean


def subsample_groups(tr_rows, frac: float, fold: int):
    if frac >= 0.999:
        return tr_rows
    groups = sorted({str(r["group_id"]) for r in tr_rows})
    random.Random(7000 + fold).shuffle(groups)
    keep = set(groups[: max(1, int(round(len(groups) * frac)))])
    return [r for r in tr_rows if str(r["group_id"]) in keep]


def evaluate(model, va_rows, img_loader, meta_store, collate_fn, device, args):
    vp, vqc, gv, _ = predict_2x_tta(model, va_rows, img_loader, meta_store, collate_fn, device,
                                    batch_size=args.batch_size * 2, workers=args.workers)
    dec = DampedFrankWolfeBayesianDecoder(dfw_steps=20, eta_doc1=0.50, max_clean_p0_fpr=0.008).fit(vp, vqc, gv)
    ev = evaluate_qingyu_dual_metrics(dec.decode(vp, vqc), gv)
    ece = compute_ece_6x4(vp, gv)
    return ev, ece, 0.5 * ev["doc1_total"] + 0.5 * ev["doc2_strict_total"] - 2.0 * ece


def train(args):
    setup_sys_path_and_cwd(chdir_repo=True)
    enforce_deterministic_runtime(args.seed + args.fold)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    tr_rows, va_rows = load_fold_rows(os.path.join(args.split_dir, f"fold_{args.fold}.csv"))
    tr_rows = subsample_groups(tr_rows, args.train_frac, args.fold)
    use_aux = bool(args.aux_kinds) or args.clean_neg

    extra_dirs = [d for d in (args.aux_img_dir if use_aux else "", args.extra_test_dir) if d]
    img_loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=extra_dirs or None)
    meta_store = PrecomputedDualEngineStore(args.aux_meta_npz if use_aux else args.dual_engine_npz,
                                            lazy_extractor=DualEnginePhysicsExtractor(device=str(device), allow_depth_download=True))

    model = QingyuVisionQualityTower(backbone_key=args.backbone, phys_dim=PHYS_FEAT_DIM,
                                     unfreeze_last_k=args.unfreeze_blocks).to(device)
    tr_phys = np.stack([meta_store.get(str(r["name"]))["phys_vec"] for r in tr_rows], axis=0)   # labelled rows only
    model.head.set_physics_scaler(tr_phys.mean(axis=0), tr_phys.std(axis=0) + 1e-5)

    base_collate = make_tier1_collate_fn(model.mode_tag, naflex_processor=getattr(model, "naflex_processor", None),
                                         max_num_patches=784)
    collate = make_collate(base_collate)

    tag_head = None
    if args.lambda_tag > 0:
        cnt: Dict[str, int] = {}
        for r in tr_rows:
            cnt[pipeline_tag(str(r["name"]))] = cnt.get(pipeline_tag(str(r["name"])), 0) + 1
        vocab = ["other"] + sorted(t for t, c in cnt.items() if c >= 20)
        for r in tr_rows:
            t = pipeline_tag(str(r["name"]))
            r["tag_idx"] = vocab.index(t) if t in vocab else 0
        tag_head = nn.Linear(model.head.proj_dim, len(vocab)).to(device)
        print(f"  [tag-aux] {len(vocab)} pipeline-tag classes (train-only head)", flush=True)

    dl_tr = DataLoader(R55Dataset(tr_rows, img_loader, meta_store, model.mode_tag, train=True),
                       batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate,
                       pin_memory=True, drop_last=True)

    aux_iter = None
    if use_aux:
        a_rows, a_tuples, a_singles = build_aux_catalog(args, tr_rows, meta_store)
        assert a_rows, "auxiliary flags are set but no auxiliary rows survived the filters (run prep_aux.py?)"
        sampler = AuxBatchSampler(a_tuples, a_singles, args.aux_tuples, args.aux_singles, seed=args.seed * 100 + args.fold)
        aux_iter = iter(DataLoader(R55Dataset(a_rows, img_loader, meta_store, model.mode_tag, train=True),
                                   batch_sampler=sampler, num_workers=args.aux_workers, collate_fn=collate, pin_memory=True))
        print(f"  [aux] rows={len(a_rows)} tuples={len(a_tuples)} singles={len(a_singles)} "
              f"clean_neg={sum(int(r.get('is_clean', 0)) for r in a_rows)} kinds={args.aux_kinds}", flush=True)

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    if tag_head is not None:
        head_params += list(tag_head.parameters())
    optimizer = torch.optim.AdamW([{"params": backbone_params, "lr": args.lr_backbone, "weight_decay": 0.02},
                                   {"params": head_params, "lr": args.lr_head, "weight_decay": 0.01}])
    total_steps = max(1, len(dl_tr) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr_backbone, args.lr_head],
                                                    total_steps=total_steps, pct_start=0.10)
    criterion = QingyuUnimodalCompositeLoss(soft_sigma=0.35, lambda_rps=0.45, lambda_rank=0.25, lambda_delta=1e-2,
                                            xbm_capacity=512).to(device)

    out_dir = os.path.join(REPO_ROOT, args.out_dir, args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_best = os.path.join(out_dir, f"fold_{args.fold}_best.pt")
    print(f"=== [R55 testbed] {args.backbone} | fold {args.fold} | seed {args.seed} | train={len(tr_rows)} "
          f"val={len(va_rows)} | frac={args.train_frac} | aux={use_aux} ===", flush=True)

    best_obj, best_sd, swa_sds = -1e9, None, []
    for ep in range(args.epochs):
        model.train()
        criterion.train()
        t0, acc, n_steps = time.time(), np.zeros(5), 0
        for b in dl_tr:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = forward_batch_on_device(model, b, device)
                loss = criterion(out, b["grades"].to(device), b["ghash"].to(device))["total"]
                parts = [float(loss.item()), 0.0, 0.0, 0.0, 0.0]
                if tag_head is not None:
                    l_tag = F.cross_entropy(tag_head(out["z_norm"].float()), b["tag_idx"].to(device))
                    loss = loss + args.lambda_tag * l_tag
                    parts[4] = float(l_tag.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            del out, loss
            if aux_iter is not None:
                # Second forward/backward on the auxiliary batch; gradients accumulate into the same step.
                # Doing it after the first backward keeps peak memory at max(real, aux), not their sum.
                ab = next(aux_iter)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    l_rank, l_kd, l_clean = aux_losses(forward_batch_on_device(model, ab, device), ab, device, args.rank_tau)
                    l_aux = args.lambda_tuple * l_rank + args.lambda_aux_kd * l_kd + args.lambda_clean_neg * l_clean
                l_aux.backward()
                parts[1:4] = [float(l_rank.item()), float(l_kd.item()), float(l_clean.item())]
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            if scheduler.last_epoch < total_steps - 1:
                scheduler.step()
            acc += np.array(parts)
            n_steps += 1

        n_proto = 0
        if ep >= max(1, args.epochs - 3):
            n_proto = populate_prototype_bank(model, tr_rows, img_loader, meta_store, base_collate, device,
                                              batch_size=args.batch_size * 2, workers=args.workers)
        ev, ece, val_obj = evaluate(model, va_rows, img_loader, meta_store, base_collate, device, args)
        m = acc / max(1, n_steps)
        print(f"  [{args.backbone} F{args.fold}] Ep {ep + 1}/{args.epochs} | loss={m[0]:.4f} rank={m[1]:.3f} kd={m[2]:.3f} "
              f"clean={m[3]:.3f} tag={m[4]:.3f} | Doc1={ev['doc1_total']:.2f} | Doc2={ev['doc2_strict_total']:.2f} | "
              f"ECE={ece:.4f} | protos={n_proto} ({time.time() - t0:.0f}s)", flush=True)
        cur_sd = model.trainable_state_dict()
        if ep >= max(0, args.epochs - 2):
            swa_sds.append(cur_sd)
        if val_obj > best_obj:
            best_obj, best_sd = val_obj, cur_sd
            torch.save(best_sd, ckpt_best)

    if best_sd is not None and len(swa_sds) >= 2:
        cands = swa_sds + [best_sd]
        avg = {k: (sum(sd[k].float() for sd in cands) / len(cands)).to(v.dtype) if v.is_floating_point() else v
               for k, v in best_sd.items()}
        model.load_state_dict(avg, strict=False)
        populate_prototype_bank(model, tr_rows, img_loader, meta_store, base_collate, device,
                                batch_size=args.batch_size * 2, workers=args.workers)
        _, _, obj_s = evaluate(model, va_rows, img_loader, meta_store, base_collate, device, args)
        if obj_s >= best_obj - 0.08:
            best_sd = model.trainable_state_dict()
            torch.save(best_sd, ckpt_best)
    model.load_state_dict(best_sd, strict=False)
    populate_prototype_bank(model, tr_rows, img_loader, meta_store, base_collate, device,
                            batch_size=args.batch_size * 2, workers=args.workers)
    torch.save(model.trainable_state_dict(), ckpt_best)

    bs2 = args.batch_size * 2
    va_probs, va_qclean, gva, va_names = predict_2x_tta(model, va_rows, img_loader, meta_store, base_collate, device, bs2, args.workers)
    va_p0, va_p1, va_p2 = simplex_6x4_to_p012(va_probs)
    t150 = load_test_template_names("results/v2_full/template_150.json")
    t_probs, t_qclean, _, _ = predict_2x_tta(model, [{"name": n, "group_id": n, "grades": [0] * 6} for n in t150],
                                             img_loader, meta_store, base_collate, device, bs2, args.workers)
    t_p0, t_p1, t_p2 = simplex_6x4_to_p012(t_probs)
    save = dict(fold=np.int64(args.fold), va_probs_6x4=va_probs, va_qclean=va_qclean, va_p0=va_p0, va_p1=va_p1, va_p2=va_p2,
                gva=gva, va_names=np.array(va_names), t150_probs_6x4=t_probs, t150_qclean=t_qclean, t150_p0=t_p0,
                t150_p1=t_p1, t150_p2=t_p2, t150_names=np.array(t150))
    if args.extra_test_template and os.path.exists(args.extra_test_template):
        ext = load_test_template_names(args.extra_test_template)
        e_probs, e_qc, _, _ = predict_2x_tta(model, [{"name": n, "group_id": n, "grades": [0] * 6} for n in ext],
                                             img_loader, meta_store, base_collate, device, bs2, args.workers)
        save.update(extra_probs_6x4=e_probs, extra_qclean=e_qc, extra_names=np.array(ext))
    out_npz = os.path.join(out_dir, f"fold_{args.fold}_probs.npz")
    np.savez_compressed(out_npz, **save)
    print(f"[R55 testbed] DONE fold {args.fold} -> {out_npz}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="pe_core_large",
                    choices=["convnextv2_base", "pe_core_large", "dinov3_vitl16", "siglip2_so400m_naflex", "cradio_v4_h"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", required=True, help="e.g. results/next55/towers/r55_rank_s2026")
    ap.add_argument("--split-dir", default="data/qingyu_splits")
    ap.add_argument("--dual-engine-npz", default="results/qingyu/dual_engine_meta.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr-backbone", type=float, default=4e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--unfreeze-blocks", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extra-test-template", default="")
    ap.add_argument("--extra-test-dir", default="")
    ap.add_argument("--train-frac", type=float, default=1.0)
    # auxiliary data
    ap.add_argument("--aux-kinds", nargs="*", default=[], choices=["rank", "overgen", "ext"])
    ap.add_argument("--kd-synth", action="store_true")
    ap.add_argument("--clean-neg", action="store_true")
    ap.add_argument("--aux-manifest", default=AUX_MANIFEST)
    ap.add_argument("--aux-img-dir", default=AUX_IMG)
    ap.add_argument("--aux-meta-npz", default=AUX_META)
    ap.add_argument("--aux-teacher-npz", default=AUX_TEACHER)
    ap.add_argument("--clean-list", default=AUX_CLEAN)
    ap.add_argument("--aux-tuples", type=int, default=2, help="tuples per step (4 images each)")
    ap.add_argument("--aux-singles", type=int, default=4, help="external images per step")
    ap.add_argument("--aux-workers", type=int, default=6)
    ap.add_argument("--lambda-tuple", type=float, default=0.5)
    ap.add_argument("--rank-tau", type=float, default=0.5)
    ap.add_argument("--lambda-aux-kd", type=float, default=0.5)
    ap.add_argument("--lambda-clean-neg", type=float, default=0.1)
    ap.add_argument("--lambda-tag", type=float, default=0.0)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
