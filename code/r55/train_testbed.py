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
POOL_KINDS = ("ext", "kvq", "glitch", "synth")   # single-image pools: ext = KonIQ/Inter4K photos, kvq = Kuaishou short-video frames, glitch = simulated 花屏 (build_glitch.py)
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
        it["aux_det"] = int(r.get("det", 0))                       # 1: the defect `dim` is present by construction, grade unknown
        mk = r.get("mask")
        it["aux_mask"] = torch.from_numpy(np.asarray(mk, dtype=np.float32)) if mk is not None else torch.ones(6)
        it["tag_idx"] = int(r.get("tag_idx", 0))
        return it


class CropJitterStore:
    """Training-only view of the meta store: re-draws the four guided 448-px crop boxes on every access.

    The Tier-1 dataset shows the model the SAME five views of an image in every epoch (only h-flip differs), and the
    training loss falls to ~0.05 by epoch 8 while validation ECE rises.  With probability `p_rand` a crop is replaced
    by a uniformly random native-pixel 448 box; otherwise it is shifted by up to +-`jitter` px.  Crops stay 1:1 native
    pixels, so no resampling touches sharpness/texture.  Validation, the prototype bank and test inference keep the
    deterministic boxes (they use the plain store)."""
    def __init__(self, base, p_rand: float, jitter: int):
        self.base, self.p_rand, self.jitter, self.map = base, float(p_rand), int(jitter), base.map

    def get(self, name, pil_im=None):
        meta = self.base.get(name, pil_im=pil_im)
        h, w = [int(v) for v in meta["orig_hw"]]
        boxes = np.array(meta["crops_448_boxes"], dtype=np.int32).copy()
        for k in range(boxes.shape[0]):
            x0, y0 = int(boxes[k, 0]), int(boxes[k, 1])
            if random.random() < self.p_rand:
                x0 = random.randint(0, w - 448) if w > 448 else x0
                y0 = random.randint(0, h - 448) if h > 448 else y0
            elif self.jitter > 0:
                x0 = min(max(0, x0 + random.randint(-self.jitter, self.jitter)), max(0, w - 448)) if w > 448 else x0
                y0 = min(max(0, y0 + random.randint(-self.jitter, self.jitter)), max(0, h - 448)) if h > 448 else y0
            boxes[k] = (x0, y0, x0 + 448, y0 + 448)
        out = dict(meta)
        out["crops_448_boxes"] = boxes
        return out


class HighPassLoader:
    """View of the image loader for a "forensic" tower (research roadmap of 2026-11-13, item 3): every frame is replaced
    by its amplified high-pass residual  127.5 + gain * (x - GaussianBlur_sigma(x))  at native resolution, before the
    views are cut, for training, validation and inference alike.  Content and colour are removed; what is left is
    what 过锐 halos, invented texture (过度生成) and missing texture (纹理损失) differ in.  The guided crop boxes still
    come from the meta store of the original frame."""
    def __init__(self, base, sigma: float, gain: float):
        self.base, self.sigma, self.gain = base, float(sigma), float(gain)

    def __getattr__(self, k):
        return getattr(self.base, k)

    def load_pil(self, name):
        from PIL import Image, ImageFilter
        im = self.base.load_pil(name).convert("RGB")
        x = np.asarray(im, dtype=np.float32)
        b = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=self.sigma)), dtype=np.float32)
        return Image.fromarray(np.clip(127.5 + self.gain * (x - b), 0, 255).astype(np.uint8))


class BenignAugLoader:
    """Training-only view of the image loader: with probability `p` the frame goes through a mild, label-preserving
    processing step before the views are cut (JPEG re-quantisation at quality 88-96 and / or a +-7 % tone change).
    In-place augmentation: no extra rows, no repeated labels (adding processed COPIES with hard labels hurt: benign1)."""
    def __init__(self, base, p: float):
        self.base, self.p = base, float(p)

    def __getattr__(self, k):
        return getattr(self.base, k)

    def load_pil(self, name):
        im = self.base.load_pil(name)
        if random.random() >= self.p:
            return im
        import io
        from PIL import Image
        im = im.convert("RGB")
        k = random.random()
        if k < 0.7:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=random.randint(88, 96), subsampling=random.choice([0, 2]))
            buf.seek(0)
            im = Image.open(buf).convert("RGB")
        if k > 0.4:
            g = random.uniform(0.93, 1.08)
            im = im.point([min(255, int(round(255.0 * (v / 255.0) ** g))) for v in range(256)] * 3)
        return im


class TwinAugLoader:
    """Training-only view of the image loader: with probability `p` a frame is replaced by its label-preserving twin
    built by synth_build.py (same content, e.g. after a high-quality x264 transcode).  In place: the row, its labels
    and its crop boxes stay those of the original, so nothing is repeated and no label is asserted twice."""
    def __init__(self, base, recipe: str, p: float, img_dir: str):
        self.base, self.p, self.dir = base, float(p), img_dir
        self.twins = {}
        for r in csv.DictReader(open(os.path.join(REPO_ROOT, f"data/r55_aux/manifest_{recipe}.csv"), encoding="utf-8")):
            if int(r["level"]) == 0 and os.path.exists(os.path.join(img_dir, r["name"])):
                self.twins.setdefault(r["seed"], []).append(r["name"])

    def __getattr__(self, k):
        return getattr(self.base, k)

    def load_pil(self, name):
        t = self.twins.get(str(name))
        if t and random.random() < self.p:
            from PIL import Image
            return Image.open(os.path.join(self.dir, random.choice(t))).convert("RGB")
        return self.base.load_pil(name)


def make_collate(base):
    def _c(batch):
        out = base(batch)
        out["aux_teacher"] = torch.stack([b["aux_teacher"] for b in batch])
        for k in ("aux_has_t", "aux_tuple", "aux_level", "aux_dim", "aux_clean", "tag_idx", "aux_det"):
            out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
        out["aux_mask"] = torch.stack([b["aux_mask"] for b in batch])
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
           (args.clean_neg and r["kind"] in POOL_KINDS)]
    man = [r for r in man if int(r["seed_fold"]) != args.fold and os.path.exists(os.path.join(args.aux_img_dir, r["name"]))]
    man = [r for r in man if r["name"] in meta_store.map]
    teacher, t_mask, t_det = {}, {}, {}
    if args.kd_synth or any(k in args.aux_kinds for k in POOL_KINDS):
        z = np.load(args.aux_teacher_npz, allow_pickle=True)
        pos = {str(n): i for i, n in enumerate(z["names"])}
        pf = z["probs_folds"][min(args.fold, z["probs_folds"].shape[0] - 1)]
        teacher = {n: pf[i] for n, i in pos.items()}
        t_mask = {n: z["mask"][i] for n, i in pos.items()} if "mask" in z.files else {}       # SynthLab: per-defect supervision mask
        t_det = {n: int(z["det"][i]) for n, i in pos.items()} if "det" in z.files else {}      # SynthLab: present-by-construction rows
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
    n_leak = 0
    for r in man:
        if r["kind"] not in POOL_KINDS:
            continue
        if r["kind"] == "synth" and r["seed"] not in train_by_name:
            n_leak += 1                                # SynthLab frames are made from labelled images: the seed must be a
            continue                                   # TRAINING image of this fold, whatever the manifest's fold column says
        is_c = int(r["name"] in clean)
        if is_c and args.clean_cert > 0:
            # Per-fold certificate: the candidate must also be called clean in every dimension by THIS fold's teacher
            # (models that never saw the validation fold), so the clean list itself can be label-free (e.g. high MOS).
            t = teacher.get(r["name"])
            is_c = int(t is not None and float(np.min(np.asarray(t)[:, 0])) >= args.clean_cert)
        use_t = r["kind"] in args.aux_kinds and r["name"] in teacher
        if not (is_c or use_t):
            continue
        singles.append(len(rows))
        rows.append(dict(name=r["name"], group_id=r["name"], grades=[0] * 6, is_clean=is_c,
                         teacher=teacher[r["name"]] if use_t else None, dim=int(r["dim"]) if str(r.get("dim", "")).lstrip("-").isdigit() else 0,
                         det=t_det.get(r["name"], 0) if use_t else 0, mask=t_mask.get(r["name"]) if use_t else None))
    if n_leak:
        print(f"  [aux] dropped {n_leak} synthetic frames whose seed is not a training image of this fold", flush=True)
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
        mk = b["aux_mask"].to(device).float()[has_t]              # (n, 6): 0 = this defect is not supervised on this row
        det = b["aux_det"].to(device)[has_t].bool()
        dmt = dm[has_t]
        terms = []
        for d in FULL_DIMS:
            q = tp[:, d] / tp[:, d].sum(-1, keepdim=True)
            m = det & (dmt == d)
            if m.any():
                # "present, grade unknown": the target is the model's own grade distribution with the clean mass
                # removed, so KL(q || p) = -log(1 - p_clean): it teaches presence and says nothing about the grade.
                ps = F.softmax(lg[m, d], dim=-1).detach().clone()
                ps[:, 0] = 0.0
                q = q.clone()
                q[m] = ps / ps.sum(-1, keepdim=True).clamp_min(1e-6)
            kl = F.kl_div(F.log_softmax(lg[:, d], dim=-1), q, reduction="none").sum(-1)
            terms.append((kl * mk[:, d]).sum() / mk[:, d].sum().clamp_min(1.0))
        for d in P0_ONLY_DIMS:
            q = tp[:, d][:, [0, 3]]
            q = q / q.sum(-1, keepdim=True)
            kl = F.kl_div(F.log_softmax(lg[:, d][:, [0, 3]], dim=-1), q, reduction="none").sum(-1)
            terms.append((kl * mk[:, d]).sum() / mk[:, d].sum().clamp_min(1.0))
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


def load_pool_teacher(dirs, fold: int) -> Dict[str, np.ndarray]:
    """Mean of `pool_probs_6x4` over teacher tower dirs (written by runs with --save-pool) for this fold.
    The caller decides which rows may use it: a teacher trained on half A may only label rows outside A."""
    acc: Dict[str, List[np.ndarray]] = {}
    for d in dirs:
        z = np.load(os.path.join(REPO_ROOT, d, f"fold_{fold}_probs.npz"), allow_pickle=True)
        for n, p in zip(z["pool_names"], z["pool_probs_6x4"]):
            acc.setdefault(str(n), []).append(p.astype(np.float32))
    return {n: np.mean(v, axis=0) for n, v in acc.items() if len(v) == len(dirs)}


class CrossDefectInvariance:
    """Disentangling penalty (2026-11-17, owner direction "B"): at a FIXED human grade of defect d, the predicted
    severity of d should not move with the human grades of the other graded defects.  On v7 nested OOF the pooled
    过度生成 P(P0) is too high on frames with other defects (pred 0.35-0.70 -> observed 0.35-0.41) and too low on
    clean-ish frames (pred 0.20-0.50 -> observed 0.43-0.54).

    For every current row i and every memory row k with g_d(i) == g_d(k) the pair contributes
    (e_d(i) - e_d(k)) and (s_j(i) - s_j(k)) for each other graded defect j, where e = expected severity under the
    model (0..3) and s_j the human grade.  The loss is sum_j slope_j^2 with slope_j the least-squares slope of the
    severity gap on the grade gap of j.  Memory rows are detached (an XBM-style queue), so a batch of 6 still sees a
    few hundred same-grade pairs.  Labels only; nothing at inference changes."""
    GRADED = (0, 1, 2, 4)

    def __init__(self, dims, capacity=512, device="cuda"):
        self.dims, self.cap, self.device = list(dims), int(capacity), device
        self.e = torch.zeros(0, 6, device=device)
        self.g = torch.zeros(0, 6, dtype=torch.long, device=device)

    def __call__(self, probs, grades):
        e = (probs.float() * torch.arange(4, device=probs.device, dtype=torch.float32)).sum(-1)      # (B, 6)
        g = grades.long()
        loss = e.sum() * 0.0
        if self.e.shape[0] >= 32:
            for d in self.dims:
                same = (g[:, d][:, None] == self.g[:, d][None, :]).float()                          # (B, M)
                if same.sum() < 8:
                    continue
                de = e[:, d][:, None] - self.e[:, d][None, :]
                for j in self.GRADED:
                    if j == d:
                        continue
                    ds = (g[:, j][:, None] - self.g[:, j][None, :]).float()
                    den = (same * ds * ds).sum()
                    if den < 1.0:
                        continue
                    loss = loss + ((same * de * ds).sum() / den) ** 2
        with torch.no_grad():
            self.e = torch.cat([e.detach(), self.e], 0)[: self.cap]
            self.g = torch.cat([g.detach(), self.g], 0)[: self.cap]
        return loss


def evaluate(model, va_rows, img_loader, meta_store, collate_fn, device, args):
    vp, vqc, gv, _ = predict_2x_tta(model, va_rows, img_loader, meta_store, collate_fn, device,
                                    batch_size=args.batch_size * 2, workers=args.workers)
    dec = DampedFrankWolfeBayesianDecoder(dfw_steps=20, eta_doc1=0.50, max_clean_p0_fpr=0.008).fit(vp, vqc, gv)
    ev = evaluate_qingyu_dual_metrics(dec.decode(vp, vqc), gv)
    ece = compute_ece_6x4(vp, gv)
    return ev, ece, 0.5 * ev["doc1_total"] + 0.5 * ev["doc2_strict_total"] - 2.0 * ece


def sibling_transition(rows):
    """Audit 2026-11-14 (P4a): per-dimension grade-transition matrix T[d, y, y'] = share of ordered pairs of frames of one
    video (training rows of this fold only) where the first has grade y and the second y'.  Rows without any sibling
    pair fall back to the identity."""
    import itertools
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[str(r["group_id"])].append([int(g) for g in r["grades"]])
    C = np.zeros((6, 4, 4))
    for v in by.values():
        for a, b in itertools.permutations(range(len(v)), 2):
            for d in range(6):
                C[d, v[a][d], v[b][d]] += 1
    n = C.sum(2, keepdims=True)
    T = np.where(n > 0, C / np.maximum(n, 1), np.eye(4)[None])
    print("  [noise-T] sibling pairs per (dim, grade):", n[..., 0].astype(int).tolist(), flush=True)
    return T


def train(args):
    if args.pe_taps:                                    # P6: read by backbones_qingyu at model construction
        os.environ["QY_PE_TAPS"] = args.pe_taps
    if args.global_letterbox:                           # P6: read by the Tier-1 dataset (workers inherit the environment)
        os.environ["QY_GLOBAL_LETTERBOX"] = "1"
    setup_sys_path_and_cwd(chdir_repo=True)
    enforce_deterministic_runtime(args.seed + args.fold)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    tr_all, va_rows = load_fold_rows(os.path.join(args.split_dir, f"fold_{args.fold}.csv"))
    half_a = {str(r["name"]) for r in subsample_groups(tr_all, 0.5, args.fold)}      # fixed group-wise half of the pool
    if args.half == "A":
        tr_rows = [r for r in tr_all if str(r["name"]) in half_a]
    elif args.half == "B":
        tr_rows = [r for r in tr_all if str(r["name"]) not in half_a]
    else:
        tr_rows = subsample_groups(tr_all, args.train_frac, args.fold)
    if args.train_batch != "all":
        # Release-batch study (first batch "orig" 1,902 images / supplement "supp" 1,671): train on one batch only.
        # Validation rows are untouched, so each batch's validation images show what the other batch contributes.
        rb = {r["name"]: r["split"] for r in csv.DictReader(open(os.path.join(REPO_ROOT, "data/manifest_combined.csv"), encoding="utf-8"))}
        tr_rows = [r for r in tr_rows if rb.get(str(r["name"])) == args.train_batch]
        print(f"  [train-batch] {args.train_batch}: {len(tr_rows)} of {len(tr_all)} training rows kept", flush=True)
    use_aux = bool(args.aux_kinds) or args.clean_neg
    if args.boost_proc_clean > 1:
        # Clean frames that went through a processing pipeline are 2 % of the processed frames (31 images) and are
        # flagged ~10x more often than other clean images.  Repeat them in the training list.  The pipeline tag in
        # the file name is used ONLY to pick which training rows to repeat; it is never a model input.
        extra = [r for r in tr_rows if not any(r["grades"]) and pipeline_tag(str(r["name"])) != "NA"]
        tr_rows = tr_rows + [dict(r) for r in extra for _ in range(args.boost_proc_clean - 1)]
        print(f"  [boost] {len(extra)} clean+processed training rows repeated x{args.boost_proc_clean}", flush=True)

    # Cross-fitted soft labels (no leak): a row of half A is labelled only by teachers trained on half B and vice
    # versa; every teacher was trained inside this fold's training pool, so none has seen the validation fold.
    t_from_a = load_pool_teacher(args.kd_a_dirs, args.fold) if args.kd_a_dirs else {}
    t_from_b = load_pool_teacher(args.kd_b_dirs, args.fold) if args.kd_b_dirs else {}
    if args.lambda_kd_lab > 0:
        n_t = 0
        for r in tr_rows:
            t = (t_from_b if str(r["name"]) in half_a else t_from_a).get(str(r["name"]))
            if t is not None:
                r["teacher"] = t
                n_t += 1
        assert n_t > 0, "--lambda-kd-lab needs --kd-a-dirs/--kd-b-dirs with pool predictions"
        print(f"  [kd-lab] {n_t}/{len(tr_rows)} labelled rows carry a cross-fitted teacher label", flush=True)
    ssl_rows = []
    if args.ssl:
        assert args.half == "A" and t_from_a, "--ssl needs --half A and --kd-a-dirs"
        ssl_rows = [dict(name=r["name"], group_id=r["group_id"], grades=[0] * 6, teacher=t_from_a[str(r["name"])])
                    for r in tr_all if str(r["name"]) not in half_a and str(r["name"]) in t_from_a]

    extra_dirs = [d for d in (args.aux_img_dir if use_aux else "", args.extra_test_dir) if d]
    img_loader = LosslessNativeImageLoader(REPO_ROOT, extra_dirs=extra_dirs or None)
    if args.input_hp > 0:
        img_loader = HighPassLoader(img_loader, args.input_hp, args.hp_gain)
        print(f"  [input] high-pass residual, sigma {args.input_hp}, gain {args.hp_gain}", flush=True)
    meta_store = PrecomputedDualEngineStore(args.aux_meta_npz if use_aux else args.dual_engine_npz,
                                            lazy_extractor=DualEnginePhysicsExtractor(device=str(device), allow_depth_download=True))

    if args.native_tiles != "off":
        from native_tiles import NativeTileTower
        Tier1DualEngineDataset.TILE_CAP_TRAIN = args.tile_cap
        Tier1DualEngineDataset.TILE_GLOBAL = (args.native_tiles == "tiles+global")
        model = NativeTileTower(backbone_key=args.backbone, phys_dim=PHYS_FEAT_DIM, unfreeze_last_k=args.unfreeze_blocks,
                                use_global=(args.native_tiles == "tiles+global")).to(device)
        print(f"  [native-tiles] {args.native_tiles}: every pixel at native resolution, 448 tiles, <= {args.tile_cap} tiles per frame in training, all tiles at inference", flush=True)
    else:
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

    tr_store = CropJitterStore(meta_store, args.crop_rand, args.crop_jitter) if (args.crop_rand > 0 or args.crop_jitter > 0) else meta_store
    tr_loader = BenignAugLoader(img_loader, args.aug_benign) if args.aug_benign > 0 else img_loader
    if args.aug_twins:
        tr_loader = TwinAugLoader(tr_loader, args.aug_twins, args.aug_twin_p, args.aux_img_dir)
        print(f"  [twins] recipe {args.aug_twins}: {sum(1 for r in tr_rows if str(r['name']) in tr_loader.twins)} of {len(tr_rows)} training frames have a twin, p={args.aug_twin_p}", flush=True)
    dl_tr = DataLoader(R55Dataset(tr_rows, tr_loader, tr_store, model.mode_tag, train=True),
                       batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate,
                       pin_memory=True, drop_last=True,
                       persistent_workers=(args.workers > 0), prefetch_factor=(2 if args.workers > 0 else None))

    aux_iter = None
    if ssl_rows:
        # The other half of the pool, labels hidden: teacher soft labels only (how much is in-domain unlabelled data worth?)
        sampler = AuxBatchSampler([], list(range(len(ssl_rows))), 0, args.aux_singles, seed=args.seed * 100 + args.fold)
        aux_iter = iter(DataLoader(R55Dataset(ssl_rows, img_loader, tr_store, model.mode_tag, train=True),
                                   batch_sampler=sampler, num_workers=args.aux_workers, collate_fn=collate, pin_memory=True))
        print(f"  [ssl] labelled={len(tr_rows)} unlabelled(teacher soft labels)={len(ssl_rows)} per-step={args.aux_singles}", flush=True)
    elif use_aux:
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
    noise_T = sibling_transition(tr_rows) if args.noise_alpha > 0 else None
    criterion = QingyuUnimodalCompositeLoss(soft_sigma=0.35, lambda_rps=0.45, lambda_rank=0.25, lambda_delta=1e-2,
                                            xbm_capacity=512, noise_T=noise_T, noise_alpha=args.noise_alpha,
                                            rank_adj_weight=args.rank_adj_w, rank_same_group=args.rank_same_group).to(device)

    if args.dim_w:
        criterion.dim_w = [float(x) for x in args.dim_w.split(",")]
        assert len(criterion.dim_w) == 6
        print(f"[sprint] per-dim loss weights (canonical order) = {criterion.dim_w}", flush=True)
    dis = None
    if args.lambda_dis > 0:
        dis = CrossDefectInvariance([int(x) for x in args.dis_dims.split(",")], capacity=args.dis_mem, device=device)
        print(f"[dis] cross-defect invariance on dims {dis.dims}, lambda {args.lambda_dis}, memory {args.dis_mem}", flush=True)

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
                if args.lambda_kd_lab > 0:
                    l_kdl = aux_losses(out, b, device, args.rank_tau)[1]
                    loss = loss + args.lambda_kd_lab * l_kdl
                    parts[2] = float(l_kdl.item())
                if dis is not None:
                    l_dis = dis(out["probs_6x4"], b["grades"].to(device))
                    loss = loss + args.lambda_dis * l_dis
                    parts[1] = float(l_dis.item())          # logged in the "rank" slot
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
        need_proto = (
            (max(1, args.epochs - 3) <= ep < args.epochs - 1)
            if (args.fast_fixed and args.select == "fixed")
            else (ep >= max(1, args.epochs - 3))
        )
        if need_proto:
            n_proto = populate_prototype_bank(model, tr_rows, img_loader, meta_store, base_collate, device,
                                              batch_size=args.batch_size * 2, workers=args.workers)
        if (args.fast_fixed and args.select == "fixed") or len(va_rows) == 0:
            ev, ece, val_obj = {"doc1_total": 0.0, "doc2_strict_total": 0.0}, 0.0, float(ep)
        else:
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
            if not args.no_save_ckpt and not (args.fast_fixed and args.select == "fixed"):
                torch.save(best_sd, ckpt_best)

    if args.select == "fixed":
        # No validation-based choice: the final weights are the mean of the last two epochs, as a full-fit run
        # (which has no validation fold) must do.  The per-epoch validation prints above are for logging only.
        best_sd = {k: (sum(sd[k].float() for sd in swa_sds) / len(swa_sds)).to(v.dtype) if v.is_floating_point() else v
                   for k, v in swa_sds[-1].items()}
    elif best_sd is not None and len(swa_sds) >= 2:
        cands = swa_sds + [best_sd]
        avg = {k: (sum(sd[k].float() for sd in cands) / len(cands)).to(v.dtype) if v.is_floating_point() else v
               for k, v in best_sd.items()}
        model.load_state_dict(avg, strict=False)
        populate_prototype_bank(model, tr_rows, img_loader, meta_store, base_collate, device,
                                batch_size=args.batch_size * 2, workers=args.workers)
        _, _, obj_s = evaluate(model, va_rows, img_loader, meta_store, base_collate, device, args)
        if obj_s >= best_obj - 0.08:
            best_sd = model.trainable_state_dict()
            if not args.no_save_ckpt:
                torch.save(best_sd, ckpt_best)
    model.load_state_dict(best_sd, strict=False)
    populate_prototype_bank(model, tr_rows, img_loader, meta_store, base_collate, device,
                            batch_size=args.batch_size * 2, workers=args.workers)
    if not args.no_save_ckpt:
        torch.save(model.trainable_state_dict(), ckpt_best)

    bs2 = args.batch_size * 2
    if len(va_rows) > 0:
        va_probs, va_qclean, gva, va_names = predict_2x_tta(model, va_rows, img_loader, meta_store, base_collate, device, bs2, args.workers)
        va_p0, va_p1, va_p2 = simplex_6x4_to_p012(va_probs)
    else:
        va_probs = np.zeros((0, 6, 4), dtype=np.float32)
        va_qclean = np.zeros((0,), dtype=np.float32)
        gva = np.zeros((0, 6), dtype=np.int64)
        va_names = []
        va_p0 = va_p1 = va_p2 = np.zeros((0, 6), dtype=np.float32)
    t150 = load_test_template_names("results/v2_full/template_150.json")
    if args.skip_t150:
        t_probs = np.zeros((len(t150), 6, 4), dtype=np.float32)
        t_probs[:, :, 0] = 1.0
        t_qclean = np.ones((len(t150),), dtype=np.float32)
        t_p0 = t_p1 = t_p2 = np.zeros((len(t150), 6), dtype=np.float32)
    else:
        t_probs, t_qclean, _, _ = predict_2x_tta(model, [{"name": n, "group_id": n, "grades": [0] * 6} for n in t150],
                                                 img_loader, meta_store, base_collate, device, bs2, args.workers)
        t_p0, t_p1, t_p2 = simplex_6x4_to_p012(t_probs)
    save = dict(fold=np.int64(args.fold), va_probs_6x4=va_probs, va_qclean=va_qclean, va_p0=va_p0, va_p1=va_p1, va_p2=va_p2,
                gva=gva, va_names=np.array(va_names), t150_probs_6x4=t_probs, t150_qclean=t_qclean, t150_p0=t_p0,
                t150_p1=t_p1, t150_p2=t_p2, t150_names=np.array(t150))
    if args.save_pool:
        p_probs, _, _, p_names = predict_2x_tta(model, tr_all, img_loader, meta_store, base_collate, device, bs2, args.workers)
        save.update(pool_probs_6x4=p_probs, pool_names=np.array(p_names))
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
    ap.add_argument("--aux-kinds", nargs="*", default=[], choices=["rank", "overgen", "ext", "kvq", "glitch", "synth"])
    ap.add_argument("--kd-synth", action="store_true")
    ap.add_argument("--clean-neg", action="store_true")
    ap.add_argument("--aux-manifest", default=AUX_MANIFEST)
    ap.add_argument("--aux-img-dir", default=AUX_IMG)
    ap.add_argument("--aux-meta-npz", default=AUX_META)
    ap.add_argument("--aux-teacher-npz", default=AUX_TEACHER)
    ap.add_argument("--clean-list", default=AUX_CLEAN)
    ap.add_argument("--clean-cert", type=float, default=0.0,
                    help=">0: a clean-list frame counts as a clean negative only if this fold's teacher gives P(clean) >= value in all six dims")
    ap.add_argument("--aux-tuples", type=int, default=2, help="tuples per step (4 images each)")
    ap.add_argument("--aux-singles", type=int, default=4, help="external images per step")
    ap.add_argument("--aux-workers", type=int, default=6)
    ap.add_argument("--lambda-tuple", type=float, default=0.5)
    ap.add_argument("--rank-tau", type=float, default=0.5)
    ap.add_argument("--lambda-aux-kd", type=float, default=0.5)
    ap.add_argument("--lambda-clean-neg", type=float, default=0.1)
    ap.add_argument("--lambda-tag", type=float, default=0.0)
    # regularisation / protocol (all default-off: no flag = Tier-1 baseline)
    ap.add_argument("--crop-rand", type=float, default=0.0, help="P(a guided crop is replaced by a random native 448 box), train only")
    ap.add_argument("--crop-jitter", type=int, default=0, help="max +-px shift of a guided crop box, train only")
    ap.add_argument("--aug-twins", default="", help="SynthLab label-preserving recipe whose frames replace their seeds in place")
    ap.add_argument("--aug-twin-p", type=float, default=0.5)
    ap.add_argument("--aug-benign", type=float, default=0.0, help="probability of a mild label-preserving JPEG / tone step on a training frame")
    ap.add_argument("--boost-proc-clean", type=int, default=1, help="repeat clean rows that carry a pipeline tag this many times (train only)")
    ap.add_argument("--half", default="none", choices=["none", "A", "B"], help="train on one fixed group-wise half of the fold's pool")
    ap.add_argument("--save-pool", action="store_true", help="also store predictions for every row of the fold's training pool")
    ap.add_argument("--kd-a-dirs", nargs="*", default=[], help="tower dirs trained with --half A --save-pool (label rows outside A)")
    ap.add_argument("--kd-b-dirs", nargs="*", default=[], help="tower dirs trained with --half B --save-pool (label rows of A)")
    ap.add_argument("--lambda-kd-lab", type=float, default=0.0, help="KL weight to the cross-fitted teacher on labelled rows")
    ap.add_argument("--ssl", action="store_true", help="with --half A: use the other half as unlabelled data with teacher soft labels")
    ap.add_argument("--native-tiles", default="off", choices=["off", "tiles", "tiles+global"],
                    help="see code/r55/native_tiles.py: the tower sees every pixel at native resolution as 448x448 tiles")
    ap.add_argument("--tile-cap", type=int, default=16, help="max tiles per frame while training (random subset); inference uses all")
    ap.add_argument("--input-hp", type=float, default=0.0, help="> 0: feed the high-pass residual of every frame (Gaussian sigma in px), see HighPassLoader")
    ap.add_argument("--hp-gain", type=float, default=4.0)
    ap.add_argument("--train-batch", default="all", choices=["all", "orig", "supp"],
                    help="train only on rows of one release batch (data/manifest_combined.csv column `split`)")
    ap.add_argument("--pe-taps", default="", help="P6: comma list of PE-Core block indices for the three gated taps")
    ap.add_argument("--global-letterbox", action="store_true", help="P6: aspect-preserving global view instead of the 448x448 squash")
    ap.add_argument("--noise-alpha", type=float, default=0.0, help="P4a: mix sibling grade-transition rows into the soft targets")
    ap.add_argument("--rank-adj-w", type=float, default=2.0, help="P4b: ranking weight of adjacent-grade pairs (2.0 = original)")
    ap.add_argument("--rank-same-group", action="store_true", help="P4b: also rank pairs of frames from the same video")
    ap.add_argument("--select", default="val", choices=["val", "fixed"],
                    help="val = Tier-1 behaviour (checkpoint chosen on the validation fold); fixed = mean of the last 2 epochs")
    ap.add_argument("--fast-fixed", action="store_true",
                    help="under --select fixed, skip logging-only per-epoch val TTA + pre-SWA E-1 prototype populate (keeps live E-3/E-2 populates)")
    ap.add_argument("--skip-t150", action="store_true",
                    help="skip redundant per-fold t150 inference (preserves t150_* keys as valid simplex placeholders for downstream readers)")
    ap.add_argument("--dim-w", default="", help="sprint: comma list of 6 per-dim loss weights in canonical DIMS order")
    ap.add_argument("--lambda-dis", type=float, default=0.0, help="B: cross-defect invariance weight (0 = off)")
    ap.add_argument("--dis-dims", default="4", help="B: comma list of graded dims the invariance applies to")
    ap.add_argument("--dis-mem", type=int, default=512, help="B: memory rows for the invariance pairs")
    ap.add_argument("--no-save-ckpt", action="store_true",
                    help="skip writing fold_*_best.pt to conserve disk during OOF sweeps (still writes fold_*_probs.npz)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
