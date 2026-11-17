#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/train_p12_specialist.py
================================
P1 + P2 Semantic-ROI & Boundary-Contrastive Multi-Subspace Specialist Trainer.

Key breakthroughs over `train_testbed.py`:
  1. Data View (`--dual-engine-npz results/r55/semantic_roi_meta.npz`):
     Feeds ALL 5 semantic views `[global_native_or_letterbox, roi_subtitle_ocr_448, roi_face_anat_448,
     roi_hand_swirl_448, roi_bokeh_halo_448]` + 48-dim forensic physics vector (including even in
     `native_aspect_plus_crops` mode where Tier-1 previously dropped crop #3!).
  2. Architecture (`PerDefectRoutedOrdinalHead`):
     Replaces `MultiViewAttentionAggregator` + 1D `UnimodalQuadraticResidualHead` with per-defect
     multi-view cross-attention + cross-defect self-attention + unconstrained 4-class readout + TipAdapter(128).
     Also unlocks `siglip2_giant` (`models/siglip2-giant` 40-layer 1536d vision encoder, 1.1B = 2.2GB bf16!)
     and view-chunked `internvit_6b`.
  3. Training Paradigm (`P12CompositeOrdinalLoss`):
     - `BoundaryMarginXBMLoss(512)` supervising all 3 exact log-odds boundaries (`Pos vs Clean`, `P1+ vs P2`,
       `P0 vs P1`) across distinct videos.
     - Leak-free Nested Cross-Family Soft-Target Distillation (`--nested-kd-npz` + `--lambda-kd`).
     - Fixed-epoch SWA (`--select fixed --epochs E`) aligned with full-fit delivery (zero Epoch-8 memorization!).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(REPO, "code"), os.path.join(REPO, "code/next55"), os.path.join(REPO, "code/r55")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qingyu.config_and_paths import (
    LosslessNativeImageLoader,
    enforce_deterministic_runtime,
    evaluate_qingyu_dual_metrics,
    load_test_template_names,
    setup_sys_path_and_cwd,
)
from qingyu.dual_engine_preprocess import (
    PrecomputedDualEngineStore,
    extract_1to1_crop_pil,
    resize_native_aspect_pil,
)
from qingyu.backbones_qingyu import (
    IntermediateThreeLayerGatedTap,
    MaskedTokenGatedABMILPool,
    QingyuVisionQualityTower,
)
from qingyu.certainty_gated_stacker import compute_ece_6x4, simplex_6x4_to_p012
from qingyu.dfw_bayesian_decoder import DampedFrankWolfeBayesianDecoder
from qingyu.train_tier1_5backbones_8fold import load_fold_rows
from qingyu.unimodal_xbm_losses import hash_group_id_int64

from p12_routed_head import P12CompositeOrdinalLoss, PerDefectRoutedOrdinalHead
from semantic_roi_meta import SEMANTIC_PHYS_TOTAL_DIM, SemanticRoiLazyExtractor


class P12SemanticRoiTower(nn.Module):
    """
    Wraps any supported vision encoder (`pe_core_large`, `cradio_v4_h`, `siglip2_so400m_naflex`,
    `siglip2_giant`, `internvit_6b`) with `PerDefectRoutedOrdinalHead` over uncollapsed `[B, V, D]` views.
    """
    def __init__(
        self,
        backbone_key: str,
        phys_dim: int = SEMANTIC_PHYS_TOTAL_DIM,
        unfreeze_last_k: int = 8,
        lora_rank: int = 24,
    ):
        super().__init__()
        self.backbone_key = backbone_key
        self.naflex_processor = None

        if backbone_key == "siglip2_giant":
            self.backbone, self._encode_single_view, mean, std, single_view_dim, self.mode_tag = (
                self._build_siglip2_giant(unfreeze_last_k)
            )
        elif backbone_key == "pe_core_g14":
            self.backbone, self._encode_single_view, mean, std, single_view_dim, self.mode_tag = (
                self._build_pe_core_g14(unfreeze_last_k)
            )
        else:
            # Reuse QingyuVisionQualityTower's tested backbone builder, discarding its collapsed head
            tmp = QingyuVisionQualityTower(
                backbone_key=backbone_key,
                phys_dim=0,
                unfreeze_last_k=unfreeze_last_k,
                use_lora_6b=(backbone_key == "internvit_6b"),
                lora_rank=lora_rank,
            )
            self.backbone = tmp.backbone
            self._encode_single_view = tmp._encode_single_view
            self.naflex_processor = getattr(tmp, "naflex_processor", None)
            mean = tuple(float(x) for x in tmp.img_mean.view(-1).tolist())
            std = tuple(float(x) for x in tmp.img_std.view(-1).tolist())
            single_view_dim = tmp.head.in_dim // 3
            self.mode_tag = tmp.mode_tag
            del tmp

        self.register_buffer("img_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        hidden_dim = 640 if single_view_dim >= 3000 else 512
        self.head = PerDefectRoutedOrdinalHead(
            single_view_dim=single_view_dim,
            phys_dim=phys_dim,
            hidden_dim=hidden_dim,
            proj_dim=512,
            num_heads=8,
            max_views=8,
            dropout=0.20,
            max_prototypes=128,
        ).float()

    def _build_siglip2_giant(self, unfreeze_last_k: int):
        from safetensors.torch import load_file
        from transformers import SiglipConfig, SiglipVisionModel

        repo = os.path.join(REPO, "models", "siglip2-giant")
        cfg = SiglipConfig.from_pretrained(repo)
        cfg.vision_config._attn_implementation = "sdpa"
        vis = SiglipVisionModel(cfg.vision_config)
        expected_keys = set(vis.state_dict().keys())
        has_vm_prefix = any(k.startswith("vision_model.") for k in expected_keys)
        sd = {}
        for fn in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            fp = os.path.join(repo, fn)
            for k, v in load_file(fp).items():
                if k.startswith("vision_model."):
                    nk = k if has_vm_prefix else k[len("vision_model."):]
                    sd[nk] = v
        missing, unexpected = vis.load_state_dict(sd, strict=False)
        del sd
        assert len(missing) == 0, f"SigLIP2-Giant missing keys: {missing[:5]}"
        print(f"[SigLIP2-Giant] Loaded vision tower (40L, 1536d): missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        vm = vis.vision_model if hasattr(vis, "vision_model") else vis
        vis = vis.to(torch.bfloat16)
        blocks = vm.encoder.layers
        for p in vis.parameters():
            p.requires_grad = False
        k_eff = min(len(blocks), max(1, unfreeze_last_k))
        for blk in list(blocks)[-k_eff:]:
            for p in blk.parameters():
                p.requires_grad = True
        for n, p in vis.named_parameters():
            if any(t in n for t in ("post_layernorm", "head")):
                p.requires_grad = True

        # Enable gradient checkpointing on SigLIP2-Giant encoder
        vm.encoder.gradient_checkpointing = True
        def _grad_hook(_m, _i, out):
            if torch.is_grad_enabled() and isinstance(out, torch.Tensor) and not out.requires_grad:
                return out.requires_grad_(True)
            return out
        vm.embeddings.register_forward_hook(_grad_hook)

        emb = vm.embeddings
        orig_emb_fwd = emb.forward
        emb.forward = lambda pixel_values, **kw: orig_emb_fwd(pixel_values, interpolate_pos_encoding=True)

        hid = cfg.vision_config.hidden_size  # 1536
        vis.tap3 = IntermediateThreeLayerGatedTap(
            blocks, hid, tap_indices=[len(blocks) - 10, len(blocks) - 5, len(blocks) - 2]
        ).float()
        vis.masked_abmil = MaskedTokenGatedABMILPool(hid).float()

        def fwd_sig_giant(x: torch.Tensor) -> torch.Tensor:
            out = vis(pixel_values=x.to(torch.bfloat16))
            amp_dev = "cuda" if out.last_hidden_state.is_cuda else "cpu"
            with torch.autocast(device_type=amp_dev, enabled=False):
                t = vis.tap3.fuse(out.last_hidden_state.float())
                return torch.cat([t.mean(dim=1), vis.masked_abmil(t, mask=None)], dim=1)

        return vis, fwd_sig_giant, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), hid * 2, "native_aspect_plus_crops"

    def _build_pe_core_g14(self, unfreeze_last_k: int):
        import timm
        from safetensors.torch import load_file

        shm_w = "/dev/shm/pe_core_g14_bf16.safetensors"
        nfs_w = os.path.join(REPO, "models", "pe-core-g14", "model.safetensors")
        w_path = shm_w if os.path.exists(shm_w) else nfs_w

        m = timm.create_model("vit_pe_core_gigantic_patch14_448", pretrained=False, num_classes=0, dynamic_img_size=True)
        sd = load_file(w_path)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        del sd
        assert len(missing) == 0, f"PE-Core-G14 missing keys: {missing[:5]}"
        print(f"[PE-Core-G14-448] Loaded giant vision tower ({len(m.blocks)}L, {m.embed_dim}d) from {w_path}: "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        m = m.to(torch.bfloat16)
        m.set_grad_checkpointing(True)
        for p in m.parameters():
            p.requires_grad = False
        k_eff = min(len(m.blocks), max(10, unfreeze_last_k))
        for blk in list(m.blocks)[-k_eff:]:
            for p in blk.parameters():
                p.requires_grad = True
        for n, p in m.named_parameters():
            if any(t in n for t in ("norm", "fc_norm", "attn_pool")):
                p.requires_grad = True

        def _grad_hook(_mod, _inp, out):
            if torch.is_grad_enabled() and isinstance(out, torch.Tensor) and not out.requires_grad:
                return out.requires_grad_(True)
            return out
        m.patch_embed.register_forward_hook(_grad_hook)

        hid = int(m.embed_dim)  # 1536
        # 50 blocks: proportional 67%/83%/96% depths -> [33, 41, 48]
        m.tap3 = IntermediateThreeLayerGatedTap(m.blocks, hid, tap_indices=[33, 41, 48]).float()
        m.masked_abmil = MaskedTokenGatedABMILPool(hid).float()

        def fwd_pe_g14(x: torch.Tensor) -> torch.Tensor:
            raw_t = m.forward_features(x.to(torch.bfloat16))
            map_feat = m.forward_head(raw_t, pre_logits=True).float()
            amp_dev = "cuda" if raw_t.is_cuda else "cpu"
            with torch.autocast(device_type=amp_dev, enabled=False):
                t = m.tap3.fuse(raw_t.float())
                return torch.cat([map_feat, m.masked_abmil(t, mask=None)], dim=1)

        return m, fwd_pe_g14, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), hid * 2, "multi_crop_448"

    def _norm_img(self, x_01: torch.Tensor) -> torch.Tensor:
        return (x_01 - self.img_mean) / self.img_std

    def _encode_chunked(self, flat: torch.Tensor, chunk_sz: int) -> torch.Tensor:
        if flat.shape[0] <= chunk_sz:
            return self._encode_single_view(flat)
        return torch.cat([self._encode_single_view(flat[s : s + chunk_sz]) for s in range(0, flat.shape[0], chunk_sz)], dim=0)

    def forward(
        self,
        views_01: Optional[torch.Tensor] = None,
        phys_feat: Optional[torch.Tensor] = None,
        native_global_list: Optional[List[torch.Tensor]] = None,
        naflex_views_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        chunk_sz = 12 if self.backbone_key in ("internvit_6b", "siglip2_giant", "pe_core_g14") else 48
        if self.mode_tag == "naflex_dict" and naflex_views_list is not None:
            v_stack = torch.stack([self._encode_single_view(vd) for vd in naflex_views_list], dim=1)
        elif self.mode_tag == "native_aspect_plus_crops" and native_global_list is not None:
            B = len(native_global_list)
            g_feats = [None] * B
            shape_groups: Dict[Tuple[int, int], List[int]] = {}
            for bi, gt in enumerate(native_global_list):
                hw = (int(gt.shape[-2]), int(gt.shape[-1]))
                shape_groups.setdefault(hw, []).append(bi)
            for _, idxs in shape_groups.items():
                gx = torch.cat([native_global_list[i] for i in idxs], dim=0)
                gf = self._encode_chunked(self._norm_img(gx), chunk_sz)
                for loc_j, orig_i in enumerate(idxs):
                    g_feats[orig_i] = gf[loc_j]
            g_stack = torch.stack(g_feats, dim=0).unsqueeze(1)

            if views_01 is not None and views_01.shape[1] > 0:
                B_, Vc, C, H, W = views_01.shape
                cf = self._encode_chunked(self._norm_img(views_01.reshape(B_ * Vc, C, H, W)), chunk_sz).view(B_, Vc, -1)
                v_stack = torch.cat([g_stack, cf], dim=1)
            else:
                v_stack = g_stack
        else:
            assert views_01 is not None
            B, V, C, H, W = views_01.shape
            flat = self._norm_img(views_01.reshape(B * V, C, H, W))
            v_stack = self._encode_chunked(flat, chunk_sz).view(B, V, -1)

        amp_dev = "cuda" if v_stack.is_cuda else "cpu"
        with torch.autocast(device_type=amp_dev, enabled=False):
            return self.head(v_stack.float(), phys_feat=phys_feat.float() if phys_feat is not None else None)

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        req = {n for n, p in self.named_parameters() if p.requires_grad}
        out = {}
        for k, v in self.state_dict().items():
            if (k in req) or k.startswith("head.") or ("tap3" in k) or ("dual_tap" in k) or ("abmil" in k):
                out[k] = v.detach().cpu()
        return out


class P12SemanticDataset(Dataset):
    """
    Feeds ALL 4 semantic 1:1 448x448 crops `[subtitle_ocr, face_anat, hand_swirl, bokeh_halo]` + global view
    across ALL backbone modes (`multi_crop_448`, `native_aspect_plus_crops`, `naflex_dict`).
    """
    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        img_loader: LosslessNativeImageLoader,
        meta_store: PrecomputedDualEngineStore,
        mode_tag: str,
        train: bool = False,
        hflip_tta: bool = False,
        crop_jitter: int = 32,
        teacher_map: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.rows = list(rows)
        self.loader = img_loader
        self.meta_store = meta_store
        self.mode_tag = mode_tag
        self.train = train
        self.hflip_tta = hflip_tta
        self.crop_jitter = int(crop_jitter) if train else 0
        self.teacher_map = teacher_map or {}

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _pil_to_chw01(im: Image.Image) -> torch.Tensor:
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy())

    @staticmethod
    def _letterbox_448(im: Image.Image) -> Image.Image:
        sc = 448.0 / max(im.size)
        nw = max(1, int(round(im.size[0] * sc)))
        nh = max(1, int(round(im.size[1] * sc)))
        g_in = im.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (448, 448))
        canvas.paste(g_in, ((448 - nw) // 2, (448 - nh) // 2))
        return canvas

    def __getitem__(self, idx: int) -> Dict[str, object]:
        r = self.rows[idx]
        nm = str(r["name"])
        im = self.loader.load_pil(nm).convert("RGB")
        meta = self.meta_store.get(nm, pil_im=im)
        orig_w, orig_h = im.size

        do_flip = self.hflip_tta or (self.train and random.random() < 0.5)
        c_boxes = meta["crops_448_boxes"].copy()
        if do_flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            old_x0 = c_boxes[:, 0].copy()
            old_x1 = c_boxes[:, 2].copy()
            c_boxes[:, 0] = np.maximum(0, orig_w - old_x1)
            c_boxes[:, 2] = np.minimum(orig_w, c_boxes[:, 0] + 448)

        if self.crop_jitter > 0:
            for k in range(c_boxes.shape[0]):
                dx = random.randint(-self.crop_jitter, self.crop_jitter) if orig_w > 448 else 0
                dy = random.randint(-self.crop_jitter, self.crop_jitter) if orig_h > 448 else 0
                x0 = min(max(0, int(c_boxes[k, 0]) + dx), max(0, orig_w - 448))
                y0 = min(max(0, int(c_boxes[k, 1]) + dy), max(0, orig_h - 448))
                c_boxes[k] = (x0, y0, min(orig_w, x0 + 448), min(orig_h, y0 + 448))

        if self.train:
            if random.random() < 0.30:
                im = ImageEnhance.Brightness(im).enhance(1.0 + random.uniform(-0.04, 0.04))
            if random.random() < 0.30:
                im = ImageEnhance.Contrast(im).enhance(1.0 + random.uniform(-0.04, 0.04))

        phys = torch.from_numpy(np.asarray(meta["phys_vec"], dtype=np.float32).copy())
        grades = torch.tensor(r["grades"], dtype=torch.long)
        ghash = hash_group_id_int64(str(r.get("group_id", nm)))
        tp = self.teacher_map.get(nm, None)
        t_tensor = torch.from_numpy(tp.copy()) if tp is not None else torch.zeros((6, 4), dtype=torch.float32)
        has_t = 1 if tp is not None else 0

        crops_pil = [extract_1to1_crop_pil(im, c_boxes[k], 448) for k in range(4)]

        if self.mode_tag == "naflex_dict":
            im_nat = resize_native_aspect_pil(im, max_pixels=784 * 256, min_pixels=256 * 256, patch_mult=16)
            return {
                "pil_views": [im_nat] + crops_pil,
                "phys": phys,
                "grades": grades,
                "ghash": ghash,
                "name": nm,
                "teacher_6x4": t_tensor,
                "has_teacher": has_t,
            }

        if self.mode_tag == "native_aspect_plus_crops":
            im_nat = resize_native_aspect_pil(im, max_pixels=784 * 256, min_pixels=256 * 256, patch_mult=16)
            return {
                "nat_global": self._pil_to_chw01(im_nat),
                "views_01": torch.stack([self._pil_to_chw01(c) for c in crops_pil], dim=0),
                "phys": phys,
                "grades": grades,
                "ghash": ghash,
                "name": nm,
                "teacher_6x4": t_tensor,
                "has_teacher": has_t,
            }

        # Direct bicubic 448x448 global resize (LEDGER E1 proved black-padded letterbox harms PE-Core by +0.0012 RPS)
        g_448 = im.resize((448, 448), Image.BICUBIC)
        views = [g_448] + crops_pil
        return {
            "views_01": torch.stack([self._pil_to_chw01(v) for v in views], dim=0),
            "phys": phys,
            "grades": grades,
            "ghash": ghash,
            "name": nm,
            "teacher_6x4": t_tensor,
            "has_teacher": has_t,
        }


def make_p12_collate(mode_tag: str, naflex_processor=None, max_num_patches: int = 784):
    def _collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        out: Dict[str, object] = {
            "phys": torch.stack([b["phys"] for b in batch], dim=0),
            "grades": torch.stack([b["grades"] for b in batch], dim=0),
            "ghash": torch.tensor([b["ghash"] for b in batch], dtype=torch.long),
            "names": [b["name"] for b in batch],
            "teacher_6x4": torch.stack([b["teacher_6x4"] for b in batch], dim=0),
            "has_teacher": torch.tensor([b["has_teacher"] for b in batch], dtype=torch.long),
        }
        if mode_tag == "naflex_dict":
            n_v = len(batch[0]["pil_views"])
            out["naflex_views"] = [
                dict(naflex_processor(images=[b["pil_views"][vi] for b in batch], return_tensors="pt", max_num_patches=max_num_patches))
                for vi in range(n_v)
            ]
        elif mode_tag == "native_aspect_plus_crops":
            out["nat_global"] = [b["nat_global"].unsqueeze(0) for b in batch]
            out["views_01"] = torch.stack([b["views_01"] for b in batch], dim=0)
        else:
            out["views_01"] = torch.stack([b["views_01"] for b in batch], dim=0)
        return out

    return _collate


def forward_p12_batch(model: P12SemanticRoiTower, batch: Dict[str, object], device: torch.device):
    phys = batch["phys"].to(device, non_blocking=True)
    if model.mode_tag == "naflex_dict":
        nv = [{k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in vd.items()}
              for vd in batch["naflex_views"]]
        return model(naflex_views_list=nv, phys_feat=phys)
    if model.mode_tag == "native_aspect_plus_crops":
        ng = [t.to(device, non_blocking=True) for t in batch["nat_global"]]
        v01 = batch["views_01"].to(device, non_blocking=True)
        return model(views_01=v01, native_global_list=ng, phys_feat=phys)
    v01 = batch["views_01"].to(device, non_blocking=True)
    return model(views_01=v01, phys_feat=phys)


@torch.no_grad()
def populate_p12_prototypes(model, rows, loader, store, collate, device, batch_size, workers) -> int:
    model.eval()
    model.head.proto_bank.num_active.fill_(0)
    dl = DataLoader(P12SemanticDataset(rows, loader, store, model.mode_tag, train=False),
                    batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate, pin_memory=True)
    zs, ys, ps = [], [], []
    for b in dl:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = forward_p12_batch(model, b, device)
        zs.append(out["z_norm"].float().cpu().numpy())
        ys.append(b["grades"].numpy())
        ps.append(b["phys"].numpy())
    return model.head.proto_bank.fit_from_train_embeddings(
        np.concatenate(zs, 0), np.concatenate(ys, 0), np.concatenate(ps, 0)
    )


@torch.no_grad()
def predict_p12_tta(model, rows, loader, store, collate, device, batch_size, workers):
    model.eval()
    probs_2, qc_2, g_ref, nm_ref = [], [], None, None
    for hf in (False, True):
        dl = DataLoader(P12SemanticDataset(rows, loader, store, model.mode_tag, train=False, hflip_tta=hf),
                        batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate, pin_memory=True)
        ps, qs, gs, nms = [], [], [], []
        for b in dl:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = forward_p12_batch(model, b, device)
            ps.append(out["probs_6x4"].float().cpu().numpy())
            qs.append(torch.sigmoid(out["clean_logit"].float()).cpu().numpy())
            gs.append(b["grades"].numpy())
            nms.extend(b["names"])
        probs_2.append(np.concatenate(ps, 0))
        qc_2.append(np.concatenate(qs, 0))
        if g_ref is None:
            g_ref = np.concatenate(gs, 0)
            nm_ref = nms
    return 0.5 * (probs_2[0] + probs_2[1]), 0.5 * (qc_2[0] + qc_2[1]), g_ref, nm_ref


def load_nested_teacher_map(npz_path: str, outer_fold: int) -> Dict[str, np.ndarray]:
    if not npz_path or not os.path.exists(npz_path):
        return {}
    z = np.load(npz_path, allow_pickle=True)
    names = [str(x) for x in z["names"]]
    if f"fold_{outer_fold}_probs" in z:
        arr = z[f"fold_{outer_fold}_probs"]
    elif "probs_folds" in z:
        arr = z["probs_folds"][outer_fold]
    else:
        arr = z["probs_6x4"]
    out = {}
    for nm, p in zip(names, arr):
        if not np.isnan(p).any() and float(p.sum()) > 0.5:
            out[nm] = np.asarray(p, dtype=np.float32)
    return out


def train(args):
    setup_sys_path_and_cwd(chdir_repo=True)
    enforce_deterministic_runtime(args.seed + args.fold * 13, deterministic_algos=False)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    tr_rows, va_rows = load_fold_rows(os.path.join(args.split_dir, f"fold_{args.fold}.csv"))
    if args.exclude_fold >= 0:
        # Inner-fold mode for building strictly leak-free (K-1)-fold nested teachers!
        _, held_out = load_fold_rows(os.path.join(args.split_dir, f"fold_{args.exclude_fold}.csv"))
        held_names = {str(r["name"]) for r in held_out}
        tr_rows = [r for r in tr_rows if str(r["name"]) not in held_names]

    img_loader = LosslessNativeImageLoader(REPO)
    meta_store = PrecomputedDualEngineStore(args.dual_engine_npz, lazy_extractor=SemanticRoiLazyExtractor())
    sample_phys = meta_store.get(str(tr_rows[0]["name"]))["phys_vec"]
    phys_dim = int(sample_phys.shape[0])

    teacher_map = load_nested_teacher_map(args.nested_kd_npz, args.fold) if args.lambda_kd > 0 else {}
    if args.lambda_kd > 0:
        cov = sum(1 for r in tr_rows if str(r["name"]) in teacher_map)
        print(f"  [Nested-KD] Loaded teacher soft targets for {cov}/{len(tr_rows)} training rows (lambda_kd={args.lambda_kd})", flush=True)

    model = P12SemanticRoiTower(
        backbone_key=args.backbone,
        phys_dim=phys_dim,
        unfreeze_last_k=args.unfreeze_blocks,
        lora_rank=args.lora_rank,
    ).to(device)

    tr_phys = np.stack([meta_store.get(str(r["name"]))["phys_vec"] for r in tr_rows], axis=0)
    model.head.set_physics_scaler(tr_phys.mean(0), tr_phys.std(0) + 1e-5)

    collate = make_p12_collate(model.mode_tag, naflex_processor=model.naflex_processor, max_num_patches=784)
    dl_tr = DataLoader(
        P12SemanticDataset(tr_rows, img_loader, meta_store, model.mode_tag, train=True,
                           crop_jitter=args.crop_jitter, teacher_map=teacher_map),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=(2 if args.workers > 0 else None),
    )

    bb_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    hd_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": args.lr_backbone, "weight_decay": 0.02},
        {"params": hd_params, "lr": args.lr_head, "weight_decay": 0.01},
    ])
    total_steps = max(1, (len(dl_tr) // max(1, args.grad_accum)) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr_backbone, args.lr_head], total_steps=total_steps, pct_start=0.12
    )
    criterion = P12CompositeOrdinalLoss(
        soft_sigma=args.soft_sigma,
        lambda_rps=0.50,
        lambda_bmargin=args.lambda_bmargin,
        lambda_clean=0.25,
        kd_temp=1.25,
        cond_b1=args.cond_b1,
    ).to(device)

    out_dir = os.path.join(REPO, args.out_dir, args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_best = os.path.join(out_dir, f"fold_{args.fold}_best.pt")

    print(f"=== [P12 Specialist] {args.backbone} | fold {args.fold} (excl={args.exclude_fold}) | seed {args.seed} "
          f"| train={len(tr_rows)} val={len(va_rows)} | epochs={args.epochs} ({args.select}) | phys={phys_dim} ===", flush=True)

    best_obj, best_sd, swa_sds = -1e9, None, []
    bs_eval = max(2, args.batch_size * 2)

    for ep in range(args.epochs):
        model.train()
        criterion.train()
        t0 = time.time()
        acc = np.zeros(4, dtype=np.float64)
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step_i, b in enumerate(dl_tr):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = forward_p12_batch(model, b, device)
                losses = criterion(
                    out,
                    b["grades"].to(device, non_blocking=True),
                    b["ghash"].to(device, non_blocking=True),
                    teacher_probs_6x4=b["teacher_6x4"].to(device, non_blocking=True),
                    has_teacher_mask=b["has_teacher"].to(device, non_blocking=True),
                    lambda_kd=args.lambda_kd,
                )
                loss = losses["total"] / max(1, args.grad_accum)
            loss.backward()

            if (step_i + 1) % args.grad_accum == 0 or (step_i + 1) == len(dl_tr):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler.last_epoch < total_steps - 1:
                    scheduler.step()

            acc += np.array([
                float(losses["total"].item()),
                float(losses["rps"].item()),
                float(losses["bmargin"].item()),
                float(losses["kd"].item()),
            ])
            n_steps += 1

        n_proto = 0
        need_proto = (
            (max(1, args.epochs - 2) <= ep < args.epochs - 1)
            if (args.fast_fixed and args.select == "fixed")
            else (ep >= max(1, args.epochs - 2))
        )
        if need_proto:
            n_proto = populate_p12_prototypes(model, tr_rows, img_loader, meta_store, collate, device, bs_eval, args.workers)

        if len(va_rows) > 0 and not (args.fast_fixed and args.select == "fixed"):
            vp, vqc, gv, _ = predict_p12_tta(model, va_rows, img_loader, meta_store, collate, device, bs_eval, args.workers)
            ece = compute_ece_6x4(vp, gv)
            dec = DampedFrankWolfeBayesianDecoder(dfw_steps=20, eta_doc1=0.50, max_clean_p0_fpr=0.008).fit(vp, vqc, gv)
            ev = evaluate_qingyu_dual_metrics(dec.decode(vp, vqc), gv)
            oh = np.eye(4)[gv]
            rps_val = float(np.mean((np.cumsum(vp, -1)[..., :3] - np.cumsum(oh, -1)[..., :3]) ** 2))
            val_obj = -rps_val  # strictly proper scoring rule if --select val is ever used
        else:
            rps_val, ece, val_obj = 0.0, 0.0, float(ep)
            ev = {"doc2_strict_total": 0.0, "doc1_total": 0.0}
        m = acc / max(1, n_steps)
        print(f"  [{args.backbone} F{args.fold}] Ep {ep + 1}/{args.epochs} | loss={m[0]:.4f} rps_l={m[1]:.4f} "
              f"bm={m[2]:.4f} kd={m[3]:.4f} | val_RPS={rps_val:.5f} Doc2={ev['doc2_strict_total']:.2f} "
              f"ECE={ece:.4f} protos={n_proto} ({time.time() - t0:.0f}s)", flush=True)

        cur_sd = model.trainable_state_dict()
        if ep >= max(0, args.epochs - 2):
            swa_sds.append(cur_sd)
        if val_obj > best_obj:
            best_obj, best_sd = val_obj, cur_sd

    if args.select == "fixed":
        best_sd = {
            k: (sum(sd[k].float() for sd in swa_sds) / len(swa_sds)).to(v.dtype) if v.is_floating_point() else v
            for k, v in swa_sds[-1].items()
        }

    model.load_state_dict(best_sd, strict=False)
    populate_p12_prototypes(model, tr_rows, img_loader, meta_store, collate, device, bs_eval, args.workers)
    if not args.no_save_ckpt:
        torch.save(model.trainable_state_dict(), ckpt_best)

    out_npz = os.path.join(out_dir, f"fold_{args.fold}_probs.npz")
    if len(va_rows) > 0:
        va_probs, va_qclean, gva, va_names = predict_p12_tta(model, va_rows, img_loader, meta_store, collate, device, bs_eval, args.workers)
        va_p0, va_p1, va_p2 = simplex_6x4_to_p012(va_probs)
        np.savez_compressed(
            out_npz,
            fold=np.int64(args.fold),
            va_probs_6x4=va_probs,
            va_qclean=va_qclean,
            va_p0=va_p0,
            va_p1=va_p1,
            va_p2=va_p2,
            gva=gva,
            va_names=np.array(va_names),
        )
    else:
        np.savez_compressed(out_npz, fold=np.int64(args.fold))
    print(f"[P12 Specialist] DONE fold {args.fold} -> {out_npz}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="pe_core_large",
                    choices=["pe_core_large", "pe_core_g14", "cradio_v4_h", "siglip2_so400m_naflex", "siglip2_giant", "internvit_6b"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--exclude-fold", type=int, default=-1,
                    help=">=0: drop this outer fold from training too (for strictly leak-free nested teacher building)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-dir", default="data/qingyu_splits")
    ap.add_argument("--dual-engine-npz", default="results/r55/semantic_roi_meta.npz")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr-backbone", type=float, default=4e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--unfreeze-blocks", type=int, default=8)
    ap.add_argument("--lora-rank", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crop-jitter", type=int, default=32)
    ap.add_argument("--soft-sigma", type=float, default=0.30)
    ap.add_argument("--lambda-bmargin", type=float, default=0.30)
    ap.add_argument("--nested-kd-npz", default="")
    ap.add_argument("--lambda-kd", type=float, default=0.0)
    ap.add_argument("--select", default="fixed", choices=["fixed", "val"])
    ap.add_argument("--fast-fixed", action="store_true",
                    help="Under --select fixed, skip logging-only per-epoch val TTA + pre-SWA E-1 prototype populate (keeps live E-2 populate)")
    ap.add_argument("--cond-b1", action="store_true",
                    help="Use conditional moderate+ boundary log-odds logsumexp(z_{2..3}) - z_1 in BoundaryMarginXBMLoss")
    ap.add_argument("--no-save-ckpt", action="store_true")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
