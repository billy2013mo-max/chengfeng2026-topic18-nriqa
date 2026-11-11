#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/phoenix/train_pillar3_hires_physics.py
===========================================
Breakthrough 2 & 3: Pillar-3 1:1 High-Frequency & Optical Physics Specialist (`~1.555B`).

Architecture Components (100% Orthogonal to VLMs):
  1. `DINOv3-ViT-L/16` (`0.304B`) + `C-RADIOv4-H` (`0.630B`) with 4-Depth Intermediate
     Hypercolumn Taps (`25%, 50%, 75%, 100%`) + Full-Depth DoRA -> preserves early/mid-layer
     oversharpening overshoot ripples (`过锐`) & smooth skin/grass smearing (`纹理损失`)
     before deep semantic pooling discards them!
  2. `ConvNeXtV2-Large` (`0.200B`) + `1:1` 2D Haar Wavelet & Bipolar Overshoot ConvStem
     -> directly ingests unscaled `1:1` high-frequency wavelet subbands `(LH, HL, HH)`,
     USM bipolar halo envelope, and `8x8` macroblock glitch maps (`花屏`).
  3. `FGResQ` Neural Restoration Degradation Adapter (`0.350B` budget) + `PAL4VST` Spatial
     Artifact Saliency Probe (`0.046B`) + `CoC Depth-Defocus Jacobian Field` (`0.025B` frozen
     `Depth-Anything-V2-Small`) -> pinpoints `虚化突兀` (same-depth sharpness cliffs) and
     `过度生成` (localized generative implausibility).
  4. `PhoenixUnimodalDeQAHead` + `DeQAOrdinalDistributionLoss` + `CrossBatchThurstoneRL2RLoss`
     + `BasinAlignedSWACollector` (`cv5`, `stage2-rl2r`, `fulldata-swa`, and `--smoke-test`).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(THIS_DIR) not in sys.path:
    sys.path.insert(0, os.path.dirname(THIS_DIR))

from phoenix.config import (
    PHOENIX_CKPT_DIR,
    PHOENIX_OOF_DIR,
    PHOENIX_SPLITS_DIR,
    PureRealImageResolver,
    evaluate_official_metrics,
    resolve_local_or_hf_model,
    set_deterministic_seed,
)
from phoenix.losses_and_heads import (
    BasinAlignedSWACollector,
    CrossBatchThurstoneRL2RLoss,
    DeQAOrdinalDistributionLoss,
    PhoenixUnimodalDeQAHead,
)
from phoenix.native_physics_and_views import (
    CoCDepthDefocusProbe,
    compute_1to1_wavelet_and_artifact_maps,
    resize_native_aspect_pil,
    select_scrutiny_1to1_patches,
)
from phoenix.stage0_real_restoration_tapt import (
    FGResQDegradationProjector,
    PAL4VSTArtifactSaliencyProbe,
)
from phoenix.train_pillar12_native_vlm import load_batch_shift_init, load_split_rows


class HaarWaveletCoCPhysicsStem(nn.Module):
    """
    1.8M-param explicit physical inductive-bias stem encoding:
      - `haar_hf` (3 ch: LH, HL, HH) + `bipolar_halo` (1 ch) + `macroblock_glitch` (1 ch)
        + `ui_text_mask` (1 ch) + `coc_jacobian_field` (3 ch: Z_hat, E_HF, CoC_Cliff) = 9 channels!
    Outputs `[B, 36, out_dim]` spatial tokens aligned to the Query2Defect head.
    """

    def __init__(self, in_channels: int = 9, out_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.Conv2d(192, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, out_dim),
            nn.GELU(),
        )

    def forward(self, phys_9ch: torch.Tensor) -> torch.Tensor:
        fmap = self.net(phys_9ch)
        pooled = F.adaptive_avg_pool2d(fmap, output_size=(6, 6))
        return pooled.flatten(2).transpose(1, 2)  # [B, 36, out_dim]


class MultiLayerHypercolumnVisionTower(nn.Module):
    """
    Loads `DINOv3-ViT-L/16`, `C-RADIOv4-H`, and `ConvNeXtV2-Large`, tapping 4 intermediate
    stages `(25%, 50%, 75%, 100%)` from each tower and projecting them into a unified `out_dim`
    hypercolumn token sequence. Supports offline mock mode for unit testing.
    """

    def __init__(self, out_dim: int = 512, mock_backbone: bool = False) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.mock_backbone = mock_backbone

        if mock_backbone:
            self.mock_conv = nn.Sequential(
                nn.Conv2d(3, 128, kernel_size=7, stride=4, padding=3),
                nn.GELU(),
                nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d((6, 6)),
            )
        else:
            self.dinov3, d_dim = self._load_hf_vision("pillar3_dinov3_l")
            self.cradio, r_dim = self._load_hf_vision("pillar3_cradio_v4_h")
            self.convnext, c_dim = self._load_convnext("pillar3_convnextv2_l")
            self.proj_dino = nn.Linear(d_dim * 4, out_dim)
            self.proj_radio = nn.Linear(r_dim * 4, out_dim)
            self.proj_convnext = nn.Linear(c_dim, out_dim)

    @staticmethod
    def _load_hf_vision(model_key: str):
        from transformers import AutoConfig, AutoModel

        path = resolve_local_or_hf_model(model_key)
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=dtype)
        # Enable full-depth tuning via DoRA if PEFT is available, else unfreeze LayerNorm + QKV
        try:
            from peft import LoraConfig, get_peft_model

            linear_names = sorted(
                {
                    n.split(".")[-1]
                    for n, m in model.named_modules()
                    if isinstance(m, nn.Linear) and any(k in n for k in ("q", "k", "v", "proj", "fc1", "fc2"))
                }
            )
            if linear_names:
                model = get_peft_model(
                    model,
                    LoraConfig(r=16, lora_alpha=32, target_modules=linear_names, use_dora=True),
                )
        except Exception:
            pass
        hdim = int(getattr(cfg, "hidden_size", 1024))
        return model, hdim

    @staticmethod
    def _load_convnext(model_key: str):
        path = resolve_local_or_hf_model(model_key)
        try:
            import timm

            if os.path.isdir(path):
                from transformers import AutoConfig, AutoModel

                cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
                m = AutoModel.from_pretrained(path, trust_remote_code=True)
                dims = getattr(cfg, "hidden_sizes", [192, 384, 768, 1536])
                return m, int(dims[-1])
            m = timm.create_model(path, pretrained=True, num_classes=0, global_pool="")
            return m, int(m.num_features)
        except Exception:
            from transformers import AutoConfig, AutoModel

            cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
            m = AutoModel.from_pretrained(path, trust_remote_code=True)
            dims = getattr(cfg, "hidden_sizes", [192, 384, 768, 1536])
            return m, int(dims[-1])

    @staticmethod
    def _extract_4layer_hypercolumn(model: nn.Module, x: torch.Tensor, target_tokens: int = 36) -> torch.Tensor:
        out = model(pixel_values=x.to(next(model.parameters()).dtype), output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        n = len(hs)
        taps = [hs[max(0, n // 4)], hs[max(0, n // 2)], hs[max(0, (3 * n) // 4)], hs[-1]]
        pooled_taps = []
        for t in taps:
            tf = t.float()
            if tf.ndim == 4:  # [B, C, H, W]
                p = F.adaptive_avg_pool2d(tf, output_size=(6, 6)).flatten(2).transpose(1, 2)
            else:             # [B, L, C]
                p = F.adaptive_avg_pool1d(tf.transpose(1, 2), output_size=target_tokens).transpose(1, 2)
            pooled_taps.append(p)
        return torch.cat(pooled_taps, dim=-1)  # [B, 36, 4*D]

    def forward(self, views_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
          views_tensor: `[B, 3_views, 3, 384, 384]` normalized RGB tensor (View 0 = aspect-padded
                        global, View 1 & 2 = unscaled 1:1 384x384 scrutiny patches).
        Returns:
          `[B, T_vis, out_dim]` visual hypercolumn tokens.
        """
        bsz, n_v, c, h, w = views_tensor.shape
        flat_v = views_tensor.reshape(bsz * n_v, c, h, w)

        if self.mock_backbone:
            fmap = self.mock_conv(flat_v)  # [B*3, out_dim, 6, 6]
            return fmap.flatten(2).transpose(1, 2).reshape(bsz, n_v * 36, self.out_dim)

        hc_dino = self.proj_dino(self._extract_4layer_hypercolumn(self.dinov3, flat_v, 36))
        hc_radio = self.proj_radio(self._extract_4layer_hypercolumn(self.cradio, flat_v, 36))

        try:
            cx_out = self.convnext(flat_v.to(next(self.convnext.parameters()).dtype))
            if hasattr(cx_out, "last_hidden_state"):
                cx_feat = cx_out.last_hidden_state.float()
            else:
                cx_feat = cx_out.float()
            if cx_feat.ndim == 4:
                cx_tok = self.proj_convnext(
                    F.adaptive_avg_pool2d(cx_feat, output_size=(6, 6)).flatten(2).transpose(1, 2)
                )
            else:
                cx_tok = self.proj_convnext(
                    F.adaptive_avg_pool1d(cx_feat.transpose(1, 2), output_size=36).transpose(1, 2)
                )
        except Exception:
            cx_tok = torch.zeros_like(hc_dino)

        fused_per_view = (hc_dino + hc_radio + cx_tok) / 3.0  # [B*3, 36, out_dim]
        return fused_per_view.reshape(bsz, n_v * 36, self.out_dim)


class Pillar3HiResPhysicsSpecialist(nn.Module):
    """Complete Pillar-3 model combining Vision Hypercolumns + Wavelet/CoC Physics + Restoration Probes + DeQA Head."""

    def __init__(
        self,
        init_batch_shift: Optional[np.ndarray] = None,
        stage0_ckpt_path: Optional[str] = None,
        mock_backbone: bool = False,
    ) -> None:
        super().__init__()
        token_dim = 512
        self.vision_tower = MultiLayerHypercolumnVisionTower(out_dim=token_dim, mock_backbone=mock_backbone)
        self.physics_stem = HaarWaveletCoCPhysicsStem(in_channels=9, out_dim=token_dim)
        self.fgresq_branch = FGResQDegradationProjector(in_channels=8, feat_dim=256)
        self.pal4vst_probe = PAL4VSTArtifactSaliencyProbe(in_channels=8, hidden_dim=192)
        self.probe_proj = nn.Linear(256, token_dim)

        if stage0_ckpt_path and os.path.exists(stage0_ckpt_path):
            ckpt = torch.load(stage0_ckpt_path, map_location="cpu")
            if "fgresq_state_dict" in ckpt:
                self.fgresq_branch.load_state_dict(ckpt["fgresq_state_dict"], strict=False)

        self.head = PhoenixUnimodalDeQAHead(
            in_dim=token_dim,
            hidden_dim=512,
            num_heads=8,
            dropout=0.10,
            init_batch_shift=init_batch_shift,
        )

    def forward(
        self,
        views_tensor: torch.Tensor,
        phys_9ch: torch.Tensor,
        probe_8ch: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
        apply_batch_bias: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        vis_tokens = self.vision_tower(views_tensor)                # [B, 108, 512]
        phys_tokens = self.physics_stem(phys_9ch)                   # [B, 36, 512]
        fg_tokens = self.probe_proj(self.fgresq_branch(probe_8ch)["tokens"])      # [B, 16, 512]
        pal_tokens = self.probe_proj(self.pal4vst_probe(probe_8ch)["artifact_tokens"])  # [B, 16, 512]

        all_tokens = torch.cat([vis_tokens, phys_tokens, fg_tokens, pal_tokens], dim=1)
        return self.head(all_tokens, batch_ids=batch_ids, apply_batch_bias=apply_batch_bias)


def build_pillar3_sample_tensors(
    pil_img,
    coc_probe: CoCDepthDefocusProbe,
    patch_size: int = 384,
) -> Dict[str, object]:
    """
    Constructs exact single-image tensors for Pillar-3 without distorting aspect ratio:
      - Global view is resized preserving aspect ratio inside `384x384` with neutral reflection padding
      - Patch 1 & Patch 2 are strictly `1:1` unscaled `384x384` crops
      - `phys_9ch` and `probe_8ch` capture native half-res Haar wavelets, bipolar overshoot,
        macroblock glitches, and Optical CoC Depth-Defocus cliffs resized cleanly to `256x256`.
    """
    rgb_np = np.asarray(pil_img, dtype=np.uint8)
    phys_maps = compute_1to1_wavelet_and_artifact_maps(rgb_np)
    coc_field = coc_probe.compute_coc_jacobian_field(pil_img, phys_maps)
    crops, crop_meta = select_scrutiny_1to1_patches(
        pil_img=pil_img,
        phys_maps=phys_maps,
        coc_field=coc_field,
        patch_size=patch_size,
        num_patches=2,
    )

    # Aspect-preserving global canvas (zero stretching!)
    g_pil = resize_native_aspect_pil(pil_img, max_pixels=patch_size * patch_size, grid_factor=16)
    gw, gh = g_pil.size
    g_arr = np.asarray(g_pil, dtype=np.uint8)
    canvas = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    g_clip = g_arr[:patch_size, :patch_size]
    ch_h, ch_w, _ = g_clip.shape
    y_off = (patch_size - ch_h) // 2
    x_off = (patch_size - ch_w) // 2
    canvas[y_off : y_off + ch_h, x_off : x_off + ch_w] = g_clip

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    view_arrays = []
    for arr_u8 in (canvas, np.asarray(crops[0], dtype=np.uint8), np.asarray(crops[1], dtype=np.uint8)):
        normed = (arr_u8.astype(np.float32) / 255.0 - mean) / std
        view_arrays.append(normed.transpose(2, 0, 1))
    views_tensor = torch.from_numpy(np.stack(view_arrays, axis=0)).float()  # [3, 3, 384, 384]

    # Stack 9-channel physical tensor & 8-channel restoration tensor
    phys_9_np = np.concatenate(
        [
            phys_maps["haar_hf"],                        # 3
            phys_maps["bipolar_halo"][:, :, None],       # 1
            phys_maps["macroblock_glitch"][:, :, None],  # 1
            phys_maps["ui_text_mask"][:, :, None],       # 1
            coc_field,                                   # 3
        ],
        axis=-1,
    )
    phys_9_t = F.interpolate(
        torch.from_numpy(phys_9_np.transpose(2, 0, 1)).unsqueeze(0).float(),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    probe_8_t = F.interpolate(
        PAL4VSTArtifactSaliencyProbe.build_input_tensor(rgb_np).unsqueeze(0),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    return {
        "views_tensor": views_tensor,
        "phys_9ch": phys_9_t,
        "probe_8ch": probe_8_t,
        "crop_meta": crop_meta,
    }


class Pillar3Dataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        resolver: PureRealImageResolver,
        coc_probe: CoCDepthDefocusProbe,
        is_train: bool = True,
        oof_teacher_probs: Optional[np.ndarray] = None,
    ) -> None:
        self.rows = list(rows)
        self.resolver = resolver
        self.coc_probe = coc_probe
        self.is_train = is_train
        self.oof_teacher_probs = oof_teacher_probs

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        from PIL import Image

        row = self.rows[idx]
        pil_img = self.resolver.load_rgb_pil(str(row["name"]))
        if self.is_train and np.random.rand() < 0.5:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        pack = build_pillar3_sample_tensors(pil_img, self.coc_probe)
        item: Dict[str, object] = {
            "name": str(row["name"]),
            "views_tensor": pack["views_tensor"],
            "phys_9ch": pack["phys_9ch"],
            "probe_8ch": pack["probe_8ch"],
            "batch_id": torch.tensor(int(row.get("batch_id", 0)), dtype=torch.long),
            "noisy_flag": torch.tensor(int(row.get("noisy_flag", 0)), dtype=torch.long),
            "grades": torch.tensor(row.get("grades", [0] * 6), dtype=torch.long),
        }
        if self.oof_teacher_probs is not None:
            item["teacher_probs"] = torch.from_numpy(self.oof_teacher_probs[idx]).float()
        return item


@torch.no_grad()
def predict_pillar3_dataset(
    model: Pillar3HiResPhysicsSpecialist,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    names_all: List[str] = []
    probs_l, logits_l, clean_l, p_clean_l, mu_l, var_l, emb_l, gold_l = [], [], [], [], [], [], [], []
    for batch in loader:
        names_all.extend(batch["name"])
        out = model(
            views_tensor=batch["views_tensor"].to(device),
            phys_9ch=batch["phys_9ch"].to(device),
            probe_8ch=batch["probe_8ch"].to(device),
            batch_ids=None,
            apply_batch_bias=False,
        )
        probs_l.append(out["probs_6x4"].cpu().numpy())
        logits_l.append(out["logits_6x4"].cpu().numpy())
        clean_l.append(out["clean_logit"].cpu().numpy())
        p_clean_l.append(out["p_clean_joint"].cpu().numpy())
        mu_l.append(out["mu_6d"].cpu().numpy())
        var_l.append(out["var_6d"].cpu().numpy())
        emb_l.append(out["quality_emb"].cpu().numpy())
        gold_l.append(batch["grades"].numpy())

    return {
        "names": np.array(names_all, dtype=object),
        "probs_6x4": np.concatenate(probs_l, axis=0).astype(np.float32),
        "logits_6x4": np.concatenate(logits_l, axis=0).astype(np.float32),
        "clean_logit": np.concatenate(clean_l, axis=0).astype(np.float32),
        "p_clean_joint": np.concatenate(p_clean_l, axis=0).astype(np.float32),
        "mu_6d": np.concatenate(mu_l, axis=0).astype(np.float32),
        "var_6d": np.concatenate(var_l, axis=0).astype(np.float32),
        "quality_emb": np.concatenate(emb_l, axis=0).astype(np.float32),
        "gold_6d": np.concatenate(gold_l, axis=0).astype(np.int64),
    }


def train_pillar3_run(
    model: Pillar3HiResPhysicsSpecialist,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr_head: float = 2.0e-4,
    lr_backbone: float = 3.0e-5,
    swa_start_epoch: Optional[int] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    tower_ids = {id(p) for p in model.vision_tower.parameters()}
    fast_params = [p for p in model.parameters() if p.requires_grad and id(p) not in tower_ids]
    slow_params = [p for p in model.vision_tower.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(
        [
            {"params": fast_params, "lr": lr_head, "weight_decay": 1e-2},
            {"params": slow_params, "lr": lr_backbone, "weight_decay": 1e-2},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs * len(train_loader)), eta_min=lr_backbone * 0.1
    )
    deqa_loss_fn = DeQAOrdinalDistributionLoss().to(device)
    rl2r_loss_fn = CrossBatchThurstoneRL2RLoss(queue_size=1024, emb_dim=256).to(device)
    swa = BasinAlignedSWACollector() if swa_start_epoch is not None else None

    for ep in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            y_6d = batch["grades"].to(device)
            b_ids = batch["batch_id"].to(device)
            noisy = batch["noisy_flag"].to(device)
            t_probs = batch.get("teacher_probs")
            if t_probs is not None:
                t_probs = t_probs.to(device)

            out = model(
                views_tensor=batch["views_tensor"].to(device),
                phys_9ch=batch["phys_9ch"].to(device),
                probe_8ch=batch["probe_8ch"].to(device),
                batch_ids=b_ids,
                apply_batch_bias=True,
            )
            loss_dict = deqa_loss_fn(
                head_out=out,
                y_6d=y_6d,
                noisy_flags=noisy,
                oof_teacher_probs=t_probs,
                supp_bias_param=model.head.supp_batch_bias,
            )
            loss = loss_dict["loss"] + 0.35 * rl2r_loss_fn(out, y_6d)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            running += float(loss.item())

        if swa is not None and ep >= swa_start_epoch:
            swa.update(model, weight=1.0)
        print(f"  [pillar3] Epoch {ep+1}/{epochs} | loss={running / max(1, len(train_loader)):.4f}")

    return swa.export_state_dict() if (swa is not None and swa.count > 0) else None


def run_smoke_test() -> None:
    from PIL import Image

    set_deterministic_seed(2026)
    device = torch.device("cpu")
    coc_probe = CoCDepthDefocusProbe(model_path=None, device="cpu")
    samples = []
    for i in range(6):
        pil_img = Image.fromarray(np.random.randint(0, 255, (720, 405, 3), dtype=np.uint8))
        pack = build_pillar3_sample_tensors(pil_img, coc_probe)
        samples.append(
            {
                "name": f"p3_smoke_{i:02d}.png",
                "views_tensor": pack["views_tensor"],
                "phys_9ch": pack["phys_9ch"],
                "probe_8ch": pack["probe_8ch"],
                "batch_id": torch.tensor(i % 2, dtype=torch.long),
                "noisy_flag": torch.tensor(0, dtype=torch.long),
                "grades": torch.tensor([i % 4, 0, (i + 1) % 4, 0, 0, 3 if i == 4 else 0], dtype=torch.long),
            }
        )

    dl = DataLoader(samples, batch_size=3, shuffle=False)
    model = Pillar3HiResPhysicsSpecialist(mock_backbone=True).to(device)
    swa_sd = train_pillar3_run(model, dl, device, epochs=2, swa_start_epoch=0)
    assert swa_sd is not None
    model.load_state_dict(swa_sd)
    pred = predict_pillar3_dataset(model, dl, device)
    assert pred["probs_6x4"].shape == (6, 6, 4)
    # Single-image bit-exact independence check
    dl_one = DataLoader(samples[:1], batch_size=1, shuffle=False)
    pred_one = predict_pillar3_dataset(model, dl_one, device)
    diff = float(np.max(np.abs(pred_one["probs_6x4"][0] - pred["probs_6x4"][0])))
    assert diff < 1e-6, f"Pillar-3 single-image independence failed: diff={diff}"
    print(f"[Pillar-3 HiRes Physics Smoke Test PASS] probs={pred['probs_6x4'].shape} | single_img_diff={diff:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix Pillar-3 1:1 Hi-Res & Optical Physics Specialist")
    parser.add_argument("--mode", choices=["cv5", "stage2-rl2r", "fulldata-swa"], default="cv5")
    parser.add_argument("--split-csv", default=str(PHOENIX_SPLITS_DIR / "phoenix_5fold.csv"))
    parser.add_argument("--audit-json", default=str(PHOENIX_SPLITS_DIR / "batch_shift_and_audit.json"))
    parser.add_argument("--stage0-ckpt", default=str(PHOENIX_CKPT_DIR / "stage0_tapt" / "stage0_fgresq_tapt.pt"))
    parser.add_argument("--teacher-oof-npz", default="")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=2.0e-4)
    parser.add_argument("--lr-backbone", type=float, default=3.0e-5)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--out-dir", default=str(PHOENIX_OOF_DIR))
    parser.add_argument("--ckpt-dir", default=str(PHOENIX_CKPT_DIR))
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    set_deterministic_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_split_rows(args.split_csv)
    batch_shift = load_batch_shift_init(args.audit_json)
    resolver = PureRealImageResolver()
    coc_probe = CoCDepthDefocusProbe(
        model_path=resolve_local_or_hf_model("pillar3_depth_anything_s"),
        device=str(device),
    )

    teacher_probs_all = None
    if args.teacher_oof_npz and os.path.exists(args.teacher_oof_npz):
        t_npz = np.load(args.teacher_oof_npz, allow_pickle=True)
        t_map = {str(nm): i for i, nm in enumerate(t_npz["names"])}
        teacher_probs_all = np.stack([t_npz["probs_6x4"][t_map[str(r["name"])]] for r in rows], axis=0)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.mode in ("cv5", "stage2-rl2r"):
        n = len(rows)
        oof_probs = np.zeros((n, 6, 4), dtype=np.float32)
        oof_logits = np.zeros((n, 6, 4), dtype=np.float32)
        oof_clean_l = np.zeros(n, dtype=np.float32)
        oof_p_clean = np.zeros(n, dtype=np.float32)
        oof_mu = np.zeros((n, 6), dtype=np.float32)
        oof_var = np.zeros((n, 6), dtype=np.float32)
        oof_emb = np.zeros((n, 256), dtype=np.float32)
        gold_all = np.array([r["grades"] for r in rows], dtype=np.int64)
        folds_all = np.array([int(r["fold"]) for r in rows], dtype=np.int64)

        target_folds = list(range(5)) if args.fold < 0 else [args.fold]
        for f_idx in target_folds:
            tr_idx = [i for i, r in enumerate(rows) if int(r["fold"]) != f_idx]
            va_idx = [i for i, r in enumerate(rows) if int(r["fold"]) == f_idx]
            tr_teacher = teacher_probs_all[tr_idx] if teacher_probs_all is not None else None

            ds_tr = Pillar3Dataset([rows[i] for i in tr_idx], resolver, coc_probe, True, tr_teacher)
            ds_va = Pillar3Dataset([rows[i] for i in va_idx], resolver, coc_probe, False)
            dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4)
            dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4)

            model = Pillar3HiResPhysicsSpecialist(
                init_batch_shift=batch_shift,
                stage0_ckpt_path=args.stage0_ckpt,
            ).to(device)
            swa_sd = train_pillar3_run(
                model=model,
                train_loader=dl_tr,
                device=device,
                epochs=args.epochs,
                lr_head=args.lr_head,
                lr_backbone=args.lr_backbone,
                swa_start_epoch=max(2, args.epochs - 3),
            )
            if swa_sd is not None:
                model.load_state_dict(swa_sd)

            va_out = predict_pillar3_dataset(model, dl_va, device)
            oof_probs[va_idx] = va_out["probs_6x4"]
            oof_logits[va_idx] = va_out["logits_6x4"]
            oof_clean_l[va_idx] = va_out["clean_logit"]
            oof_p_clean[va_idx] = va_out["p_clean_joint"]
            oof_mu[va_idx] = va_out["mu_6d"]
            oof_var[va_idx] = va_out["var_6d"]
            oof_emb[va_idx] = va_out["quality_emb"]

            m = evaluate_official_metrics(np.argmax(va_out["probs_6x4"], axis=-1), gold_all[va_idx])
            print(f"[pillar3 Fold {f_idx}] Naive Argmax Strict={m['strict_total']:.2f}")

        tag = f"pillar3_{args.mode}_oof.npz" if args.fold < 0 else f"pillar3_{args.mode}_fold{args.fold}.npz"
        out_npz = os.path.join(args.out_dir, tag)
        np.savez_compressed(
            out_npz,
            names=np.array([r["name"] for r in rows], dtype=object),
            folds=folds_all,
            gold_6d=gold_all,
            probs_6x4=oof_probs,
            logits_6x4=oof_logits,
            clean_logit=oof_clean_l,
            p_clean_joint=oof_p_clean,
            mu_6d=oof_mu,
            var_6d=oof_var,
            quality_emb=oof_emb,
        )
        print(f"[pillar3] Saved OOF predictions -> {out_npz}")

    elif args.mode == "fulldata-swa":
        ds_full = Pillar3Dataset(rows, resolver, coc_probe, True, teacher_probs_all)
        model = Pillar3HiResPhysicsSpecialist(
            init_batch_shift=batch_shift,
            stage0_ckpt_path=args.stage0_ckpt,
        ).to(device)
        dl_warm = DataLoader(ds_full, batch_size=args.batch_size, shuffle=True, num_workers=4)
        warm_epochs = max(2, args.epochs - 3)
        train_pillar3_run(model, dl_warm, device, epochs=warm_epochs, lr_head=args.lr_head, lr_backbone=args.lr_backbone)
        warm_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        basin_swa = BasinAlignedSWACollector()
        for b_idx, seed in enumerate((2026, 2027, 2028)):
            set_deterministic_seed(seed)
            model.load_state_dict(warm_state)
            dl_b = DataLoader(ds_full, batch_size=args.batch_size, shuffle=True, num_workers=4)
            train_pillar3_run(
                model,
                dl_b,
                device,
                epochs=3,
                lr_head=args.lr_head * 0.30,
                lr_backbone=args.lr_backbone * 0.30,
            )
            basin_swa.update(model, weight=1.0)
            print(f"[pillar3 FullData Basin-SWA] Collected branch {b_idx+1}/3 (seed={seed})")

        ckpt_path = os.path.join(args.ckpt_dir, "pillar3_fulldata_swa.pt")
        torch.save({"pillar": "pillar3", "state_dict": basin_swa.export_state_dict()}, ckpt_path)
        print(f"[pillar3] Exported SINGLE Basin-Aligned SWA checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
