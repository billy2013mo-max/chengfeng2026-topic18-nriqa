#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/r55/native_tiles.py — a tier-1 tower that sees EVERY pixel of the frame at its original resolution.

Motivation (owner, 2026-11-13): the deployed towers look at a whole-frame view shrunk 2-3x (pe_core_large squashes the
frame to 448x448) plus four guided 448-px crops that cover about a fifth of a 720p frame each, so small or fine local
defects outside the crops are seen only at reduced resolution.  Here the frame is cut into 448x448 tiles at native
resolution (stride 448, the last row / column aligned to the frame edge, so every pixel is covered, some twice), every tile goes
through the shared backbone, and a masked attention / max / mean pooling over tiles feeds the usual head.

  mode "tiles"         tile features only; the "global" slot of the aggregator is the masked mean over tiles
  mode "tiles+global"  additionally the whole frame squashed to 448x448 (as the deployed pe_core_large tower), as slot 0

Training: at most `cap` tiles per frame, drawn at random each time (720p = 6 tiles, 1080p = 15, 4K = 45);
validation / inference: all tiles.  Nothing is resized or re-encoded: tiles are cut from the decoded frame.
Hooks: Tier1DualEngineDataset / make_tier1_collate_fn / forward_batch_on_device in
qingyu/train_tier1_5backbones_8fold.py carry the additive `native_tiles` branches; `train_testbed.py --native-tiles`.
"""
import torch
import torch.nn as nn

from qingyu.backbones_qingyu import QingyuVisionQualityTower

TILE = 448


def tile_starts(n: int, size: int = TILE):
    if n <= size:
        return [0]
    s = list(range(0, n - size + 1, size))
    if s[-1] != n - size:
        s.append(n - size)
    return s


class NativeTileTower(QingyuVisionQualityTower):
    def __init__(self, *a, use_global: bool = False, **k):
        super().__init__(*a, **k)
        self.native_tiles = True
        self.use_global = bool(use_global)
        self.mode_tag = "native_tiles"
        self.global_slot_dim = self.view_agg.gate[0].normalized_shape[0]
        self.capture_tiles = False
        self._tile_capture_buf = []
        self.tile_chunk = 16

    def forward(self, views_01=None, phys_feat=None, tile_mask=None, global_01=None, **_):
        B, T = views_01.shape[:2]
        valid = tile_mask.bool()
        flat = self._norm_img(views_01[valid])                               # (N, 3, 448, 448): only real tiles, no padding
        csz = max(1, int(getattr(self, "tile_chunk", 16)))
        chunks = [self._encode_single_view(flat[s:s + csz]) for s in range(0, flat.shape[0], csz)]
        emb = torch.cat(chunks, 0)
        feats = emb.new_zeros((B, T, emb.shape[-1]))
        feats[valid] = emb
        g = self._encode_single_view(self._norm_img(global_01)) if self.use_global else None
        with torch.autocast(device_type="cuda" if feats.is_cuda else "cpu", enabled=False):
            f = feats.float()
            m = valid.float()
            mean = (f * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
            lg = self.view_agg.gate(f).squeeze(-1).masked_fill(~valid, -1e4)
            w = torch.softmax(lg, dim=1).unsqueeze(-1)
            attn = (w * f).sum(1)
            mx = f.masked_fill(~valid.unsqueeze(-1), -1e4).amax(1)
            slot0 = g.float() if g is not None else mean
            vis = torch.cat([slot0, attn, mx], dim=-1)
            out = self.head(vis, phys_feat=phys_feat.float() if phys_feat is not None else None)
            if getattr(self, "capture_tiles", False):
                for bi in range(B):
                    ft = f[bi, valid[bi]]
                    Tv = ft.shape[0]
                    vis_tile = torch.cat([slot0[bi : bi + 1].expand(Tv, -1), ft, ft], dim=-1)
                    pf_t = phys_feat[bi : bi + 1].float().expand(Tv, -1) if phys_feat is not None else None
                    out_tile = self.head(vis_tile, phys_feat=pf_t)
                    wv = w[bi, valid[bi], 0]
                    wv = wv / wv.sum().clamp(min=1e-8)
                    self._tile_capture_buf.append(
                        (
                            out_tile["probs_6x4"].detach().float().cpu().numpy(),
                            out_tile["severity_s"].detach().float().cpu().numpy(),
                            wv.detach().float().cpu().numpy(),
                        )
                    )
            return out
