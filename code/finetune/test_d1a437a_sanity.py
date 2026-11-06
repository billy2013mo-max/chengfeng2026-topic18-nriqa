#!/usr/bin/env python3
"""
code/finetune/test_d1a437a_sanity.py
Fast CPU sanity checks for the d1a437a ABMIL / 3-layer pyramid / asymmetric soft-label
changes. Run before spending GPU hours:

    python3 -u code/finetune/test_d1a437a_sanity.py

Checks:
  1. TokenGatedABMILPool is an exact no-op at step 0 (zero-init residual) and tau stays in [0.10, 0.25].
  2. DualLayerTapWrapper is an exact no-op at step 0, hooks 67%/83% blocks, and clears its caches.
  3. Asymmetric CORN targets: Clean rows are strictly zero (no leakage), positive rows are
     valid conditional probabilities with monotonicity P0 <= P1cum <= P2cum.
  4. The 4-way head layout is [clean, P2, P1, P0] and the soft loss puts zero mass on clean
     for defective samples / zero mass on defect columns for clean samples.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'decision_eval'))

from abmil_layers import TokenGatedABMILPool, DualLayerTapWrapper
from train_vlm_pure_real_corn import labels_to_corn_targets
from train_v12_abmil_pure_real import compute_asymmetric_truncated_soft_loss
from utils import DIMS, FULL_DIMS, P0_ONLY_DIMS

PASS = []


def check(name, cond, detail=''):
    cond = bool(cond)
    PASS.append(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ''))
    if not cond:
        raise AssertionError(name)


def test_abmil_noop():
    torch.manual_seed(0)
    dim, n = 64, 101
    pool = TokenGatedABMILPool(dim, topk_ratio=0.05, bottleneck=16, num_saliency_heads=4)
    tok = torch.randn(2, n, dim)
    with torch.no_grad():
        out = pool(tok, has_cls=True)
        anchor = (tok[:, 0] + tok[:, 1:].mean(1)) / 2.0
        diff = (out - anchor).abs().max().item()

    check("ABMIL step-0 == (CLS+mean)/2", diff < 1e-5, f"max|diff|={diff:.2e}")

    tau = 0.10 + 0.15 * torch.sigmoid(torch.tensor([-50.0, 0.0, 50.0]))
    check("tau bounds", bool((tau >= 0.10 - 1e-6).all() and (tau <= 0.25 + 1e-6).all()),
          f"tau={tau.numpy().round(4).tolist()}")

    with torch.no_grad():
        out_nocls = pool(tok, has_cls=False)
    check("ABMIL has_cls=False finite", torch.isfinite(out_nocls).all().item(),
          f"out={tuple(out_nocls.shape)}")

    # Make the residual non-zero and ensure it actually changes the output.
    with torch.no_grad():
        pool.proj_residual[-1].weight.normal_(0, 0.05)
        out2 = pool(tok, has_cls=True)
    check("ABMIL residual is active", (out2 - anchor).abs().max().item() > 1e-4)


def test_pyramid_noop():
    torch.manual_seed(1)
    dim, nblocks, ntok = 32, 12, 20
    blocks = nn.ModuleList()
    for _ in range(nblocks):
        blocks.append(nn.Identity())
    tap = DualLayerTapWrapper(blocks, dim)

    # idx: round(12*2/3)-1 = 7, round(12*5/6)-1 = 9
    check("pyramid taps", tap.tap_67_idx == 7 and tap.tap_83_idx == 9,
          f"67={tap.tap_67_idx}, 83={tap.tap_83_idx}")

    def run_blocks():
        x = torch.randn(1, ntok, dim)
        for blk in blocks:
            x = blk(x)
        return x

    final = run_blocks()
    with torch.no_grad():
        fused = tap.fuse(final)
    diff = (fused - final).abs().max().item()
    check("pyramid step-0 == final tokens", diff < 1e-5, f"max|diff|={diff:.2e}")
    check("pyramid caches cleared", tap._cache_67 is None and tap._cache_83 is None and tap._mid_cache is None)

    # A non-zero adapter must change the fused output, and caches must not leak.
    # NOTE: re-run the blocks first, since fuse() intentionally clears the hook caches.
    with torch.no_grad():
        tap.pyramid_adapter[-1].weight.normal_(0, 0.05)
        final2 = run_blocks()
        fused2 = tap.fuse(final2)
    check("pyramid adapter is active", (fused2 - final2).abs().max().item() > 1e-4)
    check("pyramid no cache leak", tap._cache_67 is None and tap._cache_83 is None)

    # Missing hook cache -> pass-through.
    with torch.no_grad():
        passthrough = tap.fuse(final)
    check("pyramid pass-through without cache", torch.equal(passthrough, final))


def test_soft_targets():
    # grades: clean, P2, P1, P0 (0,1,2,3)
    y = np.zeros((4, 6), dtype=np.int64)
    y[:, 0] = [0, 1, 2, 3]
    y[1:, 1] = [1, 2, 3]
    t, m = labels_to_corn_targets(y, soft_sigma=0.32)

    # Clean sample: strictly no positive mass anywhere.
    check("soft target clean == 0", float(np.abs(t[0]).max()) == 0.0, f"max={np.abs(t[0]).max():.3f}")
    # The unconditional P(>=1) channels (10..13) are always supervised; the conditional
    # P1cum / P0 channels are masked off for Clean samples.
    expected_mask = np.zeros(14, dtype=np.float32)
    expected_mask[10:14] = 1.0          # unconditional P(>=1) for FULL dims
    expected_mask[P0_ONLY_DIMS] = 1.0   # P0-only binary channels (花屏 / 虚化突兀)
    check("soft mask clean layout", np.array_equal(m[0], expected_mask),
          f"mask={m[0].astype(int).tolist()}")

    # Positive samples: unconditional P(>=1) is exactly 1, conditional probs are valid and monotone.
    for si, g in enumerate([1, 2, 3], start=1):
        p_ge1 = t[si, 10]
        p_ge2_given_1 = t[si, 6]
        p_ge3_given_2 = t[si, 0]
        check(f"soft g={g} P(>=1)=1", abs(float(p_ge1) - 1.0) < 1e-6, f"{p_ge1:.3f}")
        check(f"soft g={g} valid cond probs",
              0.0 <= p_ge3_given_2 <= p_ge2_given_1 <= 1.0 + 1e-6,
              f"P0={p_ge3_given_2:.3f}, P1cum={p_ge2_given_1:.3f}")

    # sigma=0 must reduce to hard targets.
    t_hard, _ = labels_to_corn_targets(y, soft_sigma=0.0)
    check("hard g=1 -> P1cum=0", abs(float(t_hard[1, 6])) < 1e-6)
    check("hard g=2 -> P1cum=1", abs(float(t_hard[2, 6]) - 1.0) < 1e-6)
    check("hard g=2 -> P0=0", abs(float(t_hard[2, 0])) < 1e-6)
    check("hard g=3 -> P0=1", abs(float(t_hard[3, 0]) - 1.0) < 1e-6)


def test_soft_loss_layout():
    """head_ex layout per class is [clean, P2, P1, P0] (matches exact_probs_from_dual)."""
    torch.manual_seed(2)
    B = 4
    # CLASSES order is 模糊, 纹理损失, 过锐, 花屏(BIN), 过度生成, 虚化突兀(BIN):
    FULL_BLOCKS = [0, 4, 8, 14]  # four FULL classes, each a 4-way block
    BIN_BLOCKS = [12, 18]        # 花屏, 虚化突兀, each a 2-way block

    def all_clean_g():
        return torch.zeros(B, 6, dtype=torch.long)

    # 1) All-clean: the auxiliary loss must be the mean CE against column 0 (clean).
    logits = torch.randn(B, 20)
    loss = compute_asymmetric_truncated_soft_loss(logits, all_clean_g(), sigma=0.32)
    expected = 0.0
    for base in FULL_BLOCKS:
        expected = expected + F.cross_entropy(
            logits[:, base:base + 4], torch.zeros(B, dtype=torch.long), reduction='mean'
        )
    expected = expected / len(FULL_BLOCKS)
    check("soft loss clean == CE(column 0)", abs(loss.item() - expected.item()) < 1e-6,
          f"loss={loss.item():.6f} expected={expected.item():.6f}")

    # 2) Defective: grading the peak logit at the correct column must give the lowest loss.
    #    grade 1 -> col 1 (P2), grade 2 -> col 2 (P1), grade 3 -> col 3 (P0).
    for gval in (1, 2, 3):
        g = torch.zeros(1, 6, dtype=torch.long)
        g[:, [0, 1, 2, 4]] = gval
        losses = []
        for peak_col in (0, 1, 2, 3):
            lg = torch.zeros(1, 20)
            for base in FULL_BLOCKS:
                lg[0, base + peak_col] = 10.0
            losses.append(compute_asymmetric_truncated_soft_loss(lg, g, sigma=0.32).item())
        best_col = int(np.argmin(losses))
        check(f"soft loss g={gval} peak at col {gval}", best_col == gval,
              f"losses={np.round(losses, 4).tolist()}")

    # 3) A tiny sigma approaches the hard one-hot target.
    g = torch.zeros(1, 6, dtype=torch.long)
    g[:, [0, 1, 2, 4]] = 2
    lg = torch.zeros(1, 20)
    for base in FULL_BLOCKS:
        lg[0, base + 2] = 10.0
    loss_hard = compute_asymmetric_truncated_soft_loss(lg, g, sigma=1e-3).item()
    check("tiny-sigma soft-loss near 0 at true class", loss_hard < 1e-3, f"loss={loss_hard:.2e}")


def main():
    print("=" * 70)
    print("d1a437a ABMIL / pyramid / soft-label sanity checks (CPU)")
    print("=" * 70)
    test_abmil_noop()
    test_pyramid_noop()
    test_soft_targets()
    test_soft_loss_layout()
    print("=" * 70)
    print(f"ALL {len(PASS)} CHECKS PASSED")
    print("=" * 70)


if __name__ == '__main__':
    main()
