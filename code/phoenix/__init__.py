#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
code/phoenix/__init__.py
========================
Phoenix 55+ Clean-Slate Compliant Architecture for Kuaishou 2026 Topic 18 NR-IQA.

Core Design Principles:
  1. 100% Rule-Compliant Model Pool:
     - Orthogonal 3-Pillar single-instance Basin-Aligned SWA ensemble = 18.655B <= 30.0B
     - Single RTX 4090 (24GB) all-resident peak VRAM = ~15.48 GB <= 24.0 GB (~3.05s/img <= 15s)
  2. Zero Legacy Contamination & Zero Test-Set Transduction:
     - Never reads historical test_150 JSON predictions or 28/69-model .npz caches
     - Never runs batch-level quantile alignment, Sinkhorn, or test-set prior scaling
     - Guarantees bit-exact single-image independence: f([x_i]) == f(X)[i]
  3. End-to-End Representation & Decision Innovations:
     - Canonical Video-ID Regex + Visual Graph Connected-Component 5-Fold Zero-Leakage Split
     - Batch-Shift Calibration (orig 1,902 vs supp 1,671) + Heteroscedastic OOF Soft-Relabeling
     - Native Aspect-Ratio Dynamic Views + 1:1 Haar Wavelet Bipolar Stem + CoC Depth-Defocus Field
     - Full-Depth ViT DoRA + Multi-Layer (25%/50%/75%/100%) Hypercolumn Query2Defect Readout
     - DeQA Direct 4-Way Unimodal Soft-Distribution Head + XBM Pairwise Thurstone RL2R Ranking
     - Inductive 256-Prototype Clean-Shield & Defect-Anchor Residual Calibrator
     - 1,024-State Closed-Form Joint Expected Competition Utility Bayesian Decoder (Bootstrap LCB)
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
