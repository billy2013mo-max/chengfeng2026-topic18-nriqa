# -*- coding: utf-8 -*-
"""
Qingyu Breakthrough NR-IQA Engineering Package (Topic 18).
Implements the 5-Pillar Architecture from `无参画质检测瓶颈破局方案.pdf`:
  - Pillar 1: 8-Fold Scene-Clustered Stratified Split + Pure-Real XBM(512) Thurstone Ranking + Tip-Adapter Prototypes
  - Pillar 2: Dual-Engine 1:1 Physics (MEOR + UI Stroke Mask + Depth-Anything-V2 S_fg & A_cutout + 4x448 Crops + 2x384 ROIs)
  - Pillar 3: Multi-Image Interleaved-MRoPE Tier-3 (Clef-Flash-9B System-1 JointSchemaHead + Qwen3-VL-8B) + Certainty-Gated Stacking
  - Pillar 4: Unimodal Quadratic-Residual Ordinal Head + 30-Step Damped Frank-Wolfe Analytical Jacobian 3-Candidate Bayesian Decoder
  - Pillar 5: Single-RTX-4090 Full-Resident Model Soup (25.22B <= 30B, ~20.07GB <= 24GB) + KV-Cache Grounded Attribution + Video Hysteresis
"""

__version__ = "1.0.0"
