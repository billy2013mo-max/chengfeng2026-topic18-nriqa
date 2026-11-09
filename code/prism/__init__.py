"""
PRISM: Pairwise Reference-anchored Integrated Severity Microscope
=================================================================
Next-generation architecture for Topic-18 No-Reference Image Quality Assessment (NR-IQA),
replacing the legacy [6B + 4 Domain Specialists + Dual-Subtraction] pipeline with:
  1. Sequential Conditional Hazard Decomposition (`heads_and_losses.py`):
     Eliminates P1 interval squeezing (`s1 - s0`) and Gaussian soft-label smearing (`sigma=0.32`).
  2. Scene-Matched Reference-Anchored Comparator (`reference_anchor_net.py`):
     Resolves adjacent severity grade confusion (`P2 <-> P1` and `P1 <-> P0`) by comparing
     each image against its Top-K scene-matched real training anchors of known grades.
  3. Real-Defect Multi-Band Poisson Ladder & Differential Calibration (`real_defect_synthesizer.py`):
     Uses synthetic and real-blended quadruplets strictly in the pairwise differential operator
     with Gradient Reversal domain invariance, preventing single-image prior shift.
  4. Three Integrated Full-Spectrum Microscope Towers (`train_prism_quad.py`, `train_prism_vit6b.py`,
     `train_prism_vlm8b.py`) + Unified OOF Reference-Anchored Evaluator (`evaluate_prism.py`).
"""

from .heads_and_losses import (
    CLASSES,
    FULL_CLASSES,
    P0_ONLY_CLASSES,
    GradientReversalLayer,
    ConditionalHazardOrdinalHead,
    PrismExactGradeLoss,
    PrismDifferentialCalibratorLoss,
    convert_p012_to_conditional_hazards,
    convert_conditional_hazards_to_p012,
)
from .reference_anchor_net import (
    SceneMatchedAnchorBank,
    AnchorCrossAttentionComparator,
    fit_and_apply_reference_anchor_calibrator,
)

__all__ = [
    "CLASSES",
    "FULL_CLASSES",
    "P0_ONLY_CLASSES",
    "GradientReversalLayer",
    "ConditionalHazardOrdinalHead",
    "PrismExactGradeLoss",
    "PrismDifferentialCalibratorLoss",
    "convert_p012_to_conditional_hazards",
    "convert_conditional_hazards_to_p012",
    "SceneMatchedAnchorBank",
    "AnchorCrossAttentionComparator",
    "fit_and_apply_reference_anchor_calibrator",
]
