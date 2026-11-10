"""
PRISM: Pairwise Reference-anchored Integrated Severity Microscope
=================================================================
Next-generation architecture for Topic-18 No-Reference Image Quality Assessment (NR-IQA).
"""
import os
import sys

_PRISM_DIR = os.path.dirname(os.path.abspath(__file__))
if _PRISM_DIR not in sys.path:
    sys.path.insert(0, _PRISM_DIR)

from heads_and_losses import (
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
from reference_anchor_net import (
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
