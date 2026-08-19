"""
Band readout: three states, not a scale.

Companion to `band-not-scale.md`. Where `hands_lie_detector.scoring` sums seven
monotone categories — so its composite rises with mean thickness and ranks a
saturated, undifferentiated hand highest — this package reads the thickness map
for CONTRAST, which separates all three states.

Scope, per `readout-channel.md`: this reads the INTEGRATED map. It does not
decompose it into domains.
"""

from .contrast import (
    DEFAULT_THRESHOLDS,
    MANAGEMENT_ACTS,
    BandReadout,
    ContrastThresholds,
    HandState,
    ThicknessReading,
    interpret_acute_damage,
    monotone_score,
    read_band,
)

__all__ = [
    "BandReadout",
    "ContrastThresholds",
    "DEFAULT_THRESHOLDS",
    "HandState",
    "MANAGEMENT_ACTS",
    "ThicknessReading",
    "interpret_acute_damage",
    "monotone_score",
    "read_band",
]
