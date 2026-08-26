"""
Palmar thickness-map readout: concentration, not skill.

Companion to `band-not-scale.md`. The scoring rubric is monotone in thickness
and ranks a saturated hand highest. This package reads the map's dispersion
instead — but dispersion measures the geometric CONCENTRATION of the load
history, not competence, and a thickness map alone cannot separate a
variable-geometry generalist from a saturated hand. That separation is a
functional sensing test.

Scope, per `readout-channel.md`: reads the INTEGRATED palmar map. It does not
decompose it into domains, and it does not read dorsal marks — those are events
on a different clock (`integration.event_log`).
"""

from .capture import (
    POSITIONS,
    TIER_2_POSITION,
    CaptureSession,
    Position,
    PositionSpec,
    Trigger,
)
from .contrast import (
    DEFAULT_THRESHOLDS,
    NEUTRAL_CALIBRATION,
    BiologicalCalibration,
    Sex,
    MANAGEMENT_ACTS,
    BandPosition,
    BandReadout,
    ContrastThresholds,
    LightCondition,
    MapState,
    Sensing,
    ThicknessReading,
    band_position,
    interpret_acute_damage,
    monotone_score,
    read_band,
)

__all__ = [
    "Trigger",
    "PositionSpec",
    "Position",
    "CaptureSession",
    "TIER_2_POSITION",
    "POSITIONS",
    "BiologicalCalibration",
    "NEUTRAL_CALIBRATION",
    "Sex",
    "BandPosition",
    "BandReadout",
    "ContrastThresholds",
    "DEFAULT_THRESHOLDS",
    "LightCondition",
    "MANAGEMENT_ACTS",
    "MapState",
    "Sensing",
    "ThicknessReading",
    "band_position",
    "interpret_acute_damage",
    "monotone_score",
    "read_band",
]
