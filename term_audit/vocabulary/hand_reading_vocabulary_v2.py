"""
term_audit/vocabulary/hand_reading_vocabulary_v2.py

Structured vocabulary for describing hands as substrate evidence.
v0.2 adds:
- Biological sex calibration (female skin elasticity)
- Observation quality (lighting, camera, manipulation)
- Falsification method (passive vs. induced observation)
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional

# ===========================================================================
# Enums (v0.2)
# ===========================================================================

class CleanlinessLevel(Enum):
    EMBEDDED_DIRT = "embedded_dirt"
    WORK_STAINED = "work_stained"
    CLEAN_BUT_WORKED = "clean_but_worked"
    CLEAN_UNWORKED = "clean_unworked"
    UNKNOWN = "unknown"

class CallusPattern(Enum):
    NONE_VISIBLE = "none_visible"
    SINGLE_DEEP = "single_deep"
    DISTRIBUTED_MODERATE = "distributed_moderate"
    DISTRIBUTED_VARIED = "distributed_varied"
    HEAVY_GENERAL = "heavy_general"
    UNKNOWN = "unknown"

class CallusZone(Enum):
    THUMB_CROTCH = "thumb_crotch"
    INDEX_SIDE = "index_side"
    PALM_BELOW_INDEX = "palm_below_index"
    HEEL_OF_PALM = "heel_of_palm"
    FINGERTIP_PADS = "fingertip_pads"
    ACROSS_PALM_CREASE = "across_palm_crease"
    BASE_OF_FINGERS = "base_of_fingers"
    THUMB_PAD = "thumb_pad"
    OUTER_PALM_EDGE = "outer_palm_edge"

class ScarringType(Enum):
    NONE_VISIBLE = "none_visible"
    FEW_SMALL = "few_small"
    MULTIPLE_SMALL = "multiple_small"
    DISTRIBUTED_VARIED = "distributed_varied"
    SEVERE_FEW = "severe_few"
    UNKNOWN = "unknown"

class KnuckleCondition(Enum):
    NORMAL_RANGE = "normal_range"
    THICKENED = "thickened"
    REDUCED_EXTENSION = "reduced_extension"
    ENLARGED_AND_STIFF = "enlarged_and_stiff"
    UNKNOWN = "unknown"

class NailCondition(Enum):
    NORMAL = "normal"
    THICKENED = "thickened"
    RIDGED = "ridged"
    SEPARATED = "separated"
    DAMAGED_IRREGULAR = "damaged_irregular"
    CLEAN_PROFESSIONAL = "clean_professional"
    UNKNOWN = "unknown"

class SkinTexture(Enum):
    SMOOTH_UNWORKED = "smooth_unworked"
    THICKENED = "thickened"
    LEATHERY = "leathery"
    CALLUSED_BUT_CLEAN = "callused_but_clean"
    EMBEDDED_STAINING = "embedded_staining"
    UNKNOWN = "unknown"

class GripAsymmetry(Enum):
    SYMMETRICAL = "symmetrical"
    MODERATE_ASYMMETRY = "moderate_asymmetry"
    MARKED_ASYMMETRY = "marked_asymmetry"
    TASK_SPECIFIC_ASYMMETRY = "task_specific_asymmetry"
    UNKNOWN = "unknown"

class HandSizeAndShape(Enum):
    AVERAGE = "average"
    LARGE_THICK = "large_thick"
    LEAN_FUNCTIONAL = "lean_functional"
    SLENDER = "slender"
    UNKNOWN = "unknown"

class GloveUsage(Enum):
    NEVER = "never"
    OCCASIONAL = "occasional"
    TASK_APPROPRIATE = "task_appropriate"
    CONSTANT = "constant"
    UNKNOWN = "unknown"

# New enums for v0.2:
class BiologicalSex(Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"

class ObservationQuality(Enum):
    PHOTO_HARSH_LED = "photo_harsh_led"
    PHOTO_SOFT_AMBIENT = "photo_soft_ambient"
    PHOTO_OUTDOOR = "photo_outdoor"
    PHYSICAL_INSPECTION = "physical_inspection"
    INDUCED_OBSERVATION = "induced_observation"  # e.g., peeled calluses
    UNKNOWN = "unknown"

class FalsificationMethod(Enum):
    PASSIVE_PHOTO = "passive_photo"
    INDUCED_PEEL = "induced_peel"
    INDUCED_PINCH = "induced_pinch"
    PHYSICAL_PROBE = "physical_probe"
    UNKNOWN = "unknown"

# ===========================================================================
# Main Data Class (v0.2)
# ===========================================================================

@dataclass
class HandDescription:
    # Existing fields...
    observer_relationship: str
    observation_date: Optional[str] = None
    practitioner_context: Optional[str] = None

    cleanliness: CleanlinessLevel = CleanlinessLevel.UNKNOWN
    callus_pattern: CallusPattern = CallusPattern.UNKNOWN
    callus_zones: List[CallusZone] = field(default_factory=list)
    scarring: ScarringType = ScarringType.UNKNOWN
    knuckles: KnuckleCondition = KnuckleCondition.UNKNOWN
    nails: NailCondition = NailCondition.UNKNOWN
    skin_texture: SkinTexture = SkinTexture.UNKNOWN
    grip_asymmetry: GripAsymmetry = GripAsymmetry.UNKNOWN
    hand_shape: HandSizeAndShape = HandSizeAndShape.UNKNOWN
    glove_usage: GloveUsage = GloveUsage.UNKNOWN
    years_gloved_work: Optional[float] = None

    # New v0.2 fields:
    biological_sex: BiologicalSex = BiologicalSex.UNKNOWN
    observation_quality: ObservationQuality = ObservationQuality.UNKNOWN
    falsification_method: FalsificationMethod = FalsificationMethod.UNKNOWN

    notes: str = ""

    # ... existing property methods (observation_confidence, has_sufficient_evidence, etc.) ...

    def inferred_e_x_probability(self) -> float:
        """
        v0.2: Sex-based calibration added.
        Female calluses are less visible but still present.
        Induced observation adds a bonus.
        """
        score = 0.0

        # Existing logic...
        if self.callus_pattern == CallusPattern.DISTRIBUTED_VARIED:
            score += 0.4
        elif self.callus_pattern == CallusPattern.HEAVY_GENERAL:
            score += 0.5
        if len(self.callus_zones) >= 4:
            score += 0.3
        elif len(self.callus_zones) >= 3:
            score += 0.2
        if self.scarring == ScarringType.DISTRIBUTED_VARIED:
            score += 0.2
        if self.cleanliness == CleanlinessLevel.CLEAN_BUT_WORKED:
            score += 0.1
        if self.grip_asymmetry == GripAsymmetry.TASK_SPECIFIC_ASYMMETRY:
            score += 0.1
        if self.glove_usage == GloveUsage.CONSTANT and self.callus_pattern not in (CallusPattern.NONE_VISIBLE, CallusPattern.UNKNOWN):
            score += 0.1

        # --- v0.2 Calibration: Biological Sex ---
        if self.biological_sex == BiologicalSex.FEMALE:
            # Female calluses are less visible; any evidence is significant.
            if self.skin_texture == SkinTexture.CALLUSED_BUT_CLEAN:
                score += 0.2
            if self.callus_pattern == CallusPattern.DISTRIBUTED_MODERATE:
                score += 0.15
            if self.callus_pattern == CallusPattern.DISTRIBUTED_VARIED:
                score += 0.25

        # --- v0.2 Calibration: Observation Quality ---
        if self.observation_quality == ObservationQuality.INDUCED_OBSERVATION:
            # Observer had to manipulate the hand to reveal evidence.
            # This suggests the evidence is real but hidden.
            if self.callus_pattern != CallusPattern.NONE_VISIBLE:
                score += 0.1

        if self.falsification_method == FalsificationMethod.INDUCED_PEEL:
            # Peeling calluses is a high-reliability observation.
            if len(self.callus_zones) >= 2:
                score += 0.1

        return min(1.0, score)

    # ... rest of existing methods (inferred_specialist_probability, inferred_domains, to_ai_context, etc.) ...

    def to_ai_context(self) -> str:
        # Include the new v0.2 fields in the output.
        # (Same as before, but with biological_sex and observation_quality added.)
        pass
