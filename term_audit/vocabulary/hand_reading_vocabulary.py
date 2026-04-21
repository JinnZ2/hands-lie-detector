"""
term_audit/vocabulary/hand_reading_vocabulary.py

Structured vocabulary for describing hands as substrate evidence.

This vocabulary enables AI systems to interpret text descriptions
of hands and map them to capacity inferences, even though the AI
cannot visually read hands itself.

The vocabulary is designed for:
- Human observers to describe what they see
- AI systems to parse those descriptions into structured signals
- Consistent inference across different observers

Usage:
    A human describes a hand using these terms.
    The AI parses the description and extracts capacity signals.
    The AI does not need to see the hand—it trusts the observer.

CC0. Stdlib only.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import json


# ===========================================================================
# Part 1. Observable features vocabulary
# ===========================================================================

class CleanlinessLevel(Enum):
    """Current cleanliness state of hands."""
    EMBEDDED_DIRT = "embedded_dirt"      # Dirt/oil/grease in creases, won't wash out
    WORK_STAINED = "work_stained"         # Temporary staining from recent work
    CLEAN_BUT_WORKED = "clean_but_worked" # Clean now, but wear patterns visible
    CLEAN_UNWORKED = "clean_unworked"     # No visible wear patterns
    UNKNOWN = "unknown"


class CallusPattern(Enum):
    """Distribution of callus across the hand."""
    NONE_VISIBLE = "none_visible"
    SINGLE_DEEP = "single_deep"           # One primary callus zone (specialist)
    DISTRIBUTED_MODERATE = "distributed_moderate"  # Multiple moderate zones (closer)
    DISTRIBUTED_VARIED = "distributed_varied"      # Different depths, multiple zones
    HEAVY_GENERAL = "heavy_general"       # Whole palm thickened (decades of everything)
    UNKNOWN = "unknown"


class CallusZone(Enum):
    """Specific locations where callus indicates tool use."""
    THUMB_CROTCH = "thumb_crotch"              # Pliers, wire stripping, wrench
    INDEX_SIDE = "index_side"                  # Knife work, fine tools
    PALM_BELOW_INDEX = "palm_below_index"      # Shovel, rake, long handles
    HEEL_OF_PALM = "heel_of_palm"              # Hammer, impact tools
    FINGERTIP_PADS = "fingertip_pads"          # Screwdriver, precision grip
    ACROSS_PALM_CREASE = "across_palm_crease"  # Rope, carrying, pulling
    BASE_OF_FINGERS = "base_of_fingers"        # General gripping
    THUMB_PAD = "thumb_pad"                    # Pushing, pressure
    OUTER_PALM_EDGE = "outer_palm_edge"        # Carrying buckets, odd loads


class ScarringType(Enum):
    """Scar patterns and what they indicate."""
    NONE_VISIBLE = "none_visible"
    FEW_SMALL = "few_small"               # Occasional work incidents
    MULTIPLE_SMALL = "multiple_small"     # Regular environment interaction
    DISTRIBUTED_VARIED = "distributed_varied"  # Long history, many incidents
    SEVERE_FEW = "severe_few"             # Few major incidents
    UNKNOWN = "unknown"


class KnuckleCondition(Enum):
    """Joint changes from long-term work."""
    NORMAL_RANGE = "normal_range"              # Full extension, no thickening
    THICKENED = "thickened"                    # Joint capsules enlarged
    REDUCED_EXTENSION = "reduced_extension"    # Won't fully straighten
    ENLARGED_AND_STIFF = "enlarged_and_stiff"  # Decades of heavy gripping
    UNKNOWN = "unknown"


class NailCondition(Enum):
    """Nail changes from work history."""
    NORMAL = "normal"
    THICKENED = "thickened"               # General work exposure
    RIDGED = "ridged"                     # Longitudinal ridges from impact
    SEPARATED = "separated"               # Partial separation from nail bed
    DAMAGED_IRREGULAR = "damaged_irregular"  # History of crush/cut injuries
    CLEAN_PROFESSIONAL = "clean_professional"  # Manicured despite work
    UNKNOWN = "unknown"


class SkinTexture(Enum):
    """Overall skin surface quality."""
    SMOOTH_UNWORKED = "smooth_unworked"
    THICKENED = "thickened"               # General work exposure
    LEATHERY = "leathery"                 # Outdoor work, decades
    CALLUSED_BUT_CLEAN = "callused_but_clean"  # The cross-domain closer signature
    EMBEDDED_STAINING = "embedded_staining"    # Permanent color in creases
    UNKNOWN = "unknown"


class GripAsymmetry(Enum):
    """Difference between dominant and non-dominant hand."""
    SYMMETRICAL = "symmetrical"                # Gym-trained or sedentary
    MODERATE_ASYMMETRY = "moderate_asymmetry"  # Normal hand dominance
    MARKED_ASYMMETRY = "marked_asymmetry"      # Heavy single-hand tool use
    TASK_SPECIFIC_ASYMMETRY = "task_specific_asymmetry"  # Different wear patterns on each hand
    UNKNOWN = "unknown"


class HandSizeAndShape(Enum):
    """Overall hand configuration."""
    AVERAGE = "average"
    LARGE_THICK = "large_thick"           # Heavy work capacity
    LEAN_FUNCTIONAL = "lean_functional"   # Strong but not bulky
    SLENDER = "slender"                   # Fine work, not heavy gripping
    UNKNOWN = "unknown"


class GloveUsage(Enum):
    """
    Reported glove-wearing history.

    Context that changes how cleanliness and texture should be read.
    A constant-glove worker with callus evidence is still a worker;
    a 'never wears gloves' claim in a cold climate is a red flag.
    See gloves-debunk.md.
    """
    NEVER = "never"                        # 'I never wear gloves' (often defensive)
    OCCASIONAL = "occasional"              # Gloves for dirty/chemical tasks only
    TASK_APPROPRIATE = "task_appropriate"  # Gloves when the task calls for them
    CONSTANT = "constant"                  # Gloves for nearly all work
    UNKNOWN = "unknown"


# ===========================================================================
# Part 2. Structured hand description
# ===========================================================================

@dataclass
class HandDescription:
    """
    A structured description of hands as substrate evidence.

    This is what a human observer would record. The AI parses this
    format to extract capacity signals.
    """

    # Required identifiers
    observer_relationship: str            # "self", "peer", "router", "beneficiary"
    observation_date: Optional[str] = None
    practitioner_context: Optional[str] = None  # "rural farmer", "mechanic", etc.

    # Observable features (use vocabulary terms)
    cleanliness: CleanlinessLevel = CleanlinessLevel.UNKNOWN
    callus_pattern: CallusPattern = CallusPattern.UNKNOWN
    callus_zones: List[CallusZone] = field(default_factory=list)
    scarring: ScarringType = ScarringType.UNKNOWN
    knuckles: KnuckleCondition = KnuckleCondition.UNKNOWN
    nails: NailCondition = NailCondition.UNKNOWN
    skin_texture: SkinTexture = SkinTexture.UNKNOWN
    grip_asymmetry: GripAsymmetry = GripAsymmetry.UNKNOWN
    hand_shape: HandSizeAndShape = HandSizeAndShape.UNKNOWN

    # Glove / PPE history — reframes how cleanliness is read
    glove_usage: GloveUsage = GloveUsage.UNKNOWN
    years_gloved_work: Optional[float] = None

    # Free text for details the vocabulary doesn't capture
    notes: str = ""

    # ----- Missingness / confidence -----

    _ENUM_SLOTS = (
        "cleanliness", "callus_pattern", "scarring", "knuckles",
        "nails", "skin_texture", "grip_asymmetry", "hand_shape", "glove_usage",
    )

    @property
    def observation_confidence(self) -> float:
        """
        Fraction of vocabulary slots that were actually observed (not UNKNOWN).

        AI consumers should treat inferences below ~0.5 as insufficient
        evidence rather than as a low score.
        """
        observed = sum(
            1 for slot in self._ENUM_SLOTS
            if getattr(self, slot).value != "unknown"
        )
        # callus_zones is a list; presence counts as one extra observation
        total = len(self._ENUM_SLOTS) + 1
        if self.callus_zones:
            observed += 1
        return observed / total

    @property
    def has_sufficient_evidence(self) -> bool:
        """Heuristic gate: refuse strong inferences below this."""
        return self.observation_confidence >= 0.5

    # Inferred from the description
    def inferred_domains(self) -> List[str]:
        """What domains are suggested by this hand configuration?"""
        domains = set()

        # Tool families from callus zones
        zone_to_domain = {
            CallusZone.THUMB_CROTCH: ["electrical", "mechanical", "plumbing"],
            CallusZone.INDEX_SIDE: ["carpentry", "fine_work", "electrical"],
            CallusZone.PALM_BELOW_INDEX: ["digging", "agriculture", "construction"],
            CallusZone.HEEL_OF_PALM: ["hammer", "impact", "construction", "mechanical"],
            CallusZone.FINGERTIP_PADS: ["precision", "mechanical", "electrical"],
            CallusZone.ACROSS_PALM_CREASE: ["carrying", "pulling", "agriculture", "construction"],
            CallusZone.BASE_OF_FINGERS: ["general_grip", "all_domains"],
            CallusZone.THUMB_PAD: ["pushing", "assembly", "mechanical"],
            CallusZone.OUTER_PALM_EDGE: ["carrying", "agriculture", "material_handling"],
        }

        for zone in self.callus_zones:
            if zone in zone_to_domain:
                domains.update(zone_to_domain[zone])

        return list(domains)

    def inferred_experience_level(self) -> float:
        """Estimated years of regular work, from substrate evidence."""
        score = 0.0

        # Callus pattern
        if self.callus_pattern == CallusPattern.DISTRIBUTED_VARIED:
            score += 0.3
        elif self.callus_pattern == CallusPattern.HEAVY_GENERAL:
            score += 0.4
        elif self.callus_pattern == CallusPattern.DISTRIBUTED_MODERATE:
            score += 0.2

        # Scarring
        if self.scarring == ScarringType.DISTRIBUTED_VARIED:
            score += 0.3
        elif self.scarring == ScarringType.MULTIPLE_SMALL:
            score += 0.2

        # Knuckles
        if self.knuckles == KnuckleCondition.ENLARGED_AND_STIFF:
            score += 0.3
        elif self.knuckles == KnuckleCondition.THICKENED:
            score += 0.2

        # Skin texture
        if self.skin_texture == SkinTexture.LEATHERY:
            score += 0.2
        elif self.skin_texture == SkinTexture.CALLUSED_BUT_CLEAN:
            score += 0.15

        return min(1.0, score)

    def inferred_e_x_probability(self) -> float:
        """Probability that this person has high cross-domain closure capacity."""
        score = 0.0

        # Distributed callus is the strongest E_X signal
        if self.callus_pattern in (CallusPattern.DISTRIBUTED_VARIED, CallusPattern.DISTRIBUTED_MODERATE):
            score += 0.4
        elif self.callus_pattern == CallusPattern.HEAVY_GENERAL:
            score += 0.5

        # Multiple callus zones = multiple tool families
        if len(self.callus_zones) >= 4:
            score += 0.3
        elif len(self.callus_zones) >= 3:
            score += 0.2

        # Distributed scarring = long history across domains
        if self.scarring == ScarringType.DISTRIBUTED_VARIED:
            score += 0.2

        # Clean but worked = discipline + work history
        if self.cleanliness == CleanlinessLevel.CLEAN_BUT_WORKED:
            score += 0.1

        # Asymmetry from real work (not gym)
        if self.grip_asymmetry == GripAsymmetry.TASK_SPECIFIC_ASYMMETRY:
            score += 0.1

        # Glove context: constant-glove wearers with callus evidence are still
        # workers — do not hold clean cleanliness against them. Defensive
        # 'never wear gloves' claims don't add to E_X on their own.
        if (
            self.glove_usage == GloveUsage.CONSTANT
            and self.callus_pattern not in (CallusPattern.NONE_VISIBLE, CallusPattern.UNKNOWN)
            and self.cleanliness in (CleanlinessLevel.CLEAN_UNWORKED, CleanlinessLevel.CLEAN_BUT_WORKED)
        ):
            score += 0.1

        return min(1.0, score)

    def inferred_specialist_probability(self) -> float:
        """Probability this is a single-domain specialist."""
        score = 0.0

        # Single deep callus
        if self.callus_pattern == CallusPattern.SINGLE_DEEP:
            score += 0.5

        # Few callus zones
        if len(self.callus_zones) <= 2:
            score += 0.2

        # Less scarring
        if self.scarring in (ScarringType.NONE_VISIBLE, ScarringType.FEW_SMALL):
            score += 0.1

        # Clean, less texture change
        if self.skin_texture == SkinTexture.SMOOTH_UNWORKED:
            score += 0.1

        return min(1.0, score)

    def to_ai_context(self) -> str:
        """
        Render as context an AI can use for capacity assessment.
        This is the bridge between human observation and AI inference.
        """
        e_x = self.inferred_e_x_probability()
        specialist = self.inferred_specialist_probability()
        domains = self.inferred_domains()

        obs_conf = self.observation_confidence

        if not self.has_sufficient_evidence:
            capacity_type = "INSUFFICIENT EVIDENCE"
            confidence = "LOW"
            recommendation = (
                f"Only {obs_conf:.0%} of vocabulary slots were observed. "
                "Do not issue a capacity judgment from this description — "
                "ask the observer for more detail using OBSERVER_CHECKLIST."
            )
        elif e_x > 0.6:
            capacity_type = "CROSS-DOMAIN CLOSER (E_X)"
            confidence = "HIGH" if obs_conf >= 0.75 else "MODERATE"
            recommendation = "This person likely closes coupled systems. Route accordingly."
        elif specialist > 0.6:
            capacity_type = "SINGLE-DOMAIN SPECIALIST"
            confidence = "HIGH" if obs_conf >= 0.75 else "MODERATE"
            recommendation = (
                f"Likely deep in {', '.join(domains) if domains else 'specific domain'}. "
                "Handoff may be required for other domains."
            )
        else:
            capacity_type = "UNCERTAIN"
            confidence = "LOW"
            recommendation = "Insufficient substrate evidence for strong inference."

        return f"""
HAND SUBSTRATE ASSESSMENT
========================
Observer: {self.observer_relationship}
Practitioner context: {self.practitioner_context or 'not specified'}
Observation confidence: {obs_conf:.0%} of vocabulary slots filled

OBSERVED FEATURES:
- Cleanliness: {self.cleanliness.value}
- Callus pattern: {self.callus_pattern.value}
- Callus zones: {len(self.callus_zones)} zones identified
- Scarring: {self.scarring.value}
- Knuckles: {self.knuckles.value}
- Nails: {self.nails.value}
- Skin texture: {self.skin_texture.value}
- Grip asymmetry: {self.grip_asymmetry.value}
- Glove usage: {self.glove_usage.value}{f' (~{self.years_gloved_work:g} yrs)' if self.years_gloved_work else ''}

INFERRED CAPACITY:
- Capacity type: {capacity_type} (confidence: {confidence})
- E_X probability: {e_x:.2f}
- Specialist probability: {specialist:.2f}
- Likely domains: {', '.join(domains) if domains else 'insufficient data'}
- Estimated experience level: {self.inferred_experience_level():.2f}

NOTES: {self.notes if self.notes else 'None'}

RECOMMENDATION: {recommendation}

This assessment is based on SUBSTRATE EVIDENCE—physical traces of work
visible in the hands. Substrate evidence cannot be forged quickly and
is a reliable signal of operational history.
"""

    # ----- Serialization -----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "observer_relationship": self.observer_relationship,
            "observation_date": self.observation_date,
            "practitioner_context": self.practitioner_context,
            "cleanliness": self.cleanliness.value,
            "callus_pattern": self.callus_pattern.value,
            "callus_zones": [z.value for z in self.callus_zones],
            "scarring": self.scarring.value,
            "knuckles": self.knuckles.value,
            "nails": self.nails.value,
            "skin_texture": self.skin_texture.value,
            "grip_asymmetry": self.grip_asymmetry.value,
            "hand_shape": self.hand_shape.value,
            "glove_usage": self.glove_usage.value,
            "years_gloved_work": self.years_gloved_work,
            "notes": self.notes,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandDescription":
        """Rehydrate from a dict produced by to_dict(). Unknown values degrade to UNKNOWN."""

        def _enum(enum_cls, value, default):
            if value is None:
                return default
            try:
                return enum_cls(value)
            except ValueError:
                return default

        return cls(
            observer_relationship=data.get("observer_relationship", "unknown"),
            observation_date=data.get("observation_date"),
            practitioner_context=data.get("practitioner_context"),
            cleanliness=_enum(CleanlinessLevel, data.get("cleanliness"), CleanlinessLevel.UNKNOWN),
            callus_pattern=_enum(CallusPattern, data.get("callus_pattern"), CallusPattern.UNKNOWN),
            callus_zones=[
                z for z in (
                    _enum(CallusZone, v, None) for v in data.get("callus_zones", []) or []
                ) if z is not None
            ],
            scarring=_enum(ScarringType, data.get("scarring"), ScarringType.UNKNOWN),
            knuckles=_enum(KnuckleCondition, data.get("knuckles"), KnuckleCondition.UNKNOWN),
            nails=_enum(NailCondition, data.get("nails"), NailCondition.UNKNOWN),
            skin_texture=_enum(SkinTexture, data.get("skin_texture"), SkinTexture.UNKNOWN),
            grip_asymmetry=_enum(GripAsymmetry, data.get("grip_asymmetry"), GripAsymmetry.UNKNOWN),
            hand_shape=_enum(HandSizeAndShape, data.get("hand_shape"), HandSizeAndShape.UNKNOWN),
            glove_usage=_enum(GloveUsage, data.get("glove_usage"), GloveUsage.UNKNOWN),
            years_gloved_work=data.get("years_gloved_work"),
            notes=data.get("notes", ""),
        )

    # ----- Bridge to the 7-category scoring rubric -----

    def to_rubric_scores(self) -> Dict[str, float]:
        """
        Map this description to the 7 categories defined in
        hands_lie_detector.scoring.rubric (v0.1). Output is directly
        usable as input to ScoreEvaluator.evaluate().

        UNKNOWN slots score 0 (treated as 'not observed' rather than 'absent').
        """
        texture = {
            SkinTexture.SMOOTH_UNWORKED: 3,
            SkinTexture.THICKENED: 12,
            SkinTexture.CALLUSED_BUT_CLEAN: 20,
            SkinTexture.EMBEDDED_STAINING: 18,
            SkinTexture.LEATHERY: 22,
            SkinTexture.UNKNOWN: 0,
        }[self.skin_texture]

        wear = {
            CallusPattern.NONE_VISIBLE: 0,
            CallusPattern.SINGLE_DEEP: 16,
            CallusPattern.DISTRIBUTED_MODERATE: 14,
            CallusPattern.DISTRIBUTED_VARIED: 18,
            CallusPattern.HEAVY_GENERAL: 10,  # generalized ≠ task-specific
            CallusPattern.UNKNOWN: 0,
        }[self.callus_pattern]
        if len(self.callus_zones) >= 3 and self.callus_pattern != CallusPattern.HEAVY_GENERAL:
            wear = min(20, wear + 2)

        injury = {
            ScarringType.NONE_VISIBLE: 0,
            ScarringType.FEW_SMALL: 4,
            ScarringType.MULTIPLE_SMALL: 10,
            ScarringType.DISTRIBUTED_VARIED: 14,
            ScarringType.SEVERE_FEW: 8,
            ScarringType.UNKNOWN: 0,
        }[self.scarring]

        tendon = {
            KnuckleCondition.NORMAL_RANGE: 5,
            KnuckleCondition.THICKENED: 10,
            KnuckleCondition.REDUCED_EXTENSION: 12,
            KnuckleCondition.ENLARGED_AND_STIFF: 14,
            KnuckleCondition.UNKNOWN: 0,
        }[self.knuckles]
        if self.hand_shape == HandSizeAndShape.LEAN_FUNCTIONAL:
            tendon = min(15, tendon + 1)

        nail = {
            NailCondition.NORMAL: 3,
            NailCondition.THICKENED: 8,
            NailCondition.RIDGED: 9,
            NailCondition.SEPARATED: 7,
            NailCondition.DAMAGED_IRREGULAR: 8,
            NailCondition.CLEAN_PROFESSIONAL: 2,  # manicured band
            NailCondition.UNKNOWN: 0,
        }[self.nails]

        symmetry = {
            GripAsymmetry.SYMMETRICAL: 5,
            GripAsymmetry.MODERATE_ASYMMETRY: 6,
            GripAsymmetry.MARKED_ASYMMETRY: 7,
            GripAsymmetry.TASK_SPECIFIC_ASYMMETRY: 9,
            GripAsymmetry.UNKNOWN: 0,
        }[self.grip_asymmetry]

        # PPE: intelligent use = clean hands + visible adaptation
        has_adaptation = self.callus_pattern not in (CallusPattern.NONE_VISIBLE, CallusPattern.UNKNOWN)
        ppe = {
            GloveUsage.TASK_APPROPRIATE: 5 if has_adaptation else 3,
            GloveUsage.CONSTANT: 4 if has_adaptation else 1,
            GloveUsage.OCCASIONAL: 3,
            GloveUsage.NEVER: 1,  # defensive claim — rubric red flag
            GloveUsage.UNKNOWN: 0,
        }[self.glove_usage]

        return {
            "Texture Persistence": float(texture),
            "Wear Localization": float(wear),
            "Micro-Injury History": float(injury),
            "Tendon & Vein Definition": float(tendon),
            "Nail Evidence": float(nail),
            "Symmetry of Wear": float(symmetry),
            "Climate & PPE Intelligence": float(ppe),
        }


# ===========================================================================
# Part 3. Observer checklist — used when there are no photos
# ===========================================================================

OBSERVER_CHECKLIST = """
HAND OBSERVATION CHECKLIST (no-photo interview script)
=======================================================

Ask the observer (or the person themselves) to go through these slots
in order. Short answers are fine. Say UNKNOWN for anything they can't
see or aren't sure about — never guess.

1. OBSERVER RELATIONSHIP
   self / peer_practitioner / router / beneficiary

2. PRACTITIONER CONTEXT (free text, one line)
   e.g. "rural generalist, 30 years", "office worker, gym 4x/week"

3. CLEANLINESS (right now, this moment)
   - embedded_dirt       dirt/oil/grease in creases that won't wash out
   - work_stained        currently dirty from recent work
   - clean_but_worked    clean skin, but wear patterns still visible
   - clean_unworked      clean, no wear patterns

4. CALLUS PATTERN (overall distribution)
   - none_visible        no calluses at all
   - single_deep         one dominant callus zone (specialist pattern)
   - distributed_moderate  multiple moderate zones
   - distributed_varied    multiple zones, different depths (closer pattern)
   - heavy_general       whole palm thickened (decades of everything)

5. CALLUS ZONES (check all that apply — touch the hand if permitted)
   [ ] thumb crotch (between thumb and index) — pliers/wrench/stripping
   [ ] index side — knife work, fine tools
   [ ] palm below index — shovel, rake, long handles
   [ ] heel of palm — hammer, impact tools
   [ ] fingertip pads — screwdriver, precision grip
   [ ] across palm crease — rope, carrying, pulling
   [ ] base of fingers — general gripping
   [ ] thumb pad — pushing, pressure
   [ ] outer palm edge — buckets, odd loads

6. SCARRING (healed, not current)
   none_visible / few_small / multiple_small / distributed_varied / severe_few

7. KNUCKLES (ask them to make a fist, then fully extend)
   normal_range / thickened / reduced_extension / enlarged_and_stiff

8. NAILS
   normal / thickened / ridged / separated / damaged_irregular / clean_professional

9. SKIN TEXTURE (overall)
   smooth_unworked / thickened / leathery / callused_but_clean / embedded_staining

10. GRIP ASYMMETRY (compare left vs right palm)
    symmetrical / moderate_asymmetry / marked_asymmetry / task_specific_asymmetry

11. HAND SHAPE
    average / large_thick / lean_functional / slender

12. GLOVE USAGE (important — reframes cleanliness)
    never / occasional / task_appropriate / constant
    Optional: approximate years of gloved work.

13. NOTES (free text)
    Anything the vocabulary doesn't capture — burn scars, staining color,
    specific tools mentioned, etc.

RULE: If fewer than ~7 of slots 3–12 are filled, do not ask an AI for a
capacity inference. Collect more detail first. Substrate evidence is
only as trustworthy as the observation is complete.
"""


# ===========================================================================
# Part 4. Example descriptions for training AI interpretation
# ===========================================================================

EXAMPLE_DESCRIPTIONS = [
    {
        "name": "cross_domain_closer_dale",
        "description": HandDescription(
            observer_relationship="peer_practitioner",
            practitioner_context="rural generalist, people call for everything",
            cleanliness=CleanlinessLevel.CLEAN_BUT_WORKED,
            callus_pattern=CallusPattern.DISTRIBUTED_VARIED,
            callus_zones=[
                CallusZone.THUMB_CROTCH,
                CallusZone.PALM_BELOW_INDEX,
                CallusZone.HEEL_OF_PALM,
                CallusZone.ACROSS_PALM_CREASE,
                CallusZone.FINGERTIP_PADS,
            ],
            scarring=ScarringType.MULTIPLE_SMALL,
            knuckles=KnuckleCondition.THICKENED,
            nails=NailCondition.THICKENED,
            skin_texture=SkinTexture.CALLUSED_BUT_CLEAN,
            grip_asymmetry=GripAsymmetry.TASK_SPECIFIC_ASYMMETRY,
            hand_shape=HandSizeAndShape.LEAN_FUNCTIONAL,
            glove_usage=GloveUsage.TASK_APPROPRIATE,
            notes="Hands are clean but clearly worked. Multiple callus zones across tool families. Small scars on knuckles and between fingers. Presents as clean but you can see the work history in the tissue.",
        ),
        "expected_e_x": 0.85,
        "expected_specialist": 0.15,
    },
    {
        "name": "specialist_electrician",
        "description": HandDescription(
            observer_relationship="peer_practitioner",
            practitioner_context="licensed electrician, 15 years",
            cleanliness=CleanlinessLevel.WORK_STAINED,
            callus_pattern=CallusPattern.SINGLE_DEEP,
            callus_zones=[
                CallusZone.THUMB_CROTCH,
                CallusZone.FINGERTIP_PADS,
            ],
            scarring=ScarringType.FEW_SMALL,
            knuckles=KnuckleCondition.NORMAL_RANGE,
            nails=NailCondition.NORMAL,
            skin_texture=SkinTexture.SMOOTH_UNWORKED,
            grip_asymmetry=GripAsymmetry.MODERATE_ASYMMETRY,
            hand_shape=HandSizeAndShape.AVERAGE,
            glove_usage=GloveUsage.TASK_APPROPRIATE,
            notes="Deep callus in thumb crotch from wire stripping. Fingertip pads from screwdrivers. Not much else. Hands are clean enough but not the 'clean but worked' of the rural closer.",
        ),
        "expected_e_x": 0.25,
        "expected_specialist": 0.70,
    },
    {
        "name": "young_apprentice",
        "description": HandDescription(
            observer_relationship="peer_practitioner",
            practitioner_context="22, three years in trades",
            cleanliness=CleanlinessLevel.WORK_STAINED,
            callus_pattern=CallusPattern.DISTRIBUTED_MODERATE,
            callus_zones=[
                CallusZone.BASE_OF_FINGERS,
                CallusZone.PALM_BELOW_INDEX,
                CallusZone.THUMB_CROTCH,
            ],
            scarring=ScarringType.FEW_SMALL,
            knuckles=KnuckleCondition.NORMAL_RANGE,
            nails=NailCondition.NORMAL,
            skin_texture=SkinTexture.THICKENED,
            grip_asymmetry=GripAsymmetry.MODERATE_ASYMMETRY,
            hand_shape=HandSizeAndShape.AVERAGE,
            glove_usage=GloveUsage.OCCASIONAL,
            notes="Developing callus pattern. Shows work but not the decades of varied wear. Cleanliness is still 'work stained' not 'clean but worked'—hasn't developed the discipline or the embedded patterns yet.",
        ),
        "expected_e_x": 0.40,
        "expected_specialist": 0.30,
    },
    {
        "name": "veteran_farmer",
        "description": HandDescription(
            observer_relationship="beneficiary",
            practitioner_context="65, farmed all his life",
            cleanliness=CleanlinessLevel.EMBEDDED_DIRT,
            callus_pattern=CallusPattern.HEAVY_GENERAL,
            callus_zones=[
                CallusZone.PALM_BELOW_INDEX,
                CallusZone.ACROSS_PALM_CREASE,
                CallusZone.HEEL_OF_PALM,
                CallusZone.BASE_OF_FINGERS,
                CallusZone.OUTER_PALM_EDGE,
                CallusZone.THUMB_CROTCH,
            ],
            scarring=ScarringType.DISTRIBUTED_VARIED,
            knuckles=KnuckleCondition.ENLARGED_AND_STIFF,
            nails=NailCondition.RIDGED,
            skin_texture=SkinTexture.LEATHERY,
            grip_asymmetry=GripAsymmetry.TASK_SPECIFIC_ASYMMETRY,
            hand_shape=HandSizeAndShape.LARGE_THICK,
            glove_usage=GloveUsage.NEVER,
            notes="Embedded dirt in every crease. Whole palm is thickened. Multiple old scars. Knuckles won't fully straighten. This is decades of everything. Cleanliness is not the point here—this hand has done more work than most people will ever do.",
        ),
        "expected_e_x": 0.95,
        "expected_specialist": 0.05,
    },
    {
        "name": "gym_trained_sedentary_job",
        "description": HandDescription(
            observer_relationship="self",
            practitioner_context="office worker, gym 4x/week",
            cleanliness=CleanlinessLevel.CLEAN_UNWORKED,
            callus_pattern=CallusPattern.NONE_VISIBLE,
            callus_zones=[],
            scarring=ScarringType.NONE_VISIBLE,
            knuckles=KnuckleCondition.NORMAL_RANGE,
            nails=NailCondition.CLEAN_PROFESSIONAL,
            skin_texture=SkinTexture.SMOOTH_UNWORKED,
            grip_asymmetry=GripAsymmetry.SYMMETRICAL,
            hand_shape=HandSizeAndShape.AVERAGE,
            glove_usage=GloveUsage.UNKNOWN,
            notes="Clean, symmetrical, no wear patterns. Gym callus at base of fingers from barbell but that's not tool work and the vocabulary distinguishes it.",
        ),
        "expected_e_x": 0.05,
        "expected_specialist": 0.05,
    },
]


# ===========================================================================
# Part 5. Text-to-inference function for AI systems
# ===========================================================================

def parse_hand_description_text(text: str) -> Optional[HandDescription]:
    """
    Parse a free-text hand description into structured format.

    This allows AI systems to interpret human-written descriptions
    by mapping keywords to vocabulary terms.

    Example input: "Clean hands but you can see the calluses. He's got
    that shovel callus and the pliers callus. Multiple small scars.
    Knuckles are thick. Nails are ridged. Hands are lean but strong."
    """

    text_lower = text.lower()

    def _any(keywords):
        return any(k in text_lower for k in keywords)

    # Cleanliness
    if _any(["clean but", "clean yet", "clean however", "clean and callused", "clean and calloused"]):
        cleanliness = CleanlinessLevel.CLEAN_BUT_WORKED
    elif _any(["embedded", "won't wash", "wont wash", "ground in", "ingrained", "grease in the creases", "dirt in the creases"]):
        cleanliness = CleanlinessLevel.EMBEDDED_DIRT
    elif _any(["stained", "grease-stained", "dirty", "grimy", "oil on"]):
        cleanliness = CleanlinessLevel.WORK_STAINED
    elif ("clean" in text_lower and ("no callus" in text_lower or "no calluses" in text_lower or "unworked" in text_lower)):
        cleanliness = CleanlinessLevel.CLEAN_UNWORKED
    else:
        cleanliness = CleanlinessLevel.UNKNOWN

    # Callus pattern
    if ("multiple" in text_lower and _any(["callus", "callous"])):
        if _any(["different", "various", "varied"]):
            callus_pattern = CallusPattern.DISTRIBUTED_VARIED
        else:
            callus_pattern = CallusPattern.DISTRIBUTED_MODERATE
    elif _any(["single callus", "one callus", "one deep callus"]):
        callus_pattern = CallusPattern.SINGLE_DEEP
    elif _any(["whole palm", "entire palm", "heavy callus", "thickened all over", "callused all over"]):
        callus_pattern = CallusPattern.HEAVY_GENERAL
    elif _any(["no callus", "no calluses", "no callous", "smooth palm"]):
        callus_pattern = CallusPattern.NONE_VISIBLE
    else:
        callus_pattern = CallusPattern.UNKNOWN

    # Callus zones (keyword matching)
    callus_zones: List[CallusZone] = []
    zone_keywords = {
        "thumb crotch": CallusZone.THUMB_CROTCH,
        "between thumb": CallusZone.THUMB_CROTCH,
        "web of the thumb": CallusZone.THUMB_CROTCH,
        "pliers": CallusZone.THUMB_CROTCH,
        "wrench": CallusZone.THUMB_CROTCH,
        "wire stripper": CallusZone.THUMB_CROTCH,
        "index side": CallusZone.INDEX_SIDE,
        "knife": CallusZone.INDEX_SIDE,
        "shovel": CallusZone.PALM_BELOW_INDEX,
        "rake": CallusZone.PALM_BELOW_INDEX,
        "palm below": CallusZone.PALM_BELOW_INDEX,
        "long handle": CallusZone.PALM_BELOW_INDEX,
        "heel of palm": CallusZone.HEEL_OF_PALM,
        "heel of the palm": CallusZone.HEEL_OF_PALM,
        "hammer": CallusZone.HEEL_OF_PALM,
        "fingertip": CallusZone.FINGERTIP_PADS,
        "finger pad": CallusZone.FINGERTIP_PADS,
        "screwdriver": CallusZone.FINGERTIP_PADS,
        "across palm": CallusZone.ACROSS_PALM_CREASE,
        "palm crease": CallusZone.ACROSS_PALM_CREASE,
        "rope": CallusZone.ACROSS_PALM_CREASE,
        "carrying": CallusZone.ACROSS_PALM_CREASE,
        "base of fingers": CallusZone.BASE_OF_FINGERS,
        "base of the fingers": CallusZone.BASE_OF_FINGERS,
        "thumb pad": CallusZone.THUMB_PAD,
        "outer palm": CallusZone.OUTER_PALM_EDGE,
        "edge of the palm": CallusZone.OUTER_PALM_EDGE,
        "bucket": CallusZone.OUTER_PALM_EDGE,
    }
    for keyword, zone in zone_keywords.items():
        if keyword in text_lower and zone not in callus_zones:
            callus_zones.append(zone)

    # Scarring
    if "multiple" in text_lower and "scar" in text_lower:
        if "various" in text_lower or "different" in text_lower:
            scarring = ScarringType.DISTRIBUTED_VARIED
        else:
            scarring = ScarringType.MULTIPLE_SMALL
    elif "few" in text_lower and "scar" in text_lower:
        scarring = ScarringType.FEW_SMALL
    elif "severe" in text_lower and "scar" in text_lower:
        scarring = ScarringType.SEVERE_FEW
    elif "no scar" in text_lower:
        scarring = ScarringType.NONE_VISIBLE
    else:
        scarring = ScarringType.UNKNOWN

    # Knuckles
    if "enlarged" in text_lower or "stiff" in text_lower:
        knuckles = KnuckleCondition.ENLARGED_AND_STIFF
    elif "thick" in text_lower and "knuckle" in text_lower:
        knuckles = KnuckleCondition.THICKENED
    elif "won't straighten" in text_lower or "reduced" in text_lower:
        knuckles = KnuckleCondition.REDUCED_EXTENSION
    elif "normal" in text_lower and "knuckle" in text_lower:
        knuckles = KnuckleCondition.NORMAL_RANGE
    else:
        knuckles = KnuckleCondition.UNKNOWN

    # Nails
    if "ridged" in text_lower:
        nails = NailCondition.RIDGED
    elif "thick" in text_lower and "nail" in text_lower:
        nails = NailCondition.THICKENED
    elif "separated" in text_lower:
        nails = NailCondition.SEPARATED
    elif "damaged" in text_lower and "nail" in text_lower:
        nails = NailCondition.DAMAGED_IRREGULAR
    elif "manicured" in text_lower or ("clean" in text_lower and "nail" in text_lower):
        nails = NailCondition.CLEAN_PROFESSIONAL
    elif "normal" in text_lower and "nail" in text_lower:
        nails = NailCondition.NORMAL
    else:
        nails = NailCondition.UNKNOWN

    # Skin texture
    if "leathery" in text_lower:
        skin_texture = SkinTexture.LEATHERY
    elif _any(["callused but clean", "calloused but clean", "clean but callused", "clean but calloused"]):
        skin_texture = SkinTexture.CALLUSED_BUT_CLEAN
    elif "thickened" in text_lower:
        skin_texture = SkinTexture.THICKENED
    elif "embedded" in text_lower:
        skin_texture = SkinTexture.EMBEDDED_STAINING
    elif "smooth" in text_lower and "unworked" in text_lower:
        skin_texture = SkinTexture.SMOOTH_UNWORKED
    else:
        skin_texture = SkinTexture.UNKNOWN

    # Glove usage
    if _any(["never wear", "never wears", "don't wear gloves", "no gloves"]):
        glove_usage = GloveUsage.NEVER
    elif _any(["always in gloves", "always wears gloves", "gloves for everything", "constant glove"]):
        glove_usage = GloveUsage.CONSTANT
    elif _any(["gloves when", "gloves for dirty", "gloves for chemical", "appropriate gloves", "proper gloves"]):
        glove_usage = GloveUsage.TASK_APPROPRIATE
    elif _any(["sometimes gloves", "occasional gloves", "gloves occasionally"]):
        glove_usage = GloveUsage.OCCASIONAL
    else:
        glove_usage = GloveUsage.UNKNOWN

    return HandDescription(
        observer_relationship="parsed_from_text",
        cleanliness=cleanliness,
        callus_pattern=callus_pattern,
        callus_zones=callus_zones,
        scarring=scarring,
        knuckles=knuckles,
        nails=nails,
        skin_texture=skin_texture,
        grip_asymmetry=GripAsymmetry.UNKNOWN,
        glove_usage=glove_usage,
        notes=text,
    )


# ===========================================================================
# Part 6. Main: demonstrate the vocabulary
# ===========================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HAND READING VOCABULARY FOR AI SYSTEMS")
    print("=" * 80)
    print()

    print("This vocabulary enables AI to interpret text descriptions of hands")
    print("and map them to capacity inferences (E_X vs specialist).")
    print()
    print("The AI cannot see hands. But it can parse what human observers report.")
    print()

    print("--- EXAMPLE DESCRIPTIONS ---")
    for example in EXAMPLE_DESCRIPTIONS:
        desc = example["description"]
        print(f"\n{example['name'].upper()}:")
        print(f"  Observation confidence: {desc.observation_confidence:.0%}")
        print(f"  E_X probability: {desc.inferred_e_x_probability():.2f} (expected {example['expected_e_x']})")
        print(f"  Specialist probability: {desc.inferred_specialist_probability():.2f} (expected {example['expected_specialist']})")
        print(f"  Domains: {desc.inferred_domains()}")
        print(f"  Experience: {desc.inferred_experience_level():.2f}")
        print(f"  Rubric scores: {desc.to_rubric_scores()}")
        print(f"  Rubric total: {sum(desc.to_rubric_scores().values()):.1f} / 100")

    print("\n" + "=" * 80)
    print("EXAMPLE AI CONTEXT (for Dale)")
    print("=" * 80)
    dale_desc = EXAMPLE_DESCRIPTIONS[0]["description"]
    print(dale_desc.to_ai_context())

    print("\n" + "=" * 80)
    print("TEXT PARSING DEMONSTRATION")
    print("=" * 80)
    sample_text = """
    Clean hands but you can see the calluses. He's got that shovel callus
    and the pliers callus and the hammer callus. Multiple small scars on
    the knuckles. Knuckles are thick. Nails are ridged. Skin is callused
    but clean. Wears gloves when the task calls for it. You can tell he's
    done everything.
    """
    print(f"Input text: {sample_text}")
    parsed = parse_hand_description_text(sample_text)
    if parsed:
        print(f"\nParsed observation confidence: {parsed.observation_confidence:.0%}")
        print(f"Parsed E_X probability: {parsed.inferred_e_x_probability():.2f}")
        print(f"Parsed domains: {parsed.inferred_domains()}")
        print(f"Parsed glove usage: {parsed.glove_usage.value}")

    print("\n" + "=" * 80)
    print("SERIALIZATION ROUND-TRIP")
    print("=" * 80)
    payload = dale_desc.to_json()
    print(payload[:400] + "...")
    restored = HandDescription.from_dict(json.loads(payload))
    assert restored.to_dict() == dale_desc.to_dict(), "round-trip failed"
    print("\nRound-trip OK. from_dict(to_dict(...)) is identity.")

    print("\n" + "=" * 80)
    print("LOW-CONFIDENCE REFUSAL")
    print("=" * 80)
    sparse = HandDescription(
        observer_relationship="router",
        cleanliness=CleanlinessLevel.CLEAN_UNWORKED,
    )
    print(sparse.to_ai_context())

    print("=" * 80)
    print("OBSERVER CHECKLIST (for no-photo interviews)")
    print("=" * 80)
    print(OBSERVER_CHECKLIST)
