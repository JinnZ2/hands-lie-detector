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
from typing import List, Dict, Optional, Set
from enum import Enum


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

    # Free text for details the vocabulary doesn't capture
    notes: str = ""

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

        if e_x > 0.6:
            capacity_type = "CROSS-DOMAIN CLOSER (E_X)"
            confidence = "HIGH"
            recommendation = "This person likely closes coupled systems. Route accordingly."
        elif specialist > 0.6:
            capacity_type = "SINGLE-DOMAIN SPECIALIST"
            confidence = "HIGH"
            recommendation = f"Likely deep in {', '.join(domains) if domains else 'specific domain'}. Handoff may be required for other domains."
        else:
            capacity_type = "UNCERTAIN"
            confidence = "LOW"
            recommendation = "Insufficient substrate evidence for strong inference."

        return f"""
HAND SUBSTRATE ASSESSMENT
========================
Observer: {self.observer_relationship}
Practitioner context: {self.practitioner_context or 'not specified'}

OBSERVED FEATURES:
- Cleanliness: {self.cleanliness.value}
- Callus pattern: {self.callus_pattern.value}
- Callus zones: {len(self.callus_zones)} zones identified
- Scarring: {self.scarring.value}
- Knuckles: {self.knuckles.value}
- Nails: {self.nails.value}
- Skin texture: {self.skin_texture.value}
- Grip asymmetry: {self.grip_asymmetry.value}

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


# ===========================================================================
# Part 3. Example descriptions for training AI interpretation
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
            notes="Clean, symmetrical, no wear patterns. Gym callus at base of fingers from barbell but that's not tool work and the vocabulary distinguishes it.",
        ),
        "expected_e_x": 0.05,
        "expected_specialist": 0.05,
    },
]


# ===========================================================================
# Part 4. Text-to-inference function for AI systems
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

    # Cleanliness
    if "clean but" in text_lower or "clean yet" in text_lower:
        cleanliness = CleanlinessLevel.CLEAN_BUT_WORKED
    elif "embedded" in text_lower or "won't wash" in text_lower:
        cleanliness = CleanlinessLevel.EMBEDDED_DIRT
    elif "stained" in text_lower or "dirty" in text_lower:
        cleanliness = CleanlinessLevel.WORK_STAINED
    elif "clean" in text_lower and "no" in text_lower and "callus" in text_lower:
        cleanliness = CleanlinessLevel.CLEAN_UNWORKED
    else:
        cleanliness = CleanlinessLevel.UNKNOWN

    # Callus pattern
    if "multiple" in text_lower and "callus" in text_lower:
        if "different" in text_lower or "various" in text_lower:
            callus_pattern = CallusPattern.DISTRIBUTED_VARIED
        else:
            callus_pattern = CallusPattern.DISTRIBUTED_MODERATE
    elif "single" in text_lower and "callus" in text_lower:
        callus_pattern = CallusPattern.SINGLE_DEEP
    elif "heavy" in text_lower or "whole palm" in text_lower:
        callus_pattern = CallusPattern.HEAVY_GENERAL
    elif "no callus" in text_lower or "no calluses" in text_lower:
        callus_pattern = CallusPattern.NONE_VISIBLE
    else:
        callus_pattern = CallusPattern.UNKNOWN

    # Callus zones (keyword matching)
    callus_zones: List[CallusZone] = []
    zone_keywords = {
        "thumb crotch": CallusZone.THUMB_CROTCH,
        "between thumb": CallusZone.THUMB_CROTCH,
        "pliers": CallusZone.THUMB_CROTCH,
        "wrench": CallusZone.THUMB_CROTCH,
        "index side": CallusZone.INDEX_SIDE,
        "shovel": CallusZone.PALM_BELOW_INDEX,
        "rake": CallusZone.PALM_BELOW_INDEX,
        "palm below": CallusZone.PALM_BELOW_INDEX,
        "heel of palm": CallusZone.HEEL_OF_PALM,
        "hammer": CallusZone.HEEL_OF_PALM,
        "fingertip": CallusZone.FINGERTIP_PADS,
        "screwdriver": CallusZone.FINGERTIP_PADS,
        "across palm": CallusZone.ACROSS_PALM_CREASE,
        "rope": CallusZone.ACROSS_PALM_CREASE,
        "carrying": CallusZone.ACROSS_PALM_CREASE,
        "base of fingers": CallusZone.BASE_OF_FINGERS,
        "thumb pad": CallusZone.THUMB_PAD,
        "outer palm": CallusZone.OUTER_PALM_EDGE,
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
    elif "callused but clean" in text_lower or "calloused but clean" in text_lower:
        skin_texture = SkinTexture.CALLUSED_BUT_CLEAN
    elif "thickened" in text_lower:
        skin_texture = SkinTexture.THICKENED
    elif "embedded" in text_lower:
        skin_texture = SkinTexture.EMBEDDED_STAINING
    elif "smooth" in text_lower and "unworked" in text_lower:
        skin_texture = SkinTexture.SMOOTH_UNWORKED
    else:
        skin_texture = SkinTexture.UNKNOWN

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
        notes=text,
    )


# ===========================================================================
# Part 5. Main: demonstrate the vocabulary
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
        print(f"  E_X probability: {desc.inferred_e_x_probability():.2f} (expected {example['expected_e_x']})")
        print(f"  Specialist probability: {desc.inferred_specialist_probability():.2f} (expected {example['expected_specialist']})")
        print(f"  Domains: {desc.inferred_domains()}")
        print(f"  Experience: {desc.inferred_experience_level():.2f}")

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
    but clean. You can tell he's done everything.
    """
    print(f"Input text: {sample_text}")
    parsed = parse_hand_description_text(sample_text)
    if parsed:
        print(f"\nParsed E_X probability: {parsed.inferred_e_x_probability():.2f}")
        print(f"Parsed domains: {parsed.inferred_domains()}")
