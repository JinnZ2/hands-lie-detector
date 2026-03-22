"""
Scoring rubric definitions for the Hands Lie Detector.

Encodes the 7-category, 100-point scoring system for estimating
experience probability from persistent physical markers.
"""

from dataclasses import dataclass, field


@dataclass
class ScoreTier:
    """A scoring tier within a category."""

    min_score: int
    max_score: int
    label: str
    description: str


@dataclass
class ScoringCategory:
    """A single scoring category in the rubric."""

    name: str
    max_points: int
    question: str
    tiers: list[ScoreTier]
    key_insight: str = ""
    red_flags: list[str] = field(default_factory=list)
    disqualifiers: list[str] = field(default_factory=list)

    def validate_score(self, score: float) -> float:
        """Clamp score to valid range."""
        return max(0.0, min(float(score), float(self.max_points)))


@dataclass
class InterpretationBand:
    """Maps a total score range to a human-readable label."""

    min_score: int
    max_score: int
    label: str


# ---------------------------------------------------------------------------
# Default rubric (v0.1)
# ---------------------------------------------------------------------------

TEXTURE_PERSISTENCE = ScoringCategory(
    name="Texture Persistence",
    max_points=25,
    question="Does the skin retain evidence after washing?",
    tiers=[
        ScoreTier(0, 5, "Smooth", "Smooth, uniform, cosmetic-grade"),
        ScoreTier(6, 15, "Mild", "Mild texture, mostly even"),
        ScoreTier(16, 25, "Deep", "Deep creases, thickened zones, uneven elasticity"),
    ],
    key_insight="Soap removes dirt. It does not reset skin architecture.",
)

WEAR_LOCALIZATION = ScoringCategory(
    name="Wear Localization",
    max_points=20,
    question="Is wear specific or generalized?",
    tiers=[
        ScoreTier(0, 5, "None", "No wear or fully uniform"),
        ScoreTier(6, 12, "Partial", "Some thickening, poorly localized"),
        ScoreTier(13, 20, "Task-specific", "Task-specific calluses (finger bases, thumb pad, heel of palm)"),
    ],
    red_flags=["'Rough everywhere' = aesthetic roughness, not labor"],
)

MICRO_INJURY_HISTORY = ScoringCategory(
    name="Micro-Injury History",
    max_points=15,
    question="Evidence of healed damage, not current mess?",
    tiers=[
        ScoreTier(0, 5, "None", "None visible"),
        ScoreTier(6, 10, "Recent", "Few, shallow, recent-only"),
        ScoreTier(11, 15, "Layered", "Multiple healed cuts/scars at varied stages"),
    ],
    key_insight="Competent workers heal injuries repeatedly.",
    disqualifiers=[
        "Fresh injuries only (could be performative)",
        "Identical injury repeated (suggests incompetence, not experience)",
    ],
)

TENDON_VEIN_DEFINITION = ScoringCategory(
    name="Tendon & Vein Definition",
    max_points=15,
    question="Does the hand show long-term load adaptation?",
    tiers=[
        ScoreTier(0, 5, "Soft", "Soft, low definition"),
        ScoreTier(6, 10, "Moderate", "Moderate definition"),
        ScoreTier(11, 15, "Clear", "Clear tendon paths, vascular prominence"),
    ],
    key_insight="This is usage over time, not fitness flexing.",
)

NAIL_EVIDENCE = ScoringCategory(
    name="Nail Evidence",
    max_points=10,
    question="Nails as tools vs nails as decorations?",
    tiers=[
        ScoreTier(0, 3, "Manicured", "Manicured, uniform, pristine"),
        ScoreTier(4, 7, "Worn", "Short, uneven wear"),
        ScoreTier(8, 10, "Functional", "Minor damage, staining, functional trimming"),
    ],
    key_insight="The thumbnail always snitches.",
)

SYMMETRY_OF_WEAR = ScoringCategory(
    name="Symmetry of Wear",
    max_points=10,
    question="One hand working, or both?",
    tiers=[
        ScoreTier(0, 3, "Dominant only", "Dominant hand only"),
        ScoreTier(4, 7, "Partial", "Partial symmetry"),
        ScoreTier(8, 10, "Bilateral", "Clear bilateral adaptation"),
    ],
    key_insight="Two-handed work = real work.",
)

CLIMATE_PPE = ScoringCategory(
    name="Climate & PPE Intelligence",
    max_points=5,
    question="Evidence of appropriate protection use?",
    tiers=[
        ScoreTier(0, 1, "None", "No consideration / 'I never wear gloves' claim"),
        ScoreTier(2, 3, "Inconsistent", "Inconsistent protection"),
        ScoreTier(4, 5, "Intelligent", "Clean hands + structural adaptation = intelligent PPE use"),
    ],
    key_insight="In cold climates, glove use is survival, not avoidance.",
    red_flags=["Defensive pre-explanation: 'I wore gloves'"],
)

DEFAULT_CATEGORIES = [
    TEXTURE_PERSISTENCE,
    WEAR_LOCALIZATION,
    MICRO_INJURY_HISTORY,
    TENDON_VEIN_DEFINITION,
    NAIL_EVIDENCE,
    SYMMETRY_OF_WEAR,
    CLIMATE_PPE,
]

DEFAULT_BANDS = [
    InterpretationBand(0, 30, "Podcast Hands"),
    InterpretationBand(31, 55, "Dirty but Unused / Casual Hobbyist"),
    InterpretationBand(56, 75, "Working Hands (Light-Moderate)"),
    InterpretationBand(76, 90, "Experienced Trade / Field Work"),
    InterpretationBand(91, 100, "Don't Ask, Just Hand Them the Tool"),
]


class ScoringRubric:
    """The full 100-point scoring rubric for hand experience detection."""

    def __init__(
        self,
        categories: list[ScoringCategory] | None = None,
        bands: list[InterpretationBand] | None = None,
    ):
        self.categories = categories or list(DEFAULT_CATEGORIES)
        self.bands = bands or list(DEFAULT_BANDS)

    @property
    def max_total(self) -> int:
        return sum(c.max_points for c in self.categories)

    @property
    def category_names(self) -> list[str]:
        return [c.name for c in self.categories]

    def get_category(self, name: str) -> ScoringCategory | None:
        for c in self.categories:
            if c.name == name:
                return c
        return None

    def interpret(self, total: float) -> str:
        """Return the interpretation band label for a total score."""
        for band in self.bands:
            if band.min_score <= total <= band.max_score:
                return band.label
        return "Out of range"

    def summary(self) -> str:
        """Human-readable rubric summary."""
        lines = [f"Hands Lie Detector Rubric (max {self.max_total} pts)", ""]
        for cat in self.categories:
            lines.append(f"  {cat.name} (0-{cat.max_points}): {cat.question}")
        lines.append("")
        lines.append("Interpretation Bands:")
        for band in self.bands:
            lines.append(f"  {band.min_score}-{band.max_score}: {band.label}")
        return "\n".join(lines)
