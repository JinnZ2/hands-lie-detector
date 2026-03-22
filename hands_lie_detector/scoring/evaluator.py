"""
Score evaluation, aggregation, and context modifiers.
"""

from dataclasses import dataclass, field

from .rubric import ScoringRubric, ScoringCategory


@dataclass
class CategoryScore:
    """Score for a single rubric category."""

    category: str
    score: float
    max_points: int
    notes: str = ""


@dataclass
class ContextModifiers:
    """Adjustments applied after raw scoring."""

    cold_climate: bool = False          # +5 if clean hands score high on texture/localization
    glove_excuse_defensive: bool = False  # -15 if mentioned defensively before being asked
    age_decade: int | None = None       # Adjust bands +/-10 based on decade

    @property
    def total_adjustment(self) -> int:
        adj = 0
        if self.cold_climate:
            adj += 5
        if self.glove_excuse_defensive:
            adj -= 15
        return adj

    @property
    def age_band_shift(self) -> int:
        """Band shift based on age decade. Younger = stricter, older = more lenient."""
        if self.age_decade is None:
            return 0
        baseline = 4  # 40s as neutral decade
        return (self.age_decade - baseline) * 3


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    category_scores: list[CategoryScore]
    modifiers: ContextModifiers
    raw_total: float = 0.0
    adjusted_total: float = 0.0
    band_label: str = ""

    @property
    def score_breakdown(self) -> dict[str, float]:
        return {cs.category: cs.score for cs in self.category_scores}


class ScoreEvaluator:
    """Aggregates category scores, applies modifiers, and interprets results."""

    def __init__(self, rubric: ScoringRubric | None = None):
        self.rubric = rubric or ScoringRubric()

    def evaluate(
        self,
        scores: dict[str, float],
        modifiers: ContextModifiers | None = None,
        notes: dict[str, str] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate a set of category scores.

        Args:
            scores: Mapping of category name -> raw score.
            modifiers: Optional context modifiers.
            notes: Optional per-category notes.

        Returns:
            EvaluationResult with totals, adjustments, and band label.
        """
        modifiers = modifiers or ContextModifiers()
        notes = notes or {}

        category_scores = []
        for cat in self.rubric.categories:
            raw = scores.get(cat.name, 0.0)
            clamped = cat.validate_score(raw)
            category_scores.append(
                CategoryScore(
                    category=cat.name,
                    score=clamped,
                    max_points=cat.max_points,
                    notes=notes.get(cat.name, ""),
                )
            )

        raw_total = sum(cs.score for cs in category_scores)
        adjusted_total = max(0.0, min(100.0, raw_total + modifiers.total_adjustment))

        # Apply age band shift to interpretation
        shifted_total = adjusted_total + modifiers.age_band_shift
        band_label = self.rubric.interpret(shifted_total)

        return EvaluationResult(
            category_scores=category_scores,
            modifiers=modifiers,
            raw_total=raw_total,
            adjusted_total=adjusted_total,
            band_label=band_label,
        )

    def evaluate_from_list(
        self,
        score_values: list[float],
        modifiers: ContextModifiers | None = None,
    ) -> EvaluationResult:
        """Convenience: pass scores in category order as a list."""
        names = self.rubric.category_names
        if len(score_values) != len(names):
            raise ValueError(
                f"Expected {len(names)} scores, got {len(score_values)}. "
                f"Categories: {names}"
            )
        scores = dict(zip(names, score_values))
        return self.evaluate(scores, modifiers)
