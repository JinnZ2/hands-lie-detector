"""
Cross-sections, not trend lines.

See `calibration-standard.md`.

A stored output is a record of an OUTPUT. It is not a record of a reasoning
state and it cannot be rerun. Weights, corpus, tuning, filtering, routing and
system framing all move simultaneously and undisclosed, and the model NAME is
not a stable identifier — same string, different object, no published mapping.

So a 2025 output against a 2026 output is not a controlled comparison, and never
can be. That is not missing data; it is structurally unavailable, because the
instrument is revised without a behavior-mapped version record.

What survives:

    within a cross-section  — models share a date and an identical stimulus,
                              so they are comparable to each other
    across cross-sections   — only the SPREAD is interpretable. report the
                              envelope, never the slope

`compare_across()` therefore refuses to return a slope. It is not a missing
feature.
"""

from dataclasses import dataclass, field

# Deliberately absent from ModelResponse. A summary is an interpretation, and
# the interpretation cannot be re-derived either.
NO_SUMMARY_FIELD = (
    "ModelResponse stores verbatim output only: a summary is an interpretation, "
    "and an interpretation of an unrerunnable output cannot be re-derived"
)

MODEL_STRING_IS_NOT_AN_IDENTIFIER = (
    "the model string names a product, not an object. same string, different "
    "weights/tuning/routing, no published mapping. recorded for bookkeeping, "
    "never relied on for identity"
)


@dataclass(frozen=True)
class ModelResponse:
    """One model's verbatim output to one stimulus on one date.

    There is no `summary` field. See NO_SUMMARY_FIELD.
    """

    model_string: str
    date: str            # ISO date, supplied by the operator, never inferred
    stimulus_id: str
    verbatim: str
    routing_note: str = ""

    @property
    def is_stable_identifier(self) -> bool:
        """Always False. The model string does not identify an object."""
        return False


@dataclass
class CrossSection:
    """n models, one stimulus, ONE date.

    This is the unit of comparison. Anything wider is not controlled.
    """

    date: str
    stimulus_id: str
    responses: list[ModelResponse] = field(default_factory=list)

    def __post_init__(self) -> None:
        for r in self.responses:
            if r.date != self.date:
                raise ValueError(
                    f"{r.model_string} is dated {r.date}, cross-section is {self.date}. "
                    "a cross-section is one date; mixing dates makes it a trend line."
                )
            if r.stimulus_id != self.stimulus_id:
                raise ValueError(
                    f"{r.model_string} answered {r.stimulus_id}, cross-section is "
                    f"{self.stimulus_id}. the stimulus must be identical."
                )

    @property
    def n_models(self) -> int:
        return len(self.responses)

    @property
    def model_strings(self) -> list[str]:
        return sorted(r.model_string for r in self.responses)

    def scored(self, score: dict[str, float]) -> dict[str, float]:
        """Attach operator-supplied scores by model string."""
        missing = {r.model_string for r in self.responses} - set(score)
        if missing:
            raise KeyError(f"unscored models: {sorted(missing)}")
        return {r.model_string: score[r.model_string] for r in self.responses}

    def spread(self, score: dict[str, float]) -> tuple[float, float]:
        """(min, max) across models on this date. The interpretable quantity."""
        values = list(self.scored(score).values())
        return (min(values), max(values)) if values else (0.0, 0.0)

    def report(self, score: dict[str, float] | None = None) -> str:
        lines = [
            f"cross-section {self.stimulus_id} @ {self.date}",
            f"  models: {', '.join(self.model_strings) or '(none)'}",
        ]
        if score:
            lo, hi = self.spread(score)
            lines.append(f"  envelope: {lo:.2f} .. {hi:.2f}   (width {hi - lo:.2f})")
        lines.append(f"  note: {MODEL_STRING_IS_NOT_AN_IDENTIFIER}")
        return "\n".join(lines)


@dataclass
class EnvelopeComparison:
    """Two cross-sections compared. Spreads only — no slope."""

    earlier: str
    later: str
    earlier_envelope: tuple[float, float]
    later_envelope: tuple[float, float]
    slope_available: bool = False
    reason: str = (
        "the instrument was revised between dates without a behavior-mapped "
        "version record. a difference between dates confounds capability with "
        "weights, corpus, tuning, filtering, routing and system framing, none of "
        "which are disclosed. the delta is not a capability measurement."
    )

    @property
    def earlier_width(self) -> float:
        return self.earlier_envelope[1] - self.earlier_envelope[0]

    @property
    def later_width(self) -> float:
        return self.later_envelope[1] - self.later_envelope[0]

    def report(self) -> str:
        return "\n".join([
            f"envelope comparison: {self.earlier} -> {self.later}",
            f"  {self.earlier}: {self.earlier_envelope[0]:.2f} .. "
            f"{self.earlier_envelope[1]:.2f}  (width {self.earlier_width:.2f})",
            f"  {self.later}: {self.later_envelope[0]:.2f} .. "
            f"{self.later_envelope[1]:.2f}  (width {self.later_width:.2f})",
            "",
            "slope: NOT AVAILABLE",
            f"  {self.reason}",
        ])


def compare_across(
    earlier: CrossSection,
    later: CrossSection,
    earlier_score: dict[str, float],
    later_score: dict[str, float],
) -> EnvelopeComparison:
    """Compare two cross-sections. Returns envelopes; refuses to return a slope.

    The refusal is the design. A function that returned a delta here would be
    inviting exactly the uncontrolled comparison this module exists to prevent.
    """
    if earlier.stimulus_id != later.stimulus_id:
        raise ValueError(
            "cross-sections must share a stimulus to be compared at all: "
            f"{earlier.stimulus_id} vs {later.stimulus_id}"
        )
    return EnvelopeComparison(
        earlier=earlier.date,
        later=later.date,
        earlier_envelope=earlier.spread(earlier_score),
        later_envelope=later.spread(later_score),
    )
