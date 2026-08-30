"""
Healing calibration: residual mark is not proportional to trauma.

See `healing-calibration.md`.

    residual mark  =  trauma  x  (1 - healing quality)

So a scar count read straight is monotone in RESIDUAL, and residual is inversely
related to how well the carrier heals. A better-healing carrier reads as
less-injured at an identical trauma history.

That is the same shape as the thickness-baseline error in
`band.BiologicalCalibration`: a physiological parameter carrying no load
information enters a monotone count uncalibrated, and marks the carrier down for
biology. It is the FOURTH sign error documented in this scale.

Every factor below pushes the same way — same mechanical events, less visible
evidence:

    faster wound closure          -> less residual mark
    cleaner collagen remodeling   -> less hypertrophic scarring
    thinner stratum corneum       -> less visible callus per unit load
    higher skin elasticity        -> tissue recovers shape, less permanent creasing
    better vascular response      -> bruising and erythema resolve faster
    hands kept in service         -> healing prioritized; no full recovery between
                                     events, so events overlap rather than stack

None of these is noise. The direction is fixed.

THE SEAM APPLIES. Healing rate is a constitutive parameter for living tissue, so
per `economic-carve.md` the ORDERING transfers and the COEFFICIENT does not.
Every magnitude here is stipulated and needs calibrating against the body being
read.
"""

from dataclasses import dataclass
from enum import Enum

STIPULATED = "stipulated direction with a stipulated magnitude; ordering is the "\
             "defensible part, the coefficient is not"


class Turnover(str, Enum):
    HIGH = "high"
    TYPICAL = "typical"
    LOW = "low"
    UNSTATED = "unstated"


# Stipulated multipliers on residual mark. Direction defensible, values not.
_TURNOVER_RESIDUAL: dict[Turnover, float] = {
    Turnover.HIGH: 0.6,
    Turnover.TYPICAL: 1.0,
    Turnover.LOW: 1.3,
    Turnover.UNSTATED: 1.0,
}


@dataclass(frozen=True)
class HealingCalibration:
    """How much of the trauma history survives as visible mark.

    Args:
        turnover: collagen turnover / remodeling quality.
        thinner_stratum_corneum: less visible callus per unit load.
        continuous_service: hands stay in use, so healing is prioritized and no
            event fully resolves before the next one lands. Events overlap
            rather than stack into distinct marks.
        provenance: stipulated by default, and it says so.
    """

    turnover: Turnover = Turnover.UNSTATED
    thinner_stratum_corneum: bool = False
    continuous_service: bool = False
    provenance: str = STIPULATED

    @property
    def is_evidence_based(self) -> bool:
        return self.provenance != STIPULATED

    @property
    def is_neutral(self) -> bool:
        return (
            self.turnover is Turnover.UNSTATED
            and not self.thinner_stratum_corneum
            and not self.continuous_service
        )

    @property
    def residual_factor(self) -> float:
        """Fraction of trauma that survives as visible mark. Stipulated."""
        factor = _TURNOVER_RESIDUAL[self.turnover]
        if self.thinner_stratum_corneum:
            factor *= 0.85
        if self.continuous_service:
            factor *= 0.85
        return factor

    @property
    def marks_understate_history(self) -> bool:
        return self.residual_factor < 1.0

    def implied_events(self, observed_marks: int) -> float:
        """Trauma events implied by a visible mark count.

        Deliberately returns a float and deliberately has no upper bound: this
        is a DIRECTION with a stipulated coefficient, not a count.
        """
        return observed_marks / self.residual_factor if self.residual_factor else float("inf")

    def report(self, observed_marks: int | None = None) -> str:
        lines = [
            "healing calibration",
            f"  turnover              : {self.turnover.value}",
            f"  thinner stratum corneum: {self.thinner_stratum_corneum}",
            f"  continuous service    : {self.continuous_service}",
            f"  residual factor       : {self.residual_factor:.2f}",
        ]
        if self.marks_understate_history:
            lines.append(
                "\n  MARKS UNDERSTATE THE HISTORY. the load history is LARGER than "
                "the scar count suggests, and a monotone injury score reads this "
                "carrier down for healing well — a physiological parameter with no "
                "load content."
            )
        if observed_marks is not None:
            lines.append(
                f"\n  {observed_marks} visible marks -> "
                f"~{self.implied_events(observed_marks):.1f} implied events "
                "(stipulated coefficient; treat the DIRECTION as the finding)"
            )
        if not self.is_evidence_based:
            lines.append(f"\n  note: {self.provenance}")
        return "\n".join(lines)


NEUTRAL_HEALING = HealingCalibration()


# What surface imaging cannot reach at any resolution or light angle.
BELOW_THE_SURFACE: tuple[str, ...] = (
    "healed fracture callus and bone remodeling",
    "joint capsule thickening",
    "ligament laxity from prior sprain",
    "tendon adhesion or re-routing after rupture",
)

SUBSURFACE_NOTE = (
    "breaks and crushes remodel BONE and JOINT, not only skin. that adaptation "
    "is invisible to surface imaging at any light angle, so a photographic "
    "instrument systematically under-reads a history containing fractures. this "
    "is a scope limit, not a resolution problem — no capture protocol fixes it."
)
