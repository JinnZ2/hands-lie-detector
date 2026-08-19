"""
Condition coordinates: what makes a contrast point plottable.

See `contrast-case.md`.

A single characterized point against a dense distribution cannot give a
variance. It can do three things, and they are the whole job of a contrast case:

    falsify "no such axis exists"
    give the axis a DIRECTION
    give a rough MAGNITUDE for the separation

Estimating the sparse arm's spread is a later and different question, and it
does not block any of the three.

But a contrast point needs its CONDITION specified or it plots nowhere. Not the
person — the condition: which needs are self-met versus purchased, at what rate,
over what duration, and which are seasonal versus continuous.

The dense arm's condition is implicit and unstated everywhere, which is exactly
why it reads as the baseline instead of as one setting. `ARM_A_UNSTATED` ships
with every coordinate marked UNSTATED and `is_plottable` returns False for it —
the baseline fails the same check the sparse arm is being asked to pass.

Stating the sparse arm's condition explicitly is what stops it becoming a second
unstated universal.
"""

from dataclasses import dataclass, field
from enum import Enum


class Provision(str, Enum):
    SELF_MET = "self_met"
    PURCHASED = "purchased"
    MIXED = "mixed"
    UNSTATED = "unstated"


class Cadence(str, Enum):
    CONTINUOUS = "continuous"
    SEASONAL = "seasonal"
    EPISODIC = "episodic"
    UNSTATED = "unstated"


@dataclass(frozen=True)
class NeedCoordinate:
    """One need, and how it is met. The unit of a condition specification."""

    need: str
    provision: Provision = Provision.UNSTATED
    rate: str = ""                     # operator's own units; free text on purpose
    duration_years: float | None = None
    cadence: Cadence = Cadence.UNSTATED

    @property
    def specified(self) -> bool:
        return (
            self.provision is not Provision.UNSTATED
            and self.cadence is not Cadence.UNSTATED
            and bool(self.rate)
            and self.duration_years is not None
        )

    def __str__(self) -> str:
        if self.provision is Provision.UNSTATED:
            return f"{self.need}: UNSTATED"
        dur = f"{self.duration_years:g}y" if self.duration_years is not None else "?y"
        return (
            f"{self.need}: {self.provision.value}, {self.rate or '?'}, "
            f"{dur}, {self.cadence.value}"
        )


# The needs a hand-load condition has to say something about. Not exhaustive —
# an operator adds their own.
DEFAULT_NEEDS: tuple[str, ...] = (
    "heat",
    "water",
    "food",
    "structure_build",
    "structure_repair",
    "equipment_repair",
    "transport",
    "clothing",
)


@dataclass
class ConditionSpec:
    """A condition, specified. Not a person, and not a population."""

    label: str
    coordinates: list[NeedCoordinate] = field(default_factory=list)
    note: str = ""

    @property
    def unstated(self) -> list[NeedCoordinate]:
        return [c for c in self.coordinates if not c.specified]

    @property
    def is_plottable(self) -> bool:
        """A condition with unstated coordinates plots nowhere."""
        return bool(self.coordinates) and not self.unstated

    def report(self) -> str:
        lines = [f"condition: {self.label}"]
        lines += [f"  {c}" for c in self.coordinates] or ["  (no coordinates)"]
        lines += [
            "",
            f"plottable: {'yes' if self.is_plottable else 'NO'}",
        ]
        if not self.is_plottable:
            lines.append(
                f"  {len(self.unstated)} of {len(self.coordinates)} coordinates "
                "unspecified. an unspecified condition is not a baseline; it is a "
                "setting that forgot to say which one it was."
            )
        if self.note:
            lines += ["", f"note: {self.note}"]
        return "\n".join(lines)


# The dense arm, as the literature actually leaves it.
ARM_A_UNSTATED = ConditionSpec(
    label="needs met by others (the dense prior)",
    coordinates=[NeedCoordinate(need=n) for n in DEFAULT_NEEDS],
    note=(
        "every coordinate UNSTATED. this is not a criticism of the data — the "
        "measurements are accurate for the condition they were taken in. the "
        "defect is a missing scope line, which is why this reads as the baseline "
        "rather than as one setting among others."
    ),
)


@dataclass
class ContrastPoint:
    """One characterized point against a dense distribution."""

    dense_arm: ConditionSpec
    sparse_arm: ConditionSpec
    n_sparse: int = 1

    SUPPORTED: tuple[str, ...] = (
        "falsifies 'no such axis exists'",
        "gives the axis a direction",
        "gives a rough magnitude for the separation",
    )

    NOT_SUPPORTED: tuple[str, ...] = (
        "variance or spread within the sparse arm",
        "prevalence or base rate on either arm",
        "any distributional claim at all",
    )

    @property
    def usable(self) -> bool:
        """Both conditions must be specified, or the point plots nowhere."""
        return self.sparse_arm.is_plottable

    def report(self) -> str:
        lines = [
            f"contrast point: n={self.n_sparse} against a dense arm",
            "",
            "supports:",
            *[f"  + {c}" for c in self.SUPPORTED],
            "",
            "does NOT support:",
            *[f"  - {c}" for c in self.NOT_SUPPORTED],
            "",
            f"sparse arm plottable: {'yes' if self.sparse_arm.is_plottable else 'NO'}",
            f"dense arm plottable : {'yes' if self.dense_arm.is_plottable else 'NO'}",
        ]
        if not self.dense_arm.is_plottable:
            lines.append(
                "  the dense arm's condition is unstated. it is being used as an "
                "origin without having coordinates of its own."
            )
        if not self.usable:
            lines.append(
                "\nnot yet usable: specify the sparse arm's coordinates, or it "
                "becomes a second unstated universal."
            )
        return "\n".join(lines)
