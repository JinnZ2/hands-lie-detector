"""
Test A from `reference-class-empty.md`: the double-dissociation discriminator.

The question: is multi-domain integration one quantity (H1), or a family that
decomposes by channel (H2)?

The design: two manipulations on ONE carrier, each moving a different channel
while holding total load fixed. Under H1 both move the same markers, because
there is only one thing to move. Under H2 each moves its own set, and the
crossing of those sets rejects H1.

Why n=1 is enough: this is a within-subject design, so the partition that
empties the reference class (enrollment by domain) never happens. "Moved" is
defined against the carrier's own baseline variability rather than against a
between-subject standard error, which is why no cohort is required.

A single dissociation is NOT enough. If M_geom moves set A and M_time moves
nothing, that is equally well explained by set B being insensitive or already
saturated. Only the crossing carries the inference. This module refuses to
report a single dissociation as support for H2.
"""

from dataclasses import dataclass, field
from enum import Enum

from .domains import Channel


class Verdict(str, Enum):
    DOUBLE_DISSOCIATION = "double_dissociation"    # rejects H1
    SINGLE_DISSOCIATION = "single_dissociation"    # inconclusive
    NO_DISSOCIATION = "no_dissociation"            # consistent with H1
    INSUFFICIENT_MOVEMENT = "insufficient_movement"  # nothing moved; test uninformative
    INVALID_DESIGN = "invalid_design"              # manipulations don't discriminate


@dataclass(frozen=True)
class Manipulation:
    """One channel moved while the others are held fixed.

    `total_load_held_fixed` is the precondition that makes this a channel test
    rather than a dose test. If load moves too, the design cannot separate
    "this channel matters" from "more work leaves more marks."
    """

    name: str
    targets: Channel
    holds_fixed: frozenset[Channel]
    total_load_held_fixed: bool = True
    notes: str = ""

    def validate(self) -> list[str]:
        problems = []
        if not self.total_load_held_fixed:
            problems.append(
                f"{self.name}: total load not held fixed; channel effect is "
                "confounded with dose"
            )
        if self.targets in self.holds_fixed:
            problems.append(f"{self.name}: targets and holds fixed the same channel")
        return problems


# The two canonical manipulations from the analysis. Both are runnable by one
# carrier without funding, enrollment, or a cohort.
M_GEOM = Manipulation(
    name="M_geom",
    targets=Channel.CONTACT_GEOMETRY,
    holds_fixed=frozenset({Channel.TIMING, Channel.RECOVERY_BUDGET, Channel.ATTENTION}),
    notes="Same schedule, same hours, same recovery. Swap a tool handle so two "
    "domains' grip geometries newly agree, or newly conflict.",
)

M_TIME = Manipulation(
    name="M_time",
    targets=Channel.TIMING,
    holds_fixed=frozenset(
        {Channel.CONTACT_GEOMETRY, Channel.RECOVERY_BUDGET, Channel.ATTENTION}
    ),
    notes="Same tools, same geometry, same total hours. Two domains on "
    "alternating days versus both inside the same day.",
)


@dataclass
class DissociationResult:
    manipulation_a: str
    manipulation_b: str
    moved_by_a: frozenset[str]
    moved_by_b: frozenset[str]
    verdict: Verdict
    design_problems: list[str] = field(default_factory=list)

    @property
    def only_a(self) -> frozenset[str]:
        return frozenset(self.moved_by_a - self.moved_by_b)

    @property
    def only_b(self) -> frozenset[str]:
        return frozenset(self.moved_by_b - self.moved_by_a)

    @property
    def both(self) -> frozenset[str]:
        return frozenset(self.moved_by_a & self.moved_by_b)

    @property
    def rejects_h1(self) -> bool:
        return self.verdict is Verdict.DOUBLE_DISSOCIATION

    def interpretation(self) -> str:
        return {
            Verdict.DOUBLE_DISSOCIATION: (
                "Crossing found: each manipulation moves markers the other does "
                "not. H1 (one quantity) is rejected. Integration decomposes; each "
                "channel needs its own instrument and has its own empty reference "
                "class."
            ),
            Verdict.SINGLE_DISSOCIATION: (
                "One manipulation moved markers the other did not, but not vice "
                "versa. Inconclusive: equally explained by a sensitivity or "
                "ceiling difference between marker sets. Not support for H2."
            ),
            Verdict.NO_DISSOCIATION: (
                "Both manipulations move the same markers. Consistent with H1: "
                "the channels are routes to loading one scalar."
            ),
            Verdict.INSUFFICIENT_MOVEMENT: (
                "Neither manipulation moved a marker beyond baseline variability. "
                "The test is uninformative — extend the exposure, lengthen the "
                "panel, or use markers with shorter time constants."
            ),
            Verdict.INVALID_DESIGN: (
                "The manipulations do not discriminate channels. Fix the design "
                "before reading the markers."
            ),
        }[self.verdict]

    def report(self) -> str:
        width = max(len(self.manipulation_a), len(self.manipulation_b)) + 14
        lines = [
            f"Test A: {self.manipulation_a} vs {self.manipulation_b}",
            f"  {f'moved by {self.manipulation_a} only':<{width}}: {_s(self.only_a)}",
            f"  {f'moved by {self.manipulation_b} only':<{width}}: {_s(self.only_b)}",
            f"  {'moved by both':<{width}}: {_s(self.both)}",
            "",
            f"verdict: {self.verdict.value}",
            f"  {self.interpretation()}",
        ]
        if self.design_problems:
            lines += ["", "design problems:"] + [f"  - {p}" for p in self.design_problems]
        return "\n".join(lines)


def _s(names) -> str:
    return ", ".join(sorted(names)) or "(none)"


def moved_markers(
    deltas: dict[str, float],
    baseline_noise: dict[str, float],
    k: float = 2.0,
) -> frozenset[str]:
    """Markers that moved beyond the carrier's own baseline variability.

    `baseline_noise` is the per-marker spread across repeated measures taken
    with nothing manipulated. This is the within-subject replacement for a
    between-subject standard error, and it is why the design does not need a
    cohort. A marker with no baseline estimate cannot be called moved.
    """
    return frozenset(
        m
        for m, d in deltas.items()
        if m in baseline_noise and abs(d) > k * baseline_noise[m]
    )


def double_dissociation(
    deltas_a: dict[str, float],
    deltas_b: dict[str, float],
    baseline_noise: dict[str, float],
    manipulation_a: Manipulation = M_GEOM,
    manipulation_b: Manipulation = M_TIME,
    k: float = 2.0,
) -> DissociationResult:
    """Run Test A on one carrier's marker deltas under two manipulations.

    Args:
        deltas_a: per-marker change under `manipulation_a`.
        deltas_b: per-marker change under `manipulation_b`.
        baseline_noise: per-marker spread across unmanipulated repeats.
        k: how many baseline spreads count as movement.

    Returns:
        DissociationResult. Only `Verdict.DOUBLE_DISSOCIATION` rejects H1.
    """
    problems = manipulation_a.validate() + manipulation_b.validate()
    if manipulation_a.targets == manipulation_b.targets:
        problems.append(
            f"both manipulations target {manipulation_a.targets.value}; "
            "no crossing is possible"
        )

    moved_a = moved_markers(deltas_a, baseline_noise, k)
    moved_b = moved_markers(deltas_b, baseline_noise, k)

    if problems:
        verdict = Verdict.INVALID_DESIGN
    elif not moved_a and not moved_b:
        verdict = Verdict.INSUFFICIENT_MOVEMENT
    elif (moved_a - moved_b) and (moved_b - moved_a):
        verdict = Verdict.DOUBLE_DISSOCIATION
    elif (moved_a - moved_b) or (moved_b - moved_a):
        verdict = Verdict.SINGLE_DISSOCIATION
    else:
        verdict = Verdict.NO_DISSOCIATION

    return DissociationResult(
        manipulation_a=manipulation_a.name,
        manipulation_b=manipulation_b.name,
        moved_by_a=moved_a,
        moved_by_b=moved_b,
        verdict=verdict,
        design_problems=problems,
    )
