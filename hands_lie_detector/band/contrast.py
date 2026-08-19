"""
Three states, not a scale. Contrast as the primary feature.

See `band-not-scale.md`.

The monotone read — callus up, experience up — ranks a saturated, glassy,
undifferentiated hand HIGHEST. That hand is out of band on the far side: armored
past the sensing threshold, load without feedback. The monotone scorer cannot
express it. It isn't ranked low; it's ranked top. That is a sign error, not a
tuning problem.

Three states, and what separates them:

    SOFT / UNIFORM      low mean,  low contrast   no load
    BANDED              any mean,  HIGH contrast  the working hand
    THICK / GLASSY      high mean, low contrast   armored past sensing

Mean thickness cannot separate BANDED from GLASSY. Contrast can, and contrast
separates all three. So contrast replaces mean as the primary feature.

Why a boundary is hard to fake: a sharp edge records a DECISION — the hand
armored here and stayed open there. A decision requires sensing. Uniform
coverage means no decision was made. Grime is uniform, so contrast filters it
without anyone having to wash first, and props and costume never appear on the
surface at all.

All thresholds here are STIPULATED, like every other default in this repo.
"""

from dataclasses import dataclass
from enum import Enum
from statistics import pstdev

from ..integration.domains import ADJACENCY, Zone

STIPULATED = "stipulated threshold; not fitted to data; falsifiable"

# Acts that hold a hand in band. None of them has a name, a procedure, a
# certification, or an instrument — which is why the skill is invisible to every
# scorer that reads outcome instead of regulation.
MANAGEMENT_ACTS: tuple[str, ...] = (
    "paring back before saturation",
    "hydration timing",
    "tool radius choice",
    "load spacing across days",
    "when to glove and when not to",
)


class HandState(str, Enum):
    SOFT = "soft_uniform"          # no load
    BANDED = "banded"              # regulated toward a setpoint; the working hand
    GLASSY = "thick_glassy"        # armored past sensing; out of band, far side
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ContrastThresholds:
    """Where the state boundaries sit. Stipulated."""

    dispersion: float = 0.12     # population stdev of thickness across zones
    glassy_mean: float = 0.55    # above this with low contrast: armored
    decision_edge: float = 0.25  # a boundary this sharp records a decision
    provenance: str = STIPULATED

    @property
    def is_evidence_based(self) -> bool:
        return self.provenance != STIPULATED


DEFAULT_THRESHOLDS = ContrastThresholds()


@dataclass
class ThicknessReading:
    """A hand read as a thickness map, plus the features that separate states.

    Thickness is normalized 0..1, where 1 is maximally armored. The map does not
    need every zone — absent zones are simply not read.
    """

    thickness: dict[Zone, float]
    thresholds: ContrastThresholds = DEFAULT_THRESHOLDS

    def __post_init__(self) -> None:
        for zone, value in self.thickness.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"thickness at {zone.value} must be in [0, 1]")

    @property
    def mean(self) -> float:
        """The monotone feature. Cannot separate BANDED from GLASSY."""
        return sum(self.thickness.values()) / len(self.thickness) if self.thickness else 0.0

    @property
    def dispersion(self) -> float:
        """Spatial variance of the thickness map. The primary feature."""
        return pstdev(self.thickness.values()) if len(self.thickness) > 1 else 0.0

    @property
    def edges(self) -> list[tuple[Zone, Zone, float]]:
        """Thickness steps across adjacent zones, sharpest first."""
        seen: set[frozenset[Zone]] = set()
        out: list[tuple[Zone, Zone, float]] = []
        for zone, neighbours in ADJACENCY.items():
            if zone not in self.thickness:
                continue
            for other in neighbours:
                key = frozenset({zone, other})
                if other not in self.thickness or key in seen:
                    continue
                seen.add(key)
                out.append((zone, other, abs(self.thickness[zone] - self.thickness[other])))
        return sorted(out, key=lambda e: -e[2])

    @property
    def max_edge(self) -> float:
        """The sharpest boundary. One clear decision is enough to record one."""
        return self.edges[0][2] if self.edges else 0.0

    @property
    def edge_contrast(self) -> float:
        """Mean step across adjacent zones."""
        edges = self.edges
        return sum(e[2] for e in edges) / len(edges) if edges else 0.0

    @property
    def records_a_decision(self) -> bool:
        """A boundary sharp enough that the hand armored here and not there."""
        return self.max_edge >= self.thresholds.decision_edge

    @property
    def state(self) -> HandState:
        if not self.thickness:
            return HandState.INDETERMINATE
        if self.dispersion >= self.thresholds.dispersion:
            return HandState.BANDED
        if self.mean >= self.thresholds.glassy_mean:
            return HandState.GLASSY
        return HandState.SOFT

    @property
    def in_band(self) -> bool:
        return self.state is HandState.BANDED

    def sharpest_boundary(self) -> str:
        if not self.edges:
            return "(no adjacent zone pairs read)"
        a, b, step = self.edges[0]
        return f"{a.value} / {b.value}  step {step:.2f}"


def monotone_score(reading: ThicknessReading) -> float:
    """A stand-in for the monotone scorer, on a 0-100 scale.

    NOT the seven-category rubric — a model of its *shape*. Every category in
    `hands_lie_detector.scoring` rises with thickness and the total is their sum,
    so the composite is monotone in mean thickness. This function is that
    composite, isolated, so the sign error can be shown against the same input.
    """
    return 100.0 * reading.mean


@dataclass
class BandReadout:
    reading: ThicknessReading
    state: HandState
    monotone: float

    @property
    def monotone_disagrees(self) -> bool:
        """True where the monotone scorer ranks an out-of-band hand highly."""
        return self.state is HandState.GLASSY and self.monotone >= 55.0

    def interpretation(self) -> str:
        return {
            HandState.SOFT: (
                "no load. low mean, low contrast, no boundary anywhere — nothing "
                "was decided because nothing was carried."
            ),
            HandState.BANDED: (
                "regulated toward a setpoint. thick where load lands, thin where "
                "sensing is needed, edges maintained. this is the working hand, "
                "and the map survives washing because it is high-frequency."
            ),
            HandState.GLASSY: (
                "armored past the sensing threshold: load without feedback. out of "
                "band on the FAR side. a monotone scorer ranks this hand highest."
            ),
            HandState.INDETERMINATE: "no zones read.",
        }[self.state]

    def report(self) -> str:
        r = self.reading
        lines = [
            f"state: {self.state.value}",
            f"  {self.interpretation()}",
            "",
            f"  mean thickness   : {r.mean:.3f}   (monotone feature)",
            f"  dispersion       : {r.dispersion:.3f}   (primary feature)",
            f"  edge contrast    : {r.edge_contrast:.3f}",
            f"  sharpest boundary: {r.sharpest_boundary()}",
            f"  records a decision: {'yes' if r.records_a_decision else 'no'}",
            "",
            f"  monotone scorer would return: {self.monotone:.1f} / 100",
        ]
        if self.monotone_disagrees:
            lines += [
                "",
                "SIGN ERROR: the monotone scorer ranks this hand high. it is out of "
                "band on the far side. mean thickness cannot separate a regulated "
                "hand from a saturated one; only contrast can.",
            ]
        if not r.thresholds.is_evidence_based:
            lines += ["", f"note: thresholds are {r.thresholds.provenance}"]
        return "\n".join(lines)


def read_band(
    thickness: dict[Zone | str, float],
    thresholds: ContrastThresholds = DEFAULT_THRESHOLDS,
) -> BandReadout:
    """Read a thickness map as one of three states.

    Args:
        thickness: zone -> normalized thickness in [0, 1].
        thresholds: stipulated state boundaries.
    """
    normalized = {
        (Zone(z) if isinstance(z, str) else z): v for z, v in thickness.items()
    }
    reading = ThicknessReading(normalized, thresholds)
    return BandReadout(
        reading=reading, state=reading.state, monotone=monotone_score(reading)
    )


def interpret_acute_damage(state: HandState, has_acute_lesion: bool) -> str:
    """What a blister means, given band position.

    Inverts the standard reading. A hand held on the sensing side WILL blister
    where an armored hand would not — the lesion is the price of the band
    position, not evidence of failure or inexperience. A scorer that reads damage
    as incompetence has the sign wrong in the same direction as the monotone
    callus read.
    """
    if state is HandState.BANDED and has_acute_lesion:
        return (
            "consistent with band maintenance. a hand kept thin enough to sense "
            "blisters where an armored one would not. this is the cost of the band "
            "position, not a demerit."
        )
    if state is HandState.GLASSY and not has_acute_lesion:
        return (
            "expected, and not evidence of skill. armor prevents the lesion by "
            "removing the feedback that would have prevented the load."
        )
    if state is HandState.SOFT and has_acute_lesion:
        return "acute damage without adaptation. load exceeded an unprepared hand."
    return "no acute finding to interpret against band position."
