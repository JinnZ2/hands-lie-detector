"""
Thickness-map readout: concentration, not skill.

See `band-not-scale.md`.

CORRECTION, and it is a serious one. An earlier version of this module treated
dispersion of the thickness map as a proxy for skill — "high contrast = the
working hand." That is wrong, and wrong in the direction this repo exists to
oppose:

    rotary / clamp   one repeated geometry
                     -> sharp localized plates, HIGH dispersion

    firewood         every piece a different diameter, bark, weight, balance;
                     a DISTRIBUTION of geometries
                     -> diffuse spread deposit, LOW dispersion

Same hours, same force, same competence, opposite score. A fixed-geometry
specialist and a variable-geometry generalist would have been ranked as if one
were skilled and the other soft — the desk-hands error re-entering through the
replacement metric, aimed at exactly the multi-domain case.

So dispersion is demoted. It measures the geometric CONCENTRATION of the load
history and nothing else. The skill claim is removed.

What follows from that: the thickness map alone cannot separate a
variable-geometry generalist from a saturated hand. Both are thick and both are
low-dispersion. `UNIFORM_THICK` is therefore reported as AMBIGUOUS, and the
separator is a functional SENSING test, not an image.

Two further limits, both real:

- The map needs RAKING LIGHT. Backlit or overhead-flat field photos cannot
  resolve thickness or boundary sharpness at all. See `LightCondition` and the
  tier split in `band-not-scale.md`.
- Thickness integrates over keratin turnover, roughly 2-4 weeks. It is a load
  HISTORY. It says nothing about events. Dorsal event marks are a different
  instrument on a different clock — see `integration.event_log`.
"""

from dataclasses import dataclass
from enum import Enum
from statistics import pstdev

from ..integration.domains import ADJACENCY, Surface, Zone

STIPULATED = "stipulated threshold; not fitted to data; falsifiable"

# Acts that hold a hand at a workable band position. None has a name, a
# procedure, a certification, or an instrument.
MANAGEMENT_ACTS: tuple[str, ...] = (
    "paring back before saturation",
    "hydration timing",
    "tool radius choice",
    "load spacing across days",
    "when to glove and when not to",
)


class LightCondition(str, Enum):
    """Whether the thickness map is measurable at all in this frame."""

    RAKING = "raking"            # single-source oblique: thickness resolvable
    FLAT_OVERHEAD = "flat_overhead"
    BACKLIT = "backlit"
    UNKNOWN = "unknown"

    @property
    def resolves_thickness(self) -> bool:
        return self is LightCondition.RAKING


class MapState(str, Enum):
    """What the thickness map shows. NOT a competence ranking."""

    SOFT = "soft"                      # low mean, low dispersion: no load
    CONCENTRATED = "concentrated"      # high dispersion: fixed-geometry history
    UNIFORM_THICK = "uniform_thick"    # AMBIGUOUS: generalist or saturated
    INDETERMINATE = "indeterminate"

    @property
    def ambiguous(self) -> bool:
        """UNIFORM_THICK cannot be resolved from thickness alone."""
        return self is MapState.UNIFORM_THICK


class Sensing(str, Enum):
    """A functional test, not an image. See `band-not-scale.md` on the ceiling."""

    INTACT = "intact"
    DEGRADED = "degraded"
    UNTESTED = "untested"


class BandPosition(str, Enum):
    IN_BAND = "in_band"
    OUT_SOFT = "out_of_band_soft"        # capacity never built
    OUT_SATURATED = "out_of_band_saturated"  # armored past sensing
    UNRESOLVED = "unresolved"            # needs the sensing test


@dataclass(frozen=True)
class ContrastThresholds:
    dispersion: float = 0.12
    thick_mean: float = 0.45
    decision_edge: float = 0.25
    provenance: str = STIPULATED

    @property
    def is_evidence_based(self) -> bool:
        return self.provenance != STIPULATED


DEFAULT_THRESHOLDS = ContrastThresholds()


@dataclass
class ThicknessReading:
    """A palmar thickness map plus its descriptive features.

    Thickness is normalized 0..1. Palmar zones only: the dorsal surface has no
    adaptation route, so it carries no thickness map to read.
    """

    thickness: dict[Zone, float]
    thresholds: ContrastThresholds = DEFAULT_THRESHOLDS
    light: LightCondition = LightCondition.UNKNOWN

    def __post_init__(self) -> None:
        for zone, value in self.thickness.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"thickness at {zone.value} must be in [0, 1]")
            if zone.surface is Surface.DORSAL:
                raise ValueError(
                    f"{zone.value} is dorsal: no adaptation route, so no thickness "
                    "map. dorsal marks are events — see integration.event_log"
                )

    @property
    def measurable(self) -> bool:
        """False unless the frame was shot in raking light."""
        return self.light.resolves_thickness

    @property
    def mean(self) -> float:
        return sum(self.thickness.values()) / len(self.thickness) if self.thickness else 0.0

    @property
    def concentration(self) -> float:
        """Spatial dispersion of the map.

        Reads how CONCENTRATED the load history's geometry was. High means the
        contact set repeated; low means it varied. Neither is a competence claim.
        """
        return pstdev(self.thickness.values()) if len(self.thickness) > 1 else 0.0

    @property
    def edges(self) -> list[tuple[Zone, Zone, float]]:
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
        return self.edges[0][2] if self.edges else 0.0

    @property
    def state(self) -> MapState:
        if not self.thickness:
            return MapState.INDETERMINATE
        if self.concentration >= self.thresholds.dispersion:
            return MapState.CONCENTRATED
        if self.mean >= self.thresholds.thick_mean:
            return MapState.UNIFORM_THICK
        return MapState.SOFT

    def sharpest_boundary(self) -> str:
        if not self.edges:
            return "(no adjacent zone pairs read)"
        a, b, step = self.edges[0]
        return f"{a.value} / {b.value}  step {step:.2f}"


def monotone_score(reading: ThicknessReading) -> float:
    """Stand-in for the shape of the seven-category rubric, 0-100.

    Not the rubric. Every category there rises with thickness and the total is
    their sum, so the composite is monotone in mean. This isolates that property.
    """
    return 100.0 * reading.mean


def band_position(state: MapState, sensing: Sensing) -> BandPosition:
    """Resolve band position from the map plus a functional sensing test.

    The map alone is not enough. UNIFORM_THICK covers both the
    variable-geometry generalist and the saturated hand, and only sensing
    separates them.
    """
    if state is MapState.SOFT:
        return BandPosition.OUT_SOFT
    if state is MapState.INDETERMINATE:
        return BandPosition.UNRESOLVED
    if sensing is Sensing.UNTESTED:
        return BandPosition.UNRESOLVED
    if sensing is Sensing.DEGRADED:
        return BandPosition.OUT_SATURATED
    return BandPosition.IN_BAND


@dataclass
class BandReadout:
    reading: ThicknessReading
    state: MapState
    monotone: float
    sensing: Sensing = Sensing.UNTESTED

    @property
    def position(self) -> BandPosition:
        return band_position(self.state, self.sensing)

    @property
    def requires_sensing_test(self) -> bool:
        return self.state.ambiguous and self.sensing is Sensing.UNTESTED

    @property
    def monotone_disagrees(self) -> bool:
        """The monotone scorer ranks a possibly-saturated hand highly."""
        return self.state is MapState.UNIFORM_THICK and self.monotone >= 45.0

    def interpretation(self) -> str:
        return {
            MapState.SOFT: (
                "low mean, low concentration. no load history deposited here."
            ),
            MapState.CONCENTRATED: (
                "the load history's geometry REPEATED — a fixed contact set laid "
                "plate in the same places. this says concentration, not skill: a "
                "variable-geometry generalist doing identical work scores low here."
            ),
            MapState.UNIFORM_THICK: (
                "thick and evenly spread. AMBIGUOUS by construction — this is what "
                "a variable-geometry load history looks like AND what a saturated "
                "hand looks like. the thickness map cannot separate them. run the "
                "sensing test."
            ),
            MapState.INDETERMINATE: "no zones read.",
        }[self.state]

    def report(self) -> str:
        r = self.reading
        lines = [
            f"map state: {self.state.value}",
            f"  {self.interpretation()}",
            "",
            f"  mean thickness  : {r.mean:.3f}   (monotone feature)",
            f"  concentration   : {r.concentration:.3f}   (geometry repeat, NOT skill)",
            f"  sharpest edge   : {r.sharpest_boundary()}",
            "",
            f"  sensing test    : {self.sensing.value}",
            f"  band position   : {self.position.value}",
            "",
            f"  monotone scorer would return: {self.monotone:.1f} / 100",
        ]
        if not r.measurable:
            lines[1:1] = [
                f"  NOT MEASURABLE: light is {r.light.value}. thickness and boundary",
                "  sharpness need raking light. treat the state above as unsupported.",
            ]
        if self.requires_sensing_test:
            lines += [
                "",
                "AMBIGUOUS: generalist or saturated. no image resolves this. the "
                "separator is a functional test — sub-millimetre placement under "
                "near-zero force, fingertip feedback only.",
            ]
        if self.monotone_disagrees:
            lines += [
                "",
                "SIGN ERROR RISK: the monotone scorer ranks this hand high, and the "
                "map cannot rule out saturation.",
            ]
        if not r.thresholds.is_evidence_based:
            lines += ["", f"note: thresholds are {r.thresholds.provenance}"]
        return "\n".join(lines)


def read_band(
    thickness: dict[Zone | str, float],
    thresholds: ContrastThresholds = DEFAULT_THRESHOLDS,
    light: LightCondition = LightCondition.UNKNOWN,
    sensing: Sensing = Sensing.UNTESTED,
) -> BandReadout:
    """Read a palmar thickness map. Returns concentration, not competence."""
    normalized = {
        (Zone(z) if isinstance(z, str) else z): v for z, v in thickness.items()
    }
    reading = ThicknessReading(normalized, thresholds, light)
    return BandReadout(
        reading=reading,
        state=reading.state,
        monotone=monotone_score(reading),
        sensing=sensing,
    )


def interpret_acute_damage(position: BandPosition, has_acute_lesion: bool) -> str:
    """What a lesion means, given band position.

    A hand held on the sensing side WILL blister where an armored hand would
    not. The lesion is the price of the band position, not evidence of failure.
    """
    if position is BandPosition.IN_BAND and has_acute_lesion:
        return (
            "consistent with band maintenance. a hand kept thin enough to sense "
            "blisters where an armored one would not. this is the cost of the band "
            "position, not a demerit."
        )
    if position is BandPosition.OUT_SATURATED and not has_acute_lesion:
        return (
            "expected, and not evidence of skill. armor prevents the lesion by "
            "removing the feedback that would have prevented the load."
        )
    if position is BandPosition.OUT_SOFT and has_acute_lesion:
        return "acute damage without adaptation. load exceeded an unprepared hand."
    if position is BandPosition.UNRESOLVED:
        return "band position unresolved; a lesion cannot be interpreted against it."
    return "no acute finding to interpret against band position."
