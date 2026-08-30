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


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNSTATED = "unstated"


@dataclass(frozen=True)
class BiologicalCalibration:
    """Sex and age calibrate the THICKNESS BASELINE. They do not touch band position.

    The separation is the whole point:

        MEAN THICKNESS   is a calibrated quantity. absolute plate depth runs
                         lower at some baselines than others at identical load
                         and identical competence.
        CONTRAST / STATE is NOT calibrated. a banded hand at a lower baseline has
                         thinner plates with the SAME boundary sharpness, so the
                         state readout is already scale-invariant once the map is
                         normalized.

    Which means the contrast-based readout survives this confound and the
    monotone one does not — see `BandReadout.monotone_penalizes_baseline`.

    A consequence worth stating: at a lower baseline, healed lesions at load
    points are MORE informative, not less. Less gross thickness means lesions
    form closer to the sensing threshold, so their presence is stronger evidence
    of a maintained band position rather than weaker.

    The factor below is STIPULATED. It is a plausible range from general skin
    biology, not a fitted value, and it needs the calibration series in
    `calibration-standard.md` before it means anything.
    """

    sex: Sex = Sex.UNSTATED
    age: int | None = None
    female_baseline_factor: float = 0.75  # stipulated; range given as 0.7-0.8
    provenance: str = STIPULATED

    @property
    def is_evidence_based(self) -> bool:
        return self.provenance != STIPULATED

    @property
    def is_neutral(self) -> bool:
        return self.sex is Sex.UNSTATED and self.age is None

    @property
    def baseline_factor(self) -> float:
        """Multiplier the raw map is expected to sit at, relative to reference."""
        factor = 1.0
        if self.sex is Sex.FEMALE:
            factor *= self.female_baseline_factor
        if self.age is not None and self.age >= 60:
            factor *= 0.9  # stipulated: thinning with age
        return factor

    def normalize(self, thickness: dict["Zone", float]) -> dict["Zone", float]:
        """Restore a map to the reference scale so state logic reads unchanged."""
        f = self.baseline_factor
        if f == 1.0:
            return dict(thickness)
        return {z: min(1.0, t / f) for z, t in thickness.items()}


NEUTRAL_CALIBRATION = BiologicalCalibration()


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
    calibration: BiologicalCalibration = NEUTRAL_CALIBRATION

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
    def calibrated(self) -> dict[Zone, float]:
        """The map on the reference scale. State logic reads this."""
        return self.calibration.normalize(self.thickness)

    @property
    def raw_mean(self) -> float:
        """Uncalibrated mean. This is what a monotone scorer sees."""
        return sum(self.thickness.values()) / len(self.thickness) if self.thickness else 0.0

    @property
    def mean(self) -> float:
        """Calibrated mean, on the reference scale."""
        values = self.calibrated.values()
        return sum(values) / len(values) if values else 0.0

    @property
    def concentration(self) -> float:
        """Spatial dispersion of the map.

        Reads how CONCENTRATED the load history's geometry was. High means the
        contact set repeated; low means it varied. Neither is a competence claim.
        """
        values = list(self.calibrated.values())
        return pstdev(values) if len(values) > 1 else 0.0

    @property
    def edges(self) -> list[tuple[Zone, Zone, float]]:
        seen: set[frozenset[Zone]] = set()
        out: list[tuple[Zone, Zone, float]] = []
        calibrated = self.calibrated
        for zone, neighbours in ADJACENCY.items():
            if zone not in calibrated:
                continue
            for other in neighbours:
                key = frozenset({zone, other})
                if other not in calibrated or key in seen:
                    continue
                seen.add(key)
                out.append((zone, other, abs(calibrated[zone] - calibrated[other])))
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

    Deliberately reads the RAW map. The rubric has no calibration step, so a
    lower thickness baseline scores lower at identical load and identical
    competence — a third sign error in the same scale, compounding with the
    saturation one.
    """
    return 100.0 * reading.raw_mean


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
    def monotone_penalizes_baseline(self) -> bool:
        """True when an uncalibrated monotone score marks a hand down for baseline.

        The contrast readout is unaffected: the map is normalized before the
        state logic runs, so boundary sharpness and state survive the confound
        that the monotone scale walks straight into.
        """
        return not self.reading.calibration.is_neutral and (
            self.reading.calibration.baseline_factor < 1.0
        )

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
            f"  mean thickness  : {r.mean:.3f} calibrated / {r.raw_mean:.3f} raw",
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
        if self.monotone_penalizes_baseline:
            lines += [
                "",
                f"BASELINE: calibrated at x{r.calibration.baseline_factor:.2f}. the "
                "state above is read on the normalized map and is unaffected. the "
                "monotone score is NOT calibrated and marks this hand down for "
                "baseline at identical load — a separate sign error from saturation.",
                "  and at a lower baseline a healed lesion at a load point is MORE "
                "informative: it formed closer to the sensing threshold.",
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
    calibration: BiologicalCalibration = NEUTRAL_CALIBRATION,
) -> BandReadout:
    """Read a palmar thickness map. Returns concentration, not competence."""
    normalized = {
        (Zone(z) if isinstance(z, str) else z): v for z, v in thickness.items()
    }
    reading = ThicknessReading(normalized, thresholds, light, calibration)
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
