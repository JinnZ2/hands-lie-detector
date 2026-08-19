"""
Sole wear as an audit of job descriptions.

See `sole-audit.md`.

    job description   AUTHORED. by an employer, a code, an HR taxonomy.
    sole wear         NOT authored. deposited.

So the delta between the wear pattern a stated job title PREDICTS and the wear
pattern actually on the boot is the gap between what the category claims a job
is and what the body actually did. The category's prediction is on the record,
it is testable, and a single boot can falsify it.

Properties that make this the cheapest instrument in the repo:

    static · still-photographable · good light available · no disposition gate ·
    no consent problem · already exists on every working person's feet, right now

The main confound has a free fix: separate work footwear means the record is
100% work-attributable.

Nobody collects this. The denominator is absent, the same as everywhere else —
the difference is that this one is cheap to build.
"""

from dataclasses import dataclass, field
from enum import Enum


class SoleZone(str, Enum):
    TOE = "toe"
    FOREFOOT_LUGS = "forefoot_lugs"
    FLEX_LINE = "flex_line"
    MIDFOOT_SHANK = "midfoot_shank"   # non-contact in normal gait
    LATERAL_HEEL = "lateral_heel"
    MEDIAL_HEEL = "medial_heel"


class Severity(str, Enum):
    NONE = "none"
    LIGHT = "light"
    HEAVY = "heavy"
    STRUCTURAL = "structural"   # material split, not just removed

    @property
    def rank(self) -> int:
        return ["none", "light", "heavy", "structural"].index(self.value)


class FailureMode(str, Enum):
    """Two channels, and they are the same pair the palm has.

    Palm: material removal versus delamination.
    Sole: abrasion versus fatigue cracking.
    """

    ABRASIVE = "abrasive"   # material removed; lug height loss
    FATIGUE = "fatigue"     # crack initiation at a stress concentration
    NONE = "none"


FAILURE_DRIVER: dict[FailureMode, str] = {
    FailureMode.ABRASIVE: "distance x surface aggression",
    FailureMode.FATIGUE: "flex cycles at the same line, every time",
    FailureMode.NONE: "-",
}

# Agents that lower the crack-initiation threshold. Rubber swells, plasticizers
# leach, then the compound embrittles; thermal cycling to a cold extreme
# embrittles it further.
CHEMICAL_AGENTS: tuple[str, ...] = ("diesel", "hydraulic fluid", "de-icer")


@dataclass(frozen=True)
class ZoneWear:
    zone: SoleZone
    severity: Severity
    mode: FailureMode = FailureMode.NONE


# The pattern normal walking deposits: lateral heel first at strike, then medial
# forefoot at toe-off.
GAIT_SIGNATURE: dict[SoleZone, Severity] = {
    SoleZone.LATERAL_HEEL: Severity.HEAVY,
    SoleZone.MEDIAL_HEEL: Severity.LIGHT,
    SoleZone.FOREFOOT_LUGS: Severity.LIGHT,
    SoleZone.TOE: Severity.LIGHT,
    SoleZone.FLEX_LINE: Severity.NONE,
    SoleZone.MIDFOOT_SHANK: Severity.NONE,
}


@dataclass(frozen=True)
class CategoryPrediction:
    """What a stated job title predicts about a sole. STIPULATED, and testable.

    The point of writing these down is that they are falsifiable. A category
    that predicts a nearly unworn sole has made a claim, and a boot can refute
    it.
    """

    job_title: str
    predicted: dict[SoleZone, Severity]
    expected_service_life_months: float
    note: str = ""


STIPULATED = "stipulated from the category's own description; not fitted to data"

# The prediction that the specimen falsifies.
DRIVER_AS_CATEGORIZED = CategoryPrediction(
    job_title="driver (as the category describes it)",
    predicted={
        **{z: Severity.LIGHT for z in SoleZone},
        # non-contact in any gait, so the category is not claiming anything here
        SoleZone.MIDFOOT_SHANK: Severity.NONE,
    },
    expected_service_life_months=24.0,
    note="the category says 'sits all day' and therefore predicts a nearly "
         "unworn sole. that is a claim, and it is on the record.",
)

CATEGORY_PREDICTIONS: dict[str, CategoryPrediction] = {
    DRIVER_AS_CATEGORIZED.job_title: DRIVER_AS_CATEGORIZED,
}

# What the job actually contains, mechanically. Not a job title — a contact list.
ACTUAL_CONTACT_SOURCES: tuple[tuple[str, str], ...] = (
    ("cab entry/exit", "3-4 rungs several times daily; ball of foot on a narrow "
                       "edge at high force, plus a twist"),
    ("pedal work", "ball loaded, heel pivoting, continuous"),
    ("trailer / catwalk", "climbing, ladder rungs, more edge loading"),
    ("landing gear", "standing braced, ball loaded, cranking"),
    ("yard surfaces", "gravel and diamond plate — aggressively abrasive"),
)


@dataclass
class SoleReading:
    """One boot, read as a wear map."""

    boot_id: str
    zones: dict[SoleZone, ZoneWear] = field(default_factory=dict)
    service_months: float | None = None
    chemical_exposure: tuple[str, ...] = ()
    separate_work_footwear: bool = False

    def severity(self, zone: SoleZone) -> Severity:
        wear = self.zones.get(zone)
        return wear.severity if wear else Severity.NONE

    @property
    def work_attributable(self) -> bool:
        """100% only when work footwear is not also worn off the job."""
        return self.separate_work_footwear

    @property
    def matches_gait_signature(self) -> bool:
        """Normal walking wears the heel first. Does this?"""
        heel = max(self.severity(SoleZone.LATERAL_HEEL).rank,
                   self.severity(SoleZone.MEDIAL_HEEL).rank)
        forefoot = max(self.severity(SoleZone.FOREFOOT_LUGS).rank,
                       self.severity(SoleZone.TOE).rank,
                       self.severity(SoleZone.FLEX_LINE).rank)
        return heel >= forefoot

    @property
    def inverted_signature(self) -> bool:
        """Heel preserved, forefoot destroyed. Not walking."""
        return not self.matches_gait_signature

    @property
    def dominant_mode(self) -> FailureMode:
        structural = [w for w in self.zones.values()
                      if w.severity is Severity.STRUCTURAL]
        if structural:
            return structural[0].mode
        heavy = [w for w in self.zones.values() if w.severity is Severity.HEAVY]
        return heavy[0].mode if heavy else FailureMode.NONE

    @property
    def time_to_failure_supports_distance_claim(self) -> bool:
        """False when the failure is fatigue-dominant.

        Four months is fast for abrasion alone. It is not fast for high-cycle
        flex on a chemically degraded compound. So a short service life is not a
        distance reading, and treating it as one is the same error shape as
        reading few photographs as few events.
        """
        return self.dominant_mode is FailureMode.ABRASIVE

    def mechanism_note(self) -> str:
        if self.dominant_mode is not FailureMode.FATIGUE:
            return ""
        agents = ", ".join(a for a in self.chemical_exposure if a in CHEMICAL_AGENTS)
        tail = f" chemically accelerated by {agents}." if agents else ""
        return (
            "fatigue-dominant: a crack initiated at a stress concentration under "
            "repeated flex, not material removed by distance." + tail +
            " the boot did not wear out. it fatigued out."
        )


@dataclass
class JobDescriptionAudit:
    """The instrument. Predicted-by-category against deposited-by-body."""

    reading: SoleReading
    stated_job: str
    prediction: CategoryPrediction

    def deltas(self) -> dict[SoleZone, int]:
        """Observed severity rank minus predicted, per zone."""
        return {
            zone: self.reading.severity(zone).rank
            - self.prediction.predicted.get(zone, Severity.NONE).rank
            for zone in SoleZone
        }

    @property
    def category_falsified(self) -> bool:
        """True when the body deposited more than the category allows."""
        return any(d > 0 for d in self.deltas().values())

    @property
    def service_life_shortfall(self) -> float | None:
        if self.reading.service_months is None:
            return None
        return self.prediction.expected_service_life_months - self.reading.service_months

    def report(self) -> str:
        lines = [
            f"job-description audit: {self.reading.boot_id}",
            f"  stated job : {self.stated_job}",
            f"  prediction : {self.prediction.note or '(none stated)'}",
            "",
            "  zone                 predicted   observed   delta",
        ]
        for zone in SoleZone:
            pred = self.prediction.predicted.get(zone, Severity.NONE)
            obs = self.reading.severity(zone)
            d = self.deltas()[zone]
            flag = "  <-" if d > 0 else ""
            lines.append(
                f"    {zone.value:<18} {pred.value:<11} {obs.value:<10} {d:+d}{flag}"
            )
        lines += [
            "",
            f"  gait signature   : "
            f"{'matches walking' if self.reading.matches_gait_signature else 'INVERTED — heel preserved, forefoot destroyed'}",
            f"  dominant mode    : {self.reading.dominant_mode.value} "
            f"({FAILURE_DRIVER[self.reading.dominant_mode]})",
            f"  work-attributable: "
            f"{'yes — separate work footwear' if self.reading.work_attributable else 'NOT ESTABLISHED — footwear also worn off the job'}",
        ]
        shortfall = self.service_life_shortfall
        if shortfall is not None:
            lines.append(
                f"  service life     : {self.reading.service_months:g} months against "
                f"{self.prediction.expected_service_life_months:g} predicted "
                f"({shortfall:+g})"
            )
        note = self.reading.mechanism_note()
        if note:
            lines += ["", f"  {note}"]
        if not self.reading.time_to_failure_supports_distance_claim:
            lines.append(
                "  time-to-failure is NOT a distance reading here. do not convert "
                "it into miles walked."
            )
        lines += ["", "verdict:"]
        lines.append(
            "  CATEGORY FALSIFIED. the sole carries load the stated job title "
            "does not allow for. the delta is the gap between what the category "
            "claims the job is and what the body actually did."
            if self.category_falsified else
            "  no zone exceeds the category's prediction on this boot."
        )
        return "\n".join(lines)


def audit(
    reading: SoleReading,
    stated_job: str,
    registry: dict[str, CategoryPrediction] | None = None,
) -> JobDescriptionAudit:
    registry = registry or CATEGORY_PREDICTIONS
    if stated_job not in registry:
        raise KeyError(
            f"no stipulated prediction for {stated_job!r}. write one first — the "
            f"category has to make a falsifiable claim before it can be audited. "
            f"known: {sorted(registry)}"
        )
    return JobDescriptionAudit(reading, stated_job, registry[stated_job])
