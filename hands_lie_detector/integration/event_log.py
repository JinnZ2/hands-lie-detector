"""
Dorsal marks: the event log. A different instrument on a different clock.

See `wear-taxonomy.md`.

Two surfaces, two instruments, and the repo previously specced only one:

    PALMAR                          DORSAL
      thick, adaptive                 thin, mobile, over metacarpals
      callus possible                 callus impossible — no adaptation route
      shear delamination              impact / snag / edge
      INTEGRATES over weeks           MARKS a single moment
      record of load history          record of an EVENT
      clock ~ keratin turnover, 2-4wk clock ~ healing, ~2wk

A dorsal count says nothing about accumulated load and everything about
CONDITIONS. So it gets its own track rather than being folded into wear metrics,
and this module refuses to emit a load estimate at all.
"""

from dataclasses import dataclass, field
from enum import Enum

from .domains import Surface, Zone


class MarkKind(str, Enum):
    LACERATION = "laceration"    # edge, released suddenly
    ABRASION = "abrasion"        # dragged across a rough counterface
    CONTUSION = "contusion"      # blunt strike
    SPLIT = "split"              # low-elasticity skin failing in tension
    THICKENING = "thickening"    # soft-tissue remodeling at a repeated strike point
    UNKNOWN = "unknown"


# Marks that record a discrete event and heal away. The dorsum's normal output.
EVENT_MARKS: frozenset[MarkKind] = frozenset({
    MarkKind.LACERATION, MarkKind.ABRASION, MarkKind.CONTUSION, MarkKind.SPLIT,
})

# The exception: repeated direct impact at a fixed site DOES remodel dorsally.
ADAPTATION_MARKS: frozenset[MarkKind] = frozenset({MarkKind.THICKENING})


# Cold moves the hand toward the low-sensing end of the window with no change in
# callus at all — environmental de-calibration. Combined with the season's
# higher edge density (chains, hooks, latches, frozen fittings that resist then
# release), this is why dorsal marks cluster in cold work.
COLD_MECHANISM = (
    "reduced sensing + reduced skin elasticity + higher edge density. the injury "
    "is not from more force"
)


@dataclass(frozen=True)
class DorsalMark:
    """One event, at one place, on one date."""

    zone: Zone
    date: str
    kind: MarkKind = MarkKind.UNKNOWN
    note: str = ""

    def __post_init__(self) -> None:
        if self.zone.surface is not Surface.DORSAL:
            raise ValueError(
                f"{self.zone.value} is palmar. palmar markers integrate load "
                "history; this log records events. use band.read_band instead."
            )


@dataclass
class EventLog:
    """Dorsal marks for one carrier. An event record, not a load record."""

    carrier: str = ""
    marks: list[DorsalMark] = field(default_factory=list)
    sampling_gate: str = ""
    baseline_frames: int = 0   # frames taken on a schedule, nothing wrong

    @property
    def carries_grip_load_history(self) -> bool:
        """Always False, and this is the strong claim.

        The dorsum is not a grip contact surface, so grip and shear load deposit
        nothing there. Whatever the palm integrates over its 2-4 week turnover,
        the dorsum does not record. A dorsal count says nothing about how much
        was carried.
        """
        return False

    @property
    def carries_impact_history(self) -> bool:
        """True only where repeated direct impact has remodeled a fixed site.

        CORRECTION to an earlier version of this module, which returned False
        unconditionally for all load history. That was too strong. The dorsum has
        no adaptation route for GRIP, but it is a contact surface for STRIKING,
        and repeated axial impact at the same knuckle does deposit soft-tissue
        thickening there.

        So the dorsum runs two channels, not one: an event log that heals away,
        and — under repeated strike at a fixed geometry — a deposit that does not.
        """
        return any(m.kind in ADAPTATION_MARKS for m in self.marks)

    @property
    def supports_rate_claims(self) -> bool:
        """False unless the sampling gate is stated AND is not external request.

        Documentation density tracks who asked, not what happened. Absence of
        record is not absence of event.
        """
        return bool(self.sampling_gate) and "request" not in self.sampling_gate.lower()

    @property
    def has_baseline_coverage(self) -> bool:
        """False when every frame exists because something went wrong.

        The narrative gate (see `economic-carve.md`) fires at event level and
        inside every population: heavy documenters still document the anomaly,
        not the Tuesday. So steady state is missing from EVERY arm, and a log
        with no scheduled frames shows wounds and hides the band — the opposite
        of what the band readout needs.
        """
        return self.baseline_frames > 0

    def zones_marked(self) -> dict[Zone, int]:
        counts: dict[Zone, int] = {}
        for m in self.marks:
            counts[m.zone] = counts.get(m.zone, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].value)))

    def condition_signature(self) -> list[str]:
        """What the marks say about CONDITIONS. Never about accumulated load."""
        out: list[str] = []
        kinds = {m.kind for m in self.marks}
        if MarkKind.LACERATION in kinds or MarkKind.ABRASION in kinds:
            out.append("edge density in the work environment")
        if MarkKind.SPLIT in kinds:
            out.append("low skin elasticity — cold, or defatted surface")
        if MarkKind.CONTUSION in kinds:
            out.append("confined working volume; strike against enclosure")
        if len(self.zones_marked()) >= 3:
            out.append("marks spread across the dorsum: not a single fixed hazard")
        return out

    def report(self) -> str:
        lines = [
            f"dorsal event log{f' ({self.carrier})' if self.carrier else ''}",
            f"  marks: {len(self.marks)}",
        ]
        lines += [
            f"    {z.value}: {n}" for z, n in self.zones_marked().items()
        ] or ["    (none)"]
        lines += ["", "  conditions indicated:"]
        lines += [f"    - {c}" for c in self.condition_signature()] or ["    (none)"]
        lines += [
            "",
            f"  sampling gate : {self.sampling_gate or 'UNSTATED'}",
            f"  rate claims   : {'permitted' if self.supports_rate_claims else 'NOT LICENSED'}",
        ]
        if not self.supports_rate_claims:
            lines.append(
                "    the denominator is unknown and unsampled — not under-sampled. "
                "few marks does not mean few events; it means few occasions to "
                "record. no frequency or severity-distribution claim is available "
                "from this log, permanently."
            )
        lines += [
            "",
            f"  baseline frames: {self.baseline_frames}"
            f"{'' if self.has_baseline_coverage else '  <- none scheduled'}",
        ]
        if not self.has_baseline_coverage:
            lines.append(
                "    every frame here exists because something went wrong. the "
                "maintained baseline lives in the boring frames, and no incentive "
                "in any population produces those — bind capture to something "
                "already recurring instead of to noticing."
            )
        lines += [
            "",
            "  this log carries NO load history. dorsal tissue has no adaptation "
            "route, so a count here says nothing about accumulated load.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dorsal signature: count and concentration separate two histories
# ---------------------------------------------------------------------------
#
# A field discriminator, in use by an operator with a dense observational sample
# and no published reference class behind it. Provenance is TESTIMONY, but from
# repeated observation rather than from a single case.
#
#   EDGE-STRIKE FIELD (the mechanic)
#     hundreds of small scars, scattered across the dorsum and the MCP row at
#     varied sites. superficial. the hand is in a confined space, the fastener
#     releases suddenly, and the knuckle travels into an edge. every reach has a
#     different geometry, so every mark lands somewhere new.
#     -> HIGH count, LOW concentration, no remodeling.
#
#   REPEATED IMPACT (the bare-knuckle striker)
#     few marks, concentrated on the 2nd and 3rd MCP heads — the knuckles that
#     land. soft-tissue thickening at those points rather than a scar field.
#     one geometry, repeated.
#     -> LOW count, HIGH concentration, remodeling present.
#
# Which is the CONCENTRATION axis from `band-not-scale.md`, running on the dorsal
# surface. Same quantity, different tissue: it reads how varied the contact
# geometry was, and it says nothing about competence in either case.


class DorsalSignature(str, Enum):
    EDGE_STRIKE_FIELD = "edge_strike_field"
    REPEATED_IMPACT = "repeated_impact"
    SPARSE = "sparse"
    MIXED = "mixed"
    UNREADABLE = "unreadable"


SIGNATURE_READING: dict[DorsalSignature, str] = {
    DorsalSignature.EDGE_STRIKE_FIELD:
        "many superficial marks at varied sites. a confined working volume with "
        "edges in it, entered repeatedly at different geometries. records event "
        "COUNT, not load carried.",
    DorsalSignature.REPEATED_IMPACT:
        "few marks, concentrated at the striking knuckles, with soft-tissue "
        "thickening. one geometry repeated. this is the dorsal case that does "
        "deposit.",
    DorsalSignature.SPARSE:
        "too few marks to distinguish a distribution from an accident.",
    DorsalSignature.MIXED:
        "high count AND concentrated remodeling. both histories, or a "
        "misclassified site list.",
    DorsalSignature.UNREADABLE: "no marks recorded.",
}

# Stipulated. The counts are a working threshold from one operator's field use,
# not a fitted boundary.
FIELD_THRESHOLD = 12          # marks above which "many" is meaningful
CONCENTRATION_THRESHOLD = 0.6  # share of marks at the top two sites

# The knuckles that land in a closed-fist strike.
STRIKE_ZONES: frozenset[Zone] = frozenset({Zone.DORSAL_MCP_KNUCKLES})


def dorsal_signature(log: "EventLog") -> DorsalSignature:
    """Separate an edge-strike field from a repeated-impact deposit.

    Reads COUNT and CONCENTRATION, which is the same pair the palmar map uses
    and for the same reason: it distinguishes varied contact geometry from a
    fixed one. Neither reading is a competence claim.
    """
    marks = log.marks
    if not marks:
        return DorsalSignature.UNREADABLE

    counts = log.zones_marked()
    top_two = sum(sorted(counts.values(), reverse=True)[:2])
    concentration = top_two / len(marks)
    remodeled = log.carries_impact_history

    many = len(marks) >= FIELD_THRESHOLD
    concentrated = concentration >= CONCENTRATION_THRESHOLD

    if many and concentrated and remodeled:
        return DorsalSignature.MIXED
    if many and not concentrated:
        return DorsalSignature.EDGE_STRIKE_FIELD
    if remodeled and concentrated:
        return DorsalSignature.REPEATED_IMPACT
    if len(marks) < 4:
        return DorsalSignature.SPARSE
    return DorsalSignature.MIXED
