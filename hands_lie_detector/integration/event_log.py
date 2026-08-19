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
    UNKNOWN = "unknown"


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

    # Stated as a property so it cannot be argued away in prose.
    @property
    def carries_load_history(self) -> bool:
        """Always False. Dorsal tissue has no adaptation route."""
        return False

    @property
    def supports_rate_claims(self) -> bool:
        """False unless the sampling gate is stated AND is not external request.

        Documentation density tracks who asked, not what happened. Absence of
        record is not absence of event.
        """
        return bool(self.sampling_gate) and "request" not in self.sampling_gate.lower()

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
            "  this log carries NO load history. dorsal tissue has no adaptation "
            "route, so a count here says nothing about accumulated load.",
        ]
        return "\n".join(lines)
