"""
The nail plate: a third instrument, on a third clock.

See `wear-taxonomy.md`.

The repo had two clocks. This is the slowest of three, and it records events the
other two cannot:

    PALMAR   keratin turnover, 2-4 weeks    integrated load history
    DORSAL   healing, ~2 weeks              discrete events at the surface
    NAIL     plate growth, ~4-6 months      matrix trauma, DATED BY DISTANCE
                                            from the fold

A mark in the nail plate was made at the matrix and has been carried outward
since. So its position along the plate is a clock — the only self-dating marker
on the hand. Leukonychia (white transverse marks) is matrix trauma from impact
or sustained pressure, and it is a SECOND CHANNEL on the same events the palm
and dorsum record differently.

Growth rate is stipulated and varies; the ordering of marks is far more robust
than any date computed from them.
"""

from dataclasses import dataclass, field
from enum import Enum

STIPULATED = "stipulated growth rate; not measured on this carrier"

# Fingernail, roughly. Thumb is slower; toenail far slower.
MM_PER_MONTH = 3.0
FULL_REPLACEMENT_MONTHS = 5.0


class NailFinding(str, Enum):
    LEUKONYCHIA = "leukonychia"        # white transverse mark: matrix trauma
    RIDGING = "ridging"                # longitudinal; not load-specific
    BEAU_LINE = "beau_line"            # transverse groove: growth arrest
    EDGE_WEAR = "edge_wear"            # functional trimming / abrasion
    SUBUNGUAL_STAIN = "subungual_stain"
    SEPARATION = "separation"


# Which findings carry information about mechanical load, and which do not.
LOAD_BEARING: frozenset[NailFinding] = frozenset({
    NailFinding.LEUKONYCHIA,
    NailFinding.EDGE_WEAR,
    NailFinding.SUBUNGUAL_STAIN,
    NailFinding.SEPARATION,
})


@dataclass(frozen=True)
class NailMark:
    """One finding, positioned along the plate.

    `distance_from_fold_mm` is the clock. It is measured, not inferred.
    """

    digit: str
    finding: NailFinding
    distance_from_fold_mm: float | None = None
    note: str = ""

    @property
    def carries_load_information(self) -> bool:
        return self.finding in LOAD_BEARING

    @property
    def months_since_event(self) -> float | None:
        """Stipulated conversion. The ORDERING is the robust part, not this."""
        if self.distance_from_fold_mm is None:
            return None
        return self.distance_from_fold_mm / MM_PER_MONTH

    def __str__(self) -> str:
        age = (
            f"~{self.months_since_event:.1f} mo"
            if self.months_since_event is not None
            else "undated"
        )
        return f"{self.digit} {self.finding.value} ({age})"


@dataclass
class NailRecord:
    """Nail findings across the digits. Self-dating, unlike the other two tracks."""

    carrier: str = ""
    marks: list[NailMark] = field(default_factory=list)
    provenance: str = STIPULATED

    @property
    def is_evidence_based(self) -> bool:
        return self.provenance != STIPULATED

    @property
    def load_marks(self) -> list[NailMark]:
        return [m for m in self.marks if m.carries_load_information]

    @property
    def dated_marks(self) -> list[NailMark]:
        return sorted(
            (m for m in self.load_marks if m.months_since_event is not None),
            key=lambda m: m.months_since_event,
        )

    @property
    def supports_ordering(self) -> bool:
        """Two dated marks give a sequence, which is the robust output."""
        return len(self.dated_marks) >= 2

    @property
    def independent_of_palmar_map(self) -> bool:
        """Always True. Matrix trauma is a separate channel from plate thickness.

        Which makes it corroboration rather than restatement: a nail record and a
        palmar map agreeing about repeated impact are two instruments, not one
        read twice.
        """
        return True

    def report(self) -> str:
        lines = [
            f"nail record{f' ({self.carrier})' if self.carrier else ''}",
            f"  load-bearing findings: {len(self.load_marks)} of {len(self.marks)}",
        ]
        lines += [f"    {m}" for m in self.marks] or ["    (none)"]
        lines += [
            "",
            f"  ordering available: {'yes' if self.supports_ordering else 'no — needs two dated marks'}",
            "",
            "  this is a SECOND CHANNEL on impact and sustained pressure. it "
            "corroborates the palmar map rather than restating it, and unlike "
            "either other track it dates its own marks by distance from the fold.",
        ]
        if not self.is_evidence_based:
            lines.append(
                f"  note: {self.provenance}. treat the ORDERING as the finding and "
                "the months as indicative only."
            )
        return "\n".join(lines)
