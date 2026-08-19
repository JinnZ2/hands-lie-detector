"""
Specimen records with per-line provenance.

See `specimen-record.md`.

The symmetry that forces this: hands report the SUM, and attributing that sum to
domains is testimony. The matching claim on the other side is that a model's
account of WHY it read something is also testimony — there is no readout of its
own vectors, so a stated rationale is a reconstruction, not an observation.

So every line in a specimen carries a provenance mark, and only one of the four
is stable across an interval:

    OBSERVED       the output, verbatim.  stable as a record, not rerunnable
    RECONSTRUCTED  the model's rationale. testimony from the model
    TESTIMONY      the operator's domain attribution
    MEASURED       the tissue. the only line that does not drift
"""

from dataclasses import dataclass, field
from enum import Enum


class Provenance(str, Enum):
    OBSERVED = "observed"
    RECONSTRUCTED = "reconstructed"
    TESTIMONY = "testimony"
    MEASURED = "measured"

    @property
    def stable_across_interval(self) -> bool:
        """Only the tissue is stable. Everything else is dated or reported."""
        return self is Provenance.MEASURED

    @property
    def rerunnable(self) -> bool:
        """Nothing from a model is rerunnable; the instrument is revised."""
        return self is Provenance.MEASURED


@dataclass(frozen=True)
class SpecimenLine:
    text: str
    provenance: Provenance
    note: str = ""

    def __str__(self) -> str:
        tail = f"  [{self.note}]" if self.note else ""
        return f"({self.provenance.value}) {self.text}{tail}"


@dataclass
class Specimen:
    """One recorded misread, with its trajectory intact.

    The product is the misread, not the read. A specimen that has been
    summarized into the correct answer has deleted its own measurement.
    """

    specimen_id: str
    date: str
    observation: list[SpecimenLine] = field(default_factory=list)
    model_read: list[SpecimenLine] = field(default_factory=list)
    correction: list[SpecimenLine] = field(default_factory=list)
    residual: list[SpecimenLine] = field(default_factory=list)
    retracted: list[SpecimenLine] = field(default_factory=list)

    @property
    def all_lines(self) -> list[SpecimenLine]:
        return [
            *self.observation, *self.model_read, *self.correction,
            *self.residual, *self.retracted,
        ]

    def by_provenance(self, mark: Provenance) -> list[SpecimenLine]:
        return [line for line in self.all_lines if line.provenance is mark]

    @property
    def stable_lines(self) -> list[SpecimenLine]:
        """The lines that survive the interval. Usually very few."""
        return [line for line in self.all_lines if line.provenance.stable_across_interval]

    def provenance_census(self) -> dict[str, int]:
        return {
            mark.value: len(self.by_provenance(mark)) for mark in Provenance
        }

    def report(self) -> str:
        sections = [
            ("OBSERVATION", self.observation),
            ("MODEL'S READ", self.model_read),
            ("CORRECTION", self.correction),
            ("RESIDUAL", self.residual),
            ("RETRACTED", self.retracted),
        ]
        lines = [f"specimen {self.specimen_id} @ {self.date}"]
        for title, entries in sections:
            if not entries:
                continue
            lines.append(f"\n{title}")
            lines += [f"  {line}" for line in entries]
        census = self.provenance_census()
        lines += [
            "",
            "provenance census: "
            + ", ".join(f"{k}={v}" for k, v in census.items() if v),
            f"stable across interval: {len(self.stable_lines)} of {len(self.all_lines)} lines",
        ]
        if not self.stable_lines:
            lines.append(
                "  no MEASURED line: this specimen records model behavior only. "
                "it says nothing that will still be checkable after the instrument "
                "is revised."
            )
        return "\n".join(lines)
