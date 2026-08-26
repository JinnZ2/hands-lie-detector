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
    DEMONSTRATED   a capability performed under conditions, witnessed by someone
                   other than the carrier. more than self-report, less than a
                   measurement — a witness is not an instrument
    MEASURED       the tissue. the only line that does not drift
"""

from dataclasses import dataclass, field
from enum import Enum


class Provenance(str, Enum):
    OBSERVED = "observed"
    RECONSTRUCTED = "reconstructed"
    TESTIMONY = "testimony"
    DEMONSTRATED = "demonstrated"
    MEASURED = "measured"

    @property
    def stable_across_interval(self) -> bool:
        """Only the tissue is stable. Everything else is dated or reported."""
        return self is Provenance.MEASURED

    @property
    def witnessed(self) -> bool:
        """A performance a third party saw, not only a claim about one.

        DEMONSTRATED sits between TESTIMONY and MEASURED: the capability was
        performed under conditions and someone other than the carrier watched it
        happen. That is strictly more than self-report and strictly less than a
        measurement, because a witness is not an instrument.
        """
        return self is Provenance.DEMONSTRATED

    @property
    def rerunnable(self) -> bool:
        """Nothing from a model is rerunnable; the instrument is revised."""
        return self is Provenance.MEASURED


@dataclass(frozen=True)
class SpecimenLine:
    text: str
    provenance: Provenance
    note: str = ""
    demonstrable: bool = False

    @property
    def falsifiable_on_demand(self) -> bool:
        """A capability claim can be re-tested. A description cannot.

        "My hands are rough" and "I can hold sub-millimetre placement under
        near-zero force with fingertip feedback only" are both TESTIMONY, and
        they are not equally strong. The second names a performance under stated
        conditions, so it can be run again and fail. That property does not
        promote it to MEASURED — it is still the carrier's report — but it
        separates testimony that could be wrong and would show it from testimony
        that could be wrong quietly.
        """
        return self.demonstrable and self.provenance is Provenance.TESTIMONY

    def __str__(self) -> str:
        tail = f"  [{self.note}]" if self.note else ""
        mark = " (demonstrable)" if self.falsifiable_on_demand else ""
        return f"({self.provenance.value}{mark}) {self.text}{tail}"


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


# ---------------------------------------------------------------------------
# Corroboration — who verified, and which way their prior was pointing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Corroboration:
    """A third party's verification of a demonstrated capability.

    Two questions decide what it is worth, and they are independent.

    1. Did it run WITH or AGAINST the verifier's documented bias direction?
       A verifier whose documented tendency is to UNDER-attribute capability to
       this carrier, affirming the capability anyway, has affirmed it against
       its own prior. That is a hostile-witness argument and it strengthens the
       corroboration.

    2. Was there an operationalized threshold?
       Without one, "verified" is a judgment call by an unvalidated instrument —
       the same gap `attribution.ScoringInstrument` fails closed on. No amount
       of hostile-witness credit substitutes for a stated criterion.

    Only the second decides whether this reaches measurement. It does not.
    """

    verifier: str
    independent_of_carrier: bool
    bias_direction_documented: str = ""
    verification_runs_against_bias: bool = False
    operationalized_threshold: str = ""

    @property
    def has_threshold(self) -> bool:
        return bool(self.operationalized_threshold.strip())

    @property
    def reaches_measurement(self) -> bool:
        """Always False without a stated threshold. Usually False with one, too.

        A threshold is necessary and not sufficient: it also has to have been
        applied by something whose agreement with another rater is known.
        """
        return False

    @property
    def strength_note(self) -> str:
        if not self.independent_of_carrier:
            return (
                "not independent of the carrier. this is self-report with an extra "
                "step, not corroboration."
            )
        if self.verification_runs_against_bias:
            return (
                "runs AGAINST the verifier's documented bias direction, so the "
                "affirmation survived a hostile prior. worth more than the same "
                "verifier denying the capability would have been — and still not "
                "a measurement."
            )
        if self.bias_direction_documented:
            return (
                "runs WITH the verifier's documented bias direction. the "
                "affirmation is what that prior would produce anyway, so it adds "
                "little."
            )
        return "verifier bias direction undocumented; strength unassessed."

    def report(self) -> str:
        lines = [
            f"corroboration by {self.verifier}",
            f"  independent of carrier : {self.independent_of_carrier}",
            f"  bias direction         : {self.bias_direction_documented or 'undocumented'}",
            f"  runs against that bias : {self.verification_runs_against_bias}",
            f"  operationalized threshold: "
            f"{self.operationalized_threshold or 'NONE'}",
            "",
            f"  {self.strength_note}",
        ]
        if not self.has_threshold:
            lines.append(
                "  NO THRESHOLD: 'verified' here is a judgment call by an "
                "unvalidated instrument. the demonstration is real; the criterion "
                "it was judged against does not exist."
            )
        return "\n".join(lines)
