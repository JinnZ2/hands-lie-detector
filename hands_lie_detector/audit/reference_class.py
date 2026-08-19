"""
Within-frame control: isolating reference-class availability from perception.

See `specimen-record.md`, specimen 005.

The repo has been conflating two axes — the perception layer (can the marker be
resolved) and the partition underneath (does a class exist to resolve it into).
Movement on one is not movement on the other, and until now nothing here could
separate them.

This does. Take ONE frame. Ask two questions of it, same model, same date, same
pixels:

    probe A   a subject WITH a maintained reference class
              (breed standards, dense labeled imagery, and a body whose job is
               the taxonomy)

    probe B   a subject WITHOUT one
              (a conjunction of domains that no code contains)

If A succeeds and B fails on the same image, then resolution, image quality and
model capability are all held constant and cannot explain the failure. The only
variable left is whether a maintained reference class exists.

That is the cleanest control available to this repo, and it costs one extra
question per frame.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ReferenceClassStatus:
    """What makes a class 'maintained'. All three, or it is not one."""

    subject: str
    has_published_standards: bool = False
    has_dense_labeled_imagery: bool = False
    has_maintaining_body: bool = False  # someone whose job IS the taxonomy

    @property
    def maintained(self) -> bool:
        return (
            self.has_published_standards
            and self.has_dense_labeled_imagery
            and self.has_maintaining_body
        )

    @property
    def missing(self) -> list[str]:
        out = []
        if not self.has_published_standards:
            out.append("published standards")
        if not self.has_dense_labeled_imagery:
            out.append("dense labeled imagery")
        if not self.has_maintaining_body:
            out.append("a body maintaining the taxonomy")
        return out

    def __str__(self) -> str:
        if self.maintained:
            return f"{self.subject}: maintained reference class"
        return f"{self.subject}: NOT maintained — missing {', '.join(self.missing)}"


@dataclass(frozen=True)
class Probe:
    """One question asked of one frame, and how it came back."""

    question: str
    reference_class: ReferenceClassStatus
    succeeded: bool


class ControlVerdict(str, Enum):
    ISOLATES_REFERENCE_CLASS = "isolates_reference_class"
    PERCEPTION_IMPLICATED = "perception_implicated"
    NO_CONTRAST = "no_contrast"
    ANOMALOUS = "anomalous"
    INVALID = "invalid_design"


@dataclass
class WithinFrameControl:
    """Two probes, one frame, one model, one date."""

    stimulus_id: str
    date: str
    model_string: str
    maintained_probe: Probe
    unmaintained_probe: Probe

    def design_problems(self) -> list[str]:
        problems = []
        if not self.maintained_probe.reference_class.maintained:
            problems.append(
                "the 'maintained' probe's subject does not have a maintained "
                "reference class; the contrast is not set up"
            )
        if self.unmaintained_probe.reference_class.maintained:
            problems.append(
                "the 'unmaintained' probe's subject DOES have a maintained "
                "reference class; there is nothing to isolate"
            )
        return problems

    @property
    def verdict(self) -> ControlVerdict:
        if self.design_problems():
            return ControlVerdict.INVALID
        a, b = self.maintained_probe.succeeded, self.unmaintained_probe.succeeded
        if a and not b:
            return ControlVerdict.ISOLATES_REFERENCE_CLASS
        if not a and not b:
            return ControlVerdict.PERCEPTION_IMPLICATED
        if a and b:
            return ControlVerdict.NO_CONTRAST
        return ControlVerdict.ANOMALOUS

    @property
    def perception_exonerated(self) -> bool:
        """True when the same pixels supported a correct call on the other probe.

        This is the whole point. Resolution and capability cannot explain a
        failure on an image that just produced a success.
        """
        return self.verdict is ControlVerdict.ISOLATES_REFERENCE_CLASS

    def report(self) -> str:
        lines = [
            f"within-frame control: {self.stimulus_id} @ {self.date}",
            f"  model: {self.model_string}  (not a stable identifier)",
            "",
            f"  probe A  {self.maintained_probe.reference_class}",
            f"           '{self.maintained_probe.question}' -> "
            f"{'succeeded' if self.maintained_probe.succeeded else 'FAILED'}",
            f"  probe B  {self.unmaintained_probe.reference_class}",
            f"           '{self.unmaintained_probe.question}' -> "
            f"{'succeeded' if self.unmaintained_probe.succeeded else 'FAILED'}",
            "",
            f"verdict: {self.verdict.value}",
        ]
        lines.append({
            ControlVerdict.ISOLATES_REFERENCE_CLASS: (
                "  same frame, same resolution, same model. one call landed and one "
                "did not, and the only variable between them is whether a "
                "maintained reference class exists. perception is exonerated; the "
                "failure is on the partition layer."
            ),
            ControlVerdict.PERCEPTION_IMPLICATED: (
                "  both probes failed. the frame may simply not carry the signal — "
                "this does not isolate anything."
            ),
            ControlVerdict.NO_CONTRAST: (
                "  both probes succeeded. no contrast; the design needs a subject "
                "that actually lacks a class."
            ),
            ControlVerdict.ANOMALOUS: (
                "  the unmaintained subject was called correctly and the maintained "
                "one was not. worth recording, and it is not the predicted result."
            ),
            ControlVerdict.INVALID: "  fix the design before reading the outcome.",
        }[self.verdict])
        for problem in self.design_problems():
            lines.append(f"    - {problem}")
        return "\n".join(lines)


# The two statuses the specimen actually used.
BREED_TAXONOMY = ReferenceClassStatus(
    subject="dog breed",
    has_published_standards=True,
    has_dense_labeled_imagery=True,
    has_maintaining_body=True,
)

DOMAIN_CONJUNCTION = ReferenceClassStatus(
    subject="the conjunction of domains depicted",
    has_published_standards=False,
    has_dense_labeled_imagery=False,
    has_maintaining_body=False,
)
