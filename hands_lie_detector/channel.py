"""
Channel split: physical record and annotation are separate artifacts.

See `channel-split.md`.

EFFECTIVE FORWARD ONLY. Prior repo state keeps its rating; nothing below
retroactively invalidates an existing specimen.

The problem this fixes: the repo has been letting the physical record and the
author's captioning of it run as one artifact. Those are two channels with
DIFFERENT failure modes, and fusing them lets a reader dismiss both by
dismissing one.

    CH1  PHYSICAL RECORD    the measurement itself. mute. carries no caption.
                            fails by resolution, occlusion, angle — not by lying.

    CH2  ANNOTATION         the captioning of CH1.
                            fails the ordinary self-report ways: gameable,
                            memory, framing.

Rule:  hands don't lie, but hands don't caption.

The channels are CITED to each other and never merged. CH2 may point at CH1. It
may not overwrite it. A claim standing on CH1 alone survives CH2 being wrong,
and a claim standing on CH2 alone survives CH1 being unreadable — which is the
entire purpose of keeping them apart.
"""

from dataclasses import dataclass, field
from enum import Enum


class Channel(str, Enum):
    PHYSICAL_RECORD = "ch1_physical_record"
    ANNOTATION = "ch2_annotation"


# Each channel's own failure list. They do not overlap, which is the point.
FAILURE_MODES: dict[Channel, tuple[str, ...]] = {
    Channel.PHYSICAL_RECORD: (
        "resolution — the feature is below what the capture resolves",
        "occlusion — the feature is behind something",
        "angle — the light or camera geometry does not reveal it",
        "scale — no reference in frame, so the reading is not convertible",
    ),
    Channel.ANNOTATION: (
        "gameable — the caption can be written to suit a conclusion",
        "memory — the caption is recalled, not recorded at the time",
        "framing — the caption selects which features are worth naming",
        "vocabulary — the caption is written in terms that presuppose a class",
    ),
}

# What CANNOT go wrong with each, stated because it is the load-bearing half.
NOT_A_FAILURE_MODE: dict[Channel, str] = {
    Channel.PHYSICAL_RECORD:
        "lying. tissue deposits under load and has no access to the categories, "
        "so CH1 can be unreadable but not dishonest.",
    Channel.ANNOTATION:
        "resolution. a caption is not limited by light angle, which is why it "
        "reaches things CH1 cannot and why it must never be allowed to stand in "
        "for them.",
}


@dataclass(frozen=True)
class PhysicalRecord:
    """CH1. Mute by construction — there is no field for a caption."""

    record_id: str
    capture_date: str
    modality: str            # image, measurement, wear reading
    conditions: str = ""     # light, geometry, scale reference
    features: tuple[str, ...] = ()   # what is resolvable, named without inference

    @property
    def channel(self) -> Channel:
        return Channel.PHYSICAL_RECORD

    @property
    def failure_modes(self) -> tuple[str, ...]:
        return FAILURE_MODES[Channel.PHYSICAL_RECORD]

    @property
    def can_be_dishonest(self) -> bool:
        """Always False. CH1 can be unreadable; it cannot be untruthful."""
        return False


@dataclass(frozen=True)
class Annotation:
    """CH2. Points at a CH1 record by id. Cannot contain one."""

    annotation_id: str
    cites: str               # a PhysicalRecord.record_id
    text: str
    recorded_at_time_of_capture: bool = False

    @property
    def channel(self) -> Channel:
        return Channel.ANNOTATION

    @property
    def failure_modes(self) -> tuple[str, ...]:
        return FAILURE_MODES[Channel.ANNOTATION]

    @property
    def memory_exposed(self) -> bool:
        """True when the caption was written later than the capture."""
        return not self.recorded_at_time_of_capture


class Support(str, Enum):
    CH1_ONLY = "ch1_only"
    CH2_ONLY = "ch2_only"
    BOTH = "both"
    NEITHER = "neither"


@dataclass
class Claim:
    """A claim, with the channel or channels it rests on made explicit."""

    text: str
    rests_on_records: tuple[str, ...] = ()
    rests_on_annotations: tuple[str, ...] = ()

    @property
    def support(self) -> Support:
        r, a = bool(self.rests_on_records), bool(self.rests_on_annotations)
        if r and a:
            return Support.BOTH
        if r:
            return Support.CH1_ONLY
        if a:
            return Support.CH2_ONLY
        return Support.NEITHER

    @property
    def survives_ch2_failure(self) -> bool:
        """True when the claim needs no annotation to stand."""
        return self.support is Support.CH1_ONLY

    @property
    def survives_ch1_failure(self) -> bool:
        """True when the claim needs no physical record to stand."""
        return self.support is Support.CH2_ONLY

    @property
    def falls_with_either(self) -> bool:
        """The fused case: dismissing one channel dismisses the claim.

        This is what the split exists to make visible. A BOTH claim is not
        stronger for resting on two channels — it is more fragile, because it
        fails if either fails.
        """
        return self.support is Support.BOTH

    def report(self) -> str:
        lines = [
            f"claim: {self.text}",
            f"  support        : {self.support.value}",
            f"  survives CH2 wrong : {self.survives_ch2_failure}",
            f"  survives CH1 unreadable : {self.survives_ch1_failure}",
        ]
        if self.falls_with_either:
            lines.append(
                "  FUSED: rests on both channels, so dismissing either dismisses "
                "the claim. split it into the part CH1 carries and the part CH2 "
                "carries, and state them separately."
            )
        if self.support is Support.NEITHER:
            lines.append("  UNSUPPORTED: cites no record and no annotation.")
        return "\n".join(lines)


@dataclass
class ChannelSet:
    """CH1 and CH2 held apart, with citation running one direction only."""

    records: dict[str, PhysicalRecord] = field(default_factory=dict)
    annotations: dict[str, Annotation] = field(default_factory=dict)

    def add_record(self, record: PhysicalRecord) -> None:
        self.records[record.record_id] = record

    def add_annotation(self, annotation: Annotation) -> None:
        if annotation.cites not in self.records:
            raise KeyError(
                f"annotation {annotation.annotation_id!r} cites {annotation.cites!r}, "
                "which is not a record in this set. CH2 points at CH1; it cannot "
                "point at nothing."
            )
        self.annotations[annotation.annotation_id] = annotation

    @property
    def dangling_annotations(self) -> list[str]:
        return [a.annotation_id for a in self.annotations.values()
                if a.cites not in self.records]

    def annotations_for(self, record_id: str) -> list[Annotation]:
        return [a for a in self.annotations.values() if a.cites == record_id]

    def report(self) -> str:
        lines = ["channel set", f"  CH1 records    : {len(self.records)}",
                 f"  CH2 annotations: {len(self.annotations)}", ""]
        for rid, rec in sorted(self.records.items()):
            lines.append(f"  [{rid}] {rec.modality} @ {rec.capture_date}")
            for feature in rec.features:
                lines.append(f"      - {feature}")
            for ann in self.annotations_for(rid):
                mark = " (from memory)" if ann.memory_exposed else ""
                lines.append(f"      ~ CH2{mark}: {ann.text}")
        lines += [
            "",
            "  CH1 fails by: " + "; ".join(
                m.split(" — ")[0] for m in FAILURE_MODES[Channel.PHYSICAL_RECORD]),
            "  CH2 fails by: " + "; ".join(
                m.split(" — ")[0] for m in FAILURE_MODES[Channel.ANNOTATION]),
            "",
            "  the lists do not overlap. that is the reason for the split: a "
            "reader who discounts one channel has not touched the other.",
        ]
        return "\n".join(lines)
