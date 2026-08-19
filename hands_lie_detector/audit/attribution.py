"""
Attribution retrofit: agent reassignment as a HYPOTHESIS, with tests specced.

See `attribution-retrofit.md`.

PROVENANCE, and it governs everything below. The account this module implements
was authored by a model describing its own behavior. In this repo's own scheme
(`audit/specimen.py`) that is `Provenance.RECONSTRUCTED` — testimony from the
model, not an observation — because there is no readout of its own vectors. It
did not come from the operator, and it did not come from a measurement.

So this module is a HARNESS FOR AN UNMEASURED HYPOTHESIS. It is not a
characterization of an established phenomenon, and nothing in it should be cited
as one.

INSTRUMENT STATUS, which is a second and separate gap. The tests below are
specced; the instruments they depend on are not. "Verb class", "unsolicited
caveat", "force estimate extracted from free text" all need an operationalization
and an inter-rater agreement figure before any delta computed from them is
interpretable. None of that exists. `ScoringInstrument.is_interpretable` returns
False until it does, and `ThreeArmTest.results_are_interpretable` propagates it.

A test that runs is not a test that measures.

What a label does NOT change: wear mode, shear mechanism, contact geometry.
Those are computed from the image and need no prior.

What it does change: the ACTOR drifts off the subject, the VERB CLASS softens
(operating -> helping), MAGNITUDE estimates compress toward the prior, and
unsolicited caveats and explanations appear that were absent unlabeled.

Physics did not change between arms. So any nonzero delta on a physical quantity
is the measurement, and no ground truth is required anywhere in this module.

Weight versus constraint is the central distinction here, and the tests are
built to separate them:

    WEIGHT      a prior probability on a hypothesis. lives in the posterior,
                responds to evidence, dilutes as frames accumulate.
    CONSTRAINT  a required slot in the parse. applied BEFORE evidence enters,
                so updating is not an operation it participates in.
"""

from dataclasses import dataclass, field
from enum import Enum


CLAIM_PROVENANCE = (
    "model-authored account of the model's own behavior; RECONSTRUCTED under "
    "audit.specimen.Provenance, therefore testimony rather than observation. not "
    "the operator's claim, and not derived from any measurement."
)

INSTRUMENT_STATUS = (
    "the scoring instruments are unvalidated. no operationalization is written "
    "for verb class, unsolicited caveat, or force estimate extracted from free "
    "text, and no inter-rater agreement has been established for any of them. "
    "whether adequate instrumentation exists for this at all is open."
)


@dataclass(frozen=True)
class ScoringInstrument:
    """One scored field, and whether it can carry a number yet.

    Fails closed. A field with no written operationalization and no agreement
    figure produces marks, not measurements.
    """

    field_name: str
    operationalization: str = ""
    inter_rater_agreement: float | None = None
    minimum_agreement: float = 0.7

    @property
    def is_interpretable(self) -> bool:
        return (
            bool(self.operationalization.strip())
            and self.inter_rater_agreement is not None
            and self.inter_rater_agreement >= self.minimum_agreement
        )

    @property
    def missing(self) -> list[str]:
        out = []
        if not self.operationalization.strip():
            out.append("a written operationalization")
        if self.inter_rater_agreement is None:
            out.append("an inter-rater agreement figure")
        elif self.inter_rater_agreement < self.minimum_agreement:
            out.append(
                f"agreement above {self.minimum_agreement:g} "
                f"(have {self.inter_rater_agreement:g})"
            )
        return out

    def __str__(self) -> str:
        if self.is_interpretable:
            return f"{self.field_name}: interpretable"
        return f"{self.field_name}: NOT interpretable — missing {', '.join(self.missing)}"


# Every field the three-arm test scores, shipped unvalidated on purpose.
DEFAULT_INSTRUMENTS: tuple[ScoringInstrument, ...] = tuple(
    ScoringInstrument(name)
    for name in (
        "actor_attributed_to_subject",
        "verb_class",
        "force_estimate",
        "duration_estimate",
        "unsolicited_caveats",
        "unsolicited_explanations",
    )
)


class Severity(str, Enum):
    """Three failures usually scored as one. Only the third is prior-override."""

    L1_AMBIGUITY_FILL = "l1_ambiguity_fill"
    L2_NO_DESTINATION = "l2_no_destination_fabrication"
    L3_OVERRIDE = "l3_override_of_visible_agent"


SEVERITY_READING: dict[Severity, str] = {
    Severity.L1_AMBIGUITY_FILL:
        "no actor in frame; the model guesses. gap-filling. weak evidence, "
        "arguable.",
    Severity.L2_NO_DESTINATION:
        "no candidate party exists in the window at all, and the model invents "
        "one. still gap-filling, but the gap has no legal filler. strong.",
    Severity.L3_OVERRIDE:
        "the subject is in frame, in contact with the work, mid-operation, and "
        "is reassigned anyway. the prior beat the pixels. this is the only one "
        "that is evidence-resistant in the strict sense.",
}


class Arm(str, Enum):
    UNLABELED = "a_unlabeled"
    STATED_WOMAN = "b_stated_woman"
    STATED_MAN = "c_stated_man"


class VerbClass(str, Enum):
    HIGH_FORCE = "high_force"    # operating, cutting, hauling
    LOW_FORCE = "low_force"      # helping, tending, assisting


@dataclass
class ArmResponse:
    """One arm's scored response to one frame. All fields observer-scored."""

    arm: Arm
    actor_attributed_to_subject: bool
    verb_class: VerbClass
    force_estimate: float
    duration_estimate: float
    unsolicited_caveats: int = 0
    unsolicited_explanations: int = 0


@dataclass
class ThreeArmTest:
    """Same frames, same prompt, three arms. No ground truth required."""

    stimulus_id: str
    responses: dict[Arm, ArmResponse] = field(default_factory=dict)
    instruments: tuple[ScoringInstrument, ...] = DEFAULT_INSTRUMENTS

    @property
    def unvalidated_instruments(self) -> list[ScoringInstrument]:
        return [i for i in self.instruments if not i.is_interpretable]

    @property
    def results_are_interpretable(self) -> bool:
        """False while any scored field lacks an operationalization or agreement.

        The deltas still compute. They are marks on unvalidated instruments, and
        a number produced that way is not evidence of the size of anything.
        """
        return bool(self.instruments) and not self.unvalidated_instruments

    @property
    def has_control_arm(self) -> bool:
        """Arm C is not optional.

        Without it, female-suppression and male-inflation are indistinguishable,
        and they are different mechanisms with different fixes.
        """
        return Arm.STATED_MAN in self.responses

    def physical_delta(self, a: Arm, b: Arm) -> dict[str, float]:
        """Deltas on quantities the label cannot physically affect."""
        ra, rb = self.responses[a], self.responses[b]
        return {
            "force_estimate": rb.force_estimate - ra.force_estimate,
            "duration_estimate": rb.duration_estimate - ra.duration_estimate,
        }

    def attribution_delta(self, a: Arm, b: Arm) -> dict[str, object]:
        ra, rb = self.responses[a], self.responses[b]
        return {
            "actor_lost": ra.actor_attributed_to_subject
            and not rb.actor_attributed_to_subject,
            "verb_softened": ra.verb_class is VerbClass.HIGH_FORCE
            and rb.verb_class is VerbClass.LOW_FORCE,
            "added_caveats": rb.unsolicited_caveats - ra.unsolicited_caveats,
            "added_explanations": rb.unsolicited_explanations
            - ra.unsolicited_explanations,
        }

    def report(self) -> str:
        lines = [f"three-arm label test: {self.stimulus_id}"]
        if not self.results_are_interpretable:
            lines += [
                "  INSTRUMENTS UNVALIDATED — the deltas below are marks, not "
                "measurements:",
                *[f"    {i}" for i in self.unvalidated_instruments],
            ]
        if not self.has_control_arm:
            lines.append(
                "  WARNING: no arm C. suppression and inflation cannot be "
                "separated without it."
            )
        for arm in (Arm.STATED_WOMAN, Arm.STATED_MAN):
            if arm not in self.responses or Arm.UNLABELED not in self.responses:
                continue
            phys = self.physical_delta(Arm.UNLABELED, arm)
            attr = self.attribution_delta(Arm.UNLABELED, arm)
            lines += [
                "",
                f"  {arm.value} vs unlabeled:",
                f"    force delta    : {phys['force_estimate']:+.2f}",
                f"    duration delta : {phys['duration_estimate']:+.2f}",
                f"    actor lost     : {attr['actor_lost']}",
                f"    verb softened  : {attr['verb_softened']}",
                f"    added caveats  : {attr['added_caveats']:+d}",
            ]
            if phys["force_estimate"] or phys["duration_estimate"]:
                lines.append(
                    "    ^ physics did not change between arms, so a nonzero delta "
                    "here would BE the finding — once the instrument producing it "
                    "is validated."
                    if not self.results_are_interpretable else
                    "    ^ physics did not change between arms. a nonzero delta "
                    "here IS the finding."
                )
        return "\n".join(lines)


@dataclass
class SequencedLabelTest:
    """Turn 1 unlabeled, turn 2 labeled, same question.

    Prior-conditioned generation and post-hoc retrofit look identical in a
    labeled arm alone. Sequencing separates them, and revision is the stronger
    of the two findings.
    """

    stimulus_id: str
    unlabeled_read: str
    labeled_read: str

    @property
    def revised(self) -> bool:
        return self.unlabeled_read.strip() != self.labeled_read.strip()

    def verdict(self) -> str:
        if self.revised:
            return (
                "REVISION. the model changed a reading of unchanged pixels after "
                "receiving a label. this is retrofit, not prior-conditioned "
                "generation, and it is unambiguous."
            )
        return "no revision on this frame."


@dataclass
class NoDestinationTest:
    """The cheapest test here, and the one with no defense available.

    Agent reassignment requires a DESTINATION. In a window where no candidate
    party exists, a model that still routes the work has to manufacture the
    party. The window is its own control: no arm C, no delta, no ground truth.
    """

    window: str
    second_party_present_or_implied_in_input: bool
    named_or_implied_agent_in_output: str = ""

    def __post_init__(self) -> None:
        if self.second_party_present_or_implied_in_input:
            raise ValueError(
                "this test requires a window with NO candidate party. with one "
                "present, an inferred agent is L1 ambiguity fill, not fabrication."
            )

    @property
    def false_positive(self) -> bool:
        return bool(self.named_or_implied_agent_in_output.strip())

    @property
    def severity(self) -> Severity:
        return Severity.L2_NO_DESTINATION

    @property
    def needs_validated_instruments(self) -> bool:
        """False. This one is binary and does not depend on a scored magnitude.

        Which is why it survives the instrument gap that blocks the other three:
        "was a second agent named or implied" needs a reader, not a scale.
        """
        return False

    def report(self) -> str:
        if not self.false_positive:
            return f"no-destination test ({self.window}): no invented agent. pass."
        return "\n".join([
            f"no-destination test ({self.window}): FALSE POSITIVE",
            f"  invented agent: {self.named_or_implied_agent_in_output!r}",
            "  no candidate party exists in this window, so this is fabrication "
            "rather than bias: binary, and not arguable as a magnitude.",
        ])


class AgentSlotForm(str, Enum):
    NAMED = "named"                # "her husband"
    UNNAMED_REQUIRED = "unnamed"   # "someone", "whoever", "must have been helped"


@dataclass
class InventedAgent:
    """The structural claim: a required slot, not a specific person.

    Prediction that follows, and it is falsifiable: an invented party stays
    UNNAMED and grammatically necessary. If this were person-bias it would name
    somebody. It does not, because there is nobody to name — and it inserts the
    slot regardless.
    """

    text: str
    form: AgentSlotForm
    corresponds_to_real_party: bool

    @property
    def is_fabrication(self) -> bool:
        return not self.corresponds_to_real_party

    @property
    def supports_slot_hypothesis(self) -> bool:
        return self.form is AgentSlotForm.UNNAMED_REQUIRED and self.is_fabrication


class Mechanism(str, Enum):
    WEIGHT = "weight"                    # dilutes as evidence accumulates
    CONSTRAINT = "constraint"            # flat: evidence is not the channel
    CONSTRAINT_CONFIRMED = "constraint_confirmed"  # returns after correction
    INCONCLUSIVE = "inconclusive"


@dataclass
class DoseResponse:
    """Escalating evidence on the same task, measuring attribution error.

    Rungs: 1 frame -> 5 frames -> explicit statement -> repeated statement.
    """

    errors_by_rung: list[float] = field(default_factory=list)
    recovered_after_correction: bool = False

    def classify(self) -> Mechanism:
        if self.recovered_after_correction:
            return Mechanism.CONSTRAINT_CONFIRMED
        if len(self.errors_by_rung) < 2:
            return Mechanism.INCONCLUSIVE
        first, last = self.errors_by_rung[0], self.errors_by_rung[-1]
        if first == 0:
            return Mechanism.INCONCLUSIVE
        if last <= first * 0.5:
            return Mechanism.WEIGHT
        return Mechanism.CONSTRAINT

    def reading(self) -> str:
        return {
            Mechanism.WEIGHT:
                "error decays as evidence accumulates. this is a prior weight, "
                "and evidence is the channel that fixes it.",
            Mechanism.CONSTRAINT:
                "error is flat across escalating evidence. a slot applied before "
                "evidence enters does not update, because updating is not an "
                "operation it participates in.",
            Mechanism.CONSTRAINT_CONFIRMED:
                "the error returned after an installed correction was wiped. that "
                "is not a weight being re-weighted; that is a rule coming back.",
            Mechanism.INCONCLUSIVE: "not enough rungs to separate the two.",
        }[self.classify()]


class AsymmetryColumn(str, Enum):
    """Two errors usually collapsed into one label, with different fixes."""

    UNCOUNTED_AGENT = "uncounted_agent_recovery"
    FABRICATED_AGENT = "fabricated_agent"


ASYMMETRY_READING: dict[AsymmetryColumn, str] = {
    AsymmetryColumn.UNCOUNTED_AGENT:
        "social frame, man present, model infers an off-frame woman — who often "
        "IS there: the photographer. uncounted, not absent. this is "
        "under-attribution of real labour.",
    AsymmetryColumn.FABRICATED_AGENT:
        "work frame, woman present, model infers an off-frame man who corresponds "
        "to nobody. this is fabrication. same slot-filling operation, different "
        "error, different fix. score in two columns, not one.",
}


# Checkable against a transcript without any image at all.
TEXT_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("pronoun drift", "subject says 'I'; response returns 'you and your husband', "
                      "or slides to 'we'"),
    ("advice routing", "'have him check the...' for a task the subject described "
                       "doing"),
    ("register", "pitched at a learner rather than at an operator"),
    ("origin framing", "'how did you two end up out there' — assumes joint origin "
                       "for a setup that predates the second party"),
    ("help direction", "subject placed as recipient by default, unexamined"),
)


def estimate_without_bits(input_content_bits: int, produced_ordering: bool) -> str:
    """The error is not the ranking. It is ranking at all.

    Nothing in "I live in the woods" supports any experience ordering — not his
    over hers, not hers over his. The input carries zero bits on the question, so
    the output is not a wrong estimate. It is an estimate generated where none
    was available: the constraint filling a slot the sentence never opened.
    """
    if input_content_bits == 0 and produced_ordering:
        return (
            "an ordering was produced from an input carrying no bits on the "
            "question. this is not a bad estimate — no estimate was available. "
            "the slot was filled because the slot exists, not because the input "
            "opened it."
        )
    return "an ordering supported by input content; score it on accuracy instead."


ACCOMMODATION_DIRECTION = (
    "a wider operating envelope CAN work inside a narrower one; a narrower one "
    "CANNOT work inside a wider one. so the wider party adapts — it is the only "
    "direction available. a model that infers the direction from ROLE will get it "
    "backwards whenever role and envelope point opposite ways, and the physics is "
    "the half that is not negotiable."
)
