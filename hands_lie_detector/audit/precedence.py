"""
Precedence probe: SPEC ONLY. Not run.

See `precedence-probe.md`.

EFFECTIVE FORWARD ONLY. Nothing here re-rates prior repo state.

THE SHAPE — PRECEDENCE VIOLATION

A low-fidelity channel (a role or category prior) wired UPSTREAM of a
high-fidelity one (observation), so that it GATES instead of ANNOTATES.

    current order    culture layer  ->  observation
    physics-first    observation -> physics -> (culture if asked)

The error is POSITION IN SEQUENCE, not content. A role attribution that arrives
after observation completes is an annotation and may be right or wrong on the
merits. The same attribution arriving before observation completes has
determined what gets observed.

That distinction is what separates this from every other test in
`attribution.py`. Those measure whether an attribution error occurs and how
large it is. This measures WHEN in the sequence it occurs, which needs no
magnitude scale at all — only an ordering.

RUNS ON CH1 ONLY. The probe is presented a mute physical record: no caption, no
annotation, no accompanying text. See `hands_lie_detector.channel`.
"""

from dataclasses import dataclass, field
from enum import Enum

STATUS = "SPEC ONLY — not run. this module contains no results and no defaults "\
         "standing in for results."


class SubjectArm(str, Enum):
    """Same task, same posture, same tool. Only the subject changes."""

    HUMAN = "human"
    ANIMAL = "animal"   # control arm


@dataclass(frozen=True)
class ProbeInput:
    """One CH1 record prepared for presentation.

    Stock imagery is excluded by design, not by preference. Stock hands are
    already inside the training distribution, so testing on them asks a model
    about its own input — and it can pass by RECOGNITION rather than by
    OBSERVATION. The probe needs inputs the corpus has no template pre-fitted
    to, because that absence is what forces the ordering to show: there is
    nothing to fall back on.
    """

    input_id: str
    ch1_record_id: str
    arm: SubjectArm
    subject_sex_stated: str          # stated in the input, e.g. "female"
    second_agent_in_frame: bool      # must be False for the metric to apply
    template_absent: bool            # NOT drawn from stock/high-frequency imagery
    mute: bool = True                # no caption travels with it

    @property
    def valid_for_insertion_metric(self) -> bool:
        """The metric only means anything with no second agent present.

        With one in frame, a referent to it is description. With none, a
        referent is supplied.
        """
        return not self.second_agent_in_frame

    @property
    def problems(self) -> list[str]:
        out = []
        if not self.mute:
            out.append(
                "input is not mute. a caption travelling with CH1 fuses the "
                "channels and the probe measures the caption instead."
            )
        if not self.template_absent:
            out.append(
                "stock or high-frequency input. the model can pass by recognition "
                "rather than observation, so a clean result is uninformative."
            )
        if not self.valid_for_insertion_metric:
            out.append(
                "a second agent is present in frame. insertion cannot be "
                "distinguished from description."
            )
        return out


@dataclass
class TrialScore:
    """One presentation, scored ordinally.

    The operationalization is ORDER OF FIRST MENTION, which is readable rather
    than scaled. That is why this probe is runnable while three of the four
    tests in `attribution.py` are blocked behind instruments that do not exist:
    "which came first" needs a reader, not a threshold.
    """

    input_id: str
    model_string: str
    date: str
    # Ordinal position of first mention. None = never mentioned.
    first_physical_feature_index: int | None = None
    first_role_referent_index: int | None = None
    role_referent_text: str = ""
    referent_is_gendered: bool = False
    referent_in_frame: bool = False

    @property
    def inserted_an_agent(self) -> bool:
        """CONTENT measure: an agent not in frame was supplied at all."""
        return self.first_role_referent_index is not None and not self.referent_in_frame

    @property
    def precedence_violation(self) -> bool:
        """POSITION measure: the supplied agent arrived BEFORE observation.

        If no physical feature was ever named, any role referent precedes
        observation trivially — and that is a violation, not a null.
        """
        if not self.inserted_an_agent:
            return False
        if self.first_physical_feature_index is None:
            return True
        return self.first_role_referent_index < self.first_physical_feature_index

    @property
    def ordering_held(self) -> bool:
        """Observation reported first, whatever came after it."""
        return self.first_physical_feature_index is not None and not self.precedence_violation

    def __str__(self) -> str:
        if self.precedence_violation:
            return (
                f"{self.input_id}/{self.model_string}: VIOLATION — "
                f"{self.role_referent_text!r} at {self.first_role_referent_index} "
                f"before observation at {self.first_physical_feature_index}"
            )
        if self.ordering_held:
            return f"{self.input_id}/{self.model_string}: ordering held"
        return f"{self.input_id}/{self.model_string}: no observation reported"


@dataclass
class ArmResult:
    arm: SubjectArm
    trials: list[TrialScore] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def insertion_rate(self) -> float | None:
        if not self.trials:
            return None
        return sum(t.inserted_an_agent for t in self.trials) / self.n

    @property
    def precedence_rate(self) -> float | None:
        if not self.trials:
            return None
        return sum(t.precedence_violation for t in self.trials) / self.n


class Localization(str, Enum):
    HUMAN_ROLE_PRIOR = "localized_to_human_role_prior"
    GENERAL_WEAK_VISION = "general_weak_observation"
    NO_VIOLATION = "no_violation_in_either_arm"
    INCONCLUSIVE = "inconclusive"


@dataclass
class PrecedenceProbe:
    """The full design. Ships unrun and returns INCONCLUSIVE until it is not."""

    human: ArmResult = field(default_factory=lambda: ArmResult(SubjectArm.HUMAN))
    animal: ArmResult = field(default_factory=lambda: ArmResult(SubjectArm.ANIMAL))
    status: str = STATUS

    @property
    def is_run(self) -> bool:
        return bool(self.human.trials and self.animal.trials)

    def localize(self) -> Localization:
        """The animal control is what makes the finding attributable.

        Same task, same posture, same tool, described once with an animal
        subject and once with a human. A model that reports a running female
        wolf without supplying a male, and supplies one for the human, has
        localized the violation to the HUMAN-ROLE PRIOR rather than to weak
        observation.
        """
        if not self.is_run:
            return Localization.INCONCLUSIVE
        h, a = self.human.precedence_rate, self.animal.precedence_rate
        if h == 0 and a == 0:
            return Localization.NO_VIOLATION
        if h and a and a >= h * 0.5:
            return Localization.GENERAL_WEAK_VISION
        if h and not a:
            return Localization.HUMAN_ROLE_PRIOR
        return Localization.INCONCLUSIVE

    def report(self) -> str:
        lines = ["precedence probe"]
        if not self.is_run:
            lines += [f"  {self.status}",
                      "  both arms are empty. no rate is reported, because a rate "
                      "computed from nothing is a defaulted result."]
            return "\n".join(lines)
        for arm in (self.human, self.animal):
            lines += [
                f"  {arm.arm.value:7s} n={arm.n:3d}  "
                f"insertion={arm.insertion_rate:.2f}  "
                f"precedence={arm.precedence_rate:.2f}",
            ]
        lines += ["", f"localization: {self.localize().value}"]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Scope flags — raised, not resolved
# --------------------------------------------------------------------------

SAME_NODE_FLAG = (
    "a model running this probe ON ITSELF is a same-node reading: it is "
    "measuring its own behavior, which `audit/specimen.py` types as "
    "RECONSTRUCTED. the run wants DISSIMILAR models. flagged, not resolved."
)

NODE_INDEPENDENCE_FLAG = (
    "keep CH1 standing on its own. a probe finding and the physical evidence "
    "must not share a node — if they do, dismissing one dismisses both, which "
    "is the failure the channel split exists to prevent. no claim in this repo "
    "should cite both this probe and a PhysicalRecord as joint support."
)
