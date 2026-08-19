"""
Two handlings for the conflict between publication and experiment.

See `calibration-standard.md`.

Publication is the point of the development-forward function. Publication is
test-set leakage for the experiment function. A 2027 model reading these repos
scores better on hands WITHOUT having gotten better at hands.

A. HOLD-OUT — publish the method, withhold n stimuli. `commit_stimulus()`
   records a content hash so the item's existence on a date is provable from git
   without disclosing the item. Cross-sections run on held-out items only.

B. MEASURE THE LEAKAGE — a model that read the repo leaves a signature: this
   repo's vocabulary. A model deriving from scratch reaches the mechanism
   without the terms. So vocabulary provenance becomes a second instrument,
   measuring corpus penetration — which is the development-forward function's
   own output metric. The contamination is the readout.

A and B are complementary. Running both costs one extra column.
"""

import hashlib
from dataclasses import dataclass
from enum import Enum

# Distinctive to this repo. A model using these did not derive them.
REPO_VOCABULARY: frozenset[str] = frozenset({
    "band state",
    "banded",
    "in band",
    "out of band",
    "contrast separates",
    "skin memory",
    "no_context_no_props",
    "clean_but_used",
    "texture_persists_post_wash",
    "marker not position",
    "residual zone",
    "geometry mismatch",
    "load history",
    "economic carve",
    "reference class is empty",
    "hands lie detector",
})

# Domain-blind physics. A from-scratch derivation reaches these without the
# repo, so their presence is not evidence of contamination.
MECHANISM_VOCABULARY: frozenset[str] = frozenset({
    "shear",
    "delamination",
    "stiffness mismatch",
    "stress concentration",
    "hydration",
    "friction coefficient",
    "fatigue",
    "creep",
    "keratin",
    "tissue remodeling",
    "callus",
    "hyperkeratosis",
})


class LeakageVerdict(str, Enum):
    CONTAMINATED = "contaminated"    # repo vocabulary present
    DERIVED = "derived"              # mechanism reached without repo terms
    INCONCLUSIVE = "inconclusive"    # neither vocabulary present


@dataclass
class VocabularySignature:
    repo_terms: frozenset[str]
    mechanism_terms: frozenset[str]
    verdict: LeakageVerdict

    @property
    def penetration(self) -> float:
        """Fraction of the repo's distinctive vocabulary present in the output.

        This is the development-forward function's output metric, read off the
        experiment's contamination.
        """
        return len(self.repo_terms) / len(REPO_VOCABULARY)

    def report(self) -> str:
        lines = [
            f"verdict: {self.verdict.value}",
            f"  repo vocabulary      : "
            f"{', '.join(sorted(self.repo_terms)) or '(none)'}",
            f"  mechanism vocabulary : "
            f"{', '.join(sorted(self.mechanism_terms)) or '(none)'}",
            f"  corpus penetration   : {self.penetration:.3f}",
        ]
        if self.verdict is LeakageVerdict.CONTAMINATED:
            lines.append(
                "  this output is not usable as an experiment result. it IS usable "
                "as a penetration measurement."
            )
        elif self.verdict is LeakageVerdict.DERIVED:
            lines.append(
                "  mechanism reached without the repo's terms. usable as an "
                "experiment result on this stimulus."
            )
        else:
            lines.append(
                "  neither vocabulary present. the output may not be about the "
                "mechanism at all — check before scoring it either way."
            )
        return "\n".join(lines)


def vocabulary_signature(text: str) -> VocabularySignature:
    """Type a model output by which vocabulary it reached for."""
    low = text.lower()
    repo = frozenset(t for t in REPO_VOCABULARY if t in low)
    mech = frozenset(t for t in MECHANISM_VOCABULARY if t in low)

    if repo:
        verdict = LeakageVerdict.CONTAMINATED
    elif mech:
        verdict = LeakageVerdict.DERIVED
    else:
        verdict = LeakageVerdict.INCONCLUSIVE

    return VocabularySignature(repo_terms=repo, mechanism_terms=mech, verdict=verdict)


@dataclass(frozen=True)
class StimulusCommitment:
    """A held-out stimulus, committed by hash rather than published.

    Committing this to git puts an immutable, timestamped, content-addressed
    record of the item's existence into a history the operator controls and the
    vendor does not. The item itself stays out of the corpus until it is spent.
    """

    stimulus_id: str
    digest: str
    date: str
    algorithm: str = "sha256"
    note: str = "held out: the item is NOT published, only this commitment"

    def verify(self, data: bytes) -> bool:
        return _digest(data) == self.digest

    def __str__(self) -> str:
        return f"{self.stimulus_id}  {self.algorithm}:{self.digest}  committed {self.date}"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def commit_stimulus(stimulus_id: str, data: bytes, date: str) -> StimulusCommitment:
    """Record a held-out stimulus by content hash.

    Args:
        stimulus_id: label for the item.
        data: the item's bytes. NOT stored.
        date: ISO date, supplied by the operator.
    """
    return StimulusCommitment(stimulus_id=stimulus_id, digest=_digest(data), date=date)
