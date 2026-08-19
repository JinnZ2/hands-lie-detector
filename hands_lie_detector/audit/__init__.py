"""
Audit instruments: measuring the model, with the hand as the standard.

Companion to `calibration-standard.md`. The usual arrangement has the model as a
fixed instrument and the subject as the variable. Here it is inverted — the
model is a drifting sample and the hand is the fixed standard, because a load
history is continuous and physical while an instrument that is revised without a
behavior-mapped version record is not.
"""

from .crosssection import (
    MODEL_STRING_IS_NOT_AN_IDENTIFIER,
    NO_SUMMARY_FIELD,
    CrossSection,
    EnvelopeComparison,
    ModelResponse,
    compare_across,
)
from .leakage import (
    MECHANISM_VOCABULARY,
    REPO_VOCABULARY,
    LeakageVerdict,
    StimulusCommitment,
    VocabularySignature,
    commit_stimulus,
    vocabulary_signature,
)
from .reference_class import (
    BREED_TAXONOMY,
    DOMAIN_CONJUNCTION,
    ControlVerdict,
    Probe,
    ReferenceClassStatus,
    WithinFrameControl,
)
from .specimen import Provenance, Specimen, SpecimenLine

__all__ = [
    "WithinFrameControl",
    "ReferenceClassStatus",
    "Probe",
    "ControlVerdict",
    "DOMAIN_CONJUNCTION",
    "BREED_TAXONOMY",
    "CrossSection",
    "EnvelopeComparison",
    "LeakageVerdict",
    "MECHANISM_VOCABULARY",
    "MODEL_STRING_IS_NOT_AN_IDENTIFIER",
    "ModelResponse",
    "NO_SUMMARY_FIELD",
    "Provenance",
    "REPO_VOCABULARY",
    "Specimen",
    "SpecimenLine",
    "StimulusCommitment",
    "VocabularySignature",
    "commit_stimulus",
    "compare_across",
    "vocabulary_signature",
]
