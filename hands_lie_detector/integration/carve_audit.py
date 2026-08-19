"""
Retrieval protocol and boundary-alignment audit.

See `economic-carve.md`.

Two things live here.

1. The retrieval protocol. Some relations are physically carved and transfer
   across domains freely, because no conservation law knows what a job title is.
   Others are economically carved — their strata are pay codes — and transfer is
   invalid by construction. `classify_relation` types a retrieval target so the
   second kind can be refused before it is imported into a multi-domain read.

   And one seam between them: constitutive parameters for LIVING tissue.
   Stiffness, fatigue limit, adaptation rate, hydration response. The relation
   is clean, but the NUMBER came from human samples, so the carve re-enters at
   the coefficient. Relations transfer; coefficients don't. Those terms return
   RELATIONAL_ONLY — use the ordering, ratio or direction, and calibrate the
   magnitude against the body in question.

2. The boundary-alignment audit, the discriminator for whether the economic,
   clinical, and linguistic carves are independent systems or dialects of one.

   The audit ships EMPTY. No classification system has been entered, no roster
   compared. It returns INSUFFICIENT_DATA until someone does the documentation
   work — committee rosters, charter dates, revision histories. It is a harness
   for an audit that has not been run, not a result.
"""

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# 1. Retrieval protocol
# ---------------------------------------------------------------------------


class RelationKind(str, Enum):
    """Where a relation's numbers came from.

    The earlier version of this module had two kinds — mechanism transfers,
    incidence does not. That is too clean, and the seam is worth naming.
    """

    GOVERNING = "governing"                    # equations; no population term, ever
    BOUNDARY = "boundary"                      # geometry; measurable directly
    MATERIAL_ENGINEERED = "material_engineered"  # measured on the material itself
    MATERIAL_LIVING = "material_living"        # THE SEAM: sampled from humans
    INCIDENCE = "incidence"                    # pay-code stratified
    UNKNOWN = "unknown"


class TransferScope(str, Enum):
    """What survives a move across domains.

    The workable rule at the seam: relations transfer, coefficients don't.
    Ratios, orderings and directions survive. Absolute magnitudes need
    calibrating against the body in question — which is what a maintained band
    gives you, and why the dated band series in `calibration-standard.md` is not
    just provenance but the source of the missing constants.
    """

    FULL = "full"                        # relation and magnitude both travel
    RELATIONAL_ONLY = "relational_only"  # ordering/ratio/direction only
    NONE = "none"


# Governing relations and boundary conditions. No population term appears in
# their statement, so nothing about who was employed can enter.
GOVERNING_RELATIONS: frozenset[str] = frozenset({
    "shear delamination",
    "stress concentration at a rigid inclusion",
    "stress concentration",
    "strain rate dependence",
    "hysteresis",
    "creep",
    "fatigue",
    "superposition",
    "load path",
})

BOUNDARY_CONDITIONS: frozenset[str] = frozenset({
    "contact geometry",
    "contact pressure distribution",
    "tool radius",
    "grip span",
    "contact area",
})

# Properties measured ON THE MATERIAL. Steel, concrete, wood, clay: the number
# came from the substance, not from a sampled population of people.
ENGINEERED_MATERIAL_PARAMS: frozenset[str] = frozenset({
    "steel modulus",
    "concrete compressive strength",
    "wood grain strength",
    "clay plasticity",
    "coefficient of friction",
})

# THE SEAM. Constitutive parameters for LIVING tissue. These numbers came from
# human samples, and the sampling is exactly where the carve re-enters. The
# mechanism chain stays clean end to end; the moment a NUMBER is wanted out of
# it for tissue, the population term is back inside the constant.
LIVING_TISSUE_PARAMS: frozenset[str] = frozenset({
    "stiffness",
    "stiffness mismatch",
    "elastic modulus",
    "fatigue limit",
    "adaptation rate",
    "hydration response",
    "friction-hydration curve",
    "tissue remodeling rate",
    "healing rate",
    "callus growth rate",
    "keratinization rate",
})

# Relations denominated in a population, and therefore in that population's
# recruitment frame, and therefore in pay codes.
INCIDENCE_RELATIONS: frozenset[str] = frozenset({
    "incidence",
    "prevalence",
    "risk ratio",
    "odds ratio",
    "exposure limit",
    "population norm",
    "attributable fraction",
    "years of exposure",
    "occupational hours",
    "base rate",
})

# Back-compatible alias: everything whose RELATION transfers, seam included.
MECHANISM_RELATIONS: frozenset[str] = (
    GOVERNING_RELATIONS
    | BOUNDARY_CONDITIONS
    | ENGINEERED_MATERIAL_PARAMS
    | LIVING_TISSUE_PARAMS
)

# Checked longest-first so a specific term beats a substring of it:
# "fatigue limit" (living tissue, seam) must not be caught by "fatigue"
# (governing relation, clean).
_RULES: list[tuple[frozenset[str], RelationKind, TransferScope, str]] = [
    (INCIDENCE_RELATIONS, RelationKind.INCIDENCE, TransferScope.NONE,
     "denominated in a sampled population whose strata are pay codes; transfer "
     "across domains is invalid by construction."),
    (LIVING_TISSUE_PARAMS, RelationKind.MATERIAL_LIVING, TransferScope.RELATIONAL_ONLY,
     "a constitutive parameter for living tissue. the relation transfers; the "
     "COEFFICIENT does not — that number came from human samples, and the "
     "sampling is where the carve re-enters. use the ordering, ratio or "
     "direction; calibrate the magnitude against the body in question."),
    (ENGINEERED_MATERIAL_PARAMS, RelationKind.MATERIAL_ENGINEERED, TransferScope.FULL,
     "measured on the material itself, not on a population of people."),
    (BOUNDARY_CONDITIONS, RelationKind.BOUNDARY, TransferScope.FULL,
     "a boundary condition: geometry, measurable directly."),
    (GOVERNING_RELATIONS, RelationKind.GOVERNING, TransferScope.FULL,
     "a governing relation. no population term appears in it, ever."),
]


@dataclass(frozen=True)
class RelationVerdict:
    term: str
    kind: RelationKind
    transfer: TransferScope
    reason: str

    @property
    def transferable(self) -> bool:
        """True if anything at all transfers. Check `transfer` for what."""
        return self.transfer is not TransferScope.NONE

    @property
    def magnitude_transfers(self) -> bool:
        """False at the seam. Ratios survive; absolute numbers need calibration."""
        return self.transfer is TransferScope.FULL

    def __str__(self) -> str:
        mark = {
            TransferScope.FULL: "transfers",
            TransferScope.RELATIONAL_ONLY: "RELATION TRANSFERS, COEFFICIENT DOES NOT",
            TransferScope.NONE: "DOES NOT TRANSFER",
        }[self.transfer]
        return f"{self.term}: {self.kind.value} — {mark}. {self.reason}"


def classify_relation(term: str) -> RelationVerdict:
    """Type a retrieval target and say what part of it may cross a domain.

    Substring matching, deliberately loose: "lifetime incidence in welders"
    should trip the incidence rule. Longer terms are checked before shorter
    ones so "fatigue limit" is not swallowed by "fatigue". An unmatched term
    returns UNKNOWN and does not transfer, because the default has to fail
    closed.
    """
    t = term.strip().lower()

    for vocabulary, kind, scope, reason in _RULES:
        for rel in sorted(vocabulary, key=len, reverse=True):
            if rel in t:
                return RelationVerdict(
                    term=term, kind=kind, transfer=scope,
                    reason=f"matches '{rel}'. {reason}",
                )

    return RelationVerdict(
        term=term,
        kind=RelationKind.UNKNOWN,
        transfer=TransferScope.NONE,
        reason=(
            "unrecognized. failing closed: an unclassified relation is treated as "
            "non-transferable until someone states which kind it is."
        ),
    )


def transferable_across_domains(term: str) -> bool:
    """Guard to call before importing a number from one domain's literature."""
    return classify_relation(term).transferable


def retrieve_mechanism_first(terms: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split retrieval targets three ways, per the protocol and the seam.

    Returns:
        (full, relational_only, refused).

        `full` may be used as stated. `relational_only` may contribute an
        ordering, ratio or direction and must NOT contribute a magnitude
        without local calibration. `refused` may not cross a domain at all.
    """
    verdicts = [classify_relation(t) for t in terms]
    return (
        [v.term for v in verdicts if v.transfer is TransferScope.FULL],
        [v.term for v in verdicts if v.transfer is TransferScope.RELATIONAL_ONLY],
        [v.term for v in verdicts if v.transfer is TransferScope.NONE],
    )


# ---------------------------------------------------------------------------
# 2. Boundary-alignment audit
# ---------------------------------------------------------------------------


class SeamKind(str, Enum):
    """What a classification boundary is drawn on.

    The distinction that makes the audit work. Alignment alone does not show
    co-authorship — two systems also align when both track a real joint. Where
    they align is what separates the two.
    """

    PAY_CODE = "pay_code"              # employment status, billing, licensure
    FORCE_GEOMETRY = "force_geometry"  # contact mode, load path, tissue response
    OTHER = "other"


@dataclass(frozen=True)
class Seam:
    """One boundary in a classification system."""

    label: str
    kind: SeamKind
    separates: tuple[str, str]


@dataclass(frozen=True)
class ClassificationSystem:
    """A carve, with its provenance attached.

    `roster` and `charter_date` are the documented provenance the audit needs:
    committee membership and charter/revision dates for SOC/NAICS, ICD,
    specialty boards, discipline formation.
    """

    name: str
    seams: tuple[Seam, ...] = ()
    roster: frozenset[str] = frozenset()
    charter_date: str = ""
    source: str = ""


class CarveVerdict(str, Enum):
    CO_AUTHORED = "co_authored"            # align at pay-code seams
    CONVERGENT = "convergent"              # align at force/geometry seams
    INDEPENDENT = "independent"            # boundaries disagree
    MIXED = "mixed"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class AuditResult:
    system_a: str
    system_b: str
    shared_seams: list[Seam] = field(default_factory=list)
    disagreement_zones: list[Seam] = field(default_factory=list)
    roster_overlap: frozenset[str] = frozenset()
    verdict: CarveVerdict = CarveVerdict.INSUFFICIENT_DATA
    notes: str = ""

    @property
    def shared_pay_seams(self) -> list[Seam]:
        return [s for s in self.shared_seams if s.kind is SeamKind.PAY_CODE]

    @property
    def shared_physical_seams(self) -> list[Seam]:
        return [s for s in self.shared_seams if s.kind is SeamKind.FORCE_GEOMETRY]

    def report(self) -> str:
        lines = [
            f"boundary audit: {self.system_a} vs {self.system_b}",
            f"  shared seams at pay codes      : "
            f"{', '.join(s.label for s in self.shared_pay_seams) or '(none)'}",
            f"  shared seams at force/geometry : "
            f"{', '.join(s.label for s in self.shared_physical_seams) or '(none)'}",
            f"  disagreement zones             : "
            f"{', '.join(s.label for s in self.disagreement_zones) or '(none)'}",
            f"  roster overlap                 : "
            f"{', '.join(sorted(self.roster_overlap)) or '(none)'}",
            "",
            f"verdict: {self.verdict.value}",
        ]
        if self.verdict is CarveVerdict.INDEPENDENT and self.disagreement_zones:
            lines += [
                "",
                "the disagreement zones are the quantity of interest: a region one "
                "system splits and the other does not is where an unclassified "
                "quantity can sit undetected in both.",
            ]
        if self.notes:
            lines += ["", f"note: {self.notes}"]
        return "\n".join(lines)


def boundary_audit(
    a: ClassificationSystem, b: ClassificationSystem
) -> AuditResult:
    """Compare two carves for boundary alignment and shared authorship.

    Alignment at pay-code seams indicates co-authorship — nothing physical sits
    at those seams, so two systems can only agree there by inheritance.
    Alignment at force/geometry seams indicates convergence on a real joint.
    Disagreement makes the disagreement zones the interesting quantity.
    """
    if not a.seams or not b.seams:
        return AuditResult(
            system_a=a.name,
            system_b=b.name,
            notes=(
                "one or both systems have no seams entered. the audit is a "
                "documentation exercise and has not been done: enter seams typed "
                "by kind, plus rosters and charter dates, then re-run."
            ),
        )

    by_sep_a = {s.separates: s for s in a.seams}
    by_sep_b = {s.separates: s for s in b.seams}
    shared_keys = set(by_sep_a) & set(by_sep_b)

    shared = [by_sep_a[k] for k in sorted(shared_keys)]
    disagree = [
        s
        for k, s in sorted({**by_sep_a, **by_sep_b}.items())
        if k not in shared_keys
    ]
    overlap = frozenset(a.roster & b.roster)

    pay = [s for s in shared if s.kind is SeamKind.PAY_CODE]
    phys = [s for s in shared if s.kind is SeamKind.FORCE_GEOMETRY]

    if not shared:
        verdict = CarveVerdict.INDEPENDENT
    elif pay and not phys:
        verdict = CarveVerdict.CO_AUTHORED
    elif phys and not pay:
        verdict = CarveVerdict.CONVERGENT
    else:
        verdict = CarveVerdict.MIXED

    notes = ""
    if verdict is CarveVerdict.CO_AUTHORED and overlap:
        notes = "confirmed twice: pay-code alignment and overlapping rosters."
    elif verdict is CarveVerdict.CO_AUTHORED:
        notes = (
            "pay-code alignment without roster overlap. inheritance is still the "
            "likeliest route, but the roster evidence is missing — find it before "
            "treating this as settled."
        )
    elif verdict is CarveVerdict.CONVERGENT and overlap:
        notes = (
            "aligned at physical seams but rosters overlap. convergence and "
            "inheritance are not separated here; the physical seam may have been "
            "inherited too."
        )

    return AuditResult(
        system_a=a.name,
        system_b=b.name,
        shared_seams=shared,
        disagreement_zones=disagree,
        roster_overlap=overlap,
        verdict=verdict,
        notes=notes,
    )


# Ships empty on purpose. Populating this is the audit.
SYSTEM_REGISTRY: dict[str, ClassificationSystem] = {}

AUDIT_STATUS = "not run: no classification system has been entered"
