"""
Retrieval protocol and boundary-alignment audit.

See `economic-carve.md`.

Two things live here.

1. The retrieval protocol. Some relations are physically carved and transfer
   across domains freely, because no conservation law knows what a job title is.
   Others are economically carved — their strata are pay codes — and transfer is
   invalid by construction. `classify_relation` types a retrieval target so the
   second kind can be refused before it is imported into a multi-domain read.

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
    MECHANISM = "mechanism"      # domain-blind; transfers
    INCIDENCE = "incidence"      # pay-code stratified; does not transfer
    UNKNOWN = "unknown"


# Relations whose statement contains no reference to who was employed doing it.
MECHANISM_RELATIONS: frozenset[str] = frozenset({
    "shear delamination",
    "stiffness mismatch",
    "fatigue",
    "creep",
    "friction-hydration curve",
    "stress concentration at a rigid inclusion",
    "strain rate dependence",
    "hysteresis",
    "contact pressure distribution",
    "tissue remodeling rate",
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


@dataclass(frozen=True)
class RelationVerdict:
    term: str
    kind: RelationKind
    transferable: bool
    reason: str

    def __str__(self) -> str:
        mark = "transfers" if self.transferable else "DOES NOT TRANSFER"
        return f"{self.term}: {self.kind.value} — {mark}. {self.reason}"


def classify_relation(term: str) -> RelationVerdict:
    """Type a retrieval target as mechanism or incidence.

    Substring matching, deliberately loose: "lifetime incidence in welders"
    should trip the incidence rule. An unmatched term returns UNKNOWN and is
    treated as non-transferable, because the default has to fail closed.
    """
    t = term.strip().lower()

    for rel in sorted(INCIDENCE_RELATIONS):
        if rel in t:
            return RelationVerdict(
                term=term,
                kind=RelationKind.INCIDENCE,
                transferable=False,
                reason=(
                    f"matches '{rel}'. denominated in a sampled population whose "
                    "strata are pay codes; transfer across domains is invalid by "
                    "construction."
                ),
            )
    for rel in sorted(MECHANISM_RELATIONS):
        if rel in t:
            return RelationVerdict(
                term=term,
                kind=RelationKind.MECHANISM,
                transferable=True,
                reason=f"matches '{rel}'. domain-blind; no job title appears in it.",
            )
    return RelationVerdict(
        term=term,
        kind=RelationKind.UNKNOWN,
        transferable=False,
        reason=(
            "unrecognized. failing closed: an unclassified relation is treated as "
            "non-transferable until someone states which kind it is."
        ),
    )


def transferable_across_domains(term: str) -> bool:
    """Guard to call before importing a number from one domain's literature."""
    return classify_relation(term).transferable


def retrieve_mechanism_first(terms: list[str]) -> tuple[list[str], list[str]]:
    """Split retrieval targets into what may be used and what may not.

    Returns:
        (usable, refused) — mechanism relations first, per the protocol.
    """
    verdicts = [classify_relation(t) for t in terms]
    return (
        [v.term for v in verdicts if v.transferable],
        [v.term for v in verdicts if not v.transferable],
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
