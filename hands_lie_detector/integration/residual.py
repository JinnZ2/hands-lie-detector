"""
Residual-zone readout: Test C from `reference-class-empty.md`.

The convention this module exists to enforce:

    Unexplained residual defaults to integration. It is never distributed
    across the enrolled domains in proportion to their legibility.

That distribution step is the move that makes an empty reference class look
full. Here the unattributed set stays in the output, named, with its zones
listed, whether or not anything can be said about it.
"""

from dataclasses import dataclass, field
from itertools import combinations

from .domains import (
    ADJACENCY,
    DEFAULT_DOMAINS,
    GENERIC_OVERFLOW,
    POST_HOC,
    SYSTEMIC_INCOMPATIBLE,
    SYSTEMIC_MODES,
    DomainSignature,
    LoadMode,
    Zone,
)


@dataclass(frozen=True)
class GeometryConflict:
    """One zone that two enrolled domains load in incompatible modes."""

    zone: Zone
    domain_a: str
    mode_a: LoadMode
    domain_b: str
    mode_b: LoadMode
    systemic: bool = False
    provenance: str = ""

    @property
    def predicted_offsite(self) -> frozenset[Zone]:
        """Where the marker is expected to land instead of the contested zone."""
        return frozenset(ADJACENCY.get(self.zone, set()))

    @property
    def is_evidence_based(self) -> bool:
        """False for systemic conflicts, which are post-hoc. Deliberately."""
        return not self.systemic

    def __str__(self) -> str:
        kind = "systemic" if self.systemic else "shared-zone"
        return (
            f"[{kind}] {self.zone.value}: {self.domain_a}/{self.mode_a.value} "
            f"vs {self.domain_b}/{self.mode_b.value}"
        )


@dataclass
class IntegrationReadout:
    """Result of reading one hand against a set of enrolled domains."""

    enrolled: list[str]
    observed: frozenset[Zone]
    predicted: frozenset[Zone]
    residual_zones: frozenset[Zone] = frozenset()
    missing_zones: frozenset[Zone] = frozenset()
    conflicts: list[GeometryConflict] = field(default_factory=list)
    provenance_warning: str = ""

    @property
    def has_residual(self) -> bool:
        return bool(self.residual_zones)

    @property
    def pair_specific_residual(self) -> frozenset[Zone]:
        """Residual zones that a specific pairwise conflict predicts.

        Evidence for H2 (integration decomposes): the residual is *located*,
        and located by which two domains collide.
        """
        predicted_offsite: set[Zone] = set()
        for c in self.conflicts:
            predicted_offsite |= c.predicted_offsite
        return frozenset(self.residual_zones & predicted_offsite)

    @property
    def generic_residual(self) -> frozenset[Zone]:
        """Residual zones at sites any excess load would reach.

        Evidence for H1 (one quantity): the residual is present but generic.
        """
        return frozenset(self.residual_zones & GENERIC_OVERFLOW)

    @property
    def unexplained_residual(self) -> frozenset[Zone]:
        """Residual that neither hypothesis predicts. Reported, not absorbed."""
        return frozenset(
            self.residual_zones - self.pair_specific_residual - self.generic_residual
        )

    def leaning(self) -> str:
        """Direction of evidence from this single hand. Not a verdict.

        One hand cannot settle H1 vs H2 — domain load is not randomized and the
        carrier chose their own domains. This reports which way the geography
        points and nothing more.
        """
        if not self.has_residual:
            return "no residual: additive model not yet challenged by this hand"
        pair, generic = len(self.pair_specific_residual), len(self.generic_residual)
        if pair and not generic:
            return "leans H2: residual is located at pairwise conflict sites"
        if generic and not pair:
            return "leans H1: residual is present but generic"
        if pair and generic:
            return "mixed: both located and generic residual present"
        return "residual at sites neither hypothesis predicts; channel list may be wrong"

    def report(self) -> str:
        lines = [
            f"enrolled domains : {', '.join(self.enrolled) or '(none)'}",
            f"observed zones   : {_z(self.observed)}",
            f"predicted zones  : {_z(self.predicted)}",
            "",
            f"RESIDUAL (integration, unattributed): {_z(self.residual_zones) or '(none)'}",
            f"  pair-specific  : {_z(self.pair_specific_residual) or '(none)'}",
            f"  generic        : {_z(self.generic_residual) or '(none)'}",
            f"  unexplained    : {_z(self.unexplained_residual) or '(none)'}",
            "",
            f"predicted-but-absent: {_z(self.missing_zones) or '(none)'}",
        ]
        if self.conflicts:
            lines += ["", "geometry conflicts between enrolled domains:"]
            lines += [f"  {c}  -> off-site: {_z(c.predicted_offsite)}" for c in self.conflicts]
            if any(c.systemic for c in self.conflicts):
                lines += [f"  (systemic conflicts are {POST_HOC})"]
        lines += ["", f"leaning: {self.leaning()}"]
        if self.provenance_warning:
            lines += ["", f"NOTE: {self.provenance_warning}"]
        return "\n".join(lines)


def _z(zones) -> str:
    return ", ".join(sorted(z.value for z in zones))


def geometry_conflicts(domains: list[DomainSignature]) -> list[GeometryConflict]:
    """Zones that two enrolled domains load in different modes.

    This is the quantity the between-subject design deletes: it exists only in
    the pair, so it has no representation in a cohort where each subject is
    coded to one domain.
    """
    out: list[GeometryConflict] = []
    for a, b in combinations(domains, 2):
        for zone in a.zones & b.zones:
            if a.zone_modes[zone] != b.zone_modes[zone]:
                out.append(
                    GeometryConflict(
                        zone=zone,
                        domain_a=a.name,
                        mode_a=a.zone_modes[zone],
                        domain_b=b.name,
                        mode_b=b.zone_modes[zone],
                    )
                )
        out.extend(_systemic_conflicts(a, b))
        out.extend(_systemic_conflicts(b, a))
    return sorted(out, key=lambda c: (c.zone.value, c.domain_a, c.domain_b))


def _systemic_conflicts(
    carrier: DomainSignature, loaded: DomainSignature
) -> list[GeometryConflict]:
    """Conflicts where `carrier` changes tissue state and `loaded` then acts on it.

    See the provenance warning in domains.py: this mechanism was added post-hoc
    from one observation and carries no confirmatory value for it.
    """
    systemic = {m for m in carrier.zone_modes.values() if m in SYSTEMIC_MODES}
    if not systemic:
        return []
    mode_c = sorted(systemic, key=lambda m: m.value)[0]
    return [
        GeometryConflict(
            zone=zone,
            domain_a=carrier.name,
            mode_a=mode_c,
            domain_b=loaded.name,
            mode_b=mode_l,
            systemic=True,
            provenance=POST_HOC,
        )
        for zone, mode_l in sorted(loaded.zone_modes.items(), key=lambda kv: kv[0].value)
        if mode_l in SYSTEMIC_INCOMPATIBLE
    ]


def read_hand(
    observed_zones: list[Zone | str],
    enrolled_domains: list[str | DomainSignature],
    registry: dict[str, DomainSignature] | None = None,
) -> IntegrationReadout:
    """Compute the integration readout for one hand.

    Args:
        observed_zones: zones where markers are actually present.
        enrolled_domains: the domains the carrier is known to load. "Enrolled"
            is deliberate — this is the same partition a cohort would impose,
            made explicit so its residual can be measured instead of absorbed.
        registry: optional override for the stipulated default signatures.

    Returns:
        IntegrationReadout. The residual is reported as its own quantity and is
        never redistributed across `enrolled_domains`.
    """
    registry = registry or DEFAULT_DOMAINS

    sigs: list[DomainSignature] = []
    unknown: list[str] = []
    for d in enrolled_domains:
        if isinstance(d, DomainSignature):
            sigs.append(d)
        elif d in registry:
            sigs.append(registry[d])
        else:
            unknown.append(d)
    if unknown:
        raise KeyError(f"unknown domain(s): {sorted(unknown)}; known: {sorted(registry)}")

    observed = frozenset(Zone(z) if isinstance(z, str) else z for z in observed_zones)
    predicted = frozenset().union(*(s.zones for s in sigs)) if sigs else frozenset()

    stipulated = [s.name for s in sigs if not s.is_evidence_based]
    warning = ""
    if stipulated:
        warning = (
            "signatures are stipulated from mechanism, not fitted to data "
            f"({', '.join(stipulated)}). Treat as falsifiable priors, not findings."
        )

    return IntegrationReadout(
        enrolled=[s.name for s in sigs],
        observed=observed,
        predicted=predicted,
        residual_zones=frozenset(observed - predicted),
        missing_zones=frozenset(predicted - observed),
        conflicts=geometry_conflicts(sigs),
        provenance_warning=warning,
    )
