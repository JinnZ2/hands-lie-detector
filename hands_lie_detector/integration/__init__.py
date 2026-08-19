"""
Multi-domain integration readout.

Companion to `reference-class-empty.md`. Where `hands_lie_detector.scoring`
sums seven categories into one additive total — a model form in which no two
categories can interact — this package keeps the non-additive part as its own
reported quantity.

Two conventions, from the two documents:

- unexplained residual defaults to integration and is never distributed across
  the enrolled domains in proportion to their legibility
  (`reference-class-empty.md`)
- one body, one load history is the null; the partition must be argued for, and
  load is weighted by mechanism with no payment term (`economic-carve.md`)
"""

from .dissociation import (
    M_GEOM,
    M_TIME,
    DissociationResult,
    Manipulation,
    Verdict,
    double_dissociation,
    moved_markers,
)
from .carve_audit import (
    AUDIT_STATUS,
    INCIDENCE_RELATIONS,
    MECHANISM_RELATIONS,
    SYSTEM_REGISTRY,
    AuditResult,
    CarveVerdict,
    ClassificationSystem,
    LIVING_TISSUE_PARAMS,
    RelationKind,
    RelationVerdict,
    Seam,
    SeamKind,
    TransferScope,
    boundary_audit,
    classify_relation,
    retrieve_mechanism_first,
    transferable_across_domains,
)
from .domains import (
    DEFAULT_DOMAINS,
    Channel,
    DomainSignature,
    LoadMode,
    Zone,
    domain_names,
    get_domain,
)
from .load_weight import (
    NO_PAYMENT_TERM,
    DiscontinuityResult,
    LoadBlock,
    compare_readouts,
    discontinuity,
    ledger_share,
    load_share,
    relabel,
)
from .strip import (
    DEFAULT_RENDERINGS,
    MECHANICAL_UNITS,
    Rendering,
    StripResult,
    StripVerdict,
    strip,
    strip_all,
)
from .partition import (
    LoadHistory,
    PartitionClaim,
    PartitionVerdict,
    propose_partition,
)
from .residual import (
    GeometryConflict,
    IntegrationReadout,
    geometry_conflicts,
    read_hand,
)

__all__ = [
    "AUDIT_STATUS",
    "AuditResult",
    "CarveVerdict",
    "Channel",
    "ClassificationSystem",
    "DEFAULT_DOMAINS",
    "DissociationResult",
    "DiscontinuityResult",
    "DomainSignature",
    "GeometryConflict",
    "DEFAULT_RENDERINGS",
    "INCIDENCE_RELATIONS",
    "LIVING_TISSUE_PARAMS",
    "MECHANICAL_UNITS",
    "IntegrationReadout",
    "LoadBlock",
    "LoadHistory",
    "LoadMode",
    "MECHANISM_RELATIONS",
    "M_GEOM",
    "M_TIME",
    "Manipulation",
    "NO_PAYMENT_TERM",
    "PartitionClaim",
    "PartitionVerdict",
    "RelationKind",
    "RelationVerdict",
    "Rendering",
    "StripResult",
    "StripVerdict",
    "TransferScope",
    "SYSTEM_REGISTRY",
    "Seam",
    "SeamKind",
    "Verdict",
    "Zone",
    "boundary_audit",
    "classify_relation",
    "compare_readouts",
    "discontinuity",
    "domain_names",
    "double_dissociation",
    "geometry_conflicts",
    "get_domain",
    "ledger_share",
    "load_share",
    "moved_markers",
    "propose_partition",
    "read_hand",
    "relabel",
    "retrieve_mechanism_first",
    "strip",
    "strip_all",
    "transferable_across_domains",
]
