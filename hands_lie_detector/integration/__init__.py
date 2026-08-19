"""
Multi-domain integration readout.

Companion to `reference-class-empty.md`. Where `hands_lie_detector.scoring`
sums seven categories into one additive total — a model form in which no two
categories can interact — this package keeps the non-additive part as its own
reported quantity.

The convention: unexplained residual defaults to integration and is never
distributed across the enrolled domains in proportion to their legibility.
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
from .domains import (
    DEFAULT_DOMAINS,
    Channel,
    DomainSignature,
    LoadMode,
    Zone,
    domain_names,
    get_domain,
)
from .residual import (
    GeometryConflict,
    IntegrationReadout,
    geometry_conflicts,
    read_hand,
)

__all__ = [
    "Channel",
    "DEFAULT_DOMAINS",
    "DissociationResult",
    "DomainSignature",
    "GeometryConflict",
    "IntegrationReadout",
    "LoadMode",
    "M_GEOM",
    "M_TIME",
    "Manipulation",
    "Verdict",
    "Zone",
    "domain_names",
    "double_dissociation",
    "geometry_conflicts",
    "get_domain",
    "moved_markers",
    "read_hand",
]
