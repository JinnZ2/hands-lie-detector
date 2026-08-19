"""
Domain signatures for multi-domain integration readout.

Background: see `reference-class-empty.md`.

Every table in this module is STIPULATED FROM MECHANISM, NOT FITTED TO DATA.
That is the whole point of the surrounding argument: the reference class for
multi-domain load is empty, so there is nothing to fit against yet. These
tables are a falsifiable starting guess, carried with an explicit provenance
flag so they cannot quietly harden into evidence. They are wrong until
someone falsifies them.
"""

from dataclasses import dataclass, field
from enum import Enum


class Surface(str, Enum):
    PALMAR = "palmar"
    DORSAL = "dorsal"


class Zone(str, Enum):
    """Hand zones.

    The first nine match `CallusZone` in term_audit/vocabulary/ and are all
    PALMAR. That vocabulary was built for grip, which is where load is carried —
    and it therefore cannot record a marker on the back of the hand at all.

    The dorsal zones below were added after photographs showed lacerations over
    the metacarpals and MCP joints, on a hand whose palmar surface was
    unremarkable. Half the evidence had no vocabulary to land in.

    Note what follows: no shipped `DomainSignature` predicts a dorsal zone, so
    every dorsal marker is residual by construction against every enrolled
    domain. That is not a bug in the residual computation. It is the shape of an
    instrument built for grip being handed a strike.
    """

    # Palmar — grip surface.
    THUMB_CROTCH = "thumb_crotch"
    INDEX_SIDE = "index_side"
    PALM_BELOW_INDEX = "palm_below_index"
    HEEL_OF_PALM = "heel_of_palm"
    FINGERTIP_PADS = "fingertip_pads"
    ACROSS_PALM_CREASE = "across_palm_crease"
    BASE_OF_FINGERS = "base_of_fingers"
    THUMB_PAD = "thumb_pad"
    OUTER_PALM_EDGE = "outer_palm_edge"

    # Dorsal — not a grip surface. Markers here come from strike, catch,
    # abrasion against an enclosure, or cold exposure, not from holding.
    DORSAL_METACARPAL = "dorsal_metacarpal"
    DORSAL_MCP_KNUCKLES = "dorsal_mcp_knuckles"
    DORSAL_WEB_SPACE = "dorsal_web_space"
    DORSAL_PHALANX = "dorsal_phalanx"
    WRIST_TRANSITION = "wrist_transition"

    @property
    def surface(self) -> Surface:
        return (
            Surface.DORSAL
            if self.value.startswith("dorsal_") or self is Zone.WRIST_TRANSITION
            else Surface.PALMAR
        )


PALMAR_ZONES: frozenset[Zone] = frozenset(
    z for z in Zone if z.surface is Surface.PALMAR
)
DORSAL_ZONES: frozenset[Zone] = frozenset(
    z for z in Zone if z.surface is Surface.DORSAL
)


class LoadMode(str, Enum):
    """How a zone is loaded. Two domains loading one zone in different modes
    is the geometry mismatch that Test C looks for."""

    COMPRESSION = "compression"   # static grip pressure
    SHEAR = "shear"               # sliding friction across the surface
    TORSION = "torsion"           # rotational load through the grip
    VIBRATION = "vibration"       # oscillating input from powered tools
    IMPACT = "impact"             # repeated strike loading
    ABRASION = "abrasion"         # material removal against the skin
    MACERATION = "maceration"     # prolonged wet exposure, softening


class Channel(str, Enum):
    """Candidate decomposition channels for integration (H2 in the analysis).

    If integration is one quantity (H1), these are interchangeable routes to
    loading a single scalar. If it is a family (H2), each has its own
    instrument and its own empty reference class.
    """

    CONTACT_GEOMETRY = "contact_geometry"
    TIMING = "timing"
    RECOVERY_BUDGET = "recovery_budget"
    ATTENTION = "attention"


# Zones that sit adjacent to each other on the hand. Used to predict where an
# off-site marker lands when two domains conflict at a shared zone: the tissue
# recruits the neighbour rather than the contested site.
ADJACENCY: dict[Zone, set[Zone]] = {
    Zone.THUMB_CROTCH: {Zone.THUMB_PAD, Zone.PALM_BELOW_INDEX, Zone.INDEX_SIDE},
    Zone.INDEX_SIDE: {Zone.THUMB_CROTCH, Zone.PALM_BELOW_INDEX, Zone.FINGERTIP_PADS},
    Zone.PALM_BELOW_INDEX: {Zone.THUMB_CROTCH, Zone.INDEX_SIDE, Zone.BASE_OF_FINGERS, Zone.ACROSS_PALM_CREASE},
    Zone.HEEL_OF_PALM: {Zone.ACROSS_PALM_CREASE, Zone.OUTER_PALM_EDGE, Zone.THUMB_CROTCH},
    Zone.FINGERTIP_PADS: {Zone.BASE_OF_FINGERS, Zone.INDEX_SIDE, Zone.THUMB_PAD},
    Zone.ACROSS_PALM_CREASE: {Zone.BASE_OF_FINGERS, Zone.HEEL_OF_PALM, Zone.PALM_BELOW_INDEX},
    Zone.BASE_OF_FINGERS: {Zone.ACROSS_PALM_CREASE, Zone.FINGERTIP_PADS, Zone.PALM_BELOW_INDEX},
    Zone.THUMB_PAD: {Zone.THUMB_CROTCH, Zone.FINGERTIP_PADS},
    Zone.OUTER_PALM_EDGE: {Zone.HEEL_OF_PALM, Zone.BASE_OF_FINGERS},
    # Dorsal adjacency is its own graph. It touches the palmar graph only at the
    # hand's edges, which is why a dorsal marker does not get "explained" by a
    # neighbouring grip zone.
    Zone.DORSAL_METACARPAL: {Zone.DORSAL_MCP_KNUCKLES, Zone.WRIST_TRANSITION},
    Zone.DORSAL_MCP_KNUCKLES: {Zone.DORSAL_METACARPAL, Zone.DORSAL_WEB_SPACE,
                               Zone.DORSAL_PHALANX},
    Zone.DORSAL_WEB_SPACE: {Zone.DORSAL_MCP_KNUCKLES, Zone.DORSAL_PHALANX},
    Zone.DORSAL_PHALANX: {Zone.DORSAL_MCP_KNUCKLES, Zone.DORSAL_WEB_SPACE},
    Zone.WRIST_TRANSITION: {Zone.DORSAL_METACARPAL, Zone.HEEL_OF_PALM},
}

# Where load goes when it exceeds any single domain's geometry, independent of
# which domains are combined. Under H1 the whole residual lands here.
GENERIC_OVERFLOW: frozenset[Zone] = frozenset(
    {Zone.HEEL_OF_PALM, Zone.OUTER_PALM_EDGE, Zone.ACROSS_PALM_CREASE}
)

STIPULATED = "stipulated from mechanism; not fitted to data; falsifiable"


@dataclass(frozen=True)
class DomainSignature:
    """The marker signature a single domain is expected to leave on its own.

    `zone_modes` is the load-bearing part: it says not just *where* a domain
    marks but *how*, which is what makes pairwise conflict computable.
    """

    name: str
    zone_modes: dict[Zone, LoadMode]
    channel_demand: dict[Channel, float] = field(default_factory=dict)
    provenance: str = STIPULATED
    notes: str = ""

    @property
    def zones(self) -> frozenset[Zone]:
        return frozenset(self.zone_modes)

    @property
    def is_evidence_based(self) -> bool:
        """False for every default in this module, deliberately."""
        return self.provenance != STIPULATED


def _sig(name: str, zone_modes: dict[Zone, LoadMode], **demand: float) -> DomainSignature:
    return DomainSignature(
        name=name,
        zone_modes=zone_modes,
        channel_demand={Channel(k): v for k, v in demand.items()},
    )


DEFAULT_DOMAINS: dict[str, DomainSignature] = {
    d.name: d
    for d in [
        _sig(
            "rotary_hand_tool",
            {
                Zone.THUMB_CROTCH: LoadMode.TORSION,
                Zone.PALM_BELOW_INDEX: LoadMode.VIBRATION,
                Zone.BASE_OF_FINGERS: LoadMode.COMPRESSION,
            },
            contact_geometry=0.8, timing=0.4, recovery_budget=0.5, attention=0.6,
        ),
        _sig(
            "wet_task",
            {
                Zone.FINGERTIP_PADS: LoadMode.MACERATION,
                Zone.THUMB_PAD: LoadMode.MACERATION,
                Zone.INDEX_SIDE: LoadMode.SHEAR,
            },
            contact_geometry=0.3, timing=0.6, recovery_budget=0.8, attention=0.2,
        ),
        _sig(
            "wrench_turning",
            {
                Zone.PALM_BELOW_INDEX: LoadMode.TORSION,
                Zone.BASE_OF_FINGERS: LoadMode.COMPRESSION,
                Zone.THUMB_CROTCH: LoadMode.SHEAR,
            },
            contact_geometry=0.7, timing=0.3, recovery_budget=0.4, attention=0.4,
        ),
        _sig(
            "shovel_haul",
            {
                Zone.ACROSS_PALM_CREASE: LoadMode.SHEAR,
                Zone.BASE_OF_FINGERS: LoadMode.COMPRESSION,
                Zone.HEEL_OF_PALM: LoadMode.COMPRESSION,
            },
            contact_geometry=0.5, timing=0.5, recovery_budget=0.9, attention=0.1,
        ),
        _sig(
            "livestock_handling",
            {
                Zone.ACROSS_PALM_CREASE: LoadMode.SHEAR,
                Zone.FINGERTIP_PADS: LoadMode.IMPACT,
                Zone.OUTER_PALM_EDGE: LoadMode.IMPACT,
            },
            contact_geometry=0.4, timing=0.9, recovery_budget=0.7, attention=0.9,
        ),
        _sig(
            "fine_manipulation",
            {
                Zone.FINGERTIP_PADS: LoadMode.COMPRESSION,
                Zone.THUMB_PAD: LoadMode.COMPRESSION,
                Zone.INDEX_SIDE: LoadMode.ABRASION,
            },
            contact_geometry=0.6, timing=0.2, recovery_budget=0.2, attention=0.9,
        ),
        _sig(
            "rope_line_handling",
            {
                Zone.ACROSS_PALM_CREASE: LoadMode.ABRASION,
                Zone.BASE_OF_FINGERS: LoadMode.ABRASION,
                Zone.THUMB_CROTCH: LoadMode.COMPRESSION,
            },
            contact_geometry=0.5, timing=0.7, recovery_budget=0.6, attention=0.5,
        ),
        _sig(
            "keyboard",
            {Zone.FINGERTIP_PADS: LoadMode.IMPACT, Zone.HEEL_OF_PALM: LoadMode.COMPRESSION},
            contact_geometry=0.1, timing=0.1, recovery_budget=0.1, attention=0.3,
        ),
    ]
}


def get_domain(name: str) -> DomainSignature | None:
    return DEFAULT_DOMAINS.get(name)


def domain_names() -> list[str]:
    return sorted(DEFAULT_DOMAINS)


# ---------------------------------------------------------------------------
# Systemic modes
# ---------------------------------------------------------------------------
#
# PROVENANCE WARNING, stronger than the one on the tables above.
#
# The zone-collision mechanism (two domains loading one shared zone in
# different modes) produces NO conflict for the (rotary_hand_tool, wet_task)
# pair, because under the stipulated table those two domains share no zone.
# That pair is the one the pass-3 observation in `reference-class-empty.md`
# reported a geometry mismatch for. So the table, as stated, disagrees with the
# one observation on record.
#
# The mechanism below was added AFTER seeing that observation, to represent a
# domain whose effect is on tissue state rather than on a site: wet-softened
# skin tears under shear where dry skin would callus, and it does so wherever
# the shear lands, not only where the water did.
#
# Because it was added post-hoc to fit that observation, it has NO confirmatory
# value for it. It is a hypothesis generated by one data point, and it needs an
# independent test before it counts as anything. Left visible on purpose: the
# document this module accompanies is about reference classes built by
# assertion, and this is where this module is most at risk of building one.

SYSTEMIC_MODES: frozenset[LoadMode] = frozenset({LoadMode.MACERATION})

# Modes whose tissue response is altered when a systemic mode is co-present.
SYSTEMIC_INCOMPATIBLE: frozenset[LoadMode] = frozenset(
    {LoadMode.SHEAR, LoadMode.ABRASION, LoadMode.TORSION, LoadMode.VIBRATION}
)

POST_HOC = "post-hoc from a single observation; no confirmatory value; needs independent test"
