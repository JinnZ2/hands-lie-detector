"""
The MCP joint as a separate instrument.

See `knuckle-instrument.md`.

Knuckles are not grip surfaces. They are strike, catch, press and hyperextension
surfaces, and a palmar-only instrument cannot read them at all — which is why
five dorsal zones were added after specimen 003, and why this module exists on
top of them.

The critical structural point, and it is the reason this is a separate
instrument rather than an extra column:

    PALMAR LOAD AND DORSAL LOAD ARE NOT CORRELATED.

A hand can carry intense palmar load with almost no knuckle history (precision
work, controlled environment), or intense knuckle history with modest palmar
load (reaching into enclosures, strike-heavy work). Reading grip zones and
inferring the rest gets a strike-heavy operator wrong in a specific direction.

NOT A DIAGNOSTIC INSTRUMENT. Several findings here have clinical differentials —
rheumatoid nodules, Heberden and Bouchard nodes, gouty tophi — that a photograph
cannot exclude. This module reads load history. A knuckle finding that is
painful, progressive, or symmetric without a matching load history is a clinical
question, not a load-history question, and `Differential.requires_clinical_view`
says so.
"""

from dataclasses import dataclass, field
from enum import Enum

from .domains import Zone

STIPULATED = "stipulated from mechanism; not fitted to data"


class Joint(str, Enum):
    """The instrument extends past the MCP row.

    Each joint has its own dominant load mode and its own failure set, because
    each sits at a different point in the kinematic chain.
    """

    MCP = "mcp"
    PIP = "pip"
    DIP = "dip"


@dataclass(frozen=True)
class JointSpec:
    joint: Joint
    typical_load: str
    trauma_pattern: str
    note: str = ""


JOINT_SPECS: dict[Joint, JointSpec] = {
    j.joint: j
    for j in [
        JointSpec(
            Joint.MCP,
            "hyperextension, strike, abrasion",
            "sagittal band rupture ('boxer's knuckle'), collateral sprain, "
            "knuckle pads",
            "the only joint of the three that develops an adaptive pad",
        ),
        JointSpec(
            Joint.PIP,
            "lateral stress, crush between objects, hyperflexion under load",
            "boutonniere deformity (central slip), collateral ligament tears",
            "crush injuries land here: it is the joint that sits between two "
            "surfaces closing on a hand",
        ),
        JointSpec(
            Joint.DIP,
            "pinch trauma, crushing, avulsion",
            "mallet finger (terminal extensor avulsion), jersey finger (FDP "
            "avulsion), nail matrix damage",
            "the nail matrix sits here, so DIP trauma writes into the third "
            "clock — see integration/nail.py",
        ),
    ]
}


class MCPLoadMode(str, Enum):
    """Three distinct failure modes at a joint not designed to be a load surface."""

    DIRECT_IMPACT = "direct_impact"
    HYPEREXTENSION = "hyperextension"
    HYPERFLEXION_LOADED = "hyperflexion_loaded"


@dataclass(frozen=True)
class LoadModeSpec:
    mode: MCPLoadMode
    mechanism: str
    tissue_response: str
    signature_of: str


MCP_LOAD_MODES: dict[MCPLoadMode, LoadModeSpec] = {
    m.mode: m
    for m in [
        LoadModeSpec(
            MCPLoadMode.DIRECT_IMPACT,
            "knuckle contacts a hard surface — strike, or catch against an "
            "enclosure",
            "abrasion, laceration, dorsal scar, knuckle pad formation",
            "mechanics, fabricators, builders, fighters",
        ),
        LoadModeSpec(
            MCPLoadMode.HYPEREXTENSION,
            "finger forced backward beyond neutral",
            "volar plate tear, collateral ligament sprain, joint instability",
            "ball sports, falls, tool kickback, heavy lifting",
        ),
        LoadModeSpec(
            MCPLoadMode.HYPERFLEXION_LOADED,
            "gripping with the MCP flexed and loaded axially",
            "joint compression, synovitis, capsular thickening over time",
            "climbers, heavy tool users, drivers",
        ),
    ]
}


class KnuckleMarker(str, Enum):
    PAD = "knuckle_pad"                  # hyperkeratosis: ADAPTATION
    SCAR = "scar"                        # healed break: EVENT
    DIFFUSE_FULLNESS = "diffuse_fullness"  # capsular, chronic
    INSTABILITY = "instability"          # extensor/volar apparatus, acute origin
    CARBON_STAIN = "carbon_stain"        # bonded during an open-wound phase


# Which markers deposit, and which merely record.
ADAPTIVE_MARKERS: frozenset[KnuckleMarker] = frozenset({
    KnuckleMarker.PAD, KnuckleMarker.DIFFUSE_FULLNESS,
})
EVENT_MARKERS: frozenset[KnuckleMarker] = frozenset({
    KnuckleMarker.SCAR, KnuckleMarker.INSTABILITY, KnuckleMarker.CARBON_STAIN,
})


MARKER_READING: dict[KnuckleMarker, str] = {
    KnuckleMarker.PAD:
        "localized hyperkeratosis over the joint. soft tissue, not bony, and it "
        "moves with the skin. the skin thickened because the joint is loaded "
        "DORSALLY — pressed, scraped, or struck repeatedly.",
    KnuckleMarker.SCAR:
        "the dorsum was broken against something sharp, rough or hot. an event, "
        "on the healing clock, not a load history.",
    KnuckleMarker.DIFFUSE_FULLNESS:
        "capsular thickening without a discrete pad. chronic high-force grip with "
        "the MCP loaded in flexion. hard to separate from soft-tissue swelling "
        "without palpation — a vision instrument may not resolve it.",
    KnuckleMarker.INSTABILITY:
        "extensor or volar apparatus disrupted by sudden force. the finger may "
        "not extend cleanly. acute in origin even when the finding is old.",
    KnuckleMarker.CARBON_STAIN:
        "the wound was open in a carbon-rich environment and carbon bonded to "
        "healing tissue. not dirt, and it does not wash off. a permanent "
        "load-history marker with a datable origin.",
}


class Differential(str, Enum):
    """Non-load causes that a photograph cannot exclude."""

    RHEUMATOID_NODULE = "rheumatoid_nodule"
    HEBERDEN_BOUCHARD_NODE = "heberden_or_bouchard_node"
    GOUTY_TOPHUS = "gouty_tophus"

    @property
    def requires_clinical_view(self) -> bool:
        """Always True. This module reads load, not disease."""
        return True


# What separates a load marker from the differentials above, and it is a
# palpation finding rather than an image finding.
PAD_VERSUS_NODE = (
    "a knuckle pad is SOFT TISSUE: it moves with the skin and has no bony "
    "component. Heberden and Bouchard nodes are bony and do not move. that "
    "distinction is made by touch, not by photograph — so a vision instrument "
    "should report the finding and decline the differential."
)


@dataclass(frozen=True)
class KnuckleFinding:
    digit: str
    zone: Zone
    marker: KnuckleMarker
    bilateral: bool = False
    stage: str = ""  # fresh / healing / healed / scarred
    joint: Joint = Joint.MCP

    @property
    def writes_to_nail_clock(self) -> bool:
        """DIP trauma reaches the nail matrix, so it dates itself."""
        return self.joint is Joint.DIP

    @property
    def deposits(self) -> bool:
        return self.marker in ADAPTIVE_MARKERS

    def __str__(self) -> str:
        side = "bilateral" if self.bilateral else "unilateral"
        stage = f", {self.stage}" if self.stage else ""
        return f"{self.digit} {self.zone.value}: {self.marker.value} ({side}{stage})"


def scar_mechanism(zone: Zone) -> str:
    """Scar LOCATION distinguishes the posture the hand was in when struck."""
    if zone is Zone.DORSAL_MCP_KNUCKLES:
        return (
            "MCP scar: the joint was FLEXED when struck — hand forward, working "
            "in a tight space, knuckles leading."
        )
    if zone is Zone.DORSAL_METACARPAL:
        return (
            "metacarpal shaft scar: the dorsum was FLAT against a surface, or "
            "dragged across an edge."
        )
    return "location does not distinguish a posture on its own."


# Falsifiable claims, in the same form as `sole.CATEGORY_PREDICTIONS`: a marker
# pattern predicts a work characteristic, and a carrier can refute it.
KNUCKLE_WORK_PREDICTIONS: tuple[tuple[str, str], ...] = (
    ("pads on 2nd-4th MCP, bilateral",
     "repeated pressing or scraping against flat surfaces — carpet laying, "
     "tailoring, machining, grinding"),
    ("pad on the thumb MCP",
     "heavy pinching, tool use, wire work"),
    ("scars on MCP dorsum, varied digits",
     "reaching into enclosures with sharp edges — engine bays, machinery, stock"),
    ("carbon stain in a dorsal scar",
     "hot work or engine work: soot exposure during the open-wound phase"),
    ("hyperextension history",
     "falls, tool kickback, jamming injuries, heavy lifting with the hand forward"),
    ("extensor apparatus disruption",
     "direct impact to the knuckle — striking, or hammering without a guard"),
    ("diffuse MCP fullness, no discrete pad",
     "chronic heavy grip — climbing, heavy tool use, pulling"),
)


@dataclass
class KnuckleReadout:
    carrier: str = ""
    findings: list[KnuckleFinding] = field(default_factory=list)
    provenance: str = STIPULATED

    @property
    def is_evidence_based(self) -> bool:
        return self.provenance != STIPULATED

    @property
    def deposits_present(self) -> bool:
        return any(f.deposits for f in self.findings)

    @property
    def events_present(self) -> bool:
        return any(not f.deposits for f in self.findings)

    @property
    def predicts_palmar_load(self) -> bool:
        """Always False, and this is the structural claim.

        Dorsal and palmar load are not correlated. Neither surface's reading
        licenses an inference about the other, in either direction.
        """
        return False

    @property
    def joints_involved(self) -> set[Joint]:
        return {f.joint for f in self.findings}

    @property
    def corroborated_by_nail_clock(self) -> bool:
        """True when a DIP finding is present — the nail record should agree."""
        return Joint.DIP in self.joints_involved

    def report(self) -> str:
        lines = [f"knuckle readout{f' ({self.carrier})' if self.carrier else ''}"]
        lines += [f"  {f}" for f in self.findings] or ["  (no findings)"]
        lines += [
            "",
            f"  deposits (adaptation) : {'present' if self.deposits_present else 'none'}",
            f"  events (healed marks) : {'present' if self.events_present else 'none'}",
        ]
        scar_zones = {f.zone for f in self.findings if f.marker is KnuckleMarker.SCAR}
        if scar_zones:
            lines += ["", "  scar posture:"]
            lines += [f"    {scar_mechanism(z)}" for z in sorted(scar_zones, key=lambda z: z.value)]
        if self.joints_involved:
            lines += ["", "  joints: " + ", ".join(
                sorted(j.value for j in self.joints_involved))]
        if self.corroborated_by_nail_clock:
            lines.append(
                "    DIP involvement reaches the nail matrix — cross-check "
                "integration/nail.py, which dates its own marks."
            )
        lines += [
            "",
            "  this readout licenses NO inference about palmar load. the two "
            "surfaces are loaded by different mechanisms and are not correlated.",
            f"  differential: {PAD_VERSUS_NODE}",
        ]
        if not self.is_evidence_based:
            lines.append(f"  note: {self.provenance}")
        return "\n".join(lines)
