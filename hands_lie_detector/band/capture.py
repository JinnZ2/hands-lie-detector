"""
Capture protocol: the five positions, and the light that makes tier 2 possible.

See `capture-protocol.md`.

The repo's largest blocker is not analysis. It is that `ThicknessReading.
measurable` returns False for every frame on record, because thickness and
boundary sharpness need raking light and no frame has it. This module specifies
what would fix that, in a form that can be checked rather than remembered.

The trigger matters as much as the geometry. Capture gated on noticing produces
the anomaly and hides the band — see the narrative gate in `economic-carve.md`.
Bind it to something already recurring instead.
"""

from dataclasses import dataclass, field
from enum import Enum

from .contrast import LightCondition


class Position(str, Enum):
    PALM_FLAT = "1_palm_flat"
    FINGERTIPS = "2_fingertips"
    THENAR = "3_thenar"
    DORSAL = "4_dorsal"
    LATERAL_RAKING = "5_lateral_raking"


@dataclass(frozen=True)
class PositionSpec:
    position: Position
    hand: str
    camera: str
    light: str
    resolves: tuple[str, ...]


POSITIONS: dict[Position, PositionSpec] = {
    p.position: p
    for p in [
        PositionSpec(
            Position.PALM_FLAT,
            hand="palm up, flat as a table; fingers spread moderately — not "
                 "stretched, not closed; thumb abducted ~45 degrees",
            camera="directly above, 12-18 inches",
            light="single source, side, skimming",
            resolves=("overall crease map", "zone coverage", "gross girth"),
        ),
        PositionSpec(
            Position.FINGERTIPS,
            hand="relaxed, palm up, fingers slightly curled as if holding a "
                 "softball",
            camera="close, 6-8 inches, focused on the pads",
            light="from the side, skimming across the pads",
            resolves=("fingertip pad texture", "proximal phalanx pad markers"),
        ),
        PositionSpec(
            Position.THENAR,
            hand="thumb extended, palm rotated ~30 degrees toward camera, thenar "
                 "eminence facing the light",
            camera="from the thumb side, close",
            light="raking across the thenar mass",
            resolves=("thenar volume", "thumb crotch", "thumb pad"),
        ),
        PositionSpec(
            Position.DORSAL,
            hand="back of hand up, flat; fingers together or slightly spread",
            camera="directly above",
            light="from the side, same angle as position 1",
            resolves=("dorsal event marks", "vein and tendon relief", "MCP state"),
        ),
        PositionSpec(
            Position.LATERAL_RAKING,
            hand="flat, palm up",
            camera="from the wrist side, ~30 degrees off vertical",
            light="from the opposite side — camera at 6 o'clock, light at 12",
            resolves=("THICKNESS MAP", "boundary sharpness", "hypothenar relief"),
        ),
    ]
}

# The position that unblocks tier 2. Without it the state readout stays
# ambiguous no matter how many other frames exist.
TIER_2_POSITION = Position.LATERAL_RAKING


class Trigger(str, Enum):
    """Bind capture to something already recurring. No judgment call in the loop."""

    FUEL_STOP = "fuel_stop"
    ODOMETER_ROUND = "odometer_crossing_a_round_number"
    WEEK_ROLLOVER = "week_rollover"
    LOG_ENTRY = "log_entry"
    NOTICED_SOMETHING = "noticed_something"  # the failure mode, named

    @property
    def is_scheduled(self) -> bool:
        return self is not Trigger.NOTICED_SOMETHING


@dataclass
class CaptureSession:
    """One session, checked against the protocol."""

    date: str
    trigger: Trigger
    positions_shot: set[Position] = field(default_factory=set)
    both_hands: bool = False
    light: LightCondition = LightCondition.UNKNOWN
    load_log: str = ""
    paired_state: str = ""  # e.g. "post-wash", "pre-wash"

    @property
    def missing_positions(self) -> list[Position]:
        return [p for p in Position if p not in self.positions_shot]

    @property
    def resolves_tier_2(self) -> bool:
        return (
            TIER_2_POSITION in self.positions_shot
            and self.light.resolves_thickness
        )

    @property
    def problems(self) -> list[str]:
        out = []
        if not self.resolves_tier_2:
            out.append(
                f"tier 2 unresolved: needs {TIER_2_POSITION.value} shot in raking "
                "light. without it the thickness map is not measurable and the "
                "state readout stays ambiguous."
            )
        if self.missing_positions:
            out.append(
                "incomplete position set: missing "
                + ", ".join(p.value for p in self.missing_positions)
            )
        if not self.both_hands:
            out.append(
                "one hand only. bilateral comparison is its own control — "
                "asymmetric load distribution is readable only across the pair."
            )
        if not self.trigger.is_scheduled:
            out.append(
                "disposition-gated capture. this samples the anomaly and hides "
                "the maintained baseline; bind the trigger to something recurring."
            )
        if not self.load_log.strip():
            out.append(
                "no dated load log. the map integrates a load history — without "
                "the history the image is uninterpretable."
            )
        return out

    @property
    def is_complete(self) -> bool:
        return not self.problems

    def report(self) -> str:
        lines = [
            f"capture session {self.date}  (trigger: {self.trigger.value})",
            f"  positions : {', '.join(sorted(p.value for p in self.positions_shot)) or '(none)'}",
            f"  light     : {self.light.value}",
            f"  hands     : {'both' if self.both_hands else 'one'}",
            f"  state     : {self.paired_state or 'unstated'}",
            "",
            f"tier 2 resolvable: {'YES' if self.resolves_tier_2 else 'no'}",
        ]
        if self.problems:
            lines += ["", "problems:"] + [f"  - {p}" for p in self.problems]
        else:
            lines += ["", "protocol satisfied."]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Knuckle survey — three positions the palmar set does not cover
# ---------------------------------------------------------------------------
#
# Knuckle ridges run TRANSVERSE across the hand. Light running down the fingers
# grazes along them and resolves nothing; light crossing them casts shadows into
# the valleys. That is why position 4 in the main set — a flat dorsal shot with
# side light — misses knuckle detail even when the light is low.


class KnucklePosition(str, Enum):
    DORSAL_SURVEY = "a_dorsal_survey"
    LOOSE_FIST_PROFILE = "b_loose_fist_profile"
    MCP_SIDE_PROFILE = "c_mcp_side_profile"


KNUCKLE_POSITIONS: dict[KnucklePosition, PositionSpec] = {
    KnucklePosition.DORSAL_SURVEY: PositionSpec(
        Position.DORSAL,
        hand="flat on a neutral surface, dorsum up, fingers together or slightly "
             "spread",
        camera="directly above",
        light="RAKING FROM THE FINGERTIP DIRECTION toward the wrist, or from the "
              "thumb side — across the ridges, never along them",
        resolves=("scar field", "pad thickening", "ridge irregularity"),
    ),
    KnucklePosition.LOOSE_FIST_PROFILE: PositionSpec(
        Position.DORSAL,
        hand="fingers curled to ~90 degrees at the MCP, IP joints extended",
        camera="dorsal surface facing camera, slightly angled",
        light="from the side, skimming across the MCP ridge line",
        resolves=("pads and scars in relief", "the 'torn up' presentation"),
    ),
    KnucklePosition.MCP_SIDE_PROFILE: PositionSpec(
        Position.DORSAL,
        hand="lateral view, MCP joint centred",
        camera="from the side",
        light="from above or below, grazing the joint",
        resolves=("dorsal-palmar thickness at the joint", "pad bulk",
                  "volar fullness"),
    ),
}

KNUCKLE_LOG_FIELDS: tuple[str, ...] = (
    "which MCP joints carry markers (index, middle, ring, little, thumb)",
    "bilateral or asymmetric",
    "fresh / healing / healed / scarred",
    "pad vs scar vs diffuse swelling",
    "any carbon staining or embedded material",
)
