"""
Tribological wear taxonomy — the spine this repo was rebuilding by hand.

See `wear-taxonomy.md`.

Wear modes are classified by MECHANISM, never by application. Nobody sorts wear
by industry; that is the whole point of the taxonomy, and it is domain-blind
because it had to be. Reading a worn component backward to its service
condition is routine failure analysis. Running it on tissue needs no new
vocabulary.

Two transfers that do real work, and one residual that does not transfer.
"""

from dataclasses import dataclass
from enum import Enum

from .domains import LoadMode


class WearMode(str, Enum):
    """The standard taxonomy, with its tissue reading."""

    ADHESIVE = "adhesive"      # surfaces grip; junction shears below the interface
    ABRASIVE = "abrasive"      # hard particle or rough counterface ploughs
    FATIGUE = "fatigue"        # subsurface cyclic damage, failure at depth
    CORROSIVE = "corrosive"    # chemical attack of the surface layer
    FRETTING = "fretting"      # small-amplitude oscillation at a fixed contact


TISSUE_READING: dict[WearMode, str] = {
    WearMode.ADHESIVE: "friction-peak delamination: the blister roof shears below "
                       "the gripping surface, not at it",
    WearMode.ABRASIVE: "bark, grit, clay, sand ploughing the stratum corneum",
    WearMode.FATIGUE: "callus is fatigue-driven remodeling; blister is fatigue "
                      "delamination",
    WearMode.CORROSIVE: "alkali, solvent, defatting — the scrub",
    WearMode.FRETTING: "a ring band. textbook fretting geometry: fixed contact, "
                       "small-amplitude oscillation",
}

# The ad-hoc LoadMode vocabulary in domains.py, mapped onto the standard one.
LOAD_MODE_TO_WEAR: dict[LoadMode, WearMode] = {
    LoadMode.SHEAR: WearMode.ADHESIVE,
    LoadMode.ABRASION: WearMode.ABRASIVE,
    LoadMode.COMPRESSION: WearMode.FATIGUE,
    LoadMode.IMPACT: WearMode.FATIGUE,
    LoadMode.VIBRATION: WearMode.FRETTING,
    LoadMode.TORSION: WearMode.FRETTING,
    LoadMode.MACERATION: WearMode.CORROSIVE,
}


def wear_mode(mode: LoadMode) -> WearMode:
    return LOAD_MODE_TO_WEAR[mode]


# ---------------------------------------------------------------------------
# Transfer 1 — wear is a SYSTEM property, not a material property
# ---------------------------------------------------------------------------

WEAR_SYSTEM_TERMS: tuple[str, ...] = (
    "load", "velocity", "geometry", "counterface", "lubricant", "cycles",
)


@dataclass
class WearSystem:
    """A wear measurement needs both sides of the interface.

    This is not a convenience: the formalism requires the counterface. Handle
    wear and palm wear are ONE measurement taken from two sides, so a hand alone
    is an incomplete specimen and the tool carries the conjugate record.
    """

    hand_zone: str
    counterface: str = ""
    lubricant: str = ""
    cycles: int | None = None

    @property
    def is_complete_specimen(self) -> bool:
        return bool(self.counterface)

    def report(self) -> str:
        lines = [
            f"wear system at {self.hand_zone}",
            f"  counterface : {self.counterface or 'NOT RECORDED'}",
            f"  lubricant   : {self.lubricant or 'not recorded'}",
            f"  cycles      : {self.cycles if self.cycles is not None else 'not recorded'}",
        ]
        if not self.is_complete_specimen:
            lines.append(
                "\n  INCOMPLETE SPECIMEN: wear is a system property. without the "
                "counterface this is half a measurement — the tool holds the other "
                "half, and photographing the handle is not optional extra data."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transfer 2 — run-in
# ---------------------------------------------------------------------------


class RunIn(str, Enum):
    """Surfaces wear fast, conform, then settle to a low steady rate.

    The conformed state is optimal and it is MAINTAINED, not final. The band is
    run-in held.
    """

    PRE = "pre_run_in"                 # no conformity yet
    CONFORMED = "conformed"            # low wear rate, functional
    PAST_FUNCTION = "past_function"    # surface exists, interface no longer works


RUN_IN_READING: dict[RunIn, str] = {
    RunIn.PRE: "soft: no conformity yet",
    RunIn.CONFORMED: "band: run-in held. conformal, low wear rate, functional",
    RunIn.PAST_FUNCTION: "thick: run-in carried past function. the surface still "
                         "exists; the interface does not work",
}

# Where the transfer breaks, and it is the interesting part.
RESIDUAL_WITHOUT_ANALOGUE = (
    "steel only degrades. skin REMODELS toward the load — a negative feedback "
    "loop with no complete engineering analogue. tribofilms and work-hardening "
    "are partial. that residual is the part with no literature."
)


def scar_identifies_mechanism_not_application() -> str:
    """The category question, settled by the taxonomy.

    A wear scar's morphology identifies the mechanism. The mechanism does not
    know the application. The application is where the codes live.
    """
    return (
        "same load, same counterface, same cycles -> identical scar. own land or "
        "someone else's, the category has no term in the wear equation to modify. "
        "categories are not unimportant; they are downstream of a measurement the "
        "material already took, before anyone classified it."
    )


# ---------------------------------------------------------------------------
# Counterfaces — hands to tools, feet to boots
# ---------------------------------------------------------------------------
#
# `feet-lie-detector.md` as originally specced reads GAIT, and that is a
# sampling mismatch:
#
#     PALM   integrated deposit. static. one frame samples it fully.
#            the information is in the STATE.
#     GAIT   dynamic. the information is in sequence, variability and
#            perturbation recovery. a still discards all three and keeps the
#            pose, which is the least informative component.
#
# So the feet track either needs video or needs to stop reading motion. The fix
# is the same tribological move already made for hands: wear is a system
# property, so read the COUNTERFACE.

BOOT_WEAR_ITEMS: tuple[str, ...] = (
    "sole wear pattern and depth",
    "strike location, lateral vs medial",
    "upper crease lines — where the foot actually flexes",
    "lacing wear",
    "heel counter collapse",
    "midsole compression set",
)

COUNTERFACE_PROPERTIES = (
    "static, shootable at rest, good light available, no disposition gate and no "
    "coincidence gate — nobody has to decide it is worth photographing while it "
    "happens. and it is self-dating: boots get replaced, so the wear is bounded "
    "by the boot's service life."
)

CONJUGATE_PAIRS: dict[str, str] = {
    "hand": "tool handle",
    "foot": "boot",
}
