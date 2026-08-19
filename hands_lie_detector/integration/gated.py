"""
Gates, not sums. The functional form for stratified load.

See `gated-not-summed.md`.

The weighting question was posed as

    w_job · J  +  w_sub · S

which makes the two parallel commensurable inputs to one sum. The actual
structure is

    environment → capacity → job

where each layer GATES the next. A weighted sum cannot express a gate. No
coefficient assignment fixes it, including the correct one, because

    ∂(job output) / ∂(subsistence)  is MULTIPLICATIVE,

and the whole product goes to zero when a lower layer fails. Arguing about
whether the split is 60/40 or 40/60 is arguing inside the wrong form.

Same error class as the monotone callus scorer: wrong form, not wrong number.

And the ledger has the arrow reversed on the layer it does not instrument:

    RECORD                          PHYSICS
      job         = production        subsistence = production (of capacity)
      subsistence = consumption       job         = the DRAW on it

Not an omission plus a coefficient error. A sign error on the dependency
direction.
"""

from dataclasses import dataclass
from enum import Enum
from math import prod


class Stratum(str, Enum):
    ENVIRONMENT = "environment"
    CAPACITY = "capacity"
    JOB = "job"


# Base first. Each stratum gates everything above it.
ORDER: tuple[Stratum, ...] = (Stratum.ENVIRONMENT, Stratum.CAPACITY, Stratum.JOB)

# What each layer is, in each accounting.
LEDGER_SIGN: dict[Stratum, str] = {
    Stratum.ENVIRONMENT: "externality",
    Stratum.CAPACITY: "consumption",
    Stratum.JOB: "production",
}

PHYSICS_SIGN: dict[Stratum, str] = {
    Stratum.ENVIRONMENT: "production (of the conditions capacity needs)",
    Stratum.CAPACITY: "production (of the capacity job draws on)",
    Stratum.JOB: "the draw",
}


@dataclass(frozen=True)
class StratumState:
    """One layer's solvency and the draw taken out of it.

    Args:
        stratum: which layer.
        solvency: 0..1. Is this layer being maintained? 1 = fully solvent,
            0 = failed. This is a gate, not a weight.
        draw: what is being taken out of the layer, in whatever consistent
            units the operator is using.
    """

    stratum: Stratum
    solvency: float
    draw: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.solvency <= 1.0:
            raise ValueError("solvency must be in [0, 1]")
        if self.draw < 0:
            raise ValueError("draw must be non-negative")


@dataclass
class GatedStack:
    """Layers in dependency order, with output gated multiplicatively."""

    states: dict[Stratum, StratumState]

    def __post_init__(self) -> None:
        missing = [s for s in ORDER if s not in self.states]
        if missing:
            raise KeyError(f"missing strata: {[s.value for s in missing]}")

    def solvency(self, stratum: Stratum) -> float:
        return self.states[stratum].solvency

    def gate(self, stratum: Stratum) -> float:
        """Product of the solvencies of every layer BELOW this one."""
        below = ORDER[: ORDER.index(stratum)]
        return prod((self.solvency(s) for s in below), start=1.0)

    def output(self, stratum: Stratum = Stratum.JOB) -> float:
        """Gated output: the layer's own draw, multiplied by every gate under it."""
        return self.states[stratum].draw * self.gate(stratum)

    def additive_output(self, stratum: Stratum = Stratum.JOB) -> float:
        """What a weighted sum would report. The misread, computed on purpose.

        Solvencies enter as parallel additive contributions rather than as
        gates, which is exactly the form that cannot go to zero when a lower
        layer fails.
        """
        contributions = [self.solvency(s) for s in ORDER[: ORDER.index(stratum)]]
        weight = sum(contributions) / len(contributions) if contributions else 1.0
        return self.states[stratum].draw * weight

    def sensitivity(self, stratum: Stratum, of: Stratum = Stratum.JOB) -> float:
        """d(output) / d(solvency of `stratum`). Multiplicative, not constant.

        A weighted sum has a constant partial derivative — that is what makes it
        the wrong form. Here the derivative depends on every other layer, which
        is the gate showing up in the calculus.
        """
        if ORDER.index(stratum) >= ORDER.index(of):
            return 0.0
        s = self.solvency(stratum)
        return self.output(of) / s if s else self.states[of].draw * prod(
            (self.solvency(x) for x in ORDER[: ORDER.index(of)] if x is not stratum),
            start=1.0,
        )

    def failed_strata(self) -> list[Stratum]:
        return [s for s in ORDER if self.solvency(s) == 0.0]

    @property
    def collapsed(self) -> bool:
        return self.output() == 0.0 and self.states[Stratum.JOB].draw > 0.0

    def report(self) -> str:
        lines = ["gated stack (base first):"]
        for s in ORDER:
            st = self.states[s]
            lines.append(
                f"  {s.value:<12} solvency {st.solvency:.2f}   draw {st.draw:>7.2f}"
                f"   gate below {self.gate(s):.3f}"
            )
        gated, additive = self.output(), self.additive_output()
        lines += [
            "",
            f"  gated output   : {gated:.3f}",
            f"  additive output: {additive:.3f}   <- what a weighted sum reports",
        ]
        if self.collapsed:
            lines += [
                "",
                "COLLAPSED: a lower layer is at zero, so the product is zero "
                "regardless of the draw above it.",
                f"  failed: {', '.join(s.value for s in self.failed_strata())}",
                f"  the additive form still reports {additive:.3f}. no coefficient "
                "assignment repairs this — it is the wrong functional form.",
            ]
        return "\n".join(lines)


def arrow_check() -> str:
    """Print the dependency direction in both accountings.

    The ledger has the arrow reversed on the layer it does not instrument. This
    is a sign error on the dependency direction, not an omitted coefficient.
    """
    lines = ["dependency direction:", "", f"  {'stratum':<12} {'record':<14} physics"]
    for s in ORDER:
        lines.append(f"  {s.value:<12} {LEDGER_SIGN[s]:<14} {PHYSICS_SIGN[s]}")
    lines += [
        "",
        "  the record calls the base layers consumption and the top layer",
        "  production. physics has it the other way: the lower layers produce the",
        "  capacity the upper layer draws on. reversed arrow, not a missing term.",
    ]
    return "\n".join(lines)


def solvency_from_band(band_state, near_term_output: float = 1.0) -> StratumState:
    """Read capacity solvency off a band state.

    Capacity is the load-bearing ASSET, not the output. Running hands to
    saturation converts capacity into near-term output — spending the
    precondition to fund the term that depends on it. So band maintenance is not
    upkeep alongside the work; it is the same operation as keeping the lower
    layer solvent.

    Both sides of the band are insolvency, for different reasons. That two-sided
    window is what a gated structure looks like from inside the tissue.

    Args:
        band_state: a `hands_lie_detector.band.HandState`.
        near_term_output: draw currently being taken out of capacity.
    """
    from ..band import HandState

    solvency = {
        HandState.BANDED: 1.0,        # maintained; the asset is intact
        HandState.GLASSY: 0.25,       # spent: capacity converted to near-term output
        HandState.SOFT: 0.25,         # never built
        HandState.INDETERMINATE: 0.0,  # unread, and unread is not solvent
    }[band_state]
    return StratumState(Stratum.CAPACITY, solvency=solvency, draw=near_term_output)
