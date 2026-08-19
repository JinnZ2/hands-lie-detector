"""
Load weighting by mechanism, not by ledger.

See `economic-carve.md`.

    w_domain  ∝  ∫ (force × shear-cycles × geometry-mismatch) dt

There is no payment term in that integral and no way to introduce one that
conserves anything. `LoadBlock` therefore has no payment field. This is not an
omission to be fixed later — the class is frozen, and the absence is the point.

The occupational-hour denomination is implemented too, in `ledger_share()`, so
the two readouts can be printed side by side. It is here as the artifact under
examination, not as an alternative worth using.
"""

from dataclasses import dataclass, replace

# Deliberately absent from LoadBlock. Named here so a reader looking for it
# finds the reason rather than assuming an oversight.
NO_PAYMENT_TERM = (
    "payment status is not a field on LoadBlock: it does not appear in the "
    "governing equations, so it cannot appear in the weight"
)


@dataclass(frozen=True)
class LoadBlock:
    """One block of hand load, described mechanically.

    Fields are the terms of the integral and nothing else. A block is not a
    job, a shift, or an activity category — it is a stretch of time over which
    the mechanical description is roughly constant.

    Args:
        name: label for reporting. Carries no weight.
        hours: duration of the block.
        force: characteristic grip/contact force, arbitrary consistent units.
        shear_cycles_per_hour: sliding-contact cycles per hour.
        geometry_mismatch: 0..1, how far the contact geometry sits from the
            tissue's adapted configuration. 0 = matched, 1 = fully mismatched.
    """

    name: str
    hours: float
    force: float
    shear_cycles_per_hour: float
    geometry_mismatch: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.geometry_mismatch <= 1.0:
            raise ValueError("geometry_mismatch must be in [0, 1]")
        for field_name in ("hours", "force", "shear_cycles_per_hour"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")

    @property
    def weight(self) -> float:
        """Discretized ∫ (force × shear-cycles × geometry-mismatch) dt."""
        return (
            self.force
            * self.shear_cycles_per_hour
            * self.geometry_mismatch
            * self.hours
        )

    @property
    def weight_per_hour(self) -> float:
        return self.force * self.shear_cycles_per_hour * self.geometry_mismatch


def relabel(block: LoadBlock, name: str) -> LoadBlock:
    """Rename a block — e.g. from 'trade' to 'hobby'.

    Exists only to demonstrate that the weight is bit-identical afterward.
    Nothing in `weight` reads `name`.
    """
    return replace(block, name=name)


def load_share(blocks: list[LoadBlock]) -> dict[str, float]:
    """Normalized mechanical weight per block. The physical readout."""
    total = sum(b.weight for b in blocks)
    if total == 0:
        return {b.name: 0.0 for b in blocks}
    return {b.name: b.weight / total for b in blocks}


def ledger_share(blocks: list[LoadBlock], paid: set[str]) -> dict[str, float]:
    """Normalized OCCUPATIONAL-HOUR weight. The artifact, not the measurement.

    This is what an exposure record returns: hours denominated in paid time,
    unpaid blocks contributing nothing. Implemented so the discontinuity can be
    shown rather than asserted. Do not use it to weight anything.
    """
    paid_hours = sum(b.hours for b in blocks if b.name in paid)
    if paid_hours == 0:
        return {b.name: 0.0 for b in blocks}
    return {
        b.name: (b.hours / paid_hours if b.name in paid else 0.0) for b in blocks
    }


@dataclass
class DiscontinuityResult:
    """Result of moving one block across the payment line, mechanics held fixed."""

    block: str
    physical_before: float
    physical_after: float
    ledger_before: float
    ledger_after: float

    @property
    def delta_physical(self) -> float:
        return self.physical_after - self.physical_before

    @property
    def delta_ledger(self) -> float:
        return self.ledger_after - self.ledger_before

    @property
    def is_artifact(self) -> bool:
        """True when the readout moves and the mechanics do not."""
        return self.delta_physical == 0.0 and self.delta_ledger != 0.0

    def report(self) -> str:
        lines = [
            f"block moved across the payment line: {self.block}",
            "  (nothing mechanical changed: same force, geometry, cycles, hours)",
            "",
            f"  physical share : {self.physical_before:.3f} -> {self.physical_after:.3f}"
            f"   (delta {self.delta_physical:+.3f})",
            f"  ledger share   : {self.ledger_before:.3f} -> {self.ledger_after:.3f}"
            f"   (delta {self.delta_ledger:+.3f})",
        ]
        if self.is_artifact:
            lines += [
                "",
                "definitional artifact: the readout jumped across a variable that "
                "appears nowhere in the governing equations.",
                "the instrument is reading the ledger, not the body.",
            ]
        return "\n".join(lines)


def discontinuity(
    blocks: list[LoadBlock], paid: set[str], move: str
) -> DiscontinuityResult:
    """Move one block across the payment line and compare the two readouts.

    Args:
        blocks: the full set of load blocks.
        paid: names of blocks currently counted as paid.
        move: name of the block to flip.
    """
    names = {b.name for b in blocks}
    if move not in names:
        raise KeyError(f"unknown block: {move}; known: {sorted(names)}")

    after_paid = set(paid) ^ {move}
    after_name = f"{move}__relabeled"

    # Relabel the block as the classifier would ("trade" <-> "hobby") and
    # recompute from scratch. The physical readout is computed on genuinely
    # different objects, not read twice from one dict, so its invariance is
    # demonstrated rather than assumed.
    blocks_after = [relabel(b, after_name) if b.name == move else b for b in blocks]

    return DiscontinuityResult(
        block=move,
        physical_before=load_share(blocks)[move],
        physical_after=load_share(blocks_after)[after_name],
        ledger_before=ledger_share(blocks, paid)[move],
        ledger_after=ledger_share(blocks, after_paid)[move],
    )


def compare_readouts(blocks: list[LoadBlock], paid: set[str]) -> str:
    """Side-by-side physical and ledger shares for a set of blocks."""
    phys, ledg = load_share(blocks), ledger_share(blocks, paid)
    width = max((len(b.name) for b in blocks), default=4)
    lines = [f"  {'block':<{width}}  {'physical':>9}  {'ledger':>8}  paid"]
    for b in sorted(blocks, key=lambda b: -phys[b.name]):
        mark = "yes" if b.name in paid else "no"
        lines.append(
            f"  {b.name:<{width}}  {phys[b.name]:>9.3f}  {ledg[b.name]:>8.3f}  {mark}"
        )
    lines += ["", f"  note: {NO_PAYMENT_TERM}"]
    return "\n".join(lines)
