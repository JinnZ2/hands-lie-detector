"""
The strip: render a category noun into the units of the governing equation.

See `economic-carve.md`.

    "mechanic"  ->  N, m, cycles/hr, duty cycle
    "hobby"     ->  N, m, cycles/hr, duty cycle

Two outcomes, and both are diagnostic:

- both sides render to the SAME units → the category was carrying no physical
  information, and the distinction drops out cleanly.
- a category CANNOT be rendered into those units at all → it is not a physical
  class. It is a ledger class wearing one.

One line, and it runs on anything. It is the cheapest instrument in this repo
and it is the one that should be reached for first.

The registry ships with entries that can be defended from the documents.
Anything else returns UNREGISTERED rather than a guess — failing closed, the
same as `carve_audit.classify_relation`. `register()` is how an operator adds
their own.
"""

from dataclasses import dataclass
from enum import Enum

# The units of the governing equation, and the fields of
# `load_weight.LoadBlock` — the same four, deliberately.
MECHANICAL_UNITS: tuple[str, ...] = (
    "force_N",
    "displacement_m",
    "cycles_per_hour",
    "duty_cycle",
)


class StripVerdict(str, Enum):
    DROPS_OUT = "drops_out"                    # same units: no physical content
    PHYSICALLY_DISTINCT = "physically_distinct"  # different units: a real difference
    LEDGER_CLASS = "ledger_class"              # unrenderable: a pay code in costume
    UNREGISTERED = "unregistered"              # fail closed


@dataclass(frozen=True)
class Rendering:
    """A category noun, rendered into mechanical units or refused."""

    category: str
    units: tuple[str, ...] | None
    reason: str = ""

    @property
    def renderable(self) -> bool:
        return self.units is not None

    def __str__(self) -> str:
        if self.renderable:
            return f"{self.category} -> {', '.join(self.units)}"
        return f"{self.category} -> UNRENDERABLE ({self.reason})"


def _load(category: str) -> Rendering:
    return Rendering(category, MECHANICAL_UNITS)


def _ledger(category: str, reason: str) -> Rendering:
    return Rendering(category, None, reason)


NO_LOAD_REFERENT = "names an employment or billing relation, not a load"

DEFAULT_RENDERINGS: dict[str, Rendering] = {
    r.category: r
    for r in [
        # Nouns that point at load. All render identically, which is the point.
        _load("mechanic"),
        _load("welder"),
        _load("farmer"),
        _load("driver"),
        _load("hobby"),
        _load("gardening"),
        _load("homesteading"),
        _load("hauling"),
        _load("fabricating"),
        _load("milling"),
        _load("mending"),
        # Nouns that point at no load at all.
        _ledger("occupation", NO_LOAD_REFERENT),
        _ledger("employment status", NO_LOAD_REFERENT),
        _ledger("profession", NO_LOAD_REFERENT),
        _ledger("trade", "names a licensed or waged category, not a load"),
        _ledger("risk class", "an actuarial price band"),
        _ledger("billing category", NO_LOAD_REFERENT),
        _ledger("licensure", NO_LOAD_REFERENT),
        _ledger("soc code", "an economic classification"),
        _ledger("naics code", "an economic classification"),
        # This repo's own interpretation bands, run through its own diagnostic.
        _ledger("casual hobbyist", "the payment-negation classifier, as a verdict"),
        _ledger(
            "working hands",
            "seated between two employment bands in DEFAULT_BANDS, so its content "
            "there is employment position, not load",
        ),
        _ledger("experienced trade", NO_LOAD_REFERENT),
        _ledger("field work", NO_LOAD_REFERENT),
        _ledger("podcast hands", "a verdict about a person, not a load"),
    ]
}


def register(rendering: Rendering, registry: dict[str, Rendering]) -> None:
    """Add an operator's own rendering. Mutates `registry` in place."""
    registry[rendering.category.strip().lower()] = rendering


def render(category: str, registry: dict[str, Rendering] | None = None) -> Rendering | None:
    registry = DEFAULT_RENDERINGS if registry is None else registry
    return registry.get(category.strip().lower())


@dataclass
class StripResult:
    left: str
    right: str | None
    verdict: StripVerdict
    left_rendering: Rendering | None = None
    right_rendering: Rendering | None = None
    note: str = ""

    def report(self) -> str:
        lines = [f"strip: {self.left}" + (f" vs {self.right}" if self.right else "")]
        for r in (self.left_rendering, self.right_rendering):
            if r is not None:
                lines.append(f"  {r}")
        lines += ["", f"verdict: {self.verdict.value}", f"  {self.note}"]
        return "\n".join(lines)


_NOTES = {
    StripVerdict.DROPS_OUT: (
        "both render to the same units. the category was carrying no physical "
        "information; the distinction drops out cleanly and must not be weighted."
    ),
    StripVerdict.PHYSICALLY_DISTINCT: (
        "the two render to different unit signatures. the distinction is doing "
        "physical work and survives the strip."
    ),
    StripVerdict.LEDGER_CLASS: (
        "cannot be rendered into the units of the governing equation at all. this "
        "is not a physical class — it is a ledger class wearing one."
    ),
    StripVerdict.UNREGISTERED: (
        "no rendering on file. failing closed: state the noun's content in force, "
        "displacement, cycles and duty cycle, or register it as unrenderable."
    ),
}


def strip(
    category: str,
    against: str | None = None,
    registry: dict[str, Rendering] | None = None,
) -> StripResult:
    """Run the strip on one category noun, or on a pair.

    Args:
        category: the noun to render.
        against: optional second noun. Supplying one asks whether the
            DISTINCTION carries physical information, rather than whether the
            single noun does.
        registry: rendering registry; defaults to the shipped table.
    """
    left = render(category, registry)
    right = render(against, registry) if against is not None else None

    def result(verdict: StripVerdict) -> StripResult:
        return StripResult(
            left=category, right=against, verdict=verdict,
            left_rendering=left, right_rendering=right, note=_NOTES[verdict],
        )

    if left is None or (against is not None and right is None):
        return result(StripVerdict.UNREGISTERED)
    if not left.renderable or (right is not None and not right.renderable):
        return result(StripVerdict.LEDGER_CLASS)
    if right is None:
        return result(StripVerdict.PHYSICALLY_DISTINCT)
    if left.units == right.units:
        return result(StripVerdict.DROPS_OUT)
    return result(StripVerdict.PHYSICALLY_DISTINCT)


def strip_all(
    categories: list[str], registry: dict[str, Rendering] | None = None
) -> dict[str, StripVerdict]:
    """Run the single-noun strip across a list. Useful on a whole scale at once."""
    return {c: strip(c, registry=registry).verdict for c in categories}


# ---------------------------------------------------------------------------
# Cardinality reduction — the general form the strip is one case of
# ---------------------------------------------------------------------------
#
# Three moves that look different and are one operation at different layers:
#
#   FALSE BINARY      reduces the OPTION set        -> presents 2 as all
#   WELDED TERM       reduces two VARIABLES to one  -> presents the weld as a
#                                                      single quantity
#   NARROW FRAMEWORK  reduces the POPULATION        -> presents a subset's
#                                                      regularities as universal
#
# Same form each time: a cardinality reduction with the reduction step deleted
# from the output. Which is the domain carve again — partition, then forget you
# partitioned.


class ReductionKind(str, Enum):
    OPTION_SET = "false_binary"
    VARIABLE_WELD = "welded_term"
    POPULATION = "narrow_framework"


@dataclass(frozen=True)
class CardinalityReduction:
    """A reduction, and whether the reducing step survived into the output."""

    kind: ReductionKind
    label: str
    presented: str
    actual: str
    declared: bool = False

    @property
    def undeclared(self) -> bool:
        return not self.declared

    def report(self) -> str:
        lines = [
            f"{self.kind.value}: {self.label}",
            f"  presented : {self.presented}",
            f"  actual    : {self.actual}",
        ]
        if self.undeclared:
            lines.append(
                "  UNDECLARED: the reduction step is missing from the output, so "
                "the reduced set is being reported as the world."
            )
        return "\n".join(lines)


# Terms that fuse two variables and report the weld as one quantity.
WELDED_TERMS: dict[str, CardinalityReduction] = {
    r.label: r
    for r in [
        CardinalityReduction(
            ReductionKind.VARIABLE_WELD,
            "waste heat",
            "one quantity: energy lost",
            "two variables: energy leaving the intended sink, AND energy of no "
            "further use. in a heated space in winter the balance goes to the "
            "room, which is a required load — so the first is nonzero and the "
            "second is zero",
        ),
        CardinalityReduction(
            ReductionKind.VARIABLE_WELD,
            "hobby",
            "one quantity: a kind of activity",
            "two variables: the physical load, AND whether it was paid. only the "
            "second is in the definition, and only the first is in the mechanics",
        ),
        CardinalityReduction(
            ReductionKind.POPULATION,
            "strange",
            "a property of the subject",
            "a frequency claim against a denominator — and the denominator in use "
            "was constructed by excluding the subject. widen it to every human "
            "who has lived and cascading multi-domain subsistence is the MODAL "
            "condition",
        ),
    ]
}


CONFUSION_IS_A_DETECTION_EVENT = (
    "working from the physical system, the option space is bounded by physics, "
    "and physics is wide. a presented binary is narrower than the physics "
    "permits, so the arriving signal is not 'I disagree with option A' — it is "
    "'where did the other options go, since nothing removed them.' confusion is "
    "the correct output for an undeclared reduction: a detection event, not a "
    "failure to follow. and it is diagnostic about the presenter's frame rather "
    "than about the topic."
)


def unweld(term: str) -> CardinalityReduction | None:
    """Look up a term that fuses two variables into one reported quantity."""
    return WELDED_TERMS.get(term.strip().lower())
