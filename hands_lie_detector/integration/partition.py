"""
The inverted null: one body, one load history. Partition must be argued for.

See `economic-carve.md`.

The inherited default assumes the partition and makes integration argue for
itself. That default is bookkeeping, not physics — under the mechanics there is
one body carrying n loads with real coupling terms, and the domain split is the
thing that needs justifying.

Inverting the default is only worth anything if a proposed partition can FAIL.
Otherwise every partition still gets accepted, with a ceremony in front of it.
The criterion here is a permutation test: a partition is earned only if its
coverage of the observed map would rarely be reached by a same-grain partition
drawn at random from the registry — same number of domains, same registry, no
cherry-picking. "Better than the average random split" is not enough; a coin
clears that half the time. A partition that fails is a relabeling, and
`propose_partition` returns the unpartitioned history instead.

One consequence worth stating up front: with a small registry there are few
distinct same-grain partitions, so the smallest reachable p-value may sit above
alpha and NO partition can be earned at that grain. That is reported as
DEGENERATE rather than silently passed or silently failed. A coarse registry
limits what can be shown, and hiding that would reproduce the problem this
module exists for.
"""

import random
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from math import comb

from .domains import DEFAULT_DOMAINS, DomainSignature, Zone


class PartitionVerdict(str, Enum):
    EARNED = "earned"                    # split explains more than chance
    NOT_EARNED = "not_earned"            # relabeling; unpartitioned history stands
    DEGENERATE = "degenerate"            # registry too small for a null comparison


@dataclass(frozen=True)
class LoadHistory:
    """One body, one load history. The default object, unpartitioned.

    This is what the readout returns when no partition has earned itself. It is
    not an error state or a fallback — it is the correct null.
    """

    zones: frozenset[Zone]
    carrier: str = ""

    @property
    def n_zones(self) -> int:
        return len(self.zones)

    def report(self) -> str:
        who = f" ({self.carrier})" if self.carrier else ""
        return (
            f"unpartitioned load history{who}: "
            f"{', '.join(sorted(z.value for z in self.zones)) or '(no zones)'}\n"
            "  no domain partition has been argued for. this is the null, not a gap."
        )


@dataclass
class PartitionClaim:
    """A proposed domain split, scored against the same-grain null."""

    history: LoadHistory
    proposed: list[str]
    coverage: float
    null_coverage_mean: float
    null_coverage_max: float
    verdict: PartitionVerdict
    p_value: float = 1.0
    alpha: float = 0.05
    n_null_draws: int = 0
    null_exhaustive: bool = False
    unexplained: frozenset[Zone] = frozenset()
    notes: str = ""

    @property
    def gain_over_null(self) -> float:
        return self.coverage - self.null_coverage_mean

    @property
    def earned(self) -> bool:
        return self.verdict is PartitionVerdict.EARNED

    def result(self) -> "PartitionClaim | LoadHistory":
        """The object the readout should carry forward.

        An unearned partition does not degrade to a weaker partition. It
        degrades to the unpartitioned history.
        """
        return self if self.earned else self.history

    def report(self) -> str:
        kind = "exhaustive" if self.null_exhaustive else "sampled"
        lines = [
            f"proposed partition: {', '.join(self.proposed) or '(none)'}",
            f"  coverage of observed zones : {self.coverage:.3f}",
            f"  same-grain null (mean/max) : {self.null_coverage_mean:.3f}"
            f" / {self.null_coverage_max:.3f}   ({kind}, {self.n_null_draws} draws)",
            f"  gain over null mean        : {self.gain_over_null:+.3f}",
            f"  p-value                    : {self.p_value:.3f}  (alpha {self.alpha:.2f})",
            f"  unexplained zones          : "
            f"{', '.join(sorted(z.value for z in self.unexplained)) or '(none)'}",
            "",
            f"verdict: {self.verdict.value}",
        ]
        if self.verdict is PartitionVerdict.EARNED:
            lines.append(
                "  coverage is rarely reached by a random same-grain split. carried "
                "forward as a claim, with its unexplained residual attached."
            )
        elif self.verdict is PartitionVerdict.NOT_EARNED:
            lines.append(
                "  coverage is within reach of chance at this grain. the split is a "
                "relabeling. returning the unpartitioned history."
            )
        else:
            lines.append(
                "  no verdict is reachable at this grain. the unpartitioned history "
                "stands by default."
            )
        if self.notes:
            lines += ["", f"note: {self.notes}"]
        return "\n".join(lines)


def _coverage(zones: frozenset[Zone], sigs: list[DomainSignature]) -> float:
    if not zones:
        return 0.0
    predicted: frozenset[Zone] = (
        frozenset().union(*(s.zones for s in sigs)) if sigs else frozenset()
    )
    return len(zones & predicted) / len(zones)


def propose_partition(
    history: LoadHistory,
    domains: list[str],
    registry: dict[str, DomainSignature] | None = None,
    alpha: float = 0.05,
    n_null_draws: int = 2000,
    seed: int = 0,
) -> PartitionClaim:
    """Argue for a domain partition of one load history.

    Runs a permutation test: how often does a random same-grain partition from
    the registry reach this partition's coverage?

    Args:
        history: the unpartitioned default.
        domains: the proposed split, as registry keys.
        registry: signature registry; defaults to the stipulated table.
        alpha: significance level the p-value must clear.
        n_null_draws: sampled draws, used only when exhaustive enumeration of
            same-grain partitions would be larger than this.
        seed: fixed, so the verdict is reproducible.

    Returns:
        PartitionClaim. Call `.result()` to get what should be carried forward —
        the claim if earned, the unpartitioned history if not.
    """
    registry = registry or DEFAULT_DOMAINS
    unknown = [d for d in domains if d not in registry]
    if unknown:
        raise KeyError(f"unknown domain(s): {sorted(unknown)}; known: {sorted(registry)}")

    sigs = [registry[d] for d in domains]
    coverage = _coverage(history.zones, sigs)
    predicted = frozenset().union(*(s.zones for s in sigs)) if sigs else frozenset()
    unexplained = frozenset(history.zones - predicted)

    pool = sorted(registry)
    k, n = len(domains), len(pool)

    def degenerate(note: str) -> PartitionClaim:
        return PartitionClaim(
            history=history,
            proposed=list(domains),
            coverage=coverage,
            null_coverage_mean=0.0,
            null_coverage_max=0.0,
            verdict=PartitionVerdict.DEGENERATE,
            alpha=alpha,
            unexplained=unexplained,
            notes=note,
        )

    if k == 0 or k >= n:
        return degenerate(
            "a partition using every domain in the registry, or none, has no "
            "same-grain null to be compared against"
        )

    n_distinct = comb(n, k)
    if n_distinct < 1 / alpha:
        return degenerate(
            f"registry too coarse at this grain: only {n_distinct} distinct "
            f"same-grain partitions exist, so the smallest reachable p-value is "
            f"{1 / n_distinct:.3f} > alpha={alpha:.2f}. no partition of {k} "
            f"domain(s) can be earned against a registry of {n}. widen the "
            "registry or accept that this grain is unresolvable, but do not read "
            "the failure as evidence about the split."
        )

    exhaustive = n_distinct <= n_null_draws
    if exhaustive:
        draws = [
            _coverage(history.zones, [registry[name] for name in combo])
            for combo in combinations(pool, k)
        ]
    else:
        rng = random.Random(seed)
        draws = [
            _coverage(history.zones, [registry[name] for name in rng.sample(pool, k)])
            for _ in range(n_null_draws)
        ]

    at_least = sum(1 for d in draws if d >= coverage)
    p_value = at_least / len(draws) if exhaustive else (at_least + 1) / (len(draws) + 1)

    verdict = (
        PartitionVerdict.EARNED if p_value <= alpha else PartitionVerdict.NOT_EARNED
    )

    return PartitionClaim(
        history=history,
        proposed=list(domains),
        coverage=coverage,
        null_coverage_mean=sum(draws) / len(draws),
        null_coverage_max=max(draws),
        verdict=verdict,
        p_value=p_value,
        alpha=alpha,
        n_null_draws=len(draws),
        null_exhaustive=exhaustive,
        unexplained=unexplained,
        notes=(
            "null is built from the same stipulated registry, so it inherits that "
            "registry's errors. it tests whether THIS split beats a random split, "
            "not whether the registry is right."
        ),
    )
