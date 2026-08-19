# Gates, Not Sums

A correction to the functional form, and an index of the same error class
appearing three times in this repo.

---

## The weighting question was posed in the wrong form

`economic-carve.md` set the problem up as

```
w_job · J  +  w_sub · S
```

Two parallel commensurable inputs, one sum. It then argued at length that the
coefficients were wrong — that `w_sub ≈ 0` is not merely small but structurally
absent, and that the true split "could go 60/40 in either direction."

Every word of that is inside the wrong form.

The actual structure is

```
environment  →  capacity  →  job
```

where each layer **gates** the next. And a weighted sum cannot express a gate.
No coefficient assignment fixes it — **including the correct one**, because

```
∂(job output) / ∂(subsistence)   is MULTIPLICATIVE
```

and the whole product goes to zero when a lower layer fails. A sum has a
constant partial derivative; that is exactly what makes it the wrong shape here.
Arguing about whether the split is 60/40 or 40/60 is arguing about a number in an
equation that cannot hold the relationship at any value.

So the earlier correction was itself half-corrected. `economic-carve.md` was
right that the weight is not 1 : 0. It was wrong to treat "what is the split" as
the question.

---

## The ledger runs the arrow backward

```
RECORD                        PHYSICS
  job         = production      subsistence = production (of capacity)
  subsistence = consumption     job         = the DRAW on it
```

The accounting has the arrow **reversed** on the layer it does not instrument.

That is not an omission plus a coefficient error. It is a **sign error on the
dependency direction** — the record treats the producing layer as the consuming
one, which is why no amount of reweighting inside the ledger's own frame
recovers the relationship. `arrow_check()` prints both columns side by side.

---

## It closes onto the band

Capacity is the **load-bearing asset**, not the output.

Running hands to saturation converts capacity into near-term output — spending
the precondition to fund the term that depends on it. The output goes up. The
gate goes down. A summed readout sees only the first.

So **band maintenance is not upkeep alongside the work. It is the same operation
as keeping the lower layer solvent.**

And the two-sided window in `band-not-scale.md` stops being a curiosity of the
tissue and becomes the expected signature: both exits from the band are
insolvency, for different reasons.

| band state | capacity | why |
|---|---|---|
| `banded` | solvent | the asset is intact and being maintained |
| `thick / glassy` | **spent** | converted into near-term output; armored past sensing |
| `soft / uniform` | never built | no load ever deposited it |

`solvency_from_band()` reads capacity solvency directly off `HandState`. The
two-sided window is what a gated structure looks like from inside the tissue.

---

## The error class — three instances, one shape

Each of these was found separately and each turned out to be the same thing:
**wrong form, not wrong number.** None was fixable by tuning.

A fifth, adjacent: the contrast metric in `band-not-scale.md` was specced as a
skill proxy and measures geometric concentration instead. That one is a
misinterpretation rather than a form error — the quantity was fine, the label on
it was not — but it belongs in the same list because it failed the same way: it
would have ranked a multi-domain generalist below a fixed-geometry specialist at
identical competence.

| where | the form | what it cannot express |
|---|---|---|
| `ScoreEvaluator` — `sum()` of 7 categories | additive | any interaction between categories. The integration term has no slot at any coefficient. (`reference-class-empty.md`) |
| the thickness scale — score rises with mean | monotone | the far-side band exit. A saturated hand is not ranked low, it is ranked highest. (`band-not-scale.md`) |
| layer weighting — `w_job·J + w_sub·S` | summed | a gate. Output cannot go to zero when a lower layer fails. (this document) |
| domain weighting — hours × force per domain | same-sign additive | a **sign**. Depositing domains build capacity; drawing domains spend it. No coefficient makes an input into a demand. (`wear-taxonomy.md`) |

The diagnostic that catches all three, before any argument about values:

> **Ask what the form cannot represent at any parameter setting.**
> If the answer is the quantity you care about, stop tuning.

This pairs with the strip in `economic-carve.md`. The strip asks whether a
*category* carries physical information. This asks whether a *functional form*
can carry the relationship. Both are one-line checks, and both are cheaper than
the arguments they replace.

---

## Executable

```python
from hands_lie_detector.integration.gated import (
    GatedStack, Stratum, StratumState, arrow_check, solvency_from_band,
)
from hands_lie_detector.band import HandState

stack = GatedStack({
    Stratum.ENVIRONMENT: StratumState(Stratum.ENVIRONMENT, solvency=0.9),
    Stratum.CAPACITY: solvency_from_band(HandState.GLASSY),
    Stratum.JOB: StratumState(Stratum.JOB, solvency=1.0, draw=70.0),
})
print(stack.report())
```

With capacity failed and a 70-hour draw above it:

```
  gated output   : 0.000
  additive output: 31.500   <- what a weighted sum reports

COLLAPSED: a lower layer is at zero, so the product is zero regardless of
the draw above it.
  the additive form still reports 31.500. no coefficient assignment repairs
  this — it is the wrong functional form.
```

`additive_output()` exists solely to be wrong next to `output()`, the same way
`ledger_share()` sits next to `load_share()` and `monotone_score()` sits next to
`read_band()`. This repo keeps its errors runnable rather than described.

`sensitivity()` returns a derivative that depends on every other layer, which is
the gate showing up in the calculus — a summed model would return a constant
there.

---

## What this does and does not change

**`load_weight.py` is unchanged and still correct.** The integral
`∫ (force × shear-cycles × geometry-mismatch) dt` is the right form *within* a
layer: two blocks of hand load at the same stratum genuinely do add. What was
wrong was carrying that additivity **across** strata, where the relationship is a
gate.

So the paid-versus-unpaid comparison in `economic-carve.md` still stands as
stated — as a within-layer statement about mechanical share. What does not stand
is treating that share as the answer to "how much does subsistence matter,"
because subsistence is not a parallel input at the same layer. It is the layer
underneath.

---

## Status

Open: the solvency values in `solvency_from_band()` are stipulated, like every
other default here. `BANDED → 1.0` and `GLASSY → 0.25` are asserted, not
measured, and the ordering is the defensible part while the magnitudes are not —
which is the living-tissue seam from `economic-carve.md` arriving in this
module's own constants.

Not done: the scoring rubric and vision heads still sum. Three form errors are
now documented and none is fixed in the shipped scale, because each fix is a
replacement rather than an edit.
