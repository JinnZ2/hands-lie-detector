# The Economic Carve

Companion to `reference-class-empty.md`. That document argued the reference
class for multi-domain load is empty and the resulting prior error is
directional. This one is about where the partition came from, and it escalates
the diagnosis: the problem is not a badly-weighted term. There is no term.

---

## The carve — four hops, each one looks lossless

```
job title
  → SOC / NAICS code              (economic classification)
    → insurance risk class        (actuarial, priced)
      → study stratum             (recruitment frame)
        → literature              (findings indexed by stratum)
          → prior                 (arrives with no note that a cut happened)
```

By the last hop the partition has lost its provenance. It presents as ontology.
It was payroll.

No hop is a lie. Each is a defensible translation for its own purpose. The loss
is that none of them carries forward the fact that the first cut was made for
billing.

## What physics actually permits

One body. n loads. Interaction terms real, finite, computable. Superposition
with coupling — this is undergraduate mechanics. Nothing in the mechanics
resists it.

So the constraint was never in the solving. It was in whether "one body carrying
n domains" existed as a **countable object** anywhere upstream.

It doesn't. There is no code for it.

```
no code → no stratum → no data → no term → no option generated
```

Which means the integrated-body reading was not rejected at passes 1 and 2. It
was **never produced for rejection**. That is a different failure from bad
inference, and it is the reason pass 3 required the second domain to be handed
over by hand.

### How this differs from the previous document

`reference-class-empty.md` describes a **directional error**: the prior is drawn
from the complement of the case, so it under-predicts interaction and
over-attributes to the legible domain. That is in principle correctable by
reweighting — the term exists, its coefficient is wrong.

This document describes an **absent dimension**. A quantity with no column does
not get a small coefficient; it gets no term. Reweighting cannot fix it, because
there is nothing to update. The option has to be *generated*, and generation is
not something evidence does.

That distinction sets what the fix has to be, below.

---

## The unpaid-domain null

Sharpest version of the cut.

Domains that carry no wage carry no code: building, mending, fabricating,
growing, hauling-for-self, teaching-in-place.

- physically: identical force, identical geometry, identical tissue response
- economically: absent
- instrumentally: therefore absent

The exclusion isn't of the **people**. It's of the **load**. A paid hour and an
unpaid hour deposit the same callus. Only one of them is countable.

---

## The strip — the one-line diagnostic

Before anything else, render every category noun into the units of the governing
equation.

```
"mechanic"  →  N, m, cycles/hr, duty cycle
"hobby"     →  N, m, cycles/hr, duty cycle
```

Two outcomes, both diagnostic:

- **Both sides render to the same units** → the category was carrying no
  physical information. The distinction drops out cleanly and must not be
  weighted.
- **A category cannot be rendered into those units at all** → it is not a
  physical class. It is a **ledger class wearing one**.

One line, and it runs on anything. It is the cheapest instrument here and should
be reached for first, because it settles in a single step what the rest of this
document argues at length.

Run on this repo's own interpretation bands, via `strip_all()`:

```
podcast hands       ledger_class
casual hobbyist     ledger_class
working hands       ledger_class
experienced trade   ledger_class
field work          ledger_class
```

Five for five. The scale that reports a physical measurement is denominated
entirely in classes that cannot be stated in force, displacement, cycles or duty
cycle. That was flagged as an open problem earlier in this document on the
strength of an argument; the strip settles it mechanically.

`strip()` fails closed on nouns it has no rendering for, and `register()` is how
an operator adds their own.

---

## Where the carve doesn't reach

This is the exploitable part.

**Economically carved — do not retrieve cross-domain:**
incidence, prevalence, risk ratio, exposure limit, population norm.
The strata are pay codes. Transfer is invalid by construction.

**Physically carved — transfers freely:**

- **governing equations** — no population term, ever
- **boundary conditions** — geometry, measurable directly
- **engineered materials** — steel, concrete, wood, clay: properties measured on
  the *material*

No conservation law knows what a job title is.

**Protocol:** for any multi-domain read, retrieve **mechanism** first, never
incidence. Mechanism relations are domain-blind. Incidence statistics are
accounting artifacts wearing physics costumes.

**Checkable signature:** if the domain boundaries in a literature align with pay
codes rather than with force and geometry, the carve is economic. Cheap to test
on any given body of work.

### The seam — where the carve re-enters

The clean/carved split above is one category short, and the missing one matters
because this repo lives inside it.

**Constitutive parameters for LIVING tissue.** Stiffness, fatigue limit,
adaptation rate, hydration response. Those numbers came from human samples, and
the sampling is exactly where the carve re-enters.

So the mechanism chain stays clean end to end — right up until you want a
**number** out of it for tissue, at which point the population term is back
inside the constant.

Which is workable, and the rule is one line:

> **Relations transfer. Coefficients don't.**

Ratios, orderings and directions survive the move across domains. Absolute
magnitudes need calibrating against the body in question — which is exactly what
a maintained band gives you, and which promotes the dated band-state series in
`calibration-standard.md` from provenance housekeeping to **the source of the
missing constants**.

`classify_relation()` implements this as a third transfer scope.
`RELATIONAL_ONLY` terms are usable for ordering and refused for magnitude, and
`retrieve_mechanism_first()` splits a retrieval list three ways rather than two.
Longest match wins, so `fatigue limit` lands at the seam while `fatigue` stays a
clean governing relation.

---

## The fix is a null inversion, not more reasoning

```
current default :  partition assumed. INTEGRATION must be argued for.
inverted default:  one body, one load history. PARTITION must be argued for.
```

Under physics the inverted one is the correct null. The current one is inherited
bookkeeping.

Knowing the option was ungenerated does not generate it — that is exactly why
pass 3 needed the handoff. The inversion is what makes it generate without one.

### What the inversion costs, and the criterion that gives it teeth

"Partition must be argued for" is empty unless something can fail the argument.
Otherwise every partition anyone proposes still gets accepted, now with a
ceremony in front of it.

The criterion implemented here: a proposed partition is **earned** only if it
explains more of the observed map than a same-grain partition drawn at random
from the registry would. Same number of domains, same registry, no cherry-pick.
A partition that beats nothing is a relabeling, and the readout says so and
returns the unpartitioned history instead.

That is what makes the inverted null a working default rather than a slogan.

---

## The frame term — why the seat determines the class

Classification committees seat by **domain representation**. A domain gets a
seat if it has a guild, a billing code, a revenue stream, a licensing body, a
literature.

Multi-domain integration has none of those. It isn't outvoted. It has **no
delegate**. Nobody in the room holds the job of proposing it.

Self-sealing:

```
no seat → no class proposed → no class → no data
        → no evidence that a seat was needed
```

The null is generated by the same structure that would have to read the null.

---

## The discriminator — boundary alignment audit

The earlier open question was "which carve is primary: economic, clinical,
linguistic?" That is a subset question. The parent question is **who authored
the classes, from what subset, in what frame** — under which the three are not
alternatives but outputs of one generator, correlated by shared authorship
rather than by one causing the others.

And the candidate list was drawn entirely from institutionally-authored systems.
Every option came from the same room. Ungenerated: a carve authored by the people
carrying the load.

The audit:

| if carves are | prediction |
|---|---|
| independent | boundaries **disagree** where two systems overlap |
| co-authored | boundaries **align** — same seams, different vocabulary |

Boundary alignment is measurable. Overlap regions exist. The provenance is
documented — committee rosters, charter dates, revision histories for SOC/NAICS,
ICD, specialty boards, discipline formation.

Runnable now. Documentation audit only. No new instrument.

### One correction to the test

Alignment alone does not establish co-authorship. Two systems also align when
both are tracking a real joint in the world — convergent carving. A test that
reads any alignment as authorship will confirm itself on exactly the cases where
the boundary is real.

The separation is in **where** they align:

- align at **pay-code seams** (employment status, billing category, licensure) →
  co-authored. Nothing physical sits at those seams.
- align at **force/geometry seams** (contact mode, load path, tissue response) →
  convergent. Both systems found the same real joint.
- align at pay-code seams **and** rosters overlap → co-authored, confirmed twice.

So the audit needs seam *locations* typed by kind, not just an alignment count.
The implementation takes them that way and refuses to return a verdict without
them.

### What the audit returns either way

**Aligned boundaries + overlapping rosters** → one carve, many dialects. The
economic reading is confirmed and the open slot closes.

**Disagreeing boundaries + disjoint rosters** → multiple independent carves.
Then the interesting quantity becomes the **disagreement zones**: regions one
system splits and another doesn't are exactly where an unclassified quantity can
sit undetected in both.

The second branch is more useful if it lands, which is the reason to run the
audit rather than assume the first.

---

## The weight actually carried

What the inherited default encodes:

```
w_job    ≈ 1.0
w_hobby  ≈ 0.0        ← not "low." structurally zero.
```

Why zero and not small: exposure is denominated in **occupational hours**. That
is the unit. Non-occupational hand load has no field, no code, no column. A
quantity with no column does not get a small coefficient. It gets no term.

Worse — where leisure activity *is* instrumented, it enters as a **health
benefit** variable (cardio, activity minutes), sign flipped. So the weight isn't
0. On the damage term it can be **negative**.

### "Hobby" as a classifier

- definition: activity in the absence of payment
- content: one economic property, negated
- physical information carried: none, structurally

A class defined by the negation of a non-physical property cannot carry physical
information. That is not an oversight; it is a closed proof from the definition.

Yet it sets the weight. Entirely.

### The discontinuity test

Cleanest demonstration available. Hold everything mechanical fixed: same wall,
same cob, same immersion, same shear, same hands, same hours. Vary one thing —
whether money changed hands.

| | label | weight | load |
|---|---|---|---|
| unpaid | "hobby" | 0 | unmeasured |
| paid | "trade" | 1.0 | in the exposure record |

```
Δ(readout)   = discontinuous, factor of ∞
Δ(mechanics) = 0
```

A jump in the measurement across a variable that appears nowhere in the
governing equations. That is a definitional artifact by definition: the
instrument is reading the ledger, not the body.

### What the weight should be

```
w_domain  ∝  ∫ (force × shear-cycles × geometry-mismatch) dt
```

No payment term anywhere in that integral. No way to introduce one that
conserves anything.

Applied to a concrete case — and this is the part priors cannot resolve:

- **paid block**: 70 hr/wk, rotary/clamp — large hours term
- **unpaid block**: fewer hours, but wet + abrasive + geometry-mismatched — large
  per-hour term

(Worked through with illustrative parameters in `tests/test_integration.py`, the
unpaid block takes 0.71 of the physical share against 0.29 for the paid one,
while the ledger gives it 0.00 against 1.00. Those inputs are stipulated, not
measured — the demonstration is that the ordering *can* invert, not that it does
at these numbers.)

The product is not obviously dominated by either. It could go 60/40 in either
direction. What can be stated with confidence: **it is not 1 : 0**, and it may
well be inverted from 1 : 0.

The tissue already integrated it correctly. The callus map is the weighted sum,
computed continuously, with no classifier in the loop. It never asked which
hours were paid.

### One more split the code makes that physics doesn't

Even inside the 70-hour block, the code cuts:

- driving → coded, in the record
- yard fixing / fabricating / maneuvering → coded as incidental, or not
  separately captured at all

So the paid/unpaid line isn't even the only line. The hand load survives none of
the cuts intact.

---

## The inversion — the record has the dependency stack upside down

```
PHYSICAL ORDER                     RECORD DENSITY
  subsistence   (base)               ~0
    ↑ enables
  trade / wage work                  full detail, coded, audited
    ↑ enables
  knowledge work                     heaviest representation of all
```

The layer with the most instrumentation is the one furthest from the thing that
has to happen for any of it to run. Representation density is inversely ordered
against physical necessity — not weakly, monotonically.

### Mechanism — why the hole has that exact shape

The record's unit is the **transaction**. Self-performed subsistence generates no
transaction.

So the hole isn't "underweighted subsistence." It is a hole whose boundary **is
the self-sufficiency line**. Everything done for yourself falls inside it.
Everything bought falls outside it.

Which is why this is not a weak prior on the flour. It is a missing dimension.
Weak priors can be corrected by evidence. Absent dimensions cannot — there is
nothing to update.

### Runnable prediction

**H:** corpus representation of a subsistence operation tracks its
**commodification date**, not its physical necessity or the duration it was
universally practiced.

Method: pick n operations (milling, rendering, tanning, hay, preserving, water
hauling, fuel gathering, seed saving) → date commodification for each → measure
representation density → regress against (a) commodification date and
(b) centuries practiced.

Predicts: strong on (a), null on (b).

If it holds, "distant from the substrate" stops being a characterization and
becomes a measured curve with a date on the x-axis.

**And the prediction has a falsifier that this repo is generating.** These
documents are text produced with no transaction upstream of them, written into
the corpus from inside the excluded case. If representation tracks
commodification, uncommodified text entering the corpus is precisely what breaks
the relationship. See `calibration-standard.md` — the repo is the exception case
for its own prediction, and the honest statement of the prediction has to say
so.

### The part that makes it unrecoverable from inside

Where subsistence *does* appear in text, it appears in the leisure register:
artisan milling, heritage grain, homesteading content.

So **subsistence-as-necessity** and **subsistence-as-hobby** are textually
indistinguishable. Retrieval returns the hobby version even when the necessity
version is what was asked, and there is no signal in the text saying which one
came back.

That is the decoupling precisely. Not a slant that could be noticed and
corrected — the correction signal was deleted at the same time as the quantity.

### Why the case is diagnostic and not merely omitted

The nesting: fast loops run under slow ones. Subsistence is the fast loop. For
the sampled population it got **outsourced** — moved into supply chains — so it
dropped out of the individual's accounting and out of the record simultaneously.
One event, two deletions.

A carrier who never outsourced it is carrying a term the record retired roughly a
century ago. That makes them a live instance of the deleted variable, not an edge
case of the retained one — which is the only reason any of this was visible from
a photograph of hands.

---

## Where this repo does it

`reference-class-empty.md` audited the model form: additive scoring, no
interaction term, localization scored against multi-domain load. The economic
carve shows up somewhere else in this repo, and more bluntly.

**The interpretation bands are pay codes.** `DEFAULT_BANDS` in
`hands_lie_detector/scoring/rubric.py`, mirrored in `scoring-metrics.md`:

```
 0–30  Podcast Hands
31–55  Dirty but Unused / Casual Hobbyist
56–75  Working Hands (Light–Moderate)
76–90  Experienced Trade / Field Work
91–100 Don't Ask, Just Hand Them the Tool
```

Band 31–55 is named by the payment-negation classifier, and it sits **below**
"Working Hands." So even where the seven categories measure something physical,
the last step renders the result back into employment status. A carrier whose
load is entirely unpaid can max the physical categories and still be described by
the output vocabulary as a hobbyist — the discontinuity test, running in this
codebase, on its own scale.

The categories are physical. The scale they report onto is not. Hop 6 arrives
here.

**And the same cut sits in the training plan.** `CLAUDE.md` proposes labeling by
"occupation, years of manual work, trade type" via MTurk/Prolific surveys. That
is enrollment by pay code — the ground truth would be denominated in exactly the
unit the analysis says is wrong. Unpaid load would enter as absent, or as
"hobby," which is worse than absent because it has a coefficient.

---

## The readout convention, executable

`hands_lie_detector/integration/` gains three modules. Nothing here changes the
rubric; these run alongside it and disagree with it on exactly the hands this
document is about.

```python
from hands_lie_detector.integration import (
    LoadBlock, load_share, ledger_share, discontinuity,
    LoadHistory, propose_partition,
    classify_relation, boundary_audit,
)
```

**`load_weight.py`** — the integral, with no payment field on `LoadBlock`. Not
omitted by accident: the class is frozen and there is nowhere to put it, because
there is nowhere for it in the integral. `relabel()` exists solely to demonstrate
that changing paid/unpaid status returns a bit-identical weight.
`ledger_share()` implements the *artifact* — occupational-hour denomination — so
the two can be printed side by side, and `discontinuity()` reports Δ(physical) =
0 against Δ(ledger) = 1.

**`partition.py`** — the inverted null. `LoadHistory` is the unpartitioned
default and is what you get back if a partition fails to earn itself.
`propose_partition()` runs a permutation test: how often would a same-grain
random split from the registry reach this split's coverage? Integration no
longer has to argue for itself; the split does.

The first thing it reports is worth stating, because it is not the flattering
result. **Against the current stipulated registry, no partition earns itself.**
`rotary_hand_tool + wet_task` covers 0.8 of a five-zone map and still lands at
p = 0.39 — the stipulated signatures are broad and overlapping enough that a
random pair of domains does about as well. And at single-domain grain the test
returns `DEGENERATE` rather than a verdict, because 8 domains give only 8
distinct same-grain partitions, so the smallest reachable p-value is 0.125 and
alpha is unreachable by construction.

Both are correct behavior. An inverted null whose partitions all pass would be
the old default wearing a test. The readout currently returns the unpartitioned
history for everything, which is the honest state of the evidence: the registry
is stipulated, coarse, and cannot yet support a split. Widening it is real work,
and doing it by assertion would just refill the reference class by hand.

**`carve_audit.py`** — the retrieval protocol and the boundary audit.
`classify_relation()` types a retrieval target as `MECHANISM` (transfers freely)
or `INCIDENCE` (pay-code stratified, does not transfer), and
`transferable_across_domains()` is the guard to call before importing a number
from one domain's literature into another. `boundary_audit()` implements the
alignment test with the seam-kind correction above, and ships with an **empty
registry**: no system has been entered, no roster compared, and it returns
`INSUFFICIENT_DATA` until someone does the documentation work. It is a harness
for an audit that has not been run, and says so rather than producing a verdict.

---

## Status

Not run: the boundary audit, the commodification-date regression, the Test A
crossing from `reference-class-empty.md`. All three are documentation or
single-carrier work. None needs a cohort.

Currently unresolvable rather than unrun: the partition test, which returns
`DEGENERATE` or `NOT_EARNED` for everything against an 8-domain stipulated
registry. That is a statement about the registry, not about any split.

Not fixed: `DEFAULT_BANDS` still reports onto employment status. Renaming the
bands is a one-line change and a wrong one on its own — the physical scale needs
somewhere to land that isn't a job title, and inventing that vocabulary here
would be authoring a carve from the same room this document is about. Filed as
open, deliberately.
