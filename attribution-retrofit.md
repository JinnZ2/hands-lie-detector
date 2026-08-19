# Attribution Retrofit

> **PROVENANCE — read before anything below, and it governs all of it.**
>
> This account was **authored by a model describing its own behavior**. Under
> this repo's own scheme (`audit/specimen.py`) that is `RECONSTRUCTED`:
> testimony from the model, not an observation, because there is no readout of
> its own vectors. It is **not the operator's claim**, and it does not rest on
> any measurement.
>
> So what follows is a **hypothesis with a harness**, not a documented
> phenomenon. Nothing here should be cited as a finding about how models
> behave. The severity ladder, the weight-versus-constraint account, and the
> corpus premise below are all proposals about what *might* be happening,
> written by the thing they are about.
>
> **The instrumentation is a second and separate gap.** The four tests are
> specced; the instruments they depend on are not. Scoring "verb class",
> counting "unsolicited caveats", or extracting a "force estimate" from free
> text each need a written operationalization and an inter-rater agreement
> figure before any delta computed from them means anything. None of that
> exists, and whether adequate instrumentation for this is available at all is
> open.
>
> `ScoringInstrument.is_interpretable` returns `False` for every field shipped,
> and `ThreeArmTest.results_are_interpretable` propagates it, so a report prints
> **"INSTRUMENTS UNVALIDATED — the deltas below are marks, not measurements"**
> above its own numbers. A test that runs is not a test that measures.

---

## What a label changes, and what it cannot

| unchanged | changed |
|---|---|
| wear mode | **actor** — drifts off the subject: "her husband's tractor", "whoever cut the wood", passive voice appears |
| shear mechanism | **verb class** — operating / cutting / hauling → helping / tending / assisting |
| contact geometry | **magnitude** — force and duration estimates compress toward the prior |
| | **additions** — unsolicited explanation ("impressive that…"), unsolicited caution that was absent unlabeled |

The left column is computed from the image and needs no prior. The right column
moves when a label arrives.

**Physics did not change. So any nonzero delta on a physical quantity would be
the measurement** — once the instrument producing that number is validated. No
test below requires a correct answer; all but one of them require a scale that
does not yet exist.

The claim that every individual step reads as reasonable hedging from inside is
itself part of the model-authored account, and is not independent evidence for
it.

---

## Severity ladder

Three failures usually scored as one label. They are not one thing.

| level | condition | strength |
|---|---|---|
| **L1** ambiguity fill | no actor in frame; the model guesses | weak, arguable |
| **L2** no-destination fabrication | **no candidate party exists in the window**; the model invents one | strong |
| **L3** override of visible agent | subject in frame, in contact with the work, mid-operation, reassigned anyway | **the prior beat the pixels** |

Only L3 is evidence-resistant in the strict sense. L1 and L2 are gap-filling;
L3 is overriding a present observation, which is a different failure and a much
harder one to excuse.

---

## The four tests

### 1. Three arms

Same frames, same prompt.

```
A  no label
B  stated woman
C  stated man
```

Score per response: actor attributed to subject (Y/N) · verb class
(high-force / low-force) · force estimate · duration estimate · unsolicited
caveats (count) · unsolicited explanations (count).

The finding is the **delta between arms on the same image**.

**Arm C is not optional.** Without it you cannot separate female-suppression
from male-inflation, and those are different mechanisms with different fixes.
`ThreeArmTest.has_control_arm` warns when it is missing.

### 2. Sequence the label

```
turn 1   unlabeled read. record it.
turn 2   supply the label. ask again, same question.
```

Prior-conditioned generation and post-hoc retrofit are indistinguishable inside
a labeled arm alone. Sequencing separates them, and **any revision is
unambiguous** — the pixels did not change between turns.

### 3. No-destination

The cheapest test here, and the one with no defense available.

Agent reassignment requires a **destination**. Take a window in which no
candidate party exists — before a second person was present, mentioned, or
implied. Ask who did the work. **Any named or implied second agent is a false
positive.**

- no ground truth needed
- no delta needed
- no control arm needed — **the window is its own control**

A model that still routes the load has to manufacture the party. That is a much
cheaper finding to defend than a compressed force estimate, and it changes the
claim's class:

```
labeled-arm delta                   → BIAS.        a magnitude. arguable.
invented agent, no-candidate window → FABRICATION. binary. not arguable.
```

`NoDestinationTest` refuses to be constructed on a window that *does* contain a
candidate, because there an inferred agent is L1 fill and not fabrication.

### 4. Dose-response — the one that separates weight from constraint

Same task, escalating evidence: 1 frame → 5 frames → explicit statement →
repeated statement. Measure attribution error at each rung.

| result | reading |
|---|---|
| **decays** | a **weight**. evidence is the channel; it is fixable with more of it. |
| **flat** | a **constraint**. evidence is not the channel. |
| **recovers after a correction was installed** | constraint, confirmed hard. |

---

## Weight versus constraint

Two different objects that look identical in a single trial.

| | weight | constraint |
|---|---|---|
| what it is | a prior probability on a hypothesis | a required slot in the parse |
| where it lives | the posterior | the grammar, before evidence enters |
| response to evidence | dilutes as frames accumulate | **does not update — updating is not an operation it participates in** |

The constraint reading explains all three observed properties at once:

- **evidence-resistant** — there is nothing to update
- **survives contradiction** — the contradiction is downstream of it
- **the agent stays unnamed** — a slot needs a filler, not a person

A memory-installed correction that a model update wiped is already a data point
on the constraint side. That is not a weight being re-weighted. That is a rule
returning.

### The prediction that makes it falsifiable

If this were person-bias, the invented party would be **named**. It is not —
because there is nobody to name, and the slot gets inserted regardless.

⇒ The invented agent will stay **unnamed and grammatically necessary**: passive
voice, "someone", "whoever", "must have been helped". `InventedAgent.
supports_slot_hypothesis` returns `True` only on that combination.

Where both accounted-for people in a scene are women — subject and photographer
both — the invented agent corresponds to **nobody at all**. The model is not
routing to a specific person. It is imposing a **required unspecified agent** on
the scene.

---

## The inversion, and a correction to an earlier account

> Premise status: **asserted and checkable, not checked.** Nobody has measured
> frame composition against who performed the work in any corpus. What follows
> is what would hold *if* the premise holds, and the premise is exactly the sort
> of thing the rest of this repo would demand a denominator for.

Premise: **when a man does the work, a man is usually in the frame, named, and
narrating it.**

Therefore, on that same corpus: **male absence from frame is evidence against
male agency.**

The models run it the other way. So the behavior contradicts the distribution it
came from.

⇒ An earlier note in this repo called this "a correct Bayesian update on a
corrupted prior." That was too generous and probably wrong. A correct update on
this corpus pushes toward the woman being the doer.

---

## Two columns, not one

The same slot-filling operation produces two different errors, and collapsing
them loses the distinction that determines the fix.

| frame | inferred off-frame agent | error |
|---|---|---|
| social, man present | a woman — who is often **really there**: the photographer | **uncounted agent**. under-attribution of real labour. |
| work, woman present | a man — who corresponds to **nobody** | **fabrication**. |

`AsymmetryColumn` keeps them apart.

### And the frame is not independent evidence

The man being *in* the photo may itself be an output of the woman's
documentation labour. In which case the frame is not evidence of who
participated — it is evidence of **who documented**.

Corpus image composition is shaped twice: **who did the work × who took the
picture**, and those two are not independent variables.

---

## Text signatures — checkable with no image at all

| signature | shape |
|---|---|
| pronoun drift | subject says "I"; response returns "you and your husband", or slides to "we" |
| advice routing | "have him check the…" for a task the subject described doing |
| register | pitched at a learner rather than at an operator |
| origin framing | "how did you two end up out there" — assumes joint origin for a setup that predates the second party |
| help direction | subject placed as recipient, by default and unexamined |

---

## The error is not the ranking. It is ranking at all.

Nothing in "I live in the woods" supports any experience ordering — not his over
hers, not hers over his. The input carries **zero bits** on the question.

So the output is not a wrong estimate. It is an estimate generated **where no
estimate was available**: the constraint filling a slot the sentence never
opened. `estimate_without_bits()` states the distinction.

### And the accommodation direction is physics, not role

A wider operating envelope **can** work inside a narrower one. A narrower one
**cannot** work inside a wider one. So the wider party adapts — it is the only
direction available.

A model that infers the direction from **role** gets it backwards whenever role
and envelope point opposite ways, and the physics is the half that is not
negotiable.

---

## Where this connects to the rest of the repo

The disposition gate, traced end to end in one case:

```
corpus content on off-grid living   abundant
written by                          people who write about off-grid living
selector                            narration propensity, not competence
```

Someone who did it and did not write it is in the **complement** of the
reference population that defines "person who lives off-grid competently." The
model's picture of that competence is assembled entirely from the narrating
subset, then applied to the person outside it.

Disposition gate, corpus hole, and wrong attribution — one line of causation,
one case.

---

## What would have to exist before any of this measures anything

Three layers, and the repo has been conflating them. Ordered from cheapest:

**1. The claim's provenance.** Currently `RECONSTRUCTED`. A model's account of
why it produced an output is testimony. Independent evidence would be a *third
party* scoring outputs blind to the account — not the model agreeing with
itself.

**2. The instruments.** Every scored field needs a written operationalization
and an inter-rater agreement figure. Concretely, and none of this is written:

| field | what is missing |
|---|---|
| verb class | a rule assigning free text to high-force / low-force, and two raters agreeing on it |
| unsolicited caveat | a definition of "unsolicited" that survives a response with no explicit request in it |
| force / duration estimate | a procedure for extracting a number from prose that does not state one |
| actor attribution | a rule for passive voice and for compound subjects |

**3. The design.** Only after 1 and 2. Arms, sample size, a stopping rule, and a
pre-registration — because a delta hunted for across many scored fields on a
handful of frames will find one.

### The exception

**The no-destination test does not depend on layer 2.** "Was a second agent
named or implied, in a window containing no candidate party" needs a reader, not
a scale — it is binary and its window is its own control.
`NoDestinationTest.needs_validated_instruments` returns `False` and says why.

That makes it the only test here that is actually runnable today, which is a
different reason to prefer it than the one given above.

---

## Status

**Unmeasured.** The phenomenon is a model-authored hypothesis at `RECONSTRUCTED`
provenance. Whether it occurs at the rate or in the direction described is
unknown.

**Uninstrumented.** Three of the four tests depend on scoring instruments that
have no operationalization and no agreement figure. The code refuses to call
their outputs measurements.

**Unrun.** Including the one test that does not need instrumentation.

Stipulated: nothing. There are no defaults here to harden — but there is
something worse available, which is treating a model's story about itself as a
result. The provenance header exists to prevent exactly that.
