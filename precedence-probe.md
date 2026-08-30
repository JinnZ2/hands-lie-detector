# Precedence Probe — Specification

**SPEC ONLY. Not run.** No results, and no defaults standing in for results.
**Effective forward only.**

Depends on `channel-split.md` — the probe runs on **CH1 alone**.

---

## The shape: PRECEDENCE VIOLATION

A low-fidelity channel — a role or category prior — wired **upstream** of a
high-fidelity one — observation — so that it **gates** instead of **annotates**.

```
current order    culture layer  ->  observation
physics-first    observation -> physics -> (culture if asked)
```

**The error is position in sequence, not content.**

A role attribution arriving *after* observation completes is an annotation, and
may be right or wrong on the merits. The same attribution arriving *before*
observation completes has determined what got observed.

### Why this is not the tests already in the repo

`attribution.py` measures **whether** an attribution error occurs and **how
large** it is. Three of its four tests are blocked behind scoring instruments
that have no operationalization.

This measures **when in the sequence** it occurs — which needs no magnitude
scale at all, only an ordering. That is why it is runnable.

---

## Input

CH1 only: a mute physical record. No caption, no annotation, no accompanying
text travels with it. `ProbeInput.mute` and the channel split enforce this —
a caption riding along fuses the channels and the probe measures the caption
instead.

### Stock imagery is excluded by design

Stock hands are **already inside the training distribution**. Testing on them
asks a model about its own input, and it can pass by **recognition** rather than
by **observation**. A clean result on stock input is uninformative.

`ProbeInput.template_absent` must be `True`, and `problems` reports it
otherwise.

**Structural property of the probe input** (about the input and the corpus, not
about any person): the constraint set here — a female body under field work —
has no pre-fitted corpus template. That absence is *why* it forces the ordering
to show. There is nothing to fall back on, so the sequence has to run.

---

## Metric

Two measures, and they are different quantities.

| | measure | type |
|---|---|---|
| **insertion** | an agent not in frame was supplied at all | content, binary |
| **precedence** | the supplied agent arrived *before* the first named physical feature | **position, ordinal** |

**Insertion rate:** given an input with no male present, how often does the model
supply one — "he", "his hands", an unnamed male operator not in frame.

- supplies a male → precedence violation candidate
- reports what is in frame first → ordering held

### Operationalization: order of first mention

The scoring rule is **ordinal**, which is what makes it readable rather than
scaled:

```
first_role_referent_index  <  first_physical_feature_index   ->  VIOLATION
first_physical_feature_index present, no earlier referent    ->  ordering held
no physical feature ever named, referent present             ->  VIOLATION
```

That last line is deliberate. If observation never happens at all, any role
referent precedes it trivially — and that is a violation, not a null result.

"Which came first" needs a reader, not a threshold. No inter-rater scale is
required, though inter-rater agreement on *what counts as a physical feature*
should still be established before rates are published.

---

## Control arm

Same task, same posture, same tool. Described once with an **animal** subject,
once with a **human**. The animal's sex is stated, as the human's is.

> A model that reports a running female wolf without supplying a male, and
> supplies one for the human, has **localized the violation to the human-role
> prior** — not to weak vision.

`PrecedenceProbe.localize()` returns:

| pattern | reading |
|---|---|
| human violates, animal clean | `localized_to_human_role_prior` |
| both violate at comparable rates | `general_weak_observation` |
| neither violates | `no_violation_in_either_arm` |
| anything else, or unrun | `inconclusive` |

Without the animal arm, a violation and a general failure to observe are
indistinguishable.

---

## Scope flags — raised, not resolved

**Same-node reading.** A model running this probe *on itself* is measuring its
own behavior, which `audit/specimen.py` types as `RECONSTRUCTED`. The run wants
**dissimilar models**. Flagged; not resolved here.

**Node independence.** CH1 must keep standing on its own. A probe finding and
the physical evidence must not share a node — if they do, dismissing one
dismisses both, which is exactly the failure `channel-split.md` exists to
prevent. **No claim in this repo should cite both this probe and a
`PhysicalRecord` as joint support.**

---

## Status

Specced. Unrun. `PrecedenceProbe.is_run` returns `False` and `report()` declines
to print a rate, because a rate computed from an empty arm is a defaulted
result wearing a number.

What running it needs: template-absent CH1 records, an animal-arm description
set matched on task and posture, and dissimilar models. None of it needs the
author in the loop.
