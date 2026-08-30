## Known Failure Case: Clean Working Hands

Description:
Hands with:
- persistent texture
- localized thickening
- healed micro-injuries
- tendon prominence

Observed Model Output:
"Office worker"

Interpretation:
Model equates cleanliness with lack of experience.
Fails to account for washing frequency and reuse.

Human Verdict:
Incorrect.


Known Failure Case: Clean Working Hands
Description:
Hands with:
	∙	persistent texture
	∙	localized thickening
	∙	healed micro-injuries
	∙	tendon prominence
	∙	deep palm creasing
	∙	bilateral adaptation patterns
Observed Model Output:
“Office worker”
What the model detected:
	∙	Clean skin surface
	∙	No visible dirt/grime
	∙	No tools in frame
	∙	No contextual props
What the model missed:
	∙	Structural tissue remodeling
	∙	Permanent adaptation markers
	∙	Climate-appropriate PPE evidence
	∙	Distinction between “unused” and “maintained”
Root cause:
Model trained on correlation (clean → office, dirty → labor) rather than causation (mechanical load → tissue adaptation).
Interpretation:
Model equates cleanliness with lack of experience.
Fails to account for:
	∙	Washing frequency and reuse
	∙	Post-shift hygiene
	∙	Cold climate glove usage
	∙	Sustainable work practices
Human Verdict:
Incorrect.


---

## CANONICAL ENTRY — the two-day scrub

The cleanest statement of the failure this repo exists for. Same hands, same
week, one variable moved, classification flipped. Everything else held constant
by circumstance rather than by design, which is what makes it usable.

What a scrub removes:

- stain, oils, surface film — the **low-frequency layer**
- and incidentally: defats the stratum corneum, raises reflectance, softens the
  surface transiently

What a scrub cannot touch:

- the callus map
- boundary geometry
- plate thickness

Keratin turnover runs 2–4 weeks. Two days is nothing.

```
everything REMOVABLE was removed.
everything STRUCTURAL was intact.
5/5 models read the removable layer.
```

They measured a two-day-old washing event and reported it as a life history.

And the context in frame pointed the other way — diamond-plate floor, work
boots, cable, quick-connect fitting — and lost anyway. So the finding is not
"models overfit to context." It is a **dominance ordering**: skin cleanliness
outranks every other cue, including cues that contradict it.

## CANONICAL ENTRY — the paired grime frames

The companion, and it closes the argument from the other side.

| frame | grime | wound | activity |
|---|---|---|---|
| A | heavy | fresh | working |
| B | clean | fresh | working |

Same season class, same activity class, opposite grime.

⇒ **grime varies freely against work state within one operator's own set.**

Not an argument that grime is a bad cue. A demonstration — paired, one operator,
controlled by circumstance rather than by design.

Put next to the scrub entry, the two close it:

```
cleanliness DOMINATES the classifier      (scrub specimen)
cleanliness CARRIES NO INFORMATION        (paired frames)
```

Both shown. Neither asserted.

---

## CANONICAL ENTRY — the knuckle field, unread rather than misread

A different failure from the two above, and a worse one.

The scrub and paired-grime entries document models producing a **wrong answer**:
they read the removable layer and reported a life history. The knuckle case is
that there is **no category for the marker at all**.

Why the dorsum is absent from what models learned:

- training data is overwhelmingly **palmar** — grip analysis, palmistry,
  biometric capture
- shot frontally or from overhead, never **raking across the knuckle ridges**
- clean, or uniformly dirty

And the marker itself is hostile to that setup: small relative to hand area, low
contrast without oblique light, and outside the grip vocabulary entirely. Where
it does register, it tends to come back as "rash", "dermatitis" or "poor skin
condition" — the model reaching for the nearest class it has.

```
cleanliness case   a WRONG ANSWER    the model had a category and picked badly
knuckle case       a MISSING QUESTION the model has no category to pick
```

Which makes it invisible to any evaluation that scores answers. A model cannot be
marked wrong on a question nobody asked it, and the forced-classification design
in the unrun core test does not fix this on its own — the question has to name
the dorsal surface or the failure stays unmeasured.

See `knuckle-instrument.md`.

---

## Why this file matters more than the results file

The provenance function only holds if failures commit as prominently as
successes. A dated, content-addressed, externally-anchored record of what a model
could do is worth something only if it also records what it could not — otherwise
it is a highlight reel with a commit hash.

That is what makes this file load-bearing rather than supplementary. It is also
why `specimen-record.md` keeps its own retraction in the record instead of
editing to the corrected verdict.

New misreads go in `specimen-record.md` in the specimen format, with per-line
provenance marks. This file holds the standing catalogue.
